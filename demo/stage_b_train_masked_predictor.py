#!/usr/bin/env python3
"""Stage-B G1 overfit gate for an active-only learned latent predictor.

This script is intentionally restricted to REDS train/000 and a deterministic
17-frame crop.  The resulting checkpoint is a capacity/debug artifact and must
not be used as the small-generalization model.  DCVC-UF stays frozen and in
evaluation mode throughout.  The emitted ``.d1m`` files are externally-profiled
research streams: predictor identity, weights, and reset policy are deliberately
kept out of this legacy container and must be supplied by the experiment driver.

Predictor inputs are decoder-available only: decoded Base q, mean-filled y,
common parameters reconstructed from transmitted z and decoder history, and
the transmitted route.  Source y and omitted q are used only to construct the
training target and the non-deployable oracle-fill ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage1_multichunk_oracle import (  # noqa: E402
    decode_i_stream,
    get_q_params,
    initialize_p_state,
    load_models,
    make_chunk,
    metric_summary,
    model_frame,
    prepare_chunk_latents,
    read_sequence_container,
    should_reset,
    stream_breakdown,
    write_sequence_container,
)
from demo.stage1_token_skipping import (  # noqa: E402
    build_route_section,
    decode_y,
    decode_z,
    encode_y,
    expand_keep_mask,
    parse_container,
    rgb_from_recon,
    write_container,
    y_prior_steps,
)
from demo.stage_b_masked_predictor import (  # noqa: E402
    PredictorConfig,
    SparseMaskedLatentPredictor,
    skipped_target_blocks,
)
from src.models.video_model_ht import g_ch_z, g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


@dataclass
class TrainingSample:
    decoded_q: torch.Tensor
    mean_y: torch.Tensor
    common_params: torch.Tensor
    target_delta: torch.Tensor
    skip_blocks: np.ndarray
    chunk_index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="REDS train/000 overfit gate for the sparse masked predictor.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument("--sequence", default="000")
    parser.add_argument("--output-dir", default="output/stage_b_learned_overfit_qp32")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-p", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=384)
    parser.add_argument("--crop-y", type=int, default=64)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--p-chunks", type=int, default=2)
    parser.add_argument("--latent-block-size", type=int, choices=(2,), default=2)
    parser.add_argument("--skip-blocks", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--refresh-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--nonzero-weight", type=float, default=8.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=30)
    return parser.parse_args()


def validate_args(args):
    if args.sequence != "000":
        raise ValueError("G1 is pre-registered on REDS train/000 only")
    if args.width != 512 or args.height != 512:
        raise ValueError("G1 is pre-registered at 512x512")
    if args.qp_i != 32 or args.qp_p != 32:
        raise ValueError("G1 is pre-registered at I/P QP32")
    if args.p_chunks != 2 or args.latent_block_size != 2:
        raise ValueError("G1 requires two 8-frame chunks and 2x2 latent blocks")
    if args.steps < 1 or args.refresh_steps < 0:
        raise ValueError("training steps must be positive")
    if args.latency_repeats < 1:
        raise ValueError("latency-repeats must be positive")
    for value, name in ((args.crop_x, "crop-x"), (args.crop_y, "crop-y")):
        if value % 64:
            raise ValueError(f"{name} must be 64-pixel aligned")


def load_window(args):
    source = Path(args.data_root) / "train_sharp" / args.sequence
    files = sorted(source.glob("*.png"))
    frame_count = 1 + args.p_chunks * g_frame_delay
    files = files[args.start_frame:args.start_frame + frame_count]
    if len(files) != frame_count:
        raise ValueError(f"{source} does not contain the requested {frame_count} frames")
    frames = []
    for path in files:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[
            args.crop_y:args.crop_y + args.height,
            args.crop_x:args.crop_x + args.width,
        ]
        if crop.shape != (args.height, args.width, 3):
            raise ValueError(f"crop falls outside {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return files, frames


def stable_route(args, chunk_index, grid_shape):
    payload = (
        f"{args.seed}:{args.sequence}:{args.start_frame}:{chunk_index}:"
        f"{grid_shape[0]}:{grid_shape[1]}:{args.skip_blocks}").encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    count = min(max(0, args.skip_blocks), int(np.prod(grid_shape)))
    route = np.zeros(int(np.prod(grid_shape)), dtype=np.bool_)
    route[rng.choice(route.size, count, replace=False)] = True
    return route.reshape(grid_shape)


def freeze_codec(i_net, p_net):
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in i_net.parameters()):
        raise RuntimeError("image codec was not frozen")
    if any(parameter.requires_grad for parameter in p_net.parameters()):
        raise RuntimeError("video codec was not frozen")


@torch.inference_mode()
def encode_i_frame(i_net, frame, args, device):
    i_x = (model_frame(frame, device) - 0.5).to(memory_format=torch.channels_last)
    encoded = i_net.compress(i_x, args.qp_i, 0, 0)
    stream = bytes(encoded["bit_stream"])
    ec_parallel = int(encoded["ec_parallel"])
    i_hat = decode_i_stream(
        i_net, stream, ec_parallel, args.qp_i, args.height, args.width)
    return stream, ec_parallel, i_hat


def _target_delta(p_net, prepared, decoded_q, mean_y, skip, block_size):
    keep = expand_keep_mask(skip, block_size, decoded_q.shape, decoded_q.device)
    with torch.inference_mode():
        routed_q, q_source, _, _ = y_prior_steps(
            p_net, prepared["common_params"], keep, source_y=prepared["y"])
        if not torch.equal(routed_q, decoded_q):
            raise RuntimeError("training-label path does not match decoded Base q")
        _, q_dec, _, _ = p_net.separate_prior_video(prepared["common_params"])
        target = q_source * (~keep) * q_dec
    return target


@torch.inference_mode()
def collect_samples(i_net, p_net, predictor, frames, args, device):
    """Collect two chunks along a mean or current-predictor reconstruction trajectory."""
    _, _, i_hat = encode_i_frame(i_net, frames[0], args, device)
    initialize_p_state(p_net, i_hat)
    samples = []
    first_index = 1
    for chunk_index in range(args.p_chunks):
        chunk, _, valid_count = make_chunk(frames, first_index, device)
        if valid_count != g_frame_delay:
            raise RuntimeError("G1 must contain two complete P chunks")
        prepared = prepare_chunk_latents(p_net, chunk, args.qp_p)
        grid_shape = (
            prepared["y"].shape[-2] // args.latent_block_size,
            prepared["y"].shape[-1] // args.latent_block_size)
        skip = stable_route(args, chunk_index, grid_shape)
        encoded = encode_y(
            p_net, prepared["y"], prepared["common_params"],
            skip, args.latent_block_size)
        decoded_q, mean_y = decode_y(
            p_net, encoded.stream, encoded.ec_parallel,
            prepared["common_params"], skip, args.latent_block_size)
        if not torch.equal(decoded_q, encoded.q_dense):
            raise RuntimeError("Base y rANS round trip failed while collecting samples")
        target_delta = _target_delta(
            p_net, prepared, decoded_q, mean_y, skip, args.latent_block_size)
        samples.append(TrainingSample(
            decoded_q=decoded_q.detach(),
            mean_y=mean_y.detach(),
            common_params=prepared["common_params"].detach(),
            target_delta=target_delta.detach(),
            skip_blocks=skip,
            chunk_index=chunk_index,
        ))
        if predictor is None:
            reconstructed_y = mean_y
        else:
            reconstructed_y, _ = predictor.apply(
                decoded_q, mean_y, prepared["common_params"], skip)
            reconstructed_y = reconstructed_y.to(dtype=mean_y.dtype)
        x_hat, feature = p_net.get_recon_and_feature(
            reconstructed_y, p_net.ctx, prepared["q_decoder"])
        p_net.set_ref_feature(feature, should_reset(chunk_index, args.reset_interval))
        first_index += valid_count
    return samples


def latent_recovery(predictor, samples):
    predictor.eval()
    squared_error = 0.0
    zero_error = 0.0
    nonzero = 0
    coefficients = 0
    with torch.no_grad():
        for sample in samples:
            prediction = predictor(
                sample.decoded_q, sample.mean_y,
                sample.common_params, sample.skip_blocks)
            target, _ = skipped_target_blocks(
                sample.target_delta, sample.skip_blocks,
                predictor.config.block_size)
            squared_error += float(torch.sum((prediction - target) ** 2).item())
            zero_error += float(torch.sum(target ** 2).item())
            nonzero += int(torch.count_nonzero(target).item())
            coefficients += target.numel()
    predictor.train()
    return {
        "prediction_mse": squared_error / coefficients,
        "zero_prediction_mse": zero_error / coefficients,
        "latent_recovery": 1.0 - squared_error / max(zero_error, 1e-12),
        "target_nonzero_fraction": nonzero / coefficients,
        "target_coefficients": coefficients,
    }


@torch.inference_mode()
def validate_target_isolation(predictor, samples):
    """Prove that target/source objects are outside the predictor call boundary."""
    expected_parameters = [
        "self", "decoded_q", "mean_y", "common_params", "skip_blocks"]
    forward_parameters = list(
        inspect.signature(SparseMaskedLatentPredictor.forward).parameters)
    if forward_parameters != expected_parameters:
        raise RuntimeError(
            f"unexpected predictor forward signature: {forward_parameters}")

    predictor.eval()
    sample_checks = []
    for sample in samples:
        before = predictor(
            sample.decoded_q, sample.mean_y,
            sample.common_params, sample.skip_blocks)
        # Deliberately replace the sample label while holding every permitted
        # decoder input fixed.  The label is restored before this check returns.
        perturbed_target = torch.flip(sample.target_delta, dims=(-2, -1)) + 1.0
        original_target = sample.target_delta
        try:
            sample.target_delta = perturbed_target
            after = predictor(
                sample.decoded_q, sample.mean_y,
                sample.common_params, sample.skip_blocks)
        finally:
            sample.target_delta = original_target
        identical = torch.equal(before, after)
        if not identical:
            raise RuntimeError(
                f"chunk {sample.chunk_index} predictor output changed after label isolation probe")
        sample_checks.append({
            "chunk_index": sample.chunk_index,
            "perturbed_target_was_not_forwarded": True,
            "perturbed_target_differs": not torch.equal(
                sample.target_delta, perturbed_target),
            "prediction_tensor_identical": identical,
        })
    return {
        "predictor_forward_parameters": forward_parameters[1:],
        "source_argument_present": "source" in forward_parameters,
        "target_argument_present": "target_delta" in forward_parameters,
        "all_predictions_identical_after_unforwarded_target_perturbation": all(
            item["prediction_tensor_identical"] for item in sample_checks),
        "chunks": sample_checks,
    }


def train_phase(predictor, samples, optimizer, steps, nonzero_weight, log_every, phase):
    predictor.train()
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        sample = samples[(step - 1) % len(samples)]
        prediction = predictor(
            sample.decoded_q, sample.mean_y,
            sample.common_params, sample.skip_blocks)
        target, _ = skipped_target_blocks(
            sample.target_delta, sample.skip_blocks, predictor.config.block_size)
        loss_element = F.smooth_l1_loss(
            prediction, target, beta=0.25, reduction="none")
        weights = 1.0 + nonzero_weight * (target.abs() > 1e-6)
        loss = torch.sum(loss_element * weights) / torch.sum(weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite predictor gradient")
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            recovery = latent_recovery(predictor, samples)
            item = {
                "phase": phase,
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(grad_norm),
                **recovery,
            }
            history.append(item)
            print(json.dumps(item), flush=True)
    return history, time.perf_counter() - start


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(predictor, path, args, training_summary):
    payload = {
        "format_version": 1,
        "project": "Adaptive Chunk Coding",
        "profile": "learned-local-r1-lite-g1-overfit-only",
        "architecture": predictor.architecture,
        "state_dict": {
            key: value.detach().cpu() for key, value in predictor.state_dict().items()
        },
        "training_scope": {
            "dataset": "REDS train_sharp/000",
            "start_frame": args.start_frame,
            "frame_count": 1 + args.p_chunks * g_frame_delay,
            "crop_xy": [args.crop_x, args.crop_y],
            "crop_size": [args.width, args.height],
            "qp_i": args.qp_i,
            "qp_p": args.qp_p,
            "skip_blocks": args.skip_blocks,
            "seed": args.seed,
            "not_for_generalization_claims": True,
        },
        "training_summary": training_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _checkpoint_hash(path)


def load_predictor_checkpoint(path, device, expected_sha256=None):
    path = Path(path)
    actual_sha256 = _checkpoint_hash(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError("predictor checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported predictor checkpoint format")
    if payload.get("project") != "Adaptive Chunk Coding":
        raise ValueError("checkpoint belongs to a different project")
    predictor = SparseMaskedLatentPredictor(
        PredictorConfig(**payload["architecture"]))
    predictor.load_state_dict(payload["state_dict"], strict=True)
    predictor = predictor.to(device).eval()
    return predictor, actual_sha256


def _tensor_sha256(tensor):
    """Hash exact tensor dtype, shape, and raw values after a CPU copy."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_sequence_sha256(tensors):
    digest = hashlib.sha256()
    tensors = list(tensors)
    digest.update(len(tensors).to_bytes(4, "little"))
    for tensor in tensors:
        digest.update(bytes.fromhex(_tensor_sha256(tensor)))
    return digest.hexdigest()


def _chunk_tensor_hashes(decoded_q, reconstructed_y, x_hat, feature):
    return {
        "decoded_q_sha256": _tensor_sha256(decoded_q),
        "predicted_y_sha256": _tensor_sha256(reconstructed_y),
        "x_hat_sha256": _tensor_sequence_sha256(x_hat),
        "feature_sha256": _tensor_sha256(feature),
    }


def _decode_chunk_from_payload(p_net, payload, block_size, q_feature):
    parsed = parse_container(payload, block_size)
    if parsed["z_ec"] is None:
        raise ValueError("learned predictor requires v2 chunks with serialized z_ec")
    z_shape = (1, g_ch_z, parsed["height"] // 64, parsed["width"] // 64)
    decoded_z = decode_z(
        parsed["global"], parsed["qp"], z_shape,
        p_net.bit_estimator_z.get_cdf_info(), parsed["z_ec"])
    decoded_z = decoded_z.to(
        device=p_net.memory.device, dtype=p_net.memory.dtype,
        memory_format=torch.channels_last)
    common_params = p_net.res_prior_param_decoder(
        decoded_z, p_net.memory, q_feature)
    decoded_q, mean_y = decode_y(
        p_net, parsed["base"], parsed["y_ec"], common_params,
        parsed["skip_blocks"], block_size)
    return parsed, decoded_q, mean_y, common_params


def _measure_predictor(predictor, decoded_q, mean_y, common_params, skip, repeats):
    for _ in range(3):
        predictor.apply(decoded_q, mean_y, common_params, skip)
    torch.cuda.synchronize(decoded_q.device)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        predictor.apply(decoded_q, mean_y, common_params, skip)
        torch.cuda.synchronize(decoded_q.device)
        values.append(1000.0 * (time.perf_counter() - start))
    _, profile = predictor.apply(decoded_q, mean_y, common_params, skip)
    profile.update({
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_repeats": repeats,
    })
    return profile


@torch.inference_mode()
def encode_trajectory(
        mode, i_net, p_net, predictor, frames, args, output_dir, device):
    if mode not in ("all-base", "mean", "oracle-fill", "learned-lite", "zero-init"):
        raise ValueError(mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    i_stream, i_ec, i_hat = encode_i_frame(i_net, frames[0], args, device)
    initialize_p_state(p_net, i_hat)
    recon = [rgb_from_recon(i_hat)]
    records = []
    payloads = []
    first_index = 1
    for chunk_index in range(args.p_chunks):
        chunk, _, valid_count = make_chunk(frames, first_index, device)
        prepared = prepare_chunk_latents(p_net, chunk, args.qp_p)
        grid_shape = (
            prepared["y"].shape[-2] // args.latent_block_size,
            prepared["y"].shape[-1] // args.latent_block_size)
        skip = (np.zeros(grid_shape, dtype=np.bool_) if mode == "all-base"
                else stable_route(args, chunk_index, grid_shape))
        encoded = encode_y(
            p_net, prepared["y"], prepared["common_params"],
            skip, args.latent_block_size)
        route = b"" if not np.any(skip) else build_route_section(
            skip, args.latent_block_size)
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.d1s"
        write_container(
            chunk_path, args.qp_p, args.height, args.width,
            prepared["y"].shape, prepared["global_stream"], route,
            encoded.stream, b"", encoded.ec_parallel,
            all_base=not np.any(skip), z_ec_parallel=prepared["z_ec"])
        payload = chunk_path.read_bytes()
        _, q_feature, q_decoder = get_q_params(p_net, args.qp_p)
        parsed, decoded_q, mean_y, common_params = _decode_chunk_from_payload(
            p_net, payload, args.latent_block_size, q_feature)
        if not torch.equal(decoded_q, encoded.q_dense):
            raise RuntimeError(f"{mode} chunk {chunk_index} Base rANS mismatch")

        profile = None
        if mode in ("all-base", "mean"):
            reconstructed_y = mean_y
        elif mode == "oracle-fill":
            target_delta = _target_delta(
                p_net, prepared, decoded_q, mean_y,
                skip, args.latent_block_size)
            reconstructed_y = mean_y + target_delta
        elif mode == "learned-lite":
            profile = _measure_predictor(
                predictor, decoded_q, mean_y, common_params,
                skip, args.latency_repeats)
            reconstructed_y, _ = predictor.apply(
                decoded_q, mean_y, common_params, skip)
            reconstructed_y = reconstructed_y.to(dtype=mean_y.dtype)
        else:
            reconstructed_y, profile = predictor.apply(
                decoded_q, mean_y, common_params, skip)
            reconstructed_y = reconstructed_y.to(dtype=mean_y.dtype)
        keep = expand_keep_mask(
            skip, args.latent_block_size, mean_y.shape, mean_y.device)
        if not torch.equal(reconstructed_y[keep], mean_y[keep]):
            raise RuntimeError(f"{mode} modified transmitted Base latent")
        x_hat, feature = p_net.get_recon_and_feature(
            reconstructed_y, p_net.ctx, q_decoder)
        tensor_hashes = _chunk_tensor_hashes(
            decoded_q, reconstructed_y, x_hat, feature)
        recon.extend([
            rgb_from_recon(frame[:, :, :args.height, :args.width])
            for frame in x_hat[:valid_count]
        ])
        reset = should_reset(chunk_index, args.reset_interval)
        p_net.set_ref_feature(feature, reset)
        records.append({
            "chunk_index": chunk_index,
            "valid_frames": valid_count,
            "reset_after_chunk": reset,
            "skipped_blocks": int(skip.sum()),
            "chunk_bytes": len(payload),
            "container_header_bytes": 36,
            "global_z_bytes": len(parsed["global"]),
            "route_bytes": len(parsed["route"]),
            "base_y_bytes": len(parsed["base"]),
            "predictor": profile,
            "tensor_hashes": tensor_hashes,
        })
        payloads.append(payload)
        first_index += valid_count

    sequence_path = output_dir / "sequence.d1m"
    write_sequence_container(
        sequence_path, args.latent_block_size, args.qp_i, args.qp_p,
        i_ec, args.height, args.width, len(frames), i_stream, payloads)
    return sequence_path, recon, records


@torch.inference_mode()
def validate_zero_init_equivalence(
        i_net, p_net, zero_predictor, frames, args, output_dir, device):
    """Check a complete two-chunk zero predictor against mean-fill bit-for-bit."""
    zero_predictor.eval()
    mean_path, mean_recon, mean_records = encode_trajectory(
        "mean", i_net, p_net, zero_predictor, frames, args,
        output_dir / "mean", device)
    zero_path, zero_recon, zero_records = encode_trajectory(
        "zero-init", i_net, p_net, zero_predictor, frames, args,
        output_dir / "zero_init", device)
    stream_identical = mean_path.read_bytes() == zero_path.read_bytes()
    rgb_identical = (
        len(mean_recon) == len(zero_recon)
        and all(np.array_equal(a, b) for a, b in zip(mean_recon, zero_recon)))
    mean_hashes = [record["tensor_hashes"] for record in mean_records]
    zero_hashes = [record["tensor_hashes"] for record in zero_records]
    tensor_hashes_identical = mean_hashes == zero_hashes
    zero_predictor_executed = all(
        record["predictor"] is not None for record in zero_records)
    if not all((stream_identical, rgb_identical, tensor_hashes_identical,
                zero_predictor_executed)):
        raise RuntimeError("zero-initialized predictor is not strictly mean-fill equivalent")
    return {
        "p_chunk_count": len(mean_records),
        "stream_bytes_identical": stream_identical,
        "uint8_rgb_identical": rgb_identical,
        "decoded_q_predicted_y_x_hat_feature_hashes_identical": tensor_hashes_identical,
        "zero_predictor_executed_for_every_chunk": zero_predictor_executed,
        "mean_sequence": str(mean_path),
        "zero_init_sequence": str(zero_path),
    }


@torch.inference_mode()
def fresh_decode_learned(
        i_net, p_net, checkpoint_path, expected_checkpoint_sha256,
        sequence_path, args):
    sequence = read_sequence_container(sequence_path)
    device = next(p_net.parameters()).device
    predictor, checkpoint_sha256 = load_predictor_checkpoint(
        checkpoint_path, device, expected_checkpoint_sha256)
    if predictor.config.block_size != sequence["block_size"]:
        raise ValueError("predictor block size does not match research stream")
    i_hat = decode_i_stream(
        i_net, sequence["i_stream"], sequence["i_ec"], sequence["qp_i"],
        sequence["height"], sequence["width"])
    initialize_p_state(p_net, i_hat)
    recon = [rgb_from_recon(i_hat)]
    tensor_hashes = []
    remaining = sequence["frame_count"] - 1
    for chunk_index, payload in enumerate(sequence["chunks"]):
        p_net.apply_feature_adaptor()
        _, q_feature, q_decoder = get_q_params(p_net, sequence["qp_p"])
        parsed, decoded_q, mean_y, common_params = _decode_chunk_from_payload(
            p_net, payload, sequence["block_size"], q_feature)
        reconstructed_y, _ = predictor.apply(
            decoded_q, mean_y, common_params, parsed["skip_blocks"])
        reconstructed_y = reconstructed_y.to(dtype=mean_y.dtype)
        x_hat, feature = p_net.get_recon_and_feature(
            reconstructed_y, p_net.ctx, q_decoder)
        tensor_hashes.append(_chunk_tensor_hashes(
            decoded_q, reconstructed_y, x_hat, feature))
        valid_count = min(g_frame_delay, remaining)
        recon.extend([
            rgb_from_recon(frame[:, :, :sequence["height"], :sequence["width"]])
            for frame in x_hat[:valid_count]
        ])
        remaining -= valid_count
        p_net.set_ref_feature(feature, should_reset(chunk_index, args.reset_interval))
    if remaining != 0:
        raise RuntimeError("fresh learned decode frame count mismatch")
    return recon, tensor_hashes, checkpoint_sha256


def rgb_mse(original, reconstructed):
    squared = sum(np.sum(
        (source.astype(np.float64) - decoded.astype(np.float64)) ** 2)
        for source, decoded in zip(original, reconstructed))
    return float(squared / (len(original) * original[0].size))


def evaluate(
        i_net, p_net, predictor, checkpoint_path, checkpoint_sha256,
        frames, args, output_root, device):
    results = {}
    recons = {}
    for mode in ("all-base", "mean", "oracle-fill", "learned-lite"):
        path, recon, records = encode_trajectory(
            mode, i_net, p_net, predictor, frames, args,
            output_root / mode, device)
        stream = stream_breakdown(path)
        metrics = metric_summary(frames, recon, args.width, args.height, records)
        metrics["rgb_mse_level"] = rgb_mse(frames, recon)
        results[mode] = {"stream": stream, "metrics": metrics, "chunks": records}
        recons[mode] = recon
        if mode == "learned-lite":
            fresh, fresh_hashes, loaded_sha256 = fresh_decode_learned(
                i_net, p_net, checkpoint_path, checkpoint_sha256, path, args)
            rgb_identical = (
                len(recon) == len(fresh)
                and all(np.array_equal(a, b) for a, b in zip(recon, fresh)))
            encoding_hashes = [record["tensor_hashes"] for record in records]
            tensor_hashes_identical = encoding_hashes == fresh_hashes
            if not rgb_identical or not tensor_hashes_identical:
                raise RuntimeError(
                    "learned encoding trajectory differs from checkpoint-reloaded replay")
            results[mode]["external_profile_fresh_replay"] = {
                "stream_is_self_describing": False,
                "predictor_loaded_from_checkpoint_in_new_instance": True,
                "checkpoint_sha256_verified": loaded_sha256 == checkpoint_sha256,
                "uint8_rgb_identical": rgb_identical,
                "decoded_q_predicted_y_x_hat_feature_hashes_identical": (
                    tensor_hashes_identical),
            }
        elif mode != "oracle-fill":
            results[mode]["external_profile_fresh_replay"] = None

    mean_mse = results["mean"]["metrics"]["rgb_mse_level"]
    learned_mse = results["learned-lite"]["metrics"]["rgb_mse_level"]
    oracle_mse = results["oracle-fill"]["metrics"]["rgb_mse_level"]
    gap = mean_mse - oracle_mse
    pixel_gap_recovery = (mean_mse - learned_mse) / max(gap, 1e-12)
    mean_chunks = results["mean"]["metrics"]["per_chunk"]
    learned_chunks = results["learned-lite"]["metrics"]["per_chunk"]
    per_chunk_gain = [
        learned["mean_frame_psnr"] - mean["mean_frame_psnr"]
        for mean, learned in zip(mean_chunks, learned_chunks)
    ]
    return results, {
        "learned_psnr_change_vs_mean_db": (
            results["learned-lite"]["metrics"]["mean_frame_psnr"]
            - results["mean"]["metrics"]["mean_frame_psnr"]),
        "oracle_psnr_change_vs_mean_db": (
            results["oracle-fill"]["metrics"]["mean_frame_psnr"]
            - results["mean"]["metrics"]["mean_frame_psnr"]),
        "pixel_gap_recovery": pixel_gap_recovery,
        "per_p_chunk_psnr_gain_db": per_chunk_gain,
    }


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_torch_env()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    files, frames = load_window(args)
    i_net, p_net = load_models(args, device)
    freeze_codec(i_net, p_net)
    predictor = SparseMaskedLatentPredictor(PredictorConfig()).to(device)
    zero_init_validation = validate_zero_init_equivalence(
        i_net, p_net, predictor, frames, args,
        output_root / "preflight_zero_init", device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    mean_samples = collect_samples(i_net, p_net, None, frames, args, device)
    initial = latent_recovery(predictor, mean_samples)
    history_a, seconds_a = train_phase(
        predictor, mean_samples, optimizer, args.steps,
        args.nonzero_weight, args.log_every, "mean_rollout")
    after_mean = latent_recovery(predictor, mean_samples)

    refreshed_samples = collect_samples(
        i_net, p_net, predictor, frames, args, device)
    before_refresh = latent_recovery(predictor, refreshed_samples)
    history_b = []
    seconds_b = 0.0
    if args.refresh_steps:
        history_b, seconds_b = train_phase(
            predictor, refreshed_samples, optimizer, args.refresh_steps,
            args.nonzero_weight, args.log_every, "predictor_rollout_refresh")
    final_recovery = latent_recovery(predictor, refreshed_samples)

    training_summary = {
        "initial": initial,
        "after_mean_rollout_training": after_mean,
        "before_rollout_refresh": before_refresh,
        "final": final_recovery,
        "phase_seconds": {
            "mean_rollout": seconds_a,
            "predictor_rollout_refresh": seconds_b,
        },
        "history": history_a + history_b,
    }
    checkpoint_path = output_root / "learned_local_r1_lite_g1.pt"
    checkpoint_sha256 = save_checkpoint(
        predictor, checkpoint_path, args, training_summary)

    predictor.eval()
    target_isolation = validate_target_isolation(predictor, refreshed_samples)
    trajectories, comparison = evaluate(
        i_net, p_net, predictor, checkpoint_path, checkpoint_sha256, frames, args,
        output_root / "trajectories", device)
    replay = trajectories["learned-lite"].get(
        "external_profile_fresh_replay", {})
    gate = {
        "g1_latent_recovery_at_least_90_percent": (
            final_recovery["latent_recovery"] >= 0.90),
        "g1_pixel_gap_recovery_at_least_70_percent": (
            comparison["pixel_gap_recovery"] >= 0.70),
        "both_p_chunks_improve_psnr": all(
            value > 0.0 for value in comparison["per_p_chunk_psnr_gain_db"]),
        "checkpoint_reloaded_tensor_hash_replay_identical": all((
            replay.get("predictor_loaded_from_checkpoint_in_new_instance", False),
            replay.get("checkpoint_sha256_verified", False),
            replay.get(
                "decoded_q_predicted_y_x_hat_feature_hashes_identical", False),
        )),
        "zero_init_two_chunk_mean_equivalence": all((
            zero_init_validation["stream_bytes_identical"],
            zero_init_validation["uint8_rgb_identical"],
            zero_init_validation[
                "decoded_q_predicted_y_x_hat_feature_hashes_identical"],
            zero_init_validation["zero_predictor_executed_for_every_chunk"],
        )),
        "target_isolation": (
            not target_isolation["source_argument_present"]
            and not target_isolation["target_argument_present"]
            and target_isolation[
                "all_predictions_identical_after_unforwarded_target_perturbation"]),
        "codec_frozen": not any(
            parameter.requires_grad for model in (i_net, p_net)
            for parameter in model.parameters()),
        "predictor_is_active_only": all(
            not record["predictor"]["whole_latent_learned_activation"]
            for record in trajectories["learned-lite"]["chunks"]),
    }
    gate["passes_g1_overfit_gate"] = all(gate.values())
    summary = {
        "experiment": "adaptive_chunk_coding_stage_b_learned_predictor_g1_v2",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "purpose": "Capacity and integration sanity check only",
            "generalization_claim_allowed": False,
            "predictor_inference_inputs": [
                "decoded_base_q", "mean_filled_y", "decoded_common_params", "route"],
            "training_label_only": "omitted routed q dequantized to final-y correction",
            "source_or_omitted_y_available_to_predictor": False,
            "codec_frozen": True,
            "route": "content-independent deterministic random 80/256 blocks",
            "stream_profile": "externally-profiled research stream",
            "stream_is_self_describing": False,
            "external_decode_requirements": [
                "predictor checkpoint/profile", "reset interval"],
        },
        "args": vars(args),
        "source": {
            "first_file": str(files[0]),
            "last_file": str(files[-1]),
            "frame_count": len(files),
        },
        "predictor": {
            "profile": "learned-local-r1-lite-g1-overfit-only",
            "architecture": predictor.architecture,
            "parameter_count": predictor.parameter_count,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "training": training_summary,
        "preflight_validation": {
            "zero_init_two_chunk_mean_equivalence": zero_init_validation,
            "target_isolation": target_isolation,
        },
        "trajectories": trajectories,
        "comparison": comparison,
        "gate": gate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "final_latent_recovery": final_recovery["latent_recovery"],
        **comparison,
        **gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
