#!/usr/bin/env python3
"""Held-out REDS G2 gate for the Adaptive Chunk Coding Lite predictor.

The registered operational points are all-Base QP30/31/32, QP32 mean-fill
(C0), and QP32 learned sparse refinement (C1).  Every point writes and is
fresh-decoded from a self-describing ``.d2l`` sequence.  The header serializes
the predictor profile, reset schedule, and checkpoint SHA-256; all header
bytes are included in rate.  The learned checkpoint is a reusable decoder
model and is not charged per video, just as the frozen DCVC-UF weights are not.

The learned predictor receives only decoded Base q, mean-filled y, common
parameters reconstructed from transmitted z and decoder history, and route.
The source-derived routed target is evaluated separately as a non-operational
reference; it is never placed on the R-D-C frontier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.analyze_stage1_oracle_rd import interpolate_log_rate  # noqa: E402
from demo.stage1_multichunk_oracle import (  # noqa: E402
    CHUNK_LENGTH,
    decode_i_stream,
    get_q_params,
    initialize_p_state,
    load_models,
    make_chunk,
    metric_summary,
    model_frame,
    prepare_chunk_latents,
    should_reset,
)
from demo.stage1_token_skipping import (  # noqa: E402
    CONTAINER_HEADER,
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
)
from src.models.video_model_ht import g_ch_z, g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


DEV_IDS = tuple(f"{index:03d}" for index in range(6))
PROFILE_IDS = {
    "all-base": 0,
    "mean": 1,
    "learned-lite-c1": 2,
    "learned-lite-c1-a025": 3,
}
ID_TO_PROFILE = {value: key for key, value in PROFILE_IDS.items()}
LEARNED_PROFILE_ALPHA = {
    "learned-lite-c1": 1.0,
    "learned-lite-c1-a025": 0.25,
}
D2L_MAGIC = b"D2LSEQ01"
D2L_VERSION = 1
# magic, version/profile/block/qp_i/qp_p/i_ec/reset/flags, geometry/counts,
# I-payload length, and the exact reusable predictor checkpoint hash.
D2L_HEADER = struct.Struct("<8s8B4HI32s")
ZERO_SHA256 = bytes(32)
PHASE_A_CHECKPOINT_PROFILE = "stage-b-g2-train-only-sparse-masked-predictor"
PHASE_B_CHECKPOINT_PROFILE = "stage-b-g2-phase-b-rollout-refresh-predictor"
REGISTERED_TRAIN_IDS = tuple(f"{index:03d}" for index in range(1, 25))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Held-out full-rollout G2 evaluation for sparse predictor C1.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--checkpoint",
        default="output/stage_b_predictor_g2_phase_b_seed20260908/best.pt")
    parser.add_argument(
        "--output-dir", default="output/stage_b_predictor_g2_dev_qp32")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--dev-sequences", nargs="+", default=list(DEV_IDS))
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=384)
    parser.add_argument("--crop-y", type=int, default=64)
    parser.add_argument("--qp-baselines", type=int, nargs="+", default=(30, 31, 32))
    parser.add_argument("--qp-route", type=int, default=32)
    parser.add_argument("--latent-block-size", type=int, choices=(2,), default=2)
    parser.add_argument("--skip-blocks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--learned-alpha", type=float, default=1.0)
    parser.add_argument(
        "--calibrated-followup", action="store_true",
        help="Run the preregistered alpha=0.25 follow-up after the G2 diagnostic.")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--allow-smoke-protocol", action="store_true",
        help="Permit a reduced engineering smoke; such a run can never pass G2.")
    parser.add_argument(
        "--include-source-reference", action=argparse.BooleanOptionalAction,
        default=True)
    return parser.parse_args()


def validate_args(args):
    if args.width != 512 or args.height != 512:
        raise ValueError("G2 is frozen at a 512x512 crop")
    if args.crop_x % 64 or args.crop_y % 64:
        raise ValueError("G2 crop must be 64-pixel aligned")
    if args.qp_route != 32 or tuple(args.qp_baselines) != (30, 31, 32):
        raise ValueError("G2 is frozen at routed QP32 and all-Base QP30/31/32")
    if args.latent_block_size != 2 or args.skip_blocks != 64:
        raise ValueError("G2 is frozen at 2x2 blocks and 64/256 Skip")
    if args.frame_count < 2 or args.frame_count > 100:
        raise ValueError("frame-count must be in [2, 100]")
    if args.decode_repeats < 1:
        raise ValueError("decode-repeats must be positive")
    expected_alpha = 0.25 if args.calibrated_followup else 1.0
    if not math.isclose(args.learned_alpha, expected_alpha):
        raise ValueError(
            f"this protocol requires learned-alpha={expected_alpha} when "
            f"calibrated-followup={args.calibrated_followup}")
    invalid = set(args.dev_sequences) - set(DEV_IDS)
    if invalid:
        raise ValueError(f"G2 dev is restricted to REDS val/000..005, got {invalid}")
    if args.max_sequences < 0:
        raise ValueError("max-sequences cannot be negative")
    if len(args.dev_sequences) != len(set(args.dev_sequences)):
        raise ValueError("development sequence IDs must not repeat")
    if args.reset_interval != 32 and not args.allow_smoke_protocol:
        raise ValueError("registered G2 reset interval is exactly 32 frames")
    if not args.allow_smoke_protocol:
        if tuple(args.dev_sequences) != DEV_IDS or args.max_sequences != 0:
            raise ValueError("registered G2 requires val/000..005 exactly once")
        if args.frame_count != 100:
            raise ValueError("registered G2 requires all 100 frames per sequence")
        if (args.crop_x, args.crop_y) != (384, 64):
            raise ValueError("registered G2 crop origin is exactly (384, 64)")
        if args.seed != 20260908 or not math.isclose(args.skip_thres, 0.0):
            raise ValueError("registered G2 fixes seed=20260908 and skip-thres=0")
        if args.decode_repeats != 3:
            raise ValueError("registered G2 requires three decode timing repeats")
        if args.no_lpips or not args.include_source_reference:
            raise ValueError("registered G2 requires LPIPS and source reference")


def learned_profile(args):
    return (
        "learned-lite-c1-a025"
        if args.calibrated_followup else "learned-lite-c1")


def learned_label(args):
    return learned_profile(args) + "-qp32"


def is_learned_profile(profile):
    return profile in LEARNED_PROFILE_ALPHA


def apply_profile_alpha(predicted_y, mean_y, profile):
    alpha = LEARNED_PROFILE_ALPHA[profile]
    return mean_y + alpha * (predicted_y - mean_y)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    """Hash a tensor or an ordered tensor collection in canonical value order."""
    if isinstance(tensor, (list, tuple)):
        digest = hashlib.sha256()
        digest.update(type(tensor).__name__.encode())
        digest.update(str(len(tensor)).encode())
        for index, value in enumerate(tensor):
            digest.update(str(index).encode())
            digest.update(bytes.fromhex(tensor_sha256(value)))
        return digest.hexdigest()
    if not torch.is_tensor(tensor):
        raise TypeError(f"cannot tensor-hash {type(tensor).__name__}")
    canonical = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(str(tuple(canonical.shape)).encode())
    digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def raw_state_hashes(decoded_q, predicted_y, x_hat, feature, p_net=None):
    result = {
        "decoded_q_sha256": tensor_sha256(decoded_q),
        "predicted_y_sha256": tensor_sha256(predicted_y),
        "x_hat_sha256": tensor_sha256(x_hat),
        "reconstructed_feature_sha256": tensor_sha256(feature),
    }
    if p_net is not None:
        result.update({
            "propagated_ref_feature_sha256": tensor_sha256(p_net.ref_feature),
            "propagated_memory_sha256": (
                None if p_net.memory is None else tensor_sha256(p_net.memory)),
            "propagated_ctx_sha256": (
                None if p_net.ctx is None else tensor_sha256(p_net.ctx)),
        })
    return result


def load_predictor_checkpoint(path: Path, device, expected_sha256=None):
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"predictor checkpoint hash mismatch: {actual_sha256} != {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture")
    if architecture is None and "predictor" in payload:
        architecture = payload["predictor"].get("architecture")
    if architecture is None:
        raise ValueError("checkpoint does not serialize predictor architecture")
    profile = str(payload.get("profile", payload.get("predictor_profile", "")))
    if "g1" in profile.lower() or "overfit" in profile.lower():
        raise ValueError("G1 overfit checkpoint is forbidden in the G2 evaluation")
    state_dict = payload.get("state_dict", payload.get("model_state_dict"))
    if state_dict is None:
        raise ValueError("checkpoint does not contain a predictor state_dict")
    predictor = SparseMaskedLatentPredictor(
        PredictorConfig(**architecture)).to(device)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval().requires_grad_(False)
    return predictor, payload, actual_sha256


def validate_checkpoint_provenance(payload, require_phase_b):
    if payload.get("project") != "Adaptive Chunk Coding":
        raise ValueError("checkpoint project identity is invalid")
    profile = str(payload.get("profile", ""))
    if profile not in (PHASE_A_CHECKPOINT_PROFILE, PHASE_B_CHECKPOINT_PROFILE):
        raise ValueError(f"unregistered G2 predictor profile: {profile!r}")
    if require_phase_b and profile != PHASE_B_CHECKPOINT_PROFILE:
        raise ValueError("formal G2 evaluation requires the train-only Phase-B checkpoint")
    scope = payload.get("scientific_scope", {})
    training = payload.get("training", {})
    if training.get("seed") != 20260908:
        raise ValueError("checkpoint was not trained with the registered seed")
    if payload.get("architecture") != PredictorConfig().__dict__:
        raise ValueError("checkpoint architecture is not the registered Lite C1 tier")
    if not scope.get("train_only") or scope.get("validation_or_sealed_data_read"):
        raise ValueError("checkpoint lacks auditable train-only provenance")
    data = payload.get("data", {})
    if data.get("split") != "REDS train_sharp/001..024":
        raise ValueError("checkpoint data split is not train/001..024")
    if tuple(data.get("sequence_ids", ())) != REGISTERED_TRAIN_IDS:
        raise ValueError("checkpoint train sequence provenance is incomplete")
    record = {
        "profile": profile,
        "training_phase": scope.get("training_phase"),
        "train_only": True,
        "validation_or_sealed_data_read": False,
        "train_sequence_ids": list(REGISTERED_TRAIN_IDS),
    }
    if profile == PHASE_A_CHECKPOINT_PROFILE:
        if data.get("trajectory") != "mean_fill":
            raise ValueError("Phase-A checkpoint was not fit on mean-fill cache")
        record["mean_manifest_sha256"] = data.get("manifest_sha256")
        return record

    if data.get("phase") != "phase_b_rollout_refresh":
        raise ValueError("Phase-B checkpoint lacks rollout-refresh phase marker")
    if (training.get("steps_requested") != 3000
            or not math.isclose(training.get("learning_rate", -1.0), 1e-4)):
        raise ValueError("Phase-B checkpoint uses an unregistered training schedule")
    if data.get("sample_count") != 192:
        raise ValueError("Phase-B checkpoint must use 192 paired train samples")
    mean_cache = data.get("mean_chunk0_cache", {})
    refresh_cache = data.get("refresh_chunk1_cache", {})
    initialization = payload.get("initialization", {})
    if (mean_cache.get("split") != "REDS train_sharp/001..024"
            or mean_cache.get("trajectory") != "mean_fill"
            or tuple(mean_cache.get("sequence_ids", ())) != REGISTERED_TRAIN_IDS):
        raise ValueError("Phase-B mean chunk-0 cache provenance is invalid")
    if (refresh_cache.get("split") != "REDS train_sharp/001..024"
            or refresh_cache.get("trajectory")
            != "phase_a_predictor_chunk0_rollout"):
        raise ValueError("Phase-B refreshed chunk-1 cache provenance is invalid")
    if initialization.get("profile") != PHASE_A_CHECKPOINT_PROFILE:
        raise ValueError("Phase-B initializer is not a registered Phase-A model")
    if (initialization.get("file_sha256")
            != refresh_cache.get("phase_a_checkpoint_file_sha256")):
        raise ValueError("Phase-B initializer and refresh cache hashes differ")
    if (mean_cache.get("manifest_sha256")
            != refresh_cache.get("source_mean_manifest_sha256")):
        raise ValueError("Phase-B mean and refresh cache manifests differ")
    record.update({
        "mean_manifest_sha256": mean_cache.get("manifest_sha256"),
        "refresh_manifest_sha256": refresh_cache.get("manifest_sha256"),
        "phase_a_checkpoint_sha256": initialization.get("file_sha256"),
    })
    if not all(record[key] for key in (
            "mean_manifest_sha256", "refresh_manifest_sha256",
            "phase_a_checkpoint_sha256")):
        raise ValueError("Phase-B checkpoint is missing required provenance hashes")
    return record


def load_frames(args, sequence):
    root = Path(args.data_root) / "val_sharp" / sequence
    paths = sorted(root.glob("*.png"))[:args.frame_count]
    if len(paths) != args.frame_count:
        raise ValueError(f"{root} has {len(paths)} requested frames")
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[
            args.crop_y:args.crop_y + args.height,
            args.crop_x:args.crop_x + args.width,
        ]
        if crop.shape != (args.height, args.width, 3):
            raise ValueError(f"crop falls outside {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return paths, frames


def stable_route(args, chunk_index, grid_shape):
    parent_skip_blocks = int(getattr(
        args, "route_parent_skip_blocks", args.skip_blocks))
    if not 0 <= args.skip_blocks <= parent_skip_blocks <= int(np.prod(grid_shape)):
        raise ValueError(
            "route requires 0 <= skip_blocks <= route_parent_skip_blocks "
            "<= number of spatial blocks")
    payload = (
        f"{args.seed}:g2-dev-shared:{chunk_index}:"
        f"{grid_shape[0]}:{grid_shape[1]}:{parent_skip_blocks}").encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    flat = np.zeros(int(np.prod(grid_shape)), dtype=np.bool_)
    ordered_parent = rng.choice(
        flat.size, parent_skip_blocks, replace=False)
    flat[ordered_parent[:args.skip_blocks]] = True
    return flat.reshape(grid_shape)


def route_sha256(route):
    return hashlib.sha256(np.packbits(route.reshape(-1)).tobytes()).hexdigest()


def write_d2l(path, profile, block_size, qp_i, qp_p, i_ec, reset_interval,
              height, width, frame_count, i_stream, chunks, checkpoint_sha256=None):
    if profile not in PROFILE_IDS:
        raise ValueError(profile)
    hash_bytes = (
        bytes.fromhex(checkpoint_sha256)
        if is_learned_profile(profile) else ZERO_SHA256)
    header = D2L_HEADER.pack(
        D2L_MAGIC, D2L_VERSION, PROFILE_IDS[profile], block_size,
        qp_i, qp_p, i_ec, reset_interval, 0,
        height, width, frame_count, len(chunks), len(i_stream), hash_bytes)
    data = bytearray(header)
    data.extend(i_stream)
    for chunk in chunks:
        data.extend(CHUNK_LENGTH.pack(len(chunk)))
        data.extend(chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def read_d2l(path):
    data = path.read_bytes()
    if len(data) < D2L_HEADER.size:
        raise ValueError("truncated D2L sequence")
    fields = D2L_HEADER.unpack(data[:D2L_HEADER.size])
    (magic, version, profile_id, block_size, qp_i, qp_p, i_ec,
     reset_interval, flags, height, width, frame_count, chunk_count,
     i_len, checkpoint_hash) = fields
    if magic != D2L_MAGIC or version != D2L_VERSION:
        raise ValueError("unsupported D2L sequence")
    if profile_id not in ID_TO_PROFILE or flags != 0:
        raise ValueError("unsupported D2L predictor profile or flags")
    profile = ID_TO_PROFILE[profile_id]
    if is_learned_profile(profile) and checkpoint_hash == ZERO_SHA256:
        raise ValueError("learned D2L sequence has no checkpoint hash")
    if not is_learned_profile(profile) and checkpoint_hash != ZERO_SHA256:
        raise ValueError("non-learned D2L sequence unexpectedly binds a checkpoint")
    pos = D2L_HEADER.size
    i_stream = data[pos:pos + i_len]
    if len(i_stream) != i_len:
        raise ValueError("truncated D2L I payload")
    pos += i_len
    chunks = []
    for _ in range(chunk_count):
        if pos + CHUNK_LENGTH.size > len(data):
            raise ValueError("truncated D2L chunk length")
        chunk_len = CHUNK_LENGTH.unpack(data[pos:pos + CHUNK_LENGTH.size])[0]
        pos += CHUNK_LENGTH.size
        chunk = data[pos:pos + chunk_len]
        if len(chunk) != chunk_len:
            raise ValueError("truncated D2L chunk")
        chunks.append(chunk)
        pos += chunk_len
    if pos != len(data):
        raise ValueError("D2L sequence length mismatch")
    return {
        "profile": profile,
        "profile_id": profile_id,
        "block_size": block_size,
        "qp_i": qp_i,
        "qp_p": qp_p,
        "i_ec": i_ec,
        "reset_interval": reset_interval,
        "height": height,
        "width": width,
        "frame_count": frame_count,
        "checkpoint_sha256": checkpoint_hash.hex(),
        "i_stream": i_stream,
        "chunks": chunks,
    }


def d2l_stream_breakdown(path):
    sequence = read_d2l(path)
    parsed = [
        parse_container(chunk, sequence["block_size"])
        for chunk in sequence["chunks"]
    ]
    result = {
        "sequence_header_and_chunk_lengths_bytes": (
            D2L_HEADER.size + CHUNK_LENGTH.size * len(parsed)),
        "serialized_fields_within_sequence_header": {
            "predictor_profile_bytes": 1,
            "reset_schedule_bytes": 1,
            "checkpoint_hash_bytes": 32,
        },
        "i_payload_bytes": len(sequence["i_stream"]),
        "p_container_header_bytes": CONTAINER_HEADER.size * len(parsed),
        "global_z_bytes": sum(len(item["global"]) for item in parsed),
        "base_y_bytes": sum(len(item["base"]) for item in parsed),
        "route_bytes": sum(len(item["route"]) for item in parsed),
        "residual_bytes": sum(len(item["residual"]) for item in parsed),
        "chunk_count": len(parsed),
    }
    result["container_bytes"] = (
        result["sequence_header_and_chunk_lengths_bytes"]
        + result["p_container_header_bytes"])
    result["total_bytes"] = (
        result["container_bytes"] + result["i_payload_bytes"]
        + result["global_z_bytes"] + result["base_y_bytes"]
        + result["route_bytes"] + result["residual_bytes"])
    if result["total_bytes"] != path.stat().st_size:
        raise RuntimeError("D2L component bytes do not equal file size")
    return result


def decode_chunk_inputs(p_net, payload, block_size, q_feature):
    parsed = parse_container(payload, block_size)
    if parsed["z_ec"] is None:
        raise ValueError("G2 requires chunk container v2 with serialized z EC")
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


def source_route_target(p_net, prepared, decoded_q, skip, block_size):
    keep = expand_keep_mask(skip, block_size, decoded_q.shape, decoded_q.device)
    routed_q, q_source, _, _ = y_prior_steps(
        p_net, prepared["common_params"], keep, source_y=prepared["y"])
    if not torch.equal(routed_q, decoded_q):
        raise RuntimeError("source reference route differs from decoded Base q")
    _, q_decoder, _, _ = p_net.separate_prior_video(prepared["common_params"])
    return q_source * (~keep) * q_decoder


@torch.inference_mode()
def encode_trajectory(label, profile, qp, i_net, p_net, predictor, checkpoint_sha,
                      frames, args, output_dir, device, operational=True,
                      fixed_routes=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    i_x = (model_frame(frames[0], device) - 0.5).to(
        memory_format=torch.channels_last)
    qp_i = int(getattr(args, "qp_i_override", qp))
    qp_p = int(qp)
    i_encoded = i_net.compress(i_x, qp_i, 0, 0)
    i_stream = bytes(i_encoded["bit_stream"])
    i_ec = int(i_encoded["ec_parallel"])
    i_hat = decode_i_stream(
        i_net, i_stream, i_ec, qp_i, args.height, args.width)
    initialize_p_state(p_net, i_hat)
    recon = [rgb_from_recon(i_hat)]
    raw_hashes = [{"i_hat_sha256": tensor_sha256(i_hat)}]
    records = []
    payloads = []
    first_index = 1
    chunk_index = 0
    encode_started = time.perf_counter()
    while first_index < len(frames):
        chunk, _, valid_count = make_chunk(frames, first_index, device)
        prepared = prepare_chunk_latents(p_net, chunk, qp_p)
        grid_shape = (
            prepared["y"].shape[-2] // args.latent_block_size,
            prepared["y"].shape[-1] // args.latent_block_size)
        if profile == "all-base":
            skip = np.zeros(grid_shape, dtype=np.bool_)
        elif fixed_routes is None:
            skip = stable_route(args, chunk_index, grid_shape)
        else:
            if chunk_index >= len(fixed_routes):
                raise ValueError(f"{label}: missing fixed route for chunk {chunk_index}")
            skip = np.asarray(fixed_routes[chunk_index], dtype=np.bool_)
            if skip.shape != grid_shape:
                raise ValueError(
                    f"{label}: fixed route {chunk_index} shape {skip.shape}, "
                    f"expected {grid_shape}")
        encoded = encode_y(
            p_net, prepared["y"], prepared["common_params"], skip,
            args.latent_block_size)
        route = b"" if not np.any(skip) else build_route_section(
            skip, args.latent_block_size)
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.d1s"
        write_container(
            chunk_path, qp_p, args.height, args.width, prepared["y"].shape,
            prepared["global_stream"], route, encoded.stream, b"",
            encoded.ec_parallel, all_base=not np.any(skip),
            z_ec_parallel=prepared["z_ec"])
        payload = chunk_path.read_bytes()
        _, q_feature, q_decoder = get_q_params(p_net, qp_p)
        parsed, decoded_q, mean_y, common_params = decode_chunk_inputs(
            p_net, payload, args.latent_block_size, q_feature)
        if not torch.equal(decoded_q, encoded.q_dense):
            raise RuntimeError(f"{label} chunk {chunk_index} Base rANS mismatch")
        predictor_profile = None
        if profile in ("all-base", "mean"):
            predicted_y = mean_y
        elif is_learned_profile(profile):
            predicted_y, predictor_profile = predictor.apply(
                decoded_q, mean_y, common_params, skip)
            predicted_y = apply_profile_alpha(
                predicted_y.to(dtype=mean_y.dtype), mean_y, profile)
            predictor_profile["output_alpha"] = LEARNED_PROFILE_ALPHA[profile]
        elif profile == "source-route-reference":
            predicted_y = mean_y + source_route_target(
                p_net, prepared, decoded_q, skip, args.latent_block_size)
        else:
            raise ValueError(profile)
        keep = expand_keep_mask(
            skip, args.latent_block_size, mean_y.shape, mean_y.device)
        if not torch.equal(predicted_y[keep], mean_y[keep]):
            raise RuntimeError(f"{label} modified transmitted Base latent")
        x_hat, feature = p_net.get_recon_and_feature(
            predicted_y, p_net.ctx, q_decoder)
        recon.extend([
            rgb_from_recon(frame[:, :, :args.height, :args.width])
            for frame in x_hat[:valid_count]
        ])
        reset = should_reset(chunk_index, args.reset_interval)
        p_net.set_ref_feature(feature, reset)
        state = raw_state_hashes(
            decoded_q, predicted_y, x_hat, feature, p_net=p_net)
        state["chunk_index"] = chunk_index
        raw_hashes.append(state)
        records.append({
            "chunk_index": chunk_index,
            "valid_frames": valid_count,
            "reset_after_chunk": reset,
            "route_sha256": route_sha256(skip),
            "skipped_blocks": int(skip.sum()),
            "chunk_bytes": len(payload),
            "container_header_bytes": CONTAINER_HEADER.size,
            "global_z_bytes": len(parsed["global"]),
            "base_y_bytes": len(parsed["base"]),
            "route_bytes": len(parsed["route"]),
            "residual_bytes": len(parsed["residual"]),
            "predictor_activation": predictor_profile,
        })
        payloads.append(payload)
        first_index += valid_count
        chunk_index += 1
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_started
    sequence_path = None
    if operational:
        sequence_path = output_dir / "sequence.d2l"
        write_d2l(
            sequence_path, profile, args.latent_block_size, qp_i, qp_p, i_ec,
            args.reset_interval, args.height, args.width, len(frames), i_stream,
            payloads, checkpoint_sha if is_learned_profile(profile) else None)
    return {
        "label": label,
        "profile": profile,
        "qp_i": qp_i,
        "qp_p": qp_p,
        "operational": operational,
        "sequence_path": sequence_path,
        "recon": recon,
        "raw_hashes": raw_hashes,
        "chunks": records,
        "encode_wall_seconds": encode_seconds,
        "component_payload_bytes_without_sequence_wrapper": (
            len(i_stream) + sum(len(payload) for payload in payloads)),
    }


@torch.inference_mode()
def decode_d2l_once(i_net, p_net, sequence_path, checkpoint_path, device):
    sequence = read_d2l(sequence_path)
    predictor = None
    checkpoint_payload = None
    if is_learned_profile(sequence["profile"]):
        predictor, checkpoint_payload, _ = load_predictor_checkpoint(
            checkpoint_path, device, sequence["checkpoint_sha256"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    i_hat = decode_i_stream(
        i_net, sequence["i_stream"], sequence["i_ec"], sequence["qp_i"],
        sequence["height"], sequence["width"])
    initialize_p_state(p_net, i_hat)
    recon = [rgb_from_recon(i_hat)]
    torch.cuda.synchronize(device)
    hash_started = time.perf_counter()
    raw_hashes = [{"i_hat_sha256": tensor_sha256(i_hat)}]
    validation_hash_seconds = time.perf_counter() - hash_started
    profiles = []
    event_pairs = []
    remaining = sequence["frame_count"] - 1
    for chunk_index, payload in enumerate(sequence["chunks"]):
        p_net.apply_feature_adaptor()
        _, q_feature, q_decoder = get_q_params(p_net, sequence["qp_p"])
        parsed, decoded_q, mean_y, common_params = decode_chunk_inputs(
            p_net, payload, sequence["block_size"], q_feature)
        if sequence["profile"] in ("all-base", "mean"):
            predicted_y = mean_y
            profile = {
                "skipped_blocks": int(parsed["skip_blocks"].sum()),
                "linear_macs": 0,
                "learned_module_input_shapes": {},
            }
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            predicted_y, profile = predictor.apply(
                decoded_q, mean_y, common_params, parsed["skip_blocks"])
            end_event.record()
            event_pairs.append((len(profiles), start_event, end_event))
            predicted_y = apply_profile_alpha(
                predicted_y.to(dtype=mean_y.dtype), mean_y,
                sequence["profile"])
            profile["output_alpha"] = LEARNED_PROFILE_ALPHA[sequence["profile"]]
        keep = expand_keep_mask(
            parsed["skip_blocks"], sequence["block_size"],
            mean_y.shape, mean_y.device)
        if not torch.equal(predicted_y[keep], mean_y[keep]):
            raise RuntimeError("fresh D2L decode modified Base latent")
        x_hat, feature = p_net.get_recon_and_feature(
            predicted_y, p_net.ctx, q_decoder)
        valid_count = min(g_frame_delay, remaining)
        recon.extend([
            rgb_from_recon(frame[:, :, :sequence['height'], :sequence['width']])
            for frame in x_hat[:valid_count]
        ])
        remaining -= valid_count
        p_net.set_ref_feature(
            feature, should_reset(chunk_index, sequence["reset_interval"]))
        torch.cuda.synchronize(device)
        hash_started = time.perf_counter()
        state = raw_state_hashes(
            decoded_q, predicted_y, x_hat, feature, p_net=p_net)
        validation_hash_seconds += time.perf_counter() - hash_started
        state["chunk_index"] = chunk_index
        raw_hashes.append(state)
        profiles.append(profile)
    if remaining != 0:
        raise RuntimeError("fresh D2L decode frame count mismatch")
    torch.cuda.synchronize(device)
    decode_seconds_with_validation = time.perf_counter() - started
    decode_seconds = decode_seconds_with_validation - validation_hash_seconds
    for index, start_event, end_event in event_pairs:
        profiles[index]["latency_ms"] = float(start_event.elapsed_time(end_event))
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    del predictor, checkpoint_payload
    return (
        recon, raw_hashes, profiles, decode_seconds, peak_memory,
        validation_hash_seconds)


def temporal_delta_mae(original, recon):
    if len(original) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(original)):
        source_delta = (
            original[index].astype(np.float32)
            - original[index - 1].astype(np.float32))
        recon_delta = (
            recon[index].astype(np.float32)
            - recon[index - 1].astype(np.float32))
        total += float(np.mean(np.abs(source_delta - recon_delta)))
    return total / (len(original) - 1)


def rgb_mse(original, recon):
    squared = sum(np.sum(
        (source.astype(np.float64) - decoded.astype(np.float64)) ** 2)
        for source, decoded in zip(original, recon))
    return float(squared / (len(original) * original[0].size))


class LPIPSAlex:
    def __init__(self, device):
        import lpips
        self.device = device
        # Keep LPIPS on CPU between metric calls so it is absent from decoder
        # timing and CUDA peak-memory measurements.
        self.model = lpips.LPIPS(net="alex", verbose=False).eval().cpu()

    @torch.inference_mode()
    def __call__(self, original, recon, batch_size=8):
        self.model.to(self.device)
        values = []
        for start in range(0, len(original), batch_size):
            source = torch.from_numpy(np.stack(
                original[start:start + batch_size])).to(self.device).float()
            decoded = torch.from_numpy(np.stack(
                recon[start:start + batch_size])).to(self.device).float()
            values.extend(self.model(
                source / 127.5 - 1.0,
                decoded / 127.5 - 1.0,
            ).reshape(-1).detach().cpu().tolist())
        result = float(np.mean(values))
        self.model.cpu()
        torch.cuda.empty_cache()
        return result


def compute_metrics(frames, recon, records, args, lpips_metric):
    metrics = metric_summary(frames, recon, args.width, args.height, records)
    metrics["rgb_mse_level"] = rgb_mse(frames, recon)
    metrics["temporal_delta_mae_rgb_level"] = temporal_delta_mae(frames, recon)
    if lpips_metric is not None:
        metrics["lpips_alex"] = lpips_metric(frames, recon)
    return metrics


def summarize_decode(profiles_by_repeat, seconds, peaks, hash_seconds, frame_count):
    all_profiles = [profile for repeat in profiles_by_repeat for profile in repeat]
    latencies = [
        profile["latency_ms"] for profile in all_profiles
        if "latency_ms" in profile
    ]
    learned_shapes = [
        profile.get("learned_module_input_shapes", {})
        for profile in all_profiles if profile.get("learned_module_input_shapes")
    ]
    return {
        "decode_repeats": len(seconds),
        "decode_wall_seconds_median": float(np.median(seconds)),
        "decode_wall_seconds_all": seconds,
        "validation_tensor_hash_wall_seconds_all": hash_seconds,
        "decode_throughput_fps": frame_count / float(np.median(seconds)),
        "predictor_latency_ms_p50": float(np.median(latencies)) if latencies else 0.0,
        "predictor_latency_ms_p95": (
            float(np.percentile(latencies, 95)) if latencies else 0.0),
        "predictor_latency_ms_all": latencies,
        "predictor_latency_sample_count": len(latencies),
        "linear_macs_per_chunk": (
            int(all_profiles[0].get("linear_macs", 0)) if all_profiles else 0),
        "learned_module_input_shapes_observed": learned_shapes[0] if learned_shapes else {},
        "peak_cuda_allocated_bytes_max": int(max(peaks)),
    }


@torch.inference_mode()
def run_operational_mode(label, profile, qp, i_net, p_net, checkpoint_path,
                         checkpoint_sha, frames, args, output_dir, device,
                         lpips_metric, fixed_routes=None):
    predictor = None
    if is_learned_profile(profile):
        predictor, _, _ = load_predictor_checkpoint(
            checkpoint_path, device, checkpoint_sha)
    encoded = encode_trajectory(
        label, profile, qp, i_net, p_net, predictor, checkpoint_sha,
        frames, args, output_dir, device, operational=True,
        fixed_routes=fixed_routes)
    del predictor
    repeated_profiles = []
    repeated_seconds = []
    repeated_peaks = []
    repeated_hash_seconds = []
    reference_recon = None
    reference_raw = None
    for repeat in range(args.decode_repeats):
        recon, raw_hashes, profiles, seconds, peak, hash_seconds = decode_d2l_once(
            i_net, p_net, encoded["sequence_path"], checkpoint_path, device)
        if repeat == 0:
            reference_recon = recon
            reference_raw = raw_hashes
        elif raw_hashes != reference_raw:
            raise RuntimeError(f"{label} repeated raw decode hashes differ")
        repeated_profiles.append(profiles)
        repeated_seconds.append(seconds)
        repeated_peaks.append(peak)
        repeated_hash_seconds.append(hash_seconds)
    if encoded["raw_hashes"] != reference_raw:
        raise RuntimeError(f"{label} encoding trajectory and fresh raw decode differ")
    if any(not np.array_equal(a, b) for a, b in zip(
            encoded["recon"], reference_recon)):
        raise RuntimeError(f"{label} encoding trajectory and fresh RGB decode differ")
    stream = d2l_stream_breakdown(encoded["sequence_path"])
    stream["total_bpp"] = (
        stream["total_bytes"] * 8
        / (len(frames) * args.width * args.height))
    metrics = compute_metrics(
        frames, reference_recon, encoded["chunks"], args, lpips_metric)
    compute = summarize_decode(
        repeated_profiles, repeated_seconds, repeated_peaks,
        repeated_hash_seconds, len(frames))
    result = {
        "label": label,
        "profile": profile,
        "profile_id_serialized": PROFILE_IDS[profile],
        "qp_i": encoded["qp_i"],
        "qp_p": encoded["qp_p"],
        "operational": True,
        "stream": stream,
        "metrics": metrics,
        "compute": compute,
        "encode_wall_seconds": encoded["encode_wall_seconds"],
        "chunks": encoded["chunks"],
        "validation": {
            "full_sequence_fresh_decode": True,
            "new_predictor_instance_loaded_from_disk": (
                is_learned_profile(profile)),
            "raw_tensor_hashes_match_encoding_trajectory": True,
            "repeated_raw_tensor_hashes_identical": True,
            "profile_reset_and_checkpoint_hash_serialized": True,
            "source_or_omitted_y_available_to_predictor": False,
            "routed_state_updated_after_chunk": profile != "all-base",
            "routed_reconstruction_propagated": (
                profile != "all-base" and len(encoded["chunks"]) > 1),
        },
    }
    return result


@torch.inference_mode()
def run_source_reference(i_net, p_net, frames, args, output_dir, device,
                         lpips_metric=None):
    encoded = encode_trajectory(
        "source-route-reference", "source-route-reference", args.qp_route,
        i_net, p_net, None, None, frames, args, output_dir, device,
        operational=False)
    metrics = compute_metrics(
        frames, encoded["recon"], encoded["chunks"], args, lpips_metric)
    return {
        "label": "source-route-reference",
        "operational": False,
        "eligible_for_rdc_frontier": False,
        "reason": (
            "Uses omitted source q as a route-relative diagnostic target; no "
            "source-free decoder can reproduce this point."),
        "metrics": metrics,
        "component_payload_bytes_without_sequence_wrapper": encoded[
            "component_payload_bytes_without_sequence_wrapper"],
        "encode_wall_seconds": encoded["encode_wall_seconds"],
        "chunks": encoded["chunks"],
    }


def run_sequence(sequence, i_net, p_net, checkpoint_path, checkpoint_sha,
                 args, device, lpips_metric):
    paths, frames = load_frames(args, sequence)
    sequence_root = Path(args.output_dir) / sequence
    results = []
    for qp in args.qp_baselines:
        label = f"all-base-qp{qp}"
        print(json.dumps({"sequence": sequence, "running": label}), flush=True)
        results.append(run_operational_mode(
            label, "all-base", qp, i_net, p_net, checkpoint_path,
            checkpoint_sha, frames, args, sequence_root / label,
            device, lpips_metric))
    selected_learned_profile = learned_profile(args)
    selected_learned_label = learned_label(args)
    for label, profile in (("mean-c0-qp32", "mean"),
                           (selected_learned_label, selected_learned_profile)):
        print(json.dumps({"sequence": sequence, "running": label}), flush=True)
        results.append(run_operational_mode(
            label, profile, args.qp_route, i_net, p_net, checkpoint_path,
            checkpoint_sha, frames, args, sequence_root / label,
            device, lpips_metric))
    source_reference = None
    if args.include_source_reference:
        print(json.dumps({
            "sequence": sequence, "running": "source-route-reference"}), flush=True)
        source_reference = run_source_reference(
            i_net, p_net, frames, args,
            sequence_root / "source-route-reference", device)
    by_label = {item["label"]: item for item in results}
    mean = by_label["mean-c0-qp32"]
    learned = by_label[selected_learned_label]
    mean_routes = [
        (record["route_sha256"], record["skipped_blocks"])
        for record in mean["chunks"]]
    learned_routes = [
        (record["route_sha256"], record["skipped_blocks"])
        for record in learned["chunks"]]
    if learned_routes != mean_routes:
        raise RuntimeError("mean and learned trajectories used different routes")
    if any(count != args.skip_blocks for _, count in mean_routes):
        raise RuntimeError("a routed G2 chunk does not contain exactly 64 Skip blocks")
    if source_reference is not None:
        reference_routes = [
            (record["route_sha256"], record["skipped_blocks"])
            for record in source_reference["chunks"]]
        if reference_routes != mean_routes:
            raise RuntimeError("source reference and operational routes differ")

    baseline_points = [
        (qp, by_label[f"all-base-qp{qp}"]["metrics"]["mean_frame_psnr"],
         by_label[f"all-base-qp{qp}"]["stream"]["total_bpp"])
        for qp in args.qp_baselines]
    interpolation = interpolate_log_rate(
        baseline_points, learned["metrics"]["mean_frame_psnr"])
    if interpolation is None:
        per_sequence_matched_rd = {
            "status": "outside_all_base_quality_range"}
    else:
        matched_bpp, lower_qp, upper_qp, weight = interpolation
        per_sequence_matched_rd = {
            "status": "ok",
            "matched_all_base_bpp": matched_bpp,
            "learned_rate_change_percent": 100.0 * (
                learned["stream"]["total_bpp"] / matched_bpp - 1.0),
            "lower_qp": lower_qp,
            "upper_qp": upper_qp,
            "interpolation_weight": weight,
        }
    comparison = {
        "learned_psnr_change_vs_mean_db": (
            learned["metrics"]["mean_frame_psnr"]
            - mean["metrics"]["mean_frame_psnr"]),
        "learned_aggregate_psnr_change_vs_mean_db": (
            learned["metrics"]["aggregate_psnr"]
            - mean["metrics"]["aggregate_psnr"]),
        "learned_bytes_change_vs_mean": (
            learned["stream"]["total_bytes"] - mean["stream"]["total_bytes"]),
        "learned_rate_change_vs_mean_percent": 100.0 * (
            learned["stream"]["total_bytes"] / mean["stream"]["total_bytes"] - 1.0),
        "learned_lpips_change_vs_mean": (
            learned["metrics"].get("lpips_alex", float("nan"))
            - mean["metrics"].get("lpips_alex", float("nan"))),
        "learned_temporal_change_vs_mean": (
            learned["metrics"]["temporal_delta_mae_rgb_level"]
            - mean["metrics"]["temporal_delta_mae_rgb_level"]),
        "matched_rd_mean_frame_psnr": per_sequence_matched_rd,
    }
    summary = {
        "sequence": sequence,
        "split": "REDS val_sharp development 000..005",
        "first_source_file": str(paths[0]),
        "last_source_file": str(paths[-1]),
        "frame_count": len(frames),
        "crop_xy": [args.crop_x, args.crop_y],
        "crop_size": [args.width, args.height],
        "route_schedule_sha256": [
            item["route_sha256"]
            for item in mean["chunks"]
        ],
        "operational_points": results,
        "source_route_reference": source_reference,
        "comparison": comparison,
    }
    sequence_root.mkdir(parents=True, exist_ok=True)
    (sequence_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del frames
    return summary


def aggregate_results(summaries, args, checkpoint_path, checkpoint_sha, payload,
                      checkpoint_provenance):
    registered_route_schedule = summaries[0]["route_schedule_sha256"]
    for sequence in summaries:
        if sequence["route_schedule_sha256"] != registered_route_schedule:
            raise RuntimeError("G2 route schedule changed across development content")
        expected_chunks = math.ceil((sequence["frame_count"] - 1) / g_frame_delay)
        expected_last = (sequence["frame_count"] - 1) % g_frame_delay or g_frame_delay
        for point in sequence["operational_points"]:
            chunk_records = point["chunks"]
            if len(chunk_records) != expected_chunks:
                raise RuntimeError("G2 development mode has the wrong chunk count")
            expected_valid = [g_frame_delay] * expected_chunks
            expected_valid[-1] = expected_last
            if [record["valid_frames"] for record in chunk_records] != expected_valid:
                raise RuntimeError("G2 mode has a wrong valid/padded frame schedule")
            for record in chunk_records:
                expected_reset = should_reset(
                    record["chunk_index"], args.reset_interval)
                if record["reset_after_chunk"] != expected_reset:
                    raise RuntimeError(
                        "G2 reset schedule differs from the registered rule")
    labels = [item["label"] for item in summaries[0]["operational_points"]]
    total_frames = sum(item["frame_count"] for item in summaries)
    total_pixels = total_frames * args.width * args.height
    aggregate = []
    for label in labels:
        points = [
            next(point for point in sequence["operational_points"]
                 if point["label"] == label)
            for sequence in summaries
        ]
        total_bytes = sum(point["stream"]["total_bytes"] for point in points)
        frame_weighted_psnr = sum(
            point["metrics"]["mean_frame_psnr"] * sequence["frame_count"]
            for point, sequence in zip(points, summaries)) / total_frames
        mse = sum(
            point["metrics"]["rgb_mse_level"] * sequence["frame_count"]
            for point, sequence in zip(points, summaries)) / total_frames
        temporal_weight = sum(sequence["frame_count"] - 1 for sequence in summaries)
        temporal = sum(
            point["metrics"]["temporal_delta_mae_rgb_level"]
            * (sequence["frame_count"] - 1)
            for point, sequence in zip(points, summaries)) / temporal_weight
        lpips_values = [point["metrics"].get("lpips_alex") for point in points]
        lpips = None if any(value is None for value in lpips_values) else sum(
            value * sequence["frame_count"]
            for value, sequence in zip(lpips_values, summaries)) / total_frames
        pooled_predictor_latencies = [
            value for point in points
            for value in point["compute"]["predictor_latency_ms_all"]]
        predictor_ms = (
            float(np.median(pooled_predictor_latencies))
            if pooled_predictor_latencies else 0.0)
        predictor_p95 = (
            float(np.percentile(pooled_predictor_latencies, 95))
            if pooled_predictor_latencies else 0.0)
        repeat_count = min(
            len(point["compute"]["decode_wall_seconds_all"])
            for point in points)
        pooled_decode_seconds = [
            sum(point["compute"]["decode_wall_seconds_all"][repeat]
                for point in points)
            for repeat in range(repeat_count)]
        median_decode_seconds = float(np.median(pooled_decode_seconds))
        aggregate.append({
            "label": label,
            "profile": points[0]["profile"],
            "total_bytes": total_bytes,
            "total_bpp": total_bytes * 8 / total_pixels,
            "frame_weighted_mean_psnr": frame_weighted_psnr,
            "aggregate_psnr": (
                float("inf") if mse == 0
                else 10.0 * math.log10(255.0 * 255.0 / mse)),
            "rgb_mse_level": mse,
            "frame_weighted_lpips_alex": lpips,
            "transition_weighted_temporal_delta_mae": temporal,
            "predictor_latency_ms_per_chunk": predictor_ms,
            "predictor_latency_ms_per_chunk_p95": predictor_p95,
            "linear_macs_per_chunk": points[0]["compute"]["linear_macs_per_chunk"],
            "decode_wall_seconds_by_aligned_repeat": pooled_decode_seconds,
            "decode_throughput_fps": total_frames / median_decode_seconds,
            "system_decode_ms_per_frame": 1000.0 * median_decode_seconds / total_frames,
            "peak_cuda_allocated_bytes_max": max(
                point["compute"]["peak_cuda_allocated_bytes_max"] for point in points),
        })
    by_label = {item["label"]: item for item in aggregate}
    mean = by_label["mean-c0-qp32"]
    selected_learned_label = learned_label(args)
    learned = by_label[selected_learned_label]
    positive_sequences = sum(
        sequence["comparison"]["learned_psnr_change_vs_mean_db"] > 0.0
        for sequence in summaries)
    oracle_mse = None
    if all(sequence["source_route_reference"] is not None for sequence in summaries):
        oracle_mse = sum(
            sequence["source_route_reference"]["metrics"]["rgb_mse_level"]
            * sequence["frame_count"] for sequence in summaries) / total_frames
    pixel_gap_recovery = None
    source_gap_valid = (
        oracle_mse is not None and mean["rgb_mse_level"] > oracle_mse)
    if source_gap_valid:
        pixel_gap_recovery = (
            (mean["rgb_mse_level"] - learned["rgb_mse_level"])
            / (mean["rgb_mse_level"] - oracle_mse))
    lpips_not_worse = (
        mean["frame_weighted_lpips_alex"] is None
        or learned["frame_weighted_lpips_alex"]
        <= mean["frame_weighted_lpips_alex"] * 1.005)
    temporal_not_worse = (
        learned["transition_weighted_temporal_delta_mae"]
        <= mean["transition_weighted_temporal_delta_mae"] * 1.005)
    at_least_one_secondary_improves = (
        (mean["frame_weighted_lpips_alex"] is not None
         and learned["frame_weighted_lpips_alex"]
         < mean["frame_weighted_lpips_alex"])
        or learned["transition_weighted_temporal_delta_mae"]
        < mean["transition_weighted_temporal_delta_mae"])
    learned_points = [
        next(point for point in sequence["operational_points"]
             if point["label"] == selected_learned_label)
        for sequence in summaries
    ]
    active_only_verified = all(
        record["skipped_blocks"] == args.skip_blocks
        and record["predictor_activation"]["learned_module_input_shapes"][
            "q_projector"][0] == args.skip_blocks
        and record["predictor_activation"]["learned_module_input_shapes"][
            "common_projector"][0] == args.skip_blocks
        and record["predictor_activation"]["learned_module_input_shapes"][
            "mean_projector"][0] == args.skip_blocks
        and record["predictor_activation"]["learned_module_input_shapes"][
            "trunk"][0] == args.skip_blocks
        and record["predictor_activation"]["learned_module_output_shape"][0]
        == args.skip_blocks
        for point in learned_points for record in point["chunks"])
    lpips_reported = (
        mean["frame_weighted_lpips_alex"] is not None
        and learned["frame_weighted_lpips_alex"] is not None)
    source_reference_reported = oracle_mse is not None
    gate = {
        "c1_psnr_gain_at_least_0p015_db": (
            learned["frame_weighted_mean_psnr"]
            - mean["frame_weighted_mean_psnr"] >= 0.015),
        "c1_positive_on_at_least_5_of_6_dev_sequences": (
            positive_sequences >= 5 and len(summaries) == 6),
        "c1_total_bytes_within_mean_plus_0p5_percent": (
            learned["total_bytes"] <= mean["total_bytes"] * 1.005),
        "c1_lpips_not_worse_by_more_than_0p5_percent": lpips_not_worse,
        "c1_temporal_not_worse_by_more_than_0p5_percent": temporal_not_worse,
        "c1_improves_lpips_or_temporal": at_least_one_secondary_improves,
        "full_registered_six_sequence_dev_run": (
            tuple(sequence["sequence"] for sequence in summaries) == DEV_IDS),
        "full_registered_100_frames_per_sequence": all(
            sequence["frame_count"] == 100 for sequence in summaries),
        "lpips_reported": lpips_reported,
        "source_route_reference_reported": source_reference_reported,
        "source_route_reference_has_positive_gap": source_gap_valid,
        "active_only_learned_shapes_verified": active_only_verified,
        "positive_measured_predictor_compute": (
            learned["predictor_latency_ms_per_chunk"] > 0.0
            and learned["linear_macs_per_chunk"] > 0),
        "not_a_reduced_smoke_protocol": not args.allow_smoke_protocol,
    }
    boolean_gate_values = [value for value in gate.values() if isinstance(value, bool)]
    gate["passes_c1_g2_gate"] = all(boolean_gate_values)

    candidates = []
    for point in aggregate:
        candidates.append({
            "label": point["label"],
            "bpp": point["total_bpp"],
            "psnr": point["frame_weighted_mean_psnr"],
            "system_decode_ms_per_frame": point["system_decode_ms_per_frame"],
            "predictor_overhead_ms_per_chunk": point[
                "predictor_latency_ms_per_chunk"],
        })
    for point in candidates:
        dominators = []
        for other in candidates:
            if other is point:
                continue
            no_worse = (
                other["bpp"] <= point["bpp"]
                and other["psnr"] >= point["psnr"]
                and other["system_decode_ms_per_frame"]
                <= point["system_decode_ms_per_frame"])
            strict = (
                other["bpp"] < point["bpp"]
                or other["psnr"] > point["psnr"]
                or other["system_decode_ms_per_frame"]
                < point["system_decode_ms_per_frame"])
            if no_worse and strict:
                dominators.append(other["label"])
        point["sampled_operational_point_dominated"] = bool(dominators)
        point["sampled_operational_point_dominated_by"] = dominators
    baseline_points = [
        (int(item["label"].removeprefix("all-base-qp")),
         item["frame_weighted_mean_psnr"], item["total_bpp"])
        for item in aggregate if item["label"].startswith("all-base-qp")
    ]
    interpolation = interpolate_log_rate(
        baseline_points, learned["frame_weighted_mean_psnr"])
    if interpolation is None:
        matched_rd = {"status": "outside_all_base_quality_range"}
    else:
        matched_bpp, lower_qp, upper_qp, weight = interpolation
        matched_rd = {
            "status": "ok",
            "matched_all_base_bpp": matched_bpp,
            "learned_rate_change_percent": 100.0 * (
                learned["total_bpp"] / matched_bpp - 1.0),
            "lower_qp": lower_qp,
            "upper_qp": upper_qp,
            "interpolation_weight": weight,
        }
    per_sequence_matched = {
        sequence["sequence"]: sequence["comparison"][
            "matched_rd_mean_frame_psnr"]
        for sequence in summaries}
    matched_sequence_count = sum(
        item["status"] == "ok" for item in per_sequence_matched.values())
    improved_matched_sequence_count = sum(
        item.get("learned_rate_change_percent", float("inf")) < 0.0
        for item in per_sequence_matched.values())
    learned_candidate = next(
        item for item in candidates if item["label"] == selected_learned_label)
    predictor_feasibility_gate = dict(gate)
    predictor_feasibility_gate["passes_predictor_feasibility_gate"] = (
        predictor_feasibility_gate.pop("passes_c1_g2_gate"))
    pareto_gate = {
        "aggregate_matched_rd_available": matched_rd["status"] == "ok",
        "aggregate_matched_rate_improves": (
            matched_rd.get("learned_rate_change_percent", float("inf")) < 0.0),
        "per_sequence_matched_rd_available_for_all_six": (
            matched_sequence_count == 6),
        "matched_rate_improves_on_at_least_four_of_six": (
            improved_matched_sequence_count >= 4),
        "not_dominated_by_sampled_operational_points": (
            not learned_candidate["sampled_operational_point_dominated"]),
    }
    pareto_gate["passes_pareto_extension_gate"] = all(pareto_gate.values())
    return {
        "experiment": (
            "adaptive_chunk_coding_stage_b_predictor_g2_calibrated_followup_v1"
            if args.calibrated_followup
            else "adaptive_chunk_coding_stage_b_predictor_g2_dev_v1"),
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "claim_if_passed": (
                "A reusable decoder-only sparse C1 predictor provides a stable "
                "development-set gain over conditional mean under the frozen "
                "protocol; a calibrated follow-up is not an independent-sequence "
                "generalization test because alpha was chosen on the same dev split."),
            "sealed_test_not_accessed": "REDS val/024..029",
            "legacy_sequences_not_used_for_model_selection": True,
            "source_route_reference_is_not_an_operational_point": True,
            "calibrated_followup": args.calibrated_followup,
        },
        "protocol": {
            "train_split": "REDS train_sharp/001..024",
            "dev_split": "REDS val_sharp/000..005",
            "sealed_split": "REDS val_sharp/024..029 (not accessed)",
            "frame_count": args.frame_count,
            "crop_xy": [args.crop_x, args.crop_y],
            "crop_size": [args.width, args.height],
            "routing_unit": [2, 2, 256, 8],
            "route": "shared content-independent 64/256 mask per global chunk index",
            "qp": "routed QP32; all-Base QP30/31/32",
            "reset_interval_frames": args.reset_interval,
        },
        "predictor": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes_not_charged_per_video": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "profile": payload.get("profile", payload.get("predictor_profile")),
            "operational_profile": learned_profile(args),
            "output_alpha": args.learned_alpha,
            "architecture": payload.get("architecture"),
            "verified_training_provenance": checkpoint_provenance,
        },
        "aggregate_operational_points": aggregate,
        "rdc_sampled_points": candidates,
        "matched_rd_aggregate": matched_rd,
        "matched_rd_per_sequence": per_sequence_matched,
        "predictor_feasibility_gate": predictor_feasibility_gate,
        "pareto_extension_gate": pareto_gate,
        "source_reference_diagnostic": {
            "aggregate_rgb_mse_level": oracle_mse,
            "pixel_gap_recovery": pixel_gap_recovery,
            "valid_positive_mean_to_reference_gap": (
                oracle_mse is not None
                and mean["rgb_mse_level"] > oracle_mse),
        },
        "per_sequence": summaries,
    }


def write_csv(summary, path):
    rows = []
    for sequence in summary["per_sequence"]:
        for point in sequence["operational_points"]:
            rows.append({
                "sequence": sequence["sequence"],
                "label": point["label"],
                "total_bytes": point["stream"]["total_bytes"],
                "container_bytes": point["stream"]["container_bytes"],
                "i_bytes": point["stream"]["i_payload_bytes"],
                "z_bytes": point["stream"]["global_z_bytes"],
                "y_bytes": point["stream"]["base_y_bytes"],
                "route_bytes": point["stream"]["route_bytes"],
                "residual_bytes": point["stream"]["residual_bytes"],
                "mean_frame_psnr": point["metrics"]["mean_frame_psnr"],
                "aggregate_psnr": point["metrics"]["aggregate_psnr"],
                "lpips_alex": point["metrics"].get("lpips_alex", ""),
                "temporal_delta_mae": point["metrics"][
                    "temporal_delta_mae_rgb_level"],
                "decode_wall_seconds": point["compute"]["decode_wall_seconds_median"],
                "decode_throughput_fps": point["compute"]["decode_throughput_fps"],
                "predictor_ms_per_chunk": point["compute"][
                    "predictor_latency_ms_p50"],
                "linear_macs_per_chunk": point["compute"]["linear_macs_per_chunk"],
                "peak_cuda_allocated_bytes": point["compute"][
                    "peak_cuda_allocated_bytes_max"],
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_rdc_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["rdc_sampled_points"]
    latencies = [point["system_decode_ms_per_frame"] for point in points]
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    scatter = axis.scatter(
        [point["bpp"] for point in points],
        [point["psnr"] for point in points],
        c=latencies, cmap="viridis", s=100, edgecolors="black")
    for point in points:
        marker = (
            " dominated" if point["sampled_operational_point_dominated"] else "")
        axis.annotate(
            point["label"] + marker, (point["bpp"], point["psnr"]),
            xytext=(5, 6), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Actual total rate (bits/pixel/frame)")
    axis.set_ylabel("Frame-weighted mean PSNR (dB)")
    axis.grid(alpha=0.25)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Measured system decode time (ms/frame)")
    axis.set_title("Adaptive Chunk Coding REDS development R-D-C points")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_torch_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    metadata_predictor, checkpoint_payload, checkpoint_sha = load_predictor_checkpoint(
        checkpoint_path, device)
    checkpoint_provenance = validate_checkpoint_provenance(
        checkpoint_payload, require_phase_b=not args.allow_smoke_protocol)
    del metadata_predictor
    torch.cuda.empty_cache()
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    lpips_metric = None if args.no_lpips else LPIPSAlex(device)
    sequences = args.dev_sequences[
        :args.max_sequences if args.max_sequences else len(args.dev_sequences)]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sequence in sequences:
        summary = run_sequence(
            sequence, i_net, p_net, checkpoint_path, checkpoint_sha,
            args, device, lpips_metric)
        summaries.append(summary)
        print(json.dumps({
            "sequence": sequence,
            **summary["comparison"],
        }, indent=2), flush=True)
    aggregate = aggregate_results(
        summaries, args, checkpoint_path, checkpoint_sha,
        checkpoint_payload, checkpoint_provenance)
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(aggregate, output_root / "per_sequence_points.csv")
    save_rdc_plot(aggregate, output_root / "rdc_pareto.png")
    print(json.dumps({
        "summary": str(output_root / "summary.json"),
        "checkpoint_sha256": checkpoint_sha,
        "predictor_feasibility_gate": aggregate["predictor_feasibility_gate"],
        "pareto_extension_gate": aggregate["pareto_extension_gate"],
        "matched_rd_aggregate": aggregate["matched_rd_aggregate"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
