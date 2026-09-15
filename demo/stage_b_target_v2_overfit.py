#!/usr/bin/env python3
"""Train-only K8 capacity probe for the Stage-B C1 target-v2 hypothesis.

This is deliberately separate from the registered G2 cache/training/evaluation
programs.  It uses only REDS train_sharp/001..008, one fresh-state P chunk per
sequence, and refuses to train unless an actual-byte K8 route has enough RD
headroom.  Omitted latents and source pixels are supervision only; the sparse
predictor forward path remains decoder-only and active on exactly eight blocks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage1_multichunk_oracle import (  # noqa: E402
    decode_i_stream,
    initialize_p_state,
    load_models,
    make_chunk,
    model_frame,
    prepare_chunk_latents,
    reconstruction_float,
)
from demo.stage1_token_skipping import (  # noqa: E402
    build_route_section,
    decode_y,
    encode_y,
    expand_keep_mask,
)
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    compute_metrics,
    encode_trajectory,
    read_d2l,
    run_operational_mode,
    sha256_file,
)
from demo.stage_b_masked_predictor import (  # noqa: E402
    PredictorConfig,
    SparseMaskedLatentPredictor,
    skipped_target_blocks,
)
from demo.stage_b_prepare_predictor_cache import crop_xy  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


SEQUENCES = tuple(f"{index:03d}" for index in range(1, 9))
SEED = 20260908
ALPHA = 0.25
K = 8
STEPS = 4000
EVAL_EVERY = 250
EXPECTED_PARAMETERS = 461_824
EXPECTED_MACS = 5_776_384
CONTROL_PROFILE = "stage-b-c1-target-v1-train-only-capacity-gate"
TARGET_V2_PROFILE = "stage-b-c1-target-v2-train-only-capacity-gate"


@dataclass
class ProbeSample:
    sequence: str
    frames: list[np.ndarray]
    source_files: list[str]
    crop: tuple[int, int]
    route: np.ndarray
    decoded_q: torch.Tensor
    mean_y: torch.Tensor
    common_params: torch.Tensor
    target_delta: torch.Tensor
    rate_map: torch.Tensor
    sensitivity_map: torch.Tensor
    ctx: torch.Tensor
    q_decoder: torch.Tensor
    target_rgb: torch.Tensor
    mean_source_mse: float
    reference_source_mse: float
    allbase_y_bytes: int
    routed_y_bytes: int
    z_bytes: int
    route_bytes: int
    singleton_records: list[dict]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train-only K8 target-v2 capacity gate for Adaptive Chunk Coding.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--output-dir", default="output/stage_b_target_v2_overfit_train001_008")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.25)
    parser.add_argument("--nonzero-weight", type=float, default=8.0)
    return parser.parse_args()


def validate_args(args):
    if args.steps != STEPS or args.eval_every != EVAL_EVERY:
        raise ValueError("the registered capacity gate fixes 4000 steps and eval-every=250")
    if args.decode_repeats != 3:
        raise ValueError("the registered capacity gate requires three fresh decodes")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("the registered learning rate is 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("the registered weight decay is 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("the registered gradient clip is 1.0")
    if not math.isclose(args.smooth_l1_beta, 0.25):
        raise ValueError("the registered Smooth-L1 beta is 0.25")
    if not math.isclose(args.nonzero_weight, 8.0):
        raise ValueError("the registered nonzero weight is 8")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("the registered codec skip threshold is zero")


def ensure_new_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


def route_sha256(route: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(route, dtype=np.uint8).tobytes()).hexdigest()


def load_train_frames(data_root: Path, sequence: str):
    x, y = crop_xy(SEED, sequence, 0)
    root = data_root / "train_sharp" / sequence
    paths = sorted(root.glob("*.png"))[:9]
    if len(paths) != 9:
        raise ValueError(f"{root} does not provide the registered 9 frames")
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[y:y + 512, x:x + 512]
        if crop.shape != (512, 512, 3):
            raise ValueError(f"bad crop {x},{y} for {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return [str(path.resolve()) for path in paths], frames, (x, y)


@torch.inference_mode()
def encode_i(i_net, frame, device):
    source = (model_frame(frame, device) - 0.5).to(memory_format=torch.channels_last)
    encoded = i_net.compress(source, 32, 0, 0)
    stream = bytes(encoded["bit_stream"])
    i_hat = decode_i_stream(i_net, stream, int(encoded["ec_parallel"]), 32, 512, 512)
    return stream, int(encoded["ec_parallel"]), i_hat


def unit_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.float() - b.float()).square()).item())


@torch.inference_mode()
def collect_sample(i_net, p_net, data_root: Path, sequence: str, device):
    source_files, frames, crop = load_train_frames(data_root, sequence)
    _, _, i_hat = encode_i(i_net, frames[0], device)
    initialize_p_state(p_net, i_hat)
    chunk, target_rgb, valid_count = make_chunk(frames, 1, device)
    if valid_count != 8:
        raise RuntimeError("capacity sample is not one complete P chunk")
    prepared = prepare_chunk_latents(p_net, chunk, 32)
    y = prepared["y"]
    grid = (y.shape[-2] // 2, y.shape[-1] // 2)
    if grid != (16, 16):
        raise RuntimeError(f"unexpected route grid: {grid}")

    all_base_route = np.zeros(grid, dtype=np.bool_)
    all_base = encode_y(p_net, y, prepared["common_params"], all_base_route, 2)
    expected = np.nan_to_num(
        all_base.expected_bits_by_block.reshape(-1), nan=-1e30,
        posinf=1e30, neginf=-1e30)
    singleton_records = []
    for flat_id in range(grid[0] * grid[1]):
        route = np.zeros(grid, dtype=np.bool_)
        route.reshape(-1)[flat_id] = True
        encoded = encode_y(p_net, y, prepared["common_params"], route, 2)
        singleton_records.append({
            "flat_id": flat_id,
            "row": flat_id // grid[1],
            "col": flat_id % grid[1],
            "allbase_y_bytes": len(all_base.stream),
            "singleton_y_bytes": len(encoded.stream),
            "saved_y_bytes": len(all_base.stream) - len(encoded.stream),
            "expected_bits": float(expected[flat_id]),
        })
    ranked = sorted(
        singleton_records,
        key=lambda item: (
            -item["saved_y_bytes"], -item["expected_bits"], item["flat_id"]))
    selected = ranked[:K]
    route = np.zeros(grid, dtype=np.bool_)
    route.reshape(-1)[[item["flat_id"] for item in selected]] = True

    routed = encode_y(p_net, y, prepared["common_params"], route, 2)
    decoded_q, mean_y = decode_y(
        p_net, routed.stream, routed.ec_parallel,
        prepared["common_params"], route, 2)
    if not torch.equal(decoded_q, routed.q_dense):
        raise RuntimeError("K8 Base-y rANS round trip mismatch")

    from demo.stage_b_evaluate_predictor_g2 import source_route_target
    target_delta = source_route_target(p_net, prepared, decoded_q, route, 2)
    base_rgb, _, _ = reconstruction_float(
        p_net, all_base.y_hat, prepared["q_decoder"], valid_count)
    mean_rgb, _, _ = reconstruction_float(
        p_net, mean_y, prepared["q_decoder"], valid_count)
    reference_rgb, _, _ = reconstruction_float(
        p_net, mean_y + target_delta, prepared["q_decoder"], valid_count)
    mean_source_mse = unit_mse(mean_rgb, target_rgb)
    reference_source_mse = unit_mse(reference_rgb, target_rgb)

    rate_map = torch.zeros(grid, dtype=torch.float32, device=device)
    sensitivity_map = torch.zeros(grid, dtype=torch.float32, device=device)
    for item in selected:
        row, col = item["row"], item["col"]
        rate_map[row, col] = max(float(item["saved_y_bytes"]), 0.0)
        one_delta = torch.zeros_like(target_delta)
        one_delta[..., row * 2:(row + 1) * 2, col * 2:(col + 1) * 2] = (
            target_delta[..., row * 2:(row + 1) * 2, col * 2:(col + 1) * 2])
        single_rgb, _, _ = reconstruction_float(
            p_net, mean_y + one_delta, prepared["q_decoder"], valid_count)
        single_mse = unit_mse(single_rgb, target_rgb)
        sensitivity_map[row, col] = max(mean_source_mse - single_mse, 0.0)
        item["pixel_gain_unit_mse"] = float(sensitivity_map[row, col])

    if float((rate_map * sensitivity_map).sum()) <= 0.0:
        raise RuntimeError(f"{sequence}: no positive rate-times-sensitivity supervision")
    route_section = build_route_section(route, 2)
    return ProbeSample(
        sequence=sequence,
        frames=frames,
        source_files=source_files,
        crop=crop,
        route=route,
        decoded_q=decoded_q.detach().cpu(),
        mean_y=mean_y.detach().cpu(),
        common_params=prepared["common_params"].detach().cpu(),
        target_delta=target_delta.detach().cpu(),
        rate_map=rate_map.detach().cpu(),
        sensitivity_map=sensitivity_map.detach().cpu(),
        ctx=p_net.ctx.detach().cpu(),
        q_decoder=prepared["q_decoder"].detach().cpu(),
        target_rgb=target_rgb.detach().cpu(),
        mean_source_mse=mean_source_mse,
        reference_source_mse=reference_source_mse,
        allbase_y_bytes=len(all_base.stream),
        routed_y_bytes=len(routed.stream),
        z_bytes=len(prepared["global_stream"]),
        route_bytes=len(route_section),
        singleton_records=singleton_records,
    )


def evaluation_args(args):
    class Values:
        pass
    values = Values()
    values.model_path_i = args.model_path_i
    values.model_path_p = args.model_path_p
    values.skip_thres = args.skip_thres
    values.height = 512
    values.width = 512
    values.latent_block_size = 2
    values.reset_interval = 32
    values.decode_repeats = args.decode_repeats
    values.qp_i_override = 32
    values.qp_route = 32
    return values


def p_only_mse_from_metrics(metrics: dict) -> float:
    psnrs = metrics["per_frame_psnr"][1:]
    return float(np.mean([10.0 ** (-value / 10.0) for value in psnrs]))


def aggregate_points(results_by_sequence: dict, label: str):
    points = [results_by_sequence[sequence][label] for sequence in SEQUENCES]
    p_mse = float(np.mean([
        p_only_mse_from_metrics(point["metrics"]) for point in points]))
    components = {
        key: sum(point["stream"][key] for point in points)
        for key in (
            "sequence_header_and_chunk_lengths_bytes", "i_payload_bytes",
            "p_container_header_bytes", "global_z_bytes", "base_y_bytes",
            "route_bytes", "residual_bytes", "total_bytes")
    }
    decode_seconds = sum(point["compute"]["decode_wall_seconds_median"] for point in points)
    return {
        "label": label,
        "total_bytes": components["total_bytes"],
        "components": components,
        "p_only_unit_mse": p_mse,
        "p_only_psnr": -10.0 * math.log10(max(p_mse, 1e-30)),
        "decode_wall_seconds_sum": decode_seconds,
        "decode_fps": (len(SEQUENCES) * 9) / decode_seconds,
        "peak_cuda_allocated_bytes": max(
            point["compute"]["peak_memory_bytes"]
            if "peak_memory_bytes" in point["compute"] else
            point["compute"]["peak_cuda_allocated_bytes_max"]
            for point in points),
        "predictor_latency_ms_p50": float(np.median([
            point["compute"]["predictor_latency_ms_p50"] for point in points])),
        "linear_macs_per_chunk": max(
            point["compute"]["linear_macs_per_chunk"] for point in points),
    }


def aggregate_reference(results_by_sequence: dict):
    points = [results_by_sequence[sequence]["true-latent-reference"]
              for sequence in SEQUENCES]
    p_mse = float(np.mean([
        p_only_mse_from_metrics(point["metrics"]) for point in points]))
    return {
        "label": "true-latent-reference",
        "total_bytes": sum(
            results_by_sequence[sequence]["mean-k8"]["stream"]["total_bytes"]
            for sequence in SEQUENCES),
        "p_only_unit_mse": p_mse,
        "p_only_psnr": -10.0 * math.log10(max(p_mse, 1e-30)),
        "operational": False,
        "rate_source": "identical K8 payload represented by mean-k8 D2L",
    }


def matched_rate(baselines: list[dict], target: dict):
    ordered = sorted(baselines, key=lambda point: point["p_only_psnr"])
    quality = target["p_only_psnr"]
    if quality >= ordered[-1]["p_only_psnr"]:
        upper = ordered[-1]
        return {
            "comparable": True,
            "comparison": "directly_dominates_or_matches_highest_quality_anchor",
            "lower_label": upper["label"],
            "upper_label": upper["label"],
            "matched_allbase_bytes": upper["total_bytes"],
            "rate_change_percent": 100.0 * (
                target["total_bytes"] / upper["total_bytes"] - 1.0),
        }
    for lower, upper in zip(ordered, ordered[1:]):
        if lower["p_only_psnr"] <= quality <= upper["p_only_psnr"]:
            span = upper["p_only_psnr"] - lower["p_only_psnr"]
            fraction = 0.0 if span == 0 else (quality - lower["p_only_psnr"]) / span
            matched_log_bytes = (
                math.log(lower["total_bytes"])
                + fraction * (math.log(upper["total_bytes"])
                              - math.log(lower["total_bytes"])))
            matched_bytes = math.exp(matched_log_bytes)
            return {
                "comparable": True,
                "lower_label": lower["label"],
                "upper_label": upper["label"],
                "matched_allbase_bytes": matched_bytes,
                "rate_change_percent": 100.0 * (
                    target["total_bytes"] / matched_bytes - 1.0),
            }
    return {
        "comparable": False,
        "quality": quality,
        "baseline_quality_range": [
            ordered[0]["p_only_psnr"], ordered[-1]["p_only_psnr"]],
    }


@torch.inference_mode()
def run_precheck(i_net, p_net, samples, args, output_dir: Path, device):
    eval_args = evaluation_args(args)
    results = {}
    for sample in samples:
        sequence_root = output_dir / "precheck" / sample.sequence
        results[sample.sequence] = {}
        for qp in (30, 31, 32):
            label = f"allbase-p{qp}"
            print(json.dumps({"stage": "precheck", "sequence": sample.sequence,
                              "mode": label}), flush=True)
            results[sample.sequence][label] = run_operational_mode(
                label, "all-base", qp, i_net, p_net, None, None,
                sample.frames, eval_args, sequence_root / label, device, None)
        print(json.dumps({"stage": "precheck", "sequence": sample.sequence,
                          "mode": "mean-k8"}), flush=True)
        results[sample.sequence]["mean-k8"] = run_operational_mode(
            "mean-k8", "mean", 32, i_net, p_net, None, None,
            sample.frames, eval_args, sequence_root / "mean-k8", device, None,
            fixed_routes=[sample.route])
        reference_encoded = encode_trajectory(
            "true-latent-reference", "source-route-reference", 32,
            i_net, p_net, None, None, sample.frames, eval_args,
            sequence_root / "true-latent-reference", device,
            operational=False, fixed_routes=[sample.route])
        results[sample.sequence]["true-latent-reference"] = {
            "label": "true-latent-reference",
            "operational": False,
            "metrics": compute_metrics(
                sample.frames, reference_encoded["recon"],
                reference_encoded["chunks"], eval_args, None),
            "chunks": reference_encoded["chunks"],
        }
    aggregates = {
        label: aggregate_points(results, label)
        for label in ("allbase-p30", "allbase-p31", "allbase-p32", "mean-k8")
    }
    aggregates["true-latent-reference"] = aggregate_reference(results)
    baselines = [aggregates[f"allbase-p{qp}"] for qp in (30, 31, 32)]
    reference_match = matched_rate(baselines, aggregates["true-latent-reference"])
    same_qp_saving = 100.0 * (
        aggregates["mean-k8"]["total_bytes"]
        / aggregates["allbase-p32"]["total_bytes"] - 1.0)
    gate = {
        "k8_total_rate_change_vs_allbase_p32_percent": same_qp_saving,
        "requires_at_most_minus_1_percent_same_qp": same_qp_saving <= -1.0,
        "true_latent_matched_allbase": reference_match,
        "requires_true_latent_matched_at_most_minus_1_percent": (
            reference_match.get("comparable", False)
            and reference_match.get("rate_change_percent", float("inf")) <= -1.0),
    }
    gate["passes"] = (
        gate["requires_at_most_minus_1_percent_same_qp"]
        and gate["requires_true_latent_matched_at_most_minus_1_percent"])
    record = {"aggregates": aggregates, "gate": gate, "per_sequence": results}
    (output_dir / "precheck.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    return record


def to_device(sample: ProbeSample, device):
    return {
        "decoded_q": sample.decoded_q.to(device),
        "mean_y": sample.mean_y.to(device),
        "common_params": sample.common_params.to(device),
        "target_delta": sample.target_delta.to(device),
        "rate_map": sample.rate_map.to(device),
        "sensitivity_map": sample.sensitivity_map.to(device),
        "skip_blocks": torch.from_numpy(sample.route.astype(np.uint8)).to(device),
    }


def query_supervision(model, sample):
    target, query_ids = skipped_target_blocks(
        sample["target_delta"], sample["skip_blocks"], model.config.block_size)
    rates = sample["rate_map"].reshape(-1)[query_ids].float()
    sensitivities = sample["sensitivity_map"].reshape(-1)[query_ids].float()
    return target, rates, sensitivities, query_ids


@torch.inference_mode()
def latent_metrics(model, samples):
    error_sum = target_sum = 0.0
    weighted_error = weighted_target = 0.0
    for sample in samples:
        raw = model(
            sample["decoded_q"], sample["mean_y"],
            sample["common_params"], sample["skip_blocks"])
        target, rates, sensitivities, _ = query_supervision(model, sample)
        effective = ALPHA * raw
        block_error = (effective.float() - target.float()).square().sum(dim=1)
        block_target = target.float().square().sum(dim=1)
        weights = torch.clamp_min(rates, 0.0) * torch.clamp_min(sensitivities, 0.0)
        error_sum += float(block_error.sum())
        target_sum += float(block_target.sum())
        weighted_error += float((weights * block_error).sum())
        weighted_target += float((weights * block_target).sum())
    return {
        "latent_energy_recovery": 1.0 - error_sum / max(target_sum, 1e-30),
        "rate_sensitivity_weighted_latent_energy_recovery": (
            1.0 - weighted_error / max(weighted_target, 1e-30)),
        "error_energy": error_sum,
        "target_energy": target_sum,
        "weighted_error_energy": weighted_error,
        "weighted_target_energy": weighted_target,
    }


def make_schedule(sample_count: int):
    rng = np.random.default_rng(SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    schedule = schedule[:STEPS]
    digest = hashlib.sha256(np.asarray(schedule, dtype=np.int32).tobytes()).hexdigest()
    return schedule, digest


def control_loss(model, sample, args):
    raw = model(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"])
    target, _, _, _ = query_supervision(model, sample)
    effective = ALPHA * raw
    element = F.smooth_l1_loss(
        effective, target, beta=args.smooth_l1_beta, reduction="none")
    weights = 1.0 + args.nonzero_weight * (target.abs() > 1e-6)
    return torch.sum(element * weights) / torch.sum(weights)


def target_v2_loss(model, sample):
    raw = model(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"])
    target, rates, sensitivities, _ = query_supervision(model, sample)
    block_error = (ALPHA * raw - target).float().square().mean(dim=1)
    weights = torch.clamp_min(rates, 0.0) * torch.clamp_min(sensitivities, 0.0)
    if float(weights.sum()) <= 0.0:
        raise RuntimeError("target-v2 sample has no positive supervision weight")
    return torch.sum(weights * block_error) / torch.sum(weights)


def train_arm(name, model, samples, schedule, args, output_dir: Path):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    metrics_path = output_dir / f"{name}_metrics.jsonl"
    history = []
    initial = latent_metrics(model, samples)
    selection_key = (
        "latent_energy_recovery" if name == "control"
        else "rate_sensitivity_weighted_latent_energy_recovery")
    best_score = initial[selection_key]
    best_state = copy.deepcopy(model.state_dict())
    history.append({"step": 0, **initial})
    metrics_path.write_text(
        json.dumps(history[-1], ensure_ascii=False) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
    started = time.perf_counter()
    model.train()
    for step, index in enumerate(schedule, 1):
        sample = samples[index]
        loss = (control_loss(model, sample, args) if name == "control"
                else target_v2_loss(model, sample))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"{name}: non-finite gradient")
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            metrics = latent_metrics(model, samples)
            event = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(grad_norm),
                **metrics,
            }
            history.append(event)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps({"stage": "train", "arm": name, **event}), flush=True)
            if metrics[selection_key] > best_score:
                best_score = metrics[selection_key]
                best_state = copy.deepcopy(model.state_dict())
            model.train()
    torch.cuda.synchronize(next(model.parameters()).device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(next(model.parameters()).device))
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return {
        "selection_key": selection_key,
        "best_score": best_score,
        "best_metrics": latent_metrics(model, samples),
        "history": history,
        "training_wall_seconds": elapsed,
        "updates_per_second": STEPS / elapsed,
        "cuda_peak_allocated_bytes": peak,
    }


def save_checkpoint(path: Path, model, profile: str, training: dict,
                    args, schedule_hash: str, sample_records: list[dict]):
    payload = {
        "format_version": 1,
        "project": "Adaptive Chunk Coding",
        "profile": profile,
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "seed": SEED,
            "steps_requested": STEPS,
            "alpha": ALPHA,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "sample_schedule_sha256": schedule_hash,
            "result": training,
        },
        "data": {
            "split": "REDS train_sharp/001..008",
            "sequence_ids": list(SEQUENCES),
            "window_start": 0,
            "chunk_indices": [0],
            "sample_count": len(sample_records),
            "routes": sample_records,
        },
        "scientific_scope": {
            "train_only": True,
            "validation_or_sealed_data_read": False,
            "generalization_claim_allowed": False,
            "capacity_probe_only": True,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "model_state_sha256": model_state_sha256(model),
        "profile": profile,
    }


@torch.inference_mode()
def target_isolation(model, sample):
    parameters = list(inspect.signature(model.forward).parameters)
    before = model(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"]).clone()
    target_backup = sample["target_delta"]
    rate_backup = sample["rate_map"]
    sensitivity_backup = sample["sensitivity_map"]
    sample["target_delta"] = torch.flip(target_backup, dims=(-2, -1)) + 1
    sample["rate_map"] = rate_backup + 13
    sample["sensitivity_map"] = sensitivity_backup + 17
    after = model(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"])
    sample["target_delta"] = target_backup
    sample["rate_map"] = rate_backup
    sample["sensitivity_map"] = sensitivity_backup
    predicted_y, profile = model.apply(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"])
    effective_y = sample["mean_y"] + ALPHA * (predicted_y - sample["mean_y"])
    keep = expand_keep_mask(
        sample["skip_blocks"].detach().cpu().numpy().astype(np.bool_),
        2, sample["mean_y"].shape, sample["mean_y"].device)
    return {
        "forward_parameters": parameters,
        "supervision_perturbation_does_not_change_output": torch.equal(before, after),
        "base_positions_bit_exact": torch.equal(
            effective_y[keep], sample["mean_y"][keep]),
        "active_profile": profile,
    }


@torch.inference_mode()
def float_pixel_gate(model, cpu_samples, gpu_samples, p_net, device):
    mean_sum = reference_sum = predicted_sum = 0.0
    per_sequence = []
    for cpu, gpu in zip(cpu_samples, gpu_samples):
        predicted_y, _ = model.apply(
            gpu["decoded_q"], gpu["mean_y"],
            gpu["common_params"], gpu["skip_blocks"])
        predicted_y = gpu["mean_y"] + ALPHA * (predicted_y - gpu["mean_y"])
        p_net.ctx = cpu.ctx.to(device)
        rgb, _, _ = reconstruction_float(
            p_net, predicted_y.to(dtype=gpu["mean_y"].dtype),
            cpu.q_decoder.to(device), 8)
        predicted_mse = unit_mse(rgb, cpu.target_rgb.to(device))
        mean_sum += cpu.mean_source_mse
        reference_sum += cpu.reference_source_mse
        predicted_sum += predicted_mse
        per_sequence.append({
            "sequence": cpu.sequence,
            "mean_source_mse": cpu.mean_source_mse,
            "predicted_source_mse": predicted_mse,
            "true_fill_source_mse": cpu.reference_source_mse,
            "improves_mean": predicted_mse < cpu.mean_source_mse,
            "gap_recovery": (
                (cpu.mean_source_mse - predicted_mse)
                / max(cpu.mean_source_mse - cpu.reference_source_mse, 1e-30)),
        })
    return {
        "pooled_mean_source_mse": mean_sum / len(cpu_samples),
        "pooled_predicted_source_mse": predicted_sum / len(cpu_samples),
        "pooled_true_fill_source_mse": reference_sum / len(cpu_samples),
        "pooled_mean_to_true_fill_gap_recovery": (
            (mean_sum - predicted_sum) / max(mean_sum - reference_sum, 1e-30)),
        "positive_sequences": sum(item["improves_mean"] for item in per_sequence),
        "per_sequence": per_sequence,
    }


@torch.inference_mode()
def run_trained_modes(i_net, p_net, cpu_samples, args, output_dir, device,
                      checkpoints):
    eval_args = evaluation_args(args)
    results = {sample.sequence: {} for sample in cpu_samples}
    for sample in cpu_samples:
        for arm, checkpoint in checkpoints.items():
            label = f"{arm}-c1-a025-k8"
            print(json.dumps({"stage": "post-eval", "sequence": sample.sequence,
                              "mode": label}), flush=True)
            results[sample.sequence][label] = run_operational_mode(
                label, "learned-lite-c1-a025", 32, i_net, p_net,
                Path(checkpoint["path"]), checkpoint["file_sha256"],
                sample.frames, eval_args,
                output_dir / "post_eval" / sample.sequence / label,
                device, None, fixed_routes=[sample.route])
    return results


def verify_post_payload_identity(output_dir: Path, cpu_samples):
    """Prove that learned modes change reconstruction, not transmitted payloads."""
    records = []
    for sample in cpu_samples:
        mean_path = (
            output_dir / "precheck" / sample.sequence / "mean-k8" /
            "sequence.d2l")
        mean = read_d2l(mean_path)
        for arm in ("control", "target-v2"):
            label = f"{arm}-c1-a025-k8"
            learned_path = (
                output_dir / "post_eval" / sample.sequence / label /
                "sequence.d2l")
            learned = read_d2l(learned_path)
            i_identical = mean["i_stream"] == learned["i_stream"]
            chunks_identical = mean["chunks"] == learned["chunks"]
            size_identical = mean_path.stat().st_size == learned_path.stat().st_size
            if not (i_identical and chunks_identical and size_identical):
                raise RuntimeError(
                    f"payload mismatch for {sample.sequence} {arm}")
            records.append({
                "sequence": sample.sequence,
                "arm": arm,
                "mean_path": str(mean_path.resolve()),
                "learned_path": str(learned_path.resolve()),
                "i_payload_identical": i_identical,
                "p_chunk_payloads_identical": chunks_identical,
                "total_file_bytes_identical": size_identical,
            })
    return records


def sample_record(sample: ProbeSample):
    selected = sorted(
        (item for item in sample.singleton_records
         if sample.route.reshape(-1)[item["flat_id"]]),
        key=lambda item: item["flat_id"])
    return {
        "sequence": sample.sequence,
        "source_files": sample.source_files,
        "crop_xy": list(sample.crop),
        "route_sha256": route_sha256(sample.route),
        "selected_flat_ids": np.flatnonzero(sample.route.reshape(-1)).tolist(),
        "selected_singletons": selected,
        "allbase_y_bytes": sample.allbase_y_bytes,
        "k8_y_bytes": sample.routed_y_bytes,
        "k8_route_bytes": sample.route_bytes,
        "z_bytes": sample.z_bytes,
        "mean_source_mse": sample.mean_source_mse,
        "true_fill_source_mse": sample.reference_source_mse,
        "decoder_inputs": {
            "decoded_q_sha256": tensor_sha256(sample.decoded_q),
            "mean_y_sha256": tensor_sha256(sample.mean_y),
            "common_params_sha256": tensor_sha256(sample.common_params),
            "ctx_sha256": tensor_sha256(sample.ctx),
            "q_decoder_sha256": tensor_sha256(sample.q_decoder),
        },
        "supervision": {
            "target_delta_sha256": tensor_sha256(sample.target_delta),
            "rate_map_sha256": tensor_sha256(sample.rate_map),
            "sensitivity_map_sha256": tensor_sha256(sample.sensitivity_map),
        },
    }


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir)
    ensure_new_output(output_dir)
    set_torch_env()
    seed_everything(SEED)
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    total_started = time.perf_counter()

    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    data_root = Path(args.data_root)
    samples = []
    for sequence in SEQUENCES:
        print(json.dumps({"stage": "collect", "sequence": sequence}), flush=True)
        samples.append(collect_sample(i_net, p_net, data_root, sequence, device))
    records = [sample_record(sample) for sample in samples]
    (output_dir / "samples.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    precheck = run_precheck(i_net, p_net, samples, args, output_dir, device)
    if not precheck["gate"]["passes"]:
        summary = {
            "experiment": "adaptive_chunk_coding_stage_b_target_v2_capacity_gate",
            "project": "Adaptive Chunk Coding",
            "status": "stopped_at_preregistered_route_headroom_precheck",
            "protocol": {
                "split": "REDS train_sharp/001..008",
                "validation_or_sealed_data_read": False,
                "k": K,
                "alpha": ALPHA,
            },
            "samples": records,
            "precheck": precheck,
            "training_started": False,
            "elapsed_wall_seconds": time.perf_counter() - total_started,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")
        print(json.dumps(summary["status"]), flush=True)
        return

    gpu_samples = [to_device(sample, device) for sample in samples]
    config = PredictorConfig()
    seed_everything(SEED)
    initial_model = SparseMaskedLatentPredictor(config).to(device)
    if initial_model.parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("C1 parameter count changed")
    if initial_model.predicted_macs(K) != EXPECTED_MACS:
        raise RuntimeError("C1 K8 MAC count changed")
    initial_state = copy.deepcopy(initial_model.state_dict())
    initial_hash = model_state_sha256(initial_model)
    schedule, schedule_hash = make_schedule(len(gpu_samples))

    isolation = target_isolation(initial_model.eval(), gpu_samples[0])
    if not (isolation["supervision_perturbation_does_not_change_output"]
            and isolation["base_positions_bit_exact"]):
        raise RuntimeError("predictor target isolation or Base preservation failed")

    control = SparseMaskedLatentPredictor(config).to(device)
    control.load_state_dict(initial_state, strict=True)
    control_training = train_arm(
        "control", control, gpu_samples, schedule, args, output_dir)
    control_pixel = float_pixel_gate(control, samples, gpu_samples, p_net, device)
    control_checkpoint = save_checkpoint(
        output_dir / "control_best.pt", control, CONTROL_PROFILE,
        control_training, args, schedule_hash, records)

    target_v2 = SparseMaskedLatentPredictor(config).to(device)
    target_v2.load_state_dict(initial_state, strict=True)
    target_training = train_arm(
        "target_v2", target_v2, gpu_samples, schedule, args, output_dir)
    target_pixel = float_pixel_gate(target_v2, samples, gpu_samples, p_net, device)
    target_checkpoint = save_checkpoint(
        output_dir / "target_v2_best.pt", target_v2, TARGET_V2_PROFILE,
        target_training, args, schedule_hash, records)

    # Fresh-decode memory must exclude the in-memory training cache and all
    # simultaneously resident training models.  Operational evaluation below
    # reloads exactly one predictor from its checkpoint at a time.
    model_record = {
        "architecture": target_v2.architecture,
        "parameter_count": target_v2.parameter_count,
        "linear_macs_per_k8_chunk": target_v2.predicted_macs(K),
    }
    del gpu_samples, initial_state, initial_model, control, target_v2, i_net, p_net
    torch.cuda.empty_cache()
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)

    post = run_trained_modes(
        i_net, p_net, samples, args, output_dir, device,
        {"control": control_checkpoint, "target-v2": target_checkpoint})
    payload_identity = verify_post_payload_identity(output_dir, samples)
    post_aggregates = {}
    for arm in ("control", "target-v2"):
        label = f"{arm}-c1-a025-k8"
        post_aggregates[arm] = aggregate_points(post, label)
        post_aggregates[arm]["matched_allbase"] = matched_rate(
            [precheck["aggregates"][f"allbase-p{qp}"] for qp in (30, 31, 32)],
            post_aggregates[arm])

    target_match = post_aggregates["target-v2"]["matched_allbase"]
    active = isolation["active_profile"]
    gate_checks = {
        "route_headroom_precheck": precheck["gate"]["passes"],
        "float_rgb_gap_recovery_at_least_95_percent": (
            target_pixel["pooled_mean_to_true_fill_gap_recovery"] >= 0.95),
        "weighted_latent_recovery_at_least_95_percent": (
            target_training["best_metrics"][
                "rate_sensitivity_weighted_latent_energy_recovery"] >= 0.95),
        "at_least_7_of_8_sequences_improve_mean": (
            target_pixel["positive_sequences"] >= 7),
        "actual_d2l_matched_rate_at_most_minus_1_percent": (
            target_match.get("comparable", False)
            and target_match.get("rate_change_percent", float("inf")) <= -1.0),
        "decoder_only_target_isolation": (
            isolation["supervision_perturbation_does_not_change_output"]),
        "base_positions_bit_exact": isolation["base_positions_bit_exact"],
        "active_only_k8": (
            active["skipped_blocks"] == K
            and active["linear_macs"] == EXPECTED_MACS
            and not active["whole_latent_learned_activation"]),
        "parameter_count_unchanged": (
            model_record["parameter_count"] == EXPECTED_PARAMETERS),
        "fresh_decode_three_times": all(
            point["compute"]["decode_repeats"] == 3
            and point["validation"]["full_sequence_fresh_decode"]
            for sequence in SEQUENCES
            for point in post[sequence].values()),
        "mean_control_target_v2_payloads_identical": all(
            item["i_payload_identical"]
            and item["p_chunk_payloads_identical"]
            and item["total_file_bytes_identical"]
            for item in payload_identity),
    }
    passed = all(gate_checks.values())
    summary = {
        "experiment": "adaptive_chunk_coding_stage_b_target_v2_capacity_gate",
        "project": "Adaptive Chunk Coding",
        "status": "passed" if passed else "failed",
        "scientific_scope": {
            "capacity_probe_only": True,
            "generalization_claim_allowed": False,
            "multi_chunk_claim_allowed": False,
            "validation_or_sealed_data_read": False,
            "authorizes_train_only_short_training": passed,
            "authorizes_c3_controller_residual_or_codec_adaptation": False,
        },
        "protocol": {
            "split": "REDS train_sharp/001..008",
            "sequences": list(SEQUENCES),
            "window_start": 0,
            "chunk_indices": [0],
            "qp_i": 32,
            "qp_p": 32,
            "block_size": 2,
            "k": K,
            "alpha": ALPHA,
            "steps": STEPS,
            "seed": SEED,
            "route": "top-8 actual singleton Base-y byte saving",
            "control_loss": "alpha-aware nonzero-weighted Smooth-L1",
            "target_v2_loss": "alpha-aware rate-times-pixel-sensitivity weighted latent MSE",
        },
        "reproducibility": {
            "initial_model_state_sha256": initial_hash,
            "sample_schedule_sha256": schedule_hash,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "model": model_record,
        "samples": records,
        "precheck": precheck,
        "target_isolation": isolation,
        "training": {
            "control": control_training,
            "target_v2": target_training,
        },
        "float_rgb_capacity": {
            "control": control_pixel,
            "target_v2": target_pixel,
        },
        "checkpoints": {
            "control": control_checkpoint,
            "target_v2": target_checkpoint,
        },
        "post_eval_aggregates": post_aggregates,
        "post_eval_per_sequence": post,
        "post_eval_payload_identity": payload_identity,
        "gate_checks": gate_checks,
        "gate_passed": passed,
        "decision": (
            "authorize one preregistered train-only short-training audit"
            if passed else
            "stop target-v2 C1; do not proceed to short training, C3, Controller or Residual"),
        "elapsed_wall_seconds": time.perf_counter() - total_started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "gate_checks": gate_checks,
        "summary": str(summary_path.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
