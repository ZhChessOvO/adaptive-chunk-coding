#!/usr/bin/env python3
"""Train the Stage-B G2 masked latent predictor from the frozen HT-S cache.

This program is intentionally train-only.  It accepts exactly the cached
REDS train_sharp/001..024 split and never discovers or opens validation/test
data.  Phase A initializes the predictor from scratch after the deterministic
Torch environment has been configured.  Phase B is enabled only when both a
Phase-A checkpoint and its matching 96-sample predictor-rollout chunk-1 cache
are supplied; it trains on the paired mean-trajectory chunk0 plus refreshed
chunk1 samples.

The optimized objective is the nonzero-weighted Smooth-L1 loss used by the G1
capacity probe.  Periodic train-split evaluations additionally report:

* all-element loss and energy recovery (including false positives at zeros);
* nonzero-target-only loss and energy recovery;
* median per-skipped-block energy recovery.

These are optimization diagnostics only.  G2 acceptance still requires a
separate evaluation on the pre-registered validation sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_b_masked_predictor import (  # noqa: E402
    PredictorConfig,
    SparseMaskedLatentPredictor,
    skipped_target_blocks,
)
from src.utils.common import set_torch_env  # noqa: E402


ELIGIBLE_SEQUENCES = tuple(f"{index:03d}" for index in range(1, 25))
EXPECTED_KEYS = {
    "decoded_q", "mean_y", "common_params", "skip_blocks", "target_delta"
}
SAMPLE_NAME = re.compile(r"train_(\d{3})_s(\d+)_c([01])\.safetensors$")
PHASE_A_PROFILE = "stage-b-g2-train-only-sparse-masked-predictor"
PHASE_B_PROFILE = "stage-b-g2-phase-b-rollout-refresh-predictor"
REFRESH_FORMAT = "adaptive_chunk_coding_predictor_refresh_cache_v1"
REFRESH_TRAJECTORY = "phase_a_predictor_chunk0_rollout"
PHASE_A_DEFAULT_OUTPUT = "output/stage_b_predictor_g2_train_seed20260908"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the G2 sparse predictor from train-only SafeTensors cache.")
    parser.add_argument(
        "--cache-dir",
        default="data/REDS/cache_hts_qp32_b2_g2_v1")
    parser.add_argument(
        "--output-dir", default=PHASE_A_DEFAULT_OUTPUT)
    parser.add_argument(
        "--init-checkpoint", default="",
        help="Phase-B only: initialize from a Phase-A best/final checkpoint.")
    parser.add_argument(
        "--refresh-cache", default="",
        help="Phase-B only: 96 chunk-1 samples from predictor rollout refresh.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nonzero-weight", type=float, default=8.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--cache-device", choices=("cpu", "cuda"), default="cuda",
        help="Keep the verified 554 MB cache on CPU or the selected CUDA device.")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--q-width", type=int, default=32)
    parser.add_argument("--common-width", type=int, default=64)
    parser.add_argument("--mean-width", type=int, default=32)
    parser.add_argument("--hidden-width", type=int, default=128)
    return parser.parse_args()


def validate_args(args):
    positive = (
        (args.steps, "steps"),
        (args.learning_rate, "learning-rate"),
        (args.smooth_l1_beta, "smooth-l1-beta"),
        (args.gradient_clip, "gradient-clip"),
        (args.log_every, "log-every"),
        (args.eval_every, "eval-every"),
        (args.checkpoint_every, "checkpoint-every"),
        (args.q_width, "q-width"),
        (args.common_width, "common-width"),
        (args.mean_width, "mean-width"),
        (args.hidden_width, "hidden-width"),
    )
    for value, name in positive:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.nonzero_weight < 0 or args.weight_decay < 0 or args.radius < 0:
        raise ValueError("weights and radius must be non-negative")
    if bool(args.init_checkpoint) != bool(args.refresh_cache):
        raise ValueError(
            "Phase B requires both --init-checkpoint and --refresh-cache")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    """Hash tensor names, dtypes, shapes and bytes in a model state."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def seed_after_torch_setup(seed: int):
    """Apply the requested seed after set_torch_env(), which seeds Torch to zero."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_new_output_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_cache(cache_dir: Path):
    summary_path = cache_dir / "summary.json"
    manifest_path = cache_dir / "manifest.jsonl"
    sample_dir = cache_dir / "samples"
    if not summary_path.is_file() or not manifest_path.is_file() or not sample_dir.is_dir():
        raise FileNotFoundError(f"incomplete predictor cache: {cache_dir}")

    summary = read_json(summary_path)
    if summary.get("format") != "adaptive_chunk_coding_predictor_cache_v1":
        raise ValueError("unsupported predictor cache format")
    if summary.get("split") != "REDS train_sharp/001..024":
        raise ValueError("cache is not the registered train/001..024 split")
    if summary.get("debug_prefix_only"):
        raise ValueError("G2 training refuses a debug-prefix cache")
    if tuple(summary.get("sequences", ())) != ELIGIBLE_SEQUENCES:
        raise ValueError("cache sequence list is not exactly train/001..024")
    if summary.get("trajectory") != "mean_fill":
        raise ValueError("G2 requires the registered mean-fill cache trajectory")
    if summary.get("qp_i") != 32 or summary.get("qp_p") != 32:
        raise ValueError("G2 cache must use I/P QP32")
    if summary.get("crop_size") != [512, 512]:
        raise ValueError("G2 cache must use 512x512 crops")

    entries = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            sequence = str(entry.get("sequence", ""))
            if sequence not in ELIGIBLE_SEQUENCES:
                raise ValueError(
                    f"manifest line {line_number} references forbidden split/sequence")
            if entry.get("trajectory") != "mean_fill":
                raise ValueError(f"manifest line {line_number} is not mean-fill")
            filename = Path(str(entry.get("path", ""))).name
            match = SAMPLE_NAME.fullmatch(filename)
            if match is None or match.group(1) != sequence:
                raise ValueError(f"invalid train-cache filename at line {line_number}")
            local_path = sample_dir / filename
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            normalized = dict(entry)
            normalized["path"] = str(local_path)
            entries.append(normalized)

    if len(entries) != summary.get("sample_count"):
        raise ValueError("manifest length differs from cache summary")
    if len(entries) != 192:
        raise ValueError(f"registered G2 cache must contain 192 samples, got {len(entries)}")
    counts = Counter(entry["sequence"] for entry in entries)
    if set(counts) != set(ELIGIBLE_SEQUENCES) or set(counts.values()) != {8}:
        raise ValueError("each train sequence must contribute exactly eight cached chunks")
    if any("val" in Path(entry["path"]).name.lower() or
           "test" in Path(entry["path"]).name.lower() for entry in entries):
        raise ValueError("validation/test cache path rejected")
    return summary, entries, summary_path, manifest_path


def validate_refresh_cache(
        refresh_dir: Path, mean_manifest_hash: str, init_checkpoint_hash: str):
    summary_path = refresh_dir / "summary.json"
    manifest_path = refresh_dir / "manifest.jsonl"
    sample_dir = refresh_dir / "samples"
    if not summary_path.is_file() or not manifest_path.is_file() or not sample_dir.is_dir():
        raise FileNotFoundError(f"incomplete rollout-refresh cache: {refresh_dir}")
    summary = read_json(summary_path)
    if summary.get("format") != REFRESH_FORMAT:
        raise ValueError("unsupported rollout-refresh cache format")
    if summary.get("split") != "REDS train_sharp/001..024":
        raise ValueError("refresh cache is not train/001..024")
    if summary.get("debug_prefix_only"):
        raise ValueError("Phase B refuses a debug-prefix refresh cache")
    if tuple(summary.get("sequences", ())) != ELIGIBLE_SEQUENCES:
        raise ValueError("refresh sequence list is not exactly train/001..024")
    if summary.get("trajectory") != REFRESH_TRAJECTORY:
        raise ValueError("refresh cache has the wrong predictor trajectory")
    if summary.get("cached_chunk_indices") != [1] or summary.get("sample_count") != 96:
        raise ValueError("refresh cache must contain exactly 96 chunk-1 samples")
    source_cache = summary.get("source_mean_cache", {})
    if source_cache.get("manifest_sha256") != mean_manifest_hash:
        raise ValueError("refresh cache is not bound to the supplied mean cache")
    phase_a = summary.get("phase_a_checkpoint", {})
    if phase_a.get("file_sha256") != init_checkpoint_hash:
        raise ValueError("refresh cache and Phase-B initializer checkpoints differ")
    if str(phase_a.get("profile", "")) != PHASE_A_PROFILE:
        raise ValueError("refresh cache was not produced by a Phase-A checkpoint")

    entries = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            sequence = str(entry.get("sequence", ""))
            filename = Path(str(entry.get("path", ""))).name
            match = SAMPLE_NAME.fullmatch(filename)
            if (sequence not in ELIGIBLE_SEQUENCES or match is None
                    or match.group(1) != sequence or int(match.group(3)) != 1):
                raise ValueError(f"invalid refresh sample at manifest line {line_number}")
            if int(entry.get("chunk_index", -1)) != 1:
                raise ValueError("refresh manifest contains a non-chunk1 sample")
            if entry.get("trajectory") != REFRESH_TRAJECTORY:
                raise ValueError("refresh manifest trajectory mismatch")
            if entry.get("phase_a_checkpoint_sha256") != init_checkpoint_hash:
                raise ValueError("refresh sample checkpoint provenance mismatch")
            local_path = sample_dir / filename
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            normalized = dict(entry)
            normalized["path"] = str(local_path)
            entries.append(normalized)
    if len(entries) != 96:
        raise ValueError(f"refresh manifest must contain 96 samples, got {len(entries)}")
    expected = {
        (sequence, start, 1)
        for sequence in ELIGIBLE_SEQUENCES for start in (0, 24, 48, 72)
    }
    actual = {
        (str(entry["sequence"]), int(entry["window_start"]), int(entry["chunk_index"]))
        for entry in entries
    }
    if actual != expected or len(actual) != len(entries):
        raise ValueError("refresh cache does not contain the registered chunk-1 grid")
    return summary, entries, summary_path, manifest_path


def phase_b_entries(mean_entries, refresh_entries):
    mean_chunk0 = {
        (str(entry["sequence"]), int(entry["window_start"])): entry
        for entry in mean_entries if int(entry["chunk_index"]) == 0
    }
    mean_chunk1 = {
        (str(entry["sequence"]), int(entry["window_start"])): entry
        for entry in mean_entries if int(entry["chunk_index"]) == 1
    }
    refreshed_chunk1 = {
        (str(entry["sequence"]), int(entry["window_start"])): entry
        for entry in refresh_entries
    }
    expected = {
        (sequence, start)
        for sequence in ELIGIBLE_SEQUENCES for start in (0, 24, 48, 72)
    }
    if (set(mean_chunk0) != expected or set(mean_chunk1) != expected
            or set(refreshed_chunk1) != expected):
        raise ValueError("Phase-B mean/refresh window grids do not match")
    combined = []
    for sequence in ELIGIBLE_SEQUENCES:
        for start in (0, 24, 48, 72):
            key = (sequence, start)
            first = mean_chunk0[key]
            original_second = mean_chunk1[key]
            refreshed_second = refreshed_chunk1[key]
            for field in (
                    "crop_x", "crop_y", "first_source_file", "last_source_file",
                    "skip_blocks"):
                if original_second.get(field) != refreshed_second.get(field):
                    raise ValueError(
                        f"mean/refresh chunk1 mismatch for {sequence}/{start}/{field}")
            if (int(refreshed_second.get("chunk0_predictor_skipped_blocks", -1))
                    != int(first["skip_blocks"])):
                raise ValueError(
                    f"refresh chunk0 route mismatch for {sequence}/{start}")
            combined.extend((first, refreshed_second))
    return combined


def expanded_skip_mask(skip_blocks: torch.Tensor) -> torch.Tensor:
    skip = skip_blocks.to(dtype=torch.bool)
    return skip.repeat_interleave(2, 0).repeat_interleave(2, 1)[None, None]


def load_cached_samples(
        entries, cache_device: torch.device, expected_trajectory: str = "mean_fill"):
    samples = []
    expected_shapes = {
        "decoded_q": (1, 256, 32, 32),
        "mean_y": (1, 256, 32, 32),
        "common_params": (1, 768, 32, 32),
        "target_delta": (1, 256, 32, 32),
        "skip_blocks": (16, 16),
    }
    for entry in entries:
        path = Path(entry["path"])
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != EXPECTED_KEYS:
                raise ValueError(f"unexpected tensor keys in {path}")
            metadata = handle.metadata() or {}
            if metadata.get("sequence") != entry["sequence"]:
                raise ValueError(f"sequence metadata mismatch in {path}")
            if metadata.get("trajectory") != expected_trajectory:
                raise ValueError(f"unexpected cache trajectory in {path}")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key, shape in expected_shapes.items():
            if tuple(tensors[key].shape) != shape:
                raise ValueError(f"{path}: {key} has shape {tuple(tensors[key].shape)}")
        if tensors["decoded_q"].dtype != torch.int8:
            raise ValueError(f"{path}: decoded_q must be int8")
        if tensors["skip_blocks"].dtype != torch.uint8:
            raise ValueError(f"{path}: skip_blocks must be uint8")
        skip = expanded_skip_mask(tensors["skip_blocks"])
        if int(tensors["skip_blocks"].count_nonzero()) != int(entry["skip_blocks"]):
            raise ValueError(f"{path}: route cardinality mismatch")
        if torch.count_nonzero(tensors["decoded_q"].masked_select(skip)).item():
            raise ValueError(f"{path}: decoded_q leaks values into skipped blocks")
        if torch.count_nonzero(tensors["target_delta"].masked_select(~skip)).item():
            raise ValueError(f"{path}: target_delta is nonzero outside skipped blocks")
        samples.append({
            key: tensor.to(device=cache_device, non_blocking=False)
            for key, tensor in tensors.items()
        })
    return samples


def sample_on_device(sample, device: torch.device):
    return {
        key: value if value.device == device else value.to(device=device)
        for key, value in sample.items()
    }


def prediction_and_target(model, sample):
    prediction = model(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"])
    target, query_ids = skipped_target_blocks(
        sample["target_delta"], sample["skip_blocks"],
        model.config.block_size)
    if prediction.shape != target.shape:
        raise RuntimeError("prediction and cached target shapes differ")
    return prediction, target, query_ids


def smooth_l1_components(prediction, target, beta, nonzero_weight):
    element = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    nonzero = target.abs() > 1e-6
    weights = 1.0 + nonzero_weight * nonzero.to(dtype=element.dtype)
    weighted = torch.sum(element * weights) / torch.sum(weights)
    return weighted, element, nonzero, weights


def one_sample_metrics(prediction, target, beta, nonzero_weight):
    weighted, element, nonzero, weights = smooth_l1_components(
        prediction, target, beta, nonzero_weight)
    error_sq = (prediction.float() - target.float()).square()
    target_sq = target.float().square()
    block_error = error_sq.sum(dim=1)
    block_energy = target_sq.sum(dim=1)
    valid_blocks = block_energy > 1e-12
    block_recovery = 1.0 - block_error[valid_blocks] / block_energy[valid_blocks]
    nonzero_count = int(nonzero.sum().item())
    return {
        "optimized_weighted_smooth_l1": float(weighted.detach()),
        "all_element_smooth_l1": float(element.mean().detach()),
        "nonzero_only_smooth_l1": (
            float(element[nonzero].mean().detach()) if nonzero_count else None),
        "all_element_recovery": float(
            1.0 - error_sq.sum() / torch.clamp_min(target_sq.sum(), 1e-12)),
        "nonzero_only_recovery": (
            float(1.0 - error_sq[nonzero].sum()
                  / torch.clamp_min(target_sq[nonzero].sum(), 1e-12))
            if nonzero_count else None),
        "per_block_median_recovery": (
            float(block_recovery.median()) if block_recovery.numel() else None),
        "target_nonzero_fraction": nonzero_count / target.numel(),
        "target_coefficients": target.numel(),
        "positive_energy_blocks": int(valid_blocks.sum().item()),
        "skipped_blocks": target.shape[0],
        "weighted_denominator": float(weights.sum().detach()),
    }


@torch.no_grad()
def evaluate_train(model, samples, device, beta, nonzero_weight):
    model.eval()
    totals = {
        "weighted_loss_sum": 0.0,
        "weighted_denominator": 0.0,
        "smooth_l1_sum": 0.0,
        "element_count": 0,
        "nonzero_smooth_l1_sum": 0.0,
        "nonzero_count": 0,
        "error_sq": 0.0,
        "target_sq": 0.0,
        "nonzero_error_sq": 0.0,
        "nonzero_target_sq": 0.0,
    }
    block_recoveries = []
    skipped_blocks = 0
    positive_energy_blocks = 0
    start = time.perf_counter()
    for cached in samples:
        sample = sample_on_device(cached, device)
        prediction, target, _ = prediction_and_target(model, sample)
        _, element, nonzero, weights = smooth_l1_components(
            prediction, target, beta, nonzero_weight)
        error_sq = (prediction.float() - target.float()).square()
        target_sq = target.float().square()
        totals["weighted_loss_sum"] += float((element * weights).sum())
        totals["weighted_denominator"] += float(weights.sum())
        totals["smooth_l1_sum"] += float(element.sum())
        totals["element_count"] += element.numel()
        totals["nonzero_smooth_l1_sum"] += float(element[nonzero].sum())
        totals["nonzero_count"] += int(nonzero.sum())
        totals["error_sq"] += float(error_sq.sum())
        totals["target_sq"] += float(target_sq.sum())
        totals["nonzero_error_sq"] += float(error_sq[nonzero].sum())
        totals["nonzero_target_sq"] += float(target_sq[nonzero].sum())
        block_error = error_sq.sum(dim=1)
        block_energy = target_sq.sum(dim=1)
        valid = block_energy > 1e-12
        block_recoveries.extend(
            (1.0 - block_error[valid] / block_energy[valid]).cpu().tolist())
        skipped_blocks += target.shape[0]
        positive_energy_blocks += int(valid.sum())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    model.train()
    return {
        "optimized_weighted_smooth_l1": (
            totals["weighted_loss_sum"] / totals["weighted_denominator"]),
        "all_element_smooth_l1": (
            totals["smooth_l1_sum"] / totals["element_count"]),
        "nonzero_only_smooth_l1": (
            totals["nonzero_smooth_l1_sum"] / totals["nonzero_count"]),
        "all_element_recovery": (
            1.0 - totals["error_sq"] / max(totals["target_sq"], 1e-12)),
        "nonzero_only_recovery": (
            1.0 - totals["nonzero_error_sq"]
            / max(totals["nonzero_target_sq"], 1e-12)),
        "per_block_median_recovery": float(np.median(block_recoveries)),
        "per_block_mean_recovery": float(np.mean(block_recoveries)),
        "per_block_p10_recovery": float(np.percentile(block_recoveries, 10)),
        "per_block_p90_recovery": float(np.percentile(block_recoveries, 90)),
        "target_nonzero_fraction": (
            totals["nonzero_count"] / totals["element_count"]),
        "target_coefficients": totals["element_count"],
        "target_nonzero_coefficients": totals["nonzero_count"],
        "skipped_blocks": skipped_blocks,
        "positive_energy_blocks": positive_energy_blocks,
        "evaluation_wall_seconds": elapsed,
    }


@torch.no_grad()
def train_data_statistics(samples, entries, model, device):
    coefficients = 0
    nonzero = 0
    target_sum = 0.0
    target_sq_sum = 0.0
    target_abs_sum = 0.0
    positive_blocks = 0
    total_blocks = 0
    for cached in samples:
        sample = sample_on_device(cached, device)
        target, _ = skipped_target_blocks(
            sample["target_delta"], sample["skip_blocks"], model.config.block_size)
        mask = target.abs() > 1e-6
        coefficients += target.numel()
        nonzero += int(mask.sum())
        target_sum += float(target.sum())
        target_sq_sum += float(target.float().square().sum())
        target_abs_sum += float(target.abs().sum())
        energy = target.float().square().sum(dim=1)
        total_blocks += target.shape[0]
        positive_blocks += int((energy > 1e-12).sum())
    histogram = Counter(str(entry["skip_blocks"]) for entry in entries)
    return {
        "source": "train cache only",
        "sequence_ids": list(ELIGIBLE_SEQUENCES),
        "sample_count": len(samples),
        "skip_count_histogram": dict(sorted(histogram.items())),
        "target_coefficients": coefficients,
        "target_nonzero_coefficients": nonzero,
        "target_nonzero_fraction": nonzero / coefficients,
        "target_mean": target_sum / coefficients,
        "target_mean_abs": target_abs_sum / coefficients,
        "target_rms": math.sqrt(target_sq_sum / coefficients),
        "skipped_blocks": total_blocks,
        "positive_energy_blocks": positive_blocks,
    }


def make_schedule(sample_count: int, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < steps:
        schedule.extend(rng.permutation(sample_count).tolist())
    schedule = schedule[:steps]
    digest = hashlib.sha256(np.asarray(schedule, dtype=np.int32).tobytes()).hexdigest()
    return schedule, digest


def load_phase_a_initializer(
        path: Path, model: SparseMaskedLatentPredictor,
        mean_manifest_hash: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    file_hash = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    profile = str(payload.get("profile", ""))
    if profile != PHASE_A_PROFILE or "g1" in profile.lower():
        raise ValueError("Phase B refuses G1 and non-Phase-A checkpoints")
    scope = payload.get("scientific_scope", {})
    data = payload.get("data", {})
    if not scope.get("train_only") or scope.get("validation_or_sealed_data_read"):
        raise ValueError("initializer does not have train-only provenance")
    if data.get("split") != "REDS train_sharp/001..024":
        raise ValueError("initializer was not trained on the registered train split")
    if data.get("manifest_sha256") != mean_manifest_hash:
        raise ValueError("initializer and supplied mean cache differ")
    if payload.get("architecture") != model.architecture:
        raise ValueError("CLI architecture differs from Phase-A checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    return {
        "type": "phase_a_checkpoint",
        "path": str(path),
        "file_sha256": file_hash,
        "model_state_sha256": state_sha256(model),
        "profile": profile,
        "step": payload.get("training", {}).get("step"),
        "mean_manifest_sha256": mean_manifest_hash,
    }


def checkpoint_payload(
        model, step, metrics, args, cache_record, schedule_hash,
        initialization_record):
    phase_b = cache_record.get("phase") == "phase_b_rollout_refresh"
    return {
        "format_version": 1,
        "project": "Adaptive Chunk Coding",
        "profile": PHASE_B_PROFILE if phase_b else PHASE_A_PROFILE,
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "training": {
            "step": step,
            "seed": args.seed,
            "steps_requested": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "nonzero_weight": args.nonzero_weight,
            "smooth_l1_beta": args.smooth_l1_beta,
            "gradient_clip": args.gradient_clip,
            "metrics": metrics,
            "sample_schedule_sha256": schedule_hash,
        },
        "data": cache_record,
        "initialization": initialization_record,
        "scientific_scope": {
            "train_only": True,
            "validation_or_sealed_data_read": False,
            "generalization_claim_allowed": False,
            "training_phase": "phase_b" if phase_b else "phase_a",
        },
    }


def save_checkpoint(
        path, model, step, metrics, args, cache_record, schedule_hash,
        initialization_record):
    payload = checkpoint_payload(
        model, step, metrics, args, cache_record, schedule_hash,
        initialization_record)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "model_state_sha256": state_sha256(model),
        "step": step,
        "all_element_recovery": metrics["all_element_recovery"],
        "nonzero_only_recovery": metrics["nonzero_only_recovery"],
        "per_block_median_recovery": metrics["per_block_median_recovery"],
    }


def append_jsonl(path, item):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("G2 training requires CUDA")

    total_start = time.perf_counter()
    cache_dir = Path(args.cache_dir).resolve()
    phase_b = bool(args.init_checkpoint)
    if phase_b and args.output_dir == PHASE_A_DEFAULT_OUTPUT:
        args.output_dir = "output/stage_b_predictor_g2_phase_b_seed20260908"
    output_dir = Path(args.output_dir)
    ensure_new_output_dir(output_dir)
    cache_summary, entries, summary_path, manifest_path = validate_cache(cache_dir)
    mean_manifest_hash = sha256_file(manifest_path)
    init_checkpoint_path = (
        Path(args.init_checkpoint).resolve() if phase_b else None)
    init_checkpoint_file_hash = (
        sha256_file(init_checkpoint_path) if init_checkpoint_path else None)
    refresh_summary = None
    refresh_summary_path = None
    refresh_manifest_path = None
    refresh_entries = None
    if phase_b:
        refresh_dir = Path(args.refresh_cache).resolve()
        (refresh_summary, refresh_entries, refresh_summary_path,
         refresh_manifest_path) = validate_refresh_cache(
            refresh_dir, mean_manifest_hash, init_checkpoint_file_hash)
        combined_entries = phase_b_entries(entries, refresh_entries)
        mean_chunk0_entries = [
            entry for entry in combined_entries if int(entry["chunk_index"]) == 0]
        refreshed_chunk1_entries = [
            entry for entry in combined_entries
            if entry.get("trajectory") == REFRESH_TRAJECTORY]
        if len(mean_chunk0_entries) != 96 or len(refreshed_chunk1_entries) != 96:
            raise RuntimeError("Phase-B cache pairing did not produce 96+96 samples")

    # This call deliberately precedes the requested seed: set_torch_env() sets
    # torch.manual_seed(0), which otherwise silently defeats the CLI seed.
    set_torch_env()
    seed_after_torch_setup(args.seed)
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    cache_device = device if args.cache_device == "cuda" else torch.device("cpu")
    torch.cuda.reset_peak_memory_stats(device)

    load_start = time.perf_counter()
    if phase_b:
        mean_samples = load_cached_samples(
            mean_chunk0_entries, cache_device, "mean_fill")
        refresh_samples = load_cached_samples(
            refreshed_chunk1_entries, cache_device, REFRESH_TRAJECTORY)
        mean_by_key = {
            (entry["sequence"], int(entry["window_start"])): sample
            for entry, sample in zip(mean_chunk0_entries, mean_samples)
        }
        refresh_by_key = {
            (entry["sequence"], int(entry["window_start"])): sample
            for entry, sample in zip(refreshed_chunk1_entries, refresh_samples)
        }
        entries = []
        samples = []
        for sequence in ELIGIBLE_SEQUENCES:
            for window_start in (0, 24, 48, 72):
                key = (sequence, window_start)
                mean_index = next(
                    index for index, entry in enumerate(mean_chunk0_entries)
                    if (entry["sequence"], int(entry["window_start"])) == key)
                refresh_index = next(
                    index for index, entry in enumerate(refreshed_chunk1_entries)
                    if (entry["sequence"], int(entry["window_start"])) == key)
                entries.extend((
                    mean_chunk0_entries[mean_index],
                    refreshed_chunk1_entries[refresh_index],
                ))
                samples.extend((mean_by_key[key], refresh_by_key[key]))
    else:
        samples = load_cached_samples(entries, cache_device, "mean_fill")
    torch.cuda.synchronize(device)
    cache_load_seconds = time.perf_counter() - load_start

    config = PredictorConfig(
        block_size=2,
        radius=args.radius,
        q_width=args.q_width,
        common_width=args.common_width,
        mean_width=args.mean_width,
        hidden_width=args.hidden_width,
    )
    predictor = SparseMaskedLatentPredictor(config).to(device)
    if phase_b:
        initialization_record = load_phase_a_initializer(
            init_checkpoint_path, predictor, mean_manifest_hash)
    else:
        initialization_record = {
            "type": "random",
            "seed": args.seed,
            "applied_after_set_torch_env": True,
            "model_state_sha256": state_sha256(predictor),
        }
    initial_state_hash = state_sha256(predictor)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    schedule, schedule_hash = make_schedule(len(samples), args.steps, args.seed)
    data_stats = train_data_statistics(samples, entries, predictor, device)
    initial_metrics = evaluate_train(
        predictor, samples, device, args.smooth_l1_beta, args.nonzero_weight)

    mean_cache_record = {
        "cache_dir": str(cache_dir),
        "format": cache_summary["format"],
        "split": cache_summary["split"],
        "trajectory": cache_summary["trajectory"],
        "sample_count": 96 if phase_b else len(samples),
        "selected_chunk_indices": [0] if phase_b else [0, 1],
        "sequence_ids": list(ELIGIBLE_SEQUENCES),
        "cache_summary_sha256": sha256_file(summary_path),
        "manifest_sha256": mean_manifest_hash,
    }
    if phase_b:
        cache_record = {
            "phase": "phase_b_rollout_refresh",
            "split": "REDS train_sharp/001..024",
            "sample_count": len(samples),
            "sequence_ids": list(ELIGIBLE_SEQUENCES),
            "pairing": "mean chunk0 + Phase-A-predictor-trajectory chunk1",
            "mean_chunk0_cache": mean_cache_record,
            "refresh_chunk1_cache": {
                "cache_dir": str(Path(args.refresh_cache).resolve()),
                "format": refresh_summary["format"],
                "split": refresh_summary["split"],
                "trajectory": refresh_summary["trajectory"],
                "sample_count": len(refreshed_chunk1_entries),
                "selected_chunk_indices": [1],
                "cache_summary_sha256": sha256_file(refresh_summary_path),
                "manifest_sha256": sha256_file(refresh_manifest_path),
                "source_mean_manifest_sha256": (
                    refresh_summary["source_mean_cache"]["manifest_sha256"]),
                "phase_a_checkpoint_file_sha256": (
                    refresh_summary["phase_a_checkpoint"]["file_sha256"]),
                "phase_a_model_state_sha256": (
                    refresh_summary["phase_a_checkpoint"]["model_state_sha256"]),
            },
        }
    else:
        # Preserve the original flat Phase-A checkpoint/cache schema.
        cache_record = {
            key: value for key, value in mean_cache_record.items()
            if key != "selected_chunk_indices"
        }
    metrics_path = output_dir / "metrics.jsonl"
    append_jsonl(metrics_path, {
        "event": "train_evaluation", "step": 0, **initial_metrics})
    best_record = save_checkpoint(
        output_dir / "best.pt", predictor, 0, initial_metrics,
        args, cache_record, schedule_hash, initialization_record)
    checkpoint_records = []
    history = [{"step": 0, **initial_metrics}]
    evaluation_seconds = initial_metrics["evaluation_wall_seconds"]
    checkpoint_seconds = 0.0

    print(json.dumps({
        "event": "initialized",
        "sample_count": len(samples),
        "model": predictor.architecture,
        "parameter_count": predictor.parameter_count,
        "initial_state_sha256": initial_state_hash,
        "initial_metrics": initial_metrics,
    }, ensure_ascii=False), flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    predictor.train()
    torch.cuda.synchronize(device)
    loop_start = time.perf_counter()
    optimizer_update_seconds = 0.0

    for step, sample_index in enumerate(schedule, 1):
        cached = samples[sample_index]
        sample = sample_on_device(cached, device)
        update_start = time.perf_counter()
        prediction, target, _ = prediction_and_target(predictor, sample)
        loss, _, _, _ = smooth_l1_components(
            prediction, target, args.smooth_l1_beta, args.nonzero_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite predictor gradient")
        optimizer.step()
        optimizer_update_seconds += time.perf_counter() - update_start

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                sample_metrics = one_sample_metrics(
                    prediction.detach(), target, args.smooth_l1_beta,
                    args.nonzero_weight)
            event = {
                "event": "optimization_step",
                "step": step,
                "sample_index": sample_index,
                "sample_path": entries[sample_index]["path"],
                "gradient_norm": float(gradient_norm),
                **sample_metrics,
            }
            append_jsonl(metrics_path, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)

        needs_evaluation = (
            step % args.eval_every == 0
            or step % args.checkpoint_every == 0
            or step == args.steps)
        if not needs_evaluation:
            continue

        metrics = evaluate_train(
            predictor, samples, device,
            args.smooth_l1_beta, args.nonzero_weight)
        evaluation_seconds += metrics["evaluation_wall_seconds"]
        history.append({"step": step, **metrics})
        event = {"event": "train_evaluation", "step": step, **metrics}
        append_jsonl(metrics_path, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

        checkpoint_start = time.perf_counter()
        if metrics["all_element_recovery"] > best_record["all_element_recovery"]:
            best_record = save_checkpoint(
                output_dir / "best.pt", predictor, step, metrics,
                args, cache_record, schedule_hash, initialization_record)
        if step % args.checkpoint_every == 0 and step != args.steps:
            checkpoint_records.append(save_checkpoint(
                output_dir / f"step_{step:06d}.pt", predictor, step, metrics,
                args, cache_record, schedule_hash, initialization_record))
        checkpoint_seconds += time.perf_counter() - checkpoint_start

    torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - loop_start
    final_metrics = history[-1]
    checkpoint_start = time.perf_counter()
    final_record = save_checkpoint(
        output_dir / "final.pt", predictor, args.steps, final_metrics,
        args, cache_record, schedule_hash, initialization_record)
    checkpoint_seconds += time.perf_counter() - checkpoint_start

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total_seconds = time.perf_counter() - total_start
    summary = {
        "experiment": "adaptive_chunk_coding_stage_b_predictor_g2_train_v1",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "purpose": (
                "Phase-B train-only rollout-refresh fitting"
                if phase_b else
                "Train-only predictor fitting before registered G2 validation"),
            "generalization_claim_allowed": False,
            "validation_or_sealed_data_read": False,
            "training_phase": "phase_b" if phase_b else "phase_a",
            "codec_cache_trajectory": (
                "mean chunk0 + Phase-A-predictor chunk1"
                if phase_b else "mean_fill"),
        },
        "args": vars(args),
        "reproducibility": {
            "seed": args.seed,
            "seed_applied_after_set_torch_env": True,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "sample_schedule_sha256": schedule_hash,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "cache": cache_record,
        "initialization": initialization_record,
        "train_data_statistics": data_stats,
        "model": {
            "architecture": predictor.architecture,
            "parameter_count": predictor.parameter_count,
            "initial_state_sha256": initial_state_hash,
            "final_state_sha256": final_record["model_state_sha256"],
        },
        "training": {
            "steps_completed": args.steps,
            "samples_seen": args.steps,
            "effective_epochs": args.steps / len(samples),
            "initial_metrics": initial_metrics,
            "best_metrics": {
                key: value for key, value in best_record.items()
                if key not in ("path", "file_sha256", "model_state_sha256")
            },
            "final_metrics": final_metrics,
            "evaluation_history": history,
        },
        "checkpoints": {
            "best": best_record,
            "periodic": checkpoint_records,
            "final": final_record,
        },
        "compute": {
            "cache_device": str(cache_device),
            "cache_load_wall_seconds": cache_load_seconds,
            "training_loop_wall_seconds": loop_seconds,
            "optimizer_update_wall_seconds_unsynchronized": optimizer_update_seconds,
            "evaluation_wall_seconds_total": evaluation_seconds,
            "checkpoint_wall_seconds_total": checkpoint_seconds,
            "total_wall_seconds": total_seconds,
            "steps_per_training_loop_second": args.steps / loop_seconds,
            "cuda_baseline_allocated_bytes": baseline_allocated,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        },
        "outputs": {
            "metrics_jsonl": str(metrics_path),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    summary_path_out = output_dir / "summary.json"
    summary_path_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "summary": str(summary_path_out),
        "best": best_record,
        "final": final_record,
        "total_wall_seconds": total_seconds,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
