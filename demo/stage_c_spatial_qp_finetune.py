#!/usr/bin/env python3
"""Single-A800 spatial-QP-aware DCVC-UF fine-tuning.

This is a narrow adaptation of the released image and HT-S video checkpoints,
not a replacement for the upstream from-scratch recipe.  A training step uses
one 17-frame crop (I + two P8 coding units), spatially selects the released
QP 8/16/32 scale rows, and optimizes the matching region-weighted RD loss.

The sampler is a pure function of ``seed`` and ``global_step``.  Checkpoints
therefore need only record the last completed step, model weights and optimizer
states to resume without silently changing the future sample stream.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import clip_grad_norm_


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_spatial_quality_codec import select_scale
from src.layers.layers import QuantFunc, get_mse_yuv_rgb, mse_weighted_average
from src.models.image_model import DMCI
from src.models.video_model_ht import DMC, g_frame_delay
from src.utils.common import ModelStructure, get_state_dict, get_training_lambdas
from src.utils.transforms import rgb2ycbcr_np, ycbcr2rgb


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_TO_QP = np.asarray([16, 8, 32], dtype=np.int64)
QP_VALUES = (8, 16, 32)
ACTION_NAMES = {
    ACTION_BASE: "Base",
    ACTION_GENERATE: "Generate",
    ACTION_ENHANCE: "Enhance",
}
TEMPORAL_DISTORTION_WEIGHTS = (0.16, 0.4, 1.5)
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def disk_snapshot() -> dict[str, dict[str, int]]:
    result = {}
    for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
        usage = shutil.disk_usage(name)
        result[name] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial-QP-aware DCVC-UF single-GPU fine-tuning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    train.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    train.add_argument(
        "--reds-root", type=Path,
        default=Path("/root/autodl-fs/DCVC/data/REDS/train_sharp"))
    train.add_argument(
        "--uvg-root", type=Path,
        default=Path("/root/autodl-fs/DCVC/data/UVG_adaptation/samples"))
    train.add_argument("--max-steps", type=int, required=True)
    train.add_argument("--patch-size", type=int, choices=(256, 512), default=512)
    train.add_argument("--learning-rate", type=float, default=2e-6)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--max-grad-norm", type=float, default=0.2)
    train.add_argument("--uvg-probability", type=float, default=0.107)
    train.add_argument("--uniform-probability", type=float, default=0.25)
    train.add_argument(
        "--sampling-mode", choices=("random", "balanced"), default="random",
        help=(
            "random preserves the v1 Bernoulli sampler; balanced fixes the "
            "domain counts, balances the five UVG source sequences, and "
            "allocates uniform-QP rehearsal separately inside each domain"))
    train.add_argument("--seed", type=int, default=21260920)
    train.add_argument("--save-every", type=int, default=25)
    train.add_argument("--log-every", type=int, default=1)
    train.add_argument(
        "--amp-dtype", choices=("bfloat16", "none"), default="bfloat16")
    train.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Disable HT-S activation checkpointing at 512x512")
    train.add_argument(
        "--no-resume", action="store_true",
        help="Fail if output already contains a resume checkpoint")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--expected-steps", type=int, required=True)
    finalize.add_argument(
        "--codec-dir", type=Path, required=True,
        help="Actual spatial-QP encode/fresh-decode validation directory")
    finalize.add_argument("--regression", type=Path, required=True)
    finalize.add_argument("--mode", choices=("smoke", "train"), required=True)

    subparsers.add_parser("self-test")
    args = parser.parse_args(argv)

    if args.command == "train":
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        if args.learning_rate <= 0 or args.weight_decay < 0:
            parser.error("invalid optimizer hyperparameters")
        if args.max_grad_norm <= 0:
            parser.error("--max-grad-norm must be positive")
        if not 0 <= args.uvg_probability <= 1:
            parser.error("--uvg-probability must be in [0,1]")
        if not 0 <= args.uniform_probability <= 1:
            parser.error("--uniform-probability must be in [0,1]")
        if args.save_every < 1 or args.log_every < 1:
            parser.error("save/log intervals must be positive")
    return args


def source_inventory(reds_root: Path, uvg_root: Path) -> dict:
    def records(root: Path, dataset: str) -> list[dict]:
        if not root.is_dir():
            raise FileNotFoundError(root)
        output = []
        for directory in sorted(item for item in root.iterdir() if item.is_dir()):
            frames = sorted(directory.glob("*.png"))
            if len(frames) < 17:
                continue
            with Image.open(frames[0]) as image:
                width, height = image.size
            output.append({
                "dataset": dataset,
                "sequence": directory.name,
                "directory": str(directory.resolve()),
                "frame_count": len(frames),
                "width": width,
                "height": height,
            })
        return output

    reds = records(reds_root, "REDS-train")
    uvg = records(uvg_root, "UVG-adaptation")
    if not reds:
        raise RuntimeError("no REDS training sequences with at least 17 frames")
    if not uvg:
        raise RuntimeError("no UVG adaptation windows with at least 17 frames")
    return {
        "created_utc": utc_now(),
        "roles": {
            "REDS-train": "training; REDS train/000..239",
            "UVG-adaptation": (
                "training; 60 prepared windows from the five UVG adaptation "
                "sequences, excluding the held-out evaluation videos"),
        },
        "reds_root": str(reds_root.resolve()),
        "uvg_root": str(uvg_root.resolve()),
        "reds_sequence_count": len(reds),
        "uvg_window_count": len(uvg),
        "records": [*reds, *uvg],
    }


def frame_paths(record: dict) -> list[Path]:
    return sorted(Path(record["directory"]).glob("*.png"))


def uvg_source_sequence(name: str) -> str:
    parts = name.split("-")
    if len(parts) < 3 or parts[0] != "uvg":
        raise ValueError(f"cannot identify UVG source sequence from {name!r}")
    return parts[1]


def balanced_values(
    values: list[object], count: int, rng: np.random.Generator,
) -> list[object]:
    """Return a shuffled cycle whose value counts differ by at most one."""
    if not values:
        raise ValueError("balanced_values requires at least one value")
    output = []
    while len(output) < count:
        cycle = list(values)
        rng.shuffle(cycle)
        output.extend(cycle)
    return output[:count]


def balanced_sampling_schedule(
    inventory: dict,
    max_steps: int,
    seed: int,
    uvg_probability: float,
    uniform_probability: float,
) -> dict:
    """Build the deterministic domain-balanced v2 sampling schedule."""
    records = inventory["records"]
    reds = sorted(
        (item for item in records if item["dataset"] == "REDS-train"),
        key=lambda item: item["sequence"],
    )
    uvg = sorted(
        (item for item in records if item["dataset"] == "UVG-adaptation"),
        key=lambda item: item["sequence"],
    )
    if not reds or not uvg:
        raise RuntimeError("balanced schedule requires both REDS and UVG records")

    uvg_by_source: dict[str, list[dict]] = defaultdict(list)
    for record in uvg:
        uvg_by_source[uvg_source_sequence(record["sequence"])].append(record)
    if len(uvg_by_source) != 5:
        raise RuntimeError(
            f"balanced v2 expects five UVG source sequences, found "
            f"{sorted(uvg_by_source)}")

    domain_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    record_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    route_rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    uvg_steps = int(round(max_steps * uvg_probability))
    reds_steps = max_steps - uvg_steps
    domains = ["UVG-adaptation"] * uvg_steps + ["REDS-train"] * reds_steps
    domain_rng.shuffle(domains)

    reds_records = balanced_values(reds, reds_steps, record_rng)
    source_names = sorted(uvg_by_source)
    uvg_sources = balanced_values(source_names, uvg_steps, record_rng)
    source_occurrences = Counter(uvg_sources)
    uvg_record_queues = {
        source: deque(balanced_values(
            uvg_by_source[source], count, record_rng))
        for source, count in source_occurrences.items()
    }
    uvg_records = [uvg_record_queues[source].popleft() for source in uvg_sources]

    route_plan = {}
    for domain, domain_steps in (
            ("REDS-train", reds_steps), ("UVG-adaptation", uvg_steps)):
        uniform_steps = int(round(domain_steps * uniform_probability))
        kinds = ["uniform"] * uniform_steps + [
            "mixed"] * (domain_steps - uniform_steps)
        route_rng.shuffle(kinds)
        uniform_actions = deque(balanced_values(
            [ACTION_BASE, ACTION_GENERATE, ACTION_ENHANCE],
            uniform_steps, route_rng))
        route_plan[domain] = deque({
            "route_kind": kind,
            "uniform_action": (
                int(uniform_actions.popleft()) if kind == "uniform" else None),
        } for kind in kinds)

    record_plan = {
        "REDS-train": deque(reds_records),
        "UVG-adaptation": deque(uvg_records),
    }
    entries = []
    for step, domain in enumerate(domains):
        record = record_plan[domain].popleft()
        route = route_plan[domain].popleft()
        entries.append({
            "step": step,
            "dataset": domain,
            "record_sequence": record["sequence"],
            "uvg_source_sequence": (
                uvg_source_sequence(record["sequence"])
                if domain == "UVG-adaptation" else None),
            **route,
        })

    dataset_counts = Counter(entry["dataset"] for entry in entries)
    uniform_counts = Counter(
        entry["dataset"] for entry in entries
        if entry["route_kind"] == "uniform")
    uvg_source_counts = Counter(
        entry["uvg_source_sequence"] for entry in entries
        if entry["dataset"] == "UVG-adaptation")
    uvg_window_counts = Counter(
        entry["record_sequence"] for entry in entries
        if entry["dataset"] == "UVG-adaptation")
    reds_sequence_counts = Counter(
        entry["record_sequence"] for entry in entries
        if entry["dataset"] == "REDS-train")
    return {
        "schema_version": 1,
        "mode": "balanced",
        "seed": seed,
        "max_steps": max_steps,
        "requested_uvg_probability": uvg_probability,
        "requested_uniform_probability_per_domain": uniform_probability,
        "dataset_step_counts": dict(sorted(dataset_counts.items())),
        "uniform_step_counts": dict(sorted(uniform_counts.items())),
        "actual_uniform_probability_per_domain": {
            domain: uniform_counts[domain] / count
            for domain, count in sorted(dataset_counts.items())
        },
        "uvg_source_step_counts": dict(sorted(uvg_source_counts.items())),
        "uvg_window_step_counts": dict(sorted(uvg_window_counts.items())),
        "reds_sequence_step_count_range": [
            min(reds_sequence_counts.values()) if reds_sequence_counts else 0,
            max(reds_sequence_counts.values()) if reds_sequence_counts else 0,
        ],
        "entries": entries,
    }


def load_training_clip(
    inventory: dict,
    step: int,
    seed: int,
    patch_size: int,
    uvg_probability: float,
    scheduled_entry: dict | None = None,
) -> tuple[torch.Tensor, dict, np.random.Generator]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    records = inventory["records"]
    reds = [item for item in records if item["dataset"] == "REDS-train"]
    uvg = [item for item in records if item["dataset"] == "UVG-adaptation"]
    if scheduled_entry is None:
        use_uvg = bool(rng.random() < uvg_probability)
        pool = uvg if use_uvg else reds
        record = pool[int(rng.integers(0, len(pool)))]
    else:
        if int(scheduled_entry["step"]) != step:
            raise RuntimeError("balanced schedule step mismatch")
        matches = [
            item for item in records
            if item["dataset"] == scheduled_entry["dataset"]
            and item["sequence"] == scheduled_entry["record_sequence"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"scheduled source is not unique: {scheduled_entry}")
        record = matches[0]
    paths = frame_paths(record)
    start = int(rng.integers(0, len(paths) - 17 + 1))
    selected = paths[start:start + 17]
    width, height = int(record["width"]), int(record["height"])
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"source {record['sequence']} is {width}x{height}, smaller than "
            f"the requested {patch_size} patch")
    crop_x = int(rng.integers(0, width - patch_size + 1))
    crop_y = int(rng.integers(0, height - patch_size + 1))
    flip = bool(rng.integers(0, 2))

    tensors = []
    for path in selected:
        with Image.open(path) as opened:
            image = opened.convert("RGB").crop((
                crop_x, crop_y, crop_x + patch_size, crop_y + patch_size))
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        yuv = rgb2ycbcr_np(rgb) - 0.5
        tensors.append(torch.from_numpy(yuv).permute(2, 0, 1).contiguous())
    clip = torch.stack(tensors)
    sample = {
        "dataset": record["dataset"],
        "sequence": record["sequence"],
        "source_start": start,
        "source_files": [str(path) for path in selected],
        "crop": {
            "x": crop_x,
            "y": crop_y,
            "width": patch_size,
            "height": patch_size,
        },
        "horizontal_flip": flip,
        "sampling_mode": "balanced" if scheduled_entry is not None else "random",
    }
    if scheduled_entry is not None:
        sample["uvg_source_sequence"] = scheduled_entry["uvg_source_sequence"]
    return clip, sample, rng


def connected_cells(
    rng: np.random.Generator,
    count: int,
    forbidden: set[tuple[int, int]] | None = None,
    grid: int = 4,
) -> list[tuple[int, int]]:
    forbidden = forbidden or set()
    available = [
        (row, column) for row in range(grid) for column in range(grid)
        if (row, column) not in forbidden
    ]
    if count <= 0 or not available:
        return []
    start = available[int(rng.integers(0, len(available)))]
    chosen = [start]
    chosen_set = {start}
    while len(chosen) < min(count, len(available)):
        frontier = []
        for row, column in chosen:
            for other in ((row - 1, column), (row + 1, column),
                          (row, column - 1), (row, column + 1)):
                if (0 <= other[0] < grid and 0 <= other[1] < grid
                        and other not in forbidden and other not in chosen_set):
                    frontier.append(other)
        candidates = frontier or [item for item in available if item not in chosen_set]
        value = candidates[int(rng.integers(0, len(candidates)))]
        chosen.append(value)
        chosen_set.add(value)
    return chosen


def route_action_map(
    rng: np.random.Generator,
    uniform_probability: float,
    forced_kind: str | None = None,
    forced_uniform_action: int | None = None,
) -> tuple[np.ndarray, str]:
    if forced_kind not in (None, "uniform", "mixed"):
        raise ValueError(f"unknown forced route kind: {forced_kind}")
    use_uniform = (
        forced_kind == "uniform"
        if forced_kind is not None else rng.random() < uniform_probability)
    if use_uniform:
        action = (
            int(forced_uniform_action)
            if forced_uniform_action is not None else int(rng.integers(0, 3)))
        if action not in ACTION_NAMES:
            raise ValueError(f"invalid uniform action: {action}")
        return np.full((4, 4), action, dtype=np.int64), (
            f"uniform-{ACTION_NAMES[action].lower()}")

    actions = np.full((4, 4), ACTION_BASE, dtype=np.int64)
    generate_count = int(rng.integers(1, 5))
    generate = connected_cells(rng, generate_count)
    for row, column in generate:
        actions[row, column] = ACTION_GENERATE
    enhance_count = int(rng.integers(3, 9))
    enhance = connected_cells(rng, enhance_count, set(generate))
    for row, column in enhance:
        actions[row, column] = ACTION_ENHANCE
    return actions, "mixed-connected"


def mutate_route(
    actions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    output = actions.copy()
    if np.all(output == output.flat[0]):
        return output, 0
    change_count = int(rng.integers(0, 4))
    positions = rng.choice(16, size=change_count, replace=False)
    for position in positions:
        row, column = divmod(int(position), 4)
        alternatives = [value for value in range(3) if value != output[row, column]]
        output[row, column] = alternatives[int(rng.integers(0, 2))]
    return output, change_count


def qp_map_from_route(actions: np.ndarray, patch_size: int) -> torch.Tensor:
    syntax_grid = patch_size // 64
    if syntax_grid % 4:
        raise ValueError("patch syntax grid must be divisible by the 4x4 router grid")
    repeat = syntax_grid // 4
    values = ACTION_TO_QP[actions]
    values = np.repeat(np.repeat(values, repeat, axis=0), repeat, axis=1)
    return torch.from_numpy(values.copy())[None].long()


def action_counts(actions: np.ndarray) -> dict[str, int]:
    return {
        ACTION_NAMES[action]: int(np.count_nonzero(actions == action))
        for action in ACTION_NAMES
    }


def mixed_z_bits(
    model: DMCI | DMC,
    z_for_bit: torch.Tensor,
    qp_map: torch.Tensor,
) -> torch.Tensor:
    z_qp = F.interpolate(
        qp_map[:, None].float(), size=z_for_bit.shape[-2:], mode="nearest")
    total = torch.zeros((), device=z_for_bit.device, dtype=torch.float32)
    for qp in QP_VALUES:
        index = torch.tensor([qp], device=z_for_bit.device, dtype=torch.long)
        bits = model.get_z_bits(z_for_bit, index).float()
        total = total + torch.sum(bits * (z_qp == qp).to(bits.dtype))
    return total


def spatial_i_forward(
    model: DMCI,
    x: torch.Tensor,
    qp_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_size = (x.shape[-2] // 8, x.shape[-1] // 8)
    q_enc = select_scale(model.q_scale_enc, qp_map, feature_size, "nearest")
    y = model.enc(x, q_enc)
    q_y_enc = select_scale(model.q_scale_y_enc, qp_map, y.shape[-2:], "nearest")
    q_y_dec = select_scale(model.q_scale_y_dec, qp_map, y.shape[-2:], "nearest")
    z = model.hyper_enc(y)
    z_hat = QuantFunc.apply(z)
    params = model.y_prior_fusion(model.hyper_dec(z_hat))
    params = params[:, :, :y.shape[-2], :y.shape[-1]]
    y_res, _y_q, y_hat, scales = model.forward_prior_4x(
        y, q_y_enc, q_y_dec, params,
        model.y_spatial_prior_reduction,
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
        model.y_spatial_prior,
    )
    q_dec = select_scale(model.q_scale_dec, qp_map, feature_size, "nearest")
    x_hat = model.dec(y_hat, q_dec)
    bits_y = torch.sum(model.get_y_bits(model.add_noise(y_res), scales).float())
    bits_z = mixed_z_bits(model, model.add_noise(z), qp_map)
    return x_hat, bits_y + bits_z, bits_y, bits_z


def spatial_p_forward(
    model: DMC,
    x: torch.Tensor,
    qp_map: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.apply_feature_adaptor()
    feature_size = model.ctx.shape[-2:]
    q_enc = select_scale(model.q_encoder, qp_map, feature_size, "nearest")
    y = model.encoder(x, model.ctx, q_enc)
    z = model.hyper_encoder(y)
    z_hat = QuantFunc.apply(z)
    q_feature = select_scale(
        model.q_feature, qp_map, model.memory.shape[-2:], "nearest")
    params = model.res_prior_param_decoder(z_hat, model.memory, q_feature)
    y_res, _y_q, y_hat, scales = model.forward_prior_4x(
        y, None, None, params,
        model.y_spatial_prior_reduction,
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
        model.y_spatial_prior,
        spatial_prior_has_scales=False,
    )
    q_dec = select_scale(model.q_decoder, qp_map, feature_size, "nearest")
    x_hat, feature = model.get_recon_and_feature(y_hat, model.ctx, q_dec)
    model.set_ref_feature(feature, False)
    bits_y = torch.sum(model.get_y_bits(model.add_noise(y_res), scales).float())
    bits_z = mixed_z_bits(model, model.add_noise(z), qp_map)
    return x_hat, bits_y + bits_z, bits_y, bits_z


def spatial_weighted_distortion(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    qp_map: torch.Tensor,
    lambda_table: torch.Tensor,
) -> torch.Tensor:
    """Region-wise version of the upstream YUV/RGB distortion.

    With a uniform QP map this is numerically the upstream distortion times
    that QP's lambda.  With a mixed map, each region gets its own lambda before
    area averaging.
    """
    if source.shape[0] != 1 or reconstruction.shape[0] != 1:
        raise ValueError("the current single-A800 trainer requires batch size one")
    full_qp = F.interpolate(
        qp_map[:, None].float(), size=source.shape[-2:], mode="nearest")[:, 0]
    error_yuv = (source - reconstruction).square().float()
    source_rgb = ycbcr2rgb(source, clamp=False)
    reconstruction_rgb = ycbcr2rgb(reconstruction, clamp=False)
    error_rgb = (source_rgb - reconstruction_rgb).square().float()
    pixel_count = source.shape[-2] * source.shape[-1]
    total = torch.zeros((), device=source.device, dtype=torch.float32)
    for qp in QP_VALUES:
        mask = (full_qp == qp).float()
        count = torch.sum(mask)
        if not bool(count > 0):
            continue
        mask_channel = mask[:, None]
        mse_yuv = torch.sum(error_yuv * mask_channel, dim=(2, 3)) / count
        mse_y, mse_u, mse_v = mse_yuv[:, 0], mse_yuv[:, 1], mse_yuv[:, 2]
        combined_yuv = torch.exp(
            0.0833 * (
                10 * torch.log(torch.clamp_min(mse_y, 1e-6))
                + torch.log(torch.clamp_min(mse_u, 1e-6))
                + torch.log(torch.clamp_min(mse_v, 1e-6)))
        ) * 3.0
        combined_rgb = torch.sum(error_rgb * mask_channel) / count
        distortion = combined_yuv * 0.8 + combined_rgb * 0.2
        total = total + (count / pixel_count) * lambda_table[qp] * distortion.mean()
    return total


def p_group_distortion(
    sources: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    qp_map: torch.Tensor,
    lambda_table: torch.Tensor,
) -> torch.Tensor:
    values = [
        spatial_weighted_distortion(source, reconstruction, qp_map, lambda_table)
        for source, reconstruction in zip(sources, reconstructions)
    ]
    if len(values) != g_frame_delay:
        raise ValueError("HT-S P unit must contain eight frames")
    short, medium, long = TEMPORAL_DISTORTION_WEIGHTS
    return (
        (values[0] + values[2] + values[4] + values[6]) * short
        + (values[1] + values[3] + values[5]) * medium
        + values[7] * long
    )


def finite_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
    if not bool(torch.isfinite(loss.detach())) or not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"non-finite training value: loss={loss.detach().item()}, norm={norm.item()}")
    optimizer.step()
    return float(norm.item())


def detach_dpb(model: DMC) -> None:
    for name in ("ref_feature", "memory", "ctx"):
        value = getattr(model, name)
        if value is not None:
            setattr(model, name, value.detach())


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def immutable_config(args: argparse.Namespace) -> dict:
    config = {
        "schema_version": 1,
        "model_path_i": str(args.model_path_i.resolve()),
        "model_path_p": str(args.model_path_p.resolve()),
        "model_i_sha256": sha256_file(args.model_path_i),
        "model_p_sha256": sha256_file(args.model_path_p),
        "reds_root": str(args.reds_root.resolve()),
        "uvg_root": str(args.uvg_root.resolve()),
        "patch_size": args.patch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "uvg_probability": args.uvg_probability,
        "uniform_probability": args.uniform_probability,
        "seed": args.seed,
        "amp_dtype": args.amp_dtype,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
        "spatial_scale_interpolation": "nearest",
        "frames_per_step": 17,
        "p_units_per_step": 2,
        "full_model_finetuning": True,
        "single_gpu": True,
    }
    if args.sampling_mode == "balanced":
        config.update({
            "sampling_mode": "balanced",
            "sampling_schedule_schema_version": 1,
            "sampling_schedule_steps": args.max_steps,
        })
    return config


def save_resume(
    path: Path,
    completed_steps: int,
    i_net: DMCI,
    p_net: DMC,
    optimizer_i: torch.optim.Optimizer,
    optimizer_p: torch.optim.Optimizer,
    config: dict,
) -> float:
    started = time.perf_counter()
    atomic_torch_save(path, {
        "schema_version": 1,
        "completed_steps": completed_steps,
        "saved_utc": utc_now(),
        "config": config,
        "image_model": i_net.state_dict(),
        "video_model": p_net.state_dict(),
        "optimizer_i": optimizer_i.state_dict(),
        "optimizer_p": optimizer_p.state_dict(),
    })
    return time.perf_counter() - started


def export_models(output_dir: Path, i_net: DMCI, p_net: DMC) -> dict:
    checkpoint_dir = output_dir / "checkpoints"
    image_path = checkpoint_dir / "image_model.pth.tar"
    video_path = checkpoint_dir / "video_model_hts.pth.tar"
    atomic_torch_save(image_path, {"state_dict": i_net.state_dict()})
    atomic_torch_save(video_path, {"state_dict": p_net.state_dict()})
    return {
        "image_model": str(image_path.resolve()),
        "image_model_bytes": image_path.stat().st_size,
        "image_model_sha256": sha256_file(image_path),
        "video_model": str(video_path.resolve()),
        "video_model_bytes": video_path.stat().st_size,
        "video_model_sha256": sha256_file(video_path),
    }


def handle_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def train_main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = immutable_config(args)
    config_path = args.output_dir / "training_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "immutable training configuration differs from the existing run")
    else:
        atomic_json(config_path, config)

    inventory_path = args.output_dir / "data_manifest.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        inventory = source_inventory(args.reds_root, args.uvg_root)
        atomic_json(inventory_path, inventory)

    sampling_schedule = None
    if args.sampling_mode == "balanced":
        sampling_schedule = balanced_sampling_schedule(
            inventory, args.max_steps, args.seed,
            args.uvg_probability, args.uniform_probability)
        schedule_path = args.output_dir / "sampling_schedule.json"
        if schedule_path.exists():
            existing_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            if existing_schedule != sampling_schedule:
                raise RuntimeError(
                    "balanced sampling schedule differs from the existing run")
        else:
            atomic_json(schedule_path, sampling_schedule)

    i_net = DMCI()
    i_net.load_state_dict(get_state_dict(str(args.model_path_i)))
    p_net = DMC(ModelStructure.HTS)
    p_net.load_state_dict(get_state_dict(str(args.model_path_p)))
    i_net = i_net.to(device).to(memory_format=torch.channels_last).train()
    p_net = p_net.to(device).to(memory_format=torch.channels_last).train()
    p_net.set_use_ckpt(
        args.patch_size >= 512 and not args.no_gradient_checkpointing)
    optimizer_i = torch.optim.AdamW(
        i_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_p = torch.optim.AdamW(
        p_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    resume_path = args.output_dir / "checkpoints" / "latest.pt"
    completed_steps = 0
    if resume_path.exists():
        if args.no_resume:
            raise RuntimeError(f"resume checkpoint already exists: {resume_path}")
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume["config"] != config:
            raise RuntimeError("resume checkpoint configuration mismatch")
        i_net.load_state_dict(resume["image_model"])
        p_net.load_state_dict(resume["video_model"])
        optimizer_i.load_state_dict(resume["optimizer_i"])
        optimizer_p.load_state_dict(resume["optimizer_p"])
        optimizer_to(optimizer_i, device)
        optimizer_to(optimizer_p, device)
        completed_steps = int(resume["completed_steps"])
        print(f"resume from completed step {completed_steps}", flush=True)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    lambda_values = get_training_lambdas([1, 768], DMCI.qp_num())
    lambda_table = torch.as_tensor(
        lambda_values, device=device, dtype=torch.float32)
    amp_enabled = args.amp_dtype == "bfloat16"
    def autocast():
        if amp_enabled:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    log_path = args.output_dir / "training_log.jsonl"
    run_started = time.time()
    invocation_started = time.time()
    previous_summary_path = args.output_dir / "training_summary.json"
    previous_summary = (
        json.loads(previous_summary_path.read_text(encoding="utf-8"))
        if previous_summary_path.is_file() else {})
    peak_allocated = int(previous_summary.get("peak_cuda_allocated_bytes", 0))
    last_record = None
    last_saved_steps = completed_steps if resume_path.exists() else -1
    complete_marker = args.output_dir / "training.complete"
    if completed_steps < args.max_steps:
        complete_marker.unlink(missing_ok=True)

    for step in range(completed_steps, args.max_steps):
        if STOP_REQUESTED:
            break
        step_started = time.perf_counter()
        scheduled_entry = (
            sampling_schedule["entries"][step]
            if sampling_schedule is not None else None)
        clip, sample, rng = load_training_clip(
            inventory, step, args.seed, args.patch_size, args.uvg_probability,
            scheduled_entry=scheduled_entry)
        first_actions, map_kind = route_action_map(
            rng, args.uniform_probability,
            forced_kind=(
                scheduled_entry["route_kind"]
                if scheduled_entry is not None else None),
            forced_uniform_action=(
                scheduled_entry["uniform_action"]
                if scheduled_entry is not None else None),
        )
        second_actions, changed_tiles = mutate_route(first_actions, rng)
        qp_first = qp_map_from_route(first_actions, args.patch_size).to(device)
        qp_second = qp_map_from_route(second_actions, args.patch_size).to(device)
        clip = clip.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        frames = [clip[index:index + 1] for index in range(17)]

        torch.cuda.reset_peak_memory_stats(device)
        with autocast():
            intra, bits_i, bits_y_i, bits_z_i = spatial_i_forward(
                i_net, frames[0], qp_first)
            distortion_i = spatial_weighted_distortion(
                frames[0], intra, qp_first, lambda_table)
            bpp_i = bits_i / (args.patch_size * args.patch_size)
            loss_i = distortion_i + bpp_i
        grad_i = finite_step(
            loss_i, i_net, optimizer_i, args.max_grad_norm)

        p_net.clear_dpb()
        p_net.ref_feature = F.pixel_unshuffle(intra.detach(), 8)
        p_records = []
        for unit_index, (start, qp_map) in enumerate(
                ((1, qp_first), (9, qp_second)), start=1):
            sources = frames[start:start + g_frame_delay]
            p_input = torch.cat(sources, dim=1).to(memory_format=torch.channels_last)
            with autocast():
                reconstructions, bits_p, bits_y_p, bits_z_p = spatial_p_forward(
                    p_net, p_input, qp_map)
                distortion_p = p_group_distortion(
                    sources, reconstructions, qp_map, lambda_table)
                bpp_p = bits_p / (args.patch_size * args.patch_size)
                loss_p = distortion_p + bpp_p
            grad_p = finite_step(
                loss_p, p_net, optimizer_p, args.max_grad_norm)
            detach_dpb(p_net)
            p_records.append({
                "unit_index": unit_index,
                "loss": float(loss_p.detach().item()),
                "distortion_term": float(distortion_p.detach().item()),
                "estimated_bpp": float(bpp_p.detach().item()),
                "estimated_bits_y": float(bits_y_p.detach().item()),
                "estimated_bits_z": float(bits_z_p.detach().item()),
                "gradient_norm_before_clip": grad_p,
            })

        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        peak_step = int(torch.cuda.max_memory_allocated(device))
        peak_allocated = max(peak_allocated, peak_step)
        completed_steps = step + 1
        last_record = {
            "completed_step": completed_steps,
            "utc": utc_now(),
            "sample": sample,
            "sampling_schedule": scheduled_entry,
            "map_kind": map_kind,
            "first_action_counts": action_counts(first_actions),
            "second_action_counts": action_counts(second_actions),
            "changed_router_tiles_between_p_units": changed_tiles,
            "image": {
                "loss": float(loss_i.detach().item()),
                "distortion_term": float(distortion_i.detach().item()),
                "estimated_bpp": float(bpp_i.detach().item()),
                "estimated_bits_y": float(bits_y_i.detach().item()),
                "estimated_bits_z": float(bits_z_i.detach().item()),
                "gradient_norm_before_clip": grad_i,
            },
            "p_units": p_records,
            "mean_p_loss": float(np.mean([item["loss"] for item in p_records])),
            "step_seconds": step_seconds,
            "peak_cuda_allocated_bytes": peak_step,
        }
        if completed_steps % args.log_every == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_record, ensure_ascii=False) + "\n")
            print(json.dumps({
                "step": completed_steps,
                "source": f"{sample['dataset']}/{sample['sequence']}",
                "map": map_kind,
                "loss_i": last_record["image"]["loss"],
                "loss_p": last_record["mean_p_loss"],
                "seconds": step_seconds,
                "peak_gib": peak_step / (1024 ** 3),
            }, ensure_ascii=False), flush=True)
        atomic_json(args.output_dir / "progress.json", {
            "status": "running",
            "completed_steps": completed_steps,
            "target_steps": args.max_steps,
            "latest": last_record,
        })

        should_save = (
            completed_steps % args.save_every == 0
            or completed_steps == args.max_steps
            or STOP_REQUESTED
        )
        if should_save:
            checkpoint_seconds = save_resume(
                resume_path, completed_steps, i_net, p_net,
                optimizer_i, optimizer_p, config)
            last_saved_steps = completed_steps
            print(
                f"saved resume checkpoint at step {completed_steps} "
                f"in {checkpoint_seconds:.2f}s", flush=True)

    if completed_steps and last_saved_steps != completed_steps:
        save_resume(
            resume_path, completed_steps, i_net, p_net,
            optimizer_i, optimizer_p, config)

    exports = export_models(args.output_dir, i_net, p_net)
    complete = completed_steps >= args.max_steps
    summary = {
        "experiment": "single-A800 spatial-QP-aware DCVC-UF fine-tuning",
        "status": "complete" if complete else "interrupted-resumable",
        "completed_utc": utc_now(),
        "git_commit_at_execution": git_commit(),
        "completed_steps": completed_steps,
        "target_steps": args.max_steps,
        "frames_seen": completed_steps * 17,
        "i_updates": completed_steps,
        "p8_updates": completed_steps * 2,
        "invocation_seconds": time.time() - invocation_started,
        "wall_seconds_since_invocation_start": time.time() - run_started,
        "peak_cuda_allocated_bytes": peak_allocated,
        "last_record": last_record,
        "config": config,
        "data": {
            "reds_sequence_count": inventory["reds_sequence_count"],
            "uvg_window_count": inventory["uvg_window_count"],
            "uvg_sampling_probability": args.uvg_probability,
            "sampling_mode": args.sampling_mode,
            "balanced_schedule_summary": (
                {
                    key: value for key, value in sampling_schedule.items()
                    if key != "entries"
                } if sampling_schedule is not None else None),
        },
        "exports": exports,
        "resume_checkpoint": str(resume_path.resolve()),
        "resume_checkpoint_bytes": resume_path.stat().st_size,
        "mounts": disk_snapshot(),
        "scientific_boundary": {
            "goal": (
                "verify and adapt the codec to mixed spatial QP; this training "
                "summary alone is not a quality comparison"),
            "uniform_qp_rehearsal": True,
            "actual_stream_validation_required": True,
            "multi_gpu": False,
        },
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {
        "status": summary["status"],
        "completed_steps": completed_steps,
        "target_steps": args.max_steps,
        "latest": last_record,
    })
    if complete:
        complete_marker.write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def finalize_main(args: argparse.Namespace) -> None:
    training_path = args.run_root / "training" / "training_summary.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    encode = json.loads((args.codec_dir / "encode_summary.json").read_text())
    decode = json.loads((args.codec_dir / "decode_summary.json").read_text())
    regression = json.loads(args.regression.read_text(encoding="utf-8"))
    if training["completed_steps"] != args.expected_steps:
        raise RuntimeError("training step count differs from the requested run")
    if not regression["pixel_exact"]:
        raise RuntimeError("fine-tuned stream failed fresh-decode regression")
    files = [path for path in args.run_root.rglob("*") if path.is_file()]
    summary = {
        "experiment": "A800 spatial-QP-aware DCVC-UF fine-tuning",
        "mode": args.mode,
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit_at_execution": git_commit(),
        "single_gpu": True,
        "training_steps": training["completed_steps"],
        "frames_seen": training["frames_seen"],
        "peak_cuda_allocated_bytes": training["peak_cuda_allocated_bytes"],
        "actual_validation_stream_bytes": encode["stream_bytes"],
        "fresh_decode_pixel_exact": regression["pixel_exact"],
        "fresh_decode_max_abs_channel_error": regression[
            "maximum_absolute_channel_error"],
        "validation_frames": encode["frames"],
        "time_varying_action_maps": encode["time_varying_action_maps"],
        "training_summary": str(training_path.resolve()),
        "validation_encode_summary": str(
            (args.codec_dir / "encode_summary.json").resolve()),
        "validation_decode_summary": str(
            (args.codec_dir / "decode_summary.json").resolve()),
        "exports": training["exports"],
        "ordinary_file_count": len(files),
        "ordinary_file_bytes": sum(path.stat().st_size for path in files),
        "mounts": disk_snapshot(),
        "scientific_role": (
            "training/infrastructure smoke" if args.mode == "smoke" else
            "single-card spatial-QP codec adaptation; quality must be evaluated separately"),
        "decoder_used_only_stream_and_finetuned_checkpoints": True,
        "decode_frame_directory": decode["fresh_decode_dir"],
    }
    atomic_json(args.run_root / "run_summary.json", summary)
    (args.run_root / "run.complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def self_test() -> None:
    rng_a = np.random.default_rng(np.random.SeedSequence([1234, 7]))
    rng_b = np.random.default_rng(np.random.SeedSequence([1234, 7]))
    actions_a, kind_a = route_action_map(rng_a, 0.0)
    actions_b, kind_b = route_action_map(rng_b, 0.0)
    assert kind_a == kind_b == "mixed-connected"
    assert np.array_equal(actions_a, actions_b)
    assert qp_map_from_route(actions_a, 256).shape == (1, 4, 4)
    assert qp_map_from_route(actions_a, 512).shape == (1, 8, 8)

    synthetic_inventory = {
        "records": [
            {"dataset": "REDS-train", "sequence": f"{index:03d}"}
            for index in range(240)
        ] + [
            {
                "dataset": "UVG-adaptation",
                "sequence": f"uvg-{source}-f{window:03d}-center",
            }
            for source in ("beauty", "bosphorus", "honeybee", "jockey", "shakendry")
            for window in range(12)
        ],
    }
    schedule_a = balanced_sampling_schedule(
        synthetic_inventory, 1000, 4321, 0.25, 0.25)
    schedule_b = balanced_sampling_schedule(
        synthetic_inventory, 1000, 4321, 0.25, 0.25)
    assert schedule_a == schedule_b
    assert schedule_a["dataset_step_counts"] == {
        "REDS-train": 750, "UVG-adaptation": 250}
    assert set(schedule_a["uvg_source_step_counts"].values()) == {50}
    assert schedule_a["uniform_step_counts"] == {
        "REDS-train": 188, "UVG-adaptation": 62}
    assert sum(schedule_a["uniform_step_counts"].values()) == 250

    torch.manual_seed(3)
    source = torch.randn(1, 3, 16, 16) * 0.1
    reconstruction = source + torch.randn_like(source) * 0.02
    lambdas = torch.as_tensor(get_training_lambdas([1, 768], 64)).float()
    qp = 16
    qp_map = torch.full((1, 4, 4), qp, dtype=torch.long)
    spatial = spatial_weighted_distortion(
        source, reconstruction, qp_map, lambdas)
    mse_yuv, mse_rgb = get_mse_yuv_rgb(source, reconstruction)
    upstream = mse_weighted_average(mse_yuv, mse_rgb, 16 * 16).mean() * lambdas[qp]
    if not torch.allclose(spatial, upstream, rtol=2e-5, atol=2e-7):
        raise AssertionError(f"uniform loss mismatch: {spatial} != {upstream}")
    print(json.dumps({
        "status": "passed",
        "deterministic_step_sampler": True,
        "deterministic_balanced_schedule": True,
        "balanced_schedule_uvg_steps": 250,
        "balanced_schedule_uvg_steps_per_source": 50,
        "balanced_schedule_uniform_steps": 250,
        "route_grid": [4, 4],
        "syntax_grids": {"256_patch": [4, 4], "512_patch": [8, 8]},
        "uniform_spatial_loss_matches_upstream": True,
        "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
    }, indent=2))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train_main(args)
    elif args.command == "finalize":
        finalize_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main(sys.argv[1:])
