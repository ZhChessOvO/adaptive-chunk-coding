#!/usr/bin/env python3
"""Two-pass encoder-side Base-probe controller for low-budget routing v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_controller import (
    TARGET_NAMES,
    UtilityMLP,
    atomic_json,
    read_jsonl,
    read_teacher_rows,
    solve_actions,
)
from demo.stage_c_a800_low_budget_v2 import (
    COMPUTE_TIME_WEIGHT,
    ENSEMBLE_SIZE,
    FOLD_COUNT,
    FRAGMENT_PENALTY,
    LOW_BYTE_RATIO,
    LOW_GENERATE_TILES,
    PSNR_RISK_WEIGHT,
    TEMPORAL_RISK_WEIGHT,
    atomic_npz,
    fold_for_sequence,
    grouped_indices,
    normalize_from_train,
    solve_actions_pruned,
)
from demo.stage_c_a800_teacher import (
    FEATURE_NAMES,
    SpatialLPIPS,
    load_source,
    local_quality,
    source_features,
)
from demo.stage_c_evaluate_seedvr2_gate import load_pngs
from demo.stage_c_three_path_roi_probe import (
    ACTION_BASE,
    ACTION_ENHANCE,
    ACTION_GENERATE,
    TILE_HEADER,
    action_counts,
    tile_boxes,
)


DEFAULT_SEED = 20260918
DIRECT_TARGET_NAMES = (
    "generate_utility",
    "enhance_lpips_gain_vs_base",
    "enhance_fallback_extra_on_disk_bytes",
)
BASE_PROBE_NAMES = (
    "base_lpips_alex",
    "base_psnr_db",
    "base_rgb_mse",
    "base_temporal_delta_mae",
)
BASE_PROBE_FEATURE_NAMES = (
    tuple(FEATURE_NAMES)
    + BASE_PROBE_NAMES
    + tuple(f"sample_delta_{name}" for name in BASE_PROBE_NAMES)
    + tuple(f"sample_z_{name}" for name in BASE_PROBE_NAMES)
)
GAIN_PENALTIES = (0.0, 0.25, 0.5)


class IndependentDirectMLP(nn.Module):
    def __init__(self, input_dim: int, width: int = 32) -> None:
        super().__init__()
        self.towers = nn.ModuleList([
            UtilityMLP(input_dim, 1, width)
            for _ in DIRECT_TARGET_NAMES
        ])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat([tower(value) for tower in self.towers], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base-probe direct-utility low-budget controller v3")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    cross_validate = subparsers.add_parser("cross-validate")
    add_training_inputs(cross_validate)
    cross_validate.add_argument("--output-dir", type=Path, required=True)

    train = subparsers.add_parser("train")
    add_training_inputs(train)
    train.add_argument("--selection-summary", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)

    route = subparsers.add_parser("route")
    route.add_argument("--sample-manifest", type=Path, required=True)
    route.add_argument("--base-probe-root", type=Path, required=True)
    route.add_argument("--checkpoint", type=Path, required=True)
    route.add_argument("--output-dir", type=Path, required=True)
    route.add_argument("--lpips-batch-size", type=int, default=4)
    route.add_argument("--cuda-idx", type=int, default=0)
    route.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mode in {"cross-validate", "train"}:
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("epochs and batch size must be positive")
        if args.ensemble_size != ENSEMBLE_SIZE:
            parser.error(
                f"the preregistered ensemble size is fixed at {ENSEMBLE_SIZE}")
    elif args.lpips_batch_size < 1:
        parser.error("LPIPS batch size must be positive")
    return args


def add_training_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-teacher-manifest", type=Path, required=True)
    parser.add_argument("--train-roi-cost-manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cuda-idx", type=int, default=0)


def direct_targets(values: np.ndarray) -> np.ndarray:
    index = {name: TARGET_NAMES.index(name) for name in TARGET_NAMES}
    generate = (
        values[:, index["generate_lpips_gain_vs_base"]]
        - TEMPORAL_RISK_WEIGHT * np.maximum(
            values[:, index["generate_temporal_risk_vs_base"]], 0)
        - PSNR_RISK_WEIGHT * np.maximum(
            -values[:, index["generate_psnr_delta_db_vs_base"]], 0)
        - COMPUTE_TIME_WEIGHT * np.maximum(
            values[:, index[
                "generate_roi_seconds_measured_geometry_class"]], 0)
    )
    return np.stack([
        generate,
        values[:, index["enhance_lpips_gain_vs_base"]],
        values[:, index["enhance_fallback_extra_on_disk_bytes"]],
    ], axis=1).astype(np.float32)


def read_base_probe_training_features(
    teacher_manifest: Path, metadata: list[dict],
) -> np.ndarray:
    manifest = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    by_sample = {}
    for entry in manifest["entries"]:
        sample = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        rows = []
        for region in sorted(sample["regions"], key=lambda item: item["index"]):
            base = region["candidates"]["Base"]
            rows.append([
                base["lpips_alex"],
                base["psnr_db"],
                base["rgb_mse"],
                base["temporal_delta_mae"],
            ])
        if len(rows) != 16:
            raise ValueError(f"incomplete Base probe for {entry['sample_id']}")
        by_sample[entry["sample_id"]] = np.asarray(rows, dtype=np.float32)
    output = np.empty((len(metadata), len(BASE_PROBE_NAMES)), dtype=np.float32)
    for indices in grouped_indices(metadata):
        sample_id = metadata[int(indices[0])]["sample_id"]
        if sample_id not in by_sample:
            raise ValueError(f"Base probe is missing for {sample_id}")
        output[indices] = by_sample[sample_id]
    return output


def single_sample_base_probe_features(
    raw_features: np.ndarray, probe: np.ndarray,
) -> np.ndarray:
    if raw_features.shape != (16, len(FEATURE_NAMES)):
        raise ValueError(f"unexpected raw feature shape: {raw_features.shape}")
    if probe.shape != (16, len(BASE_PROBE_NAMES)):
        raise ValueError(f"unexpected Base probe shape: {probe.shape}")
    center = probe.mean(axis=0)
    scale = probe.std(axis=0)
    scale[scale < 1e-8] = 1.0
    delta = probe - center
    return np.concatenate(
        [raw_features, probe, delta, delta / scale], axis=1
    ).astype(np.float32)


def engineer_features(
    raw_features: np.ndarray,
    probes: np.ndarray,
    metadata: list[dict],
    kind: str,
) -> np.ndarray:
    if kind == "raw-direct":
        return raw_features.astype(np.float32, copy=True)
    if kind != "base-probe-direct":
        raise ValueError(f"unknown feature kind: {kind}")
    output = np.empty(
        (len(raw_features), len(BASE_PROBE_FEATURE_NAMES)), dtype=np.float32)
    for indices in grouped_indices(metadata):
        output[indices] = single_sample_base_probe_features(
            raw_features[indices], probes[indices])
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    prediction_x: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, float]:
    set_seed(seed)
    model = IndependentDirectMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predicted = model(batch_x)
            loss = torch.mean((predicted - batch_y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        final_loss = total / count
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(prediction_x), 4096):
            batch = torch.from_numpy(
                prediction_x[start:start + 4096]).to(device)
            chunks.append(model(batch).cpu().numpy())
    state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    return state, np.concatenate(chunks), final_loss


def out_of_fold_predictions(
    raw_features: np.ndarray,
    probes: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict],
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features = engineer_features(raw_features, probes, metadata, kind)
    folds = np.asarray([
        fold_for_sequence(item["sequence"]) for item in metadata
    ], dtype=np.int64)
    output = np.empty_like(targets)
    disagreement = np.empty_like(targets)
    records = []
    for fold in range(FOLD_COUNT):
        train_mask = folds != fold
        validation_mask = folds == fold
        normalized_x, _, _ = normalize_from_train(
            features[train_mask], features)
        normalized_y, y_mean, y_scale = normalize_from_train(
            targets[train_mask], targets)
        predictions = []
        losses = []
        for member in range(args.ensemble_size):
            seed = args.seed + fold * 1000 + member
            _, normalized_prediction, final_loss = fit_one(
                normalized_x[train_mask],
                normalized_y[train_mask],
                normalized_x[validation_mask],
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                seed,
                device,
            )
            predictions.append(normalized_prediction * y_scale + y_mean)
            losses.append(final_loss)
            print(json.dumps({
                "stage": "low-budget-v3-cross-validation",
                "feature_kind": kind,
                "fold": fold,
                "member": member,
                "seed": seed,
                "final_standardized_mse": final_loss,
            }), flush=True)
        stacked = np.stack(predictions)
        output[validation_mask] = stacked.mean(axis=0)
        disagreement[validation_mask] = stacked.std(axis=0)
        records.append({
            "fold": fold,
            "train_region_count": int(np.count_nonzero(train_mask)),
            "validation_region_count": int(np.count_nonzero(validation_mask)),
            "train_sequence_count": len(set(
                item["sequence"] for item, keep in zip(metadata, train_mask)
                if keep)),
            "validation_sequence_count": len(set(
                item["sequence"]
                for item, keep in zip(metadata, validation_mask) if keep)),
            "member_final_losses": losses,
        })
    return output, disagreement, records


def direct_to_solver(values: np.ndarray) -> np.ndarray:
    output = np.zeros((len(values), len(TARGET_NAMES)), dtype=np.float32)
    output[:, TARGET_NAMES.index("generate_lpips_gain_vs_base")] = values[:, 0]
    output[:, TARGET_NAMES.index("enhance_lpips_gain_vs_base")] = values[:, 1]
    output[:, TARGET_NAMES.index(
        "enhance_fallback_extra_on_disk_bytes")] = values[:, 2]
    return output


def solve_direct(
    values: np.ndarray, byte_budget: int,
) -> tuple[np.ndarray, dict]:
    return solve_actions_pruned(
        direct_to_solver(values),
        byte_budget,
        LOW_GENERATE_TILES,
        0.0,
        0.0,
        0.0,
        FRAGMENT_PENALTY,
    )


def direct_route_utility(actions: np.ndarray, targets: np.ndarray) -> float:
    return float(
        targets[actions == ACTION_GENERATE, 0].sum()
        + targets[actions == ACTION_ENHANCE, 1].sum())


def verify_solver(values: np.ndarray, metadata: list[dict], limit: int = 32) -> int:
    checked = 0
    for indices in grouped_indices(metadata)[:limit]:
        sample = values[indices]
        costs = np.maximum(
            np.rint(sample[:, 2]).astype(np.int64), TILE_HEADER.size + 1)
        budget = int(round(LOW_BYTE_RATIO * float(costs.sum())))
        solver_values = direct_to_solver(sample)
        reference, _ = solve_actions(
            solver_values, budget, LOW_GENERATE_TILES,
            0.0, 0.0, 0.0, FRAGMENT_PENALTY)
        pruned, _ = solve_direct(sample, budget)
        if not np.array_equal(reference, pruned):
            raise RuntimeError("pruned direct solver changed the action map")
        checked += 1
    return checked


def evaluate_candidate(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict],
    generate_alpha: float,
    enhance_alpha: float,
    residual_rmse: dict[str, float],
) -> dict:
    calibrated = predictions.copy()
    calibrated[:, 0] -= generate_alpha * residual_rmse["generate_utility"]
    calibrated[:, 1] -= enhance_alpha * residual_rmse[
        "enhance_lpips_gain_vs_base"]
    rows = []
    counts = Counter()
    harmful_g = 0
    harmful_e = 0
    selected_g = 0
    selected_e = 0
    for indices in grouped_indices(metadata):
        predicted_sample = calibrated[indices]
        target_sample = targets[indices]
        predicted_costs = np.maximum(
            np.rint(predicted_sample[:, 2]).astype(np.int64),
            TILE_HEADER.size + 1)
        predicted_budget = int(round(
            LOW_BYTE_RATIO * float(predicted_costs.sum())))
        actions, _ = solve_direct(predicted_sample, predicted_budget)
        true_costs = np.maximum(
            np.rint(target_sample[:, 2]).astype(np.int64),
            TILE_HEADER.size + 1)
        used_true_bytes = int(true_costs[actions == ACTION_ENHANCE].sum())
        oracle, _ = solve_direct(target_sample, used_true_bytes)
        utility = direct_route_utility(actions, target_sample)
        oracle_utility = direct_route_utility(oracle, target_sample)
        g_mask = actions == ACTION_GENERATE
        e_mask = actions == ACTION_ENHANCE
        harmful_g += int(np.count_nonzero(target_sample[g_mask, 0] < 0))
        harmful_e += int(np.count_nonzero(target_sample[e_mask, 1] < 0))
        selected_g += int(np.count_nonzero(g_mask))
        selected_e += int(np.count_nonzero(e_mask))
        for action, name in (
            (ACTION_BASE, "Base"),
            (ACTION_GENERATE, "Generate"),
            (ACTION_ENHANCE, "Enhance"),
        ):
            counts[name] += int(np.count_nonzero(actions == action))
        rows.append({
            "sample_id": metadata[int(indices[0])]["sample_id"],
            "utility": utility,
            "oracle_utility_at_same_true_enhance_bytes": oracle_utility,
            "oracle_regret": max(oracle_utility - utility, 0.0),
            "action_agreement_with_oracle": float(np.mean(actions == oracle)),
            "true_enhance_budget_ratio": (
                used_true_bytes / float(true_costs.sum())),
        })
    utilities = np.asarray([item["utility"] for item in rows])
    regrets = np.asarray([item["oracle_regret"] for item in rows])
    worst_count = max(1, math.ceil(0.25 * len(rows)))
    return {
        "generate_penalty_rmse_multiplier": generate_alpha,
        "enhance_penalty_rmse_multiplier": enhance_alpha,
        "absolute_generate_utility_penalty": (
            generate_alpha * residual_rmse["generate_utility"]),
        "absolute_enhance_gain_penalty": (
            enhance_alpha * residual_rmse["enhance_lpips_gain_vs_base"]),
        "sample_count": len(rows),
        "mean_oracle_regret": float(regrets.mean()),
        "median_oracle_regret": float(np.median(regrets)),
        "mean_true_utility": float(utilities.mean()),
        "worst_quartile_mean_true_utility": float(
            np.sort(utilities)[:worst_count].mean()),
        "positive_true_utility_fraction": float(np.mean(utilities > 0)),
        "mean_action_agreement_with_oracle": float(np.mean([
            item["action_agreement_with_oracle"] for item in rows
        ])),
        "mean_true_enhance_budget_ratio": float(np.mean([
            item["true_enhance_budget_ratio"] for item in rows
        ])),
        "action_counts": dict(counts),
        "harmful_selected_generate_fraction": (
            harmful_g / selected_g if selected_g else 0.0),
        "harmful_selected_enhance_fraction": (
            harmful_e / selected_e if selected_e else 0.0),
    }


def choose_candidate(candidates: list[dict]) -> dict:
    minimum = min(item["mean_oracle_regret"] for item in candidates)
    tied = [
        item for item in candidates
        if item["mean_oracle_regret"] <= minimum + 1e-6
    ]
    return min(tied, key=lambda item: (
        -item["worst_quartile_mean_true_utility"],
        item["generate_penalty_rmse_multiplier"]
        + item["enhance_penalty_rmse_multiplier"],
        item["generate_penalty_rmse_multiplier"],
    ))


def direct_regression_metrics(
    reference: np.ndarray, prediction: np.ndarray,
) -> dict:
    result = {}
    for index, name in enumerate(DIRECT_TARGET_NAMES):
        error = prediction[:, index] - reference[:, index]
        denominator = float(np.sum(
            (reference[:, index] - reference[:, index].mean()) ** 2))
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": (
                1.0 - float(np.sum(error ** 2)) / denominator
                if denominator > 0 else None),
            "target_mean": float(reference[:, index].mean()),
            "prediction_mean": float(prediction[:, index].mean()),
        }
    return result


def cross_validate_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    raw_features, indirect_targets, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    if len(metadata) != 8000:
        raise ValueError(f"expected 8000 training regions, found {len(metadata)}")
    probes = read_base_probe_training_features(
        args.train_teacher_manifest, metadata)
    targets = direct_targets(indirect_targets)
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    results = {}
    saved = {"targets": targets, "base_probe": probes}
    for kind in ("raw-direct", "base-probe-direct"):
        prediction, disagreement, folds = out_of_fold_predictions(
            raw_features, probes, targets, metadata, kind, args, device)
        saved[f"{kind}_prediction"] = prediction
        saved[f"{kind}_ensemble_std"] = disagreement
        metrics = direct_regression_metrics(targets, prediction)
        residual_rmse = {
            name: metrics[name]["rmse"] for name in DIRECT_TARGET_NAMES
        }
        candidates = [
            evaluate_candidate(
                prediction, targets, metadata, generate_alpha,
                enhance_alpha, residual_rmse)
            for generate_alpha in GAIN_PENALTIES
            for enhance_alpha in GAIN_PENALTIES
        ]
        results[kind] = {
            "feature_count": (
                len(FEATURE_NAMES) if kind == "raw-direct"
                else len(BASE_PROBE_FEATURE_NAMES)),
            "tower_width": 32,
            "regression_metrics": metrics,
            "mean_ensemble_disagreement": {
                name: float(disagreement[:, index].mean())
                for index, name in enumerate(DIRECT_TARGET_NAMES)
            },
            "out_of_fold_residual_rmse": residual_rmse,
            "folds": folds,
            "pruned_solver_action_regression_samples": verify_solver(
                prediction, metadata),
            "candidates": candidates,
            "best_candidate": choose_candidate(candidates),
        }
    raw_best = results["raw-direct"]["best_candidate"]
    probe_best = results["base-probe-direct"]["best_candidate"]
    raw_regret = raw_best["mean_oracle_regret"]
    reduction = (
        (raw_regret - probe_best["mean_oracle_regret"]) / raw_regret
        if raw_regret > 0 else 0.0)
    regret_pass = reduction >= 0.10
    tail_pass = (
        probe_best["worst_quartile_mean_true_utility"]
        >= raw_best["worst_quartile_mean_true_utility"])
    gate_pass = regret_pass and tail_pass
    summary = {
        "experiment": "A800 low-budget v3 Base-probe grouped cross-validation",
        "status": "selection-complete",
        "protocol": "docs/CLOUD_A800_LOW_BUDGET_V3.md",
        "inputs": {
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
            "region_count": len(metadata),
            "sample_count": len(grouped_indices(metadata)),
            "sequence_count": len(set(item["sequence"] for item in metadata)),
            "base_probe_source": (
                "saved uniform-Base candidate quality; reproducible from "
                "encoder-visible source plus provisional Base reconstruction"),
            "development_or_test_data_used": False,
        },
        "training": {
            "fold_rule": "int(sequence) % 5",
            "fold_count": FOLD_COUNT,
            "ensemble_size": args.ensemble_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "architecture": (
                "three independent 2x32 MLP towers for direct targets"),
        },
        "budget": {
            "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
            "generate_tile_budget": LOW_GENERATE_TILES,
            "fragment_penalty": FRAGMENT_PENALTY,
        },
        "selection_rule": {
            "primary": "minimum mean oracle regret",
            "secondary": "maximum worst-quartile mean true utility",
            "required_probe_regret_reduction": 0.10,
            "required_probe_tail_non_degradation": True,
        },
        "feature_schemas": {
            "raw-direct": list(FEATURE_NAMES),
            "base-probe-direct": list(BASE_PROBE_FEATURE_NAMES),
        },
        "direct_target_names": list(DIRECT_TARGET_NAMES),
        "models": results,
        "gate": {
            "pass": gate_pass,
            "probe_regret_reduction_fraction": reduction,
            "regret_reduction_pass": regret_pass,
            "tail_non_degradation_pass": tail_pass,
            "selected_feature_kind": "base-probe-direct" if gate_pass else None,
            "selected_candidate": probe_best if gate_pass else None,
            "next_step": (
                "train frozen full-data ensemble"
                if gate_pass else "stop before development evaluation"),
        },
        "scientific_boundary": {
            "selection_uses_training_labels_only": True,
            "folds_grouped_by_source_sequence": True,
            "base_probe_reproducible_at_encoder_from_source_and_reconstruction": True,
            "decoder_does_not_receive_source_rgb": True,
            "independent_test_012_023_not_used": True,
            "sealed_024_029_not_read": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "selection_summary.json", summary)
    atomic_npz(args.output_dir / "oof_predictions.npz", **saved)
    print(json.dumps({
        "summary": str(args.output_dir / "selection_summary.json"),
        "gate": summary["gate"],
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def train_full(
    features: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict, dict]:
    normalized_x, x_mean, x_scale = normalize_from_train(features, features)
    normalized_y, y_mean, y_scale = normalize_from_train(targets, targets)
    states = []
    predictions = []
    seeds = []
    losses = []
    for member in range(args.ensemble_size):
        seed = args.seed + 50000 + member
        state, normalized_prediction, loss = fit_one(
            normalized_x, normalized_y, normalized_x,
            args.epochs, args.batch_size, args.learning_rate,
            args.weight_decay, seed, device)
        states.append(state)
        predictions.append(normalized_prediction * y_scale + y_mean)
        seeds.append(seed)
        losses.append(loss)
        print(json.dumps({
            "stage": "low-budget-v3-full-train",
            "member": member,
            "seed": seed,
            "final_standardized_mse": loss,
        }), flush=True)
    normalization = {
        "x_mean": torch.from_numpy(x_mean),
        "x_scale": torch.from_numpy(x_scale),
        "y_mean": torch.from_numpy(y_mean),
        "y_scale": torch.from_numpy(y_scale),
    }
    training = {
        "member_seeds": seeds,
        "member_final_standardized_mse": losses,
        "ensemble_train_metrics": direct_regression_metrics(
            targets, np.stack(predictions).mean(axis=0)),
    }
    return states, normalization, training


def train_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    gate = selection.get("gate", {})
    if (not gate.get("pass")
            or gate.get("selected_feature_kind") != "base-probe-direct"):
        raise RuntimeError("v3 training-only gate did not authorize full training")
    if selection["inputs"].get("development_or_test_data_used") is not False:
        raise RuntimeError("selection summary is not training-only")
    raw, indirect, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    probes = read_base_probe_training_features(
        args.train_teacher_manifest, metadata)
    features = engineer_features(
        raw, probes, metadata, "base-probe-direct")
    targets = direct_targets(indirect)
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    states, normalization, training = train_full(
        features, targets, args, device)
    selected = gate["selected_candidate"]
    checkpoint = {
        "format": "A800 low-budget controller v3 Base-probe direct ensemble",
        "raw_feature_names": list(FEATURE_NAMES),
        "base_probe_names": list(BASE_PROBE_NAMES),
        "engineered_feature_names": list(BASE_PROBE_FEATURE_NAMES),
        "direct_target_names": list(DIRECT_TARGET_NAMES),
        "input_dim": len(BASE_PROBE_FEATURE_NAMES),
        "tower_width": 32,
        "ensemble_size": args.ensemble_size,
        "model_state_dicts": states,
        **normalization,
        "calibration": {
            "generate_penalty_rmse_multiplier": selected[
                "generate_penalty_rmse_multiplier"],
            "enhance_penalty_rmse_multiplier": selected[
                "enhance_penalty_rmse_multiplier"],
            "absolute_generate_utility_penalty": selected[
                "absolute_generate_utility_penalty"],
            "absolute_enhance_gain_penalty": selected[
                "absolute_enhance_gain_penalty"],
            "out_of_fold_residual_rmse": selection["models"][
                "base-probe-direct"]["out_of_fold_residual_rmse"],
        },
        "budget": selection["budget"],
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "base_seed": args.seed,
            "region_count": len(metadata),
            "sample_count": len(grouped_indices(metadata)),
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
            **training,
        },
        "selection": {
            "summary": str(args.selection_summary.resolve()),
            "summary_sha256": hashlib.sha256(
                args.selection_summary.read_bytes()).hexdigest(),
            "training_only_gate_passed": True,
        },
        "scientific_boundary": selection["scientific_boundary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller_v3.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    parameter_count = sum(value.numel() for value in states[0].values())
    summary = {
        "experiment": "A800 low-budget v3 full training",
        "status": "trained",
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameter_count_per_member": parameter_count,
        "parameter_count_total": parameter_count * args.ensemble_size,
        "feature_count": len(BASE_PROBE_FEATURE_NAMES),
        "calibration": checkpoint["calibration"],
        "training": checkpoint["training"],
        "selection": checkpoint["selection"],
        "wall_seconds": time.perf_counter() - started,
        "scientific_boundary": checkpoint["scientific_boundary"],
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "summary": str(args.output_dir / "training_summary.json"),
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def predict_ensemble(
    checkpoint: dict, features: np.ndarray, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = ((features - checkpoint["x_mean"].numpy())
                  / checkpoint["x_scale"].numpy()).astype(np.float32)
    y_mean = checkpoint["y_mean"].numpy()
    y_scale = checkpoint["y_scale"].numpy()
    predictions = []
    for state in checkpoint["model_state_dicts"]:
        model = IndependentDirectMLP(
            checkpoint["input_dim"], checkpoint["tower_width"]).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            value = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        predictions.append(value * y_scale + y_mean)
    stacked = np.stack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0)


def base_probe_for_route(
    originals: list[np.ndarray],
    reconstruction: list[np.ndarray],
    metric: SpatialLPIPS,
) -> np.ndarray:
    if len(originals) != 17 or len(reconstruction) != 17:
        raise ValueError("Base probe expects 17 source and reconstruction frames")
    boxes = tile_boxes(512, 512, 128)
    maps = metric.maps(originals, reconstruction)
    rows = []
    for box in boxes:
        quality = local_quality(originals, reconstruction, maps, box)
        rows.append([
            quality["lpips_alex"],
            quality["psnr_db"],
            quality["rgb_mse"],
            quality["temporal_delta_mae"],
        ])
    return np.asarray(rows, dtype=np.float32)


def route_main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != (
            "A800 low-budget controller v3 Base-probe direct ensemble"):
        raise ValueError("checkpoint is not a v3 Base-probe controller")
    if checkpoint["engineered_feature_names"] != list(BASE_PROBE_FEATURE_NAMES):
        raise ValueError("v3 feature schema mismatch")
    if checkpoint["direct_target_names"] != list(DIRECT_TARGET_NAMES):
        raise ValueError("v3 direct target schema mismatch")
    records = read_jsonl(args.sample_manifest)
    if args.limit is not None:
        records = records[:args.limit]
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    model_load_started = time.perf_counter()
    metric = SpatialLPIPS(device, args.lpips_batch_size)
    lpips_model_load_seconds = time.perf_counter() - model_load_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    variant_name = "base-probe-direct-low-v3"
    for record in records:
        sample_started = time.perf_counter()
        sample_id = record["sample_id"]
        gate_root = args.base_probe_root / sample_id / "uniform_gate"
        gate = json.loads((gate_root / "summary.json").read_text(encoding="utf-8"))
        qp16 = gate["variants"]["qp16-base"]
        if qp16["qp"] != 16:
            raise ValueError(f"Base probe is not QP16 for {sample_id}")
        stream_path = Path(qp16["path"])
        if stream_path.stat().st_size != qp16["rate"]["total_bytes"]:
            raise RuntimeError(f"Base probe stream byte check failed for {sample_id}")
        originals = load_source(record)
        reconstruction_dir = gate_root / "frames" / "qp16-base"
        reconstruction = load_pngs(reconstruction_dir)
        feature_started = time.perf_counter()
        probe = base_probe_for_route(originals, reconstruction, metric)
        boxes = tile_boxes(512, 512, 128)
        raw_records = source_features(originals, boxes)
        raw = np.asarray([
            [item[name] for name in FEATURE_NAMES] for item in raw_records
        ], dtype=np.float32)
        engineered = single_sample_base_probe_features(raw, probe)
        uncalibrated, disagreement = predict_ensemble(
            checkpoint, engineered, device)
        feature_and_controller_seconds = time.perf_counter() - feature_started
        calibrated = uncalibrated.copy()
        calibrated[:, 0] -= checkpoint["calibration"][
            "absolute_generate_utility_penalty"]
        calibrated[:, 1] -= checkpoint["calibration"][
            "absolute_enhance_gain_penalty"]
        predicted_costs = np.maximum(
            np.rint(calibrated[:, 2]).astype(np.int64),
            TILE_HEADER.size + 1)
        byte_budget = int(round(LOW_BYTE_RATIO * predicted_costs.sum()))
        actions, diagnostics = solve_direct(calibrated, byte_budget)
        result = {
            "experiment": "A800 deployable Base-probe low-budget v3 route",
            "sample": record,
            "route_kind": "two-pass-base-probe-direct-controller",
            "selected_variant": variant_name,
            "configuration": {
                "tile_size": 128,
                "tile_grid": [4, 4],
                "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
            },
            "raw_feature_names": list(FEATURE_NAMES),
            "base_probe_names": list(BASE_PROBE_NAMES),
            "engineered_feature_names": list(BASE_PROBE_FEATURE_NAMES),
            "direct_target_names": list(DIRECT_TARGET_NAMES),
            "encoder_visible_features": raw_records,
            "base_probe_features": probe.tolist(),
            "engineered_features": engineered.tolist(),
            "uncalibrated_direct_predictions": uncalibrated.tolist(),
            "ensemble_prediction_std": disagreement.tolist(),
            "direct_predictions": calibrated.tolist(),
            "calibration": checkpoint["calibration"],
            "base_probe": {
                "qp": 16,
                "temporary_stream": str(stream_path),
                "temporary_stream_bytes_not_charged_to_transmitted_rate": (
                    qp16["rate"]["total_bytes"]),
                "reconstruction_dir": str(reconstruction_dir),
                "recorded_encode_seconds": qp16["encode"]["seconds"],
                "recorded_fresh_decode_seconds_median": qp16["runtime"][
                    "fresh_decode_seconds_median"],
                "feature_and_controller_seconds": feature_and_controller_seconds,
                "shared_lpips_model_load_seconds": lpips_model_load_seconds,
                "estimated_encoder_analysis_seconds_excluding_shared_model_load": (
                    qp16["encode"]["seconds"]
                    + qp16["runtime"]["fresh_decode_seconds_median"]
                    + feature_and_controller_seconds),
                "cached_deterministic_reconstruction_reused": True,
            },
            "variants": {
                variant_name: {
                    "method": "base-probe-direct-v3",
                    "actions": actions.tolist(),
                    "action_counts": action_counts(actions),
                    "budget": {
                        "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
                        "enhance_fallback_byte_budget": byte_budget,
                        "generate_tile_budget": LOW_GENERATE_TILES,
                        "compute_time_weight_already_inside_direct_utility": True,
                    },
                    "diagnostics": diagnostics,
                }
            },
            "route_wall_seconds": time.perf_counter() - sample_started,
            "scientific_boundary": {
                "uncompressed_source_available_only_at_encoder": True,
                "base_reconstruction_computed_at_encoder": True,
                "ground_truth_or_teacher_targets_used_for_route": False,
                "decoder_receives_source_rgb": False,
                "temporary_base_stream_transmitted": False,
                "action_map_must_be_written_to_stream": True,
                "predicted_fallback_bytes_are_final_spatial_bytes": False,
                "dcvc_uf_frozen": True,
                "seedvr2_frozen": True,
            },
        }
        path = args.output_dir / f"{sample_id}.json"
        atomic_json(path, result)
        entries.append({
            "sample_id": sample_id,
            "path": str(path),
            "selected_variant": variant_name,
            "action_counts": action_counts(actions),
            "base_probe": result["base_probe"],
        })
        print(json.dumps({
            "stage": "low-budget-v3-route",
            "sample_id": sample_id,
            "selected_counts": action_counts(actions),
            "encoder_analysis_seconds": result["base_probe"][
                "estimated_encoder_analysis_seconds_excluding_shared_model_load"],
        }, ensure_ascii=False), flush=True)
    manifest = {
        "experiment": "A800 Base-probe low-budget v3 route manifest",
        "checkpoint": str(args.checkpoint),
        "sample_count": len(entries),
        "lpips_model_load_seconds_shared": lpips_model_load_seconds,
        "entries": entries,
        "scientific_boundary": {
            "development_or_test_quality_results_used_for_routing": False,
            "cached_base_reconstruction_is_encoder_reproducible": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    if args.mode == "cross-validate":
        cross_validate_main(args)
    elif args.mode == "train":
        train_main(args)
    else:
        route_main(args)


if __name__ == "__main__":
    main()
