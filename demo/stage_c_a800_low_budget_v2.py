#!/usr/bin/env python3
"""Training-only selection and deployable routing for low-budget controller v2.

The v2 controller adds within-sample context to the existing encoder-visible
features and uses grouped out-of-fold predictions to select one conservative
gain calibration.  It never reads development or test quality while selecting
the model.  DCVC-UF, SeedVR2, and the spatial-QP format remain frozen.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_controller import (
    TARGET_NAMES,
    UtilityMLP,
    atomic_json,
    generate_components,
    read_jsonl,
    read_teacher_rows,
    regression_metrics,
    solve_actions,
)
from demo.stage_c_a800_teacher import FEATURE_NAMES, load_source, source_features
from demo.stage_c_three_path_roi_probe import (
    ACTION_BASE,
    ACTION_ENHANCE,
    ACTION_GENERATE,
    TILE_HEADER,
    action_counts,
    tile_boxes,
)


CONTEXT_CONTENT_NAMES = tuple(FEATURE_NAMES[3:])
NEIGHBOR_DELTA_NAMES = (
    "luma_std",
    "gradient_mean",
    "edge_density",
    "motion_mean",
    "motion_std",
    "temporal_luma_std",
)
CONTEXT_FEATURE_NAMES = (
    tuple(FEATURE_NAMES)
    + tuple(f"sample_delta_{name}" for name in CONTEXT_CONTENT_NAMES)
    + tuple(f"sample_z_{name}" for name in CONTEXT_CONTENT_NAMES)
    + tuple(f"neighbor_delta_{name}" for name in NEIGHBOR_DELTA_NAMES)
)

DEFAULT_SEED = 20260918
FOLD_COUNT = 5
ENSEMBLE_SIZE = 3
LOW_BYTE_RATIO = 0.25
LOW_GENERATE_TILES = 4
TEMPORAL_RISK_WEIGHT = 0.002
PSNR_RISK_WEIGHT = 0.001
COMPUTE_TIME_WEIGHT = 0.001
FRAGMENT_PENALTY = 0.004
GENERATE_PENALTIES = (0.0, 0.25, 0.5, 0.75, 1.0)
ENHANCE_PENALTIES = (0.0, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grouped-CV low-budget regional controller v2")
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
    route.add_argument("--checkpoint", type=Path, required=True)
    route.add_argument("--output-dir", type=Path, required=True)
    route.add_argument("--limit", type=int)
    route.add_argument("--cuda-idx", type=int, default=0)

    args = parser.parse_args()
    if args.mode in {"cross-validate", "train"}:
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("epochs and batch size must be positive")
        if args.ensemble_size != ENSEMBLE_SIZE:
            parser.error(
                f"the preregistered ensemble size is fixed at {ENSEMBLE_SIZE}")
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


def normalize_from_train(
    train_values: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return ((values - mean) / scale).astype(np.float32), mean, scale


def grouped_indices(metadata: list[dict]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[item["sample_id"]].append(index)
    result = []
    for sample_id, indices in groups.items():
        ordered = sorted(indices, key=lambda value: metadata[value]["region"])
        regions = [metadata[value]["region"] for value in ordered]
        if regions != list(range(16)):
            raise ValueError(f"sample {sample_id} does not contain regions 0..15")
        result.append(np.asarray(ordered, dtype=np.int64))
    return result


def single_sample_context_features(features: np.ndarray) -> np.ndarray:
    if features.shape != (16, len(FEATURE_NAMES)):
        raise ValueError(
            f"expected one 16-region sample, received {features.shape}")
    content_indices = [FEATURE_NAMES.index(name) for name in CONTEXT_CONTENT_NAMES]
    content = features[:, content_indices]
    sample_mean = content.mean(axis=0)
    sample_scale = content.std(axis=0)
    sample_scale[sample_scale < 1e-8] = 1.0
    sample_delta = content - sample_mean
    sample_z = sample_delta / sample_scale

    neighbor_indices = [FEATURE_NAMES.index(name) for name in NEIGHBOR_DELTA_NAMES]
    neighbor_delta = np.empty((16, len(neighbor_indices)), dtype=np.float32)
    for region in range(16):
        row, column = divmod(region, 4)
        neighbors = [
            next_row * 4 + next_column
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            )
            if 0 <= next_row < 4 and 0 <= next_column < 4
        ]
        neighbor_delta[region] = (
            features[region, neighbor_indices]
            - features[neighbors][:, neighbor_indices].mean(axis=0)
        )
    return np.concatenate(
        [features, sample_delta, sample_z, neighbor_delta], axis=1
    ).astype(np.float32)


def engineer_features(
    features: np.ndarray, metadata: list[dict], kind: str,
) -> np.ndarray:
    if kind == "raw":
        return features.astype(np.float32, copy=True)
    if kind != "context":
        raise ValueError(f"unknown feature kind: {kind}")
    output = np.empty(
        (len(features), len(CONTEXT_FEATURE_NAMES)), dtype=np.float32)
    for indices in grouped_indices(metadata):
        output[indices] = single_sample_context_features(features[indices])
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
    width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, float]:
    set_seed(seed)
    model = UtilityMLP(train_x.shape[1], train_y.shape[1], width).to(device)
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


def fold_for_sequence(sequence: str) -> int:
    try:
        return int(sequence) % FOLD_COUNT
    except ValueError as error:
        raise ValueError(
            f"sequence must be numeric for the frozen fold rule: {sequence}"
        ) from error


def out_of_fold_predictions(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    metadata: list[dict],
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features = engineer_features(raw_x, metadata, kind)
    folds = np.asarray([
        fold_for_sequence(item["sequence"]) for item in metadata
    ], dtype=np.int64)
    width = 64 if kind == "raw" else 96
    output = np.empty_like(raw_y)
    disagreement = np.empty_like(raw_y)
    records = []
    for fold in range(FOLD_COUNT):
        train_mask = folds != fold
        validation_mask = folds == fold
        if not np.any(train_mask) or not np.any(validation_mask):
            raise ValueError(f"empty train or validation partition in fold {fold}")
        normalized_x, x_mean, x_scale = normalize_from_train(
            features[train_mask], features)
        normalized_y, y_mean, y_scale = normalize_from_train(
            raw_y[train_mask], raw_y)
        predictions = []
        losses = []
        for member in range(args.ensemble_size):
            seed = args.seed + fold * 1000 + member
            _, normalized_prediction, final_loss = fit_one(
                normalized_x[train_mask],
                normalized_y[train_mask],
                normalized_x[validation_mask],
                width,
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
                "stage": "low-budget-v2-cross-validation",
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


def target_scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    enhance = values[:, index["enhance_lpips_gain_vs_base"]]
    return generate, enhance


def solve_actions_pruned(
    predictions: np.ndarray,
    byte_budget: int,
    generate_budget: int,
    temporal_risk_weight: float,
    psnr_risk_weight: float,
    compute_time_weight: float,
    fragment_penalty: float,
) -> tuple[np.ndarray, dict]:
    """Exact v1 solver with Pareto-dominated intermediate states removed.

    For a fixed Generate count, a state that spends at least as many Enhance
    bytes and has no greater score can never improve any continuation.  Removing
    it preserves the objective and tie-breaking while making repeated grouped-CV
    route replay tractable.
    """
    index = {name: TARGET_NAMES.index(name) for name in TARGET_NAMES}
    generate_scores = (
        predictions[:, index["generate_lpips_gain_vs_base"]]
        - temporal_risk_weight * np.maximum(
            predictions[:, index["generate_temporal_risk_vs_base"]], 0)
        - psnr_risk_weight * np.maximum(
            -predictions[:, index["generate_psnr_delta_db_vs_base"]], 0)
        - compute_time_weight * np.maximum(
            predictions[:, index[
                "generate_roi_seconds_measured_geometry_class"]], 0)
    )
    enhance_scores = predictions[:, index["enhance_lpips_gain_vs_base"]]
    enhance_costs = np.maximum(
        np.rint(predictions[:, index[
            "enhance_fallback_extra_on_disk_bytes"]]).astype(np.int64),
        TILE_HEADER.size + 1,
    )
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0): (0.0, tuple())
    }
    for region in range(16):
        updated: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (used_bytes, used_generate), (score, actions) in states.items():
            choices = [(ACTION_BASE, 0, 0, 0.0)]
            if used_generate < generate_budget and generate_scores[region] > 0:
                choices.append((
                    ACTION_GENERATE, 0, 1, float(generate_scores[region])))
            if (used_bytes + enhance_costs[region] <= byte_budget
                    and enhance_scores[region] > 0):
                choices.append((
                    ACTION_ENHANCE,
                    int(enhance_costs[region]),
                    0,
                    float(enhance_scores[region]),
                ))
            for action, byte_cost, generate_cost, gain in choices:
                key = (used_bytes + byte_cost, used_generate + generate_cost)
                candidate = (score + gain, actions + (action,))
                incumbent = updated.get(key)
                if incumbent is None or candidate[0] > incumbent[0]:
                    updated[key] = candidate

        states = {}
        by_generate: dict[int, list[tuple[int, float, tuple[int, ...]]]] = (
            defaultdict(list))
        for (used_bytes, used_generate), (score, actions) in updated.items():
            by_generate[used_generate].append((used_bytes, score, actions))
        for used_generate, candidates in by_generate.items():
            best_score = -math.inf
            for used_bytes, score, actions in sorted(candidates):
                if score > best_score:
                    states[(used_bytes, used_generate)] = (score, actions)
                    best_score = score

    _, (_, selected) = max(
        states.items(), key=lambda item: (
            item[1][0], -item[0][0], -item[0][1]))
    actions = np.asarray(selected, dtype=np.uint8)
    changed = True
    while changed:
        changed = False
        grid = actions.reshape(4, 4)
        for region in np.flatnonzero(actions == ACTION_GENERATE):
            row, column = divmod(int(region), 4)
            neighbors = [
                grid[next_row, next_column]
                for next_row, next_column in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                )
                if 0 <= next_row < 4 and 0 <= next_column < 4
            ]
            if (ACTION_GENERATE not in neighbors
                    and generate_scores[region] < fragment_penalty):
                actions[region] = ACTION_BASE
                changed = True

    components = generate_components(actions)
    used_bytes = int(sum(
        enhance_costs[value]
        for value in np.flatnonzero(actions == ACTION_ENHANCE)))
    predicted_roi_seconds = np.maximum(
        predictions[:, index[
            "generate_roi_seconds_measured_geometry_class"]], 0)
    used_generate_seconds = float(predicted_roi_seconds[
        actions == ACTION_GENERATE].sum())
    return actions, {
        "predicted_generate_scores": generate_scores.tolist(),
        "predicted_enhance_scores": enhance_scores.tolist(),
        "predicted_enhance_fallback_cost_bytes": enhance_costs.tolist(),
        "predicted_generate_independent_roi_seconds": (
            predicted_roi_seconds.tolist()),
        "predicted_used_enhance_fallback_bytes": used_bytes,
        "predicted_used_generate_independent_roi_seconds": (
            used_generate_seconds),
        "generate_component_count": len(components),
        "generate_components": components,
        "generate_area_fraction": float(
            np.count_nonzero(actions == ACTION_GENERATE) / 16),
        "compute_proxy": float(
            np.count_nonzero(actions == ACTION_GENERATE) / 16
            + 0.05 * len(components)),
        "pareto_pruned_exact_solver": True,
    }


def route_utility(actions: np.ndarray, targets: np.ndarray) -> float:
    generate, enhance = target_scores(targets)
    return float(
        generate[actions == ACTION_GENERATE].sum()
        + enhance[actions == ACTION_ENHANCE].sum())


def verify_pruned_solver(
    predictions: np.ndarray, metadata: list[dict], limit: int = 32,
) -> int:
    cost_index = TARGET_NAMES.index("enhance_fallback_extra_on_disk_bytes")
    checked = 0
    for indices in grouped_indices(metadata)[:limit]:
        sample = predictions[indices]
        costs = np.maximum(
            np.rint(sample[:, cost_index]).astype(np.int64),
            TILE_HEADER.size + 1)
        byte_budget = int(round(LOW_BYTE_RATIO * float(costs.sum())))
        reference, _ = solve_actions(
            sample,
            byte_budget,
            LOW_GENERATE_TILES,
            TEMPORAL_RISK_WEIGHT,
            PSNR_RISK_WEIGHT,
            COMPUTE_TIME_WEIGHT,
            FRAGMENT_PENALTY,
        )
        pruned, _ = solve_actions_pruned(
            sample,
            byte_budget,
            LOW_GENERATE_TILES,
            TEMPORAL_RISK_WEIGHT,
            PSNR_RISK_WEIGHT,
            COMPUTE_TIME_WEIGHT,
            FRAGMENT_PENALTY,
        )
        if not np.array_equal(reference, pruned):
            raise RuntimeError(
                "Pareto-pruned solver changed a reference action map")
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
    calibrated[:, TARGET_NAMES.index("generate_lpips_gain_vs_base")] -= (
        generate_alpha * residual_rmse["generate_lpips_gain_vs_base"])
    calibrated[:, TARGET_NAMES.index("enhance_lpips_gain_vs_base")] -= (
        enhance_alpha * residual_rmse["enhance_lpips_gain_vs_base"])
    cost_index = TARGET_NAMES.index("enhance_fallback_extra_on_disk_bytes")
    sample_rows = []
    counts = Counter()
    harmful_generate = 0
    harmful_enhance = 0
    selected_generate = 0
    selected_enhance = 0
    for indices in grouped_indices(metadata):
        predicted_sample = calibrated[indices]
        target_sample = targets[indices]
        predicted_costs = np.maximum(
            np.rint(predicted_sample[:, cost_index]).astype(np.int64),
            TILE_HEADER.size + 1)
        predicted_budget = int(round(
            LOW_BYTE_RATIO * float(predicted_costs.sum())))
        actions, _ = solve_actions_pruned(
            predicted_sample,
            predicted_budget,
            LOW_GENERATE_TILES,
            TEMPORAL_RISK_WEIGHT,
            PSNR_RISK_WEIGHT,
            COMPUTE_TIME_WEIGHT,
            FRAGMENT_PENALTY,
        )
        true_costs = np.maximum(
            np.rint(target_sample[:, cost_index]).astype(np.int64),
            TILE_HEADER.size + 1)
        used_true_bytes = int(true_costs[actions == ACTION_ENHANCE].sum())
        oracle, _ = solve_actions_pruned(
            target_sample,
            used_true_bytes,
            LOW_GENERATE_TILES,
            TEMPORAL_RISK_WEIGHT,
            PSNR_RISK_WEIGHT,
            COMPUTE_TIME_WEIGHT,
            FRAGMENT_PENALTY,
        )
        utility = route_utility(actions, target_sample)
        oracle_utility = route_utility(oracle, target_sample)
        regret = max(oracle_utility - utility, 0.0)
        generate_score, enhance_score = target_scores(target_sample)
        g_mask = actions == ACTION_GENERATE
        e_mask = actions == ACTION_ENHANCE
        harmful_generate += int(np.count_nonzero(generate_score[g_mask] < 0))
        harmful_enhance += int(np.count_nonzero(enhance_score[e_mask] < 0))
        selected_generate += int(np.count_nonzero(g_mask))
        selected_enhance += int(np.count_nonzero(e_mask))
        for action, name in (
            (ACTION_BASE, "Base"),
            (ACTION_GENERATE, "Generate"),
            (ACTION_ENHANCE, "Enhance"),
        ):
            counts[name] += int(np.count_nonzero(actions == action))
        sample_rows.append({
            "sample_id": metadata[int(indices[0])]["sample_id"],
            "sequence": metadata[int(indices[0])]["sequence"],
            "utility": utility,
            "oracle_utility_at_same_true_enhance_bytes": oracle_utility,
            "oracle_regret": regret,
            "action_agreement_with_oracle": float(np.mean(actions == oracle)),
            "true_enhance_bytes": used_true_bytes,
            "true_enhance_budget_ratio": (
                used_true_bytes / float(true_costs.sum())),
        })
    utilities = np.asarray([item["utility"] for item in sample_rows])
    regrets = np.asarray([item["oracle_regret"] for item in sample_rows])
    worst_count = max(1, math.ceil(0.25 * len(utilities)))
    worst_quartile = float(np.sort(utilities)[:worst_count].mean())
    return {
        "generate_penalty_rmse_multiplier": generate_alpha,
        "enhance_penalty_rmse_multiplier": enhance_alpha,
        "absolute_generate_gain_penalty": (
            generate_alpha * residual_rmse["generate_lpips_gain_vs_base"]),
        "absolute_enhance_gain_penalty": (
            enhance_alpha * residual_rmse["enhance_lpips_gain_vs_base"]),
        "sample_count": len(sample_rows),
        "mean_oracle_regret": float(regrets.mean()),
        "median_oracle_regret": float(np.median(regrets)),
        "mean_true_utility": float(utilities.mean()),
        "worst_quartile_mean_true_utility": worst_quartile,
        "positive_true_utility_fraction": float(np.mean(utilities > 0)),
        "mean_action_agreement_with_oracle": float(np.mean([
            item["action_agreement_with_oracle"] for item in sample_rows
        ])),
        "mean_true_enhance_budget_ratio": float(np.mean([
            item["true_enhance_budget_ratio"] for item in sample_rows
        ])),
        "action_counts": dict(counts),
        "harmful_selected_generate_fraction": (
            harmful_generate / selected_generate if selected_generate else 0.0),
        "harmful_selected_enhance_fraction": (
            harmful_enhance / selected_enhance if selected_enhance else 0.0),
    }


def choose_candidate(candidates: list[dict]) -> dict:
    minimum_regret = min(item["mean_oracle_regret"] for item in candidates)
    tied = [
        item for item in candidates
        if item["mean_oracle_regret"] <= minimum_regret + 1e-6
    ]
    return min(tied, key=lambda item: (
        -item["worst_quartile_mean_true_utility"],
        item["generate_penalty_rmse_multiplier"]
        + item["enhance_penalty_rmse_multiplier"],
        item["generate_penalty_rmse_multiplier"],
    ))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def cross_validate_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    raw_x, raw_y, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    if len(metadata) != 500 * 16:
        raise ValueError(
            f"preregistered run expects 8000 training regions, found {len(metadata)}")
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    results = {}
    saved_predictions = {"targets": raw_y}
    for kind in ("raw", "context"):
        prediction, disagreement, fold_records = out_of_fold_predictions(
            raw_x, raw_y, metadata, kind, args, device)
        solver_regression_count = verify_pruned_solver(prediction, metadata)
        saved_predictions[f"{kind}_prediction"] = prediction
        saved_predictions[f"{kind}_ensemble_std"] = disagreement
        residual_rmse = {
            name: float(np.sqrt(np.mean(
                (prediction[:, index] - raw_y[:, index]) ** 2)))
            for index, name in enumerate(TARGET_NAMES)
        }
        candidates = [
            evaluate_candidate(
                prediction,
                raw_y,
                metadata,
                generate_alpha,
                enhance_alpha,
                residual_rmse,
            )
            for generate_alpha in GENERATE_PENALTIES
            for enhance_alpha in ENHANCE_PENALTIES
        ]
        best = choose_candidate(candidates)
        results[kind] = {
            "feature_count": (
                len(FEATURE_NAMES) if kind == "raw"
                else len(CONTEXT_FEATURE_NAMES)),
            "width": 64 if kind == "raw" else 96,
            "regression_metrics": regression_metrics(raw_y, prediction),
            "mean_ensemble_disagreement": {
                name: float(disagreement[:, index].mean())
                for index, name in enumerate(TARGET_NAMES)
            },
            "out_of_fold_residual_rmse": residual_rmse,
            "folds": fold_records,
            "pruned_solver_action_regression_samples": solver_regression_count,
            "candidates": candidates,
            "best_candidate": best,
        }
    raw_best = results["raw"]["best_candidate"]
    context_best = results["context"]["best_candidate"]
    baseline_regret = raw_best["mean_oracle_regret"]
    regret_reduction = (
        (baseline_regret - context_best["mean_oracle_regret"])
        / baseline_regret if baseline_regret > 0 else 0.0)
    regret_pass = regret_reduction >= 0.10
    tail_pass = (
        context_best["worst_quartile_mean_true_utility"]
        >= raw_best["worst_quartile_mean_true_utility"])
    gate_pass = regret_pass and tail_pass
    summary = {
        "experiment": "A800 low-budget controller v2 grouped cross-validation",
        "status": "selection-complete",
        "protocol": "docs/CLOUD_A800_LOW_BUDGET_V2.md",
        "inputs": {
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
            "region_count": len(metadata),
            "sample_count": len(grouped_indices(metadata)),
            "sequence_count": len(set(item["sequence"] for item in metadata)),
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
        },
        "budget": {
            "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
            "generate_tile_budget": LOW_GENERATE_TILES,
            "temporal_risk_weight": TEMPORAL_RISK_WEIGHT,
            "psnr_risk_weight": PSNR_RISK_WEIGHT,
            "compute_time_weight": COMPUTE_TIME_WEIGHT,
            "fragment_penalty": FRAGMENT_PENALTY,
        },
        "selection_rule": {
            "primary": "minimum mean oracle regret",
            "secondary": "maximum worst-quartile mean true utility",
            "required_context_regret_reduction": 0.10,
            "required_context_tail_non_degradation": True,
        },
        "feature_schemas": {
            "raw": list(FEATURE_NAMES),
            "context": list(CONTEXT_FEATURE_NAMES),
        },
        "models": results,
        "gate": {
            "pass": gate_pass,
            "context_regret_reduction_fraction": regret_reduction,
            "regret_reduction_pass": regret_pass,
            "tail_non_degradation_pass": tail_pass,
            "selected_feature_kind": "context" if gate_pass else None,
            "selected_candidate": context_best if gate_pass else None,
            "next_step": (
                "train frozen full-data ensemble"
                if gate_pass else "stop before development evaluation"),
        },
        "scientific_boundary": {
            "selection_uses_training_labels_only": True,
            "folds_are_grouped_by_source_sequence": True,
            "ground_truth_is_not_a_deployment_input": True,
            "independent_test_012_023_not_used": True,
            "sealed_024_029_not_read": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "selection_summary.json", summary)
    atomic_npz(args.output_dir / "oof_predictions.npz", **saved_predictions)
    print(json.dumps({
        "summary": str(args.output_dir / "selection_summary.json"),
        "gate": summary["gate"],
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def train_full_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    width: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], np.ndarray, dict]:
    normalized_x, x_mean, x_scale = normalize_from_train(features, features)
    normalized_y, y_mean, y_scale = normalize_from_train(targets, targets)
    states = []
    predictions = []
    losses = []
    seeds = []
    for member in range(args.ensemble_size):
        seed = args.seed + 50000 + member
        state, normalized_prediction, loss = fit_one(
            normalized_x,
            normalized_y,
            normalized_x,
            width,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            seed,
            device,
        )
        states.append(state)
        predictions.append(normalized_prediction * y_scale + y_mean)
        losses.append(loss)
        seeds.append(seed)
        print(json.dumps({
            "stage": "low-budget-v2-full-train",
            "member": member,
            "seed": seed,
            "final_standardized_mse": loss,
        }), flush=True)
    ensemble_prediction = np.stack(predictions).mean(axis=0)
    normalization = {
        "x_mean": torch.from_numpy(x_mean),
        "x_scale": torch.from_numpy(x_scale),
        "y_mean": torch.from_numpy(y_mean),
        "y_scale": torch.from_numpy(y_scale),
    }
    training = {
        "member_seeds": seeds,
        "member_final_standardized_mse": losses,
        "ensemble_train_metrics": regression_metrics(
            targets, ensemble_prediction),
    }
    return states, normalization, training


def train_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    gate = selection.get("gate", {})
    if not gate.get("pass") or gate.get("selected_feature_kind") != "context":
        raise RuntimeError(
            "cross-validation gate did not authorize a full v2 checkpoint")
    if selection["inputs"].get("development_or_test_data_used") is not False:
        raise RuntimeError("selection summary does not prove training-only selection")
    raw_x, targets, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    features = engineer_features(raw_x, metadata, "context")
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    states, normalization, training = train_full_ensemble(
        features, targets, 96, args, device)
    selected = gate["selected_candidate"]
    residual_rmse = selection["models"]["context"][
        "out_of_fold_residual_rmse"]
    checkpoint = {
        "format": "A800 regional utility controller v2 context ensemble",
        "raw_feature_names": list(FEATURE_NAMES),
        "engineered_feature_names": list(CONTEXT_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "feature_kind": "context",
        "input_dim": len(CONTEXT_FEATURE_NAMES),
        "mlp_width": 96,
        "ensemble_size": args.ensemble_size,
        "mlp_state_dicts": states,
        **normalization,
        "calibration": {
            "generate_penalty_rmse_multiplier": selected[
                "generate_penalty_rmse_multiplier"],
            "enhance_penalty_rmse_multiplier": selected[
                "enhance_penalty_rmse_multiplier"],
            "out_of_fold_residual_rmse": residual_rmse,
            "absolute_generate_gain_penalty": selected[
                "absolute_generate_gain_penalty"],
            "absolute_enhance_gain_penalty": selected[
                "absolute_enhance_gain_penalty"],
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
    checkpoint_path = args.output_dir / "controller_v2.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    parameter_count = sum(
        tensor.numel() for tensor in states[0].values())
    summary = {
        "experiment": "A800 low-budget controller v2 full training",
        "status": "trained",
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameter_count_per_member": parameter_count,
        "parameter_count_total": parameter_count * args.ensemble_size,
        "feature_count": len(CONTEXT_FEATURE_NAMES),
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
    x_mean = checkpoint["x_mean"].numpy()
    x_scale = checkpoint["x_scale"].numpy()
    y_mean = checkpoint["y_mean"].numpy()
    y_scale = checkpoint["y_scale"].numpy()
    normalized_x = ((features - x_mean) / x_scale).astype(np.float32)
    predictions = []
    for state in checkpoint["mlp_state_dicts"]:
        model = UtilityMLP(
            checkpoint["input_dim"], len(TARGET_NAMES),
            checkpoint["mlp_width"]).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            value = model(torch.from_numpy(normalized_x).to(device)).cpu().numpy()
        predictions.append(value * y_scale + y_mean)
    stacked = np.stack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0)


def route_main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != (
            "A800 regional utility controller v2 context ensemble"):
        raise ValueError("checkpoint format is not low-budget controller v2")
    if checkpoint["raw_feature_names"] != list(FEATURE_NAMES):
        raise ValueError("raw feature schema mismatch")
    if checkpoint["engineered_feature_names"] != list(CONTEXT_FEATURE_NAMES):
        raise ValueError("context feature schema mismatch")
    if checkpoint["target_names"] != list(TARGET_NAMES):
        raise ValueError("target schema mismatch")
    budget = checkpoint["budget"]
    if (budget["enhance_fallback_byte_ratio"] != LOW_BYTE_RATIO
            or budget["generate_tile_budget"] != LOW_GENERATE_TILES):
        raise ValueError("checkpoint low-budget declaration drifted")
    records = read_jsonl(args.sample_manifest)
    if args.limit is not None:
        records = records[:args.limit]
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    variant_name = "context-ensemble-low-v2"
    for record in records:
        originals = load_source(record)
        boxes = tile_boxes(512, 512, 128)
        raw_records = source_features(originals, boxes)
        raw_features = np.asarray([
            [item[name] for name in FEATURE_NAMES] for item in raw_records
        ], dtype=np.float32)
        context_features = single_sample_context_features(raw_features)
        uncalibrated, disagreement = predict_ensemble(
            checkpoint, context_features, device)
        predictions = uncalibrated.copy()
        calibration = checkpoint["calibration"]
        predictions[:, TARGET_NAMES.index(
            "generate_lpips_gain_vs_base")] -= calibration[
                "absolute_generate_gain_penalty"]
        predictions[:, TARGET_NAMES.index(
            "enhance_lpips_gain_vs_base")] -= calibration[
                "absolute_enhance_gain_penalty"]
        cost_index = TARGET_NAMES.index("enhance_fallback_extra_on_disk_bytes")
        predicted_costs = np.maximum(
            np.rint(predictions[:, cost_index]).astype(np.int64),
            TILE_HEADER.size + 1)
        byte_budget = int(round(LOW_BYTE_RATIO * predicted_costs.sum()))
        actions, diagnostics = solve_actions_pruned(
            predictions,
            byte_budget,
            LOW_GENERATE_TILES,
            TEMPORAL_RISK_WEIGHT,
            PSNR_RISK_WEIGHT,
            COMPUTE_TIME_WEIGHT,
            FRAGMENT_PENALTY,
        )
        result = {
            "experiment": "A800 deployable low-budget controller v2 route",
            "sample": record,
            "route_kind": "context-ensemble-controller",
            "selected_variant": variant_name,
            "configuration": {
                "tile_size": 128,
                "tile_grid": [4, 4],
                "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
            },
            "raw_feature_names": list(FEATURE_NAMES),
            "engineered_feature_names": list(CONTEXT_FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "encoder_visible_features": raw_records,
            "engineered_features": context_features.tolist(),
            "uncalibrated_soft_predictions": uncalibrated.tolist(),
            "ensemble_prediction_std": disagreement.tolist(),
            "soft_predictions": predictions.tolist(),
            "calibration": calibration,
            "variants": {
                variant_name: {
                    "method": "context-ensemble-v2",
                    "actions": actions.tolist(),
                    "action_counts": action_counts(actions),
                    "budget": {
                        "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
                        "enhance_fallback_byte_budget": byte_budget,
                        "generate_tile_budget": LOW_GENERATE_TILES,
                        "compute_time_weight": COMPUTE_TIME_WEIGHT,
                    },
                    "diagnostics": diagnostics,
                }
            },
            "scientific_boundary": {
                "source_rgb_used_at_encoder": True,
                "ground_truth_quality_metrics_used_for_route": False,
                "teacher_targets_used_for_route": False,
                "sample_context_uses_encoder_visible_features_only": True,
                "budget_known_before_encoding": True,
                "action_map_must_be_written_to_stream": True,
                "predicted_fallback_bytes_are_final_spatial_bytes": False,
                "dcvc_uf_frozen": True,
                "seedvr2_frozen": True,
            },
        }
        path = args.output_dir / f"{record['sample_id']}.json"
        atomic_json(path, result)
        entries.append({
            "sample_id": record["sample_id"],
            "path": str(path),
            "selected_variant": variant_name,
            "action_counts": action_counts(actions),
        })
        print(json.dumps({
            "stage": "low-budget-v2-route",
            "sample_id": record["sample_id"],
            "selected_counts": action_counts(actions),
        }, ensure_ascii=False), flush=True)
    manifest = {
        "experiment": "A800 low-budget controller v2 route manifest",
        "checkpoint": str(args.checkpoint),
        "sample_count": len(entries),
        "entries": entries,
        "scientific_boundary": {
            "development_or_test_metrics_used_for_routing": False,
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
