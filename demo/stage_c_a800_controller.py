#!/usr/bin/env python3
"""Train and deploy the lightweight regional controller for the A800 pilot.

The controller predicts soft counterfactual quantities from encoder-visible
source features.  A separate budget solver converts those predictions into
Generate/Base/Enhance actions.  Neither training nor routing changes DCVC-UF
or SeedVR2 weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import FEATURE_NAMES, load_source, source_features
from demo.stage_c_three_path_roi_probe import (
    ACTION_BASE,
    ACTION_ENHANCE,
    ACTION_GENERATE,
    ACTION_NAMES,
    TILE_HEADER,
    action_counts,
    tile_boxes,
)


TARGET_NAMES = (
    "generate_lpips_gain_vs_base",
    "enhance_lpips_gain_vs_base",
    "generate_psnr_delta_db_vs_base",
    "enhance_psnr_delta_db_vs_base",
    "generate_temporal_risk_vs_base",
    "enhance_temporal_gain_vs_base",
    "generate_roi_seconds_measured_geometry_class",
    "enhance_fallback_extra_on_disk_bytes",
)


class UtilityMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or route with the A800 lightweight controller")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--train-teacher-manifest", type=Path, required=True)
    train.add_argument("--train-roi-cost-manifest", type=Path, required=True)
    train.add_argument("--development-teacher-manifest", type=Path)
    train.add_argument("--development-roi-cost-manifest", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=240)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--ridge", type=float, default=1e-2)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--seed", type=int, default=20260917)
    train.add_argument("--cuda-idx", type=int, default=0)

    route = subparsers.add_parser("route")
    route.add_argument("--sample-manifest", type=Path, required=True)
    route.add_argument("--checkpoint", type=Path, required=True)
    route.add_argument("--output-dir", type=Path, required=True)
    route.add_argument(
        "--method", choices=("fixed", "linear", "mlp"), default="mlp")
    route.add_argument(
        "--byte-budget-ratios", type=float, nargs="+", default=(0.25, 0.5, 1.0))
    route.add_argument(
        "--generate-budget-tiles", type=int, nargs="+", default=(4, 8, 12))
    route.add_argument("--temporal-risk-weight", type=float, default=0.002)
    route.add_argument("--psnr-risk-weight", type=float, default=0.001)
    route.add_argument("--compute-time-weight", type=float, default=0.001)
    route.add_argument("--fragment-penalty", type=float, default=0.004)
    route.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mode == "train":
        if args.epochs < 1 or args.batch_size < 1 or args.width < 1:
            parser.error("epochs, batch size, and width must be positive")
        if bool(args.development_teacher_manifest) != bool(
                args.development_roi_cost_manifest):
            parser.error("development quality and ROI-cost manifests are paired")
    else:
        if len(args.byte_budget_ratios) != len(args.generate_budget_tiles):
            parser.error("byte and Generate budget lists must have equal length")
        if any(not 0 <= value <= 1 for value in args.byte_budget_ratios):
            parser.error("byte budget ratios must lie in [0, 1]")
        if any(not 0 <= value <= 16 for value in args.generate_budget_tiles):
            parser.error("Generate tile budgets must lie in [0, 16]")
    return args


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_teacher_rows(
    path: Path, roi_cost_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"teacher manifest is incomplete: {path}")
    roi_manifest = json.loads(roi_cost_path.read_text(encoding="utf-8"))
    if not roi_manifest.get("complete"):
        raise ValueError(f"ROI-cost manifest is incomplete: {roi_cost_path}")
    roi_paths = {
        entry["sample_id"]: Path(entry["path"])
        for entry in roi_manifest["entries"]
    }
    rows = []
    metadata = []
    for entry in manifest["entries"]:
        sample = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        sample_id = sample["sample"]["sample_id"]
        if sample_id not in roi_paths:
            raise ValueError(f"ROI-cost label is missing for {sample_id}")
        roi_sample = json.loads(roi_paths[sample_id].read_text(encoding="utf-8"))
        if roi_sample["sample"]["sample_id"] != sample_id:
            raise ValueError(f"ROI-cost label identity differs for {sample_id}")
        roi_regions = {
            int(region["index"]): region["targets"]
            for region in roi_sample["regions"]
        }
        if set(roi_regions) != set(range(16)):
            raise ValueError(f"ROI-cost label has incomplete regions for {sample_id}")
        if sample["configuration"]["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("teacher feature schema differs from controller schema")
        for region in sample["regions"]:
            targets = {**region["targets"], **roi_regions[region["index"]]}
            rows.append((
                [region["features"][name] for name in FEATURE_NAMES],
                [targets[name] for name in TARGET_NAMES],
            ))
            metadata.append({
                "sample_id": sample_id,
                "sequence": sample["sample"]["sequence"],
                "region": region["index"],
            })
    if not rows:
        raise ValueError(f"teacher manifest has no regions: {path}")
    x = np.asarray([row[0] for row in rows], dtype=np.float32)
    y = np.asarray([row[1] for row in rows], dtype=np.float32)
    return x, y, metadata


def normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (values - mean) / scale, mean, scale


def regression_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(TARGET_NAMES):
        error = predicted[:, index] - reference[:, index]
        denominator = float(np.sum(
            (reference[:, index] - reference[:, index].mean()) ** 2))
        r2 = (
            1.0 - float(np.sum(error ** 2)) / denominator
            if denominator > 0 else None
        )
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": r2,
            "target_mean": float(reference[:, index].mean()),
            "prediction_mean": float(predicted[:, index].mean()),
        }
    return result


def linear_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate([
        x,
        np.ones((x.shape[0], 1), dtype=x.dtype),
    ], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    left = design.astype(np.float64).T @ design.astype(np.float64) + penalty
    right = design.astype(np.float64).T @ y.astype(np.float64)
    return np.linalg.solve(left, right).astype(np.float32)


def linear_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([
        x,
        np.ones((x.shape[0], 1), dtype=x.dtype),
    ], axis=1)
    return design @ weights


def train_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_x_raw, train_y_raw, train_meta = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    train_x, x_mean, x_scale = normalize(train_x_raw)
    train_y, y_mean, y_scale = normalize(train_y_raw)

    linear_weights = linear_fit(train_x, train_y, args.ridge)
    linear_train_norm = linear_predict(train_x, linear_weights)
    linear_train = linear_train_norm * y_scale + y_mean

    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    model = UtilityMLP(len(FEATURE_NAMES), len(TARGET_NAMES), args.width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0)
    losses = []
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            loss = torch.mean((prediction - batch_y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch_x.shape[0]
            count += batch_x.shape[0]
        epoch_loss = total / count
        losses.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
            print(json.dumps({
                "stage": "controller-train",
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "standardized_mse": epoch_loss,
            }), flush=True)

    model.eval()
    with torch.inference_mode():
        mlp_train_norm = model(torch.from_numpy(train_x).to(device)).cpu().numpy()
    mlp_train = mlp_train_norm * y_scale + y_mean

    development = None
    if args.development_teacher_manifest:
        dev_x_raw, dev_y_raw, dev_meta = read_teacher_rows(
            args.development_teacher_manifest,
            args.development_roi_cost_manifest)
        dev_x = (dev_x_raw - x_mean) / x_scale
        linear_dev = linear_predict(dev_x, linear_weights) * y_scale + y_mean
        with torch.inference_mode():
            mlp_dev_norm = model(torch.from_numpy(dev_x).to(device)).cpu().numpy()
        mlp_dev = mlp_dev_norm * y_scale + y_mean
        development = {
            "region_count": len(dev_meta),
            "sequence_count": len(set(item["sequence"] for item in dev_meta)),
            "linear": regression_metrics(dev_y_raw, linear_dev),
            "mlp": regression_metrics(dev_y_raw, mlp_dev),
        }

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint = {
        "format": "A800 regional utility controller v1",
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "x_mean": torch.from_numpy(x_mean),
        "x_scale": torch.from_numpy(x_scale),
        "y_mean": torch.from_numpy(y_mean),
        "y_scale": torch.from_numpy(y_scale),
        "linear_weights": torch.from_numpy(linear_weights),
        "mlp_width": args.width,
        "mlp_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ridge": args.ridge,
            "seed": args.seed,
            "region_count": len(train_meta),
            "sample_count": len(set(item["sample_id"] for item in train_meta)),
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
        },
        "scientific_boundary": {
            "input_is_encoder_visible_source_features": True,
            "ground_truth_is_not_a_deployment_input": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "budget_solver_is_separate": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    summary = {
        "experiment": "A800 lightweight regional controller",
        "status": "trained",
        "checkpoint": str(checkpoint_path),
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "training": checkpoint["training"],
        "parameter_count": parameter_count,
        "fixed_training_schedule": True,
        "checkpoint_selected_on_development": False,
        "train_metrics": {
            "linear": regression_metrics(train_y_raw, linear_train),
            "mlp": regression_metrics(train_y_raw, mlp_train),
        },
        "development_metrics": development,
        "loss_curve": losses,
        "wall_seconds": time.perf_counter() - started,
        "scientific_boundary": checkpoint["scientific_boundary"],
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "summary": str(args.output_dir / "summary.json"),
        "checkpoint": str(checkpoint_path),
        "parameters": parameter_count,
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2))


def predict(checkpoint: dict, features: np.ndarray, method: str) -> np.ndarray:
    x_mean = checkpoint["x_mean"].numpy()
    x_scale = checkpoint["x_scale"].numpy()
    y_mean = checkpoint["y_mean"].numpy()
    y_scale = checkpoint["y_scale"].numpy()
    x = (features - x_mean) / x_scale
    if method == "linear":
        normalized = linear_predict(x, checkpoint["linear_weights"].numpy())
    elif method == "mlp":
        model = UtilityMLP(
            len(FEATURE_NAMES), len(TARGET_NAMES), checkpoint["mlp_width"])
        model.load_state_dict(checkpoint["mlp_state_dict"])
        model.eval()
        with torch.inference_mode():
            normalized = model(torch.from_numpy(x.astype(np.float32))).numpy()
    else:
        # The fixed heuristic is expressed in the same target schema so the
        # downstream budget solver and accounting remain identical.
        normalized = np.zeros((len(features), len(TARGET_NAMES)), dtype=np.float32)
        gradient = features[:, FEATURE_NAMES.index("gradient_mean")]
        motion = features[:, FEATURE_NAMES.index("motion_mean")]
        structure = gradient + motion
        normalized[:, TARGET_NAMES.index("generate_lpips_gain_vs_base")] = (
            np.median(structure) - structure)
        normalized[:, TARGET_NAMES.index("enhance_lpips_gain_vs_base")] = (
            structure - np.median(structure))
        normalized[:, TARGET_NAMES.index(
            "generate_roi_seconds_measured_geometry_class")] = 0
        normalized[:, TARGET_NAMES.index("enhance_fallback_extra_on_disk_bytes")] = 0
        # Fixed heuristic values above are already in natural units except cost.
        natural = normalized.copy()
        natural[:, TARGET_NAMES.index(
            "generate_roi_seconds_measured_geometry_class")] = (
                float(y_mean[TARGET_NAMES.index(
                    "generate_roi_seconds_measured_geometry_class")]))
        natural[:, TARGET_NAMES.index("enhance_fallback_extra_on_disk_bytes")] = (
            float(y_mean[TARGET_NAMES.index(
                "enhance_fallback_extra_on_disk_bytes")]))
        return natural
    return normalized * y_scale + y_mean


def generate_components(actions: np.ndarray) -> list[list[int]]:
    grid = actions.reshape(4, 4)
    seen = set()
    components = []
    for row, column in zip(*np.where(grid == ACTION_GENERATE)):
        start = (int(row), int(column))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current_row, current_column = stack.pop()
            component.append(current_row * 4 + current_column)
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                candidate = (next_row, next_column)
                if (0 <= next_row < 4 and 0 <= next_column < 4
                        and candidate not in seen
                        and grid[next_row, next_column] == ACTION_GENERATE):
                    seen.add(candidate)
                    stack.append(candidate)
        components.append(sorted(component))
    return components


def solve_actions(
    predictions: np.ndarray, byte_budget: int, generate_budget: int,
    temporal_risk_weight: float, psnr_risk_weight: float,
    compute_time_weight: float, fragment_penalty: float,
) -> tuple[np.ndarray, dict]:
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
                    ACTION_ENHANCE, int(enhance_costs[region]), 0,
                    float(enhance_scores[region])))
            for action, byte_cost, generate_cost, gain in choices:
                key = (used_bytes + byte_cost, used_generate + generate_cost)
                candidate = (score + gain, actions + (action,))
                incumbent = updated.get(key)
                if incumbent is None or candidate[0] > incumbent[0]:
                    updated[key] = candidate
        states = updated
    _, (_, selected) = max(
        states.items(), key=lambda item: (item[1][0], -item[0][0], -item[0][1]))
    actions = np.asarray(selected, dtype=np.uint8)

    # The measured independent-ROI time already enters each Generate score.
    # This additional structural guard discourages fragmentation because
    # connected components still incur launch/model overhead that a sum of
    # per-geometry costs cannot capture. Formal routes are timed end to end.
    changed = True
    while changed:
        changed = False
        grid = actions.reshape(4, 4)
        for region in np.flatnonzero(actions == ACTION_GENERATE):
            row, column = divmod(int(region), 4)
            neighbors = [
                grid[next_row, next_column]
                for next_row, next_column in (
                    (row - 1, column), (row + 1, column),
                    (row, column - 1), (row, column + 1),
                )
                if 0 <= next_row < 4 and 0 <= next_column < 4
            ]
            if (ACTION_GENERATE not in neighbors
                    and generate_scores[region] < fragment_penalty):
                actions[region] = ACTION_BASE
                changed = True

    components = generate_components(actions)
    used_bytes = int(sum(
        enhance_costs[index]
        for index in np.flatnonzero(actions == ACTION_ENHANCE)))
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
    }


def route_main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("checkpoint feature schema mismatch")
    if checkpoint["target_names"] != list(TARGET_NAMES):
        raise ValueError("checkpoint target schema mismatch")
    records = read_jsonl(args.sample_manifest)
    if args.limit is not None:
        records = records[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for record in records:
        originals = load_source(record)
        boxes = tile_boxes(512, 512, 128)
        feature_records = source_features(originals, boxes)
        features = np.asarray([
            [item[name] for name in FEATURE_NAMES]
            for item in feature_records
        ], dtype=np.float32)
        predictions = predict(checkpoint, features, args.method)
        cost_index = TARGET_NAMES.index("enhance_fallback_extra_on_disk_bytes")
        predicted_total_cost = float(np.maximum(
            predictions[:, cost_index], TILE_HEADER.size + 1).sum())
        variants = {}
        actions_by_budget = []
        for budget_index, (ratio, generate_tiles) in enumerate(zip(
            args.byte_budget_ratios, args.generate_budget_tiles
        )):
            byte_budget = int(round(ratio * predicted_total_cost))
            actions, diagnostics = solve_actions(
                predictions, byte_budget, generate_tiles,
                args.temporal_risk_weight, args.psnr_risk_weight,
                args.compute_time_weight, args.fragment_penalty)
            name = f"{args.method}-budget-{budget_index}"
            actions_by_budget.append(actions)
            variants[name] = {
                "method": args.method,
                "actions": actions.tolist(),
                "action_counts": action_counts(actions),
                "budget": {
                    "enhance_fallback_byte_ratio": ratio,
                    "enhance_fallback_byte_budget": byte_budget,
                    "generate_tile_budget": generate_tiles,
                    "compute_time_weight": args.compute_time_weight,
                },
                "diagnostics": diagnostics,
            }
        changes = sum(
            len(set(int(actions[index]) for actions in actions_by_budget)) > 1
            for index in range(16)
        )
        selected_variant = f"{args.method}-budget-{len(actions_by_budget) // 2}"
        result = {
            "experiment": "A800 deployable lightweight-controller route",
            "sample": record,
            "route_kind": "learned-controller" if args.method != "fixed" else "fixed-heuristic",
            "selected_variant": selected_variant,
            "configuration": {
                "tile_size": 128,
                "tile_grid": [4, 4],
                "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
            },
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "encoder_visible_features": feature_records,
            "soft_predictions": predictions.tolist(),
            "variants": variants,
            "regions_changing_action_across_budgets": changes,
            "scientific_boundary": {
                "source_rgb_used_at_encoder": True,
                "ground_truth_quality_metrics_used_for_route": False,
                "teacher_targets_used_for_route": False,
                "budget_known_before_encoding": True,
                "action_map_must_be_written_to_stream": True,
                "predicted_fallback_bytes_are_final_spatial_bytes": False,
            },
        }
        path = args.output_dir / f"{record['sample_id']}.json"
        atomic_json(path, result)
        entries.append({
            "sample_id": record["sample_id"],
            "path": str(path),
            "selected_variant": result["selected_variant"],
            "regions_changing_action_across_budgets": changes,
        })
        print(json.dumps({
            "stage": "controller-route",
            "sample_id": record["sample_id"],
            "selected": result["selected_variant"],
            "selected_counts": variants[result["selected_variant"]]["action_counts"],
            "regions_changing_action": changes,
        }, ensure_ascii=False), flush=True)
    manifest = {
        "experiment": "A800 controller route manifest",
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "sample_count": len(entries),
        "entries": entries,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_main(args)
    else:
        route_main(args)


if __name__ == "__main__":
    main()
