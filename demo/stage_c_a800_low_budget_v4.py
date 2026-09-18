#!/usr/bin/env python3
"""Anchored v1 + v2 context + v3 Base-probe low-budget controller v4.

The original v1 controller remains the anchor.  Small independent towers learn
direct corrections to its Generate utility, Enhance gain, and Enhance cost.
Grouped out-of-fold replay compares context-only, probe-only, and combined
corrections without a hard promotion gate, then freezes the best candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_controller import (
    TARGET_NAMES,
    atomic_json,
    predict as predict_v1,
    read_jsonl,
    read_teacher_rows,
)
from demo.stage_c_a800_low_budget_v2 import (
    CONTEXT_FEATURE_NAMES,
    ENSEMBLE_SIZE,
    FOLD_COUNT,
    LOW_BYTE_RATIO,
    LOW_GENERATE_TILES,
    atomic_npz,
    engineer_features as engineer_v2_features,
    fit_one as fit_indirect,
    fold_for_sequence,
    grouped_indices,
    normalize_from_train,
    single_sample_context_features,
)
from demo.stage_c_a800_low_budget_v3 import (
    BASE_PROBE_FEATURE_NAMES,
    BASE_PROBE_NAMES,
    DIRECT_TARGET_NAMES,
    GAIN_PENALTIES,
    IndependentDirectMLP,
    base_probe_for_route,
    direct_regression_metrics,
    direct_targets,
    evaluate_candidate,
    fit_one as fit_direct,
    read_base_probe_training_features,
    solve_direct,
)
from demo.stage_c_a800_teacher import (
    FEATURE_NAMES,
    SpatialLPIPS,
    load_source,
    source_features,
)
from demo.stage_c_evaluate_seedvr2_gate import load_pngs
from demo.stage_c_three_path_roi_probe import (
    TILE_HEADER,
    action_counts,
    tile_boxes,
)


PROBE_NEIGHBOR_FEATURE_NAMES = tuple(
    f"neighbor_delta_{name}" for name in BASE_PROBE_NAMES)
PROBE_CONTEXT_FEATURE_NAMES = (
    tuple(BASE_PROBE_FEATURE_NAMES) + PROBE_NEIGHBOR_FEATURE_NAMES)
HYBRID_FEATURE_NAMES = (
    tuple(CONTEXT_FEATURE_NAMES)
    + tuple(BASE_PROBE_NAMES)
    + tuple(f"sample_delta_{name}" for name in BASE_PROBE_NAMES)
    + tuple(f"sample_z_{name}" for name in BASE_PROBE_NAMES)
    + PROBE_NEIGHBOR_FEATURE_NAMES
)
BASE_DIRECT_FEATURE_NAMES = tuple(
    f"v1_{name}" for name in DIRECT_TARGET_NAMES)

FEATURE_KINDS = (
    "context-residual",
    "probe-residual",
    "hybrid-residual",
)
FEATURE_NAMES_BY_KIND = {
    "context-residual": tuple(CONTEXT_FEATURE_NAMES),
    "probe-residual": PROBE_CONTEXT_FEATURE_NAMES,
    "hybrid-residual": HYBRID_FEATURE_NAMES,
}
CORRECTION_SCALES = (0.25, 0.5, 0.75, 1.0)
BASE_SEED = 20260917
CORRECTION_SEED = 20260918
PROTOCOL_PATH = "docs/CLOUD_A800_LOW_BUDGET_V4.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anchored context + Base-probe low-budget controller v4")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    cross_validate = subparsers.add_parser("cross-validate")
    add_training_inputs(cross_validate)
    cross_validate.add_argument("--output-dir", type=Path, required=True)

    train = subparsers.add_parser("train")
    add_training_inputs(train)
    train.add_argument("--selection-summary", type=Path, required=True)
    train.add_argument("--v1-checkpoint", type=Path, required=True)
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
            parser.error(f"ensemble size is fixed at {ENSEMBLE_SIZE}")
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
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--correction-seed", type=int, default=CORRECTION_SEED)
    parser.add_argument("--cuda-idx", type=int, default=0)


def probe_neighbor_delta(probe: np.ndarray) -> np.ndarray:
    if probe.shape != (16, len(BASE_PROBE_NAMES)):
        raise ValueError(f"unexpected Base probe shape: {probe.shape}")
    output = np.empty_like(probe, dtype=np.float32)
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
        output[region] = probe[region] - probe[neighbors].mean(axis=0)
    return output


def single_sample_probe_context_features(
    raw_features: np.ndarray, probe: np.ndarray,
) -> np.ndarray:
    if raw_features.shape != (16, len(FEATURE_NAMES)):
        raise ValueError(f"unexpected raw feature shape: {raw_features.shape}")
    center = probe.mean(axis=0)
    scale = probe.std(axis=0)
    scale[scale < 1e-8] = 1.0
    delta = probe - center
    return np.concatenate([
        raw_features,
        probe,
        delta,
        delta / scale,
        probe_neighbor_delta(probe),
    ], axis=1).astype(np.float32)


def single_sample_hybrid_features(
    raw_features: np.ndarray, probe: np.ndarray,
) -> np.ndarray:
    context = single_sample_context_features(raw_features)
    center = probe.mean(axis=0)
    scale = probe.std(axis=0)
    scale[scale < 1e-8] = 1.0
    delta = probe - center
    return np.concatenate([
        context,
        probe,
        delta,
        delta / scale,
        probe_neighbor_delta(probe),
    ], axis=1).astype(np.float32)


def single_sample_features(
    raw_features: np.ndarray,
    probe: np.ndarray | None,
    kind: str,
) -> np.ndarray:
    if kind == "context-residual":
        return single_sample_context_features(raw_features)
    if probe is None:
        raise ValueError(f"{kind} requires a Base probe")
    if kind == "probe-residual":
        return single_sample_probe_context_features(raw_features, probe)
    if kind == "hybrid-residual":
        return single_sample_hybrid_features(raw_features, probe)
    raise ValueError(f"unknown correction feature kind: {kind}")


def engineer_feature_family(
    raw_features: np.ndarray,
    probes: np.ndarray,
    metadata: list[dict],
    kind: str,
) -> np.ndarray:
    if kind == "context-residual":
        return engineer_v2_features(raw_features, metadata, "context")
    names = FEATURE_NAMES_BY_KIND[kind]
    output = np.empty((len(raw_features), len(names)), dtype=np.float32)
    for indices in grouped_indices(metadata):
        output[indices] = single_sample_features(
            raw_features[indices], probes[indices], kind)
    return output


def fit_fold_v1(
    raw_features: np.ndarray,
    indirect_targets: np.ndarray,
    train_mask: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    normalized_x, _, _ = normalize_from_train(
        raw_features[train_mask], raw_features)
    normalized_y, y_mean, y_scale = normalize_from_train(
        indirect_targets[train_mask], indirect_targets)
    seed = args.base_seed + fold * 1000
    _, normalized_prediction, loss = fit_indirect(
        normalized_x[train_mask],
        normalized_y[train_mask],
        normalized_x,
        64,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        seed,
        device,
    )
    prediction = normalized_prediction * y_scale + y_mean
    return prediction.astype(np.float32), {
        "seed": seed,
        "final_standardized_mse": loss,
    }


def fit_fold_correction(
    correction_features: np.ndarray,
    residual_targets: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    kind: str,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    normalized_x, _, _ = normalize_from_train(
        correction_features[train_mask], correction_features)
    normalized_y, y_mean, y_scale = normalize_from_train(
        residual_targets[train_mask], residual_targets)
    predictions = []
    records = []
    for member in range(args.ensemble_size):
        seed = args.correction_seed + fold * 1000 + member
        _, normalized_prediction, loss = fit_direct(
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
        records.append({
            "member": member,
            "seed": seed,
            "final_standardized_mse": loss,
        })
        print(json.dumps({
            "stage": "low-budget-v4-cross-validation",
            "feature_kind": kind,
            "fold": fold,
            **records[-1],
        }), flush=True)
    stacked = np.stack(predictions)
    return (
        stacked.mean(axis=0).astype(np.float32),
        stacked.std(axis=0).astype(np.float32),
        records,
    )


def out_of_fold_predictions(
    raw_features: np.ndarray,
    probes: np.ndarray,
    indirect_targets: np.ndarray,
    metadata: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    direct_reference = direct_targets(indirect_targets)
    feature_families = {
        kind: engineer_feature_family(raw_features, probes, metadata, kind)
        for kind in FEATURE_KINDS
    }
    folds = np.asarray([
        fold_for_sequence(item["sequence"]) for item in metadata
    ], dtype=np.int64)
    base_output = np.empty_like(direct_reference)
    corrections = {
        kind: np.empty_like(direct_reference) for kind in FEATURE_KINDS
    }
    disagreements = {
        kind: np.empty_like(direct_reference) for kind in FEATURE_KINDS
    }
    fold_records = []
    for fold in range(FOLD_COUNT):
        train_mask = folds != fold
        validation_mask = folds == fold
        if not np.any(train_mask) or not np.any(validation_mask):
            raise ValueError(f"empty train or validation partition in fold {fold}")
        indirect_prediction, base_record = fit_fold_v1(
            raw_features, indirect_targets, train_mask, fold, args, device)
        base_direct = direct_targets(indirect_prediction)
        base_output[validation_mask] = base_direct[validation_mask]
        residual = direct_reference - base_direct
        correction_records = {}
        for kind in FEATURE_KINDS:
            correction_input = np.concatenate(
                [feature_families[kind], base_direct], axis=1).astype(np.float32)
            mean, std, records = fit_fold_correction(
                correction_input,
                residual,
                train_mask,
                validation_mask,
                kind,
                fold,
                args,
                device,
            )
            corrections[kind][validation_mask] = mean
            disagreements[kind][validation_mask] = std
            correction_records[kind] = records
        fold_records.append({
            "fold": fold,
            "train_region_count": int(np.count_nonzero(train_mask)),
            "validation_region_count": int(np.count_nonzero(validation_mask)),
            "train_sequence_count": len(set(
                item["sequence"] for item, keep in zip(metadata, train_mask)
                if keep)),
            "validation_sequence_count": len(set(
                item["sequence"]
                for item, keep in zip(metadata, validation_mask) if keep)),
            "v1_anchor": base_record,
            "correction_members": correction_records,
        })
    return base_output, corrections, disagreements, fold_records


def prediction_rmse(
    reference: np.ndarray, prediction: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(np.sqrt(np.mean(
            (prediction[:, index] - reference[:, index]) ** 2)))
        for index, name in enumerate(DIRECT_TARGET_NAMES)
    }


def calibrated_candidates(
    prediction: np.ndarray,
    reference: np.ndarray,
    metadata: list[dict],
    feature_kind: str,
    correction_scale: float,
) -> list[dict]:
    rmse = prediction_rmse(reference, prediction)
    result = []
    for generate_alpha in GAIN_PENALTIES:
        for enhance_alpha in GAIN_PENALTIES:
            item = evaluate_candidate(
                prediction,
                reference,
                metadata,
                generate_alpha,
                enhance_alpha,
                rmse,
            )
            item["feature_kind"] = feature_kind
            item["correction_scale"] = correction_scale
            result.append(item)
    return result


def candidate_key(item: dict) -> tuple:
    complexity = {
        "v1": 0,
        "context-residual": 1,
        "probe-residual": 2,
        "hybrid-residual": 3,
    }[item["feature_kind"]]
    return (
        item["mean_oracle_regret"],
        -item["worst_quartile_mean_true_utility"],
        item["harmful_selected_generate_fraction"],
        item["correction_scale"],
        complexity,
        item["generate_penalty_rmse_multiplier"]
        + item["enhance_penalty_rmse_multiplier"],
    )


def cross_validate_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    raw, indirect, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    if len(metadata) != 8000:
        raise ValueError(f"expected 8000 training regions, found {len(metadata)}")
    probes = read_base_probe_training_features(
        args.train_teacher_manifest, metadata)
    reference = direct_targets(indirect)
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    base, corrections, disagreements, folds = out_of_fold_predictions(
        raw, probes, indirect, metadata, args, device)

    candidates = calibrated_candidates(
        base, reference, metadata, "v1", 0.0)
    models = {
        "v1": {
            "feature_count": len(FEATURE_NAMES),
            "correction_feature_count": 0,
            "direct_regression_metrics": direct_regression_metrics(
                reference, base),
        }
    }
    saved = {
        "targets": reference,
        "base_probe": probes,
        "v1_direct_prediction": base,
    }
    for kind in FEATURE_KINDS:
        saved[f"{kind}_correction"] = corrections[kind]
        saved[f"{kind}_ensemble_std"] = disagreements[kind]
        scale_models = {}
        for correction_scale in CORRECTION_SCALES:
            prediction = base + correction_scale * corrections[kind]
            candidates.extend(calibrated_candidates(
                prediction,
                reference,
                metadata,
                kind,
                correction_scale,
            ))
            scale_models[str(correction_scale)] = {
                "direct_regression_metrics": direct_regression_metrics(
                    reference, prediction),
                "residual_rmse": prediction_rmse(reference, prediction),
            }
        models[kind] = {
            "feature_count": len(FEATURE_NAMES_BY_KIND[kind]),
            "correction_feature_count_including_v1_prediction": (
                len(FEATURE_NAMES_BY_KIND[kind]) + len(DIRECT_TARGET_NAMES)),
            "tower_width": 32,
            "mean_correction_ensemble_disagreement": {
                name: float(disagreements[kind][:, index].mean())
                for index, name in enumerate(DIRECT_TARGET_NAMES)
            },
            "scales": scale_models,
        }
    selected = min(candidates, key=candidate_key)
    for model_name in ("v1", *FEATURE_KINDS):
        eligible = [
            item for item in candidates if item["feature_kind"] == model_name
        ]
        models[model_name]["best_route_candidate"] = min(
            eligible, key=candidate_key)

    summary = {
        "experiment": "A800 low-budget anchored hybrid controller v4",
        "status": "selection-complete",
        "protocol": PROTOCOL_PATH,
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
            "correction_ensemble_size": args.ensemble_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "base_seed": args.base_seed,
            "correction_seed": args.correction_seed,
            "device": str(device),
            "architecture": (
                "v1 indirect anchor plus three independent 2x32 direct "
                "residual towers"),
        },
        "budget": {
            "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
            "generate_tile_budget": LOW_GENERATE_TILES,
        },
        "selection_policy": {
            "hard_promotion_gate": False,
            "primary": "minimum mean grouped-OOF oracle regret",
            "secondary": "maximum worst-quartile mean true utility",
            "tertiary": "minimum harmful selected Generate fraction",
            "final_tie_break": "smaller correction then simpler model",
            "development_candidates_after_selection": 1,
        },
        "feature_schemas": {
            kind: list(names) for kind, names in FEATURE_NAMES_BY_KIND.items()
        },
        "base_direct_feature_names": list(BASE_DIRECT_FEATURE_NAMES),
        "direct_target_names": list(DIRECT_TARGET_NAMES),
        "correction_scales": list(CORRECTION_SCALES),
        "folds": folds,
        "models": models,
        "candidate_count": len(candidates),
        "selected_candidate": selected,
        "next_step": (
            "train selected full-data controller and route val/000..005 once"),
        "scientific_boundary": {
            "selection_uses_training_labels_only": True,
            "folds_grouped_by_source_sequence": True,
            "v1_is_the_zero_correction_anchor": True,
            "v2_context_is_encoder_visible": True,
            "base_probe_is_encoder_reproducible": True,
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
        "selected_candidate": selected,
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def fit_full_correction(
    correction_features: np.ndarray,
    residual_targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict, dict]:
    normalized_x, x_mean, x_scale = normalize_from_train(
        correction_features, correction_features)
    normalized_y, y_mean, y_scale = normalize_from_train(
        residual_targets, residual_targets)
    states = []
    predictions = []
    losses = []
    seeds = []
    for member in range(args.ensemble_size):
        seed = args.correction_seed + 50000 + member
        state, normalized_prediction, loss = fit_direct(
            normalized_x,
            normalized_y,
            normalized_x,
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
            "stage": "low-budget-v4-full-train",
            "member": member,
            "seed": seed,
            "final_standardized_mse": loss,
        }), flush=True)
    normalization = {
        "correction_x_mean": torch.from_numpy(x_mean),
        "correction_x_scale": torch.from_numpy(x_scale),
        "correction_y_mean": torch.from_numpy(y_mean),
        "correction_y_scale": torch.from_numpy(y_scale),
    }
    training = {
        "member_seeds": seeds,
        "member_final_standardized_mse": losses,
        "ensemble_train_residual_metrics": direct_regression_metrics(
            residual_targets, np.stack(predictions).mean(axis=0)),
    }
    return states, normalization, training


def train_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    if selection["selection_policy"].get("hard_promotion_gate") is not False:
        raise RuntimeError("v4 selection unexpectedly contains a hard gate")
    if selection["inputs"].get("development_or_test_data_used") is not False:
        raise RuntimeError("selection summary is not training-only")
    selected = selection["selected_candidate"]
    selected_kind = selected["feature_kind"]
    raw, indirect, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    probes = read_base_probe_training_features(
        args.train_teacher_manifest, metadata)
    direct_reference = direct_targets(indirect)
    base_checkpoint = torch.load(
        args.v1_checkpoint, map_location="cpu", weights_only=False)
    if base_checkpoint.get("format") != "A800 regional utility controller v1":
        raise ValueError("the anchor checkpoint is not controller v1")
    if base_checkpoint["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("v1 anchor feature schema mismatch")
    base_indirect = predict_v1(base_checkpoint, raw, "mlp")
    base_direct = direct_targets(base_indirect)

    states: list[dict] = []
    normalization: dict = {}
    correction_training: dict = {
        "member_seeds": [],
        "member_final_standardized_mse": [],
        "ensemble_train_residual_metrics": None,
    }
    correction_feature_names: tuple[str, ...] = tuple()
    if selected_kind != "v1":
        engineered = engineer_feature_family(
            raw, probes, metadata, selected_kind)
        correction_input = np.concatenate(
            [engineered, base_direct], axis=1).astype(np.float32)
        residual = direct_reference - base_direct
        device = torch.device(
            f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
        states, normalization, correction_training = fit_full_correction(
            correction_input, residual, args, device)
        correction_feature_names = (
            FEATURE_NAMES_BY_KIND[selected_kind] + BASE_DIRECT_FEATURE_NAMES)

    checkpoint = {
        "format": "A800 low-budget controller v4 anchored hybrid residual",
        "protocol": PROTOCOL_PATH,
        "selected_feature_kind": selected_kind,
        "correction_scale": selected["correction_scale"],
        "raw_feature_names": list(FEATURE_NAMES),
        "base_probe_names": list(BASE_PROBE_NAMES),
        "engineered_feature_names": (
            list(FEATURE_NAMES_BY_KIND[selected_kind])
            if selected_kind != "v1" else []),
        "correction_feature_names": list(correction_feature_names),
        "direct_target_names": list(DIRECT_TARGET_NAMES),
        "correction_input_dim": len(correction_feature_names),
        "correction_tower_width": 32,
        "correction_ensemble_size": len(states),
        "correction_state_dicts": states,
        **normalization,
        "v1_anchor_checkpoint": base_checkpoint,
        "v1_anchor_source": {
            "path": str(args.v1_checkpoint.resolve()),
            "sha256": hashlib.sha256(args.v1_checkpoint.read_bytes()).hexdigest(),
        },
        "calibration": {
            "generate_penalty_rmse_multiplier": selected[
                "generate_penalty_rmse_multiplier"],
            "enhance_penalty_rmse_multiplier": selected[
                "enhance_penalty_rmse_multiplier"],
            "absolute_generate_utility_penalty": selected[
                "absolute_generate_utility_penalty"],
            "absolute_enhance_gain_penalty": selected[
                "absolute_enhance_gain_penalty"],
        },
        "budget": selection["budget"],
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "correction_seed": args.correction_seed,
            "region_count": len(metadata),
            "sample_count": len(grouped_indices(metadata)),
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
            **correction_training,
        },
        "selection": {
            "summary": str(args.selection_summary.resolve()),
            "summary_sha256": hashlib.sha256(
                args.selection_summary.read_bytes()).hexdigest(),
            "hard_promotion_gate": False,
            "selected_candidate": selected,
        },
        "scientific_boundary": selection["scientific_boundary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller_v4.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    parameter_count = (
        sum(value.numel() for value in states[0].values()) if states else 0)
    summary = {
        "experiment": "A800 low-budget v4 full training",
        "status": "trained",
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "selected_feature_kind": selected_kind,
        "correction_scale": selected["correction_scale"],
        "correction_parameter_count_per_member": parameter_count,
        "correction_parameter_count_total": parameter_count * len(states),
        "v1_anchor_parameter_count": sum(
            value.numel()
            for value in base_checkpoint["mlp_state_dict"].values()),
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
        "selected_feature_kind": selected_kind,
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def predict_correction(
    checkpoint: dict,
    correction_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if not checkpoint["correction_state_dicts"]:
        zeros = np.zeros(
            (len(correction_features), len(DIRECT_TARGET_NAMES)),
            dtype=np.float32)
        return zeros, zeros.copy()
    normalized = (
        (correction_features - checkpoint["correction_x_mean"].numpy())
        / checkpoint["correction_x_scale"].numpy()
    ).astype(np.float32)
    y_mean = checkpoint["correction_y_mean"].numpy()
    y_scale = checkpoint["correction_y_scale"].numpy()
    predictions = []
    for state in checkpoint["correction_state_dicts"]:
        model = IndependentDirectMLP(
            checkpoint["correction_input_dim"],
            checkpoint["correction_tower_width"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            value = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        predictions.append(value * y_scale + y_mean)
    stacked = np.stack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0)


def load_route_probe(
    record: dict,
    base_probe_root: Path,
    metric: SpatialLPIPS,
) -> tuple[np.ndarray, dict]:
    sample_id = record["sample_id"]
    gate_root = base_probe_root / sample_id / "uniform_gate"
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
    feature_seconds = time.perf_counter() - feature_started
    return probe, {
        "qp": 16,
        "temporary_stream": str(stream_path),
        "temporary_stream_bytes_not_charged_to_transmitted_rate": (
            qp16["rate"]["total_bytes"]),
        "reconstruction_dir": str(reconstruction_dir),
        "recorded_encode_seconds": qp16["encode"]["seconds"],
        "recorded_fresh_decode_seconds_median": qp16["runtime"][
            "fresh_decode_seconds_median"],
        "probe_metric_seconds_before_controller": feature_seconds,
        "cached_deterministic_reconstruction_reused": True,
    }


def route_main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != (
            "A800 low-budget controller v4 anchored hybrid residual"):
        raise ValueError("checkpoint is not low-budget controller v4")
    if checkpoint["raw_feature_names"] != list(FEATURE_NAMES):
        raise ValueError("v4 raw feature schema mismatch")
    if checkpoint["direct_target_names"] != list(DIRECT_TARGET_NAMES):
        raise ValueError("v4 direct target schema mismatch")
    selected_kind = checkpoint["selected_feature_kind"]
    if selected_kind not in {"v1", *FEATURE_KINDS}:
        raise ValueError(f"unknown selected v4 feature kind: {selected_kind}")
    needs_probe = selected_kind in {"probe-residual", "hybrid-residual"}
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    metric = None
    lpips_model_load_seconds = 0.0
    if needs_probe:
        model_started = time.perf_counter()
        metric = SpatialLPIPS(device, args.lpips_batch_size)
        lpips_model_load_seconds = time.perf_counter() - model_started
    records = read_jsonl(args.sample_manifest)
    if args.limit is not None:
        records = records[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    variant_name = "anchored-hybrid-low-v4"
    for record in records:
        sample_started = time.perf_counter()
        sample_id = record["sample_id"]
        originals = load_source(record)
        boxes = tile_boxes(512, 512, 128)
        raw_records = source_features(originals, boxes)
        raw = np.asarray([
            [item[name] for name in FEATURE_NAMES] for item in raw_records
        ], dtype=np.float32)
        base_indirect = predict_v1(
            checkpoint["v1_anchor_checkpoint"], raw, "mlp")
        base_direct = direct_targets(base_indirect)

        probe = None
        probe_record = None
        if needs_probe:
            assert metric is not None
            probe, probe_record = load_route_probe(
                record, args.base_probe_root, metric)
        controller_started = time.perf_counter()
        if selected_kind == "v1":
            engineered = np.empty((16, 0), dtype=np.float32)
            correction_input = np.empty((16, 0), dtype=np.float32)
            correction = np.zeros_like(base_direct)
            disagreement = np.zeros_like(base_direct)
        else:
            engineered = single_sample_features(raw, probe, selected_kind)
            correction_input = np.concatenate(
                [engineered, base_direct], axis=1).astype(np.float32)
            correction, disagreement = predict_correction(
                checkpoint, correction_input, device)
        direct_prediction = (
            base_direct + checkpoint["correction_scale"] * correction)
        calibrated = direct_prediction.copy()
        calibrated[:, 0] -= checkpoint["calibration"][
            "absolute_generate_utility_penalty"]
        calibrated[:, 1] -= checkpoint["calibration"][
            "absolute_enhance_gain_penalty"]
        predicted_costs = np.maximum(
            np.rint(calibrated[:, 2]).astype(np.int64),
            TILE_HEADER.size + 1)
        byte_budget = int(round(LOW_BYTE_RATIO * predicted_costs.sum()))
        actions, diagnostics = solve_direct(calibrated, byte_budget)
        controller_seconds = time.perf_counter() - controller_started
        if probe_record is not None:
            probe_record["feature_and_controller_seconds"] = (
                probe_record.pop("probe_metric_seconds_before_controller")
                + controller_seconds)
            probe_record["shared_lpips_model_load_seconds"] = (
                lpips_model_load_seconds)
            probe_record[
                "estimated_encoder_analysis_seconds_excluding_shared_model_load"
            ] = (
                probe_record["recorded_encode_seconds"]
                + probe_record["recorded_fresh_decode_seconds_median"]
                + probe_record["feature_and_controller_seconds"])
        result = {
            "experiment": "A800 deployable anchored hybrid low-budget v4 route",
            "sample": record,
            "route_kind": "v1-anchored-context-base-probe-residual-controller",
            "selected_variant": variant_name,
            "selected_feature_kind": selected_kind,
            "configuration": {
                "tile_size": 128,
                "tile_grid": [4, 4],
                "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
            },
            "raw_feature_names": list(FEATURE_NAMES),
            "base_probe_names": list(BASE_PROBE_NAMES),
            "engineered_feature_names": checkpoint["engineered_feature_names"],
            "correction_feature_names": checkpoint["correction_feature_names"],
            "direct_target_names": list(DIRECT_TARGET_NAMES),
            "encoder_visible_features": raw_records,
            "base_probe_features": probe.tolist() if probe is not None else None,
            "engineered_features": engineered.tolist(),
            "v1_indirect_predictions": base_indirect.tolist(),
            "v1_direct_anchor_predictions": base_direct.tolist(),
            "uncalibrated_residual_correction": correction.tolist(),
            "residual_ensemble_std": disagreement.tolist(),
            "correction_scale": checkpoint["correction_scale"],
            "uncalibrated_direct_predictions": direct_prediction.tolist(),
            "direct_predictions": calibrated.tolist(),
            "calibration": checkpoint["calibration"],
            "base_probe": probe_record,
            "variants": {
                variant_name: {
                    "method": "anchored-hybrid-residual-v4",
                    "actions": actions.tolist(),
                    "action_counts": action_counts(actions),
                    "budget": {
                        "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
                        "enhance_fallback_byte_budget": byte_budget,
                        "generate_tile_budget": LOW_GENERATE_TILES,
                    },
                    "diagnostics": diagnostics,
                }
            },
            "route_wall_seconds": time.perf_counter() - sample_started,
            "scientific_boundary": {
                "v1_is_the_default_anchor": True,
                "v2_context_uses_encoder_visible_regions_only": True,
                "base_probe_uses_encoder_source_and_reconstruction": needs_probe,
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
            "selected_feature_kind": selected_kind,
            "action_counts": action_counts(actions),
            "base_probe": probe_record,
        })
        print(json.dumps({
            "stage": "low-budget-v4-route",
            "sample_id": sample_id,
            "selected_feature_kind": selected_kind,
            "selected_counts": action_counts(actions),
            "encoder_analysis_seconds": (
                probe_record[
                    "estimated_encoder_analysis_seconds_excluding_shared_model_load"
                ] if probe_record is not None else controller_seconds),
        }, ensure_ascii=False), flush=True)
    manifest = {
        "experiment": "A800 anchored hybrid low-budget v4 route manifest",
        "checkpoint": str(args.checkpoint),
        "sample_count": len(entries),
        "selected_feature_kind": selected_kind,
        "lpips_model_load_seconds_shared": lpips_model_load_seconds,
        "entries": entries,
        "scientific_boundary": {
            "development_or_test_quality_results_used_for_routing": False,
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
