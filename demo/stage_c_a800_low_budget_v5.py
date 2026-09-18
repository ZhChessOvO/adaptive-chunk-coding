#!/usr/bin/env python3
"""Conservative v1 correction from independent context and Base-probe experts.

v5 keeps the deployed v1 controller as the default.  The v2 context residual
and v3 Base-probe residual are trained independently.  A target is corrected
only when both experts agree on its direction and both lower confidence bounds
remain non-zero.  The applied magnitude is the smaller lower bound, so neither
expert can unilaterally make an aggressive change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    atomic_json,
    predict as predict_v1,
    read_jsonl,
    read_teacher_rows,
)
from demo.stage_c_a800_low_budget_v2 import (
    ENSEMBLE_SIZE,
    LOW_BYTE_RATIO,
    LOW_GENERATE_TILES,
    grouped_indices,
)
from demo.stage_c_a800_low_budget_v3 import (
    BASE_PROBE_NAMES,
    DIRECT_TARGET_NAMES,
    GAIN_PENALTIES,
    IndependentDirectMLP,
    direct_regression_metrics,
    direct_targets,
    evaluate_candidate,
    read_base_probe_training_features,
    solve_direct,
)
from demo.stage_c_a800_low_budget_v4 import (
    BASE_DIRECT_FEATURE_NAMES,
    FEATURE_NAMES_BY_KIND,
    engineer_feature_family,
    fit_full_correction,
    load_route_probe,
    prediction_rmse,
    single_sample_features,
)
from demo.stage_c_a800_teacher import FEATURE_NAMES, SpatialLPIPS, load_source, source_features
from demo.stage_c_three_path_roi_probe import TILE_HEADER, action_counts, tile_boxes


EXPERT_KINDS = ("context-residual", "probe-residual")
CONFIDENCE_Z_VALUES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
CORRECTION_SCALES = (0.25, 0.5, 0.75, 1.0)
BASE_SEED = 20260917
CORRECTION_SEED = 20260919
PROTOCOL_PATH = "docs/CLOUD_A800_LOW_BUDGET_V5.md"
CHECKPOINT_FORMAT = "A800 low-budget controller v5 conservative consensus"
VARIANT_NAME = "conservative-consensus-low-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative context + Base-probe consensus controller v5")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--oof-predictions", type=Path, required=True)
    select.add_argument("--train-teacher-manifest", type=Path, required=True)
    select.add_argument("--train-roi-cost-manifest", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)

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
    if args.mode == "train":
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("epochs and batch size must be positive")
        if args.ensemble_size != ENSEMBLE_SIZE:
            parser.error(f"ensemble size is fixed at {ENSEMBLE_SIZE}")
    elif args.mode == "route" and args.lpips_batch_size < 1:
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


def conservative_consensus(
    context_mean: np.ndarray,
    context_std: np.ndarray,
    probe_mean: np.ndarray,
    probe_std: np.ndarray,
    confidence_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the smaller same-sign lower confidence bound per target."""
    arrays = (context_mean, context_std, probe_mean, probe_std)
    if any(value.shape != context_mean.shape for value in arrays):
        raise ValueError("expert correction arrays do not share a shape")
    if confidence_z < 0:
        raise ValueError("confidence_z must be non-negative")
    sign_agreement = (
        (np.sign(context_mean) == np.sign(probe_mean))
        & (np.sign(context_mean) != 0)
    )
    context_lower = np.maximum(
        np.abs(context_mean) - confidence_z * context_std, 0.0)
    probe_lower = np.maximum(
        np.abs(probe_mean) - confidence_z * probe_std, 0.0)
    stable = (context_lower > 0) & (probe_lower > 0)
    active = sign_agreement & stable
    magnitude = np.minimum(context_lower, probe_lower)
    correction = np.where(
        active, np.sign(context_mean) * magnitude, 0.0).astype(np.float32)
    return correction, active, sign_agreement, stable


def calibrated_candidates(
    prediction: np.ndarray,
    reference: np.ndarray,
    metadata: list[dict],
    method: str,
    confidence_z: float | None,
    correction_scale: float,
    diagnostics: dict,
) -> list[dict]:
    rmse = prediction_rmse(reference, prediction)
    output = []
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
            item.update({
                "method": method,
                "confidence_z": confidence_z,
                "correction_scale": correction_scale,
                **diagnostics,
            })
            output.append(item)
    return output


def candidate_key(item: dict) -> tuple:
    return (
        item["mean_oracle_regret"],
        -item["worst_quartile_mean_true_utility"],
        item["harmful_selected_generate_fraction"],
        item["mean_absolute_applied_correction"],
        item["correction_scale"],
        item["generate_penalty_rmse_multiplier"]
        + item["enhance_penalty_rmse_multiplier"],
    )


def selection_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    _, indirect, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    if len(metadata) != 8000:
        raise ValueError(f"expected 8000 training regions, found {len(metadata)}")
    arrays = np.load(args.oof_predictions)
    required = (
        "targets",
        "v1_direct_prediction",
        "context-residual_correction",
        "context-residual_ensemble_std",
        "probe-residual_correction",
        "probe-residual_ensemble_std",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"v4 OOF archive is missing arrays: {missing}")
    reference = arrays["targets"]
    if not np.array_equal(reference, direct_targets(indirect)):
        raise RuntimeError("v4 OOF targets do not match current teacher manifests")
    base = arrays["v1_direct_prediction"]
    context_mean = arrays["context-residual_correction"]
    context_std = arrays["context-residual_ensemble_std"]
    probe_mean = arrays["probe-residual_correction"]
    probe_std = arrays["probe-residual_ensemble_std"]

    candidates = calibrated_candidates(
        base,
        reference,
        metadata,
        "v1",
        None,
        0.0,
        {
            "raw_sign_agreement_fraction": 0.0,
            "stable_target_fraction": 0.0,
            "active_target_fraction": 0.0,
            "mean_absolute_applied_correction": 0.0,
        },
    )
    grid = []
    for confidence_z in CONFIDENCE_Z_VALUES:
        correction, active, agreement, stable = conservative_consensus(
            context_mean,
            context_std,
            probe_mean,
            probe_std,
            confidence_z,
        )
        for correction_scale in CORRECTION_SCALES:
            applied = correction_scale * correction
            diagnostics = {
                "raw_sign_agreement_fraction": float(agreement.mean()),
                "stable_target_fraction": float(stable.mean()),
                "active_target_fraction": float(active.mean()),
                "mean_absolute_applied_correction": float(
                    np.mean(np.abs(applied))),
            }
            current = calibrated_candidates(
                base + applied,
                reference,
                metadata,
                "context-probe-consensus",
                confidence_z,
                correction_scale,
                diagnostics,
            )
            candidates.extend(current)
            grid.append({
                "confidence_z": confidence_z,
                "correction_scale": correction_scale,
                **diagnostics,
                "best_candidate": min(current, key=candidate_key),
            })
    selected = min(candidates, key=candidate_key)
    v1_best = min(
        (item for item in candidates if item["method"] == "v1"),
        key=candidate_key,
    )
    summary = {
        "experiment": "A800 low-budget conservative consensus controller v5",
        "status": "selection-complete",
        "protocol": PROTOCOL_PATH,
        "inputs": {
            "oof_predictions": str(args.oof_predictions.resolve()),
            "oof_predictions_sha256": hashlib.sha256(
                args.oof_predictions.read_bytes()).hexdigest(),
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
            "region_count": len(metadata),
            "sample_count": len(grouped_indices(metadata)),
            "development_or_evaluation_data_used": False,
        },
        "method": {
            "experts": list(EXPERT_KINDS),
            "agreement_rule": "same sign for each direct target",
            "uncertainty_rule": (
                "both abs(mean)-confidence_z*ensemble_std values are positive"),
            "magnitude_rule": "smaller of the two lower confidence bounds",
            "v1_is_default_when_inactive": True,
        },
        "selection_policy": {
            "hard_promotion_gate": False,
            "primary": "minimum mean grouped-OOF oracle regret",
            "secondary": "maximum worst-quartile mean true utility",
            "tertiary": "minimum harmful selected Generate fraction",
            "final_tie_break": "smaller applied correction",
            "development_candidates_after_selection": 1,
        },
        "confidence_z_values": list(CONFIDENCE_Z_VALUES),
        "correction_scales": list(CORRECTION_SCALES),
        "calibration_multipliers": list(GAIN_PENALTIES),
        "candidate_count": len(candidates),
        "v1_best_candidate": v1_best,
        "grid": grid,
        "selected_candidate": selected,
        "training_oof_regret_change_vs_v1": (
            selected["mean_oracle_regret"]
            - v1_best["mean_oracle_regret"]),
        "next_step": (
            "train both full-data experts and route val/000..005 once"),
        "scientific_boundary": {
            "selection_uses_training_labels_only": True,
            "source_sequences_are_grouped_in_saved_oof_predictions": True,
            "v1_is_the_default_anchor": True,
            "context_and_probe_experts_are_independent": True,
            "decoder_does_not_receive_source_rgb": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "selection_summary.json", summary)
    print(json.dumps({
        "summary": str(args.output_dir / "selection_summary.json"),
        "selected_candidate": selected,
        "v1_mean_oracle_regret": v1_best["mean_oracle_regret"],
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def train_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    if selection["selection_policy"].get("hard_promotion_gate") is not False:
        raise RuntimeError("v5 selection unexpectedly contains a hard gate")
    selected = selection["selected_candidate"]
    raw, indirect, metadata = read_teacher_rows(
        args.train_teacher_manifest, args.train_roi_cost_manifest)
    probes = read_base_probe_training_features(
        args.train_teacher_manifest, metadata)
    reference = direct_targets(indirect)
    base_checkpoint = torch.load(
        args.v1_checkpoint, map_location="cpu", weights_only=False)
    if base_checkpoint.get("format") != "A800 regional utility controller v1":
        raise ValueError("the anchor checkpoint is not controller v1")
    if base_checkpoint["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("v1 anchor feature schema mismatch")
    base_direct = direct_targets(predict_v1(base_checkpoint, raw, "mlp"))
    residual = reference - base_direct
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")

    experts = {}
    if selected["method"] != "v1":
        for expert_index, kind in enumerate(EXPERT_KINDS):
            engineered = engineer_feature_family(raw, probes, metadata, kind)
            correction_input = np.concatenate(
                [engineered, base_direct], axis=1).astype(np.float32)
            expert_args = argparse.Namespace(**vars(args))
            expert_args.correction_seed = (
                args.correction_seed + expert_index * 10000)
            states, normalization, training = fit_full_correction(
                correction_input, residual, expert_args, device)
            experts[kind] = {
                "engineered_feature_names": list(FEATURE_NAMES_BY_KIND[kind]),
                "correction_feature_names": list(
                    FEATURE_NAMES_BY_KIND[kind] + BASE_DIRECT_FEATURE_NAMES),
                "correction_input_dim": correction_input.shape[1],
                "correction_tower_width": 32,
                "correction_ensemble_size": len(states),
                "correction_state_dicts": states,
                **normalization,
                "training": training,
            }

    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "protocol": PROTOCOL_PATH,
        "selected_method": selected["method"],
        "confidence_z": selected["confidence_z"],
        "correction_scale": selected["correction_scale"],
        "raw_feature_names": list(FEATURE_NAMES),
        "base_probe_names": list(BASE_PROBE_NAMES),
        "direct_target_names": list(DIRECT_TARGET_NAMES),
        "experts": experts,
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
        "budget": {
            "enhance_fallback_byte_ratio": LOW_BYTE_RATIO,
            "generate_tile_budget": LOW_GENERATE_TILES,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "base_seed": args.base_seed,
            "correction_seed": args.correction_seed,
            "region_count": len(metadata),
            "sample_count": len(grouped_indices(metadata)),
            "quality_teacher_manifest": str(args.train_teacher_manifest),
            "roi_cost_teacher_manifest": str(args.train_roi_cost_manifest),
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
    checkpoint_path = args.output_dir / "controller_v5.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    expert_parameter_counts = {
        kind: (
            sum(value.numel() for value in expert["correction_state_dicts"][0].values())
            * expert["correction_ensemble_size"]
        )
        for kind, expert in experts.items()
    }
    summary = {
        "experiment": "A800 low-budget v5 full expert training",
        "status": "trained",
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "selected_method": selected["method"],
        "confidence_z": selected["confidence_z"],
        "correction_scale": selected["correction_scale"],
        "expert_parameter_counts_total": expert_parameter_counts,
        "v1_anchor_parameter_count": sum(
            value.numel()
            for value in base_checkpoint["mlp_state_dict"].values()),
        "calibration": checkpoint["calibration"],
        "training": checkpoint["training"],
        "expert_training": {
            kind: expert["training"] for kind, expert in experts.items()
        },
        "selection": checkpoint["selection"],
        "wall_seconds": time.perf_counter() - started,
        "scientific_boundary": checkpoint["scientific_boundary"],
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "summary": str(args.output_dir / "training_summary.json"),
        "selected_method": selected["method"],
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def predict_expert(
    expert: dict,
    correction_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = (
        (correction_features - expert["correction_x_mean"].numpy())
        / expert["correction_x_scale"].numpy()
    ).astype(np.float32)
    y_mean = expert["correction_y_mean"].numpy()
    y_scale = expert["correction_y_scale"].numpy()
    predictions = []
    for state in expert["correction_state_dicts"]:
        model = IndependentDirectMLP(
            expert["correction_input_dim"],
            expert["correction_tower_width"],
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            value = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        predictions.append(value * y_scale + y_mean)
    stacked = np.stack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0)


def route_main(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not conservative controller v5")
    if checkpoint["raw_feature_names"] != list(FEATURE_NAMES):
        raise ValueError("v5 raw feature schema mismatch")
    if checkpoint["direct_target_names"] != list(DIRECT_TARGET_NAMES):
        raise ValueError("v5 direct target schema mismatch")
    selected_method = checkpoint["selected_method"]
    if selected_method not in {"v1", "context-probe-consensus"}:
        raise ValueError(f"unknown selected v5 method: {selected_method}")
    needs_probe = selected_method != "v1"
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
        expert_predictions = {}
        controller_started = time.perf_counter()
        if selected_method == "v1":
            correction = np.zeros_like(base_direct)
            active = np.zeros_like(base_direct, dtype=bool)
            agreement = np.zeros_like(base_direct, dtype=bool)
            stable = np.zeros_like(base_direct, dtype=bool)
        else:
            assert metric is not None
            probe, probe_record = load_route_probe(
                record, args.base_probe_root, metric)
            for kind in EXPERT_KINDS:
                engineered = single_sample_features(raw, probe, kind)
                correction_input = np.concatenate(
                    [engineered, base_direct], axis=1).astype(np.float32)
                mean, std = predict_expert(
                    checkpoint["experts"][kind], correction_input, device)
                expert_predictions[kind] = {
                    "engineered_features": engineered.tolist(),
                    "correction_mean": mean.tolist(),
                    "correction_ensemble_std": std.tolist(),
                }
            context = expert_predictions["context-residual"]
            probe_expert = expert_predictions["probe-residual"]
            correction, active, agreement, stable = conservative_consensus(
                np.asarray(context["correction_mean"], dtype=np.float32),
                np.asarray(
                    context["correction_ensemble_std"], dtype=np.float32),
                np.asarray(probe_expert["correction_mean"], dtype=np.float32),
                np.asarray(
                    probe_expert["correction_ensemble_std"], dtype=np.float32),
                checkpoint["confidence_z"],
            )
        direct_prediction = (
            base_direct + checkpoint["correction_scale"] * correction)
        calibrated = direct_prediction.copy()
        calibrated[:, 0] -= checkpoint["calibration"][
            "absolute_generate_utility_penalty"]
        calibrated[:, 1] -= checkpoint["calibration"][
            "absolute_enhance_gain_penalty"]
        predicted_costs = np.maximum(
            np.rint(calibrated[:, 2]).astype(np.int64), TILE_HEADER.size + 1)
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
            "experiment": "A800 deployable conservative low-budget v5 route",
            "sample": record,
            "route_kind": "v1-anchored-context-probe-consensus-controller",
            "selected_variant": VARIANT_NAME,
            "selected_method": selected_method,
            "selected_feature_kind": "context-probe-consensus",
            "configuration": {
                "tile_size": 128,
                "tile_grid": [4, 4],
                "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
            },
            "raw_feature_names": list(FEATURE_NAMES),
            "base_probe_names": list(BASE_PROBE_NAMES),
            "direct_target_names": list(DIRECT_TARGET_NAMES),
            "encoder_visible_features": raw_records,
            "base_probe_features": probe.tolist() if probe is not None else None,
            "v1_indirect_predictions": base_indirect.tolist(),
            "v1_direct_anchor_predictions": base_direct.tolist(),
            "expert_predictions": expert_predictions,
            "consensus": {
                "confidence_z": checkpoint["confidence_z"],
                "correction_scale": checkpoint["correction_scale"],
                "raw_sign_agreement": agreement.tolist(),
                "uncertainty_stable": stable.tolist(),
                "active": active.tolist(),
                "active_target_count": int(np.count_nonzero(active)),
                "active_target_fraction": float(active.mean()),
                "uncalibrated_residual_correction": correction.tolist(),
            },
            "uncalibrated_direct_predictions": direct_prediction.tolist(),
            "direct_predictions": calibrated.tolist(),
            "calibration": checkpoint["calibration"],
            "base_probe": probe_record,
            "variants": {
                VARIANT_NAME: {
                    "method": "v1-anchored-conservative-consensus-v5",
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
                "context_and_probe_must_agree": selected_method != "v1",
                "context_uses_encoder_visible_regions_only": True,
                "base_probe_uses_encoder_source_and_reconstruction": needs_probe,
                "ground_truth_or_teacher_targets_used_for_route": False,
                "decoder_receives_source_rgb": False,
                "temporary_base_stream_transmitted": False,
                "action_map_must_be_written_to_stream": True,
                "predicted_fallback_bytes_are_final_spatial_bytes": False,
                "dcvc_uf_frozen": True,
                "seedvr2_frozen": True,
                "spatial_qp_codec_frozen": True,
            },
        }
        path = args.output_dir / f"{sample_id}.json"
        atomic_json(path, result)
        entries.append({
            "sample_id": sample_id,
            "path": str(path),
            "selected_variant": VARIANT_NAME,
            "selected_method": selected_method,
            "action_counts": action_counts(actions),
            "active_target_count": int(np.count_nonzero(active)),
            "base_probe": probe_record,
        })
        print(json.dumps({
            "stage": "low-budget-v5-route",
            "sample_id": sample_id,
            "selected_method": selected_method,
            "active_target_count": int(np.count_nonzero(active)),
            "selected_counts": action_counts(actions),
        }, ensure_ascii=False), flush=True)
    manifest = {
        "experiment": "A800 conservative low-budget v5 route manifest",
        "checkpoint": str(args.checkpoint),
        "sample_count": len(entries),
        "selected_method": selected_method,
        "selected_feature_kind": "context-probe-consensus",
        "lpips_model_load_seconds_shared": lpips_model_load_seconds,
        "entries": entries,
        "scientific_boundary": {
            "development_or_evaluation_quality_results_used_for_routing": False,
            "single_gpu": True,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)


def main() -> None:
    args = parse_args()
    if args.mode == "select":
        selection_main(args)
    elif args.mode == "train":
        train_main(args)
    else:
        route_main(args)


if __name__ == "__main__":
    main()
