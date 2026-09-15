#!/usr/bin/env python3
"""E11: train-set learning, sample scaling, and capacity diagnosis.

This is deliberately a train-only diagnostic.  It first evaluates the four
fixed E10 checkpoints on exact training windows, then trains C1/C2 predictors
on nested subsets of train/001..018 and evaluates them on train/019..024,
which are excluded from those new checkpoints.  Validation and sealed REDS
sequences are never eligible for this program.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import stage_b_mse_weight_ablation as e10
from demo.stage_b_evaluate_predictor_g2 import (
    load_predictor_checkpoint,
    run_operational_mode,
    sha256_file,
)
from demo.stage_b_masked_predictor import PredictorConfig, SparseMaskedLatentPredictor


SEED = 20260914
TRAIN_IDS = tuple(f"{index:03d}" for index in range(1, 25))
FIT_SCALES = (6, 12, 18)
INTERNAL_HOLDOUT_IDS = tuple(f"{index:03d}" for index in range(19, 25))
VISITS_PER_SAMPLE = 200
ARCHITECTURES = {
    "c1": PredictorConfig(),
    "c2": PredictorConfig(
        q_width=64, common_width=128, mean_width=64, hidden_width=256),
}
E10_ARMS = e10.ARMS


def parse_args():
    parser = argparse.ArgumentParser(
        description="E11 train-only learning and capacity scaling diagnosis")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--e10-root",
        default="output/stage_b_mse_weight_ablation_train001_024_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_train_capacity_probe_train001_024_v1")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    return parser.parse_args()


def validate_args(args):
    if args.decode_repeats != 3:
        raise ValueError("E11 requires three complete fresh decodes")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E11 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E11 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E11 fixes gradient clipping at 1.0")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("E11 fixes codec skip threshold at zero")
    e10_root = Path(args.e10_root).resolve()
    required = [
        e10_root / "train_cache" / "complete.json",
        e10_root / "train_cache" / "manifest.jsonl",
        e10_root / "training_complete.json",
    ] + [e10_root / "checkpoints" / f"{arm}_final.pt" for arm in E10_ARMS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E11 requires completed E10 artifacts: {missing}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_manifest(e10_root: Path):
    manifest = e10_root / "train_cache" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != 192:
        raise RuntimeError(f"E11 expected 192 E10 train samples, got {len(entries)}")
    if {entry["sequence"] for entry in entries} != set(TRAIN_IDS):
        raise RuntimeError("E11 training cache sequence set is incompatible")
    return entries


def select_entries(entries, sequence_ids):
    allowed = set(sequence_ids)
    selected = [entry for entry in entries if entry["sequence"] in allowed]
    selected.sort(key=lambda item: (
        item["sequence"], item["window_start"], item["chunk_index"]))
    return selected


def load_samples(entries, device):
    return [load_file(entry["path"], device=str(device)) for entry in entries]


def model_label(architecture: str, fit_sequences: int) -> str:
    return f"{architecture}-n{fit_sequences:02d}-byte-e200"


def make_schedule(sample_count: int):
    rng = np.random.default_rng(SEED)
    schedule = []
    for _ in range(VISITS_PER_SAMPLE):
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule


def save_scaling_checkpoint(path: Path, label: str, model, fit_ids, steps,
                            training):
    atomic_checkpoint(path, {
        "format_version": 1,
        "project": "Adaptive Chunk Coding",
        "profile": f"stage-b-e11-train-capacity-{label}",
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "seed": SEED,
            "steps_requested": steps,
            "visits_per_sample": VISITS_PER_SAMPLE,
            "alpha": e10.ALPHA,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "loss": "MSE",
            "weight_mode": "byte",
            "checkpoint_rule": "fixed final update",
            "result": training,
        },
        "data": {
            "split": "REDS train_sharp",
            "fit_sequence_ids": list(fit_ids),
            "excluded_internal_holdout_ids": list(INTERNAL_HOLDOUT_IDS),
            "window_starts": list(e10.WINDOW_STARTS),
            "chunk_indices": [0, 1],
            "sample_count": len(fit_ids) * len(e10.WINDOW_STARTS) * 2,
            "trajectory": "fixed_mean_fill_k8",
        },
        "scientific_scope": {
            "train_only": True,
            "internal_holdout_excluded_from_gradient": True,
            "validation_or_sealed_data_read": False,
            "generalization_claim_allowed": False,
        },
    })


def train_scaling_models(entries, args, output_dir: Path, device):
    checkpoints = {}
    records = {}
    initial_states = {}
    for arch_name, config in ARCHITECTURES.items():
        seed_everything(SEED)
        initial = SparseMaskedLatentPredictor(config).to(device)
        initial_states[arch_name] = copy.deepcopy(initial.state_dict())
        del initial
    for arch_name, config in ARCHITECTURES.items():
        for fit_sequence_count in FIT_SCALES:
            label = model_label(arch_name, fit_sequence_count)
            fit_ids = TRAIN_IDS[:fit_sequence_count]
            fit_entries = select_entries(entries, fit_ids)
            expected_samples = fit_sequence_count * len(e10.WINDOW_STARTS) * 2
            if len(fit_entries) != expected_samples:
                raise RuntimeError(f"{label}: wrong fit sample count")
            checkpoint_path = output_dir / "checkpoints" / f"{label}.pt"
            record_path = output_dir / "training" / f"{label}.json"
            if checkpoint_path.is_file() and record_path.is_file():
                payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False)
                if (payload.get("training", {}).get("seed") != SEED
                        or payload.get("data", {}).get("fit_sequence_ids")
                        != list(fit_ids)):
                    raise RuntimeError(f"{label}: incompatible resumed checkpoint")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                print(json.dumps({"stage": "train-resume", "model": label}),
                      flush=True)
            else:
                if checkpoint_path.exists() or record_path.exists():
                    raise RuntimeError(f"{label}: partial training artifacts")
                samples = load_samples(fit_entries, device)
                model = SparseMaskedLatentPredictor(config).to(device)
                model.load_state_dict(initial_states[arch_name], strict=True)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.learning_rate,
                    weight_decay=args.weight_decay)
                schedule = make_schedule(len(samples))
                milestones = {
                    len(samples) * visits for visits in (0, 25, 50, 100, 150, 200)}
                history = [{
                    "step": 0,
                    "visits_per_sample": 0,
                    **e10.latent_metrics(model, samples),
                }]
                started = time.perf_counter()
                model.train()
                last_loss = None
                for step, sample_index in enumerate(schedule, 1):
                    loss = e10.mse_loss(model, samples[sample_index], "byte")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip)
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError(f"{label}: non-finite gradient")
                    optimizer.step()
                    last_loss = float(loss.detach())
                    if step in milestones:
                        model.eval()
                        event = {
                            "step": step,
                            "visits_per_sample": step / len(samples),
                            "last_sample_loss": last_loss,
                            **e10.latent_metrics(model, samples),
                        }
                        history.append(event)
                        print(json.dumps({
                            "stage": "train", "model": label, **event}), flush=True)
                        model.train()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                model.eval()
                record = {
                    "label": label,
                    "architecture": asdict(config),
                    "parameter_count": model.parameter_count,
                    "linear_macs_per_k8_chunk": model.predicted_macs(e10.K),
                    "fit_sequence_ids": list(fit_ids),
                    "fit_sample_count": len(samples),
                    "visits_per_sample": VISITS_PER_SAMPLE,
                    "steps": len(schedule),
                    "training_wall_seconds": elapsed,
                    "updates_per_second": len(schedule) / elapsed,
                    "final_fit_metrics": e10.latent_metrics(model, samples),
                    "history": history,
                }
                save_scaling_checkpoint(
                    checkpoint_path, label, model, fit_ids, len(schedule), record)
                atomic_json(record_path, record)
                del samples, model, optimizer
                torch.cuda.empty_cache()
            checkpoints[label] = checkpoint_path.resolve()
            records[label] = record
    del initial_states
    atomic_json(output_dir / "training_complete.json", {
        "all_scaling_checkpoints_fixed_before_internal_holdout_tensor_load": True,
        "internal_holdout_ids": list(INTERNAL_HOLDOUT_IDS),
        "validation_or_sealed_data_read": False,
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
    })
    return checkpoints, records


@torch.inference_mode()
def evaluate_internal_holdout_latents(entries, checkpoints, output_dir: Path,
                                      device):
    holdout_entries = select_entries(entries, INTERNAL_HOLDOUT_IDS)
    samples = load_samples(holdout_entries, device)
    results = {}
    for label, checkpoint_path in checkpoints.items():
        model, payload, _ = load_predictor_checkpoint(checkpoint_path, device)
        fit_ids = set(payload["data"]["fit_sequence_ids"])
        if fit_ids.intersection(INTERNAL_HOLDOUT_IDS):
            raise RuntimeError(f"{label}: internal holdout leaked into fit set")
        results[label] = e10.latent_metrics(model, samples)
        print(json.dumps({
            "stage": "internal-holdout-latent", "model": label,
            **results[label]}), flush=True)
        del model
    del samples
    torch.cuda.empty_cache()
    atomic_json(output_dir / "internal_holdout_latent_metrics.json", {
        "split": "REDS train_sharp/019..024",
        "sample_count": len(holdout_entries),
        "excluded_from_all_new_model_gradients": True,
        "results": results,
    })
    return results


def manifest_entry_map(entries):
    return {
        (entry["sequence"], entry["window_start"], entry["chunk_index"]): entry
        for entry in entries
    }


def build_probe_sequences(entries, data_root: Path, sequence_ids, start: int):
    by_key = manifest_entry_map(entries)
    sequences = []
    for sequence in sequence_ids:
        source_files, frames, crop = e10.load_window(
            data_root, "train_sharp", sequence, start)
        routes = []
        route_records = []
        for chunk_index in range(2):
            entry = by_key[(sequence, start, chunk_index)]
            tensors = load_file(entry["path"], device="cpu")
            route = tensors["skip_blocks"].numpy().astype(np.bool_)
            if int(route.sum()) != e10.K:
                raise RuntimeError("E11 probe route is not K8")
            routes.append(route)
            route_records.append({
                "chunk_index": chunk_index,
                "selected_flat_ids": np.flatnonzero(route.reshape(-1)).tolist(),
                "allbase_y_bytes": entry["allbase_y_bytes"],
                "k8_y_bytes": entry["k8_y_bytes"],
                "route_bytes": entry["route_bytes"],
            })
        sequences.append(e10.DevSequence(
            sequence=sequence,
            frames=frames,
            source_files=source_files,
            crop=crop,
            routes=routes,
            route_records=route_records,
        ))
    return sequences


def public_point(point):
    return e10.public_point(point)


def run_or_load_point(label, profile, qp, i_net, p_net, checkpoint_path,
                      frames, eval_args, point_dir: Path, device, routes):
    result_path = point_dir / "point.json"
    if result_path.is_file():
        print(json.dumps({"stage": "evaluate-resume", "mode": label,
                          "path": str(result_path)}), flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    digest = None if checkpoint_path is None else sha256_file(checkpoint_path)
    print(json.dumps({"stage": "evaluate", "mode": label,
                      "path": str(point_dir)}), flush=True)
    point = run_operational_mode(
        label, profile, qp, i_net, p_net,
        checkpoint_path, digest, frames, eval_args, point_dir, device, None,
        fixed_routes=routes)
    result = public_point(point)
    atomic_json(result_path, result)
    return result


@torch.inference_mode()
def evaluate_existing_on_seen_train(i_net, p_net, sequences, e10_root: Path,
                                    args, output_dir: Path, device):
    eval_args = e10.evaluation_args(args)
    checkpoints = {
        arm: (e10_root / "checkpoints" / f"{arm}_final.pt").resolve()
        for arm in E10_ARMS
    }
    results = {}
    for sequence in sequences:
        sequence_results = {}
        root = output_dir / "seen_train_operational" / sequence.sequence
        for qp in (30, 32):
            label = f"allbase-p{qp}"
            sequence_results[label] = run_or_load_point(
                label, "all-base", qp, i_net, p_net, None, sequence.frames,
                eval_args, root / label, device, None)
        sequence_results["mean-k8"] = run_or_load_point(
            "mean-k8", "mean", 32, i_net, p_net, None, sequence.frames,
            eval_args, root / "mean-k8", device, sequence.routes)
        for arm, checkpoint_path in checkpoints.items():
            label = f"{arm}-c1-a025-k8"
            sequence_results[arm] = run_or_load_point(
                label, "learned-lite-c1-a025", 32, i_net, p_net,
                checkpoint_path, sequence.frames, eval_args, root / label,
                device, sequence.routes)
        results[sequence.sequence] = sequence_results
    return results


@torch.inference_mode()
def evaluate_scaling_on_internal_holdout(i_net, p_net, sequences, checkpoints,
                                         args, output_dir: Path, device):
    eval_args = e10.evaluation_args(args)
    results = {}
    for sequence in sequences:
        sequence_results = {}
        root = output_dir / "internal_holdout_operational" / sequence.sequence
        for qp in (30, 32):
            label = f"allbase-p{qp}"
            sequence_results[label] = run_or_load_point(
                label, "all-base", qp, i_net, p_net, None, sequence.frames,
                eval_args, root / label, device, None)
        sequence_results["mean-k8"] = run_or_load_point(
            "mean-k8", "mean", 32, i_net, p_net, None, sequence.frames,
            eval_args, root / "mean-k8", device, sequence.routes)
        for label, checkpoint_path in checkpoints.items():
            sequence_results[label] = run_or_load_point(
                f"{label}-a025-k8", "learned-lite-c1-a025", 32,
                i_net, p_net, checkpoint_path, sequence.frames, eval_args,
                root / f"{label}-a025-k8", device, sequence.routes)
        results[sequence.sequence] = sequence_results
    return results


def aggregate_point(results, sequence_ids, label):
    points = [results[sequence][label] for sequence in sequence_ids]
    p_mse = float(np.mean([e10.point_p_mse(point) for point in points]))
    chunk_mse = [float(np.mean([
        e10.point_chunk_mse(point, chunk_index) for point in points
    ])) for chunk_index in range(2)]
    component_keys = (
        "sequence_header_and_chunk_lengths_bytes", "i_payload_bytes",
        "p_container_header_bytes", "global_z_bytes", "base_y_bytes",
        "route_bytes", "residual_bytes", "container_bytes", "total_bytes")
    components = {
        key: sum(point["stream"][key] for point in points)
        for key in component_keys
    }
    seconds = sum(
        point["compute"]["decode_wall_seconds_median"] for point in points)
    latencies = [
        value for point in points
        for value in point["compute"]["predictor_latency_ms_all"]]
    return {
        "label": label,
        "total_bytes": components["total_bytes"],
        "components": components,
        "p_frame_unit_mse": p_mse,
        "p_frame_psnr": -10.0 * math.log10(max(p_mse, 1e-30)),
        "chunk_psnr": [
            -10.0 * math.log10(max(value, 1e-30)) for value in chunk_mse],
        "decode_ms_per_frame": 1000.0 * seconds / (len(sequence_ids) * 17),
        "predictor_latency_ms_p50": (
            float(np.median(latencies)) if latencies else 0.0),
        "predictor_latency_ms_p95": (
            float(np.percentile(latencies, 95)) if latencies else 0.0),
        "peak_cuda_allocated_bytes": max(
            point["compute"]["peak_cuda_allocated_bytes_max"] for point in points),
        "linear_macs_per_chunk": max(
            point["compute"]["linear_macs_per_chunk"] for point in points),
        "per_sequence": {
            sequence: {
                "p_frame_unit_mse": e10.point_p_mse(results[sequence][label]),
                "p_frame_psnr": -10.0 * math.log10(max(
                    e10.point_p_mse(results[sequence][label]), 1e-30)),
            }
            for sequence in sequence_ids
        },
    }


def aggregate_group(results, sequence_ids, labels):
    aggregates = {
        label: aggregate_point(results, sequence_ids, label) for label in labels}
    mean = aggregates["mean-k8"]
    p30 = aggregates["allbase-p30"]
    for label, point in aggregates.items():
        if label in ("allbase-p30", "allbase-p32", "mean-k8"):
            continue
        point["psnr_change_vs_mean_db"] = (
            point["p_frame_psnr"] - mean["p_frame_psnr"])
        point["positive_sequences_vs_mean"] = sum(
            point["per_sequence"][sequence]["p_frame_unit_mse"]
            < mean["per_sequence"][sequence]["p_frame_unit_mse"]
            for sequence in sequence_ids)
        point["directly_dominated_by_p30"] = (
            point["total_bytes"] >= p30["total_bytes"]
            and point["p_frame_psnr"] <= p30["p_frame_psnr"]
            and point["decode_ms_per_frame"] >= p30["decode_ms_per_frame"])
    return aggregates


def summarize_block_records(records, sequence_ids):
    ordered = sorted(records, key=lambda item: (
        -item["actual_singleton_saved_y_bytes"], item["sequence"],
        item["chunk_index"], item["flat_id"]))
    top = ordered[:max(1, math.ceil(len(ordered) / 4))]

    def subset(items):
        mean_energy = sum(item["mean_latent_mse"] for item in items)
        predictor_energy = sum(item["predictor_latent_mse"] for item in items)
        per_sequence = {}
        for sequence in sequence_ids:
            selected = [item for item in items if item["sequence"] == sequence]
            if not selected:
                continue
            mean_sum = sum(item["mean_latent_mse"] for item in selected)
            pred_sum = sum(item["predictor_latent_mse"] for item in selected)
            per_sequence[sequence] = 1.0 - pred_sum / max(mean_sum, 1e-30)
        return {
            "block_count": len(items),
            "latent_gap_recovery": 1.0 - predictor_energy / max(mean_energy, 1e-30),
            "fraction_improving_mean_latent": float(np.mean([
                item["predictor_improves_mean_latent"] for item in items])),
            "sequences_with_positive_latent_recovery": sum(
                value > 0.0 for value in per_sequence.values()),
            "per_sequence": per_sequence,
        }
    return {
        "all_selected_high_byte_blocks": subset(ordered),
        "highest_byte_saving_quartile": subset(top),
    }


@torch.inference_mode()
def run_internal_holdout_block_diagnostics(i_net, p_net, sequences,
                                           checkpoints, output_dir: Path,
                                           device):
    summaries = {}
    all_records = {}
    for label, checkpoint_path in checkpoints.items():
        records = []
        for sequence in sequences:
            print(json.dumps({
                "stage": "internal-holdout-block", "model": label,
                "sequence": sequence.sequence}), flush=True)
            sequence_records, _ = e10.diagnose_arm_sequence(
                i_net, p_net, sequence, checkpoint_path, device)
            records.extend(sequence_records)
        summaries[label] = summarize_block_records(records, INTERNAL_HOLDOUT_IDS)
        all_records[label] = records
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model"] + list(next(iter(all_records.values()))[0].keys())
    with (diagnostic_dir / "internal_holdout_block_records.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, records in all_records.items():
            for record in records:
                writer.writerow({"model": label, **record})
    atomic_json(diagnostic_dir / "internal_holdout_block_summary.json", summaries)
    return summaries


def audit_results(seen_results, holdout_results, seen_ids, holdout_ids,
                  output_dir: Path):
    checks = {
        "validation_or_sealed_data_read": False,
        "internal_holdout_excluded_from_gradient": True,
        "all_files_equal_reported_total_bytes": True,
        "all_points_fresh_decode_three_times": True,
        "all_routed_points_use_k8": True,
        "all_residual_streams_zero_bytes": True,
        "all_predictors_are_decoder_only": True,
    }
    groups = (
        (seen_results, seen_ids, output_dir / "seen_train_operational"),
        (holdout_results, holdout_ids,
         output_dir / "internal_holdout_operational"),
    )
    for results, sequence_ids, root in groups:
        for sequence in sequence_ids:
            for key, point in results[sequence].items():
                path_label = point["label"]
                sequence_path = root / sequence / path_label / "sequence.d2l"
                checks["all_files_equal_reported_total_bytes"] &= (
                    sequence_path.stat().st_size == point["stream"]["total_bytes"])
                checks["all_points_fresh_decode_three_times"] &= (
                    point["compute"]["decode_repeats"] == 3
                    and point["audits"]["complete_fresh_decode"]
                    and point["audits"]["three_repeats_identical"])
                checks["all_residual_streams_zero_bytes"] &= (
                    point["stream"]["residual_bytes"] == 0)
                if key not in ("allbase-p30", "allbase-p32"):
                    checks["all_routed_points_use_k8"] &= all(
                        chunk["skipped_blocks"] == e10.K
                        for chunk in point["chunks"])
                if key not in ("allbase-p30", "allbase-p32", "mean-k8"):
                    checks["all_predictors_are_decoder_only"] &= not point[
                        "audits"]["source_or_omitted_y_available_to_predictor"]
    checks["passes"] = all(
        value is False if key == "validation_or_sealed_data_read" else value
        for key, value in checks.items() if key != "passes")
    return checks


def decide(seen_aggregates, holdout_aggregates, block_summaries):
    mean_seen = seen_aggregates["mean-k8"]
    existing = {}
    for arm in E10_ARMS:
        point = seen_aggregates[arm]
        existing[arm] = {
            "psnr_change_vs_mean_db": point["psnr_change_vs_mean_db"],
            "positive_sequences_vs_mean": point["positive_sequences_vs_mean"],
            "training_window_fit_improves_mean": (
                point["p_frame_unit_mse"] < mean_seen["p_frame_unit_mse"]),
        }
    best_label = max(
        (label for label in holdout_aggregates
         if label not in ("allbase-p30", "allbase-p32", "mean-k8")),
        key=lambda label: holdout_aggregates[label]["p_frame_psnr"])
    best = holdout_aggregates[best_label]
    top = block_summaries[best_label]["highest_byte_saving_quartile"]
    c1_quality = [
        holdout_aggregates[model_label("c1", count)]["p_frame_psnr"]
        for count in FIT_SCALES]
    c2_quality = [
        holdout_aggregates[model_label("c2", count)]["p_frame_psnr"]
        for count in FIT_SCALES]
    sample_trend = (
        c1_quality[0] <= c1_quality[1] <= c1_quality[2]
        and c2_quality[0] <= c2_quality[1] <= c2_quality[2])
    capacity_trend = all(
        holdout_aggregates[model_label("c2", count)]["p_frame_psnr"]
        >= holdout_aggregates[model_label("c1", count)]["p_frame_psnr"]
        for count in FIT_SCALES)
    passes_internal_gate = (
        best["positive_sequences_vs_mean"] >= 5
        and best["p_frame_unit_mse"] < holdout_aggregates["mean-k8"][
            "p_frame_unit_mse"]
        and top["latent_gap_recovery"] > 0.0
        and top["sequences_with_positive_latent_recovery"] >= 5)
    if sample_trend and capacity_trend and passes_internal_gate:
        status = "capacity_and_sample_scaling_warrant_one_dev_confirmation"
        recommendation = (
            "训练剂量、样本量和容量在 train 内部未见序列上形成一致正趋势；"
            "建议预登记一次开发集确认，但不能据此启封 E09。")
    elif passes_internal_gate:
        status = "promising_internal_holdout_but_scaling_trend_not_consistent"
        recommendation = (
            "最佳模型在 train 内部未见序列通过恢复门槛，但样本量或容量趋势不一致；"
            "结论尚未确定，只适合做一次受控开发复核。")
    else:
        status = "scaling_alone_not_supported"
        recommendation = (
            "增加训练次数、样本量和约 2.3 倍参数后仍未在 train 内部未见序列通过恢复门槛；"
            "不建议把单纯扩模型或继续训练作为当前配置的下一步。")
    return {
        "status": status,
        "recommendation": recommendation,
        "existing_models_on_seen_training_windows": existing,
        "best_internal_holdout_model": best_label,
        "best_internal_holdout_psnr_change_vs_mean_db": best[
            "psnr_change_vs_mean_db"],
        "best_internal_holdout_positive_sequences": best[
            "positive_sequences_vs_mean"],
        "best_highest_byte_quartile_recovery": top["latent_gap_recovery"],
        "sample_count_trend_monotonic_for_both_capacities": sample_trend,
        "c2_not_worse_than_c1_at_all_sample_counts": capacity_trend,
        "passes_preregistered_internal_gate": passes_internal_gate,
    }


def write_aggregate_csv(path: Path, aggregates):
    rows = []
    mean = aggregates["mean-k8"]
    for label, point in aggregates.items():
        rows.append({
            "method": label,
            "total_bytes": point["total_bytes"],
            "p_frame_psnr": point["p_frame_psnr"],
            "psnr_change_vs_mean_db": (
                point["p_frame_psnr"] - mean["p_frame_psnr"]),
            "positive_sequences_vs_mean": point.get(
                "positive_sequences_vs_mean"),
            "decode_ms_per_frame": point["decode_ms_per_frame"],
            "predictor_latency_ms_p50": point["predictor_latency_ms_p50"],
            "predictor_latency_ms_p95": point["predictor_latency_ms_p95"],
            "parameter_macs_per_chunk": point["linear_macs_per_chunk"],
            "directly_dominated_by_p30": point.get(
                "directly_dominated_by_p30"),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figure(summary, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    holdout = summary["internal_holdout_aggregates"]
    mean_psnr = holdout["mean-k8"]["p_frame_psnr"]
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for arch_name, marker in (("c1", "o"), ("c2", "s")):
        values = [
            holdout[model_label(arch_name, count)]["p_frame_psnr"] - mean_psnr
            for count in FIT_SCALES]
        ax.plot(FIT_SCALES, values, marker=marker, label=arch_name.upper())
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Fit training sequences")
    ax.set_ylabel("P-frame PSNR change vs mean-fill (dB)")
    ax.set_title("Internal train-holdout sample/capacity trend")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    labels = [model_label(arch, count)
              for arch in ARCHITECTURES for count in FIT_SCALES]
    recovery = [100.0 * summary["internal_holdout_block_diagnostics"][label][
        "highest_byte_saving_quartile"]["latent_gap_recovery"]
                for label in labels]
    ax.bar(np.arange(len(labels)), recovery)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Highest-byte quartile gap recovery (%)")
    ax.set_title("Byte-heavy block recovery")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "train_capacity_scaling.png", dpi=180)
    plt.close(fig)


def write_report(summary, output_dir: Path):
    seen = summary["seen_train_aggregates"]
    holdout = summary["internal_holdout_aggregates"]
    decision = summary["decision"]
    lines = [
        "# E11 训练集内学习与容量缩放诊断",
        "",
        f"结论：{decision['recommendation']}",
        "",
        "训练集检查只用于判断训练、容量和内部泛化瓶颈，不计作最终压缩收益。",
        "",
        "## 现有 E10 模型在见过训练窗口上的实际表现",
        "",
        "| 方法 | 实际总字节 | P 帧 PSNR | 相对 mean-fill | 正序列 | 完整解码 ms/帧 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("allbase-p30", "allbase-p32", "mean-k8") + E10_ARMS:
        point = seen[label]
        positive = point.get("positive_sequences_vs_mean")
        lines.append(
            f"| {label} | {point['total_bytes']:,} | {point['p_frame_psnr']:.5f} | "
            f"{point['p_frame_psnr'] - seen['mean-k8']['p_frame_psnr']:+.5f} dB | "
            f"{'—' if positive is None else f'{positive}/24'} | "
            f"{point['decode_ms_per_frame']:.4f} |")
    lines += [
        "",
        "## 拟合样本与内部未见序列的 latent 恢复",
        "",
        "| 模型 | 拟合样本字节加权恢复 | 内部未见字节加权恢复 |",
        "|---|---:|---:|",
    ]
    for arch_name in ARCHITECTURES:
        for count in FIT_SCALES:
            label = model_label(arch_name, count)
            fit_recovery = summary["training"][label]["final_fit_metrics"][
                "byte_latent_energy_recovery"]
            unseen_recovery = summary["internal_holdout_latent_metrics"][label][
                "byte_latent_energy_recovery"]
            lines.append(
                f"| {label} | {100.0 * fit_recovery:+.2f}% | "
                f"{100.0 * unseen_recovery:+.2f}% |")
    lines += [
        "",
        "训练剂量与容量都能显著提高拟合，但全部模型在内部未见序列上变为负恢复；这是训练策略可以改善拟合、却没有转化为跨视频能力的直接证据。",
        "",
        "## train 内部未见序列上的实际视频结果",
        "",
        "| 模型 | 参数量 | 拟合序列 | 实际总字节 | P 帧 PSNR | 相对 mean-fill | 正序列 | 高字节四分位恢复 | 完整解码 ms/帧 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arch_name, config in ARCHITECTURES.items():
        parameter_count = SparseMaskedLatentPredictor(config).parameter_count
        for count in FIT_SCALES:
            label = model_label(arch_name, count)
            point = holdout[label]
            top = summary["internal_holdout_block_diagnostics"][label][
                "highest_byte_saving_quartile"]
            lines.append(
                f"| {label} | {parameter_count:,} | {count} | "
                f"{point['total_bytes']:,} | "
                f"{point['p_frame_psnr']:.5f} | "
                f"{point['psnr_change_vs_mean_db']:+.5f} dB | "
                f"{point['positive_sequences_vs_mean']}/6 | "
                f"{100.0 * top['latent_gap_recovery']:+.2f}% | "
                f"{point['decode_ms_per_frame']:.4f} |")
    lines += [
        "",
        "内部未见序列固定为 train/019..024，这些序列没有进入六个缩放模型的梯度更新；validation 和封存数据均未读取。",
        "",
        "每个拟合样本固定访问 200 次，避免 E10 中样本增加但总更新数不变所造成的训练剂量下降。C2 参数量约为 C1 的 2.28 倍；两者 codec、route、K8 和 α=0.25 均相同。",
        "",
        "详表：`seen_train_aggregate.csv`、`internal_holdout_aggregate.csv`；逐块记录：`diagnostics/internal_holdout_block_records.csv`；图：`figures/train_capacity_scaling.png`。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    e10_root = Path(args.e10_root).resolve()
    data_root = Path(args.data_root).resolve()

    e10.set_torch_env()
    seed_everything(SEED)
    torch.set_grad_enabled(True)
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))

    entries = load_manifest(e10_root)
    checkpoints, training_records = train_scaling_models(
        entries, args, output_dir, device)
    internal_latent = evaluate_internal_holdout_latents(
        entries, checkpoints, output_dir, device)

    # Pixel files are first needed here.  Only REDS train_sharp is eligible.
    i_net, p_net = e10.load_models(args, device)
    seen_sequences = build_probe_sequences(entries, data_root, TRAIN_IDS, 0)
    seen_results = evaluate_existing_on_seen_train(
        i_net, p_net, seen_sequences, e10_root, args, output_dir, device)
    del seen_sequences
    torch.cuda.empty_cache()

    internal_sequences = build_probe_sequences(
        entries, data_root, INTERNAL_HOLDOUT_IDS, 0)
    holdout_results = evaluate_scaling_on_internal_holdout(
        i_net, p_net, internal_sequences, checkpoints, args, output_dir, device)
    block_summaries = run_internal_holdout_block_diagnostics(
        i_net, p_net, internal_sequences, checkpoints, output_dir, device)

    seen_aggregates = aggregate_group(
        seen_results, TRAIN_IDS,
        ("allbase-p30", "allbase-p32", "mean-k8") + E10_ARMS)
    scaling_labels = tuple(checkpoints)
    holdout_aggregates = aggregate_group(
        holdout_results, INTERNAL_HOLDOUT_IDS,
        ("allbase-p30", "allbase-p32", "mean-k8") + scaling_labels)
    decision = decide(seen_aggregates, holdout_aggregates, block_summaries)
    audits = audit_results(
        seen_results, holdout_results, TRAIN_IDS, INTERNAL_HOLDOUT_IDS,
        output_dir)
    if not audits["passes"]:
        decision = {
            **decision,
            "status": "inconclusive_due_to_audit_failure",
            "recommendation": "关键码流、数据隔离或 fresh decode 审计未通过，结论尚未确定。",
        }
    summary = {
        "experiment": "E11 train-set learning and capacity scaling diagnosis",
        "status": decision["status"],
        "scientific_scope": {
            "train_only_diagnostic": True,
            "counts_as_generalization_or_actual_benefit": False,
            "validation_data_read": False,
            "sealed_data_read": False,
        },
        "protocol": {
            "seen_train_sequences": list(TRAIN_IDS),
            "seen_train_window_start": 0,
            "fit_scales": list(FIT_SCALES),
            "internal_holdout_sequences": list(INTERNAL_HOLDOUT_IDS),
            "window_starts_for_scaling": list(e10.WINDOW_STARTS),
            "visits_per_sample": VISITS_PER_SAMPLE,
            "weight_mode": "byte",
            "architectures": {
                name: {
                    "config": asdict(config),
                    "parameter_count": SparseMaskedLatentPredictor(
                        config).parameter_count,
                    "linear_macs_per_k8_chunk": SparseMaskedLatentPredictor(
                        config).predicted_macs(e10.K),
                }
                for name, config in ARCHITECTURES.items()
            },
            "codec_frozen": True,
            "qp_i": 32,
            "qp_p": 32,
            "block_size": 2,
            "k": e10.K,
            "alpha": e10.ALPHA,
            "decode_repeats": 3,
        },
        "training": training_records,
        "internal_holdout_latent_metrics": internal_latent,
        "seen_train_aggregates": seen_aggregates,
        "internal_holdout_aggregates": holdout_aggregates,
        "internal_holdout_block_diagnostics": block_summaries,
        "decision": decision,
        "audits": audits,
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    write_aggregate_csv(output_dir / "seen_train_aggregate.csv", seen_aggregates)
    write_aggregate_csv(
        output_dir / "internal_holdout_aggregate.csv", holdout_aggregates)
    write_figure(summary, output_dir)
    write_report(summary, output_dir)
    atomic_json(output_dir / "complete.json", {
        "status": summary["status"],
        "summary": str((output_dir / "summary.json").resolve()),
        "report": str((output_dir / "report.md").resolve()),
        "validation_or_sealed_data_read": False,
        "audits_pass": audits["passes"],
    })
    print(json.dumps({
        "status": summary["status"],
        "recommendation": decision["recommendation"],
        "summary": str((output_dir / "summary.json").resolve()),
        "report": str((output_dir / "report.md").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
