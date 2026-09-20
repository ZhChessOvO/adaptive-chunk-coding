#!/usr/bin/env python3
"""Frozen-vs-fine-tuned DCVC-UF codec-only evaluation.

The protocol freezes all routes before looking at fine-tuned quality.  It
re-encodes the 37 existing REDS+UVG development/evaluation samples with the
v6 combined action maps under both checkpoints, and rehearses uniform QP
8/16/32 on six fixed clips.  SeedVR2 is intentionally excluded here so that
the first comparison isolates the codec adaptation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_feather_diagnostic import boundary_metrics
from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import panel
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant


ACTION_BY_QP = {8: 1, 16: 0, 32: 2}
ACTION_COLORS = np.asarray(
    ((74, 144, 226), (242, 160, 42), (70, 170, 92)), dtype=np.uint8)
QUALITY_KEYS = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
BOUNDARY_KEYS = (
    "gradient_error_mae", "output_jump_mae", "reference_jump_mae",
    "boundary_band_rgb_mae",
)
UNIFORM_SAMPLE_IDS = (
    "dev-s000-f00-x384-y096",
    "test-s012-f00-x384-y096",
    "eval-s024-f00-x384-y096",
    "uvg-beauty-f00-center512",
    "uvg-readysetgo-f00-center512",
    "uvg-yachtride-f00-center512",
)
VISUAL_SAMPLE_IDS = (
    "dev-s000-f00-x384-y096",
    "uvg-readysetgo-f00-center512",
    "uvg-yachtride-f00-center512",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--sample-manifest", type=Path, required=True)
    prepare.add_argument("--v6-plan", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--frozen-image", type=Path, required=True)
    prepare.add_argument("--frozen-video", type=Path, required=True)
    prepare.add_argument("--tuned-image", type=Path, required=True)
    prepare.add_argument("--tuned-video", type=Path, required=True)

    listing = subparsers.add_parser("list-tasks")
    listing.add_argument("--protocol", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--protocol", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    resource = subparsers.add_parser("resource-snapshot")
    resource.add_argument("--output-dir", type=Path, required=True)
    resource.add_argument("--output", type=Path)

    subparsers.add_parser("self-test")
    return parser.parse_args(argv)


def uniform_route(source: dict, qp: int) -> dict:
    if qp not in ACTION_BY_QP:
        raise ValueError(qp)
    name = f"uniform-qp{qp}"
    action = ACTION_BY_QP[qp]
    return {
        "experiment": "spatial-QP fine-tune uniform regression",
        "sample": source["sample"],
        "route_kind": "fixed-uniform-quality-regression",
        "selected_variant": name,
        "configuration": source["configuration"],
        "variants": {
            name: {
                "method": "uniform action through spatial-QP syntax",
                "actions": [action] * 16,
                "action_counts": {
                    "Base": 16 if action == 0 else 0,
                    "Generate": 16 if action == 1 else 0,
                    "Enhance": 16 if action == 2 else 0,
                },
            },
        },
        "scientific_boundary": {
            "route_fixed_before_finetuned_quality": True,
            "uniform_qp_regression": True,
            "quality_index": qp,
        },
    }


def prepare_main(args: argparse.Namespace) -> None:
    records = [
        json.loads(line) for line in args.sample_manifest.read_text().splitlines()
        if line.strip()
    ]
    plan = read_json(args.v6_plan)
    if len(records) != 37 or len(plan["samples"]) != 37:
        raise RuntimeError("expected the frozen 30 REDS + 7 UVG protocol")
    by_id = {record["sample_id"]: record for record in records}
    plan_by_id = {record["sample_id"]: record for record in plan["samples"]}
    if set(by_id) != set(plan_by_id):
        raise RuntimeError("sample manifest and v6 plan IDs differ")
    if not set(UNIFORM_SAMPLE_IDS).issubset(by_id):
        raise RuntimeError("uniform regression sample is absent")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gates_dir = args.output_dir / "gates"
    routes_dir = args.output_dir / "uniform_routes"
    gates_dir.mkdir(parents=True, exist_ok=True)
    routes_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "frozen": {
            "image": str(args.frozen_image.resolve()),
            "video": str(args.frozen_video.resolve()),
            "checkpoint_role": "frozen-pretrained",
            "image_sha256": sha256_file(args.frozen_image),
            "video_sha256": sha256_file(args.frozen_video),
        },
        "tuned": {
            "image": str(args.tuned_image.resolve()),
            "video": str(args.tuned_video.resolve()),
            "checkpoint_role": "spatial-qp-finetuned",
            "image_sha256": sha256_file(args.tuned_image),
            "video_sha256": sha256_file(args.tuned_video),
        },
    }
    tasks = []
    sample_records = []
    for sample in plan["samples"]:
        sample_id = sample["sample_id"]
        gate_path = gates_dir / f"{sample_id}.json"
        atomic_json(gate_path, by_id[sample_id])
        combined = sample["variants"]["combined"]
        mixed_route = Path(combined["route"])
        if not mixed_route.is_file():
            raise FileNotFoundError(mixed_route)
        sample_records.append({
            "sample_id": sample_id,
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "data_role": sample["data_role"],
            "gate": str(gate_path.resolve()),
            "mixed_route": str(mixed_route.resolve()),
            "mixed_actions": combined["actions"],
        })
        for role, checkpoint in checkpoints.items():
            tasks.append({
                "task_id": f"{sample_id}--mixed--{role}",
                "sample_id": sample_id,
                "dataset": sample["dataset"],
                "data_role": sample["data_role"],
                "family": "mixed-v6-route",
                "qp": None,
                "model_role": role,
                "gate": str(gate_path.resolve()),
                "route": str(mixed_route.resolve()),
                "output_rel": f"evaluation/{sample_id}/mixed/{role}",
                **checkpoint,
            })
        if sample_id in UNIFORM_SAMPLE_IDS:
            source_route = read_json(mixed_route)
            for qp in (8, 16, 32):
                route_path = routes_dir / f"{sample_id}-qp{qp}.json"
                atomic_json(route_path, uniform_route(source_route, qp))
                for role, checkpoint in checkpoints.items():
                    tasks.append({
                        "task_id": f"{sample_id}--uniform-qp{qp}--{role}",
                        "sample_id": sample_id,
                        "dataset": sample["dataset"],
                        "data_role": sample["data_role"],
                        "family": "uniform-regression",
                        "qp": qp,
                        "model_role": role,
                        "gate": str(gate_path.resolve()),
                        "route": str(route_path.resolve()),
                        "output_rel": (
                            f"evaluation/{sample_id}/uniform-qp{qp}/{role}"),
                        **checkpoint,
                    })
    protocol = {
        "experiment": "frozen vs spatial-QP-finetuned DCVC-UF codec-only comparison",
        "status": "frozen-before-finetuned-evaluation",
        "created_utc": utc_now(),
        "sample_manifest": str(args.sample_manifest.resolve()),
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "v6_plan": str(args.v6_plan.resolve()),
        "v6_plan_sha256": sha256_file(args.v6_plan),
        "sample_count": len(sample_records),
        "dataset_counts": {
            "REDS": sum(item["dataset"] == "REDS" for item in sample_records),
            "UVG": sum(item["dataset"] == "UVG" for item in sample_records),
        },
        "uniform_sample_ids": list(UNIFORM_SAMPLE_IDS),
        "mixed_task_count": 37 * 2,
        "uniform_task_count": len(UNIFORM_SAMPLE_IDS) * 3 * 2,
        "total_task_count": len(tasks),
        "checkpoints": checkpoints,
        "samples": sample_records,
        "tasks": tasks,
        "scientific_boundary": {
            "all_samples_previously_used_by_project": True,
            "role": "development and cross-distribution codec candidate comparison",
            "routes_frozen_before_quality": True,
            "same_route_between_models": True,
            "seedvr2_excluded_to_isolate_codec": True,
            "no_hard_acceptance_threshold": True,
            "actual_stream_and_fresh_decode_required": True,
        },
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps({
        "protocol": str((args.output_dir / "protocol.json").resolve()),
        "samples": len(sample_records),
        "tasks": len(tasks),
    }, indent=2))


def list_tasks_main(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    fields = (
        "task_id", "sample_id", "family", "model_role", "gate", "route",
        "image", "video", "checkpoint_role", "output_rel",
    )
    for task in protocol["tasks"]:
        print("\t".join(str(task[field]) for field in fields))


def finite_mean(values: list[float | None]) -> float | None:
    selected = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(selected)) if selected else None


def aggregate(records: list[dict]) -> dict:
    return {
        "sample_count": len(records),
        "stream_bytes_total": sum(record["stream_bytes"] for record in records),
        "stream_bytes_mean": finite_mean([
            record["stream_bytes"] for record in records]),
        "aggregate_bpp": (
            sum(record["stream_bytes"] for record in records) * 8
            / (len(records) * 17 * 512 * 512)),
        "quality_mean": {
            key: finite_mean([record["quality"][key] for record in records])
            for key in QUALITY_KEYS
        },
        "boundary_mean": {
            key: finite_mean([record["boundary"][key] for record in records])
            for key in BOUNDARY_KEYS
        },
        "peak_cuda_allocated_bytes_max": max(
            record["peak_cuda_allocated_bytes"] for record in records),
    }


def paired_comparison(records: list[dict]) -> dict:
    roles = {record["model_role"]: record for record in records}
    frozen, tuned = roles["frozen"], roles["tuned"]
    return {
        "sample_id": frozen["sample_id"],
        "dataset": frozen["dataset"],
        "family": frozen["family"],
        "qp": frozen["qp"],
        "stream_byte_delta_tuned_minus_frozen": (
            tuned["stream_bytes"] - frozen["stream_bytes"]),
        **{
            f"{key}_delta_tuned_minus_frozen": (
                tuned["quality"][key] - frozen["quality"][key])
            for key in QUALITY_KEYS
        },
        **{
            f"boundary_{key}_delta_tuned_minus_frozen": (
                None if tuned["boundary"][key] is None else
                tuned["boundary"][key] - frozen["boundary"][key])
            for key in BOUNDARY_KEYS
        },
    }


def comparison_summary(pairs: list[dict]) -> dict:
    return {
        "pair_count": len(pairs),
        "mean_stream_byte_delta": finite_mean([
            row["stream_byte_delta_tuned_minus_frozen"] for row in pairs]),
        "mean_lpips_delta": finite_mean([
            row["lpips_alex_delta_tuned_minus_frozen"] for row in pairs]),
        "mean_psnr_delta_db": finite_mean([
            row["psnr_db_delta_tuned_minus_frozen"] for row in pairs]),
        "mean_temporal_delta_mae_delta": finite_mean([
            row["temporal_delta_mae_delta_tuned_minus_frozen"] for row in pairs]),
        "lpips_improved_count": sum(
            row["lpips_alex_delta_tuned_minus_frozen"] < 0 for row in pairs),
        "psnr_improved_count": sum(
            row["psnr_db_delta_tuned_minus_frozen"] > 0 for row in pairs),
        "temporal_improved_count": sum(
            row["temporal_delta_mae_delta_tuned_minus_frozen"] < 0
            for row in pairs),
        "mean_boundary_gradient_error_delta": finite_mean([
            row["boundary_gradient_error_mae_delta_tuned_minus_frozen"]
            for row in pairs]),
    }


def bd_rate_percent(
    reference: list[dict],
    candidate: list[dict],
    quality_key: str,
    lower_quality_is_better: bool = False,
) -> float | None:
    """Estimate candidate bitrate change at equal quality from three QP points."""
    def points(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        values = []
        for record in records:
            quality = float(record["quality"][quality_key])
            if lower_quality_is_better:
                quality = -quality
            rate = float(record["stream_bytes"])
            if math.isfinite(quality) and rate > 0:
                values.append((quality, math.log(rate)))
        values.sort()
        deduplicated = {}
        for quality, log_rate in values:
            deduplicated[quality] = log_rate
        return (
            np.asarray(list(deduplicated), dtype=np.float64),
            np.asarray(list(deduplicated.values()), dtype=np.float64),
        )

    ref_quality, ref_log_rate = points(reference)
    can_quality, can_log_rate = points(candidate)
    if len(ref_quality) < 2 or len(can_quality) < 2:
        return None
    lower = max(float(ref_quality.min()), float(can_quality.min()))
    upper = min(float(ref_quality.max()), float(can_quality.max()))
    if not upper > lower:
        return None
    ref_degree = min(2, len(ref_quality) - 1)
    can_degree = min(2, len(can_quality) - 1)
    ref_integral = np.polyint(np.polyfit(ref_quality, ref_log_rate, ref_degree))
    can_integral = np.polyint(np.polyfit(can_quality, can_log_rate, can_degree))
    ref_area = float(np.polyval(ref_integral, upper) - np.polyval(ref_integral, lower))
    can_area = float(np.polyval(can_integral, upper) - np.polyval(can_integral, lower))
    mean_log_rate_delta = (can_area - ref_area) / (upper - lower)
    return float((math.exp(mean_log_rate_delta) - 1.0) * 100.0)


def uniform_bd_rate_summary(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["family"] == "uniform-regression":
            grouped[(record["sample_id"], record["model_role"])].append(record)
    rows = []
    sample_ids = sorted({sample_id for sample_id, _ in grouped})
    for sample_id in sample_ids:
        frozen = grouped.get((sample_id, "frozen"), [])
        tuned = grouped.get((sample_id, "tuned"), [])
        if len(frozen) != 3 or len(tuned) != 3:
            raise RuntimeError(f"uniform QP curve is incomplete: {sample_id}")
        rows.append({
            "sample_id": sample_id,
            "dataset": frozen[0]["dataset"],
            "psnr_bd_rate_percent": bd_rate_percent(
                frozen, tuned, "psnr_db"),
            "lpips_bd_rate_percent": bd_rate_percent(
                frozen, tuned, "lpips_alex", lower_quality_is_better=True),
        })

    def summarize(selected: list[dict]) -> dict:
        output = {"sample_count": len(selected)}
        for key in ("psnr_bd_rate_percent", "lpips_bd_rate_percent"):
            values = [row[key] for row in selected if row[key] is not None]
            output[f"mean_{key}"] = finite_mean(values)
            output[f"median_{key}"] = (
                float(np.median(values)) if values else None)
            output[f"improved_count_{key}"] = sum(value < 0 for value in values)
            output[f"valid_count_{key}"] = len(values)
        return output

    return {
        "definition": (
            "tuned versus frozen bitrate change at equal quality; negative is better; "
            "quadratic log-rate integration over the overlapping three-QP range"),
        "combined": summarize(rows),
        "REDS": summarize([row for row in rows if row["dataset"] == "REDS"]),
        "UVG": summarize([row for row in rows if row["dataset"] == "UVG"]),
        "per_sample": rows,
    }


def write_visual(
    path: Path,
    source: list[np.ndarray],
    frozen: list[np.ndarray],
    tuned: list[np.ndarray],
    frozen_quality: dict,
    tuned_quality: dict,
    actions: list[int],
) -> None:
    frame_index = 8
    action_array = np.asarray(actions, dtype=np.int64).reshape(4, 4)
    action_image = np.repeat(
        np.repeat(ACTION_COLORS[action_array], 128, axis=0), 128, axis=1)
    difference = np.clip(
        np.abs(tuned[frame_index].astype(np.int16)
               - frozen[frame_index].astype(np.int16)) * 8,
        0, 255,
    ).astype(np.uint8)
    panels = [
        panel(source[frame_index], "GT | frame 9", None),
        panel(frozen[frame_index], "Frozen UF | mixed v6 route", frozen_quality),
        panel(tuned[frame_index], "Fine-tuned UF | same route", tuned_quality),
        panel(difference, "|tuned - frozen| x8", None),
        panel(action_image, "Same B/G/E action map", None),
    ]
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new("RGB", (3 * width, 2 * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % 3) * width, (index // 3) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def summarize_main(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    metric = LPIPSAlex(True)
    source_cache: dict[str, list[np.ndarray]] = {}
    decoded_cache: dict[tuple[str, str, str], list[np.ndarray]] = {}
    records = []
    task_by_key = {}
    for task in protocol["tasks"]:
        output = args.output_dir / task["output_rel"]
        encode = read_json(output / "encode_summary.json")
        decode = read_json(output / "decode_summary.json")
        regression = read_json(output / "fresh_decode_regression.json")
        stream = Path(encode["stream"])
        if stream.stat().st_size != encode["stream_bytes"]:
            raise RuntimeError(f"stream byte mismatch: {task['task_id']}")
        if not regression["pixel_exact"]:
            raise RuntimeError(f"fresh decode mismatch: {task['task_id']}")
        if encode["model_checkpoints"]["role"] != task["checkpoint_role"]:
            raise RuntimeError(f"checkpoint role mismatch: {task['task_id']}")
        sample_id = task["sample_id"]
        if sample_id not in source_cache:
            source_cache[sample_id] = load_source(read_json(Path(task["gate"])), 17)
        decoded = load_pngs(output / "fresh_decode")
        if len(decoded) != 17:
            raise RuntimeError(f"decoded frame count mismatch: {task['task_id']}")
        quality = evaluate_variant(source_cache[sample_id], decoded, metric)
        route = read_json(Path(task["route"]))
        variant = route["variants"][route["selected_variant"]]
        actions = variant["actions"]
        boundary = boundary_metrics(
            source_cache[sample_id], decoded,
            np.asarray(actions, dtype=np.int64).reshape(4, 4),
            int(route["configuration"]["tile_size"]),
        )
        record = {
            "task_id": task["task_id"],
            "sample_id": sample_id,
            "dataset": task["dataset"],
            "data_role": task["data_role"],
            "family": task["family"],
            "qp": task["qp"],
            "model_role": task["model_role"],
            "stream": str(stream.resolve()),
            "stream_bytes": encode["stream_bytes"],
            "quality": quality,
            "boundary": boundary,
            "fresh_decode_pixel_exact": True,
            "peak_cuda_allocated_bytes": max(
                encode["peak_cuda_allocated_bytes"],
                decode["peak_cuda_allocated_bytes"]),
            "actions": actions,
        }
        atomic_json(output / "quality_summary.json", record)
        records.append(record)
        key = (sample_id, task["family"], task["qp"], task["model_role"])
        task_by_key[key] = record
        decoded_cache[(sample_id, task["family"], task["model_role"])] = decoded

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        for dataset in ("combined", record["dataset"]):
            grouped[(record["family"], record["qp"], dataset,
                     record["model_role"])].append(record)
    aggregates = {}
    for key, values in grouped.items():
        family, qp, dataset, role = key
        name = family if qp is None else f"{family}-qp{qp}"
        aggregates.setdefault(name, {}).setdefault(dataset, {})[role] = aggregate(values)

    paired_groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        paired_groups[(record["sample_id"], record["family"], record["qp"])].append(record)
    pairs = [paired_comparison(values) for values in paired_groups.values()]
    comparisons = {}
    for family in ("mixed-v6-route", "uniform-regression"):
        family_pairs = [row for row in pairs if row["family"] == family]
        comparisons[family] = {
            "combined": comparison_summary(family_pairs),
            "REDS": comparison_summary([
                row for row in family_pairs if row["dataset"] == "REDS"]),
            "UVG": comparison_summary([
                row for row in family_pairs if row["dataset"] == "UVG"]),
        }
        if family == "uniform-regression":
            comparisons[family]["by_qp"] = {
                str(qp): comparison_summary([
                    row for row in family_pairs if row["qp"] == qp])
                for qp in (8, 16, 32)
            }

    visual_paths = []
    sample_protocol = {item["sample_id"]: item for item in protocol["samples"]}
    for sample_id in VISUAL_SAMPLE_IDS:
        frozen_key = (sample_id, "mixed-v6-route", "frozen")
        tuned_key = (sample_id, "mixed-v6-route", "tuned")
        frozen_record = task_by_key[(sample_id, "mixed-v6-route", None, "frozen")]
        tuned_record = task_by_key[(sample_id, "mixed-v6-route", None, "tuned")]
        target = args.output_dir / "visuals" / f"{sample_id}.png"
        write_visual(
            target, source_cache[sample_id], decoded_cache[frozen_key],
            decoded_cache[tuned_key], frozen_record["quality"],
            tuned_record["quality"], sample_protocol[sample_id]["mixed_actions"])
        visual_paths.append(str(target.resolve()))

    uniform_bd_rate = uniform_bd_rate_summary(records)

    files = [path for path in args.output_dir.rglob("*") if path.is_file()]
    summary = {
        "experiment": protocol["experiment"],
        "status": "complete",
        "completed_utc": utc_now(),
        "protocol": str(args.protocol.resolve()),
        "sample_count": protocol["sample_count"],
        "task_count": len(records),
        "primary_metric": "LPIPS Alex, lower is better",
        "aggregates": aggregates,
        "comparisons_tuned_minus_frozen": comparisons,
        "uniform_bd_rate_tuned_vs_frozen": uniform_bd_rate,
        "records": records,
        "visuals": visual_paths,
        "ordinary_file_count_before_summary_artifacts": len(files),
        "ordinary_file_bytes_before_summary_artifacts": sum(
            path.stat().st_size for path in files),
        "final_resource_snapshot": str(
            (args.output_dir / "final_resource_snapshot.json").resolve()),
        "mounts": {
            name: dict(zip(
                ("total_bytes", "used_bytes", "free_bytes"),
                shutil.disk_usage(name)))
            for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs")
        },
        "scientific_boundary": protocol["scientific_boundary"],
    }
    atomic_json(args.output_dir / "summary.json", summary)

    mixed = comparisons["mixed-v6-route"]
    uniform = comparisons["uniform-regression"]
    markdown = f"""# Spatial-QP codec fine-tune comparison

This is a codec-only development comparison. All routes were frozen before
reading fine-tuned quality; SeedVR2 is excluded.

| Set | Pairs | LPIPS delta | PSNR delta | Byte delta | LPIPS improved |
|---|---:|---:|---:|---:|---:|
| Mixed combined | {mixed['combined']['pair_count']} | {mixed['combined']['mean_lpips_delta']:+.6f} | {mixed['combined']['mean_psnr_delta_db']:+.4f} dB | {mixed['combined']['mean_stream_byte_delta']:+.1f} B | {mixed['combined']['lpips_improved_count']}/{mixed['combined']['pair_count']} |
| Mixed REDS | {mixed['REDS']['pair_count']} | {mixed['REDS']['mean_lpips_delta']:+.6f} | {mixed['REDS']['mean_psnr_delta_db']:+.4f} dB | {mixed['REDS']['mean_stream_byte_delta']:+.1f} B | {mixed['REDS']['lpips_improved_count']}/{mixed['REDS']['pair_count']} |
| Mixed UVG | {mixed['UVG']['pair_count']} | {mixed['UVG']['mean_lpips_delta']:+.6f} | {mixed['UVG']['mean_psnr_delta_db']:+.4f} dB | {mixed['UVG']['mean_stream_byte_delta']:+.1f} B | {mixed['UVG']['lpips_improved_count']}/{mixed['UVG']['pair_count']} |
| Uniform rehearsal | {uniform['combined']['pair_count']} | {uniform['combined']['mean_lpips_delta']:+.6f} | {uniform['combined']['mean_psnr_delta_db']:+.4f} dB | {uniform['combined']['mean_stream_byte_delta']:+.1f} B | {uniform['combined']['lpips_improved_count']}/{uniform['combined']['pair_count']} |

Uniform QP 8/16/32 rate-distortion integration (negative BD-rate is better):

| Set | Samples | PSNR BD-rate | LPIPS BD-rate |
|---|---:|---:|---:|
| Combined | {uniform_bd_rate['combined']['sample_count']} | {uniform_bd_rate['combined']['mean_psnr_bd_rate_percent']:+.2f}% | {uniform_bd_rate['combined']['mean_lpips_bd_rate_percent']:+.2f}% |
| REDS | {uniform_bd_rate['REDS']['sample_count']} | {uniform_bd_rate['REDS']['mean_psnr_bd_rate_percent']:+.2f}% | {uniform_bd_rate['REDS']['mean_lpips_bd_rate_percent']:+.2f}% |
| UVG | {uniform_bd_rate['UVG']['sample_count']} | {uniform_bd_rate['UVG']['mean_psnr_bd_rate_percent']:+.2f}% | {uniform_bd_rate['UVG']['mean_lpips_bd_rate_percent']:+.2f}% |
"""
    atomic_text(args.output_dir / "summary.md", markdown)

    csv_path = args.output_dir / "records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample_id", "dataset", "family", "qp", "model_role",
            "stream_bytes", *QUALITY_KEYS, *BOUNDARY_KEYS,
        ))
        for record in records:
            writer.writerow((
                record["sample_id"], record["dataset"], record["family"],
                record["qp"], record["model_role"], record["stream_bytes"],
                *(record["quality"][key] for key in QUALITY_KEYS),
                *(record["boundary"][key] for key in BOUNDARY_KEYS),
            ))
    print(json.dumps({
        "summary": str((args.output_dir / "summary.json").resolve()),
        "mixed_comparison": mixed,
        "uniform_comparison": uniform,
    }, ensure_ascii=False, indent=2))


def resource_snapshot_main(args: argparse.Namespace) -> None:
    """Write an exact, stable count of regular files after the run is quiet."""
    root = args.output_dir.resolve()
    output = (args.output or root / "final_resource_snapshot.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        files = [
            path for path in root.rglob("*")
            if path.is_file() and path.resolve() != output
        ]
        base_bytes = sum(path.stat().st_size for path in files)
        snapshot = {
            "captured_utc": utc_now(),
            "run_root": str(root),
            "accounting": "sum of st_size for every regular file under run_root",
            "ordinary_file_count_including_snapshot": len(files) + 1,
            "ordinary_file_bytes_including_snapshot": 0,
            "snapshot_file_bytes": 0,
            "mounts": {
                name: dict(zip(
                    ("total_bytes", "used_bytes", "free_bytes"),
                    shutil.disk_usage(name)))
                for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs")
            },
        }
        for _ in range(20):
            payload = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
            payload_bytes = len(payload.encode("utf-8"))
            total_bytes = base_bytes + payload_bytes
            if (snapshot["snapshot_file_bytes"] == payload_bytes
                    and snapshot["ordinary_file_bytes_including_snapshot"]
                    == total_bytes):
                break
            snapshot["snapshot_file_bytes"] = payload_bytes
            snapshot["ordinary_file_bytes_including_snapshot"] = total_bytes
        atomic_text(output, payload)
        final_files = [path for path in root.rglob("*") if path.is_file()]
        final_bytes = sum(path.stat().st_size for path in final_files)
        if (len(final_files) == snapshot["ordinary_file_count_including_snapshot"]
                and final_bytes
                == snapshot["ordinary_file_bytes_including_snapshot"]
                and output.stat().st_size == snapshot["snapshot_file_bytes"]):
            return
    raise RuntimeError("run files changed while taking the final resource snapshot")


def self_test() -> None:
    source = {
        "sample": {"sample_id": "sample"},
        "configuration": {
            "tile_size": 128,
            "tile_grid": [4, 4],
            "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
        },
    }
    for qp, action in ACTION_BY_QP.items():
        route = uniform_route(source, qp)
        variant = route["variants"][route["selected_variant"]]
        assert variant["actions"] == [action] * 16
    values = [
        {"model_role": "frozen", "sample_id": "x", "dataset": "REDS",
         "family": "mixed-v6-route", "qp": None, "stream_bytes": 100,
         "quality": {key: 1.0 for key in QUALITY_KEYS},
         "boundary": {key: 1.0 for key in BOUNDARY_KEYS}},
        {"model_role": "tuned", "sample_id": "x", "dataset": "REDS",
         "family": "mixed-v6-route", "qp": None, "stream_bytes": 95,
         "quality": {**{key: 1.0 for key in QUALITY_KEYS}, "lpips_alex": 0.9},
         "boundary": {key: 0.9 for key in BOUNDARY_KEYS}},
    ]
    pair = paired_comparison(values)
    assert pair["stream_byte_delta_tuned_minus_frozen"] == -5
    assert math.isclose(pair["lpips_alex_delta_tuned_minus_frozen"], -0.1)
    reference_curve = [
        {"stream_bytes": rate, "quality": {"psnr_db": quality}}
        for rate, quality in ((100, 20), (200, 25), (400, 30))
    ]
    candidate_curve = [
        {"stream_bytes": rate * 0.8, "quality": {"psnr_db": quality}}
        for rate, quality in ((100, 20), (200, 25), (400, 30))
    ]
    assert math.isclose(
        bd_rate_percent(reference_curve, candidate_curve, "psnr_db"),
        -20.0, rel_tol=1e-6, abs_tol=1e-6)
    print(json.dumps({
        "status": "passed",
        "uniform_route_profiles": [8, 16, 32],
        "paired_delta_sign": "tuned minus frozen",
        "bd_rate_sign": "negative means tuned needs fewer bytes at equal quality",
        "mixed_sample_count": 37,
        "uniform_sample_count": len(UNIFORM_SAMPLE_IDS),
    }, indent=2))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_main(args)
    elif args.command == "list-tasks":
        list_tasks_main(args)
    elif args.command == "summarize":
        summarize_main(args)
    elif args.command == "resource-snapshot":
        resource_snapshot_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main(sys.argv[1:])
