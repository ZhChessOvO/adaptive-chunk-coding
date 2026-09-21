#!/usr/bin/env python3
"""Build and summarize a real multi-budget curve for the selected LoRA route.

The selected balanced system keeps the frozen v6 controller and spatial
objective, uses SeedVR2 LoRA at strength 0.50, and changes only the Enhance
fallback byte budget known before encoding.  Exact action-map matches may be
reused; every new map must produce a real spatial-QP stream and fresh decode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from demo.stage_c_a800_feather_diagnostic import boundary_metrics
from demo.stage_c_a800_spatial_consistency import (
    action_counts,
    atomic_json,
    generate_boundary_edges,
    generate_components,
    reroute_record,
)
from demo.stage_c_evaluate_seedvr2_gate import load_source
from demo.stage_c_evaluate_spatial_quality_codec import panel
from demo.stage_c_seedvr2_lora_router_evaluation import (
    ACTION_COLORS,
    BOUNDARY_NAMES,
    QUALITY_NAMES,
    UVG_ADAPTATION_SEQUENCES,
    UVG_HOLDOUT_SEQUENCES,
    atomic_text,
    mean,
    outside_generate_unchanged,
    read,
    read_jsonl,
    route_manifest,
    selected_actions,
    sha256_file,
    validate_frames,
)


DEFAULT_RATIOS = (0.0, 0.25, 0.5, 1.0)
SCALAR_VARIANTS = ("scalar-qp8", "scalar-qp16", "scalar-qp24", "scalar-qp32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--sample-manifest", type=Path, required=True)
    plan.add_argument("--base-routes", type=Path, required=True)
    plan.add_argument("--base-evaluation-summary", type=Path, required=True)
    plan.add_argument("--joint-summary", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--lora-strength", type=float, default=0.5)
    plan.add_argument("--route-output-root", type=Path, required=True)
    plan.add_argument("--enhance-budget-ratios", type=float, nargs="+",
                      default=DEFAULT_RATIOS)
    plan.add_argument("--generate-tile-budget", type=int, default=4)
    plan.add_argument("--spatial-lambda", type=float, default=0.004)
    plan.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--joint-formal-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.mode == "plan":
        if (
            not math.isfinite(args.lora_strength)
            or args.lora_strength < 0
            or not math.isfinite(args.spatial_lambda)
            or args.spatial_lambda < 0
        ):
            parser.error("LoRA strength and spatial lambda must be finite and nonnegative")
        if (
            len(set(args.enhance_budget_ratios))
            != len(args.enhance_budget_ratios)
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in args.enhance_budget_ratios)
        ):
            parser.error("Enhance budget ratios must be unique values in [0, 1]")
        if not 0 <= args.generate_tile_budget <= 16:
            parser.error("Generate tile budget must be between zero and 16")
        if not any(math.isclose(value, 0.25, abs_tol=1e-12)
                   for value in args.enhance_budget_ratios):
            parser.error("the 0.25 regression/reuse point is required")
    return args


def rate_key(ratio: float) -> str:
    return f"enhance-{int(round(ratio * 100)):03d}"


def action_key(actions: list[int]) -> str:
    return "".join(map(str, actions))


def write_route_manifest(
    base_manifest: Path,
    base_routes: dict[str, Path],
    samples: list[dict],
    output_dir: Path,
    ratio: float,
    generate_budget: int,
    spatial_lambda: float,
) -> dict:
    entries = []
    comparisons = []
    for sample in samples:
        sample_id = sample["sample_id"]
        route = read(base_routes[sample_id])
        output, comparison = reroute_record(
            route, spatial_lambda, ratio, generate_budget)
        route_path = output_dir / f"{sample_id}.json"
        atomic_json(route_path, output)
        actions = comparison["candidate_actions"]
        entries.append({
            "sample_id": sample_id,
            "path": str(route_path.resolve()),
            "selected_variant": output["selected_variant"],
            "action_counts": action_counts(actions),
            "changed_tile_count": comparison["changed_tile_count"],
            "generate_boundary_edges": comparison["generate_boundary_edges"],
            "generate_component_count": comparison["generate_component_count"],
        })
        comparisons.append(comparison)
    manifest = {
        "experiment": "SeedVR2 LoRA selected-route Enhance-budget curve",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(base_manifest),
        "input_manifest_sha256": sha256_file(base_manifest),
        "enhance_budget_ratio": ratio,
        "generate_tile_budget": generate_budget,
        "spatial_lambda": spatial_lambda,
        "sample_count": len(entries),
        "entries": entries,
        "comparisons": comparisons,
        "scientific_boundary": {
            "controller_predictions_unchanged": True,
            "only_pre_encoding_budget_changes": True,
            "ground_truth_metrics_used_for_routes": False,
            "single_gpu_required_for_route_solver": False,
            "dcvc_uf_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def make_plan(args: argparse.Namespace) -> None:
    sample_manifest = args.sample_manifest.resolve()
    base_manifest = args.base_routes.resolve()
    base_summary_path = args.base_evaluation_summary.resolve()
    joint_summary_path = args.joint_summary.resolve()
    adapter = args.lora_checkpoint.resolve()
    route_root = args.route_output_root.resolve()
    samples = read_jsonl(sample_manifest)
    base_routes = route_manifest(base_manifest)
    base_summary = read(base_summary_path)
    joint_summary = read(joint_summary_path)
    if (
        len(samples) != 37
        or base_summary.get("sample_count") != 37
        or joint_summary.get("sample_count") != 37
        or not adapter.is_file()
    ):
        raise RuntimeError("expected the fixed 37-sample inputs and adapter")
    sample_ids = [sample["sample_id"] for sample in samples]
    base_by_id = {sample["sample_id"]: sample
                  for sample in base_summary["samples"]}
    if (
        len(set(sample_ids)) != 37
        or set(sample_ids) != set(base_routes)
        or set(sample_ids) != set(base_by_id)
    ):
        raise RuntimeError("sample, base route, and base evaluation IDs differ")

    ratios = sorted(float(value) for value in args.enhance_budget_ratios)
    point_manifests = {}
    route_maps = {}
    rate_points = []
    for ratio in ratios:
        key = rate_key(ratio)
        output_dir = route_root / key
        manifest = write_route_manifest(
            base_manifest, base_routes, samples, output_dir, ratio,
            args.generate_tile_budget, args.spatial_lambda)
        point_path = output_dir / "manifest.json"
        point_manifests[key] = point_path
        route_maps[key] = route_manifest(point_path)
        counts = Counter()
        for entry in manifest["entries"]:
            counts.update({
                0: entry["action_counts"]["Base"],
                1: entry["action_counts"]["Generate"],
                2: entry["action_counts"]["Enhance"],
            })
        rate_points.append({
            "key": key,
            "enhance_budget_ratio": ratio,
            "generate_tile_budget": args.generate_tile_budget,
            "spatial_lambda": args.spatial_lambda,
            "route_manifest": str(point_path),
            "route_manifest_sha256": sha256_file(point_path),
            "action_counts_total": {
                "Base": counts[0], "Generate": counts[1],
                "Enhance": counts[2],
            },
            "generate_boundary_edges_total": sum(
                entry["generate_boundary_edges"]
                for entry in manifest["entries"]),
            "generate_components_total": sum(
                entry["generate_component_count"]
                for entry in manifest["entries"]),
        })

    records = []
    tasks = []
    low_regression_count = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        base_route, base_actions = selected_actions(base_routes[sample_id])
        del base_route
        base_record = base_by_id[sample_id]["variants"]["old-route-lora050"]
        if list(map(int, base_record["actions"])) != base_actions:
            raise RuntimeError(f"base route and LoRA result differ: {sample_id}")
        stream = Path(base_record["stream"]).resolve()
        output_frames = Path(base_record["output_frames"]).resolve()
        if (
            not stream.is_file()
            or stream.stat().st_size != int(base_record["actual_on_disk_bytes"])
            or not output_frames.is_dir()
        ):
            raise RuntimeError(f"base LoRA result is incomplete: {sample_id}")
        known = {action_key(base_actions): {
            "kind": "prior-base",
            "rate_key": None,
        }}
        points = {}
        for point in rate_points:
            key = point["key"]
            route_path = route_maps[key][sample_id]
            _, actions = selected_actions(route_path)
            if math.isclose(point["enhance_budget_ratio"], 0.25, abs_tol=1e-12):
                if actions != base_actions:
                    raise RuntimeError(
                        f"0.25 budget point did not reproduce v6: {sample_id}")
                low_regression_count += 1
            key_actions = action_key(actions)
            reuse = known.get(key_actions)
            if reuse is None:
                reuse = None
                known[key_actions] = {"kind": "rate-point", "rate_key": key}
                tasks.append({
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "rate_key": key,
                    "route": str(route_path),
                    "seed": int(sample["seed"]),
                    "has_generate": 1 in actions,
                })
            points[key] = {
                "route": str(route_path),
                "actions": actions,
                "action_key": key_actions,
                "has_generate": 1 in actions,
                "reuse": copy.deepcopy(reuse),
            }
        records.append({
            "sample_id": sample_id,
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "data_role": sample["data_role"],
            "seed": int(sample["seed"]),
            "base": {
                "route": str(base_routes[sample_id]),
                "actions": base_actions,
                "stream": str(stream),
                "stream_bytes": stream.stat().st_size,
                "stream_sha256": sha256_file(stream),
                "output_frames": str(output_frames),
            },
            "points": points,
        })

    value = {
        "experiment": "selected v6 route + SeedVR2 LoRA 0.50 budget curve",
        "status": "frozen-before-real-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "sample_count": 37,
        "dataset_counts": dict(Counter(sample["dataset"] for sample in samples)),
        "sample_manifest": str(sample_manifest),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "base_routes": str(base_manifest),
        "base_routes_sha256": sha256_file(base_manifest),
        "base_evaluation_summary": str(base_summary_path),
        "base_evaluation_summary_sha256": sha256_file(base_summary_path),
        "joint_summary": str(joint_summary_path),
        "joint_summary_sha256": sha256_file(joint_summary_path),
        "lora_checkpoint": str(adapter),
        "lora_checkpoint_sha256": sha256_file(adapter),
        "lora_strength": float(args.lora_strength),
        "rate_points": rate_points,
        "logical_evaluation_count": 37 * len(rate_points),
        "physical_backend_task_count": len(tasks),
        "quarter_budget_exact_action_regression_count": low_regression_count,
        "samples": records,
        "tasks": tasks,
        "reuse_rule": (
            "Reuse only inside the same sample and seed when all 16 actions "
            "match.  The 0.25 point must exactly reproduce the already "
            "verified old-v6-route plus LoRA-0.50 result."),
        "scientific_boundary": {
            "same_fixed_37_development_samples": True,
            "only_pre_encoding_enhance_budget_changes": True,
            "generate_tile_budget_fixed": args.generate_tile_budget,
            "spatial_lambda_fixed": args.spatial_lambda,
            "controller_and_lora_weights_fixed": True,
            "all_new_actions_require_real_stream_and_fresh_decode": True,
            "no_hard_promotion_gate": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "rate_points": [point["key"] for point in rate_points],
        "logical_evaluation_count": value["logical_evaluation_count"],
        "physical_backend_task_count": len(tasks),
        "quarter_budget_exact_action_regression_count": low_regression_count,
    }, ensure_ascii=False, indent=2))


def materialized_record(
    run_root: Path,
    plan: dict,
    plan_sample: dict,
    rate_key_value: str,
    joint_formal_root: Path,
) -> dict:
    sample_id = plan_sample["sample_id"]
    route_info = plan_sample["points"][rate_key_value]
    root = run_root / "formal" / "evaluation" / sample_id / rate_key_value
    backend_root = root / "lora050"
    codec = root / "codec"
    if not (backend_root / "backend.complete").is_file():
        raise RuntimeError(f"backend is incomplete: {backend_root}")
    encode = read(codec / "encode_summary.json")
    decode = read(codec / "decode_summary.json")
    roi = read(root / "roi" / "manifest.json")
    batch = read(backend_root / "restore" / "roi_batch_metadata.json")
    evaluation = read(backend_root / "evaluation" / "summary.json")
    stream = Path(encode["stream"]).resolve()
    stream_bytes = int(encode["stream_bytes"])
    if (
        not stream.is_file()
        or stream.stat().st_size != stream_bytes
        or int(decode["stream_bytes"]) != stream_bytes
        or evaluation["fresh_decode_regression"].get("pixel_exact") is not True
        or batch.get("complete") is not True
        or batch.get("lora_checkpoint_sha256")
        != plan["lora_checkpoint_sha256"]
        or not math.isclose(
            float(batch.get("lora_strength", -1.0)),
            float(plan["lora_strength"]), abs_tol=1e-12)
    ):
        raise RuntimeError(f"stream, decode, or adapter audit failed: {root}")
    actions = list(map(int, route_info["actions"]))
    roi_actions = np.asarray(roi["actions"], dtype=np.int64).reshape(-1).tolist()
    if actions != roi_actions:
        raise RuntimeError(f"route and ROI actions differ: {root}")
    gate = read(joint_formal_root / sample_id / "uniform_gate" / "summary.json")
    reference = load_source(gate, 17)
    decoded = validate_frames(codec / "fresh_decode")
    output_path = Path(evaluation["output_frames"]).resolve()
    output = validate_frames(output_path)
    if not outside_generate_unchanged(decoded, output, actions):
        raise RuntimeError(f"non-Generate pixels changed: {root}")
    boundary = boundary_metrics(
        reference, output, np.asarray(actions).reshape(4, 4), 128)
    return {
        "logical_variant": rate_key_value,
        "physical_source": str(backend_root),
        "reused": False,
        "reuse_source": None,
        "route": route_info["route"],
        "actions": actions,
        "action_counts": action_counts(actions),
        "generate_boundary_edges": generate_boundary_edges(actions),
        "generate_component_count": len(generate_components(actions)),
        "quality": {
            name: evaluation["quality"]["roi-spatial-bge-stitched"][name]
            for name in QUALITY_NAMES
        },
        "boundary": boundary,
        "actual_on_disk_bytes": stream_bytes,
        "stream": str(stream),
        "runtime_seconds": float(
            evaluation["runtime"]["full_roi_pipeline_seconds"]),
        "peak_cuda_allocated_bytes": int(
            evaluation["runtime"]["peak_cuda_allocated_bytes"]),
        "fresh_decode_pixel_exact": True,
        "outside_generate_pixels_unchanged": True,
        "component_count": len(generate_components(actions)),
        "output_frames": str(output_path),
        "visual": evaluation["visual"],
        "backend": "lora050",
    }


def prior_record(source: dict, plan_sample: dict, point: dict) -> dict:
    value = copy.deepcopy(source)
    stream = Path(value["stream"]).resolve()
    base = plan_sample["base"]
    if (
        not stream.is_file()
        or stream.stat().st_size != int(base["stream_bytes"])
        or sha256_file(stream) != base["stream_sha256"]
        or list(map(int, value["actions"])) != point["actions"]
    ):
        raise RuntimeError(f"prior base result changed: {plan_sample['sample_id']}")
    value.update({
        "logical_variant": "enhance-025",
        "reused": True,
        "reuse_source": "verified old-v6-route + LoRA-0.50 result",
        "reuse_reason": "all 16 actions match the frozen 0.25 budget point",
        "route": point["route"],
        "backend": "lora050",
    })
    return value


def reused_record(source: dict, rate_key_value: str, point: dict) -> dict:
    value = copy.deepcopy(source)
    if list(map(int, value["actions"])) != point["actions"]:
        raise RuntimeError("attempted to reuse a different action map")
    value.update({
        "logical_variant": rate_key_value,
        "reused": True,
        "reuse_source": source["logical_variant"],
        "reuse_reason": "all 16 actions match inside the same sample and seed",
        "route": point["route"],
    })
    return value


def action_image(actions: list[int]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.int64).reshape(4, 4)
    return np.repeat(np.repeat(ACTION_COLORS[values], 128, axis=0), 128, axis=1)


def save_visual(
    path: Path,
    reference: list[np.ndarray],
    point_order: list[dict],
    variants: dict[str, dict],
) -> None:
    frame_index = 8
    frames = {
        point["key"]: validate_frames(
            Path(variants[point["key"]]["output_frames"]))
        for point in point_order
    }
    items = [panel(reference[frame_index], "GT", None)]
    for point in point_order:
        key = point["key"]
        title = f"Enhance budget {point['enhance_budget_ratio']:.0%}"
        items.append(panel(
            frames[key][frame_index], title, variants[key]["quality"]))
    for point in point_order:
        key = point["key"]
        title = f"Actions {point['enhance_budget_ratio']:.0%}"
        items.append(panel(action_image(variants[key]["actions"]), title, None))
    width = max(item.width for item in items)
    height = max(item.height for item in items)
    columns = 5
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), (232, 232, 232))
    for offset, item in enumerate(items):
        canvas.paste(item, ((offset % columns) * width, (offset // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def group_samples(samples: list[dict]) -> dict[str, list[dict]]:
    return {
        "combined": samples,
        "REDS": [sample for sample in samples if sample["dataset"] == "REDS"],
        "UVG": [sample for sample in samples if sample["dataset"] == "UVG"],
        "UVG-adaptation-sequences": [
            sample for sample in samples
            if sample["dataset"] == "UVG"
            and sample["sequence"] in UVG_ADAPTATION_SEQUENCES
        ],
        "UVG-holdout-sequences": [
            sample for sample in samples
            if sample["dataset"] == "UVG"
            and sample["sequence"] in UVG_HOLDOUT_SEQUENCES
        ],
    }


def aggregate(samples: list[dict], keys: list[str]) -> dict:
    pixels = len(samples) * 17 * 512 * 512
    result = {}
    for key in keys:
        rows = [sample["variants"][key] for sample in samples]
        total_bytes = sum(int(row["actual_on_disk_bytes"]) for row in rows)
        boundaries = [
            row["boundary"] for row in rows
            if row["boundary"]["cross_generate_boundary_edge_count"] > 0
        ]
        counts = Counter()
        for row in rows:
            counts.update(row["actions"])
        result[key] = {
            "sample_count": len(rows),
            "quality_mean": {
                metric: mean([row["quality"][metric] for row in rows])
                for metric in QUALITY_NAMES
            },
            "actual_on_disk_bytes_total": total_bytes,
            "actual_on_disk_bytes_mean": total_bytes / len(rows),
            "aggregate_bpp": 8.0 * total_bytes / pixels,
            "runtime_seconds_mean": mean([row["runtime_seconds"] for row in rows]),
            "peak_cuda_allocated_bytes_max": max(
                int(row["peak_cuda_allocated_bytes"]) for row in rows),
            "action_counts_total": {
                "Base": counts[0], "Generate": counts[1], "Enhance": counts[2],
            },
            "generate_boundary_edges_total": sum(
                row["generate_boundary_edges"] for row in rows),
            "generate_components_total": sum(
                row["generate_component_count"] for row in rows),
            "boundary_sample_count": len(boundaries),
            "boundary_mean": {
                metric: mean([row[metric] for row in boundaries])
                for metric in BOUNDARY_NAMES
            },
            "logical_reuse_count": sum(bool(row.get("reused")) for row in rows),
        }
    return result


def budget_response(samples: list[dict], point_order: list[dict]) -> dict:
    keys = [point["key"] for point in point_order]
    per_sample_bytes_monotonic = 0
    per_sample_distinct = []
    changed_from_previous = []
    for sample in samples:
        values = [sample["variants"][key] for key in keys]
        byte_values = [int(value["actual_on_disk_bytes"]) for value in values]
        per_sample_bytes_monotonic += int(all(
            first <= second for first, second in zip(byte_values, byte_values[1:])))
        per_sample_distinct.append(len({action_key(value["actions"]) for value in values}))
    for first, second in zip(keys, keys[1:]):
        changed_from_previous.append({
            "before": first,
            "after": second,
            "sample_count_with_action_change": sum(
                sample["variants"][first]["actions"]
                != sample["variants"][second]["actions"]
                for sample in samples),
            "changed_tile_count": sum(
                sum(a != b for a, b in zip(
                    sample["variants"][first]["actions"],
                    sample["variants"][second]["actions"]))
                for sample in samples),
        })
    return {
        "sample_count": len(samples),
        "per_sample_actual_bytes_nondecreasing_count": per_sample_bytes_monotonic,
        "mean_distinct_action_maps_per_sample": mean(per_sample_distinct),
        "samples_with_more_than_one_action_map": sum(
            value > 1 for value in per_sample_distinct),
        "adjacent_point_action_changes": changed_from_previous,
    }


def markdown(result: dict) -> str:
    lines = [
        "# 旧 v6 route + SeedVR2 LoRA 0.50：真实多预算曲线", "",
        "固定 controller、Generate 上限 4、空间项 0.004 与 LoRA 0.50，只改变编码前已知的 Enhance 字节预算。LPIPS 越低越好。", "",
    ]
    for group, title in (("combined", "合并 37 条"), ("REDS", "REDS 30 条"), ("UVG", "UVG 7 条")):
        lines.extend([
            f"## {title}", "",
            "| Enhance 预算 | 平均真实字节 | bpp | LPIPS | PSNR | 时序误差 | 平均时间 | B/G/E |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for point in result["rate_points"]:
            value = result["aggregate"][group][point["key"]]
            counts = value["action_counts_total"]
            lines.append(
                f"| {point['enhance_budget_ratio']:.0%} | "
                f"{value['actual_on_disk_bytes_mean']:.1f} | "
                f"{value['aggregate_bpp']:.6f} | "
                f"{value['quality_mean']['lpips_alex']:.6f} | "
                f"{value['quality_mean']['psnr_db']:.3f} | "
                f"{value['quality_mean']['temporal_delta_mae']:.3f} | "
                f"{value['runtime_seconds_mean']:.3f} | "
                f"{counts['Base']}/{counts['Generate']}/{counts['Enhance']} |")
        lines.append("")
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    plan = read(args.plan.resolve())
    joint_formal_root = args.joint_formal_root.resolve()
    if plan.get("status") != "frozen-before-real-evaluation":
        raise RuntimeError("budget curve plan was not frozen")
    for key, hash_key in (
        ("sample_manifest", "sample_manifest_sha256"),
        ("base_routes", "base_routes_sha256"),
        ("base_evaluation_summary", "base_evaluation_summary_sha256"),
        ("joint_summary", "joint_summary_sha256"),
        ("lora_checkpoint", "lora_checkpoint_sha256"),
    ):
        if sha256_file(Path(plan[key])) != plan[hash_key]:
            raise RuntimeError(f"frozen input changed: {key}")
    for point in plan["rate_points"]:
        if sha256_file(Path(point["route_manifest"])) != point["route_manifest_sha256"]:
            raise RuntimeError(f"route manifest changed: {point['key']}")
    base_summary = read(Path(plan["base_evaluation_summary"]))
    base_by_id = {sample["sample_id"]: sample for sample in base_summary["samples"]}
    joint_summary = read(Path(plan["joint_summary"]))
    point_order = plan["rate_points"]
    keys = [point["key"] for point in point_order]
    samples = []
    visuals = run_root / "formal" / "visuals"
    for plan_sample in plan["samples"]:
        sample_id = plan_sample["sample_id"]
        records = {}
        for point in point_order:
            key = point["key"]
            point_info = plan_sample["points"][key]
            reuse = point_info["reuse"]
            if reuse is None:
                record = materialized_record(
                    run_root, plan, plan_sample, key, joint_formal_root)
            elif reuse["kind"] == "prior-base":
                record = prior_record(
                    base_by_id[sample_id]["variants"]["old-route-lora050"],
                    plan_sample, point_info)
                record["logical_variant"] = key
            elif reuse["kind"] == "rate-point":
                source_key = reuse["rate_key"]
                if source_key not in records:
                    raise RuntimeError("reuse source was not materialized first")
                record = reused_record(records[source_key], key, point_info)
            else:
                raise RuntimeError("unknown reuse kind")
            records[key] = record
        gate = read(joint_formal_root / sample_id / "uniform_gate" / "summary.json")
        reference = load_source(gate, 17)
        visual = visuals / f"{sample_id}.png"
        save_visual(visual, reference, point_order, records)
        samples.append({
            "sample_id": sample_id,
            "dataset": plan_sample["dataset"],
            "sequence": plan_sample["sequence"],
            "data_role": plan_sample["data_role"],
            "seed": plan_sample["seed"],
            "variants": records,
            "fixed_visual": str(visual),
        })
        print(json.dumps({
            "stage": "budget-curve-summary-sample",
            "sample_id": sample_id,
            "bytes": {key: records[key]["actual_on_disk_bytes"] for key in keys},
            "lpips": {key: records[key]["quality"]["lpips_alex"] for key in keys},
        }, ensure_ascii=False), flush=True)

    groups = group_samples(samples)
    if [len(groups[name]) for name in (
        "combined", "REDS", "UVG", "UVG-adaptation-sequences",
        "UVG-holdout-sequences",
    )] != [37, 30, 7, 5, 2]:
        raise RuntimeError("evaluation group counts differ")
    aggregate_value = {name: aggregate(rows, keys) for name, rows in groups.items()}
    response = {name: budget_response(rows, point_order)
                for name, rows in groups.items()}
    scalar_baselines = {
        group: {name: copy.deepcopy(joint_summary["aggregate"][group][name])
                for name in SCALAR_VARIANTS}
        for group in ("combined", "REDS", "UVG")
    }
    nearest_scalar = {}
    for group in ("combined", "REDS", "UVG"):
        nearest_scalar[group] = {}
        for key in keys:
            point = aggregate_value[group][key]
            name = min(
                SCALAR_VARIANTS,
                key=lambda candidate: abs(
                    scalar_baselines[group][candidate]["actual_on_disk_bytes_mean"]
                    - point["actual_on_disk_bytes_mean"]))
            baseline = scalar_baselines[group][name]
            nearest_scalar[group][key] = {
                "variant": name,
                "mean_byte_delta": (
                    point["actual_on_disk_bytes_mean"]
                    - baseline["actual_on_disk_bytes_mean"]),
                "lpips_delta": (
                    point["quality_mean"]["lpips_alex"]
                    - baseline["quality_mean"]["lpips_alex"]),
                "psnr_delta_db": (
                    point["quality_mean"]["psnr_db"]
                    - baseline["quality_mean"]["psnr_db"]),
            }
    markers = list(run_root.glob(
        "formal/evaluation/*/*/lora050/backend.complete"))
    if len(markers) != plan["physical_backend_task_count"]:
        raise RuntimeError("physical backend completion count differs")
    all_records = [sample["variants"][key] for sample in samples for key in keys]
    result = {
        "experiment": "selected v6 route + SeedVR2 LoRA 0.50 budget curve",
        "status": "complete",
        "sample_count": 37,
        "dataset_counts": dict(Counter(sample["dataset"] for sample in samples)),
        "primary_metric": "LPIPS Alex (lower is better)",
        "feather_pixels": 16,
        "rate_points": point_order,
        "plan": str(args.plan.resolve()),
        "aggregate": aggregate_value,
        "budget_response": response,
        "scalar_dcvc_uf_baselines": scalar_baselines,
        "nearest_scalar_comparisons": nearest_scalar,
        "samples": samples,
        "verification": {
            "physical_backend_tasks": len(markers),
            "logical_evaluations": len(all_records),
            "quarter_budget_exact_action_regression_count": plan[
                "quarter_budget_exact_action_regression_count"],
            "all_stream_sizes_rechecked_against_files": True,
            "all_fresh_decodes_pixel_exact": all(
                record["fresh_decode_pixel_exact"] for record in all_records),
            "all_non_generate_pixels_unchanged": all(
                record["outside_generate_pixels_unchanged"]
                for record in all_records),
            "adapter_sha256_and_strength_match_frozen_plan": True,
            "exact_action_reuse_only": True,
            "fixed_visual_count": len(samples),
            "fixed_visuals_exist": all(
                Path(sample["fixed_visual"]).is_file() for sample in samples),
        },
        "notion_visual_candidates": {
            sample["sequence"]: sample["fixed_visual"]
            for sample in samples
            if sample["sequence"] in {"Beauty", "Jockey", "YachtRide"}
        },
        "scientific_boundary": {
            "same_37_samples_are_development_and_error_analysis": True,
            "only_pre_encoding_enhance_budget_changes": True,
            "generate_tile_budget_fixed_at_four": True,
            "spatial_lambda_fixed_at_0_004": True,
            "controller_and_lora_weights_fixed": True,
            "all_rate_values_are_real_final_file_bytes": True,
            "source_rgb_not_read_by_decoder": True,
            "dcvc_uf_and_spatial_qp_codec_frozen": True,
            "training_or_finetuning_during_evaluation": False,
            "single_gpu": True,
        },
    }
    atomic_json(args.output.resolve(), result)
    atomic_text(args.output.with_suffix(".md"), markdown(result) + "\n")
    rows = []
    for group_name, variants in aggregate_value.items():
        for point in point_order:
            key = point["key"]
            value = variants[key]
            rows.append({
                "group": group_name,
                "rate_key": key,
                "enhance_budget_ratio": point["enhance_budget_ratio"],
                "generate_tile_budget": point["generate_tile_budget"],
                "sample_count": value["sample_count"],
                "mean_actual_bytes": value["actual_on_disk_bytes_mean"],
                "aggregate_bpp": value["aggregate_bpp"],
                **value["quality_mean"],
                **{f"actions_{name.lower()}": count
                   for name, count in value["action_counts_total"].items()},
                "generate_boundary_edges_total": value[
                    "generate_boundary_edges_total"],
                "generate_components_total": value[
                    "generate_components_total"],
                "runtime_seconds_mean": value["runtime_seconds_mean"],
            })
    csv_path = args.output.with_suffix(".csv")
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    print(json.dumps({
        "summary": str(args.output.resolve()),
        "combined": aggregate_value["combined"],
        "budget_response": response["combined"],
        "verification": result["verification"],
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    assert rate_key(0.0) == "enhance-000"
    assert rate_key(0.25) == "enhance-025"
    assert rate_key(1.0) == "enhance-100"
    assert action_key([0, 1, 2, 0]) == "0120"
    samples = [{
        "variants": {
            "a": {"actual_on_disk_bytes": 10, "actions": [0, 0]},
            "b": {"actual_on_disk_bytes": 11, "actions": [0, 2]},
        }
    }]
    result = budget_response(samples, [{"key": "a"}, {"key": "b"}])
    assert result["per_sample_actual_bytes_nondecreasing_count"] == 1
    assert result["samples_with_more_than_one_action_map"] == 1
    print(json.dumps({"stage": "self-test", "status": "passed"}))


def main() -> None:
    args = parse_args()
    if args.mode == "plan":
        make_plan(args)
    elif args.mode == "summarize":
        summarize(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
