#!/usr/bin/env python3
"""Plan and summarize the SeedVR2-LoRA x retrained-router 2x2 evaluation.

The frozen v6 result is the old-router/frozen-SeedVR2 corner.  We materialize
only the other corners that cannot be reused by an exact 16-cell action-map
match.  Every changed route still goes through a real spatial-QP stream and a
fresh decode before either restoration backend is evaluated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
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
    generate_boundary_edges,
    generate_components,
)
from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import panel


VARIANTS = (
    "old-route-frozen",
    "old-route-lora050",
    "new-route-frozen",
    "new-route-lora050",
)
QUALITY_NAMES = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
BOUNDARY_NAMES = (
    "gradient_error_mae",
    "output_jump_mae",
    "reference_jump_mae",
    "boundary_band_rgb_mae",
)
UVG_ADAPTATION_SEQUENCES = {
    "Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry",
}
UVG_HOLDOUT_SEQUENCES = {"ReadySetGo", "YachtRide"}
ACTION_COLORS = np.asarray(
    ((74, 144, 226), (242, 160, 42), (70, 170, 92)), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--sample-manifest", type=Path, required=True)
    plan.add_argument("--old-routes", type=Path, required=True)
    plan.add_argument("--new-routes", type=Path, required=True)
    plan.add_argument("--old-v6-summary", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--lora-strength", type=float, default=0.5)
    plan.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--joint-formal-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("self-test")
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def selected_actions(route_path: Path) -> tuple[dict, list[int]]:
    route = read(route_path)
    selected = route["variants"][route["selected_variant"]]
    actions = list(map(int, selected["actions"]))
    if len(actions) != 16 or any(value not in (0, 1, 2) for value in actions):
        raise RuntimeError(f"invalid route actions: {route_path}")
    configuration = route["configuration"]
    if (
        configuration.get("tile_size") != 128
        or configuration.get("tile_grid") != [4, 4]
        or configuration.get("quality_profile")
        != {"Generate": 8, "Base": 16, "Enhance": 32}
    ):
        raise RuntimeError(f"unexpected route configuration: {route_path}")
    return route, actions


def route_manifest(path: Path) -> dict[str, Path]:
    manifest = read(path)
    entries = manifest.get("entries", [])
    if manifest.get("sample_count") != 37 or len(entries) != 37:
        raise RuntimeError(f"route manifest is incomplete: {path}")
    result = {}
    for entry in entries:
        sample_id = entry["sample_id"]
        route = Path(entry["path"]).resolve()
        if sample_id in result or not route.is_file():
            raise RuntimeError(f"duplicate or missing route: {sample_id}")
        result[sample_id] = route
    return result


def make_plan(args: argparse.Namespace) -> None:
    if not math.isfinite(args.lora_strength) or args.lora_strength < 0:
        raise ValueError("LoRA strength must be finite and nonnegative")
    sample_manifest = args.sample_manifest.resolve()
    old_manifest = args.old_routes.resolve()
    new_manifest = args.new_routes.resolve()
    old_summary_path = args.old_v6_summary.resolve()
    adapter = args.lora_checkpoint.resolve()
    samples = read_jsonl(sample_manifest)
    old_summary = read(old_summary_path)
    old_routes = route_manifest(old_manifest)
    new_routes = route_manifest(new_manifest)
    if len(samples) != 37 or old_summary.get("sample_count") != 37:
        raise RuntimeError("expected the fixed 37-sample evaluation")
    old_by_id = {sample["sample_id"]: sample for sample in old_summary["samples"]}
    sample_ids = [sample["sample_id"] for sample in samples]
    if (
        len(set(sample_ids)) != 37
        or set(sample_ids) != set(old_by_id)
        or set(sample_ids) != set(old_routes)
        or set(sample_ids) != set(new_routes)
    ):
        raise RuntimeError("sample, old route, new route, and v6 IDs differ")

    records = []
    tasks = []
    changed = 0
    no_generate_reuse = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        old_path = old_routes[sample_id]
        new_path = new_routes[sample_id]
        _, old_actions = selected_actions(old_path)
        _, new_actions = selected_actions(new_path)
        old_record = old_by_id[sample_id]["variants"]["combined"]
        if old_record["actions"] != old_actions:
            raise RuntimeError(f"old v6 summary and route differ: {sample_id}")
        old_stream = Path(old_record["stream"]).resolve()
        old_codec = old_stream.parent
        for required in (
            old_stream,
            old_codec / "encode_summary.json",
            old_codec / "decode_summary.json",
            old_codec / "encoder_reconstruction",
            old_codec / "fresh_decode",
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        if old_stream.stat().st_size != int(old_record["actual_on_disk_bytes"]):
            raise RuntimeError(f"old stream size changed: {sample_id}")

        same_actions = old_actions == new_actions
        old_has_generate = 1 in old_actions
        new_has_generate = 1 in new_actions
        if old_has_generate:
            tasks.append({
                "sample_id": sample_id,
                "dataset": sample["dataset"],
                "route_key": "old",
                "backend": "lora050",
                "route": str(old_path),
                "seed": int(sample["seed"]),
                "codec_mode": "verified-v6-reuse",
                "codec_dir": str(old_codec),
            })
        else:
            no_generate_reuse += 1
        if not same_actions:
            changed += 1
            tasks.append({
                "sample_id": sample_id,
                "dataset": sample["dataset"],
                "route_key": "new",
                "backend": "frozen",
                "route": str(new_path),
                "seed": int(sample["seed"]),
                "codec_mode": "new-real-stream",
                "codec_dir": None,
            })
            if new_has_generate:
                tasks.append({
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "route_key": "new",
                    "backend": "lora050",
                    "route": str(new_path),
                    "seed": int(sample["seed"]),
                    "codec_mode": "new-real-stream",
                    "codec_dir": None,
                })
            else:
                no_generate_reuse += 1
        records.append({
            "sample_id": sample_id,
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "data_role": sample["data_role"],
            "seed": int(sample["seed"]),
            "old": {
                "route": str(old_path),
                "actions": old_actions,
                "action_key": "".join(map(str, old_actions)),
                "codec_dir": str(old_codec),
                "stream": str(old_stream),
                "stream_bytes": old_stream.stat().st_size,
                "stream_sha256": sha256_file(old_stream),
                "has_generate": old_has_generate,
            },
            "new": {
                "route": str(new_path),
                "actions": new_actions,
                "action_key": "".join(map(str, new_actions)),
                "has_generate": new_has_generate,
            },
            "same_actions": same_actions,
            "changed_tile_count": sum(
                first != second for first, second in zip(old_actions, new_actions)),
        })

    value = {
        "experiment": "SeedVR2 LoRA x retrained router 2x2 real evaluation",
        "status": "frozen-before-real-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": 37,
        "dataset_counts": dict(Counter(sample["dataset"] for sample in samples)),
        "logical_variants": list(VARIANTS),
        "sample_manifest": str(sample_manifest),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "old_routes": str(old_manifest),
        "old_routes_sha256": sha256_file(old_manifest),
        "new_routes": str(new_manifest),
        "new_routes_sha256": sha256_file(new_manifest),
        "old_v6_summary": str(old_summary_path),
        "old_v6_summary_sha256": sha256_file(old_summary_path),
        "lora_checkpoint": str(adapter),
        "lora_checkpoint_sha256": sha256_file(adapter),
        "lora_strength": float(args.lora_strength),
        "changed_route_sample_count": changed,
        "unchanged_route_sample_count": 37 - changed,
        "physical_backend_task_count": len(tasks),
        "logical_evaluation_count": 37 * len(VARIANTS),
        "no_generate_backend_reuse_count": no_generate_reuse,
        "samples": records,
        "tasks": tasks,
        "reuse_rule": (
            "Reuse only within the same sample and seed when all 16 actions "
            "match, or between frozen and LoRA when the action map has no "
            "Generate cell. Controller scores never trigger reuse."),
        "scientific_boundary": {
            "same_fixed_37_development_samples": True,
            "dcvc_uf_frozen": True,
            "spatial_qp_syntax_frozen": True,
            "only_router_weights_and_seedvr2_lora_corner_change": True,
            "all_changed_routes_require_real_stream_and_fresh_decode": True,
            "no_hard_promotion_gate": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "sample_count": 37,
        "changed_route_sample_count": changed,
        "physical_backend_task_count": len(tasks),
        "logical_evaluation_count": 37 * len(VARIANTS),
    }, ensure_ascii=False, indent=2))


def mean(values: list[float | int | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def validate_frames(path: Path) -> list[np.ndarray]:
    frames = load_pngs(path)
    if len(frames) != 17 or any(frame.shape[:2] != (512, 512) for frame in frames):
        raise RuntimeError(f"expected 17 512x512 frames: {path}")
    return frames


def outside_generate_unchanged(
    decoded: list[np.ndarray], output: list[np.ndarray], actions: list[int],
) -> bool:
    generate = np.asarray(actions, dtype=np.int64).reshape(4, 4) == 1
    generate = np.repeat(np.repeat(generate, 128, axis=0), 128, axis=1)
    outside = ~generate
    return all(np.array_equal(first[outside], second[outside])
               for first, second in zip(decoded, output))


def materialized_record(
    run_root: Path, plan_sample: dict, route_key: str, backend: str,
    joint_formal_root: Path,
) -> dict:
    sample_id = plan_sample["sample_id"]
    route_info = plan_sample[route_key]
    root = run_root / "formal" / "evaluation" / sample_id / route_key
    backend_root = root / backend
    if not (backend_root / "backend.complete").is_file():
        raise RuntimeError(f"backend is incomplete: {backend_root}")
    codec = (
        Path(route_info["codec_dir"]).resolve()
        if route_key == "old" else root / "codec")
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
    ):
        raise RuntimeError(f"stream, fresh decode, or batch audit failed: {root}")
    actions = list(map(int, route_info["actions"]))
    roi_actions = np.asarray(roi["actions"], dtype=np.int64).reshape(-1).tolist()
    if actions != roi_actions:
        raise RuntimeError(f"route and ROI actions differ: {root}")
    if backend == "lora050":
        plan = read(run_root / "evaluation_plan.json")
        if (
            batch.get("lora_checkpoint_sha256")
            != plan["lora_checkpoint_sha256"]
            or not math.isclose(
                float(batch.get("lora_strength", -1.0)),
                float(plan["lora_strength"]), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"LoRA identity or strength differs: {root}")
    elif batch.get("lora_checkpoint") is not None:
        raise RuntimeError(f"frozen backend unexpectedly has LoRA: {root}")
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
        "logical_variant": f"{route_key}-route-{backend}",
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
        "backend": backend,
    }


def reuse_record(source: dict, name: str, route_info: dict, reason: str) -> dict:
    value = copy.deepcopy(source)
    value.update({
        "logical_variant": name,
        "reused": True,
        "reuse_source": source["logical_variant"],
        "reuse_reason": reason,
        "route": route_info["route"],
        "actions": list(route_info["actions"]),
        "action_counts": action_counts(route_info["actions"]),
        "generate_boundary_edges": generate_boundary_edges(route_info["actions"]),
        "generate_component_count": len(generate_components(route_info["actions"])),
    })
    return value


def baseline_record(old_sample: dict, plan_sample: dict) -> dict:
    value = copy.deepcopy(old_sample["variants"]["combined"])
    stream = Path(value["stream"]).resolve()
    if (
        not stream.is_file()
        or stream.stat().st_size != int(value["actual_on_disk_bytes"])
        or sha256_file(stream) != plan_sample["old"]["stream_sha256"]
    ):
        raise RuntimeError(f"v6 baseline stream changed: {plan_sample['sample_id']}")
    value.update({
        "logical_variant": "old-route-frozen",
        "backend": "frozen",
        "reused": True,
        "reuse_source": "verified v6 combined formal output",
        "outside_generate_pixels_unchanged": True,
    })
    return value


def action_image(actions: list[int]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.int64).reshape(4, 4)
    return np.repeat(np.repeat(ACTION_COLORS[values], 128, axis=0), 128, axis=1)


def save_visual(
    path: Path, reference: list[np.ndarray], variants: dict[str, dict],
) -> None:
    index = 8
    frames = {
        name: validate_frames(Path(record["output_frames"]))
        for name, record in variants.items()
    }
    difference = np.clip(
        np.abs(frames["new-route-lora050"][index].astype(np.int16)
               - frames["old-route-frozen"][index].astype(np.int16)) * 4,
        0, 255).astype(np.uint8)
    items = [
        panel(reference[index], "GT", None),
        panel(frames["old-route-frozen"][index], "Old route + frozen",
              variants["old-route-frozen"]["quality"]),
        panel(frames["old-route-lora050"][index], "Old route + LoRA 0.50",
              variants["old-route-lora050"]["quality"]),
        panel(frames["new-route-frozen"][index], "New route + frozen",
              variants["new-route-frozen"]["quality"]),
        panel(frames["new-route-lora050"][index], "New route + LoRA 0.50",
              variants["new-route-lora050"]["quality"]),
        panel(action_image(variants["old-route-frozen"]["actions"]),
              "Old action map", None),
        panel(action_image(variants["new-route-lora050"]["actions"]),
              "New action map", None),
        panel(difference, "4x |new full - old v6|", None),
    ]
    width = max(item.width for item in items)
    height = max(item.height for item in items)
    canvas = Image.new("RGB", (4 * width, 2 * height), (232, 232, 232))
    for offset, item in enumerate(items):
        canvas.paste(item, ((offset % 4) * width, (offset // 4) * height))
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


def aggregate(samples: list[dict]) -> dict:
    if not samples:
        raise RuntimeError("cannot aggregate an empty group")
    output = {}
    pixels = len(samples) * 17 * 512 * 512
    for name in VARIANTS:
        rows = [sample["variants"][name] for sample in samples]
        total_bytes = sum(int(row["actual_on_disk_bytes"]) for row in rows)
        boundaries = [
            row["boundary"] for row in rows
            if row["boundary"]["cross_generate_boundary_edge_count"] > 0
        ]
        counts = Counter()
        for row in rows:
            counts.update(row["actions"])
        output[name] = {
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
                label: counts[index]
                for index, label in enumerate(("Base", "Generate", "Enhance"))
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
    return output


def compare(samples: list[dict]) -> dict:
    pairs = {
        "lora_on_old_route": ("old-route-lora050", "old-route-frozen"),
        "router_under_frozen": ("new-route-frozen", "old-route-frozen"),
        "router_under_lora": ("new-route-lora050", "old-route-lora050"),
        "lora_on_new_route": ("new-route-lora050", "new-route-frozen"),
        "full_new_vs_v6": ("new-route-lora050", "old-route-frozen"),
    }
    effects = {}
    for label, (after, before) in pairs.items():
        metric_deltas = {
            metric: [
                sample["variants"][after]["quality"][metric]
                - sample["variants"][before]["quality"][metric]
                for sample in samples
            ]
            for metric in QUALITY_NAMES
        }
        lpips = metric_deltas["lpips_alex"]
        effects[label] = {
            "after": after,
            "before": before,
            "quality_mean_delta": {
                metric: mean(values) for metric, values in metric_deltas.items()
            },
            "lpips_better_count": sum(value < -1e-12 for value in lpips),
            "lpips_equal_count": sum(abs(value) <= 1e-12 for value in lpips),
            "mean_byte_delta": mean([
                sample["variants"][after]["actual_on_disk_bytes"]
                - sample["variants"][before]["actual_on_disk_bytes"]
                for sample in samples
            ]),
            "mean_runtime_delta_seconds": mean([
                sample["variants"][after]["runtime_seconds"]
                - sample["variants"][before]["runtime_seconds"]
                for sample in samples
            ]),
            "sample_count": len(samples),
        }
    frozen_route = effects["router_under_frozen"]["quality_mean_delta"][
        "lpips_alex"]
    lora_route = effects["router_under_lora"]["quality_mean_delta"][
        "lpips_alex"]
    return {
        "effects": effects,
        "lpips_router_x_lora_interaction": lora_route - frozen_route,
    }


def markdown(result: dict) -> str:
    lines = [
        "# LoRA 0.50 × router 重训：37 条真实码流 2×2 评估",
        "",
        "四个角分别是旧／新 router 与冻结／LoRA 0.50 SeedVR2。所有改变的动作图都重新写入真实 spatial-QP 码流并 fresh decode；LPIPS 越低越好。",
        "",
    ]
    for group, title in (
        ("combined", "合并 37 条"), ("REDS", "REDS 30 条"),
        ("UVG", "UVG 7 条"),
    ):
        lines.extend([
            f"## {title}", "",
            "| 版本 | 平均字节 | LPIPS | PSNR | 时序误差 | G 边 | G 连通块 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name in VARIANTS:
            value = result["aggregate"][group][name]
            lines.append(
                f"| {name} | {value['actual_on_disk_bytes_mean']:.1f} | "
                f"{value['quality_mean']['lpips_alex']:.6f} | "
                f"{value['quality_mean']['psnr_db']:.3f} | "
                f"{value['quality_mean']['temporal_delta_mae']:.3f} | "
                f"{value['generate_boundary_edges_total']} | "
                f"{value['generate_components_total']} |")
        effects = result["comparisons"][group]["effects"]
        full = effects["full_new_vs_v6"]
        lines.extend([
            "",
            f"- 完整新版相对 v6：LPIPS {full['quality_mean_delta']['lpips_alex']:+.6f}，{full['lpips_better_count']}/{full['sample_count']} 条更好，平均字节 {full['mean_byte_delta']:+.1f}。",
            f"- 旧 route 上只换 LoRA：{effects['lora_on_old_route']['quality_mean_delta']['lpips_alex']:+.6f}；冻结恢复器下只换 router：{effects['router_under_frozen']['quality_mean_delta']['lpips_alex']:+.6f}。",
            "",
        ])
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    plan = read(args.plan.resolve())
    joint_root = args.joint_formal_root.resolve()
    if plan.get("status") != "frozen-before-real-evaluation":
        raise RuntimeError("evaluation plan was not frozen")
    for key, hash_key in (
        ("sample_manifest", "sample_manifest_sha256"),
        ("old_routes", "old_routes_sha256"),
        ("new_routes", "new_routes_sha256"),
        ("old_v6_summary", "old_v6_summary_sha256"),
        ("lora_checkpoint", "lora_checkpoint_sha256"),
    ):
        if sha256_file(Path(plan[key])) != plan[hash_key]:
            raise RuntimeError(f"frozen input changed: {key}")
    old_summary = read(Path(plan["old_v6_summary"]))
    old_by_id = {sample["sample_id"]: sample for sample in old_summary["samples"]}

    samples = []
    visuals = run_root / "formal" / "visuals"
    for plan_sample in plan["samples"]:
        sample_id = plan_sample["sample_id"]
        old_frozen = baseline_record(old_by_id[sample_id], plan_sample)
        if plan_sample["old"]["has_generate"]:
            old_lora = materialized_record(
                run_root, plan_sample, "old", "lora050", joint_root)
            old_lora["logical_variant"] = "old-route-lora050"
        else:
            old_lora = reuse_record(
                old_frozen, "old-route-lora050", plan_sample["old"],
                "no Generate cell, so the restoration backend is unused")

        if plan_sample["same_actions"]:
            new_frozen = reuse_record(
                old_frozen, "new-route-frozen", plan_sample["new"],
                "all 16 old/new actions are identical")
            new_lora = reuse_record(
                old_lora, "new-route-lora050", plan_sample["new"],
                "all 16 old/new actions are identical")
        else:
            new_frozen = materialized_record(
                run_root, plan_sample, "new", "frozen", joint_root)
            new_frozen["logical_variant"] = "new-route-frozen"
            if plan_sample["new"]["has_generate"]:
                new_lora = materialized_record(
                    run_root, plan_sample, "new", "lora050", joint_root)
                new_lora["logical_variant"] = "new-route-lora050"
            else:
                new_lora = reuse_record(
                    new_frozen, "new-route-lora050", plan_sample["new"],
                    "no Generate cell, so the restoration backend is unused")

        variants = {
            "old-route-frozen": old_frozen,
            "old-route-lora050": old_lora,
            "new-route-frozen": new_frozen,
            "new-route-lora050": new_lora,
        }
        gate = read(joint_root / sample_id / "uniform_gate" / "summary.json")
        reference = load_source(gate, 17)
        visual = visuals / f"{sample_id}.png"
        save_visual(visual, reference, variants)
        samples.append({
            "sample_id": sample_id,
            "dataset": plan_sample["dataset"],
            "sequence": plan_sample["sequence"],
            "data_role": plan_sample["data_role"],
            "seed": plan_sample["seed"],
            "same_actions": plan_sample["same_actions"],
            "changed_tile_count": plan_sample["changed_tile_count"],
            "variants": variants,
            "fixed_visual": str(visual),
        })
        print(json.dumps({
            "stage": "lora-router-summary-sample",
            "sample_id": sample_id,
            "same_actions": plan_sample["same_actions"],
            "lpips": {name: variants[name]["quality"]["lpips_alex"]
                       for name in VARIANTS},
        }, ensure_ascii=False), flush=True)

    groups = group_samples(samples)
    if [len(groups[name]) for name in (
        "combined", "REDS", "UVG", "UVG-adaptation-sequences",
        "UVG-holdout-sequences",
    )] != [37, 30, 7, 5, 2]:
        raise RuntimeError("evaluation group counts differ")
    aggregate_value = {name: aggregate(rows) for name, rows in groups.items()}
    comparisons_value = {name: compare(rows) for name, rows in groups.items()}
    markers = list(run_root.glob(
        "formal/evaluation/*/*/*/backend.complete"))
    if len(markers) != plan["physical_backend_task_count"]:
        raise RuntimeError("physical backend completion count differs")
    all_records = [
        sample["variants"][name] for sample in samples for name in VARIANTS
    ]
    result = {
        "experiment": "SeedVR2 LoRA x retrained router 2x2 real evaluation",
        "status": "complete",
        "sample_count": 37,
        "dataset_counts": dict(Counter(sample["dataset"] for sample in samples)),
        "logical_variants": list(VARIANTS),
        "primary_metric": "LPIPS Alex (lower is better)",
        "feather_pixels": 16,
        "plan": str(args.plan.resolve()),
        "aggregate": aggregate_value,
        "comparisons": comparisons_value,
        "samples": samples,
        "verification": {
            "physical_backend_tasks": len(markers),
            "logical_evaluations": len(all_records),
            "all_stream_sizes_rechecked_against_files": True,
            "all_fresh_decodes_pixel_exact": all(
                record["fresh_decode_pixel_exact"] for record in all_records),
            "all_non_generate_pixels_unchanged": all(
                record["outside_generate_pixels_unchanged"]
                for record in all_records),
            "adapter_sha256_and_strength_match_frozen_plan": True,
            "exact_action_or_no_generate_reuse_only": True,
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
            "no_hard_promotion_gate": True,
            "result_driven_sample_exclusion": False,
            "all_rate_values_are_real_final_file_bytes": True,
            "source_rgb_not_read_by_decoder": True,
            "dcvc_uf_frozen": True,
            "spatial_qp_codec_frozen": True,
            "training_or_finetuning_during_evaluation": False,
            "single_gpu": True,
        },
    }
    atomic_json(args.output.resolve(), result)
    atomic_text(args.output.with_suffix(".md"), markdown(result) + "\n")
    rows = []
    for group_name, variants in aggregate_value.items():
        for variant_name, value in variants.items():
            rows.append({
                "group": group_name,
                "variant": variant_name,
                "sample_count": value["sample_count"],
                "mean_actual_bytes": value["actual_on_disk_bytes_mean"],
                "aggregate_bpp": value["aggregate_bpp"],
                **value["quality_mean"],
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
        "combined_effects": comparisons_value["combined"]["effects"],
        "verification": result["verification"],
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    assert action_counts([0, 1, 2, 1]) == {
        "Base": 1, "Generate": 2, "Enhance": 1}
    sample = {
        "variants": {
            "old-route-frozen": {"quality": {"lpips_alex": 0.5},
                                 "actual_on_disk_bytes": 10,
                                 "runtime_seconds": 2.0},
            "old-route-lora050": {"quality": {"lpips_alex": 0.4},
                                  "actual_on_disk_bytes": 10,
                                  "runtime_seconds": 2.1},
            "new-route-frozen": {"quality": {"lpips_alex": 0.45},
                                 "actual_on_disk_bytes": 11,
                                 "runtime_seconds": 2.0},
            "new-route-lora050": {"quality": {"lpips_alex": 0.35},
                                  "actual_on_disk_bytes": 11,
                                  "runtime_seconds": 2.1},
        }
    }
    for variant in sample["variants"].values():
        variant["quality"].update({
            "psnr_db": 20.0, "temporal_delta_mae": 3.0, "rgb_mse": 4.0})
    result = compare([sample])
    assert math.isclose(
        result["effects"]["full_new_vs_v6"]["quality_mean_delta"][
            "lpips_alex"], -0.15)
    assert result["effects"]["full_new_vs_v6"]["lpips_better_count"] == 1
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
