#!/usr/bin/env python3
"""Aggregate the v6 adaptation x spatial-consistency real evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
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
    "baseline-v5-feather16",
    "adaptation-only",
    "spatial-only",
    "combined",
)
TARGET_VARIANTS = VARIANTS[1:]
QUALITY_NAMES = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
BOUNDARY_NAMES = (
    "gradient_error_mae", "output_jump_mae", "reference_jump_mae",
    "boundary_band_rgb_mae",
)
SCALAR_NAMES = tuple(f"scalar-qp{qp}" for qp in (8, 16, 24, 32))
ACTION_COLORS = np.asarray(
    ((74, 144, 226), (242, 160, 42), (70, 170, 92)), dtype=np.uint8)
UVG_ADAPTATION_SEQUENCES = {
    "Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry",
}
UVG_HOLDOUT_SEQUENCES = {"ReadySetGo", "YachtRide"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--joint-summary", type=Path, required=True)
    parser.add_argument("--feather-summary", type=Path, required=True)
    parser.add_argument("--joint-formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def mean(values: list[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def checked_stream(path: str, expected_bytes: int) -> str:
    stream = Path(path).resolve()
    if not stream.is_file() or stream.stat().st_size != expected_bytes:
        raise RuntimeError(f"stream byte check failed: {stream}")
    return str(stream)


def validate_frames(path: Path) -> list[np.ndarray]:
    frames = load_pngs(path)
    if len(frames) != 17 or any(frame.shape[:2] != (512, 512) for frame in frames):
        raise RuntimeError(f"expected 17 512x512 frames: {path}")
    return frames


def action_image(actions: list[int]) -> np.ndarray:
    array = np.asarray(actions, dtype=np.int64).reshape(4, 4)
    return np.repeat(np.repeat(ACTION_COLORS[array], 128, axis=0), 128, axis=1)


def baseline_record(
    plan_sample: dict, feather_sample: dict, joint_sample: dict,
) -> dict:
    logical = plan_sample["variants"]["baseline-v5-feather16"]
    feather = feather_sample["variants"]["16"]
    joint = joint_sample["variants"]["final-joint"]
    actions = list(map(int, logical["actions"]))
    if feather_sample["action_counts"] != action_counts(actions):
        raise RuntimeError(
            f"baseline action counts differ: {plan_sample['sample_id']}")
    stream_bytes = int(joint["actual_on_disk_bytes"])
    return {
        "logical_variant": "baseline-v5-feather16",
        "canonical_variant": "baseline-v5-feather16",
        "reused": True,
        "reuse_source": "frozen v5 real stream plus verified 16px recompose",
        "route": logical["route"],
        "selected_route_variant": logical["selected_variant"],
        "actions": actions,
        "action_counts": action_counts(actions),
        "generate_boundary_edges": generate_boundary_edges(actions),
        "generate_component_count": len(generate_components(actions)),
        "quality": {
            name: feather["quality"][name] for name in QUALITY_NAMES
        },
        "boundary": feather["boundary"],
        "actual_on_disk_bytes": stream_bytes,
        "stream": checked_stream(joint["stream_path"], stream_bytes),
        "runtime_seconds": float(joint["runtime_seconds"]),
        "peak_cuda_allocated_bytes": int(joint["peak_cuda_allocated_bytes"]),
        "fresh_decode_pixel_exact": True,
        "component_count": len(generate_components(actions)),
        "output_frames": str(Path(feather["frames"]).resolve()),
        "visual": feather_sample["fixed_visual"],
    }


def target_record(
    *, run_root: Path, joint_formal_root: Path, plan_sample: dict,
    canonical_variant: str,
) -> dict:
    sample_id = plan_sample["sample_id"]
    logical = plan_sample["variants"][canonical_variant]
    root = run_root / "formal" / "evaluation" / sample_id / canonical_variant
    if not (root / "variant.complete").is_file():
        raise RuntimeError(f"canonical evaluation is incomplete: {root}")
    evaluation = read(root / "evaluation" / "summary.json")
    encode = read(root / "codec" / "encode_summary.json")
    decode = read(root / "codec" / "decode_summary.json")
    roi = read(root / "roi" / "manifest.json")
    batch = read(root / "roi" / "roi_batch_metadata.json")
    if not evaluation["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"fresh decode differs: {root}")
    if not batch.get("complete"):
        raise RuntimeError(f"ROI batch is incomplete: {root}")
    stream_bytes = int(encode["stream_bytes"])
    if int(decode["stream_bytes"]) != stream_bytes:
        raise RuntimeError(f"encode/decode byte count differs: {root}")
    stream_path = checked_stream(encode["stream"], stream_bytes)
    actions = list(map(int, logical["actions"]))
    roi_actions = np.asarray(roi["actions"], dtype=np.int64).reshape(-1).tolist()
    if roi_actions != actions:
        raise RuntimeError(f"ROI and frozen route actions differ: {root}")
    expected_components = len(generate_components(actions))
    if (
        len(roi["components"]) != expected_components
        or int(evaluation["component_count"]) != expected_components
        or int(batch["component_count"]) != expected_components
    ):
        raise RuntimeError(f"Generate component count differs: {root}")
    gate_path = joint_formal_root / sample_id / "uniform_gate" / "summary.json"
    reference = load_source(read(gate_path), 17)
    frame_path = Path(evaluation["output_frames"]).resolve()
    frames = validate_frames(frame_path)
    boundary = boundary_metrics(
        reference, frames, np.asarray(actions, dtype=np.int64).reshape(4, 4), 128)
    return {
        "logical_variant": canonical_variant,
        "canonical_variant": canonical_variant,
        "reused": False,
        "reuse_source": None,
        "route": logical["route"],
        "selected_route_variant": logical["selected_variant"],
        "actions": actions,
        "action_counts": action_counts(actions),
        "generate_boundary_edges": generate_boundary_edges(actions),
        "generate_component_count": expected_components,
        "quality": {
            name: evaluation["quality"]["roi-spatial-bge-stitched"][name]
            for name in QUALITY_NAMES
        },
        "boundary": boundary,
        "actual_on_disk_bytes": stream_bytes,
        "stream": stream_path,
        "runtime_seconds": float(
            evaluation["runtime"]["full_roi_pipeline_seconds"]),
        "peak_cuda_allocated_bytes": int(
            evaluation["runtime"]["peak_cuda_allocated_bytes"]),
        "fresh_decode_pixel_exact": True,
        "component_count": expected_components,
        "output_frames": str(frame_path),
        "visual": evaluation["visual"],
    }


def logical_record(
    canonical: dict, plan_sample: dict, logical_variant: str,
) -> dict:
    value = dict(canonical)
    planned = plan_sample["variants"][logical_variant]
    if planned["action_key"] != plan_sample["variants"][
            canonical["canonical_variant"]]["action_key"]:
        raise RuntimeError("reuse attempted without exact action match")
    value.update({
        "logical_variant": logical_variant,
        "canonical_variant": canonical["canonical_variant"],
        "reused": logical_variant != canonical["canonical_variant"],
        "reuse_source": (
            canonical["canonical_variant"]
            if logical_variant != canonical["canonical_variant"] else None),
        "route": planned["route"],
        "selected_route_variant": planned["selected_variant"],
        "actions": planned["actions"],
        "action_counts": action_counts(planned["actions"]),
        "generate_boundary_edges": generate_boundary_edges(planned["actions"]),
        "generate_component_count": len(
            generate_components(planned["actions"])),
    })
    return value


def save_comparison_visual(
    path: Path, reference: list[np.ndarray], variants: dict[str, dict],
) -> None:
    frame_index = 8
    labels = {
        "baseline-v5-feather16": "v5 + feather16",
        "adaptation-only": "UVG adaptation",
        "spatial-only": "Spatial only",
        "combined": "Adaptation + spatial",
    }
    panels = [panel(reference[frame_index], "GT", None)]
    loaded = {}
    for name in VARIANTS:
        loaded[name] = validate_frames(Path(variants[name]["output_frames"]))
        panels.append(panel(
            loaded[name][frame_index], labels[name], variants[name]["quality"]))
    for name in VARIANTS:
        panels.append(panel(
            action_image(variants[name]["actions"]),
            f"Map: {labels[name]}", None))
    panel_width = max(item.width for item in panels)
    panel_height = max(item.height for item in panels)
    canvas = Image.new(
        "RGB", (3 * panel_width, 3 * panel_height), (232, 232, 232))
    for index, item in enumerate(panels):
        canvas.paste(
            item, ((index % 3) * panel_width, (index // 3) * panel_height))
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
        "UVG-v6-holdout-sequences": [
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
        total_bytes = sum(row["actual_on_disk_bytes"] for row in rows)
        boundary_rows = [
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
            "runtime_seconds_mean": mean(
                [row["runtime_seconds"] for row in rows]),
            "peak_cuda_allocated_bytes_max": max(
                row["peak_cuda_allocated_bytes"] for row in rows),
            "action_counts_total": {
                action: counts[index]
                for index, action in enumerate(("Base", "Generate", "Enhance"))
            },
            "generate_boundary_edges_total": sum(
                row["generate_boundary_edges"] for row in rows),
            "generate_components_total": sum(
                row["generate_component_count"] for row in rows),
            "boundary_sample_count": len(boundary_rows),
            "boundary_mean": {
                metric: mean([row[metric] for row in boundary_rows])
                for metric in BOUNDARY_NAMES
            },
            "logical_reuse_count": sum(row["reused"] for row in rows),
        }
    return output


def comparisons(samples: list[dict], joint_by_id: dict[str, dict]) -> dict:
    per_variant = {}
    for name in TARGET_VARIANTS:
        rows = []
        for sample in samples:
            baseline = sample["variants"]["baseline-v5-feather16"]
            target = sample["variants"][name]
            old = joint_by_id[sample["sample_id"]]
            nearest_name = min(
                SCALAR_NAMES,
                key=lambda scalar: abs(
                    old["variants"][scalar]["actual_on_disk_bytes"]
                    - target["actual_on_disk_bytes"]),
            )
            scalar = old["variants"][nearest_name]
            rows.append({
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "sequence": sample["sequence"],
                "target_variant": name,
                "canonical_variant": target["canonical_variant"],
                "target_minus_baseline_lpips": (
                    target["quality"]["lpips_alex"]
                    - baseline["quality"]["lpips_alex"]),
                "target_minus_baseline_bytes": (
                    target["actual_on_disk_bytes"]
                    - baseline["actual_on_disk_bytes"]),
                "target_minus_nearest_scalar_lpips": (
                    target["quality"]["lpips_alex"]
                    - scalar["quality"]["lpips_alex"]),
                "nearest_scalar": nearest_name,
                "target_generate_boundary_edges": target[
                    "generate_boundary_edges"],
                "baseline_generate_boundary_edges": baseline[
                    "generate_boundary_edges"],
            })
        per_variant[name] = {
            "sample_count": len(rows),
            "mean_target_minus_baseline_lpips": mean([
                row["target_minus_baseline_lpips"] for row in rows]),
            "mean_target_minus_baseline_bytes": mean([
                row["target_minus_baseline_bytes"] for row in rows]),
            "target_better_lpips_than_baseline_count": sum(
                row["target_minus_baseline_lpips"] < -1e-12 for row in rows),
            "target_equal_lpips_to_baseline_count": sum(
                abs(row["target_minus_baseline_lpips"]) <= 1e-12
                for row in rows),
            "mean_target_minus_nearest_scalar_lpips": mean([
                row["target_minus_nearest_scalar_lpips"] for row in rows]),
            "target_better_lpips_than_nearest_scalar_count": sum(
                row["target_minus_nearest_scalar_lpips"] < 0 for row in rows),
            "mean_generate_boundary_edge_delta_vs_baseline": mean([
                row["target_generate_boundary_edges"]
                - row["baseline_generate_boundary_edges"] for row in rows]),
            "samples": rows,
        }

    effects = {}
    pairs = {
        "adaptation_without_spatial": (
            "adaptation-only", "baseline-v5-feather16"),
        "spatial_without_adaptation": (
            "spatial-only", "baseline-v5-feather16"),
        "adaptation_with_spatial": ("combined", "spatial-only"),
        "spatial_with_adaptation": ("combined", "adaptation-only"),
        "combined_vs_baseline": ("combined", "baseline-v5-feather16"),
    }
    for label, (after_name, before_name) in pairs.items():
        deltas = [
            sample["variants"][after_name]["quality"]["lpips_alex"]
            - sample["variants"][before_name]["quality"]["lpips_alex"]
            for sample in samples
        ]
        byte_deltas = [
            sample["variants"][after_name]["actual_on_disk_bytes"]
            - sample["variants"][before_name]["actual_on_disk_bytes"]
            for sample in samples
        ]
        effects[label] = {
            "after": after_name,
            "before": before_name,
            "mean_lpips_delta": mean(deltas),
            "better_count": sum(value < -1e-12 for value in deltas),
            "equal_count": sum(abs(value) <= 1e-12 for value in deltas),
            "mean_byte_delta": mean(byte_deltas),
            "sample_count": len(samples),
        }
    return {"targets_vs_baseline": per_variant, "factorial_effects": effects}


def markdown(result: dict) -> str:
    lines = [
        "# A800 单卡 v6 真实画质评估",
        "",
        "四个版本使用同一组 30 条 REDS + 7 条 UVG。所有新动作图都写入真实 spatial-QP 码流并 fresh decode；Generate 使用 16 像素羽化。LPIPS 越低越好。",
        "",
    ]
    for group, title in (
        ("combined", "合并 37 条"), ("REDS", "REDS 30 条"),
        ("UVG", "UVG 7 条"),
        ("UVG-adaptation-sequences", "UVG 适配侧 5 条"),
        ("UVG-v6-holdout-sequences", "UVG 未参与 v6 训练 2 条"),
    ):
        lines.extend([
            f"## {title}", "",
            "| 版本 | 平均字节 | LPIPS | PSNR | G 边总数 | G 连通块 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name in VARIANTS:
            value = result["aggregate"][group][name]
            lines.append(
                f"| {name} | {value['actual_on_disk_bytes_mean']:.1f} | "
                f"{value['quality_mean']['lpips_alex']:.6f} | "
                f"{value['quality_mean']['psnr_db']:.3f} | "
                f"{value['generate_boundary_edges_total']} | "
                f"{value['generate_components_total']} |")
        effect = result["comparisons"][group]["factorial_effects"]
        combined = effect["combined_vs_baseline"]
        lines.extend([
            "",
            f"- 组合版相对基线：LPIPS {combined['mean_lpips_delta']:+.6f}，{combined['better_count']}/{combined['sample_count']} 条更好，平均字节 {combined['mean_byte_delta']:+.1f}。",
            f"- 只适配：LPIPS {effect['adaptation_without_spatial']['mean_lpips_delta']:+.6f}；只加空间项：{effect['spatial_without_adaptation']['mean_lpips_delta']:+.6f}。",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    joint_formal_root = args.joint_formal_root.resolve()
    plan = read(args.plan.resolve())
    joint = read(args.joint_summary.resolve())
    feather = read(args.feather_summary.resolve())
    if plan["status"] != "frozen-before-quality-evaluation":
        raise RuntimeError("evaluation plan was not frozen")
    if plan["sample_count"] != 37 or joint["sample_count"] != 37:
        raise RuntimeError("sample counts differ")
    if feather["sample_count"] != 37 or feather["status"] != "complete":
        raise RuntimeError("frozen feather summary is incomplete")
    if not feather["verification"][
            "all_8px_recompositions_match_frozen_v5_pixel_exact"]:
        raise RuntimeError("frozen feather verification differs")

    joint_by_id = {sample["sample_id"]: sample for sample in joint["samples"]}
    feather_by_id = {
        sample["sample_id"]: sample for sample in feather["samples"]}
    plan_ids = {sample["sample_id"] for sample in plan["samples"]}
    if set(joint_by_id) != set(feather_by_id) or set(joint_by_id) != plan_ids:
        raise RuntimeError("plan, joint, and feather sample IDs differ")

    samples = []
    canonical_cache: dict[tuple[str, str], dict] = {}
    visuals_root = run_root / "formal" / "visuals"
    for plan_sample in plan["samples"]:
        sample_id = plan_sample["sample_id"]
        baseline = baseline_record(
            plan_sample, feather_by_id[sample_id], joint_by_id[sample_id])
        canonical_cache[(sample_id, "baseline-v5-feather16")] = baseline
        variants = {"baseline-v5-feather16": baseline}
        for name in TARGET_VARIANTS:
            canonical_name = plan_sample["variants"][name]["canonical_variant"]
            key = (sample_id, canonical_name)
            if key not in canonical_cache:
                canonical_cache[key] = target_record(
                    run_root=run_root,
                    joint_formal_root=joint_formal_root,
                    plan_sample=plan_sample,
                    canonical_variant=canonical_name,
                )
            variants[name] = logical_record(
                canonical_cache[key], plan_sample, name)

        gate = read(
            joint_formal_root / sample_id / "uniform_gate" / "summary.json")
        reference = load_source(gate, 17)
        visual_path = visuals_root / f"{sample_id}.png"
        save_comparison_visual(visual_path, reference, variants)
        samples.append({
            "sample_id": sample_id,
            "dataset": plan_sample["dataset"],
            "sequence": plan_sample["sequence"],
            "data_role": plan_sample["data_role"],
            "seed": plan_sample["seed"],
            "variants": variants,
            "fixed_visual": str(visual_path),
        })
        print(json.dumps({
            "stage": "v6-summary-sample",
            "sample_id": sample_id,
            "lpips": {
                name: variants[name]["quality"]["lpips_alex"]
                for name in VARIANTS
            },
        }, ensure_ascii=False), flush=True)

    groups = group_samples(samples)
    if [len(groups[name]) for name in (
        "combined", "REDS", "UVG", "UVG-adaptation-sequences",
        "UVG-v6-holdout-sequences",
    )] != [37, 30, 7, 5, 2]:
        raise RuntimeError("evaluation group counts differ")
    aggregate_value = {
        name: aggregate(members) for name, members in groups.items()
    }
    comparison_value = {
        name: comparisons(members, joint_by_id)
        for name, members in groups.items()
    }
    task_markers = sorted(
        run_root.glob("formal/evaluation/*/*/variant.complete"))
    if len(task_markers) != plan["new_real_evaluation_task_count"]:
        raise RuntimeError("new task completion marker count differs")
    target_records = [
        sample["variants"][name] for sample in samples for name in TARGET_VARIANTS
    ]
    result = {
        "experiment": "A800 v6 adaptation x spatial-consistency evaluation",
        "status": "complete",
        "sample_count": len(samples),
        "dataset_counts": dict(Counter(
            sample["dataset"] for sample in samples)),
        "logical_variants": list(VARIANTS),
        "primary_metric": "LPIPS Alex (lower is better)",
        "feather_pixels": 16,
        "plan": str(args.plan.resolve()),
        "source_joint_summary": str(args.joint_summary.resolve()),
        "source_feather_summary": str(args.feather_summary.resolve()),
        "aggregate": aggregate_value,
        "comparisons": comparison_value,
        "samples": samples,
        "verification": {
            "new_real_evaluation_tasks": len(task_markers),
            "logical_target_evaluations": len(target_records),
            "logical_target_reuse_count": sum(
                record["reused"] for record in target_records),
            "all_new_stream_sizes_rechecked_against_files": True,
            "all_new_spatial_fresh_decodes_pixel_exact": all(
                record["fresh_decode_pixel_exact"] for record in target_records),
            "fixed_visual_count": len(samples),
            "fixed_visuals_exist": all(
                Path(sample["fixed_visual"]).is_file() for sample in samples),
            "baseline_feather16_previously_verified": True,
            "exact_action_reuse_only": True,
        },
        "notion_visual_candidates": {
            sample["sequence"]: sample["fixed_visual"]
            for sample in samples
            if sample["sequence"] in {"Beauty", "Jockey", "YachtRide"}
        },
        "scientific_boundary": {
            "same_37_samples_are_v6_development_and_error_analysis": True,
            "no_hard_promotion_gate": True,
            "result_driven_sample_exclusion": False,
            "all_rate_values_are_real_final_file_bytes": True,
            "source_rgb_not_read_by_decoder": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
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
        "sample_count": len(samples),
        "combined_effects": comparison_value["combined"][
            "factorial_effects"],
        "uvg_effects": comparison_value["UVG"]["factorial_effects"],
        "verification": result["verification"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
