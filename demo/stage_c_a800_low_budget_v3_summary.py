#!/usr/bin/env python3
"""Aggregate the frozen six-sample development check for Base-probe v3."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from demo.stage_c_a800_followup_summary import (
    aggregate,
    atomic_text,
    checked_record,
    generic_variant,
    mean,
    roi_variant,
)


SCALAR_NAMES = ("scalar-qp8", "scalar-qp16", "scalar-qp24", "scalar-qp32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-followup-summary", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--routes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample(root: Path, previous: dict, route_path: Path) -> dict:
    sample_id = root.name
    variants = {
        name: checked_record(previous["variants"][name])
        for name in (*SCALAR_NAMES, "all-generate")
    }
    variants["low-v1"] = checked_record(previous["variants"]["low-joint"])
    variants["low-v3"] = roi_variant(root / "spatial" / "low-joint")
    variants["low-v3-no-generate"] = generic_variant(
        root / "spatial" / "low-no-generate" / "evaluation" /
        "evaluation.json")
    variants["low-v3-no-enhance"] = roi_variant(
        root / "spatial" / "low-no-enhance")
    route = read(route_path)
    selected_name = route["selected_variant"]
    if selected_name != "base-probe-direct-low-v3":
        raise RuntimeError(f"unexpected v3 route in {route_path}")
    selected = route["variants"][selected_name]
    boundary = route["scientific_boundary"]
    if boundary.get("ground_truth_or_teacher_targets_used_for_route") is not False:
        raise RuntimeError(f"v3 route boundary is incomplete in {route_path}")
    visual = read(root / "visuals" / "manifest.json")
    if not Path(visual["visual"]).is_file():
        raise RuntimeError(f"fixed visual is missing for {sample_id}")
    if not (root / "sample.complete").is_file():
        raise RuntimeError(f"sample completion marker is missing: {sample_id}")
    nearest_name = min(
        SCALAR_NAMES,
        key=lambda name: abs(
            variants[name]["actual_on_disk_bytes"]
            - variants["low-v3"]["actual_on_disk_bytes"]),
    )
    return {
        "sample_id": sample_id,
        "source_role": previous["source_role"],
        "variants": variants,
        "route": {
            "selected_variant": selected_name,
            "action_counts": selected["action_counts"],
            "budget": selected["budget"],
        },
        "base_probe": route["base_probe"],
        "nearest_scalar_to_low_v3": nearest_name,
        "fixed_visual": visual,
    }


def comparisons(samples: list[dict], aggregate_value: dict) -> dict:
    rows = []
    for sample in samples:
        variants = sample["variants"]
        v3 = variants["low-v3"]
        v1 = variants["low-v1"]
        nearest_name = sample["nearest_scalar_to_low_v3"]
        nearest = variants[nearest_name]
        rows.append({
            "sample_id": sample["sample_id"],
            "nearest_scalar": nearest_name,
            "low_v3_bytes": v3["actual_on_disk_bytes"],
            "low_v1_bytes": v1["actual_on_disk_bytes"],
            "nearest_scalar_bytes": nearest["actual_on_disk_bytes"],
            "low_v3_minus_v1_lpips": (
                v3["quality"]["lpips_alex"] - v1["quality"]["lpips_alex"]),
            "low_v3_minus_nearest_scalar_lpips": (
                v3["quality"]["lpips_alex"]
                - nearest["quality"]["lpips_alex"]),
            "no_generate_minus_v3_lpips": (
                variants["low-v3-no-generate"]["quality"]["lpips_alex"]
                - v3["quality"]["lpips_alex"]),
            "no_enhance_minus_v3_lpips": (
                variants["low-v3-no-enhance"]["quality"]["lpips_alex"]
                - v3["quality"]["lpips_alex"]),
            "v3_faster_than_all_generate": (
                v3["runtime_seconds"]
                < variants["all-generate"]["runtime_seconds"]),
        })
    delta_names = (
        "low_v3_minus_v1_lpips",
        "low_v3_minus_nearest_scalar_lpips",
        "no_generate_minus_v3_lpips",
        "no_enhance_minus_v3_lpips",
    )
    return {
        "per_sample": rows,
        "mean_of_per_sample_deltas": {
            name: mean([item[name] for item in rows]) for name in delta_names
        },
        "sample_consistency_counts": {
            "low_v3_better_than_v1": sum(
                item["low_v3_minus_v1_lpips"] < 0 for item in rows),
            "low_v3_better_than_nearest_scalar": sum(
                item["low_v3_minus_nearest_scalar_lpips"] < 0
                for item in rows),
            "generate_ablation_worse": sum(
                item["no_generate_minus_v3_lpips"] > 0 for item in rows),
            "enhance_ablation_worse": sum(
                item["no_enhance_minus_v3_lpips"] > 0 for item in rows),
            "low_v3_faster_than_all_generate": sum(
                item["v3_faster_than_all_generate"] for item in rows),
            "sample_count": len(rows),
        },
        "aggregate_runtime": {
            "low_v3_seconds_mean": aggregate_value["low-v3"][
                "runtime_seconds_mean"],
            "all_generate_seconds_mean": aggregate_value["all-generate"][
                "runtime_seconds_mean"],
        },
    }


def route_summary(samples: list[dict]) -> dict:
    totals = defaultdict(int)
    for sample in samples:
        for name, count in sample["route"]["action_counts"].items():
            totals[name] += int(count)
    return {
        "selected_variant": "base-probe-direct-low-v3",
        "action_counts_total": dict(totals),
        "per_sample": [
            {
                "sample_id": sample["sample_id"],
                "action_counts": sample["route"]["action_counts"],
            }
            for sample in samples
        ],
    }


def encoder_analysis_summary(samples: list[dict]) -> dict:
    fields = (
        "recorded_encode_seconds",
        "recorded_fresh_decode_seconds_median",
        "feature_and_controller_seconds",
        "estimated_encoder_analysis_seconds_excluding_shared_model_load",
        "temporary_stream_bytes_not_charged_to_transmitted_rate",
    )
    return {
        f"{name}_mean": mean([sample["base_probe"][name] for sample in samples])
        for name in fields
    }


def acceptance(
    aggregate_value: dict, comparison: dict, route_value: dict,
    sample_count: int,
) -> dict:
    v3 = aggregate_value["low-v3"]
    v1 = aggregate_value["low-v1"]
    deltas = comparison["mean_of_per_sample_deltas"]
    counts = comparison["sample_consistency_counts"]
    actions = route_value["action_counts_total"]
    criteria = {
        "mean_lpips_better_than_v1": (
            v3["quality_mean"]["lpips_alex"]
            < v1["quality_mean"]["lpips_alex"]),
        "mean_bytes_at_most_110_percent_of_v1": (
            v3["actual_on_disk_bytes_mean"]
            <= 1.10 * v1["actual_on_disk_bytes_mean"]),
        "mean_lpips_better_than_nearest_scalar": (
            deltas["low_v3_minus_nearest_scalar_lpips"] < 0),
        "at_least_four_of_six_better_than_nearest_scalar": (
            counts["low_v3_better_than_nearest_scalar"] >= 4),
        "all_three_actions_used": all(
            actions.get(name, 0) > 0
            for name in ("Base", "Generate", "Enhance")),
        "generate_ablation_mean_worse": (
            deltas["no_generate_minus_v3_lpips"] > 0),
        "generate_ablation_at_least_four_of_six_worse": (
            counts["generate_ablation_worse"] >= 4),
        "enhance_ablation_mean_worse": (
            deltas["no_enhance_minus_v3_lpips"] > 0),
        "enhance_ablation_at_least_four_of_six_worse": (
            counts["enhance_ablation_worse"] >= 4),
        "mean_runtime_below_all_generate": (
            v3["runtime_seconds_mean"]
            < aggregate_value["all-generate"]["runtime_seconds_mean"]),
        "all_six_samples_complete_and_fresh_decode_exact": sample_count == 6,
    }
    return {
        "overall_pass": all(criteria.values()),
        "criteria": criteria,
        "next_step": (
            "stop and request authorization for a new independent test"
            if all(criteria.values())
            else "stop and discuss the negative development result"),
    }


def markdown(result: dict) -> str:
    lines = [
        "# A800 单卡低预算 Base 预分析 v3 结果",
        "",
        "所有码率均为最终 spatial-QP 真实落盘字节；临时 Base 流不传输，其编码端时间单列。LPIPS 越低越好。",
        "",
        "| 方法 | 平均字节/17帧 | LPIPS | PSNR dB | T-MAE | 平均完整秒 | 峰值显存 GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "scalar-qp8", "scalar-qp16", "scalar-qp24", "scalar-qp32",
        "all-generate", "low-v1", "low-v3", "low-v3-no-generate",
        "low-v3-no-enhance",
    ):
        value = result["aggregate"][name]
        quality = value["quality_mean"]
        lines.append(
            f"| {name} | {value['actual_on_disk_bytes_mean']:.1f} | "
            f"{quality['lpips_alex']:.6f} | {quality['psnr_db']:.3f} | "
            f"{quality['temporal_delta_mae']:.3f} | "
            f"{value['runtime_seconds_mean']:.3f} | "
            f"{value['peak_cuda_allocated_bytes_max'] / 2**30:.3f} |")
    deltas = result["comparisons"]["mean_of_per_sample_deltas"]
    counts = result["comparisons"]["sample_consistency_counts"]
    actions = result["route"]["action_counts_total"]
    encoder = result["encoder_analysis"]
    lines.extend([
        "",
        "## 固定判据",
        "",
        f"- 总判定：{'通过' if result['acceptance']['overall_pass'] else '未通过'}。",
        f"- v3 相对 v1 平均 LPIPS 差：{deltas['low_v3_minus_v1_lpips']:+.6f}。",
        f"- v3 相对逐样本最近均匀 QP 平均 LPIPS 差：{deltas['low_v3_minus_nearest_scalar_lpips']:+.6f}，{counts['low_v3_better_than_nearest_scalar']}/6 条更好。",
        f"- 关闭 Generate／Enhance 后平均 LPIPS 差：{deltas['no_generate_minus_v3_lpips']:+.6f}／{deltas['no_enhance_minus_v3_lpips']:+.6f}。",
        f"- 96 个区域动作合计：Base {actions.get('Base', 0)}、Generate {actions.get('Generate', 0)}、Enhance {actions.get('Enhance', 0)}。",
        f"- Base 预分析编码端额外时间均值（不含共享 LPIPS 模型加载）：{encoder['estimated_encoder_analysis_seconds_excluding_shared_model_load_mean']:.3f} 秒。",
        "",
        "逐项判据：",
        "",
    ])
    for name, passed in result["acceptance"]["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    previous = read(args.v1_followup_summary)
    previous_by_id = {
        sample["sample_id"]: sample for sample in previous["samples"]
    }
    roots = sorted(
        path.parent.parent for path in args.formal_root.glob(
            "*/visuals/manifest.json"))
    if len(roots) != 6:
        raise RuntimeError(f"expected six development samples, found {len(roots)}")
    if {root.name for root in roots} != set(previous_by_id):
        raise RuntimeError("v3 development identities differ from v1")
    samples = [
        load_sample(
            root,
            previous_by_id[root.name],
            args.routes_root / f"{root.name}.json",
        )
        for root in roots
    ]
    aggregate_value = aggregate(samples)
    comparison_value = comparisons(samples, aggregate_value)
    route_value = route_summary(samples)
    encoder_value = encoder_analysis_summary(samples)
    selection = read(args.selection_summary)
    if not selection["gate"]["pass"]:
        raise RuntimeError("training-only v3 gate did not pass")
    acceptance_value = acceptance(
        aggregate_value, comparison_value, route_value, len(samples))
    result = {
        "experiment": "A800 single-card low-budget Base-probe controller v3",
        "status": "complete",
        "sample_count": len(samples),
        "data_role": "REDS val/000..005 development set; not independent test",
        "protocol": "docs/CLOUD_A800_LOW_BUDGET_V3.md",
        "selection_summary": str(args.selection_summary.resolve()),
        "v1_followup_summary": str(args.v1_followup_summary.resolve()),
        "primary_metric": "LPIPS Alex (lower is better)",
        "rate_accounting": (
            "final spatial-QP on-disk bytes; encoder-only Base probe is not "
            "transmitted"),
        "runtime_accounting": (
            "decoder runtime is fresh codec decode plus persistent SeedVR2 "
            "ROI and compositing; encoder Base-probe overhead is separate"),
        "verification": {
            "all_recorded_stream_sizes_rechecked_against_files": True,
            "all_new_spatial_fresh_decodes_pixel_exact": True,
            "sample_complete_markers": 6,
            "fixed_visual_frame_one_based": 9,
            "fixed_visuals_created_for_all_six_samples": True,
        },
        "training_only_gate": selection["gate"],
        "samples": samples,
        "aggregate": aggregate_value,
        "comparisons": comparison_value,
        "route": route_value,
        "encoder_analysis": encoder_value,
        "acceptance": acceptance_value,
        "scientific_boundary": {
            "development_sequences": "REDS val/000..005 only",
            "development_run_once_after_frozen_training_selection": True,
            "base_probe_uses_encoder_visible_source_and_reconstruction": True,
            "temporary_base_stream_transmitted": False,
            "validation_012_023_used_for_selection_or_tuning": False,
            "sealed_validation_024_029_read": False,
            "new_teacher_labels_generated": False,
            "controller_retrained_on_training_labels_only": True,
            "route_retuned_after_development_results": False,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "multi_gpu_used": False,
        },
    }
    atomic_text(args.output.with_suffix(".md"), markdown(result))
    rows = []
    for name, value in result["aggregate"].items():
        rows.append({
            "variant": name,
            "mean_actual_bytes": value["actual_on_disk_bytes_mean"],
            "aggregate_bpp": value["aggregate_bpp"],
            **value["quality_mean"],
            "runtime_seconds_mean": value["runtime_seconds_mean"],
            "peak_cuda_allocated_mib": (
                value["peak_cuda_allocated_bytes_max"] / 2**20),
        })
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    atomic_text(
        args.output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "summary": str(args.output),
        "acceptance": acceptance_value,
        "route_action_counts": route_value["action_counts_total"],
        "encoder_analysis": encoder_value,
        "key_comparisons": comparison_value,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
