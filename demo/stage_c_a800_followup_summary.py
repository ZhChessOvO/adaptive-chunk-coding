#!/usr/bin/env python3
"""Aggregate the fixed six-sample A800 follow-up evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


METRICS = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
BASELINE_NAMES = (
    "scalar-qp8", "scalar-qp16", "scalar-qp24", "scalar-qp32",
    "all-generate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--routes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def mean(values: list[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(value)
    ]
    return sum(finite) / len(finite) if finite else None


def checked_record(value: dict) -> dict:
    byte_count = int(value["actual_on_disk_bytes"])
    if byte_count <= 0:
        raise RuntimeError("formal stream byte count must be positive")
    stream_path = Path(value["stream_path"])
    if not stream_path.is_file() or stream_path.stat().st_size != byte_count:
        raise RuntimeError(f"formal stream byte check failed: {stream_path}")
    quality = value["quality"]
    return {
        "quality": {name: quality.get(name) for name in METRICS},
        "actual_on_disk_bytes": byte_count,
        "runtime_seconds": float(value["runtime_seconds"]),
        "peak_cuda_allocated_bytes": int(
            value["peak_cuda_allocated_bytes"]),
        "stream_path": str(stream_path),
        "visual": value.get("visual"),
    }


def generic_variant(path: Path) -> dict:
    value = read(path)
    if not value["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"non-exact fresh decode in {path}")
    return checked_record({
        "quality": value["quality"],
        "actual_on_disk_bytes": value["stream"]["actual_on_disk_bytes"],
        "runtime_seconds": value["runtime"][
            "complete_fresh_decode_pipeline_process_seconds"],
        "peak_cuda_allocated_bytes": value["runtime"][
            "peak_cuda_allocated_bytes"],
        "stream_path": value["stream"]["path"],
        "visual": value["visual"],
    })


def roi_variant(root: Path, codec_root: Path | None = None) -> dict:
    value = read(root / "evaluation" / "summary.json")
    if not value["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"non-exact fresh decode in {root}")
    codec = codec_root or root / "codec"
    encode = read(codec / "encode_summary.json")
    return checked_record({
        "quality": value["quality"]["roi-spatial-bge-stitched"],
        "actual_on_disk_bytes": encode["stream_bytes"],
        "runtime_seconds": value["runtime"]["full_roi_pipeline_seconds"],
        "peak_cuda_allocated_bytes": value["runtime"][
            "peak_cuda_allocated_bytes"],
        "stream_path": encode["stream"],
        "visual": value["visual"],
    })


def aggregate(samples: list[dict]) -> dict:
    names = list(samples[0]["variants"])
    if any(set(sample["variants"]) != set(names) for sample in samples):
        raise RuntimeError("follow-up samples contain different variants")
    output = {}
    pixels = len(samples) * 17 * 512 * 512
    for name in names:
        rows = [sample["variants"][name] for sample in samples]
        total_bytes = sum(row["actual_on_disk_bytes"] for row in rows)
        output[name] = {
            "sample_count": len(rows),
            "quality_mean": {
                metric: mean([row["quality"][metric] for row in rows])
                for metric in METRICS
            },
            "actual_on_disk_bytes_total": total_bytes,
            "actual_on_disk_bytes_mean": total_bytes / len(rows),
            "aggregate_bpp": 8.0 * total_bytes / pixels,
            "runtime_seconds_mean": mean(
                [row["runtime_seconds"] for row in rows]),
            "peak_cuda_allocated_bytes_max": max(
                row["peak_cuda_allocated_bytes"] for row in rows),
        }
    return output


def load_sample(
    followup_root: Path, baseline_sample: dict, baseline_root: Path,
    route_path: Path, legacy_route_path: Path,
) -> dict:
    sample_id = baseline_sample["sample_id"]
    baseline_variants = baseline_sample["variants"]
    variants = {
        name: checked_record(baseline_variants[name])
        for name in BASELINE_NAMES
    }
    variants["middle-legacy"] = checked_record(
        baseline_variants["learned-joint"])
    variants["middle-persistent"] = roi_variant(
        followup_root / "spatial" / "middle-persistent",
        baseline_root / "spatial" / "learned-joint" / "codec")
    variants["low-joint"] = roi_variant(
        followup_root / "spatial" / "low-joint")
    variants["low-no-generate"] = generic_variant(
        followup_root / "spatial" / "low-no-generate" / "evaluation" /
        "evaluation.json")
    variants["low-no-enhance"] = roi_variant(
        followup_root / "spatial" / "low-no-enhance")

    regression = read(
        followup_root / "spatial" / "middle-persistent" /
        "exact_regression.json")
    if not regression.get("pixel_exact"):
        raise RuntimeError(f"middle-route output regression failed: {sample_id}")

    route = read(route_path)
    if route["selected_variant"] != "mlp-budget-0":
        raise RuntimeError(f"unexpected selected route for {sample_id}")
    selected = route["variants"][route["selected_variant"]]
    legacy_route = read(legacy_route_path)
    legacy_low = legacy_route["variants"]["mlp-budget-0"]
    if selected["actions"] != legacy_low["actions"]:
        raise RuntimeError(f"low route changed from preregistered route: {sample_id}")
    if selected["action_counts"].get("Generate") != 4:
        raise RuntimeError(f"low route Generate budget drifted: {sample_id}")

    visual = read(followup_root / "visuals" / "manifest.json")
    if not Path(visual["visual"]).is_file():
        raise RuntimeError(f"follow-up fixed visual is missing: {sample_id}")

    scalar_names = [name for name in variants if name.startswith("scalar-qp")]
    nearest_name = min(
        scalar_names,
        key=lambda name: abs(
            variants[name]["actual_on_disk_bytes"]
            - variants["low-joint"]["actual_on_disk_bytes"]),
    )
    return {
        "sample_id": sample_id,
        "source_role": baseline_sample["source_role"],
        "variants": variants,
        "route": {
            "selected_variant": route["selected_variant"],
            "action_counts": selected["action_counts"],
            "budget": selected["budget"],
            "actions_match_original_preregistered_budget_0": True,
        },
        "middle_executor_regression": regression,
        "nearest_scalar_to_low_joint": nearest_name,
        "fixed_visual": visual,
    }


def comparisons(samples: list[dict], aggregate_value: dict) -> dict:
    rows = []
    for sample in samples:
        variants = sample["variants"]
        low = variants["low-joint"]
        nearest_name = sample["nearest_scalar_to_low_joint"]
        nearest = variants[nearest_name]
        legacy = variants["middle-legacy"]
        persistent = variants["middle-persistent"]
        all_generate = variants["all-generate"]
        rows.append({
            "sample_id": sample["sample_id"],
            "nearest_scalar": nearest_name,
            "low_joint_bytes": low["actual_on_disk_bytes"],
            "nearest_scalar_bytes": nearest["actual_on_disk_bytes"],
            "low_joint_minus_nearest_scalar_lpips": (
                low["quality"]["lpips_alex"]
                - nearest["quality"]["lpips_alex"]),
            "low_joint_minus_all_generate_lpips": (
                low["quality"]["lpips_alex"]
                - all_generate["quality"]["lpips_alex"]),
            "low_no_generate_minus_joint_lpips": (
                variants["low-no-generate"]["quality"]["lpips_alex"]
                - low["quality"]["lpips_alex"]),
            "low_no_enhance_minus_joint_lpips": (
                variants["low-no-enhance"]["quality"]["lpips_alex"]
                - low["quality"]["lpips_alex"]),
            "middle_persistent_minus_legacy_lpips": (
                persistent["quality"]["lpips_alex"]
                - legacy["quality"]["lpips_alex"]),
            "middle_persistent_speedup_fraction_vs_legacy": (
                1.0 - persistent["runtime_seconds"] /
                legacy["runtime_seconds"]),
            "middle_persistent_speedup_fraction_vs_all_generate": (
                1.0 - persistent["runtime_seconds"] /
                all_generate["runtime_seconds"]),
        })
    delta_names = (
        "low_joint_minus_nearest_scalar_lpips",
        "low_joint_minus_all_generate_lpips",
        "low_no_generate_minus_joint_lpips",
        "low_no_enhance_minus_joint_lpips",
        "middle_persistent_minus_legacy_lpips",
        "middle_persistent_speedup_fraction_vs_legacy",
        "middle_persistent_speedup_fraction_vs_all_generate",
    )
    middle = aggregate_value["middle-persistent"]
    legacy = aggregate_value["middle-legacy"]
    all_generate = aggregate_value["all-generate"]
    return {
        "per_sample": rows,
        "sample_consistency_counts": {
            "low_joint_better_lpips_than_nearest_scalar": sum(
                row["low_joint_minus_nearest_scalar_lpips"] < 0
                for row in rows),
            "low_joint_better_lpips_than_all_generate": sum(
                row["low_joint_minus_all_generate_lpips"] < 0
                for row in rows),
            "generate_ablation_worse_lpips_than_joint": sum(
                row["low_no_generate_minus_joint_lpips"] > 0
                for row in rows),
            "enhance_ablation_worse_lpips_than_joint": sum(
                row["low_no_enhance_minus_joint_lpips"] > 0
                for row in rows),
            "middle_persistent_faster_than_legacy": sum(
                row["middle_persistent_speedup_fraction_vs_legacy"] > 0
                for row in rows),
            "middle_persistent_faster_than_all_generate": sum(
                row["middle_persistent_speedup_fraction_vs_all_generate"] > 0
                for row in rows),
            "sample_count": len(rows),
        },
        "mean_of_per_sample_deltas": {
            name: mean([row[name] for row in rows]) for name in delta_names
        },
        "aggregate_runtime": {
            "middle_persistent_seconds_mean": middle["runtime_seconds_mean"],
            "middle_legacy_seconds_mean": legacy["runtime_seconds_mean"],
            "all_generate_seconds_mean": all_generate["runtime_seconds_mean"],
            "middle_persistent_speedup_fraction_vs_legacy": (
                1.0 - middle["runtime_seconds_mean"] /
                legacy["runtime_seconds_mean"]),
            "middle_persistent_speedup_fraction_vs_all_generate": (
                1.0 - middle["runtime_seconds_mean"] /
                all_generate["runtime_seconds_mean"]),
        },
        "interpretation": (
            "LPIPS is lower-is-better. Positive no-branch-minus-joint deltas "
            "mean the removed branch helped. Positive runtime speedup means "
            "the persistent executor is faster."),
    }


def route_summary(samples: list[dict]) -> dict:
    totals = defaultdict(int)
    for sample in samples:
        for name, count in sample["route"]["action_counts"].items():
            totals[name] += int(count)
    return {
        "selected_variant": "mlp-budget-0",
        "action_counts_total": dict(totals),
        "all_routes_match_original_preregistered_budget_0": True,
        "per_sample": [
            {
                "sample_id": sample["sample_id"],
                "action_counts": sample["route"]["action_counts"],
            }
            for sample in samples
        ],
    }


def startup_cache_observation(run_root: Path) -> dict | None:
    smoke_root = run_root / "smoke" / "persistent_middle"
    cold_path = (
        smoke_root / "dev-s000-f00-x384-y096" / "roi_batch_metadata.json")
    warm_path = (
        smoke_root / "dev-s001-f00-x384-y096" / "roi_batch_metadata.json")
    if not cold_path.is_file() or not warm_path.is_file():
        return None
    cold = read(cold_path)
    warm = read(warm_path)
    return {
        "scope": (
            "two pre-formal regression examples; excluded from the formal "
            "six-sample mean"),
        "cold_after_server_restart": {
            "sample_id": "dev-s000-f00-x384-y096",
            "model_load_seconds": cold["model_load_seconds_this_process"],
            "roi_batch_seconds": cold["total_after_argument_parse_seconds"],
        },
        "warm_filesystem_cache": {
            "sample_id": "dev-s001-f00-x384-y096",
            "model_load_seconds": warm["model_load_seconds_this_process"],
            "roi_batch_seconds": warm["total_after_argument_parse_seconds"],
        },
        "interpretation": (
            "A true cold boot can be dominated by checkpoint I/O. Formal "
            "comparisons include process/model loading but use the warmed "
            "filesystem-cache regime shared by the existing baselines."),
    }


def markdown(result: dict) -> str:
    lines = [
        "# A800 单卡后续复验结果",
        "",
        "所有码率都按真实落盘字节统计；LPIPS 越低越好。运行时间包含 fresh decode、SeedVR2（如有）和拼接。",
        "",
        "| 方法 | 平均字节/17帧 | bpp | LPIPS | PSNR dB | T-MAE | 平均完整解码秒 | 峰值显存 GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    preferred = [
        "scalar-qp8", "scalar-qp16", "scalar-qp24", "scalar-qp32",
        "all-generate", "middle-legacy", "middle-persistent",
        "low-joint", "low-no-generate", "low-no-enhance",
    ]
    for name in preferred:
        value = result["aggregate"][name]
        quality = value["quality_mean"]
        lines.append(
            f"| {name} | {value['actual_on_disk_bytes_mean']:.1f} | "
            f"{value['aggregate_bpp']:.6f} | {quality['lpips_alex']:.6f} | "
            f"{quality['psnr_db']:.3f} | {quality['temporal_delta_mae']:.3f} | "
            f"{value['runtime_seconds_mean']:.3f} | "
            f"{value['peak_cuda_allocated_bytes_max'] / 2**30:.3f} |")
    delta = result["comparisons"]["mean_of_per_sample_deltas"]
    runtime = result["comparisons"]["aggregate_runtime"]
    consistency = result["comparisons"]["sample_consistency_counts"]
    counts = result["route"]["action_counts_total"]
    lines.extend([
        "",
        "## 关键结论数字",
        "",
        f"- 低预算路由共 96 个区域：Base {counts.get('Base', 0)}、Generate {counts.get('Generate', 0)}、Enhance {counts.get('Enhance', 0)}。",
        f"- 低预算联合路由相对逐样本最近均匀 QP 的 LPIPS 差值：{delta['low_joint_minus_nearest_scalar_lpips']:+.6f}。",
        f"- 上述最近均匀 QP 比较为 {consistency['low_joint_better_lpips_than_nearest_scalar']}/{consistency['sample_count']} 个样本更好；相对全 Generate 为 {consistency['low_joint_better_lpips_than_all_generate']}/{consistency['sample_count']} 个样本更好。",
        f"- 关闭 Generate 后减联合路由：{delta['low_no_generate_minus_joint_lpips']:+.6f}；关闭 Enhance 后减联合路由：{delta['low_no_enhance_minus_joint_lpips']:+.6f}。",
        f"- 中预算常驻执行与旧执行逐像素一致；平均完整时间从 {runtime['middle_legacy_seconds_mean']:.3f} 秒降到 {runtime['middle_persistent_seconds_mean']:.3f} 秒。",
        f"- 中预算常驻执行相对全画面 Generate 的平均时间加速比例：{runtime['middle_persistent_speedup_fraction_vs_all_generate']:+.2%}，逐样本为 {consistency['middle_persistent_faster_than_all_generate']}/{consistency['sample_count']} 更快。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = read(args.baseline_summary)
    baseline_by_id = {
        sample["sample_id"]: sample for sample in baseline["samples"]
    }
    followup_roots = sorted(
        path.parent.parent for path in args.formal_root.glob(
            "*/visuals/manifest.json"))
    if len(followup_roots) != 6:
        raise RuntimeError(
            f"expected six fixed development samples, found {len(followup_roots)}")
    sample_ids = [root.name for root in followup_roots]
    if set(sample_ids) != set(baseline_by_id):
        raise RuntimeError("follow-up sample identities differ from baseline")

    baseline_run_root = args.baseline_summary.parent.parent
    baseline_formal_root = baseline_run_root / "formal" / "development"
    legacy_routes_root = baseline_run_root / "pilot" / "routes" / "mlp"
    samples = [
        load_sample(
            followup_root=root,
            baseline_sample=baseline_by_id[root.name],
            baseline_root=baseline_formal_root / root.name,
            route_path=args.routes_root / f"{root.name}.json",
            legacy_route_path=legacy_routes_root / f"{root.name}.json",
        )
        for root in followup_roots
    ]
    aggregate_value = aggregate(samples)
    route_value = route_summary(samples)
    if route_value["action_counts_total"] != {
            "Base": 24, "Generate": 24, "Enhance": 48}:
        raise RuntimeError(
            f"preregistered aggregate route changed: {route_value}")
    comparison_value = comparisons(samples, aggregate_value)
    exact = all(
        sample["middle_executor_regression"]["pixel_exact"]
        for sample in samples)
    result = {
        "experiment": "A800 single-card low-budget and persistent-ROI follow-up",
        "status": "complete",
        "sample_count": len(samples),
        "data_role": "REDS val/000..005 development set; not an independent test",
        "baseline_summary": str(args.baseline_summary.resolve()),
        "primary_metric": "LPIPS Alex (lower is better)",
        "rate_accounting": "all actual on-disk stream bytes including headers and maps",
        "runtime_accounting": (
            "fresh codec decode plus one persistent SeedVR2 process per route "
            "when Generate is present, plus CPU compositing"),
        "verification": {
            "all_recorded_stream_sizes_rechecked_against_files": True,
            "all_new_spatial_fresh_decodes_pixel_exact": True,
            "middle_executor_outputs_pixel_exact_to_legacy": exact,
            "middle_executor_exact_sample_count": sum(
                sample["middle_executor_regression"]["pixel_exact"]
                for sample in samples),
            "fixed_visual_frame_one_based": 9,
            "fixed_visuals_created_for_all_six_samples": True,
        },
        "samples": samples,
        "aggregate": aggregate_value,
        "comparisons": comparison_value,
        "route": route_value,
        "startup_cache_observation": startup_cache_observation(
            args.output.parent.parent),
        "scientific_boundary": {
            "development_sequences": "REDS val/000..005 only",
            "independent_test": False,
            "validation_012_023_read": False,
            "sealed_validation_024_029_read": False,
            "new_teacher_labels_generated": False,
            "controller_retrained": False,
            "route_retuned_after_results": False,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "multi_gpu_used": False,
            "lpips_primary_psnr_diagnostic": True,
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
    # JSON is the orchestration marker and is published last.
    atomic_text(
        args.output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "summary": str(args.output),
        "markdown": str(args.output.with_suffix('.md')),
        "csv": str(csv_path),
        "sample_count": len(samples),
        "route_action_counts": route_value["action_counts_total"],
        "key_comparisons": comparison_value,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
