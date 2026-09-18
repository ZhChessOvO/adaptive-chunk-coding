#!/usr/bin/env python3
"""Aggregate the preregistered 12-sample A800 independent test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


METRICS = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
SCALAR_NAMES = tuple(f"scalar-qp{qp}" for qp in (8, 16, 24, 32))
BASIC_NAMES = tuple(f"basicvsrpp-qp{qp}" for qp in (8, 16, 24, 32))
BUDGETS = ("low", "middle")
EXPECTED_SEQUENCES = tuple(f"{index:03d}" for index in range(12, 24))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--routes-root", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
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


def assert_frames_exact(first_dir: Path, second_dir: Path) -> None:
    first = sorted(first_dir.glob("*.png"))
    second = sorted(second_dir.glob("*.png"))
    if len(first) != 17 or len(second) != 17:
        raise RuntimeError(
            f"fresh-decode comparison needs 17 frames: {first_dir}, {second_dir}")
    maximum = 0
    for first_path, second_path in zip(first, second):
        first_array = np.asarray(
            Image.open(first_path).convert("RGB"), dtype=np.int16)
        second_array = np.asarray(
            Image.open(second_path).convert("RGB"), dtype=np.int16)
        maximum = max(
            maximum, int(np.max(np.abs(first_array - second_array))))
    if maximum != 0:
        raise RuntimeError(
            f"isolated scalar decode differs by {maximum}: "
            f"{first_dir}, {second_dir}")


def checked_record(
    *, quality: dict, byte_count: int, runtime_seconds: float,
    peak_bytes: int, stream_path: str, visual: str | None,
) -> dict:
    path = Path(stream_path)
    if byte_count <= 0 or not path.is_file() or path.stat().st_size != byte_count:
        raise RuntimeError(f"formal stream byte check failed: {path}")
    return {
        "quality": {name: quality.get(name) for name in METRICS},
        "actual_on_disk_bytes": int(byte_count),
        "runtime_seconds": float(runtime_seconds),
        "peak_cuda_allocated_bytes": int(peak_bytes),
        "stream_path": str(path.resolve()),
        "visual": visual,
    }


def generic_variant(path: Path) -> dict:
    value = read(path)
    if not value["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"non-exact fresh decode in {path}")
    return checked_record(
        quality=value["quality"],
        byte_count=int(value["stream"]["actual_on_disk_bytes"]),
        runtime_seconds=float(value["runtime"][
            "complete_fresh_decode_pipeline_process_seconds"]),
        peak_bytes=int(value["runtime"]["peak_cuda_allocated_bytes"]),
        stream_path=value["stream"]["path"],
        visual=value.get("visual"),
    )


def roi_variant(root: Path) -> dict:
    value = read(root / "evaluation" / "summary.json")
    if not value["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"non-exact fresh decode in {root}")
    encode = read(root / "codec" / "encode_summary.json")
    return checked_record(
        quality=value["quality"]["roi-spatial-bge-stitched"],
        byte_count=int(encode["stream_bytes"]),
        runtime_seconds=float(value["runtime"]["full_roi_pipeline_seconds"]),
        peak_bytes=int(value["runtime"]["peak_cuda_allocated_bytes"]),
        stream_path=encode["stream"],
        visual=value.get("visual"),
    )


def scalar_variants(root: Path) -> dict[str, dict]:
    gate_root = root / "uniform_gate"
    gate = read(gate_root / "summary.json")
    output = {}
    for qp in (8, 16, 24, 32):
        for suffix, label in (("base", "scalar"),
                              ("basicvsrpp", "basicvsrpp")):
            source = gate["variants"][f"qp{qp}-{suffix}"]
            benchmark = read(
                gate_root / "isolated_fresh_decode" /
                f"{label}-qp{qp}.json")
            assert_frames_exact(
                gate_root / "frames" / f"qp{qp}-{suffix}",
                Path(benchmark["fresh_decode_frames"]))
            output[f"{label}-qp{qp}"] = checked_record(
                quality=source["quality"],
                byte_count=int(source["rate"]["total_bytes"]),
                runtime_seconds=float(benchmark[
                    "complete_process_seconds_after_argument_parse"]),
                peak_bytes=int(benchmark["peak_cuda_allocated_bytes"]),
                stream_path=source["path"],
                visual=gate.get("visual"),
            )
    return output


def verify_route(
    controller_route: dict, route_path: Path, source_variant: str,
) -> dict:
    route = read(route_path)
    if route["provenance"]["source_selected_variant"] != source_variant:
        raise RuntimeError(f"route provenance changed: {route_path}")
    selected = route["variants"][route["selected_variant"]]
    expected = controller_route["variants"][source_variant]
    if selected["actions"] != expected["actions"]:
        raise RuntimeError(f"route actions changed: {route_path}")
    return {
        "source_variant": source_variant,
        "action_counts": selected["action_counts"],
        "actions_match_frozen_controller_output": True,
    }


def load_sample(
    root: Path, routes_root: Path, expected_record: dict,
) -> dict:
    if not (root / "sample.complete").is_file():
        raise RuntimeError(f"sample marker is missing: {root}")
    sample_id = expected_record["sample_id"]
    if root.name != sample_id:
        raise RuntimeError(f"sample identity differs: {root.name}, {sample_id}")
    gate = read(root / "uniform_gate" / "summary.json")
    if "one-shot independent test" not in gate.get("source_role", ""):
        raise RuntimeError(f"wrong data role in {root}")

    variants = scalar_variants(root)
    spatial = root / "spatial"
    variants["all-generate"] = generic_variant(
        spatial / "all-generate" / "evaluation" / "evaluation.json")
    variants["enhance-only"] = generic_variant(
        spatial / "enhance-only" / "evaluation" / "evaluation.json")
    for budget in BUDGETS:
        variants[f"{budget}-joint"] = roi_variant(
            spatial / f"{budget}-joint")
        variants[f"{budget}-no-generate"] = generic_variant(
            spatial / f"{budget}-no-generate" / "evaluation" /
            "evaluation.json")
        variants[f"{budget}-no-enhance"] = roi_variant(
            spatial / f"{budget}-no-enhance")

    controller_route = read(routes_root / "controller" / f"{sample_id}.json")
    route_records = {
        "low": verify_route(
            controller_route,
            root / "routes" / "low" / "learned-joint.json",
            "mlp-budget-0"),
        "middle": verify_route(
            controller_route,
            root / "routes" / "middle" / "learned-joint.json",
            "mlp-budget-1"),
    }

    visual = read(root / "visuals" / "manifest.json")
    if not Path(visual["visual"]).is_file():
        raise RuntimeError(f"fixed visual is missing: {sample_id}")

    nearest = {}
    for budget in BUDGETS:
        joint = variants[f"{budget}-joint"]
        nearest[budget] = min(
            SCALAR_NAMES,
            key=lambda name: abs(
                variants[name]["actual_on_disk_bytes"]
                - joint["actual_on_disk_bytes"]),
        )
    return {
        "sample_id": sample_id,
        "sequence": expected_record["sequence"],
        "source_role": expected_record["source_role"],
        "variants": variants,
        "routes": route_records,
        "nearest_scalar": nearest,
        "fixed_visual": visual,
    }


def aggregate(samples: list[dict]) -> dict:
    names = list(samples[0]["variants"])
    if any(set(sample["variants"]) != set(names) for sample in samples):
        raise RuntimeError("independent-test samples contain different variants")
    pixels = len(samples) * 17 * 512 * 512
    output = {}
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


def compare_budget(
    samples: list[dict], aggregate_value: dict, budget: str,
) -> dict:
    rows = []
    for sample in samples:
        variants = sample["variants"]
        joint = variants[f"{budget}-joint"]
        nearest_name = sample["nearest_scalar"][budget]
        nearest = variants[nearest_name]
        all_generate = variants["all-generate"]
        rows.append({
            "sample_id": sample["sample_id"],
            "nearest_scalar": nearest_name,
            "joint_bytes": joint["actual_on_disk_bytes"],
            "nearest_scalar_bytes": nearest["actual_on_disk_bytes"],
            "joint_minus_nearest_scalar_lpips": (
                joint["quality"]["lpips_alex"]
                - nearest["quality"]["lpips_alex"]),
            "joint_minus_all_generate_lpips": (
                joint["quality"]["lpips_alex"]
                - all_generate["quality"]["lpips_alex"]),
            "joint_minus_all_generate_bytes": (
                joint["actual_on_disk_bytes"]
                - all_generate["actual_on_disk_bytes"]),
            "no_generate_minus_joint_lpips": (
                variants[f"{budget}-no-generate"]["quality"]["lpips_alex"]
                - joint["quality"]["lpips_alex"]),
            "no_enhance_minus_joint_lpips": (
                variants[f"{budget}-no-enhance"]["quality"]["lpips_alex"]
                - joint["quality"]["lpips_alex"]),
            "runtime_speedup_fraction_vs_all_generate": (
                1.0 - joint["runtime_seconds"]
                / all_generate["runtime_seconds"]),
        })

    delta_names = (
        "joint_minus_nearest_scalar_lpips",
        "joint_minus_all_generate_lpips",
        "joint_minus_all_generate_bytes",
        "no_generate_minus_joint_lpips",
        "no_enhance_minus_joint_lpips",
        "runtime_speedup_fraction_vs_all_generate",
    )
    delta = {
        name: mean([row[name] for row in rows]) for name in delta_names
    }
    counts = {
        "joint_better_lpips_than_nearest_scalar": sum(
            row["joint_minus_nearest_scalar_lpips"] < 0 for row in rows),
        "joint_better_lpips_than_all_generate": sum(
            row["joint_minus_all_generate_lpips"] < 0 for row in rows),
        "generate_ablation_worse_lpips_than_joint": sum(
            row["no_generate_minus_joint_lpips"] > 0 for row in rows),
        "enhance_ablation_worse_lpips_than_joint": sum(
            row["no_enhance_minus_joint_lpips"] > 0 for row in rows),
        "joint_faster_than_all_generate": sum(
            row["runtime_speedup_fraction_vs_all_generate"] > 0
            for row in rows),
        "sample_count": len(rows),
    }
    joint_runtime = aggregate_value[f"{budget}-joint"][
        "runtime_seconds_mean"]
    all_generate_runtime = aggregate_value["all-generate"][
        "runtime_seconds_mean"]
    runtime_speedup = 1.0 - joint_runtime / all_generate_runtime
    criteria = {
        "quality_vs_nearest_scalar": (
            delta["joint_minus_nearest_scalar_lpips"] < 0
            and counts["joint_better_lpips_than_nearest_scalar"] >= 8),
        "generate_branch_contribution": (
            delta["no_generate_minus_joint_lpips"] > 0
            and counts["generate_ablation_worse_lpips_than_joint"] >= 8),
        "enhance_branch_contribution": (
            delta["no_enhance_minus_joint_lpips"] > 0
            and counts["enhance_ablation_worse_lpips_than_joint"] >= 8),
        "runtime_vs_all_generate": (
            runtime_speedup > 0
            and counts["joint_faster_than_all_generate"] >= 8),
    }
    return {
        "per_sample": rows,
        "mean_of_per_sample_deltas": delta,
        "sample_consistency_counts": counts,
        "aggregate_runtime": {
            "joint_seconds_mean": joint_runtime,
            "all_generate_seconds_mean": all_generate_runtime,
            "joint_speedup_fraction_vs_all_generate": runtime_speedup,
        },
        "preregistered_criteria": criteria,
        "supports_development_conclusion": all(criteria.values()),
        "interpretation": (
            "LPIPS is lower-is-better. Positive ablation-minus-joint values "
            "mean the removed branch helped. Nearest scalar QP is a discrete "
            "measured point, not matched-rate interpolation."),
    }


def route_summary(samples: list[dict]) -> dict:
    output = {}
    for budget, source_variant in (("low", "mlp-budget-0"),
                                   ("middle", "mlp-budget-1")):
        totals = defaultdict(int)
        for sample in samples:
            route = sample["routes"][budget]
            if route["source_variant"] != source_variant:
                raise RuntimeError(f"unexpected {budget} source variant")
            for name, count in route["action_counts"].items():
                totals[name] += int(count)
        output[budget] = {
            "source_variant": source_variant,
            "action_counts_total": dict(totals),
            "per_sample": [
                {
                    "sample_id": sample["sample_id"],
                    "action_counts": sample["routes"][budget]["action_counts"],
                }
                for sample in samples
            ],
        }
    return output


def markdown(result: dict) -> str:
    lines = [
        "# A800 单卡一次性独立测试结果",
        "",
        "测试集为此前未读取的 REDS val/012..023。所有码率均为真实落盘字节；LPIPS 越低越好。",
        "",
        "| 方法 | 平均字节/17帧 | bpp | LPIPS | PSNR dB | T-MAE | 平均完整秒 | 峰值显存 GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    preferred = [
        *SCALAR_NAMES, *BASIC_NAMES, "all-generate", "enhance-only",
        "low-joint", "low-no-generate", "low-no-enhance",
        "middle-joint", "middle-no-generate", "middle-no-enhance",
    ]
    for name in preferred:
        value = result["aggregate"][name]
        quality = value["quality_mean"]
        lines.append(
            f"| {name} | {value['actual_on_disk_bytes_mean']:.1f} | "
            f"{value['aggregate_bpp']:.6f} | {quality['lpips_alex']:.6f} | "
            f"{quality['psnr_db']:.3f} | "
            f"{quality['temporal_delta_mae']:.3f} | "
            f"{value['runtime_seconds_mean']:.3f} | "
            f"{value['peak_cuda_allocated_bytes_max'] / 2**30:.3f} |")

    for budget, title in (("low", "低预算"), ("middle", "中预算")):
        comparison = result["comparisons"][budget]
        delta = comparison["mean_of_per_sample_deltas"]
        counts = comparison["sample_consistency_counts"]
        runtime = comparison["aggregate_runtime"]
        actions = result["route"][budget]["action_counts_total"]
        criteria = comparison["preregistered_criteria"]
        lines.extend([
            "",
            f"## {title}",
            "",
            f"- 动作总数：Base {actions.get('Base', 0)}、Generate {actions.get('Generate', 0)}、Enhance {actions.get('Enhance', 0)}。",
            f"- 相对逐样本最近均匀 QP 的 LPIPS 差值为 {delta['joint_minus_nearest_scalar_lpips']:+.6f}，{counts['joint_better_lpips_than_nearest_scalar']}/12 条更好。",
            f"- 相对全 Generate 的 LPIPS 差值为 {delta['joint_minus_all_generate_lpips']:+.6f}，{counts['joint_better_lpips_than_all_generate']}/12 条更好；这不是同码率比较。",
            f"- 关闭 Generate 后减联合路由为 {delta['no_generate_minus_joint_lpips']:+.6f}，{counts['generate_ablation_worse_lpips_than_joint']}/12 条变差。",
            f"- 关闭 Enhance 后减联合路由为 {delta['no_enhance_minus_joint_lpips']:+.6f}，{counts['enhance_ablation_worse_lpips_than_joint']}/12 条变差。",
            f"- 完整时间相对全 Generate 的加速为 {runtime['joint_speedup_fraction_vs_all_generate']:+.2%}，{counts['joint_faster_than_all_generate']}/12 条更快。",
            f"- 预注册四项判据：质量 {criteria['quality_vs_nearest_scalar']}，Generate 贡献 {criteria['generate_branch_contribution']}，Enhance 贡献 {criteria['enhance_branch_contribution']}，速度 {criteria['runtime_vs_all_generate']}。",
            f"- 总结：支持开发结论 = {comparison['supports_development_conclusion']}。",
        ])
    lines.extend([
        "",
        "## 边界",
        "",
        "本轮没有新 teacher、训练、调参或模型微调。val/012..023 已被本次一次性测试消费，不能再作为独立测试；val/024..029 仍未读取。",
        "",
    ])
    return "\n".join(lines)


def write_csv(path: Path, result: dict) -> None:
    rows = []
    for name, value in result["aggregate"].items():
        quality = value["quality_mean"]
        rows.append({
            "variant": name,
            "samples": value["sample_count"],
            "mean_bytes": value["actual_on_disk_bytes_mean"],
            "aggregate_bpp": value["aggregate_bpp"],
            "lpips_alex": quality["lpips_alex"],
            "psnr_db": quality["psnr_db"],
            "temporal_delta_mae": quality["temporal_delta_mae"],
            "rgb_mse": quality["rgb_mse"],
            "runtime_seconds": value["runtime_seconds_mean"],
            "peak_cuda_allocated_bytes": value[
                "peak_cuda_allocated_bytes_max"],
        })
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    expected = [
        json.loads(line)
        for line in args.sample_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(expected) != 12:
        raise RuntimeError(f"expected 12 test records, found {len(expected)}")
    if tuple(record["sequence"] for record in expected) != EXPECTED_SEQUENCES:
        raise RuntimeError("test sequences differ from preregistration")
    roots = [args.formal_root / record["sample_id"] for record in expected]
    samples = [
        load_sample(root, args.routes_root, record)
        for root, record in zip(roots, expected)
    ]
    aggregate_value = aggregate(samples)
    route_value = route_summary(samples)
    comparisons = {
        budget: compare_budget(samples, aggregate_value, budget)
        for budget in BUDGETS
    }
    result = {
        "experiment": "A800 single-card one-shot independent test",
        "status": "complete",
        "sample_count": len(samples),
        "data_role": (
            "REDS val/012..023 one-shot independent test; consumed after "
            "this evaluation"),
        "controller_checkpoint": str(args.controller_checkpoint.resolve()),
        "primary_metric": "LPIPS Alex (lower is better)",
        "rate_accounting": (
            "all actual on-disk stream bytes including headers and maps"),
        "runtime_accounting": (
            "fresh codec decode plus a separate SeedVR2 process per route "
            "when Generate is present, plus CPU compositing"),
        "verification": {
            "all_recorded_stream_sizes_rechecked_against_files": True,
            "all_scalar_and_spatial_fresh_decodes_pixel_exact": True,
            "fixed_visual_frame_one_based": 9,
            "fixed_visuals_created_for_all_twelve_samples": True,
        },
        "samples": samples,
        "aggregate": aggregate_value,
        "comparisons": comparisons,
        "route": route_value,
        "preregistered_overall": {
            "low_supports_development_conclusion": comparisons["low"][
                "supports_development_conclusion"],
            "middle_supports_development_conclusion": comparisons["middle"][
                "supports_development_conclusion"],
        },
        "scientific_boundary": {
            "test_sequences": "REDS val/012..023 only",
            "independent_test_before_this_run": True,
            "test_set_consumed_after_this_run": True,
            "sealed_validation_024_029_read": False,
            "new_teacher_labels_generated": False,
            "controller_retrained": False,
            "route_retuned_after_results": False,
            "sample_or_crop_excluded_after_results": False,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "multi_gpu_used": False,
        },
    }
    atomic_text(
        args.output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    atomic_text(args.output.with_suffix(".md"), markdown(result))
    write_csv(args.output.with_suffix(".csv"), result)
    print(json.dumps({
        "summary": str(args.output.resolve()),
        "sample_count": len(samples),
        "preregistered_overall": result["preregistered_overall"],
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
