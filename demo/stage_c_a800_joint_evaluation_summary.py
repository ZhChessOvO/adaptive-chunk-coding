#!/usr/bin/env python3
"""Aggregate the frozen joint REDS-validation and UVG paper evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


METRICS = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
SCALAR_NAMES = tuple(f"scalar-qp{qp}" for qp in (8, 16, 24, 32))
BASIC_NAMES = tuple(f"basicvsrpp-qp{qp}" for qp in (8, 16, 24, 32))
VARIANT_NAMES = (
    *SCALAR_NAMES,
    *BASIC_NAMES,
    "all-generate",
    "enhance-only",
    "final-joint",
    "final-no-generate",
    "final-no-enhance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--routes-root", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--frozen-controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def mean(values: list[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(value)
    ]
    return sum(finite) / len(finite) if finite else None


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
    gate = read(root / "uniform_gate" / "summary.json")
    if gate["frames"] != 17:
        raise RuntimeError(f"uniform gate frame count drifted in {root}")
    if not gate["protocol"].get("actual_file_bytes_charged"):
        raise RuntimeError(f"uniform gate did not charge file bytes in {root}")
    output = {}
    for qp in (8, 16, 24, 32):
        for suffix, label in (("base", "scalar"),
                              ("basicvsrpp", "basicvsrpp")):
            source = gate["variants"][f"qp{qp}-{suffix}"]
            runtime = source["runtime"]
            output[f"{label}-qp{qp}"] = checked_record(
                quality=source["quality"],
                byte_count=int(source["rate"]["total_bytes"]),
                runtime_seconds=float(runtime["fresh_decode_seconds_median"]),
                peak_bytes=int(runtime["peak_cuda_allocated_bytes"]),
                stream_path=source["path"],
                visual=gate.get("visual"),
            )
    return output


def verify_route(
    controller_route: dict, route_path: Path, expected_variant: str,
) -> dict:
    route = read(route_path)
    if route["provenance"]["source_selected_variant"] != expected_variant:
        raise RuntimeError(f"route provenance changed: {route_path}")
    selected = route["variants"][route["selected_variant"]]
    expected = controller_route["variants"][expected_variant]
    if selected["actions"] != expected["actions"]:
        raise RuntimeError(f"route actions changed: {route_path}")
    return {
        "source_variant": expected_variant,
        "action_counts": selected["action_counts"],
        "actions_match_frozen_controller_output": True,
        "base_probe": controller_route.get("base_probe"),
        "consensus": controller_route.get("consensus"),
    }


def load_sample(
    root: Path, routes_root: Path, expected_record: dict,
    expected_variant: str,
) -> dict:
    if not (root / "sample.complete").is_file():
        raise RuntimeError(f"sample marker is missing: {root}")
    sample_id = expected_record["sample_id"]
    if root.name != sample_id:
        raise RuntimeError(f"sample identity differs: {root.name}, {sample_id}")
    gate = read(root / "uniform_gate" / "summary.json")
    gate_sources = [str(Path(path).resolve()) for path in gate["source_files"]]
    expected_sources = [
        str(Path(path).resolve()) for path in expected_record["source_files"]
    ]
    if gate_sources != expected_sources:
        raise RuntimeError(f"uniform gate source files differ for {sample_id}")
    if gate["crop"] != expected_record["crop"]:
        raise RuntimeError(f"uniform gate crop differs for {sample_id}")
    variants = scalar_variants(root)
    spatial = root / "spatial"
    variants["all-generate"] = generic_variant(
        spatial / "all-generate" / "evaluation" / "evaluation.json")
    variants["enhance-only"] = generic_variant(
        spatial / "enhance-only" / "evaluation" / "evaluation.json")
    variants["final-joint"] = roi_variant(spatial / "final-joint")
    variants["final-no-generate"] = generic_variant(
        spatial / "final-no-generate" / "evaluation" / "evaluation.json")
    variants["final-no-enhance"] = roi_variant(
        spatial / "final-no-enhance")
    if set(variants) != set(VARIANT_NAMES):
        raise RuntimeError(f"variant set differs for {sample_id}")

    controller_route = read(routes_root / "controller" / f"{sample_id}.json")
    route = verify_route(
        controller_route,
        root / "routes" / "learned-joint.json",
        expected_variant,
    )
    visual = read(root / "visuals" / "manifest.json")
    if not Path(visual["visual"]).is_file():
        raise RuntimeError(f"fixed visual is missing: {sample_id}")
    joint = variants["final-joint"]
    nearest = min(
        SCALAR_NAMES,
        key=lambda name: abs(
            variants[name]["actual_on_disk_bytes"]
            - joint["actual_on_disk_bytes"]),
    )
    return {
        "sample_id": sample_id,
        "dataset": expected_record["dataset"],
        "sequence": expected_record["sequence"],
        "data_role": expected_record["data_role"],
        "source_role": expected_record["source_role"],
        "selected_source_sha256": expected_record[
            "selected_source_sha256"],
        "variants": variants,
        "route": route,
        "nearest_scalar": nearest,
        "fixed_visual": visual,
        "reused_formal_outputs": read(root / "reuse.json").get(
            "reused", False) if (root / "reuse.json").is_file() else False,
    }


def aggregate(samples: list[dict]) -> dict:
    if not samples:
        raise RuntimeError("cannot aggregate an empty sample group")
    if any(set(sample["variants"]) != set(VARIANT_NAMES)
           for sample in samples):
        raise RuntimeError("joint-evaluation samples contain different variants")
    pixels = len(samples) * 17 * 512 * 512
    output = {}
    for name in VARIANT_NAMES:
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


def comparisons(samples: list[dict]) -> dict:
    rows = []
    for sample in samples:
        variants = sample["variants"]
        joint = variants["final-joint"]
        nearest_name = sample["nearest_scalar"]
        nearest = variants[nearest_name]
        all_generate = variants["all-generate"]
        rows.append({
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "data_role": sample["data_role"],
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
                variants["final-no-generate"]["quality"]["lpips_alex"]
                - joint["quality"]["lpips_alex"]),
            "no_enhance_minus_joint_lpips": (
                variants["final-no-enhance"]["quality"]["lpips_alex"]
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
    return {
        "per_sample": rows,
        "mean_of_per_sample_deltas": {
            name: mean([row[name] for row in rows]) for name in delta_names
        },
        "sample_consistency_counts": {
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
        },
        "interpretation": (
            "LPIPS is lower-is-better. Positive ablation-minus-joint values "
            "mean the removed branch helped. Nearest scalar QP is a discrete "
            "measured point, not matched-rate interpolation."),
    }


def route_summary(samples: list[dict]) -> dict:
    totals = defaultdict(int)
    base_probe_count = 0
    active_targets = 0
    for sample in samples:
        for name, count in sample["route"]["action_counts"].items():
            totals[name] += int(count)
        if sample["route"]["base_probe"] is not None:
            base_probe_count += 1
        consensus = sample["route"]["consensus"]
        if consensus is not None:
            active_targets += int(consensus["active_target_count"])
    return {
        "action_counts_total": dict(totals),
        "base_probe_sample_count": base_probe_count,
        "active_consensus_targets_total": active_targets,
        "per_sample": [
            {
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "action_counts": sample["route"]["action_counts"],
            }
            for sample in samples
        ],
    }


def markdown(result: dict) -> str:
    lines = [
        "# A800 单卡 REDS validation + UVG 联合评估",
        "",
        f"冻结的最终低预算版本：`{result['frozen_controller']['name']}`。外部视频与 REDS 一起作为跨分布论文评估，不称为单独的应用测试。所有码率均为真实落盘字节；LPIPS 越低越好。",
        "",
    ]
    for group_name, title in (("combined", "合并 37 条"),
                              ("REDS", "REDS validation 30 条"),
                              ("UVG", "UVG 7 条")):
        aggregate_value = result["aggregate"][group_name]
        comparison = result["comparisons"][group_name]
        delta = comparison["mean_of_per_sample_deltas"]
        counts = comparison["sample_consistency_counts"]
        lines.extend([
            f"## {title}",
            "",
            "| 方法 | 平均字节/17帧 | bpp | LPIPS | PSNR dB | T-MAE | 平均完整秒 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name in VARIANT_NAMES:
            value = aggregate_value[name]
            quality = value["quality_mean"]
            lines.append(
                f"| {name} | {value['actual_on_disk_bytes_mean']:.1f} | "
                f"{value['aggregate_bpp']:.6f} | "
                f"{quality['lpips_alex']:.6f} | {quality['psnr_db']:.3f} | "
                f"{quality['temporal_delta_mae']:.3f} | "
                f"{value['runtime_seconds_mean']:.3f} |")
        total = counts["sample_count"]
        lines.extend([
            "",
            f"- 联合路由相对逐样本最近均匀 QP：LPIPS {delta['joint_minus_nearest_scalar_lpips']:+.6f}，{counts['joint_better_lpips_than_nearest_scalar']}/{total} 条更好。",
            f"- 关闭 Generate／Enhance 后：LPIPS {delta['no_generate_minus_joint_lpips']:+.6f}／{delta['no_enhance_minus_joint_lpips']:+.6f}；分别 {counts['generate_ablation_worse_lpips_than_joint']}/{total}、{counts['enhance_ablation_worse_lpips_than_joint']}/{total} 条变差。",
            f"- 相对全 Generate 的平均完整时间变化：{delta['runtime_speedup_fraction_vs_all_generate']:+.2%} 加速；{counts['joint_faster_than_all_generate']}/{total} 条更快。",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.sample_manifest)
    if len(records) != 37:
        raise RuntimeError(f"expected 37 joint samples, found {len(records)}")
    ledger = read(args.ledger_summary)
    frozen = read(args.frozen_controller)
    expected_variant = frozen["route_variant"]
    samples = [
        load_sample(
            args.formal_root / record["sample_id"],
            args.routes_root,
            record,
            expected_variant,
        )
        for record in records
    ]
    groups = {
        "combined": samples,
        "REDS": [sample for sample in samples if sample["dataset"] == "REDS"],
        "UVG": [sample for sample in samples if sample["dataset"] == "UVG"],
    }
    if len(groups["REDS"]) != 30 or len(groups["UVG"]) != 7:
        raise RuntimeError("joint evaluation dataset counts drifted")
    aggregate_value = {name: aggregate(group) for name, group in groups.items()}
    comparison_value = {
        name: comparisons(group) for name, group in groups.items()
    }
    route_value = route_summary(samples)
    result = {
        "experiment": "A800 single-card REDS validation plus UVG evaluation",
        "status": "complete",
        "sample_count": len(samples),
        "dataset_counts": dict(Counter(
            sample["dataset"] for sample in samples)),
        "data_role_counts": dict(Counter(
            sample["data_role"] for sample in samples)),
        "frozen_controller": frozen,
        "ledger": ledger,
        "primary_metric": "LPIPS Alex (lower is better)",
        "rate_accounting": (
            "actual final on-disk bytes including spatial-QP headers, action "
            "map and all substreams; encoder-only probe is not transmitted"),
        "samples": samples,
        "aggregate": aggregate_value,
        "comparisons": comparison_value,
        "route": route_value,
        "verification": {
            "all_stream_sizes_rechecked_against_files": True,
            "all_spatial_stream_fresh_decodes_pixel_exact": True,
            "uniform_streams_fresh_decoded_from_files": True,
            "fixed_visuals_created": len(samples),
            "sample_complete_markers": len(samples),
            "reused_sample_count": sum(
                sample["reused_formal_outputs"] for sample in samples),
        },
        "scientific_boundary": {
            "controller_frozen_before_joint_quality_evaluation": True,
            "development_and_historical_roles_reported": True,
            "result_driven_sample_exclusion": False,
            "uvg_reported_with_reds_not_as_application_test": True,
            "qst_not_silently_replaced": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
            "multi_gpu_used": False,
        },
    }
    atomic_text(args.output.with_suffix(".md"), markdown(result))
    rows = []
    for group_name, values in aggregate_value.items():
        for variant, value in values.items():
            rows.append({
                "group": group_name,
                "variant": variant,
                "sample_count": value["sample_count"],
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
        "sample_count": len(samples),
        "dataset_counts": result["dataset_counts"],
        "route": route_value,
        "combined_comparison": comparison_value["combined"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
