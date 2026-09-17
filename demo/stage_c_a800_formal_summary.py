#!/usr/bin/env python3
"""Aggregate the six fixed A800 development clips into one formal result."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--controller-summary", type=Path, required=True)
    parser.add_argument("--routes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        maximum = max(maximum, int(np.max(np.abs(first_array - second_array))))
    if maximum != 0:
        raise RuntimeError(
            f"isolated scalar decode differs by {maximum}: "
            f"{first_dir}, {second_dir}")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def record(
    *, quality: dict, byte_count: int, runtime_seconds: float | None,
    peak_bytes: int | None, stream_path: str | None, visual: str | None,
) -> dict:
    if byte_count <= 0:
        raise ValueError("formal byte count must be positive")
    if stream_path is not None:
        path = Path(stream_path)
        if not path.is_file() or path.stat().st_size != byte_count:
            raise RuntimeError(f"formal byte verification failed: {path}")
    return {
        "quality": {name: quality.get(name) for name in METRICS},
        "actual_on_disk_bytes": int(byte_count),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_bytes": peak_bytes,
        "stream_path": stream_path,
        "visual": visual,
    }


def generic_variant(path: Path) -> dict:
    value = read(path)
    if not value["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"non-exact fresh decode in {path}")
    return record(
        quality=value["quality"],
        byte_count=int(value["stream"]["actual_on_disk_bytes"]),
        runtime_seconds=float(value["runtime"][
            "complete_fresh_decode_pipeline_process_seconds"]),
        peak_bytes=int(value["runtime"]["peak_cuda_allocated_bytes"]),
        stream_path=value["stream"]["path"],
        visual=value["visual"],
    )


def roi_variant(root: Path) -> dict:
    value = read(root / "evaluation" / "summary.json")
    if not value["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"non-exact fresh decode in {root}")
    encode = read(root / "codec" / "encode_summary.json")
    stream_path = encode["stream"]
    return record(
        quality=value["quality"]["roi-spatial-bge-stitched"],
        byte_count=int(encode["stream_bytes"]),
        runtime_seconds=float(value["runtime"]["full_roi_pipeline_seconds"]),
        peak_bytes=int(value["runtime"]["peak_cuda_allocated_bytes"]),
        stream_path=stream_path,
        visual=value["visual"],
    )


def load_sample(root: Path) -> dict:
    gate = read(root / "uniform_gate" / "summary.json")
    variants = {}
    for qp in gate["configuration"]["qps"]:
        for suffix, label in (("base", "scalar"), ("basicvsrpp", "basicvsrpp")):
            source = gate["variants"][f"qp{qp}-{suffix}"]
            benchmark = read(
                root / "uniform_gate" / "isolated_fresh_decode" /
                f"{label}-qp{qp}.json")
            expected_frames = (
                root / "uniform_gate" / "frames" /
                f"qp{qp}-{'base' if suffix == 'base' else 'basicvsrpp'}")
            assert_frames_exact(
                expected_frames, Path(benchmark["fresh_decode_frames"]))
            variants[f"{label}-qp{qp}"] = record(
                quality=source["quality"],
                byte_count=int(source["rate"]["total_bytes"]),
                runtime_seconds=float(benchmark[
                    "complete_process_seconds_after_argument_parse"]),
                peak_bytes=int(benchmark["peak_cuda_allocated_bytes"]),
                stream_path=source["path"],
                visual=gate.get("visual"),
            )

    seed = read(root / "ordinary_qp8_seed" / "evaluation" / "evaluation.json")
    seed_metadata = seed["seedvr2_metadata"]
    seed_quality = seed["variants"]["QP8 SeedVR2"]["quality"]
    qp8_decode = read(
        root / "uniform_gate" / "isolated_fresh_decode" / "scalar-qp8.json")
    variants["ordinary-qp8-seedvr2"] = record(
        quality=seed_quality,
        byte_count=int(seed["rate"]["dcvc_uf_file_bytes"]),
        runtime_seconds=(
            float(qp8_decode["complete_process_seconds_after_argument_parse"])
            + float(seed_metadata["total_after_argument_parse_seconds"])),
        peak_bytes=max(
            int(qp8_decode["peak_cuda_allocated_bytes"]),
            int(seed_metadata["peak_cuda_allocated_bytes"])),
        stream_path=gate["variants"]["qp8-base"]["path"],
        visual=str(root / "ordinary_qp8_seed" / "evaluation" / "visuals" /
                   "frame_00009.png"),
    )
    spatial = root / "spatial"
    variants["enhance-only"] = generic_variant(
        spatial / "enhance-only" / "evaluation" / "evaluation.json")
    variants["all-generate"] = generic_variant(
        spatial / "all-generate" / "evaluation" / "evaluation.json")
    variants["learned-joint"] = roi_variant(spatial / "learned-joint")
    variants["same-route-no-generate"] = generic_variant(
        spatial / "same-route-no-generate" / "evaluation" / "evaluation.json")
    variants["same-route-no-enhance"] = roi_variant(
        spatial / "same-route-no-enhance")
    fixed_visual = read(root / "visuals" / "manifest.json")
    if not Path(fixed_visual["visual"]).is_file():
        raise RuntimeError(f"fixed visual is missing for {root.name}")
    return {
        "sample_id": root.name,
        "source_role": gate["source_role"],
        "variants": variants,
        "fixed_visual": fixed_visual,
    }


def aggregate(samples: list[dict]) -> dict:
    names = list(samples[0]["variants"])
    if any(set(sample["variants"]) != set(names) for sample in samples):
        raise RuntimeError("formal samples contain different variant sets")
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
            "runtime_seconds_mean": mean([row["runtime_seconds"] for row in rows]),
            "peak_cuda_allocated_bytes_max": max(
                row["peak_cuda_allocated_bytes"] or 0 for row in rows),
        }
    return output


def delta_summary(samples: list[dict]) -> dict:
    rows = []
    for sample in samples:
        variants = sample["variants"]
        joint = variants["learned-joint"]
        scalar_names = [name for name in variants if name.startswith("scalar-qp")]
        nearest_name = min(
            scalar_names,
            key=lambda name: abs(
                variants[name]["actual_on_disk_bytes"]
                - joint["actual_on_disk_bytes"]),
        )
        nearest = variants[nearest_name]
        rows.append({
            "sample_id": sample["sample_id"],
            "nearest_scalar": nearest_name,
            "joint_bytes": joint["actual_on_disk_bytes"],
            "nearest_scalar_bytes": nearest["actual_on_disk_bytes"],
            "joint_minus_nearest_scalar_lpips": (
                joint["quality"]["lpips_alex"] - nearest["quality"]["lpips_alex"]),
            "joint_minus_all_generate_lpips": (
                joint["quality"]["lpips_alex"]
                - variants["all-generate"]["quality"]["lpips_alex"]),
            "no_generate_minus_joint_lpips": (
                variants["same-route-no-generate"]["quality"]["lpips_alex"]
                - joint["quality"]["lpips_alex"]),
            "no_enhance_minus_joint_lpips": (
                variants["same-route-no-enhance"]["quality"]["lpips_alex"]
                - joint["quality"]["lpips_alex"]),
        })
    return {
        "per_sample": rows,
        "mean": {
            key: mean([row[key] for row in rows])
            for key in (
                "joint_minus_nearest_scalar_lpips",
                "joint_minus_all_generate_lpips",
                "no_generate_minus_joint_lpips",
                "no_enhance_minus_joint_lpips",
            )
        },
        "interpretation": (
            "For LPIPS deltas, negative is better for the first two comparisons; "
            "positive is evidence that the removed branch helped for the ablations."
        ),
    }


def route_summary(routes_root: Path, samples: list[dict]) -> dict:
    action_totals = defaultdict(int)
    changed = []
    per_sample = []
    for sample in samples:
        route = read(routes_root / "mlp" / f"{sample['sample_id']}.json")
        selected = route["variants"][route["selected_variant"]]
        for name, count in selected["action_counts"].items():
            action_totals[name] += int(count)
        changed.append(int(route["regions_changing_action_across_budgets"]))
        per_sample.append({
            "sample_id": sample["sample_id"],
            "selected_variant": route["selected_variant"],
            "selected_action_counts": selected["action_counts"],
            "regions_changing_action_across_budgets": changed[-1],
        })
    return {
        "selected_action_counts_total": dict(action_totals),
        "regions_changing_action_across_budgets_total": sum(changed),
        "regions_changing_action_across_budgets_mean": mean(changed),
        "per_sample": per_sample,
    }


def markdown(result: dict) -> str:
    lines = [
        "# A800 单卡试跑正式结果",
        "",
        "下面所有码率均为真实落盘字节；LPIPS 越低越好，PSNR 仅作诊断。",
        "",
        "| 方法 | 平均字节/17帧 | bpp | LPIPS | PSNR dB | T-MAE | 平均完整解码秒 | 峰值显存 GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    preferred = [
        "scalar-qp8", "scalar-qp16", "scalar-qp24", "scalar-qp32",
        "basicvsrpp-qp8", "basicvsrpp-qp16", "basicvsrpp-qp24",
        "basicvsrpp-qp32", "ordinary-qp8-seedvr2", "enhance-only",
        "all-generate", "learned-joint", "same-route-no-generate",
        "same-route-no-enhance",
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
    delta = result["comparisons"]["mean"]
    lines.extend([
        "",
        "## 关键差值",
        "",
        f"- 联合策略相对逐样本最近均匀 QP 的 LPIPS 差值：{delta['joint_minus_nearest_scalar_lpips']:+.6f}。",
        f"- 联合策略相对全 Generate 的 LPIPS 差值：{delta['joint_minus_all_generate_lpips']:+.6f}。",
        f"- 关闭 Generate 后减联合策略：{delta['no_generate_minus_joint_lpips']:+.6f}。",
        f"- 关闭 Enhance 后减联合策略：{delta['no_enhance_minus_joint_lpips']:+.6f}。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    roots = sorted(path.parent.parent for path in args.formal_root.glob(
        "*/uniform_gate/summary.json"))
    if len(roots) != 6:
        raise RuntimeError(f"expected six fixed development samples, found {len(roots)}")
    samples = [load_sample(root) for root in roots]
    controller = read(args.controller_summary)
    result = {
        "experiment": "A800 single-card pilot formal development evaluation",
        "status": "complete",
        "sample_count": len(samples),
        "data_role": "REDS val/000..005 development set; not an independent test",
        "primary_metric": "LPIPS Alex (lower is better)",
        "rate_accounting": "all actual on-disk stream bytes including headers and maps",
        "verification": {
            "all_recorded_stream_sizes_rechecked_against_files": True,
            "all_spatial_fresh_decodes_pixel_exact": True,
            "fixed_visual_frame_one_based": 9,
            "fixed_visuals_created_for_all_six_samples": True,
        },
        "samples": samples,
        "aggregate": aggregate(samples),
        "comparisons": delta_summary(samples),
        "controller": {
            "training": controller["training"],
            "parameter_count": controller["parameter_count"],
            "train_metrics": controller["train_metrics"],
            "development_metrics": controller["development_metrics"],
            "route_budget_response": route_summary(args.routes_root, samples),
        },
        "scientific_boundary": {
            "development_sequences": "REDS val/000..005 only",
            "independent_test": False,
            "validation_012_023_read": False,
            "sealed_validation_024_029_read": False,
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
            "peak_cuda_allocated_mib": value["peak_cuda_allocated_bytes_max"] / 2**20,
        })
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    # The JSON is the orchestration marker, so publish it only after the
    # human-readable and tabular companions are safely on disk.
    atomic_text(args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "summary": str(args.output),
        "markdown": str(args.output.with_suffix('.md')),
        "csv": str(csv_path),
        "sample_count": len(samples),
        "key_deltas": result["comparisons"]["mean"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
