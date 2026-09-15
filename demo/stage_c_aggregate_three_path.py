#!/usr/bin/env python3
"""Aggregate E19 three-path summaries without re-reading source frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="E19 aggregate")
    return parser.parse_args()


def aggregate_variant(records: list[dict], pixel_count: int) -> dict:
    total_bytes = sum(item["rate"]["total_bytes"] for item in records)
    mean_mse = sum(item["quality"]["rgb_mse"] for item in records) / len(records)
    psnr = 10.0 * math.log10(255.0 ** 2 / mean_mse)
    lpips_values = [
        item["quality"]["lpips_alex"] for item in records
        if item["quality"]["lpips_alex"] is not None
    ]
    action_counts = {
        key: sum(item["action_counts"][key] for item in records)
        for key in ("Base", "Generate", "Enhance")
    }
    return {
        "total_bytes": total_bytes,
        "bits_per_pixel": total_bytes * 8 / pixel_count,
        "rgb_mse": mean_mse,
        "psnr_db": psnr,
        "lpips_alex": (
            sum(lpips_values) / len(lpips_values) if lpips_values else None),
        "temporal_delta_mae": sum(
            item["quality"]["temporal_delta_mae"] for item in records
        ) / len(records),
        "fresh_decode_seconds_sum": sum(
            item["runtime"]["fresh_decode_seconds_median"] for item in records),
        "peak_cuda_allocated_bytes_max": max(
            item["runtime"]["peak_cuda_allocated_bytes"] for item in records),
        "action_counts": action_counts,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if not summaries:
        raise ValueError("at least one summary is required")
    first = summaries[0]
    variant_names = list(first["variants"])
    if any(list(item["variants"]) != variant_names for item in summaries):
        raise ValueError("variant sets or ordering differ")
    signature = (
        first["frames"], first["crop"]["width"], first["crop"]["height"],
        first["configuration"]["base_qp"], first["configuration"]["enhance_qp"],
        first["configuration"]["tile_size"],
        round(first["configuration"]["enhance_budget_ratio_of_base_stream"], 3),
        first["configuration"]["restorer"],
    )
    for item in summaries[1:]:
        current = (
            item["frames"], item["crop"]["width"], item["crop"]["height"],
            item["configuration"]["base_qp"], item["configuration"]["enhance_qp"],
            item["configuration"]["tile_size"],
            round(item["configuration"]["enhance_budget_ratio_of_base_stream"], 3),
            item["configuration"]["restorer"],
        )
        if current != signature:
            raise ValueError("protocol signatures differ")

    frame_count, width, height = signature[:3]
    pixel_count = len(summaries) * frame_count * width * height
    variants = {
        name: aggregate_variant(
            [item["variants"][name] for item in summaries], pixel_count)
        for name in variant_names
    }
    base = variants["base-only"]
    joint = variants["joint-three-path-oracle"]
    qp40 = variants.get("ordinary-dcvc-qp40")
    conclusion = (
        f"联合 Oracle 相对 Base 为 {joint['psnr_db'] - base['psnr_db']:+.5f} dB，"
        f"LPIPS 变化 {joint['lpips_alex'] - base['lpips_alex']:+.5f}。"
    )
    if qp40 is not None:
        conclusion += (
            f"相对普通 QP40，联合方案字节变化 "
            f"{joint['total_bytes'] - qp40['total_bytes']:+d}，PSNR 变化 "
            f"{joint['psnr_db'] - qp40['psnr_db']:+.5f} dB。")

    result = {
        "experiment": args.label,
        "sequence_count": len(summaries),
        "sequences": [item["sequence"] for item in summaries],
        "frames_per_sequence": frame_count,
        "crop": first["crop"],
        "configuration": first["configuration"],
        "aggregation": {
            "bytes": "sum of actual on-disk files",
            "psnr": "convert equal-pixel-count mean RGB MSE to dB",
            "lpips_and_temporal": "equal-frame-count mean",
            "fresh_decode_time": "sum of per-sequence medians",
            "peak_memory": "maximum across sequences",
        },
        "variants": variants,
        "conclusion": conclusion,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [{"variant": name, **record} for name, record in variants.items()]
    for row in rows:
        row["action_counts"] = json.dumps(row["action_counts"], ensure_ascii=False)
    write_csv(args.output_dir / "per_variant.csv", rows)

    lines = [
        f"# {args.label}", "", conclusion, "",
        "| 方案 | 总字节 | bpp | PSNR | LPIPS | 时序差分误差 | 解码秒（六段和） | 峰值显存 MiB | B/G/E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, record in variants.items():
        counts = record["action_counts"]
        lines.append(
            f"| {name} | {record['total_bytes']} | {record['bits_per_pixel']:.6f} | "
            f"{record['psnr_db']:.5f} | {record['lpips_alex']:.6f} | "
            f"{record['temporal_delta_mae']:.5f} | "
            f"{record['fresh_decode_seconds_sum']:.5f} | "
            f"{record['peak_cuda_allocated_bytes_max'] / 1048576:.1f} | "
            f"{counts['Base']}/{counts['Generate']}/{counts['Enhance']} |")
    lines += [
        "", "聚合只读取各序列结果 JSON，不再读取源图。路由仍是编码端 Oracle，不能作为控制器泛化结果。",
    ]
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(args.output_dir / "summary.json"),
        "conclusion": conclusion,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
