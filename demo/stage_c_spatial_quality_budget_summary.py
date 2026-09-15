#!/usr/bin/env python3
"""Aggregate E24's exact spatial-QP budget curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--labels", nargs=3, default=("0.25x", "0.50x", "1.00x"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    scalar = json.loads(
        Path("output/e21_seedvr2_three_path_jockey_b16_e32_t128_b050/summary.json")
        .read_text(encoding="utf-8"))["ordinary_scalar_qp_baselines"]
    scalar_points = sorted(
        ((value["rate"]["total_bytes"], value["quality"]["lpips_alex"],
          value["quality"]["temporal_delta_mae"], name)
         for name, value in scalar.items()),
        key=lambda item: item[0],
    )

    spatial_bytes = [record["stream"]["actual_on_disk_bytes"] for record in records]
    stitched_lpips = [record["variants"]["spatial-bge-stitched"]["quality"]["lpips_alex"]
                       for record in records]
    raw_lpips = [record["variants"]["spatial-qp-raw"]["quality"]["lpips_alex"]
                 for record in records]
    all_generate_lpips = [
        record["variants"]["all-generate-spatial-input"]["quality"]["lpips_alex"]
        for record in records]
    old_bytes = [record["rate_comparison"]["old_tile_fallback_bytes"]
                 for record in records]
    old_lpips = [record["rate_comparison"]["old_tile_fallback_lpips"]
                 for record in records]
    stitched_temporal = [
        record["variants"]["spatial-bge-stitched"]["quality"]["temporal_delta_mae"]
        for record in records]
    all_generate_temporal = [
        record["variants"]["all-generate-spatial-input"]["quality"]["temporal_delta_mae"]
        for record in records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    scalar_x = [point[0] for point in scalar_points]
    scalar_lpips = [point[1] for point in scalar_points]
    scalar_temporal = [point[2] for point in scalar_points]

    axes[0].plot(scalar_x, scalar_lpips, "o-", label="Scalar-QP DCVC-UF")
    axes[0].plot(old_bytes, old_lpips, "s--", label="E21 independent-tile B/G/E")
    axes[0].plot(spatial_bytes, raw_lpips, "d--", label="E24 spatial-QP raw")
    axes[0].plot(spatial_bytes, stitched_lpips, "o-", linewidth=2.5,
                 label="E24 spatial-QP B/G/E")
    axes[0].plot(spatial_bytes, all_generate_lpips, "*:", markersize=13,
                 label="SeedVR2 everywhere, same spatial input")
    for x, y, label in zip(spatial_bytes, stitched_lpips, args.labels):
        axes[0].annotate(label, (x, y), xytext=(6, 6), textcoords="offset points")
    axes[0].set_title("Perceptual rate-quality (LPIPS primary)")
    axes[0].set_xlabel("Actual on-disk bytes, 17 x 512 x 512")
    axes[0].set_ylabel("LPIPS Alex (lower is better)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].plot(scalar_x, scalar_temporal, "o-", label="Scalar-QP DCVC-UF")
    axes[1].plot(spatial_bytes, stitched_temporal, "o-", linewidth=2.5,
                 label="E24 spatial-QP B/G/E")
    axes[1].plot(spatial_bytes, all_generate_temporal, "*:", markersize=13,
                 label="SeedVR2 everywhere")
    for x, y, label in zip(spatial_bytes, stitched_temporal, args.labels):
        axes[1].annotate(label, (x, y), xytext=(6, 6), textcoords="offset points")
    axes[1].set_title("Temporal diagnostic")
    axes[1].set_xlabel("Actual on-disk bytes, 17 x 512 x 512")
    axes[1].set_ylabel("Temporal delta MAE (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    figure.tight_layout()
    visual = visual_dir / "e24_exact_spatial_budget_curve.png"
    figure.savefig(visual, dpi=170)
    plt.close(figure)

    result = {
        "experiment": "E24 exact one-shot spatial-QP budget summary",
        "labels": list(args.labels),
        "points": [
            {
                "label": label,
                "stream_bytes": byte_count,
                "spatial_raw_lpips": raw,
                "spatial_bge_lpips": stitched,
                "all_generate_lpips": all_generate,
                "spatial_bge_temporal_delta_mae": temporal,
                "all_generate_temporal_delta_mae": generated_temporal,
                "fresh_decode_pixel_exact": record["fresh_decode_regression"]["pixel_exact"],
            }
            for label, byte_count, raw, stitched, all_generate, temporal,
            generated_temporal, record in zip(
                args.labels, spatial_bytes, raw_lpips, stitched_lpips,
                all_generate_lpips, stitched_temporal, all_generate_temporal, records)
        ],
        "visual": str(visual),
        "psnr_role": "report only; see individual E24 summaries",
        "all_stream_bytes_are_actual_on_disk": True,
    }
    summary = args.output_dir / "summary.json"
    summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
