#!/usr/bin/env python3
"""Summarize the 8-sample actual ROI-cost calibration and continuation gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-cost-dir", type=Path, required=True)
    parser.add_argument("--quality-calibration-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=500)
    return parser.parse_args()


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def disk(path: Path) -> dict:
    value = shutil.disk_usage(path)
    return {
        "total_bytes": value.total,
        "used_bytes": value.used,
        "free_bytes": value.free,
        "used_percent": 100.0 * value.used / value.total,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.roi_cost_dir / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads(
        args.quality_calibration_summary.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest["completed_sample_count"] < 8:
        raise ValueError("ROI cost calibration needs at least 8 complete samples")
    samples = [
        json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        for entry in manifest["entries"]
    ]
    sample_seconds = [item["runtime"]["sample_total_seconds"] for item in samples]
    names = list(manifest["measurement_classes"])
    geometry = {
        name: stats([
            item["measurements"][name]["seconds_model_load_excluded"]
            for item in samples
        ])
        for name in names
    }
    median = statistics.median(sample_seconds)
    projected_seconds = median * args.target_count
    quality_seconds = quality["projected"]["wall_seconds_for_target_at_median"]
    current_disk = {
        "system": disk(Path("/root")),
        "fast": disk(Path("/root/autodl-tmp")),
        "file_store": disk(Path("/root/autodl-fs")),
    }
    result = {
        "experiment": "A800 actual ROI-cost throughput calibration",
        "completed_samples": len(samples),
        "target_samples": args.target_count,
        "sample_seconds": stats(sample_seconds),
        "geometry_seconds": geometry,
        "projected": {
            "roi_cost_wall_seconds_for_target_at_median": projected_seconds,
            "roi_cost_wall_hours_for_target_at_median": projected_seconds / 3600,
            "quality_teacher_wall_seconds_for_target_at_median": quality_seconds,
            "combined_teacher_wall_hours_at_medians": (
                projected_seconds + quality_seconds) / 3600,
            "cost_json_bytes_for_target": int(round(
                sum(Path(entry["path"]).stat().st_size
                    for entry in manifest["entries"])
                / len(manifest["entries"]) * args.target_count)),
        },
        "disk": current_disk,
        "gates": {
            "at_least_8_samples": len(samples) >= 8,
            "combined_teacher_projection_within_12_hours": (
                projected_seconds + quality_seconds <= 12 * 3600),
            "all_disks_below_80_percent": max(
                item["used_percent"] for item in current_disk.values()) < 80,
        },
        "measurement_boundary": (
            "Four actual sample-specific SeedVR2 geometry classes are measured; "
            "regions with identical ROI geometry share that measurement. Final "
            "connected routes are remeasured end to end rather than area-prorated."
        ),
    }
    result["gates"]["continue_to_500"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
