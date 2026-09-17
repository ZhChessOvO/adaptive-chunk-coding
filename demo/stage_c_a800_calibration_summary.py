#!/usr/bin/env python3
"""Summarize the 8--16 sample A800 teacher throughput calibration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


TARGETS = (
    "generate_lpips_gain_vs_base",
    "enhance_lpips_gain_vs_base",
    "generate_temporal_risk_vs_base",
    "enhance_fallback_extra_on_disk_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    return parser.parse_args()


def statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def disk(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percent": 100.0 * usage.used / usage.total,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    manifest_path = args.teacher_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if not 8 <= len(entries) <= 16:
        raise ValueError("calibration must contain 8..16 completed samples")
    sample_values = [
        json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        for entry in entries
    ]
    target_values = {name: [] for name in TARGETS}
    winner_counts = {"Generate": 0, "Base": 0, "Enhance": 0}
    for sample in sample_values:
        for region in sample["regions"]:
            for name in TARGETS:
                target_values[name].append(float(region["targets"][name]))
            winner = min(
                ("Generate", "Base", "Enhance"),
                key=lambda name: region["candidates"][name]["lpips_alex"])
            winner_counts[winner] += 1

    sample_seconds = [entry["seconds"] for entry in entries]
    seed_seconds = [entry["seedvr2_seconds"] for entry in entries]
    calibration_bytes = tree_bytes(args.teacher_dir)
    label_bytes = sum(
        Path(entry["path"]).stat().st_size for entry in entries)
    per_label_bytes = label_bytes / len(entries)
    projected_seconds = (
        manifest.get("model_load_seconds", 0.0)
        + statistics(sample_seconds)["p50"] * args.target_count
    )
    disks = {
        "system": disk(Path("/root")),
        "fast": disk(Path("/root/autodl-tmp")),
        "file_store": disk(Path("/root/autodl-fs")),
    }
    label_statistics = {
        name: statistics(values) for name, values in target_values.items()
    }
    enough_variance = (
        label_statistics["generate_lpips_gain_vs_base"]["std"] > 1e-4
        and label_statistics["enhance_lpips_gain_vs_base"]["std"] > 1e-4
        and len([count for count in winner_counts.values() if count > 0]) >= 2
    )
    under_time_limit = projected_seconds <= args.max_wall_hours * 3600
    under_disk_limit = max(
        value["used_percent"] for value in disks.values()) < 80.0
    result = {
        "experiment": "A800 teacher throughput calibration",
        "completed_samples": len(entries),
        "target_samples": args.target_count,
        "sample_seconds": statistics(sample_seconds),
        "seedvr2_seconds": statistics(seed_seconds),
        "label_statistics": label_statistics,
        "best_lpips_action_counts": winner_counts,
        "calibration_output_bytes": calibration_bytes,
        "sample_json_bytes": label_bytes,
        "mean_sample_json_bytes": per_label_bytes,
        "projected": {
            "wall_seconds_for_target_at_median": projected_seconds,
            "wall_hours_for_target_at_median": projected_seconds / 3600,
            "compact_label_bytes_for_target": int(per_label_bytes * args.target_count),
            "all_candidate_frames_retained": False,
            "streams_retained": False,
            "fixed_visuals_only": True,
        },
        "resumability": {
            "one_atomic_json_per_sample": True,
            "completed_samples_are_skipped": True,
            "manifest_can_be_rebuilt_from_sample_json": True,
        },
        "disk": disks,
        "gates": {
            "enough_label_variance": enough_variance,
            "projected_within_12_hours": under_time_limit,
            "all_disks_below_80_percent": under_disk_limit,
            "continue_to_500": (
                enough_variance and under_time_limit and under_disk_limit),
        },
        "known_limit": (
            "Regional Enhance bytes are exact for the legal independent-tile "
            "fallback; final one-shot spatial routes are remeasured separately."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
