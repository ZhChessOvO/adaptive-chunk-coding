#!/usr/bin/env python3
"""Summarize all quality and actual ROI-cost teacher labels."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


QUALITY_TARGETS = (
    "generate_lpips_gain_vs_base",
    "enhance_lpips_gain_vs_base",
    "generate_psnr_delta_db_vs_base",
    "enhance_psnr_delta_db_vs_base",
    "generate_temporal_risk_vs_base",
    "enhance_temporal_gain_vs_base",
    "enhance_fallback_extra_on_disk_bytes",
)
ROI_TARGETS = (
    "generate_roi_seconds_measured_geometry_class",
    "generate_roi_processing_pixels_per_frame",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-quality-manifest", type=Path, required=True)
    parser.add_argument("--train-roi-cost-manifest", type=Path, required=True)
    parser.add_argument("--development-quality-manifest", type=Path, required=True)
    parser.add_argument("--development-roi-cost-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def summarize(quality_path: Path, roi_path: Path) -> dict:
    quality_manifest = read(quality_path)
    roi_manifest = read(roi_path)
    if not quality_manifest.get("complete") or not roi_manifest.get("complete"):
        raise ValueError("teacher manifests must be complete")
    roi_entries = {entry["sample_id"]: entry for entry in roi_manifest["entries"]}
    target_values = {name: [] for name in QUALITY_TARGETS + ROI_TARGETS}
    uniform_bytes = {name: [] for name in ("Generate", "Base", "Enhance")}
    geometry_seconds: dict[str, list[float]] = {}
    best_actions: Counter[str] = Counter()
    sequences: Counter[str] = Counter()
    quality_seconds = []
    roi_seconds = []
    for entry in quality_manifest["entries"]:
        sample = read(Path(entry["path"]))
        sample_id = sample["sample"]["sample_id"]
        roi = read(Path(roi_entries[sample_id]["path"]))
        sequences[sample["sample"]["sequence"]] += 1
        quality_seconds.append(float(sample["runtime"]["sample_total_seconds"]))
        roi_seconds.append(float(roi["runtime"]["sample_total_seconds"]))
        for action, value in sample["rate"]["uniform_stream_bytes"].items():
            uniform_bytes[action].append(float(value))
        roi_regions = {
            int(region["index"]): region["targets"] for region in roi["regions"]
        }
        for region in sample["regions"]:
            for name in QUALITY_TARGETS:
                target_values[name].append(float(region["targets"][name]))
            for name in ROI_TARGETS:
                target_values[name].append(float(roi_regions[region["index"]][name]))
            best = min(
                region["candidates"],
                key=lambda action: region["candidates"][action]["lpips_alex"])
            best_actions[best] += 1
        for name, measurement in roi["measurements"].items():
            geometry_seconds.setdefault(name, []).append(float(
                measurement["seconds_model_load_excluded"]))
    return {
        "sample_count": quality_manifest["completed_sample_count"],
        "region_count": sum(len(read(Path(entry["path"]))["regions"])
                            for entry in quality_manifest["entries"]),
        "sequence_sample_counts": dict(sorted(sequences.items())),
        "quality_teacher_sample_seconds": stats(quality_seconds),
        "roi_cost_teacher_sample_seconds": stats(roi_seconds),
        "uniform_actual_stream_bytes": {
            name: stats(values) for name, values in uniform_bytes.items()
        },
        "targets": {name: stats(values) for name, values in target_values.items()},
        "roi_geometry_seconds": {
            name: stats(values) for name, values in geometry_seconds.items()
        },
        "best_lpips_action_counts": dict(best_actions),
        "quality_label_tree_bytes": tree_bytes(quality_path.parent),
        "roi_cost_label_tree_bytes": tree_bytes(roi_path.parent),
    }


def main() -> None:
    args = parse_args()
    result = {
        "experiment": "A800 complete teacher-label summary",
        "train": summarize(
            args.train_quality_manifest, args.train_roi_cost_manifest),
        "development": summarize(
            args.development_quality_manifest,
            args.development_roi_cost_manifest),
        "scientific_boundary": {
            "train_source": "REDS train/000..239",
            "development_source": "REDS val/000..005",
            "development_is_independent_test": False,
            "validation_012_023_read": False,
            "sealed_validation_024_029_read": False,
            "actual_uniform_stream_bytes": True,
            "actual_sample_specific_roi_geometry_compute": True,
            "all_candidate_frames_retained": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "train_samples": result["train"]["sample_count"],
        "development_samples": result["development"]["sample_count"],
        "train_best_actions": result["train"]["best_lpips_action_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
