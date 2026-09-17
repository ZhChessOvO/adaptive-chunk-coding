#!/usr/bin/env python3
"""Validate the A800 pilot deliverables and write a resource snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_bytes(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return sum(item.stat().st_size for item in files), len(files)


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
    required_markers = (
        "stage0_smoke.complete",
        "teacher_calibration.complete",
        "full_teacher.complete",
        "roi_cost_teacher.complete",
        "controller.complete",
        "formal_evaluation.complete",
    )
    marker_status = {
        name: (args.run_root / name).is_file() for name in required_markers
    }
    if not all(marker_status.values()):
        raise RuntimeError(f"pilot markers are incomplete: {marker_status}")

    pilot = args.run_root / "pilot"
    train = read(pilot / "teacher_train" / "manifest.json")
    development = read(pilot / "teacher_development" / "manifest.json")
    roi_train = read(pilot / "roi_cost_train" / "manifest.json")
    roi_development = read(pilot / "roi_cost_development" / "manifest.json")
    controller = read(pilot / "controller" / "summary.json")
    formal = read(args.run_root / "formal" / "formal_summary.json")
    if not (
        train["complete"] and train["completed_sample_count"] == 500
        and development["complete"] and development["completed_sample_count"] == 6
        and roi_train["complete"] and roi_train["completed_sample_count"] == 500
        and roi_development["complete"]
        and roi_development["completed_sample_count"] == 6
        and formal["status"] == "complete"
    ):
        raise RuntimeError("one or more final manifests are incomplete")

    quality_peaks = [
        int(read(Path(entry["path"]))["runtime"]["peak_cuda_allocated_bytes"])
        for entry in train["entries"] + development["entries"]
    ]
    roi_peaks = [
        int(read(Path(entry["path"]))["runtime"]["peak_cuda_allocated_bytes"])
        for entry in roi_train["entries"] + roi_development["entries"]
    ]
    formal_peak = max(
        int(value["peak_cuda_allocated_bytes_max"])
        for value in formal["aggregate"].values())
    run_bytes, run_files = tree_bytes(args.run_root)
    formal_bytes, formal_files = tree_bytes(args.run_root / "formal")
    gpu = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    result = {
        "experiment": "A800 single-card pilot final resource snapshot",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "markers": marker_status,
        "hardware": {
            "nvidia_smi": gpu,
            "gpu_count_used": 1,
            "multi_gpu_used": False,
        },
        "software": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "labels": {
            "quality_teacher_train_samples": train["completed_sample_count"],
            "quality_teacher_development_samples": development[
                "completed_sample_count"],
            "roi_cost_train_samples": roi_train["completed_sample_count"],
            "roi_cost_development_samples": roi_development[
                "completed_sample_count"],
            "quality_teacher_sample_seconds_sum": sum(
                float(entry["seconds"])
                for entry in train["entries"] + development["entries"]),
            "roi_cost_sample_seconds_sum": sum(
                float(entry["seconds"])
                for entry in roi_train["entries"] + roi_development["entries"]),
        },
        "controller": {
            "parameter_count": controller["parameter_count"],
            "training_wall_seconds": controller["wall_seconds"],
            "fixed_training_schedule": controller["fixed_training_schedule"],
        },
        "formal": {
            "development_samples": formal["sample_count"],
            "variant_count": len(formal["aggregate"]),
            "summary": str(args.run_root / "formal" / "formal_summary.json"),
            "tree_bytes": formal_bytes,
            "file_count": formal_files,
        },
        "storage": {
            "run_tree_bytes": run_bytes,
            "run_file_count": run_files,
            "system": disk(Path("/root")),
            "fast": disk(Path("/root/autodl-tmp")),
            "file_store": disk(Path("/root/autodl-fs")),
        },
        "peak_cuda_allocated_bytes": {
            "quality_teacher": max(quality_peaks),
            "roi_cost_teacher": max(roi_peaks),
            "formal_evaluation": formal_peak,
            "overall": max(max(quality_peaks), max(roi_peaks), formal_peak),
        },
        "data_boundary": formal["scientific_boundary"],
    }
    # Include this snapshot itself in the reported run-tree size.  Rewriting
    # usually changes the decimal byte count once, so iterate to a fixed point.
    atomic_json(args.output, result)
    for _ in range(4):
        current_bytes, current_files = tree_bytes(args.run_root)
        if (result["storage"]["run_tree_bytes"] == current_bytes
                and result["storage"]["run_file_count"] == current_files):
            break
        result["storage"]["run_tree_bytes"] = current_bytes
        result["storage"]["run_file_count"] = current_files
        atomic_json(args.output, result)
    final_bytes, final_files = tree_bytes(args.run_root)
    if (result["storage"]["run_tree_bytes"] != final_bytes
            or result["storage"]["run_file_count"] != final_files):
        raise RuntimeError("run-tree byte snapshot did not reach a fixed point")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
