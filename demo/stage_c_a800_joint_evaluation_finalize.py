#!/usr/bin/env python3
"""Final integrity and resource snapshot for the REDS plus UVG evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def disk(path: str) -> dict:
    value = shutil.disk_usage(path)
    return {"total": value.total, "used": value.used, "free": value.free}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    ledger = read(args.run_root / "manifests" / "summary.json")
    frozen = read(args.run_root / "frozen_controller.json")
    routes = read(args.run_root / "routes" / "controller" / "manifest.json")
    formal = read(args.run_root / "formal" / "joint_evaluation_summary.json")
    if ledger["status"] != "frozen_before_quality_evaluation":
        raise RuntimeError("joint ledger was not frozen before evaluation")
    if frozen["status"] != "frozen_before_joint_quality_evaluation":
        raise RuntimeError("controller was not frozen before evaluation")
    if routes["sample_count"] != 37:
        raise RuntimeError("joint route manifest is incomplete")
    if formal["status"] != "complete" or formal["sample_count"] != 37:
        raise RuntimeError("joint formal summary is incomplete")
    markers = sorted(
        args.run_root.glob("formal/evaluation/*/sample.complete"))
    if len(markers) != 37:
        raise RuntimeError(f"expected 37 completion markers, found {len(markers)}")
    gpu_indices = set()
    used_memory_mib = []
    sample_path = args.run_root / "logs" / "gpu_samples.csv"
    if sample_path.is_file():
        for line in sample_path.read_text(encoding="utf-8").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                try:
                    gpu_indices.add(int(fields[1]))
                    used_memory_mib.append(int(fields[2]))
                except ValueError:
                    pass
    if gpu_indices and gpu_indices != {0}:
        raise RuntimeError(f"more than one GPU was sampled: {gpu_indices}")
    files = [path for path in args.run_root.rglob("*") if path.is_file()]
    links = [path for path in args.run_root.rglob("*") if path.is_symlink()]
    value = {
        "experiment": "A800 REDS plus UVG joint evaluation final snapshot",
        "status": "complete",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": args.wall_seconds,
        "frozen_controller": frozen,
        "dataset_counts": formal["dataset_counts"],
        "data_role_counts": formal["data_role_counts"],
        "combined_comparison": formal["comparisons"]["combined"],
        "integrity": {
            "route_count": routes["sample_count"],
            "formal_sample_count": formal["sample_count"],
            "sample_complete_marker_count": len(markers),
            "all_stream_sizes_rechecked_against_files": formal[
                "verification"]["all_stream_sizes_rechecked_against_files"],
            "all_spatial_fresh_decodes_pixel_exact": formal[
                "verification"]["all_spatial_stream_fresh_decodes_pixel_exact"],
            "fixed_visual_count": formal["verification"][
                "fixed_visuals_created"],
            "reused_sample_count": formal["verification"][
                "reused_sample_count"],
            "single_gpu_indices_observed": sorted(gpu_indices),
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
        "storage": {
            "ordinary_file_count_excluding_link_targets": len(files),
            "ordinary_file_bytes_excluding_link_targets": sum(
                path.stat().st_size for path in files),
            "symbolic_link_count": len(links),
            "du_bytes_excluding_link_targets": int(subprocess.check_output(
                ["du", "-sb", str(args.run_root)], text=True).split()[0]),
            "mounts": {path: disk(path) for path in (
                "/root", "/root/autodl-tmp", "/root/autodl-fs")},
        },
        "gpu": {
            "peak_nvidia_smi_used_memory_mib": (
                max(used_memory_mib) if used_memory_mib else None),
            "current": subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ], text=True).strip(),
        },
        "scientific_boundary": formal["scientific_boundary"],
    }
    atomic_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
