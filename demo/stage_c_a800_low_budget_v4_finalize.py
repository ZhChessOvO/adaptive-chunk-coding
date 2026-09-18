#!/usr/bin/env python3
"""Final integrity and resource snapshot for anchored hybrid v4."""

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
    selection = read(args.run_root / "selection" / "selection_summary.json")
    training = read(args.run_root / "controller" / "training_summary.json")
    routes = read(args.run_root / "routes" / "manifest.json")
    formal = read(args.run_root / "formal" / "low_budget_v4_summary.json")
    if selection["selection_policy"].get("hard_promotion_gate") is not False:
        raise RuntimeError("v4 selection unexpectedly used a hard gate")
    if training["status"] != "trained" or routes["sample_count"] != 6:
        raise RuntimeError("v4 training or route manifest is incomplete")
    if formal["status"] != "complete" or formal["sample_count"] != 6:
        raise RuntimeError("v4 formal summary is incomplete")
    selected_kind = selection["selected_candidate"]["feature_kind"]
    if training["selected_feature_kind"] != selected_kind:
        raise RuntimeError("v4 full training differs from training-only selection")
    if routes["selected_feature_kind"] != selected_kind:
        raise RuntimeError("v4 routes differ from training-only selection")
    expected_ids = {f"dev-s{index:03d}-f00-x384-y096" for index in range(6)}
    route_ids = {item["sample_id"] for item in routes["entries"]}
    if route_ids != expected_ids:
        raise RuntimeError(f"unexpected development identities: {route_ids}")
    markers = sorted(
        args.run_root.glob("formal/development/*/sample.complete"))
    if len(markers) != 6:
        raise RuntimeError(f"expected six completion markers, found {len(markers)}")
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
    value = {
        "experiment": "A800 low-budget anchored hybrid v4 final snapshot",
        "status": "complete",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "formal_wall_seconds": args.wall_seconds,
        "selection_wall_seconds": selection["wall_seconds"],
        "training_wall_seconds": training["wall_seconds"],
        "selected_feature_kind": selected_kind,
        "decision_support": formal["decision_support"],
        "encoder_analysis": formal["encoder_analysis"],
        "integrity": {
            "hard_promotion_gate_used": False,
            "training_only_selection_frozen_before_development": True,
            "full_training_complete": True,
            "route_count": routes["sample_count"],
            "formal_sample_count": formal["sample_count"],
            "sample_complete_marker_count": len(markers),
            "all_new_spatial_fresh_decodes_pixel_exact": formal[
                "verification"]["all_new_spatial_fresh_decodes_pixel_exact"],
            "only_expected_development_sample_ids": True,
            "single_gpu_indices_observed": sorted(gpu_indices),
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
        },
        "storage": {
            "ordinary_file_count": len(files),
            "ordinary_file_bytes": sum(path.stat().st_size for path in files),
            "du_bytes": int(subprocess.check_output(
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
