#!/usr/bin/env python3
"""Validate the A800 follow-up deliverables and snapshot resource use."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


NEW_VARIANTS = (
    "middle-persistent", "low-joint", "low-no-generate", "low-no-enhance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_bytes(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return sum(item.stat().st_size for item in files), len(files)


def du_apparent_bytes(path: Path) -> int:
    output = subprocess.check_output(
        ["du", "-sb", str(path)], text=True).splitlines()[0]
    return int(output.split()[0])


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
    formal_marker = args.run_root / "formal_evaluation.complete"
    if not formal_marker.is_file():
        raise RuntimeError("follow-up formal marker is missing")
    sample_markers = sorted(
        (args.run_root / "formal" / "development").glob(
            "*/sample.complete"))
    if len(sample_markers) != 6:
        raise RuntimeError(
            f"expected six completed follow-up samples, found {len(sample_markers)}")
    summary_path = args.run_root / "formal" / "followup_summary.json"
    summary = read(summary_path)
    if summary.get("status") != "complete" or summary.get("sample_count") != 6:
        raise RuntimeError("follow-up summary is incomplete")
    if not summary["verification"][
            "middle_executor_outputs_pixel_exact_to_legacy"]:
        raise RuntimeError("persistent executor regression is not pixel exact")

    run_bytes, run_files = tree_bytes(args.run_root)
    formal_bytes, formal_files = tree_bytes(args.run_root / "formal")
    gpu = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    result = {
        "experiment": "A800 single-card follow-up final resource snapshot",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "orchestration_wall_seconds": args.wall_seconds,
        "markers": {
            "formal_evaluation.complete": True,
            "completed_sample_markers": len(sample_markers),
        },
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
        "formal": {
            "development_samples": summary["sample_count"],
            "new_variant_count": len(NEW_VARIANTS),
            "summary": str(summary_path),
            "tree_bytes": formal_bytes,
            "du_apparent_bytes": du_apparent_bytes(args.run_root / "formal"),
            "file_count": formal_files,
            "new_complete_pipeline_seconds_sum": sum(
                sample["variants"][name]["runtime_seconds"]
                for sample in summary["samples"] for name in NEW_VARIANTS),
        },
        "storage": {
            "run_tree_bytes": run_bytes,
            "du_apparent_bytes": du_apparent_bytes(args.run_root),
            "run_file_count": run_files,
            "system": disk(Path("/root")),
            "fast": disk(Path("/root/autodl-tmp")),
            "file_store": disk(Path("/root/autodl-fs")),
        },
        "peak_cuda_allocated_bytes": {
            name: summary["aggregate"][name][
                "peak_cuda_allocated_bytes_max"]
            for name in NEW_VARIANTS
        },
        "route_action_counts": summary["route"]["action_counts_total"],
        "data_boundary": summary["scientific_boundary"],
    }
    result["peak_cuda_allocated_bytes"]["overall_new_formal"] = max(
        result["peak_cuda_allocated_bytes"][name] for name in NEW_VARIANTS)

    # Include this snapshot itself in the byte count and converge to a fixed
    # point because the decimal count can change the JSON length once.
    atomic_json(args.output, result)
    for _ in range(4):
        current_bytes, current_files = tree_bytes(args.run_root)
        current_du_bytes = du_apparent_bytes(args.run_root)
        if (result["storage"]["run_tree_bytes"] == current_bytes
                and result["storage"]["run_file_count"] == current_files
                and result["storage"]["du_apparent_bytes"]
                == current_du_bytes):
            break
        result["storage"]["run_tree_bytes"] = current_bytes
        result["storage"]["run_file_count"] = current_files
        result["storage"]["du_apparent_bytes"] = current_du_bytes
        atomic_json(args.output, result)
    final_bytes, final_files = tree_bytes(args.run_root)
    final_du_bytes = du_apparent_bytes(args.run_root)
    if (result["storage"]["run_tree_bytes"] != final_bytes
            or result["storage"]["run_file_count"] != final_files
            or result["storage"]["du_apparent_bytes"] != final_du_bytes):
        raise RuntimeError("follow-up run-tree snapshot did not reach a fixed point")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
