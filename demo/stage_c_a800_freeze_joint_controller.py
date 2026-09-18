#!/usr/bin/env python3
"""Record the final low-budget controller chosen before joint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("v1", "v5"), required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-summary", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    v1 = read(args.v1_summary)
    v5 = read(args.v5_summary)
    decision = v5["decision_support"]
    if args.kind == "v5":
        checkpoint = args.v5_checkpoint.resolve()
        name = "v5 conservative context-probe consensus low-budget controller"
        route_variant = "conservative-consensus-low-v5"
        reason = (
            "v5 was preferred after the fixed six-video development comparison "
            "using joint quality, rate, cross-video consistency and compute."
        )
    else:
        checkpoint = args.v1_checkpoint.resolve()
        name = "v1 low-budget controller"
        route_variant = "mlp-budget-0"
        reason = (
            "v1 remained the better overall choice after the fixed six-video "
            "v5 comparison; v5 remains a reported ablation."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    value = {
        "experiment": "A800 final low-budget controller freeze",
        "status": "frozen_before_joint_quality_evaluation",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "route_variant": route_variant,
        "git_commit": args.git_commit,
        "selection_basis": reason,
        "development_evidence": {
            "sample_count": 6,
            "data_role": "REDS val/000..005 development",
            "v1": {
                "mean_actual_bytes": v1["aggregate"]["low-joint"][
                    "actual_on_disk_bytes_mean"],
                "mean_lpips": v1["aggregate"]["low-joint"][
                    "quality_mean"]["lpips_alex"],
            },
            "v5": {
                "mean_actual_bytes": v5["aggregate"]["low-v5"][
                    "actual_on_disk_bytes_mean"],
                "mean_lpips": v5["aggregate"]["low-v5"][
                    "quality_mean"]["lpips_alex"],
                "mean_lpips_delta_vs_v1": decision[
                    "quality_vs_v1"]["mean_lpips_delta"],
                "mean_byte_delta_vs_v1": decision[
                    "rate_vs_v1"]["mean_actual_byte_delta"],
                "better_video_count_vs_v1": decision[
                    "quality_vs_v1"]["better_video_count"],
            },
            "hard_pass_fail_gate": False,
        },
        "evaluation_policy": {
            "controller_or_settings_may_change_after_joint_results": False,
            "per_dataset_and_combined_reporting": True,
            "development_and_historical_roles_retained": True,
            "uvg_or_qst_called_application_test": False,
        },
        "scientific_boundary": {
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
            "single_gpu": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
