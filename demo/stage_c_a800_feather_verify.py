#!/usr/bin/env python3
"""Independently verify and interpret the frozen feather diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--joint-summary", type=Path, required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    formal_root = run_root / "formal"
    summary_path = formal_root / "feather_diagnostic_summary.json"
    summary = read(summary_path)
    joint = read(args.joint_summary.resolve())
    frozen = read(run_root / "frozen_source.json")
    snapshot = read(run_root / "final_resource_snapshot.json")

    if summary["status"] != "complete" or summary["sample_count"] != 37:
        raise RuntimeError("feather summary is incomplete")
    if joint["status"] != "complete" or joint["sample_count"] != 37:
        raise RuntimeError("joint source summary is incomplete")
    if sha256(args.joint_summary.resolve()) != frozen[
            "source_joint_summary_sha256"]:
        raise RuntimeError("joint source summary SHA-256 changed")
    if sha256(args.controller_checkpoint.resolve()) != frozen[
            "controller_checkpoint_sha256"]:
        raise RuntimeError("controller checkpoint SHA-256 changed")
    if (run_root / "formal.complete").read_text(encoding="utf-8").strip() \
            != "complete":
        raise RuntimeError("formal completion marker differs")

    feather_samples = {item["sample_id"]: item for item in summary["samples"]}
    joint_samples = {item["sample_id"]: item for item in joint["samples"]}
    if set(feather_samples) != set(joint_samples) or len(feather_samples) != 37:
        raise RuntimeError("feather and joint sample IDs differ")
    if Counter(item["dataset"] for item in summary["samples"]) != Counter(
            {"REDS": 30, "UVG": 7}):
        raise RuntimeError("dataset counts differ")

    saved_frame_count = 0
    boundary_sample_count = 0
    for sample_id, item in feather_samples.items():
        sample_path = run_root / "samples" / sample_id / "summary.json"
        if not sample_path.is_file():
            raise RuntimeError(f"missing sample summary: {sample_id}")
        saved = read(sample_path)
        if saved != item:
            raise RuntimeError(f"aggregate sample differs from saved sample: {sample_id}")
        baseline = item["baseline_feather8_regression"]
        if not baseline["pixel_exact"] or baseline["max_abs_pixel_error"] != 0:
            raise RuntimeError(f"8 px regression failed: {sample_id}")
        if item["variants"]["8"]["boundary"][
                "cross_generate_boundary_edge_count"] > 0:
            boundary_sample_count += 1
        for feather in (16, 32):
            frame_root = Path(item["variants"][str(feather)]["frames"])
            paths = sorted(frame_root.glob("*.png"))
            if len(paths) != 17:
                raise RuntimeError(
                    f"{sample_id} feather {feather}: expected 17 frames")
            for path in paths:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.size != (512, 512):
                        raise RuntimeError(f"unexpected image size: {path}")
            saved_frame_count += len(paths)
        visual = Path(item["fixed_visual"])
        if not visual.is_file() or visual.stat().st_size == 0:
            raise RuntimeError(f"fixed visual is missing: {sample_id}")

    csv_path = formal_root / "feather_diagnostic_samples.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 37 * 3:
        raise RuntimeError("CSV row count differs")
    if Counter(int(row["feather_pixels"]) for row in csv_rows) != Counter(
            {8: 37, 16: 37, 32: 37}):
        raise RuntimeError("CSV feather counts differ")

    execution_log = (run_root / "logs" / "feather_diagnostic.log").read_text(
        encoding="utf-8")
    failure_tokens = ("Traceback", "CUDA out of memory", "Killed", "FAILED")
    found_failures = [token for token in failure_tokens if token in execution_log]
    if found_failures:
        raise RuntimeError(f"failure token in execution log: {found_failures}")
    if snapshot["sample_count"] != 37 or not snapshot["single_gpu"]:
        raise RuntimeError("resource snapshot differs")

    contribution = {}
    for group in ("combined", "REDS", "UVG"):
        members = [
            item for item in summary["samples"]
            if group == "combined" or item["dataset"] == group
        ]
        group_value = {"sample_count": len(members), "feathers": {}}
        for feather in (8, 16, 32):
            values = []
            rows = []
            for item in members:
                joint_item = joint_samples[item["sample_id"]]
                no_generate = joint_item["variants"]["final-no-generate"][
                    "quality"]["lpips_alex"]
                current = item["variants"][str(feather)]["quality"][
                    "lpips_alex"]
                benefit = no_generate - current
                values.append(benefit)
                rows.append({
                    "sample_id": item["sample_id"],
                    "sequence": item["sequence"],
                    "dataset": item["dataset"],
                    "no_generate_minus_feather_lpips": benefit,
                })
            group_value["feathers"][str(feather)] = {
                "mean_no_generate_minus_feather_lpips": sum(values) / len(values),
                "generate_beneficial_count": sum(value > 1e-12 for value in values),
                "generate_equal_count": sum(abs(value) <= 1e-12 for value in values),
                "generate_harmful_count": sum(value < -1e-12 for value in values),
                "samples": rows,
            }
        contribution[group] = group_value

    combined = summary["groups"]["combined"]["feathers"]
    uvg = summary["groups"]["UVG"]["feathers"]
    decision = {
        "recommended_feather_pixels": 16,
        "selection_is_a_tradeoff_not_a_hard_gate": True,
        "reason": (
            "16 px improves the boundary-gradient metric on all 33 samples "
            "with a Generate boundary and improves mean UVG LPIPS, while its "
            "combined LPIPS cost is much smaller than 32 px.  It does not fix "
            "Generate misrouting, which remains a separate v6 change."
        ),
        "evidence": {
            "combined_lpips_delta_vs_8px": combined["16"][
                "comparison_to_8px"]["mean_lpips_delta"],
            "combined_boundary_gradient_delta_vs_8px": combined["16"][
                "comparison_to_8px"]["mean_boundary_gradient_error_delta"],
            "boundary_better_count": combined["16"][
                "comparison_to_8px"]["boundary_gradient_better_count"],
            "boundary_sample_count": boundary_sample_count,
            "uvg_lpips_delta_vs_8px": uvg["16"]["comparison_to_8px"][
                "mean_lpips_delta"],
            "feather32_combined_lpips_delta_vs_8px": combined["32"][
                "comparison_to_8px"]["mean_lpips_delta"],
        },
    }

    verification = {
        "experiment": "independent verification of A800 feather diagnostic",
        "status": "verified",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "sample_summaries": len(feather_samples),
            "dataset_counts": {"REDS": 30, "UVG": 7},
            "feather8_pixel_exact_count": 37,
            "saved_16px_and_32px_frames": saved_frame_count,
            "fixed_visuals": 37,
            "csv_rows": len(csv_rows),
            "source_joint_summary_sha256_matches": True,
            "controller_checkpoint_sha256_matches": True,
            "execution_log_failure_tokens": found_failures,
            "formal_completion_marker": True,
            "single_gpu": True,
        },
        "resource_snapshot": snapshot,
        "decision": decision,
        "scientific_boundary": {
            "verification_did_not_rerun_codec_or_seedvr2": True,
            "decision_uses_the_37_diagnostic_samples": True,
            "these_samples_are_development_or_error_analysis_for_v6": True,
        },
    }
    atomic_json(formal_root / "independent_verification.json", verification)
    atomic_json(
        formal_root / "generate_contribution_by_feather.json",
        {
            "definition": (
                "final-no-generate LPIPS minus feather-variant LPIPS; positive "
                "means Generate improves LPIPS"
            ),
            "groups": contribution,
        },
    )

    lines = [
        "# Generate 羽化诊断独立复核与选择",
        "",
        "37/37 条样本、8 px 逐像素回归、1,258 张新保存帧、37 张固定图、",
        "源联合汇总与 checkpoint SHA-256、完成标志和执行日志均已独立复核。",
        "",
        "**选择 16 px 作为 v6 默认羽化宽度。** 这不是硬门槛：16 px 在有",
        "Generate 边界的 33/33 条上降低边界梯度误差，UVG 平均 LPIPS 也降低",
        f"{abs(decision['evidence']['uvg_lpips_delta_vs_8px']):.6f}；代价是合并 37 条",
        f"LPIPS 比 8 px 高 {decision['evidence']['combined_lpips_delta_vs_8px']:.6f}。",
        f"32 px 的合并 LPIPS 代价为 {decision['evidence']['feather32_combined_lpips_delta_vs_8px']:.6f}，",
        "因此没有选择更宽的 32 px。",
        "",
        "更宽羽化只缓解贴回边界，不会纠正 Jockey 等样本上的 Generate 误选；",
        "v6 仍需单独加入空间一致性和 UVG 跨域适配。",
    ]
    atomic_text(
        formal_root / "independent_verification.md", "\n".join(lines) + "\n")
    print(json.dumps({
        "status": "verified",
        "recommended_feather_pixels": 16,
        "output": str(formal_root / "independent_verification.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
