#!/usr/bin/env python3
"""Create the frozen REDS-validation plus UVG paper-evaluation ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image


FRAME_COUNT = 17
SEED_BASE = 20260918 + 3_000_000
UVG_NAMES = (
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySetGo",
    "ShakeNDry",
    "YachtRide",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reds-root", type=Path, required=True)
    parser.add_argument("--uvg-root", type=Path, required=True)
    parser.add_argument("--controller-name", required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def selected_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def verify_paths(
    paths: list[Path], expected_count: int, expected_size: tuple[int, int],
) -> None:
    if len(paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} PNG files, found {len(paths)}")
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != expected_size:
                raise RuntimeError(
                    f"unexpected frame size {image.size} in {path}")


def reds_role(index: int) -> tuple[str, str]:
    if index <= 5:
        return (
            "development",
            "REDS validation development sample used repeatedly during design",
        )
    if index <= 11:
        return (
            "historically_used_evaluation",
            "REDS validation sample that influenced earlier experiments",
        )
    if index <= 23:
        return (
            "previously_consumed_evaluation",
            "REDS validation sample used in the earlier one-shot evaluation",
        )
    return (
        "frozen_new_evaluation",
        "REDS validation sample frozen before this joint paper evaluation",
    )


def reds_sample_id(index: int, sequence: str) -> str:
    """Keep historical IDs where formal outputs already exist."""
    if index <= 5:
        prefix = "dev"
    elif 12 <= index <= 23:
        prefix = "test"
    else:
        prefix = "eval"
    return f"{prefix}-s{sequence}-f00-x384-y096"


def reds_seed(index: int) -> int:
    if index <= 5:
        return 20260917 + 1_000_000 + index
    if 12 <= index <= 23:
        return 20260917 + 2_000_000 + index
    return SEED_BASE + index


def main() -> None:
    args = parse_args()
    reds_root = args.reds_root.resolve()
    uvg_root = args.uvg_root.resolve()
    checkpoint = args.controller_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    records = []

    for index in range(30):
        sequence = f"{index:03d}"
        source_dir = reds_root / sequence
        paths = sorted(source_dir.glob("*.png"))
        verify_paths(paths, 100, (1280, 720))
        selected = paths[:FRAME_COUNT]
        role, description = reds_role(index)
        records.append({
            "sample_id": reds_sample_id(index, sequence),
            "dataset": "REDS",
            "split": "validation",
            "data_role": role,
            "source_role": description,
            "sequence": sequence,
            "source_dir": str(source_dir),
            "frame_start": 0,
            "frame_count": FRAME_COUNT,
            "source_files": [str(path.resolve()) for path in selected],
            "selected_source_sha256": selected_digest(selected),
            "crop": {"x": 384, "y": 96, "width": 512, "height": 512},
            "seed": reds_seed(index),
        })

    for name in UVG_NAMES:
        source_dir = uvg_root / name
        paths = sorted(source_dir.glob("*.png"))
        verify_paths(paths, FRAME_COUNT, (512, 512))
        source_manifest_path = source_dir / "source_manifest.json"
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("official_sequence") != name:
            raise RuntimeError(f"UVG source manifest mismatch for {name}")
        records.append({
            "sample_id": f"uvg-{name.lower()}-f00-center512",
            "dataset": "UVG",
            "split": "official_1080p_sequences",
            "data_role": (
                "historically_used_external_evaluation"
                if name in {"Jockey", "ShakeNDry"}
                else "cross_distribution_evaluation_not_claimed_independent"
            ),
            "source_role": (
                "UVG cross-distribution paper evaluation; reported together "
                "with REDS validation, not as a separate application test"
            ),
            "sequence": name,
            "source_dir": str(source_dir),
            "frame_start": 0,
            "frame_count": FRAME_COUNT,
            "source_files": [str(path.resolve()) for path in paths],
            "selected_source_sha256": selected_digest(paths),
            "official_source_manifest": str(source_manifest_path.resolve()),
            "official_source_url": source_manifest["official_source_url"],
            "original_crop": source_manifest["evaluation_sample"]["crop"],
            "crop": {"x": 0, "y": 0, "width": 512, "height": 512},
            "seed": SEED_BASE + len(records),
        })

    ids = [record["sample_id"] for record in records]
    if len(records) != 37 or len(set(ids)) != 37:
        raise RuntimeError("joint evaluation ledger must contain 37 unique samples")
    manifest_path = args.output_dir / "joint_samples.jsonl"
    atomic_text(
        manifest_path,
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records),
    )
    summary = {
        "experiment": "A800 frozen REDS plus UVG joint paper evaluation ledger",
        "status": "frozen_before_quality_evaluation",
        "created_date": "2026-09-18",
        "sample_count": len(records),
        "dataset_counts": dict(Counter(
            record["dataset"] for record in records)),
        "data_role_counts": dict(Counter(
            record["data_role"] for record in records)),
        "sample_shape": {"frame_count": 17, "width": 512, "height": 512},
        "controller": {
            "name": args.controller_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "code": {"git_commit_before_evaluation": args.git_commit},
        "manifest": str(manifest_path.resolve()),
        "reporting": {
            "per_dataset": True,
            "combined_pool": True,
            "external_sequences_called_application_test": False,
            "result_driven_sample_exclusion_allowed": False,
            "historical_roles_reported_per_sample": True,
        },
        "qst_status": {
            "included": False,
            "reason": (
                "The official QST release is a full Baidu Pan archive and the "
                "old local sky alias lacks an official clip ID. No ambiguous "
                "or unofficial replacement is silently mixed into this ledger."
            ),
            "future_addition_rule": (
                "Record official clip ID, frame window, crop, source URL and "
                "historical role, then report QST as another dataset stratum."
            ),
        },
        "scientific_boundary": {
            "manifest_fixed_before_any_024_029_or_uvg_quality_result": True,
            "controller_fixed_before_joint_evaluation": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
            "single_gpu": True,
        },
    }
    atomic_text(
        args.output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
