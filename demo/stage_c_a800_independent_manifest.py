#!/usr/bin/env python3
"""Create the preregistered one-shot REDS independent-test ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


TEST_SEQUENCES = tuple(f"{index:03d}" for index in range(12, 24))
FRAME_COUNT = 17
CROP_X = 384
CROP_Y = 96
CROP_WIDTH = 512
CROP_HEIGHT = 512
SEED_BASE = 20260917 + 2_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the frozen REDS val/012..023 independent-test ledger")
    parser.add_argument(
        "--validation-root", type=Path,
        default=Path("data/REDS/val_sharp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validation_root = args.validation_root.resolve()
    records = []
    for sequence in TEST_SEQUENCES:
        source_dir = validation_root / sequence
        paths = sorted(source_dir.glob("*.png"))
        if len(paths) != 100:
            raise RuntimeError(
                f"expected 100 frames in {source_dir}, found {len(paths)}")
        with Image.open(paths[0]) as image:
            if image.size != (1280, 720):
                raise RuntimeError(
                    f"unexpected REDS frame size {image.size} in {source_dir}")
        selected = paths[:FRAME_COUNT]
        sample_id = f"test-s{sequence}-f00-x{CROP_X:03d}-y{CROP_Y:03d}"
        records.append({
            "sample_id": sample_id,
            "split": "independent_test",
            "source_role": (
                "one-shot independent test; REDS val/012..023; consumed "
                "after this evaluation and never used for tuning"),
            "sequence": sequence,
            "source_dir": str(source_dir),
            "frame_start": 0,
            "frame_count": FRAME_COUNT,
            "source_files": [str(path.resolve()) for path in selected],
            "crop": {
                "x": CROP_X,
                "y": CROP_Y,
                "width": CROP_WIDTH,
                "height": CROP_HEIGHT,
            },
            "seed": SEED_BASE + int(sequence),
        })

    sample_ids = [record["sample_id"] for record in records]
    if len(records) != 12 or len(set(sample_ids)) != 12:
        raise RuntimeError("independent-test ledger must contain 12 unique samples")

    manifest_path = args.output_dir / "test_samples.jsonl"
    atomic_text(
        manifest_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n"
                for record in records))
    summary = {
        "experiment": "A800 one-shot independent-test sample ledger",
        "status": "complete",
        "authorization_date": "2026-09-18",
        "data_role": "REDS val/012..023 one-shot independent test",
        "sample_count": len(records),
        "sequences": list(TEST_SEQUENCES),
        "frame_start": 0,
        "frame_count": FRAME_COUNT,
        "crop": {
            "x": CROP_X,
            "y": CROP_Y,
            "width": CROP_WIDTH,
            "height": CROP_HEIGHT,
        },
        "seed_rule": "20260917 + 2000000 + integer sequence number",
        "manifest": str(manifest_path.resolve()),
        "scientific_boundary": {
            "samples_selected_before_image_content_was_read": True,
            "result_driven_sample_exclusion_allowed": False,
            "validation_012_023_consumed_by_this_test": True,
            "sealed_validation_024_029_read": False,
        },
    }
    atomic_text(
        args.output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
