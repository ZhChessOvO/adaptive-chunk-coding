#!/usr/bin/env python3
"""Build the fixed REDS sample ledger for the single-A800 controller pilot.

The manifest is intentionally created before any model result is observed.  It
uses only REDS train/000..239 for teacher generation and REDS val/000..005 for
development.  No other validation directory is listed or inspected.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image


TRAIN_SEQUENCES = tuple(f"{index:03d}" for index in range(240))
DEV_SEQUENCES = tuple(f"{index:03d}" for index in range(6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the fixed train/development ledger for the A800 pilot")
    parser.add_argument(
        "--train-root", type=Path,
        default=Path("data/REDS/train_sharp"))
    parser.add_argument(
        "--dev-root", type=Path,
        default=Path("data/REDS/val_sharp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-alignment", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if not 500 <= args.train_count <= 1000:
        parser.error("--train-count must remain inside the authorized 500..1000 pilot")
    if args.frame_count != 17:
        parser.error("the authorized first pilot uses exactly 17-frame samples")
    if args.crop_width != 512 or args.crop_height != 512:
        parser.error("the authorized first pilot uses exactly 512x512 crops")
    if args.crop_alignment < 1:
        parser.error("--crop-alignment must be positive")
    return args


def frame_paths(sequence_dir: Path) -> list[Path]:
    paths = sorted(sequence_dir.glob("*.png"))
    if len(paths) != 100:
        raise ValueError(f"expected 100 PNG frames in {sequence_dir}, found {len(paths)}")
    return paths


def aligned_choices(limit: int, alignment: int) -> list[int]:
    if limit < 0:
        raise ValueError("crop is larger than the REDS frame")
    values = list(range(0, limit + 1, alignment))
    if values[-1] != limit and limit % alignment == 0:
        values.append(limit)
    return values


def make_record(
    *, sample_id: str, split: str, sequence: str, source_dir: Path,
    paths: list[Path], start: int, crop_x: int, crop_y: int,
    crop_width: int, crop_height: int, seed: int,
) -> dict:
    selected = paths[start:start + 17]
    if len(selected) != 17:
        raise ValueError(f"{sample_id} does not contain 17 frames")
    return {
        "sample_id": sample_id,
        "split": split,
        "source_role": (
            "training teacher-label source; REDS train only"
            if split == "train" else
            "development source; REDS val/000..005, never an independent test"
        ),
        "sequence": sequence,
        "source_dir": str(source_dir.resolve()),
        "frame_start": start,
        "frame_count": 17,
        "source_files": [str(path.resolve()) for path in selected],
        "crop": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_width,
            "height": crop_height,
        },
        "seed": seed,
    }


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    train_root = args.train_root.resolve()
    dev_root = args.dev_root.resolve()
    train_paths = {
        sequence: frame_paths(train_root / sequence)
        for sequence in TRAIN_SEQUENCES
    }
    # Deliberately address only the six authorized development directories.
    dev_paths = {
        sequence: frame_paths(dev_root / sequence)
        for sequence in DEV_SEQUENCES
    }

    first = Image.open(train_paths["000"][0])
    frame_width, frame_height = first.size
    first.close()
    x_choices = aligned_choices(frame_width - args.crop_width, args.crop_alignment)
    y_choices = aligned_choices(frame_height - args.crop_height, args.crop_alignment)

    starts = (0, 40, 80)
    train_records = []
    for index in range(args.train_count):
        sequence = TRAIN_SEQUENCES[index % len(TRAIN_SEQUENCES)]
        repeat = index // len(TRAIN_SEQUENCES)
        if repeat >= len(starts):
            # Counts above 720 remain deterministic while using a shifted window.
            start = (repeat * 23 + index) % (101 - args.frame_count)
        else:
            start = starts[repeat]
        sample_seed = args.seed + index * 1009
        generator = random.Random(sample_seed)
        crop_x = generator.choice(x_choices)
        crop_y = generator.choice(y_choices)
        train_records.append(make_record(
            sample_id=f"train-{index:04d}-s{sequence}-f{start:02d}-x{crop_x:03d}-y{crop_y:03d}",
            split="train", sequence=sequence,
            source_dir=train_root / sequence, paths=train_paths[sequence],
            start=start, crop_x=crop_x, crop_y=crop_y,
            crop_width=args.crop_width, crop_height=args.crop_height,
            seed=sample_seed,
        ))

    center_x = ((frame_width - args.crop_width) // 2 // args.crop_alignment
                * args.crop_alignment)
    center_y = ((frame_height - args.crop_height) // 2 // args.crop_alignment
                * args.crop_alignment)
    dev_records = [
        make_record(
            sample_id=f"dev-s{sequence}-f00-x{center_x:03d}-y{center_y:03d}",
            split="development", sequence=sequence,
            source_dir=dev_root / sequence, paths=dev_paths[sequence],
            start=0, crop_x=center_x, crop_y=center_y,
            crop_width=args.crop_width, crop_height=args.crop_height,
            seed=args.seed + 1_000_000 + int(sequence),
        )
        for sequence in DEV_SEQUENCES
    ]

    all_ids = [record["sample_id"] for record in train_records + dev_records]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("sample IDs are not unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_samples.jsonl"
    dev_path = args.output_dir / "development_samples.jsonl"
    write_jsonl_atomic(train_path, train_records)
    write_jsonl_atomic(dev_path, dev_records)
    summary = {
        "experiment": "A800 single-card pilot fixed sample ledger",
        "protocol_version": 1,
        "seed": args.seed,
        "train": {
            "path": str(train_path),
            "sample_count": len(train_records),
            "allowed_sequences": "REDS train/000..239",
            "sequence_sample_counts": dict(sorted(Counter(
                record["sequence"] for record in train_records).items())),
        },
        "development": {
            "path": str(dev_path),
            "sample_count": len(dev_records),
            "allowed_sequences": "REDS val/000..005",
            "independent_test": False,
        },
        "sample_shape": {
            "frames": 17,
            "width": args.crop_width,
            "height": args.crop_height,
        },
        "forbidden_or_sealed_validation_opened": False,
        "file_hashes_recorded": False,
    }
    summary_path = args.output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "train_samples": len(train_records),
        "development_samples": len(dev_records),
        "frame_size": [frame_width, frame_height],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
