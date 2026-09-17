#!/usr/bin/env python3
"""Write an atomic pixel-exact regression record for two PNG sequences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    reference = pngs(args.reference_dir)
    candidate = pngs(args.candidate_dir)
    if len(reference) != args.expected_frames:
        raise RuntimeError(
            f"reference has {len(reference)} frames, expected {args.expected_frames}")
    if len(candidate) != args.expected_frames:
        raise RuntimeError(
            f"candidate has {len(candidate)} frames, expected {args.expected_frames}")

    maximum = 0
    unequal_pixels = 0
    for reference_path, candidate_path in zip(reference, candidate):
        reference_image = np.asarray(
            Image.open(reference_path).convert("RGB"), dtype=np.int16)
        candidate_image = np.asarray(
            Image.open(candidate_path).convert("RGB"), dtype=np.int16)
        if reference_image.shape != candidate_image.shape:
            raise RuntimeError(
                f"frame shapes differ: {reference_path}, {candidate_path}")
        difference = np.abs(reference_image - candidate_image)
        maximum = max(maximum, int(difference.max()))
        unequal_pixels += int(np.count_nonzero(np.any(difference != 0, axis=2)))

    result = {
        "experiment": "PNG sequence pixel-exact regression",
        "reference_dir": str(args.reference_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "frame_count": args.expected_frames,
        "maximum_absolute_channel_error": maximum,
        "unequal_rgb_pixel_count": unequal_pixels,
        "pixel_exact": maximum == 0 and unequal_pixels == 0,
    }
    if not result["pixel_exact"]:
        raise RuntimeError(
            f"frame regression failed: max={maximum}, pixels={unequal_pixels}")
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
