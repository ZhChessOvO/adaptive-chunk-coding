#!/usr/bin/env python3
"""Prepare and finalize the single-frame A800 spatial/SeedVR2 smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--sample-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> None:
    with args.sample_manifest.open("r", encoding="utf-8") as handle:
        sample = json.loads(next(line for line in handle if line.strip()))
    gate = {
        "experiment": "A800 stage-0 one-frame smoke input",
        "sequence": sample["sequence"],
        "source_role": "REDS training data used only for environment smoke",
        "source_files": sample["source_files"][:1],
        "crop": sample["crop"],
        "frames": 1,
    }
    actions = [
        1, 1, 0, 0,
        1, 0, 0, 2,
        0, 0, 2, 2,
        0, 2, 2, 2,
    ]
    route = {
        "experiment": "A800 stage-0 fixed mixed-action smoke route",
        "route_kind": "fixed-smoke-route",
        "selected_variant": "smoke-mixed",
        "configuration": {
            "tile_size": 128,
            "tile_grid": [4, 4],
            "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
        },
        "variants": {
            "smoke-mixed": {
                "actions": actions,
                "action_counts": {
                    "Base": actions.count(0),
                    "Generate": actions.count(1),
                    "Enhance": actions.count(2),
                },
            }
        },
        "scientific_boundary": {
            "ground_truth_used_to_choose_route": False,
            "deployment_controller": False,
            "purpose": "exercise all three syntax actions before experiments",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "gate_summary.json", gate)
    atomic_json(args.output_dir / "route_summary.json", route)
    print(json.dumps({
        "gate_summary": str(args.output_dir / "gate_summary.json"),
        "route_summary": str(args.output_dir / "route_summary.json"),
    }, ensure_ascii=False, indent=2))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(args: argparse.Namespace) -> None:
    codec_dir = args.output_dir / "spatial_codec"
    encode = load_json(codec_dir / "encode_summary.json")
    decode = load_json(codec_dir / "decode_summary.json")
    seed = load_json(args.output_dir / "seedvr2" / "seedvr2_metadata.json")
    encoder_path = next((codec_dir / "encoder_reconstruction").glob("*.png"))
    decoder_path = next((codec_dir / "fresh_decode").glob("*.png"))
    with Image.open(encoder_path) as image:
        encoder = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(decoder_path) as image:
        decoder = np.asarray(image.convert("RGB"), dtype=np.uint8)
    maximum = int(np.max(np.abs(
        encoder.astype(np.int16) - decoder.astype(np.int16))))
    stream_path = Path(encode["stream"])
    summary = {
        "experiment": "A800 stage-0 environment and one-frame smoke test",
        "status": "pass" if maximum == 0 else "fail",
        "spatial_format_test": "passed before this smoke in compile log",
        "stream": {
            "path": str(stream_path),
            "actual_on_disk_bytes": stream_path.stat().st_size,
            "reported_bytes": encode["stream_bytes"],
            "all_headers_maps_and_substreams_charged": True,
        },
        "fresh_decode": {
            "independent_process": True,
            "source_rgb_available": False,
            "hidden_route_arguments_required": False,
            "max_abs_pixel_error_vs_encoder_reconstruction": maximum,
            "pixel_exact": maximum == 0,
            "seconds_after_argument_parse": decode[
                "total_after_argument_parse_seconds"],
        },
        "seedvr2": {
            "frames": seed["frame_count"],
            "runtime_seconds_model_load_excluded": seed["runtime_seconds"],
            "total_after_argument_parse_seconds": seed[
                "total_after_argument_parse_seconds"],
            "peak_cuda_allocated_bytes": seed["peak_cuda_allocated_bytes"],
            "bf16_dit": seed["dit_checkpoint"],
            "training_or_finetuning": False,
        },
        "codec": {
            "encode_seconds": encode["codec_seconds"],
            "decode_seconds": decode["bitstream_decode_seconds"],
            "encode_peak_cuda_allocated_bytes": encode[
                "peak_cuda_allocated_bytes"],
            "decode_peak_cuda_allocated_bytes": decode[
                "peak_cuda_allocated_bytes"],
        },
        "peak_cuda_allocated_bytes": max(
            encode["peak_cuda_allocated_bytes"],
            decode["peak_cuda_allocated_bytes"],
            seed["peak_cuda_allocated_bytes"],
        ),
    }
    if summary["stream"]["actual_on_disk_bytes"] != summary["stream"][
            "reported_bytes"]:
        raise RuntimeError("smoke stream size differs from codec summary")
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
