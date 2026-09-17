#!/usr/bin/env python3
"""Evaluate lossless SeedVR2 outputs against an E20 restoration-gate clip."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--seedvr2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-frame", type=int, default=3)
    return parser.parse_args()


def load_pngs(path: Path) -> list[np.ndarray]:
    paths = sorted(path.glob("*.png"))
    if not paths:
        raise ValueError(f"no PNG frames in {path}")
    return [
        np.asarray(Image.open(item).convert("RGB"), dtype=np.uint8).copy()
        for item in paths
    ]


def load_source(summary: dict, count: int) -> list[np.ndarray]:
    crop = summary["crop"]
    output = []
    for value in summary["source_files"][:count]:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        output.append(frame[
            crop["y"]:crop["y"] + crop["height"],
            crop["x"]:crop["x"] + crop["width"],
        ].copy())
    if len(output) != count:
        raise ValueError("gate summary does not contain enough source frames")
    return output


def panel(frame: np.ndarray, title: str, quality: dict | None) -> Image.Image:
    image = Image.fromarray(frame)
    banner_height = 68
    result = Image.new("RGB", (image.width, image.height + banner_height), "white")
    result.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(result)
    title_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    metric_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    draw.text((8, 7), title, fill="black", font=title_font)
    if quality is not None:
        draw.text(
            (8, 37),
            f"PSNR {quality['psnr_db']:.3f} dB | "
            f"LPIPS {quality['lpips_alex']:.4f} | "
            f"T-MAE {quality['temporal_delta_mae']:.3f}",
            fill="black",
            font=metric_font,
        )
    return result


def write_visual(
    path: Path,
    frame_index: int,
    reference: list[np.ndarray],
    variants: dict[str, dict],
) -> None:
    images = [panel(reference[frame_index], "GT", None)]
    for name, record in variants.items():
        images.append(panel(record["frames"][frame_index], name, record["quality"]))
    columns = 2
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (230, 230, 230))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_temporal_visual(
    path: Path,
    center_index: int,
    reference: list[np.ndarray],
    variants: dict[str, dict],
) -> None:
    start = max(0, center_index - 1)
    stop = min(len(reference), center_index + 2)
    indices = list(range(start, stop))
    rows = [("GT", {"frames": reference, "quality": None}), *variants.items()]
    panels = [
        panel(record["frames"][frame_index], f"{name} | frame {frame_index + 1}",
              record["quality"])
        for name, record in rows
        for frame_index in indices
    ]
    width = max(image.width for image in panels)
    height = max(image.height for image in panels)
    sheet = Image.new(
        "RGB", (len(indices) * width, len(rows) * height), (230, 230, 230))
    for index, image in enumerate(panels):
        sheet.paste(
            image,
            ((index % len(indices)) * width, (index // len(indices)) * height),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    if args.qp not in gate["configuration"]["qps"]:
        raise ValueError(f"QP{args.qp} is absent from the gate summary")
    seed_frames = load_pngs(args.seedvr2_dir)
    count = len(seed_frames)
    if count > gate["frames"]:
        raise ValueError("SeedVR2 output is longer than the E20 clip")
    base_dir = args.gate_summary.parent / "frames" / f"qp{args.qp}-base"
    basic_dir = args.gate_summary.parent / "frames" / f"qp{args.qp}-basicvsrpp"
    base_frames = load_pngs(base_dir)[:count]
    basic_frames = load_pngs(basic_dir)[:count]
    reference = load_source(gate, count)
    expected_shape = reference[0].shape
    for name, frames in (
        ("Base", base_frames),
        ("BasicVSR++", basic_frames),
        ("SeedVR2", seed_frames),
    ):
        if len(frames) != count or any(frame.shape != expected_shape for frame in frames):
            raise ValueError(f"{name} frame count or shape differs from GT")

    lpips = LPIPSAlex(True)
    variants = {
        f"QP{args.qp} Base": {"frames": base_frames},
        f"QP{args.qp} BasicVSR++": {"frames": basic_frames},
        f"QP{args.qp} SeedVR2": {"frames": seed_frames},
    }
    for record in variants.values():
        record["quality"] = evaluate_variant(reference, record["frames"], lpips)

    metadata_path = args.seedvr2_dir / "seedvr2_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_runtime = gate["variants"][f"qp{args.qp}-base"]["runtime"]
    full_clip = count == gate["frames"]
    result = {
        "experiment": "E20 SeedVR2 restoration gate evaluation",
        "qp": args.qp,
        "frames": count,
        "full_gate_clip": full_clip,
        "source_role": gate["source_role"],
        "variants": {
            name: {"quality": record["quality"]}
            for name, record in variants.items()
        },
        "seedvr2_vs_base": {
            key: (
                variants[f"QP{args.qp} SeedVR2"]["quality"][metric]
                - variants[f"QP{args.qp} Base"]["quality"][metric]
            )
            for key, metric in (
                ("psnr_delta_db", "psnr_db"),
                ("lpips_delta", "lpips_alex"),
                ("temporal_delta_mae_change", "temporal_delta_mae"),
            )
        },
        "rate": {
            "dcvc_uf_file_bytes": (
                gate["variants"][f"qp{args.qp}-base"]["rate"]["total_bytes"]
                if full_clip else None),
            "seedvr2_additional_stream_bytes": 0,
            "partial_clip_rate_warning": (
                None if full_clip else
                "The stored DCVC-UF stream has more frames; do not quote its bytes for this smoke clip."),
        },
        "runtime": {
            "dcvc_fresh_decode_seconds": (
                base_runtime["fresh_decode_seconds_median"] if full_clip else None),
            "seedvr2_seconds": metadata["runtime_seconds"],
            "sequential_total_seconds": (
                base_runtime["fresh_decode_seconds_median"]
                + metadata["runtime_seconds"] if full_clip else None),
            "seedvr2_peak_cuda_allocated_bytes": metadata[
                "peak_cuda_allocated_bytes"],
            "combined_pipeline_peak_not_yet_measured": True,
        },
        "seedvr2_metadata": metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "evaluation.json"
    frame_index = min(max(args.visual_frame, 1), count) - 1
    visual_path = args.output_dir / "visuals" / f"frame_{frame_index + 1:05d}.png"
    write_visual(visual_path, frame_index, reference, variants)
    temporal_path = args.output_dir / "visuals" / "three_consecutive_frames.png"
    write_temporal_visual(
        temporal_path, frame_index, reference, variants)
    atomic_json(output_path, result)
    print(json.dumps({
        "evaluation": str(output_path),
        "visual": str(visual_path),
        "temporal_visual": str(temporal_path),
        "seedvr2_vs_base": result["seedvr2_vs_base"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
