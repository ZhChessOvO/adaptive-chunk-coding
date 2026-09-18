#!/usr/bin/env python3
"""Build the fixed frame-9 comparison sheet for low-budget controller v3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import panel
from demo.stage_c_spatial_quality_codec import selected_route_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sample-root", type=Path, required=True)
    parser.add_argument("--v1-followup-sample-root", type=Path, required=True)
    parser.add_argument("--v3-sample-root", type=Path, required=True)
    parser.add_argument("--visual-frame", type=int, default=9)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def roi_frames(root: Path) -> list[np.ndarray]:
    return load_pngs(
        root / "evaluation" / "frames" / "roi-spatial-bge-stitched")


def roi_quality(root: Path) -> dict:
    return read(root / "evaluation" / "summary.json")["quality"][
        "roi-spatial-bge-stitched"]


def main() -> None:
    args = parse_args()
    baseline = args.baseline_sample_root
    previous = args.v1_followup_sample_root
    current = args.v3_sample_root
    gate_root = baseline / "uniform_gate"
    gate = read(gate_root / "summary.json")
    frame_index = min(max(args.visual_frame, 1), gate["frames"]) - 1
    reference = load_source(gate, gate["frames"])
    baseline_spatial = baseline / "spatial"
    previous_spatial = previous / "spatial"
    current_spatial = current / "spatial"
    all_generate_eval = read(
        baseline_spatial / "all-generate" / "evaluation" / "evaluation.json")
    no_generate_eval = read(
        current_spatial / "low-no-generate" / "evaluation" /
        "evaluation.json")
    variants = [
        ("GT", reference, None),
        ("Uniform QP8", load_pngs(gate_root / "frames" / "qp8-base"),
         gate["variants"]["qp8-base"]["quality"]),
        ("Uniform QP16", load_pngs(gate_root / "frames" / "qp16-base"),
         gate["variants"]["qp16-base"]["quality"]),
        ("Uniform QP24", load_pngs(gate_root / "frames" / "qp24-base"),
         gate["variants"]["qp24-base"]["quality"]),
        ("Uniform QP32", load_pngs(gate_root / "frames" / "qp32-base"),
         gate["variants"]["qp32-base"]["quality"]),
        ("All Generate", load_pngs(
            baseline_spatial / "all-generate" / "seedvr2"),
         all_generate_eval["quality"]),
        ("Low v1 G/B/E", roi_frames(previous_spatial / "low-joint"),
         roi_quality(previous_spatial / "low-joint")),
        ("Low v3 Base-probe", roi_frames(current_spatial / "low-joint"),
         roi_quality(current_spatial / "low-joint")),
        ("Low v3, no Generate", load_pngs(
            current_spatial / "low-no-generate" / "codec" / "fresh_decode"),
         no_generate_eval["quality"]),
        ("Low v3, no Enhance", roi_frames(
            current_spatial / "low-no-enhance"),
         roi_quality(current_spatial / "low-no-enhance")),
    ]
    if any(len(frames) != gate["frames"] for _, frames, _ in variants):
        raise RuntimeError("v3 visual has a frame-count mismatch")

    route = read(current / "routes" / "learned-joint.json")
    _, selected = selected_route_variant(route)
    rows, columns = route["configuration"]["tile_grid"]
    tile_size = int(route["configuration"]["tile_size"])
    actions = np.asarray(
        selected["actions"], dtype=np.int64).reshape(rows, columns)
    colors = np.asarray([
        (74, 144, 226),
        (242, 160, 42),
        (70, 170, 92),
    ], dtype=np.uint8)
    action_map = np.repeat(
        np.repeat(colors[actions], tile_size, axis=0), tile_size, axis=1)
    panels = [
        panel(frames[frame_index], name, quality)
        for name, frames, quality in variants
    ]
    panels.append(panel(action_map, "Low v3 action map (B/G/E)", None))
    sheet_columns = 4
    sheet_rows = (len(panels) + sheet_columns - 1) // sheet_columns
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new(
        "RGB", (sheet_columns * width, sheet_rows * height),
        (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, (
            (index % sheet_columns) * width,
            (index // sheet_columns) * height,
        ))
    output_dir = current / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual = output_dir / f"low_budget_v3_frame_{frame_index + 1:05d}.png"
    canvas.save(visual, optimize=True)
    result = {
        "experiment": "A800 low-budget controller v3 fixed visual",
        "one_based_frame": frame_index + 1,
        "visual": str(visual.resolve()),
        "panel_order": [name for name, _, _ in variants] + [
            "Low v3 action map (B/G/E)"],
        "action_colors_rgb": {
            "Base": colors[0].tolist(),
            "Generate": colors[1].tolist(),
            "Enhance": colors[2].tolist(),
        },
    }
    manifest = output_dir / "manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
