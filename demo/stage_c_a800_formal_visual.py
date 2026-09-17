#!/usr/bin/env python3
"""Build the fixed, method-complete visual sheet for one A800 dev clip."""

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
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--visual-frame", type=int, default=9)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    gate_root = args.sample_root / "uniform_gate"
    gate = read(gate_root / "summary.json")
    frame_index = min(max(args.visual_frame, 1), gate["frames"]) - 1
    reference = load_source(gate, gate["frames"])
    spatial = args.sample_root / "spatial"
    seed_eval = read(
        args.sample_root / "ordinary_qp8_seed" / "evaluation" / "evaluation.json")
    enhance = read(spatial / "enhance-only" / "evaluation" / "evaluation.json")
    all_generate = read(
        spatial / "all-generate" / "evaluation" / "evaluation.json")
    joint = read(spatial / "learned-joint" / "evaluation" / "summary.json")
    no_generate = read(
        spatial / "same-route-no-generate" / "evaluation" / "evaluation.json")
    no_enhance = read(
        spatial / "same-route-no-enhance" / "evaluation" / "summary.json")

    variants = [
        ("GT", reference, None),
        ("Uniform QP8", load_pngs(gate_root / "frames" / "qp8-base"),
         gate["variants"]["qp8-base"]["quality"]),
        ("Uniform QP16", load_pngs(gate_root / "frames" / "qp16-base"),
         gate["variants"]["qp16-base"]["quality"]),
        ("Uniform QP32", load_pngs(gate_root / "frames" / "qp32-base"),
         gate["variants"]["qp32-base"]["quality"]),
        ("QP8 + BasicVSR++",
         load_pngs(gate_root / "frames" / "qp8-basicvsrpp"),
         gate["variants"]["qp8-basicvsrpp"]["quality"]),
        ("Ordinary QP8 + SeedVR2",
         load_pngs(args.sample_root / "ordinary_qp8_seed" / "restored"),
         seed_eval["variants"]["QP8 SeedVR2"]["quality"]),
        ("Enhance-only",
         load_pngs(spatial / "enhance-only" / "codec" / "fresh_decode"),
         enhance["quality"]),
        ("All Generate",
         load_pngs(spatial / "all-generate" / "seedvr2"),
         all_generate["quality"]),
        ("Learned G/B/E",
         load_pngs(spatial / "learned-joint" / "evaluation" / "frames" /
                   "roi-spatial-bge-stitched"),
         joint["quality"]["roi-spatial-bge-stitched"]),
        ("Same route, no Generate",
         load_pngs(spatial / "same-route-no-generate" / "codec" /
                   "fresh_decode"),
         no_generate["quality"]),
        ("Same route, no Enhance",
         load_pngs(spatial / "same-route-no-enhance" / "evaluation" / "frames" /
                   "roi-spatial-bge-stitched"),
         no_enhance["quality"]["roi-spatial-bge-stitched"]),
    ]
    if any(len(frames) != gate["frames"] for _, frames, _ in variants):
        raise ValueError("fixed visual has a frame-count mismatch")

    route = read(args.sample_root / "routes" / "learned-joint.json")
    _, selected = selected_route_variant(route)
    actions = np.asarray(selected["actions"], dtype=np.int64).reshape(4, 4)
    colors = np.asarray([
        (74, 144, 226),
        (242, 160, 42),
        (70, 170, 92),
    ], dtype=np.uint8)
    action_map = np.repeat(np.repeat(colors[actions], 128, axis=0), 128, axis=1)

    panels = [
        panel(frames[frame_index], name, quality)
        for name, frames, quality in variants
    ]
    panels.append(panel(action_map, "Learned action map (B/G/E)", None))
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new(
        "RGB", (columns * width, rows * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % columns) * width, (index // columns) * height))

    output_dir = args.sample_root / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual = output_dir / f"formal_frame_{frame_index + 1:05d}.png"
    canvas.save(visual, optimize=True)
    result = {
        "experiment": "A800 fixed formal visual comparison",
        "one_based_frame": frame_index + 1,
        "visual": str(visual.resolve()),
        "panel_order": [name for name, _, _ in variants] + [
            "Learned action map (B/G/E)"],
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
