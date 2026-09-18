#!/usr/bin/env python3
"""Build the fixed frame-9 sheet for one independent-test sample."""

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


ACTION_COLORS = np.asarray([
    (74, 144, 226),
    (242, 160, 42),
    (70, 170, 92),
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--visual-frame", type=int, default=9)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generic(root: Path) -> tuple[list[np.ndarray], dict]:
    value = read(root / "evaluation" / "evaluation.json")
    return load_pngs(Path(value["output_frames"])), value["quality"]


def roi(root: Path) -> tuple[list[np.ndarray], dict]:
    value = read(root / "evaluation" / "summary.json")
    frames = load_pngs(
        root / "evaluation" / "frames" / "roi-spatial-bge-stitched")
    return frames, value["quality"]["roi-spatial-bge-stitched"]


def action_map(route_path: Path) -> np.ndarray:
    route = read(route_path)
    _, selected = selected_route_variant(route)
    rows, columns = route["configuration"]["tile_grid"]
    tile_size = int(route["configuration"]["tile_size"])
    actions = np.asarray(
        selected["actions"], dtype=np.int64).reshape(rows, columns)
    return np.repeat(
        np.repeat(ACTION_COLORS[actions], tile_size, axis=0),
        tile_size, axis=1)


def main() -> None:
    args = parse_args()
    root = args.sample_root
    gate_root = root / "uniform_gate"
    gate = read(gate_root / "summary.json")
    frame_index = min(max(args.visual_frame, 1), gate["frames"]) - 1
    reference = load_source(gate, gate["frames"])

    variants: list[tuple[str, list[np.ndarray], dict | None]] = [
        ("GT", reference, None),
    ]
    for qp in (8, 16, 24, 32):
        variants.append((
            f"Uniform QP{qp}",
            load_pngs(gate_root / "frames" / f"qp{qp}-base"),
            gate["variants"][f"qp{qp}-base"]["quality"],
        ))
    for qp in (16, 24):
        variants.append((
            f"QP{qp} + BasicVSR++",
            load_pngs(gate_root / "frames" / f"qp{qp}-basicvsrpp"),
            gate["variants"][f"qp{qp}-basicvsrpp"]["quality"],
        ))

    spatial = root / "spatial"
    frames, quality = generic(spatial / "all-generate")
    variants.append(("All Generate", frames, quality))
    frames, quality = generic(spatial / "enhance-only")
    variants.append(("Enhance-only", frames, quality))

    for budget, title in (("low", "Low"), ("middle", "Middle")):
        frames, quality = roi(spatial / f"{budget}-joint")
        variants.append((f"{title} G/B/E", frames, quality))
        frames, quality = generic(spatial / f"{budget}-no-generate")
        variants.append((f"{title}, no Generate", frames, quality))
        frames, quality = roi(spatial / f"{budget}-no-enhance")
        variants.append((f"{title}, no Enhance", frames, quality))
        variants.append((
            f"{title} action map (B/G/E)",
            [action_map(root / "routes" / budget / "learned-joint.json")]
            * gate["frames"],
            None,
        ))

    if len(variants) != 17:
        raise RuntimeError(f"expected 17 visual panels, found {len(variants)}")
    if any(len(frames) != gate["frames"] for _, frames, _ in variants):
        raise RuntimeError("independent-test visual has a frame-count mismatch")

    panels = [
        panel(frames[frame_index], name, quality)
        for name, frames, quality in variants
    ]
    columns = 5
    rows = (len(panels) + columns - 1) // columns
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new(
        "RGB", (columns * width, rows * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(
            item,
            ((index % columns) * width, (index // columns) * height))

    output_dir = root / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual = output_dir / f"independent_frame_{frame_index + 1:05d}.png"
    canvas.save(visual, optimize=True)
    result = {
        "experiment": "A800 independent-test fixed visual comparison",
        "one_based_frame": frame_index + 1,
        "visual": str(visual.resolve()),
        "panel_order": [name for name, _, _ in variants],
        "action_colors_rgb": {
            "Base": ACTION_COLORS[0].tolist(),
            "Generate": ACTION_COLORS[1].tolist(),
            "Enhance": ACTION_COLORS[2].tolist(),
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
