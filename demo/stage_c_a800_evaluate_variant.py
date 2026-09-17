#!/usr/bin/env python3
"""Evaluate one formal A800 route from its real stream and fresh decode."""

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


from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import (
    exact_frame_comparison,
    local_metrics,
    panel,
)
from demo.stage_c_spatial_quality_codec import selected_route_variant
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant


ACTION_COLORS = {
    0: (74, 144, 226),
    1: (242, 160, 42),
    2: (70, 170, 92),
}
ACTION_NAMES = {0: "Base", 1: "Generate", 2: "Enhance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an A800 formal route with actual stream bytes")
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--restoration-metadata", type=Path)
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-frame", type=int, default=9)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def action_map_image(actions: np.ndarray, tile_size: int) -> Image.Image:
    rows, columns = actions.shape
    image = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    for row in range(rows):
        for column in range(columns):
            action = int(actions[row, column])
            x0, y0 = column * tile_size, row * tile_size
            x1, y1 = x0 + tile_size, y0 + tile_size
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=ACTION_COLORS[action])
            label = ACTION_NAMES[action][0]
            draw.text((x0 + 8, y0 + 7), label, fill="white", font=font)
    return image


def save_visual(
    path: Path, frame_index: int, reference: list[np.ndarray],
    decoded: list[np.ndarray], output: list[np.ndarray], actions: np.ndarray,
    tile_size: int, raw_quality: dict, output_quality: dict, label: str,
) -> None:
    items = [
        panel(reference[frame_index], "GT", None),
        panel(decoded[frame_index], "Fresh spatial decode", raw_quality),
        panel(output[frame_index], label, output_quality),
        panel(np.asarray(action_map_image(actions, tile_size)), "Action map", None),
    ]
    width = max(item.width for item in items)
    height = max(item.height for item in items)
    canvas = Image.new("RGB", (2 * width, 2 * height), (230, 230, 230))
    for index, item in enumerate(items):
        canvas.paste(item, ((index % 2) * width, (index // 2) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    route = json.loads(args.route_summary.read_text(encoding="utf-8"))
    encode = json.loads(
        (args.codec_dir / "encode_summary.json").read_text(encoding="utf-8"))
    decode = json.loads(
        (args.codec_dir / "decode_summary.json").read_text(encoding="utf-8"))
    stream = Path(encode["stream"])
    if not stream.is_absolute():
        stream = args.codec_dir / stream
    actual_bytes = stream.stat().st_size
    if actual_bytes != int(encode["stream_bytes"]):
        raise RuntimeError("encode summary byte count differs from on-disk stream")
    if actual_bytes != int(decode["stream_bytes"]):
        raise RuntimeError("decode summary byte count differs from on-disk stream")

    frame_count = int(decode["frames"])
    reference = load_source(gate, frame_count)
    encoder = load_pngs(args.codec_dir / "encoder_reconstruction")
    decoded = load_pngs(args.codec_dir / "fresh_decode")
    frames_dir = args.frames_dir or args.codec_dir / "fresh_decode"
    output = load_pngs(frames_dir)
    if not all(len(value) == frame_count for value in (reference, encoder, decoded, output)):
        raise ValueError("formal variant frame counts differ")

    selected_name, selected = selected_route_variant(route)
    rows, columns = route["configuration"]["tile_grid"]
    tile_size = int(route["configuration"]["tile_size"])
    actions = np.asarray(selected["actions"], dtype=np.int64).reshape(rows, columns)
    metric = LPIPSAlex(True)
    raw_quality = evaluate_variant(reference, decoded, metric)
    output_quality = evaluate_variant(reference, output, metric)
    fresh_regression = exact_frame_comparison(encoder, decoded)
    if not fresh_regression.get("pixel_exact"):
        raise RuntimeError("fresh spatial decode differs from encoder reconstruction")

    restoration_path = args.restoration_metadata
    if restoration_path is None:
        candidate = frames_dir / "seedvr2_metadata.json"
        restoration_path = candidate if candidate.is_file() else None
    restoration = None
    if restoration_path is not None:
        restoration = json.loads(restoration_path.read_text(encoding="utf-8"))

    decode_total = float(decode["total_after_argument_parse_seconds"])
    restoration_total = (
        float(restoration["total_after_argument_parse_seconds"])
        if restoration is not None else 0.0)
    peak = int(decode["peak_cuda_allocated_bytes"])
    if restoration is not None:
        peak = max(peak, int(restoration["peak_cuda_allocated_bytes"]))

    visual_frame = min(max(args.visual_frame, 1), frame_count) - 1
    visual = args.output_dir / "visuals" / "gt_codec_output_action_map.png"
    save_visual(
        visual, visual_frame, reference, decoded, output, actions, tile_size,
        raw_quality, output_quality, args.variant_label)
    result = {
        "experiment": "A800 formal real-stream variant evaluation",
        "variant": args.variant_label,
        "source_role": gate["source_role"],
        "frames": frame_count,
        "route": {
            "summary": str(args.route_summary.resolve()),
            "selected_variant": selected_name,
            "kind": route.get("route_kind"),
            "actions": actions.tolist(),
            "action_counts": {
                name: int(np.count_nonzero(actions == action))
                for action, name in ACTION_NAMES.items()
            },
        },
        "stream": {
            "path": str(stream.resolve()),
            "actual_on_disk_bytes": actual_bytes,
            "bpp": 8.0 * actual_bytes / (frame_count * encode["width"] * encode["height"]),
            "sps_bytes": encode["sps_bytes"],
            "coding_units": encode["unit_bytes"],
            "maps_headers_and_all_substreams_charged": True,
        },
        "fresh_decode_regression": fresh_regression,
        "quality": output_quality,
        "codec_only_quality": raw_quality,
        "local_quality_by_action": local_metrics(
            metric, reference, output, actions, tile_size),
        "runtime": {
            "codec_encode_seconds": encode["codec_seconds"],
            "codec_encode_process_seconds": encode["total_after_argument_parse_seconds"],
            "codec_bitstream_decode_seconds": decode["bitstream_decode_seconds"],
            "codec_decode_process_seconds": decode_total,
            "restoration_core_seconds": (
                restoration["runtime_seconds"] if restoration is not None else None),
            "restoration_process_seconds": restoration_total if restoration else None,
            "complete_fresh_decode_pipeline_process_seconds": (
                decode_total + restoration_total),
            "peak_cuda_allocated_bytes": peak,
        },
        "scientific_boundary": {
            "actual_spatial_entropy_stream": True,
            "fresh_decoder_reads_source_rgb": False,
            "route_uses_ground_truth_metrics": False,
            "different_qp_latents_spliced": False,
            "area_prorated_bytes": False,
            "lpips_is_primary": True,
            "training_or_finetuning": False,
        },
        "output_frames": str(frames_dir.resolve()),
        "visual": str(visual.resolve()),
    }
    atomic_json(args.output_dir / "evaluation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
