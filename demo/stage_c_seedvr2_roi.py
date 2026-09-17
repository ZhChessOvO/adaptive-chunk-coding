#!/usr/bin/env python3
"""Prepare and evaluate connected-component SeedVR2 Generate ROIs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import (
    exact_frame_comparison,
    generate_composite,
    panel,
)
from demo.stage_c_spatial_quality_codec import selected_route_variant
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant
from demo.stage_c_a800_teacher import PersistentSeedVR2


ACTION_GENERATE = 1


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--route-summary", type=Path, required=True)
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--context-pixels", type=int, default=64)
    prepare.add_argument("--processing-scale", type=float, default=1.5)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--output-root", type=Path, required=True)
    restore.add_argument("--seed", type=int, required=True)
    restore.add_argument(
        "--upstream-root", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2")
    restore.add_argument(
        "--dit-checkpoint", type=Path,
        default=(REPO_ROOT / "third_party" / "SeedVR2" / "ckpts" /
                 "seedvr2_ema_3b_bf16.safetensors"))
    restore.add_argument(
        "--vae-checkpoint", type=Path,
        default=(REPO_ROOT / "third_party" / "SeedVR2" / "ckpts" /
                 "ema_vae.pth"))
    restore.add_argument(
        "--positive-embedding", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2" / "pos_emb.pt")
    restore.add_argument(
        "--negative-embedding", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2" / "neg_emb.pt")
    restore.add_argument("--sample-steps", type=int, default=1)
    restore.add_argument("--cfg-scale", type=float, default=1.0)
    restore.add_argument(
        "--dit-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    restore.add_argument("--cuda-idx", type=int, default=0)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--gate-summary", type=Path, required=True)
    evaluate.add_argument("--codec-dir", type=Path, required=True)
    evaluate.add_argument("--restored-root", type=Path, required=True)
    evaluate.add_argument(
        "--full-seedvr2-dir", type=Path,
        help=("Optional full-frame SeedVR2 control. The connected-ROI result "
              "remains evaluable without running this extra control."))
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--feather-pixels", type=int, default=8)
    return parser.parse_args()


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=np.bool_)
    output = []
    for start_row, start_column in zip(*np.where(mask)):
        if seen[start_row, start_column]:
            continue
        queue = deque([(int(start_row), int(start_column))])
        seen[start_row, start_column] = True
        component = []
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for next_row, next_column in (
                (row - 1, column), (row + 1, column),
                (row, column - 1), (row, column + 1),
            ):
                if (0 <= next_row < mask.shape[0]
                        and 0 <= next_column < mask.shape[1]
                        and mask[next_row, next_column]
                        and not seen[next_row, next_column]):
                    seen[next_row, next_column] = True
                    queue.append((next_row, next_column))
        output.append(component)
    return output


def round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def prepare(args: argparse.Namespace) -> None:
    if args.context_pixels < 0 or args.processing_scale <= 0:
        raise ValueError("context and processing scale must be nonnegative/positive")
    route = json.loads(args.route_summary.read_text(encoding="utf-8"))
    frames = load_pngs(args.input_dir)
    height, width = frames[0].shape[:2]
    tile_size = int(route["configuration"]["tile_size"])
    rows, columns = route["configuration"]["tile_grid"]
    variant_name, variant = selected_route_variant(route)
    actions = np.asarray(variant["actions"], dtype=np.int64).reshape(rows, columns)
    components = connected_components(actions == ACTION_GENERATE)
    records = []
    for index, component in enumerate(components):
        component_rows = [item[0] for item in component]
        component_columns = [item[1] for item in component]
        x0 = max(0, min(component_columns) * tile_size - args.context_pixels)
        y0 = max(0, min(component_rows) * tile_size - args.context_pixels)
        x1 = min(width, (max(component_columns) + 1) * tile_size + args.context_pixels)
        y1 = min(height, (max(component_rows) + 1) * tile_size + args.context_pixels)
        # All current cells/context are multiples of 16.  Keep that explicit
        # because SeedVR2 requires a multiple-of-16 output geometry.
        if any(value % 16 for value in (x0, y0, x1, y1)):
            raise ValueError("ROI bounds must be divisible by 16")
        crop_dir = args.output_dir / f"component_{index:02d}" / "input"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(frames, start=1):
            Image.fromarray(frame[y0:y1, x0:x1]).save(
                crop_dir / f"im{frame_index:05d}.png")
        crop_height, crop_width = y1 - y0, x1 - x0
        records.append({
            "index": index,
            "cells": component,
            "crop": {"x": x0, "y": y0, "width": crop_width, "height": crop_height},
            "input_dir": str(crop_dir),
            "expected_restored_dir": str(
                args.output_dir / f"component_{index:02d}" / "restored"),
            "processing_height": round_up(crop_height * args.processing_scale, 16),
            "processing_width": round_up(crop_width * args.processing_scale, 16),
        })
    total_processing_pixels = sum(
        record["processing_height"] * record["processing_width"]
        for record in records)
    full_processing_height = round_up(height * args.processing_scale, 16)
    full_processing_width = round_up(width * args.processing_scale, 16)
    result = {
        "route_summary": str(args.route_summary),
        "route_variant": variant_name,
        "input_dir": str(args.input_dir),
        "frame_count": len(frames),
        "frame_height": height,
        "frame_width": width,
        "tile_size": tile_size,
        "actions": actions.tolist(),
        "context_pixels": args.context_pixels,
        "processing_scale": args.processing_scale,
        "components": records,
        "total_roi_processing_pixels_per_frame": total_processing_pixels,
        "full_processing_pixels_per_frame": full_processing_height * full_processing_width,
        "roi_to_full_processing_pixel_ratio": (
            total_processing_pixels / (full_processing_height * full_processing_width)),
        "scientific_boundary": {
            "roi_is_derived_only_from_transmitted_action_map": True,
            "source_rgb_used_for_roi_geometry": False,
            "model_weights_changed": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    atomic_json(manifest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def save_frames(path: Path, frames: list[np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        Image.fromarray(frame).save(path / f"im{index:05d}.png")


def valid_component_metadata(
    path: Path, record: dict, seed: int, frame_count: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    crop = record["crop"]
    return (
        value.get("persistent_roi_session") is True
        and value.get("seed") == seed
        and value.get("frame_count") == frame_count
        and value.get("processing_height") == record["processing_height"]
        and value.get("processing_width") == record["processing_width"]
        and value.get("output_height") == crop["height"]
        and value.get("output_width") == crop["width"]
    )


def valid_batch_metadata(
    path: Path, manifest_path: Path, output_root: Path,
    components: list[dict], seed: int, frame_count: int,
) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not value.get("complete")
        or value.get("component_count") != len(components)
        or value.get("completed_component_count") != len(components)
        or value.get("base_seed") != seed
        or Path(value.get("manifest", "")).resolve() != manifest_path.resolve()
    ):
        return None
    for record in components:
        index = int(record["index"])
        component_seed = seed + index * 100003
        metadata_path = (
            output_root / f"component_{index:02d}" / "restored" /
            "seedvr2_metadata.json")
        if not valid_component_metadata(
                metadata_path, record, component_seed, frame_count):
            return None
    return value


def restore(args: argparse.Namespace) -> None:
    process_started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    components = manifest["components"]
    frame_count = int(manifest["frame_count"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    batch_metadata_path = args.output_root / "roi_batch_metadata.json"
    existing = valid_batch_metadata(
        batch_metadata_path, args.manifest, args.output_root, components,
        args.seed, frame_count)
    if existing is not None:
        print(json.dumps({
            "stage": "persistent-roi-batch-resume-skip",
            "output": str(batch_metadata_path),
            "total_after_argument_parse_seconds": existing[
                "total_after_argument_parse_seconds"],
        }), flush=True)
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return

    # A batch without a valid final marker may have been interrupted after one
    # or more component files were written.  Recompute the whole small batch in
    # one process so its outputs and wall-clock measurement describe the same
    # execution.  Completed batches remain immediately resumable above.
    model = None
    model_load_seconds = 0.0
    if components:
        torch.cuda.set_device(args.cuda_idx)
        model = PersistentSeedVR2(args)
        model_load_seconds = model.model_load_seconds

    completed = []
    peak = 0
    inference_seconds = 0.0
    component_wall_seconds = 0.0
    for record in components:
        index = int(record["index"])
        component_seed = args.seed + index * 100003
        output_dir = args.output_root / f"component_{index:02d}" / "restored"
        metadata_path = output_dir / "seedvr2_metadata.json"
        if model is None:
            raise RuntimeError("ROI component has no loaded model")
        component_started = time.perf_counter()
        frames = load_pngs(Path(record["input_dir"]))
        crop = record["crop"]
        restored, runtime = model.restore(
            frames,
            component_seed,
            processing_height=int(record["processing_height"]),
            processing_width=int(record["processing_width"]),
            output_height=int(crop["height"]),
            output_width=int(crop["width"]),
        )
        save_frames(output_dir, restored)
        wall_seconds = time.perf_counter() - component_started
        metadata = {
            "model": "SeedVR2-3B persistent connected-ROI session",
            "dit_checkpoint": str(args.dit_checkpoint),
            "input_dir": record["input_dir"],
            "output_dir": str(output_dir),
            "frame_count": len(restored),
            "input_height": int(crop["height"]),
            "input_width": int(crop["width"]),
            "processing_height": int(record["processing_height"]),
            "processing_width": int(record["processing_width"]),
            "output_height": int(crop["height"]),
            "output_width": int(crop["width"]),
            "seed": component_seed,
            "sample_steps": args.sample_steps,
            "cfg_scale": args.cfg_scale,
            "dit_dtype": args.dit_dtype,
            "runtime_seconds": runtime["seconds_model_load_excluded"],
            "component_wall_seconds": wall_seconds,
            "total_after_argument_parse_seconds": wall_seconds,
            "peak_cuda_allocated_bytes": runtime[
                "peak_cuda_allocated_bytes"],
            "model_load_in_timing": False,
            "persistent_roi_session": True,
            "shared_model_load_seconds": model_load_seconds,
            "input_and_output_png_lossless": True,
            "training_or_finetuning": False,
            "actual_compute_scope": runtime["actual_compute_scope"],
        }
        atomic_json(metadata_path, metadata)
        print(json.dumps({
            "stage": "persistent-roi-component-complete",
            "component": index,
            "runtime_seconds": metadata["runtime_seconds"],
            "component_wall_seconds": wall_seconds,
            "peak_cuda_mib": metadata["peak_cuda_allocated_bytes"] / 1048576,
        }), flush=True)
        inference_seconds += float(metadata["runtime_seconds"])
        component_wall_seconds += float(metadata.get(
            "component_wall_seconds", metadata["total_after_argument_parse_seconds"]))
        peak = max(peak, int(metadata["peak_cuda_allocated_bytes"]))
        completed.append({
            "index": index,
            "metadata": str(metadata_path),
            "runtime_seconds": metadata["runtime_seconds"],
            "component_wall_seconds": metadata.get(
                "component_wall_seconds",
                metadata["total_after_argument_parse_seconds"]),
        })

    result = {
        "experiment": "persistent single-process SeedVR2 ROI restoration",
        "manifest": str(args.manifest),
        "base_seed": args.seed,
        "component_count": len(components),
        "completed_component_count": len(completed),
        "complete": len(completed) == len(components),
        "model_load_seconds_this_process": model_load_seconds,
        "component_inference_seconds_sum": inference_seconds,
        "component_wall_seconds_sum": component_wall_seconds,
        "total_after_argument_parse_seconds": (
            time.perf_counter() - process_started),
        "peak_cuda_allocated_bytes": peak,
        "components": completed,
        "scientific_boundary": {
            "one_model_load_shared_by_all_pending_components": True,
            "components_processed_sequentially_in_one_process": True,
            "component_seeds_match_legacy_runner": True,
            "training_or_finetuning": False,
        },
    }
    atomic_json(batch_metadata_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    decode = json.loads((args.codec_dir / "decode_summary.json").read_text())
    reference = load_source(gate, manifest["frame_count"])
    encoder_reconstruction = load_pngs(args.codec_dir / "encoder_reconstruction")
    decoded = load_pngs(args.codec_dir / "fresh_decode")
    fresh_regression = exact_frame_comparison(encoder_reconstruction, decoded)
    if not fresh_regression.get("pixel_exact"):
        raise RuntimeError("fresh spatial decode differs from encoder reconstruction")
    full_generated = (
        load_pngs(args.full_seedvr2_dir)
        if args.full_seedvr2_dir is not None else None)
    actions = np.asarray(manifest["actions"], dtype=np.int64)
    tile_size = int(manifest["tile_size"])
    generated_canvas = [frame.copy() for frame in decoded]
    roi_core_seconds = 0.0
    roi_total_seconds = 0.0
    roi_peak = 0
    component_runtime = []
    for record in manifest["components"]:
        restored_dir = args.restored_root / f"component_{record['index']:02d}" / "restored"
        restored = load_pngs(restored_dir)
        metadata = json.loads((restored_dir / "seedvr2_metadata.json").read_text())
        crop = record["crop"]
        x, y = crop["x"], crop["y"]
        height, width = crop["height"], crop["width"]
        for frame_index, frame in enumerate(restored):
            generated_canvas[frame_index][y:y + height, x:x + width] = frame
        roi_core_seconds += float(metadata["runtime_seconds"])
        roi_total_seconds += float(metadata["total_after_argument_parse_seconds"])
        roi_peak = max(roi_peak, int(metadata["peak_cuda_allocated_bytes"]))
        component_runtime.append({
            "index": record["index"],
            "runtime_seconds_model_load_excluded": metadata["runtime_seconds"],
            "total_after_argument_parse_seconds": metadata[
                "total_after_argument_parse_seconds"],
            "peak_cuda_allocated_bytes": metadata["peak_cuda_allocated_bytes"],
        })
    batch_metadata_path = args.restored_root / "roi_batch_metadata.json"
    batch_metadata = (
        json.loads(batch_metadata_path.read_text(encoding="utf-8"))
        if batch_metadata_path.is_file() else None)
    if batch_metadata is not None:
        if (not batch_metadata.get("complete")
                or batch_metadata.get("component_count")
                != len(manifest["components"])):
            raise RuntimeError("persistent ROI batch metadata is incomplete")
        roi_total_seconds = float(
            batch_metadata["total_after_argument_parse_seconds"])
        roi_peak = max(
            roi_peak, int(batch_metadata["peak_cuda_allocated_bytes"]))
    started = time.perf_counter()
    stitched = generate_composite(
        decoded, generated_canvas, actions, tile_size, args.feather_pixels)
    composite_seconds = time.perf_counter() - started
    full_stitched = None
    full_composite_seconds = None
    if full_generated is not None:
        full_started = time.perf_counter()
        full_stitched = generate_composite(
            decoded, full_generated, actions, tile_size, args.feather_pixels)
        full_composite_seconds = time.perf_counter() - full_started
    output_frames = args.output_dir / "frames" / "roi-spatial-bge-stitched"
    output_frames.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(stitched, start=1):
        Image.fromarray(frame).save(output_frames / f"im{index:05d}.png")
    metric = LPIPSAlex(True)
    roi_quality = evaluate_variant(reference, stitched, metric)
    full_quality = (
        evaluate_variant(reference, full_generated, metric)
        if full_generated is not None else None)
    full_stitched_quality = (
        evaluate_variant(reference, full_stitched, metric)
        if full_stitched is not None else None)
    full_metadata = (
        json.loads((args.full_seedvr2_dir / "seedvr2_metadata.json").read_text())
        if args.full_seedvr2_dir is not None else None)
    full_pipeline_seconds = (
        decode["total_after_argument_parse_seconds"]
        + full_metadata["total_after_argument_parse_seconds"]
        + full_composite_seconds
        if full_metadata is not None else None)
    roi_pipeline_seconds = (
        decode["total_after_argument_parse_seconds"]
        + roi_total_seconds + composite_seconds)
    visual_frame = min(8, len(reference) - 1)
    action_colors = np.asarray([
        (74, 144, 226),
        (242, 160, 42),
        (70, 170, 92),
    ], dtype=np.uint8)
    action_map_frame = np.repeat(
        np.repeat(action_colors[actions], tile_size, axis=0),
        tile_size, axis=1)
    visual_panels = [
        panel(reference[visual_frame], "GT", None),
        panel(decoded[visual_frame], "Spatial QP raw", evaluate_variant(
            reference, decoded, metric)),
        panel(action_map_frame, "Action map (B / G / E)", None),
        panel(stitched[visual_frame], "Connected-ROI SeedVR2 B/G/E", roi_quality),
    ]
    if full_stitched is not None:
        visual_panels.insert(2, panel(
            full_stitched[visual_frame], "Full-frame SeedVR2 B/G/E",
            full_stitched_quality))
    panel_width = max(item.width for item in visual_panels)
    panel_height = max(item.height for item in visual_panels)
    visual_rows = (len(visual_panels) + 1) // 2
    visual_image = Image.new(
        "RGB", (2 * panel_width, visual_rows * panel_height), (230, 230, 230))
    for index, item in enumerate(visual_panels):
        visual_image.paste(
            item, ((index % 2) * panel_width, (index // 2) * panel_height))
    visual_path = args.output_dir / "visuals" / "roi_vs_full_seedvr2.png"
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    visual_image.save(visual_path, optimize=True)
    result = {
        "experiment": "E25 connected-ROI SeedVR2 compute probe",
        "status": "no-training-connected-component-roi-complete",
        "manifest": str(args.manifest),
        "component_count": len(manifest["components"]),
        "processing_pixel_ratio_vs_full": manifest["roi_to_full_processing_pixel_ratio"],
        "fresh_decode_regression": fresh_regression,
        "quality": {
            "roi-spatial-bge-stitched": roi_quality,
            "full-frame-seedvr2-spatial-bge-stitched": full_stitched_quality,
            "full-seedvr2": full_quality,
        },
        "runtime": {
            "components": component_runtime,
            "roi_seedvr2_inference_seconds_model_load_excluded_sum": roi_core_seconds,
            "roi_seedvr2_total_after_argument_parse_seconds_sum": roi_total_seconds,
            "persistent_roi_batch_metadata": (
                str(batch_metadata_path) if batch_metadata is not None else None),
            "persistent_single_process_roi": batch_metadata is not None,
            "roi_composite_seconds_cpu": composite_seconds,
            "codec_fresh_decode_after_argument_parse_seconds": decode[
                "total_after_argument_parse_seconds"],
            "full_roi_pipeline_seconds": roi_pipeline_seconds,
            "full_frame_seedvr2_pipeline_seconds": full_pipeline_seconds,
            "pipeline_speedup_fraction_vs_full_frame": (
                1.0 - roi_pipeline_seconds / full_pipeline_seconds
                if full_pipeline_seconds is not None else None),
            "seedvr2_core_speedup_fraction_vs_full_frame": (
                1.0 - roi_core_seconds / full_metadata["runtime_seconds"]
                if full_metadata is not None else None),
            "peak_cuda_allocated_bytes": max(
                roi_peak, int(decode["peak_cuda_allocated_bytes"])),
        },
        "scientific_boundary": {
            "diffusion_executed_only_on_generate_component_crop": True,
            "generate_geometry_available_from_bitstream": True,
            "source_rgb_read_by_decoder": False,
            "training_or_finetuning": False,
            "multi_component_model_reload_not_optimized": batch_metadata is None,
            "one_model_load_shared_across_components": batch_metadata is not None,
            "optional_full_frame_control_run": full_metadata is not None,
        },
        "output_frames": str(output_frames),
        "visual": str(visual_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "summary.json"
    atomic_json(summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "restore":
        restore(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
