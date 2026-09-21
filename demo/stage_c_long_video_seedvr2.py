#!/usr/bin/env python3
"""Overlapping-window SeedVR2 restoration for a continuous spatial-QP stream.

DCVC-UF is decoded once as one continuous video.  SeedVR2 receives configurable
4k+1-frame clips (17 frames with stride eight by default).  Overlapping
predictions are combined with deterministic triangular time weights, then
pasted only inside the per-frame Generate action mask.  Every ROI component
has its own atomic completion record, so an interrupted long run resumes from
the next unfinished component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import exact_frame_comparison
from demo.stage_c_seedvr2_roi import connected_components, round_up, save_frames
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant


ACTION_GENERATE = 1
WINDOW_LENGTH = 17
WINDOW_STRIDE = 8


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def common_seedvr2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upstream-root", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2")
    parser.add_argument(
        "--dit-checkpoint", type=Path,
        default=(REPO_ROOT / "third_party" / "SeedVR2" / "ckpts" /
                 "seedvr2_ema_3b_bf16.safetensors"))
    parser.add_argument(
        "--lora-checkpoint", type=Path,
        help="Optional SeedVR2 project LoRA adapter")
    parser.add_argument(
        "--lora-strength", type=float, default=1.0,
        help="Multiplier for the optional LoRA residual (default: 1.0)")
    parser.add_argument(
        "--vae-checkpoint", type=Path,
        default=(REPO_ROOT / "third_party" / "SeedVR2" / "ckpts" /
                 "ema_vae.pth"))
    parser.add_argument(
        "--positive-embedding", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2" / "pos_emb.pt")
    parser.add_argument(
        "--negative-embedding", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2" / "neg_emb.pt")
    parser.add_argument("--sample-steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--dit-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cuda-idx", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-video overlapping SeedVR2 ROI pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--codec-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--context-pixels", type=int, default=64)
    prepare.add_argument("--processing-scale", type=float, default=1.5)
    prepare.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    prepare.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)

    restore = commands.add_parser("restore")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--output-root", type=Path, required=True)
    restore.add_argument("--seed", type=int, required=True)
    common_seedvr2_args(restore)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--gate-summary", type=Path, required=True)
    evaluate.add_argument("--codec-dir", type=Path, required=True)
    evaluate.add_argument("--restored-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--feather-pixels", type=int, default=8)

    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.context_pixels < 0 or args.processing_scale <= 0:
            parser.error("context must be nonnegative and scale must be positive")
        if args.window_length < 1 or (args.window_length - 1) % 4:
            parser.error("SeedVR2 window length must have the form 4k+1")
        if not 0 < args.window_stride < args.window_length:
            parser.error("window stride must be positive and shorter than the window")
    if args.command == "restore":
        if not math.isfinite(args.lora_strength) or args.lora_strength < 0:
            parser.error("--lora-strength must be finite and nonnegative")
        if args.lora_checkpoint is None and args.lora_strength != 1.0:
            parser.error("--lora-strength only has meaning with --lora-checkpoint")
    if args.command == "evaluate" and args.feather_pixels < 0:
        parser.error("feather pixels must be nonnegative")
    return args


def restoration_window_starts(
    frame_count: int,
    window_length: int = WINDOW_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> list[int]:
    if frame_count < window_length:
        raise ValueError(
            f"long-video restoration needs at least {window_length} frames")
    starts = list(range(0, frame_count - window_length + 1, stride))
    final_start = frame_count - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def triangular_weights(length: int = WINDOW_LENGTH) -> list[float]:
    if length < 1 or length % 2 == 0:
        raise ValueError("temporal window length must be a positive odd number")
    center = length // 2
    denominator = center + 1
    return [
        (denominator - abs(index - center)) / denominator
        for index in range(length)
    ]


def frame_actions_from_units(summary: dict) -> list[np.ndarray]:
    units = summary.get("coding_unit_action_maps")
    if not units:
        raise ValueError("codec summary has no coding-unit action maps")
    expected_shape = tuple(map(int, summary["action_map_shape"]))
    frames = []
    expected_start = 0
    for unit_index, record in enumerate(units):
        if int(record["unit_index"]) != unit_index:
            raise ValueError("codec action-map unit indexes are not contiguous")
        frame_start = int(record["frame_start"])
        frame_count = int(record["frame_count"])
        if frame_start != expected_start:
            raise ValueError("codec action-map frame ranges are not contiguous")
        actions = np.asarray(record["actions"], dtype=np.int64)
        if actions.shape != expected_shape:
            raise ValueError("codec action map differs from declared shape")
        frames.extend(actions.copy() for _ in range(frame_count))
        expected_start += frame_count
    if len(frames) != int(summary["frames"]):
        raise ValueError("codec action maps do not cover every decoded frame")
    return frames


def component_crop(
    component: list[tuple[int, int]],
    cell_size: int,
    context: int,
    height: int,
    width: int,
) -> dict[str, int]:
    rows = [item[0] for item in component]
    columns = [item[1] for item in component]
    x0 = max(0, min(columns) * cell_size - context)
    y0 = max(0, min(rows) * cell_size - context)
    x1 = min(width, (max(columns) + 1) * cell_size + context)
    y1 = min(height, (max(rows) + 1) * cell_size + context)
    if any(value % 16 for value in (x0, y0, x1, y1)):
        raise ValueError("SeedVR2 ROI bounds must be divisible by 16")
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def prepare_main(args: argparse.Namespace) -> None:
    encode_path = args.codec_dir / "encode_summary.json"
    decode_path = args.codec_dir / "decode_summary.json"
    encode = json.loads(encode_path.read_text(encoding="utf-8"))
    decode = json.loads(decode_path.read_text(encoding="utf-8"))
    if encode["frames"] != decode["frames"]:
        raise ValueError("encode and decode summaries disagree on frame count")
    encode_frame_actions = frame_actions_from_units(encode)
    decode_frame_actions = frame_actions_from_units(decode)
    if any(
        not np.array_equal(first, second)
        for first, second in zip(encode_frame_actions, decode_frame_actions)
    ):
        raise ValueError("fresh decoder recovered different coding-unit action maps")
    decoded = load_pngs(args.codec_dir / "fresh_decode")
    frame_count = len(decoded)
    if frame_count != int(encode["frames"]):
        raise ValueError("decoded PNG count differs from codec summary")
    height, width = decoded[0].shape[:2]
    cell_size = int(encode["cell_size"])
    if args.context_pixels % 16:
        raise ValueError("context pixels must be divisible by 16")
    frame_actions = encode_frame_actions
    window_length = int(args.window_length)
    window_stride = int(args.window_stride)
    starts = restoration_window_starts(
        frame_count, window_length=window_length, stride=window_stride)
    weights = triangular_weights(window_length)
    windows = []
    total_processing_pixel_frames = 0
    full_processing_height = round_up(height * args.processing_scale, 16)
    full_processing_width = round_up(width * args.processing_scale, 16)
    for window_index, start in enumerate(starts):
        stop = start + window_length
        union_generate = np.logical_or.reduce([
            actions == ACTION_GENERATE
            for actions in frame_actions[start:stop]
        ])
        components = connected_components(union_generate)
        records = []
        for component_index, component in enumerate(components):
            crop = component_crop(
                component, cell_size, args.context_pixels, height, width)
            input_dir = (
                args.output_dir / f"window_{window_index:03d}_f{start:05d}" /
                f"component_{component_index:02d}" / "input")
            input_dir.mkdir(parents=True, exist_ok=True)
            x, y = crop["x"], crop["y"]
            crop_width, crop_height = crop["width"], crop["height"]
            for local_index, frame in enumerate(decoded[start:stop], start=1):
                Image.fromarray(
                    frame[y:y + crop_height, x:x + crop_width]
                ).save(input_dir / f"im{local_index:05d}.png")
            processing_height = round_up(
                crop_height * args.processing_scale, 16)
            processing_width = round_up(
                crop_width * args.processing_scale, 16)
            total_processing_pixel_frames += (
                processing_height * processing_width * window_length)
            records.append({
                "component_index": component_index,
                "component_id": f"w{window_index:03d}-c{component_index:02d}",
                "cells_in_window_union": component,
                "crop": crop,
                "input_dir": str(input_dir.resolve()),
                "processing_height": processing_height,
                "processing_width": processing_width,
            })
        windows.append({
            "window_index": window_index,
            "frame_start": start,
            "frame_count": window_length,
            "temporal_weights": weights,
            "generate_union_cell_count": int(np.count_nonzero(union_generate)),
            "components": records,
        })
    full_pixel_frames = (
        len(windows) * window_length
        * full_processing_height * full_processing_width)
    manifest = {
        "experiment": (
            f"overlapping {window_length}-frame SeedVR2 long-video ROI plan"),
        "codec_dir": str(args.codec_dir.resolve()),
        "encode_summary": str(encode_path.resolve()),
        "decode_summary": str(decode_path.resolve()),
        "frame_count": frame_count,
        "frame_height": height,
        "frame_width": width,
        "cell_size": cell_size,
        "action_map_shape": list(frame_actions[0].shape),
        "frame_actions": [value.tolist() for value in frame_actions],
        "window_length": window_length,
        "window_stride": window_stride,
        "window_count": len(windows),
        "windows": windows,
        "context_pixels": args.context_pixels,
        "processing_scale": args.processing_scale,
        "total_roi_processing_pixel_frames": total_processing_pixel_frames,
        "full_processing_pixel_frames": full_pixel_frames,
        "roi_to_full_processing_pixel_ratio": (
            total_processing_pixel_frames / full_pixel_frames),
        "scientific_boundary": {
            "codec_was_decoded_once_with_continuous_reference_state": True,
            "seedvr2_window_length": window_length,
            "seedvr2_window_stride": window_stride,
            "overlap_predictions_use_fixed_triangular_weights": True,
            "roi_geometry_uses_only_transmitted_action_maps": True,
            "source_rgb_used_for_roi_geometry": False,
            "training_or_finetuning": False,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "frames": frame_count,
        "windows": len(windows),
        "components": sum(len(window["components"]) for window in windows),
        "roi_to_full_processing_pixel_ratio": manifest[
            "roi_to_full_processing_pixel_ratio"],
    }, ensure_ascii=False, indent=2))


def flatten_components(manifest: dict) -> list[dict]:
    output = []
    for window in manifest["windows"]:
        for component in window["components"]:
            output.append({
                **component,
                "window_index": int(window["window_index"]),
                "window_frame_start": int(window["frame_start"]),
                "window_frame_count": int(window["frame_count"]),
            })
    return output


def component_seed(base_seed: int, record: dict) -> int:
    return (
        base_seed
        + int(record["window_index"]) * 1_000_003
        + int(record["component_index"]) * 100_003
    )


def component_output_dir(root: Path, record: dict) -> Path:
    return root / record["component_id"] / "restored"


def valid_component(
    metadata_path: Path,
    output_dir: Path,
    record: dict,
    seed: int,
    manifest_hash: str,
    lora_checkpoint: str | None,
    lora_checkpoint_sha256: str | None,
    lora_strength: float | None,
) -> dict | None:
    if not metadata_path.is_file():
        return None
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    crop = record["crop"]
    expected = {
        "manifest_sha256": manifest_hash,
        "component_id": record["component_id"],
        "seed": seed,
        "frame_count": record["window_frame_count"],
        "processing_height": record["processing_height"],
        "processing_width": record["processing_width"],
        "output_height": crop["height"],
        "output_width": crop["width"],
        "lora_checkpoint": lora_checkpoint,
        "lora_checkpoint_sha256": lora_checkpoint_sha256,
        "lora_strength": lora_strength,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return None
    if len(list(output_dir.glob("*.png"))) != record["window_frame_count"]:
        return None
    return value


def restore_main(args: argparse.Namespace) -> None:
    process_started = time.perf_counter()
    invocation_id = f"pid-{os.getpid()}-start-{time.time_ns()}"
    manifest_path = args.manifest.resolve()
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lora_path = (
        args.lora_checkpoint.resolve()
        if args.lora_checkpoint is not None else None)
    lora_checkpoint = str(lora_path) if lora_path is not None else None
    lora_checkpoint_sha256 = (
        file_sha256(lora_path) if lora_path is not None else None)
    lora_strength = (
        float(args.lora_strength) if lora_path is not None else None)
    components = flatten_components(manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    complete_records = {}
    pending = []
    for record in components:
        seed = component_seed(args.seed, record)
        output_dir = component_output_dir(args.output_root, record)
        metadata = valid_component(
            output_dir / "seedvr2_metadata.json", output_dir, record,
            seed, manifest_hash, lora_checkpoint, lora_checkpoint_sha256,
            lora_strength)
        if metadata is None:
            pending.append(record)
        else:
            complete_records[record["component_id"]] = metadata
    print(json.dumps({
        "stage": "long-video-resume-scan",
        "complete_components": len(complete_records),
        "pending_components": len(pending),
    }), flush=True)

    batch_metadata_path = args.output_root / "long_roi_batch_metadata.json"
    if not pending and batch_metadata_path.is_file():
        existing_batch = json.loads(
            batch_metadata_path.read_text(encoding="utf-8"))
        if (
            existing_batch.get("status") == "complete"
            and existing_batch.get("manifest_sha256") == manifest_hash
            and existing_batch.get("base_seed") == args.seed
            and existing_batch.get("component_count") == len(components)
            and existing_batch.get("lora_checkpoint") == lora_checkpoint
            and existing_batch.get("lora_checkpoint_sha256")
            == lora_checkpoint_sha256
            and existing_batch.get("lora_strength") == lora_strength
        ):
            print(json.dumps({
                "stage": "long-video-restoration-resume-skip",
                "output": str(batch_metadata_path),
                "completed_component_count": len(complete_records),
            }), flush=True)
            print(json.dumps(existing_batch, ensure_ascii=False, indent=2))
            return

    model = None
    model_load_seconds = 0.0
    if pending:
        torch.cuda.set_device(args.cuda_idx)
        model = PersistentSeedVR2(args)
        model_load_seconds = model.model_load_seconds
    for record in pending:
        if model is None:
            raise RuntimeError("pending component has no loaded SeedVR2 model")
        seed = component_seed(args.seed, record)
        output_dir = component_output_dir(args.output_root, record)
        started = time.perf_counter()
        frames = load_pngs(Path(record["input_dir"]))
        crop = record["crop"]
        restored, runtime = model.restore(
            frames,
            seed,
            processing_height=int(record["processing_height"]),
            processing_width=int(record["processing_width"]),
            output_height=int(crop["height"]),
            output_width=int(crop["width"]),
        )
        save_frames(output_dir, restored)
        wall_seconds = time.perf_counter() - started
        metadata = {
            "model": "SeedVR2-3B overlapping long-video ROI",
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "component_id": record["component_id"],
            "window_index": record["window_index"],
            "window_frame_start": record["window_frame_start"],
            "input_dir": record["input_dir"],
            "output_dir": str(output_dir.resolve()),
            "frame_count": len(restored),
            "input_height": int(crop["height"]),
            "input_width": int(crop["width"]),
            "processing_height": int(record["processing_height"]),
            "processing_width": int(record["processing_width"]),
            "output_height": int(crop["height"]),
            "output_width": int(crop["width"]),
            "seed": seed,
            "sample_steps": args.sample_steps,
            "cfg_scale": args.cfg_scale,
            "dit_dtype": args.dit_dtype,
            "dit_checkpoint": str(args.dit_checkpoint.resolve()),
            "lora_checkpoint": lora_checkpoint,
            "lora_checkpoint_sha256": lora_checkpoint_sha256,
            "lora_strength": lora_strength,
            "lora_adapter": model.runner.lora_adapter_info,
            "runtime_seconds": runtime["seconds_model_load_excluded"],
            "component_wall_seconds": wall_seconds,
            "peak_cuda_allocated_bytes": runtime[
                "peak_cuda_allocated_bytes"],
            "model_load_in_timing": False,
            "restore_invocation_id": invocation_id,
            "shared_model_load_seconds_this_process": model_load_seconds,
            "resumable_component": True,
            "training_or_finetuning": False,
            "actual_compute_scope": runtime["actual_compute_scope"],
        }
        atomic_json(output_dir / "seedvr2_metadata.json", metadata)
        complete_records[record["component_id"]] = metadata
        atomic_json(args.output_root / "progress.json", {
            "manifest_sha256": manifest_hash,
            "completed_component_count": len(complete_records),
            "component_count": len(components),
            "last_completed_component": record["component_id"],
        })
        print(json.dumps({
            "stage": "long-video-component-complete",
            "component_id": record["component_id"],
            "completed": len(complete_records),
            "total": len(components),
            "runtime_seconds": metadata["runtime_seconds"],
            "peak_cuda_mib": metadata["peak_cuda_allocated_bytes"] / 1048576,
        }), flush=True)

    if len(complete_records) != len(components):
        raise RuntimeError("not every long-video ROI component completed")
    ordered = [complete_records[record["component_id"]] for record in components]
    result = {
        "experiment": "resumable overlapping-window SeedVR2 restoration",
        "status": "complete",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "base_seed": args.seed,
        "lora_checkpoint": lora_checkpoint,
        "lora_checkpoint_sha256": lora_checkpoint_sha256,
        "lora_strength": lora_strength,
        "lora_adapter": ordered[0].get("lora_adapter") if ordered else None,
        "component_count": len(components),
        "completed_component_count": len(ordered),
        "resumed_component_count": len(components) - len(pending),
        "computed_component_count_this_process": len(pending),
        "model_load_seconds_this_process": model_load_seconds,
        "component_inference_seconds_sum": sum(
            float(value["runtime_seconds"]) for value in ordered),
        "component_wall_seconds_sum": sum(
            float(value["component_wall_seconds"]) for value in ordered),
        "this_process_wall_seconds": time.perf_counter() - process_started,
        "peak_cuda_allocated_bytes": max(
            (int(value["peak_cuda_allocated_bytes"]) for value in ordered),
            default=0),
        "components": [
            {
                "component_id": value["component_id"],
                "metadata": str(
                    component_output_dir(args.output_root, record)
                    / "seedvr2_metadata.json"),
                "runtime_seconds": value["runtime_seconds"],
            }
            for record, value in zip(components, ordered)
        ],
        "scientific_boundary": {
            "completed_components_survive_process_restart": True,
            "one_model_load_shared_by_pending_components": True,
            "model_load_may_repeat_after_restart": True,
            "training_or_finetuning": False,
            "base_seedvr2_weights_frozen": True,
            "lora_adapter_inference_only": lora_path is not None,
        },
    }
    atomic_json(batch_metadata_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def spatial_alpha(
    actions: np.ndarray,
    cell_size: int,
    feather: int,
    height: int,
    width: int,
) -> np.ndarray:
    if feather > cell_size:
        raise ValueError("feather cannot exceed the action cell size")
    rows, columns = actions.shape
    if rows * cell_size != height or columns * cell_size != width:
        raise ValueError("action map does not cover the frame")
    alpha = np.zeros((height, width), dtype=np.float32)
    for row, column in zip(*np.where(actions == ACTION_GENERATE)):
        y, x = int(row) * cell_size, int(column) * cell_size
        tile = np.ones((cell_size, cell_size), dtype=np.float32)
        if feather:
            ramp = np.arange(feather, dtype=np.float32) / feather

            def generated(next_row: int, next_column: int) -> bool:
                return (
                    0 <= next_row < rows and 0 <= next_column < columns
                    and actions[next_row, next_column] == ACTION_GENERATE)

            if not generated(row, column - 1):
                tile[:, :feather] *= ramp[None]
            if not generated(row, column + 1):
                tile[:, -feather:] *= ramp[::-1][None]
            if not generated(row - 1, column):
                tile[:feather] *= ramp[:, None]
            if not generated(row + 1, column):
                tile[-feather:] *= ramp[::-1, None]
        alpha[y:y + cell_size, x:x + cell_size] = tile
    return alpha


def action_visual(actions: np.ndarray, cell_size: int) -> np.ndarray:
    colors = np.asarray([
        (74, 144, 226),
        (242, 160, 42),
        (70, 170, 92),
    ], dtype=np.uint8)
    return np.repeat(
        np.repeat(colors[actions], cell_size, axis=0), cell_size, axis=1)


def visual_panel(frame: np.ndarray, title: str) -> Image.Image:
    image = Image.fromarray(frame)
    banner = 42
    output = Image.new("RGB", (image.width, image.height + banner), "white")
    output.paste(image, (0, banner))
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    draw.text((8, 10), title, fill="black", font=font)
    return output


def write_boundary_visual(
    path: Path,
    reference: list[np.ndarray],
    decoded: list[np.ndarray],
    hard_stitched: list[np.ndarray],
    stitched: list[np.ndarray],
    frame_actions: list[np.ndarray],
    cell_size: int,
    center: int,
) -> None:
    indices = sorted(set(
        max(0, min(len(reference) - 1, value))
        for value in (center - 1, center, center + 1)))
    rows = []
    for label, frames in (
        ("GT", reference),
        ("Spatial-QP decoded", decoded),
        ("Action map B/G/E", [
            action_visual(value, cell_size) for value in frame_actions]),
        ("Hard window selection", hard_stitched),
        ("Overlap SeedVR2 result", stitched),
    ):
        rows.append([
            visual_panel(frames[index], f"{label} | frame {index}")
            for index in indices
        ])
    panel_width = max(panel.width for row in rows for panel in row)
    panel_height = max(panel.height for row in rows for panel in row)
    canvas = Image.new(
        "RGB", (len(indices) * panel_width, len(rows) * panel_height),
        (230, 230, 230))
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row):
            canvas.paste(
                panel, (column_index * panel_width, row_index * panel_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def evaluate_main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    batch_path = args.restored_root / "long_roi_batch_metadata.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if batch.get("status") != "complete":
        raise RuntimeError("long-video SeedVR2 restoration is incomplete")
    encode = json.loads(
        (args.codec_dir / "encode_summary.json").read_text(encoding="utf-8"))
    decode = json.loads(
        (args.codec_dir / "decode_summary.json").read_text(encoding="utf-8"))
    encoder_frames = load_pngs(args.codec_dir / "encoder_reconstruction")
    decoded = load_pngs(args.codec_dir / "fresh_decode")
    fresh_regression = exact_frame_comparison(encoder_frames, decoded)
    if not fresh_regression.get("pixel_exact"):
        raise RuntimeError("fresh decode differs from encoder reconstruction")
    frame_count = len(decoded)
    height, width = decoded[0].shape[:2]
    frame_actions = [
        np.asarray(value, dtype=np.int64)
        for value in manifest["frame_actions"]
    ]
    compose_started = time.perf_counter()
    numerator = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    denominator = np.zeros((frame_count, height, width, 1), dtype=np.float32)
    hard_generated = np.stack(decoded).astype(np.float32)
    hard_weight = np.zeros((frame_count, height, width), dtype=np.float32)
    frame_window_coverage = np.zeros(frame_count, dtype=np.int64)
    for window in manifest["windows"]:
        start = int(window["frame_start"])
        weights = list(map(float, window["temporal_weights"]))
        for local_index in range(int(window["frame_count"])):
            frame_window_coverage[start + local_index] += 1
        for component in window["components"]:
            restored_dir = component_output_dir(args.restored_root, component)
            restored = load_pngs(restored_dir)
            if len(restored) != int(window["frame_count"]):
                raise ValueError("restored ROI has the wrong frame count")
            crop = component["crop"]
            x, y = int(crop["x"]), int(crop["y"])
            crop_width, crop_height = int(crop["width"]), int(crop["height"])
            for local_index, frame in enumerate(restored):
                frame_index = start + local_index
                weight = weights[local_index]
                numerator[
                    frame_index, y:y + crop_height, x:x + crop_width
                ] += frame.astype(np.float32) * weight
                denominator[
                    frame_index, y:y + crop_height, x:x + crop_width
                ] += weight
                best = hard_weight[
                    frame_index, y:y + crop_height, x:x + crop_width]
                take = weight > best
                hard_region = hard_generated[
                    frame_index, y:y + crop_height, x:x + crop_width]
                hard_region[take] = frame.astype(np.float32)[take]
                best[take] = weight
    generated = np.stack(decoded).astype(np.float32)
    covered = denominator[..., 0] > 0
    generated[covered] = (
        numerator[covered] / denominator[covered])
    stitched = []
    hard_stitched = []
    cell_size = int(manifest["cell_size"])
    for frame_index, (base, actions) in enumerate(zip(decoded, frame_actions)):
        alpha = spatial_alpha(
            actions, cell_size, args.feather_pixels, height, width)
        generate_pixels = np.repeat(
            np.repeat(actions == ACTION_GENERATE, cell_size, axis=0),
            cell_size, axis=1)
        if np.any(generate_pixels & ~covered[frame_index]):
            raise RuntimeError(
                f"frame {frame_index} has Generate pixels without restoration")
        value = (
            base.astype(np.float32)
            + alpha[:, :, None]
            * (generated[frame_index] - base.astype(np.float32)))
        stitched.append(np.clip(value, 0, 255).round().astype(np.uint8))
        hard_value = (
            base.astype(np.float32)
            + alpha[:, :, None]
            * (hard_generated[frame_index] - base.astype(np.float32)))
        hard_stitched.append(
            np.clip(hard_value, 0, 255).round().astype(np.uint8))
    output_frames = args.output_dir / "frames" / "overlap-seedvr2-stitched"
    save_frames(output_frames, stitched)
    hard_output_frames = (
        args.output_dir / "frames" / "hard-window-seedvr2-stitched")
    save_frames(hard_output_frames, hard_stitched)
    composite_seconds = time.perf_counter() - compose_started

    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    reference = load_source(gate, frame_count)
    metric = LPIPSAlex(True)
    decoded_quality = evaluate_variant(reference, decoded, metric)
    hard_quality = evaluate_variant(reference, hard_stitched, metric)
    stitched_quality = evaluate_variant(reference, stitched, metric)
    boundary_frames = [
        index for index, count in enumerate(frame_window_coverage) if count > 1
    ]
    primary_windows = []
    for frame_index in range(frame_count):
        candidates = []
        for window in manifest["windows"]:
            local_index = frame_index - int(window["frame_start"])
            if 0 <= local_index < int(window["frame_count"]):
                candidates.append((
                    float(window["temporal_weights"][local_index]),
                    -int(window["window_index"]),
                ))
        primary_windows.append(-max(candidates)[1])
    hard_window_transition_frames = [
        index for index in range(1, frame_count)
        if primary_windows[index] != primary_windows[index - 1]
    ]
    action_transition_frames = [
        index for index in range(1, frame_count)
        if not np.array_equal(frame_actions[index], frame_actions[index - 1])
    ]

    def boundary_temporal_error(frames: list[np.ndarray]) -> float | None:
        if not hard_window_transition_frames:
            return None
        values = []
        for index in hard_window_transition_frames:
            reference_delta = (
                reference[index].astype(np.float32)
                - reference[index - 1].astype(np.float32))
            output_delta = (
                frames[index].astype(np.float32)
                - frames[index - 1].astype(np.float32))
            values.append(float(np.mean(np.abs(
                reference_delta - output_delta))))
        return float(np.mean(values))

    center = (
        action_transition_frames[0]
        if action_transition_frames else frame_count // 2)
    visual_path = args.output_dir / "visuals" / "long_video_boundary.png"
    write_boundary_visual(
        visual_path, reference, decoded, hard_stitched, stitched, frame_actions,
        cell_size, center)
    result = {
        "experiment": "continuous-codec overlapping-window SeedVR2 smoke",
        "status": "complete",
        "source_role": gate["source_role"],
        "frames": frame_count,
        "stream": encode["stream"],
        "actual_stream_bytes": int(encode["stream_bytes"]),
        "continuous_codec_reference": True,
        "time_varying_action_maps": bool(encode[
            "time_varying_action_maps"]),
        "fresh_decode_regression": fresh_regression,
        "windowing": {
            "length": int(manifest["window_length"]),
            "stride": int(manifest["window_stride"]),
            "window_count": int(manifest["window_count"]),
            "temporal_weights": list(map(
                float, manifest["windows"][0]["temporal_weights"])),
            "per_frame_window_coverage": frame_window_coverage.tolist(),
            "overlap_frame_indices": boundary_frames,
            "hard_selected_window_per_frame": primary_windows,
            "hard_window_transition_frames": hard_window_transition_frames,
            "action_transition_frames": action_transition_frames,
        },
        "quality": {
            "spatial_qp_decoded": decoded_quality,
            "hard_window_seedvr2_stitched": hard_quality,
            "overlap_seedvr2_stitched": stitched_quality,
            "delta_stitched_minus_decoded": {
                "psnr_db": (
                    stitched_quality["psnr_db"] - decoded_quality["psnr_db"]),
                "lpips_alex": (
                    stitched_quality["lpips_alex"]
                    - decoded_quality["lpips_alex"]),
                "temporal_delta_mae": (
                    stitched_quality["temporal_delta_mae"]
                    - decoded_quality["temporal_delta_mae"]),
            },
            "delta_overlap_minus_hard_window": {
                "psnr_db": (
                    stitched_quality["psnr_db"] - hard_quality["psnr_db"]),
                "lpips_alex": (
                    stitched_quality["lpips_alex"]
                    - hard_quality["lpips_alex"]),
                "temporal_delta_mae": (
                    stitched_quality["temporal_delta_mae"]
                    - hard_quality["temporal_delta_mae"]),
            },
            "hard_switch_pair_temporal_delta_mae": {
                "hard_window": boundary_temporal_error(hard_stitched),
                "overlap_blend": boundary_temporal_error(stitched),
            },
        },
        "runtime": {
            "codec_encode_seconds": encode["codec_seconds"],
            "codec_fresh_decode_seconds": decode["bitstream_decode_seconds"],
            "seedvr2_component_inference_seconds_sum": batch[
                "component_inference_seconds_sum"],
            "seedvr2_component_wall_seconds_sum": batch[
                "component_wall_seconds_sum"],
            "seedvr2_model_load_seconds": batch[
                "model_load_seconds_this_process"],
            "seedvr2_successful_process_seconds": batch[
                "this_process_wall_seconds"],
            "overlap_and_hard_composite_seconds_cpu": composite_seconds,
            "sequential_fresh_decode_seedvr2_overlap_seconds": (
                decode["total_after_argument_parse_seconds"]
                + batch["this_process_wall_seconds"]
                + composite_seconds),
            "evaluation_this_process_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": max(
                int(encode["peak_cuda_allocated_bytes"]),
                int(decode["peak_cuda_allocated_bytes"]),
                int(batch["peak_cuda_allocated_bytes"])),
        },
        "storage": {
            "stream_bytes": int(encode["stream_bytes"]),
            "output_frames": str(output_frames.resolve()),
            "hard_window_output_frames": str(hard_output_frames.resolve()),
        },
        "scientific_boundary": {
            "development_mechanism_smoke_not_independent_test": True,
            "router_and_backbones_frozen": batch.get("lora_checkpoint") is None,
            "router_and_base_backbones_frozen": True,
            "seedvr2_lora_adapter_applied": batch.get("lora_checkpoint") is not None,
            "seedvr2_lora_strength": batch.get("lora_strength"),
            "source_rgb_read_by_decoder": False,
            "generate_geometry_available_from_bitstream": True,
            "seedvr2_predictions_temporally_blended_before_spatial_paste": True,
            "training_or_finetuning": False,
        },
        "visual": str(visual_path.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def self_test() -> None:
    assert restoration_window_starts(17) == [0]
    assert restoration_window_starts(25) == [0, 8]
    assert restoration_window_starts(33) == [0, 8, 16]
    assert restoration_window_starts(33, 9, 4) == [0, 4, 8, 12, 16, 20, 24]
    assert restoration_window_starts(33, 17, 8) == [0, 8, 16]
    assert restoration_window_starts(33, 33, 16) == [0]
    weights = triangular_weights()
    assert len(weights) == 17 and weights[8] == 1.0
    assert weights[0] == weights[-1] == 1 / 9
    for length in (9, 17, 33):
        values = triangular_weights(length)
        assert len(values) == length
        assert values[length // 2] == 1.0
        assert values[0] == values[-1]
    summary = {
        "frames": 17,
        "action_map_shape": [2, 3],
        "coding_unit_action_maps": [
            {
                "unit_index": 0, "frame_start": 0, "frame_count": 1,
                "actions": [[0, 1, 0], [0, 0, 0]],
            },
            {
                "unit_index": 1, "frame_start": 1, "frame_count": 8,
                "actions": [[0, 0, 0], [0, 1, 0]],
            },
            {
                "unit_index": 2, "frame_start": 9, "frame_count": 8,
                "actions": [[0, 0, 1], [0, 0, 0]],
            },
        ],
    }
    actions = frame_actions_from_units(summary)
    assert len(actions) == 17
    assert actions[0][0, 1] == ACTION_GENERATE
    assert actions[8][1, 1] == ACTION_GENERATE
    assert actions[9][0, 2] == ACTION_GENERATE
    alpha = spatial_alpha(
        np.asarray([[1, 1], [0, 0]], dtype=np.int64), 16, 4, 32, 32)
    assert math.isclose(float(alpha[8, 15]), 1.0)
    assert float(alpha[0, 0]) == 0.0
    print(json.dumps({
        "status": "passed",
        "33_frame_window_starts": restoration_window_starts(33),
        "33_frame_sensitivity_starts": {
            "9": restoration_window_starts(33, 9, 4),
            "17": restoration_window_starts(33, 17, 8),
            "33": restoration_window_starts(33, 33, 16),
        },
        "triangular_center_weight": weights[8],
        "triangular_endpoint_weight": weights[0],
        "per_frame_action_mapping": True,
        "spatial_feather": True,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_main(args)
    elif args.command == "restore":
        restore_main(args)
    elif args.command == "evaluate":
        evaluate_main(args)
    else:
        self_test()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
