#!/usr/bin/env python3
"""Freeze and summarize the SeedVR2 LoRA-0.50 ROI/long-video check.

The experiment deliberately reuses the completed 33-frame continuous codec
stream, transmitted action maps, ROI plan, source crop, and noise seeds.  The
only changed variable is the inference-only SeedVR2 LoRA residual multiplier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_GENERATE = 1
FORMAT_VERSION = 1
QUALITY_KEYS = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
BOUNDARY_KEYS = (
    "gradient_error_mae",
    "output_jump_mae",
    "reference_jump_mae",
    "boundary_band_rgb_mae",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_paths(path: Path, expected: int = 33) -> list[Path]:
    values = sorted(path.glob("*.png"))
    if len(values) != expected:
        raise RuntimeError(f"expected {expected} PNGs in {path}, found {len(values)}")
    return values


def frame_digest(path: Path, expected: int = 33) -> str:
    digest = hashlib.sha256()
    for item in frame_paths(path, expected):
        digest.update(item.name.encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--base-run", type=Path, required=True)
    plan.add_argument("--strength-summary", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--strength", type=float, default=0.50)
    plan.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--lora-evaluation", type=Path, required=True)
    summarize.add_argument("--lora-restored-root", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "plan":
        if not math.isfinite(args.strength) or args.strength < 0:
            parser.error("--strength must be finite and nonnegative")
    return args


def plan_main(args: argparse.Namespace) -> None:
    base = args.base_run.resolve()
    required = {
        "run_complete": base / "run.complete",
        "run_summary": base / "run_summary.json",
        "roi_manifest": base / "seedvr2_roi_plan" / "manifest.json",
        "gate_summary": base / "plan" / "long_gate.json",
        "codec_encode": base / "codec" / "encode_summary.json",
        "codec_decode": base / "codec" / "decode_summary.json",
        "stream": base / "codec" / "streams" / "continuous_33f_spatial_qp.dqvc",
        "base_evaluation": base / "evaluation" / "summary.json",
        "base_batch": base / "seedvr2_roi_restore" / "long_roi_batch_metadata.json",
    }
    required["strength_summary"] = args.strength_summary.resolve()
    required["lora_checkpoint"] = args.lora_checkpoint.resolve()
    for path in required.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    run = read_json(required["run_summary"])
    manifest = read_json(required["roi_manifest"])
    gate = read_json(required["gate_summary"])
    encode = read_json(required["codec_encode"])
    decode = read_json(required["codec_decode"])
    evaluation = read_json(required["base_evaluation"])
    batch = read_json(required["base_batch"])
    strengths = read_json(required["strength_summary"])
    if run.get("status") != "complete" or evaluation.get("status") != "complete":
        raise RuntimeError("base long-video run is incomplete")
    if batch.get("status") != "complete" or batch.get("component_count") != 3:
        raise RuntimeError("base long-video ROI restoration is incomplete")
    if int(manifest.get("frame_count", -1)) != 33 or manifest.get("window_count") != 3:
        raise RuntimeError("expected the frozen 33-frame, three-window ROI plan")
    if encode.get("stream_bytes") != required["stream"].stat().st_size:
        raise RuntimeError("base stream byte count differs")
    if decode.get("source_rgb_read_by_decoder") is not False:
        raise RuntimeError("base fresh decoder crossed the source-RGB boundary")
    if evaluation.get("fresh_decode_regression", {}).get("pixel_exact") is not True:
        raise RuntimeError("base fresh decode is not pixel exact")
    if strengths.get("status") != "complete" or strengths.get("sample_count") != 37:
        raise RuntimeError("LoRA strength selection summary is incomplete")
    available = {float(row["strength"]) for row in strengths["metric_only_ranking"]}
    if float(args.strength) not in available:
        raise RuntimeError("selected strength was not part of the fixed sweep")

    frozen_frames = Path(evaluation["storage"]["output_frames"]).resolve()
    source_files = [Path(value).resolve() for value in gate["source_files"]]
    if len(source_files) != 33:
        raise RuntimeError("expected 33 frozen source paths")
    value = {
        "experiment": "SeedVR2 LoRA-0.50 Generate-ROI and long-video reintegration",
        "format_version": FORMAT_VERSION,
        "status": "frozen-before-lora-roi-output",
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "selected_lora_strength": float(args.strength),
        "selection_is_a_practical_choice_not_a_hard_gate": True,
        "lora_checkpoint": str(required["lora_checkpoint"]),
        "lora_checkpoint_sha256": sha256_file(required["lora_checkpoint"]),
        "strength_summary": str(required["strength_summary"]),
        "strength_summary_sha256": sha256_file(required["strength_summary"]),
        "base_run": str(base),
        "base_git_commit": run["git_commit_at_execution"],
        "source_role": gate["source_role"],
        "frame_count": 33,
        "base_seed": int(batch["base_seed"]),
        "feather_pixels": 16,
        "base_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in required.items()
            if name not in {"run_complete", "lora_checkpoint", "strength_summary"}
        },
        "base_frozen_output": {
            "frames_dir": str(frozen_frames),
            "frames_sha256": frame_digest(frozen_frames),
        },
        "source_files_sha256": source_digest(source_files),
        "protocol": {
            "same_continuous_codec_stream": True,
            "same_transmitted_time_varying_action_maps": True,
            "same_roi_geometry_and_16px_feather": True,
            "same_17_frame_windows_stride_8_and_triangular_blend": True,
            "same_per_component_noise_seeds": True,
            "only_changed_variable": "SeedVR2 LoRA inference strength 0 -> 0.50",
            "fresh_decode_reused_after_hash_verification": True,
            "development_mechanism_check_not_independent_benchmark": True,
        },
    }
    atomic_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def load_pngs(path: Path, expected: int = 33) -> list[np.ndarray]:
    return [np.asarray(Image.open(item).convert("RGB"), dtype=np.uint8)
            for item in frame_paths(path, expected)]


def load_reference(gate: dict) -> list[np.ndarray]:
    crop = gate["crop"]
    box = (
        int(crop["x"]), int(crop["y"]),
        int(crop["x"]) + int(crop["width"]),
        int(crop["y"]) + int(crop["height"]),
    )
    return [
        np.asarray(Image.open(path).convert("RGB").crop(box), dtype=np.uint8)
        for path in gate["source_files"]
    ]


def generated_mask(actions: np.ndarray, cell_size: int) -> np.ndarray:
    return np.repeat(
        np.repeat(actions == ACTION_GENERATE, cell_size, axis=0),
        cell_size, axis=1)


def pixel_scope_stats(
    frozen: list[np.ndarray], lora: list[np.ndarray],
    frame_actions: list[np.ndarray], cell_size: int,
) -> dict:
    generated_differences = []
    outside_max = 0
    changed_inside = 0
    inside_values = 0
    for first, second, actions in zip(frozen, lora, frame_actions):
        difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
        mask = generated_mask(actions, cell_size)
        outside = difference[~mask]
        if outside.size:
            outside_max = max(outside_max, int(outside.max()))
        inside = difference[mask]
        if inside.size:
            generated_differences.append(float(inside.mean()))
            changed_inside += int(np.count_nonzero(np.any(difference[mask] > 0, axis=1)))
            inside_values += int(inside.shape[0])
    return {
        "outside_generate_max_abs_pixel_difference": outside_max,
        "outside_generate_pixel_exact": outside_max == 0,
        "inside_generate_mean_abs_pixel_difference": float(np.mean(
            generated_differences)) if generated_differences else 0.0,
        "inside_generate_changed_rgb_pixel_count": changed_inside,
        "inside_generate_rgb_pixel_count": inside_values,
    }


def temporal_pair_delta_error(
    reference: list[np.ndarray], output: list[np.ndarray], indices: list[int],
) -> float | None:
    values = []
    for index in indices:
        if not 0 < index < len(reference):
            continue
        reference_delta = (
            reference[index].astype(np.float32)
            - reference[index - 1].astype(np.float32))
        output_delta = (
            output[index].astype(np.float32)
            - output[index - 1].astype(np.float32))
        values.append(float(np.mean(np.abs(output_delta - reference_delta))))
    return float(np.mean(values)) if values else None


def boundary_aggregate(
    reference: list[np.ndarray], output: list[np.ndarray],
    frame_actions: list[np.ndarray], cell_size: int,
) -> dict:
    from demo.stage_c_a800_feather_diagnostic import boundary_metrics

    totals = {key: 0.0 for key in BOUNDARY_KEYS}
    edges = 0
    frames = 0
    for source, value, actions in zip(reference, output, frame_actions):
        row = boundary_metrics([source], [value], actions, cell_size)
        count = int(row["cross_generate_boundary_edge_count"])
        if count == 0:
            continue
        frames += 1
        edges += count
        for key in BOUNDARY_KEYS:
            totals[key] += float(row[key]) * count
    return {
        "frame_count_with_generate_boundary": frames,
        "cross_generate_boundary_edge_frame_count": edges,
        **{key: totals[key] / edges if edges else None for key in BOUNDARY_KEYS},
    }


def action_visual(actions: np.ndarray, cell_size: int) -> np.ndarray:
    colors = np.asarray(
        ((74, 144, 226), (242, 160, 42), (70, 170, 92)), dtype=np.uint8)
    return np.repeat(np.repeat(colors[actions], cell_size, axis=0), cell_size, axis=1)


def visual_panel(frame: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    image = Image.fromarray(frame)
    banner = 54
    panel = Image.new("RGB", (image.width, image.height + banner), "white")
    panel.paste(image, (0, banner))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    draw.text((8, 7), title, fill="black", font=font)
    if subtitle:
        draw.text((8, 29), subtitle, fill=(55, 55, 55), font=font)
    return panel


def save_boundary_visual(
    path: Path, reference: list[np.ndarray], decoded: list[np.ndarray],
    frozen: list[np.ndarray], lora: list[np.ndarray],
    frame_actions: list[np.ndarray], center: int,
) -> None:
    indices = [max(0, center - 1), center, min(len(reference) - 1, center + 1)]
    rows: list[list[Image.Image]] = []
    for label, frames in (
        ("GT", reference),
        ("Spatial-QP fresh decode", decoded),
        ("Frozen SeedVR2 ROI", frozen),
        ("LoRA 0.50 ROI", lora),
    ):
        rows.append([visual_panel(frames[index], f"{label} | frame {index}")
                     for index in indices])
    rows.append([
        visual_panel(
            np.clip(np.abs(lora[index].astype(np.int16)
                           - frozen[index].astype(np.int16)) * 6, 0, 255).astype(np.uint8),
            f"|LoRA - frozen| x6 | frame {index}")
        for index in indices
    ])
    rows.append([
        visual_panel(action_visual(frame_actions[index], 64),
                     f"Action map B/G/E | frame {index}")
        for index in indices
    ])
    width = max(panel.width for row in rows for panel in row)
    height = max(panel.height for row in rows for panel in row)
    canvas = Image.new("RGB", (len(indices) * width, len(rows) * height),
                       (230, 230, 230))
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row):
            canvas.paste(panel, (column_index * width, row_index * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def save_video_frames(
    root: Path, reference: list[np.ndarray], decoded: list[np.ndarray],
    frozen: list[np.ndarray], lora: list[np.ndarray],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(len(reference)):
        difference = np.clip(
            np.abs(lora[index].astype(np.int16) - frozen[index].astype(np.int16)) * 6,
            0, 255).astype(np.uint8)
        panels = [
            visual_panel(reference[index], "GT", f"frame {index}"),
            visual_panel(decoded[index], "Spatial-QP", f"frame {index}"),
            visual_panel(frozen[index], "Frozen ROI", f"frame {index}"),
            visual_panel(lora[index], "LoRA 0.50 ROI", f"frame {index}"),
            visual_panel(difference, "|LoRA-frozen| x6", f"frame {index}"),
        ]
        canvas = Image.new("RGB", (sum(panel.width for panel in panels),
                                   max(panel.height for panel in panels)), "white")
        left = 0
        for panel in panels:
            canvas.paste(panel, (left, 0))
            left += panel.width
        target = root / f"frame_{index:03d}.png"
        temporary = target.with_name(target.name + ".tmp")
        canvas.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, target)


def summarize_main(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    if plan.get("status") != "frozen-before-lora-roi-output":
        raise RuntimeError("ROI/long-video plan is not frozen")
    if plan.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("unsupported ROI/long-video plan format")
    for item in plan["base_artifacts"].values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"base artifact changed after plan freeze: {path}")
    if sha256_file(Path(plan["lora_checkpoint"])) != plan["lora_checkpoint_sha256"]:
        raise RuntimeError("LoRA checkpoint changed after plan freeze")
    if sha256_file(Path(plan["strength_summary"])) != plan["strength_summary_sha256"]:
        raise RuntimeError("strength summary changed after plan freeze")
    if frame_digest(Path(plan["base_frozen_output"]["frames_dir"])) != (
            plan["base_frozen_output"]["frames_sha256"]):
        raise RuntimeError("base frozen long-video output changed after plan freeze")

    base = Path(plan["base_run"])
    base_eval = read_json(base / "evaluation" / "summary.json")
    lora_eval = read_json(args.lora_evaluation)
    manifest = read_json(base / "seedvr2_roi_plan" / "manifest.json")
    gate = read_json(base / "plan" / "long_gate.json")
    batch_path = args.lora_restored_root / "long_roi_batch_metadata.json"
    batch = read_json(batch_path)
    if lora_eval.get("status") != "complete" or batch.get("status") != "complete":
        raise RuntimeError("LoRA ROI evaluation is incomplete")
    if batch.get("completed_component_count") != 3:
        raise RuntimeError("LoRA ROI component count differs")
    if batch.get("lora_checkpoint_sha256") != plan["lora_checkpoint_sha256"]:
        raise RuntimeError("LoRA ROI used another adapter")
    if not math.isclose(float(batch.get("lora_strength", -1)),
                        float(plan["selected_lora_strength"]), abs_tol=1e-12):
        raise RuntimeError("LoRA ROI used another inference strength")
    if int(batch.get("base_seed", -1)) != int(plan["base_seed"]):
        raise RuntimeError("LoRA ROI used another base seed")
    if lora_eval.get("fresh_decode_regression", {}).get("pixel_exact") is not True:
        raise RuntimeError("LoRA evaluation lacks pixel-exact fresh decode")
    if int(lora_eval["actual_stream_bytes"]) != int(base_eval["actual_stream_bytes"]):
        raise RuntimeError("LoRA evaluation did not reuse the same stream bytes")

    reference = load_reference(gate)
    decoded = load_pngs(base / "codec" / "fresh_decode")
    frozen = load_pngs(Path(base_eval["storage"]["output_frames"]))
    lora = load_pngs(Path(lora_eval["storage"]["output_frames"]))
    frame_actions = [np.asarray(value, dtype=np.int64)
                     for value in manifest["frame_actions"]]
    if len(frame_actions) != 33:
        raise RuntimeError("time-varying action map count differs")
    scope = pixel_scope_stats(frozen, lora, frame_actions, int(manifest["cell_size"]))
    if not scope["outside_generate_pixel_exact"]:
        raise RuntimeError("LoRA changed pixels outside transmitted Generate regions")

    frozen_quality = base_eval["quality"]["overlap_seedvr2_stitched"]
    lora_quality = lora_eval["quality"]["overlap_seedvr2_stitched"]
    quality_delta = {
        key: float(lora_quality[key]) - float(frozen_quality[key])
        for key in QUALITY_KEYS
    }
    frozen_boundary = boundary_aggregate(
        reference, frozen, frame_actions, int(manifest["cell_size"]))
    lora_boundary = boundary_aggregate(
        reference, lora, frame_actions, int(manifest["cell_size"]))
    boundary_delta = {
        key: float(lora_boundary[key]) - float(frozen_boundary[key])
        for key in BOUNDARY_KEYS
    }
    hard_transitions = list(map(
        int, lora_eval["windowing"]["hard_window_transition_frames"]))
    action_transitions = list(map(
        int, lora_eval["windowing"]["action_transition_frames"]))
    transition = {
        "hard_window_transition_frames": hard_transitions,
        "action_transition_frames": action_transitions,
        "hard_window_temporal_delta_error": {
            "frozen": temporal_pair_delta_error(reference, frozen, hard_transitions),
            "lora_050": temporal_pair_delta_error(reference, lora, hard_transitions),
        },
        "action_map_temporal_delta_error": {
            "frozen": temporal_pair_delta_error(reference, frozen, action_transitions),
            "lora_050": temporal_pair_delta_error(reference, lora, action_transitions),
        },
    }
    for values in (
        transition["hard_window_temporal_delta_error"],
        transition["action_map_temporal_delta_error"],
    ):
        values["delta_lora_minus_frozen"] = values["lora_050"] - values["frozen"]

    center = action_transitions[0] if action_transitions else 17
    visual = args.output_dir / "visuals" / "lora050_long_video_boundary.png"
    save_boundary_visual(
        visual, reference, decoded, frozen, lora, frame_actions, center)
    video_frames = args.output_dir / "comparison_frames"
    save_video_frames(video_frames, reference, decoded, frozen, lora)
    result = {
        "experiment": "SeedVR2 LoRA-0.50 Generate-ROI and long-video reintegration",
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit": git_commit(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "scientific_role": plan["source_role"],
        "frames": 33,
        "stream_bytes": int(lora_eval["actual_stream_bytes"]),
        "lora_strength": float(plan["selected_lora_strength"]),
        "quality": {
            "frozen_seedvr2_roi": frozen_quality,
            "lora_050_seedvr2_roi": lora_quality,
            "delta_lora_minus_frozen": quality_delta,
        },
        "spatial_generate_boundary": {
            "frozen_seedvr2_roi": frozen_boundary,
            "lora_050_seedvr2_roi": lora_boundary,
            "delta_lora_minus_frozen": boundary_delta,
        },
        "temporal_transitions": transition,
        "pixel_scope_regression": scope,
        "runtime": {
            "lora_component_inference_seconds_sum": batch[
                "component_inference_seconds_sum"],
            "lora_component_wall_seconds_sum": batch["component_wall_seconds_sum"],
            "lora_model_load_seconds": batch["model_load_seconds_this_process"],
            "lora_restore_process_seconds": batch["this_process_wall_seconds"],
            "lora_evaluation_process_seconds": lora_eval["runtime"][
                "evaluation_this_process_seconds"],
            "peak_cuda_allocated_bytes": lora_eval["runtime"][
                "peak_cuda_allocated_bytes"],
        },
        "verification": {
            "same_continuous_stream_and_actual_bytes": True,
            "same_time_varying_action_maps_and_roi_geometry": True,
            "same_component_noise_seeds": True,
            "fresh_decode_pixel_exact": True,
            "base_frozen_output_hash_unchanged": True,
            "adapter_and_strength_match_frozen_plan": True,
            "outside_generate_pixels_unchanged": True,
            "only_one_visible_gpu_expected_by_runner": True,
        },
        "artifacts": {
            "boundary_visual": str(visual.resolve()),
            "comparison_frames": str(video_frames.resolve()),
            "base_frozen_output_frames_sha256": frame_digest(
                Path(base_eval["storage"]["output_frames"])),
            "lora_050_output_frames_sha256": frame_digest(
                Path(lora_eval["storage"]["output_frames"])),
            "lora_evaluation": str(args.lora_evaluation.resolve()),
            "lora_batch": str(batch_path.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", result)
    write_markdown(args.output_dir / "summary.md", result)
    atomic_text(args.output_dir / "summary.complete", "complete\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def write_markdown(path: Path, result: dict) -> None:
    quality = result["quality"]
    boundary = result["spatial_generate_boundary"]
    transition = result["temporal_transitions"]
    lines = [
        "# SeedVR2 LoRA 0.50 ROI / long-video reintegration",
        "",
        "同一条 33 帧 spatial-QP 码流、同一路由、同一 ROI、同一 seed；只把 SeedVR2 LoRA 从 0 改为 0.50。",
        "",
        "| 输出 | LPIPS | PSNR | 时序误差 |",
        "|---|---:|---:|---:|",
        (f"| 冻结 SeedVR2 ROI | {quality['frozen_seedvr2_roi']['lpips_alex']:.6f} | "
         f"{quality['frozen_seedvr2_roi']['psnr_db']:.4f} | "
         f"{quality['frozen_seedvr2_roi']['temporal_delta_mae']:.4f} |"),
        (f"| LoRA 0.50 ROI | {quality['lora_050_seedvr2_roi']['lpips_alex']:.6f} | "
         f"{quality['lora_050_seedvr2_roi']['psnr_db']:.4f} | "
         f"{quality['lora_050_seedvr2_roi']['temporal_delta_mae']:.4f} |"),
        (f"| LoRA - frozen | {quality['delta_lora_minus_frozen']['lpips_alex']:+.6f} | "
         f"{quality['delta_lora_minus_frozen']['psnr_db']:+.4f} | "
         f"{quality['delta_lora_minus_frozen']['temporal_delta_mae']:+.4f} |"),
        "",
        ("Generate 边界带 RGB MAE 变化："
         f"{boundary['delta_lora_minus_frozen']['boundary_band_rgb_mae']:+.6f}；"
         "LoRA 与冻结版在所有非 Generate 像素上逐像素一致。"),
        ("动作图切换处时序误差变化："
         f"{transition['action_map_temporal_delta_error']['delta_lora_minus_frozen']:+.6f}；"
         "负数更好。"),
    ]
    atomic_text(path, "\n".join(lines) + "\n")


def self_test() -> None:
    frozen = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    lora = [value.copy() for value in frozen]
    lora[0][:2, :2] = 3
    lora[1][:2, :2] = 5
    actions = [np.asarray([[1, 0], [0, 0]], dtype=np.int64) for _ in range(2)]
    scope = pixel_scope_stats(frozen, lora, actions, 2)
    assert scope["outside_generate_pixel_exact"]
    assert scope["inside_generate_changed_rgb_pixel_count"] == 8
    reference = [value.copy() for value in frozen]
    assert temporal_pair_delta_error(reference, frozen, [1]) == 0.0
    assert temporal_pair_delta_error(reference, lora, [1]) == 0.5
    print(json.dumps({
        "status": "passed",
        "pixel_scope_regression": True,
        "transition_metric": True,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan_main(args)
    elif args.command == "summarize":
        summarize_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
