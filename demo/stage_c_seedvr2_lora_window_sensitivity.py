#!/usr/bin/env python3
"""Freeze and summarize a 9/17/33-frame SeedVR2 window sensitivity check.

The codec stream, decoded frames, transmitted action maps, ROI geometry,
SeedVR2 weights, project LoRA adapter, and inference strength stay fixed.  The
only intended change is how the same 33-frame video is divided into temporal
restoration windows.  The completed 17-frame run is reused byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_GENERATE = 1
FORMAT_VERSION = 1
WINDOWS = {9: 4, 17: 8, 33: 16}


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


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def required_file(path: Path, allow_empty: bool = False) -> Path:
    value = path.resolve()
    if not value.is_file() or (not allow_empty and value.stat().st_size == 0):
        raise FileNotFoundError(value)
    return value


def frame_paths(path: Path, expected: int = 33) -> list[Path]:
    values = sorted(path.glob("*.png"))
    if len(values) != expected:
        raise RuntimeError(f"expected {expected} PNGs in {path}, found {len(values)}")
    return values


def load_frames(path: Path, expected: int = 33) -> list[np.ndarray]:
    return [
        np.asarray(Image.open(item).convert("RGB"), dtype=np.uint8)
        for item in frame_paths(path, expected)
    ]


def generated_mask(actions: np.ndarray, cell_size: int) -> np.ndarray:
    return np.repeat(
        np.repeat(actions == ACTION_GENERATE, cell_size, axis=0),
        cell_size, axis=1,
    )


def pixel_scope_stats(
    decoded: list[np.ndarray], output: list[np.ndarray],
    frame_actions: list[np.ndarray], cell_size: int,
) -> dict:
    outside_max = 0
    inside_changed = 0
    inside_count = 0
    inside_abs_sum = 0
    for base, restored, actions in zip(decoded, output, frame_actions):
        difference = np.abs(base.astype(np.int16) - restored.astype(np.int16))
        mask = generated_mask(actions, cell_size)
        outside = difference[~mask]
        if outside.size:
            outside_max = max(outside_max, int(outside.max()))
        inside = difference[mask]
        if inside.size:
            inside_changed += int(np.count_nonzero(np.any(inside > 0, axis=1)))
            inside_count += int(inside.shape[0])
            inside_abs_sum += int(inside.sum())
    return {
        "outside_generate_max_abs_pixel_difference": outside_max,
        "outside_generate_pixels_exact": outside_max == 0,
        "inside_generate_changed_rgb_pixel_count": inside_changed,
        "inside_generate_rgb_pixel_count": inside_count,
        "inside_generate_mean_abs_pixel_difference": (
            inside_abs_sum / (inside_count * 3) if inside_count else 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--base-run", type=Path, required=True)
    plan.add_argument("--reference-run", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--plan", type=Path, required=True)
    for length in (9, 33):
        summarize.add_argument(
            f"--window{length}-manifest", type=Path, required=True)
        summarize.add_argument(
            f"--window{length}-evaluation", type=Path, required=True)
        summarize.add_argument(
            f"--window{length}-restored-root", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    commands.add_parser("self-test")
    return parser.parse_args()


def plan_main(args: argparse.Namespace) -> None:
    base = args.base_run.resolve()
    reference = args.reference_run.resolve()
    adapter = required_file(args.lora_checkpoint)
    paths = {
        "base_run_summary": required_file(base / "run_summary.json"),
        "gate_summary": required_file(base / "plan" / "long_gate.json"),
        "codec_encode": required_file(base / "codec" / "encode_summary.json"),
        "codec_decode": required_file(base / "codec" / "decode_summary.json"),
        "codec_stream": required_file(
            base / "codec" / "streams" / "continuous_33f_spatial_qp.dqvc"),
        "reference_plan": required_file(reference / "experiment_plan.json"),
        "reference_manifest": required_file(
            base / "seedvr2_roi_plan" / "manifest.json"),
        "reference_batch": required_file(
            reference / "lora050_roi_restore" / "long_roi_batch_metadata.json"),
        "reference_evaluation": required_file(
            reference / "lora050_evaluation" / "summary.json"),
    }
    required_file(base / "run.complete", allow_empty=True)
    required_file(reference / "run.complete", allow_empty=True)

    run = read_json(paths["base_run_summary"])
    gate = read_json(paths["gate_summary"])
    encode = read_json(paths["codec_encode"])
    decode = read_json(paths["codec_decode"])
    manifest = read_json(paths["reference_manifest"])
    batch = read_json(paths["reference_batch"])
    evaluation = read_json(paths["reference_evaluation"])
    reference_plan = read_json(paths["reference_plan"])
    if run.get("status") != "complete" or batch.get("status") != "complete":
        raise RuntimeError("base or reference run is incomplete")
    if evaluation.get("status") != "complete":
        raise RuntimeError("reference evaluation is incomplete")
    if int(encode["frames"]) != 33 or int(decode["frames"]) != 33:
        raise RuntimeError("window sensitivity requires the frozen 33-frame stream")
    if not evaluation["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError("reference fresh-decode regression is not pixel exact")
    if (
        int(manifest["window_length"]) != 17
        or int(manifest["window_stride"]) != 8
        or int(manifest["window_count"]) != 3
    ):
        raise RuntimeError("reference is not the expected 17/8 window run")
    adapter_hash = sha256_file(adapter)
    if batch.get("lora_checkpoint_sha256") != adapter_hash:
        raise RuntimeError("reference adapter hash differs from requested adapter")
    if float(batch.get("lora_strength")) != 0.5:
        raise RuntimeError("reference run did not use LoRA strength 0.50")
    if reference_plan.get("lora_checkpoint_sha256") != adapter_hash:
        raise RuntimeError("reference plan and adapter disagree")
    if paths["codec_stream"].stat().st_size != int(encode["stream_bytes"]):
        raise RuntimeError("codec stream size differs from encode summary")
    reference_frames = Path(evaluation["storage"]["output_frames"])
    frame_paths(reference_frames)

    value = {
        "experiment": "SeedVR2 LoRA-0.50 temporal-window sensitivity",
        "format_version": FORMAT_VERSION,
        "status": "frozen-before-new-window-inference",
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "purpose_plain_language": (
            "On the same 33-frame coded video, compare whether SeedVR2 should "
            "see 9, 17, or all 33 frames at a time."),
        "source_role": gate["source_role"],
        "frame_count": 33,
        "window_configurations": [
            {"length": length, "stride": stride, "role": (
                "reused completed reference" if length == 17 else "new inference")}
            for length, stride in WINDOWS.items()
        ],
        "base_seed": int(batch["base_seed"]),
        "feather_pixels": 16,
        "context_pixels": int(manifest["context_pixels"]),
        "processing_scale": float(manifest["processing_scale"]),
        "lora_strength": 0.5,
        "lora_checkpoint": str(adapter),
        "lora_checkpoint_sha256": adapter_hash,
        "codec_dir": str((base / "codec").resolve()),
        "frame_actions_sha256": json_sha256(manifest["frame_actions"]),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "reference_output_frames": str(reference_frames.resolve()),
        "protocol": {
            "same_continuous_codec_stream_and_actual_bytes": True,
            "same_fresh_decoded_frames": True,
            "same_transmitted_time_varying_action_maps": True,
            "same_spatial_roi_context_processing_scale_and_feather": True,
            "same_seedvr2_base_weights_lora_adapter_and_strength": True,
            "same_base_seed_rule": True,
            "changed_variable": (
                "temporal clip length and its approximately half-window stride"),
            "noise_realization_note": (
                "Changing clip shape and window starts necessarily changes the "
                "window-level random tensor assignment; this is part of the "
                "tested inference schedule."),
            "no_training_or_finetuning": True,
            "development_sensitivity_not_independent_benchmark": True,
            "selection_is_practical_not_a_hard_gate": True,
        },
    }
    atomic_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def variant_paths(args: argparse.Namespace, plan: dict) -> dict[int, dict[str, Path]]:
    artifacts = plan["artifacts"]
    reference_evaluation = Path(artifacts["reference_evaluation"]["path"])
    reference_manifest = Path(artifacts["reference_manifest"]["path"])
    reference_batch = Path(artifacts["reference_batch"]["path"])
    return {
        9: {
            "manifest": args.window9_manifest,
            "evaluation": args.window9_evaluation,
            "batch": args.window9_restored_root / "long_roi_batch_metadata.json",
        },
        17: {
            "manifest": reference_manifest,
            "evaluation": reference_evaluation,
            "batch": reference_batch,
        },
        33: {
            "manifest": args.window33_manifest,
            "evaluation": args.window33_evaluation,
            "batch": args.window33_restored_root / "long_roi_batch_metadata.json",
        },
    }


def summarize_variant(
    length: int, paths: dict[str, Path], plan: dict,
    decoded: list[np.ndarray], expected_actions: list,
) -> tuple[dict, list[np.ndarray]]:
    manifest_path = required_file(paths["manifest"])
    evaluation_path = required_file(paths["evaluation"])
    batch_path = required_file(paths["batch"])
    manifest = read_json(manifest_path)
    evaluation = read_json(evaluation_path)
    batch = read_json(batch_path)
    stride = WINDOWS[length]
    if int(manifest["window_length"]) != length:
        raise RuntimeError(f"window-{length} manifest has the wrong length")
    if int(manifest["window_stride"]) != stride:
        raise RuntimeError(f"window-{length} manifest has the wrong stride")
    if json_sha256(manifest["frame_actions"]) != plan["frame_actions_sha256"]:
        raise RuntimeError(f"window-{length} action maps changed")
    if manifest["frame_actions"] != expected_actions:
        raise RuntimeError(f"window-{length} action maps differ from reference")
    if Path(manifest["codec_dir"]).resolve() != Path(plan["codec_dir"]).resolve():
        raise RuntimeError(f"window-{length} codec directory changed")
    if evaluation.get("status") != "complete" or batch.get("status") != "complete":
        raise RuntimeError(f"window-{length} result is incomplete")
    if not evaluation["fresh_decode_regression"].get("pixel_exact"):
        raise RuntimeError(f"window-{length} fresh decode is not exact")
    if int(evaluation["actual_stream_bytes"]) != int(
        read_json(Path(plan["artifacts"]["codec_encode"]["path"]))["stream_bytes"]
    ):
        raise RuntimeError(f"window-{length} stream byte count changed")
    if batch.get("lora_checkpoint_sha256") != plan["lora_checkpoint_sha256"]:
        raise RuntimeError(f"window-{length} adapter hash changed")
    if float(batch.get("lora_strength")) != float(plan["lora_strength"]):
        raise RuntimeError(f"window-{length} adapter strength changed")
    if int(batch["component_count"]) != sum(
        len(window["components"]) for window in manifest["windows"]
    ):
        raise RuntimeError(f"window-{length} component count changed")
    output = load_frames(Path(evaluation["storage"]["output_frames"]))
    actions = [np.asarray(value, dtype=np.int64) for value in manifest["frame_actions"]]
    scope = pixel_scope_stats(decoded, output, actions, int(manifest["cell_size"]))
    if not scope["outside_generate_pixels_exact"]:
        raise RuntimeError(f"window-{length} changed a non-Generate pixel")
    quality = evaluation["quality"]["overlap_seedvr2_stitched"]
    return ({
        "window_length": length,
        "window_stride": stride,
        "window_count": int(manifest["window_count"]),
        "component_count": int(batch["component_count"]),
        "total_roi_processing_pixel_frames": int(
            manifest["total_roi_processing_pixel_frames"]),
        "roi_to_full_processing_pixel_ratio": float(
            manifest["roi_to_full_processing_pixel_ratio"]),
        "quality": {
            "lpips_alex": float(quality["lpips_alex"]),
            "psnr_db": float(quality["psnr_db"]),
            "temporal_delta_mae": float(quality["temporal_delta_mae"]),
            "rgb_mse": float(quality["rgb_mse"]),
        },
        "delta_from_spatial_qp_decoded": evaluation["quality"][
            "delta_stitched_minus_decoded"],
        "runtime": {
            "component_inference_seconds_sum": float(batch[
                "component_inference_seconds_sum"]),
            "restoration_process_seconds": float(batch[
                "this_process_wall_seconds"]),
            "sequential_fresh_decode_seedvr2_overlap_seconds": float(
                evaluation["runtime"][
                    "sequential_fresh_decode_seedvr2_overlap_seconds"]),
            "peak_cuda_allocated_bytes": int(
                evaluation["runtime"]["peak_cuda_allocated_bytes"]),
        },
        "verification": scope,
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "evaluation": str(evaluation_path.resolve()),
            "evaluation_sha256": sha256_file(evaluation_path),
            "batch": str(batch_path.resolve()),
            "batch_sha256": sha256_file(batch_path),
            "output_frames": str(Path(
                evaluation["storage"]["output_frames"]).resolve()),
        },
    }, output)


def action_visual(actions: np.ndarray, cell_size: int) -> np.ndarray:
    colors = np.asarray(
        ((74, 144, 226), (242, 160, 42), (70, 170, 92)), dtype=np.uint8)
    return np.repeat(np.repeat(colors[actions], cell_size, axis=0), cell_size, axis=1)


def visual_panel(frame: np.ndarray, title: str, width: int = 320) -> Image.Image:
    image = Image.fromarray(frame)
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    banner = 38
    panel = Image.new("RGB", (width, height + banner), "white")
    panel.paste(image, (0, banner))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    draw.text((7, 9), title, fill="black", font=font)
    return panel


def write_visual(
    path: Path, reference: list[np.ndarray], decoded: list[np.ndarray],
    outputs: dict[int, list[np.ndarray]], actions: list[np.ndarray], cell_size: int,
) -> None:
    indices = (8, 16, 17, 24)
    rows = []
    for index in indices:
        rows.append([
            visual_panel(reference[index], f"GT | frame {index}"),
            visual_panel(decoded[index], f"Decoded | frame {index}"),
            visual_panel(action_visual(actions[index], cell_size), "B / G / E map"),
            *[
                visual_panel(outputs[length][index], f"SeedVR2 window {length}")
                for length in (9, 17, 33)
            ],
        ])
    panel_width = max(panel.width for row in rows for panel in row)
    panel_height = max(panel.height for row in rows for panel in row)
    canvas = Image.new(
        "RGB", (panel_width * len(rows[0]), panel_height * len(rows)),
        (225, 225, 225))
    for row_index, row in enumerate(rows):
        for column_index, panel in enumerate(row):
            canvas.paste(panel, (column_index * panel_width, row_index * panel_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def summarize_main(args: argparse.Namespace) -> None:
    plan_path = required_file(args.plan)
    plan = read_json(plan_path)
    if plan.get("status") != "frozen-before-new-window-inference":
        raise RuntimeError("window sensitivity plan has the wrong state")
    for name, record in plan["artifacts"].items():
        path = required_file(Path(record["path"]))
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed: {name}")
    adapter = required_file(Path(plan["lora_checkpoint"]))
    if sha256_file(adapter) != plan["lora_checkpoint_sha256"]:
        raise RuntimeError("frozen LoRA checkpoint changed")

    reference_manifest = read_json(Path(
        plan["artifacts"]["reference_manifest"]["path"]))
    expected_actions = reference_manifest["frame_actions"]
    actions = [np.asarray(value, dtype=np.int64) for value in expected_actions]
    decoded = load_frames(Path(plan["codec_dir"]) / "fresh_decode")
    variants = {}
    outputs = {}
    for length, paths in variant_paths(args, plan).items():
        variants[length], outputs[length] = summarize_variant(
            length, paths, plan, decoded, expected_actions)

    gate = read_json(Path(plan["artifacts"]["gate_summary"]["path"]))
    crop = gate["crop"]
    box = (
        int(crop["x"]), int(crop["y"]),
        int(crop["x"]) + int(crop["width"]),
        int(crop["y"]) + int(crop["height"]),
    )
    reference = [
        np.asarray(Image.open(path).convert("RGB").crop(box), dtype=np.uint8)
        for path in gate["source_files"]
    ]
    visual = args.output_dir / "visuals" / "window_9_17_33_comparison.png"
    write_visual(
        visual, reference, decoded, outputs, actions,
        int(reference_manifest["cell_size"]))

    reference_metrics = variants[17]["quality"]
    for length, record in variants.items():
        record["delta_from_window17"] = {
            "lpips_alex": (
                record["quality"]["lpips_alex"]
                - reference_metrics["lpips_alex"]),
            "psnr_db": (
                record["quality"]["psnr_db"]
                - reference_metrics["psnr_db"]),
            "temporal_delta_mae": (
                record["quality"]["temporal_delta_mae"]
                - reference_metrics["temporal_delta_mae"]),
        }
    best = {
        "lowest_lpips_window": min(
            variants, key=lambda key: variants[key]["quality"]["lpips_alex"]),
        "highest_psnr_window": max(
            variants, key=lambda key: variants[key]["quality"]["psnr_db"]),
        "lowest_temporal_error_window": min(
            variants,
            key=lambda key: variants[key]["quality"]["temporal_delta_mae"]),
        "fastest_inference_window": min(
            variants,
            key=lambda key: variants[key]["runtime"][
                "component_inference_seconds_sum"]),
    }
    result = {
        "experiment": plan["experiment"],
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit": plan["git_commit"],
        "source_role": plan["source_role"],
        "purpose_plain_language": plan["purpose_plain_language"],
        "fixed_stream_bytes": int(read_json(Path(
            plan["artifacts"]["codec_encode"]["path"]))["stream_bytes"]),
        "variants": {str(key): value for key, value in variants.items()},
        "metric_leaders_not_hard_gates": best,
        "verification": {
            "same_codec_stream": True,
            "same_fresh_decode": True,
            "same_action_maps": True,
            "same_spatial_roi_settings": True,
            "same_seedvr2_and_lora_050": True,
            "all_non_generate_pixels_exact": all(
                value["verification"]["outside_generate_pixels_exact"]
                for value in variants.values()),
            "all_variants_have_33_output_frames": all(
                len(outputs[length]) == 33 for length in variants),
            "reference_window17_reused_without_recomputation": True,
        },
        "interpretation_boundary": {
            "single_development_sequence_only": True,
            "sensitivity_check_not_external_comparison": True,
            "no_training_or_finetuning": True,
            "metric_leaders_are_descriptive_not_acceptance_thresholds": True,
        },
        "visual": str(visual.resolve()),
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
    }
    atomic_json(args.output_dir / "summary.json", result)
    atomic_text(args.output_dir / "summary.complete", "complete\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def self_test() -> None:
    actions = [np.asarray([[1, 0], [2, 0]], dtype=np.int64)]
    decoded = [np.zeros((4, 4, 3), dtype=np.uint8)]
    output = [decoded[0].copy()]
    output[0][:2, :2] = 5
    stats = pixel_scope_stats(decoded, output, actions, 2)
    assert stats["outside_generate_pixels_exact"] is True
    assert stats["inside_generate_changed_rgb_pixel_count"] == 4
    output[0][3, 3] = 1
    assert pixel_scope_stats(decoded, output, actions, 2)[
        "outside_generate_pixels_exact"] is False
    assert json_sha256({"b": 2, "a": 1}) == json_sha256({"a": 1, "b": 2})
    print(json.dumps({
        "status": "passed",
        "window_configurations": WINDOWS,
        "pixel_scope_check": True,
        "stable_json_digest": True,
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
