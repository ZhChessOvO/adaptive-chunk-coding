#!/usr/bin/env python3
"""Recompose frozen v5 Generate ROIs with wider feather bands.

This is a post-hoc diagnostic.  It reuses the already decoded spatial-QP
frames and already generated SeedVR2 ROI crops.  It never runs the codec or
SeedVR2 again and never changes the frozen action map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_pngs
from demo.stage_c_evaluate_spatial_quality_codec import (
    exact_frame_comparison,
    generate_composite,
    panel,
)
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_COLORS = np.asarray(
    ((74, 144, 226), (242, 160, 42), (70, 170, 92)), dtype=np.uint8)
QUALITY_NAMES = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
BOUNDARY_NAMES = (
    "gradient_error_mae",
    "output_jump_mae",
    "reference_jump_mae",
    "boundary_band_rgb_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feather-pixels", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--visual-frame", type=int, default=9)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_png(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(frame).save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_generated_canvas(
    decoded: list[np.ndarray], roi_root: Path, manifest: dict,
) -> list[np.ndarray]:
    generated = [frame.copy() for frame in decoded]
    for record in manifest["components"]:
        restored_root = roi_root / f"component_{int(record['index']):02d}" / "restored"
        restored = load_pngs(restored_root)
        if len(restored) != len(decoded):
            raise RuntimeError(f"restored frame count differs: {restored_root}")
        crop = record["crop"]
        x, y = int(crop["x"]), int(crop["y"])
        width, height = int(crop["width"]), int(crop["height"])
        for frame_index, frame in enumerate(restored):
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"restored crop shape differs: {restored_root}")
            generated[frame_index][y:y + height, x:x + width] = frame
    return generated


def cross_generate_edges(
    actions: np.ndarray, tile_size: int,
) -> list[tuple[str, int, int, int]]:
    rows, columns = actions.shape
    edges: list[tuple[str, int, int, int]] = []
    for row in range(rows):
        for column in range(columns - 1):
            if ((actions[row, column] == ACTION_GENERATE)
                    != (actions[row, column + 1] == ACTION_GENERATE)):
                edges.append(("vertical", (column + 1) * tile_size,
                              row * tile_size, (row + 1) * tile_size))
    for row in range(rows - 1):
        for column in range(columns):
            if ((actions[row, column] == ACTION_GENERATE)
                    != (actions[row + 1, column] == ACTION_GENERATE)):
                edges.append(("horizontal", (row + 1) * tile_size,
                              column * tile_size, (column + 1) * tile_size))
    return edges


def boundary_metrics(
    reference: list[np.ndarray], output: list[np.ndarray],
    actions: np.ndarray, tile_size: int, band_radius: int = 48,
) -> dict:
    edges = cross_generate_edges(actions, tile_size)
    if not edges:
        return {
            "cross_generate_boundary_edge_count": 0,
            **{name: None for name in BOUNDARY_NAMES},
        }
    gradient_errors = []
    output_jumps = []
    reference_jumps = []
    band_errors = []
    for source, restored in zip(reference, output):
        source_float = source.astype(np.float32)
        restored_float = restored.astype(np.float32)
        for orientation, coordinate, start, end in edges:
            if orientation == "vertical":
                source_edge_gradient = (
                    source_float[start:end, coordinate]
                    - source_float[start:end, coordinate - 1])
                output_edge_gradient = (
                    restored_float[start:end, coordinate]
                    - restored_float[start:end, coordinate - 1])
                left = max(0, coordinate - band_radius)
                right = min(source.shape[1], coordinate + band_radius)
                band_source = source_float[start:end, left:right]
                band_output = restored_float[start:end, left:right]
                source_normal_gradient = np.diff(band_source, axis=1)
                output_normal_gradient = np.diff(band_output, axis=1)
            else:
                source_edge_gradient = (
                    source_float[coordinate, start:end]
                    - source_float[coordinate - 1, start:end])
                output_edge_gradient = (
                    restored_float[coordinate, start:end]
                    - restored_float[coordinate - 1, start:end])
                top = max(0, coordinate - band_radius)
                bottom = min(source.shape[0], coordinate + band_radius)
                band_source = source_float[top:bottom, start:end]
                band_output = restored_float[top:bottom, start:end]
                source_normal_gradient = np.diff(band_source, axis=0)
                output_normal_gradient = np.diff(band_output, axis=0)
            gradient_errors.append(float(np.mean(np.abs(
                output_normal_gradient - source_normal_gradient))))
            output_jumps.append(float(np.mean(np.abs(output_edge_gradient))))
            reference_jumps.append(float(np.mean(np.abs(source_edge_gradient))))
            band_errors.append(float(np.mean(np.abs(band_output - band_source))))
    return {
        "cross_generate_boundary_edge_count": len(edges),
        "gradient_error_mae": float(np.mean(gradient_errors)),
        "output_jump_mae": float(np.mean(output_jumps)),
        "reference_jump_mae": float(np.mean(reference_jumps)),
        "boundary_band_rgb_mae": float(np.mean(band_errors)),
    }


def fixed_visual(
    path: Path, frame_index: int, reference: list[np.ndarray],
    outputs: dict[int, list[np.ndarray]], qualities: dict[int, dict],
    actions: np.ndarray, tile_size: int,
) -> None:
    panels = [panel(reference[frame_index], "GT", None)]
    for width in sorted(outputs):
        panels.append(panel(
            outputs[width][frame_index], f"Generate feather {width}px",
            qualities[width]))
    action_image = ACTION_COLORS[actions]
    action_image = np.repeat(np.repeat(action_image, tile_size, axis=0),
                             tile_size, axis=1)
    panels.append(panel(action_image, "Frozen action map (B/G/E)", None))
    panel_width = max(item.width for item in panels)
    panel_height = max(item.height for item in panels)
    canvas = Image.new("RGB", (3 * panel_width, 2 * panel_height), (232, 232, 232))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % 3) * panel_width,
                            (index // 3) * panel_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def valid_completed_sample(path: Path, widths: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        value = read(path)
    except (OSError, ValueError):
        return False
    return (
        value.get("status") == "complete"
        and tuple(sorted(map(int, value.get("feather_pixels", ())))) == widths
        and value.get("baseline_feather8_regression", {}).get("pixel_exact") is True
    )


def process_sample(
    record: dict, joint_root: Path, output_root: Path,
    widths: tuple[int, ...], visual_frame: int, metric: LPIPSAlex,
) -> dict:
    sample_id = record["sample_id"]
    source_root = joint_root / "formal" / "evaluation" / sample_id
    spatial_root = source_root / "spatial" / "final-joint"
    roi_root = spatial_root / "roi"
    manifest_path = roi_root / "manifest.json"
    manifest = read(manifest_path)
    reference = load_pngs(source_root / "uniform_gate" / "frames" / "original")
    decoded = load_pngs(spatial_root / "codec" / "fresh_decode")
    if len(reference) != 17 or len(decoded) != 17:
        raise RuntimeError(f"expected 17 frames for {sample_id}")
    generated = reconstruct_generated_canvas(decoded, roi_root, manifest)
    actions = np.asarray(manifest["actions"], dtype=np.int64)
    tile_size = int(manifest["tile_size"])
    if actions.shape != (4, 4) or tile_size != 128:
        raise RuntimeError(f"unexpected action geometry for {sample_id}")

    sample_root = output_root / "samples" / sample_id
    frozen_evaluation_root = spatial_root / "evaluation"
    frozen_summary = read(frozen_evaluation_root / "summary.json")
    frozen_frames_root = (
        frozen_evaluation_root / "frames" / "roi-spatial-bge-stitched")
    frozen_frames = load_pngs(frozen_frames_root)
    outputs: dict[int, list[np.ndarray]] = {}
    qualities: dict[int, dict] = {}
    variants = {}
    for width in widths:
        stitched = generate_composite(decoded, generated, actions, tile_size, width)
        quality = (
            frozen_summary["quality"]["roi-spatial-bge-stitched"]
            if width == 8 else evaluate_variant(reference, stitched, metric))
        boundary = boundary_metrics(reference, stitched, actions, tile_size)
        if width == 8:
            frame_root = frozen_frames_root
        else:
            frame_root = sample_root / f"feather_{width:02d}" / "frames"
            for index, frame in enumerate(stitched, start=1):
                save_png(frame, frame_root / f"im{index:05d}.png")
        outputs[width] = stitched
        qualities[width] = quality
        variants[str(width)] = {
            "quality": quality,
            "boundary": boundary,
            "frames": str(frame_root.resolve()),
            "reused_frozen_frames_and_quality": width == 8,
        }

    baseline = exact_frame_comparison(frozen_frames, outputs[8])
    if not baseline.get("pixel_exact"):
        raise RuntimeError(f"8-pixel recomposition differs from frozen v5: {sample_id}")
    visual_path = sample_root / "visuals" / "feather_frame_00009.png"
    fixed_visual(
        visual_path, visual_frame - 1, reference, outputs, qualities,
        actions, tile_size)
    action_counts = Counter(int(value) for value in actions.ravel())
    result = {
        "experiment": "A800 frozen-v5 Generate feather diagnostic",
        "status": "complete",
        "sample_id": sample_id,
        "dataset": record["dataset"],
        "sequence": record["sequence"],
        "data_role_before_diagnostic": record["data_role"],
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "feather_pixels": list(widths),
        "action_counts": {
            "Base": action_counts[ACTION_BASE],
            "Generate": action_counts[ACTION_GENERATE],
            "Enhance": action_counts[ACTION_ENHANCE],
        },
        "baseline_feather8_regression": baseline,
        "variants": variants,
        "fixed_visual": str(visual_path.resolve()),
        "scientific_boundary": {
            "seedvr2_rerun": False,
            "codec_rerun": False,
            "action_map_changed": False,
            "only_decoder_side_feather_changed": True,
            "joint_evaluation_result_overwritten": False,
        },
    }
    atomic_json(sample_root / "summary.json", result)
    return result


def finite_mean(values: list[float | None]) -> float | None:
    selected = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(selected) / len(selected) if selected else None


def aggregate_group(records: list[dict], widths: tuple[int, ...]) -> dict:
    output = {"sample_count": len(records), "feathers": {}}
    for width in widths:
        key = str(width)
        quality = {
            name: finite_mean([
                record["variants"][key]["quality"].get(name)
                for record in records
            ])
            for name in QUALITY_NAMES
        }
        boundary = {
            name: finite_mean([
                record["variants"][key]["boundary"].get(name)
                for record in records
            ])
            for name in BOUNDARY_NAMES
        }
        boundary["sample_count_with_generate_boundary"] = sum(
            record["variants"][key]["boundary"][
                "cross_generate_boundary_edge_count"] > 0
            for record in records)
        lpips_delta = [
            record["variants"][key]["quality"]["lpips_alex"]
            - record["variants"]["8"]["quality"]["lpips_alex"]
            for record in records
        ]
        boundary_delta = [
            record["variants"][key]["boundary"]["gradient_error_mae"]
            - record["variants"]["8"]["boundary"]["gradient_error_mae"]
            for record in records
            if record["variants"][key]["boundary"]["gradient_error_mae"] is not None
        ]
        output["feathers"][key] = {
            "quality": quality,
            "boundary": boundary,
            "comparison_to_8px": {
                "mean_lpips_delta": finite_mean(lpips_delta),
                "lpips_better_count": sum(value < 0 for value in lpips_delta),
                "lpips_equal_count": sum(value == 0 for value in lpips_delta),
                "mean_boundary_gradient_error_delta": finite_mean(boundary_delta),
                "boundary_gradient_better_count": sum(
                    value < 0 for value in boundary_delta),
            },
        }
    return output


def write_aggregate(
    output_root: Path, records: list[dict], widths: tuple[int, ...],
    joint_root: Path,
) -> None:
    aggregate = {
        "experiment": "A800 frozen-v5 Generate feather diagnostic",
        "status": "complete",
        "sample_count": len(records),
        "feather_pixels": list(widths),
        "groups": {
            "combined": aggregate_group(records, widths),
            "REDS": aggregate_group(
                [record for record in records if record["dataset"] == "REDS"],
                widths),
            "UVG": aggregate_group(
                [record for record in records if record["dataset"] == "UVG"],
                widths),
        },
        "source_joint_evaluation": str(joint_root.resolve()),
        "verification": {
            "all_8px_recompositions_match_frozen_v5_pixel_exact": all(
                record["baseline_feather8_regression"].get("pixel_exact")
                for record in records),
            "seedvr2_or_codec_rerun": False,
            "action_maps_changed": False,
        },
        "scientific_boundary": {
            "diagnostic_uses_joint_evaluation_outputs": True,
            "samples_become_development_or_error_analysis_if_used_for_v6": True,
            "future_v6_requires_new_evidence_or_sequence_held_out_reporting": True,
        },
        "samples": records,
    }
    formal_root = output_root / "formal"
    atomic_json(formal_root / "feather_diagnostic_summary.json", aggregate)

    csv_path = formal_root / "feather_diagnostic_samples.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample_id", "dataset", "data_role_before_diagnostic", "feather_pixels",
            "lpips_alex", "psnr_db", "temporal_delta_mae",
            "boundary_gradient_error_mae", "boundary_band_rgb_mae",
            "cross_generate_boundary_edge_count",
        ))
        for record in records:
            for width in widths:
                variant = record["variants"][str(width)]
                writer.writerow((
                    record["sample_id"], record["dataset"],
                    record["data_role_before_diagnostic"], width,
                    variant["quality"]["lpips_alex"],
                    variant["quality"]["psnr_db"],
                    variant["quality"]["temporal_delta_mae"],
                    variant["boundary"]["gradient_error_mae"],
                    variant["boundary"]["boundary_band_rgb_mae"],
                    variant["boundary"]["cross_generate_boundary_edge_count"],
                ))
    os.replace(temporary, csv_path)

    lines = [
        "# A800 冻结 v5 Generate 边界融合诊断",
        "",
        "只复用既有 spatial-QP 解码和 SeedVR2 ROI；没有重新运行 codec／SeedVR2，",
        "也没有改变 action map。8 px 输出必须与冻结 v5 逐像素一致。LPIPS 与边界误差越低越好。",
        "",
        "| 数据 | feather | LPIPS | 相对 8px | LPIPS 改善数 | 边界梯度误差 | 相对 8px | 边界改善数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name in ("combined", "REDS", "UVG"):
        group = aggregate["groups"][group_name]
        for width in widths:
            value = group["feathers"][str(width)]
            comparison = value["comparison_to_8px"]
            lines.append(
                f"| {group_name} ({group['sample_count']}) | {width} | "
                f"{value['quality']['lpips_alex']:.6f} | "
                f"{comparison['mean_lpips_delta']:+.6f} | "
                f"{comparison['lpips_better_count']}/{group['sample_count']} | "
                f"{value['boundary']['gradient_error_mae']:.6f} | "
                f"{comparison['mean_boundary_gradient_error_delta']:+.6f} | "
                f"{comparison['boundary_gradient_better_count']}/"
                f"{value['boundary']['sample_count_with_generate_boundary']} |")
    lines.extend((
        "",
        "边界梯度误差只统计 Generate／非 Generate 相邻的 tile 边；没有 Generate 边界的样本不进入该均值。",
        "本诊断一旦用于设计 v6，对应 37 条样本应改记为开发／误差分析材料。",
        "",
    ))
    atomic_text(formal_root / "feather_diagnostic_summary.md", "\n".join(lines))
    atomic_text(output_root / "formal.complete", "complete\n")


def main() -> None:
    args = parse_args()
    widths = tuple(sorted(set(args.feather_pixels)))
    if widths != tuple(args.feather_pixels) or 8 not in widths:
        raise ValueError("feather widths must be unique, sorted, and include 8")
    if any(width < 0 or width > 64 for width in widths):
        raise ValueError("feather widths must lie in [0, 64]")
    if not 1 <= args.visual_frame <= 17:
        raise ValueError("visual frame must lie in [1, 17]")
    manifest_path = args.joint_root / "manifests" / "joint_samples.jsonl"
    manifest = read_jsonl(manifest_path)
    if len(manifest) != 37 or len({record["sample_id"] for record in manifest}) != 37:
        raise RuntimeError("joint sample manifest is not the fixed 37-sample ledger")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = [
        record for record in manifest
        if not valid_completed_sample(
            args.output_root / "samples" / record["sample_id"] / "summary.json",
            widths)
    ]
    metric = LPIPSAlex(True) if pending else None
    for index, record in enumerate(manifest, start=1):
        summary_path = (
            args.output_root / "samples" / record["sample_id"] / "summary.json")
        if valid_completed_sample(summary_path, widths):
            print(f"SKIP {index}/37 {record['sample_id']}", flush=True)
            continue
        if metric is None:
            raise RuntimeError("LPIPS metric was not initialized")
        print(f"START {index}/37 {record['sample_id']}", flush=True)
        process_sample(
            record, args.joint_root, args.output_root, widths,
            args.visual_frame, metric)
        print(f"COMPLETE {index}/37 {record['sample_id']}", flush=True)
    records = [
        read(args.output_root / "samples" / record["sample_id"] / "summary.json")
        for record in manifest
    ]
    write_aggregate(args.output_root, records, widths, args.joint_root)
    print(json.dumps({
        "status": "complete",
        "sample_count": len(records),
        "output": str((args.output_root / "formal" /
                       "feather_diagnostic_summary.json").resolve()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
