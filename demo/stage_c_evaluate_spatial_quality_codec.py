#!/usr/bin/env python3
"""Evaluate E24's legal spatial-QP stream and SeedVR2 three-path output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_three_path_roi_probe import (
    LPIPSAlex,
    crop_tile_frames,
    evaluate_variant,
)


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_NAMES = {
    ACTION_BASE: "Base",
    ACTION_GENERATE: "Generate",
    ACTION_ENHANCE: "Enhance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--seedvr2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feather-pixels", type=int, default=8)
    parser.add_argument("--visual-frame", type=int, default=9)
    return parser.parse_args()


def generate_composite(
    decoded: list[np.ndarray],
    generated: list[np.ndarray],
    actions: np.ndarray,
    tile_size: int,
    feather: int,
) -> list[np.ndarray]:
    """Use SeedVR2 only for Generate cells with a fixed decoder-side blend."""
    rows, columns = actions.shape
    outputs = [frame.astype(np.float32).copy() for frame in decoded]
    for row in range(rows):
        for column in range(columns):
            if actions[row, column] != ACTION_GENERATE:
                continue
            y, x = row * tile_size, column * tile_size
            alpha = np.ones((tile_size, tile_size), dtype=np.float32)
            if feather:
                ramp = np.arange(feather, dtype=np.float32) / feather

                def is_generate(r: int, c: int) -> bool:
                    return 0 <= r < rows and 0 <= c < columns and (
                        actions[r, c] == ACTION_GENERATE)

                if not is_generate(row, column - 1):
                    alpha[:, :feather] *= ramp[None]
                if not is_generate(row, column + 1):
                    alpha[:, -feather:] *= ramp[::-1][None]
                if not is_generate(row - 1, column):
                    alpha[:feather] *= ramp[:, None]
                if not is_generate(row + 1, column):
                    alpha[-feather:] *= ramp[::-1, None]
            alpha = alpha[:, :, None]
            for index, source in enumerate(generated):
                old = decoded[index][y:y + tile_size, x:x + tile_size].astype(np.float32)
                new = source[y:y + tile_size, x:x + tile_size].astype(np.float32)
                outputs[index][y:y + tile_size, x:x + tile_size] = (
                    old + alpha * (new - old))
    return [np.clip(frame, 0, 255).round().astype(np.uint8) for frame in outputs]


def local_metrics(
    metric: LPIPSAlex,
    reference: list[np.ndarray],
    reconstruction: list[np.ndarray],
    actions: np.ndarray,
    tile_size: int,
) -> dict:
    result = {}
    for action, name in ACTION_NAMES.items():
        values = []
        for row, column in zip(*np.where(actions == action)):
            box = (column * tile_size, row * tile_size, tile_size, tile_size)
            values.append(metric.evaluate(
                crop_tile_frames(reference, box),
                crop_tile_frames(reconstruction, box),
            ))
        result[name] = {
            "tile_count": len(values),
            "mean_lpips_alex": float(np.mean(values)) if values else None,
        }
    return result


def save_frames(path: Path, frames: list[np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        Image.fromarray(frame).save(path / f"im{index:05d}.png")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def panel(frame: np.ndarray, title: str, quality: dict | None) -> Image.Image:
    image = Image.fromarray(frame)
    banner = 68
    output = Image.new("RGB", (image.width, image.height + banner), "white")
    output.paste(image, (0, banner))
    draw = ImageDraw.Draw(output)
    draw.text((8, 7), title, fill="black", font=font(18, True))
    if quality:
        draw.text(
            (8, 39),
            f"LPIPS {quality['lpips_alex']:.4f} | "
            f"PSNR {quality['psnr_db']:.3f} | "
            f"T {quality['temporal_delta_mae']:.3f}",
            fill=(50, 50, 50), font=font(14))
    return output


def comparison_visual(
    path: Path,
    frame_index: int,
    reference: list[np.ndarray],
    variants: dict[str, dict],
) -> None:
    order = (
        ("GT", reference, None),
        ("Scalar Base QP16", variants["scalar-base-qp16"]["frames"],
         variants["scalar-base-qp16"]["quality"]),
        ("Spatial QP raw (8/16/32)", variants["spatial-qp-raw"]["frames"],
         variants["spatial-qp-raw"]["quality"]),
        ("SeedVR2 full output", variants["all-generate-spatial-input"]["frames"],
         variants["all-generate-spatial-input"]["quality"]),
        ("Spatial B/G/E stitched", variants["spatial-bge-stitched"]["frames"],
         variants["spatial-bge-stitched"]["quality"]),
        ("Old independent-tile fallback", variants["old-tile-fallback"]["frames"],
         variants["old-tile-fallback"]["quality"]),
    )
    panels = [panel(frames[frame_index], title, quality)
              for title, frames, quality in order]
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new("RGB", (3 * width, 2 * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % 3) * width, (index // 3) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def exact_frame_comparison(first: list[np.ndarray], second: list[np.ndarray]) -> dict:
    if len(first) != len(second):
        return {"frame_count_equal": False, "max_abs_pixel_error": None}
    maximum = max(
        int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        for a, b in zip(first, second)
    )
    return {
        "frame_count_equal": True,
        "max_abs_pixel_error": maximum,
        "pixel_exact": maximum == 0,
    }


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    route = json.loads(args.route_summary.read_text(encoding="utf-8"))
    encode = json.loads((args.codec_dir / "encode_summary.json").read_text())
    decode = json.loads((args.codec_dir / "decode_summary.json").read_text())
    seed = json.loads((args.seedvr2_dir / "seedvr2_metadata.json").read_text())
    frame_count = encode["frames"]
    reference = load_source(gate, frame_count)
    encoder_reconstruction = load_pngs(args.codec_dir / "encoder_reconstruction")
    decoded = load_pngs(args.codec_dir / "fresh_decode")
    generated = load_pngs(args.seedvr2_dir)
    base = load_pngs(args.gate_summary.parent / "frames" / "qp16-base")
    old_fallback = load_pngs(
        args.route_summary.parent / "frames" / "joint-three-path-oracle")
    if not all(len(value) == frame_count for value in (
            encoder_reconstruction, decoded, generated, base, old_fallback)):
        raise ValueError("variant frame count mismatch")

    tile_size = int(route["configuration"]["tile_size"])
    grid_height, grid_width = route["configuration"]["tile_grid"]
    actions = np.asarray(
        route["variants"]["joint-three-path-oracle"]["actions"], dtype=np.int64,
    ).reshape(grid_height, grid_width)
    stitched = generate_composite(
        decoded, generated, actions, tile_size, args.feather_pixels)
    save_frames(args.output_dir / "frames" / "spatial-bge-stitched", stitched)

    metric = LPIPSAlex(True)
    frames_by_name = {
        "scalar-base-qp16": base,
        "spatial-qp-raw": decoded,
        "all-generate-spatial-input": generated,
        "spatial-bge-stitched": stitched,
        "old-tile-fallback": old_fallback,
    }
    variants = {
        name: {
            "frames": frames,
            "quality": evaluate_variant(reference, frames, metric),
            "local_by_action": local_metrics(
                metric, reference, frames, actions, tile_size),
        }
        for name, frames in frames_by_name.items()
    }

    stream_bytes = int(encode["stream_bytes"])
    scalar_candidates = route["ordinary_scalar_qp_baselines"]
    nearest_name, nearest = min(
        scalar_candidates.items(),
        key=lambda item: abs(item[1]["rate"]["total_bytes"] - stream_bytes),
    )
    old = route["variants"]["joint-three-path-oracle"]
    spatial_quality = variants["spatial-bge-stitched"]["quality"]
    nearest_quality = nearest["quality"]
    seed_total = seed.get("total_after_argument_parse_seconds")
    full_seconds = (
        decode["total_after_argument_parse_seconds"] + seed_total
        if seed_total is not None else None)
    full_peak = max(
        int(decode["peak_cuda_allocated_bytes"]),
        int(seed["peak_cuda_allocated_bytes"]),
    )
    visual = args.output_dir / "visuals" / "gt_scalar_spatial_seed_stitched.png"
    frame_index = min(max(args.visual_frame, 1), frame_count) - 1
    comparison_visual(visual, frame_index, reference, variants)

    serializable_variants = {
        name: {key: value for key, value in record.items() if key != "frames"}
        for name, record in variants.items()
    }
    result = {
        "experiment": "E24 legal one-shot spatial-QP stream plus SeedVR2",
        "status": "no-training-codec-roundtrip-and-three-path-complete",
        "source_role": gate["source_role"],
        "source_files": gate["source_files"][:frame_count],
        "frames": frame_count,
        "stream": {
            "path": encode["stream"],
            "actual_on_disk_bytes": stream_bytes,
            "sps_bytes": encode["sps_bytes"],
            "coding_units": encode["unit_bytes"],
            "maps_headers_and_all_substreams_charged": True,
        },
        "route": {
            "source": "E21 encoder-side local-LPIPS Oracle",
            "tile_size": tile_size,
            "actions": actions.tolist(),
            "action_counts": route["variants"]["joint-three-path-oracle"]["action_counts"],
            "budget_known_before_encoding": True,
            "controller_trained": False,
        },
        "quality_profile": encode["quality_profile"],
        "scale_interpolation": encode["scale_interpolation"],
        "fresh_decode_regression": exact_frame_comparison(
            encoder_reconstruction, decoded),
        "variants": serializable_variants,
        "rate_comparison": {
            "nearest_scalar_name": nearest_name,
            "nearest_scalar_bytes": nearest["rate"]["total_bytes"],
            "nearest_scalar_quality": nearest_quality,
            "spatial_stream_byte_delta": stream_bytes - nearest["rate"]["total_bytes"],
            "stitched_lpips_delta": (
                spatial_quality["lpips_alex"] - nearest_quality["lpips_alex"]),
            "stitched_psnr_delta_db_report_only": (
                spatial_quality["psnr_db"] - nearest_quality["psnr_db"]),
            "stitched_temporal_delta_mae_delta": (
                spatial_quality["temporal_delta_mae"]
                - nearest_quality["temporal_delta_mae"]),
            "old_tile_fallback_bytes": old["rate"]["total_bytes"],
            "old_tile_fallback_lpips": old["quality"]["lpips_alex"],
            "new_spatial_byte_delta_vs_old_fallback": (
                stream_bytes - old["rate"]["total_bytes"]),
            "new_spatial_lpips_delta_vs_old_fallback": (
                spatial_quality["lpips_alex"] - old["quality"]["lpips_alex"]),
        },
        "runtime": {
            "codec_fresh_decode_after_argument_parse_seconds": (
                decode["total_after_argument_parse_seconds"]),
            "codec_bitstream_decode_seconds": decode["bitstream_decode_seconds"],
            "seedvr2_after_argument_parse_seconds": seed_total,
            "seedvr2_inference_seconds_model_load_excluded": seed["runtime_seconds"],
            "full_serial_decode_plus_generate_seconds": full_seconds,
            "full_peak_cuda_allocated_bytes": full_peak,
            "execution": "codec and SeedVR2 executed serially in fresh processes",
        },
        "scientific_boundary": {
            "actual_spatial_entropy_stream": True,
            "fresh_decoder_reads_source_rgb": False,
            "generate_outside_reference_loop": True,
            "different_qp_latents_spliced": False,
            "area_prorated_bytes": False,
            "true_fill_used": False,
            "omitted_latent_prediction_used": False,
            "psnr_used_for_routing": False,
            "lpips_is_primary": True,
            "training_or_finetuning": False,
            "route_is_oracle_not_learned_controller": True,
        },
        "visual": str(visual),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "summary.json"
    summary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary),
        "visual": str(visual),
        "stream_bytes": stream_bytes,
        "fresh_decode_regression": result["fresh_decode_regression"],
        "quality": {
            name: record["quality"] for name, record in serializable_variants.items()
        },
        "rate_comparison": result["rate_comparison"],
        "runtime": result["runtime"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
