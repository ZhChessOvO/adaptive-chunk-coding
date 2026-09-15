#!/usr/bin/env python3
"""E21: perceptual Base / Generate / Enhance response probe.

This diagnostic deliberately stays on the legal fallback format established
by E19: one ordinary full-frame DCVC-UF Base stream plus independently coded
DCVC-UF enhancement tile streams.  No different-QP latents are spliced.  The
action map, container header, descriptors, Base stream, and selected
enhancement streams are all charged from the on-disk container.

Generate uses a deterministic SeedVR2 result produced only from the decoded
Base RGB frames.  Ground truth is visible only to the encoder-side Oracle that
measures local action response.  The decoder never receives source RGB or an
omitted latent.  PSNR is reported but is not used for action selection; local
LPIPS reduction is the primary utility.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_three_path_roi_probe import (
    ACTION_BASE,
    ACTION_ENHANCE,
    ACTION_GENERATE,
    ACTION_NAMES,
    TILE_HEADER,
    LPIPSAlex,
    TilePayload,
    action_counts,
    crop_tile_frames,
    dcvc_stream_breakdown,
    encode_dcvc_stream,
    evaluate_variant,
    fresh_decode_baseline,
    fresh_decode_container,
    load_codecs,
    read_three_path_container,
    save_frames,
    select_actions,
    tile_boxes,
    write_three_path_container,
)
from src.utils.common import set_torch_env


COLORS = {
    ACTION_BASE: (45, 105, 210),
    ACTION_GENERATE: (245, 145, 35),
    ACTION_ENHANCE: (40, 170, 90),
}
SHORT_NAMES = {
    ACTION_BASE: "B",
    ACTION_GENERATE: "G",
    ACTION_ENHANCE: "E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E21 perceptual three-path Oracle with SeedVR2")
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--seedvr2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-qp", type=int, default=16)
    parser.add_argument("--enhance-qp", type=int, default=32)
    parser.add_argument(
        "--ordinary-qps", type=int, nargs="+",
        default=(8, 16, 24, 28, 29, 30, 32))
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--feather-pixels", type=int, default=8)
    parser.add_argument("--enhance-budget-ratio", type=float, default=1.0)
    parser.add_argument("--max-generate-tiles", type=int, default=8)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--visual-frame", type=int, default=9)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--reset-interval", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.base_qp < args.enhance_qp < 64:
        parser.error("require 0 <= Base QP < Enhance QP < 64")
    args.ordinary_qps = sorted(set(args.ordinary_qps))
    if any(not 0 <= value < 64 for value in args.ordinary_qps):
        parser.error("ordinary QPs must be inside [0, 63]")
    if args.tile_size < 64:
        parser.error("independent DCVC-UF tiles must be at least 64 pixels")
    if not 0 <= args.feather_pixels <= args.tile_size // 4:
        parser.error("feather width must be in [0, tile_size/4]")
    if args.enhance_budget_ratio < 0:
        parser.error("--enhance-budget-ratio must be nonnegative")
    if args.max_generate_tiles < 0 or args.decode_repeats < 1:
        parser.error("generate limit must be nonnegative and repeats positive")
    return args


class SeedVR2Replay:
    """Replay a recorded deterministic result while timing it separately.

    This object is used only to verify the saved container's codec streams and
    compositing.  The result JSON replaces the replay duration with the actual
    SeedVR2 inference duration from its metadata before reporting full decode.
    """

    name = "seedvr2-3b-deterministic-replay-for-container-verification"
    local_compute = False

    def __init__(
        self, expected_base: list[np.ndarray], generated: list[np.ndarray]
    ) -> None:
        self.expected_base = expected_base
        self.generated = generated

    def restore(self, frames, selected_tiles, boxes):
        del selected_tiles, boxes
        if len(frames) != len(self.expected_base) or any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(frames, self.expected_base)
        ):
            raise RuntimeError(
                "fresh Base decode differs from SeedVR2's recorded input")
        return [frame.copy() for frame in self.generated]


def local_lpips(
    metric: LPIPSAlex,
    reference: list[np.ndarray],
    reconstruction: list[np.ndarray],
    box: tuple[int, int, int, int],
) -> float:
    return float(metric.evaluate(
        crop_tile_frames(reference, box),
        crop_tile_frames(reconstruction, box),
    ))


def paste_tile(
    base: list[np.ndarray],
    tile_frames: list[np.ndarray],
    box: tuple[int, int, int, int],
) -> list[np.ndarray]:
    x, y, width, height = box
    result = [frame.copy() for frame in base]
    for index, tile in enumerate(tile_frames):
        result[index][y:y + height, x:x + width] = tile
    return result


def feathered_composite_actions(
    base: list[np.ndarray],
    generated: list[np.ndarray],
    enhanced_tiles: dict[int, list[np.ndarray]],
    actions: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    grid_width: int,
    feather: int,
) -> list[np.ndarray]:
    """Blend action boundaries through Base using a fixed decoder rule.

    Adjacent Generate cells share one continuous restored frame and therefore
    need no seam treatment.  Every boundary touching an independently coded
    Enhance tile is feathered because its codec context is tile-local.
    """
    outputs = [frame.astype(np.float32).copy() for frame in base]
    grid_height = len(actions) // grid_width
    for tile_index, (action, box) in enumerate(zip(actions.tolist(), boxes)):
        if action == ACTION_BASE:
            continue
        x, y, width, height = box
        source_frames = (
            generated if action == ACTION_GENERATE
            else enhanced_tiles[tile_index]
        )
        alpha = np.ones((height, width), dtype=np.float32)
        if feather:
            row, column = divmod(tile_index, grid_width)
            ramp = np.arange(feather, dtype=np.float32) / feather

            def same_continuous_generate(neighbor: int) -> bool:
                return (
                    action == ACTION_GENERATE
                    and actions[neighbor] == ACTION_GENERATE
                )

            if column > 0 and not same_continuous_generate(tile_index - 1):
                alpha[:, :feather] *= ramp[None, :]
            if (column + 1 < grid_width
                    and not same_continuous_generate(tile_index + 1)):
                alpha[:, -feather:] *= ramp[::-1][None, :]
            if row > 0 and not same_continuous_generate(tile_index - grid_width):
                alpha[:feather, :] *= ramp[:, None]
            if (row + 1 < grid_height
                    and not same_continuous_generate(tile_index + grid_width)):
                alpha[-feather:, :] *= ramp[::-1][:, None]
        alpha = alpha[:, :, None]
        for frame_index, source in enumerate(source_frames):
            source_tile = (
                source[y:y + height, x:x + width]
                if action == ACTION_GENERATE else source
            ).astype(np.float32)
            base_tile = base[frame_index][
                y:y + height, x:x + width].astype(np.float32)
            outputs[frame_index][y:y + height, x:x + width] = (
                base_tile + alpha * (source_tile - base_tile))
    return [np.clip(frame, 0, 255).round().astype(np.uint8) for frame in outputs]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(frame)
    banner = 68
    result = Image.new("RGB", (image.width, image.height + banner), "white")
    result.paste(image, (0, banner))
    draw = ImageDraw.Draw(result)
    draw.text((8, 7), title, fill="black", font=font(18, True))
    draw.text((8, 39), subtitle, fill=(50, 50, 50), font=font(14))
    return result


def metric_line(quality: dict) -> str:
    return (
        f"LPIPS {quality['lpips_alex']:.4f} | "
        f"PSNR {quality['psnr_db']:.3f} | "
        f"T {quality['temporal_delta_mae']:.3f}"
    )


def make_comparison(
    path: Path,
    frame_index: int,
    reference: list[np.ndarray],
    outputs: dict[str, list[np.ndarray]],
    variants: dict[str, dict],
) -> None:
    names = [
        "base-only",
        "all-generate",
        "deterministic-control",
        "enhance-only-oracle",
        "joint-three-path-oracle",
    ]
    panels = [panel(reference[frame_index], "GT", "source reference")]
    titles = {
        "base-only": "Base QP16",
        "all-generate": "Generate: SeedVR2 raw",
        "deterministic-control": "Base + BasicVSR++",
        "enhance-only-oracle": "Enhance-only Oracle",
        "joint-three-path-oracle": "Stitched B/G/E Oracle",
    }
    for name in names:
        panels.append(panel(
            outputs[name][frame_index], titles[name],
            metric_line(variants[name]["quality"])))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new("RGB", (columns * width, rows * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def make_action_map(
    path: Path,
    frame: np.ndarray,
    actions: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    record: dict,
) -> None:
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, (action, box) in enumerate(zip(actions.tolist(), boxes)):
        x, y, width, height = box
        color = COLORS[action]
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            fill=(*color, 54), outline=(*color, 255), width=4)
        draw.rounded_rectangle(
            (x + 6, y + 6, x + 38, y + 38), radius=6,
            fill=(*color, 235))
        draw.text(
            (x + 22, y + 8), SHORT_NAMES[action], fill="white",
            font=font(21, True), anchor="ma")
        draw.text(
            (x + width - 7, y + 7), str(index), fill="white",
            stroke_width=2, stroke_fill="black", font=font(12), anchor="ra")
    merged = Image.alpha_composite(image, overlay).convert("RGB")
    banner = 78
    canvas = Image.new("RGB", (merged.width, merged.height + banner), "white")
    canvas.paste(merged, (0, banner))
    draw = ImageDraw.Draw(canvas)
    counts = record["action_counts"]
    draw.text((8, 6), "E21 action map", fill="black", font=font(18, True))
    draw.text(
        (8, 35),
        f"Blue B={counts['Base']}  Orange G={counts['Generate']}  "
        f"Green E={counts['Enhance']}  |  {record['rate']['total_bytes']} bytes",
        fill=(50, 50, 50), font=font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E21 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(codec_stream)

    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    if args.base_qp not in gate["configuration"]["qps"]:
        raise ValueError("Base QP is absent from the E20 gate")
    crop = gate["crop"]
    width, height = crop["width"], crop["height"]
    if width % args.tile_size or height % args.tile_size:
        raise ValueError("tile size must divide the E20 crop")
    reference = load_source(gate, gate["frames"])
    base = load_pngs(
        args.gate_summary.parent / "frames" / f"qp{args.base_qp}-base")
    deterministic = load_pngs(
        args.gate_summary.parent / "frames" / f"qp{args.base_qp}-basicvsrpp")
    generated = load_pngs(args.seedvr2_dir)
    if not (
        len(reference) == len(base) == len(deterministic) == len(generated)
        == gate["frames"]
    ):
        raise ValueError("input frame counts differ")
    base_path = Path(gate["variants"][f"qp{args.base_qp}-base"]["path"])
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    base_stream = base_path.read_bytes()
    if dcvc_stream_breakdown(base_stream, len(reference))["total_bytes"] != len(base_stream):
        raise RuntimeError("Base stream byte accounting failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    boxes = tile_boxes(width, height, args.tile_size)
    base_i_net, base_p_net = load_codecs(args, device)
    tile_i_net, tile_p_net = load_codecs(args, device)
    # The released CUDA proxy is created lazily by compress().  Re-encoding
    # once initializes the matching decoder instances and also proves that the
    # E20 Base payload can be reproduced byte-for-byte from the registered
    # source and configuration.
    reproduced_base_stream, reproduced_base_encode = encode_dcvc_stream(
        reference, args.base_qp, args.base_qp,
        base_i_net, base_p_net, device, args.reset_interval)
    if reproduced_base_stream != base_stream:
        raise RuntimeError("reproduced Base stream differs from the E20 file")
    payloads: list[TilePayload] = []
    enhanced: dict[int, list[np.ndarray]] = {}
    candidate_encode_seconds = 0.0
    for index, box in enumerate(boxes):
        source_tile = crop_tile_frames(reference, box)
        stream, encode = encode_dcvc_stream(
            source_tile, args.enhance_qp, args.enhance_qp,
            tile_i_net, tile_p_net, device, args.reset_interval)
        x, y, tile_width, tile_height = box
        path = (
            args.output_dir / "candidate_enhancement_streams"
            / f"tile_{index:03d}_x{x}_y{y}_qp{args.enhance_qp}.dcvc")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stream)
        decoded, _ = fresh_decode_baseline(
            path, len(reference), 1, tile_i_net, tile_p_net,
            device, codec_stream)
        dcvc_stream_breakdown(stream, len(reference))
        payloads.append(TilePayload(
            index, x, y, tile_width, tile_height,
            args.enhance_qp, args.enhance_qp, stream))
        enhanced[index] = decoded
        candidate_encode_seconds += encode["seconds"]

    # LPIPS is intentionally the route utility.  PSNR and temporal metrics are
    # evaluated only after the complete output has been reconstructed.
    lpips = LPIPSAlex(True)
    region_rows = []
    generate_gains = []
    enhance_gains = []
    enhance_costs = []
    for index, (box, payload) in enumerate(zip(boxes, payloads)):
        base_lpips = local_lpips(lpips, reference, base, box)
        generated_lpips = local_lpips(lpips, reference, generated, box)
        enhanced_full = paste_tile(base, enhanced[index], box)
        enhanced_lpips = local_lpips(lpips, reference, enhanced_full, box)
        generate_gain = base_lpips - generated_lpips
        enhance_gain = base_lpips - enhanced_lpips
        enhance_cost = TILE_HEADER.size + len(payload.stream)
        generate_gains.append(generate_gain)
        enhance_gains.append(enhance_gain)
        enhance_costs.append(enhance_cost)
        region_rows.append({
            "tile_index": index,
            "x": box[0],
            "y": box[1],
            "width": box[2],
            "height": box[3],
            "base_lpips": base_lpips,
            "generate_lpips": generated_lpips,
            "enhance_lpips": enhanced_lpips,
            "generate_lpips_reduction": generate_gain,
            "enhance_lpips_reduction": enhance_gain,
            "enhance_stream_bytes": len(payload.stream),
            "enhance_descriptor_bytes": TILE_HEADER.size,
            "enhance_cost_bytes": enhance_cost,
        })

    enhance_budget = int(round(len(base_stream) * args.enhance_budget_ratio))
    joint_actions = select_actions(
        generate_gains, enhance_gains, enhance_costs,
        enhance_budget, args.max_generate_tiles, True, True)
    joint_no_generate = joint_actions.copy()
    joint_no_generate[joint_no_generate == ACTION_GENERATE] = ACTION_BASE
    joint_no_enhance = joint_actions.copy()
    joint_no_enhance[joint_no_enhance == ACTION_ENHANCE] = ACTION_BASE
    actions_by_name = {
        "base-only": np.full(len(boxes), ACTION_BASE, dtype=np.uint8),
        "all-generate": np.full(len(boxes), ACTION_GENERATE, dtype=np.uint8),
        "generate-only-oracle": select_actions(
            generate_gains, enhance_gains, enhance_costs,
            0, args.max_generate_tiles, True, False),
        "enhance-only-oracle": select_actions(
            generate_gains, enhance_gains, enhance_costs,
            enhance_budget, 0, False, True),
        "joint-no-generate-fixed-route": joint_no_generate,
        "joint-no-enhance-fixed-route": joint_no_enhance,
        "joint-three-path-oracle": joint_actions,
    }

    seed_metadata = json.loads(
        (args.seedvr2_dir / "seedvr2_metadata.json").read_text(encoding="utf-8"))
    seed_seconds = float(seed_metadata["runtime_seconds"])
    seed_peak = int(seed_metadata["peak_cuda_allocated_bytes"])
    replay = SeedVR2Replay(base, generated)
    variants: dict[str, dict] = {}
    outputs: dict[str, list[np.ndarray]] = {}
    for name, actions in actions_by_name.items():
        container_path = args.output_dir / "containers" / f"{name}.d3r"
        rate = write_three_path_container(
            container_path, width, height, len(reference), args.tile_size,
            args.base_qp, args.enhance_qp, actions, base_stream, payloads)
        hard_output, replay_runtime = fresh_decode_container(
            container_path, args.decode_repeats, replay,
            base_i_net, base_p_net, tile_i_net, tile_p_net,
            device, codec_stream)
        del hard_output
        parsed = read_three_path_container(container_path)
        if parsed["file_bytes"] != container_path.stat().st_size:
            raise RuntimeError("fresh parser byte count differs from disk")
        uses_generate = bool(np.any(actions == ACTION_GENERATE))
        composite_started = time.perf_counter()
        output = feathered_composite_actions(
            base, generated, enhanced, actions, boxes,
            width // args.tile_size, args.feather_pixels)
        feathered_composite_seconds = time.perf_counter() - composite_started
        non_seed_seconds = (
            replay_runtime["fresh_decode_seconds_median"]
            - replay_runtime["restorer_seconds_median"]
            + feathered_composite_seconds
        )
        runtime = {
            **replay_runtime,
            "recorded_seedvr2_inference_seconds": (
                seed_seconds if uses_generate else 0.0),
            "full_decode_component_sum_seconds": (
                non_seed_seconds + seed_seconds
                if uses_generate else non_seed_seconds),
            "full_decode_peak_cuda_allocated_bytes": max(
                replay_runtime["peak_cuda_allocated_bytes"],
                seed_peak if uses_generate else 0),
            "seedvr2_model_load_in_timing": False,
            "codec_model_load_in_timing": False,
            "container_file_io_in_timing": True,
            "replay_duration_not_reported_as_seedvr2_compute": True,
            "feathered_composite_seconds_cpu": feathered_composite_seconds,
        }
        variants[name] = {
            "path": str(container_path),
            "actions": actions.tolist(),
            "action_counts": action_counts(actions),
            "rate": rate,
            "quality": evaluate_variant(reference, output, lpips),
            "runtime": runtime,
        }
        outputs[name] = output
        save_frames(args.output_dir / "frames" / name, output)

    # The deterministic full-frame reference uses the exact same Base stream
    # and therefore has no additional transmitted bytes.
    ordinary_base_rate = gate["variants"][
        f"qp{args.base_qp}-base"]["rate"]
    basic_runtime = gate["variants"][
        f"qp{args.base_qp}-basicvsrpp"]["runtime"]
    variants["deterministic-control"] = {
        "path": gate["variants"][f"qp{args.base_qp}-base"]["path"],
        "actions": None,
        "action_counts": {
            "Base": len(boxes), "Generate": 0, "Enhance": 0},
        "rate": ordinary_base_rate,
        "quality": evaluate_variant(reference, deterministic, lpips),
        "runtime": {
            "full_decode_component_sum_seconds": basic_runtime[
                "fresh_decode_seconds_median"],
            "full_decode_peak_cuda_allocated_bytes": basic_runtime[
                "peak_cuda_allocated_bytes"],
            "model_load_in_timing": False,
        },
    }
    outputs["deterministic-control"] = deterministic

    baseline_variants = {}
    for qp in args.ordinary_qps:
        name = f"ordinary-dcvc-qp{qp}"
        if qp in gate["configuration"]["qps"]:
            record = gate["variants"][f"qp{qp}-base"]
            baseline_variants[name] = {
                "path": record["path"],
                "rate": record["rate"],
                "quality": record["quality"],
                "runtime": record["runtime"],
                "source": "E20 registered stream",
            }
            continue
        stream, encode_stats = encode_dcvc_stream(
            reference, qp, qp, base_i_net, base_p_net,
            device, args.reset_interval)
        path = args.output_dir / "ordinary_streams" / f"all_base_qp{qp}.dcvc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stream)
        decoded, runtime = fresh_decode_baseline(
            path, len(reference), args.decode_repeats,
            base_i_net, base_p_net, device, codec_stream)
        rate = dcvc_stream_breakdown(stream, len(reference))
        rate["bpp"] = 8.0 * rate["total_bytes"] / (
            len(reference) * width * height)
        baseline_variants[name] = {
            "path": str(path),
            "rate": rate,
            "quality": evaluate_variant(reference, decoded, lpips),
            "runtime": runtime,
            "encode": encode_stats,
            "source": "E21 exact scalar-QP stream",
        }

    frame_index = min(max(args.visual_frame, 1), len(reference)) - 1
    save_frames(args.output_dir / "frames" / "original", reference)
    comparison_path = args.output_dir / "visuals" / "gt_generate_stitched.png"
    make_comparison(
        comparison_path, frame_index, reference, outputs, variants)
    action_path = args.output_dir / "visuals" / "joint_action_map.png"
    make_action_map(
        action_path, reference[frame_index],
        actions_by_name["joint-three-path-oracle"], boxes,
        variants["joint-three-path-oracle"])

    base_quality = variants["base-only"]["quality"]
    all_generate_quality = variants["all-generate"]["quality"]
    joint_quality = variants["joint-three-path-oracle"]["quality"]
    joint_no_generate_quality = variants[
        "joint-no-generate-fixed-route"]["quality"]
    joint_no_enhance_quality = variants[
        "joint-no-enhance-fixed-route"]["quality"]
    joint_bytes = variants["joint-three-path-oracle"]["rate"]["total_bytes"]
    nearest_scalar_name, nearest_scalar = min(
        baseline_variants.items(),
        key=lambda item: abs(item[1]["rate"]["total_bytes"] - joint_bytes))
    summary = {
        "experiment": "E21 perceptual Base/Generate/Enhance response probe",
        "status": "no-training-oracle-diagnostic-complete",
        "sequence": gate["sequence"],
        "source_role": gate["source_role"],
        "source_files": gate["source_files"],
        "crop": crop,
        "frames": len(reference),
        "protocol": {
            "budget_known_before_encoding": True,
            "route_utility": "mean local LPIPS reduction; larger is better",
            "psnr_role": "report-only diagnostic; not used for selection",
            "source_rgb_visible_to_encoder_oracle_only": True,
            "source_rgb_available_to_decoder": False,
            "true_fill_used": False,
            "omitted_latent_used": False,
            "different_qp_latents_spliced": False,
            "full_high_quality_video_area_prorated": False,
            "all_on_disk_container_bytes_charged": True,
            "generate_input": "fresh decoded Base RGB only",
            "generate_outside_codec_reference_loop": True,
        },
        "codec_format": {
            "released_spatial_qp_supported": False,
            "released_scalable_layer_supported": False,
            "fallback": (
                "ordinary full-frame Base DCVC-UF stream plus independently "
                "coded enhancement tile streams and a two-bit action map"),
            "fallback_is_final_spatial_qp_contribution": False,
        },
        "configuration": {
            "base_qp": args.base_qp,
            "enhance_qp": args.enhance_qp,
            "tile_size": args.tile_size,
            "tile_grid": [height // args.tile_size, width // args.tile_size],
            "fixed_decoder_feather_pixels": args.feather_pixels,
            "enhance_budget_bytes": enhance_budget,
            "enhance_budget_ratio_of_base_stream": args.enhance_budget_ratio,
            "max_generate_tiles": args.max_generate_tiles,
            "decode_repeats": args.decode_repeats,
            "ordinary_qps": args.ordinary_qps,
            "seedvr2": seed_metadata,
        },
        "candidate_search": {
            "enhancement_candidate_count": len(payloads),
            "candidate_encode_seconds": candidate_encode_seconds,
            "candidate_streams_not_all_transmitted": True,
            "base_stream_reproduced_byte_exact": True,
            "base_reproduction_encode_seconds": reproduced_base_encode["seconds"],
        },
        "variants": variants,
        "ordinary_scalar_qp_baselines": baseline_variants,
        "comparisons": {
            "all_generate_vs_base": {
                "lpips_delta": (
                    all_generate_quality["lpips_alex"]
                    - base_quality["lpips_alex"]),
                "psnr_delta_db": (
                    all_generate_quality["psnr_db"]
                    - base_quality["psnr_db"]),
                "temporal_delta_mae_change": (
                    all_generate_quality["temporal_delta_mae"]
                    - base_quality["temporal_delta_mae"]),
            },
            "joint_vs_base": {
                "lpips_delta": (
                    joint_quality["lpips_alex"]
                    - base_quality["lpips_alex"]),
                "psnr_delta_db": (
                    joint_quality["psnr_db"]
                    - base_quality["psnr_db"]),
                "temporal_delta_mae_change": (
                    joint_quality["temporal_delta_mae"]
                    - base_quality["temporal_delta_mae"]),
            },
            "joint_vs_all_generate": {
                "lpips_delta": (
                    joint_quality["lpips_alex"]
                    - all_generate_quality["lpips_alex"]),
                "psnr_delta_db": (
                    joint_quality["psnr_db"]
                    - all_generate_quality["psnr_db"]),
                "temporal_delta_mae_change": (
                    joint_quality["temporal_delta_mae"]
                    - all_generate_quality["temporal_delta_mae"]),
            },
            "generate_contribution_on_fixed_joint_route": {
                "lpips_delta": (
                    joint_quality["lpips_alex"]
                    - joint_no_generate_quality["lpips_alex"]),
                "psnr_delta_db": (
                    joint_quality["psnr_db"]
                    - joint_no_generate_quality["psnr_db"]),
                "temporal_delta_mae_change": (
                    joint_quality["temporal_delta_mae"]
                    - joint_no_generate_quality["temporal_delta_mae"]),
                "byte_delta": 0,
            },
            "enhancement_contribution_on_fixed_joint_route": {
                "lpips_delta": (
                    joint_quality["lpips_alex"]
                    - joint_no_enhance_quality["lpips_alex"]),
                "psnr_delta_db": (
                    joint_quality["psnr_db"]
                    - joint_no_enhance_quality["psnr_db"]),
                "temporal_delta_mae_change": (
                    joint_quality["temporal_delta_mae"]
                    - joint_no_enhance_quality["temporal_delta_mae"]),
                "byte_delta": (
                    joint_bytes
                    - variants["joint-no-enhance-fixed-route"]["rate"][
                        "total_bytes"]),
            },
            "joint_vs_nearest_scalar_qp": {
                "scalar_variant": nearest_scalar_name,
                "byte_delta": joint_bytes - nearest_scalar["rate"]["total_bytes"],
                "lpips_delta": (
                    joint_quality["lpips_alex"]
                    - nearest_scalar["quality"]["lpips_alex"]),
                "psnr_delta_db": (
                    joint_quality["psnr_db"]
                    - nearest_scalar["quality"]["psnr_db"]),
                "temporal_delta_mae_change": (
                    joint_quality["temporal_delta_mae"]
                    - nearest_scalar["quality"]["temporal_delta_mae"]),
            },
        },
        "runtime_note": (
            "Full decode time is a measured sequential component sum: fresh "
            "container parse/codec decode/composite plus the actual SeedVR2 "
            "inference duration, with both models already loaded.  The replay "
            "duration is retained only for bit-exact container verification."),
        "limitations": [
            "The route is an encoder-side Oracle, not a learned controller.",
            "Independent enhancement tiles repeat intra and boundary information.",
            "The fixed feather reduces display seams but cannot recover codec context lost at tile boundaries.",
            "SeedVR2 is still evaluated full-frame, so this probe does not claim selective-compute speedup.",
            "Local LPIPS on 128-pixel tiles is a response proxy, not the final importance metric.",
        ],
        "visuals": {
            "gt_generate_stitched": str(comparison_path),
            "joint_action_map": str(action_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_csv(args.output_dir / "per_region.csv", region_rows)
    write_csv(args.output_dir / "per_variant.csv", [
        {
            "variant": name,
            "file_bytes": record["rate"]["total_bytes"],
            "psnr_db": record["quality"]["psnr_db"],
            "lpips_alex": record["quality"]["lpips_alex"],
            "temporal_delta_mae": record["quality"]["temporal_delta_mae"],
            "base_tiles": record["action_counts"]["Base"],
            "generate_tiles": record["action_counts"]["Generate"],
            "enhance_tiles": record["action_counts"]["Enhance"],
            "full_decode_seconds": record["runtime"][
                "full_decode_component_sum_seconds"],
            "peak_cuda_allocated_bytes": record["runtime"][
                "full_decode_peak_cuda_allocated_bytes"],
        }
        for name, record in variants.items()
    ])
    print(json.dumps({
        "summary": str(summary_path),
        "comparison_visual": str(comparison_path),
        "action_visual": str(action_path),
        "joint_actions": actions_by_name[
            "joint-three-path-oracle"].tolist(),
        "joint_action_counts": variants[
            "joint-three-path-oracle"]["action_counts"],
        "comparisons": summary["comparisons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
