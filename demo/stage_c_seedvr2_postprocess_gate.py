#!/usr/bin/env python3
"""No-training perceptual controls for SeedVR2 restoration output.

The decoder-side candidates use only the decoded Base frames and SeedVR2
frames.  Ground truth is loaded solely for diagnostic evaluation and is never
an input to blending or wavelet reconstruction.
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant, save_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--seedvr2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0.1, 0.25, 0.5, 0.75))
    parser.add_argument("--visual-frame", type=int, default=3)
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.alphas):
        parser.error("--alphas must be inside (0, 1)")
    return args


def as_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    data = np.stack(frames).copy()
    return torch.from_numpy(data).permute(0, 3, 1, 2).float().div_(255.0)


def as_frames(tensor: torch.Tensor) -> list[np.ndarray]:
    data = tensor.clamp(0, 1).mul(255).round().byte()
    data = data.permute(0, 2, 3, 1).cpu().numpy()
    return [frame.copy() for frame in data]


def wavelet_blur(frames: torch.Tensor, radius: int) -> torch.Tensor:
    values = torch.tensor(
        [[0.0625, 0.125, 0.0625],
         [0.125, 0.25, 0.125],
         [0.0625, 0.125, 0.0625]],
        dtype=frames.dtype,
        device=frames.device,
    )
    kernel = values[None, None].repeat(frames.shape[1], 1, 1, 1)
    padded = F.pad(frames, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(
        padded, kernel, groups=frames.shape[1], dilation=radius)


def wavelet_decomposition(
    frames: torch.Tensor, levels: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    high = torch.zeros_like(frames)
    current = frames
    for level in range(levels):
        low = wavelet_blur(current, 2 ** level)
        high = high + current - low
        current = low
    return high, current


def wavelet_reconstruction(
    generated: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    generated_high, _ = wavelet_decomposition(generated)
    _, base_low = wavelet_decomposition(base)
    return generated_high + base_low


def blend(
    base: torch.Tensor, restoration: torch.Tensor, alpha: float
) -> torch.Tensor:
    return base + alpha * (restoration - base)


def make_panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(frame)
    banner = 68
    panel = Image.new("RGB", (image.width, image.height + banner), "white")
    panel.paste(image, (0, banner))
    draw = ImageDraw.Draw(panel)
    title_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    metric_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((8, 7), title, fill="black", font=title_font)
    draw.text((8, 38), subtitle, fill="black", font=metric_font)
    return panel


def make_visual(
    path: Path,
    frame_index: int,
    reference: list[np.ndarray],
    variants: dict[str, dict],
    names: list[str],
) -> None:
    panels = [make_panel(reference[frame_index], "GT", "source RGB")]
    for name in names:
        quality = variants[name]["quality"]
        subtitle = (
            f"PSNR {quality['psnr_db']:.3f} | "
            f"LPIPS {quality['lpips_alex']:.4f} | "
            f"T-MAE {quality['temporal_delta_mae']:.3f}"
        )
        panels.append(make_panel(
            variants[name]["frames"][frame_index], name, subtitle))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (columns * width, rows * height), (230, 230, 230))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    args = parse_args()
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    seed_frames = load_pngs(args.seedvr2_dir)
    count = len(seed_frames)
    base_frames = load_pngs(
        args.gate_summary.parent / "frames" / f"qp{args.qp}-base")[:count]
    basic_frames = load_pngs(
        args.gate_summary.parent / "frames" / f"qp{args.qp}-basicvsrpp")[:count]
    reference = load_source(gate, count)
    if not (len(base_frames) == len(basic_frames) == len(reference) == count):
        raise ValueError("frame counts differ")

    base = as_tensor(base_frames)
    seed = as_tensor(seed_frames)
    started = time.perf_counter()
    wavelet = wavelet_reconstruction(seed, base)
    wavelet_seconds = time.perf_counter() - started

    tensors: dict[str, torch.Tensor] = {
        "Base": base,
        "BasicVSR++": as_tensor(basic_frames),
        "SeedVR2 raw": seed,
        "SeedVR2 wavelet": wavelet,
    }
    for alpha in args.alphas:
        tensors[f"raw alpha={alpha:g}"] = blend(base, seed, alpha)
        tensors[f"wavelet alpha={alpha:g}"] = blend(base, wavelet, alpha)

    lpips = LPIPSAlex(True)
    variants = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensors.items():
        frames = as_frames(tensor)
        quality = evaluate_variant(reference, frames, lpips)
        variants[name] = {"frames": frames, "quality": quality}
        safe_name = name.lower().replace(" ", "-").replace("=", "")
        save_frames(args.output_dir / "frames" / safe_name, frames)

    base_quality = variants["Base"]["quality"]
    rows = []
    for name, record in variants.items():
        quality = record["quality"]
        rows.append({
            "variant": name,
            **quality,
            "psnr_delta_db": quality["psnr_db"] - base_quality["psnr_db"],
            "lpips_delta": quality["lpips_alex"] - base_quality["lpips_alex"],
            "temporal_delta_mae_change": (
                quality["temporal_delta_mae"]
                - base_quality["temporal_delta_mae"]),
        })

    perceptual_candidates = [
        row for row in rows
        if row["variant"] not in ("Base", "BasicVSR++")
        and row["lpips_delta"] < 0
        and row["temporal_delta_mae_change"] <= 0.5
    ]
    perceptual_candidates.sort(key=lambda row: row["lpips_alex"])
    perceptual_and_temporal = [
        row for row in perceptual_candidates
        if row["temporal_delta_mae_change"] <= 0
    ]
    perceptual_and_temporal.sort(key=lambda row: row["lpips_alex"])
    joint_name = (
        perceptual_and_temporal[0]["variant"]
        if perceptual_and_temporal else None
    )
    perceptual_name = (
        perceptual_candidates[0]["variant"]
        if perceptual_candidates else None
    )
    recommended = joint_name or perceptual_name
    result = {
        "experiment": "E20 SeedVR2 no-training fidelity-control gate",
        "frames": count,
        "qp": args.qp,
        "source_role": gate["source_role"],
        "decoder_inputs": ["decoded Base RGB", "SeedVR2 restored RGB"],
        "ground_truth_used_by_postprocess": False,
        "ground_truth_used_for_diagnostic_selection": True,
        "additional_stream_bytes": 0,
        "wavelet_levels": 5,
        "wavelet_postprocess_seconds_cpu": wavelet_seconds,
        "perceptual_gate": {
            "primary_metric": "LPIPS (lower is better)",
            "psnr_role": "report-only diagnostic; not a selection guard",
            "maximum_temporal_delta_mae_change": 0.5,
            "best_lpips_with_nonworse_temporal_variant": joint_name,
            "best_lpips_within_temporal_guard_variant": perceptual_name,
            "recommended_variant_on_this_clip": recommended,
            "important_region_fidelity": (
                "not applicable to this uniform restoration gate; required "
                "after the spatial action map is introduced"
            ),
        },
        "variants": {row["variant"]: row for row in rows},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_variant.csv").open(
        "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frame_index = min(max(args.visual_frame, 1), count) - 1
    visual_names = ["Base", "BasicVSR++", "SeedVR2 raw", "SeedVR2 wavelet"]
    for name in (joint_name, perceptual_name):
        if name is not None and name not in visual_names:
            visual_names.append(name)
    visual_path = args.output_dir / "visuals" / f"frame_{frame_index + 1:05d}.png"
    make_visual(visual_path, frame_index, reference, variants, visual_names)
    print(json.dumps({
        "summary": str(args.output_dir / "summary.json"),
        "visual": str(visual_path),
        "recommended": recommended,
        "perceptual_candidates": perceptual_candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
