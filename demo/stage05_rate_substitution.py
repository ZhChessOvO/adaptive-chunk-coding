#!/usr/bin/env python3
"""Stage 0.5: no-training rate substitution experiment.

The pixel-space texture suppression used here is a proxy for future latent
token skipping. It is applied before DCVC-UF so that any rate saving is real.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


MAGIC = b"S05R"
HEADER_FMT = "<4sHHHHBBBBII"
GENERIC_PROMPT = "high quality natural video frame, realistic fine texture"
GENERIC_NEGATIVE_PROMPT = "text, watermark, deformation, artifacts, oversmoothed"


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def frame_paths(directory: Path, count: int = -1) -> list[Path]:
    paths = sorted(directory.glob("*.png"), key=natural_key)
    if count > 0:
        paths = paths[:count]
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {directory}")
    return paths


def load_frames(directory: Path, count: int = -1) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in frame_paths(directory, count)]


def save_frame(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), "RGB").save(path)


def block_slices(height: int, width: int, size: int):
    rows, cols = math.ceil(height / size), math.ceil(width / size)
    for row in range(rows):
        for col in range(cols):
            yield row, col, slice(row * size, min((row + 1) * size, height)), slice(col * size, min((col + 1) * size, width))


def expand_blocks(blocks: np.ndarray, height: int, width: int, size: int) -> np.ndarray:
    return np.repeat(np.repeat(blocks, size, axis=0), size, axis=1)[:height, :width]


def grass_pixels(rgb: np.ndarray, min_y_ratio: float) -> np.ndarray:
    height = rgb.shape[0]
    values = rgb.astype(np.float32)
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    y_ok = np.arange(height)[:, None] >= int(round(height * min_y_ratio))
    # Broad enough for sunlit/yellow-green grass, while excluding brown horse,
    # white rails and low-saturation background.
    green = (g > r * 1.025) & (g > b * 1.10) & ((g - np.minimum(r, b)) > 8)
    return green & y_ok


def build_temporal_grass_mask(
    frames: list[np.ndarray], block_size: int, min_y_ratio: float,
    min_green_fraction: float, temporal_quantile: float,
) -> np.ndarray:
    height, width = frames[0].shape[:2]
    rows, cols = math.ceil(height / block_size), math.ceil(width / block_size)
    fractions = np.zeros((len(frames), rows, cols), dtype=np.float32)
    for index, frame in enumerate(frames):
        green = grass_pixels(frame, min_y_ratio)
        for row, col, ys, xs in block_slices(height, width, block_size):
            fractions[index, row, col] = float(np.mean(green[ys, xs]))
    stable_fraction = np.quantile(fractions, temporal_quantile, axis=0)
    return stable_fraction >= min_green_fraction


def suppress_texture(frame: np.ndarray, alpha: np.ndarray, radius: float, downsample: int) -> np.ndarray:
    image = Image.fromarray(frame, "RGB")
    small = image.resize(
        (max(1, image.width // downsample), max(1, image.height // downsample)),
        Image.Resampling.BOX,
    )
    lowpass = small.resize(image.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius))
    low = np.asarray(lowpass, dtype=np.float32)
    result = frame.astype(np.float32) * (1.0 - alpha[..., None]) + low * alpha[..., None]
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(frame)
    color[..., 1] = 220
    out = frame.copy().astype(np.float32)
    out[mask] = out[mask] * 0.45 + color[mask] * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare(args: argparse.Namespace) -> None:
    frames = load_frames(args.source_dir, args.frame_count)
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("all source frames must have the same shape")
    height, width = frames[0].shape[:2]
    blocks = build_temporal_grass_mask(
        frames, args.mask_block_size, args.min_y_ratio,
        args.min_green_fraction, args.temporal_quantile,
    )
    binary = expand_blocks(blocks, height, width, args.mask_block_size)
    alpha_img = Image.fromarray(binary.astype(np.uint8) * 255, "L").filter(
        ImageFilter.GaussianBlur(args.feather_radius)
    )
    alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0

    input_dir = args.output_dir / "texture_suppressed_input" / args.sequence
    overlay_dir = args.output_dir / "mask_overlay"
    for index, frame in enumerate(frames, start=1):
        skeleton = suppress_texture(frame, alpha, args.blur_radius, args.downsample)
        save_frame(input_dir / f"im{index:05d}.png", skeleton)
        save_frame(overlay_dir / f"im{index:05d}.png", overlay_mask(frame, binary))
    Image.fromarray(binary.astype(np.uint8) * 255, "L").save(args.output_dir / "grass_mask.png")
    Image.fromarray(np.asarray(alpha_img), "L").save(args.output_dir / "grass_mask_feathered.png")
    np.save(args.output_dir / "grass_mask_blocks.npy", blocks)

    config = {
        "root_path": str((args.output_dir / "texture_suppressed_input").resolve()),
        "test_classes": {
            args.dataset: {
                "test": 1,
                "base_path": "",
                "src_type": "png",
                "sequences": {
                    args.sequence: {
                        "width": width, "height": height,
                        "frames": len(frames), "intra_period": -1,
                    }
                },
            }
        },
    }
    (args.output_dir / "texture_suppressed_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    metadata = {
        "frames": len(frames), "resolution": [width, height],
        "mask_block_size": args.mask_block_size,
        "grass_block_count": int(blocks.sum()),
        "total_block_count": int(blocks.size),
        "grass_block_ratio": float(blocks.mean()),
        "grass_pixel_ratio": float(binary.mean()),
        "min_y_ratio": args.min_y_ratio,
        "min_green_fraction": args.min_green_fraction,
        "temporal_quantile": args.temporal_quantile,
        "feather_radius": args.feather_radius,
        "blur_radius": args.blur_radius,
        "downsample": args.downsample,
    }
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def psnr(reference: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    delta = reference.astype(np.float64) - value.astype(np.float64)
    if mask is not None:
        if not mask.any():
            return None
        delta = delta[mask]
    error = float(np.mean(delta * delta))
    return None if error == 0 else 10.0 * math.log10(255.0 ** 2 / error)


def temporal_delta_mae(reference: list[np.ndarray], value: list[np.ndarray], mask: np.ndarray | None = None) -> float | None:
    if len(reference) < 2:
        return None
    errors = []
    for i in range(1, len(reference)):
        ref = reference[i].astype(np.int16) - reference[i - 1].astype(np.int16)
        val = value[i].astype(np.int16) - value[i - 1].astype(np.int16)
        err = np.abs(ref.astype(np.int32) - val.astype(np.int32))
        if mask is not None:
            err = err[mask]
        errors.append(float(np.mean(err)))
    return float(np.mean(errors))


def load_lpips(device: str):
    import lpips
    return lpips.LPIPS(net="alex").to(device).eval()


def lpips_mean(model, reference: list[np.ndarray], value: list[np.ndarray], device: str) -> float:
    import torch
    scores = []
    with torch.inference_mode():
        for ref, val in zip(reference, value):
            a = torch.from_numpy(ref.copy()).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            b = torch.from_numpy(val.copy()).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            scores.append(float(model(a, b).item()))
    return float(np.mean(scores))


def load_generator(args: argparse.Namespace):
    import torch
    from diffusers import AutoPipelineForInpainting
    kwargs = {"torch_dtype": torch.float16, "local_files_only": True, "use_safetensors": True}
    if args.variant:
        kwargs["variant"] = args.variant
    pipe = AutoPipelineForInpainting.from_pretrained(args.model, **kwargs).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_frames(args: argparse.Namespace, pipe, bases: list[np.ndarray], mask: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    import torch
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    # A soft composite hides block boundaries, while the binary mask remains
    # the transmitted route description and the evaluation region.
    blend = np.asarray(mask_image.filter(ImageFilter.GaussianBlur(args.composite_feather)), dtype=np.float32) / 255.0
    outputs, times = [], []
    for index, base in enumerate(bases):
        start = time.perf_counter()
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        image = pipe(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            image=Image.fromarray(base, "RGB"), mask_image=mask_image,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            generator=generator, height=base.shape[0], width=base.shape[1],
        ).images[0].convert("RGB")
        times.append(time.perf_counter() - start)
        generated = np.asarray(image, dtype=np.float32)
        mixed = base.astype(np.float32) * (1.0 - blend[..., None]) + generated * blend[..., None]
        outputs.append(np.clip(np.round(mixed), 0, 255).astype(np.uint8))
        print(f"[{index + 1:04d}/{len(bases):04d}] SDXL {times[-1]:.3f}s")
    return outputs, times


def mask_payload(block_mask: np.ndarray) -> bytes:
    packed = np.packbits(block_mask.reshape(-1).astype(np.uint8), bitorder="little")
    return zlib.compress(packed.tobytes(), level=9)


def make_residual_candidates(
    originals: list[np.ndarray], bases: list[np.ndarray], gen_mask: np.ndarray,
    block_size: int, quant_step: int,
) -> list[tuple[float, bytes, int, int, np.ndarray]]:
    height, width = originals[0].shape[:2]
    candidates = []
    cols = math.ceil(width / block_size)
    for frame_idx, (original, base) in enumerate(zip(originals, bases)):
        delta = original.astype(np.int16) - base.astype(np.int16)
        for row, col, ys, xs in block_slices(height, width, block_size):
            # Residual is reserved for protected content, not generated grass.
            if np.any(gen_mask[ys, xs]):
                continue
            q = np.clip(np.round(delta[ys, xs] / quant_step), -127, 127).astype(np.int8)
            reconstructed = np.clip(base[ys, xs].astype(np.int16) + q.astype(np.int16) * quant_step, 0, 255)
            old_error = np.square(original[ys, xs].astype(np.float32) - base[ys, xs].astype(np.float32)).sum()
            new_error = np.square(original[ys, xs].astype(np.float32) - reconstructed.astype(np.float32)).sum()
            benefit = float(old_error - new_error)
            if benefit <= 0 or not np.any(q):
                continue
            block_index = row * cols + col
            record = struct.pack("<HH", frame_idx, block_index) + q.tobytes()
            estimated = len(zlib.compress(record, level=6)) + 1
            candidates.append((benefit / estimated, record, frame_idx, block_index, q))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def build_stream(
    block_mask: np.ndarray, frame_count: int, height: int, width: int,
    mask_block_size: int, residual_block_size: int, quant_step: int,
    records: list[bytes],
) -> bytes:
    mask_data = mask_payload(block_mask)
    residual_data = zlib.compress(b"".join(records), level=9)
    header = struct.pack(
        HEADER_FMT, MAGIC, frame_count, height, width, len(records),
        mask_block_size, residual_block_size, quant_step, 0,
        len(mask_data), len(residual_data),
    )
    return header + mask_data + residual_data


def select_budgeted_stream(
    candidates, block_mask, frame_count, height, width,
    mask_block_size, residual_block_size, quant_step, budget_bytes,
) -> tuple[bytes, int]:
    selected: list[bytes] = []
    best = build_stream(block_mask, frame_count, height, width, mask_block_size, residual_block_size, quant_step, selected)
    for candidate in candidates:
        trial = build_stream(
            block_mask, frame_count, height, width, mask_block_size,
            residual_block_size, quant_step, selected + [candidate[1]],
        )
        if len(trial) <= budget_bytes:
            selected.append(candidate[1])
            best = trial
    return best, len(selected)


def decode_stream(payload: bytes, bases: list[np.ndarray], generated: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, int]:
    header_size = struct.calcsize(HEADER_FMT)
    (magic, frame_count, height, width, record_count, mask_bs, residual_bs,
     quant_step, _, mask_len, residual_len) = struct.unpack(HEADER_FMT, payload[:header_size])
    if magic != MAGIC or frame_count != len(bases):
        raise ValueError("invalid Stage 0.5 side stream")
    cursor = header_size
    packed = np.frombuffer(zlib.decompress(payload[cursor:cursor + mask_len]), dtype=np.uint8)
    cursor += mask_len
    rows, cols = math.ceil(height / mask_bs), math.ceil(width / mask_bs)
    blocks = np.unpackbits(packed, bitorder="little")[: rows * cols].reshape(rows, cols).astype(bool)
    mask = expand_blocks(blocks, height, width, mask_bs)
    outputs = []
    for base, gen in zip(bases, generated):
        output = base.copy()
        output[mask] = gen[mask]
        outputs.append(output)
    residual_raw = zlib.decompress(payload[cursor:cursor + residual_len])
    value_count = residual_bs * residual_bs * 3
    record_size = 4 + value_count
    if len(residual_raw) != record_count * record_size:
        raise ValueError("corrupt residual records")
    residual_cols = math.ceil(width / residual_bs)
    for offset in range(0, len(residual_raw), record_size):
        frame_idx, block_index = struct.unpack("<HH", residual_raw[offset:offset + 4])
        q = np.frombuffer(residual_raw[offset + 4:offset + record_size], dtype=np.int8).reshape(residual_bs, residual_bs, 3)
        row, col = divmod(block_index, residual_cols)
        y0, x0 = row * residual_bs, col * residual_bs
        y1, x1 = min(y0 + residual_bs, height), min(x0 + residual_bs, width)
        q = q[:y1 - y0, :x1 - x0]
        corrected = bases[frame_idx][y0:y1, x0:x1].astype(np.int16) + q.astype(np.int16) * quant_step
        outputs[frame_idx][y0:y1, x0:x1] = np.clip(corrected, 0, 255).astype(np.uint8)
    return outputs, mask, record_count


def label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 26), "white")
    canvas.paste(image, (0, 26))
    ImageDraw.Draw(canvas).text((7, 6), text, fill="black", font=ImageFont.load_default())
    return canvas


def panel(arrays: list[np.ndarray], labels: list[str]) -> Image.Image:
    items = [label(Image.fromarray(x, "RGB"), name) for x, name in zip(arrays, labels)]
    canvas = Image.new("RGB", (sum(x.width for x in items), items[0].height), "white")
    x = 0
    for item in items:
        canvas.paste(item, (x, 0)); x += item.width
    return canvas


def top_residual_routes(
    originals: list[np.ndarray], bases: list[np.ndarray], gen_blocks: np.ndarray,
    block_size: int, residual_ratio: float,
) -> np.ndarray:
    """Build per-frame qualitative routes without a rate-savings constraint."""
    routes = []
    residual_count = int(round(gen_blocks.size * residual_ratio))
    for original, base in zip(originals, bases):
        route = np.zeros_like(gen_blocks, dtype=np.uint8)
        route[gen_blocks] = 1
        scores = np.full(gen_blocks.shape, -np.inf, dtype=np.float64)
        delta = original.astype(np.float32) - base.astype(np.float32)
        for row, col, ys, xs in block_slices(original.shape[0], original.shape[1], block_size):
            if not gen_blocks[row, col]:
                scores[row, col] = float(np.mean(np.square(delta[ys, xs])))
        available = np.flatnonzero(np.isfinite(scores.reshape(-1)))
        count = min(residual_count, available.size)
        if count:
            chosen = available[np.argsort(scores.reshape(-1)[available])[-count:]]
            route.reshape(-1)[chosen] = 2
        routes.append(route)
    return np.stack(routes)


def qualitative(args: argparse.Namespace) -> None:
    # Reuse the Stage-0 stream implementation so the residual branch is an
    # actual lossless side stream and the visual layout remains identical.
    from stage0_selective_demo import (
        ROUTE_BASE, ROUTE_GENERATE, ROUTE_RESIDUAL,
        decode_side_stream, make_panel, route_overlay, write_side_stream,
    )

    originals = load_frames(args.original_dir, args.frame_count)
    bases = load_frames(args.base_dir, args.frame_count)
    if len(originals) != len(bases):
        raise ValueError("original/base frame counts do not match")
    height, width = originals[0].shape[:2]
    blocks = np.load(args.mask_blocks).astype(bool)
    gen_mask = expand_blocks(blocks, height, width, args.mask_block_size)
    pipe = load_generator(args)
    generated, generator_times = generate_frames(args, pipe, bases, gen_mask)
    del pipe

    routes = top_residual_routes(
        originals, bases, blocks, args.mask_block_size, args.residual_ratio
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    side_path = output_dir / "qualitative_side_stream.bin"
    stream_stats = write_side_stream(side_path, routes, originals, bases, args.mask_block_size)
    finals, decoded_routes = decode_side_stream(side_path, bases, generated)
    if not np.array_equal(routes, decoded_routes):
        raise AssertionError("qualitative route round trip failed")

    panels = []
    frame_rows = []
    for index, (original, base, gen, final, route) in enumerate(
        zip(originals, bases, generated, finals, routes), start=1
    ):
        filename = f"im{index:05d}.png"
        overlay = route_overlay(base, route, args.mask_block_size)
        save_frame(output_dir / "routes" / filename, overlay)
        view = make_panel(original, base, gen, final, overlay)
        panel_path = output_dir / "panels" / filename
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        view.save(panel_path)
        preview = view.copy()
        preview.thumbnail((1280, 300), Image.Resampling.LANCZOS)
        panels.append(preview)
        frame_rows.append({
            "frame": index,
            "base_psnr": psnr(original, base),
            "generated_psnr": psnr(original, gen),
            "final_psnr": psnr(original, final),
            "base_gen_region_psnr": psnr(original, base, gen_mask),
            "generated_gen_region_psnr": psnr(original, gen, gen_mask),
            "generator_seconds": generator_times[index - 1],
        })
    panels[0].save(
        output_dir / "preview.gif", save_all=True, append_images=panels[1:],
        duration=250, loop=0, optimize=False,
    )
    contact = Image.new("RGB", (panels[0].width, sum(x.height for x in panels[:4])), "white")
    y = 0
    for item in panels[:4]:
        contact.paste(item, (0, y)); y += item.height
    contact.save(output_dir / "contact_sheet.png")
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0])); writer.writeheader(); writer.writerows(frame_rows)

    pixel_count = len(originals) * height * width
    base_bytes = args.base_bitstream.stat().st_size
    route_ratios = {
        "base": float(np.mean(routes == ROUTE_BASE)),
        "generate": float(np.mean(routes == ROUTE_GENERATE)),
        "residual": float(np.mean(routes == ROUTE_RESIDUAL)),
    }
    quality = {
        "base_psnr_full": float(np.mean([psnr(a, b) for a, b in zip(originals, bases)])),
        "generated_psnr_full": float(np.mean([psnr(a, b) for a, b in zip(originals, generated)])),
        "final_psnr_full": float(np.mean([psnr(a, b) for a, b in zip(originals, finals)])),
        "base_psnr_gen_region": float(np.mean([psnr(a, b, gen_mask) for a, b in zip(originals, bases)])),
        "generated_psnr_gen_region": float(np.mean([psnr(a, b, gen_mask) for a, b in zip(originals, generated)])),
        "base_temporal_delta_mae": temporal_delta_mae(originals, bases),
        "generated_temporal_delta_mae": temporal_delta_mae(originals, generated),
        "final_temporal_delta_mae": temporal_delta_mae(originals, finals),
    }
    if args.lpips:
        model = load_lpips(args.device)
        quality["lpips_alex"] = {
            "base": lpips_mean(model, originals, bases, args.device),
            "generated": lpips_mean(model, originals, generated, args.device),
            "final": lpips_mean(model, originals, finals, args.device),
        }
    summary = {
        "stage": "Stage 0.5 low-QP qualitative gen+res routing",
        "purpose": "Qualitative routing visualization; residual is intentionally not constrained by generated-region rate savings.",
        "frames": len(originals), "resolution": [width, height],
        "route": {**route_ratios, "block_size": args.mask_block_size, "shared_generate_mask": True},
        "rate": {
            "base_bytes": base_bytes, "base_bpp": base_bytes * 8 / pixel_count,
            "side_stream_bytes": stream_stats.total_bytes,
            "side_stream_bpp": stream_stats.total_bytes * 8 / pixel_count,
            "route_mask_bytes": stream_stats.mask_bytes,
            "lossless_residual_bytes": stream_stats.residual_bytes,
            "total_bpp": (base_bytes + stream_stats.total_bytes) * 8 / pixel_count,
        },
        "quality": quality,
        "runtime": {
            "generator_total_seconds": float(sum(generator_times)),
            "generator_seconds_per_frame": float(np.mean(generator_times)),
        },
        "route_legend": {"base_blue": [74, 103, 169], "generate_green": [89, 161, 79], "residual_red": [225, 87, 89]},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    originals = load_frames(args.original_dir, args.frame_count)
    skeletons = load_frames(args.skeleton_dir, args.frame_count)
    base_a = load_frames(args.original_base_dir, args.frame_count)
    base_b = load_frames(args.skeleton_base_dir, args.frame_count)
    counts = {len(originals), len(skeletons), len(base_a), len(base_b)}
    if len(counts) != 1:
        raise ValueError("input frame counts do not match")
    height, width = originals[0].shape[:2]
    blocks = np.load(args.mask_blocks).astype(bool)
    gen_mask = expand_blocks(blocks, height, width, args.mask_block_size)
    protected_mask = ~gen_mask
    rate_a = args.original_bitstream.stat().st_size
    rate_b = args.skeleton_bitstream.stat().st_size

    pipe = load_generator(args)
    generated, generator_times = generate_frames(args, pipe, base_b, gen_mask)
    del pipe

    # C carries the shared route mask but no residual records.
    stream_c = build_stream(blocks, len(originals), height, width, args.mask_block_size, args.residual_block_size, args.residual_quant_step, [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "route_only.bin").write_bytes(stream_c)
    final_c, decoded_mask, _ = decode_stream(stream_c, base_b, generated)
    if not np.array_equal(decoded_mask, gen_mask):
        raise AssertionError("generation mask round trip failed")

    # D may spend at most the bytes saved relative to A, including route mask.
    side_budget = max(0, rate_a - rate_b)
    candidates = make_residual_candidates(
        originals, base_b, gen_mask, args.residual_block_size, args.residual_quant_step
    )
    stream_d, selected_count = select_budgeted_stream(
        candidates, blocks, len(originals), height, width,
        args.mask_block_size, args.residual_block_size,
        args.residual_quant_step, side_budget,
    )
    (args.output_dir / "budget_matched_side_stream.bin").write_bytes(stream_d)
    final_d, decoded_mask_d, decoded_count = decode_stream(stream_d, base_b, generated)
    if not np.array_equal(decoded_mask_d, gen_mask) or selected_count != decoded_count:
        raise AssertionError("budgeted side-stream round trip failed")

    variants = {
        "A_original_dcvc": base_a,
        "B_suppressed_dcvc": base_b,
        "C_suppressed_plus_gen": final_c,
        "D_budget_matched_gen_res": final_d,
    }
    lpips_model = load_lpips(args.device) if args.lpips else None
    metrics = {}
    for name, frames in variants.items():
        metrics[name] = {
            "psnr_full": float(np.mean([psnr(a, b) for a, b in zip(originals, frames)])),
            "psnr_gen_region": float(np.mean([psnr(a, b, gen_mask) for a, b in zip(originals, frames)])),
            "psnr_protected_region": float(np.mean([psnr(a, b, protected_mask) for a, b in zip(originals, frames)])),
            "temporal_delta_mae_full": temporal_delta_mae(originals, frames),
            "temporal_delta_mae_gen_region": temporal_delta_mae(originals, frames, gen_mask),
            "temporal_delta_mae_protected_region": temporal_delta_mae(originals, frames, protected_mask),
        }
        if lpips_model is not None:
            metrics[name]["lpips_alex_full"] = lpips_mean(lpips_model, originals, frames, args.device)

    pixel_count = len(originals) * height * width
    rates = {
        "A_original_dcvc": {"bytes": rate_a, "bpp": rate_a * 8 / pixel_count},
        "B_suppressed_dcvc": {"bytes": rate_b, "bpp": rate_b * 8 / pixel_count},
        "C_suppressed_plus_gen": {"bytes": rate_b + len(stream_c), "bpp": (rate_b + len(stream_c)) * 8 / pixel_count},
        "D_budget_matched_gen_res": {"bytes": rate_b + len(stream_d), "bpp": (rate_b + len(stream_d)) * 8 / pixel_count},
    }
    summary = {
        "stage": "Stage 0.5 no-training rate substitution",
        "proxy_warning": "Pixel-space temporal low-pass is a proxy for future latent-token skipping.",
        "frames": len(originals), "resolution": [width, height],
        "mask": {
            "block_size": args.mask_block_size,
            "block_count": int(blocks.sum()), "total_blocks": int(blocks.size),
            "block_ratio": float(blocks.mean()), "pixel_ratio": float(gen_mask.mean()),
            "shared_across_frames": True,
        },
        "rate": rates,
        "rate_substitution": {
            "dcvc_bytes_saved_A_minus_B": rate_a - rate_b,
            "dcvc_rate_reduction_percent": (rate_a - rate_b) / rate_a * 100,
            "route_only_stream_bytes": len(stream_c),
            "budget_matched_stream_bytes": len(stream_d),
            "budget_bytes_A_minus_B": side_budget,
            "D_within_A_budget": rate_b + len(stream_d) <= rate_a,
        },
        "residual": {
            "block_size": args.residual_block_size,
            "quant_step": args.residual_quant_step,
            "candidate_blocks": len(candidates), "selected_blocks": selected_count,
        },
        "quality": metrics,
        "runtime": {
            "generator_total_seconds": float(sum(generator_times)),
            "generator_seconds_per_frame": float(np.mean(generator_times)),
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for i in range(len(originals)):
        row = {"frame": i + 1, "generator_seconds": generator_times[i]}
        for name, frames in variants.items():
            row[f"{name}_psnr"] = psnr(originals[i], frames[i])
        rows.append(row)
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    panels = []
    for i in range(len(originals)):
        filename = f"im{i + 1:05d}.png"
        arrays = [originals[i], base_a[i], skeletons[i], base_b[i], final_c[i], final_d[i]]
        labels = ["original", "A: original DCVC", "suppressed input", "B: suppressed DCVC", "C: B + grass gen", "D: budget gen + res"]
        for folder, array in zip(("original", "A_base", "suppressed_input", "B_base", "C_generated", "D_budget_matched"), arrays):
            save_frame(args.output_dir / folder / filename, array)
        view = panel(arrays, labels)
        (args.output_dir / "panels").mkdir(parents=True, exist_ok=True)
        view.save(args.output_dir / "panels" / filename)
        preview = view.copy(); preview.thumbnail((1536, 300), Image.Resampling.LANCZOS); panels.append(preview)
    panels[0].save(args.output_dir / "preview.gif", save_all=True, append_images=panels[1:], duration=250, loop=0)
    contact = Image.new("RGB", (panels[0].width, sum(x.height for x in panels[:4])), "white")
    y = 0
    for item in panels[:4]: contact.paste(item, (0, y)); y += item.height
    contact.save(args.output_dir / "contact_sheet.png")
    print(json.dumps({"rate": rates, "rate_substitution": summary["rate_substitution"], "quality": metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--frame-count", type=int, default=17)
    prep.add_argument("--dataset", default="JockeyStage05")
    prep.add_argument("--sequence", default="jockey_stage05")
    prep.add_argument("--mask-block-size", type=int, default=32)
    prep.add_argument("--min-y-ratio", type=float, default=0.50)
    prep.add_argument("--min-green-fraction", type=float, default=0.72)
    prep.add_argument("--temporal-quantile", type=float, default=0.10)
    prep.add_argument("--feather-radius", type=float, default=5.0)
    prep.add_argument("--blur-radius", type=float, default=2.0)
    prep.add_argument("--downsample", type=int, default=16)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--original-dir", type=Path, required=True)
    ev.add_argument("--skeleton-dir", type=Path, required=True)
    ev.add_argument("--original-base-dir", type=Path, required=True)
    ev.add_argument("--skeleton-base-dir", type=Path, required=True)
    ev.add_argument("--original-bitstream", type=Path, required=True)
    ev.add_argument("--skeleton-bitstream", type=Path, required=True)
    ev.add_argument("--mask-blocks", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)
    ev.add_argument("--frame-count", type=int, default=17)
    ev.add_argument("--mask-block-size", type=int, default=32)
    ev.add_argument("--residual-block-size", type=int, default=8)
    ev.add_argument("--residual-quant-step", type=int, default=16)
    ev.add_argument("--model", type=str, default="checkpoints/sdxl-inpainting-0.1")
    ev.add_argument("--variant", type=str, default="fp16")
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--prompt", default=GENERIC_PROMPT)
    ev.add_argument("--negative-prompt", default=GENERIC_NEGATIVE_PROMPT)
    ev.add_argument("--steps", type=int, default=20)
    ev.add_argument("--guidance-scale", type=float, default=5.0)
    ev.add_argument("--seed", type=int, default=20260903)
    ev.add_argument("--composite-feather", type=float, default=5.0)
    ev.add_argument("--lpips", action="store_true")

    qual = sub.add_parser("qualitative")
    qual.add_argument("--original-dir", type=Path, required=True)
    qual.add_argument("--base-dir", type=Path, required=True)
    qual.add_argument("--base-bitstream", type=Path, required=True)
    qual.add_argument("--mask-blocks", type=Path, required=True)
    qual.add_argument("--output-dir", type=Path, required=True)
    qual.add_argument("--frame-count", type=int, default=17)
    qual.add_argument("--mask-block-size", type=int, default=32)
    qual.add_argument("--residual-ratio", type=float, default=0.10)
    qual.add_argument("--model", type=str, default="checkpoints/sdxl-inpainting-0.1")
    qual.add_argument("--variant", type=str, default="fp16")
    qual.add_argument("--device", default="cuda")
    qual.add_argument("--prompt", default=GENERIC_PROMPT)
    qual.add_argument("--negative-prompt", default=GENERIC_NEGATIVE_PROMPT)
    qual.add_argument("--steps", type=int, default=20)
    qual.add_argument("--guidance-scale", type=float, default=5.0)
    qual.add_argument("--seed", type=int, default=20260903)
    qual.add_argument("--composite-feather", type=float, default=5.0)
    qual.add_argument("--lpips", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    elif arguments.command == "evaluate":
        evaluate(arguments)
    else:
        qualitative(arguments)
