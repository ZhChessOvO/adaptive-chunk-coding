#!/usr/bin/env python3
"""Stage-0 demo for selective generation and selective residual coding.

This script intentionally does not train anything.  It consumes original RGB
PNG frames and the decoded PNG frames produced by DCVC-UF, calls a frozen
inpainting/refinement backend only on a subset of blocks, and uses an oracle
rate-distortion-compute decision to route every block through one of:

    base-only / generate / residual

The residual branch is not just painted from the original for visualization:
an actual zlib-compressed side stream is written and decoded again before the
final frames are saved.  It is still a Stage-0 proxy, not the learned entropy
model intended for the final method.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROUTE_BASE = 0
ROUTE_GENERATE = 1
ROUTE_RESIDUAL = 2
ROUTE_NAMES = ("base", "generate", "residual")
ROUTE_COLORS = np.asarray(
    ((74, 103, 169), (89, 161, 79), (225, 87, 89)), dtype=np.uint8
)
SIDE_STREAM_MAGIC = b"S0R1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-training selective generation + residual visualization demo."
    )
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-bitstream", type=Path, default=None)
    parser.add_argument("--frame-count", type=int, default=-1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--candidate-ratio",
        type=float,
        default=0.30,
        help="Fraction of highest base-error blocks offered to the generator.",
    )
    parser.add_argument(
        "--max-generate-ratio", type=float, default=0.20,
        help="Hard upper bound on blocks finally routed to generation."
    )
    parser.add_argument(
        "--min-generate-ratio", type=float, default=0.0,
        help="Optional qualitative-demo floor; force the best generator candidates even when RGB-MSE does not improve."
    )
    parser.add_argument(
        "--max-residual-ratio", type=float, default=0.10,
        help="Hard upper bound on blocks finally routed to the residual stream."
    )
    parser.add_argument(
        "--lambda-rate", type=float, default=32.0,
        help="Rate weight in oracle residual cost: lambda * proxy bits/pixel."
    )
    parser.add_argument(
        "--gamma-compute", type=float, default=0.0,
        help="Per-generated-block compute penalty in RGB-MSE units."
    )
    parser.add_argument(
        "--backend", choices=("unsharp", "diffusers"), default="unsharp",
        help="unsharp is a dependency-free pipeline smoke test; diffusers is the real pretrained path."
    )
    parser.add_argument(
        "--model", type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Local directory or Hugging Face model id for the diffusers backend."
    )
    parser.add_argument(
        "--variant", type=str, default="fp16",
        help="Diffusers weight variant; use an empty string for the repository default."
    )
    parser.add_argument("--prompt", type=str, default="high quality natural video frame, realistic fine texture")
    parser.add_argument("--negative-prompt", type=str, default="text, watermark, deformation, artifacts, oversmoothed")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--allow-download", action="store_true",
        help="Allow diffusers to access the network. By default only local model files are accepted."
    )
    parser.add_argument("--lpips", action="store_true", help="Also evaluate LPIPS (optional package/weights).")
    parser.add_argument("--gif-fps", type=int, default=4)
    args = parser.parse_args()

    for name in ("candidate_ratio", "min_generate_ratio", "max_generate_ratio", "max_residual_ratio"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.min_generate_ratio > args.max_generate_ratio:
        parser.error("--min-generate-ratio cannot exceed --max-generate-ratio")
    if args.min_generate_ratio > args.candidate_ratio:
        parser.error("--min-generate-ratio cannot exceed --candidate-ratio")
    if args.frame_count == 0 or args.frame_count < -1:
        parser.error("--frame-count must be -1 or positive")
    return args


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def find_frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"frame directory does not exist: {directory}")
    frames = sorted(directory.glob("*.png"), key=natural_key)
    if not frames:
        raise FileNotFoundError(f"no PNG frames found in: {directory}")
    return frames


def paired_frames(original_dir: Path, base_dir: Path, frame_count: int) -> list[tuple[Path, Path]]:
    originals = find_frames(original_dir)
    bases = find_frames(base_dir)
    if frame_count > 0:
        originals = originals[:frame_count]
        bases = bases[:frame_count]
    if len(originals) != len(bases):
        raise ValueError(
            f"frame count mismatch: original={len(originals)}, base={len(bases)}. "
            "Use --frame-count if the base run decoded only a prefix."
        )
    return list(zip(originals, bases))


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def grid_shape(height: int, width: int, block_size: int) -> tuple[int, int]:
    return math.ceil(height / block_size), math.ceil(width / block_size)


def block_slices(height: int, width: int, block_size: int):
    rows, cols = grid_shape(height, width, block_size)
    for row in range(rows):
        y0, y1 = row * block_size, min((row + 1) * block_size, height)
        for col in range(cols):
            x0, x1 = col * block_size, min((col + 1) * block_size, width)
            yield row, col, slice(y0, y1), slice(x0, x1)


def block_mse(reference: np.ndarray, reconstruction: np.ndarray, block_size: int) -> np.ndarray:
    height, width = reference.shape[:2]
    result = np.empty(grid_shape(height, width, block_size), dtype=np.float64)
    delta = reference.astype(np.float32) - reconstruction.astype(np.float32)
    for row, col, ys, xs in block_slices(height, width, block_size):
        result[row, col] = float(np.mean(np.square(delta[ys, xs])))
    return result


def residual_proxy_bpp(reference: np.ndarray, base: np.ndarray, block_size: int) -> np.ndarray:
    """Independent per-block zlib rate proxy, including local stream overhead."""
    height, width = reference.shape[:2]
    result = np.empty(grid_shape(height, width, block_size), dtype=np.float64)
    delta = reference.astype(np.int16) - base.astype(np.int16)
    for row, col, ys, xs in block_slices(height, width, block_size):
        values = np.asarray(delta[ys, xs], dtype="<i2")
        pixel_count = values.shape[0] * values.shape[1]
        result[row, col] = len(zlib.compress(values.tobytes(), level=6)) * 8 / pixel_count
    return result


def top_fraction_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    flat = scores.reshape(-1)
    count = min(flat.size, max(0, int(round(flat.size * ratio))))
    result = np.zeros(flat.size, dtype=bool)
    if count:
        selected = np.argpartition(flat, flat.size - count)[flat.size - count:]
        result[selected] = True
    return result.reshape(scores.shape)


def expand_blocks(block_map: np.ndarray, height: int, width: int, block_size: int) -> np.ndarray:
    expanded = np.repeat(np.repeat(block_map, block_size, axis=0), block_size, axis=1)
    return expanded[:height, :width]


class RefinementBackend(Protocol):
    def refine(self, base: Image.Image, mask: Image.Image, frame_index: int) -> Image.Image: ...


class UnsharpBackend:
    """A deterministic smoke-test backend; it is not the claimed generative model."""

    def refine(self, base: Image.Image, mask: Image.Image, frame_index: int) -> Image.Image:
        del frame_index
        sharpened = base.filter(ImageFilter.UnsharpMask(radius=2.0, percent=180, threshold=2))
        return Image.composite(sharpened, base, mask)


class DiffusersBackend:
    def __init__(self, args: argparse.Namespace):
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:
            raise RuntimeError(
                "diffusers backend needs: pip install diffusers transformers accelerate safetensors"
            ) from exc

        self.torch = torch
        self.args = args
        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        load_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": not args.allow_download,
            "use_safetensors": True,
        }
        if args.variant:
            load_kwargs["variant"] = args.variant
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(args.model, **load_kwargs)
        except OSError as exc:
            raise RuntimeError(
                f"model is not available locally: {args.model!r}. Download it first or pass --allow-download."
            ) from exc
        self.pipe = self.pipe.to(args.device)
        self.pipe.set_progress_bar_config(disable=True)

    def refine(self, base: Image.Image, mask: Image.Image, frame_index: int) -> Image.Image:
        # Reusing the seed across adjacent frames makes the noise realization less
        # erratic, although an image model still cannot guarantee temporal coherence.
        generator = self.torch.Generator(device=self.args.device).manual_seed(self.args.seed)
        result = self.pipe(
            prompt=self.args.prompt,
            negative_prompt=self.args.negative_prompt,
            image=base,
            mask_image=mask,
            num_inference_steps=self.args.steps,
            guidance_scale=self.args.guidance_scale,
            generator=generator,
            height=base.height,
            width=base.width,
        ).images[0].convert("RGB")
        # Diffusers may alter pixels outside the mask.  The proposed codec is
        # local, so enforce exact base pixels outside the candidate region.
        return Image.composite(result, base, mask)


def make_backend(args: argparse.Namespace) -> RefinementBackend:
    if args.backend == "unsharp":
        return UnsharpBackend()
    return DiffusersBackend(args)


def cap_route(route: np.ndarray, route_id: int, max_ratio: float, benefit: np.ndarray) -> None:
    locations = np.flatnonzero(route.reshape(-1) == route_id)
    limit = int(round(route.size * max_ratio))
    if locations.size <= limit:
        return
    flat_benefit = benefit.reshape(-1)
    order = locations[np.argsort(flat_benefit[locations])[::-1]]
    route.reshape(-1)[order[limit:]] = ROUTE_BASE


def floor_generate_route(
    route: np.ndarray,
    candidate: np.ndarray,
    min_ratio: float,
    benefit: np.ndarray,
) -> None:
    """Force a small qualitative sample while keeping the choice oracle-ranked."""
    minimum = int(round(route.size * min_ratio))
    current = int(np.sum(route == ROUTE_GENERATE))
    if current >= minimum:
        return
    available = np.flatnonzero(
        candidate.reshape(-1) & (route.reshape(-1) == ROUTE_BASE)
    )
    count = min(minimum - current, available.size)
    if count:
        order = available[np.argsort(benefit.reshape(-1)[available])[::-1]]
        route.reshape(-1)[order[:count]] = ROUTE_GENERATE


def choose_routes(
    base_cost: np.ndarray,
    generated_cost: np.ndarray,
    residual_bpp: np.ndarray,
    candidate: np.ndarray,
    lambda_rate: float,
    gamma_compute: float,
    min_generate_ratio: float,
    max_generate_ratio: float,
    max_residual_ratio: float,
) -> np.ndarray:
    gen_cost = generated_cost + gamma_compute
    gen_cost = np.where(candidate, gen_cost, np.inf)
    res_cost = residual_bpp * lambda_rate
    stacked = np.stack((base_cost, gen_cost, res_cost), axis=0)
    route = np.argmin(stacked, axis=0).astype(np.uint8)

    cap_route(route, ROUTE_GENERATE, max_generate_ratio, base_cost - gen_cost)
    cap_route(route, ROUTE_RESIDUAL, max_residual_ratio, base_cost - res_cost)
    floor_generate_route(route, candidate, min_generate_ratio, base_cost - gen_cost)
    return route


def pack_routes(routes: np.ndarray) -> bytes:
    flat = routes.astype(np.uint8).reshape(-1)
    padded = np.pad(flat, (0, (-flat.size) % 4))
    packed = padded[0::4] | (padded[1::4] << 2) | (padded[2::4] << 4) | (padded[3::4] << 6)
    return packed.tobytes()


def unpack_routes(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    flat = np.empty(packed.size * 4, dtype=np.uint8)
    flat[0::4] = packed & 0x03
    flat[1::4] = (packed >> 2) & 0x03
    flat[2::4] = (packed >> 4) & 0x03
    flat[3::4] = (packed >> 6) & 0x03
    return flat[: math.prod(shape)].reshape(shape)


@dataclass
class SideStreamStats:
    total_bytes: int
    header_bytes: int
    mask_bytes: int
    residual_bytes: int


def write_side_stream(
    path: Path,
    routes: np.ndarray,
    originals: list[np.ndarray],
    bases: list[np.ndarray],
    block_size: int,
) -> SideStreamStats:
    height, width = originals[0].shape[:2]
    route_raw = pack_routes(routes)
    route_compressed = zlib.compress(route_raw, level=9)
    residual_chunks = []
    residual_value_count = 0
    for index, (original, base) in enumerate(zip(originals, bases)):
        mask = expand_blocks(routes[index] == ROUTE_RESIDUAL, height, width, block_size)
        values = (original.astype(np.int16) - base.astype(np.int16))[mask]
        values = np.asarray(values, dtype="<i2")
        residual_value_count += values.size
        residual_chunks.append(values.tobytes())
    residual_compressed = zlib.compress(b"".join(residual_chunks), level=9)
    header = {
        "version": 1,
        "frame_shape": [len(originals), height, width, 3],
        "route_shape": list(routes.shape),
        "block_size": block_size,
        "residual_value_count": residual_value_count,
        "route_codec": "2-bit-pack+zlib",
        "residual_codec": "little-endian-int16+zlib",
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<4sIII", SIDE_STREAM_MAGIC, len(header_bytes), len(route_compressed), len(residual_compressed))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prefix + header_bytes + route_compressed + residual_compressed)
    return SideStreamStats(
        total_bytes=path.stat().st_size,
        header_bytes=len(prefix) + len(header_bytes),
        mask_bytes=len(route_compressed),
        residual_bytes=len(residual_compressed),
    )


def decode_side_stream(path: Path, bases: list[np.ndarray], generated: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    payload = path.read_bytes()
    prefix_size = struct.calcsize("<4sIII")
    magic, header_size, route_size, residual_size = struct.unpack("<4sIII", payload[:prefix_size])
    if magic != SIDE_STREAM_MAGIC:
        raise ValueError(f"invalid Stage-0 side stream: {path}")
    cursor = prefix_size
    header = json.loads(payload[cursor:cursor + header_size])
    cursor += header_size
    route_raw = zlib.decompress(payload[cursor:cursor + route_size])
    cursor += route_size
    residual_raw = zlib.decompress(payload[cursor:cursor + residual_size])
    routes = unpack_routes(route_raw, tuple(header["route_shape"]))
    residuals = np.frombuffer(residual_raw, dtype="<i2")
    if residuals.size != header["residual_value_count"]:
        raise ValueError("corrupt residual payload")

    _, height, width, _ = header["frame_shape"]
    block_size = header["block_size"]
    outputs = []
    value_cursor = 0
    for index, base in enumerate(bases):
        gen_mask = expand_blocks(routes[index] == ROUTE_GENERATE, height, width, block_size)
        res_mask = expand_blocks(routes[index] == ROUTE_RESIDUAL, height, width, block_size)
        output = base.copy()
        output[gen_mask] = generated[index][gen_mask]
        value_count = int(res_mask.sum()) * 3
        values = residuals[value_cursor:value_cursor + value_count].reshape(-1, 3)
        corrected = base[res_mask].astype(np.int16) + values
        output[res_mask] = np.clip(corrected, 0, 255).astype(np.uint8)
        value_cursor += value_count
        outputs.append(output)
    if value_cursor != residuals.size:
        raise ValueError("unused residual values after side-stream decode")
    return outputs, routes


def mse(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    delta = reference.astype(np.float64) - reconstruction.astype(np.float64)
    return float(np.mean(np.square(delta)))


def psnr(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    value = mse(reference, reconstruction)
    return float("inf") if value == 0 else 10.0 * math.log10((255.0 ** 2) / value)


def temporal_delta_mae(reference: list[np.ndarray], reconstruction: list[np.ndarray]) -> float | None:
    if len(reference) < 2:
        return None
    errors = []
    for index in range(1, len(reference)):
        ref_delta = reference[index].astype(np.int16) - reference[index - 1].astype(np.int16)
        rec_delta = reconstruction[index].astype(np.int16) - reconstruction[index - 1].astype(np.int16)
        errors.append(np.mean(np.abs(ref_delta.astype(np.int32) - rec_delta.astype(np.int32))))
    return float(np.mean(errors))


def evaluate_lpips(reference: list[np.ndarray], variants: dict[str, list[np.ndarray]], device: str) -> dict[str, float]:
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError("--lpips needs: pip install lpips") from exc
    model = lpips.LPIPS(net="alex").to(device).eval()
    scores: dict[str, list[float]] = {name: [] for name in variants}
    with torch.inference_mode():
        for frame_index, original in enumerate(reference):
            target = torch.from_numpy(original.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
            for name, frames in variants.items():
                value = torch.from_numpy(frames[frame_index].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
                scores[name].append(float(model(target, value).item()))
    return {name: float(np.mean(values)) for name, values in scores.items()}


def route_overlay(base: np.ndarray, route_blocks: np.ndarray, block_size: int) -> np.ndarray:
    route_pixels = expand_blocks(route_blocks, base.shape[0], base.shape[1], block_size)
    colors = ROUTE_COLORS[route_pixels]
    return np.round(base.astype(np.float32) * 0.45 + colors.astype(np.float32) * 0.55).astype(np.uint8)


def label_panel(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill="black", font=ImageFont.load_default())
    return canvas


def make_panel(original: np.ndarray, base: np.ndarray, generated: np.ndarray, final: np.ndarray, overlay: np.ndarray) -> Image.Image:
    items = (
        label_panel(Image.fromarray(original), "original"),
        label_panel(Image.fromarray(base), "DCVC-UF base"),
        label_panel(Image.fromarray(generated), "generator candidate"),
        label_panel(Image.fromarray(final), "selective final"),
        label_panel(Image.fromarray(overlay), "route: blue=base green=gen red=res"),
    )
    panel = Image.new("RGB", (sum(item.width for item in items), items[0].height), "white")
    x = 0
    for item in items:
        panel.paste(item, (x, 0))
        x += item.width
    return panel


def safe_number(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    pairs = paired_frames(args.original_dir, args.base_dir, args.frame_count)
    originals = [read_rgb(pair[0]) for pair in pairs]
    bases = [read_rgb(pair[1]) for pair in pairs]
    shape = originals[0].shape
    if any(frame.shape != shape for frame in originals + bases):
        raise ValueError("all original and base frames must have the same RGB dimensions")
    height, width = shape[:2]
    backend = make_backend(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_costs = []
    candidates = []
    generated = []
    generator_times = []
    for index, (original, base) in enumerate(zip(originals, bases)):
        base_cost = block_mse(original, base, args.block_size)
        candidate = top_fraction_mask(base_cost, args.candidate_ratio)
        candidate_pixels = expand_blocks(candidate, height, width, args.block_size)
        mask_image = Image.fromarray(candidate_pixels.astype(np.uint8) * 255, mode="L")
        start = time.perf_counter()
        refined = backend.refine(Image.fromarray(base), mask_image, index)
        generator_times.append(time.perf_counter() - start)
        refined_array = np.asarray(refined, dtype=np.uint8).copy()
        refined_array[~candidate_pixels] = base[~candidate_pixels]
        base_costs.append(base_cost)
        candidates.append(candidate)
        generated.append(refined_array)
        print(f"[{index + 1:04d}/{len(pairs):04d}] generator {generator_times[-1]:.3f}s")

    route_frames = []
    frame_rows = []
    for index, (original, base, refined) in enumerate(zip(originals, bases, generated)):
        base_cost = base_costs[index]
        gen_cost = block_mse(original, refined, args.block_size)
        rate_proxy = residual_proxy_bpp(original, base, args.block_size)
        routes = choose_routes(
            base_cost, gen_cost, rate_proxy, candidates[index], args.lambda_rate,
            args.gamma_compute, args.min_generate_ratio, args.max_generate_ratio,
            args.max_residual_ratio,
        )
        route_frames.append(routes)
        frame_rows.append({
            "frame": index + 1,
            "base_psnr": psnr(original, base),
            "generated_candidate_psnr": psnr(original, refined),
            "candidate_ratio": float(np.mean(candidates[index])),
            "generate_ratio": float(np.mean(routes == ROUTE_GENERATE)),
            "residual_ratio": float(np.mean(routes == ROUTE_RESIDUAL)),
            "generator_seconds": generator_times[index],
        })
    routes_array = np.stack(route_frames)

    side_stream_path = args.output_dir / "selective_side_stream.bin"
    stream_stats = write_side_stream(side_stream_path, routes_array, originals, bases, args.block_size)
    finals, decoded_routes = decode_side_stream(side_stream_path, bases, generated)
    if not np.array_equal(routes_array, decoded_routes):
        raise AssertionError("side-stream route round trip failed")
    for original, base, final, routes in zip(originals, bases, finals, route_frames):
        residual_mask = expand_blocks(routes == ROUTE_RESIDUAL, height, width, args.block_size)
        if not np.array_equal(final[residual_mask], original[residual_mask]):
            raise AssertionError("decoded residual pixels are not lossless")
        base_mask = expand_blocks(routes == ROUTE_BASE, height, width, args.block_size)
        if not np.array_equal(final[base_mask], base[base_mask]):
            raise AssertionError("base-only pixels changed")

    panels = []
    for index, (original, base, refined, final, routes) in enumerate(
        zip(originals, bases, generated, finals, route_frames), start=1
    ):
        filename = f"im{index:05d}.png"
        save_rgb(args.output_dir / "original" / filename, original)
        save_rgb(args.output_dir / "base" / filename, base)
        save_rgb(args.output_dir / "generated" / filename, refined)
        save_rgb(args.output_dir / "final" / filename, final)
        overlay = route_overlay(base, routes, args.block_size)
        save_rgb(args.output_dir / "routes" / filename, overlay)
        panel = make_panel(original, base, refined, final, overlay)
        panel_path = args.output_dir / "panels" / filename
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path)
        panels.append(panel)
        frame_rows[index - 1]["final_psnr"] = psnr(original, final)

    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    preview_frames = []
    for panel in panels:
        preview = panel.copy()
        preview.thumbnail((1280, 300), Image.Resampling.LANCZOS)
        preview_frames.append(preview)
    preview_frames[0].save(
        args.output_dir / "preview.gif", save_all=True, append_images=preview_frames[1:],
        duration=max(1, round(1000 / args.gif_fps)), loop=0, optimize=False,
    )
    contact_count = min(4, len(preview_frames))
    contact = Image.new("RGB", (preview_frames[0].width, sum(frame.height for frame in preview_frames[:contact_count])), "white")
    y = 0
    for frame in preview_frames[:contact_count]:
        contact.paste(frame, (0, y))
        y += frame.height
    contact.save(args.output_dir / "contact_sheet.png")

    pixel_count = len(originals) * height * width
    base_bytes = args.base_bitstream.stat().st_size if args.base_bitstream else None
    summary = {
        "stage": "Stage 0: no-training selective generation + selective residual",
        "oracle_warning": "Routing uses original frames and is not decoder-available; this is an upper-bound/proof-of-concept experiment.",
        "backend": args.backend,
        "model": args.model if args.backend == "diffusers" else None,
        "frames": len(originals),
        "resolution": [width, height],
        "block_size": args.block_size,
        "routing": {
            "candidate_ratio": float(np.mean(np.stack(candidates))),
            "base_ratio": float(np.mean(routes_array == ROUTE_BASE)),
            "generate_ratio": float(np.mean(routes_array == ROUTE_GENERATE)),
            "residual_ratio": float(np.mean(routes_array == ROUTE_RESIDUAL)),
            "lambda_rate": args.lambda_rate,
            "gamma_compute": args.gamma_compute,
        },
        "rate": {
            "base_bitstream_bytes": base_bytes,
            "base_bpp": None if base_bytes is None else base_bytes * 8 / pixel_count,
            "side_stream_bytes": stream_stats.total_bytes,
            "side_stream_bpp": stream_stats.total_bytes * 8 / pixel_count,
            "side_stream_header_bytes": stream_stats.header_bytes,
            "compressed_route_mask_bytes": stream_stats.mask_bytes,
            "compressed_residual_bytes": stream_stats.residual_bytes,
            "total_bpp": None if base_bytes is None else (base_bytes + stream_stats.total_bytes) * 8 / pixel_count,
        },
        "quality": {
            "base_psnr": safe_number(float(np.mean([psnr(a, b) for a, b in zip(originals, bases)]))),
            "generated_candidate_psnr": safe_number(float(np.mean([psnr(a, b) for a, b in zip(originals, generated)]))),
            "final_psnr": safe_number(float(np.mean([psnr(a, b) for a, b in zip(originals, finals)]))),
            "base_temporal_delta_mae": temporal_delta_mae(originals, bases),
            "generated_temporal_delta_mae": temporal_delta_mae(originals, generated),
            "final_temporal_delta_mae": temporal_delta_mae(originals, finals),
        },
        "runtime": {
            "generator_total_seconds": float(sum(generator_times)),
            "generator_seconds_per_frame": float(np.mean(generator_times)),
            "note": "Image backends still execute a full network call per frame; candidate ratio is routed area, not measured FLOP saving.",
        },
        "route_legend": dict(zip(ROUTE_NAMES, ROUTE_COLORS.tolist())),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if args.lpips:
        summary["quality"]["lpips_alex"] = evaluate_lpips(
            originals, {"base": bases, "generated_candidate": generated, "final": finals}, args.device
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "base_psnr": summary["quality"]["base_psnr"],
        "final_psnr": summary["quality"]["final_psnr"],
        "generate_ratio": summary["routing"]["generate_ratio"],
        "residual_ratio": summary["routing"]["residual_ratio"],
        "side_stream_bpp": summary["rate"]["side_stream_bpp"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
