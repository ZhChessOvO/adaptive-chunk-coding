#!/usr/bin/env python3
"""Generate resumable regional counterfactual labels for the A800 pilot.

Each 17-frame 512x512 sample produces three frozen counterfactuals:

* Generate: a real QP8 DCVC-UF stream followed by frozen SeedVR2;
* Base: a real QP16 DCVC-UF stream;
* Enhance: a real QP32 DCVC-UF stream.

The three uniform streams are materialized and read back before decoding, so
their complete on-disk byte counts include headers and entropy payloads.  Each
region also receives the complete byte cost of an independently decodable
QP32 enhancement-tile stream plus its descriptor.  That regional number is an
exact cost for the legal fallback container, not a claim about the cheaper
one-shot spatial-quality format.  Final controller evaluation must therefore
measure the learned route again with the real one-shot format.

DCVC-UF and SeedVR2 are inference-only.  Source RGB is used only at the encoder
to build supervision and encoder-visible features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_seedvr2_bridge import (
    configure_runner,
    pad_temporal,
    resize_and_normalize,
    resize_output,
)
from demo.stage_c_three_path_roi_probe import (
    CONTAINER_HEADER,
    TILE_HEADER,
    dcvc_stream_breakdown,
    decode_dcvc_stream,
    encode_dcvc_stream,
    load_codecs,
    tile_boxes,
)
from src.utils.common import set_torch_env


PROTOCOL_VERSION = 1
FEATURE_NAMES = (
    "x_center",
    "y_center",
    "border_region",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "luma_mean",
    "luma_std",
    "gradient_mean",
    "edge_density",
    "motion_mean",
    "motion_std",
    "temporal_luma_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen DCVC-UF/SeedVR2 regional teacher for the A800 pilot")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scratch-dir", type=Path,
        default=Path("/root/autodl-tmp/DCVC/tmp/a800_teacher"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--generate-qp", type=int, default=8)
    parser.add_argument("--base-qp", type=int, default=16)
    parser.add_argument("--enhance-qp", type=int, default=32)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument(
        "--upstream-root", type=Path,
        default=Path("third_party/SeedVR2"))
    parser.add_argument(
        "--dit-checkpoint", type=Path,
        default=Path(
            "third_party/SeedVR2/ckpts/seedvr2_ema_3b_bf16.safetensors"))
    parser.add_argument(
        "--lora-checkpoint", type=Path,
        help="Optional SeedVR2 project LoRA adapter")
    parser.add_argument(
        "--vae-checkpoint", type=Path,
        default=Path("third_party/SeedVR2/ckpts/ema_vae.pth"))
    parser.add_argument(
        "--positive-embedding", type=Path,
        default=Path("third_party/SeedVR2/pos_emb.pt"))
    parser.add_argument(
        "--negative-embedding", type=Path,
        default=Path("third_party/SeedVR2/neg_emb.pt"))
    parser.add_argument("--sample-steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--dit-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--visual-count", type=int, default=4)
    parser.add_argument("--max-wall-seconds", type=int, default=12 * 60 * 60)
    parser.add_argument("--disk-stop-percent", type=float, default=80.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.tile_size != 128:
        parser.error("the fixed first pilot uses a 4x4 grid of 128-pixel regions")
    if not 0 <= args.generate_qp < args.base_qp < args.enhance_qp < 64:
        parser.error("require Generate < Base < Enhance inside [0, 63]")
    if args.sample_steps < 1 or args.lpips_batch_size < 1:
        parser.error("sample steps and LPIPS batch size must be positive")
    if args.max_wall_seconds < 60:
        parser.error("--max-wall-seconds must be at least one minute")
    return args


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"empty sample manifest: {path}")
    return records


def load_source(record: dict) -> list[np.ndarray]:
    crop = record["crop"]
    frames = []
    for value in record["source_files"]:
        path = Path(value)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        cropped = rgb[
            crop["y"]:crop["y"] + crop["height"],
            crop["x"]:crop["x"] + crop["width"],
        ]
        if cropped.shape != (512, 512, 3):
            raise ValueError(f"invalid crop for {path}: {cropped.shape}")
        frames.append(cropped.copy())
    if len(frames) != 17:
        raise ValueError(f"{record['sample_id']} has {len(frames)} frames, expected 17")
    return frames


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def stream_roundtrip(
    *, stream: bytes, path: Path, frame_count: int, i_net, p_net,
    device: torch.device,
) -> tuple[list[np.ndarray], int, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(stream)
    os.replace(temporary, path)
    on_disk_bytes = path.stat().st_size
    data = path.read_bytes()
    if len(data) != on_disk_bytes or data != stream:
        raise RuntimeError("materialized DCVC-UF stream differs from encoder output")
    decoded = decode_dcvc_stream(data, frame_count, i_net, p_net, device)
    breakdown = dcvc_stream_breakdown(data, frame_count)
    if breakdown["total_bytes"] != on_disk_bytes:
        raise RuntimeError("stream byte accounting differs from the file size")
    path.unlink()
    return decoded, on_disk_bytes, breakdown


def materialized_size(stream: bytes, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(stream)
    os.replace(temporary, path)
    size = path.stat().st_size
    if path.read_bytes() != stream:
        raise RuntimeError("materialized regional stream differs from encoder output")
    path.unlink()
    return size


def to_lpips_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    values = np.stack(frames)
    tensor = torch.from_numpy(values.copy()).permute(0, 3, 1, 2)
    return tensor.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)


class SpatialLPIPS:
    def __init__(self, device: torch.device, batch_size: int) -> None:
        import lpips

        self.model = lpips.LPIPS(
            net="alex", spatial=True, verbose=False).eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def maps(
        self, reference: list[np.ndarray], reconstruction: list[np.ndarray]
    ) -> np.ndarray:
        first = to_lpips_tensor(reference, self.device)
        second = to_lpips_tensor(reconstruction, self.device)
        outputs = []
        for start in range(0, first.shape[0], self.batch_size):
            outputs.append(self.model(
                first[start:start + self.batch_size],
                second[start:start + self.batch_size],
            ).float().cpu())
        return torch.cat(outputs).squeeze(1).numpy()


def source_features(
    frames: list[np.ndarray], boxes: list[tuple[int, int, int, int]]
) -> list[dict[str, float]]:
    values = np.stack(frames).astype(np.float32) / 255.0
    luma = (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )
    gradient_x = np.abs(np.diff(luma, axis=2, prepend=luma[:, :, :1]))
    gradient_y = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1, :]))
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    motion = np.abs(np.diff(luma, axis=0))
    result = []
    for index, (x, y, width, height) in enumerate(boxes):
        tile = values[:, y:y + height, x:x + width]
        tile_luma = luma[:, y:y + height, x:x + width]
        tile_gradient = gradient[:, y:y + height, x:x + width]
        tile_motion = motion[:, y:y + height, x:x + width]
        row, column = divmod(index, 4)
        features = {
            "x_center": (x + width / 2) / values.shape[2],
            "y_center": (y + height / 2) / values.shape[1],
            "border_region": float(row in (0, 3) or column in (0, 3)),
            "mean_r": float(tile[..., 0].mean()),
            "mean_g": float(tile[..., 1].mean()),
            "mean_b": float(tile[..., 2].mean()),
            "std_r": float(tile[..., 0].std()),
            "std_g": float(tile[..., 1].std()),
            "std_b": float(tile[..., 2].std()),
            "luma_mean": float(tile_luma.mean()),
            "luma_std": float(tile_luma.std()),
            "gradient_mean": float(tile_gradient.mean()),
            "edge_density": float((tile_gradient > 0.08).mean()),
            "motion_mean": float(tile_motion.mean()),
            "motion_std": float(tile_motion.std()),
            "temporal_luma_std": float(tile_luma.mean(axis=(1, 2)).std()),
        }
        if tuple(features) != FEATURE_NAMES:
            raise RuntimeError("feature order differs from the registered schema")
        result.append(features)
    return result


def local_quality(
    reference: list[np.ndarray], reconstruction: list[np.ndarray],
    lpips_maps: np.ndarray, box: tuple[int, int, int, int],
) -> dict[str, float]:
    x, y, width, height = box
    source = np.stack([
        frame[y:y + height, x:x + width] for frame in reference
    ]).astype(np.float32)
    decoded = np.stack([
        frame[y:y + height, x:x + width] for frame in reconstruction
    ]).astype(np.float32)
    mse = float(np.mean((source - decoded) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    if len(source) > 1:
        source_delta = np.diff(source, axis=0)
        decoded_delta = np.diff(decoded, axis=0)
        temporal = float(np.mean(np.abs(source_delta - decoded_delta)))
    else:
        temporal = 0.0
    lpips_value = float(lpips_maps[:, y:y + height, x:x + width].mean())
    return {
        "lpips_alex": lpips_value,
        "psnr_db": psnr,
        "rgb_mse": mse,
        "temporal_delta_mae": temporal,
    }


def arrays_from_seed_tensor(sample: torch.Tensor, frame_count: int) -> list[np.ndarray]:
    if sample.ndim == 3:
        sample = sample.unsqueeze(1)
    if sample.ndim != 4:
        raise ValueError(
            f"SeedVR2 output must be C,T,H,W (or C,H,W for T=1), got {sample.shape}")
    sample = sample[:, :frame_count]
    sample = sample.permute(1, 2, 3, 0).float().cpu()
    values = sample.clamp(-1, 1).add_(1).mul_(127.5).round_().byte().numpy()
    return [frame.copy() for frame in values]


class PersistentSeedVR2:
    def __init__(self, args: argparse.Namespace) -> None:
        bridge_args = SimpleNamespace(
            upstream_root=resolve(args.upstream_root),
            dit_checkpoint=resolve(args.dit_checkpoint),
            lora_checkpoint=(
                resolve(args.lora_checkpoint)
                if getattr(args, "lora_checkpoint", None) is not None else None),
            lora_strength=float(getattr(args, "lora_strength", 1.0)),
            vae_checkpoint=resolve(args.vae_checkpoint),
            positive_embedding=resolve(args.positive_embedding),
            negative_embedding=resolve(args.negative_embedding),
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
            dit_dtype=args.dit_dtype,
        )
        started = time.perf_counter()
        self.runner, self.device = configure_runner(bridge_args)
        self.dtype = getattr(torch, args.dit_dtype)
        self.runner.dit.requires_grad_(False).eval().to(
            device=self.device, dtype=self.dtype)
        self.runner.vae.to(self.device)
        self.positive = torch.load(
            bridge_args.positive_embedding, map_location=self.device)
        self.negative = torch.load(
            bridge_args.negative_embedding, map_location=self.device)
        self.cfg_scale = args.cfg_scale
        self.sample_steps = args.sample_steps
        self.model_load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def restore(
        self, frames: list[np.ndarray], seed: int,
        processing_height: int | None = None,
        processing_width: int | None = None,
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        values = np.stack(frames)
        input_height, input_width = values.shape[1:3]
        processing_height = processing_height or input_height
        processing_width = processing_width or input_width
        output_height = output_height or input_height
        output_width = output_width or input_width
        if any(value % 16 for value in (processing_height, processing_width)):
            raise ValueError("SeedVR2 processing dimensions must be divisible by 16")
        tensor = torch.from_numpy(values.copy()).permute(0, 3, 1, 2)
        tensor = tensor.float().div_(255.0)
        condition, original_length = pad_temporal(
            resize_and_normalize(
                tensor, processing_height, processing_width, self.device))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=self.dtype == torch.bfloat16,
        ):
            cond_latents = self.runner.vae_encode([condition])
            noises = [torch.randn_like(cond_latents[0])]
            conditions = [self.runner.get_condition(
                noises[0], task="sr", latent_blur=cond_latents[0])]
            outputs = self.runner.inference(
                noises=noises,
                conditions=conditions,
                texts_pos=[self.positive],
                texts_neg=[self.negative],
                cfg_scale=self.cfg_scale,
                dit_offload=False,
            )
            outputs = [
                resize_output(sample, output_height, output_width)
                for sample in outputs
            ]
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(self.device))
        is_full_frame = (
            input_height == 512
            and input_width == 512
            and processing_height == 512
            and processing_width == 512
            and output_height == 512
            and output_width == 512
        )
        return arrays_from_seed_tensor(outputs[0], original_length), {
            "seconds_model_load_excluded": elapsed,
            "peak_cuda_allocated_bytes": peak,
            "input_shape": [original_length, input_height, input_width],
            "processing_shape": [
                original_length, processing_height, processing_width],
            "output_shape": [original_length, output_height, output_width],
            "seed": seed,
            "sample_steps": self.sample_steps,
            "actual_compute_scope": "full-frame" if is_full_frame else "roi-crop",
            "full_frame_actual_compute": is_full_frame,
            "roi_compute_measured_here": not is_full_frame,
        }


def save_visual(
    path: Path, reference: list[np.ndarray], variants: dict[str, list[np.ndarray]],
) -> None:
    frame_index = 8
    images = [Image.fromarray(reference[frame_index])]
    images.extend(Image.fromarray(frames[frame_index]) for frames in variants.values())
    canvas = Image.new("RGB", (512 * len(images), 512), "white")
    for index, image in enumerate(images):
        canvas.paste(image, (512 * index, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def disk_percent(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total


def build_manifest(
    output_dir: Path, records: list[dict], started: float,
    model_load_seconds: float,
) -> dict:
    entries = []
    for record in records:
        path = output_dir / "samples" / f"{record['sample_id']}.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "sample_id": record["sample_id"],
            "path": str(path),
            "split": record["split"],
            "sequence": record["sequence"],
            "seconds": value["runtime"]["sample_total_seconds"],
            "seedvr2_seconds": value["runtime"]["seedvr2"][
                "seconds_model_load_excluded"],
            "uniform_stream_bytes": value["rate"]["uniform_stream_bytes"],
        })
    manifest = {
        "experiment": "A800 frozen counterfactual teacher labels",
        "protocol_version": PROTOCOL_VERSION,
        "requested_sample_count": len(records),
        "completed_sample_count": len(entries),
        "complete": len(entries) == len(records),
        "entries": entries,
        "feature_names": list(FEATURE_NAMES),
        "model_load_seconds": model_load_seconds,
        "current_process_elapsed_seconds": time.perf_counter() - started,
        "scientific_boundary": {
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "source_rgb_used_for_teacher_only": True,
            "uniform_stream_bytes_are_actual_on_disk": True,
            "regional_enhance_cost_format": (
                "actual independent QP32 tile stream plus descriptor; legal fallback"
            ),
            "regional_enhance_cost_is_one_shot_spatial_stream_cost": False,
            "one_shot_spatial_stream_requires_final_remeasurement": True,
            "true_fill_used": False,
            "validation_outside_000_005_read": False,
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    records = read_jsonl(args.sample_manifest)
    if args.limit is not None:
        records = records[:args.limit]
    if any(record["frame_count"] != 17 for record in records):
        raise ValueError("sample manifest contains a non-17-frame record")
    if any(record["crop"]["width"] != 512 or record["crop"]["height"] != 512
           for record in records):
        raise ValueError("sample manifest contains a non-512x512 crop")
    supported_splits = {"train", "development", "v6_adaptation_train"}
    unknown_splits = sorted({
        record["split"] for record in records
        if record["split"] not in supported_splits
    })
    if unknown_splits:
        raise ValueError(
            f"unknown sample split(s) in sample manifest: {unknown_splits}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = [
        args.output_dir / "samples" / f"{record['sample_id']}.json"
        for record in records
    ]
    if existing_paths and all(path.is_file() for path in existing_paths):
        for path, record in zip(existing_paths, records):
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (existing.get("protocol_version") != PROTOCOL_VERSION
                    or existing.get("sample", {}).get("sample_id")
                    != record["sample_id"]):
                raise RuntimeError(f"incompatible resumed label: {path}")
        previous_manifest_path = args.output_dir / "manifest.json"
        previous_model_load = 0.0
        if previous_manifest_path.is_file():
            previous_model_load = json.loads(
                previous_manifest_path.read_text(encoding="utf-8")
            ).get("model_load_seconds", 0.0)
        manifest = build_manifest(
            args.output_dir, records, process_started, previous_model_load)
        print(json.dumps({
            "stage": "teacher-resume-all-complete",
            "manifest": str(previous_manifest_path),
            "requested": manifest["requested_sample_count"],
            "completed": manifest["completed_sample_count"],
        }, ensure_ascii=False, indent=2))
        return
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_args = SimpleNamespace(
        model_path_i=resolve(args.model_path_i),
        model_path_p=resolve(args.model_path_p),
        skip_thres=args.skip_thres,
    )
    codec_load_started = time.perf_counter()
    i_net, p_net = load_codecs(codec_args, device)
    codec_load_seconds = time.perf_counter() - codec_load_started
    # DCVC-UF captures its optimized proxy kernels into CUDA graphs on first
    # use.  CUDA 13 cannot begin that capture on the legacy/default stream, so
    # keep every codec encode/decode on one dedicated non-default stream.
    codec_stream = torch.cuda.Stream(device=device)
    seedvr2 = PersistentSeedVR2(args)
    metric = SpatialLPIPS(device, args.lpips_batch_size)
    boxes = tile_boxes(512, 512, args.tile_size)
    if len(boxes) != 16:
        raise RuntimeError("first pilot must have exactly 16 regions")

    completed_this_process = 0
    for record_index, record in enumerate(records):
        output_path = args.output_dir / "samples" / f"{record['sample_id']}.json"
        if output_path.is_file():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (existing.get("protocol_version") != PROTOCOL_VERSION
                    or existing.get("sample", {}).get("sample_id")
                    != record["sample_id"]):
                raise RuntimeError(f"incompatible resumed label: {output_path}")
            print(json.dumps({
                "stage": "teacher-resume-skip",
                "sample_id": record["sample_id"],
                "completed": record_index + 1,
                "requested": len(records),
            }), flush=True)
            continue
        if time.perf_counter() - process_started >= args.max_wall_seconds:
            print(json.dumps({
                "stage": "teacher-time-limit",
                "elapsed_seconds": time.perf_counter() - process_started,
                "completed_this_process": completed_this_process,
            }), flush=True)
            break
        usage = {
            "system": disk_percent(Path("/root")),
            "fast": disk_percent(Path("/root/autodl-tmp")),
            "file_store": disk_percent(Path("/root/autodl-fs")),
        }
        if max(usage.values()) >= args.disk_stop_percent:
            print(json.dumps({
                "stage": "teacher-disk-limit",
                "disk_percent": usage,
            }), flush=True)
            break

        sample_started = time.perf_counter()
        sample_scratch = args.scratch_dir / record["sample_id"]
        if sample_scratch.exists():
            shutil.rmtree(sample_scratch)
        sample_scratch.mkdir(parents=True, exist_ok=False)
        originals = load_source(record)
        candidates: dict[str, list[np.ndarray]] = {}
        rate: dict[str, object] = {
            "uniform_stream_bytes": {},
            "uniform_stream_breakdown": {},
        }
        encode_runtime = {}
        for action, qp in (
            ("Generate", args.generate_qp),
            ("Base", args.base_qp),
            ("Enhance", args.enhance_qp),
        ):
            with torch.cuda.stream(codec_stream):
                stream, stats = encode_dcvc_stream(
                    originals, qp, qp, i_net, p_net, device,
                    args.reset_interval)
                decoded, byte_count, breakdown = stream_roundtrip(
                    stream=stream,
                    path=sample_scratch / f"uniform_qp{qp}.dcvc",
                    frame_count=17, i_net=i_net, p_net=p_net, device=device)
            candidates[action] = decoded
            rate["uniform_stream_bytes"][action] = byte_count
            rate["uniform_stream_breakdown"][action] = breakdown
            encode_runtime[action] = stats

        generated, seed_runtime = seedvr2.restore(
            candidates["Generate"], int(record["seed"]))
        candidates["Generate"] = generated

        regional_enhance_bytes = []
        tile_encode_seconds = []
        for tile_index, box in enumerate(boxes):
            x, y, width, height = box
            tile_frames = [
                frame[y:y + height, x:x + width].copy()
                for frame in originals
            ]
            with torch.cuda.stream(codec_stream):
                tile_stream, stats = encode_dcvc_stream(
                    tile_frames, args.enhance_qp, args.enhance_qp,
                    i_net, p_net, device, args.reset_interval)
            tile_bytes = materialized_size(
                tile_stream, sample_scratch / f"enhance_tile_{tile_index:02d}.dcvc")
            regional_enhance_bytes.append(tile_bytes + TILE_HEADER.size)
            tile_encode_seconds.append(float(stats["seconds"]))

        features = source_features(originals, boxes)
        lpips_maps = {
            action: metric.maps(originals, frames)
            for action, frames in candidates.items()
        }
        regions = []
        for index, box in enumerate(boxes):
            quality = {
                action: local_quality(
                    originals, candidates[action], lpips_maps[action], box)
                for action in ("Generate", "Base", "Enhance")
            }
            base = quality["Base"]
            generate = quality["Generate"]
            enhance = quality["Enhance"]
            row, column = divmod(index, 4)
            regions.append({
                "index": index,
                "row": row,
                "column": column,
                "box": list(box),
                "features": features[index],
                "candidates": quality,
                "targets": {
                    "generate_lpips_gain_vs_base": (
                        base["lpips_alex"] - generate["lpips_alex"]),
                    "enhance_lpips_gain_vs_base": (
                        base["lpips_alex"] - enhance["lpips_alex"]),
                    "generate_psnr_delta_db_vs_base": (
                        generate["psnr_db"] - base["psnr_db"]),
                    "enhance_psnr_delta_db_vs_base": (
                        enhance["psnr_db"] - base["psnr_db"]),
                    "generate_temporal_risk_vs_base": max(
                        0.0,
                        generate["temporal_delta_mae"]
                        - base["temporal_delta_mae"]),
                    "enhance_temporal_gain_vs_base": (
                        base["temporal_delta_mae"]
                        - enhance["temporal_delta_mae"]),
                    "enhance_fallback_extra_on_disk_bytes": (
                        regional_enhance_bytes[index]),
                    "generate_region_area_pixels": args.tile_size ** 2,
                    "generate_region_area_fraction": 1.0 / len(boxes),
                },
            })

        rate.update({
            "action_map_bytes_for_16_cells": (len(boxes) + 3) // 4,
            "legal_fallback_container_header_bytes": CONTAINER_HEADER.size,
            "legal_fallback_tile_descriptor_bytes_each": TILE_HEADER.size,
            "regional_enhance_extra_on_disk_bytes": regional_enhance_bytes,
            "regional_cost_warning": (
                "Exact for the legal independent-tile fallback only; final learned "
                "routes must be re-encoded and measured with the one-shot spatial format."
            ),
        })
        sample_seconds = time.perf_counter() - sample_started
        result = {
            "experiment": "A800 regional frozen-teacher counterfactual",
            "protocol_version": PROTOCOL_VERSION,
            "sample": record,
            "configuration": {
                "quality_profile": {
                    "Generate": args.generate_qp,
                    "Base": args.base_qp,
                    "Enhance": args.enhance_qp,
                },
                "tile_size": args.tile_size,
                "tile_grid": [4, 4],
                "feature_names": list(FEATURE_NAMES),
                "seedvr2_sample_steps": args.sample_steps,
                "seedvr2_cfg_scale": args.cfg_scale,
            },
            "rate": rate,
            "regions": regions,
            "runtime": {
                "sample_total_seconds": sample_seconds,
                "codec_model_load_seconds_process_level": codec_load_seconds,
                "seedvr2_model_load_seconds_process_level": (
                    seedvr2.model_load_seconds),
                "uniform_encode": encode_runtime,
                "enhance_tile_encode_seconds": tile_encode_seconds,
                "seedvr2": seed_runtime,
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)),
                "disk_percent_before_sample": usage,
            },
            "scientific_boundary": {
                "dcvc_uf_frozen": True,
                "seedvr2_frozen": True,
                "training_or_finetuning": False,
                "source_rgb_visible_to_teacher": True,
                "source_rgb_visible_to_decoder": False,
                "uniform_streams_materialized_then_read_for_decode": True,
                "actual_uniform_on_disk_bytes": True,
                "regional_enhance_cost_is_actual_legal_fallback_bytes": True,
                "regional_enhance_cost_is_one_shot_spatial_bytes": False,
                "generate_full_frame_compute_measured": True,
                "generate_roi_compute_measured": False,
                "true_fill_used": False,
            },
        }
        atomic_json(output_path, result)
        if record_index < args.visual_count:
            save_visual(
                args.output_dir / "visuals" / f"{record['sample_id']}.png",
                originals,
                {name: candidates[name] for name in ("Generate", "Base", "Enhance")},
            )
        if any(sample_scratch.iterdir()):
            raise RuntimeError(f"scratch files remain after {record['sample_id']}")
        sample_scratch.rmdir()
        completed_this_process += 1
        print(json.dumps({
            "stage": "teacher-sample-complete",
            "sample_id": record["sample_id"],
            "completed": record_index + 1,
            "requested": len(records),
            "sample_seconds": sample_seconds,
            "seedvr2_seconds": seed_runtime["seconds_model_load_excluded"],
            "peak_cuda_mib": result["runtime"]["peak_cuda_allocated_bytes"] / 1048576,
            "disk_percent": usage,
        }, ensure_ascii=False), flush=True)
        if completed_this_process == 1 or completed_this_process % 10 == 0:
            build_manifest(
                args.output_dir, records, process_started,
                seedvr2.model_load_seconds)

    manifest = build_manifest(
        args.output_dir, records, process_started, seedvr2.model_load_seconds)
    print(json.dumps({
        "manifest": str(args.output_dir / "manifest.json"),
        "requested": manifest["requested_sample_count"],
        "completed": manifest["completed_sample_count"],
        "complete": manifest["complete"],
        "elapsed_seconds": time.perf_counter() - process_started,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("the A800 pilot teacher supports one GPU only")
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
