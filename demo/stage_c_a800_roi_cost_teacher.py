#!/usr/bin/env python3
"""Measure frozen SeedVR2 ROI cost labels for the A800 controller pilot.

Each sample measures four actual connected-crop geometries: corner, horizontal
edge, vertical edge, and interior.  Every one of the 16 regions is assigned the
measurement with the same crop geometry.  This keeps the run bounded while
storing an actual, sample-specific ROI measurement rather than area-prorating
the full-frame runtime.  Final learned routes are still measured end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import (
    PROTOCOL_VERSION,
    PersistentSeedVR2,
    atomic_json,
    disk_percent,
    load_source,
)
from demo.stage_c_three_path_roi_probe import (
    decode_dcvc_stream,
    encode_dcvc_stream,
    load_codecs,
)
from src.utils.common import set_torch_env


ROI_CONTEXT = 64
TILE_SIZE = 128
GEOMETRY_REPRESENTATIVES = {
    "corner-192x192": 0,
    "horizontal-edge-256x192": 1,
    "vertical-edge-192x256": 4,
    "interior-256x256": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actual SeedVR2 ROI compute-cost teacher")
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generate-qp", type=int, default=8)
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
        "--upstream-root", type=Path, default=Path("third_party/SeedVR2"))
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
    parser.add_argument("--max-wall-seconds", type=int, default=10 * 60 * 60)
    parser.add_argument("--disk-stop-percent", type=float, default=80.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def roi_bounds(index: int) -> tuple[int, int, int, int]:
    row, column = divmod(index, 4)
    x0 = max(0, column * TILE_SIZE - ROI_CONTEXT)
    y0 = max(0, row * TILE_SIZE - ROI_CONTEXT)
    x1 = min(512, (column + 1) * TILE_SIZE + ROI_CONTEXT)
    y1 = min(512, (row + 1) * TILE_SIZE + ROI_CONTEXT)
    return x0, y0, x1 - x0, y1 - y0


def geometry_name(index: int) -> str:
    _, _, width, height = roi_bounds(index)
    if (width, height) == (192, 192):
        return "corner-192x192"
    if (width, height) == (256, 192):
        return "horizontal-edge-256x192"
    if (width, height) == (192, 256):
        return "vertical-edge-192x256"
    if (width, height) == (256, 256):
        return "interior-256x256"
    raise RuntimeError(f"unexpected ROI geometry {(width, height)}")


def build_manifest(output_dir: Path, entries: list[dict], requested: int,
                   process_started: float, model_load_seconds: float) -> dict:
    completed = []
    for entry in entries:
        path = output_dir / "samples" / f"{entry['sample_id']}.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        completed.append({
            "sample_id": entry["sample_id"],
            "path": str(path),
            "seconds": value["runtime"]["sample_total_seconds"],
            "geometry_seconds": {
                name: record["seconds_model_load_excluded"]
                for name, record in value["measurements"].items()
            },
        })
    result = {
        "experiment": "A800 actual SeedVR2 ROI compute-cost labels",
        "protocol_version": PROTOCOL_VERSION,
        "requested_sample_count": requested,
        "completed_sample_count": len(completed),
        "complete": len(completed) == requested,
        "entries": completed,
        "model_load_seconds": model_load_seconds,
        "current_process_elapsed_seconds": time.perf_counter() - process_started,
        "measurement_classes": GEOMETRY_REPRESENTATIVES,
        "scientific_boundary": {
            "actual_seedvr2_roi_inference_measured": True,
            "one_measurement_per_sample_and_geometry_class": True,
            "region_uses_measurement_with_identical_geometry": True,
            "full_route_runtime_area_prorated": False,
            "final_connected_routes_remeasured_end_to_end": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
        },
    }
    atomic_json(output_dir / "manifest.json", result)
    return result


def validate_existing(path: Path, sample_id: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("protocol_version") != PROTOCOL_VERSION
            or value.get("sample", {}).get("sample_id") != sample_id):
        raise RuntimeError(f"incompatible resumed ROI-cost label: {path}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    if not teacher.get("complete"):
        raise ValueError("quality teacher manifest must be complete")
    entries = teacher["entries"]
    if args.limit is not None:
        entries = entries[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        args.output_dir / "samples" / f"{entry['sample_id']}.json"
        for entry in entries
    ]
    if output_paths and all(path.is_file() for path in output_paths):
        for path, entry in zip(output_paths, entries):
            validate_existing(path, entry["sample_id"])
        manifest = build_manifest(
            args.output_dir, entries, len(entries), process_started,
            json.loads((args.output_dir / "manifest.json").read_text()).get(
                "model_load_seconds", 0.0)
            if (args.output_dir / "manifest.json").is_file() else 0.0)
        print(json.dumps({
            "stage": "roi-cost-resume-all-complete",
            "completed": manifest["completed_sample_count"],
        }), flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("ROI cost teacher requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(codec_stream):
        i_net, p_net = load_codecs(args, device)

    seed_args = SimpleNamespace(
        upstream_root=resolve(args.upstream_root),
        dit_checkpoint=resolve(args.dit_checkpoint),
        lora_checkpoint=(
            resolve(args.lora_checkpoint)
            if args.lora_checkpoint is not None else None),
        vae_checkpoint=resolve(args.vae_checkpoint),
        positive_embedding=resolve(args.positive_embedding),
        negative_embedding=resolve(args.negative_embedding),
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        dit_dtype=args.dit_dtype,
    )
    seedvr2 = PersistentSeedVR2(seed_args)
    completed_this_process = 0
    for entry_index, entry in enumerate(entries):
        output_path = output_paths[entry_index]
        if output_path.is_file():
            validate_existing(output_path, entry["sample_id"])
            print(json.dumps({
                "stage": "roi-cost-resume-skip",
                "sample_id": entry["sample_id"],
                "completed": entry_index + 1,
                "requested": len(entries),
            }), flush=True)
            continue
        if time.perf_counter() - process_started >= args.max_wall_seconds:
            break
        usage = {
            "system": disk_percent(Path("/root")),
            "fast": disk_percent(Path("/root/autodl-tmp")),
            "file_store": disk_percent(Path("/root/autodl-fs")),
        }
        if max(usage.values()) >= args.disk_stop_percent:
            break

        sample_started = time.perf_counter()
        quality_label = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        record = quality_label["sample"]
        originals = load_source(record)
        with torch.cuda.stream(codec_stream):
            stream, encode_stats = encode_dcvc_stream(
                originals, args.generate_qp, args.generate_qp,
                i_net, p_net, device, args.reset_interval)
            decoded = decode_dcvc_stream(stream, 17, i_net, p_net, device)

        measurements = {}
        for geometry_index, (name, representative) in enumerate(
                GEOMETRY_REPRESENTATIVES.items()):
            x, y, width, height = roi_bounds(representative)
            crop = [
                frame[y:y + height, x:x + width].copy()
                for frame in decoded
            ]
            _, runtime = seedvr2.restore(
                crop,
                int(record["seed"]) + (geometry_index + 1) * 100003,
                processing_height=height,
                processing_width=width,
                output_height=height,
                output_width=width,
            )
            measurements[name] = {
                **runtime,
                "representative_region": representative,
                "crop": {"x": x, "y": y, "width": width, "height": height},
                "processing_pixels_per_frame": width * height,
            }

        regions = []
        for region_index in range(16):
            x, y, width, height = roi_bounds(region_index)
            name = geometry_name(region_index)
            measured = measurements[name]
            regions.append({
                "index": region_index,
                "geometry_class": name,
                "crop": {"x": x, "y": y, "width": width, "height": height},
                "targets": {
                    "generate_roi_seconds_measured_geometry_class": measured[
                        "seconds_model_load_excluded"],
                    "generate_roi_processing_pixels_per_frame": width * height,
                    "generate_roi_context_pixels": ROI_CONTEXT,
                    "generate_roi_independent_component_count": 1,
                    "generate_roi_fragment_boundary_edges": 4,
                },
            })

        result = {
            "experiment": "A800 per-sample actual SeedVR2 ROI cost label",
            "protocol_version": PROTOCOL_VERSION,
            "sample": record,
            "measurements": measurements,
            "regions": regions,
            "runtime": {
                "sample_total_seconds": time.perf_counter() - sample_started,
                "generate_qp8_encode_seconds": encode_stats["seconds"],
                "seedvr2_model_load_seconds_process_level": (
                    seedvr2.model_load_seconds),
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)),
                "disk_percent_before_sample": usage,
            },
            "scientific_boundary": {
                "actual_roi_compute_measured": True,
                "same_geometry_regions_share_one_sample_specific_measurement": True,
                "measurement_input_is_fresh_qp8_decode": True,
                "source_rgb_visible_to_decoder": False,
                "training_or_finetuning": False,
                "area_prorated_from_full_frame": False,
            },
        }
        atomic_json(output_path, result)
        completed_this_process += 1
        print(json.dumps({
            "stage": "roi-cost-sample-complete",
            "sample_id": entry["sample_id"],
            "completed": entry_index + 1,
            "requested": len(entries),
            "sample_seconds": result["runtime"]["sample_total_seconds"],
            "geometry_seconds": {
                name: value["seconds_model_load_excluded"]
                for name, value in measurements.items()
            },
            "peak_cuda_mib": result["runtime"]["peak_cuda_allocated_bytes"] / 2**20,
            "disk_percent": usage,
        }), flush=True)
        if completed_this_process == 1 or completed_this_process % 10 == 0:
            build_manifest(
                args.output_dir, entries, len(entries), process_started,
                seedvr2.model_load_seconds)

    manifest = build_manifest(
        args.output_dir, entries, len(entries), process_started,
        seedvr2.model_load_seconds)
    print(json.dumps({
        "manifest": str(args.output_dir / "manifest.json"),
        "requested": len(entries),
        "completed": manifest["completed_sample_count"],
        "complete": manifest["complete"],
        "completed_this_process": completed_this_process,
        "elapsed_seconds": time.perf_counter() - process_started,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("ROI cost teacher supports one GPU only")
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
