#!/usr/bin/env python3
"""E20: uniform-quality DCVC-UF restoration gate.

This is deliberately narrower than the three-path controller experiment.  It
first writes and fresh-decodes ordinary DCVC-UF streams at a small set of
quality indexes, then applies one deterministic full-frame video restorer to
the decoded RGB frames.  The resulting folders are also the inputs for the
separate SeedVR2 diffusion bridge.

No source frame, omitted latent, or oracle map is available to either decoder
or restorer.  DCVC-UF file size is charged from the actual file on disk; the
restorer sends no additional bits.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_three_path_roi_probe import (
    BasicVSRPPRestorer,
    LPIPSAlex,
    dcvc_stream_breakdown,
    decode_dcvc_stream,
    encode_dcvc_stream,
    evaluate_variant,
    fresh_decode_baseline,
    load_codecs,
    load_source_frames,
    save_frames,
)
from src.utils.common import set_torch_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E20 DCVC-UF uniform-QP restoration gate")
    parser.add_argument("--sequence-name", default="jockey")
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("data/test_sequences/PNG/jockey"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/e20_restorer_gate_jockey"))
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=0)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument(
        "--qps", type=int, nargs="+", default=(8, 16, 24, 32),
        help="DCVC-UF quality indexes; larger means higher quality/bitrate")
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument(
        "--basicvsrpp-checkpoint", type=Path,
        default=Path(
            "third_party/mmagic/checkpoints/"
            "basicvsr_plusplus_c128n25_ntire_decompress_track1_"
            "20210223-7b2eba02.pth"))
    parser.add_argument("--basicvsrpp-spatial-tile", type=int, default=512)
    parser.add_argument(
        "--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-frame", type=int, default=9)
    args = parser.parse_args()

    args.qps = sorted(set(args.qps))
    if not args.qps or any(not 0 <= value < 64 for value in args.qps):
        parser.error("--qps must contain DCVC-UF indexes in [0, 63]")
    if args.frame_count < 1 or args.decode_repeats < 1:
        parser.error("frame count and decode repeats must be positive")
    if not 1 <= args.visual_frame <= args.frame_count:
        parser.error("--visual-frame must be inside the evaluated clip")
    return args


def fresh_decode_and_restore(
    path: Path,
    frame_count: int,
    repeats: int,
    restorer: BasicVSRPPRestorer,
    i_net,
    p_net,
    device: torch.device,
    codec_stream: torch.cuda.Stream,
) -> tuple[list[np.ndarray], dict]:
    times = []
    codec_times = []
    restorer_times = []
    peaks = []
    canonical = None
    for _ in range(repeats):
        torch.cuda.set_stream(codec_stream)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        data = path.read_bytes()
        codec_started = time.perf_counter()
        decoded = decode_dcvc_stream(data, frame_count, i_net, p_net, device)
        torch.cuda.synchronize(device)
        codec_seconds = time.perf_counter() - codec_started

        torch.cuda.set_stream(torch.cuda.default_stream(device))
        restorer_started = time.perf_counter()
        restored = restorer.restore(decoded, [], [])
        torch.cuda.synchronize(device)
        restorer_seconds = time.perf_counter() - restorer_started
        times.append(time.perf_counter() - started)
        codec_times.append(codec_seconds)
        restorer_times.append(restorer_seconds)
        peaks.append(int(torch.cuda.max_memory_allocated(device)))
        if canonical is None:
            canonical = restored
        elif any(
            not np.array_equal(first, second)
            for first, second in zip(canonical, restored)
        ):
            raise RuntimeError("deterministic restorer changed across repeats")
    return canonical, {
        "fresh_decode_repeats": repeats,
        "fresh_decode_seconds_median": float(statistics.median(times)),
        "fresh_decode_seconds_all": times,
        "codec_decode_seconds_median": float(statistics.median(codec_times)),
        "restorer_seconds_median": float(statistics.median(restorer_times)),
        "peak_cuda_allocated_bytes": max(peaks),
        "file_io_in_timing": True,
        "model_load_in_timing": False,
        "source_rgb_available_to_decoder": False,
    }


def label_panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(frame)
    banner_height = 54
    panel = Image.new("RGB", (image.width, image.height + banner_height), "white")
    panel.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 7), title, fill="black", font=font)
    draw.text((8, 28), subtitle, fill="black", font=font)
    return panel


def make_visual(
    path: Path,
    originals: list[np.ndarray],
    decoded: dict[int, list[np.ndarray]],
    restored: dict[int, list[np.ndarray]],
    records: dict[str, dict],
    qps: list[int],
    frame_index: int,
) -> None:
    panels = [label_panel(originals[frame_index], "GT", "source RGB")]
    for qp in qps:
        for kind, frames in (("Base", decoded[qp]), ("BasicVSR++", restored[qp])):
            quality = records[f"qp{qp}-{kind.lower().replace('+', 'p')}"]["quality"]
            panels.append(label_panel(
                frames[frame_index], f"QP{qp} {kind}",
                f"PSNR {quality['psnr_db']:.3f} dB | LPIPS {quality['lpips_alex']:.4f}"))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (columns * width, rows * height), (230, 230, 230))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_csv(path: Path, records: dict[str, dict]) -> None:
    rows = []
    for name, record in records.items():
        quality = record["quality"]
        runtime = record["runtime"]
        rows.append({
            "variant": name,
            "qp": record["qp"],
            "restorer": record["restorer"],
            "file_bytes": record["rate"]["total_bytes"],
            "bpp": record["rate"]["bpp"],
            "psnr_db": quality["psnr_db"],
            "lpips_alex": quality["lpips_alex"],
            "temporal_delta_mae": quality["temporal_delta_mae"],
            "fresh_decode_seconds": runtime["fresh_decode_seconds_median"],
            "peak_cuda_allocated_mib": (
                runtime["peak_cuda_allocated_bytes"] / 1048576),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E20 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(codec_stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths, originals = load_source_frames(args)
    i_net, p_net = load_codecs(args, device)
    lpips_metric = LPIPSAlex(args.lpips)
    pixel_count = len(originals) * args.width * args.height
    records = {}
    decoded_by_qp = {}
    restored_by_qp = {}

    for qp in args.qps:
        stream, encode_stats = encode_dcvc_stream(
            originals, qp, qp, i_net, p_net, device, args.reset_interval)
        stream_path = args.output_dir / "streams" / f"all_base_qp{qp}.dcvc"
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_bytes(stream)
        decoded, runtime = fresh_decode_baseline(
            stream_path, len(originals), args.decode_repeats,
            i_net, p_net, device, codec_stream)
        rate = dcvc_stream_breakdown(stream_path.read_bytes(), len(originals))
        if rate["total_bytes"] != stream_path.stat().st_size:
            raise RuntimeError("bitstream accounting differs from file size")
        rate["bpp"] = 8.0 * rate["total_bytes"] / pixel_count
        records[f"qp{qp}-base"] = {
            "qp": qp,
            "restorer": "none",
            "path": str(stream_path),
            "rate": rate,
            "quality": evaluate_variant(originals, decoded, lpips_metric),
            "runtime": runtime,
            "encode": encode_stats,
        }
        decoded_by_qp[qp] = decoded
        if args.save_frames:
            save_frames(args.output_dir / "frames" / f"qp{qp}-base", decoded)

    restorer = BasicVSRPPRestorer(
        args.basicvsrpp_checkpoint, device, args.basicvsrpp_spatial_tile)
    for qp in args.qps:
        base_record = records[f"qp{qp}-base"]
        restored, runtime = fresh_decode_and_restore(
            Path(base_record["path"]), len(originals), args.decode_repeats,
            restorer, i_net, p_net, device, codec_stream)
        records[f"qp{qp}-basicvsrpp"] = {
            "qp": qp,
            "restorer": restorer.name,
            "path": base_record["path"],
            "rate": base_record["rate"],
            "quality": evaluate_variant(originals, restored, lpips_metric),
            "runtime": runtime,
        }
        restored_by_qp[qp] = restored
        if args.save_frames:
            save_frames(
                args.output_dir / "frames" / f"qp{qp}-basicvsrpp", restored)

    if args.save_frames:
        save_frames(args.output_dir / "frames" / "original", originals)

    comparisons = {}
    for qp in args.qps:
        base = records[f"qp{qp}-base"]["quality"]
        restored = records[f"qp{qp}-basicvsrpp"]["quality"]
        comparisons[str(qp)] = {
            "psnr_delta_db": restored["psnr_db"] - base["psnr_db"],
            "lpips_delta": (
                None if base["lpips_alex"] is None
                else restored["lpips_alex"] - base["lpips_alex"]),
            "temporal_delta_mae_change": (
                restored["temporal_delta_mae"] - base["temporal_delta_mae"]),
        }

    summary = {
        "experiment": "E20 uniform-quality DCVC-UF restoration gate",
        "status": "deterministic_control_complete_diffusion_pending",
        "sequence": args.sequence_name,
        "source_role": "previously used regression/development data",
        "source_files": [str(path) for path in source_paths],
        "crop": {
            "x": args.crop_x,
            "y": args.crop_y,
            "width": args.width,
            "height": args.height,
        },
        "frames": len(originals),
        "protocol": {
            "budget_known_before_encoding": True,
            "ordinary_stock_dcvc_uf_stream_per_qp": True,
            "actual_file_bytes_charged": True,
            "restoration_additional_bytes": 0,
            "source_or_omitted_latent_available_to_decoder": False,
            "model_load_in_decode_timing": False,
            "file_io_codec_decode_and_rgb_restoration_in_timing": True,
            "dcvc_uf_quality_index_direction": "larger is higher quality/bitrate",
        },
        "configuration": {
            "qps": args.qps,
            "decode_repeats": args.decode_repeats,
            "restorer": restorer.name,
            "reset_interval": args.reset_interval,
        },
        "variants": records,
        "deterministic_restoration_deltas": comparisons,
        "next_gate": (
            "Run the same decoded folders through full-frame SeedVR2, then "
            "compare Base, BasicVSR++, and SeedVR2 before any controller training."),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_csv(args.output_dir / "per_variant.csv", records)
    make_visual(
        args.output_dir / "visuals" / f"frame_{args.visual_frame:05d}.png",
        originals, decoded_by_qp, restored_by_qp, records, args.qps,
        args.visual_frame - 1)
    print(json.dumps({
        "summary": str(summary_path),
        "visual": str(
            args.output_dir / "visuals" / f"frame_{args.visual_frame:05d}.png"),
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
