#!/usr/bin/env python3
"""Isolated complete-process fresh decode for one scalar A800 baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_restorer_gate import fresh_decode_and_restore
from demo.stage_c_three_path_roi_probe import (
    BasicVSRPPRestorer,
    fresh_decode_baseline,
    load_codecs,
    save_frames,
)
from src.utils.common import set_torch_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument(
        "--restorer", choices=("none", "basicvsrpp"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-frames-dir", type=Path, required=True)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--model-path-i", type=Path,
        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument(
        "--model-path-p", type=Path,
        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument(
        "--basicvsrpp-checkpoint", type=Path,
        default=Path(
            "third_party/mmagic/checkpoints/"
            "basicvsr_plusplus_c128n25_ntire_decompress_track1_"
            "20210223-7b2eba02.pth"))
    parser.add_argument("--basicvsrpp-spatial-tile", type=int, default=512)
    args = parser.parse_args()
    if args.frame_count < 1 or args.decode_repeats < 1:
        parser.error("frame count and repeats must be positive")
    return args


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def initialize_intra_decoder_proxy(i_net) -> None:
    """Initialize the released CUDA decoder without running an encoder.

    Upstream creates the DMCI proxy lazily in ``compress`` but assumes it
    already exists in ``decompress``.  A genuinely fresh decoder process has
    no source RGB to perform a warm-up encode, so construct the same proxy
    directly from the frozen state and entropy CDFs.
    """

    from inference_extensions_cuda import DMCIProxy

    state_dict = i_net.add_cdf_to_state_dict(i_net.state_dict())
    i_net.proxy = DMCIProxy()
    i_net.proxy.set_param(state_dict, i_net.gaussian_encoder.skip_thres)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("formal fresh decode requires CUDA")
    if not args.stream.is_file():
        raise FileNotFoundError(args.stream)
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(codec_stream):
        i_net, p_net = load_codecs(args, device)
        initialize_intra_decoder_proxy(i_net)
    torch.cuda.synchronize(device)
    restorer = None
    if args.restorer == "basicvsrpp":
        restorer = BasicVSRPPRestorer(
            args.basicvsrpp_checkpoint, device, args.basicvsrpp_spatial_tile)
    setup_seconds = time.perf_counter() - process_started
    if restorer is None:
        frames, runtime = fresh_decode_baseline(
            args.stream, args.frame_count, args.decode_repeats,
            i_net, p_net, device, codec_stream)
    else:
        frames, runtime = fresh_decode_and_restore(
            args.stream, args.frame_count, args.decode_repeats,
            restorer, i_net, p_net, device, codec_stream)
    save_frames(args.output_frames_dir, frames)
    result = {
        "experiment": "A800 isolated scalar fresh decode",
        "stream": str(args.stream.resolve()),
        "actual_on_disk_bytes": args.stream.stat().st_size,
        "frame_count": args.frame_count,
        "restorer": args.restorer,
        "setup_seconds": setup_seconds,
        "warm_runtime": runtime,
        "complete_process_seconds_after_argument_parse": (
            time.perf_counter() - process_started),
        "peak_cuda_allocated_bytes": runtime["peak_cuda_allocated_bytes"],
        "fresh_decode_frames": str(args.output_frames_dir.resolve()),
        "protocol": {
            "independent_process": True,
            "file_read_and_complete_decode_timed": True,
            "model_load_in_complete_process_time": True,
            "model_load_in_warm_runtime": False,
            "source_rgb_available": False,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
