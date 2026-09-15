#!/usr/bin/env python3
"""Benchmark each E19 saved variant in an isolated decoder process.

The parent launches one child per saved stream/container.  A child loads only
the codec contexts and restorer required by that file's action map, then times
file I/O plus complete decode/compositing.  Model loading is deliberately
outside the timed interval but its resident memory is included in the CUDA
peak, matching a deployed warm decoder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_three_path_roi_probe import (
    ACTION_ENHANCE,
    ACTION_GENERATE,
    BasicVSRPPRestorer,
    SDXLTileRestorer,
    fresh_decode_baseline,
    fresh_decode_container,
    load_codecs,
    read_three_path_container,
)
from src.utils.common import set_torch_env


class NoOpRestorer:
    name = "not-loaded-no-generate-action"
    local_compute = True

    def restore(self, frames, selected_tiles, boxes):
        if selected_tiles:
            raise RuntimeError("NoOpRestorer received Generate actions")
        return [frame.copy() for frame in frames]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-repeats", type=int)
    parser.add_argument("--restorer", choices=("basicvsrpp", "sdxl"),
                        default="basicvsrpp")
    parser.add_argument(
        "--basicvsrpp-checkpoint", type=Path,
        default=Path("third_party/mmagic/checkpoints/"
                     "basicvsr_plusplus_c128n25_ntire_decompress_track1_"
                     "20210223-7b2eba02.pth"))
    parser.add_argument("--basicvsrpp-spatial-tile", type=int, default=512)
    parser.add_argument("--sdxl-model", type=Path,
                        default=Path("checkpoints/sdxl-inpainting-0.1"))
    parser.add_argument("--sdxl-steps", type=int, default=20)
    parser.add_argument("--sdxl-guidance", type=float, default=5.0)
    parser.add_argument("--sdxl-prompt", default=(
        "high quality natural video frame, faithful realistic texture"))
    parser.add_argument("--sdxl-negative-prompt", default=(
        "text, watermark, deformation, hallucinated objects, artifacts"))
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--model-path-i", type=Path,
                        default=Path("checkpoints/cvpr2026_image.pth.tar"))
    parser.add_argument("--model-path-p", type=Path,
                        default=Path("checkpoints/cvpr2026_video_hts.pth.tar"))
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--single-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--single-kind", choices=("baseline", "container"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--frame-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_path is None and args.summary is None:
        parser.error("--summary is required")
    if args.single_path is not None and (
            args.single_kind is None or args.frame_count is None):
        parser.error("internal single mode requires kind and frame count")
    return args


def initialize_i_decoder(model) -> None:
    """Create the released CUDA proxy without encoding a source image."""
    if model.proxy is not None:
        return
    from inference_extensions_cuda import DMCIProxy

    state = model.add_cdf_to_state_dict(model.state_dict())
    model.proxy = DMCIProxy()
    model.proxy.set_param(state, model.gaussian_encoder.skip_thres)


@torch.inference_mode()
def single(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(codec_stream)
    base_i, base_p = load_codecs(args, device)
    initialize_i_decoder(base_i)

    if args.single_kind == "baseline":
        _, runtime = fresh_decode_baseline(
            args.single_path, args.frame_count, args.decode_repeats,
            base_i, base_p, device, codec_stream)
        runtime["loaded_components"] = ["base-codec"]
        return runtime

    preview = read_three_path_container(args.single_path)
    needs_tile = bool(np.any(preview["actions"] == ACTION_ENHANCE))
    needs_restorer = bool(np.any(preview["actions"] == ACTION_GENERATE))
    if needs_tile:
        tile_i, tile_p = load_codecs(args, device)
        initialize_i_decoder(tile_i)
    else:
        tile_i, tile_p = base_i, base_p

    if needs_restorer and args.restorer == "basicvsrpp":
        restorer = BasicVSRPPRestorer(
            args.basicvsrpp_checkpoint, device, args.basicvsrpp_spatial_tile)
    elif needs_restorer:
        restorer = SDXLTileRestorer(args, device)
    else:
        restorer = NoOpRestorer()

    _, runtime = fresh_decode_container(
        args.single_path, args.decode_repeats, restorer,
        base_i, base_p, tile_i, tile_p, device, codec_stream)
    components = ["base-codec"]
    if needs_tile:
        components.append("roi-codec")
    if needs_restorer:
        components.append(args.restorer)
    runtime["loaded_components"] = components
    return runtime


def child_command(args: argparse.Namespace, path: Path, kind: str,
                  frame_count: int, repeats: int) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--single-path", str(path.resolve()),
        "--single-kind", kind,
        "--frame-count", str(frame_count),
        "--decode-repeats", str(repeats),
        "--restorer", args.restorer,
        "--basicvsrpp-checkpoint", str(args.basicvsrpp_checkpoint.resolve()),
        "--basicvsrpp-spatial-tile", str(args.basicvsrpp_spatial_tile),
        "--sdxl-model", str(args.sdxl_model.resolve()),
        "--sdxl-steps", str(args.sdxl_steps),
        "--sdxl-guidance", str(args.sdxl_guidance),
        "--sdxl-prompt", args.sdxl_prompt,
        "--sdxl-negative-prompt", args.sdxl_negative_prompt,
        "--seed", str(args.seed),
        "--cuda-idx", str(args.cuda_idx),
        "--model-path-i", str(args.model_path_i.resolve()),
        "--model-path-p", str(args.model_path_p.resolve()),
        "--skip-thres", str(args.skip_thres),
    ]
    return command


def parent(args: argparse.Namespace) -> dict:
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    repeats = args.decode_repeats or source["configuration"]["decode_repeats"]
    benchmarks = {}
    for name, record in source["variants"].items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        kind = "container" if path.suffix == ".d3r" else "baseline"
        completed = subprocess.run(
            child_command(args, path, kind, source["frames"], repeats),
            cwd=REPO_ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=sys.stderr)
        runtime = json.loads(completed.stdout)
        benchmarks[name] = runtime
        print(json.dumps({
            "variant": name,
            "seconds": runtime["fresh_decode_seconds_median"],
            "peak_mib": runtime["peak_cuda_allocated_bytes"] / 1048576,
            "loaded": runtime["loaded_components"],
        }, ensure_ascii=False), flush=True)

    result = {
        "experiment": source["experiment"],
        "source_summary": str(args.summary),
        "protocol": {
            "one_new_process_per_variant": True,
            "file_io_in_timing": True,
            "model_load_in_timing": False,
            "resident_required_model_memory_in_peak": True,
            "source_rgb_available_to_decoder": False,
            "decode_repeats": repeats,
        },
        "variants": benchmarks,
    }
    output = args.output or args.summary.with_name("fresh_decode_benchmark.json")
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    benchmarked = json.loads(json.dumps(source))
    for name, runtime in benchmarks.items():
        benchmarked["variants"][name]["runtime"] = runtime
    copy_path = args.summary.with_name("summary_benchmarked.json")
    benchmarked["isolated_fresh_decode_benchmark"] = str(output)
    copy_path.write_text(
        json.dumps(benchmarked, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {"benchmark": str(output), "summary": str(copy_path)}


def main() -> None:
    args = parse_args()
    result = single(args) if args.single_path is not None else parent(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
