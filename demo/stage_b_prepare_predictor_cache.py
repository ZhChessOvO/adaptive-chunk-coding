#!/usr/bin/env python3
"""Prepare the frozen-codec REDS cache for the Stage-B G2 predictor.

Only REDS train_sharp/001..024 are eligible.  Each cached sample contains
decoder inputs and a separate supervised target.  The codec trajectory is
mean-fill, never all-Base or source-frame teacher forcing.  Route masks and
64-aligned crops are deterministic functions of metadata and do not inspect
source pixels or omitted latents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage1_multichunk_oracle import (  # noqa: E402
    decode_i_stream,
    initialize_p_state,
    load_models,
    make_chunk,
    model_frame,
    prepare_chunk_latents,
    should_reset,
)
from demo.stage1_token_skipping import (  # noqa: E402
    decode_y,
    encode_y,
    expand_keep_mask,
    y_prior_steps,
)
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


TRAIN_SEQUENCE_IDS = tuple(f"{index:03d}" for index in range(1, 25))
WINDOW_STARTS = (0, 24, 48, 72)
SKIP_COUNTS = (32, 64, 96)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare active-only predictor cache from REDS train/001..024.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--cache-dir",
        default="data/REDS/cache_hts_qp32_b2_g2_v1")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-p", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--latent-block-size", type=int, choices=(2,), default=2)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--max-sequences", type=int, default=0,
        help="Debug-only prefix length; 0 prepares the registered 24 sequences.")
    return parser.parse_args()


def validate_args(args):
    if args.qp_i != 32 or args.qp_p != 32:
        raise ValueError("G2 cache is frozen at I/P QP32")
    if args.width != 512 or args.height != 512 or args.latent_block_size != 2:
        raise ValueError("G2 cache is frozen at 512x512 and 2x2 latent blocks")
    if args.max_sequences < 0 or args.max_sequences > len(TRAIN_SEQUENCE_IDS):
        raise ValueError("max-sequences must be in [0, 24]")


def stable_u64(*parts):
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def crop_xy(seed, sequence, start):
    rng = np.random.default_rng(stable_u64(seed, "crop", sequence, start))
    x_choices = np.arange(0, 1280 - 512 + 1, 64)
    y_choices = np.arange(0, 720 - 512 + 1, 64)
    return int(rng.choice(x_choices)), int(rng.choice(y_choices))


def skip_count(seed, sequence, start, chunk_index):
    index = stable_u64(seed, "count", sequence, start, chunk_index) % len(SKIP_COUNTS)
    return SKIP_COUNTS[index]


def route_mask(seed, sequence, start, chunk_index, grid_shape, count):
    rng = np.random.default_rng(
        stable_u64(seed, "route", sequence, start, chunk_index, count))
    flat = np.zeros(int(np.prod(grid_shape)), dtype=np.bool_)
    flat[rng.choice(flat.size, count, replace=False)] = True
    return flat.reshape(grid_shape)


def load_window(root, sequence, start, crop_x, crop_y):
    source = root / "train_sharp" / sequence
    paths = sorted(source.glob("*.png"))[start:start + 17]
    if len(paths) != 17:
        raise ValueError(f"{source} start={start} does not provide 17 frames")
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[crop_y:crop_y + 512, crop_x:crop_x + 512]
        if crop.shape != (512, 512, 3):
            raise ValueError(f"invalid crop for {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return paths, frames


@torch.inference_mode()
def encode_i(i_net, frame, qp, device):
    source = (model_frame(frame, device) - 0.5).to(memory_format=torch.channels_last)
    encoded = i_net.compress(source, qp, 0, 0)
    stream = bytes(encoded["bit_stream"])
    ec = int(encoded["ec_parallel"])
    return decode_i_stream(i_net, stream, ec, qp, 512, 512)


@torch.inference_mode()
def prepare_window(i_net, p_net, data_root, sequence, start, args, device):
    crop_x, crop_y = crop_xy(args.seed, sequence, start)
    paths, frames = load_window(data_root, sequence, start, crop_x, crop_y)
    i_hat = encode_i(i_net, frames[0], args.qp_i, device)
    initialize_p_state(p_net, i_hat)
    samples = []
    first_index = 1
    for chunk_index in range(2):
        chunk, _, valid_count = make_chunk(frames, first_index, device)
        if valid_count != g_frame_delay:
            raise RuntimeError("cache window contains an incomplete P chunk")
        prepared = prepare_chunk_latents(p_net, chunk, args.qp_p)
        grid_shape = (
            prepared["y"].shape[-2] // args.latent_block_size,
            prepared["y"].shape[-1] // args.latent_block_size)
        count = skip_count(args.seed, sequence, start, chunk_index)
        skip = route_mask(
            args.seed, sequence, start, chunk_index, grid_shape, count)
        encoded = encode_y(
            p_net, prepared["y"], prepared["common_params"],
            skip, args.latent_block_size)
        decoded_q, mean_y = decode_y(
            p_net, encoded.stream, encoded.ec_parallel,
            prepared["common_params"], skip, args.latent_block_size)
        if not torch.equal(decoded_q, encoded.q_dense):
            raise RuntimeError("cache Base y rANS round trip mismatch")
        keep = expand_keep_mask(
            skip, args.latent_block_size, decoded_q.shape, decoded_q.device)
        routed_q, q_source, _, _ = y_prior_steps(
            p_net, prepared["common_params"], keep, source_y=prepared["y"])
        if not torch.equal(routed_q, decoded_q):
            raise RuntimeError("cache target coordinate differs from decoded Base q")
        _, q_dec, _, _ = p_net.separate_prior_video(prepared["common_params"])
        target_delta = q_source * (~keep) * q_dec
        samples.append({
            "decoded_q": decoded_q.detach().to(dtype=torch.int8).cpu().contiguous(),
            "mean_y": mean_y.detach().cpu().contiguous(),
            "common_params": prepared["common_params"].detach().cpu().contiguous(),
            "target_delta": target_delta.detach().cpu().contiguous(),
            "skip_blocks": torch.from_numpy(skip.astype(np.uint8)),
            "metadata": {
                "sequence": sequence,
                "window_start": start,
                "chunk_index": chunk_index,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "skip_blocks": count,
                "first_source_file": str(paths[0]),
                "last_source_file": str(paths[-1]),
                "trajectory": "mean_fill",
            },
        })
        x_hat, feature = p_net.get_recon_and_feature(
            mean_y, p_net.ctx, prepared["q_decoder"])
        p_net.set_ref_feature(feature, should_reset(chunk_index, args.reset_interval))
        first_index += valid_count
    return samples


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    sample_dir = cache_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sequences = TRAIN_SEQUENCE_IDS[
        :args.max_sequences if args.max_sequences else len(TRAIN_SEQUENCE_IDS)]

    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    manifest = []
    for sequence in sequences:
        for start in WINDOW_STARTS:
            for sample in prepare_window(
                    i_net, p_net, data_root, sequence, start, args, device):
                metadata = sample.pop("metadata")
                filename = (
                    f"train_{sequence}_s{start:02d}_c{metadata['chunk_index']}.safetensors")
                path = sample_dir / filename
                save_file(sample, path, metadata={
                    key: str(value) for key, value in metadata.items()
                })
                entry = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    **metadata,
                }
                manifest.append(entry)
                print(json.dumps(entry), flush=True)

    manifest_path = cache_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item) + "\n" for item in manifest), encoding="utf-8")
    summary = {
        "format": "adaptive_chunk_coding_predictor_cache_v1",
        "split": "REDS train_sharp/001..024",
        "debug_prefix_only": bool(args.max_sequences),
        "sequences": list(sequences),
        "window_starts": list(WINDOW_STARTS),
        "chunks_per_window": 2,
        "sample_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "qp_i": args.qp_i,
        "qp_p": args.qp_p,
        "crop_size": [args.width, args.height],
        "crop_alignment": 64,
        "route_policy": {
            "source_independent": True,
            "skip_block_counts": list(SKIP_COUNTS),
            "seed": args.seed,
        },
        "trajectory": "mean_fill",
        "predictor_input_keys": [
            "decoded_q", "mean_y", "common_params", "skip_blocks"],
        "training_label_keys": ["target_delta"],
        "manifest": str(manifest_path),
    }
    (cache_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
