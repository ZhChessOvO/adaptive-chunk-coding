#!/usr/bin/env python3
"""Build the train-only Phase-B rollout-refresh cache.

For each registered REDS train/001..024 17-frame window, chunk 0 is encoded
with the original content-independent route, reconstructed by the frozen
Phase-A predictor, and propagated through the frozen HT-S codec.  Only the
resulting predictor-trajectory chunk-1 tensors are saved.  Crop coordinates
and both route masks are loaded from the original mean-fill cache; no
validation or sealed sequence is discovered or read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
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
from demo.stage_b_masked_predictor import (  # noqa: E402
    PredictorConfig,
    SparseMaskedLatentPredictor,
)
from demo.stage_b_train_predictor_g2 import (  # noqa: E402
    ELIGIBLE_SEQUENCES,
    seed_after_torch_setup,
    sha256_file,
    state_sha256,
    validate_cache,
)
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


REFRESH_TRAJECTORY = "phase_a_predictor_chunk0_rollout"
PHASE_A_PROFILE = "stage-b-g2-train-only-sparse-masked-predictor"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refresh train chunk-1 cache along the Phase-A predictor trajectory.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--mean-cache",
        default="data/REDS/cache_hts_qp32_b2_g2_v1")
    parser.add_argument(
        "--phase-a-checkpoint",
        default="output/stage_b_predictor_g2_train_seed20260908/best.pt")
    parser.add_argument(
        "--output-dir",
        default="data/REDS/cache_hts_qp32_b2_g2_phase_b_refresh_v1")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-p", type=int, default=32)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument(
        "--max-sequences", type=int, default=0,
        help="Smoke-only prefix; zero writes the registered 24-sequence cache.")
    return parser.parse_args()


def validate_args(args):
    if args.qp_i != 32 or args.qp_p != 32:
        raise ValueError("rollout refresh is frozen at I/P QP32")
    if args.seed != 20260908:
        raise ValueError("rollout refresh uses the registered seed 20260908")
    if not 0 <= args.max_sequences <= len(ELIGIBLE_SEQUENCES):
        raise ValueError("max-sequences must be in [0, 24]")


def ensure_new_output_dir(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty refresh cache: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "samples").mkdir(exist_ok=True)


def load_phase_a_checkpoint(path: Path, mean_manifest_hash: str, device):
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_hash = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    profile = str(payload.get("profile", ""))
    if profile != PHASE_A_PROFILE or "g1" in profile.lower():
        raise ValueError("refresh requires a Phase-A G2 checkpoint, never a G1 checkpoint")
    scope = payload.get("scientific_scope", {})
    data = payload.get("data", {})
    if not scope.get("train_only") or scope.get("validation_or_sealed_data_read"):
        raise ValueError("Phase-A checkpoint does not carry train-only provenance")
    if data.get("split") != "REDS train_sharp/001..024":
        raise ValueError("Phase-A checkpoint has the wrong training split")
    if data.get("manifest_sha256") != mean_manifest_hash:
        raise ValueError("Phase-A checkpoint is not bound to the supplied mean cache")
    architecture = payload.get("architecture")
    predictor = SparseMaskedLatentPredictor(
        PredictorConfig(**architecture)).to(device)
    predictor.load_state_dict(payload["state_dict"], strict=True)
    predictor.eval().requires_grad_(False)
    return predictor, payload, {
        "path": str(path),
        "file_sha256": checkpoint_hash,
        "model_state_sha256": state_sha256(predictor),
        "profile": profile,
        "step": payload.get("training", {}).get("step"),
        "architecture": architecture,
    }


def entry_key(entry):
    return str(entry["sequence"]), int(entry["window_start"]), int(entry["chunk_index"])


def pair_mean_entries(entries):
    indexed = {}
    for entry in entries:
        key = entry_key(entry)
        if key in indexed:
            raise ValueError(f"duplicate mean-cache key: {key}")
        indexed[key] = entry
    expected = {
        (sequence, start, chunk)
        for sequence in ELIGIBLE_SEQUENCES
        for start in (0, 24, 48, 72)
        for chunk in (0, 1)
    }
    if set(indexed) != expected:
        raise ValueError("mean cache does not contain the registered window/chunk grid")
    pairs = []
    for sequence in ELIGIBLE_SEQUENCES:
        for start in (0, 24, 48, 72):
            first = indexed[(sequence, start, 0)]
            second = indexed[(sequence, start, 1)]
            for key in ("crop_x", "crop_y", "first_source_file", "last_source_file"):
                if first.get(key) != second.get(key):
                    raise ValueError(f"mean-cache window metadata mismatch for {sequence}/{start}")
            pairs.append((first, second))
    return pairs


def load_cached_tensors(entry):
    path = Path(entry["path"])
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("sequence") != entry["sequence"]:
            raise ValueError(f"cache metadata mismatch in {path}")
        return {key: handle.get_tensor(key) for key in handle.keys()}


def load_window(data_root: Path, first_entry):
    sequence = str(first_entry["sequence"])
    if sequence not in ELIGIBLE_SEQUENCES:
        raise ValueError("refresh attempted to leave train/001..024")
    start = int(first_entry["window_start"])
    source = data_root / "train_sharp" / sequence
    paths = sorted(source.glob("*.png"))[start:start + 17]
    if len(paths) != 17:
        raise ValueError(f"{source} does not provide the registered 17-frame window")
    crop_x = int(first_entry["crop_x"])
    crop_y = int(first_entry["crop_y"])
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[crop_y:crop_y + 512, crop_x:crop_x + 512]
        if crop.shape != (512, 512, 3):
            raise ValueError(f"invalid registered crop for {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    expected_first = Path(str(first_entry["first_source_file"])).name
    expected_last = Path(str(first_entry["last_source_file"])).name
    if paths[0].name != expected_first or paths[-1].name != expected_last:
        raise ValueError("source window differs from the mean-cache manifest")
    return paths, frames


@torch.inference_mode()
def encode_i(i_net, frame, qp, device):
    source = (model_frame(frame, device) - 0.5).to(memory_format=torch.channels_last)
    encoded = i_net.compress(source, qp, 0, 0)
    stream = bytes(encoded["bit_stream"])
    return decode_i_stream(
        i_net, stream, int(encoded["ec_parallel"]), qp, 512, 512)


def assert_matches_mean_chunk0(cached, decoded_q, mean_y, common_params):
    comparisons = {
        "decoded_q": decoded_q.detach().to(dtype=torch.int8).cpu(),
        "mean_y": mean_y.detach().cpu(),
        "common_params": common_params.detach().cpu(),
    }
    for key, actual in comparisons.items():
        if not torch.equal(actual, cached[key]):
            raise RuntimeError(f"recomputed chunk0 differs from mean cache: {key}")


def training_target(p_net, prepared, decoded_q, skip):
    keep = expand_keep_mask(skip, 2, decoded_q.shape, decoded_q.device)
    routed_q, q_source, _, _ = y_prior_steps(
        p_net, prepared["common_params"], keep, source_y=prepared["y"])
    if not torch.equal(routed_q, decoded_q):
        raise RuntimeError("refresh target coordinate differs from decoded Base q")
    _, q_dec, _, _ = p_net.separate_prior_video(prepared["common_params"])
    return q_source * (~keep) * q_dec


@torch.inference_mode()
def refresh_window(i_net, p_net, predictor, data_root, pair, args, device):
    chunk0_entry, chunk1_entry = pair
    paths, frames = load_window(data_root, chunk0_entry)
    i_hat = encode_i(i_net, frames[0], args.qp_i, device)
    initialize_p_state(p_net, i_hat)
    first_index = 1
    output = None
    chunk0_base_invariant = None
    for chunk_index, entry in enumerate((chunk0_entry, chunk1_entry)):
        cached = load_cached_tensors(entry)
        skip = cached["skip_blocks"].numpy().astype(np.bool_)
        if int(skip.sum()) != int(entry["skip_blocks"]):
            raise ValueError("cached route cardinality differs from manifest")
        chunk, _, valid_count = make_chunk(frames, first_index, device)
        if valid_count != g_frame_delay:
            raise RuntimeError("refresh requires two complete P chunks")
        prepared = prepare_chunk_latents(p_net, chunk, args.qp_p)
        encoded = encode_y(p_net, prepared["y"], prepared["common_params"], skip, 2)
        decoded_q, mean_y = decode_y(
            p_net, encoded.stream, encoded.ec_parallel,
            prepared["common_params"], skip, 2)
        if not torch.equal(decoded_q, encoded.q_dense):
            raise RuntimeError("refresh Base y rANS round trip failed")

        if chunk_index == 0:
            assert_matches_mean_chunk0(
                cached, decoded_q, mean_y, prepared["common_params"])
            reconstructed_y, profile = predictor.apply(
                decoded_q, mean_y, prepared["common_params"], skip)
            reconstructed_y = reconstructed_y.to(dtype=mean_y.dtype)
            keep = expand_keep_mask(skip, 2, mean_y.shape, mean_y.device)
            chunk0_base_invariant = torch.equal(
                reconstructed_y[keep], mean_y[keep])
            if not chunk0_base_invariant:
                raise RuntimeError("Phase-A predictor changed transmitted Base latent")
            x_hat, feature = p_net.get_recon_and_feature(
                reconstructed_y, p_net.ctx, prepared["q_decoder"])
            p_net.set_ref_feature(
                feature, should_reset(chunk_index, args.reset_interval))
        else:
            target_delta = training_target(p_net, prepared, decoded_q, skip)
            output = {
                "decoded_q": decoded_q.detach().to(dtype=torch.int8).cpu().contiguous(),
                "mean_y": mean_y.detach().cpu().contiguous(),
                "common_params": prepared["common_params"].detach().cpu().contiguous(),
                "target_delta": target_delta.detach().cpu().contiguous(),
                "skip_blocks": torch.from_numpy(skip.astype(np.uint8)),
                "metadata": {
                    "sequence": str(entry["sequence"]),
                    "window_start": int(entry["window_start"]),
                    "chunk_index": 1,
                    "crop_x": int(entry["crop_x"]),
                    "crop_y": int(entry["crop_y"]),
                    "skip_blocks": int(entry["skip_blocks"]),
                    "first_source_file": str(paths[0]),
                    "last_source_file": str(paths[-1]),
                    "trajectory": REFRESH_TRAJECTORY,
                    "chunk0_predictor_skipped_blocks": profile["skipped_blocks"],
                    "chunk0_base_invariant": chunk0_base_invariant,
                },
            }
        first_index += valid_count
    if output is None:
        raise RuntimeError("refresh window did not produce chunk1")
    return output


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("rollout refresh requires CUDA")
    start_time = time.perf_counter()
    data_root = Path(args.data_root).resolve()
    mean_cache = Path(args.mean_cache).resolve()
    checkpoint_path = Path(args.phase_a_checkpoint).resolve()
    output_dir = Path(args.output_dir)
    ensure_new_output_dir(output_dir)

    mean_summary, mean_entries, mean_summary_path, mean_manifest_path = validate_cache(
        mean_cache)
    pairs = pair_mean_entries(mean_entries)
    selected_sequences = ELIGIBLE_SEQUENCES[
        :args.max_sequences if args.max_sequences else len(ELIGIBLE_SEQUENCES)]
    pairs = [pair for pair in pairs if pair[0]["sequence"] in selected_sequences]

    set_torch_env()
    seed_after_torch_setup(args.seed)
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    predictor, _, checkpoint_record = load_phase_a_checkpoint(
        checkpoint_path, sha256_file(mean_manifest_path), device)
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(device)

    manifest = []
    for pair in pairs:
        sample = refresh_window(
            i_net, p_net, predictor, data_root, pair, args, device)
        metadata = sample.pop("metadata")
        metadata.update({
            "phase_a_checkpoint_sha256": checkpoint_record["file_sha256"],
            "phase_a_model_state_sha256": checkpoint_record["model_state_sha256"],
            "source_mean_manifest_sha256": sha256_file(mean_manifest_path),
        })
        filename = (
            f"train_{metadata['sequence']}_s{metadata['window_start']:02d}_c1."
            "safetensors")
        path = output_dir / "samples" / filename
        save_file(sample, path, metadata={
            key: str(value) for key, value in metadata.items()
        })
        entry = {"path": str(path.resolve()), "bytes": path.stat().st_size, **metadata}
        manifest.append(entry)
        print(json.dumps(entry), flush=True)

    manifest_path = output_dir / "manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    temporary_manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in manifest), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    summary = {
        "format": "adaptive_chunk_coding_predictor_refresh_cache_v1",
        "split": "REDS train_sharp/001..024",
        "debug_prefix_only": bool(args.max_sequences),
        "sequences": list(selected_sequences),
        "window_starts": [0, 24, 48, 72],
        "cached_chunk_indices": [1],
        "sample_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "qp_i": args.qp_i,
        "qp_p": args.qp_p,
        "crop_size": [512, 512],
        "route_source": "original mean-fill cache",
        "trajectory": REFRESH_TRAJECTORY,
        "source_mean_cache": {
            "path": str(mean_cache),
            "format": mean_summary["format"],
            "summary_sha256": sha256_file(mean_summary_path),
            "manifest_sha256": sha256_file(mean_manifest_path),
        },
        "phase_a_checkpoint": checkpoint_record,
        "predictor_input_keys": [
            "decoded_q", "mean_y", "common_params", "skip_blocks"],
        "training_label_keys": ["target_delta"],
        "scientific_scope": {
            "train_only": True,
            "validation_or_sealed_data_read": False,
            "chunk0_saved": False,
            "chunk1_saved": True,
        },
        "compute": {
            "wall_seconds": time.perf_counter() - start_time,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "manifest": str(manifest_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
