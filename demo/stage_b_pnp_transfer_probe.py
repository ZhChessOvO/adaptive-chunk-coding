#!/usr/bin/env python3
"""E12: frozen PnP-VCVE feature transfer probe on registered REDS train videos.

The experiment deliberately separates fitting (train/001..018) from an
internal unseen screen (train/019..024).  It compares a released PnP-VCVE
backbone with an identically shaped, deterministically random backbone.  Both
backbones stay frozen and both use the same zero-output latent head, sample
order, optimizer, and update budget.

Only decoder-visible tensors enter the predictor: mean-fill RGB frames,
decoded Base q, entropy-model common parameters, mean y, and the transmitted
route.  Source frames and true omitted latent values are labels only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage1_multichunk_oracle import (  # noqa: E402
    CHUNK_LENGTH,
    SEQ_HEADER,
    initialize_p_state,
    load_models,
    make_chunk,
    prepare_chunk_latents,
    reconstruction_float,
    should_reset,
)
from demo.stage1_token_skipping import (  # noqa: E402
    CONTAINER_HEADER,
    build_route_section,
    decode_y,
    encode_y,
)
from demo.stage_b_masked_predictor import (  # noqa: E402
    blocks_to_tensor,
    skipped_target_blocks,
    tensor_to_blocks,
)
from demo.stage_b_mse_weight_ablation import (  # noqa: E402
    SEED,
    encode_i,
    load_window,
)
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


FIT_IDS = tuple(f"{index:03d}" for index in range(1, 19))
SCREEN_IDS = tuple(f"{index:03d}" for index in range(19, 25))
ALL_IDS = FIT_IDS + SCREEN_IDS
WINDOW_STARTS = (0, 24, 48, 72)
ARMS = ("pretrained", "random")
STEPS = 5000
EVAL_EVERY = 500
QP = 32
BLOCK_SIZE = 2
K = 8
RANDOM_BACKBONE_SEED = 2026091301
HEAD_SEED = 2026091302


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--e10-dir",
        default="output/stage_b_mse_weight_ablation_train001_024_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_pnp_transfer_probe_train001_024_v1")
    parser.add_argument(
        "--pnp-root", default="third_party/PnP-VCVE")
    parser.add_argument(
        "--pnp-checkpoint",
        default="third_party/PnP-VCVE/checkpoint/HR_davis_LR_128x128.pth")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--skip-operational-screen", action="store_true",
        help="Stop after the latent screen; intended only for implementation smoke tests.")
    return parser.parse_args()


def validate_args(args):
    if args.steps != STEPS:
        raise ValueError(f"E12 fixes the paired update count at {STEPS}")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E12 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E12 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E12 fixes gradient clipping at 1.0")
    pnp_root = Path(args.pnp_root).resolve()
    if not (pnp_root / "standalone_backbone.py").is_file():
        raise FileNotFoundError(pnp_root / "standalone_backbone.py")
    if not Path(args.pnp_checkpoint).is_file():
        raise FileNotFoundError(args.pnp_checkpoint)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_pnp_module(pnp_root: Path):
    path = pnp_root / "standalone_backbone.py"
    spec = importlib.util.spec_from_file_location("e12_pnp_standalone", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_backbones(args, device):
    module = load_pnp_module(Path(args.pnp_root).resolve())
    pretrained = module.PnPFeatureBackbone()
    load_audit = pretrained.load_released_checkpoint(args.pnp_checkpoint)
    random_model = module.make_random_backbone_like(RANDOM_BACKBONE_SEED)
    models = {"pretrained": pretrained, "random": random_model}
    for model in models.values():
        model.eval().requires_grad_(False).to(device)
    if any(parameter.requires_grad for model in models.values()
           for parameter in model.parameters()):
        raise RuntimeError("a PnP backbone was not frozen")
    return models, load_audit


def e10_entries(e10_dir: Path):
    manifest = e10_dir / "train_cache" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != 192:
        raise RuntimeError(f"E12 expected 192 E10 samples, got {len(entries)}")
    expected = {
        f"train_{sequence}_s{start:02d}_c{chunk}"
        for sequence in ALL_IDS for start in WINDOW_STARTS for chunk in range(2)}
    by_key = {entry["key"]: entry for entry in entries}
    if set(by_key) != expected:
        raise RuntimeError("E10 manifest keys do not match the registered E12 split")
    return by_key


def feature_path(output_dir: Path, role: str, key: str) -> Path:
    return (output_dir / "feature_cache" / role / "samples" / f"{key}.safetensors")


def pnp_features(
        backbone, mean_rgb: torch.Tensor, device,
        move_to_cpu: bool = True) -> torch.Tensor:
    resized = F.interpolate(
        mean_rgb, size=(128, 128), mode="bilinear", align_corners=False
    ).reshape(1, g_frame_delay, 3, 128, 128)
    qps = torch.full(
        (1, g_frame_delay, 1, 1, 1), QP / 255.0,
        device=device, dtype=torch.float32)
    # The 2x2 latent route corresponds to an 8x8 area after the 4x RGB
    # downsample.  PnP's released loader scaled its one-hot partition maps by
    # 1/255, so channel 2 is the closest decoder-side fixed analogue.
    partitions = torch.zeros(
        1, g_frame_delay, 3, 128, 128, device=device, dtype=torch.float32)
    partitions[:, :, 2] = 1.0 / 255.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        _, features = backbone(resized, qps, partitions)
    pooled = F.avg_pool2d(
        features.reshape(g_frame_delay, 64, 128, 128), 4)
    pooled = pooled.detach().to(dtype=torch.float16).contiguous()
    return pooled.cpu() if move_to_cpu else pooled


def load_codec(args, device):
    class Values:
        pass
    values = Values()
    values.model_path_i = args.model_path_i
    values.model_path_p = args.model_path_p
    values.skip_thres = 0.0
    i_net, p_net = load_models(values, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    return i_net, p_net


@torch.inference_mode()
def prepare_feature_cache(args, role: str, sequence_ids, base_entries, device):
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "feature_cache" / role
    complete_path = cache_dir / "complete.json"
    manifest_path = cache_dir / "manifest.jsonl"
    expected_count = len(sequence_ids) * len(WINDOW_STARTS) * 2
    if complete_path.is_file() and manifest_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if (complete.get("sample_count") != expected_count
                or complete.get("sequence_ids") != list(sequence_ids)
                or len(entries) != expected_count
                or not all(Path(entry["feature_path"]).is_file() for entry in entries)):
            raise RuntimeError(f"incompatible or incomplete E12 {role} cache")
        return entries, complete["pnp_checkpoint_load_audit"]

    i_net, p_net = load_codec(args, device)
    backbones, load_audit = make_backbones(args, device)
    entries = []
    started = time.perf_counter()
    for sequence in sequence_ids:
        for start in WINDOW_STARTS:
            _, frames, crop = load_window(
                Path(args.data_root), "train_sharp", sequence, start)
            _, _, i_hat = encode_i(i_net, frames[0], device)
            initialize_p_state(p_net, i_hat)
            first_index = 1
            for chunk_index in range(2):
                key = f"train_{sequence}_s{start:02d}_c{chunk_index}"
                base = load_file(base_entries[key]["path"], device=str(device))
                route = base["skip_blocks"].cpu().numpy().astype(np.bool_)
                if int(route.sum()) != K:
                    raise RuntimeError(f"{key} does not use K{K}")
                chunk, _, valid_count = make_chunk(frames, first_index, device)
                prepared = prepare_chunk_latents(p_net, chunk, QP)
                routed = encode_y(
                    p_net, prepared["y"], prepared["common_params"], route,
                    BLOCK_SIZE)
                decoded_q, mean_y = decode_y(
                    p_net, routed.stream, routed.ec_parallel,
                    prepared["common_params"], route, BLOCK_SIZE)
                if (not torch.equal(decoded_q.cpu(), base["decoded_q"].cpu())
                        or not torch.equal(mean_y.cpu(), base["mean_y"].cpu())):
                    raise RuntimeError(f"E12 replay disagrees with E10 for {key}")
                mean_rgb, _, feature = reconstruction_float(
                    p_net, mean_y, prepared["q_decoder"], valid_count)
                tensors = {
                    arm: pnp_features(backbone, mean_rgb, device)
                    for arm, backbone in backbones.items()
                }
                path = feature_path(output_dir, role, key)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, path, metadata={
                    "sequence": sequence,
                    "window_start": str(start),
                    "chunk_index": str(chunk_index),
                    "source": "decoder-visible mean-fill RGB only",
                })
                entry = {
                    "key": key,
                    "sequence": sequence,
                    "window_start": start,
                    "chunk_index": chunk_index,
                    "crop_xy": list(crop),
                    "latent_path": base_entries[key]["path"],
                    "feature_path": str(path.resolve()),
                    "feature_shape": [g_frame_delay, 64, 32, 32],
                    "feature_dtype": "float16",
                    "backbone_input": "mean-fill RGB resized from 512x512 to 128x128",
                    "source_or_true_latent_in_backbone_input": False,
                }
                entries.append(entry)
                print(json.dumps({"stage": f"{role}-feature-cache", "key": key}),
                      flush=True)
                p_net.set_ref_feature(
                    feature, should_reset(chunk_index, 32))
                first_index += valid_count
    elapsed = time.perf_counter() - started
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries),
        encoding="utf-8")
    os.replace(temporary, manifest_path)
    complete = {
        "format": "e12_pnp_feature_cache_v1",
        "role": role,
        "split": f"REDS train_sharp/{sequence_ids[0]}..{sequence_ids[-1]}",
        "sequence_ids": list(sequence_ids),
        "window_starts": list(WINDOW_STARTS),
        "chunks_per_window": 2,
        "sample_count": len(entries),
        "trajectory": "fixed E10 mean-fill K8",
        "backbone_input": "decoder-visible mean-fill RGB only",
        "pnp_input_resolution": [128, 128],
        "cached_feature_resolution": [32, 32],
        "pnp_qp_condition": QP / 255.0,
        "pnp_partition_proxy": "channel-2 (8x8) fixed at 1/255",
        "pnp_motion_vectors": "zero / identity alignment",
        "pnp_checkpoint_load_audit": load_audit,
        "elapsed_seconds": elapsed,
        "sealed_or_validation_data_read": False,
    }
    atomic_json(complete_path, complete)
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return entries, load_audit


def neighbor_table(query_ids: torch.Tensor, height: int, width: int):
    offsets = torch.arange(-1, 2, device=query_ids.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    rows = torch.div(query_ids, width, rounding_mode="floor")[:, None] + dy.reshape(1, -1)
    cols = torch.remainder(query_ids, width)[:, None] + dx.reshape(1, -1)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows.clamp(0, height - 1)
    cols = cols.clamp(0, width - 1)
    return rows * width + cols, valid


def projector(inputs: int, outputs: int):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.LayerNorm(outputs), nn.GELU())


class PnPLatentHead(nn.Module):
    """Active-block head over frozen PnP features and DCVC decoder conditions."""

    def __init__(self):
        super().__init__()
        block_area = BLOCK_SIZE * BLOCK_SIZE
        self.q_projector = projector(256 * block_area, 16)
        self.common_projector = projector(768 * block_area, 64)
        self.mean_projector = projector(256 * block_area, 32)
        self.video_projector = projector(g_frame_delay * 64 * block_area, 128)
        input_width = 9 * 16 + 9 + 64 + 32 + 128
        self.trunk = nn.Sequential(
            nn.Linear(input_width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256 * block_area),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, decoded_q, mean_y, common_params, skip_blocks, video_features):
        if isinstance(skip_blocks, np.ndarray):
            route = torch.from_numpy(skip_blocks)
        elif torch.is_tensor(skip_blocks):
            route = skip_blocks
        else:
            raise TypeError("skip_blocks must be a NumPy array or Torch tensor")
        route = route.to(device=decoded_q.device, dtype=torch.bool)
        query_ids = torch.nonzero(route.reshape(-1), as_tuple=False).reshape(-1)
        if query_ids.numel() != K:
            raise ValueError(f"E12 expects exactly K{K} queries")
        grid_h, grid_w = route.shape
        neighbor_ids, valid = neighbor_table(query_ids, grid_h, grid_w)
        observed = valid & ~route.reshape(-1)[neighbor_ids]
        q_blocks = tensor_to_blocks(decoded_q, BLOCK_SIZE).float()
        mean_blocks = tensor_to_blocks(mean_y, BLOCK_SIZE).float()
        common_blocks = tensor_to_blocks(common_params, BLOCK_SIZE).float()
        video = video_features.reshape(
            1, g_frame_delay * 64, video_features.shape[-2], video_features.shape[-1])
        video_blocks = tensor_to_blocks(video, BLOCK_SIZE).float()
        q_neighbors = q_blocks[neighbor_ids] * observed[:, :, None]
        q_embedding = self.q_projector(q_neighbors / 8.0).reshape(K, -1)
        features = torch.cat((
            q_embedding,
            observed.to(dtype=torch.float32),
            self.common_projector(common_blocks[query_ids]),
            self.mean_projector(mean_blocks[query_ids]),
            self.video_projector(video_blocks[query_ids]),
        ), dim=1)
        return self.trunk(features), query_ids

    def apply(self, decoded_q, mean_y, common_params, skip_blocks, video_features):
        prediction, query_ids = self(
            decoded_q, mean_y, common_params, skip_blocks, video_features)
        blocks = mean_y.new_zeros(tensor_to_blocks(mean_y, BLOCK_SIZE).shape)
        blocks[query_ids] = prediction.to(dtype=mean_y.dtype)
        delta = blocks_to_tensor(blocks, tuple(mean_y.shape), BLOCK_SIZE)
        return mean_y + delta


def load_samples(entries, arm: str, device):
    samples = []
    for entry in entries:
        latent = load_file(entry["latent_path"], device=str(device))
        features = load_file(entry["feature_path"], device=str(device))[arm]
        samples.append({
            **latent,
            "video_features": features,
            "sequence": entry["sequence"],
            "window_start": entry["window_start"],
            "chunk_index": entry["chunk_index"],
        })
    return samples


def sample_prediction(model, sample):
    prediction, query_ids = model(
        sample["decoded_q"], sample["mean_y"], sample["common_params"],
        sample["skip_blocks"], sample["video_features"])
    target, target_ids = skipped_target_blocks(
        sample["target_delta"], sample["skip_blocks"], BLOCK_SIZE)
    if not torch.equal(query_ids, target_ids):
        raise RuntimeError("prediction and target query order differ")
    return prediction, target, query_ids


def latent_summary(model, samples, include_records=False):
    records = []
    with torch.inference_mode():
        for sample in samples:
            prediction, target, query_ids = sample_prediction(model, sample)
            rates = sample["rate_map"].reshape(-1)[query_ids].float()
            for local, query_id in enumerate(query_ids):
                target_energy = float(target[local].float().square().sum())
                error = float((prediction[local] - target[local]).float().square().sum())
                records.append({
                    "sequence": sample["sequence"],
                    "window_start": sample["window_start"],
                    "chunk_index": sample["chunk_index"],
                    "flat_id": int(query_id),
                    "saved_y_bytes": float(rates[local]),
                    "target_energy": target_energy,
                    "prediction_error": error,
                })

    def summarize(items):
        target = sum(item["target_energy"] for item in items)
        error = sum(item["prediction_error"] for item in items)
        return {
            "block_count": len(items),
            "latent_gap_recovery": 1.0 - error / max(target, 1e-30),
            "prediction_mse": error / max(len(items) * 1024, 1),
            "mean_fill_mse": target / max(len(items) * 1024, 1),
            "fraction_blocks_better_than_mean": float(np.mean([
                item["prediction_error"] < item["target_energy"] for item in items])),
        }

    ordered = sorted(
        records, key=lambda item: (-item["saved_y_bytes"], item["sequence"],
                                  item["window_start"], item["chunk_index"],
                                  item["flat_id"]))
    top_count = max(1, math.ceil(len(ordered) / 4))
    result = {
        "overall": summarize(records),
        "highest_byte_quartile": summarize(ordered[:top_count]),
        "per_sequence": {
            sequence: summarize([item for item in records
                                 if item["sequence"] == sequence])
            for sequence in sorted({item["sequence"] for item in records})
        },
        "per_chunk_position": {
            str(chunk): summarize([item for item in records
                                   if item["chunk_index"] == chunk])
            for chunk in range(2)
        },
    }
    if include_records:
        result["records"] = records
    return result


def make_schedule(sample_count: int):
    rng = np.random.default_rng(SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:STEPS]


def train_one_arm(arm, samples, initial_state, args, output_dir, device):
    checkpoint_path = output_dir / "checkpoints" / f"{arm}_head_final.pt"
    summary_path = output_dir / "training" / f"{arm}_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (payload["arm"] != arm or payload["steps"] != STEPS
                or payload["head_seed"] != HEAD_SEED):
            raise RuntimeError(f"incompatible existing E12 {arm} checkpoint")
        return checkpoint_path, json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists() or summary_path.exists():
        raise RuntimeError(f"partial E12 training output for {arm}")

    model = PnPLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = make_schedule(len(samples))
    history = [{"step": 0, **latent_summary(model, samples)["overall"]}]
    log_path = output_dir / "training" / f"{arm}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history[0]) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step, index in enumerate(schedule, 1):
        prediction, target, _ = sample_prediction(model, samples[index])
        loss = (prediction - target).float().square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"{arm} produced a non-finite gradient")
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            event = {
                "step": step,
                "last_sample_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **latent_summary(model, samples)["overall"],
            }
            history.append(event)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps({"stage": "train", "arm": arm, **event}), flush=True)
            model.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    final_fit = latent_summary(model, samples)
    summary = {
        "arm": arm,
        "steps": STEPS,
        "loss": "unweighted direct-delta MSE",
        "checkpoint_rule": "fixed final update before screen tensors are loaded",
        "head_parameters": model.parameter_count,
        "frozen_backbone_parameters": 4_559_885,
        "fit_metrics": final_fit,
        "history": history,
        "training_wall_seconds": elapsed,
        "updates_per_second": STEPS / elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    atomic_checkpoint(checkpoint_path, {
        "format": "e12_pnp_latent_head_v1",
        "arm": arm,
        "steps": STEPS,
        "head_seed": HEAD_SEED,
        "random_backbone_seed": RANDOM_BACKBONE_SEED if arm == "random" else None,
        "state_dict": {key: value.detach().cpu()
                       for key, value in model.state_dict().items()},
        "data": {
            "fit": "REDS train_sharp/001..018",
            "screen_not_loaded": True,
            "window_starts": list(WINDOW_STARTS),
            "chunks_per_window": 2,
        },
        "training": summary,
    })
    atomic_json(summary_path, summary)
    return checkpoint_path, summary


def load_head(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PnPLatentHead().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def paired_training(args, fit_entries, output_dir, device):
    seed_everything(HEAD_SEED)
    initial = PnPLatentHead().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    records = {}
    checkpoints = {}
    for arm in ARMS:
        samples = load_samples(fit_entries, arm, device)
        path, record = train_one_arm(
            arm, samples, initial_state, args, output_dir, device)
        checkpoints[arm] = path
        records[arm] = record
        del samples
        torch.cuda.empty_cache()
    del initial, initial_state
    lock = {
        "all_final_checkpoints_fixed_before_screen_tensor_load": True,
        "fit_ids": list(FIT_IDS),
        "screen_ids_not_loaded_during_training": list(SCREEN_IDS),
        "checkpoints": {arm: str(path.resolve()) for arm, path in checkpoints.items()},
        "fixed_at_unix_time": time.time(),
    }
    atomic_json(output_dir / "checkpoint_lock_before_screen.json", lock)
    return checkpoints, records


def screen_latents(checkpoints, screen_entries, output_dir, device):
    summaries = {}
    all_records = []
    for arm in ARMS:
        model = load_head(checkpoints[arm], device)
        samples = load_samples(screen_entries, arm, device)
        summary = latent_summary(model, samples, include_records=True)
        records = summary.pop("records")
        for record in records:
            all_records.append({"arm": arm, **record})
        summaries[arm] = summary
        del model, samples
        torch.cuda.empty_cache()
    diagnostics = output_dir / "screen"
    diagnostics.mkdir(parents=True, exist_ok=True)
    atomic_json(diagnostics / "latent_summary.json", summaries)
    with (diagnostics / "block_records.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    return summaries


def latent_gate(summaries):
    pretrained = summaries["pretrained"]
    random_arm = summaries["random"]
    positive_sequences = sum(
        item["latent_gap_recovery"] > 0
        for item in pretrained["per_sequence"].values())
    beats_random_sequences = sum(
        pretrained["per_sequence"][sequence]["prediction_mse"]
        < random_arm["per_sequence"][sequence]["prediction_mse"]
        for sequence in SCREEN_IDS)
    checks = {
        "pretrained_positive_on_at_least_5_of_6_sequences": positive_sequences >= 5,
        "pretrained_overall_beats_random": (
            pretrained["overall"]["prediction_mse"]
            < random_arm["overall"]["prediction_mse"]),
        "pretrained_beats_random_on_at_least_4_of_6_sequences": (
            beats_random_sequences >= 4),
        "highest_byte_quartile_recovery_positive": (
            pretrained["highest_byte_quartile"]["latent_gap_recovery"] > 0),
        "second_chunk_recovery_positive": (
            pretrained["per_chunk_position"]["1"]["latent_gap_recovery"] > 0),
        "unseen_screen_recovery_positive": (
            pretrained["overall"]["latent_gap_recovery"] > 0),
    }
    return {
        "passes": all(checks.values()),
        "positive_sequences": positive_sequences,
        "pretrained_beats_random_sequences": beats_random_sequences,
        "checks": checks,
    }


@torch.inference_mode()
def operational_screen(args, checkpoints, base_entries, output_dir, device):
    i_net, p_net = load_codec(args, device)
    backbones, _ = make_backbones(args, device)
    heads = {arm: load_head(checkpoints[arm], device) for arm in ARMS}
    rows = []
    latency = defaultdict(list)
    methods = ("mean",) + ARMS
    for sequence in SCREEN_IDS:
        for start in WINDOW_STARTS:
            _, frames, _ = load_window(
                Path(args.data_root), "train_sharp", sequence, start)
            i_stream, _, i_hat = encode_i(i_net, frames[0], device)
            for method in methods:
                initialize_p_state(p_net, i_hat)
                first_index = 1
                sum_squared_error = 0.0
                pixel_count = 0
                chunk_bytes = []
                chunk_mse = []
                for chunk_index in range(2):
                    key = f"train_{sequence}_s{start:02d}_c{chunk_index}"
                    static = load_file(base_entries[key]["path"], device="cpu")
                    route = static["skip_blocks"].numpy().astype(np.bool_)
                    chunk, target_rgb, valid_count = make_chunk(
                        frames, first_index, device)
                    prepared = prepare_chunk_latents(p_net, chunk, QP)
                    encoded = encode_y(
                        p_net, prepared["y"], prepared["common_params"], route,
                        BLOCK_SIZE)
                    decoded_q, mean_y = decode_y(
                        p_net, encoded.stream, encoded.ec_parallel,
                        prepared["common_params"], route, BLOCK_SIZE)
                    if method == "mean":
                        predicted_y = mean_y
                    else:
                        mean_rgb, _, _ = reconstruction_float(
                            p_net, mean_y, prepared["q_decoder"], valid_count)
                        torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        features = pnp_features(
                            backbones[method], mean_rgb, device,
                            move_to_cpu=False)
                        predicted_y = heads[method].apply(
                            decoded_q, mean_y, prepared["common_params"], route,
                            features)
                        torch.cuda.synchronize(device)
                        latency[method].append(1000.0 * (time.perf_counter() - tick))
                    rgb, _, ref_feature = reconstruction_float(
                        p_net, predicted_y, prepared["q_decoder"], valid_count)
                    squared = float((rgb.float() - target_rgb).square().sum())
                    count = target_rgb.numel()
                    mse = squared / count
                    sum_squared_error += squared
                    pixel_count += count
                    chunk_mse.append(mse)
                    route_section = build_route_section(route, BLOCK_SIZE)
                    chunk_bytes.append(
                        CONTAINER_HEADER.size + len(prepared["global_stream"])
                        + len(route_section) + len(encoded.stream))
                    p_net.set_ref_feature(
                        ref_feature, should_reset(chunk_index, 32))
                    first_index += valid_count
                total_bytes = (
                    SEQ_HEADER.size + len(i_stream)
                    + len(chunk_bytes) * CHUNK_LENGTH.size + sum(chunk_bytes))
                rows.append({
                    "method": method,
                    "sequence": sequence,
                    "window_start": start,
                    "p_frame_mse": sum_squared_error / pixel_count,
                    "p_frame_psnr": -10.0 * math.log10(
                        max(sum_squared_error / pixel_count, 1e-30)),
                    "chunk0_mse": chunk_mse[0],
                    "chunk1_mse": chunk_mse[1],
                    "actual_sequence_bytes": total_bytes,
                })
                print(json.dumps({"stage": "operational-screen", **rows[-1]}),
                      flush=True)

    aggregate = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        mean_mse = float(np.mean([row["p_frame_mse"] for row in selected]))
        per_sequence = {}
        for sequence in SCREEN_IDS:
            sequence_rows = [row for row in selected if row["sequence"] == sequence]
            sequence_mse = float(np.mean([row["p_frame_mse"] for row in sequence_rows]))
            per_sequence[sequence] = {
                "p_frame_mse": sequence_mse,
                "p_frame_psnr": -10.0 * math.log10(max(sequence_mse, 1e-30)),
                "actual_bytes": sum(row["actual_sequence_bytes"] for row in sequence_rows),
            }
        times = latency.get(method, [])
        aggregate[method] = {
            "windows": len(selected),
            "p_frame_mse": mean_mse,
            "p_frame_psnr": -10.0 * math.log10(max(mean_mse, 1e-30)),
            "chunk0_mse": float(np.mean([row["chunk0_mse"] for row in selected])),
            "chunk1_mse": float(np.mean([row["chunk1_mse"] for row in selected])),
            "actual_bytes": sum(row["actual_sequence_bytes"] for row in selected),
            "backbone_plus_head_ms_per_chunk_median": (
                float(np.median(times)) if times else 0.0),
            "backbone_plus_head_ms_per_chunk_p95": (
                float(np.percentile(times, 95)) if times else 0.0),
            "per_sequence": per_sequence,
        }
    screen_dir = output_dir / "screen"
    with (screen_dir / "operational_windows.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(screen_dir / "operational_summary.json", aggregate)
    del heads, backbones, i_net, p_net
    torch.cuda.empty_cache()
    return aggregate


def final_decision(latent, gate, operational):
    if not gate["passes"]:
        return {
            "status": "stop_pnp_transfer_route",
            "recommendation": (
                "冻结 PnP-VCVE 表示没有同时通过内部未见视频、随机骨干和高字节块门槛；"
                "不租卡、不扩大头部，也不因此读取 validation。"),
            "validation_or_sealed_data_should_be_read": False,
        }
    if operational is None:
        return {
            "status": "latent_gate_passed_but_rgb_screen_missing",
            "recommendation": "潜变量门槛通过，但缺少 RGB 闭环，结论尚未确定。",
            "validation_or_sealed_data_should_be_read": False,
        }
    pre = operational["pretrained"]
    mean = operational["mean"]
    random_arm = operational["random"]
    rgb_positive = sum(
        pre["per_sequence"][sequence]["p_frame_mse"]
        < mean["per_sequence"][sequence]["p_frame_mse"]
        for sequence in SCREEN_IDS)
    rgb_beats_random = sum(
        pre["per_sequence"][sequence]["p_frame_mse"]
        < random_arm["per_sequence"][sequence]["p_frame_mse"]
        for sequence in SCREEN_IDS)
    if (pre["p_frame_mse"] < mean["p_frame_mse"]
            and pre["p_frame_mse"] < random_arm["p_frame_mse"]
            and rgb_positive >= 5 and rgb_beats_random >= 4
            and pre["chunk1_mse"] <= mean["chunk1_mse"]):
        return {
            "status": "continue_to_registered_validation_replication",
            "recommendation": (
                "预训练骨干在潜变量和 RGB 两层都显示内部未见视频迁移信号；"
                "下一步才值得在已登记 validation 开发样本做真实码流、公平 all-Base 和完整解码复核。"),
            "validation_or_sealed_data_should_be_read": True,
            "rgb_positive_sequences": rgb_positive,
            "rgb_beats_random_sequences": rgb_beats_random,
        }
    return {
        "status": "stop_pnp_transfer_route_after_rgb_screen",
        "recommendation": (
            "潜变量探针即使局部通过，也没有在内部未见视频的 RGB 闭环中稳定胜过 mean-fill 与随机骨干；"
            "停止 PnP-VCVE 迁移路线，不租卡、不读 validation。"),
        "validation_or_sealed_data_should_be_read": False,
        "rgb_positive_sequences": rgb_positive,
        "rgb_beats_random_sequences": rgb_beats_random,
    }


def write_report(summary, output_dir):
    latent = summary["latent_screen"]
    operational = summary.get("operational_screen")
    lines = [
        "# E12 冻结 PnP-VCVE 骨干迁移探针",
        "",
        f"结论：{summary['decision']['recommendation']}",
        "",
        "这里的内部未见筛查使用 REDS train/019..024；它们不参与梯度或 checkpoint 选择，"
        "但仍不是独立测试集。validation 与封存数据均未读取。",
        "",
        "## 潜变量结果",
        "",
        "| 骨干 | 拟合恢复 | 内部未见恢复 | 高字节四分位恢复 | 正恢复视频数 | 第二段恢复 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        fit = summary["training"][arm]["fit_metrics"]["overall"]["latent_gap_recovery"]
        screen = latent[arm]
        positive = sum(item["latent_gap_recovery"] > 0
                       for item in screen["per_sequence"].values())
        lines.append(
            f"| {arm} | {100*fit:+.3f}% | "
            f"{100*screen['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*screen['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
            f"{positive}/6 | "
            f"{100*screen['per_chunk_position']['1']['latent_gap_recovery']:+.3f}% |")
    if operational is not None:
        lines += [
            "",
            "## RGB 闭环（内部筛查）",
            "",
            "| 方法 | P 帧 PSNR | 实际窗口总字节 | 相对 mean PSNR | 骨干+头中位 ms/段 |",
            "|---|---:|---:|---:|---:|",
        ]
        mean_psnr = operational["mean"]["p_frame_psnr"]
        for method in ("mean",) + ARMS:
            point = operational[method]
            lines.append(
                f"| {method} | {point['p_frame_psnr']:.5f} | "
                f"{point['actual_bytes']:,} | {point['p_frame_psnr']-mean_psnr:+.5f} dB | "
                f"{point['backbone_plus_head_ms_per_chunk_median']:.3f} |")
        lines += [
            "",
            "字节数包含序列头、I 帧、两段 z、Base-y、K8 route 和段头；未包含共享模型权重。"
            "计时这里只统计额外骨干和输出头，不冒充完整解码耗时。",
        ]
    lines += [
        "",
        "## 边界",
        "",
        "- PnP 输入来自 mean-fill 解码 RGB；真实源视频和 true-fill 只作标签。",
        "- DCVC 与 PnP 骨干均冻结；只训练直接预测 latent 修正量的输出头。",
        "- 随机骨干与预训练骨干同结构、同头部初始化、同样本顺序和同训练预算。",
        "- PnP 原生 H.264 运动向量在 DCVC 中没有同构字段，本探针使用零运动；分区图使用固定 8×8 代理。",
        "- 本轮不把训练拟合、true-fill 或内部筛查写成真实压缩收益。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("E12 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_entries = e10_entries(Path(args.e10_dir).resolve())

    # Strict order: fit features -> paired final checkpoints -> screen features.
    fit_entries, load_audit = prepare_feature_cache(
        args, "fit", FIT_IDS, base_entries, device)
    checkpoints, training = paired_training(
        args, fit_entries, output_dir, device)
    if not (output_dir / "checkpoint_lock_before_screen.json").is_file():
        raise RuntimeError("screen access attempted before checkpoint lock")
    screen_entries, screen_load_audit = prepare_feature_cache(
        args, "screen", SCREEN_IDS, base_entries, device)
    if screen_load_audit != load_audit:
        raise RuntimeError("PnP checkpoint audit changed between fit and screen")
    latent = screen_latents(checkpoints, screen_entries, output_dir, device)
    gate = latent_gate(latent)
    operational = None
    if not args.skip_operational_screen:
        operational = operational_screen(
            args, checkpoints, base_entries, output_dir, device)
    decision = final_decision(latent, gate, operational)
    summary = {
        "experiment": "E12 frozen PnP-VCVE backbone transfer probe",
        "status": decision["status"],
        "scientific_scope": {
            "fit_split": "REDS train_sharp/001..018",
            "internal_unseen_screen": "REDS train_sharp/019..024",
            "independent_test_claim_allowed": False,
            "validation_read": False,
            "sealed_data_read": False,
            "true_fill_counts_as_result": False,
            "codec_modified": False,
        },
        "protocol": {
            "window_starts": list(WINDOW_STARTS),
            "chunks_per_window": 2,
            "qp_i": QP,
            "qp_p": QP,
            "block_size": BLOCK_SIZE,
            "k": K,
            "steps": STEPS,
            "loss": "unweighted direct-delta MSE",
            "head_seed": HEAD_SEED,
            "random_backbone_seed": RANDOM_BACKBONE_SEED,
            "checkpoint_fixed_before_screen_tensor_load": True,
        },
        "official_asset": {
            "repository": "https://github.com/ZeldaM1/PnP-VCVE",
            "checkpoint": str(Path(args.pnp_checkpoint).resolve()),
            "checkpoint_bytes": Path(args.pnp_checkpoint).stat().st_size,
            "checkpoint_load_audit": load_audit,
            "upstream_license_file_present": False,
            "official_environment_not_installed": (
                "Python 3.7 / torch 1.8 / CUDA 11.1 would conflict with the working DCVC environment"),
        },
        "model": {
            "frozen_pnp_parameters": 4_559_885,
            "head_parameters": PnPLatentHead().parameter_count,
            "pnp_input": "8 mean-fill RGB frames at 128x128",
            "feature": "8x64x32x32",
            "motion_proxy": "zero flow",
            "partition_proxy": "fixed 8x8 channel at 1/255",
        },
        "training": training,
        "latent_screen": latent,
        "latent_gate": gate,
        "operational_screen": operational,
        "decision": decision,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    write_report(summary, output_dir)
    print(json.dumps({
        "stage": "complete", "status": decision["status"],
        "summary": str((output_dir / "summary.json").resolve()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
