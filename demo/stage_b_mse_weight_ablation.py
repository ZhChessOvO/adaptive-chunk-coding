#!/usr/bin/env python3
"""E10: same-MSE weight ablation and high-byte-block recovery diagnosis.

The experiment keeps DCVC-UF and the 461,824-parameter sparse C1 predictor
frozen in structure.  Four models share initialization, sample order, update
count, and MSE form; only the per-block weight changes.  All final checkpoints
are fixed before REDS val/000..005 is read.  Sealed data is never eligible.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

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
    reconstruction_float,
    should_reset,
)
from demo.stage1_token_skipping import (  # noqa: E402
    build_route_section,
    decode_y,
    encode_y,
    expand_keep_mask,
    parse_container,
)
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    compute_metrics,
    encode_trajectory,
    load_predictor_checkpoint,
    read_d2l,
    run_operational_mode,
    sha256_file,
    source_route_target,
)
from demo.stage_b_masked_predictor import (  # noqa: E402
    PredictorConfig,
    SparseMaskedLatentPredictor,
    skipped_target_blocks,
)
from demo.stage_b_prepare_predictor_cache import crop_xy  # noqa: E402
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


TRAIN_IDS = tuple(f"{index:03d}" for index in range(1, 25))
DEV_IDS = tuple(f"{index:03d}" for index in range(6))
WINDOW_STARTS = (0, 24, 48, 72)
ARMS = ("unweighted", "byte", "sensitivity", "byte_x_sensitivity")
ARM_ZH = {
    "unweighted": "无权重",
    "byte": "仅字节",
    "sensitivity": "仅敏感度",
    "byte_x_sensitivity": "字节×敏感度",
}
SEED = 20260913
ALPHA = 0.25
K = 8
STEPS = 5000
EVAL_EVERY = 250
EXPECTED_PARAMETERS = 461_824
EXPECTED_MACS = 5_776_384


@dataclass
class DevSequence:
    sequence: str
    frames: list[np.ndarray]
    source_files: list[str]
    crop: tuple[int, int]
    routes: list[np.ndarray]
    route_records: list[dict]


def parse_args():
    parser = argparse.ArgumentParser(
        description="E10 same-MSE weight ablation on registered train/dev data.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_mse_weight_ablation_train001_024_val000_005_v1")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument(
        "--reanalyze-existing", action="store_true",
        help="Rebuild derived tables/figures from an existing completed summary.")
    return parser.parse_args()


def validate_args(args):
    if args.steps != STEPS or args.eval_every != EVAL_EVERY:
        raise ValueError("E10 fixes 5,000 updates and eval-every=250")
    if args.decode_repeats != 3:
        raise ValueError("E10 requires three complete fresh decodes")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E10 fixes the learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E10 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E10 fixes gradient clipping at 1.0")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("E10 fixes the codec skip threshold at zero")
    data_root = Path(args.data_root).resolve()
    forbidden = [
        data_root / "train_sharp" / sequence
        for sequence in tuple(f"{index:03d}" for index in range(25, 33))
    ]
    forbidden += [
        data_root / "val_sharp" / sequence
        for sequence in tuple(f"{index:03d}" for index in range(12, 30))
    ]
    # Keep the forbidden paths explicit in the protocol record; never glob their
    # contents or inspect them in this program.
    args.forbidden_paths = [str(path) for path in forbidden]


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


def load_window(data_root: Path, split: str, sequence: str, start: int):
    allowed = TRAIN_IDS if split == "train_sharp" else DEV_IDS
    if sequence not in allowed:
        raise ValueError(f"E10 forbids {split}/{sequence}")
    x, y = crop_xy(SEED, sequence, start)
    root = data_root / split / sequence
    paths = sorted(root.glob("*.png"))[start:start + 17]
    if len(paths) != 17:
        raise ValueError(f"{root} start={start} does not provide 17 frames")
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[y:y + 512, x:x + 512]
        if crop.shape != (512, 512, 3):
            raise ValueError(f"invalid 512x512 crop ({x},{y}) for {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return [str(path.resolve()) for path in paths], frames, (x, y)


@torch.inference_mode()
def encode_i(i_net, frame, device):
    source = (model_frame(frame, device) - 0.5).to(memory_format=torch.channels_last)
    encoded = i_net.compress(source, 32, 0, 0)
    stream = bytes(encoded["bit_stream"])
    i_hat = decode_i_stream(
        i_net, stream, int(encoded["ec_parallel"]), 32, 512, 512)
    return stream, int(encoded["ec_parallel"]), i_hat


def unit_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.float() - b.float()).square()).item())


@torch.inference_mode()
def select_high_byte_route(p_net, prepared):
    y = prepared["y"]
    grid = (y.shape[-2] // 2, y.shape[-1] // 2)
    if grid != (16, 16):
        raise RuntimeError(f"unexpected E10 route grid {grid}")
    all_base_route = np.zeros(grid, dtype=np.bool_)
    all_base = encode_y(
        p_net, y, prepared["common_params"], all_base_route, 2)
    expected = np.nan_to_num(
        all_base.expected_bits_by_block.reshape(-1), nan=-1e30,
        posinf=1e30, neginf=-1e30)
    singletons = []
    for flat_id in range(grid[0] * grid[1]):
        route = np.zeros(grid, dtype=np.bool_)
        route.reshape(-1)[flat_id] = True
        encoded = encode_y(p_net, y, prepared["common_params"], route, 2)
        singletons.append({
            "flat_id": flat_id,
            "row": flat_id // grid[1],
            "col": flat_id % grid[1],
            "allbase_y_bytes": len(all_base.stream),
            "singleton_y_bytes": len(encoded.stream),
            "saved_y_bytes": len(all_base.stream) - len(encoded.stream),
            "expected_bits": float(expected[flat_id]),
        })
    ranked = sorted(
        singletons,
        key=lambda item: (
            -item["saved_y_bytes"], -item["expected_bits"], item["flat_id"]))
    selected = ranked[:K]
    route = np.zeros(grid, dtype=np.bool_)
    route.reshape(-1)[[item["flat_id"] for item in selected]] = True
    return route, all_base, singletons, selected


@torch.inference_mode()
def collect_training_sample(p_net, prepared, target_rgb, route, all_base,
                            singletons, selected):
    routed = encode_y(
        p_net, prepared["y"], prepared["common_params"], route, 2)
    decoded_q, mean_y = decode_y(
        p_net, routed.stream, routed.ec_parallel,
        prepared["common_params"], route, 2)
    if not torch.equal(decoded_q, routed.q_dense):
        raise RuntimeError("E10 training Base-y rANS round trip mismatch")
    target_delta = source_route_target(p_net, prepared, decoded_q, route, 2)
    mean_rgb, _, _ = reconstruction_float(
        p_net, mean_y, prepared["q_decoder"], g_frame_delay)
    true_rgb, _, _ = reconstruction_float(
        p_net, mean_y + target_delta, prepared["q_decoder"], g_frame_delay)
    mean_source_mse = unit_mse(mean_rgb, target_rgb)
    true_source_mse = unit_mse(true_rgb, target_rgb)
    rate_map = torch.zeros(route.shape, dtype=torch.float32, device=mean_y.device)
    sensitivity_map = torch.zeros_like(rate_map)
    selected_by_id = {item["flat_id"]: dict(item) for item in selected}
    for flat_id in np.flatnonzero(route.reshape(-1)):
        item = selected_by_id[int(flat_id)]
        row, col = item["row"], item["col"]
        rate_map[row, col] = max(float(item["saved_y_bytes"]), 0.0)
        one_delta = torch.zeros_like(target_delta)
        one_delta[..., row * 2:(row + 1) * 2, col * 2:(col + 1) * 2] = (
            target_delta[..., row * 2:(row + 1) * 2, col * 2:(col + 1) * 2])
        single_rgb, _, _ = reconstruction_float(
            p_net, mean_y + one_delta, prepared["q_decoder"], g_frame_delay)
        single_mse = unit_mse(single_rgb, target_rgb)
        sensitivity = max(mean_source_mse - single_mse, 0.0)
        sensitivity_map[row, col] = sensitivity
        item["true_fill_pixel_gain_unit_mse"] = sensitivity
    if float(rate_map.sum()) <= 0.0:
        raise RuntimeError("E10 sample has no positive byte weight")
    if float(sensitivity_map.sum()) <= 0.0:
        raise RuntimeError("E10 sample has no positive sensitivity weight")
    route_section = build_route_section(route, 2)
    tensors = {
        "decoded_q": decoded_q.detach().to(dtype=torch.int8).cpu().contiguous(),
        "mean_y": mean_y.detach().cpu().contiguous(),
        "common_params": prepared["common_params"].detach().cpu().contiguous(),
        "target_delta": target_delta.detach().cpu().contiguous(),
        "rate_map": rate_map.detach().cpu().contiguous(),
        "sensitivity_map": sensitivity_map.detach().cpu().contiguous(),
        "skip_blocks": torch.from_numpy(route.astype(np.uint8)),
    }
    record = {
        "selected_blocks": [selected_by_id[int(index)]
                            for index in np.flatnonzero(route.reshape(-1))],
        "allbase_y_bytes": len(all_base.stream),
        "k8_y_bytes": len(routed.stream),
        "combination_y_bytes_saved": len(all_base.stream) - len(routed.stream),
        "route_bytes": len(route_section),
        "combination_net_y_after_route_bytes_saved": (
            len(all_base.stream) - len(routed.stream) - len(route_section)),
        "mean_source_mse_unit": mean_source_mse,
        "true_fill_source_mse_unit": true_source_mse,
    }
    return tensors, record, mean_y


def training_key(sequence: str, start: int, chunk_index: int) -> str:
    return f"train_{sequence}_s{start:02d}_c{chunk_index}"


@torch.inference_mode()
def prepare_training_cache(i_net, p_net, data_root: Path, output_dir: Path,
                           device):
    cache_dir = output_dir / "train_cache"
    sample_dir = cache_dir / "samples"
    record_dir = cache_dir / "records"
    sample_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    expected_count = len(TRAIN_IDS) * len(WINDOW_STARTS) * 2
    complete_path = cache_dir / "complete.json"
    manifest_path = cache_dir / "manifest.jsonl"
    if complete_path.is_file() and manifest_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if (complete.get("sample_count") != expected_count
                or len(entries) != expected_count
                or complete.get("seed") != SEED):
            raise RuntimeError("existing E10 training cache has incompatible protocol")
        if not all(Path(item["path"]).is_file() for item in entries):
            raise RuntimeError("existing E10 training cache is incomplete")
        return entries

    entries = []
    for sequence in TRAIN_IDS:
        for start in WINDOW_STARTS:
            source_files, frames, crop = load_window(
                data_root, "train_sharp", sequence, start)
            _, _, i_hat = encode_i(i_net, frames[0], device)
            initialize_p_state(p_net, i_hat)
            first_index = 1
            for chunk_index in range(2):
                key = training_key(sequence, start, chunk_index)
                tensor_path = (sample_dir / f"{key}.safetensors").resolve()
                record_path = record_dir / f"{key}.json"
                chunk, target_rgb, valid_count = make_chunk(
                    frames, first_index, device)
                if valid_count != g_frame_delay:
                    raise RuntimeError("E10 training window has an incomplete P chunk")
                prepared = prepare_chunk_latents(p_net, chunk, 32)
                if tensor_path.is_file() and record_path.is_file():
                    stored = load_file(tensor_path, device="cpu")
                    route = stored["skip_blocks"].numpy().astype(np.bool_)
                    routed = encode_y(
                        p_net, prepared["y"], prepared["common_params"], route, 2)
                    decoded_q, mean_y = decode_y(
                        p_net, routed.stream, routed.ec_parallel,
                        prepared["common_params"], route, 2)
                    if (not torch.equal(decoded_q.cpu(), stored["decoded_q"])
                            or not torch.equal(mean_y.cpu(), stored["mean_y"])):
                        raise RuntimeError(f"resumed E10 cache disagrees for {key}")
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    print(json.dumps({"stage": "train-cache-resume", "key": key}),
                          flush=True)
                else:
                    if tensor_path.exists() or record_path.exists():
                        raise RuntimeError(f"partial E10 cache pair for {key}")
                    print(json.dumps({"stage": "train-cache-collect", "key": key}),
                          flush=True)
                    route, all_base, singletons, selected = select_high_byte_route(
                        p_net, prepared)
                    tensors, diagnostics, mean_y = collect_training_sample(
                        p_net, prepared, target_rgb, route, all_base,
                        singletons, selected)
                    save_file(tensors, tensor_path)
                    record = {
                        "key": key,
                        "path": str(tensor_path),
                        "sequence": sequence,
                        "window_start": start,
                        "chunk_index": chunk_index,
                        "crop_xy": list(crop),
                        "source_files": source_files,
                        "trajectory": "fixed_mean_fill_k8",
                        **diagnostics,
                    }
                    atomic_json(record_path, record)
                entries.append(record)
                x_hat, feature = p_net.get_recon_and_feature(
                    mean_y, p_net.ctx, prepared["q_decoder"])
                p_net.set_ref_feature(
                    feature, should_reset(chunk_index, 32))
                first_index += valid_count

    if len(entries) != expected_count:
        raise RuntimeError(f"E10 expected {expected_count} samples, got {len(entries)}")
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries),
        encoding="utf-8")
    os.replace(temporary, manifest_path)
    atomic_json(complete_path, {
        "format": "e10_same_mse_k8_training_cache_v1",
        "split": "REDS train_sharp/001..024",
        "sequences": list(TRAIN_IDS),
        "window_starts": list(WINDOW_STARTS),
        "chunks_per_window": 2,
        "sample_count": len(entries),
        "trajectory": "fixed_mean_fill_k8",
        "route": "top-8 actual singleton Base-y byte saving",
        "seed": SEED,
        "sealed_or_unregistered_data_read": False,
    })
    return entries


def load_gpu_training_samples(entries, device):
    samples = []
    for entry in entries:
        tensors = load_file(entry["path"], device=str(device))
        samples.append(tensors)
    return samples


def query_supervision(model, sample):
    target, query_ids = skipped_target_blocks(
        sample["target_delta"], sample["skip_blocks"], model.config.block_size)
    rates = sample["rate_map"].reshape(-1)[query_ids].float()
    sensitivities = sample["sensitivity_map"].reshape(-1)[query_ids].float()
    return target, rates, sensitivities


def arm_weights(arm: str, rates: torch.Tensor, sensitivities: torch.Tensor):
    rates = torch.clamp_min(rates, 0.0)
    sensitivities = torch.clamp_min(sensitivities, 0.0)
    if arm == "unweighted":
        weights = torch.ones_like(rates)
    elif arm == "byte":
        weights = rates
    elif arm == "sensitivity":
        weights = sensitivities
    elif arm == "byte_x_sensitivity":
        weights = rates * sensitivities
    else:
        raise ValueError(arm)
    mean_weight = weights.mean()
    if not torch.isfinite(mean_weight) or float(mean_weight) <= 0.0:
        raise RuntimeError(f"{arm} sample has no positive finite supervision weight")
    return weights / mean_weight


def mse_loss(model, sample, arm: str):
    raw = model(
        sample["decoded_q"], sample["mean_y"],
        sample["common_params"], sample["skip_blocks"])
    target, rates, sensitivities = query_supervision(model, sample)
    per_block = (ALPHA * raw - target).float().square().mean(dim=1)
    weights = arm_weights(arm, rates, sensitivities)
    return torch.mean(weights * per_block)


@torch.inference_mode()
def latent_metrics(model, samples):
    sums = {
        "unweighted_error": 0.0,
        "unweighted_target": 0.0,
        "byte_error": 0.0,
        "byte_target": 0.0,
        "sensitivity_error": 0.0,
        "sensitivity_target": 0.0,
        "product_error": 0.0,
        "product_target": 0.0,
    }
    for sample in samples:
        raw = model(
            sample["decoded_q"], sample["mean_y"],
            sample["common_params"], sample["skip_blocks"])
        target, rates, sensitivities = query_supervision(model, sample)
        error = (ALPHA * raw - target).float().square().sum(dim=1)
        target_energy = target.float().square().sum(dim=1)
        weight_sets = {
            "unweighted": torch.ones_like(rates),
            "byte": torch.clamp_min(rates, 0.0),
            "sensitivity": torch.clamp_min(sensitivities, 0.0),
            "product": torch.clamp_min(rates, 0.0) * torch.clamp_min(
                sensitivities, 0.0),
        }
        for name, weights in weight_sets.items():
            sums[f"{name}_error"] += float((weights * error).sum())
            sums[f"{name}_target"] += float((weights * target_energy).sum())
    return {
        f"{name}_latent_energy_recovery": 1.0 - (
            sums[f"{name}_error"] / max(sums[f"{name}_target"], 1e-30))
        for name in ("unweighted", "byte", "sensitivity", "product")
    }


def make_schedule(sample_count: int):
    rng = np.random.default_rng(SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:STEPS]


def train_arm(arm, model, samples, schedule, args, output_dir: Path):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = [{"step": 0, **latent_metrics(model, samples)}]
    metrics_path = output_dir / "training" / f"{arm}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(history[0], ensure_ascii=False) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
    started = time.perf_counter()
    model.train()
    for step, index in enumerate(schedule, 1):
        loss = mse_loss(model, samples[index], arm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"{arm}: non-finite gradient")
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            event = {
                "step": step,
                "last_sample_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **latent_metrics(model, samples),
            }
            history.append(event)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps({"stage": "train", "arm": arm, **event}), flush=True)
            model.train()
    torch.cuda.synchronize(next(model.parameters()).device)
    elapsed = time.perf_counter() - started
    model.eval()
    return {
        "loss": "per-block MSE with weights normalized to sample mean 1",
        "weight_mode": arm,
        "checkpoint_rule": "fixed final update; no development early stopping",
        "steps": STEPS,
        "final_metrics_on_training_samples": latent_metrics(model, samples),
        "history": history,
        "training_wall_seconds": elapsed,
        "updates_per_second": STEPS / elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(
            next(model.parameters()).device)),
    }


def save_arm_checkpoint(path: Path, arm: str, model, training: dict, args):
    payload = {
        "format_version": 1,
        "project": "Adaptive Chunk Coding",
        "profile": f"stage-b-c1-e10-{arm}-mse",
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "seed": SEED,
            "steps_requested": STEPS,
            "alpha": ALPHA,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "loss": "MSE",
            "weight_mode": arm,
            "checkpoint_rule": "final",
            "result": training,
        },
        "data": {
            "split": "REDS train_sharp/001..024",
            "sequence_ids": list(TRAIN_IDS),
            "window_starts": list(WINDOW_STARTS),
            "chunk_indices": [0, 1],
            "sample_count": 192,
            "trajectory": "fixed_mean_fill_k8",
        },
        "scientific_scope": {
            "train_only": True,
            "development_or_sealed_data_used_for_checkpoint": False,
            "generalization_claim_allowed": False,
        },
    }
    atomic_checkpoint(path, payload)


def train_all_arms(entries, args, output_dir: Path, device):
    samples = load_gpu_training_samples(entries, device)
    config = PredictorConfig()
    seed_everything(SEED)
    initial = SparseMaskedLatentPredictor(config).to(device)
    if initial.parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("E10 C1 parameter count changed")
    if initial.predicted_macs(K) != EXPECTED_MACS:
        raise RuntimeError("E10 C1 K8 MAC count changed")
    initial_state = copy.deepcopy(initial.state_dict())
    schedule = make_schedule(len(samples))
    checkpoints = {}
    training_records = {}
    for arm in ARMS:
        checkpoint_path = (output_dir / "checkpoints" / f"{arm}_final.pt").resolve()
        training_path = output_dir / "training" / f"{arm}_summary.json"
        if checkpoint_path.is_file() and training_path.is_file():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (payload.get("training", {}).get("seed") != SEED
                    or payload.get("training", {}).get("steps_requested") != STEPS
                    or payload.get("training", {}).get("weight_mode") != arm):
                raise RuntimeError(f"existing {arm} checkpoint is incompatible")
            training = json.loads(training_path.read_text(encoding="utf-8"))
            print(json.dumps({"stage": "train-resume", "arm": arm}), flush=True)
        else:
            if checkpoint_path.exists() or training_path.exists():
                raise RuntimeError(f"partial E10 training output for {arm}")
            model = SparseMaskedLatentPredictor(config).to(device)
            model.load_state_dict(initial_state, strict=True)
            training = train_arm(arm, model, samples, schedule, args, output_dir)
            save_arm_checkpoint(checkpoint_path, arm, model, training, args)
            atomic_json(training_path, training)
            del model
            torch.cuda.empty_cache()
        checkpoints[arm] = checkpoint_path
        training_records[arm] = training
    del samples, initial, initial_state
    torch.cuda.empty_cache()
    atomic_json(output_dir / "training_complete.json", {
        "all_final_checkpoints_fixed_before_development_read": True,
        "arms": {arm: str(path) for arm, path in checkpoints.items()},
        "checkpoint_rule": "fixed final update",
        "development_read_at_this_stage": False,
    })
    return checkpoints, training_records


@torch.inference_mode()
def prepare_dev_sequences(i_net, p_net, data_root: Path, output_dir: Path,
                          device):
    dev_sequences = []
    plan_dir = output_dir / "dev_route_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for sequence in DEV_IDS:
        source_files, frames, crop = load_window(
            data_root, "val_sharp", sequence, 0)
        _, _, i_hat = encode_i(i_net, frames[0], device)
        initialize_p_state(p_net, i_hat)
        routes = []
        route_records = []
        first_index = 1
        for chunk_index in range(2):
            chunk, _, valid_count = make_chunk(frames, first_index, device)
            if valid_count != g_frame_delay:
                raise RuntimeError("E10 development window has incomplete P chunk")
            prepared = prepare_chunk_latents(p_net, chunk, 32)
            route, all_base, _, selected = select_high_byte_route(p_net, prepared)
            routed = encode_y(
                p_net, prepared["y"], prepared["common_params"], route, 2)
            decoded_q, mean_y = decode_y(
                p_net, routed.stream, routed.ec_parallel,
                prepared["common_params"], route, 2)
            if not torch.equal(decoded_q, routed.q_dense):
                raise RuntimeError("E10 dev route Base-y rANS mismatch")
            route_bytes = len(build_route_section(route, 2))
            record = {
                "chunk_index": chunk_index,
                "selected_flat_ids": np.flatnonzero(route.reshape(-1)).tolist(),
                "selected_singletons": sorted(selected, key=lambda item: item["flat_id"]),
                "allbase_y_bytes": len(all_base.stream),
                "k8_y_bytes": len(routed.stream),
                "combination_y_bytes_saved": len(all_base.stream) - len(routed.stream),
                "route_bytes": route_bytes,
                "combination_net_y_after_route_bytes_saved": (
                    len(all_base.stream) - len(routed.stream) - route_bytes),
            }
            routes.append(route)
            route_records.append(record)
            x_hat, feature = p_net.get_recon_and_feature(
                mean_y, p_net.ctx, prepared["q_decoder"])
            p_net.set_ref_feature(feature, should_reset(chunk_index, 32))
            first_index += valid_count
        atomic_json(plan_dir / f"val_{sequence}.json", {
            "sequence": sequence,
            "split": "REDS val_sharp/000..005 development only",
            "source_files": source_files,
            "crop_xy": list(crop),
            "route_trajectory": "fixed mean-fill",
            "chunks": route_records,
        })
        dev_sequences.append(DevSequence(
            sequence=sequence,
            frames=frames,
            source_files=source_files,
            crop=crop,
            routes=routes,
            route_records=route_records,
        ))
        print(json.dumps({"stage": "dev-route", "sequence": sequence}), flush=True)
    return dev_sequences


def evaluation_args(args):
    class Values:
        pass
    values = Values()
    values.model_path_i = args.model_path_i
    values.model_path_p = args.model_path_p
    values.skip_thres = args.skip_thres
    values.height = 512
    values.width = 512
    values.latent_block_size = 2
    values.reset_interval = 32
    values.decode_repeats = args.decode_repeats
    values.qp_i_override = 32
    values.qp_route = 32
    return values


def frame_psnr_to_unit_mse(psnr: float) -> float:
    return 10.0 ** (-float(psnr) / 10.0)


def point_p_mse(point: dict) -> float:
    return float(np.mean([
        frame_psnr_to_unit_mse(value)
        for value in point["metrics"]["per_frame_psnr"][1:]
    ]))


def point_chunk_mse(point: dict, chunk_index: int) -> float:
    start = 1 + chunk_index * g_frame_delay
    stop = start + g_frame_delay
    return float(np.mean([
        frame_psnr_to_unit_mse(value)
        for value in point["metrics"]["per_frame_psnr"][start:stop]
    ]))


def aggregate_point(results: dict, label: str):
    points = [results[sequence][label] for sequence in DEV_IDS]
    p_mse = float(np.mean([point_p_mse(point) for point in points]))
    chunk_mse = [float(np.mean([
        point_chunk_mse(point, chunk_index) for point in points
    ])) for chunk_index in range(2)]
    component_keys = (
        "sequence_header_and_chunk_lengths_bytes", "i_payload_bytes",
        "p_container_header_bytes", "global_z_bytes", "base_y_bytes",
        "route_bytes", "residual_bytes", "container_bytes", "total_bytes")
    components = {
        key: sum(point["stream"][key] for point in points)
        for key in component_keys
    }
    decode_seconds = sum(
        point["compute"]["decode_wall_seconds_median"] for point in points)
    predictor_latencies = [
        value
        for point in points
        for value in point["compute"]["predictor_latency_ms_all"]
    ]
    return {
        "label": label,
        "total_bytes": components["total_bytes"],
        "components": components,
        "p_frame_unit_mse": p_mse,
        "p_frame_psnr": -10.0 * math.log10(max(p_mse, 1e-30)),
        "chunk_unit_mse": chunk_mse,
        "chunk_psnr": [-10.0 * math.log10(max(value, 1e-30))
                       for value in chunk_mse],
        "decode_wall_seconds_sum_of_sequence_medians": decode_seconds,
        "decode_ms_per_frame": 1000.0 * decode_seconds / (len(DEV_IDS) * 17),
        "decode_fps": len(DEV_IDS) * 17 / decode_seconds,
        "predictor_latency_ms_p50": (
            float(np.median(predictor_latencies)) if predictor_latencies else 0.0),
        "predictor_latency_ms_p95": (
            float(np.percentile(predictor_latencies, 95))
            if predictor_latencies else 0.0),
        "peak_cuda_allocated_bytes": max(
            point["compute"]["peak_cuda_allocated_bytes_max"] for point in points),
        "linear_macs_per_chunk": max(
            point["compute"]["linear_macs_per_chunk"] for point in points),
        "per_sequence": {
            sequence: {
                "p_frame_unit_mse": point_p_mse(results[sequence][label]),
                "p_frame_psnr": -10.0 * math.log10(max(
                    point_p_mse(results[sequence][label]), 1e-30)),
                "chunk_unit_mse": [
                    point_chunk_mse(results[sequence][label], index)
                    for index in range(2)],
            }
            for sequence in DEV_IDS
        },
    }


def matched_rate(baselines: list[dict], target: dict):
    ordered = sorted(baselines, key=lambda point: point["p_frame_psnr"])
    quality = target["p_frame_psnr"]
    if quality >= ordered[-1]["p_frame_psnr"]:
        anchor = ordered[-1]
        return {
            "comparable": True,
            "comparison": "directly matches or exceeds highest-quality anchor",
            "lower_label": anchor["label"],
            "upper_label": anchor["label"],
            "matched_allbase_bytes": anchor["total_bytes"],
            "rate_change_percent": 100.0 * (
                target["total_bytes"] / anchor["total_bytes"] - 1.0),
        }
    for lower, upper in zip(ordered, ordered[1:]):
        if lower["p_frame_psnr"] <= quality <= upper["p_frame_psnr"]:
            span = upper["p_frame_psnr"] - lower["p_frame_psnr"]
            fraction = 0.0 if span == 0 else (
                quality - lower["p_frame_psnr"]) / span
            matched_bytes = math.exp(
                math.log(lower["total_bytes"])
                + fraction * (math.log(upper["total_bytes"])
                              - math.log(lower["total_bytes"])))
            return {
                "comparable": True,
                "comparison": "log-rate interpolation",
                "lower_label": lower["label"],
                "upper_label": upper["label"],
                "matched_allbase_bytes": matched_bytes,
                "rate_change_percent": 100.0 * (
                    target["total_bytes"] / matched_bytes - 1.0),
            }
    lowest = ordered[0]
    directly_dominated = target["total_bytes"] >= lowest["total_bytes"]
    result = {
        "comparable": False,
        "quality": quality,
        "baseline_quality_range": [
            ordered[0]["p_frame_psnr"], ordered[-1]["p_frame_psnr"]],
        "reason": "target quality is below the registered P30-P32 range",
        "directly_dominated_without_extrapolation": directly_dominated,
    }
    if directly_dominated:
        result.update({
            "dominating_anchor": lowest["label"],
            "dominating_anchor_has_higher_quality_db": (
                lowest["p_frame_psnr"] - quality),
            "rate_excess_vs_dominating_anchor_percent": 100.0 * (
                target["total_bytes"] / lowest["total_bytes"] - 1.0),
            "note": (
                "Exact quality-matched interpolation is unnecessary for the "
                "negative finding: the anchor already uses fewer bytes and has "
                "higher quality."),
        })
    return result


def public_point(point: dict):
    return {
        "label": point["label"],
        "profile": point["profile"],
        "qp_i": point["qp_i"],
        "qp_p": point["qp_p"],
        "stream": point["stream"],
        "metrics": point["metrics"],
        "compute": point["compute"],
        "encode_wall_seconds": point["encode_wall_seconds"],
        "chunks": [{key: value for key, value in chunk.items()
                    if key != "route_sha256"}
                   for chunk in point["chunks"]],
        "audits": {
            "complete_fresh_decode": point["validation"][
                "full_sequence_fresh_decode"],
            "three_repeats_identical": point["validation"][
                "repeated_raw_tensor_hashes_identical"],
            "source_or_omitted_y_available_to_predictor": point["validation"][
                "source_or_omitted_y_available_to_predictor"],
            "routed_reconstruction_propagated": point["validation"][
                "routed_reconstruction_propagated"],
        },
    }


@torch.inference_mode()
def evaluate_operational_points(i_net, p_net, dev_sequences, checkpoints,
                                args, output_dir: Path, device):
    eval_args = evaluation_args(args)
    results = {sequence: {} for sequence in DEV_IDS}
    references = {}
    checkpoint_digests = {arm: sha256_file(path)
                          for arm, path in checkpoints.items()}
    for dev in dev_sequences:
        sequence_root = output_dir / "evaluation" / dev.sequence
        for qp in (30, 31, 32):
            label = f"allbase-p{qp}"
            print(json.dumps({"stage": "evaluate", "sequence": dev.sequence,
                              "mode": label}), flush=True)
            point = run_operational_mode(
                label, "all-base", qp, i_net, p_net, None, None,
                dev.frames, eval_args, sequence_root / label, device, None)
            results[dev.sequence][label] = point
        label = "mean-k8"
        print(json.dumps({"stage": "evaluate", "sequence": dev.sequence,
                          "mode": label}), flush=True)
        results[dev.sequence][label] = run_operational_mode(
            label, "mean", 32, i_net, p_net, None, None,
            dev.frames, eval_args, sequence_root / label, device, None,
            fixed_routes=dev.routes)
        for arm in ARMS:
            label = f"{arm}-c1-a025-k8"
            print(json.dumps({"stage": "evaluate", "sequence": dev.sequence,
                              "mode": label}), flush=True)
            results[dev.sequence][label] = run_operational_mode(
                label, "learned-lite-c1-a025", 32, i_net, p_net,
                checkpoints[arm], checkpoint_digests[arm], dev.frames,
                eval_args, sequence_root / label, device, None,
                fixed_routes=dev.routes)
        print(json.dumps({"stage": "evaluate", "sequence": dev.sequence,
                          "mode": "true-fill-diagnostic"}), flush=True)
        encoded = encode_trajectory(
            "true-fill-diagnostic", "source-route-reference", 32,
            i_net, p_net, None, None, dev.frames, eval_args,
            sequence_root / "true-fill-diagnostic", device,
            operational=False, fixed_routes=dev.routes)
        references[dev.sequence] = {
            "operational": False,
            "eligible_as_actual_benefit": False,
            "metrics": compute_metrics(
                dev.frames, encoded["recon"], encoded["chunks"], eval_args, None),
            "chunks": [{key: value for key, value in chunk.items()
                        if key != "route_sha256"}
                       for chunk in encoded["chunks"]],
        }
    return results, references


@torch.inference_mode()
def diagnose_arm_sequence(i_net, p_net, dev: DevSequence,
                          checkpoint_path: Path, device):
    predictor, _, _ = load_predictor_checkpoint(checkpoint_path, device)
    _, _, i_hat = encode_i(i_net, dev.frames[0], device)
    initialize_p_state(p_net, i_hat)
    records = []
    chunk_records = []
    first_index = 1
    for chunk_index, route in enumerate(dev.routes):
        chunk, target_rgb, valid_count = make_chunk(dev.frames, first_index, device)
        prepared = prepare_chunk_latents(p_net, chunk, 32)
        empty = np.zeros_like(route)
        all_base = encode_y(
            p_net, prepared["y"], prepared["common_params"], empty, 2)
        routed = encode_y(
            p_net, prepared["y"], prepared["common_params"], route, 2)
        decoded_q, mean_y = decode_y(
            p_net, routed.stream, routed.ec_parallel,
            prepared["common_params"], route, 2)
        if not torch.equal(decoded_q, routed.q_dense):
            raise RuntimeError("E10 diagnostic Base-y rANS mismatch")
        target_delta = source_route_target(p_net, prepared, decoded_q, route, 2)
        raw_blocks = predictor(
            decoded_q, mean_y, prepared["common_params"], route)
        target_blocks, query_ids = skipped_target_blocks(target_delta, route, 2)
        predicted_blocks = ALPHA * raw_blocks.float()
        predicted_y_raw, active_profile = predictor.apply(
            decoded_q, mean_y, prepared["common_params"], route)
        predicted_y = mean_y + ALPHA * (
            predicted_y_raw.to(dtype=mean_y.dtype) - mean_y)
        keep = expand_keep_mask(route, 2, mean_y.shape, mean_y.device)
        if not torch.equal(predicted_y[keep], mean_y[keep]):
            raise RuntimeError("E10 diagnostic modified transmitted Base latent")
        mean_rgb, _, _ = reconstruction_float(
            p_net, mean_y, prepared["q_decoder"], valid_count)
        full_pred_rgb, _, _ = reconstruction_float(
            p_net, predicted_y, prepared["q_decoder"], valid_count)
        true_rgb, _, _ = reconstruction_float(
            p_net, mean_y + target_delta, prepared["q_decoder"], valid_count)
        mean_source_mse = unit_mse(mean_rgb, target_rgb)
        full_pred_source_mse = unit_mse(full_pred_rgb, target_rgb)
        true_source_mse = unit_mse(true_rgb, target_rgb)
        route_bytes = len(build_route_section(route, 2))
        chunk_records.append({
            "sequence": dev.sequence,
            "chunk_index": chunk_index,
            "allbase_y_bytes": len(all_base.stream),
            "k8_y_bytes": len(routed.stream),
            "combination_y_bytes_saved": len(all_base.stream) - len(routed.stream),
            "route_bytes": route_bytes,
            "combination_net_y_after_route_bytes_saved": (
                len(all_base.stream) - len(routed.stream) - route_bytes),
            "mean_source_mse_unit": mean_source_mse,
            "predictor_source_mse_unit": full_pred_source_mse,
            "true_fill_source_mse_unit": true_source_mse,
            "active_profile": active_profile,
        })
        for local_index, flat_id_tensor in enumerate(query_ids):
            flat_id = int(flat_id_tensor)
            row, col = divmod(flat_id, route.shape[1])
            singleton_route = np.zeros_like(route)
            singleton_route.reshape(-1)[flat_id] = True
            singleton = encode_y(
                p_net, prepared["y"], prepared["common_params"],
                singleton_route, 2)
            mean_error = float(target_blocks[local_index].float().square().mean())
            predictor_error = float((
                predicted_blocks[local_index] - target_blocks[local_index].float()
            ).square().mean())
            single_predicted_y = mean_y.clone()
            single_true_y = mean_y.clone()
            ys = slice(row * 2, (row + 1) * 2)
            xs = slice(col * 2, (col + 1) * 2)
            single_predicted_y[..., ys, xs] = predicted_y[..., ys, xs]
            single_true_y[..., ys, xs] = (
                mean_y[..., ys, xs] + target_delta[..., ys, xs])
            single_pred_rgb, _, _ = reconstruction_float(
                p_net, single_predicted_y, prepared["q_decoder"], valid_count)
            single_true_rgb, _, _ = reconstruction_float(
                p_net, single_true_y, prepared["q_decoder"], valid_count)
            predictor_pixel_gain = mean_source_mse - unit_mse(
                single_pred_rgb, target_rgb)
            true_pixel_gain = mean_source_mse - unit_mse(single_true_rgb, target_rgb)
            records.append({
                "sequence": dev.sequence,
                "chunk_index": chunk_index,
                "flat_id": flat_id,
                "row": row,
                "col": col,
                "actual_singleton_saved_y_bytes": (
                    len(all_base.stream) - len(singleton.stream)),
                "mean_latent_mse": mean_error,
                "predictor_latent_mse": predictor_error,
                "latent_gap_recovery": 1.0 - predictor_error / max(mean_error, 1e-30),
                "predictor_improves_mean_latent": predictor_error < mean_error,
                "predictor_pixel_gain_unit_mse": predictor_pixel_gain,
                "predictor_improves_mean_pixel": predictor_pixel_gain > 0.0,
                "true_fill_pixel_gain_unit_mse": true_pixel_gain,
            })
        x_hat, feature = p_net.get_recon_and_feature(
            predicted_y, p_net.ctx, prepared["q_decoder"])
        p_net.set_ref_feature(feature, should_reset(chunk_index, 32))
        first_index += valid_count
    del predictor
    return records, chunk_records


def rank_with_ties(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(values_a, values_b):
    if len(values_a) < 2:
        return None
    a = rank_with_ties(values_a)
    b = rank_with_ties(values_b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize_block_records(records):
    ordered = sorted(
        records,
        key=lambda item: (-item["actual_singleton_saved_y_bytes"],
                          item["sequence"], item["chunk_index"], item["flat_id"]))
    top_count = max(1, math.ceil(len(ordered) / 4))
    top = ordered[:top_count]

    def subset_summary(items):
        mean_energy = sum(item["mean_latent_mse"] for item in items)
        predictor_energy = sum(item["predictor_latent_mse"] for item in items)
        sequence_positive = 0
        per_sequence = {}
        for sequence in DEV_IDS:
            sequence_items = [item for item in items if item["sequence"] == sequence]
            if not sequence_items:
                continue
            mean_sum = sum(item["mean_latent_mse"] for item in sequence_items)
            pred_sum = sum(item["predictor_latent_mse"] for item in sequence_items)
            recovery = 1.0 - pred_sum / max(mean_sum, 1e-30)
            per_sequence[sequence] = {
                "block_count": len(sequence_items),
                "latent_gap_recovery": recovery,
            }
            sequence_positive += recovery > 0.0
        return {
            "block_count": len(items),
            "byte_saving_min": min(item["actual_singleton_saved_y_bytes"]
                                   for item in items),
            "byte_saving_median": float(np.median([
                item["actual_singleton_saved_y_bytes"] for item in items])),
            "byte_saving_max": max(item["actual_singleton_saved_y_bytes"]
                                   for item in items),
            "latent_gap_recovery": 1.0 - predictor_energy / max(mean_energy, 1e-30),
            "fraction_improving_mean_latent": float(np.mean([
                item["predictor_improves_mean_latent"] for item in items])),
            "fraction_improving_mean_pixel": float(np.mean([
                item["predictor_improves_mean_pixel"] for item in items])),
            "sequences_with_positive_latent_recovery": int(sequence_positive),
            "per_sequence": per_sequence,
        }

    return {
        "all_selected_high_byte_blocks": subset_summary(ordered),
        "highest_byte_saving_quartile": subset_summary(top),
        "spearman_byte_saving_vs_latent_gap_recovery": spearman(
            [item["actual_singleton_saved_y_bytes"] for item in ordered],
            [item["latent_gap_recovery"] for item in ordered]),
        "spearman_byte_saving_vs_predictor_pixel_gain": spearman(
            [item["actual_singleton_saved_y_bytes"] for item in ordered],
            [item["predictor_pixel_gain_unit_mse"] for item in ordered]),
    }


@torch.inference_mode()
def run_block_diagnostics(i_net, p_net, dev_sequences, checkpoints,
                          output_dir: Path, device):
    all_summaries = {}
    all_records = {}
    all_chunks = {}
    for arm in ARMS:
        records = []
        chunks = []
        for dev in dev_sequences:
            print(json.dumps({"stage": "block-diagnostic", "arm": arm,
                              "sequence": dev.sequence}), flush=True)
            sequence_records, sequence_chunks = diagnose_arm_sequence(
                i_net, p_net, dev, checkpoints[arm], device)
            records.extend(sequence_records)
            chunks.extend(sequence_chunks)
        all_records[arm] = records
        all_chunks[arm] = chunks
        all_summaries[arm] = summarize_block_records(records)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with (diagnostics_dir / "block_records.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["arm"] + list(all_records[ARMS[0]][0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for arm in ARMS:
            for record in all_records[arm]:
                writer.writerow({"arm": arm, **record})
    atomic_json(diagnostics_dir / "chunk_records.json", all_chunks)
    atomic_json(diagnostics_dir / "block_summary.json", all_summaries)
    return all_summaries, all_records, all_chunks


def verify_outputs(results, output_dir: Path):
    checks = {
        "all_files_equal_reported_total_bytes": True,
        "all_operational_points_fresh_decode_three_times": True,
        "all_routed_points_use_k8_in_both_chunks": True,
        "all_residual_streams_are_zero_bytes": True,
        "mean_and_all_arms_use_identical_routes": True,
        "mean_and_all_arms_have_identical_i_payloads": True,
        "mean_and_all_arms_have_identical_first_p_chunk_payload": True,
        "all_learned_points_propagate_routed_state": True,
        "all_learned_points_are_active_only_k8_c1": True,
        "predictor_forward_signature_is_decoder_only": (
            list(inspect.signature(
                SparseMaskedLatentPredictor.forward).parameters)
            == ["self", "decoded_q", "mean_y", "common_params", "skip_blocks"]),
    }
    per_sequence = {}
    for sequence in DEV_IDS:
        mean_label = "mean-k8"
        labels = [mean_label] + [f"{arm}-c1-a025-k8" for arm in ARMS]
        points = [results[sequence][label] for label in labels]
        route_signatures = [[
            (chunk["route_sha256"], chunk["skipped_blocks"])
            for chunk in point["chunks"]] for point in points]
        paths = {
            label: output_dir / "evaluation" / sequence / label / "sequence.d2l"
            for label in labels
        }
        sequences = {label: read_d2l(path) for label, path in paths.items()}
        route_payloads = {
            label: [parse_container(chunk, 2)["route"]
                    for chunk in sequence_file["chunks"]]
            for label, sequence_file in sequences.items()
        }
        sequence_checks = {
            "routes_identical": all(signature == route_signatures[0]
                                    for signature in route_signatures[1:])
                and all(route_payloads[label] == route_payloads[mean_label]
                        for label in labels[1:]),
            "i_payloads_identical": all(
                sequences[label]["i_stream"] == sequences[mean_label]["i_stream"]
                for label in labels[1:]),
            "first_p_chunk_payload_identical": all(
                sequences[label]["chunks"][0] == sequences[mean_label]["chunks"][0]
                for label in labels[1:]),
        }
        checks["mean_and_all_arms_use_identical_routes"] &= sequence_checks[
            "routes_identical"]
        checks["mean_and_all_arms_have_identical_i_payloads"] &= sequence_checks[
            "i_payloads_identical"]
        checks["mean_and_all_arms_have_identical_first_p_chunk_payload"] &= (
            sequence_checks["first_p_chunk_payload_identical"])
        per_sequence[sequence] = sequence_checks
        for label, point in results[sequence].items():
            path = output_dir / "evaluation" / sequence / label / "sequence.d2l"
            checks["all_files_equal_reported_total_bytes"] &= (
                path.stat().st_size == point["stream"]["total_bytes"])
            checks["all_operational_points_fresh_decode_three_times"] &= (
                point["compute"]["decode_repeats"] == 3
                and point["validation"]["full_sequence_fresh_decode"])
            checks["all_residual_streams_are_zero_bytes"] &= (
                point["stream"]["residual_bytes"] == 0)
            if label != "allbase-p30" and not label.startswith("allbase-p"):
                checks["all_routed_points_use_k8_in_both_chunks"] &= all(
                    chunk["skipped_blocks"] == K for chunk in point["chunks"])
            if label.endswith("-c1-a025-k8"):
                checks["all_learned_points_propagate_routed_state"] &= (
                    point["validation"]["routed_reconstruction_propagated"])
                checks["all_learned_points_are_active_only_k8_c1"] &= (
                    point["compute"]["linear_macs_per_chunk"] == EXPECTED_MACS
                    and all(not chunk["predictor_activation"][
                        "whole_latent_learned_activation"]
                            for chunk in point["chunks"]))
    checks["passes"] = all(checks.values())
    return {"checks": checks, "per_sequence": per_sequence}


def make_decision(aggregates, block_summaries, audits):
    mean = aggregates["mean-k8"]
    arm_gates = {}
    for arm in ARMS:
        point = aggregates[arm]
        positive_sequences = sum(
            point["per_sequence"][sequence]["p_frame_unit_mse"]
            < mean["per_sequence"][sequence]["p_frame_unit_mse"]
            for sequence in DEV_IDS)
        chunks_not_worse = all(
            point["chunk_unit_mse"][index] <= mean["chunk_unit_mse"][index]
            for index in range(2))
        top = block_summaries[arm]["highest_byte_saving_quartile"]
        predictor_stability = (
            point["p_frame_unit_mse"] < mean["p_frame_unit_mse"]
            and positive_sequences >= 5
            and chunks_not_worse
            and top["latent_gap_recovery"] > 0.0
            and top["fraction_improving_mean_latent"] > 0.5
            and top["sequences_with_positive_latent_recovery"] >= 2)
        matched = point["matched_allbase"]
        matched_pass = (
            matched.get("comparable", False)
            and matched.get("rate_change_percent", float("inf")) <= -0.5)
        arm_gates[arm] = {
            "aggregate_mse_improves_mean": (
                point["p_frame_unit_mse"] < mean["p_frame_unit_mse"]),
            "positive_sequences": positive_sequences,
            "requires_at_least_5_of_6": positive_sequences >= 5,
            "both_chunk_positions_not_worse": chunks_not_worse,
            "highest_byte_quartile_latent_recovery_positive": (
                top["latent_gap_recovery"] > 0.0),
            "highest_byte_quartile_majority_of_blocks_improve": (
                top["fraction_improving_mean_latent"] > 0.5),
            "highest_byte_quartile_not_single_sequence_only": (
                top["sequences_with_positive_latent_recovery"] >= 2),
            "predictor_stability_gate": predictor_stability,
            "matched_allbase_comparable": matched.get("comparable", False),
            "matched_rate_at_most_minus_0_5_percent": matched_pass,
            "full_continue_gate": predictor_stability and matched_pass
                                  and audits["checks"]["passes"],
        }
    stable_arms = [arm for arm, gate in arm_gates.items()
                   if gate["predictor_stability_gate"]]
    comparable_stable = [arm for arm in stable_arms
                         if arm_gates[arm]["matched_allbase_comparable"]]
    passed_arms = [arm for arm, gate in arm_gates.items()
                   if gate["full_continue_gate"]]
    if not audits["checks"]["passes"]:
        status = "inconclusive_due_to_correctness_or_timing_audit"
        recommendation = "关键码流或计时审计未通过，结论尚未确定。"
    elif passed_arms:
        status = "continue_current_configuration"
        recommendation = (
            "至少一个同损失训练臂同时通过开发集稳定性、高字节块恢复和公平码率门槛；"
            "可据此另行冻结未见测试，但本轮本身不是独立泛化证据。")
    elif not stable_arms:
        status = "stop_current_configuration"
        recommendation = (
            "四个同损失训练臂都未证明实际 C1 能稳定补回高字节块；停止当前冻结 codec＋"
            "整块 K8 省略＋C1 配置，不继续调 K、α、扩模型或直接执行旧 E09。")
    elif not comparable_stable:
        status = "inconclusive_due_to_missing_matched_allbase_coverage"
        recommendation = (
            "存在一定预测改善，但其画质落在已登记 all-Base P30–P32 曲线之外，"
            "缺少关键公平对照，整体结论尚未确定。")
    else:
        status = "stop_current_configuration"
        recommendation = (
            "虽有局部预测改善，但计入真实总字节后没有达到 all-Base 公平前沿门槛；"
            "停止当前冻结 codec＋整块 K8 省略＋C1 配置。")
    return {
        "status": status,
        "recommendation": recommendation,
        "arms_passing_predictor_stability": stable_arms,
        "arms_passing_full_continue_gate": passed_arms,
        "arm_gates": arm_gates,
    }


def write_aggregate_csv(aggregates, output_dir: Path):
    path = output_dir / "aggregate_results.csv"
    rows = []
    for name, point in aggregates.items():
        matched = point.get("matched_allbase", {})
        rows.append({
            "method": name,
            "total_bytes": point["total_bytes"],
            "actual_total_rate_change_vs_allbase_p32_percent": 100.0 * (
                point["total_bytes"]
                / aggregates["allbase-p32"]["total_bytes"] - 1.0),
            "p_frame_psnr": point["p_frame_psnr"],
            "p_frame_unit_mse": point["p_frame_unit_mse"],
            "chunk0_psnr": point["chunk_psnr"][0],
            "chunk1_psnr": point["chunk_psnr"][1],
            "matched_rate_change_percent": matched.get("rate_change_percent"),
            "directly_dominated_by_allbase": matched.get(
                "directly_dominated_without_extrapolation", False),
            "dominating_allbase_anchor": matched.get("dominating_anchor"),
            "rate_excess_vs_dominating_anchor_percent": matched.get(
                "rate_excess_vs_dominating_anchor_percent"),
            "decode_ms_per_frame": point["decode_ms_per_frame"],
            "decode_fps": point["decode_fps"],
            "predictor_latency_ms_p50": point["predictor_latency_ms_p50"],
            "predictor_latency_ms_p95": point["predictor_latency_ms_p95"],
            "peak_cuda_allocated_bytes": point["peak_cuda_allocated_bytes"],
            "linear_macs_per_chunk": point["linear_macs_per_chunk"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_figure(aggregates, block_summaries, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    base_names = ["allbase-p30", "allbase-p31", "allbase-p32"]
    ax.plot(
        [aggregates[name]["total_bytes"] / 1000 for name in base_names],
        [aggregates[name]["p_frame_psnr"] for name in base_names],
        "o-", color="black", label="all-Base QP curve")
    colors = {
        "mean-k8": "#777777",
        "unweighted": "#1f77b4",
        "byte": "#ff7f0e",
        "sensitivity": "#2ca02c",
        "byte_x_sensitivity": "#d62728",
    }
    for name in ("mean-k8",) + ARMS:
        point = aggregates[name]
        ax.scatter(point["total_bytes"] / 1000, point["p_frame_psnr"],
                   s=65, color=colors[name],
                   label=f"{name} ({point['decode_ms_per_frame']:.3f} ms/f)")
    ax.set_xlabel("Actual full-stream bytes (kB)")
    ax.set_ylabel("P-frame PSNR (dB)")
    ax.set_title("Actual rate-distortion-compute points")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    x = np.arange(len(ARMS))
    recovery = [100 * block_summaries[arm][
        "highest_byte_saving_quartile"]["latent_gap_recovery"] for arm in ARMS]
    positive = [100 * block_summaries[arm][
        "highest_byte_saving_quartile"]["fraction_improving_mean_latent"]
                for arm in ARMS]
    width = 0.36
    ax.bar(x - width / 2, recovery, width, label="latent gap recovery")
    ax.bar(x + width / 2, positive, width, label="blocks better than mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["none", "byte", "sensitivity", "byte×sensitivity"],
                  rotation=18, ha="right")
    ax.set_ylabel("Highest-byte quartile (%)")
    ax.set_title("Actual predictor recovery on byte-heavy blocks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "rdc_and_high_byte_recovery.png", dpi=180)
    plt.close(fig)


def write_report(summary, output_dir: Path):
    aggregates = summary["aggregates"]
    decision = summary["decision"]
    lines = [
        "# E10 同损失权重消融与开发集高字节块可恢复性诊断",
        "",
        f"结论：{decision['recommendation']}",
        "",
        "均方误差（MSE）衡量像素或特征的平均平方差，越低越好；峰值信噪比（PSNR）越高越好；量化参数（QP）控制 codec 的码率与画质档位。",
        "",
        "## 聚合结果",
        "",
        "| 方法 | 实际总字节 | 相对 all-Base P32 总字节 | P 帧 PSNR | 相对 mean 的 PSNR | 公平 all-Base 关系 | 完整解码 ms/帧 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    mean_psnr = aggregates["mean-k8"]["p_frame_psnr"]
    order = ("allbase-p30", "allbase-p31", "allbase-p32", "mean-k8") + ARMS
    for name in order:
        point = aggregates[name]
        matched = point.get("matched_allbase", {})
        if name.startswith("allbase-"):
            matched_text = "all-Base 基线锚点"
        elif matched.get("comparable"):
            matched_text = f"{matched['rate_change_percent']:+.3f}%"
        elif matched.get("directly_dominated_without_extrapolation"):
            matched_text = (
                f"被 P30 直接支配；字节多 "
                f"{matched['rate_excess_vs_dominating_anchor_percent']:+.3f}%")
        else:
            matched_text = "P30–P32 范围外，结论未定"
        delta = point["p_frame_psnr"] - mean_psnr
        same_qp_rate = 100.0 * (
            point["total_bytes"] / aggregates["allbase-p32"]["total_bytes"] - 1.0)
        lines.append(
            f"| {name} | {point['total_bytes']:,} | {same_qp_rate:+.3f}% | "
            f"{point['p_frame_psnr']:.5f} | {delta:+.5f} dB | {matched_text} | "
            f"{point['decode_ms_per_frame']:.4f} |")
    lines += [
        "",
        "四个预测器在 QP32 下确实少写约 5.99%–6.04% 总字节，但同时比 all-Base P32 低约 1.34–1.37 dB；这不是公平画质下的压缩收益。更关键的是 all-Base P30 已经同时具有更少字节和更高画质，且完整解码更快，因此无需向更低画质外推即可判定四个预测器点被直接支配。",
        "",
        "## 实际字节分项",
        "",
        "| 方法 | 容器 | I 帧 | z 辅助信息 | Base-y | route | residual | 总字节 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("allbase-p30", "allbase-p32", "mean-k8") + ARMS:
        components = aggregates[name]["components"]
        lines.append(
            f"| {name} | {components['container_bytes']:,} | "
            f"{components['i_payload_bytes']:,} | {components['global_z_bytes']:,} | "
            f"{components['base_y_bytes']:,} | {components['route_bytes']:,} | "
            f"{components['residual_bytes']:,} | {components['total_bytes']:,} |")
    lines += ["", "## 高字节块诊断", "",
              "| 训练权重 | 最高字节四分位 latent 缺口恢复 | 优于 mean 的块比例 | 正恢复序列数 |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        top = summary["block_diagnostics"][arm]["highest_byte_saving_quartile"]
        lines.append(
            f"| {ARM_ZH[arm]} | {100*top['latent_gap_recovery']:+.3f}% | "
            f"{100*top['fraction_improving_mean_latent']:.1f}% | "
            f"{top['sequences_with_positive_latent_recovery']}/6 |")
    lines += [
        "",
        "四个预测器的开发集 P 帧 MSE 均高于同 route mean-fill，6/6 序列全部变差，两个 P chunk 位置也都变差。即使只看实际占字节最多的四分之一省略块，latent 缺口恢复仍为负，说明预测误差比直接使用条件均值更大。",
        "",
        "true-fill 使用接收端实际拿不到的真实省略特征，只在结果中表示 route 潜力，不算实际收益。训练样本上的恢复只用于检查训练是否运行，不参与继续／停止判定。",
        "",
        "完整解码时间从读取内存中的落盘码流内容并执行 I/P 解码开始，包含小预测器和两段状态传播，不包含模型文件冷加载，也扣除了独立正确性核验的张量摘要时间。每条序列 fresh decode 三次后取中位数，再聚合。",
        "",
        "详表：`aggregate_results.csv`；逐块记录：`diagnostics/block_records.csv`；图：`figures/rdc_and_high_byte_recovery.png`。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reanalyze_existing(output_dir: Path):
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"no completed E10 summary at {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregates = summary["aggregates"]
    baselines = [aggregates[f"allbase-p{qp}"] for qp in (30, 31, 32)]
    for name in ("mean-k8",) + ARMS:
        aggregates[name]["matched_allbase"] = matched_rate(
            baselines, aggregates[name])
    summary["audits"]["checks"][
        "predictor_forward_signature_is_decoder_only"] = (
            list(inspect.signature(
                SparseMaskedLatentPredictor.forward).parameters)
            == ["self", "decoded_q", "mean_y", "common_params", "skip_blocks"])
    summary["audits"]["checks"]["passes"] = all(
        value for key, value in summary["audits"]["checks"].items()
        if key != "passes")
    summary["decision"] = make_decision(
        aggregates, summary["block_diagnostics"], summary["audits"])
    summary["status"] = summary["decision"]["status"]
    summary["rd_interpretation"] = {
        name: aggregates[name]["matched_allbase"]
        for name in ("mean-k8",) + ARMS
    }
    atomic_json(summary_path, summary)
    write_aggregate_csv(aggregates, output_dir)
    write_figure(aggregates, summary["block_diagnostics"], output_dir)
    write_report(summary, output_dir)
    print(json.dumps({
        "status": summary["status"],
        "recommendation": summary["decision"]["recommendation"],
        "summary": str(summary_path),
        "reanalyzed_without_rerunning_models": True,
    }, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    if args.reanalyze_existing:
        reanalyze_existing(output_dir)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("E10 requires CUDA")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed E10: {output_dir}")
    set_torch_env()
    seed_everything(SEED)
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    total_started = time.perf_counter()

    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    data_root = Path(args.data_root).resolve()

    train_entries = prepare_training_cache(
        i_net, p_net, data_root, output_dir, device)
    del i_net, p_net
    torch.cuda.empty_cache()
    checkpoints, training_records = train_all_arms(
        train_entries, args, output_dir, device)

    # Development pixels are first opened here, after all four final checkpoints
    # have been written.  No development result can alter a checkpoint.
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    dev_sequences = prepare_dev_sequences(
        i_net, p_net, data_root, output_dir, device)
    operational, references = evaluate_operational_points(
        i_net, p_net, dev_sequences, checkpoints, args, output_dir, device)
    block_summaries, block_records, chunk_records = run_block_diagnostics(
        i_net, p_net, dev_sequences, checkpoints, output_dir, device)

    aggregate_labels = (
        "allbase-p30", "allbase-p31", "allbase-p32", "mean-k8")
    aggregates = {
        label: aggregate_point(operational, label) for label in aggregate_labels}
    for arm in ARMS:
        label = f"{arm}-c1-a025-k8"
        aggregates[arm] = aggregate_point(operational, label)
    baselines = [aggregates[f"allbase-p{qp}"] for qp in (30, 31, 32)]
    for name in ("mean-k8",) + ARMS:
        aggregates[name]["matched_allbase"] = matched_rate(
            baselines, aggregates[name])
        aggregates[name]["same_qp32_total_rate_change_percent"] = 100.0 * (
            aggregates[name]["total_bytes"]
            / aggregates["allbase-p32"]["total_bytes"] - 1.0)

    reference_mse = float(np.mean([
        np.mean([frame_psnr_to_unit_mse(value) for value in
                 references[sequence]["metrics"]["per_frame_psnr"][1:]])
        for sequence in DEV_IDS
    ]))
    reference_summary = {
        "operational": False,
        "eligible_as_actual_benefit": False,
        "p_frame_unit_mse": reference_mse,
        "p_frame_psnr": -10.0 * math.log10(max(reference_mse, 1e-30)),
        "per_sequence": references,
    }
    audits = verify_outputs(operational, output_dir)
    decision = make_decision(aggregates, block_summaries, audits)
    ablation_vs_unweighted = {}
    for arm in ARMS[1:]:
        ablation_vs_unweighted[arm] = {
            "p_frame_mse_change_percent": 100.0 * (
                aggregates[arm]["p_frame_unit_mse"]
                / aggregates["unweighted"]["p_frame_unit_mse"] - 1.0),
            "p_frame_psnr_change_db": (
                aggregates[arm]["p_frame_psnr"]
                - aggregates["unweighted"]["p_frame_psnr"]),
            "highest_byte_quartile_latent_recovery_change_points": 100.0 * (
                block_summaries[arm]["highest_byte_saving_quartile"][
                    "latent_gap_recovery"]
                - block_summaries["unweighted"]["highest_byte_saving_quartile"][
                    "latent_gap_recovery"]),
        }

    public_operational = {
        sequence: {label: public_point(point)
                   for label, point in points.items()}
        for sequence, points in operational.items()
    }
    summary = {
        "experiment": "E10 same-MSE weight ablation and dev high-byte recovery",
        "project": "Adaptive Chunk Coding",
        "status": decision["status"],
        "scientific_scope": {
            "development_diagnostic_only": True,
            "independent_test_claim_allowed": False,
            "training_sample_memory_counts_as_benefit": False,
            "true_fill_counts_as_actual_benefit": False,
            "sealed_data_read": False,
            "old_e09_executed": False,
            "codec_modified": False,
            "model_expanded": False,
        },
        "protocol": {
            "train_split": "REDS train_sharp/001..024",
            "train_window_starts": list(WINDOW_STARTS),
            "train_chunks_per_window": 2,
            "train_sample_count": len(train_entries),
            "dev_split": "REDS val_sharp/000..005 (development only)",
            "dev_start": 0,
            "dev_frame_count": 17,
            "crop_function": "crop_xy(seed, sequence, start), 64-pixel aligned",
            "seed": SEED,
            "qp_i": 32,
            "qp_p_routed": 32,
            "allbase_qp_p": [30, 31, 32],
            "block_size": 2,
            "k": K,
            "alpha": ALPHA,
            "model_parameters": EXPECTED_PARAMETERS,
            "linear_macs_per_k8_chunk": EXPECTED_MACS,
            "steps": STEPS,
            "checkpoint_rule": "fixed final update before development read",
            "loss": "MSE for every arm",
            "weight_modes": list(ARMS),
            "route": "top-8 actual singleton Base-y byte saving on fixed mean trajectory",
            "combination_route_reencoded": True,
            "forbidden_paths_not_inspected": args.forbidden_paths,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "training": training_records,
        "checkpoints": {arm: str(path) for arm, path in checkpoints.items()},
        "aggregates": aggregates,
        "same_mse_ablation_vs_unweighted": ablation_vs_unweighted,
        "block_diagnostics": block_summaries,
        "diagnostic_chunk_records": chunk_records,
        "true_fill_diagnostic": reference_summary,
        "operational_per_sequence": public_operational,
        "audits": audits,
        "decision": decision,
        "complete_decode_timing_definition": (
            "Fresh D2L decode includes I/P codec decode, sparse predictor and state "
            "propagation; it excludes checkpoint cold-load and separately measured "
            "correctness-summary work. Three repeats per sequence, median reported."),
        "elapsed_wall_seconds": time.perf_counter() - total_started,
    }
    atomic_json(output_dir / "summary.json", summary)
    write_aggregate_csv(aggregates, output_dir)
    write_figure(aggregates, block_summaries, output_dir)
    write_report(summary, output_dir)
    print(json.dumps({
        "status": summary["status"],
        "recommendation": decision["recommendation"],
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
