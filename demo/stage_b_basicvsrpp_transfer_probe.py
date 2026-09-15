#!/usr/bin/env python3
"""E13: all-train BasicVSR++ transfer probe with train/dev evaluation.

The released NTIRE compressed-video BasicVSR++ backbone and an identically
shaped random backbone are frozen.  Both feed the same small latent head.  The
head is fitted on every registered REDS train/001..024 sample, then its final
checkpoint is locked before REDS val/000..005 is loaded.  Train metrics are a
learnability diagnostic only; development metrics measure generalization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo.stage_b_pnp_transfer_probe as e12  # noqa: E402
from demo.stage_b_evaluate_predictor_g2 import source_route_target  # noqa: E402
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


TRAIN_IDS = tuple(f"{index:03d}" for index in range(1, 25))
DEV_IDS = tuple(f"{index:03d}" for index in range(6))
WINDOW_STARTS = (0, 24, 48, 72)
ARMS = ("pretrained", "random")
STEPS = 5000
EVAL_EVERY = 500
QP = 32
BLOCK_SIZE = 2
K = 8
HEAD_SEED = 2026091302
RANDOM_BACKBONE_SEED = 2026091303
BACKBONE_PARAMETERS = 44_075_631


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--e10-dir",
        default="output/stage_b_mse_weight_ablation_train001_024_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_basicvsrpp_transfer_train001_024_val000_005_v1")
    parser.add_argument("--mmagic-root", default="demo")
    parser.add_argument(
        "--checkpoint",
        default=("third_party/mmagic/checkpoints/"
                 "basicvsr_plusplus_c128n25_ntire_decompress_track1_"
                 "20210223-7b2eba02.pth"))
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--skip-rgb-diagnostic", action="store_true",
        help="Only for smoke testing; the registered run keeps RGB diagnostics.")
    return parser.parse_args()


def validate_args(args):
    if args.steps != STEPS:
        raise ValueError(f"E13 fixes the update count at {STEPS}")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E13 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E13 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E13 fixes gradient clipping at 1.0")
    adapter = Path(args.mmagic_root).resolve() / "standalone_basicvsrpp.py"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)


def load_adapter(mmagic_root: Path):
    path = mmagic_root / "standalone_basicvsrpp.py"
    spec = importlib.util.spec_from_file_location("e13_basicvsrpp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_backbones(args, device):
    module = load_adapter(Path(args.mmagic_root).resolve())
    pretrained = module.BasicVSRPlusPlusFeatureBackbone()
    audit = pretrained.load_released_checkpoint(args.checkpoint)
    random_model = module.make_random_backbone(RANDOM_BACKBONE_SEED)
    models = {"pretrained": pretrained, "random": random_model}
    for model in models.values():
        model.eval().requires_grad_(False).to(device)
    if any(parameter.requires_grad for model in models.values()
           for parameter in model.parameters()):
        raise RuntimeError("E13 backbones must remain frozen")
    counts = {name: sum(p.numel() for p in model.parameters())
              for name, model in models.items()}
    if set(counts.values()) != {BACKBONE_PARAMETERS}:
        raise RuntimeError(f"unexpected BasicVSR++ parameter counts: {counts}")
    return models, audit


def basic_features(backbone, mean_rgb, device, move_to_cpu=True,
                   use_autocast=True, output_dtype=torch.float16,
                   autocast_dtype=torch.float16):
    resized = F.interpolate(
        mean_rgb.float(), size=(256, 256), mode="bilinear", align_corners=False)
    frames = resized.reshape(1, g_frame_delay, 3, 256, 256)
    if use_autocast:
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=autocast_dtype):
            features = backbone.forward_features(frames).squeeze(0)
    else:
        with torch.inference_mode():
            features = backbone.forward_features(frames).squeeze(0)
    features = features.detach().to(dtype=output_dtype).contiguous()
    return features.cpu() if move_to_cpu else features


def train_feature_path(output_dir: Path, key: str):
    return output_dir / "feature_cache" / "train" / "samples" / f"{key}.safetensors"


def dev_sample_path(output_dir: Path, key: str):
    return output_dir / "feature_cache" / "development" / "samples" / f"{key}.safetensors"


@torch.inference_mode()
def prepare_train_features(args, base_entries, device):
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "feature_cache" / "train"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    if manifest_path.is_file() and complete_path.is_file():
        entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (len(entries) != 192 or complete.get("sample_count") != 192
                or complete.get("sequence_ids") != list(TRAIN_IDS)
                or not all(Path(entry["feature_path"]).is_file()
                           for entry in entries)):
            raise RuntimeError("incompatible E13 train feature cache")
        return entries, complete["checkpoint_load_audit"]

    i_net, p_net = e12.load_codec(args, device)
    backbones, audit = make_backbones(args, device)
    entries = []
    started = time.perf_counter()
    for sequence in TRAIN_IDS:
        for start in WINDOW_STARTS:
            _, frames, crop = e12.load_window(
                Path(args.data_root), "train_sharp", sequence, start)
            _, _, i_hat = e12.encode_i(i_net, frames[0], device)
            e12.initialize_p_state(p_net, i_hat)
            first_index = 1
            for chunk_index in range(2):
                key = f"train_{sequence}_s{start:02d}_c{chunk_index}"
                base = load_file(base_entries[key]["path"], device=str(device))
                route = base["skip_blocks"].cpu().numpy().astype(np.bool_)
                chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
                prepared = e12.prepare_chunk_latents(p_net, chunk, QP)
                routed = e12.encode_y(
                    p_net, prepared["y"], prepared["common_params"], route,
                    BLOCK_SIZE)
                decoded_q, mean_y = e12.decode_y(
                    p_net, routed.stream, routed.ec_parallel,
                    prepared["common_params"], route, BLOCK_SIZE)
                if (not torch.equal(decoded_q.cpu(), base["decoded_q"].cpu())
                        or not torch.equal(mean_y.cpu(), base["mean_y"].cpu())):
                    raise RuntimeError(f"E13 train replay disagrees with E10: {key}")
                mean_rgb, _, ref_feature = e12.reconstruction_float(
                    p_net, mean_y, prepared["q_decoder"], valid_count)
                features = {
                    arm: basic_features(model, mean_rgb, device)
                    for arm, model in backbones.items()
                }
                path = train_feature_path(output_dir, key)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(features, path, metadata={
                    "split": "train",
                    "source": "decoder-visible mean-fill RGB only",
                })
                entries.append({
                    "key": key,
                    "split": "train",
                    "sequence": sequence,
                    "window_start": start,
                    "chunk_index": chunk_index,
                    "crop_xy": list(crop),
                    "latent_path": base_entries[key]["path"],
                    "feature_path": str(path.resolve()),
                    "feature_shape": [g_frame_delay, 64, 32, 32],
                    "source_or_true_latent_in_backbone_input": False,
                })
                p_net.set_ref_feature(
                    ref_feature, e12.should_reset(chunk_index, 32))
                first_index += valid_count
            print(json.dumps({
                "stage": "train-feature-cache", "sequence": sequence,
                "window_start": start, "samples": len(entries)}), flush=True)
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in entries),
        encoding="utf-8")
    os.replace(temporary, manifest_path)
    complete = {
        "format": "e13_basicvsrpp_train_features_v1",
        "split": "REDS train_sharp/001..024",
        "sequence_ids": list(TRAIN_IDS),
        "window_starts": list(WINDOW_STARTS),
        "chunks_per_window": 2,
        "sample_count": len(entries),
        "trajectory": "fixed E10 mean-fill K8",
        "backbone_input": "decoder-visible mean-fill RGB only",
        "input_resolution": [256, 256],
        "feature_shape": [g_frame_delay, 64, 32, 32],
        "checkpoint_load_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
        "development_or_sealed_data_read": False,
    }
    e12.atomic_json(complete_path, complete)
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return entries, audit


def load_train_samples(entries, arm, device):
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


def train_one_arm(arm, samples, initial_state, args, output_dir, device):
    checkpoint_path = output_dir / "checkpoints" / f"{arm}_head_final.pt"
    summary_path = output_dir / "training" / f"{arm}_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != "e13_basicvsrpp_latent_head_v1"
                or payload.get("arm") != arm or payload.get("steps") != STEPS):
            raise RuntimeError(f"incompatible existing E13 checkpoint: {arm}")
        return checkpoint_path, json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists() or summary_path.exists():
        raise RuntimeError(f"partial E13 training output: {arm}")

    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = e12.make_schedule(len(samples))
    initial = e12.latent_summary(model, samples)["overall"]
    history = [{"step": 0, **initial}]
    log_path = output_dir / "training" / f"{arm}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history[0]) + "\n", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step, index in enumerate(schedule, 1):
        prediction, target, _ = e12.sample_prediction(model, samples[index])
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
                **e12.latent_summary(model, samples)["overall"],
            }
            history.append(event)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps({"stage": "train", "arm": arm, **event}), flush=True)
            model.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    metrics = e12.latent_summary(model, samples)
    summary = {
        "arm": arm,
        "steps": STEPS,
        "loss": "unweighted direct-delta MSE",
        "checkpoint_rule": "fixed final update before development tensors are read",
        "head_parameters": model.parameter_count,
        "frozen_backbone_parameters": BACKBONE_PARAMETERS,
        "train_metrics": metrics,
        "history": history,
        "training_wall_seconds": elapsed,
        "updates_per_second": STEPS / elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    e12.atomic_checkpoint(checkpoint_path, {
        "format": "e13_basicvsrpp_latent_head_v1",
        "arm": arm,
        "steps": STEPS,
        "head_seed": HEAD_SEED,
        "random_backbone_seed": RANDOM_BACKBONE_SEED if arm == "random" else None,
        "state_dict": {key: value.detach().cpu()
                       for key, value in model.state_dict().items()},
        "data": {
            "train": "REDS train_sharp/001..024, starts 0/24/48/72",
            "development_not_loaded": True,
        },
        "training": summary,
    })
    e12.atomic_json(summary_path, summary)
    return checkpoint_path, summary


def paired_training(args, train_entries, output_dir, device):
    e12.seed_everything(HEAD_SEED)
    initial = e12.PnPLatentHead().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    checkpoints = {}
    summaries = {}
    for arm in ARMS:
        samples = load_train_samples(train_entries, arm, device)
        checkpoint, summary = train_one_arm(
            arm, samples, initial_state, args, output_dir, device)
        checkpoints[arm] = checkpoint
        summaries[arm] = summary
        del samples
        torch.cuda.empty_cache()
    del initial, initial_state
    lock = {
        "all_final_checkpoints_fixed_before_development_tensor_load": True,
        "all_registered_train_ids_used_for_fitting": list(TRAIN_IDS),
        "development_ids_not_loaded_during_training": list(DEV_IDS),
        "checkpoints": {arm: str(path.resolve())
                        for arm, path in checkpoints.items()},
        "fixed_at_unix_time": time.time(),
    }
    e12.atomic_json(output_dir / "checkpoint_lock_before_development.json", lock)
    return checkpoints, summaries


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def evaluate_train_latents(checkpoints, entries, output_dir, device):
    result = {}
    for arm in ARMS:
        model = load_head(checkpoints[arm], device)
        samples = load_train_samples(entries, arm, device)
        result[arm] = e12.latent_summary(model, samples)
        del model, samples
        torch.cuda.empty_cache()
    e12.atomic_json(output_dir / "evaluation" / "train_latent_summary.json", result)
    return result


def load_dev_plan(e10_dir: Path, sequence: str):
    path = e10_dir / "dev_route_plans" / f"val_{sequence}.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("sequence") != sequence or len(plan.get("chunks", [])) != 2
            or plan.get("route_trajectory") != "fixed mean-fill"):
        raise RuntimeError(f"invalid registered E10 route plan: {path}")
    return plan


def route_from_record(record):
    route = np.zeros((16, 16), dtype=np.bool_)
    route.reshape(-1)[record["selected_flat_ids"]] = True
    if int(route.sum()) != K:
        raise RuntimeError("development route is not K8")
    return route


@torch.inference_mode()
def prepare_development_cache(args, device):
    output_dir = Path(args.output_dir).resolve()
    if not (output_dir / "checkpoint_lock_before_development.json").is_file():
        raise RuntimeError("development access attempted before checkpoint lock")
    cache_dir = output_dir / "feature_cache" / "development"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    if manifest_path.is_file() and complete_path.is_file():
        entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (len(entries) != 12 or complete.get("sample_count") != 12
                or complete.get("sequence_ids") != list(DEV_IDS)
                or not all(Path(entry["sample_path"]).is_file()
                           for entry in entries)):
            raise RuntimeError("incompatible E13 development cache")
        return entries, complete["checkpoint_load_audit"]

    i_net, p_net = e12.load_codec(args, device)
    backbones, audit = make_backbones(args, device)
    entries = []
    e10_dir = Path(args.e10_dir).resolve()
    started = time.perf_counter()
    for sequence in DEV_IDS:
        plan = load_dev_plan(e10_dir, sequence)
        _, frames, crop = e12.load_window(
            Path(args.data_root), "val_sharp", sequence, 0)
        _, _, i_hat = e12.encode_i(i_net, frames[0], device)
        e12.initialize_p_state(p_net, i_hat)
        first_index = 1
        for chunk_index, route_record in enumerate(plan["chunks"]):
            key = f"val_{sequence}_s00_c{chunk_index}"
            route = route_from_record(route_record)
            chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
            prepared = e12.prepare_chunk_latents(p_net, chunk, QP)
            routed = e12.encode_y(
                p_net, prepared["y"], prepared["common_params"], route,
                BLOCK_SIZE)
            decoded_q, mean_y = e12.decode_y(
                p_net, routed.stream, routed.ec_parallel,
                prepared["common_params"], route, BLOCK_SIZE)
            target_delta = source_route_target(
                p_net, prepared, decoded_q, route, BLOCK_SIZE)
            mean_rgb, _, ref_feature = e12.reconstruction_float(
                p_net, mean_y, prepared["q_decoder"], valid_count)
            features = {
                arm: basic_features(model, mean_rgb, device)
                for arm, model in backbones.items()
            }
            rate_map = torch.zeros((16, 16), dtype=torch.float32)
            for block in route_record["selected_singletons"]:
                rate_map.reshape(-1)[block["flat_id"]] = block["saved_y_bytes"]
            tensors = {
                "decoded_q": decoded_q.detach().cpu().contiguous(),
                "mean_y": mean_y.detach().cpu().to(torch.float16).contiguous(),
                "common_params": prepared["common_params"].detach().cpu().to(
                    torch.float16).contiguous(),
                "target_delta": target_delta.detach().cpu().to(
                    torch.float16).contiguous(),
                "skip_blocks": torch.from_numpy(
                    route.astype(np.uint8)).contiguous(),
                "rate_map": rate_map.contiguous(),
                **features,
            }
            path = dev_sample_path(output_dir, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, path, metadata={
                "split": "development",
                "source": "decoder-visible inputs; true latent is label only",
            })
            entries.append({
                "key": key,
                "split": "development",
                "sequence": sequence,
                "window_start": 0,
                "chunk_index": chunk_index,
                "crop_xy": list(crop),
                "sample_path": str(path.resolve()),
                "feature_shape": [g_frame_delay, 64, 32, 32],
                "route_plan": str((e10_dir / "dev_route_plans" /
                                   f"val_{sequence}.json").resolve()),
                "source_or_true_latent_in_backbone_input": False,
            })
            p_net.set_ref_feature(
                ref_feature, e12.should_reset(chunk_index, 32))
            first_index += valid_count
        print(json.dumps({
            "stage": "development-cache", "sequence": sequence,
            "samples": len(entries)}), flush=True)
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in entries),
        encoding="utf-8")
    os.replace(temporary, manifest_path)
    complete = {
        "format": "e13_basicvsrpp_development_cache_v1",
        "split": "REDS val_sharp/000..005 development only",
        "sequence_ids": list(DEV_IDS),
        "window_starts": [0],
        "chunks_per_window": 2,
        "sample_count": len(entries),
        "trajectory": "registered E10 fixed mean-fill K8 routes",
        "checkpoint_load_audit": audit,
        "checkpoint_locked_before_any_development_source_read": True,
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_data_read": False,
    }
    e12.atomic_json(complete_path, complete)
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return entries, audit


def load_development_samples(entries, arm, device):
    samples = []
    for entry in entries:
        tensors = load_file(entry["sample_path"], device=str(device))
        samples.append({
            "decoded_q": tensors["decoded_q"],
            "mean_y": tensors["mean_y"],
            "common_params": tensors["common_params"],
            "target_delta": tensors["target_delta"],
            "skip_blocks": tensors["skip_blocks"],
            "rate_map": tensors["rate_map"],
            "video_features": tensors[arm],
            "sequence": entry["sequence"],
            "window_start": entry["window_start"],
            "chunk_index": entry["chunk_index"],
        })
    return samples


def evaluate_development_latents(checkpoints, entries, output_dir, device):
    result = {}
    records = []
    for arm in ARMS:
        model = load_head(checkpoints[arm], device)
        samples = load_development_samples(entries, arm, device)
        summary = e12.latent_summary(model, samples, include_records=True)
        for record in summary.pop("records"):
            records.append({"arm": arm, **record})
        result[arm] = summary
        del model, samples
        torch.cuda.empty_cache()
    evaluation = output_dir / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    e12.atomic_json(evaluation / "development_latent_summary.json", result)
    with (evaluation / "development_block_records.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return result


def fixed_route(split, sequence, start, chunk_index, base_entries, e10_dir):
    if split == "train":
        key = f"train_{sequence}_s{start:02d}_c{chunk_index}"
        static = load_file(base_entries[key]["path"], device="cpu")
        return static["skip_blocks"].numpy().astype(np.bool_)
    plan = load_dev_plan(e10_dir, sequence)
    return route_from_record(plan["chunks"][chunk_index])


@torch.inference_mode()
def rgb_diagnostic(args, split, checkpoints, base_entries, output_dir, device):
    sequence_ids = TRAIN_IDS if split == "train" else DEV_IDS
    starts = WINDOW_STARTS if split == "train" else (0,)
    source_split = "train_sharp" if split == "train" else "val_sharp"
    i_net, p_net = e12.load_codec(args, device)
    backbones, _ = make_backbones(args, device)
    heads = {arm: load_head(checkpoints[arm], device) for arm in ARMS}
    methods = ("mean",) + ARMS
    rows = []
    latency = defaultdict(list)
    e10_dir = Path(args.e10_dir).resolve()
    for sequence in sequence_ids:
        for start in starts:
            _, frames, _ = e12.load_window(
                Path(args.data_root), source_split, sequence, start)
            i_stream, _, i_hat = e12.encode_i(i_net, frames[0], device)
            for method in methods:
                e12.initialize_p_state(p_net, i_hat)
                first_index = 1
                total_squared_error = 0.0
                pixel_count = 0
                chunk_mse = []
                chunk_bytes = []
                for chunk_index in range(2):
                    route = fixed_route(
                        split, sequence, start, chunk_index, base_entries, e10_dir)
                    chunk, target_rgb, valid_count = e12.make_chunk(
                        frames, first_index, device)
                    prepared = e12.prepare_chunk_latents(p_net, chunk, QP)
                    encoded = e12.encode_y(
                        p_net, prepared["y"], prepared["common_params"], route,
                        BLOCK_SIZE)
                    decoded_q, mean_y = e12.decode_y(
                        p_net, encoded.stream, encoded.ec_parallel,
                        prepared["common_params"], route, BLOCK_SIZE)
                    if method == "mean":
                        predicted_y = mean_y
                    else:
                        mean_rgb, _, _ = e12.reconstruction_float(
                            p_net, mean_y, prepared["q_decoder"], valid_count)
                        torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        features = basic_features(
                            backbones[method], mean_rgb, device, move_to_cpu=False)
                        predicted_y = heads[method].apply(
                            decoded_q, mean_y, prepared["common_params"], route,
                            features)
                        torch.cuda.synchronize(device)
                        latency[method].append(
                            1000.0 * (time.perf_counter() - tick))
                    rgb, _, ref_feature = e12.reconstruction_float(
                        p_net, predicted_y, prepared["q_decoder"], valid_count)
                    squared = float((rgb.float() - target_rgb).square().sum())
                    count = target_rgb.numel()
                    total_squared_error += squared
                    pixel_count += count
                    chunk_mse.append(squared / count)
                    route_section = e12.build_route_section(route, BLOCK_SIZE)
                    chunk_bytes.append(
                        e12.CONTAINER_HEADER.size + len(prepared["global_stream"])
                        + len(route_section) + len(encoded.stream))
                    p_net.set_ref_feature(
                        ref_feature, e12.should_reset(chunk_index, 32))
                    first_index += valid_count
                mse = total_squared_error / pixel_count
                rows.append({
                    "split": split,
                    "method": method,
                    "sequence": sequence,
                    "window_start": start,
                    "p_frame_mse": mse,
                    "p_frame_psnr": -10.0 * math.log10(max(mse, 1e-30)),
                    "chunk0_mse": chunk_mse[0],
                    "chunk1_mse": chunk_mse[1],
                    "component_counted_sequence_bytes": (
                        e12.SEQ_HEADER.size + len(i_stream)
                        + len(chunk_bytes) * e12.CHUNK_LENGTH.size
                        + sum(chunk_bytes)),
                })
            print(json.dumps({
                "stage": f"{split}-rgb-diagnostic", "sequence": sequence,
                "window_start": start}), flush=True)

    aggregate = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        mse = float(np.mean([row["p_frame_mse"] for row in selected]))
        per_sequence = {}
        for sequence in sequence_ids:
            subset = [row for row in selected if row["sequence"] == sequence]
            sequence_mse = float(np.mean([row["p_frame_mse"] for row in subset]))
            per_sequence[sequence] = {
                "p_frame_mse": sequence_mse,
                "p_frame_psnr": -10.0 * math.log10(max(sequence_mse, 1e-30)),
            }
        times = latency.get(method, [])
        aggregate[method] = {
            "windows": len(selected),
            "p_frame_mse": mse,
            "p_frame_psnr": -10.0 * math.log10(max(mse, 1e-30)),
            "chunk0_mse": float(np.mean([row["chunk0_mse"] for row in selected])),
            "chunk1_mse": float(np.mean([row["chunk1_mse"] for row in selected])),
            "component_counted_bytes": sum(
                row["component_counted_sequence_bytes"] for row in selected),
            "backbone_plus_head_ms_per_chunk_median": (
                float(np.median(times)) if times else 0.0),
            "backbone_plus_head_ms_per_chunk_p95": (
                float(np.percentile(times, 95)) if times else 0.0),
            "per_sequence": per_sequence,
        }
    evaluation = output_dir / "evaluation"
    with (evaluation / f"{split}_rgb_windows.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    e12.atomic_json(evaluation / f"{split}_rgb_summary.json", aggregate)
    del heads, backbones, i_net, p_net
    torch.cuda.empty_cache()
    return aggregate


def classify(train_latent, dev_latent, train_rgb, dev_rgb):
    train_pre = train_latent["pretrained"]
    dev_pre = dev_latent["pretrained"]
    dev_random = dev_latent["random"]
    train_learned = (
        train_pre["overall"]["latent_gap_recovery"] > 0.30
        and train_pre["highest_byte_quartile"]["latent_gap_recovery"] > 0)
    positive_dev_sequences = sum(
        item["latent_gap_recovery"] > 0
        for item in dev_pre["per_sequence"].values())
    beats_random_sequences = sum(
        dev_pre["per_sequence"][sequence]["prediction_mse"]
        < dev_random["per_sequence"][sequence]["prediction_mse"]
        for sequence in DEV_IDS)
    latent_generalizes = (
        dev_pre["overall"]["latent_gap_recovery"] > 0
        and dev_pre["highest_byte_quartile"]["latent_gap_recovery"] > 0
        and dev_pre["per_chunk_position"]["1"]["latent_gap_recovery"] > 0
        and dev_pre["overall"]["prediction_mse"]
        < dev_random["overall"]["prediction_mse"]
        and positive_dev_sequences >= 5 and beats_random_sequences >= 4)
    rgb_generalizes = False
    if dev_rgb is not None:
        rgb_positive = sum(
            dev_rgb["pretrained"]["per_sequence"][sequence]["p_frame_mse"]
            < dev_rgb["mean"]["per_sequence"][sequence]["p_frame_mse"]
            for sequence in DEV_IDS)
        rgb_generalizes = (
            dev_rgb["pretrained"]["p_frame_mse"] < dev_rgb["mean"]["p_frame_mse"]
            and dev_rgb["pretrained"]["p_frame_mse"]
            < dev_rgb["random"]["p_frame_mse"]
            and dev_rgb["pretrained"]["chunk1_mse"]
            <= dev_rgb["mean"]["chunk1_mse"]
            and rgb_positive >= 5)
    else:
        rgb_positive = None
    if latent_generalizes and rgb_generalizes:
        status = "development_signal_passed_needs_full_controls"
        recommendation = (
            "训练与开发集都出现稳定正恢复；下一步补真实落盘、all-Base 调 QP、"
            "公平增强和完整重新解码耗时，补齐前不能声称真实净收益。")
    elif train_learned:
        status = "learnable_but_not_generalizing"
        recommendation = (
            "全部 train 上能学会、开发集没有稳定复现，当前主要矛盾是泛化而非头部完全"
            "训不动。保留思路，下一步优先增加已登记训练视频、受控微调与正则化；"
            "现阶段不把训练拟合写成收益。")
    else:
        status = "not_learned_under_fixed_probe"
        recommendation = (
            "全部 train 上也没有形成足够恢复，当前固定骨干接口或目标设计本身更可疑；"
            "先改目标/接口或换更适合部分传输的 codec，不宜直接靠租卡放大。")
    return {
        "status": status,
        "train_learnability_gate": train_learned,
        "development_latent_gate": latent_generalizes,
        "development_rgb_gate": rgb_generalizes,
        "positive_development_sequences": positive_dev_sequences,
        "pretrained_beats_random_development_sequences": beats_random_sequences,
        "positive_development_rgb_sequences": rgb_positive,
        "recommendation": recommendation,
        "actual_net_benefit_claim_allowed": False,
        "sealed_data_should_be_read": False,
    }


def write_report(summary, output_dir):
    train = summary["train_latent"]
    dev = summary["development_latent"]
    lines = [
        "# E13 BasicVSR++：全部 train 训练，train/validation 双评估",
        "",
        f"结论：{summary['decision']['recommendation']}",
        "",
        "训练结果只诊断能否学会；validation/000..005 是已使用的开发集，不是封存测试。",
        "",
        "## Latent（压缩特征）恢复",
        "",
        "| 骨干 | train 恢复 | train 高字节四分位 | validation 恢复 | validation 高字节四分位 | validation 正恢复视频 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        positive = sum(x["latent_gap_recovery"] > 0
                       for x in dev[arm]["per_sequence"].values())
        lines.append(
            f"| {arm} | {100*train[arm]['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*train[arm]['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev[arm]['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev[arm]['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
            f"{positive}/6 |")
    for split, title in (("train_rgb", "train"), ("development_rgb", "validation")):
        rgb = summary.get(split)
        if rgb is None:
            continue
        lines += [
            "",
            f"## RGB 闭环：{title}",
            "",
            "| 方法 | P 帧 PSNR | 相对 mean-fill | 组件计数字节 | 额外骨干+头中位耗时/段 |",
            "|---|---:|---:|---:|---:|",
        ]
        mean_psnr = rgb["mean"]["p_frame_psnr"]
        for method in ("mean",) + ARMS:
            point = rgb[method]
            lines.append(
                f"| {method} | {point['p_frame_psnr']:.5f} dB | "
                f"{point['p_frame_psnr']-mean_psnr:+.5f} dB | "
                f"{point['component_counted_bytes']:,} | "
                f"{point['backbone_plus_head_ms_per_chunk_median']:.2f} ms |")
    lines += [
        "",
        "## 结果边界",
        "",
        "- BasicVSR++ 与 DCVC 均冻结；只训练 934,864 参数的 latent 修正头。",
        "- 预训练与随机骨干同结构、同头部初值、同训练顺序和同 5,000 次更新。",
        "- validation 在两个最终 checkpoint 锁定后才读取，没有用来选步数或调参数。",
        "- RGB 字节来自真实熵编码组件相加，但没有落盘后完整重新解码，只作闭环诊断。",
        "- 未读取 train/025..240、val/006..029 或任何封存数据；true-fill 不计结果。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("E13 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_entries = e12.e10_entries(Path(args.e10_dir).resolve())

    # Strict order: all train work -> checkpoint lock -> development read.
    train_entries, load_audit = prepare_train_features(args, base_entries, device)
    checkpoints, training = paired_training(
        args, train_entries, output_dir, device)
    train_latent = evaluate_train_latents(
        checkpoints, train_entries, output_dir, device)
    dev_entries, dev_audit = prepare_development_cache(args, device)
    if dev_audit != load_audit:
        raise RuntimeError("checkpoint load audit changed before development")
    dev_latent = evaluate_development_latents(
        checkpoints, dev_entries, output_dir, device)

    train_rgb = None
    dev_rgb = None
    if not args.skip_rgb_diagnostic:
        train_rgb = rgb_diagnostic(
            args, "train", checkpoints, base_entries, output_dir, device)
        dev_rgb = rgb_diagnostic(
            args, "development", checkpoints, base_entries, output_dir, device)
    decision = classify(train_latent, dev_latent, train_rgb, dev_rgb)
    summary = {
        "experiment": "E13 frozen BasicVSR++ all-train transfer probe",
        "status": decision["status"],
        "scientific_scope": {
            "fit_split": "REDS train_sharp/001..024, starts 0/24/48/72",
            "train_evaluation_role": "learnability and memorization diagnostic only",
            "development_split": "REDS val_sharp/000..005, start 0",
            "development_is_independent_test": False,
            "sealed_data_read": False,
            "true_fill_counts_as_result": False,
            "codec_modified": False,
        },
        "protocol": {
            "train_samples": 192,
            "development_samples": 12,
            "chunks_per_window": 2,
            "qp_i": QP,
            "qp_p": QP,
            "block_size": BLOCK_SIZE,
            "k": K,
            "steps": STEPS,
            "loss": "unweighted direct-delta MSE",
            "head_seed": HEAD_SEED,
            "random_backbone_seed": RANDOM_BACKBONE_SEED,
            "checkpoint_fixed_before_development_read": True,
        },
        "official_asset": {
            "repository": "https://github.com/open-mmlab/mmagic",
            "model": "BasicVSR++ NTIRE 2021 compressed-video enhancement Track 1",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
            "checkpoint_load_audit": load_audit,
            "license": "Apache-2.0",
            "compatibility": (
                "PyTorch-only adapter using torchvision deform_conv2d; no mmcv install or environment downgrade"),
        },
        "model": {
            "frozen_backbone_parameters": BACKBONE_PARAMETERS,
            "head_parameters": e12.PnPLatentHead().parameter_count,
            "input": "8 decoder-visible mean-fill RGB frames resized to 256x256",
            "feature": "8x64x32x32",
        },
        "training": training,
        "train_latent": train_latent,
        "development_latent": dev_latent,
        "train_rgb": train_rgb,
        "development_rgb": dev_rgb,
        "decision": decision,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torchvision": __import__("torchvision").__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    e12.atomic_json(output_dir / "summary.json", summary)
    write_report(summary, output_dir)
    print(json.dumps({
        "stage": "complete", "status": decision["status"],
        "summary": str((output_dir / "summary.json").resolve()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
