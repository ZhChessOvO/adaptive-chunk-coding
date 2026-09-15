#!/usr/bin/env python3
"""E14: separate video diversity from sample-count effects for BasicVSR++.

The registered comparison keeps the frozen codec, frozen pretrained/random
BasicVSR++ backbones, latent head, route rule, optimizer, and 5,000 updates
fixed.  It contrasts 96 videos x one window (192 P chunks, matching E13's
sample count) with all 240 REDS train videos (000..239) x one window
(480 P chunks).
All four checkpoints are fixed before the already-used development cache is
loaded.  No sealed validation sequence is eligible.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo.stage_b_basicvsrpp_transfer_probe as e13  # noqa: E402
import demo.stage_b_mse_weight_ablation as e10  # noqa: E402
import demo.stage_b_pnp_transfer_probe as e12  # noqa: E402
from demo.stage_b_evaluate_predictor_g2 import source_route_target  # noqa: E402
from demo.stage_b_prepare_predictor_cache import crop_xy  # noqa: E402
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


ALL_TRAIN_IDS = tuple(f"{index:03d}" for index in range(240))
D96_IDS = tuple(f"{index:03d}" for index in range(1, 97))
CONFIG_IDS = {"d96": D96_IDS, "s240": ALL_TRAIN_IDS}
ARMS = ("pretrained", "random")
STEPS = 5000
EVAL_EVERY = 500
QP = 32
BLOCK_SIZE = 2
K = 8
HEAD_SEED = 2026091302
RANDOM_BACKBONE_SEED = e13.RANDOM_BACKBONE_SEED
E13_DEV_RECOVERY = -0.2145970911688173


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--e10-dir",
        default="output/stage_b_mse_weight_ablation_train001_024_val000_005_v1")
    parser.add_argument(
        "--e13-dir",
        default="output/stage_b_basicvsrpp_transfer_train001_024_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_basicvsrpp_data_scale_train001_240_val000_005_v1")
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
    return parser.parse_args()


def validate_args(args):
    if args.steps != STEPS:
        raise ValueError(f"E14 fixes {STEPS} updates for every head")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E14 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E14 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E14 fixes gradient clipping at 1.0")
    e13.validate_args(args)
    e13_dir = Path(args.e13_dir).resolve()
    required = (
        e13_dir / "feature_cache" / "train" / "manifest.jsonl",
        e13_dir / "feature_cache" / "development" / "manifest.jsonl",
        e13_dir / "feature_cache" / "development" / "complete.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)


def load_any_train_window(data_root: Path, sequence: str):
    if sequence not in ALL_TRAIN_IDS:
        raise ValueError(f"E14 only permits registered train/000..239: {sequence}")
    start = 0
    x, y = crop_xy(e10.SEED, sequence, start)
    root = data_root / "train_sharp" / sequence
    paths = sorted(root.glob("*.png"))[start:start + 17]
    if len(paths) != 17:
        raise ValueError(f"{root} does not provide the registered 17 frames")
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[y:y + 512, x:x + 512]
        if crop.shape != (512, 512, 3):
            raise ValueError(f"invalid crop ({x}, {y}) for {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return [str(path.resolve()) for path in paths], frames, (x, y)


def prior_entries(args):
    e10_entries = e12.e10_entries(Path(args.e10_dir).resolve())
    manifest = (Path(args.e13_dir).resolve() / "feature_cache" / "train" /
                "manifest.jsonl")
    features = {
        item["key"]: item
        for item in (json.loads(line) for line in manifest.read_text(
            encoding="utf-8").splitlines() if line.strip())
    }
    result = {}
    for sequence in e13.TRAIN_IDS:
        for chunk_index in range(2):
            key = f"train_{sequence}_s00_c{chunk_index}"
            result[key] = {
                "key": key,
                "sequence": sequence,
                "window_start": 0,
                "chunk_index": chunk_index,
                "latent_path": e10_entries[key]["path"],
                "feature_path": features[key]["feature_path"],
                "reused_from_e10_e13": True,
            }
    return result


def new_sample_path(output_dir: Path, key: str):
    return output_dir / "train_cache" / "samples" / f"{key}.safetensors"


def new_record_path(output_dir: Path, key: str):
    return output_dir / "train_cache" / "records" / f"{key}.json"


@torch.inference_mode()
def collect_new_sample(p_net, prepared, route, all_base, selected,
                       backbones, device):
    encoded = e12.encode_y(
        p_net, prepared["y"], prepared["common_params"], route, BLOCK_SIZE)
    decoded_q, mean_y = e12.decode_y(
        p_net, encoded.stream, encoded.ec_parallel,
        prepared["common_params"], route, BLOCK_SIZE)
    if not torch.equal(decoded_q, encoded.q_dense):
        raise RuntimeError("E14 Base-y entropy round trip mismatch")
    target_delta = source_route_target(
        p_net, prepared, decoded_q, route, BLOCK_SIZE)
    mean_rgb, _, ref_feature = e12.reconstruction_float(
        p_net, mean_y, prepared["q_decoder"], g_frame_delay)
    features = {
        arm: e13.basic_features(model, mean_rgb, device)
        for arm, model in backbones.items()
    }
    rate_map = torch.zeros((16, 16), dtype=torch.float32)
    selected_by_id = {item["flat_id"]: item for item in selected}
    for flat_id in np.flatnonzero(route.reshape(-1)):
        rate_map.reshape(-1)[int(flat_id)] = max(
            float(selected_by_id[int(flat_id)]["saved_y_bytes"]), 0.0)
    tensors = {
        "decoded_q": decoded_q.detach().cpu().to(torch.int8).contiguous(),
        "mean_y": mean_y.detach().cpu().to(torch.float16).contiguous(),
        "common_params": prepared["common_params"].detach().cpu().to(
            torch.float16).contiguous(),
        "target_delta": target_delta.detach().cpu().to(torch.float16).contiguous(),
        "rate_map": rate_map.contiguous(),
        "skip_blocks": torch.from_numpy(route.astype(np.uint8)).contiguous(),
        **features,
    }
    record = {
        "selected_blocks": [selected_by_id[int(index)]
                            for index in np.flatnonzero(route.reshape(-1))],
        "allbase_y_bytes": len(all_base.stream),
        "k8_y_bytes": len(encoded.stream),
        "combination_y_bytes_saved": len(all_base.stream) - len(encoded.stream),
        "route_bytes": len(e12.build_route_section(route, BLOCK_SIZE)),
    }
    record["combination_net_y_after_route_bytes_saved"] = (
        record["combination_y_bytes_saved"] - record["route_bytes"])
    return tensors, record, mean_y, ref_feature


@torch.inference_mode()
def prepare_all_train_cache(args, device):
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "train_cache"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    if manifest_path.is_file() and complete_path.is_file():
        entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (len(entries) != 480 or complete.get("sample_count") != 480
                or complete.get("sequence_ids") != list(ALL_TRAIN_IDS)):
            raise RuntimeError("incompatible E14 all-train cache")
        for entry in entries:
            if entry.get("sample_path"):
                if not Path(entry["sample_path"]).is_file():
                    raise RuntimeError(f"missing E14 sample: {entry['key']}")
            elif not (Path(entry["latent_path"]).is_file()
                      and Path(entry["feature_path"]).is_file()):
                raise RuntimeError(f"missing reused sample: {entry['key']}")
        return entries, complete["checkpoint_load_audit"]

    reused = prior_entries(args)
    entries = [reused[key] for key in sorted(reused)]
    i_net, p_net = e12.load_codec(args, device)
    backbones, audit = e13.make_backbones(args, device)
    started = time.perf_counter()
    data_root = Path(args.data_root).resolve()
    for sequence in ALL_TRAIN_IDS:
        if sequence in e13.TRAIN_IDS:
            continue
        source_files, frames, crop = load_any_train_window(data_root, sequence)
        _, _, i_hat = e12.encode_i(i_net, frames[0], device)
        e12.initialize_p_state(p_net, i_hat)
        first_index = 1
        for chunk_index in range(2):
            key = f"train_{sequence}_s00_c{chunk_index}"
            sample_path = new_sample_path(output_dir, key).resolve()
            record_path = new_record_path(output_dir, key)
            chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
            prepared = e12.prepare_chunk_latents(p_net, chunk, QP)
            if sample_path.is_file() and record_path.is_file():
                stored = load_file(sample_path, device=str(device))
                route = stored["skip_blocks"].cpu().numpy().astype(np.bool_)
                encoded = e12.encode_y(
                    p_net, prepared["y"], prepared["common_params"], route,
                    BLOCK_SIZE)
                decoded_q, mean_y = e12.decode_y(
                    p_net, encoded.stream, encoded.ec_parallel,
                    prepared["common_params"], route, BLOCK_SIZE)
                if (not torch.equal(decoded_q.cpu(), stored["decoded_q"].cpu())
                        or not torch.equal(mean_y.cpu(), stored["mean_y"].cpu())):
                    raise RuntimeError(f"resumed E14 sample disagrees: {key}")
                _, _, ref_feature = e12.reconstruction_float(
                    p_net, mean_y, prepared["q_decoder"], valid_count)
                record = json.loads(record_path.read_text(encoding="utf-8"))
            else:
                route, all_base, _, selected = e10.select_high_byte_route(
                    p_net, prepared)
                tensors, diagnostics, mean_y, ref_feature = collect_new_sample(
                    p_net, prepared, route, all_base, selected,
                    backbones, device)
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, sample_path, metadata={
                    "split": "train",
                    "source": "decoder-visible inputs; true latent is label only",
                })
                record = {
                    "key": key,
                    "sequence": sequence,
                    "window_start": 0,
                    "chunk_index": chunk_index,
                    "crop_xy": list(crop),
                    "source_files": source_files,
                    "route_trajectory": "fixed mean-fill",
                    "route_rule": "top-8 actual singleton Base-y byte saving",
                    **diagnostics,
                }
                e12.atomic_json(record_path, record)
            entries.append({
                "key": key,
                "sequence": sequence,
                "window_start": 0,
                "chunk_index": chunk_index,
                "sample_path": str(sample_path),
                "record_path": str(record_path.resolve()),
                "reused_from_e10_e13": False,
            })
            p_net.set_ref_feature(
                ref_feature, e12.should_reset(chunk_index, 32))
            first_index += valid_count
        print(json.dumps({
            "stage": "all-train-cache", "sequence": sequence,
            "samples": len(entries)}), flush=True)

    entries.sort(key=lambda item: item["key"])
    if len(entries) != 480:
        raise RuntimeError(f"E14 expected 480 samples, got {len(entries)}")
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in entries),
        encoding="utf-8")
    os.replace(temporary, manifest_path)
    complete = {
        "format": "e14_basicvsrpp_all_train_start0_v1",
        "split": "REDS train_sharp/000..239",
        "sequence_ids": list(ALL_TRAIN_IDS),
        "window_starts": [0],
        "chunks_per_window": 2,
        "sample_count": len(entries),
        "reused_samples_001_024": 48,
        "new_samples_000_and_025_239": 432,
        "trajectory": "fixed mean-fill K8",
        "route": "top-8 actual singleton Base-y byte saving",
        "checkpoint_load_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_or_sealed_data_read": False,
    }
    e12.atomic_json(complete_path, complete)
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return entries, audit


@torch.inference_mode()
def ensure_stable_features(args, entries, load_audit, device):
    """Re-extract every E14 feature with bfloat16 before head training.

    Half precision is stable on E13's first 24 videos but overflows on a
    subset of the full REDS train split.  Bfloat16 retains float32's exponent
    range and the tensor-core execution path.  Applying one uniform rule to all
    480 samples avoids both cherry-picking and train/eval dtype mismatch.
    """
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "feature_cache_bfloat16"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    if manifest_path.is_file() and complete_path.is_file():
        feature_entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        by_key = {item["key"]: item for item in feature_entries}
        if (len(by_key) != 480 or complete.get("sample_count") != 480
                or complete.get("extraction_dtype") != "bfloat16"
                or complete.get("checkpoint_load_audit") != load_audit
                or not all(Path(item["feature_stable_path"]).is_file()
                           for item in feature_entries)):
            raise RuntimeError("incompatible E14 bfloat16 feature cache")
        return [{**entry, **by_key[entry["key"]]} for entry in entries]

    entry_by_key = {entry["key"]: entry for entry in entries}
    i_net, p_net = e12.load_codec(args, device)
    backbones, audit = e13.make_backbones(args, device)
    if audit != load_audit:
        raise RuntimeError("checkpoint audit changed before float32 extraction")
    started = time.perf_counter()
    feature_entries = []
    data_root = Path(args.data_root).resolve()
    maxima = {arm: 0.0 for arm in ARMS}
    for sequence in ALL_TRAIN_IDS:
        _, frames, _ = load_any_train_window(data_root, sequence)
        _, _, i_hat = e12.encode_i(i_net, frames[0], device)
        e12.initialize_p_state(p_net, i_hat)
        first_index = 1
        for chunk_index in range(2):
            key = f"train_{sequence}_s00_c{chunk_index}"
            entry = entry_by_key[key]
            if entry.get("sample_path"):
                static = load_file(entry["sample_path"], device="cpu")
            else:
                static = load_file(entry["latent_path"], device="cpu")
            route = static["skip_blocks"].numpy().astype(np.bool_)
            chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
            prepared = e12.prepare_chunk_latents(p_net, chunk, QP)
            encoded = e12.encode_y(
                p_net, prepared["y"], prepared["common_params"], route,
                BLOCK_SIZE)
            decoded_q, mean_y = e12.decode_y(
                p_net, encoded.stream, encoded.ec_parallel,
                prepared["common_params"], route, BLOCK_SIZE)
            if (not torch.equal(decoded_q.cpu(), static["decoded_q"])
                    or not torch.equal(mean_y.cpu(), static["mean_y"])):
                raise RuntimeError(f"E14 float32 replay disagrees: {key}")
            mean_rgb, _, ref_feature = e12.reconstruction_float(
                p_net, mean_y, prepared["q_decoder"], valid_count)
            features = {
                arm: e13.basic_features(
                    model, mean_rgb, device, use_autocast=True,
                    output_dtype=torch.bfloat16,
                    autocast_dtype=torch.bfloat16)
                for arm, model in backbones.items()
            }
            for arm, feature in features.items():
                if not bool(torch.isfinite(feature).all()):
                    raise RuntimeError(f"non-finite bfloat16 feature: {key}/{arm}")
                maxima[arm] = max(maxima[arm], float(feature.abs().max()))
            path = (cache_dir / "samples" / f"{key}.safetensors").resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(features, path, metadata={
                "split": "train",
                "extraction_dtype": "bfloat16",
                "source": "decoder-visible mean-fill RGB only",
            })
            feature_entries.append({
                "key": key,
                "feature_stable_path": str(path),
                "feature_dtype": "bfloat16",
            })
            p_net.set_ref_feature(
                ref_feature, e12.should_reset(chunk_index, 32))
            first_index += valid_count
        print(json.dumps({
            "stage": "bfloat16-feature-cache", "sequence": sequence,
            "samples": len(feature_entries)}), flush=True)
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item) + "\n" for item in feature_entries), encoding="utf-8")
    os.replace(temporary, manifest_path)
    e12.atomic_json(complete_path, {
        "format": "e14_basicvsrpp_all_train_bfloat16_features_v1",
        "sample_count": len(feature_entries),
        "sequence_ids": list(ALL_TRAIN_IDS),
        "extraction_dtype": "bfloat16",
        "stored_dtype": "bfloat16",
        "feature_shape": [g_frame_delay, 64, 32, 32],
        "max_abs_feature": maxima,
        "checkpoint_load_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
        "development_or_sealed_data_read": False,
    })
    by_key = {item["key"]: item for item in feature_entries}
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return [{**entry, **by_key[entry["key"]]} for entry in entries]


def load_samples(entries, arm, device):
    samples = []
    for entry in entries:
        if "feature_stable_path" not in entry:
            raise RuntimeError("E14 training requires uniform bfloat16 features")
        if entry.get("sample_path"):
            stored = load_file(entry["sample_path"], device=str(device))
            sample = {
                key: stored[key]
                for key in ("decoded_q", "mean_y", "common_params",
                            "target_delta", "rate_map", "skip_blocks")
            }
        else:
            latent = load_file(entry["latent_path"], device=str(device))
            sample = {**latent}
        features = load_file(entry["feature_stable_path"], device=str(device))
        sample["video_features"] = features[arm]
        sample.update({
            "sequence": entry["sequence"],
            "window_start": 0,
            "chunk_index": entry["chunk_index"],
        })
        samples.append(sample)
    return samples


def train_candidate(name, arm, samples, initial_state, args, output_dir, device):
    checkpoint_path = output_dir / "checkpoints" / f"{name}_{arm}_head_final.pt"
    summary_path = output_dir / "training" / f"{name}_{arm}_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != "e14_basicvsrpp_data_scale_head_v1"
                or payload.get("candidate") != name
                or payload.get("arm") != arm
                or payload.get("steps") != STEPS):
            raise RuntimeError(f"incompatible E14 checkpoint: {name}/{arm}")
        return checkpoint_path, json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists() or summary_path.exists():
        raise RuntimeError(f"partial E14 training output: {name}/{arm}")

    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = e12.make_schedule(len(samples))
    history = [{"step": 0, **e12.latent_summary(model, samples)["overall"]}]
    log_path = output_dir / "training" / f"{name}_{arm}.jsonl"
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
            raise RuntimeError(f"non-finite gradient: {name}/{arm}")
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
            print(json.dumps({
                "stage": "train", "candidate": name, "arm": arm, **event}),
                flush=True)
            model.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    train_metrics = e12.latent_summary(model, samples)
    summary = {
        "candidate": name,
        "arm": arm,
        "steps": STEPS,
        "sample_count": len(samples),
        "sequence_count": len(CONFIG_IDS[name]),
        "train_metrics": train_metrics,
        "history": history,
        "training_wall_seconds": elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    e12.atomic_checkpoint(checkpoint_path, {
        "format": "e14_basicvsrpp_data_scale_head_v1",
        "candidate": name,
        "arm": arm,
        "steps": STEPS,
        "head_seed": HEAD_SEED,
        "state_dict": {key: value.detach().cpu()
                       for key, value in model.state_dict().items()},
        "development_not_loaded": True,
        "training": summary,
    })
    e12.atomic_json(summary_path, summary)
    return checkpoint_path, summary


def train_all_candidates(args, entries, output_dir, device):
    e12.seed_everything(HEAD_SEED)
    initial = e12.PnPLatentHead().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    checkpoints = {}
    summaries = {}
    for name, ids in CONFIG_IDS.items():
        selected = [entry for entry in entries if entry["sequence"] in ids]
        if len(selected) != len(ids) * 2:
            raise RuntimeError(f"wrong sample count for {name}: {len(selected)}")
        checkpoints[name] = {}
        summaries[name] = {}
        for arm in ARMS:
            samples = load_samples(selected, arm, device)
            checkpoint, summary = train_candidate(
                name, arm, samples, initial_state, args, output_dir, device)
            checkpoints[name][arm] = checkpoint
            summaries[name][arm] = summary
            del samples
            torch.cuda.empty_cache()
    del initial, initial_state
    e12.atomic_json(output_dir / "checkpoint_lock_before_development.json", {
        "all_four_final_checkpoints_fixed_before_development_cache_load": True,
        "candidates": {
            name: {
                "train_ids": [ids[0], ids[-1]],
                "sample_count": len(ids) * 2,
                "checkpoints": {arm: str(path.resolve())
                                for arm, path in checkpoints[name].items()},
            }
            for name, ids in CONFIG_IDS.items()
        },
        "development_ids_not_loaded_during_training": list(e13.DEV_IDS),
        "fixed_at_unix_time": time.time(),
    })
    return checkpoints, summaries


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def load_dev_entries_after_lock(args, output_dir):
    if not (output_dir / "checkpoint_lock_before_development.json").is_file():
        raise RuntimeError("development cache load attempted before checkpoint lock")
    e13_dir = Path(args.e13_dir).resolve()
    complete = json.loads((e13_dir / "feature_cache" / "development" /
                           "complete.json").read_text(encoding="utf-8"))
    if (complete.get("sequence_ids") != list(e13.DEV_IDS)
            or complete.get("sample_count") != 12
            or complete.get("sealed_data_read") is not False):
        raise RuntimeError("incompatible E13 development cache")
    manifest = e13_dir / "feature_cache" / "development" / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != 12:
        raise RuntimeError("E14 expected 12 development samples")
    return entries, complete["checkpoint_load_audit"]


@torch.inference_mode()
def ensure_development_stable_features(args, entries, load_audit, output_dir,
                                       device):
    cache_dir = output_dir / "feature_cache_bfloat16" / "development"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    if manifest_path.is_file() and complete_path.is_file():
        feature_entries = [json.loads(line) for line in manifest_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        by_key = {item["key"]: item for item in feature_entries}
        if (len(by_key) != 12 or complete.get("sample_count") != 12
                or complete.get("extraction_dtype") != "bfloat16"
                or complete.get("checkpoint_load_audit") != load_audit
                or not all(Path(item["feature_stable_path"]).is_file()
                           for item in feature_entries)):
            raise RuntimeError("incompatible E14 development float32 cache")
        return [{**entry, **by_key[entry["key"]]} for entry in entries]

    entry_by_key = {entry["key"]: entry for entry in entries}
    i_net, p_net = e12.load_codec(args, device)
    backbones, audit = e13.make_backbones(args, device)
    if audit != load_audit:
        raise RuntimeError("checkpoint audit changed for development bfloat16 features")
    feature_entries = []
    maxima = {arm: 0.0 for arm in ARMS}
    e10_dir = Path(args.e10_dir).resolve()
    for sequence in e13.DEV_IDS:
        plan = e13.load_dev_plan(e10_dir, sequence)
        _, frames, _ = e12.load_window(
            Path(args.data_root), "val_sharp", sequence, 0)
        _, _, i_hat = e12.encode_i(i_net, frames[0], device)
        e12.initialize_p_state(p_net, i_hat)
        first_index = 1
        for chunk_index, route_record in enumerate(plan["chunks"]):
            key = f"val_{sequence}_s00_c{chunk_index}"
            static = load_file(entry_by_key[key]["sample_path"], device="cpu")
            route = e13.route_from_record(route_record)
            chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
            prepared = e12.prepare_chunk_latents(p_net, chunk, QP)
            encoded = e12.encode_y(
                p_net, prepared["y"], prepared["common_params"], route,
                BLOCK_SIZE)
            decoded_q, mean_y = e12.decode_y(
                p_net, encoded.stream, encoded.ec_parallel,
                prepared["common_params"], route, BLOCK_SIZE)
            if (not torch.equal(decoded_q.cpu(), static["decoded_q"])
                    or not torch.equal(mean_y.cpu(), static["mean_y"])):
                raise RuntimeError(f"E14 development replay disagrees: {key}")
            mean_rgb, _, ref_feature = e12.reconstruction_float(
                p_net, mean_y, prepared["q_decoder"], valid_count)
            features = {
                arm: e13.basic_features(
                    model, mean_rgb, device, use_autocast=True,
                    output_dtype=torch.bfloat16,
                    autocast_dtype=torch.bfloat16)
                for arm, model in backbones.items()
            }
            for arm, feature in features.items():
                if not bool(torch.isfinite(feature).all()):
                    raise RuntimeError(f"non-finite dev bfloat16 feature: {key}/{arm}")
                maxima[arm] = max(maxima[arm], float(feature.abs().max()))
            path = (cache_dir / "samples" / f"{key}.safetensors").resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(features, path, metadata={
                "split": "development", "extraction_dtype": "bfloat16"})
            feature_entries.append({
                "key": key, "feature_stable_path": str(path),
                "feature_dtype": "bfloat16",
            })
            p_net.set_ref_feature(
                ref_feature, e12.should_reset(chunk_index, 32))
            first_index += valid_count
        print(json.dumps({
            "stage": "development-bfloat16-feature-cache", "sequence": sequence}),
            flush=True)
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item) + "\n" for item in feature_entries), encoding="utf-8")
    os.replace(temporary, manifest_path)
    e12.atomic_json(complete_path, {
        "format": "e14_basicvsrpp_development_bfloat16_features_v1",
        "sample_count": 12,
        "sequence_ids": list(e13.DEV_IDS),
        "extraction_dtype": "bfloat16",
        "stored_dtype": "bfloat16",
        "max_abs_feature": maxima,
        "checkpoint_load_audit": audit,
        "all_e14_checkpoints_locked_before_source_read": True,
        "sealed_data_read": False,
    })
    by_key = {item["key"]: item for item in feature_entries}
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return [{**entry, **by_key[entry["key"]]} for entry in entries]


def load_development_samples_stable(entries, arm, device):
    samples = []
    for entry in entries:
        static = load_file(entry["sample_path"], device=str(device))
        features = load_file(entry["feature_stable_path"], device=str(device))
        samples.append({
            "decoded_q": static["decoded_q"],
            "mean_y": static["mean_y"],
            "common_params": static["common_params"],
            "target_delta": static["target_delta"],
            "skip_blocks": static["skip_blocks"],
            "rate_map": static["rate_map"],
            "video_features": features[arm],
            "sequence": entry["sequence"],
            "window_start": 0,
            "chunk_index": entry["chunk_index"],
        })
    return samples


def evaluate_development(checkpoints, dev_entries, output_dir, device):
    result = {}
    records = []
    for name in CONFIG_IDS:
        result[name] = {}
        for arm in ARMS:
            model = load_head(checkpoints[name][arm], device)
            samples = load_development_samples_stable(dev_entries, arm, device)
            summary = e12.latent_summary(model, samples, include_records=True)
            for record in summary.pop("records"):
                records.append({"candidate": name, "arm": arm, **record})
            result[name][arm] = summary
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


@torch.inference_mode()
def development_rgb(args, checkpoints, output_dir, device):
    i_net, p_net = e12.load_codec(args, device)
    backbones, _ = e13.make_backbones(args, device)
    heads = {
        f"{name}_{arm}": load_head(checkpoints[name][arm], device)
        for name in CONFIG_IDS for arm in ARMS
    }
    methods = ("mean",) + tuple(heads)
    rows = []
    latency = defaultdict(list)
    e10_dir = Path(args.e10_dir).resolve()
    for sequence in e13.DEV_IDS:
        plan = e13.load_dev_plan(e10_dir, sequence)
        _, frames, _ = e12.load_window(
            Path(args.data_root), "val_sharp", sequence, 0)
        i_stream, _, i_hat = e12.encode_i(i_net, frames[0], device)
        for method in methods:
            e12.initialize_p_state(p_net, i_hat)
            first_index = 1
            total_error = 0.0
            pixel_count = 0
            chunk_mse = []
            chunk_bytes = []
            for chunk_index, route_record in enumerate(plan["chunks"]):
                route = e13.route_from_record(route_record)
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
                    _, arm = method.split("_", 1)
                    mean_rgb, _, _ = e12.reconstruction_float(
                        p_net, mean_y, prepared["q_decoder"], valid_count)
                    torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    features = e13.basic_features(
                        backbones[arm], mean_rgb, device, move_to_cpu=False,
                        use_autocast=True, output_dtype=torch.bfloat16,
                        autocast_dtype=torch.bfloat16)
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
                total_error += squared
                pixel_count += count
                chunk_mse.append(squared / count)
                route_section = e12.build_route_section(route, BLOCK_SIZE)
                chunk_bytes.append(
                    e12.CONTAINER_HEADER.size + len(prepared["global_stream"])
                    + len(route_section) + len(encoded.stream))
                p_net.set_ref_feature(
                    ref_feature, e12.should_reset(chunk_index, 32))
                first_index += valid_count
            mse = total_error / pixel_count
            rows.append({
                "method": method,
                "sequence": sequence,
                "p_frame_mse": mse,
                "p_frame_psnr": -10 * math.log10(max(mse, 1e-30)),
                "chunk0_mse": chunk_mse[0],
                "chunk1_mse": chunk_mse[1],
                "component_counted_sequence_bytes": (
                    e12.SEQ_HEADER.size + len(i_stream)
                    + len(chunk_bytes) * e12.CHUNK_LENGTH.size
                    + sum(chunk_bytes)),
            })
        print(json.dumps({
            "stage": "development-rgb", "sequence": sequence}), flush=True)

    aggregate = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        mse = float(np.mean([row["p_frame_mse"] for row in selected]))
        times = latency.get(method, [])
        aggregate[method] = {
            "p_frame_mse": mse,
            "p_frame_psnr": -10 * math.log10(max(mse, 1e-30)),
            "chunk0_mse": float(np.mean([row["chunk0_mse"] for row in selected])),
            "chunk1_mse": float(np.mean([row["chunk1_mse"] for row in selected])),
            "component_counted_bytes": sum(
                row["component_counted_sequence_bytes"] for row in selected),
            "backbone_plus_head_ms_per_chunk_median": (
                float(np.median(times)) if times else 0.0),
            "backbone_plus_head_ms_per_chunk_p95": (
                float(np.percentile(times, 95)) if times else 0.0),
            "positive_vs_mean_sequences": None,
        }
    for method in heads:
        aggregate[method]["positive_vs_mean_sequences"] = sum(
            row["p_frame_mse"] < next(
                item["p_frame_mse"] for item in rows
                if item["method"] == "mean" and item["sequence"] == row["sequence"])
            for row in rows if row["method"] == method)
    evaluation = output_dir / "evaluation"
    with (evaluation / "development_rgb_windows.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    e12.atomic_json(evaluation / "development_rgb_summary.json", aggregate)
    del heads, backbones, i_net, p_net
    torch.cuda.empty_cache()
    return aggregate


def classify(development_latent, development_rgb):
    points = {}
    for name in CONFIG_IDS:
        pre = development_latent[name]["pretrained"]
        random_arm = development_latent[name]["random"]
        recovery = pre["overall"]["latent_gap_recovery"]
        rgb = development_rgb[f"{name}_pretrained"]
        points[name] = {
            "development_latent_recovery": recovery,
            "change_vs_e13_percentage_points": 100 * (recovery - E13_DEV_RECOVERY),
            "pretrained_beats_random_latent": (
                pre["overall"]["prediction_mse"]
                < random_arm["overall"]["prediction_mse"]),
            "development_rgb_delta_vs_mean_db": (
                rgb["p_frame_psnr"] - development_rgb["mean"]["p_frame_psnr"]),
            "development_rgb_positive_sequences": rgb["positive_vs_mean_sequences"],
        }
    d96_helped = points["d96"]["change_vs_e13_percentage_points"] >= 3.0
    s240_helped = points["s240"]["change_vs_e13_percentage_points"] >= 3.0
    s240_passed = (
        points["s240"]["development_latent_recovery"] > 0
        and points["s240"]["development_rgb_delta_vs_mean_db"] > 0
        and points["s240"]["development_rgb_positive_sequences"] >= 5)
    if s240_passed:
        status = "data_scale_development_signal_passed"
        recommendation = (
            "全 train 训练在开发集的 latent 与 RGB 都转正；下一步补真实落盘、"
            "all-Base 调 QP、公平增强和完整解码耗时。")
    elif d96_helped or s240_helped:
        status = "data_diversity_or_scale_helped_but_not_passed"
        recommendation = (
            "更多训练视频带来可见改善，但尚未稳定胜过 mean-fill；保留路线，下一步"
            "做受控适配器微调，并继续把 validation 当开发集。")
    else:
        status = "data_scale_did_not_resolve_generalization"
        recommendation = (
            "在固定 5,000 步下，扩大训练视频多样性和样本量都没有带来至少 3 个百分点"
            "的开发恢复改善。单靠继续加冻结特征数据不够；下一步应测试小范围骨干微调"
            "或重新设计预测目标，而不是把训练拟合当收益。")
    return {
        "status": status,
        "points": points,
        "d96_helped_by_registered_3pp_rule": d96_helped,
        "s240_helped_by_registered_3pp_rule": s240_helped,
        "s240_passed_development": s240_passed,
        "recommendation": recommendation,
        "actual_net_benefit_claim_allowed": False,
        "sealed_data_should_be_read": False,
    }


def write_report(summary, output_dir):
    lines = [
        "# E14 训练视频多样性与样本量复核",
        "",
        f"结论：{summary['decision']['recommendation']}",
        "",
        "| 训练方案 | 骨干 | train latent 恢复 | validation latent 恢复 | 高字节四分位 | 第二段 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in CONFIG_IDS:
        for arm in ARMS:
            train = summary["training"][name][arm]["train_metrics"]
            dev = summary["development_latent"][name][arm]
            lines.append(
                f"| {name} | {arm} | {100*train['overall']['latent_gap_recovery']:+.3f}% | "
                f"{100*dev['overall']['latent_gap_recovery']:+.3f}% | "
                f"{100*dev['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
                f"{100*dev['per_chunk_position']['1']['latent_gap_recovery']:+.3f}% |")
    rgb = summary["development_rgb"]
    lines += [
        "",
        "## validation RGB 闭环",
        "",
        "| 方法 | P 帧 PSNR | 相对 mean-fill | 正改善视频 | 组件计数字节 | 额外中位耗时/段 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    mean_psnr = rgb["mean"]["p_frame_psnr"]
    for method, point in rgb.items():
        positive = "—" if method == "mean" else f"{point['positive_vs_mean_sequences']}/6"
        lines.append(
            f"| {method} | {point['p_frame_psnr']:.5f} dB | "
            f"{point['p_frame_psnr']-mean_psnr:+.5f} dB | {positive} | "
            f"{point['component_counted_bytes']:,} | "
            f"{point['backbone_plus_head_ms_per_chunk_median']:.2f} ms |")
    lines += [
        "",
        "## 边界",
        "",
        "- train/000..239 只用于梯度和训练集诊断；训练拟合不算实际收益。",
        "- validation/000..005 已在 E10/E13 使用，本轮只是开发迭代，不是独立测试。",
        "- 码流字节是内存内真实熵编码组件相加，未做落盘 fresh decode。",
        "- 未读取 val/006..029 或任何封存数据，未使用 true-fill 作为结果。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("E14 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries, load_audit = prepare_all_train_cache(args, device)
    entries = ensure_stable_features(args, entries, load_audit, device)
    checkpoints, training = train_all_candidates(
        args, entries, output_dir, device)
    dev_entries, dev_audit = load_dev_entries_after_lock(args, output_dir)
    if dev_audit != load_audit:
        raise RuntimeError("BasicVSR++ load audit changed between train and development")
    dev_entries = ensure_development_stable_features(
        args, dev_entries, dev_audit, output_dir, device)
    development_latent = evaluate_development(
        checkpoints, dev_entries, output_dir, device)
    development_rgb_result = development_rgb(
        args, checkpoints, output_dir, device)
    decision = classify(development_latent, development_rgb_result)
    summary = {
        "experiment": "E14 BasicVSR++ train diversity and sample scale probe",
        "status": decision["status"],
        "scientific_scope": {
            "d96": "REDS train_sharp/001..096, start 0, 192 P chunks",
            "s240": "REDS train_sharp/000..239, start 0, 480 P chunks",
            "train_metrics_are_actual_benefit": False,
            "development": "REDS val_sharp/000..005, already-used development set",
            "sealed_data_read": False,
        },
        "protocol": {
            "steps_per_head": STEPS,
            "head_parameters": e12.PnPLatentHead().parameter_count,
            "frozen_backbone_parameters": e13.BACKBONE_PARAMETERS,
            "loss": "unweighted direct-delta MSE",
            "qp_i": QP,
            "qp_p": QP,
            "k": K,
            "route": "top-8 actual singleton Base-y byte saving",
            "all_checkpoints_fixed_before_development_cache_load": True,
        },
        "official_asset": {
            "model": "BasicVSR++ NTIRE 2021 compressed-video enhancement Track 1",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
            "checkpoint_load_audit": load_audit,
            "license": "Apache-2.0",
        },
        "training": training,
        "development_latent": development_latent,
        "development_rgb": development_rgb_result,
        "decision": decision,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
