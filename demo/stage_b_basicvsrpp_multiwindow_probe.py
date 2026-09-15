#!/usr/bin/env python3
"""E16: complete-train, four-window BasicVSR++ generalization probe.

All REDS train/000..239 videos contribute windows starting at 0, 24, 48,
and 72 (1,920 eight-frame P chunks).  The codec and pretrained/random
BasicVSR++ backbones remain frozen.  Each unchanged latent head receives
50,000 updates, preserving E15's 26.04 updates per sample.  Final checkpoints
are locked before the already-used development split is loaded.
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
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo.stage_b_basicvsrpp_data_scale_probe as e14  # noqa: E402
import demo.stage_b_basicvsrpp_equal_exposure_probe as e15  # noqa: E402
import demo.stage_b_basicvsrpp_transfer_probe as e13  # noqa: E402
import demo.stage_b_mse_weight_ablation as e10  # noqa: E402
import demo.stage_b_pnp_transfer_probe as e12  # noqa: E402
import demo.stage1_token_skipping as stage1  # noqa: E402
from demo.stage_b_evaluate_predictor_g2 import source_route_target  # noqa: E402
from demo.stage_b_prepare_predictor_cache import crop_xy  # noqa: E402
from src.models.video_model_ht import g_frame_delay  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


TRAIN_IDS = tuple(f"{index:03d}" for index in range(240))
WINDOW_STARTS = (0, 24, 48, 72)
ARMS = ("pretrained", "random")
SAMPLE_COUNT = 1_920
STEPS = 50_000
EVAL_EVERY = 2_500
HEAD_SEED = e14.HEAD_SEED
E14_BEST_DEV_RECOVERY = -0.0528556
E14_BEST_RGB_DELTA_DB = -0.0233822


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
        "--e14-dir",
        default="output/stage_b_basicvsrpp_data_scale_train001_240_val000_005_v1")
    parser.add_argument(
        "--e15-dir",
        default="output/stage_b_basicvsrpp_equal_exposure_train000_239_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_basicvsrpp_multiwindow_train000_239_val000_005_v1")
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
        raise ValueError(f"E16 fixes the update count at {STEPS}")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E16 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E16 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E16 fixes gradient clipping at 1.0")
    adapter = Path(args.mmagic_root).resolve() / "standalone_basicvsrpp.py"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    for directory in (args.e10_dir, args.e13_dir, args.e14_dir, args.e15_dir):
        if not Path(directory).resolve().is_dir():
            raise FileNotFoundError(directory)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def load_train_window(data_root: Path, sequence: str, start: int):
    if sequence not in TRAIN_IDS or start not in WINDOW_STARTS:
        raise ValueError(f"E16 forbids train/{sequence} start={start}")
    x, y = crop_xy(e10.SEED, sequence, start)
    root = data_root / "train_sharp" / sequence
    paths = sorted(root.glob("*.png"))[start:start + 17]
    if len(paths) != 17:
        raise ValueError(f"{root} start={start} does not provide 17 frames")
    frames = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crop = image[y:y + 512, x:x + 512]
        if crop.shape != (512, 512, 3):
            raise ValueError(f"invalid crop ({x}, {y}) for {path}")
        frames.append(crop.transpose(2, 0, 1).copy())
    return [str(path.resolve()) for path in paths], frames, (x, y)


def static_path(output_dir, key):
    return output_dir / "train_cache" / "samples" / f"{key}.safetensors"


def record_path(output_dir, key):
    return output_dir / "train_cache" / "records" / f"{key}.json"


def prior_static_entries(args):
    e14_root = Path(args.e14_dir).resolve()
    e14_entries = read_jsonl(e14_root / "train_cache" / "manifest.jsonl")
    result = {(item["sequence"], 0, item["chunk_index"]): item
              for item in e14_entries}
    if len(result) != 480:
        raise RuntimeError("E16 requires all 480 E14 start=0 entries")
    e13_root = Path(args.e13_dir).resolve()
    e13_entries = read_jsonl(
        e13_root / "feature_cache" / "train" / "manifest.jsonl")
    for item in e13_entries:
        if item["window_start"] == 0:
            continue
        result[(item["sequence"], item["window_start"], item["chunk_index"])] = {
            "key": item["key"],
            "sequence": item["sequence"],
            "window_start": item["window_start"],
            "chunk_index": item["chunk_index"],
            "latent_path": item["latent_path"],
            "reused_from": "E10/E13",
        }
    if len(result) != 624:
        raise RuntimeError(f"wrong E16 reusable static count: {len(result)}")
    return result


def load_static(entry, device="cpu"):
    path = entry.get("sample_path") or entry.get("latent_path")
    if not path:
        raise RuntimeError(f"no static tensor path for {entry['key']}")
    return load_file(path, device=str(device))


@torch.inference_mode()
def select_high_byte_route_batched(p_net, prepared, batch_size=32):
    """Exact E10 singleton-byte ranking with batched neural-prior forwards.

    The rANS stream for every singleton is still produced independently with
    the same coder.  Only the 256 repeated neural prior evaluations are grouped
    into batches.  A startup audit compares every byte count against E10's
    scalar implementation before this helper is accepted for the run.
    """
    y = prepared["y"]
    common_params = prepared["common_params"]
    grid = (y.shape[-2] // e14.BLOCK_SIZE, y.shape[-1] // e14.BLOCK_SIZE)
    if grid != (16, 16):
        raise RuntimeError(f"unexpected E16 route grid: {grid}")
    all_base_route = np.zeros(grid, dtype=np.bool_)
    all_base = e12.encode_y(
        p_net, y, common_params, all_base_route, e14.BLOCK_SIZE)
    expected = np.nan_to_num(
        all_base.expected_bits_by_block.reshape(-1), nan=-1e30,
        posinf=1e30, neginf=-1e30)
    cdf_info = p_net.gaussian_encoder.get_cdf_info()
    singletons = []
    for first in range(0, grid[0] * grid[1], batch_size):
        flat_ids = np.arange(first, min(first + batch_size, grid[0] * grid[1]))
        routes = np.zeros((len(flat_ids), *grid), dtype=np.bool_)
        routes.reshape(len(flat_ids), -1)[np.arange(len(flat_ids)), flat_ids] = True
        skip = torch.from_numpy(routes).to(device=y.device)
        skip = skip.repeat_interleave(e14.BLOCK_SIZE, 1).repeat_interleave(
            e14.BLOCK_SIZE, 2)
        keep = (~skip)[:, None].expand(-1, y.shape[1], -1, -1)
        y_batch = y.expand(len(flat_ids), -1, -1, -1)
        common_batch = common_params.expand(len(flat_ids), -1, -1, -1)
        q_dense, _, _, scales = stage1.y_prior_steps(
            p_net, common_batch, keep, source_y=y_batch)
        active = keep & (scales > p_net.gaussian_encoder.skip_thres)
        q_numpy = q_dense.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        scale_numpy = scales.detach().permute(
            0, 2, 3, 1).contiguous().float().cpu().numpy()
        active_numpy = active.detach().permute(
            0, 2, 3, 1).contiguous().cpu().numpy()
        for local, flat_id in enumerate(flat_ids.tolist()):
            active_flat = active_numpy[local].reshape(-1)
            q_active = q_numpy[local].reshape(-1)[active_flat].astype(np.int16)
            indexes = stage1.scale_to_index(
                scale_numpy[local].reshape(-1))[active_flat].astype(np.int16)
            combined = (q_active * 256 + indexes).astype(np.int16)
            ec_parallel = stage1.compute_ec_parallel(int(combined.size))
            coder = stage1.make_rans_encoder(cdf_info, 1, ec_parallel)
            coder.encode_y(combined)
            coder.flush()
            singleton_bytes = len(np.asarray(
                coder.get_encoded_stream(), dtype=np.uint8))
            singletons.append({
                "flat_id": flat_id,
                "row": flat_id // grid[1],
                "col": flat_id % grid[1],
                "allbase_y_bytes": len(all_base.stream),
                "singleton_y_bytes": singleton_bytes,
                "saved_y_bytes": len(all_base.stream) - singleton_bytes,
                "expected_bits": float(expected[flat_id]),
            })
        del q_dense, scales, active, y_batch, common_batch, keep
    ranked = sorted(singletons, key=lambda item: (
        -item["saved_y_bytes"], -item["expected_bits"], item["flat_id"]))
    selected = ranked[:e14.K]
    route = np.zeros(grid, dtype=np.bool_)
    route.reshape(-1)[[item["flat_id"] for item in selected]] = True
    return route, all_base, singletons, selected


@torch.inference_mode()
def collect_static(p_net, prepared, route, all_base, selected):
    encoded = e12.encode_y(
        p_net, prepared["y"], prepared["common_params"], route, e14.BLOCK_SIZE)
    decoded_q, mean_y = e12.decode_y(
        p_net, encoded.stream, encoded.ec_parallel,
        prepared["common_params"], route, e14.BLOCK_SIZE)
    if not torch.equal(decoded_q, encoded.q_dense):
        raise RuntimeError("E16 Base-y entropy round trip mismatch")
    target_delta = source_route_target(
        p_net, prepared, decoded_q, route, e14.BLOCK_SIZE)
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
    }
    route_bytes = len(e12.build_route_section(route, e14.BLOCK_SIZE))
    record = {
        "selected_blocks": [selected_by_id[int(index)]
                            for index in np.flatnonzero(route.reshape(-1))],
        "allbase_y_bytes": len(all_base.stream),
        "k8_y_bytes": len(encoded.stream),
        "combination_y_bytes_saved": len(all_base.stream) - len(encoded.stream),
        "route_bytes": route_bytes,
        "combination_net_y_after_route_bytes_saved": (
            len(all_base.stream) - len(encoded.stream) - route_bytes),
    }
    return tensors, record, mean_y


@torch.inference_mode()
def prepare_static_cache(args, device):
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "train_cache"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    if manifest_path.is_file() and complete_path.is_file():
        entries = read_jsonl(manifest_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (len(entries) != SAMPLE_COUNT
                or complete.get("sample_count") != SAMPLE_COUNT
                or complete.get("sequence_ids") != list(TRAIN_IDS)
                or complete.get("window_starts") != list(WINDOW_STARTS)):
            raise RuntimeError("incompatible E16 static cache")
        if not all(Path(item.get("sample_path") or item.get("latent_path")).is_file()
                   for item in entries):
            raise RuntimeError("missing E16 static sample")
        return entries, complete["codec_checkpoint_audit"]

    prior = prior_static_entries(args)
    entries = list(prior.values())
    batch_audit_path = cache_dir / "batched_singleton_exactness_audit.json"
    batched_audited = batch_audit_path.is_file()
    if batched_audited:
        batch_audit = json.loads(batch_audit_path.read_text(encoding="utf-8"))
        if (batch_audit.get("all_256_singleton_byte_counts_equal") is not True
                or batch_audit.get("selected_route_equal") is not True
                or batch_audit.get("allbase_stream_equal") is not True):
            raise RuntimeError("invalid E16 batched singleton audit")
    i_net, p_net = e12.load_codec(args, device)
    codec_audit = {
        "image_checkpoint": str(Path(args.model_path_i).resolve()),
        "video_checkpoint": str(Path(args.model_path_p).resolve()),
    }
    data_root = Path(args.data_root).resolve()
    started = time.perf_counter()
    for sequence in TRAIN_IDS:
        for start in WINDOW_STARTS:
            if start == 0 or (sequence in e13.TRAIN_IDS and start != 0):
                continue
            source_files, frames, crop = load_train_window(
                data_root, sequence, start)
            _, _, i_hat = e12.encode_i(i_net, frames[0], device)
            e12.initialize_p_state(p_net, i_hat)
            first_index = 1
            for chunk_index in range(2):
                key = f"train_{sequence}_s{start:02d}_c{chunk_index}"
                sample_path = static_path(output_dir, key).resolve()
                metadata_path = record_path(output_dir, key)
                chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
                prepared = e12.prepare_chunk_latents(p_net, chunk, e14.QP)
                if sample_path.is_file() and metadata_path.is_file():
                    stored = load_file(sample_path, device="cpu")
                    route = stored["skip_blocks"].numpy().astype(np.bool_)
                    encoded = e12.encode_y(
                        p_net, prepared["y"], prepared["common_params"], route,
                        e14.BLOCK_SIZE)
                    decoded_q, mean_y = e12.decode_y(
                        p_net, encoded.stream, encoded.ec_parallel,
                        prepared["common_params"], route, e14.BLOCK_SIZE)
                    if (not torch.equal(decoded_q.cpu(), stored["decoded_q"])
                            or not torch.equal(mean_y.cpu(), stored["mean_y"])):
                        raise RuntimeError(f"resumed E16 sample disagrees: {key}")
                else:
                    if not batched_audited:
                        scalar = e10.select_high_byte_route(p_net, prepared)
                        batched = select_high_byte_route_batched(p_net, prepared)
                        scalar_by_id = {item["flat_id"]: item for item in scalar[2]}
                        batch_by_id = {item["flat_id"]: item for item in batched[2]}
                        byte_equal = all(
                            scalar_by_id[index]["singleton_y_bytes"]
                            == batch_by_id[index]["singleton_y_bytes"]
                            for index in range(256))
                        route_equal = np.array_equal(scalar[0], batched[0])
                        allbase_equal = scalar[1].stream == batched[1].stream
                        batch_audit = {
                            "audited_key": key,
                            "batch_size": 32,
                            "all_256_singleton_byte_counts_equal": byte_equal,
                            "selected_route_equal": route_equal,
                            "allbase_stream_equal": allbase_equal,
                        }
                        e12.atomic_json(batch_audit_path, batch_audit)
                        if not (byte_equal and route_equal and allbase_equal):
                            raise RuntimeError(
                                "batched singleton ranking disagrees with E10")
                        route, all_base, _, selected = batched
                        batched_audited = True
                    else:
                        route, all_base, _, selected = (
                            select_high_byte_route_batched(p_net, prepared))
                    tensors, diagnostics, mean_y = collect_static(
                        p_net, prepared, route, all_base, selected)
                    sample_path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, sample_path, metadata={
                        "split": "train",
                        "source": "decoder-visible inputs; true latent is label only",
                    })
                    e12.atomic_json(metadata_path, {
                        "key": key,
                        "sequence": sequence,
                        "window_start": start,
                        "chunk_index": chunk_index,
                        "crop_xy": list(crop),
                        "source_files": source_files,
                        "trajectory": "fixed mean-fill",
                        "route_rule": "top-8 actual singleton Base-y byte saving",
                        **diagnostics,
                    })
                _, _, ref_feature = e12.reconstruction_float(
                    p_net, mean_y, prepared["q_decoder"], valid_count)
                entries.append({
                    "key": key,
                    "sequence": sequence,
                    "window_start": start,
                    "chunk_index": chunk_index,
                    "sample_path": str(sample_path),
                    "record_path": str(metadata_path.resolve()),
                    "reused_from": None,
                })
                p_net.set_ref_feature(
                    ref_feature, e12.should_reset(chunk_index, 32))
                first_index += valid_count
        print(json.dumps({
            "stage": "multiwindow-static-cache", "sequence": sequence,
            "samples": len(entries)}), flush=True)
    entries.sort(key=lambda item: item["key"])
    if len(entries) != SAMPLE_COUNT:
        raise RuntimeError(f"E16 expected {SAMPLE_COUNT} samples, got {len(entries)}")
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item) + "\n" for item in entries), encoding="utf-8")
    os.replace(temporary, manifest_path)
    complete = {
        "format": "e16_basicvsrpp_complete_train_four_windows_static_v1",
        "split": "REDS train_sharp/000..239",
        "sequence_ids": list(TRAIN_IDS),
        "window_starts": list(WINDOW_STARTS),
        "chunks_per_window": 2,
        "sample_count": len(entries),
        "reused_start0_from_e14": 480,
        "reused_other_windows_001_024_from_e10_e13": 144,
        "new_samples": 1_296,
        "trajectory": "fixed mean-fill K8",
        "route": "top-8 actual singleton Base-y byte saving",
        "codec_checkpoint_audit": codec_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "development_or_sealed_data_read": False,
    }
    e12.atomic_json(complete_path, complete)
    del i_net, p_net
    torch.cuda.empty_cache()
    return entries, codec_audit


@torch.inference_mode()
def prepare_feature_cache(args, entries, device):
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "feature_cache_bfloat16"
    manifest_path = cache_dir / "manifest.jsonl"
    complete_path = cache_dir / "complete.json"
    e14_root = Path(args.e14_dir).resolve()
    e14_complete = json.loads((e14_root / "feature_cache_bfloat16" /
                               "complete.json").read_text(encoding="utf-8"))
    load_audit = e14_complete["checkpoint_load_audit"]
    if manifest_path.is_file() and complete_path.is_file():
        feature_entries = read_jsonl(manifest_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        by_key = {item["key"]: item for item in feature_entries}
        if (len(by_key) != SAMPLE_COUNT
                or complete.get("sample_count") != SAMPLE_COUNT
                or complete.get("sequence_ids") != list(TRAIN_IDS)
                or complete.get("window_starts") != list(WINDOW_STARTS)
                or complete.get("extraction_dtype") != "bfloat16"
                or complete.get("checkpoint_load_audit") != load_audit
                or not all(Path(item["feature_stable_path"]).is_file()
                           for item in feature_entries)):
            raise RuntimeError("incompatible E16 feature cache")
        return [{**item, **by_key[item["key"]]} for item in entries], load_audit

    e14_features = {
        item["key"]: item for item in read_jsonl(
            e14_root / "feature_cache_bfloat16" / "manifest.jsonl")
    }
    if len(e14_features) != 480:
        raise RuntimeError("E16 requires all E14 start=0 bfloat16 features")
    feature_entries = [
        {**item, "reused_from": "E14 start=0"}
        for item in e14_features.values()
    ]
    entry_by_key = {item["key"]: item for item in entries}
    i_net, p_net = e12.load_codec(args, device)
    backbones, audit = e13.make_backbones(args, device)
    if audit != load_audit:
        raise RuntimeError("BasicVSR++ checkpoint audit changed for E16")
    maxima = {arm: 0.0 for arm in ARMS}
    started = time.perf_counter()
    data_root = Path(args.data_root).resolve()
    for sequence in TRAIN_IDS:
        for start in WINDOW_STARTS[1:]:
            _, frames, _ = load_train_window(data_root, sequence, start)
            _, _, i_hat = e12.encode_i(i_net, frames[0], device)
            e12.initialize_p_state(p_net, i_hat)
            first_index = 1
            for chunk_index in range(2):
                key = f"train_{sequence}_s{start:02d}_c{chunk_index}"
                static = load_static(entry_by_key[key], device="cpu")
                route = static["skip_blocks"].numpy().astype(np.bool_)
                chunk, _, valid_count = e12.make_chunk(frames, first_index, device)
                prepared = e12.prepare_chunk_latents(p_net, chunk, e14.QP)
                encoded = e12.encode_y(
                    p_net, prepared["y"], prepared["common_params"], route,
                    e14.BLOCK_SIZE)
                decoded_q, mean_y = e12.decode_y(
                    p_net, encoded.stream, encoded.ec_parallel,
                    prepared["common_params"], route, e14.BLOCK_SIZE)
                if (not torch.equal(decoded_q.cpu(), static["decoded_q"])
                        or not torch.equal(mean_y.cpu(), static["mean_y"])):
                    raise RuntimeError(f"E16 feature replay disagrees: {key}")
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
                        raise RuntimeError(f"non-finite E16 feature: {key}/{arm}")
                    maxima[arm] = max(maxima[arm], float(feature.abs().max()))
                path = (cache_dir / "samples" / f"{key}.safetensors").resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(features, path, metadata={
                    "split": "train", "extraction_dtype": "bfloat16",
                    "source": "decoder-visible mean-fill RGB only",
                })
                feature_entries.append({
                    "key": key,
                    "feature_stable_path": str(path),
                    "feature_dtype": "bfloat16",
                    "reused_from": None,
                })
                p_net.set_ref_feature(
                    ref_feature, e12.should_reset(chunk_index, 32))
                first_index += valid_count
        print(json.dumps({
            "stage": "multiwindow-bfloat16-features", "sequence": sequence,
            "samples": len(feature_entries)}), flush=True)
    feature_entries.sort(key=lambda item: item["key"])
    if len(feature_entries) != SAMPLE_COUNT:
        raise RuntimeError(
            f"E16 expected {SAMPLE_COUNT} features, got {len(feature_entries)}")
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(
        json.dumps(item) + "\n" for item in feature_entries), encoding="utf-8")
    os.replace(temporary, manifest_path)
    e12.atomic_json(complete_path, {
        "format": "e16_basicvsrpp_complete_train_four_windows_bfloat16_v1",
        "sample_count": len(feature_entries),
        "sequence_ids": list(TRAIN_IDS),
        "window_starts": list(WINDOW_STARTS),
        "extraction_dtype": "bfloat16",
        "stored_dtype": "bfloat16",
        "feature_shape": [g_frame_delay, 64, 32, 32],
        "reused_start0_from_e14": 480,
        "newly_extracted": 1_440,
        "max_abs_new_feature": maxima,
        "checkpoint_load_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
        "development_or_sealed_data_read": False,
    })
    by_key = {item["key"]: item for item in feature_entries}
    del backbones, i_net, p_net
    torch.cuda.empty_cache()
    return [{**item, **by_key[item["key"]]} for item in entries], audit


def make_schedule(sample_count):
    rng = np.random.default_rng(e12.SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:STEPS]


def train_one_arm(arm, samples, initial_state, args, output_dir, device):
    checkpoint_path = output_dir / "checkpoints" / f"multiwindow_{arm}_head_final.pt"
    summary_path = output_dir / "training" / f"multiwindow_{arm}_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != "e16_basicvsrpp_multiwindow_head_v1"
                or payload.get("arm") != arm or payload.get("steps") != STEPS):
            raise RuntimeError(f"incompatible E16 checkpoint: {arm}")
        return checkpoint_path, json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists() or summary_path.exists():
        raise RuntimeError(f"partial E16 training output: {arm}")
    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = make_schedule(len(samples))
    history = [{"step": 0, **e12.latent_summary(model, samples)["overall"]}]
    log_path = output_dir / "training" / f"multiwindow_{arm}.jsonl"
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
            raise RuntimeError(f"non-finite E16 gradient: {arm}")
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
    train_metrics = e12.latent_summary(model, samples)
    summary = {
        "arm": arm,
        "steps": STEPS,
        "sample_count": len(samples),
        "sequence_count": len(TRAIN_IDS),
        "windows_per_sequence": len(WINDOW_STARTS),
        "updates_per_sample": STEPS / len(samples),
        "train_metrics": train_metrics,
        "history": history,
        "training_wall_seconds": elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    e12.atomic_checkpoint(checkpoint_path, {
        "format": "e16_basicvsrpp_multiwindow_head_v1",
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


def train_both_arms(args, entries, output_dir, device):
    e12.seed_everything(HEAD_SEED)
    initial = e12.PnPLatentHead().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    checkpoints = {}
    summaries = {}
    for arm in ARMS:
        samples = e14.load_samples(entries, arm, device)
        checkpoint, summary = train_one_arm(
            arm, samples, initial_state, args, output_dir, device)
        checkpoints[arm] = checkpoint
        summaries[arm] = summary
        del samples
        torch.cuda.empty_cache()
    del initial, initial_state
    e12.atomic_json(output_dir / "checkpoint_lock_before_development.json", {
        "both_final_checkpoints_fixed_before_development_artifacts_loaded": True,
        "train": "REDS train_sharp/000..239, starts 0/24/48/72, 1920 P chunks",
        "steps_per_head": STEPS,
        "checkpoints": {arm: str(path.resolve())
                        for arm, path in checkpoints.items()},
        "development_ids_not_loaded_during_training": list(e13.DEV_IDS),
        "fixed_at_unix_time": time.time(),
    })
    return checkpoints, summaries


def classify(training, development, rgb):
    pre = development["pretrained"]
    random_arm = development["random"]
    positive_sequences = sum(
        point["latent_gap_recovery"] > 0 for point in pre["per_sequence"].values())
    beats_random_sequences = sum(
        pre["per_sequence"][sequence]["prediction_mse"]
        < random_arm["per_sequence"][sequence]["prediction_mse"]
        for sequence in e13.DEV_IDS)
    rgb_delta = rgb["pretrained"]["p_frame_psnr"] - rgb["mean"]["p_frame_psnr"]
    checks = {
        "development_overall_positive": pre["overall"]["latent_gap_recovery"] > 0,
        "development_high_byte_quartile_positive": (
            pre["highest_byte_quartile"]["latent_gap_recovery"] > 0),
        "development_second_chunk_positive": (
            pre["per_chunk_position"]["1"]["latent_gap_recovery"] > 0),
        "development_positive_on_at_least_5_of_6_sequences": (
            positive_sequences >= 5),
        "pretrained_beats_random_overall": (
            pre["overall"]["prediction_mse"]
            < random_arm["overall"]["prediction_mse"]),
        "pretrained_beats_random_on_at_least_4_of_6_sequences": (
            beats_random_sequences >= 4),
        "rgb_beats_mean": rgb_delta > 0,
        "rgb_positive_on_at_least_5_of_6_sequences": (
            rgb["pretrained"]["positive_vs_mean_sequences"] >= 5),
    }
    passed = all(checks.values())
    dev_change_vs_e14_pp = 100 * (
        pre["overall"]["latent_gap_recovery"] - E14_BEST_DEV_RECOVERY)
    rgb_change_vs_e14_db = rgb_delta - E14_BEST_RGB_DELTA_DB
    if passed:
        status = "multiwindow_development_signal_passed"
        recommendation = (
            "更多独立训练窗口使开发门槛通过；下一步做小范围适配器复核和真实码流公平闭环。")
    elif dev_change_vs_e14_pp >= 3.0 and rgb_change_vs_e14_db > 0:
        status = "multiwindow_improved_previous_best_but_not_passed"
        recommendation = (
            "多窗口训练超过此前最佳开发结果但尚未转正；路线暂保留，下一步只加轻量适配器或正则。")
    else:
        status = "multiwindow_did_not_resolve_generalization"
        recommendation = (
            "把训练段增至 1,920 且保持等暴露后仍未超过此前最佳开发结果；"
            "不再只扩数据、步数或同类预测头，优先改变可训练接口或 codec 表示。")
    return {
        "status": status,
        "checks": checks,
        "development_passed": passed,
        "development_positive_sequences": positive_sequences,
        "pretrained_beats_random_sequences": beats_random_sequences,
        "development_change_vs_e14_best_percentage_points": dev_change_vs_e14_pp,
        "development_rgb_delta_vs_mean_db": rgb_delta,
        "rgb_change_vs_e14_best_db": rgb_change_vs_e14_db,
        "recommendation": recommendation,
        "actual_net_benefit_claim_allowed": False,
        "sealed_data_should_be_read": False,
    }


def write_report(summary, output_dir):
    lines = [
        "# E16 完整 train 的多窗口泛化复核",
        "",
        f"结论：{summary['decision']['recommendation']}",
        "",
        "| 骨干 | train latent 恢复 | validation latent 恢复 | 高字节四分位 | 第一段 | 第二段 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        train = summary["training"][arm]["train_metrics"]
        dev = summary["development_latent"][arm]
        lines.append(
            f"| {arm} | {100*train['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['per_chunk_position']['0']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['per_chunk_position']['1']['latent_gap_recovery']:+.3f}% |")
    rgb = summary["development_rgb"]
    mean_psnr = rgb["mean"]["p_frame_psnr"]
    lines += [
        "",
        "## validation RGB 闭环",
        "",
        "| 方法 | P 帧 PSNR | 相对 mean-fill | 正改善视频 | 组件计数字节 | 额外中位耗时/段 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
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
        "- train/000..239 全部参与训练；train 指标不算实际收益。",
        "- validation/000..005 是已使用的开发集，不是独立测试。",
        "- 字节为相同 K8 路由下的内存内真实熵编码组件计数，未做落盘 fresh decode。",
        "- 未读取 val/006..029 或任何封存数据，未使用 true-fill 作为结果。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("E16 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries, codec_audit = prepare_static_cache(args, device)
    entries, backbone_audit = prepare_feature_cache(args, entries, device)
    checkpoints, training = train_both_arms(
        args, entries, output_dir, device)
    development_entries = e15.load_development_entries_after_lock(
        args, output_dir)
    development = e15.evaluate_development(
        checkpoints, development_entries, output_dir, device)
    rgb = e15.evaluate_rgb(args, checkpoints, output_dir, device)
    decision = classify(training, development, rgb)
    summary = {
        "experiment": "E16 BasicVSR++ complete-train multiwindow probe",
        "status": decision["status"],
        "scientific_scope": {
            "train": "REDS train_sharp/000..239, starts 0/24/48/72, 1920 P chunks",
            "train_metrics_are_actual_benefit": False,
            "development": "REDS val_sharp/000..005, already-used development set",
            "sealed_data_read": False,
        },
        "protocol": {
            "steps_per_head": STEPS,
            "updates_per_sample": STEPS / SAMPLE_COUNT,
            "sample_count": SAMPLE_COUNT,
            "head_parameters": e12.PnPLatentHead().parameter_count,
            "frozen_backbone_parameters": e13.BACKBONE_PARAMETERS,
            "loss": "unweighted direct-delta MSE",
            "qp_i": e14.QP,
            "qp_p": e14.QP,
            "k": e14.K,
            "route": "top-8 actual singleton Base-y byte saving",
            "all_checkpoints_fixed_before_development_artifacts_loaded": True,
        },
        "cache": {
            "codec_checkpoint_audit": codec_audit,
            "backbone_checkpoint_load_audit": backbone_audit,
        },
        "training": training,
        "development_latent": development,
        "development_rgb": rgb,
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
