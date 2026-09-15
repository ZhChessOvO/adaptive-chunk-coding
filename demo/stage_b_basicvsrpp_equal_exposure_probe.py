#!/usr/bin/env python3
"""E15: equal-exposure training check on the complete REDS train split.

E14 used 5,000 updates for both 192- and 480-sample configurations.  This
probe keeps the complete 480-sample train set, frozen codec/backbones, head,
optimizer, loss, and routes unchanged, but raises the update count to 12,500.
That matches the 26.04 updates per sample used by the 192-sample experiments.
Both final checkpoints are fixed before the already-used development set is
loaded.  No sealed validation data is eligible.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo.stage_b_basicvsrpp_data_scale_probe as e14  # noqa: E402
import demo.stage_b_basicvsrpp_transfer_probe as e13  # noqa: E402
import demo.stage_b_pnp_transfer_probe as e12  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


ARMS = ("pretrained", "random")
STEPS = 12_500
EVAL_EVERY = 500
HEAD_SEED = e14.HEAD_SEED
E14_STEPS = 5_000
TRAIN_FIT_THRESHOLD = 0.45
TRAIN_CHANGE_THRESHOLD = 0.15
TRAIN_CAPACITY_THRESHOLD = 0.30


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
        "--output-dir",
        default="output/stage_b_basicvsrpp_equal_exposure_train000_239_val000_005_v1")
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
        raise ValueError(f"E15 fixes the update count at {STEPS}")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E15 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E15 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E15 fixes gradient clipping at 1.0")
    adapter = Path(args.mmagic_root).resolve() / "standalone_basicvsrpp.py"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    e14_dir = Path(args.e14_dir).resolve()
    required = (
        e14_dir / "summary.json",
        e14_dir / "train_cache" / "manifest.jsonl",
        e14_dir / "train_cache" / "complete.json",
        e14_dir / "feature_cache_bfloat16" / "manifest.jsonl",
        e14_dir / "feature_cache_bfloat16" / "complete.json",
        e14_dir / "feature_cache_bfloat16" / "development" / "manifest.jsonl",
        e14_dir / "feature_cache_bfloat16" / "development" / "complete.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def load_training_entries(args):
    """Load only E14 train artifacts; development paths are not touched here."""
    root = Path(args.e14_dir).resolve()
    base_complete = json.loads((root / "train_cache" / "complete.json").read_text(
        encoding="utf-8"))
    feature_complete = json.loads((root / "feature_cache_bfloat16" /
                                   "complete.json").read_text(encoding="utf-8"))
    if (base_complete.get("sequence_ids") != list(e14.ALL_TRAIN_IDS)
            or base_complete.get("sample_count") != 480
            or base_complete.get("validation_or_sealed_data_read") is not False):
        raise RuntimeError("E15 requires the complete E14 train-only cache")
    if (feature_complete.get("sequence_ids") != list(e14.ALL_TRAIN_IDS)
            or feature_complete.get("sample_count") != 480
            or feature_complete.get("extraction_dtype") != "bfloat16"
            or feature_complete.get("development_or_sealed_data_read") is not False):
        raise RuntimeError("E15 requires E14's uniform bfloat16 train features")
    entries = read_jsonl(root / "train_cache" / "manifest.jsonl")
    features = {
        item["key"]: item
        for item in read_jsonl(root / "feature_cache_bfloat16" / "manifest.jsonl")
    }
    if len(entries) != 480 or len(features) != 480:
        raise RuntimeError("E15 expected 480 E14 training entries")
    merged = [{**item, **features[item["key"]]} for item in entries]
    if ({item["sequence"] for item in merged} != set(e14.ALL_TRAIN_IDS)
            or not all(Path(item["feature_stable_path"]).is_file()
                       for item in merged)):
        raise RuntimeError("E15 train manifest is incomplete")
    return merged


def make_schedule(sample_count):
    rng = np.random.default_rng(e12.SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:STEPS]


def train_one_arm(arm, samples, initial_state, args, output_dir, device):
    checkpoint_path = output_dir / "checkpoints" / f"s240_{arm}_head_final.pt"
    summary_path = output_dir / "training" / f"s240_{arm}_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != "e15_basicvsrpp_equal_exposure_head_v1"
                or payload.get("arm") != arm or payload.get("steps") != STEPS
                or payload.get("head_seed") != HEAD_SEED):
            raise RuntimeError(f"incompatible E15 checkpoint: {arm}")
        return checkpoint_path, json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists() or summary_path.exists():
        raise RuntimeError(f"partial E15 training output: {arm}")

    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = make_schedule(len(samples))
    history = [{"step": 0, **e12.latent_summary(model, samples)["overall"]}]
    log_path = output_dir / "training" / f"s240_{arm}.jsonl"
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
            raise RuntimeError(f"non-finite E15 gradient: {arm}")
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
        "sequence_count": len(e14.ALL_TRAIN_IDS),
        "updates_per_sample": STEPS / len(samples),
        "train_metrics": train_metrics,
        "history": history,
        "training_wall_seconds": elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    e12.atomic_checkpoint(checkpoint_path, {
        "format": "e15_basicvsrpp_equal_exposure_head_v1",
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
        "train": "REDS train_sharp/000..239, start=0, 480 P chunks",
        "steps_per_head": STEPS,
        "checkpoints": {arm: str(path.resolve())
                        for arm, path in checkpoints.items()},
        "development_ids_not_loaded_during_training": list(e13.DEV_IDS),
        "fixed_at_unix_time": time.time(),
    })
    return checkpoints, summaries


def load_development_entries_after_lock(args, output_dir):
    if not (output_dir / "checkpoint_lock_before_development.json").is_file():
        raise RuntimeError("development load attempted before E15 checkpoint lock")
    e13_root = Path(args.e13_dir).resolve()
    e14_root = Path(args.e14_dir).resolve()
    base_complete = json.loads((e13_root / "feature_cache" / "development" /
                                "complete.json").read_text(encoding="utf-8"))
    feature_complete = json.loads((e14_root / "feature_cache_bfloat16" /
                                   "development" / "complete.json").read_text(
                                       encoding="utf-8"))
    if (base_complete.get("sequence_ids") != list(e13.DEV_IDS)
            or base_complete.get("sample_count") != 12
            or base_complete.get("sealed_data_read") is not False):
        raise RuntimeError("incompatible E13 development cache")
    if (feature_complete.get("sequence_ids") != list(e13.DEV_IDS)
            or feature_complete.get("sample_count") != 12
            or feature_complete.get("extraction_dtype") != "bfloat16"
            or feature_complete.get("sealed_data_read") is not False):
        raise RuntimeError("incompatible E14 development feature cache")
    entries = read_jsonl(
        e13_root / "feature_cache" / "development" / "manifest.jsonl")
    features = {
        item["key"]: item for item in read_jsonl(
            e14_root / "feature_cache_bfloat16" / "development" / "manifest.jsonl")
    }
    if len(entries) != 12 or len(features) != 12:
        raise RuntimeError("E15 expected 12 development samples")
    return [{**item, **features[item["key"]]} for item in entries]


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def evaluate_development(checkpoints, entries, output_dir, device):
    summaries = {}
    records = []
    for arm in ARMS:
        model = load_head(checkpoints[arm], device)
        samples = e14.load_development_samples_stable(entries, arm, device)
        summary = e12.latent_summary(model, samples, include_records=True)
        for record in summary.pop("records"):
            records.append({"arm": arm, **record})
        summaries[arm] = summary
        del model, samples
        torch.cuda.empty_cache()
    evaluation = output_dir / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    e12.atomic_json(evaluation / "development_latent_summary.json", summaries)
    with (evaluation / "development_block_records.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return summaries


@torch.inference_mode()
def evaluate_rgb(args, checkpoints, output_dir, device):
    i_net, p_net = e12.load_codec(args, device)
    backbones, _ = e13.make_backbones(args, device)
    heads = {arm: load_head(checkpoints[arm], device) for arm in ARMS}
    methods = ("mean",) + ARMS
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
                prepared = e12.prepare_chunk_latents(p_net, chunk, e14.QP)
                encoded = e12.encode_y(
                    p_net, prepared["y"], prepared["common_params"], route,
                    e14.BLOCK_SIZE)
                decoded_q, mean_y = e12.decode_y(
                    p_net, encoded.stream, encoded.ec_parallel,
                    prepared["common_params"], route, e14.BLOCK_SIZE)
                if method == "mean":
                    predicted_y = mean_y
                else:
                    mean_rgb, _, _ = e12.reconstruction_float(
                        p_net, mean_y, prepared["q_decoder"], valid_count)
                    torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    features = e13.basic_features(
                        backbones[method], mean_rgb, device, move_to_cpu=False,
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
                route_section = e12.build_route_section(route, e14.BLOCK_SIZE)
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
        print(json.dumps({"stage": "development-rgb", "sequence": sequence}),
              flush=True)
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
    for method in ARMS:
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


def classify(training, development, rgb, e14_summary):
    train_pre = training["pretrained"]["train_metrics"]["overall"][
        "latent_gap_recovery"]
    old_train_pre = e14_summary["training"]["s240"]["pretrained"][
        "train_metrics"]["overall"]["latent_gap_recovery"]
    dev_pre = development["pretrained"]
    dev_random = development["random"]
    positive_sequences = sum(
        item["latent_gap_recovery"] > 0
        for item in dev_pre["per_sequence"].values())
    beats_random_sequences = sum(
        dev_pre["per_sequence"][sequence]["prediction_mse"]
        < dev_random["per_sequence"][sequence]["prediction_mse"]
        for sequence in e13.DEV_IDS)
    rgb_delta = rgb["pretrained"]["p_frame_psnr"] - rgb["mean"]["p_frame_psnr"]
    checks = {
        "train_recovery_at_least_45_percent": train_pre >= TRAIN_FIT_THRESHOLD,
        "train_recovery_gain_at_least_15pp_vs_e14_s240_5k": (
            train_pre - old_train_pre >= TRAIN_CHANGE_THRESHOLD),
        "development_overall_positive": (
            dev_pre["overall"]["latent_gap_recovery"] > 0),
        "development_high_byte_quartile_positive": (
            dev_pre["highest_byte_quartile"]["latent_gap_recovery"] > 0),
        "development_second_chunk_positive": (
            dev_pre["per_chunk_position"]["1"]["latent_gap_recovery"] > 0),
        "development_positive_on_at_least_5_of_6_sequences": (
            positive_sequences >= 5),
        "pretrained_beats_random_overall": (
            dev_pre["overall"]["prediction_mse"]
            < dev_random["overall"]["prediction_mse"]),
        "pretrained_beats_random_on_at_least_4_of_6_sequences": (
            beats_random_sequences >= 4),
        "rgb_beats_mean": rgb_delta > 0,
        "rgb_positive_on_at_least_5_of_6_sequences": (
            rgb["pretrained"]["positive_vs_mean_sequences"] >= 5),
    }
    development_passed = all(checks[key] for key in (
        "development_overall_positive",
        "development_high_byte_quartile_positive",
        "development_second_chunk_positive",
        "development_positive_on_at_least_5_of_6_sequences",
        "pretrained_beats_random_overall",
        "pretrained_beats_random_on_at_least_4_of_6_sequences",
        "rgb_beats_mean",
        "rgb_positive_on_at_least_5_of_6_sequences",
    ))
    if development_passed:
        status = "equal_exposure_development_signal_passed"
        recommendation = (
            "等训练暴露后 train 与开发门槛均通过；下一步锁定方案并补真实落盘、"
            "all-Base 调 QP、增强器公平对照和完整解码耗时。")
    elif checks["train_recovery_gain_at_least_15pp_vs_e14_s240_5k"]:
        status = "more_training_improved_fit_but_not_generalization"
        recommendation = (
            "更多更新明显改善 train，但开发门槛仍未通过；训练不足只解释了拟合，"
            "不能解释泛化。停止单纯加步数，下一轮只测试小范围适配器微调或接口变化。")
    elif train_pre < TRAIN_CAPACITY_THRESHOLD:
        status = "equal_exposure_still_underfits_train"
        recommendation = (
            "等训练暴露后 train 恢复仍低于 30%；下一轮可单独提高预测头容量或调整优化，"
            "保持完整 train 和开发协议不变。")
    else:
        status = "equal_exposure_did_not_resolve_development"
        recommendation = (
            "等训练暴露没有带来足够的 train 或开发改善；继续堆训练步数的依据不足，"
            "应优先改变可训练接口。")
    return {
        "status": status,
        "checks": checks,
        "train_recovery": train_pre,
        "train_recovery_change_vs_e14_s240_5k_percentage_points": (
            100 * (train_pre - old_train_pre)),
        "development_positive_sequences": positive_sequences,
        "pretrained_beats_random_sequences": beats_random_sequences,
        "development_rgb_delta_vs_mean_db": rgb_delta,
        "development_passed": development_passed,
        "recommendation": recommendation,
        "actual_net_benefit_claim_allowed": False,
        "sealed_data_should_be_read": False,
    }


def write_report(summary, output_dir):
    lines = [
        "# E15 完整 train 的等训练剂量复核",
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
        raise RuntimeError("E15 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_entries = load_training_entries(args)
    checkpoints, training = train_both_arms(
        args, training_entries, output_dir, device)
    development_entries = load_development_entries_after_lock(args, output_dir)
    development = evaluate_development(
        checkpoints, development_entries, output_dir, device)
    rgb = evaluate_rgb(args, checkpoints, output_dir, device)
    e14_summary = json.loads((Path(args.e14_dir).resolve() /
                              "summary.json").read_text(encoding="utf-8"))
    decision = classify(training, development, rgb, e14_summary)
    summary = {
        "experiment": "E15 BasicVSR++ complete-train equal-exposure probe",
        "status": decision["status"],
        "scientific_scope": {
            "train": "REDS train_sharp/000..239, start 0, 480 P chunks",
            "train_metrics_are_actual_benefit": False,
            "development": "REDS val_sharp/000..005, already-used development set",
            "sealed_data_read": False,
        },
        "protocol": {
            "steps_per_head": STEPS,
            "updates_per_sample": STEPS / 480,
            "e14_s240_steps": E14_STEPS,
            "e14_s240_updates_per_sample": E14_STEPS / 480,
            "e13_d96_reference_updates_per_sample": E14_STEPS / 192,
            "head_parameters": e12.PnPLatentHead().parameter_count,
            "frozen_backbone_parameters": e13.BACKBONE_PARAMETERS,
            "loss": "unweighted direct-delta MSE",
            "qp_i": e14.QP,
            "qp_p": e14.QP,
            "k": e14.K,
            "route": "top-8 actual singleton Base-y byte saving",
            "all_checkpoints_fixed_before_development_artifacts_loaded": True,
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
