#!/usr/bin/env python3
"""E17: preregistered learning curve on E16's complete multiwindow train set."""

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
import demo.stage_b_basicvsrpp_equal_exposure_probe as e15  # noqa: E402
import demo.stage_b_basicvsrpp_multiwindow_probe as e16  # noqa: E402
import demo.stage_b_basicvsrpp_transfer_probe as e13  # noqa: E402
import demo.stage_b_pnp_transfer_probe as e12  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


ARMS = ("pretrained", "random")
MILESTONES = (5_000, 12_500, 25_000, 50_000)
STEPS = MILESTONES[-1]
HEAD_SEED = e16.HEAD_SEED
SAMPLE_COUNT = e16.SAMPLE_COUNT
E14_BEST_DEV_RECOVERY = e16.E14_BEST_DEV_RECOVERY
E14_BEST_RGB_DELTA_DB = e16.E14_BEST_RGB_DELTA_DB


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
        "--e16-dir",
        default="output/stage_b_basicvsrpp_multiwindow_train000_239_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_basicvsrpp_learning_curve_train000_239_val000_005_v1")
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
        raise ValueError(f"E17 fixes the update count at {STEPS}")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E17 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E17 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E17 fixes gradient clipping at 1.0")
    if not (Path(args.mmagic_root).resolve() /
            "standalone_basicvsrpp.py").is_file():
        raise FileNotFoundError(args.mmagic_root)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    root = Path(args.e16_dir).resolve()
    for path in (
        root / "summary.json",
        root / "train_cache" / "manifest.jsonl",
        root / "train_cache" / "complete.json",
        root / "feature_cache_bfloat16" / "manifest.jsonl",
        root / "feature_cache_bfloat16" / "complete.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def load_training_entries(args):
    root = Path(args.e16_dir).resolve()
    static_complete = json.loads((root / "train_cache" /
                                  "complete.json").read_text(encoding="utf-8"))
    feature_complete = json.loads((root / "feature_cache_bfloat16" /
                                   "complete.json").read_text(encoding="utf-8"))
    if (static_complete.get("sample_count") != SAMPLE_COUNT
            or static_complete.get("sequence_ids") != list(e16.TRAIN_IDS)
            or static_complete.get("window_starts") != list(e16.WINDOW_STARTS)
            or static_complete.get("development_or_sealed_data_read") is not False):
        raise RuntimeError("E17 requires E16's complete train-only static cache")
    if (feature_complete.get("sample_count") != SAMPLE_COUNT
            or feature_complete.get("sequence_ids") != list(e16.TRAIN_IDS)
            or feature_complete.get("window_starts") != list(e16.WINDOW_STARTS)
            or feature_complete.get("extraction_dtype") != "bfloat16"
            or feature_complete.get("development_or_sealed_data_read") is not False):
        raise RuntimeError("E17 requires E16's complete bfloat16 train features")
    entries = read_jsonl(root / "train_cache" / "manifest.jsonl")
    features = {
        item["key"]: item for item in read_jsonl(
            root / "feature_cache_bfloat16" / "manifest.jsonl")
    }
    if len(entries) != SAMPLE_COUNT or len(features) != SAMPLE_COUNT:
        raise RuntimeError("E17 expected 1,920 cached training samples")
    merged = [{**item, **features[item["key"]]} for item in entries]
    if not all(Path(item["feature_stable_path"]).is_file() for item in merged):
        raise RuntimeError("E17 feature manifest is incomplete")
    return merged


def make_schedule(sample_count):
    rng = np.random.default_rng(e12.SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:STEPS]


def checkpoint_path(output_dir, arm, step):
    return output_dir / "checkpoints" / f"step{step:05d}_{arm}_head.pt"


def train_one_arm(arm, samples, initial_state, args, output_dir, device):
    summary_path = output_dir / "training" / f"{arm}_summary.json"
    expected_paths = [checkpoint_path(output_dir, arm, step)
                      for step in MILESTONES]
    if summary_path.is_file() and all(path.is_file() for path in expected_paths):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("steps") != STEPS
                or summary.get("milestones") != list(MILESTONES)):
            raise RuntimeError(f"incompatible E17 training output: {arm}")
        return {step: path for step, path in zip(MILESTONES, expected_paths)}, summary
    if summary_path.exists() or any(path.exists() for path in expected_paths):
        raise RuntimeError(f"partial E17 training output: {arm}")

    model = e12.PnPLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = make_schedule(len(samples))
    history = [{"step": 0, **e12.latent_summary(model, samples)["overall"]}]
    paths = {}
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
            raise RuntimeError(f"non-finite E17 gradient: {arm}")
        optimizer.step()
        if step in MILESTONES:
            model.eval()
            train_metrics = e12.latent_summary(model, samples)
            event = {
                "step": step,
                "last_sample_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **train_metrics["overall"],
            }
            history.append(event)
            path = checkpoint_path(output_dir, arm, step)
            e12.atomic_checkpoint(path, {
                "format": "e17_basicvsrpp_learning_curve_head_v1",
                "arm": arm,
                "step": step,
                "head_seed": HEAD_SEED,
                "state_dict": {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                "development_not_loaded": True,
                "train_metrics": train_metrics,
            })
            paths[step] = path
            print(json.dumps({"stage": "train-milestone", "arm": arm, **event}),
                  flush=True)
            model.train()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "arm": arm,
        "steps": STEPS,
        "milestones": list(MILESTONES),
        "sample_count": len(samples),
        "updates_per_sample_at_milestones": {
            str(step): step / len(samples) for step in MILESTONES},
        "history": history,
        "training_wall_seconds": elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    e12.atomic_json(summary_path, summary)
    return paths, summary


def train_all_checkpoints(args, entries, output_dir, device):
    e12.seed_everything(HEAD_SEED)
    initial = e12.PnPLatentHead().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    checkpoints = {}
    summaries = {}
    for arm in ARMS:
        samples = e14.load_samples(entries, arm, device)
        paths, summary = train_one_arm(
            arm, samples, initial_state, args, output_dir, device)
        checkpoints[arm] = paths
        summaries[arm] = summary
        del samples
        torch.cuda.empty_cache()
    del initial, initial_state
    e12.atomic_json(output_dir / "checkpoint_lock_before_development.json", {
        "all_eight_checkpoints_fixed_before_development_artifacts_loaded": True,
        "milestones": list(MILESTONES),
        "train": "REDS train_sharp/000..239, starts 0/24/48/72, 1920 P chunks",
        "checkpoints": {
            arm: {str(step): str(path.resolve())
                  for step, path in paths.items()}
            for arm, paths in checkpoints.items()
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


def verify_e16_final(checkpoints, args):
    reference_root = Path(args.e16_dir).resolve() / "checkpoints"
    result = {}
    for arm in ARMS:
        candidate = torch.load(checkpoints[arm][STEPS], map_location="cpu",
                               weights_only=False)["state_dict"]
        reference = torch.load(
            reference_root / f"multiwindow_{arm}_head_final.pt",
            map_location="cpu", weights_only=False)["state_dict"]
        max_difference = max(float((candidate[key] - reference[key]).abs().max())
                             for key in reference)
        result[arm] = {
            "all_tensors_bit_equal": all(torch.equal(candidate[key], reference[key])
                                         for key in reference),
            "max_abs_difference": max_difference,
        }
        if not result[arm]["all_tensors_bit_equal"]:
            raise RuntimeError(f"E17 failed to reproduce E16 final state: {arm}")
    return result


def evaluate_development(checkpoints, entries, output_dir, device):
    summaries = {arm: {} for arm in ARMS}
    records = []
    for arm in ARMS:
        samples = e14.load_development_samples_stable(entries, arm, device)
        for step in MILESTONES:
            model = load_head(checkpoints[arm][step], device)
            summary = e12.latent_summary(model, samples, include_records=True)
            for record in summary.pop("records"):
                records.append({"arm": arm, "step": step, **record})
            summaries[arm][str(step)] = summary
            del model
            torch.cuda.empty_cache()
        del samples
    evaluation = output_dir / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    e12.atomic_json(evaluation / "development_latent_curve.json", summaries)
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
    heads = {
        f"step{step}_{arm}": load_head(checkpoints[arm][step], device)
        for arm in ARMS for step in MILESTONES
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
                    arm = method.rsplit("_", 1)[1]
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
        print(json.dumps({"stage": "development-rgb-curve", "sequence": sequence}),
              flush=True)
    aggregate = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        mse = float(np.mean([row["p_frame_mse"] for row in selected]))
        times = latency.get(method, [])
        aggregate[method] = {
            "p_frame_mse": mse,
            "p_frame_psnr": -10 * math.log10(max(mse, 1e-30)),
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
    with (evaluation / "development_rgb_curve_windows.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    e12.atomic_json(evaluation / "development_rgb_curve.json", aggregate)
    del heads, backbones, i_net, p_net
    torch.cuda.empty_cache()
    return aggregate


def classify(training, development, rgb):
    mean_psnr = rgb["mean"]["p_frame_psnr"]
    points = {}
    for step in MILESTONES:
        key = str(step)
        pre = development["pretrained"][key]
        random_arm = development["random"][key]
        method = f"step{step}_pretrained"
        positive_sequences = sum(
            item["latent_gap_recovery"] > 0
            for item in pre["per_sequence"].values())
        beats_random_sequences = sum(
            pre["per_sequence"][sequence]["prediction_mse"]
            < random_arm["per_sequence"][sequence]["prediction_mse"]
            for sequence in e13.DEV_IDS)
        rgb_delta = rgb[method]["p_frame_psnr"] - mean_psnr
        checks = {
            "overall_positive": pre["overall"]["latent_gap_recovery"] > 0,
            "high_byte_positive": (
                pre["highest_byte_quartile"]["latent_gap_recovery"] > 0),
            "second_chunk_positive": (
                pre["per_chunk_position"]["1"]["latent_gap_recovery"] > 0),
            "latent_positive_on_5_of_6": positive_sequences >= 5,
            "pretrained_beats_random_overall": (
                pre["overall"]["prediction_mse"]
                < random_arm["overall"]["prediction_mse"]),
            "pretrained_beats_random_on_4_of_6": beats_random_sequences >= 4,
            "rgb_positive": rgb_delta > 0,
            "rgb_positive_on_5_of_6": (
                rgb[method]["positive_vs_mean_sequences"] >= 5),
        }
        points[key] = {
            "train_recovery": next(
                event["latent_gap_recovery"] for event in training["pretrained"]["history"]
                if event["step"] == step),
            "development_recovery": pre["overall"]["latent_gap_recovery"],
            "development_high_byte_recovery": (
                pre["highest_byte_quartile"]["latent_gap_recovery"]),
            "development_second_chunk_recovery": (
                pre["per_chunk_position"]["1"]["latent_gap_recovery"]),
            "development_positive_sequences": positive_sequences,
            "pretrained_beats_random_sequences": beats_random_sequences,
            "rgb_delta_vs_mean_db": rgb_delta,
            "rgb_positive_sequences": rgb[method]["positive_vs_mean_sequences"],
            "passes": all(checks.values()),
            "checks": checks,
        }
    best_step = max(MILESTONES, key=lambda step: points[str(step)][
        "development_recovery"])
    best = points[str(best_step)]
    improved_e14 = (
        best["development_recovery"] - E14_BEST_DEV_RECOVERY >= 0.03
        and best["rgb_delta_vs_mean_db"] > E14_BEST_RGB_DELTA_DB)
    if any(point["passes"] for point in points.values()):
        status = "learning_curve_development_signal_passed"
        recommendation = (
            "预登记训练剂量中存在完整开发门槛通过点；下一步锁定日程并做真实码流公平闭环。")
    elif improved_e14:
        status = "learning_curve_improved_previous_best_but_not_passed"
        recommendation = (
            "多窗口早期 checkpoint 超过此前最佳但尚未通过完整门槛；路线暂保留，"
            "下一步测试轻量适配器或更强正则。")
    else:
        status = "learning_curve_found_no_usable_training_dose"
        recommendation = (
            "四个预登记训练剂量都未超过此前最佳开发结果；停止继续扫描步数、"
            "扩同类数据或预测头，下一步只测试可训练适配器/接口变化，或转 codec。")
    return {
        "status": status,
        "points": points,
        "best_development_step": best_step,
        "best_development_point": best,
        "improved_e14_by_registered_rule": improved_e14,
        "recommendation": recommendation,
        "actual_net_benefit_claim_allowed": False,
        "sealed_data_should_be_read": False,
    }


def write_report(summary, output_dir):
    lines = [
        "# E17 多窗口训练剂量曲线",
        "",
        f"结论：{summary['decision']['recommendation']}",
        "",
        "| 步数 | 骨干 | train latent 恢复 | validation latent 恢复 | 高字节四分位 | 第二段 | RGB vs mean | 正 RGB 视频 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    mean_psnr = summary["development_rgb"]["mean"]["p_frame_psnr"]
    for step in MILESTONES:
        for arm in ARMS:
            dev = summary["development_latent"][arm][str(step)]
            train = next(event for event in summary["training"][arm]["history"]
                         if event["step"] == step)
            method = f"step{step}_{arm}"
            rgb = summary["development_rgb"][method]
            lines.append(
                f"| {step:,} | {arm} | {100*train['latent_gap_recovery']:+.3f}% | "
                f"{100*dev['overall']['latent_gap_recovery']:+.3f}% | "
                f"{100*dev['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
                f"{100*dev['per_chunk_position']['1']['latent_gap_recovery']:+.3f}% | "
                f"{rgb['p_frame_psnr']-mean_psnr:+.5f} dB | "
                f"{rgb['positive_vs_mean_sequences']}/6 |")
    lines += [
        "",
        "## 边界",
        "",
        "- 所有 8 个 checkpoint 均在 development 载入前固定。",
        "- train/000..239 全部参与训练；train 指标不算实际收益。",
        "- validation/000..005 是已使用的开发集，不是独立测试。",
        "- 字节为相同 K8 路由下的内存内真实熵编码组件计数，未做落盘 fresh decode。",
        "- 未读取 val/006..029 或封存数据，未使用 true-fill 作为结果。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("E17 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_training_entries(args)
    checkpoints, training = train_all_checkpoints(
        args, entries, output_dir, device)
    reproduction = verify_e16_final(checkpoints, args)
    development_entries = e15.load_development_entries_after_lock(
        args, output_dir)
    development = evaluate_development(
        checkpoints, development_entries, output_dir, device)
    rgb = evaluate_rgb(args, checkpoints, output_dir, device)
    decision = classify(training, development, rgb)
    summary = {
        "experiment": "E17 BasicVSR++ multiwindow learning curve",
        "status": decision["status"],
        "scientific_scope": {
            "train": "REDS train_sharp/000..239, starts 0/24/48/72, 1920 P chunks",
            "train_metrics_are_actual_benefit": False,
            "development": "REDS val_sharp/000..005, already-used development set",
            "sealed_data_read": False,
        },
        "protocol": {
            "milestones": list(MILESTONES),
            "sample_count": SAMPLE_COUNT,
            "head_parameters": e12.PnPLatentHead().parameter_count,
            "frozen_backbone_parameters": e13.BACKBONE_PARAMETERS,
            "loss": "unweighted direct-delta MSE",
            "qp_i": e14.QP,
            "qp_p": e14.QP,
            "k": e14.K,
            "all_eight_checkpoints_fixed_before_development_artifacts_loaded": True,
        },
        "e16_final_reproduction": reproduction,
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
        "best_step": decision["best_development_step"],
        "summary": str((output_dir / "summary.json").resolve()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
