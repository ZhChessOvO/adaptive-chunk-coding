#!/usr/bin/env python3
"""E18: tiny spatiotemporal adapter on frozen BasicVSR++ features."""

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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo.stage_b_basicvsrpp_data_scale_probe as e14  # noqa: E402
import demo.stage_b_basicvsrpp_equal_exposure_probe as e15  # noqa: E402
import demo.stage_b_basicvsrpp_learning_curve_probe as e17  # noqa: E402
import demo.stage_b_basicvsrpp_multiwindow_probe as e16  # noqa: E402
import demo.stage_b_basicvsrpp_transfer_probe as e13  # noqa: E402
import demo.stage_b_pnp_transfer_probe as e12  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


ARMS = ("pretrained", "random")
STEPS = 5_000
EVAL_EVERY = 1_000
HEAD_SEED = e16.HEAD_SEED
SAMPLE_COUNT = e16.SAMPLE_COUNT


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
        "--e17-dir",
        default="output/stage_b_basicvsrpp_learning_curve_train000_239_val000_005_v1")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_basicvsrpp_temporal_adapter_train000_239_val000_005_v1")
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
        raise ValueError(f"E18 fixes the update count at {STEPS}")
    if not math.isclose(args.learning_rate, 1e-3):
        raise ValueError("E18 fixes learning rate at 1e-3")
    if not math.isclose(args.weight_decay, 1e-4):
        raise ValueError("E18 fixes weight decay at 1e-4")
    if not math.isclose(args.gradient_clip, 1.0):
        raise ValueError("E18 fixes gradient clipping at 1.0")
    if not Path(args.e17_dir).resolve().joinpath("summary.json").is_file():
        raise FileNotFoundError(args.e17_dir)
    if not (Path(args.mmagic_root).resolve() /
            "standalone_basicvsrpp.py").is_file():
        raise FileNotFoundError(args.mmagic_root)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    root = Path(args.e16_dir).resolve()
    for path in (
        root / "train_cache" / "manifest.jsonl",
        root / "train_cache" / "complete.json",
        root / "feature_cache_bfloat16" / "manifest.jsonl",
        root / "feature_cache_bfloat16" / "complete.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


class ResidualSpatiotemporalBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        residual = self.pointwise(
            self.activation(self.depthwise(self.norm(x))))
        return x + residual


class AdaptedLatentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Sequential(
            ResidualSpatiotemporalBlock(),
            ResidualSpatiotemporalBlock(),
        )
        self.head = e12.PnPLatentHead()

    @property
    def adapter_parameter_count(self):
        return sum(parameter.numel() for parameter in self.adapter.parameters())

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def adapt(self, video_features):
        x = video_features.permute(1, 0, 2, 3).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            adapted = self.adapter(x)
        return adapted.squeeze(0).permute(1, 0, 2, 3).contiguous()

    def forward(self, decoded_q, mean_y, common_params, skip_blocks,
                video_features):
        return self.head(
            decoded_q, mean_y, common_params, skip_blocks,
            self.adapt(video_features))

    def apply(self, decoded_q, mean_y, common_params, skip_blocks,
              video_features):
        return self.head.apply(
            decoded_q, mean_y, common_params, skip_blocks,
            self.adapt(video_features))


def make_schedule(sample_count):
    rng = np.random.default_rng(e12.SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:STEPS]


def train_one_arm(arm, samples, initial_state, args, output_dir, device):
    checkpoint_path = output_dir / "checkpoints" / f"{arm}_adapter_head_final.pt"
    summary_path = output_dir / "training" / f"{arm}_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != "e18_basicvsrpp_temporal_adapter_v1"
                or payload.get("arm") != arm or payload.get("steps") != STEPS):
            raise RuntimeError(f"incompatible E18 checkpoint: {arm}")
        return checkpoint_path, json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists() or summary_path.exists():
        raise RuntimeError(f"partial E18 output: {arm}")

    model = AdaptedLatentHead().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = make_schedule(len(samples))
    history = [{"step": 0, **e12.latent_summary(model, samples)["overall"]}]
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
            raise RuntimeError(f"non-finite E18 gradient: {arm}")
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
        "train_metrics": train_metrics,
        "history": history,
        "head_parameters": model.head.parameter_count,
        "adapter_parameters": model.adapter_parameter_count,
        "total_trainable_parameters": model.parameter_count,
        "training_wall_seconds": elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    e12.atomic_checkpoint(checkpoint_path, {
        "format": "e18_basicvsrpp_temporal_adapter_v1",
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


def train_both(args, entries, output_dir, device):
    e12.seed_everything(HEAD_SEED)
    initial = AdaptedLatentHead().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    checkpoints = {}
    summaries = {}
    for arm in ARMS:
        samples = e14.load_samples(entries, arm, device)
        path, summary = train_one_arm(
            arm, samples, initial_state, args, output_dir, device)
        checkpoints[arm] = path
        summaries[arm] = summary
        del samples
        torch.cuda.empty_cache()
    del initial, initial_state
    e12.atomic_json(output_dir / "checkpoint_lock_before_development.json", {
        "both_final_checkpoints_fixed_before_development_artifacts_loaded": True,
        "train": "REDS train_sharp/000..239, starts 0/24/48/72, 1920 P chunks",
        "steps_per_model": STEPS,
        "checkpoints": {arm: str(path.resolve())
                        for arm, path in checkpoints.items()},
        "development_ids_not_loaded_during_training": list(e13.DEV_IDS),
        "fixed_at_unix_time": time.time(),
    })
    return checkpoints, summaries


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = AdaptedLatentHead().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def evaluate_development(checkpoints, entries, output_dir, device):
    summaries = {}
    records = []
    for arm in ARMS:
        model = load_model(checkpoints[arm], device)
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
    models = {arm: load_model(checkpoints[arm], device) for arm in ARMS}
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
                    predicted_y = models[method].apply(
                        decoded_q, mean_y, prepared["common_params"], route,
                        features)
                    torch.cuda.synchronize(device)
                    latency[method].append(
                        1000.0 * (time.perf_counter() - tick))
                rgb, _, ref_feature = e12.reconstruction_float(
                    p_net, predicted_y, prepared["q_decoder"], valid_count)
                total_error += float((rgb.float() - target_rgb).square().sum())
                pixel_count += target_rgb.numel()
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
            "component_counted_bytes": sum(
                row["component_counted_sequence_bytes"] for row in selected),
            "backbone_plus_adapter_head_ms_per_chunk_median": (
                float(np.median(times)) if times else 0.0),
            "backbone_plus_adapter_head_ms_per_chunk_p95": (
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
    del models, backbones, i_net, p_net
    torch.cuda.empty_cache()
    return aggregate


def classify(development, rgb, identity):
    pre = development["pretrained"]
    random_arm = development["random"]
    identity_pre = identity["development_latent"]["pretrained"]["5000"]
    positive_sequences = sum(
        item["latent_gap_recovery"] > 0 for item in pre["per_sequence"].values())
    beats_random_sequences = sum(
        pre["per_sequence"][sequence]["prediction_mse"]
        < random_arm["per_sequence"][sequence]["prediction_mse"]
        for sequence in e13.DEV_IDS)
    mean_psnr = rgb["mean"]["p_frame_psnr"]
    rgb_delta = rgb["pretrained"]["p_frame_psnr"] - mean_psnr
    identity_rgb = identity["development_rgb"]["step5000_pretrained"]
    checks = {
        "pretrained_beats_identity_latent": (
            pre["overall"]["prediction_mse"]
            < identity_pre["overall"]["prediction_mse"]),
        "pretrained_beats_identity_rgb": (
            rgb["pretrained"]["p_frame_psnr"] > identity_rgb["p_frame_psnr"]),
        "pretrained_beats_random_overall": (
            pre["overall"]["prediction_mse"]
            < random_arm["overall"]["prediction_mse"]),
        "pretrained_beats_random_on_4_of_6": beats_random_sequences >= 4,
        "development_overall_positive": pre["overall"]["latent_gap_recovery"] > 0,
        "development_high_byte_positive": (
            pre["highest_byte_quartile"]["latent_gap_recovery"] > 0),
        "development_second_chunk_positive": (
            pre["per_chunk_position"]["1"]["latent_gap_recovery"] > 0),
        "rgb_positive": rgb_delta > 0,
        "rgb_positive_on_5_of_6": (
            rgb["pretrained"]["positive_vs_mean_sequences"] >= 5),
    }
    passed = all(checks.values())
    if passed:
        status = "temporal_adapter_transfer_signal_passed"
        recommendation = (
            "轻量时空适配器使预训练版本稳定胜过 identity、随机骨干和 mean-fill；"
            "下一步做真实码流公平闭环。")
    elif (checks["pretrained_beats_identity_latent"]
          or checks["pretrained_beats_identity_rgb"]):
        status = "temporal_adapter_helped_but_transfer_gate_failed"
        recommendation = (
            "适配器有局部帮助但未形成稳定预训练优势；不直接扩大预测头，"
            "若继续只做一次更贴近 backbone 后部的低秩微调，否则转 codec 表示。")
    else:
        status = "temporal_adapter_did_not_help"
        recommendation = (
            "适配器未超过 identity 5k；停止扩大同类冻结特征预测器，"
            "下一步转向骨干后部微调或更适合部分传输的 codec。")
    return {
        "status": status,
        "checks": checks,
        "passes": passed,
        "development_positive_sequences": positive_sequences,
        "pretrained_beats_random_sequences": beats_random_sequences,
        "latent_change_vs_identity_percentage_points": 100 * (
            pre["overall"]["latent_gap_recovery"]
            - identity_pre["overall"]["latent_gap_recovery"]),
        "rgb_delta_vs_mean_db": rgb_delta,
        "rgb_change_vs_identity_db": (
            rgb["pretrained"]["p_frame_psnr"] - identity_rgb["p_frame_psnr"]),
        "recommendation": recommendation,
        "actual_net_benefit_claim_allowed": False,
        "sealed_data_should_be_read": False,
    }


def write_report(summary, output_dir):
    lines = [
        "# E18 轻量时空适配器复核",
        "",
        f"结论：{summary['decision']['recommendation']}",
        "",
        "| 骨干 | train latent 恢复 | validation latent 恢复 | 高字节四分位 | 第二段 | RGB vs mean | 正 RGB 视频 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    mean_psnr = summary["development_rgb"]["mean"]["p_frame_psnr"]
    for arm in ARMS:
        train = summary["training"][arm]["train_metrics"]
        dev = summary["development_latent"][arm]
        rgb = summary["development_rgb"][arm]
        lines.append(
            f"| {arm} | {100*train['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['overall']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['highest_byte_quartile']['latent_gap_recovery']:+.3f}% | "
            f"{100*dev['per_chunk_position']['1']['latent_gap_recovery']:+.3f}% | "
            f"{rgb['p_frame_psnr']-mean_psnr:+.5f} dB | "
            f"{rgb['positive_vs_mean_sequences']}/6 |")
    lines += [
        "",
        "## 边界",
        "",
        "- 两个最终 checkpoint 均在 development 载入前固定。",
        "- train/000..239 全部参与训练；train 指标不算实际收益。",
        "- validation/000..005 是已使用的开发集，不是独立测试。",
        "- 未读取 val/006..029 或封存数据，未使用 true-fill 作为结果。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("E18 requires CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = e17.load_training_entries(args)
    checkpoints, training = train_both(args, entries, output_dir, device)
    development_entries = e15.load_development_entries_after_lock(
        args, output_dir)
    development = evaluate_development(
        checkpoints, development_entries, output_dir, device)
    rgb = evaluate_rgb(args, checkpoints, output_dir, device)
    identity = json.loads((Path(args.e17_dir).resolve() /
                           "summary.json").read_text(encoding="utf-8"))
    decision = classify(development, rgb, identity)
    probe = AdaptedLatentHead()
    summary = {
        "experiment": "E18 frozen BasicVSR++ temporal adapter probe",
        "status": decision["status"],
        "scientific_scope": {
            "train": "REDS train_sharp/000..239, starts 0/24/48/72, 1920 P chunks",
            "train_metrics_are_actual_benefit": False,
            "development": "REDS val_sharp/000..005, already-used development set",
            "sealed_data_read": False,
        },
        "protocol": {
            "steps": STEPS,
            "head_parameters": probe.head.parameter_count,
            "adapter_parameters": probe.adapter_parameter_count,
            "total_trainable_parameters": probe.parameter_count,
            "adapter_initial_function": "exact identity",
            "frozen_backbone_parameters": e13.BACKBONE_PARAMETERS,
            "loss": "unweighted direct-delta MSE",
            "qp_i": e14.QP,
            "qp_p": e14.QP,
            "k": e14.K,
            "both_checkpoints_fixed_before_development_artifacts_loaded": True,
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
