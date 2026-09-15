#!/usr/bin/env python3
"""Fresh-state route-density bracket for Adaptive Chunk Coding Stage B.

This is a development screening experiment, not a multi-chunk Pareto claim.
It freezes the trained Lite C1 predictor and its alpha=0.25 output scale, then
uses one content-independent ordered K64 route to form nested K={4,8,16,32}
subsets.  REDS val/006..011 is kept separate from the alpha-development split.

All points share a QP32 I-frame reconstruction.  The all-Base anchors vary
only P QP (30/31/32), so the eight P-frame primary metric is not confounded by
different I-frame quality.  Every point is written as a self-describing D2L
stream and fresh-decoded three times.  Actual total bytes include the sequence
header, I payload, P container, z, y, route, and residual sections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.analyze_stage1_oracle_rd import interpolate_log_rate  # noqa: E402
from demo.stage1_multichunk_oracle import load_models  # noqa: E402
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    LPIPSAlex,
    load_frames,
    load_predictor_checkpoint,
    read_d2l,
    run_operational_mode,
    stable_route,
    validate_checkpoint_provenance,
)
from src.utils.common import set_torch_env  # noqa: E402


SCREEN_IDS = tuple(f"{index:03d}" for index in range(6, 12))
K_VALUES = (4, 8, 16, 32)
P_QPS = (30, 31, 32)
PREDICTOR_PROFILE = "learned-lite-c1-a025"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested K bracket using the calibrated Stage-B Lite predictor.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--checkpoint",
        default="output/stage_b_predictor_g2_phase_b_seed20260908/best.pt")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_route_density_bracket_a025_val006_011")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--screen-sequences", nargs="+", default=list(SCREEN_IDS))
    parser.add_argument("--frame-count", type=int, default=9)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=384)
    parser.add_argument("--crop-y", type=int, default=64)
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--p-qps", type=int, nargs="+", default=list(P_QPS))
    parser.add_argument("--qp-route", type=int, default=32)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--route-parent-skip-blocks", type=int, default=64)
    parser.add_argument("--latent-block-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--cuda-idx", type=int, default=0)
    return parser.parse_args()


def validate_args(args):
    expected = {
        "screen_sequences": list(SCREEN_IDS),
        "frame_count": 9,
        "width": 512,
        "height": 512,
        "crop_x": 384,
        "crop_y": 64,
        "qp_i": 32,
        "p_qps": list(P_QPS),
        "qp_route": 32,
        "k_values": list(K_VALUES),
        "route_parent_skip_blocks": 64,
        "latent_block_size": 2,
        "seed": 20260908,
        "reset_interval": 32,
        "decode_repeats": 3,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"registered K bracket requires {name}={value!r}")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("registered K bracket requires skip-thres=0")
    if set(args.screen_sequences) & set(f"{index:03d}" for index in range(24, 30)):
        raise ValueError("sealed REDS val/024..029 must not be accessed")


def alpha_label(k):
    return f"learned-c1-a025-k{k:02d}-i32p32"


def mean_label(k):
    return f"mean-c0-k{k:02d}-i32p32"


def base_label(qp):
    return f"all-base-i32p{qp}"


def psnr_to_mse(value):
    return 0.0 if math.isinf(value) else 255.0 ** 2 / (10.0 ** (value / 10.0))


def p_only_metrics(point):
    values = point["metrics"]["per_frame_psnr"][1:]
    if len(values) != 8:
        raise RuntimeError("K bracket requires exactly eight P-frame metrics")
    mse = float(np.mean([psnr_to_mse(value) for value in values]))
    return {
        "mean_frame_psnr": float(np.mean(values)),
        "aggregate_psnr": (
            float("inf") if mse == 0.0
            else 10.0 * math.log10(255.0 ** 2 / mse)),
        "rgb_mse_level": mse,
        "per_frame_psnr": values,
    }


def stream_identity(point):
    stream = point["stream"]
    return {
        "container_bytes": stream["container_bytes"],
        "i_payload_bytes": stream["i_payload_bytes"],
        "global_z_bytes": stream["global_z_bytes"],
        "base_y_bytes": stream["base_y_bytes"],
        "route_bytes": stream["route_bytes"],
        "residual_bytes": stream["residual_bytes"],
        "total_bytes": stream["total_bytes"],
    }


def verify_nested_routes(args):
    # At 512x512, HT-S y is 32x32; 2x2 routing gives a 16x16 grid.
    grid_shape = (args.height // 32, args.width // 32)
    masks = {}
    hashes = {}
    for k in (*args.k_values, args.route_parent_skip_blocks):
        args.skip_blocks = k
        mask = stable_route(args, 0, grid_shape)
        if int(mask.sum()) != k:
            raise RuntimeError(f"nested route K{k} has the wrong population")
        masks[k] = mask
        hashes[str(k)] = hashlib.sha256(
            np.packbits(mask.reshape(-1)).tobytes()).hexdigest()
    for smaller, larger in zip(args.k_values, args.k_values[1:]):
        if np.any(masks[smaller] & ~masks[larger]):
            raise RuntimeError(f"K{smaller} is not a subset of K{larger}")
    if np.any(masks[args.k_values[-1]] & ~masks[args.route_parent_skip_blocks]):
        raise RuntimeError("K32 is not a subset of the registered K64 parent")
    return hashes


def verify_point_pair(mean, learned, k):
    if stream_identity(mean) != stream_identity(learned):
        raise RuntimeError(
            f"K{k} mean and C1 must have identical fresh-chunk stream bytes")
    mean_file = read_d2l(Path(mean["sequence_path"])) if "sequence_path" in mean else None
    learned_file = (
        read_d2l(Path(learned["sequence_path"]))
        if "sequence_path" in learned else None)
    if mean_file is not None and learned_file is not None:
        if mean_file["i_stream"] != learned_file["i_stream"]:
            raise RuntimeError(f"K{k} mean and C1 I streams differ")
        if mean_file["chunks"] != learned_file["chunks"]:
            raise RuntimeError(f"K{k} mean and C1 P bitstreams differ")
    if mean["chunks"][0]["route_sha256"] != learned["chunks"][0]["route_sha256"]:
        raise RuntimeError(f"K{k} mean and C1 routes differ")
    if mean["chunks"][0]["skipped_blocks"] != k:
        raise RuntimeError(f"K{k} stream has the wrong number of Skip blocks")
    activation = learned["chunks"][0]["predictor_activation"]
    shapes = activation["learned_module_input_shapes"]
    if (activation.get("output_alpha") != 0.25
            or shapes["q_projector"][0] != k
            or shapes["common_projector"][0] != k
            or shapes["mean_projector"][0] != k
            or shapes["trunk"][0] != k
            or activation["learned_module_output_shape"][0] != k):
        raise RuntimeError(f"K{k} learned activation is not active-only alpha=0.25")


def attach_sequence_path(point, root):
    # Keep the path in the screening JSON for explicit byte-level audits.
    result = dict(point)
    result["sequence_path"] = str(root / "sequence.d2l")
    result["p_only_metrics"] = p_only_metrics(point)
    return result


@torch.inference_mode()
def run_sequence(sequence, i_net, p_net, checkpoint_path, checkpoint_sha,
                 args, device, lpips_metric, output_root):
    paths, frames = load_frames(args, sequence)
    sequence_root = output_root / sequence
    points = []
    args.skip_blocks = args.route_parent_skip_blocks
    for qp in args.p_qps:
        label = base_label(qp)
        target = sequence_root / label
        print(json.dumps({"sequence": sequence, "running": label}), flush=True)
        point = run_operational_mode(
            label, "all-base", qp, i_net, p_net, checkpoint_path,
            checkpoint_sha, frames, args, target, device, lpips_metric)
        points.append(attach_sequence_path(point, target))

    for k in args.k_values:
        args.skip_blocks = k
        mean_name = mean_label(k)
        learned_name = alpha_label(k)
        mean_root = sequence_root / mean_name
        learned_root = sequence_root / learned_name
        print(json.dumps({"sequence": sequence, "running": mean_name}), flush=True)
        mean = attach_sequence_path(run_operational_mode(
            mean_name, "mean", args.qp_route, i_net, p_net,
            checkpoint_path, checkpoint_sha, frames, args, mean_root,
            device, lpips_metric), mean_root)
        print(json.dumps({"sequence": sequence, "running": learned_name}), flush=True)
        learned = attach_sequence_path(run_operational_mode(
            learned_name, PREDICTOR_PROFILE, args.qp_route, i_net, p_net,
            checkpoint_path, checkpoint_sha, frames, args, learned_root,
            device, lpips_metric), learned_root)
        verify_point_pair(mean, learned, k)
        points.extend((mean, learned))

    i_streams = [read_d2l(Path(point["sequence_path"]))["i_stream"] for point in points]
    if any(stream != i_streams[0] for stream in i_streams[1:]):
        raise RuntimeError(f"{sequence}: not all points share the exact QP32 I stream")
    summary = {
        "sequence": sequence,
        "split": "REDS val_sharp K-screen 006..011",
        "source_files": [str(paths[0]), str(paths[-1])],
        "points": points,
    }
    sequence_root.mkdir(parents=True, exist_ok=True)
    (sequence_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def aggregate_point(summaries, label, args):
    points = [
        next(point for point in sequence["points"] if point["label"] == label)
        for sequence in summaries]
    total_frames = len(summaries) * args.frame_count
    total_p_frames = len(summaries) * 8
    total_bytes = sum(point["stream"]["total_bytes"] for point in points)
    total_p_bytes = sum(point["chunks"][0]["chunk_bytes"] for point in points)
    full_mse = sum(
        point["metrics"]["rgb_mse_level"] * args.frame_count for point in points
    ) / total_frames
    p_mse = sum(
        point["p_only_metrics"]["rgb_mse_level"] * 8 for point in points
    ) / total_p_frames
    repeat_count = min(len(point["compute"]["decode_wall_seconds_all"])
                       for point in points)
    aligned_seconds = [
        sum(point["compute"]["decode_wall_seconds_all"][repeat]
            for point in points)
        for repeat in range(repeat_count)]
    predictor_latencies = [
        value for point in points
        for value in point["compute"]["predictor_latency_ms_all"]]
    components = {
        key: sum(point["stream"][key] for point in points)
        for key in ("container_bytes", "i_payload_bytes", "global_z_bytes",
                    "base_y_bytes", "route_bytes", "residual_bytes")
    }
    if sum(components.values()) != total_bytes:
        raise RuntimeError(f"{label}: aggregate components do not equal total bytes")
    return {
        "label": label,
        "profile": points[0]["profile"],
        "qp_i": points[0]["qp_i"],
        "qp_p": points[0]["qp_p"],
        "total_bytes": total_bytes,
        "total_bpp": total_bytes * 8 / (
            total_frames * args.width * args.height),
        "p_chunk_bytes_including_d1s_container": total_p_bytes,
        "stream_components": components,
        "full_9_frame": {
            "frame_weighted_mean_psnr": float(np.mean([
                value for point in points
                for value in point["metrics"]["per_frame_psnr"]])),
            "aggregate_psnr": 10.0 * math.log10(255.0 ** 2 / full_mse),
            "rgb_mse_level": full_mse,
            "frame_weighted_lpips_alex": float(np.mean([
                point["metrics"]["lpips_alex"] for point in points])),
            "transition_weighted_temporal_delta_mae": float(np.mean([
                point["metrics"]["temporal_delta_mae_rgb_level"]
                for point in points])),
        },
        "p_only_8_frame": {
            "frame_weighted_mean_psnr": float(np.mean([
                value for point in points
                for value in point["p_only_metrics"]["per_frame_psnr"]])),
            "aggregate_psnr": 10.0 * math.log10(255.0 ** 2 / p_mse),
            "rgb_mse_level": p_mse,
        },
        "compute": {
            "decode_wall_seconds_by_aligned_repeat": aligned_seconds,
            "system_decode_ms_per_frame": (
                1000.0 * float(np.median(aligned_seconds)) / total_frames),
            "decode_throughput_fps": (
                total_frames / float(np.median(aligned_seconds))),
            "predictor_latency_ms_p50": (
                float(np.median(predictor_latencies))
                if predictor_latencies else 0.0),
            "predictor_latency_ms_p95": (
                float(np.percentile(predictor_latencies, 95))
                if predictor_latencies else 0.0),
            "linear_macs_per_chunk": points[0]["compute"]["linear_macs_per_chunk"],
            "peak_cuda_allocated_bytes_max": max(
                point["compute"]["peak_cuda_allocated_bytes_max"]
                for point in points),
        },
    }


def matched_rate(baselines, point, quality_key):
    anchors = [
        (baseline["qp_p"], baseline[quality_key]["frame_weighted_mean_psnr"],
         baseline["total_bpp"])
        for baseline in baselines]
    interpolation = interpolate_log_rate(
        anchors, point[quality_key]["frame_weighted_mean_psnr"])
    if interpolation is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = interpolation
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (point["total_bpp"] / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def mean_ladder_matched_rate(aggregate_by_label, point, quality_key):
    anchors = [
        (0,
         aggregate_by_label[base_label(32)][quality_key][
             "frame_weighted_mean_psnr"],
         aggregate_by_label[base_label(32)]["total_bpp"]),
    ]
    anchors.extend((
        k,
        aggregate_by_label[mean_label(k)][quality_key][
            "frame_weighted_mean_psnr"],
        aggregate_by_label[mean_label(k)]["total_bpp"],
    ) for k in K_VALUES)
    interpolation = interpolate_log_rate(
        anchors, point[quality_key]["frame_weighted_mean_psnr"])
    if interpolation is None:
        return {"status": "outside_mean_ladder_quality_range"}
    matched_bpp, lower_k, upper_k, weight = interpolation
    return {
        "status": "ok",
        "matched_mean_ladder_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (point["total_bpp"] / matched_bpp - 1.0),
        "lower_k": lower_k,
        "upper_k": upper_k,
        "interpolation_weight": weight,
    }


def sequence_matched_rate(sequence, label, quality_source):
    by_label = {point["label"]: point for point in sequence["points"]}
    target = by_label[label]
    anchors = []
    for qp in P_QPS:
        point = by_label[base_label(qp)]
        quality = (
            point["metrics"]["mean_frame_psnr"]
            if quality_source == "full" else
            point["p_only_metrics"]["mean_frame_psnr"])
        anchors.append((qp, quality, point["stream"]["total_bpp"]))
    target_quality = (
        target["metrics"]["mean_frame_psnr"]
        if quality_source == "full" else
        target["p_only_metrics"]["mean_frame_psnr"])
    interpolation = interpolate_log_rate(anchors, target_quality)
    if interpolation is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = interpolation
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (
            target["stream"]["total_bpp"] / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def analyze_k(k, aggregate_by_label, summaries):
    baselines = [aggregate_by_label[base_label(qp)] for qp in P_QPS]
    all_base_32 = aggregate_by_label[base_label(32)]
    mean = aggregate_by_label[mean_label(k)]
    learned = aggregate_by_label[alpha_label(k)]
    per_sequence = {}
    positive_count = 0
    for sequence in summaries:
        by_label = {point["label"]: point for point in sequence["points"]}
        mean_point = by_label[mean_label(k)]
        learned_point = by_label[alpha_label(k)]
        gain = (
            learned_point["p_only_metrics"]["mean_frame_psnr"]
            - mean_point["p_only_metrics"]["mean_frame_psnr"])
        positive_count += gain > 0.0
        per_sequence[sequence["sequence"]] = {
            "c1_p_only_psnr_gain_vs_mean_db": gain,
            "full_matched_rd": sequence_matched_rate(
                sequence, alpha_label(k), "full"),
            "p_only_quality_matched_rd": sequence_matched_rate(
                sequence, alpha_label(k), "p"),
        }
    full_match = matched_rate(baselines, learned, "full_9_frame")
    p_match = matched_rate(baselines, learned, "p_only_8_frame")
    mean_ladder_full_match = mean_ladder_matched_rate(
        aggregate_by_label, learned, "full_9_frame")
    mean_ladder_p_match = mean_ladder_matched_rate(
        aggregate_by_label, learned, "p_only_8_frame")
    full_available = sum(
        value["full_matched_rd"]["status"] == "ok"
        for value in per_sequence.values())
    full_improved = sum(
        value["full_matched_rd"].get("rate_change_percent", float("inf")) < 0.0
        for value in per_sequence.values())
    p_available = sum(
        value["p_only_quality_matched_rd"]["status"] == "ok"
        for value in per_sequence.values())
    p_improved = sum(
        value["p_only_quality_matched_rd"].get(
            "rate_change_percent", float("inf")) < 0.0
        for value in per_sequence.values())
    psnr_gain = (
        learned["p_only_8_frame"]["frame_weighted_mean_psnr"]
        - mean["p_only_8_frame"]["frame_weighted_mean_psnr"])
    lpips_change = (
        learned["full_9_frame"]["frame_weighted_lpips_alex"]
        - mean["full_9_frame"]["frame_weighted_lpips_alex"])
    temporal_change = (
        learned["full_9_frame"]["transition_weighted_temporal_delta_mae"]
        - mean["full_9_frame"]["transition_weighted_temporal_delta_mae"])
    lpips_ok = (
        learned["full_9_frame"]["frame_weighted_lpips_alex"]
        <= mean["full_9_frame"]["frame_weighted_lpips_alex"] * 1.005)
    temporal_ok = (
        learned["full_9_frame"]["transition_weighted_temporal_delta_mae"]
        <= mean["full_9_frame"]["transition_weighted_temporal_delta_mae"] * 1.005)
    secondary_improves = lpips_change < 0.0 or temporal_change < 0.0
    same_qp_saved = all_base_32["total_bytes"] - learned["total_bytes"]
    gates = {
        "c1_p_only_psnr_gain_at_least_0p015_db": psnr_gain >= 0.015,
        "c1_positive_on_at_least_5_of_6_sequences": positive_count >= 5,
        "lpips_not_worse_by_more_than_0p5_percent": lpips_ok,
        "temporal_not_worse_by_more_than_0p5_percent": temporal_ok,
        "lpips_or_temporal_improves": secondary_improves,
        "full_quality_inside_all_base_range": full_match["status"] == "ok",
        "p_only_quality_inside_all_base_range": p_match["status"] == "ok",
        "aggregate_full_matched_rate_improves_at_least_1_percent": (
            full_match.get("rate_change_percent", float("inf")) <= -1.0),
        "aggregate_p_only_matched_rate_improves_at_least_1_percent": (
            p_match.get("rate_change_percent", float("inf")) <= -1.0),
        "full_per_sequence_matched_available_on_all_6": full_available == 6,
        "p_only_per_sequence_matched_available_on_all_6": p_available == 6,
        "full_matched_rate_improves_on_at_least_4_of_6": full_improved >= 4,
        "p_only_matched_rate_improves_on_at_least_4_of_6": p_improved >= 4,
        "c1_vs_mean_ladder_full_matched_rate_improves_at_least_0p5_percent": (
            mean_ladder_full_match.get(
                "rate_change_percent", float("inf")) <= -0.5),
        "c1_vs_mean_ladder_p_only_matched_rate_improves_at_least_0p5_percent": (
            mean_ladder_p_match.get(
                "rate_change_percent", float("inf")) <= -0.5),
        "same_qp32_actual_total_bytes_decrease": same_qp_saved > 0,
        "mean_and_c1_actual_stream_bytes_identical": (
            mean["total_bytes"] == learned["total_bytes"]
            and mean["stream_components"] == learned["stream_components"]),
        "positive_active_only_compute": (
            learned["compute"]["linear_macs_per_chunk"] > 0
            and learned["compute"]["predictor_latency_ms_p50"] > 0.0),
    }
    return {
        "k": k,
        "mean_label": mean_label(k),
        "learned_label": alpha_label(k),
        "same_qp32_actual_total_bytes_saved": same_qp_saved,
        "same_qp32_actual_rate_change_percent": 100.0 * (
            learned["total_bytes"] / all_base_32["total_bytes"] - 1.0),
        "c1_vs_mean": {
            "p_only_psnr_gain_db": psnr_gain,
            "full_9_frame_psnr_gain_db": (
                learned["full_9_frame"]["frame_weighted_mean_psnr"]
                - mean["full_9_frame"]["frame_weighted_mean_psnr"]),
            "lpips_change": lpips_change,
            "temporal_delta_mae_change": temporal_change,
            "positive_sequence_count": positive_count,
            "total_bytes_change": learned["total_bytes"] - mean["total_bytes"],
        },
        "aggregate_full_matched_rd": full_match,
        "aggregate_p_only_quality_matched_rd": p_match,
        "c1_vs_mean_ladder_full_matched_rd": mean_ladder_full_match,
        "c1_vs_mean_ladder_p_only_matched_rd": mean_ladder_p_match,
        "per_sequence": per_sequence,
        "gate": {**gates, "passes_strict_bracket_gate": all(gates.values())},
    }


def write_csv(summary, path):
    rows = []
    for point in summary["aggregate_points"]:
        rows.append({
            "label": point["label"],
            "profile": point["profile"],
            "qp_i": point["qp_i"],
            "qp_p": point["qp_p"],
            "total_bytes": point["total_bytes"],
            "total_bpp": point["total_bpp"],
            "p_chunk_bytes": point["p_chunk_bytes_including_d1s_container"],
            "full_mean_psnr": point["full_9_frame"]["frame_weighted_mean_psnr"],
            "p_only_mean_psnr": point["p_only_8_frame"]["frame_weighted_mean_psnr"],
            "lpips_alex": point["full_9_frame"]["frame_weighted_lpips_alex"],
            "temporal_delta_mae": point["full_9_frame"][
                "transition_weighted_temporal_delta_mae"],
            "system_decode_ms_per_frame": point["compute"]["system_decode_ms_per_frame"],
            "predictor_ms_p50": point["compute"]["predictor_latency_ms_p50"],
            "linear_macs_per_chunk": point["compute"]["linear_macs_per_chunk"],
            "peak_cuda_allocated_bytes": point["compute"][
                "peak_cuda_allocated_bytes_max"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for point in summary["aggregate_points"]:
        label = point["label"]
        if label.startswith("all-base"):
            marker, color, size = "o", "black", 80
        elif label.startswith("mean"):
            marker, color, size = "x", "tab:blue", 70
        else:
            marker, color, size = "^", "tab:orange", 80
        axis.scatter(
            point["total_bpp"],
            point["p_only_8_frame"]["frame_weighted_mean_psnr"],
            marker=marker, color=color, s=size)
        axis.annotate(
            label, (point["total_bpp"],
                    point["p_only_8_frame"]["frame_weighted_mean_psnr"]),
            xytext=(4, 5), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Actual total rate (bits/pixel/frame, I32 + one P chunk)")
    axis.set_ylabel("P-only frame-weighted mean PSNR (dB)")
    axis.set_title("Adaptive Chunk Coding nested route-density bracket")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_torch_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    args.qp_i_override = args.qp_i
    nested_route_hashes = verify_nested_routes(args)
    checkpoint_path = Path(args.checkpoint)
    predictor, checkpoint_payload, checkpoint_sha = load_predictor_checkpoint(
        checkpoint_path, device)
    del predictor
    checkpoint_provenance = validate_checkpoint_provenance(
        checkpoint_payload, require_phase_b=True)
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    lpips_metric = LPIPSAlex(device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sequences = []
    for sequence in args.screen_sequences:
        result = run_sequence(
            sequence, i_net, p_net, checkpoint_path, checkpoint_sha,
            args, device, lpips_metric, output_root)
        sequences.append(result)
    labels = [point["label"] for point in sequences[0]["points"]]
    aggregate_points = [
        aggregate_point(sequences, label, args) for label in labels]
    aggregate_by_label = {point["label"]: point for point in aggregate_points}
    analyses = [
        analyze_k(k, aggregate_by_label, sequences) for k in args.k_values]
    eligible = [item for item in analyses if item["gate"]["passes_strict_bracket_gate"]]
    selected = None
    if eligible:
        best_margin = min(
            item["aggregate_p_only_quality_matched_rd"]["rate_change_percent"]
            for item in eligible)
        near_best = [
            item for item in eligible
            if item["aggregate_p_only_quality_matched_rd"]["rate_change_percent"]
            <= best_margin + 0.25]
        selected = min(near_best, key=lambda item: item["k"])
    torch.cuda.synchronize(device)
    summary = {
        "experiment": "adaptive_chunk_coding_stage_b_nested_k_bracket_v1",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "purpose": (
                "Select at most one route density whose calibrated C1 point is "
                "quality-bracketed by all-Base anchors before any larger predictor."),
            "development_screen_only": True,
            "multi_chunk_pareto_claim_allowed": False,
            "alpha_development_split_not_reused": "REDS val/000..005",
            "screen_split": "REDS val/006..011",
            "sealed_test_not_accessed": "REDS val/024..029",
            "weights_updated": False,
        },
        "protocol": {
            "sequence_ids": list(SCREEN_IDS),
            "frames_per_sequence": 9,
            "primary_quality_frames": "eight P frames; I frame excluded",
            "actual_rate_scope": (
                "complete D2L sequence: header + I + P header/z/y/route/residual"),
            "shared_i_qp": 32,
            "all_base_p_qps": list(P_QPS),
            "routed_p_qp": 32,
            "k_values": list(K_VALUES),
            "route_parent_k": args.route_parent_skip_blocks,
            "nested_route_hashes": nested_route_hashes,
            "predictor_profile": PREDICTOR_PROFILE,
            "predictor_alpha": 0.25,
            "decode_repeats": args.decode_repeats,
            "crop_xy": [args.crop_x, args.crop_y],
            "crop_size": [args.width, args.height],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "bytes_not_charged_per_video": checkpoint_path.stat().st_size,
            "verified_training_provenance": checkpoint_provenance,
        },
        "strict_gate_definition": {
            "predictor_feasibility": (
                "P-only PSNR >= +0.015 dB and positive on >=5/6; LPIPS and "
                "temporal each within +0.5%, with at least one improved."),
            "quality_bracket": (
                "Both full-nine-frame and P-only quality must lie inside measured "
                "all-Base I32/P30..32 ranges; interpolation never extrapolates."),
            "rate": (
                "Both aggregate matched-rate views improve by >=1%; all six "
                "per-sequence matches exist and >=4/6 improve; actual same-QP32 "
                "total bytes decrease."),
            "selection": (
                "Among strict-pass K values, choose the best aggregate P-only "
                "matched-rate margin; margins within 0.25 percentage points tie "
                "in favor of the smaller-compute K."),
            "compute_for_bandwidth": (
                "C1 must improve matched rate over the mean-fill K ladder by "
                ">=0.5% in both full and P-only quality views."),
        },
        "aggregate_points": aggregate_points,
        "k_analysis": analyses,
        "screening_decision": {
            "eligible_k": [item["k"] for item in eligible],
            "selected_k": None if selected is None else selected["k"],
            "passes_screen": selected is not None,
            "next_action": (
                "Run one preregistered 6x100 multi-chunk confirmation for the "
                f"single selected K={selected['k']} on a separate unselected split."
                if selected is not None else
                "Stop before/delay C3; inspect route choice or training target before "
                "adding predictor capacity."),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "per_sequence": sequences,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(summary, output_root / "aggregate_points.csv")
    save_plot(summary, output_root / "route_density_bracket.png")
    print(json.dumps({
        "summary": str(summary_path),
        "screening_decision": summary["screening_decision"],
        "k_analysis": [{
            "k": item["k"],
            "c1_vs_mean": item["c1_vs_mean"],
            "full_matched": item["aggregate_full_matched_rd"],
            "p_only_matched": item["aggregate_p_only_quality_matched_rd"],
            "mean_ladder_p_only_matched": item[
                "c1_vs_mean_ladder_p_only_matched_rd"],
            "strict_pass": item["gate"]["passes_strict_bracket_gate"],
        } for item in analyses],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
