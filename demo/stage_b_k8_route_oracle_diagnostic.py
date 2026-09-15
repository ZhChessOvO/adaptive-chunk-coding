#!/usr/bin/env python3
"""Predictor-aware K8 route Oracle diagnostic for Adaptive Chunk Coding.

This is an encoder-side, source-aware feasibility probe, not a deployable
Controller.  For every REDS val/006..011 fresh-state P chunk it evaluates all
256 possible one-block removals with the frozen C1 alpha=0.25 predictor.  It
ranks candidates by the established estimated-bit / source-MSE-delta proxy and
uses the first eight as one fixed route.  The selected route is then serialized
and fresh-decoded for both mean C0 and C1.

The search reuses the K screen split, updates no weights, consumes no new split,
and never reads the sealed val/024..029 sequences.
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
from demo.stage1_multichunk_oracle import (  # noqa: E402
    decode_i_stream,
    initialize_p_state,
    load_models,
    make_chunk,
    model_frame,
    mse_against_target,
    prepare_chunk_latents,
    psnr_from_unit_mse,
    reconstruction_float,
)
from demo.stage1_token_skipping import (  # noqa: E402
    CONTAINER_HEADER,
    build_route_section,
    encode_y,
)
from demo.stage_b_bracket_route_density import (  # noqa: E402
    aggregate_point,
    p_only_metrics,
)
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    LPIPSAlex,
    apply_profile_alpha,
    load_frames,
    load_predictor_checkpoint,
    read_d2l,
    route_sha256,
    run_operational_mode,
    source_route_target,
    validate_checkpoint_provenance,
)
from src.utils.common import set_torch_env  # noqa: E402


SCREEN_IDS = tuple(f"{index:03d}" for index in range(6, 12))
P_QPS = (30, 31, 32)
K = 8
C1_PROFILE = "learned-lite-c1-a025"
ORACLE_MEAN_LABEL = "oracle-mean-c0-k08-i32p32"
ORACLE_C1_LABEL = "oracle-c1-a025-k08-i32p32"
RANDOM_C1_LABEL = "learned-c1-a025-k08-i32p32"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Registered predictor-aware single-block-ranking K8 Oracle.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--checkpoint",
        default="output/stage_b_predictor_g2_phase_b_seed20260908/best.pt")
    parser.add_argument(
        "--bracket-summary",
        default=(
            "output/stage_b_route_density_bracket_a025_val006_011_v2/"
            "summary.json"))
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_k8_predictor_aware_route_oracle_val006_011_v2")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--screen-sequences", nargs="+", default=list(SCREEN_IDS))
    parser.add_argument("--frame-count", type=int, default=9)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=384)
    parser.add_argument("--crop-y", type=int, default=64)
    parser.add_argument("--qp-i", type=int, default=32)
    parser.add_argument("--qp-route", type=int, default=32)
    parser.add_argument("--latent-block-size", type=int, default=2)
    parser.add_argument("--skip-blocks", type=int, default=K)
    parser.add_argument("--candidate-pool", type=int, default=256)
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
        "qp_route": 32,
        "latent_block_size": 2,
        "skip_blocks": K,
        "candidate_pool": 256,
        "seed": 20260908,
        "reset_interval": 32,
        "decode_repeats": 3,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"registered K8 route Oracle requires {name}={value!r}")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("registered K8 route Oracle requires skip-thres=0")
    if set(args.screen_sequences) & set(f"{index:03d}" for index in range(24, 30)):
        raise ValueError("sealed REDS val/024..029 must not be accessed")


def actual_chunk_bytes(prepared, route, encoded_y):
    return (
        CONTAINER_HEADER.size + len(prepared["global_stream"])
        + len(route) + len(encoded_y.stream))


@torch.inference_mode()
def search_route(sequence, frames, i_net, p_net, predictor, args, device):
    i_x = (model_frame(frames[0], device) - 0.5).to(
        memory_format=torch.channels_last)
    i_encoded = i_net.compress(i_x, args.qp_i, 0, 0)
    i_stream = bytes(i_encoded["bit_stream"])
    i_hat = decode_i_stream(
        i_net, i_stream, int(i_encoded["ec_parallel"]), args.qp_i,
        args.height, args.width)
    initialize_p_state(p_net, i_hat)
    chunk, target_rgb, valid_count = make_chunk(frames, 1, device)
    if valid_count != 8:
        raise RuntimeError("route Oracle requires exactly eight P frames")
    prepared = prepare_chunk_latents(p_net, chunk, args.qp_route)
    grid_shape = (
        prepared["y"].shape[-2] // args.latent_block_size,
        prepared["y"].shape[-1] // args.latent_block_size)
    if int(np.prod(grid_shape)) != 256:
        raise RuntimeError("route Oracle requires the registered 16x16 route grid")
    all_base_mask = np.zeros(grid_shape, dtype=np.bool_)
    base_y = encode_y(
        p_net, prepared["y"], prepared["common_params"], all_base_mask,
        args.latent_block_size)
    base_rgb, _, _ = reconstruction_float(
        p_net, base_y.y_hat, prepared["q_decoder"], valid_count)
    base_mse = mse_against_target(base_rgb, target_rgb)
    estimated_bits = np.nan_to_num(
        base_y.expected_bits_by_block.reshape(-1), nan=0.0,
        posinf=1e12, neginf=0.0)
    candidate_ids = np.argsort(
        estimated_bits, kind="stable")[::-1][:args.candidate_pool]
    base_chunk_bytes = actual_chunk_bytes(prepared, b"", base_y)
    candidates = []
    started = time.perf_counter()
    for flat_index in candidate_ids:
        skip = np.zeros(grid_shape, dtype=np.bool_)
        skip.reshape(-1)[flat_index] = True
        trial = encode_y(
            p_net, prepared["y"], prepared["common_params"], skip,
            args.latent_block_size)
        raw_predicted_y, _ = predictor.apply(
            trial.q_dense, trial.y_hat, prepared["common_params"], skip)
        c1_y = apply_profile_alpha(
            raw_predicted_y.to(dtype=trial.y_hat.dtype), trial.y_hat,
            C1_PROFILE)
        trial_rgb, _, _ = reconstruction_float(
            p_net, c1_y, prepared["q_decoder"], valid_count)
        trial_mse = mse_against_target(trial_rgb, target_rgb)
        delta_mse = trial_mse - base_mse
        route = build_route_section(skip, args.latent_block_size)
        y_bytes_saved = len(base_y.stream) - len(trial.stream)
        # Route cost is fixed for the final K8 map.  Rank by the established
        # per-block estimated rate divided by source distortion; validate the
        # selected prefix with actual combined rANS and route bytes below.
        score = float(estimated_bits[flat_index]) / max(delta_mse, 1e-12)
        candidates.append({
            "flat_index": int(flat_index),
            "row": int(flat_index // grid_shape[1]),
            "col": int(flat_index % grid_shape[1]),
            "estimated_all_base_bits": float(estimated_bits[flat_index]),
            "c1_delta_unit_mse_vs_all_base": float(delta_mse),
            "c1_single_block_psnr": psnr_from_unit_mse(trial_mse),
            "actual_y_bytes_saved_single_block": int(y_bytes_saved),
            "actual_net_bytes_saved_single_block": int(
                base_chunk_bytes - actual_chunk_bytes(prepared, route, trial)),
            "score": score,
        })
    candidates.sort(
        key=lambda item: (item["score"], item["estimated_all_base_bits"]),
        reverse=True)
    selected_ids = [item["flat_index"] for item in candidates[:K]]
    selected = np.zeros(grid_shape, dtype=np.bool_)
    selected.reshape(-1)[selected_ids] = True
    selected_y = encode_y(
        p_net, prepared["y"], prepared["common_params"], selected,
        args.latent_block_size)
    selected_route = build_route_section(selected, args.latent_block_size)
    mean_rgb, _, _ = reconstruction_float(
        p_net, selected_y.y_hat, prepared["q_decoder"], valid_count)
    mean_mse = mse_against_target(mean_rgb, target_rgb)
    raw_selected_y, activation = predictor.apply(
        selected_y.q_dense, selected_y.y_hat, prepared["common_params"], selected)
    c1_selected_y = apply_profile_alpha(
        raw_selected_y.to(dtype=selected_y.y_hat.dtype), selected_y.y_hat,
        C1_PROFILE)
    c1_rgb, _, _ = reconstruction_float(
        p_net, c1_selected_y, prepared["q_decoder"], valid_count)
    c1_mse = mse_against_target(c1_rgb, target_rgb)
    # Source is used only after the C1 route and output have been frozen.
    source_delta = source_route_target(
        p_net, prepared, selected_y.q_dense, selected,
        args.latent_block_size)
    source_rgb, _, _ = reconstruction_float(
        p_net, selected_y.y_hat + source_delta,
        prepared["q_decoder"], valid_count)
    source_mse = mse_against_target(source_rgb, target_rgb)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return selected, {
        "sequence": sequence,
        "oracle_kind": (
            "source-aware predictor-aware full single-block ranking plus "
            "fixed K8 prefix validation"),
        "mathematical_upper_bound": False,
        "deployable_controller": False,
        "source_used_as_predictor_input": False,
        "candidate_count": len(candidates),
        "ranking": "estimated all-Base bits / max(C1 source delta-MSE, 1e-12)",
        "selected_flat_indices": selected_ids,
        "selected_route_sha256": route_sha256(selected),
        "all_base": {
            "p_chunk_bytes": base_chunk_bytes,
            "base_y_bytes": len(base_y.stream),
            "unit_mse": base_mse,
            "psnr": psnr_from_unit_mse(base_mse),
        },
        "selected_prefix": {
            "p_chunk_bytes": actual_chunk_bytes(
                prepared, selected_route, selected_y),
            "global_z_bytes": len(prepared["global_stream"]),
            "base_y_bytes": len(selected_y.stream),
            "route_bytes": len(selected_route),
            "actual_net_bytes_saved_vs_all_base_chunk": (
                base_chunk_bytes - actual_chunk_bytes(
                    prepared, selected_route, selected_y)),
            "mean_psnr": psnr_from_unit_mse(mean_mse),
            "c1_psnr": psnr_from_unit_mse(c1_mse),
            "source_latent_ceiling_psnr": psnr_from_unit_mse(source_mse),
            "predictor_activation": activation,
        },
        "search_wall_seconds": elapsed,
        "single_block_ranking": candidates,
    }


def attach(point, root):
    result = dict(point)
    result["sequence_path"] = str(root / "sequence.d2l")
    result["p_only_metrics"] = p_only_metrics(point)
    return result


def verify_operational_pair(mean, c1, route):
    keys = (
        "container_bytes", "i_payload_bytes", "global_z_bytes",
        "base_y_bytes", "route_bytes", "residual_bytes", "total_bytes")
    if {key: mean["stream"][key] for key in keys} != {
            key: c1["stream"][key] for key in keys}:
        raise RuntimeError("Oracle mean and C1 stream components differ")
    mean_file = read_d2l(Path(mean["sequence_path"]))
    c1_file = read_d2l(Path(c1["sequence_path"]))
    if mean_file["i_stream"] != c1_file["i_stream"]:
        raise RuntimeError("Oracle mean and C1 I streams differ")
    if mean_file["chunks"] != c1_file["chunks"]:
        raise RuntimeError("Oracle mean and C1 P bitstreams differ")
    expected_hash = route_sha256(route)
    if (mean["chunks"][0]["route_sha256"] != expected_hash
            or c1["chunks"][0]["route_sha256"] != expected_hash):
        raise RuntimeError("Operational route differs from Oracle-selected mask")
    if mean["chunks"][0]["skipped_blocks"] != K:
        raise RuntimeError("Operational Oracle stream does not contain K8")
    shapes = c1["chunks"][0]["predictor_activation"][
        "learned_module_input_shapes"]
    if any(shapes[name][0] != K for name in (
            "q_projector", "common_projector", "mean_projector", "trunk")):
        raise RuntimeError("Operational C1 activation is not K8 active-only")


@torch.inference_mode()
def run_sequence(sequence, formal_sequence, i_net, p_net,
                 checkpoint_path, checkpoint_sha, args, device, lpips_metric,
                 output_root):
    paths, frames = load_frames(args, sequence)
    search_predictor, _, _ = load_predictor_checkpoint(
        checkpoint_path, device, checkpoint_sha)
    selected, search = search_route(
        sequence, frames, i_net, p_net, search_predictor, args, device)
    del search_predictor
    torch.cuda.empty_cache()
    sequence_root = output_root / sequence
    sequence_root.mkdir(parents=True, exist_ok=True)
    np.save(sequence_root / "oracle_route.npy", selected)
    (sequence_root / "oracle_search.json").write_text(
        json.dumps(search, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    mean_root = sequence_root / ORACLE_MEAN_LABEL
    c1_root = sequence_root / ORACLE_C1_LABEL
    mean = attach(run_operational_mode(
        ORACLE_MEAN_LABEL, "mean", args.qp_route, i_net, p_net,
        checkpoint_path, checkpoint_sha, frames, args, mean_root, device,
        lpips_metric, fixed_routes=[selected]), mean_root)
    c1 = attach(run_operational_mode(
        ORACLE_C1_LABEL, C1_PROFILE, args.qp_route, i_net, p_net,
        checkpoint_path, checkpoint_sha, frames, args, c1_root, device,
        lpips_metric, fixed_routes=[selected]), c1_root)
    verify_operational_pair(mean, c1, selected)
    if search["selected_prefix"]["p_chunk_bytes"] != mean["chunks"][0]["chunk_bytes"]:
        raise RuntimeError("search and operational selected P chunk bytes differ")

    formal_by_label = {point["label"]: point for point in formal_sequence["points"]}
    random_c1 = formal_by_label[RANDOM_C1_LABEL]
    return {
        "sequence": sequence,
        "source_files": [str(paths[0]), str(paths[-1])],
        "search": search,
        "points": [mean, c1],
        "random_k8_reference": {
            "total_bytes": random_c1["stream"]["total_bytes"],
            "p_only_psnr": random_c1["p_only_metrics"]["mean_frame_psnr"],
            "route_sha256": random_c1["chunks"][0]["route_sha256"],
        },
    }


def sequence_matched_rate(formal_sequence, target):
    by_label = {point["label"]: point for point in formal_sequence["points"]}
    anchors = [(
        qp,
        by_label[f"all-base-i32p{qp}"]["p_only_metrics"]["mean_frame_psnr"],
        by_label[f"all-base-i32p{qp}"]["stream"]["total_bpp"],
    ) for qp in P_QPS]
    match = interpolate_log_rate(
        anchors, target["p_only_metrics"]["mean_frame_psnr"])
    if match is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = match
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (
            target["stream"]["total_bpp"] / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def aggregate_matched_rate(formal_by_label, target):
    anchors = [(
        qp,
        formal_by_label[f"all-base-i32p{qp}"]["p_only_8_frame"][
            "frame_weighted_mean_psnr"],
        formal_by_label[f"all-base-i32p{qp}"]["total_bpp"],
    ) for qp in P_QPS]
    match = interpolate_log_rate(
        anchors, target["p_only_8_frame"]["frame_weighted_mean_psnr"])
    if match is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = match
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (
            target["total_bpp"] / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def write_csv(points, path):
    rows = []
    for point in points:
        rows.append({
            "label": point["label"],
            "total_bytes": point["total_bytes"],
            "total_bpp": point["total_bpp"],
            "p_chunk_bytes": point["p_chunk_bytes_including_d1s_container"],
            "p_only_psnr": point["p_only_8_frame"]["frame_weighted_mean_psnr"],
            "full_psnr": point["full_9_frame"]["frame_weighted_mean_psnr"],
            "lpips_alex": point["full_9_frame"]["frame_weighted_lpips_alex"],
            "temporal_delta_mae": point["full_9_frame"][
                "transition_weighted_temporal_delta_mae"],
            "decode_ms_per_frame": point["compute"]["system_decode_ms_per_frame"],
            "predictor_ms_p50": point["compute"]["predictor_latency_ms_p50"],
            "linear_macs_per_chunk": point["compute"]["linear_macs_per_chunk"],
            "peak_cuda_allocated_bytes": point["compute"][
                "peak_cuda_allocated_bytes_max"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(formal_by_label, oracle_points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for qp in P_QPS:
        point = formal_by_label[f"all-base-i32p{qp}"]
        axis.scatter(
            point["total_bpp"],
            point["p_only_8_frame"]["frame_weighted_mean_psnr"],
            color="black", marker="o", s=70)
        axis.annotate(
            f"all-Base P{qp}",
            (point["total_bpp"], point["p_only_8_frame"][
                "frame_weighted_mean_psnr"]), xytext=(5, 5),
            textcoords="offset points", fontsize=8)
    random_c1 = formal_by_label[RANDOM_C1_LABEL]
    axis.scatter(
        random_c1["total_bpp"],
        random_c1["p_only_8_frame"]["frame_weighted_mean_psnr"],
        color="tab:gray", marker="s", s=75, label="random K8 C1")
    colors = {ORACLE_MEAN_LABEL: "tab:blue", ORACLE_C1_LABEL: "tab:orange"}
    labels = {ORACLE_MEAN_LABEL: "Oracle-route K8 mean",
              ORACLE_C1_LABEL: "Oracle-route K8 C1"}
    for point in oracle_points:
        axis.scatter(
            point["total_bpp"],
            point["p_only_8_frame"]["frame_weighted_mean_psnr"],
            color=colors[point["label"]], marker="^", s=90,
            label=labels[point["label"]])
    axis.set_xlabel("Actual total rate (bits/pixel/frame, I32 + one P chunk)")
    axis.set_ylabel("P-only frame-weighted mean PSNR (dB)")
    axis.set_title("K8 predictor-aware route Oracle diagnostic")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bracket_path = Path(args.bracket_summary)
    bracket = json.loads(bracket_path.read_text(encoding="utf-8"))
    if bracket["screening_decision"]["passes_screen"]:
        raise RuntimeError("route Oracle expects the registered K screen to fail")
    formal_sequences = {
        sequence["sequence"]: sequence for sequence in bracket["per_sequence"]}
    if set(formal_sequences) != set(SCREEN_IDS):
        raise RuntimeError("bracket summary is not the registered val006..011 screen")

    set_torch_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    args.qp_i_override = args.qp_i
    checkpoint_path = Path(args.checkpoint)
    predictor, checkpoint_payload, checkpoint_sha = load_predictor_checkpoint(
        checkpoint_path, device)
    provenance = validate_checkpoint_provenance(
        checkpoint_payload, require_phase_b=True)
    del predictor
    torch.cuda.empty_cache()
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    lpips_metric = LPIPSAlex(device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    sequences = []
    for sequence in args.screen_sequences:
        print(json.dumps({"sequence": sequence, "searching_candidates": 256}),
              flush=True)
        result = run_sequence(
            sequence, formal_sequences[sequence], i_net, p_net,
            checkpoint_path, checkpoint_sha, args, device, lpips_metric,
            output_root)
        sequences.append(result)
        by_label = {point["label"]: point for point in result["points"]}
        print(json.dumps({
            "sequence": sequence,
            "selected": result["search"]["selected_flat_indices"],
            "search_seconds": result["search"]["search_wall_seconds"],
            "total_bytes": by_label[ORACLE_C1_LABEL]["stream"]["total_bytes"],
            "mean_p_psnr": by_label[ORACLE_MEAN_LABEL]["p_only_metrics"][
                "mean_frame_psnr"],
            "c1_p_psnr": by_label[ORACLE_C1_LABEL]["p_only_metrics"][
                "mean_frame_psnr"],
        }), flush=True)

    aggregate_inputs = [
        {"points": sequence["points"]} for sequence in sequences]
    oracle_points = [
        aggregate_point(aggregate_inputs, label, args)
        for label in (ORACLE_MEAN_LABEL, ORACLE_C1_LABEL)]
    oracle_by_label = {point["label"]: point for point in oracle_points}
    formal_by_label = {
        point["label"]: point for point in bracket["aggregate_points"]}
    mean_match = aggregate_matched_rate(
        formal_by_label, oracle_by_label[ORACLE_MEAN_LABEL])
    c1_match = aggregate_matched_rate(
        formal_by_label, oracle_by_label[ORACLE_C1_LABEL])
    per_sequence_match = {}
    for result in sequences:
        target_by_label = {point["label"]: point for point in result["points"]}
        per_sequence_match[result["sequence"]] = {
            "mean": sequence_matched_rate(
                formal_sequences[result["sequence"]],
                target_by_label[ORACLE_MEAN_LABEL]),
            "c1": sequence_matched_rate(
                formal_sequences[result["sequence"]],
                target_by_label[ORACLE_C1_LABEL]),
        }
    c1_available = sum(
        value["c1"]["status"] == "ok" for value in per_sequence_match.values())
    c1_improved = sum(
        value["c1"].get("rate_change_percent", float("inf")) < 0.0
        for value in per_sequence_match.values())
    mean_pass = mean_match.get("rate_change_percent", float("inf")) <= -1.0
    c1_pass = (
        c1_match.get("rate_change_percent", float("inf")) <= -1.0
        and c1_available == 6 and c1_improved >= 4)
    if c1_pass and not mean_pass:
        diagnosis = "Predictor-aware route alignment is the primary observed bottleneck."
    elif c1_pass and mean_pass:
        diagnosis = (
            "The Oracle route passes with mean and C1; the observed gain is primarily "
            "route selection rather than evidence that added predictor compute is required.")
    else:
        diagnosis = (
            "This greedy-prefix Oracle did not find a K8 C1 route that clears the "
            "registered gate; it is not a proof that no better route exists.")

    random_c1 = formal_by_label[RANDOM_C1_LABEL]
    summary = {
        "experiment": "adaptive_chunk_coding_stage_b_k8_predictor_aware_route_oracle_v1",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "operational_route_bitstreams_fresh_decoded": True,
            "deployable_controller": False,
            "source_aware_encoder_oracle": True,
            "mathematical_route_upper_bound": False,
            "weights_updated": False,
            "screen_split_reused": "REDS val_sharp/006..011",
            "new_development_split_consumed": False,
            "sealed_test_not_accessed": "REDS val_sharp/024..029",
            "multi_chunk_pareto_claim_allowed": False,
        },
        "protocol": {
            "sequence_ids": list(SCREEN_IDS),
            "frames_per_sequence": 9,
            "primary_quality_frames": "eight P frames; I frame excluded",
            "shared_i_qp": 32,
            "p_qp": 32,
            "route_k": K,
            "candidate_count_per_sequence": 256,
            "ranking": "estimated all-Base bits / max(C1 source delta-MSE, 1e-12)",
            "selected_prefix_size": K,
            "predictor_profile": C1_PROFILE,
            "predictor_alpha": 0.25,
            "decode_repeats": args.decode_repeats,
            "matched_rd_method": (
                "linear P-only mean-frame quality / logarithmic complete-D2L bpp; "
                "no extrapolation"),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "verified_training_provenance": provenance,
        },
        "aggregate_points": oracle_points,
        "random_k8_c1_reference": {
            "total_bytes": random_c1["total_bytes"],
            "total_bpp": random_c1["total_bpp"],
            "p_only_psnr": random_c1["p_only_8_frame"][
                "frame_weighted_mean_psnr"],
        },
        "matched_rd": {
            "oracle_mean": mean_match,
            "oracle_c1": c1_match,
            "per_sequence": per_sequence_match,
            "c1_matches_available": c1_available,
            "c1_sequences_improved": c1_improved,
        },
        "decision": {
            "oracle_mean_passes_aggregate_minus_1_percent": mean_pass,
            "oracle_c1_passes_registered_diagnostic_gate": c1_pass,
            "diagnosis": diagnosis,
            "authorizes_full_rollout_or_c3": False,
        },
        "search_compute": {
            "total_candidates": 256 * len(sequences),
            "total_search_wall_seconds": sum(
                sequence["search"]["search_wall_seconds"] for sequence in sequences),
            "encoder_only_oracle_cost_not_decoder_cost": True,
        },
        "bracket_summary_sha256": hashlib.sha256(
            bracket_path.read_bytes()).hexdigest(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "per_sequence": sequences,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(oracle_points, output_root / "aggregate_points.csv")
    save_plot(formal_by_label, oracle_points, output_root / "k8_route_oracle.png")
    print(json.dumps({
        "summary": str(summary_path),
        "aggregate_points": oracle_points,
        "matched_rd": summary["matched_rd"],
        "decision": summary["decision"],
        "search_compute": summary["search_compute"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
