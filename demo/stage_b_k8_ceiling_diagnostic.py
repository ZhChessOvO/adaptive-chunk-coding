#!/usr/bin/env python3
"""Non-operational K8 true-latent reference for Adaptive Chunk Coding Stage B.

The nested-K screen showed that K8 is the first route density with enough
actual byte saving to make the registered one-percent matched-rate gate
possible.  This diagnostic keeps that exact route and bitstream fixed.  It
first runs the decoder-only C1 predictor, and only afterwards reads the true
omitted latent to measure an explicitly non-deployable reconstruction reference.

No weights are updated, no new route is selected, and the sealed split is not
read.  The true-latent reference is diagnostic evidence only and is never serialized
as an operational codec profile.
"""

from __future__ import annotations

import argparse
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
    get_q_params,
    initialize_p_state,
    load_models,
    make_chunk,
    model_frame,
    prepare_chunk_latents,
)
from demo.stage1_token_skipping import (  # noqa: E402
    CONTAINER_HEADER,
    build_route_section,
    encode_y,
    expand_keep_mask,
    rgb_from_recon,
    write_container,
)
from demo.stage_b_diagnose_predictor_alpha import (  # noqa: E402
    pixel_metrics,
    vector_diagnostics,
)
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    apply_profile_alpha,
    decode_chunk_inputs,
    load_frames,
    load_predictor_checkpoint,
    read_d2l,
    route_sha256,
    source_route_target,
    stable_route,
    validate_checkpoint_provenance,
)
from demo.stage_b_masked_predictor import skipped_target_blocks  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


SCREEN_IDS = tuple(f"{index:03d}" for index in range(6, 12))
MEAN_LABEL = "mean-c0-k08-i32p32"
C1_LABEL = "learned-c1-a025-k08-i32p32"
C1_PROFILE = "learned-lite-c1-a025"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Registered true-latent reference diagnostic at nested K8.")
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
        default="output/stage_b_k8_ceiling_diagnostic_val006_011_v2")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--screen-sequences", nargs="+", default=list(SCREEN_IDS))
    parser.add_argument("--frame-count", type=int, default=9)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=384)
    parser.add_argument("--crop-y", type=int, default=64)
    parser.add_argument("--qp-route", type=int, default=32)
    parser.add_argument("--latent-block-size", type=int, default=2)
    parser.add_argument("--skip-blocks", type=int, default=8)
    parser.add_argument("--route-parent-skip-blocks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--skip-thres", type=float, default=0.0)
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
        "qp_route": 32,
        "latent_block_size": 2,
        "skip_blocks": 8,
        "route_parent_skip_blocks": 64,
        "seed": 20260908,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"registered K8 reference diagnostic requires {name}={value!r}")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("registered K8 reference diagnostic requires skip-thres=0")
    if set(args.screen_sequences) & set(f"{index:03d}" for index in range(24, 30)):
        raise ValueError("sealed REDS val/024..029 must not be accessed")


def reconstruct_p_frames(p_net, y_hat, q_decoder, valid_count):
    x_hat, _ = p_net.get_recon_and_feature(y_hat, p_net.ctx, q_decoder)
    return [rgb_from_recon(frame) for frame in x_hat[:valid_count]]


def formal_points_by_sequence(bracket_summary):
    result = {}
    for sequence in bracket_summary["per_sequence"]:
        result[sequence["sequence"]] = {
            point["label"]: point for point in sequence["points"]}
    return result


def assert_metric_match(label, measured, formal):
    expected = formal["p_only_metrics"]
    if not np.allclose(
            measured["per_frame_psnr"], expected["per_frame_psnr"],
            rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{label}: reconstructed PSNR differs from K bracket")


@torch.inference_mode()
def run_sequence(sequence, formal_by_label, i_net, p_net, predictor, args,
                 output_root, device):
    paths, frames = load_frames(args, sequence)
    i_x = (model_frame(frames[0], device) - 0.5).to(
        memory_format=torch.channels_last)
    i_encoded = i_net.compress(i_x, args.qp_route, 0, 0)
    i_stream = bytes(i_encoded["bit_stream"])
    i_ec = int(i_encoded["ec_parallel"])
    i_hat = decode_i_stream(
        i_net, i_stream, i_ec, args.qp_route, args.height, args.width)
    initialize_p_state(p_net, i_hat)

    chunk, _, valid_count = make_chunk(frames, 1, device)
    if valid_count != 8:
        raise RuntimeError("K8 reference diagnostic requires exactly eight P frames")
    prepared = prepare_chunk_latents(p_net, chunk, args.qp_route)
    grid_shape = (
        prepared["y"].shape[-2] // args.latent_block_size,
        prepared["y"].shape[-1] // args.latent_block_size)
    skip = stable_route(args, 0, grid_shape)
    encoded = encode_y(
        p_net, prepared["y"], prepared["common_params"], skip,
        args.latent_block_size)
    route = build_route_section(skip, args.latent_block_size)
    sequence_root = output_root / sequence
    sequence_root.mkdir(parents=True, exist_ok=True)
    chunk_path = sequence_root / "chunk_0000.d1s"
    write_container(
        chunk_path, args.qp_route, args.height, args.width, prepared["y"].shape,
        prepared["global_stream"], route, encoded.stream, b"",
        encoded.ec_parallel, all_base=False, z_ec_parallel=prepared["z_ec"])
    payload = chunk_path.read_bytes()
    _, q_feature, q_decoder = get_q_params(p_net, args.qp_route)
    parsed, decoded_q, mean_y, common_params = decode_chunk_inputs(
        p_net, payload, args.latent_block_size, q_feature)
    if not torch.equal(decoded_q, encoded.q_dense):
        raise RuntimeError(f"{sequence}: decoded Base q differs from encoder q")

    # Predictor execution must finish before any source-only target is read.
    raw_predicted_y, activation = predictor.apply(
        decoded_q, mean_y, common_params, skip)
    raw_predicted_y = raw_predicted_y.to(dtype=mean_y.dtype)
    c1_y = apply_profile_alpha(raw_predicted_y, mean_y, C1_PROFILE)
    prediction_delta = raw_predicted_y - mean_y
    keep = expand_keep_mask(
        skip, args.latent_block_size, mean_y.shape, mean_y.device)
    if torch.count_nonzero(prediction_delta[keep]).item() != 0:
        raise RuntimeError("predictor modified a transmitted Base position")

    source_p = frames[1:9]
    mean_pixel = pixel_metrics(
        source_p, reconstruct_p_frames(p_net, mean_y, q_decoder, valid_count))
    c1_pixel = pixel_metrics(
        source_p, reconstruct_p_frames(p_net, c1_y, q_decoder, valid_count))

    # This is the first source-only operation in the diagnostic path.
    target_delta = source_route_target(
        p_net, prepared, decoded_q, skip, args.latent_block_size)
    source_y = mean_y + target_delta
    source_pixel = pixel_metrics(
        source_p, reconstruct_p_frames(p_net, source_y, q_decoder, valid_count))
    prediction_blocks, prediction_ids = skipped_target_blocks(
        prediction_delta, skip, args.latent_block_size)
    target_blocks, target_ids = skipped_target_blocks(
        target_delta, skip, args.latent_block_size)
    if not torch.equal(prediction_ids, target_ids):
        raise RuntimeError("prediction and source target query order differs")

    mean_formal = formal_by_label[MEAN_LABEL]
    c1_formal = formal_by_label[C1_LABEL]
    assert_metric_match(f"{sequence} mean", mean_pixel, mean_formal)
    assert_metric_match(f"{sequence} C1", c1_pixel, c1_formal)
    expected_chunk = mean_formal["chunks"][0]
    component_bytes = {
        "p_container_header_bytes": CONTAINER_HEADER.size,
        "global_z_bytes": len(parsed["global"]),
        "base_y_bytes": len(parsed["base"]),
        "route_bytes": len(parsed["route"]),
        "residual_bytes": len(parsed["residual"]),
        "p_chunk_total_bytes": len(payload),
    }
    expected_components = {
        "p_container_header_bytes": expected_chunk["container_header_bytes"],
        "global_z_bytes": expected_chunk["global_z_bytes"],
        "base_y_bytes": expected_chunk["base_y_bytes"],
        "route_bytes": expected_chunk["route_bytes"],
        "residual_bytes": expected_chunk["residual_bytes"],
        "p_chunk_total_bytes": expected_chunk["chunk_bytes"],
    }
    if component_bytes != expected_components:
        raise RuntimeError(f"{sequence}: K8 D1S bytes differ from registered screen")
    if len(i_stream) != mean_formal["stream"]["i_payload_bytes"]:
        raise RuntimeError(f"{sequence}: I32 payload differs from registered screen")
    if route_sha256(skip) != expected_chunk["route_sha256"]:
        raise RuntimeError(f"{sequence}: K8 route differs from registered screen")
    formal_payload = read_d2l(Path(mean_formal["sequence_path"]))["chunks"][0]
    if payload != formal_payload:
        raise RuntimeError(f"{sequence}: re-encoded D1S payload differs byte-for-byte")
    if mean_formal["stream"] != c1_formal["stream"]:
        raise RuntimeError(f"{sequence}: registered mean/C1 streams differ")

    return {
        "sequence": sequence,
        "source_files": [str(paths[0]), str(paths[-1])],
        "route_sha256": route_sha256(skip),
        "skipped_blocks": int(skip.sum()),
        "actual_registered_d2l_stream": mean_formal["stream"],
        "actual_reencoded_p_chunk": component_bytes,
        "mean_pixel": mean_pixel,
        "c1_alpha_0p25_pixel": c1_pixel,
        "true_latent_reference_pixel": source_pixel,
        "vector_diagnostic_for_unscaled_c1": vector_diagnostics(
            prediction_blocks, target_blocks),
        "predictor_activation": activation,
        "validation": {
            "predictor_completed_before_source_target_read": True,
            "predictor_inputs_decoder_available_only": True,
            "true_latent_reference_serialized_as_operational_profile": False,
            "mean_and_c1_metrics_match_registered_screen": True,
            "d1s_bytes_and_route_match_registered_screen": True,
        },
    }


def aggregate_pixel(sequences, key):
    per_frame_psnr = [
        value for sequence in sequences
        for value in sequence[key]["per_frame_psnr"]]
    per_frame_mse = [
        value for sequence in sequences
        for value in sequence[key]["per_frame_mse"]]
    mse = float(np.mean(per_frame_mse))
    return {
        "frame_weighted_mean_psnr": float(np.mean(per_frame_psnr)),
        "aggregate_psnr": 10.0 * math.log10(255.0 * 255.0 / mse),
        "rgb_mse_level": mse,
        "per_frame_count": len(per_frame_psnr),
    }


def quality_at_log_rate(anchors, target_rate):
    points = sorted(anchors, key=lambda item: item[2])
    for lower, upper in zip(points, points[1:]):
        if lower[2] <= target_rate <= upper[2]:
            log_span = math.log(upper[2]) - math.log(lower[2])
            if log_span <= 0.0:
                raise ValueError("all-Base rates must be strictly ordered")
            weight = (math.log(target_rate) - math.log(lower[2])) / log_span
            return {
                "status": "ok",
                "quality_db": lower[1] + weight * (upper[1] - lower[1]),
                "lower_qp": lower[0],
                "upper_qp": upper[0],
                "interpolation_weight": weight,
                "target_all_base_bpp": target_rate,
            }
    return {"status": "outside_all_base_rate_range"}


def matched_rate(anchors, quality, actual_bpp):
    match = interpolate_log_rate(anchors, quality)
    if match is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = match
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (actual_bpp / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def aggregate_results(sequences, bracket, args, checkpoint_path, checkpoint_sha,
                      provenance, elapsed_seconds):
    by_label = {point["label"]: point for point in bracket["aggregate_points"]}
    formal_k8 = by_label[C1_LABEL]
    anchors = [
        (qp,
         by_label[f"all-base-i32p{qp}"]["p_only_8_frame"][
             "frame_weighted_mean_psnr"],
         by_label[f"all-base-i32p{qp}"]["total_bpp"])
        for qp in (30, 31, 32)
    ]
    mean = aggregate_pixel(sequences, "mean_pixel")
    c1 = aggregate_pixel(sequences, "c1_alpha_0p25_pixel")
    source = aggregate_pixel(sequences, "true_latent_reference_pixel")
    actual_bpp = formal_k8["total_bpp"]
    zero_percent_threshold = quality_at_log_rate(anchors, actual_bpp)
    one_percent_threshold = quality_at_log_rate(anchors, actual_bpp / 0.99)
    if zero_percent_threshold["status"] != "ok" or one_percent_threshold["status"] != "ok":
        raise RuntimeError("registered K8 rate is outside the all-Base rate bracket")

    dot = sum(item["vector_diagnostic_for_unscaled_c1"][
        "prediction_target_dot"] for item in sequences)
    p_energy = sum(item["vector_diagnostic_for_unscaled_c1"][
        "prediction_energy"] for item in sequences)
    t_energy = sum(item["vector_diagnostic_for_unscaled_c1"][
        "target_energy"] for item in sequences)
    alpha_star = dot / max(p_energy, 1e-30)
    cosine = dot / math.sqrt(max(p_energy * t_energy, 1e-30))

    c1_gain = c1["frame_weighted_mean_psnr"] - mean["frame_weighted_mean_psnr"]
    break_even_gain = (
        zero_percent_threshold["quality_db"] - mean["frame_weighted_mean_psnr"])
    strict_gain = (
        one_percent_threshold["quality_db"] - mean["frame_weighted_mean_psnr"])
    source_gain = (
        source["frame_weighted_mean_psnr"] - mean["frame_weighted_mean_psnr"])
    c1_gap_recovery = (
        (mean["rgb_mse_level"] - c1["rgb_mse_level"])
        / max(mean["rgb_mse_level"] - source["rgb_mse_level"], 1e-30))
    route_bytes = formal_k8["stream_components"]["route_bytes"]
    route_free_bpp = (
        (formal_k8["total_bytes"] - route_bytes) * 8
        / (len(SCREEN_IDS) * args.frame_count * args.width * args.height))
    route_free_match = matched_rate(
        anchors, c1["frame_weighted_mean_psnr"], route_free_bpp)

    reference_crosses_break_even = (
        source["frame_weighted_mean_psnr"]
        >= zero_percent_threshold["quality_db"])
    reference_crosses_strict = (
        source["frame_weighted_mean_psnr"]
        >= one_percent_threshold["quality_db"])
    if reference_crosses_strict:
        diagnosis = (
            "The fixed K8 stream narrowly clears the registered rate-quality margin "
            "under a true-latent reference; the current shortfall is not explained "
            "by route bytes.  The "
            "next unresolved issue is predictor representation/training versus "
            "predictor-aware route alignment.")
    elif reference_crosses_break_even:
        diagnosis = (
            "The fixed K8 stream can reach the sampled all-Base frontier under the "
            "true-latent reference but cannot clear the registered one-percent margin.")
    else:
        diagnosis = (
            "Even the true-latent reference cannot reach the sampled all-Base frontier "
            "at this fixed K8 stream rate; route/side-information economics dominate.")

    return {
        "experiment": "adaptive_chunk_coding_stage_b_k8_true_latent_reference_v2",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "operational_rdc_point": False,
            "weights_updated": False,
            "route_selected_or_changed": False,
            "screen_split_reused_for_diagnosis": "REDS val_sharp/006..011",
            "sealed_test_not_accessed": "REDS val_sharp/024..029",
            "source_only_target_and_metrics_used_after_predictor_completed": True,
        },
        "protocol": {
            "sequence_ids": list(SCREEN_IDS),
            "frames_per_sequence": 9,
            "evaluated_p_frames_per_sequence": 8,
            "fresh_state_chunk_index0": True,
            "shared_i_qp": 32,
            "p_qp": 32,
            "nested_route_k": 8,
            "route_parent_k": 64,
            "predictor_profile": C1_PROFILE,
            "predictor_alpha": 0.25,
            "actual_rate_source": str(args.bracket_summary),
            "matched_rd_method": "linear quality / logarithmic actual-total-bpp interpolation",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "verified_training_provenance": provenance,
        },
        "registered_actual_stream": {
            "total_bytes": formal_k8["total_bytes"],
            "total_bpp": actual_bpp,
            "p_chunk_bytes_including_d1s_container": formal_k8[
                "p_chunk_bytes_including_d1s_container"],
            "components": formal_k8["stream_components"],
            "same_qp32_rate_change_percent": next(
                item["same_qp32_actual_rate_change_percent"]
                for item in bracket["k_analysis"] if item["k"] == 8),
        },
        "aggregate_p_only": {
            "mean": mean,
            "c1_alpha_0p25": c1,
            "true_latent_reference": source,
        },
        "all_base_thresholds_at_k8_rate": {
            "rd_break_even_zero_percent": zero_percent_threshold,
            "registered_one_percent_improvement": one_percent_threshold,
        },
        "distance_to_threshold": {
            "c1_gain_vs_mean_db": c1_gain,
            "gain_needed_from_mean_to_rd_break_even_db": break_even_gain,
            "gain_needed_from_mean_to_registered_one_percent_db": strict_gain,
            "true_latent_reference_gain_vs_mean_db": source_gain,
            "c1_fraction_of_psnr_gain_needed_for_rd_break_even": (
                c1_gain / max(break_even_gain, 1e-30)),
            "c1_fraction_of_psnr_gain_needed_for_one_percent": (
                c1_gain / max(strict_gain, 1e-30)),
            "one_percent_threshold_fraction_of_mean_to_source_psnr_gap": (
                strict_gain / max(source_gain, 1e-30)),
            "c1_pixel_mse_gap_recovery_vs_true_latent_reference": c1_gap_recovery,
        },
        "global_vector_diagnostic_for_unscaled_c1": {
            "prediction_target_dot": dot,
            "prediction_energy": p_energy,
            "target_energy": t_energy,
            "prediction_target_cosine": cosine,
            "closed_form_alpha_star": alpha_star,
            "closed_form_alpha_star_clipped_0_1": min(1.0, max(0.0, alpha_star)),
            "prediction_to_target_rms_ratio": math.sqrt(
                p_energy / max(t_energy, 1e-30)),
        },
        "route_overhead_counterfactual": {
            "route_bytes_removed": route_bytes,
            "route_fraction_of_total_percent": 100.0 * route_bytes / formal_k8["total_bytes"],
            "route_free_total_bpp": route_free_bpp,
            "route_free_c1_matched_rd": route_free_match,
            "operational_point": False,
        },
        "decision": {
            "true_latent_reference_crosses_zero_percent_break_even": (
                reference_crosses_break_even),
            "true_latent_reference_crosses_registered_one_percent_threshold": (
                reference_crosses_strict),
            "c1_crosses_zero_percent_break_even": (
                c1["frame_weighted_mean_psnr"]
                >= zero_percent_threshold["quality_db"]),
            "c1_crosses_registered_one_percent_threshold": (
                c1["frame_weighted_mean_psnr"]
                >= one_percent_threshold["quality_db"]),
            "diagnosis": diagnosis,
            "authorizes_c3_or_full_rollout": False,
        },
        "bracket_summary_sha256": hashlib.sha256(
            Path(args.bracket_summary).read_bytes()).hexdigest(),
        "elapsed_wall_seconds": elapsed_seconds,
        "per_sequence": sequences,
    }


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bracket_path = Path(args.bracket_summary)
    bracket = json.loads(bracket_path.read_text(encoding="utf-8"))
    if bracket["screening_decision"]["passes_screen"]:
        raise RuntimeError("K8 reference diagnostic expects the registered screen to fail")
    formal_by_sequence = formal_points_by_sequence(bracket)
    if set(formal_by_sequence) != set(SCREEN_IDS):
        raise RuntimeError("bracket summary is not the registered val006..011 screen")

    set_torch_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    torch.cuda.set_stream(torch.cuda.Stream(device=device))
    checkpoint_path = Path(args.checkpoint)
    predictor, checkpoint_payload, checkpoint_sha = load_predictor_checkpoint(
        checkpoint_path, device)
    provenance = validate_checkpoint_provenance(
        checkpoint_payload, require_phase_b=True)
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    sequences = []
    for sequence in args.screen_sequences:
        result = run_sequence(
            sequence, formal_by_sequence[sequence], i_net, p_net, predictor,
            args, output_root, device)
        sequences.append(result)
        print(json.dumps({
            "sequence": sequence,
            "mean_psnr": result["mean_pixel"]["mean_frame_psnr"],
            "c1_psnr": result["c1_alpha_0p25_pixel"]["mean_frame_psnr"],
            "true_latent_reference_psnr": result["true_latent_reference_pixel"][
                "mean_frame_psnr"],
        }), flush=True)
    torch.cuda.synchronize(device)
    summary = aggregate_results(
        sequences, bracket, args, checkpoint_path, checkpoint_sha, provenance,
        time.perf_counter() - started)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "aggregate_p_only": summary["aggregate_p_only"],
        "all_base_thresholds_at_k8_rate": summary[
            "all_base_thresholds_at_k8_rate"],
        "distance_to_threshold": summary["distance_to_threshold"],
        "global_vector_diagnostic_for_unscaled_c1": summary[
            "global_vector_diagnostic_for_unscaled_c1"],
        "decision": summary["decision"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
