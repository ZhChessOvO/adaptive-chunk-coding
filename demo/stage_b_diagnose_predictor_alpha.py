#!/usr/bin/env python3
"""Termination diagnostic for the Adaptive Chunk Coding Stage-B Lite predictor.

This is deliberately not an operational R-D-C evaluation.  It uses the true
omitted latent only after decoder-only prediction to measure direction and
calibration on the registered REDS development split.  The bitstream and all
predictor inputs remain source-free; source information is used solely for
offline diagnostics and pixel metrics.

The experiment is restricted to the first fresh-state P chunk of REDS
val/000..005.  It tests a predeclared scalar ladder applied to the learned
correction.  Alpha zero is exactly conditional mean-fill.  No weights are
trained or updated, and REDS val/024..029 remains sealed.
"""

from __future__ import annotations

import argparse
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
    parse_container,
    rgb_from_recon,
    write_container,
)
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    DEV_IDS,
    decode_chunk_inputs,
    load_frames,
    load_predictor_checkpoint,
    route_sha256,
    source_route_target,
    stable_route,
    validate_checkpoint_provenance,
)
from demo.stage_b_masked_predictor import skipped_target_blocks  # noqa: E402
from src.utils.common import set_torch_env  # noqa: E402


REGISTERED_ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fresh-state chunk-0 direction/calibration diagnostic.")
    parser.add_argument(
        "--data-root", default="data/REDS")
    parser.add_argument(
        "--checkpoint",
        default="output/stage_b_predictor_g2_phase_b_seed20260908/best.pt")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_predictor_g2_chunk0_alpha_diagnostic")
    parser.add_argument("--model-path-i", default="checkpoints/cvpr2026_image.pth.tar")
    parser.add_argument("--model-path-p", default="checkpoints/cvpr2026_video_hts.pth.tar")
    parser.add_argument("--dev-sequences", nargs="+", default=list(DEV_IDS))
    parser.add_argument("--frame-count", type=int, default=9)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--crop-x", type=int, default=384)
    parser.add_argument("--crop-y", type=int, default=64)
    parser.add_argument("--qp-route", type=int, default=32)
    parser.add_argument("--latent-block-size", type=int, default=2)
    parser.add_argument("--skip-blocks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--cuda-idx", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=REGISTERED_ALPHAS)
    return parser.parse_args()


def validate_args(args):
    expected = {
        "dev_sequences": list(DEV_IDS),
        "frame_count": 9,
        "width": 512,
        "height": 512,
        "crop_x": 384,
        "crop_y": 64,
        "qp_route": 32,
        "latent_block_size": 2,
        "skip_blocks": 64,
        "seed": 20260908,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(
                f"registered alpha diagnostic requires {name}={value!r}")
    if tuple(args.alphas) != REGISTERED_ALPHAS:
        raise ValueError(
            f"registered alpha diagnostic requires alphas={REGISTERED_ALPHAS}")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("registered alpha diagnostic requires skip-thres=0")


def vector_diagnostics(prediction, target):
    p = prediction.double().reshape(-1)
    t = target.double().reshape(-1)
    dot = torch.dot(p, t).item()
    p_energy = torch.dot(p, p).item()
    t_energy = torch.dot(t, t).item()
    error = torch.dot(p - t, p - t).item()
    cosine = dot / math.sqrt(max(p_energy * t_energy, 1e-30))
    alpha_star = dot / max(p_energy, 1e-30)
    nonzero = t != 0
    nonzero_energy = torch.dot(t[nonzero], t[nonzero]).item()
    nonzero_error = torch.dot(
        (p[nonzero] - t[nonzero]), (p[nonzero] - t[nonzero])).item()

    p_blocks = prediction.double().reshape(prediction.shape[0], -1)
    t_blocks = target.double().reshape(target.shape[0], -1)
    block_target_energy = torch.sum(t_blocks * t_blocks, dim=1)
    block_error = torch.sum((p_blocks - t_blocks) ** 2, dim=1)
    valid_blocks = block_target_energy > 0
    block_recovery = 1.0 - block_error[valid_blocks] / block_target_energy[valid_blocks]
    block_dot = torch.sum(p_blocks * t_blocks, dim=1)
    return {
        "coefficient_count": int(t.numel()),
        "nonzero_target_coefficients": int(nonzero.sum().item()),
        "target_energy": t_energy,
        "prediction_energy": p_energy,
        "prediction_to_target_rms_ratio": math.sqrt(
            p_energy / max(t_energy, 1e-30)),
        "prediction_target_dot": dot,
        "prediction_target_cosine": cosine,
        "closed_form_alpha_star": alpha_star,
        "closed_form_alpha_star_clipped_0_1": min(1.0, max(0.0, alpha_star)),
        "alpha1_all_element_recovery": 1.0 - error / max(t_energy, 1e-30),
        "alpha1_nonzero_recovery": (
            1.0 - nonzero_error / max(nonzero_energy, 1e-30)),
        "alpha1_per_block_median_recovery": float(
            torch.median(block_recovery).item()),
        "positive_block_dot_fraction": float(
            torch.mean((block_dot > 0).double()).item()),
        "prediction_mean": float(p.mean().item()),
        "target_mean": float(t.mean().item()),
    }


def pixel_metrics(source_frames, recon_frames):
    per_frame_mse = [
        float(np.mean((source.astype(np.float64) - recon.astype(np.float64)) ** 2))
        for source, recon in zip(source_frames, recon_frames)
    ]
    per_frame_psnr = [
        float("inf") if mse == 0.0
        else 10.0 * math.log10(255.0 * 255.0 / mse)
        for mse in per_frame_mse
    ]
    aggregate_mse = float(np.mean(per_frame_mse))
    return {
        "mean_frame_psnr": float(np.mean(per_frame_psnr)),
        "aggregate_psnr": (
            float("inf") if aggregate_mse == 0.0
            else 10.0 * math.log10(255.0 * 255.0 / aggregate_mse)),
        "rgb_mse_level": aggregate_mse,
        "per_frame_psnr": per_frame_psnr,
        "per_frame_mse": per_frame_mse,
    }


@torch.inference_mode()
def run_sequence(sequence, i_net, p_net, predictor, args, output_root, device):
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
        raise RuntimeError("alpha diagnostic requires exactly eight P frames")
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
    parsed_file = parse_container(payload, args.latent_block_size)
    _, q_feature, q_decoder = get_q_params(p_net, args.qp_route)
    parsed, decoded_q, mean_y, common_params = decode_chunk_inputs(
        p_net, payload, args.latent_block_size, q_feature)
    if not torch.equal(decoded_q, encoded.q_dense):
        raise RuntimeError(f"{sequence}: decoded Base q differs from encoder q")
    predicted_y, activation = predictor.apply(
        decoded_q, mean_y, common_params, skip)
    predicted_y = predicted_y.to(dtype=mean_y.dtype)
    prediction_delta = predicted_y - mean_y
    target_delta = source_route_target(
        p_net, prepared, decoded_q, skip, args.latent_block_size)
    prediction_blocks, query_ids = skipped_target_blocks(
        prediction_delta, skip, args.latent_block_size)
    target_blocks, target_query_ids = skipped_target_blocks(
        target_delta, skip, args.latent_block_size)
    if not torch.equal(query_ids, target_query_ids):
        raise RuntimeError("prediction and target query order differs")
    keep = expand_keep_mask(
        skip, args.latent_block_size, mean_y.shape, mean_y.device)
    if torch.count_nonzero(prediction_delta[keep]).item() != 0:
        raise RuntimeError("predictor modified a transmitted Base position")

    source_p = frames[1:9]
    alpha_results = {}
    for alpha in args.alphas:
        reconstructed_y = mean_y + float(alpha) * prediction_delta
        if not torch.equal(reconstructed_y[keep], mean_y[keep]):
            raise RuntimeError("alpha scaling modified transmitted Base latent")
        x_hat, _ = p_net.get_recon_and_feature(
            reconstructed_y, p_net.ctx, q_decoder)
        recon = [rgb_from_recon(frame) for frame in x_hat[:valid_count]]
        target_error = target_blocks - float(alpha) * prediction_blocks
        target_energy = torch.sum(target_blocks.double() ** 2).item()
        latent_sse = torch.sum(target_error.double() ** 2).item()
        alpha_results[f"{float(alpha):.3f}"] = {
            "alpha": float(alpha),
            "pixel": pixel_metrics(source_p, recon),
            "latent_all_element_recovery": (
                1.0 - latent_sse / max(target_energy, 1e-30)),
        }

    source_y = mean_y + target_delta
    source_x_hat, _ = p_net.get_recon_and_feature(source_y, p_net.ctx, q_decoder)
    source_recon = [rgb_from_recon(frame) for frame in source_x_hat[:valid_count]]
    vector = vector_diagnostics(prediction_blocks, target_blocks)
    component_bytes = {
        "i_payload_bytes": len(i_stream),
        "p_container_header_bytes": CONTAINER_HEADER.size,
        "global_z_bytes": len(parsed["global"]),
        "base_y_bytes": len(parsed["base"]),
        "route_bytes": len(parsed["route"]),
        "residual_bytes": len(parsed["residual"]),
        "p_chunk_total_bytes": len(payload),
        "i_plus_p_payload_bytes": len(i_stream) + len(payload),
    }
    if component_bytes["p_chunk_total_bytes"] != sum(
            value for key, value in component_bytes.items()
            if key in ("p_container_header_bytes", "global_z_bytes",
                       "base_y_bytes", "route_bytes", "residual_bytes")):
        raise RuntimeError("P chunk component bytes do not equal actual file size")
    if parsed_file["skip_blocks"].sum() != args.skip_blocks:
        raise RuntimeError("serialized route has the wrong number of Skip blocks")
    return {
        "sequence": sequence,
        "source_files": [str(paths[0]), str(paths[-1])],
        "route_sha256": route_sha256(skip),
        "skipped_blocks": int(skip.sum()),
        "checkpoint_activation": activation,
        "vector_diagnostic": vector,
        "fixed_alpha_results": alpha_results,
        "source_route_reference_pixel": pixel_metrics(source_p, source_recon),
        "actual_bitstream_components": component_bytes,
    }


def aggregate_results(sequences, args, checkpoint_path, checkpoint_payload,
                      checkpoint_sha, provenance, elapsed_seconds):
    alpha_summary = {}
    for alpha in args.alphas:
        key = f"{float(alpha):.3f}"
        per_sequence = [item["fixed_alpha_results"][key] for item in sequences]
        per_frame_mse = [
            mse for item in per_sequence for mse in item["pixel"]["per_frame_mse"]]
        per_frame_psnr = [
            psnr for item in per_sequence for psnr in item["pixel"]["per_frame_psnr"]]
        aggregate_mse = float(np.mean(per_frame_mse))
        alpha_summary[key] = {
            "alpha": float(alpha),
            "mean_frame_psnr": float(np.mean(per_frame_psnr)),
            "aggregate_psnr": 10.0 * math.log10(255.0 * 255.0 / aggregate_mse),
            "rgb_mse_level": aggregate_mse,
            "mean_latent_all_element_recovery": float(np.mean([
                item["latent_all_element_recovery"] for item in per_sequence])),
        }
    baseline = alpha_summary["0.000"]
    for key, aggregate in alpha_summary.items():
        aggregate["mean_frame_psnr_change_from_mean_db"] = (
            aggregate["mean_frame_psnr"] - baseline["mean_frame_psnr"])
        aggregate["aggregate_psnr_change_from_mean_db"] = (
            aggregate["aggregate_psnr"] - baseline["aggregate_psnr"])
        aggregate["positive_sequence_count"] = sum(
            item["fixed_alpha_results"][key]["pixel"]["mean_frame_psnr"]
            > item["fixed_alpha_results"]["0.000"]["pixel"]["mean_frame_psnr"]
            for item in sequences)

    dot = sum(item["vector_diagnostic"]["prediction_target_dot"]
              for item in sequences)
    p_energy = sum(item["vector_diagnostic"]["prediction_energy"]
                   for item in sequences)
    t_energy = sum(item["vector_diagnostic"]["target_energy"]
                   for item in sequences)
    global_cosine = dot / math.sqrt(max(p_energy * t_energy, 1e-30))
    global_alpha_star = dot / max(p_energy, 1e-30)
    candidates = [
        value for value in alpha_summary.values() if value["alpha"] > 0.0]
    best = max(candidates, key=lambda value: value["mean_frame_psnr"])
    passes = (
        best["mean_frame_psnr_change_from_mean_db"] >= 0.015
        and best["positive_sequence_count"] >= 5
        and global_cosine > 0.0)
    return {
        "experiment": "adaptive_chunk_coding_stage_b_chunk0_alpha_diagnostic_v1",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "operational_rdc_point": False,
            "source_used_only_after_decoder_only_prediction": True,
            "weights_updated": False,
            "development_split_used_for_diagnosis": "REDS val_sharp/000..005",
            "sealed_test_not_accessed": "REDS val_sharp/024..029",
            "purpose": (
                "Termination diagnostic separating correction direction failure "
                "from scalar amplitude miscalibration."),
        },
        "protocol": {
            "sequences": list(DEV_IDS),
            "frames_per_sequence": 9,
            "evaluated_p_frames_per_sequence": 8,
            "fresh_state_chunk_index": 0,
            "crop_xy": [args.crop_x, args.crop_y],
            "crop_size": [args.width, args.height],
            "qp": args.qp_route,
            "routing_unit": [2, 2, 256, 8],
            "skip_blocks": args.skip_blocks,
            "fixed_alpha_grid": list(args.alphas),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "profile": checkpoint_payload.get("profile"),
            "verified_training_provenance": provenance,
        },
        "global_vector_diagnostic": {
            "prediction_target_dot": dot,
            "prediction_energy": p_energy,
            "target_energy": t_energy,
            "prediction_target_cosine": global_cosine,
            "closed_form_alpha_star": global_alpha_star,
            "closed_form_alpha_star_clipped_0_1": min(
                1.0, max(0.0, global_alpha_star)),
            "prediction_to_target_rms_ratio": math.sqrt(
                p_energy / max(t_energy, 1e-30)),
        },
        "aggregate_fixed_alpha_results": alpha_summary,
        "termination_gate": {
            "criterion": (
                "At least one predeclared alpha>0 must improve aggregate "
                "mean-frame PSNR by >=0.015 dB, improve >=5/6 sequences, and "
                "have positive global prediction-target cosine."),
            "best_fixed_alpha": best["alpha"],
            "best_fixed_alpha_psnr_gain_db": (
                best["mean_frame_psnr_change_from_mean_db"]),
            "best_fixed_alpha_positive_sequences": best["positive_sequence_count"],
            "global_prediction_target_cosine_positive": global_cosine > 0.0,
            "passes": passes,
            "decision_if_failed": (
                "Stop the current Lite local-latent predictor family; do not "
                "scale the same architecture to C3."),
            "decision_if_passed": (
                "Run one calibrated 6x100 rollout before considering another tier."),
        },
        "elapsed_wall_seconds": elapsed_seconds,
        "per_sequence": sequences,
    }


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
    for sequence in args.dev_sequences:
        result = run_sequence(
            sequence, i_net, p_net, predictor, args, output_root, device)
        sequences.append(result)
        print(json.dumps({
            "sequence": sequence,
            "cosine": result["vector_diagnostic"]["prediction_target_cosine"],
            "alpha_star": result["vector_diagnostic"]["closed_form_alpha_star"],
            "alpha1_psnr": result["fixed_alpha_results"]["1.000"]["pixel"][
                "mean_frame_psnr"],
            "mean_psnr": result["fixed_alpha_results"]["0.000"]["pixel"][
                "mean_frame_psnr"],
        }, indent=2), flush=True)
    torch.cuda.synchronize(device)
    summary = aggregate_results(
        sequences, args, checkpoint_path, checkpoint_payload, checkpoint_sha,
        provenance, time.perf_counter() - started)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "global_vector_diagnostic": summary["global_vector_diagnostic"],
        "termination_gate": summary["termination_gate"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
