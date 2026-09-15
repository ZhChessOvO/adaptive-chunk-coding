#!/usr/bin/env python3
"""Five-point actual-byte K8 route frontier audit for Adaptive Chunk Coding.

This final no-training audit reuses the 256 singleton measurements already
saved by the predictor-aware source-route diagnostic.  It freezes two endpoints
and three all-Base-slope Lagrangian mixtures, serializes each resulting K8 mask,
and fresh-decodes mean C0 and C1 alpha=0.25.  It does not re-run candidate
search, consume a new split, update weights, or access the sealed split.
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
from demo.stage_b_bracket_route_density import (  # noqa: E402
    aggregate_point,
    p_only_metrics,
)
from demo.stage_b_evaluate_predictor_g2 import (  # noqa: E402
    LPIPSAlex,
    load_frames,
    load_predictor_checkpoint,
    read_d2l,
    route_sha256,
    run_operational_mode,
    validate_checkpoint_provenance,
)
from src.utils.common import set_torch_env  # noqa: E402


SCREEN_IDS = tuple(f"{index:03d}" for index in range(6, 12))
P_QPS = (30, 31, 32)
K = 8
C1_PROFILE = "learned-lite-c1-a025"
FAMILIES = (
    "quality",
    "lagrangian-m0p5",
    "lagrangian-m1",
    "lagrangian-m2",
    "rate",
)
MULTIPLIERS = {
    "lagrangian-m0p5": 0.5,
    "lagrangian-m1": 1.0,
    "lagrangian-m2": 2.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Registered five-mask K8 actual-rate frontier audit.")
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
        "--singleton-root",
        default="output/stage_b_k8_predictor_aware_route_oracle_val006_011_v2")
    parser.add_argument(
        "--output-dir",
        default="output/stage_b_k8_actual_rd_frontier_val006_011")
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
        "seed": 20260908,
        "reset_interval": 32,
        "decode_repeats": 3,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"registered actual-RD audit requires {name}={value!r}")
    if not math.isclose(args.skip_thres, 0.0):
        raise ValueError("registered actual-RD audit requires skip-thres=0")
    if set(args.screen_sequences) & set(f"{index:03d}" for index in range(24, 30)):
        raise ValueError("sealed REDS val/024..029 must not be accessed")


def base_label(qp):
    return f"all-base-i32p{qp}"


def mode_label(family, learned):
    suffix = "c1-a025" if learned else "mean-c0"
    return f"frontier-{family}-{suffix}-k08-i32p32"


def psnr_to_unit_mse(psnr):
    return 10.0 ** (-psnr / 10.0)


def build_routes(sequence, formal_sequence, singleton_root):
    search_path = singleton_root / sequence / "oracle_search.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    candidates = search["single_block_ranking"]
    if len(candidates) != 256 or len({item["flat_index"] for item in candidates}) != 256:
        raise RuntimeError(f"{sequence}: singleton record is not a full candidate set")
    formal_by_label = {point["label"]: point for point in formal_sequence["points"]}
    p31 = formal_by_label[base_label(31)]
    p32 = formal_by_label[base_label(32)]
    d31 = psnr_to_unit_mse(p31["p_only_metrics"]["aggregate_psnr"])
    d32 = psnr_to_unit_mse(p32["p_only_metrics"]["aggregate_psnr"])
    rate_span = p32["stream"]["total_bytes"] - p31["stream"]["total_bytes"]
    if not (d31 > d32 and rate_span > 0):
        raise RuntimeError(f"{sequence}: P31/P32 slope is not positive")
    mu = (d31 - d32) / rate_span

    def key_quality(item):
        return (
            item["c1_delta_unit_mse_vs_all_base"],
            -item["actual_y_bytes_saved_single_block"],
            item["flat_index"],
        )

    def key_rate(item):
        return (
            -item["actual_y_bytes_saved_single_block"],
            item["c1_delta_unit_mse_vs_all_base"],
            item["flat_index"],
        )

    orders = {
        "quality": sorted(candidates, key=key_quality),
        "rate": sorted(candidates, key=key_rate),
    }
    for family, multiplier in MULTIPLIERS.items():
        orders[family] = sorted(candidates, key=lambda item: (
            item["c1_delta_unit_mse_vs_all_base"]
            - multiplier * mu * item["actual_y_bytes_saved_single_block"],
            -item["actual_y_bytes_saved_single_block"],
            item["flat_index"],
        ))

    routes = {}
    records = {}
    for family in FAMILIES:
        chosen = orders[family][:K]
        mask = np.zeros((16, 16), dtype=np.bool_)
        mask.reshape(-1)[[item["flat_index"] for item in chosen]] = True
        routes[family] = mask
        records[family] = {
            "selected_flat_indices": [item["flat_index"] for item in chosen],
            "route_sha256": route_sha256(mask),
            "singleton_sum_actual_y_bytes_saved": sum(
                item["actual_y_bytes_saved_single_block"] for item in chosen),
            "singleton_sum_c1_delta_unit_mse": sum(
                item["c1_delta_unit_mse_vs_all_base"] for item in chosen),
            "lagrangian_multiplier": MULTIPLIERS.get(family),
        }
    return routes, {
        "singleton_source": str(search_path),
        "all_base_p31_p32_unit_mse_per_byte_slope_mu": mu,
        "families": records,
        "unique_route_count": len({record["route_sha256"] for record in records.values()}),
    }


def attach(point, root):
    result = dict(point)
    result["sequence_path"] = str(root / "sequence.d2l")
    result["p_only_metrics"] = p_only_metrics(point)
    return result


def verify_pair(mean, learned, route):
    keys = (
        "container_bytes", "i_payload_bytes", "global_z_bytes",
        "base_y_bytes", "route_bytes", "residual_bytes", "total_bytes")
    if {key: mean["stream"][key] for key in keys} != {
            key: learned["stream"][key] for key in keys}:
        raise RuntimeError("frontier mean/C1 stream components differ")
    mean_file = read_d2l(Path(mean["sequence_path"]))
    learned_file = read_d2l(Path(learned["sequence_path"]))
    if mean_file["i_stream"] != learned_file["i_stream"]:
        raise RuntimeError("frontier mean/C1 I payloads differ")
    if mean_file["chunks"] != learned_file["chunks"]:
        raise RuntimeError("frontier mean/C1 P payloads differ")
    expected_hash = route_sha256(route)
    if any(point["chunks"][0]["route_sha256"] != expected_hash
           for point in (mean, learned)):
        raise RuntimeError("frontier operational route differs from selected mask")
    shapes = learned["chunks"][0]["predictor_activation"][
        "learned_module_input_shapes"]
    if any(shapes[name][0] != K for name in (
            "q_projector", "common_projector", "mean_projector", "trunk")):
        raise RuntimeError("frontier C1 activation is not K8 active-only")


@torch.inference_mode()
def run_sequence(sequence, formal_sequence, routes, route_record, i_net, p_net,
                 checkpoint_path, checkpoint_sha, args, device, lpips_metric,
                 output_root):
    paths, frames = load_frames(args, sequence)
    points = []
    sequence_root = output_root / sequence
    sequence_root.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        route = routes[family]
        np.save(sequence_root / f"route_{family}.npy", route)
        mean_name = mode_label(family, False)
        c1_name = mode_label(family, True)
        mean_root = sequence_root / mean_name
        c1_root = sequence_root / c1_name
        mean = attach(run_operational_mode(
            mean_name, "mean", args.qp_route, i_net, p_net,
            checkpoint_path, checkpoint_sha, frames, args, mean_root, device,
            lpips_metric, fixed_routes=[route]), mean_root)
        learned = attach(run_operational_mode(
            c1_name, C1_PROFILE, args.qp_route, i_net, p_net,
            checkpoint_path, checkpoint_sha, frames, args, c1_root, device,
            lpips_metric, fixed_routes=[route]), c1_root)
        verify_pair(mean, learned, route)
        points.extend((mean, learned))
    i_streams = [read_d2l(Path(point["sequence_path"]))["i_stream"] for point in points]
    if any(stream != i_streams[0] for stream in i_streams[1:]):
        raise RuntimeError(f"{sequence}: frontier points do not share exact I32 payload")
    return {
        "sequence": sequence,
        "source_files": [str(paths[0]), str(paths[-1])],
        "route_construction": route_record,
        "points": points,
    }


def sequence_match(formal_sequence, target):
    formal = {point["label"]: point for point in formal_sequence["points"]}
    anchors = [(
        qp,
        formal[base_label(qp)]["p_only_metrics"]["mean_frame_psnr"],
        formal[base_label(qp)]["stream"]["total_bpp"],
    ) for qp in P_QPS]
    result = interpolate_log_rate(
        anchors, target["p_only_metrics"]["mean_frame_psnr"])
    if result is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = result
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (
            target["stream"]["total_bpp"] / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def aggregate_match(formal, target):
    anchors = [(
        qp,
        formal[base_label(qp)]["p_only_8_frame"]["frame_weighted_mean_psnr"],
        formal[base_label(qp)]["total_bpp"],
    ) for qp in P_QPS]
    result = interpolate_log_rate(
        anchors, target["p_only_8_frame"]["frame_weighted_mean_psnr"])
    if result is None:
        return {"status": "outside_all_base_quality_range"}
    matched_bpp, lower_qp, upper_qp, weight = result
    return {
        "status": "ok",
        "matched_all_base_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (
            target["total_bpp"] / matched_bpp - 1.0),
        "lower_qp": lower_qp,
        "upper_qp": upper_qp,
        "interpolation_weight": weight,
    }


def nondominated(points):
    result = []
    for point in points:
        dominated = any(
            other["rate"] <= point["rate"]
            and other["quality"] >= point["quality"]
            and (other["rate"] < point["rate"]
                 or other["quality"] > point["quality"])
            for other in points)
        if not dominated:
            result.append(point)
    result.sort(key=lambda point: point["quality"])
    return result


def match_mean_frontier(mean_points, all_base32, learned):
    raw = [{
        "id": point["label"],
        "quality": point["p_only_8_frame"]["frame_weighted_mean_psnr"],
        "rate": point["total_bpp"],
    } for point in mean_points]
    raw.append({
        "id": base_label(32),
        "quality": all_base32["p_only_8_frame"]["frame_weighted_mean_psnr"],
        "rate": all_base32["total_bpp"],
    })
    frontier = nondominated(raw)
    anchors = [(point["id"], point["quality"], point["rate"]) for point in frontier]
    result = interpolate_log_rate(
        anchors, learned["p_only_8_frame"]["frame_weighted_mean_psnr"])
    if result is None:
        return {"status": "outside_mean_frontier_quality_range", "frontier": frontier}
    matched_bpp, lower, upper, weight = result
    return {
        "status": "ok",
        "matched_mean_frontier_bpp": matched_bpp,
        "rate_change_percent": 100.0 * (
            learned["total_bpp"] / matched_bpp - 1.0),
        "lower": lower,
        "upper": upper,
        "interpolation_weight": weight,
        "frontier": frontier,
    }


def save_plot(formal, aggregates, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for qp in P_QPS:
        point = formal[base_label(qp)]
        axis.scatter(
            point["total_bpp"],
            point["p_only_8_frame"]["frame_weighted_mean_psnr"],
            marker="o", color="black", s=70)
        axis.annotate(
            f"all-Base P{qp}",
            (point["total_bpp"], point["p_only_8_frame"][
                "frame_weighted_mean_psnr"]), xytext=(4, 5),
            textcoords="offset points", fontsize=8)
    for family in FAMILIES:
        mean = aggregates[mode_label(family, False)]
        learned = aggregates[mode_label(family, True)]
        axis.plot(
            [mean["total_bpp"], learned["total_bpp"]],
            [mean["p_only_8_frame"]["frame_weighted_mean_psnr"],
             learned["p_only_8_frame"]["frame_weighted_mean_psnr"]],
            color="gray", alpha=0.5)
        axis.scatter(
            mean["total_bpp"],
            mean["p_only_8_frame"]["frame_weighted_mean_psnr"],
            marker="x", color="tab:blue", s=70)
        axis.scatter(
            learned["total_bpp"],
            learned["p_only_8_frame"]["frame_weighted_mean_psnr"],
            marker="^", color="tab:orange", s=70)
        axis.annotate(
            family,
            (learned["total_bpp"], learned["p_only_8_frame"][
                "frame_weighted_mean_psnr"]), xytext=(4, 4),
            textcoords="offset points", fontsize=7)
    axis.set_xlabel("Actual total rate (bits/pixel/frame, I32 + one P chunk)")
    axis.set_ylabel("P-only frame-weighted mean PSNR (dB)")
    axis.set_title("K8 actual-byte route frontier audit")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bracket_path = Path(args.bracket_summary)
    singleton_root = Path(args.singleton_root)
    bracket = json.loads(bracket_path.read_text(encoding="utf-8"))
    formal_sequences = {
        sequence["sequence"]: sequence for sequence in bracket["per_sequence"]}
    if set(formal_sequences) != set(SCREEN_IDS):
        raise RuntimeError("bracket summary is not the registered val006..011 screen")

    routes = {}
    route_records = {}
    for sequence in SCREEN_IDS:
        routes[sequence], route_records[sequence] = build_routes(
            sequence, formal_sequences[sequence], singleton_root)

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
    del predictor
    provenance = validate_checkpoint_provenance(
        checkpoint_payload, require_phase_b=True)
    i_net, p_net = load_models(args, device)
    i_net.eval().requires_grad_(False)
    p_net.eval().requires_grad_(False)
    lpips_metric = LPIPSAlex(device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    sequences = []
    for sequence in SCREEN_IDS:
        print(json.dumps({
            "sequence": sequence,
            "unique_routes": route_records[sequence]["unique_route_count"],
        }), flush=True)
        result = run_sequence(
            sequence, formal_sequences[sequence], routes[sequence],
            route_records[sequence], i_net, p_net, checkpoint_path,
            checkpoint_sha, args, device, lpips_metric, output_root)
        sequences.append(result)

    aggregate_inputs = [{"points": sequence["points"]} for sequence in sequences]
    aggregate_list = []
    for family in FAMILIES:
        aggregate_list.append(aggregate_point(
            aggregate_inputs, mode_label(family, False), args))
        aggregate_list.append(aggregate_point(
            aggregate_inputs, mode_label(family, True), args))
    aggregates = {point["label"]: point for point in aggregate_list}
    formal = {point["label"]: point for point in bracket["aggregate_points"]}
    all_base32 = formal[base_label(32)]
    mean_points = [aggregates[mode_label(family, False)] for family in FAMILIES]
    analyses = []
    csv_rows = []
    for family in FAMILIES:
        mean = aggregates[mode_label(family, False)]
        learned = aggregates[mode_label(family, True)]
        mean_match = aggregate_match(formal, mean)
        learned_match = aggregate_match(formal, learned)
        learned_vs_mean = match_mean_frontier(mean_points, all_base32, learned)
        per_sequence = {}
        for sequence in sequences:
            points = {point["label"]: point for point in sequence["points"]}
            formal_sequence = formal_sequences[sequence["sequence"]]
            per_sequence[sequence["sequence"]] = {
                "mean": sequence_match(
                    formal_sequence, points[mode_label(family, False)]),
                "c1": sequence_match(
                    formal_sequence, points[mode_label(family, True)]),
            }
        available = sum(
            item["c1"]["status"] == "ok" for item in per_sequence.values())
        improved = sum(
            item["c1"].get("rate_change_percent", float("inf")) < 0.0
            for item in per_sequence.values())
        learned_passes_all_base = (
            learned_match.get("rate_change_percent", float("inf")) <= -1.0
            and available == 6 and improved >= 4)
        mean_passes_all_base = (
            mean_match.get("rate_change_percent", float("inf")) <= -1.0)
        compute_bridge = (
            learned_passes_all_base and not mean_passes_all_base
            and learned_vs_mean.get("rate_change_percent", float("inf")) <= -0.5)
        analysis = {
            "family": family,
            "mean_label": mean["label"],
            "c1_label": learned["label"],
            "mean_matched_all_base": mean_match,
            "c1_matched_all_base": learned_match,
            "c1_matched_mean_c0_frontier": learned_vs_mean,
            "c1_psnr_gain_vs_same_route_mean_db": (
                learned["p_only_8_frame"]["frame_weighted_mean_psnr"]
                - mean["p_only_8_frame"]["frame_weighted_mean_psnr"]),
            "c1_positive_sequence_count_vs_mean": sum(
                next(point for point in sequence["points"]
                     if point["label"] == learned["label"])["p_only_metrics"][
                         "mean_frame_psnr"]
                > next(point for point in sequence["points"]
                       if point["label"] == mean["label"])["p_only_metrics"][
                           "mean_frame_psnr"]
                for sequence in sequences),
            "per_sequence_matched_all_base": per_sequence,
            "c1_matches_available": available,
            "c1_sequences_improved": improved,
            "mean_passes_minus_1_percent": mean_passes_all_base,
            "c1_passes_registered_all_base_gate": learned_passes_all_base,
            "supports_compute_for_bandwidth_bridge": compute_bridge,
        }
        analyses.append(analysis)
        for point, profile in ((mean, "mean"), (learned, "c1")):
            csv_rows.append({
                "family": family,
                "profile": profile,
                "total_bytes": point["total_bytes"],
                "total_bpp": point["total_bpp"],
                "base_y_bytes": point["stream_components"]["base_y_bytes"],
                "route_bytes": point["stream_components"]["route_bytes"],
                "p_only_psnr": point["p_only_8_frame"]["frame_weighted_mean_psnr"],
                "lpips_alex": point["full_9_frame"]["frame_weighted_lpips_alex"],
                "temporal_delta_mae": point["full_9_frame"][
                    "transition_weighted_temporal_delta_mae"],
                "matched_rate_percent": (
                    mean_match if profile == "mean" else learned_match).get(
                        "rate_change_percent", ""),
                "predictor_ms_p50": point["compute"]["predictor_latency_ms_p50"],
                "linear_macs_per_chunk": point["compute"]["linear_macs_per_chunk"],
            })

    any_compute_bridge = any(
        analysis["supports_compute_for_bandwidth_bridge"] for analysis in analyses)
    any_mean_pass = any(analysis["mean_passes_minus_1_percent"] for analysis in analyses)
    any_c1_pass = any(
        analysis["c1_passes_registered_all_base_gate"] for analysis in analyses)
    if any_compute_bridge:
        diagnosis = (
            "At least one predeclared route supports a C1-only compute-for-bandwidth "
            "bridge; a deployable route proxy may be investigated before any C3.")
    elif any_mean_pass or any_c1_pass:
        diagnosis = (
            "A route point clears all-Base, but the evidence does not attribute the "
            "gain to predictor compute over the C0 frontier.")
    else:
        diagnosis = (
            "None of the five predeclared actual-byte route points clears the all-Base "
            "gate; stop route search and redesign the K8/rate-weighted predictor target "
            "before any C3, Controller, Residual, or full rollout.")

    summary = {
        "experiment": "adaptive_chunk_coding_stage_b_k8_actual_rd_frontier_v1",
        "project": "Adaptive Chunk Coding",
        "scientific_scope": {
            "no_training": True,
            "singleton_candidates_reused_not_rerun": True,
            "source_aware_diagnostic_routes": True,
            "deployable_controller": False,
            "mathematical_route_optimum": False,
            "screen_split_reused": "REDS val_sharp/006..011",
            "new_split_consumed": False,
            "sealed_test_not_accessed": "REDS val_sharp/024..029",
            "multi_chunk_pareto_claim_allowed": False,
        },
        "protocol": {
            "sequence_ids": list(SCREEN_IDS),
            "frames_per_sequence": 9,
            "primary_quality_frames": "eight P frames; I excluded",
            "shared_i_qp": 32,
            "p_qp": 32,
            "route_k": K,
            "families": list(FAMILIES),
            "lagrangian_multipliers": MULTIPLIERS,
            "slope": "mu=(P31 unit MSE - P32 unit MSE)/(P32 bytes - P31 bytes)",
            "candidate_rate_proxy": "actual singleton Base-y bytes saved",
            "actual_final_rate": "complete fresh-decoded D2L bytes",
            "decode_repeats": args.decode_repeats,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "verified_training_provenance": provenance,
        },
        "aggregate_points": aggregate_list,
        "analysis": analyses,
        "decision": {
            "any_c1_minus_1_percent_all_base_pass": any_c1_pass,
            "any_mean_minus_1_percent_all_base_pass": any_mean_pass,
            "any_compute_for_bandwidth_bridge": any_compute_bridge,
            "diagnosis": diagnosis,
            "authorizes_c3_controller_residual_or_full_rollout": False,
        },
        "route_construction": route_records,
        "input_hashes": {
            "bracket_summary_sha256": hashlib.sha256(
                bracket_path.read_bytes()).hexdigest(),
            "singleton_summary_sha256": hashlib.sha256(
                (singleton_root / "summary.json").read_bytes()).hexdigest(),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "per_sequence": sequences,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_rows, output_root / "aggregate_points.csv")
    save_plot(formal, aggregates, output_root / "k8_actual_rd_frontier.png")
    print(json.dumps({
        "summary": str(summary_path),
        "analysis": analyses,
        "decision": summary["decision"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
