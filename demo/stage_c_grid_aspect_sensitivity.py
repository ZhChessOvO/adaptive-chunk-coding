#!/usr/bin/env python3
"""Grid-size and rectangular-aspect sensitivity for the selected system.

The frozen v5 controller was trained on 4x4, 128-pixel regions.  This bounded
development experiment evaluates a conservative deployment extrapolation:

* keep Generate area and the Enhance budget ratio fixed at 25 percent;
* convert every predicted regional utility/cost to 128x128-equivalent units;
* scale the Generate-boundary penalty by physical edge length;
* keep the DCVC-UF, controller, and SeedVR2 LoRA weights frozen.

The 4x4 results are not recomputed.  They are verified against the generalized
feature path and reused from the already audited LoRA-0.50 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from demo.stage_c_a800_controller import predict as predict_v1
from demo.stage_c_a800_low_budget_v2 import (
    CONTEXT_CONTENT_NAMES,
    NEIGHBOR_DELTA_NAMES,
    single_sample_context_features,
)
from demo.stage_c_a800_low_budget_v3 import (
    BASE_PROBE_NAMES,
    DIRECT_TARGET_NAMES,
    direct_targets,
)
from demo.stage_c_a800_low_budget_v4 import (
    FEATURE_NAMES_BY_KIND,
    probe_neighbor_delta,
    single_sample_features,
)
from demo.stage_c_a800_low_budget_v5 import (
    CHECKPOINT_FORMAT,
    EXPERT_KINDS,
    conservative_consensus,
    predict_expert,
)
from demo.stage_c_a800_spatial_consistency import (
    action_counts,
    generate_boundary_edges,
    generate_components,
    solve_spatial_actions,
)
from demo.stage_c_a800_teacher import (
    FEATURE_NAMES,
    SpatialLPIPS,
    local_quality,
    source_features,
)
from demo.stage_c_evaluate_seedvr2_gate import load_pngs
from demo.stage_c_seedvr2_lora_router_evaluation import QUALITY_NAMES
from demo.stage_c_three_path_roi_probe import (
    TILE_HEADER,
    LPIPSAlex,
    dcvc_stream_breakdown,
    encode_dcvc_stream,
    evaluate_variant,
    fresh_decode_baseline,
    load_codecs,
    save_frames,
    tile_boxes,
)
from src.utils.common import set_torch_env


REFERENCE_TILE_SIZE = 128
ENHANCE_BUDGET_RATIO = 0.25
GENERATE_AREA_RATIO = 0.25
REFERENCE_SPATIAL_LAMBDA = 0.004
GRID_SAMPLE_IDS = (
    "dev-s000-f00-x384-y096",
    "eval-s006-f00-x384-y096",
    "uvg-beauty-f00-center512",
    "uvg-jockey-f00-center512",
)
GRID_VARIANTS = (
    ("grid-2x2", 256),
    ("grid-4x4", 128),
    ("grid-8x8", 64),
)
ASPECT_SPECS = (
    {
        "sample_id": "aspect-s000-f00-768x512",
        "source_sample_id": "dev-s000-f00-x384-y096",
        "crop": {"x": 256, "y": 96, "width": 768, "height": 512},
        "variant": "aspect-3x2-4x6",
    },
    {
        "sample_id": "aspect-s006-f00-1024x512",
        "source_sample_id": "eval-s006-f00-x384-y096",
        "crop": {"x": 128, "y": 96, "width": 1024, "height": 512},
        "variant": "aspect-2x1-4x8",
    },
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--joint-manifest", type=Path, required=True)
    plan.add_argument("--joint-formal-root", type=Path, required=True)
    plan.add_argument("--reference-routes", type=Path, required=True)
    plan.add_argument("--reference-summary", type=Path, required=True)
    plan.add_argument("--checkpoint", type=Path, required=True)
    plan.add_argument("--route-output-root", type=Path, required=True)
    plan.add_argument("--gate-output-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    probe = commands.add_parser("probe")
    probe.add_argument("--plan", type=Path, required=True)
    probe.add_argument("--model-path-i", type=Path, required=True)
    probe.add_argument("--model-path-p", type=Path, required=True)
    probe.add_argument("--cuda-idx", type=int, default=0)

    route = commands.add_parser("route")
    route.add_argument("--plan", type=Path, required=True)
    route.add_argument("--checkpoint", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)
    route.add_argument("--lpips-batch-size", type=int, default=4)
    route.add_argument("--cuda-idx", type=int, default=0)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--route-manifest", type=Path, required=True)
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--reference-summary", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    commands.add_parser("self-test")
    return parser.parse_args()


def route_manifest(path: Path) -> dict[str, Path]:
    value = read(path)
    return {item["sample_id"]: Path(item["path"]).resolve()
            for item in value["entries"]}


def selected_actions(route: dict) -> list[int]:
    return list(map(int, route["variants"][route["selected_variant"]]["actions"]))


def reference_records(path: Path) -> dict[str, dict]:
    summary = read(path)
    return {
        item["sample_id"]: item["variants"]["old-route-lora050"]
        for item in summary["samples"]
    }


def make_plan(args: argparse.Namespace) -> None:
    paths = (
        args.joint_manifest, args.reference_routes, args.reference_summary,
        args.checkpoint,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = {item["sample_id"]: item for item in read_jsonl(args.joint_manifest)}
    routes = route_manifest(args.reference_routes)
    references = reference_records(args.reference_summary)
    if any(sample_id not in records or sample_id not in routes
           or sample_id not in references for sample_id in GRID_SAMPLE_IDS):
        raise RuntimeError("one or more fixed grid samples are absent")

    logical = []
    tasks = []
    grid_samples = []
    for sample_id in GRID_SAMPLE_IDS:
        sample = records[sample_id]
        gate = args.joint_formal_root / sample_id / "uniform_gate" / "summary.json"
        if not gate.is_file():
            raise FileNotFoundError(gate)
        reference_route = routes[sample_id]
        reference = references[sample_id]
        if selected_actions(read(reference_route)) != reference["actions"]:
            raise RuntimeError(f"reference route/result mismatch: {sample_id}")
        grid_samples.append({
            "sample": sample,
            "gate_summary": str(gate.resolve()),
            "reference_route": str(reference_route),
            "reference_physical_source": reference["physical_source"],
            "reference_output_frames": reference["output_frames"],
        })
        for variant, tile_size in GRID_VARIANTS:
            rows = sample["crop"]["height"] // tile_size
            columns = sample["crop"]["width"] // tile_size
            item = {
                "sample_id": sample_id,
                "dataset": sample["dataset"],
                "sequence": sample["sequence"],
                "data_role": sample["data_role"],
                "experiment_axis": "grid",
                "variant": variant,
                "tile_size": tile_size,
                "tile_grid": [rows, columns],
                "gate_summary": str(gate.resolve()),
                "seed": int(sample["seed"]),
                "route": (
                    str(reference_route) if tile_size == REFERENCE_TILE_SIZE
                    else str((args.route_output_root / sample_id /
                              f"{variant}.json").resolve())),
                "reuse": tile_size == REFERENCE_TILE_SIZE,
                "reuse_source": (
                    reference["physical_source"]
                    if tile_size == REFERENCE_TILE_SIZE else None),
            }
            logical.append(item)
            if not item["reuse"]:
                tasks.append(item)

    aspect_samples = []
    for offset, spec in enumerate(ASPECT_SPECS):
        source = records[spec["source_sample_id"]]
        crop = dict(spec["crop"])
        sample = {
            **source,
            "sample_id": spec["sample_id"],
            "crop": crop,
            "data_role": "development aspect-ratio sensitivity",
            "source_role": (
                "REDS validation previously used sequence; rectangular "
                "development sensitivity, not independent test evidence"),
            "seed": int(source["seed"]) + 700000 + offset,
        }
        gate = args.gate_output_root / spec["sample_id"] / "uniform_gate" / "summary.json"
        rows = crop["height"] // REFERENCE_TILE_SIZE
        columns = crop["width"] // REFERENCE_TILE_SIZE
        route_path = args.route_output_root / spec["sample_id"] / f"{spec['variant']}.json"
        item = {
            "sample_id": spec["sample_id"],
            "dataset": "REDS",
            "sequence": source["sequence"],
            "data_role": sample["data_role"],
            "experiment_axis": "aspect",
            "variant": spec["variant"],
            "tile_size": REFERENCE_TILE_SIZE,
            "tile_grid": [rows, columns],
            "gate_summary": str(gate.resolve()),
            "seed": sample["seed"],
            "route": str(route_path.resolve()),
            "reuse": False,
            "reuse_source": None,
        }
        aspect_samples.append({"sample": sample, "gate_summary": str(gate.resolve())})
        logical.append(item)
        tasks.append(item)

    value = {
        "experiment": "selected-system spatial grid and aspect sensitivity",
        "status": "frozen-plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "joint_manifest": str(args.joint_manifest.resolve()),
        "joint_manifest_sha256": sha256_file(args.joint_manifest),
        "reference_routes": str(args.reference_routes.resolve()),
        "reference_routes_sha256": sha256_file(args.reference_routes),
        "reference_summary": str(args.reference_summary.resolve()),
        "reference_summary_sha256": sha256_file(args.reference_summary),
        "controller_checkpoint": str(args.checkpoint.resolve()),
        "controller_checkpoint_sha256": sha256_file(args.checkpoint),
        "grid_samples": grid_samples,
        "aspect_samples": aspect_samples,
        "logical_evaluations": logical,
        "tasks": tasks,
        "logical_evaluation_count": len(logical),
        "physical_backend_task_count": len(tasks),
        "configuration": {
            "reference_tile_size": REFERENCE_TILE_SIZE,
            "enhance_budget_ratio": ENHANCE_BUDGET_RATIO,
            "generate_area_ratio": GENERATE_AREA_RATIO,
            "reference_spatial_lambda": REFERENCE_SPATIAL_LAMBDA,
            "roi_context_pixels": 64,
            "roi_processing_scale": 1.0,
            "feather_pixels": 16,
            "lora_strength": 0.5,
        },
        "scientific_boundary": {
            "bounded_development_sensitivity": True,
            "hard_acceptance_gate": False,
            "controller_trained_only_on_4x4_128px_regions": True,
            "alternate_grids_are_deployment_extrapolations": True,
            "utilities_and_costs_area_normalized": True,
            "generate_area_fraction_held_constant": True,
            "boundary_penalty_scaled_by_physical_edge_length": True,
            "dcvc_uf_controller_and_seedvr2_weights_frozen": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "logical_evaluations": len(logical),
        "physical_backend_tasks": len(tasks),
        "grid_samples": len(grid_samples),
        "aspect_samples": len(aspect_samples),
    }, indent=2))


def load_record_frames(record: dict) -> list[np.ndarray]:
    crop = record["crop"]
    frames = []
    for value in record["source_files"]:
        with Image.open(value) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        frame = frame[
            crop["y"]:crop["y"] + crop["height"],
            crop["x"]:crop["x"] + crop["width"],
        ]
        if frame.shape != (crop["height"], crop["width"], 3):
            raise ValueError(f"invalid crop for {value}: {frame.shape}")
        frames.append(frame.copy())
    if len(frames) != 17:
        raise ValueError(f"expected 17 frames, found {len(frames)}")
    return frames


def make_base_probes(args: argparse.Namespace) -> None:
    plan = read(args.plan)
    if not torch.cuda.is_available():
        raise RuntimeError("rectangular Base probes require CUDA")
    set_torch_env()
    torch.cuda.set_device(args.cuda_idx)
    device = torch.device(f"cuda:{args.cuda_idx}")
    codec_stream = torch.cuda.Stream(device=device)
    model_args = SimpleNamespace(
        model_path_i=args.model_path_i,
        model_path_p=args.model_path_p,
        skip_thres=0.0,
    )
    i_net, p_net = load_codecs(model_args, device)
    metric = LPIPSAlex(True)
    for entry in plan["aspect_samples"]:
        record = entry["sample"]
        summary_path = Path(entry["gate_summary"])
        if summary_path.is_file():
            value = read(summary_path)
            qp16 = value.get("variants", {}).get("qp16-base", {})
            stream = Path(qp16.get("path", ""))
            frames = summary_path.parent / "frames" / "qp16-base"
            if (stream.is_file() and stream.stat().st_size
                    == qp16.get("rate", {}).get("total_bytes")
                    and len(list(frames.glob("*.png"))) == 17):
                print(f"SKIP valid Base probe {record['sample_id']}", flush=True)
                continue
        started = time.perf_counter()
        originals = load_record_frames(record)
        stream_bytes, encode = encode_dcvc_stream(
            originals, 16, 16, i_net, p_net, device, 32)
        root = summary_path.parent
        stream_path = root / "streams" / "all_base_qp16.dcvc"
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_stream = stream_path.with_suffix(".dcvc.tmp")
        temporary_stream.write_bytes(stream_bytes)
        os.replace(temporary_stream, stream_path)
        decoded, runtime = fresh_decode_baseline(
            stream_path, 17, 1, i_net, p_net, device, codec_stream)
        rate = dcvc_stream_breakdown(stream_path.read_bytes(), 17)
        rate["bpp"] = 8.0 * rate["total_bytes"] / (
            17 * record["crop"]["height"] * record["crop"]["width"])
        save_frames(root / "frames" / "original", originals)
        save_frames(root / "frames" / "qp16-base", decoded)
        value = {
            "experiment": "rectangular scalar-QP16 Base probe",
            "status": "complete",
            "sequence": record["sequence"],
            "frame_start": record["frame_start"],
            "source_role": record["source_role"],
            "source_files": record["source_files"],
            "crop": record["crop"],
            "frames": 17,
            "configuration": {"qps": [16], "decode_repeats": 1},
            "variants": {
                "qp16-base": {
                    "qp": 16,
                    "restorer": "none",
                    "path": str(stream_path.resolve()),
                    "rate": rate,
                    "quality": evaluate_variant(originals, decoded, metric),
                    "runtime": runtime,
                    "encode": encode,
                }
            },
            "total_after_argument_parse_seconds": time.perf_counter() - started,
            "scientific_boundary": {
                "ordinary_stock_dcvc_uf_stream": True,
                "actual_file_bytes_charged": True,
                "source_rgb_available_to_decoder": False,
                "probe_not_transmitted_with_final_spatial_stream": True,
            },
        }
        atomic_json(summary_path, value)
        print(json.dumps({
            "sample_id": record["sample_id"],
            "stream_bytes": rate["total_bytes"],
            "quality": value["variants"]["qp16-base"]["quality"],
        }, ensure_ascii=False), flush=True)


def source_features_general(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    rows: int,
    columns: int,
) -> list[dict[str, float]]:
    if len(boxes) != rows * columns:
        raise ValueError("box count differs from grid")
    values = np.stack(frames).astype(np.float32) / 255.0
    luma = (0.2126 * values[..., 0] + 0.7152 * values[..., 1]
            + 0.0722 * values[..., 2])
    gradient_x = np.abs(np.diff(luma, axis=2, prepend=luma[:, :, :1]))
    gradient_y = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1, :]))
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    motion = np.abs(np.diff(luma, axis=0))
    result = []
    for index, (x, y, width, height) in enumerate(boxes):
        tile = values[:, y:y + height, x:x + width]
        tile_luma = luma[:, y:y + height, x:x + width]
        tile_gradient = gradient[:, y:y + height, x:x + width]
        tile_motion = motion[:, y:y + height, x:x + width]
        row, column = divmod(index, columns)
        features = {
            "x_center": (x + width / 2) / values.shape[2],
            "y_center": (y + height / 2) / values.shape[1],
            "border_region": float(
                row in (0, rows - 1) or column in (0, columns - 1)),
            "mean_r": float(tile[..., 0].mean()),
            "mean_g": float(tile[..., 1].mean()),
            "mean_b": float(tile[..., 2].mean()),
            "std_r": float(tile[..., 0].std()),
            "std_g": float(tile[..., 1].std()),
            "std_b": float(tile[..., 2].std()),
            "luma_mean": float(tile_luma.mean()),
            "luma_std": float(tile_luma.std()),
            "gradient_mean": float(tile_gradient.mean()),
            "edge_density": float((tile_gradient > 0.08).mean()),
            "motion_mean": float(tile_motion.mean()),
            "motion_std": float(tile_motion.std()),
            "temporal_luma_std": float(
                tile_luma.mean(axis=(1, 2)).std()),
        }
        if tuple(features) != FEATURE_NAMES:
            raise RuntimeError("feature order differs from registered schema")
        result.append(features)
    return result


def neighbor_indices(index: int, rows: int, columns: int) -> list[int]:
    row, column = divmod(index, columns)
    return [
        next_row * columns + next_column
        for next_row, next_column in (
            (row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1),
        )
        if 0 <= next_row < rows and 0 <= next_column < columns
    ]


def context_features_general(
    features: np.ndarray, rows: int, columns: int,
) -> np.ndarray:
    if features.shape != (rows * columns, len(FEATURE_NAMES)):
        raise ValueError("raw feature shape differs from grid")
    content_indices = [FEATURE_NAMES.index(name) for name in CONTEXT_CONTENT_NAMES]
    content = features[:, content_indices]
    center = content.mean(axis=0)
    scale = content.std(axis=0)
    scale[scale < 1e-8] = 1.0
    delta = content - center
    neighbor_columns = [FEATURE_NAMES.index(name) for name in NEIGHBOR_DELTA_NAMES]
    neighbor_delta = np.empty(
        (len(features), len(neighbor_columns)), dtype=np.float32)
    for index in range(len(features)):
        neighbors = neighbor_indices(index, rows, columns)
        neighbor_delta[index] = (
            features[index, neighbor_columns]
            - features[neighbors][:, neighbor_columns].mean(axis=0))
    return np.concatenate(
        [features, delta, delta / scale, neighbor_delta], axis=1
    ).astype(np.float32)


def probe_neighbor_delta_general(
    probe: np.ndarray, rows: int, columns: int,
) -> np.ndarray:
    if probe.shape != (rows * columns, len(BASE_PROBE_NAMES)):
        raise ValueError("Base-probe feature shape differs from grid")
    output = np.empty_like(probe, dtype=np.float32)
    for index in range(len(probe)):
        neighbors = neighbor_indices(index, rows, columns)
        output[index] = probe[index] - probe[neighbors].mean(axis=0)
    return output


def correction_features_general(
    raw: np.ndarray,
    probe: np.ndarray,
    kind: str,
    rows: int,
    columns: int,
) -> np.ndarray:
    if kind == "context-residual":
        output = context_features_general(raw, rows, columns)
    elif kind == "probe-residual":
        center = probe.mean(axis=0)
        scale = probe.std(axis=0)
        scale[scale < 1e-8] = 1.0
        delta = probe - center
        output = np.concatenate([
            raw, probe, delta, delta / scale,
            probe_neighbor_delta_general(probe, rows, columns),
        ], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unsupported deployed expert: {kind}")
    if output.shape[1] != len(FEATURE_NAMES_BY_KIND[kind]):
        raise RuntimeError(f"engineered schema mismatch for {kind}")
    return output


def base_probe_general(
    originals: list[np.ndarray],
    reconstruction: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    metric: SpatialLPIPS,
) -> np.ndarray:
    if len(originals) != 17 or len(reconstruction) != 17:
        raise ValueError("Base probe requires 17 source and reconstruction frames")
    maps = metric.maps(originals, reconstruction)
    output = []
    for box in boxes:
        quality = local_quality(originals, reconstruction, maps, box)
        output.append([
            quality["lpips_alex"], quality["psnr_db"],
            quality["rgb_mse"], quality["temporal_delta_mae"],
        ])
    return np.asarray(output, dtype=np.float32)


def solve_scalable_actions(
    generate_scores: list[float],
    enhance_scores: list[float],
    enhance_costs: list[int],
    byte_budget: int,
    generate_budget: int,
    spatial_lambda: float,
    rows: int,
    columns: int,
) -> tuple[list[int], dict]:
    """Solve the same spatial objective as v6 with a binary MILP.

    The original frontier DP is ideal for a four-column grid, but its state
    grows exponentially with grid width.  HiGHS handles the equivalent binary
    formulation directly and makes 8x8 and 4x8 deployment practical.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    count = rows * columns
    if not (len(generate_scores) == len(enhance_scores)
            == len(enhance_costs) == count):
        raise ValueError("score/cost count differs from grid geometry")
    edges = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if column + 1 < columns:
                edges.append((index, index + 1))
            if row + 1 < rows:
                edges.append((index, index + columns))
    variable_count = 2 * count + len(edges)
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:count] = -np.asarray(generate_scores, dtype=np.float64)
    objective[count:2 * count] = -np.asarray(
        enhance_scores, dtype=np.float64)
    objective[2 * count:] = spatial_lambda
    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.ones(variable_count, dtype=np.float64)
    for index, score in enumerate(generate_scores):
        if score <= 0:
            upper[index] = 0
    for index, score in enumerate(enhance_scores):
        if score <= 0:
            upper[count + index] = 0

    constraint_count = count + 2 + 2 * len(edges)
    matrix = lil_matrix((constraint_count, variable_count), dtype=np.float64)
    constraint_lower = np.full(constraint_count, -np.inf, dtype=np.float64)
    constraint_upper = np.zeros(constraint_count, dtype=np.float64)
    row_index = 0
    for index in range(count):
        matrix[row_index, index] = 1
        matrix[row_index, count + index] = 1
        constraint_upper[row_index] = 1
        row_index += 1
    matrix[row_index, count:2 * count] = np.asarray(enhance_costs)
    constraint_upper[row_index] = byte_budget
    row_index += 1
    matrix[row_index, :count] = 1
    constraint_upper[row_index] = generate_budget
    row_index += 1
    for edge_index, (first, second) in enumerate(edges):
        boundary = 2 * count + edge_index
        matrix[row_index, first] = 1
        matrix[row_index, second] = -1
        matrix[row_index, boundary] = -1
        row_index += 1
        matrix[row_index, first] = -1
        matrix[row_index, second] = 1
        matrix[row_index, boundary] = -1
        row_index += 1
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix.tocsr(), constraint_lower, constraint_upper),
        options={"time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"spatial MILP failed: status={result.status} message={result.message}")
    generate = result.x[:count] > 0.5
    enhance = result.x[count:2 * count] > 0.5
    actions = np.zeros(count, dtype=np.int64)
    actions[generate] = 1
    actions[enhance] = 2
    actions_list = actions.tolist()
    boundary_count = generate_boundary_edges(actions_list, rows, columns)
    unary = float(sum(
        generate_scores[index] if action == 1
        else enhance_scores[index] if action == 2 else 0.0
        for index, action in enumerate(actions_list)))
    components = generate_components(actions_list, rows, columns)
    used_bytes = int(sum(
        enhance_costs[index] for index, action in enumerate(actions_list)
        if action == 2))
    diagnostics = {
        "spatial_lambda_per_generate_boundary": spatial_lambda,
        "predicted_unary_utility": unary,
        "regularized_objective": unary - spatial_lambda * boundary_count,
        "generate_boundary_edges": boundary_count,
        "generate_component_count": len(components),
        "generate_components": components,
        "predicted_used_enhance_fallback_bytes": used_bytes,
        "used_generate_tiles": int(generate.sum()),
        "enhance_fallback_byte_budget": byte_budget,
        "generate_tile_budget": generate_budget,
        "exact_binary_milp": True,
        "milp_solver": "scipy-highs",
        "milp_status": int(result.status),
        "milp_gap": (
            float(result.mip_gap) if getattr(result, "mip_gap", None) is not None
            else None),
        "solver_seconds": time.perf_counter() - started,
        "nonpositive_generate_tiles_forbidden": True,
        "nonpositive_enhance_tiles_forbidden": True,
    }
    return actions_list, diagnostics


def infer_route(
    checkpoint: dict,
    sample: dict,
    gate_path: Path,
    tile_size: int,
    metric: SpatialLPIPS,
    device: torch.device,
) -> dict:
    gate = read(gate_path)
    crop = sample["crop"]
    height, width = int(crop["height"]), int(crop["width"])
    if height % tile_size or width % tile_size:
        raise ValueError("tile size must exactly cover the crop")
    rows, columns = height // tile_size, width // tile_size
    originals = load_record_frames(sample)
    reconstruction = load_pngs(gate_path.parent / "frames" / "qp16-base")
    if any(frame.shape[:2] != (height, width) for frame in reconstruction):
        raise ValueError("Base reconstruction geometry differs from source crop")
    qp16 = gate["variants"]["qp16-base"]
    stream_path = Path(qp16["path"])
    if (not stream_path.is_file()
            or stream_path.stat().st_size != qp16["rate"]["total_bytes"]):
        raise RuntimeError("Base-probe stream byte check failed")
    boxes = tile_boxes(width, height, tile_size)
    raw_records = source_features_general(
        originals, boxes, rows, columns)
    raw = np.asarray([
        [item[name] for name in FEATURE_NAMES] for item in raw_records
    ], dtype=np.float32)
    probe = base_probe_general(originals, reconstruction, boxes, metric)
    base_indirect = predict_v1(checkpoint["v1_anchor_checkpoint"], raw, "mlp")
    base_direct = direct_targets(base_indirect)
    expert_predictions = {}
    for kind in EXPERT_KINDS:
        engineered = correction_features_general(
            raw, probe, kind, rows, columns)
        correction_input = np.concatenate(
            [engineered, base_direct], axis=1).astype(np.float32)
        mean, std = predict_expert(
            checkpoint["experts"][kind], correction_input, device)
        expert_predictions[kind] = {
            "engineered_features": engineered,
            "correction_mean": mean,
            "correction_ensemble_std": std,
        }
    context = expert_predictions["context-residual"]
    probe_expert = expert_predictions["probe-residual"]
    correction, active, agreement, stable = conservative_consensus(
        context["correction_mean"], context["correction_ensemble_std"],
        probe_expert["correction_mean"],
        probe_expert["correction_ensemble_std"], checkpoint["confidence_z"])
    prediction = base_direct + checkpoint["correction_scale"] * correction
    calibrated = prediction.copy()
    calibrated[:, 0] -= checkpoint["calibration"][
        "absolute_generate_utility_penalty"]
    calibrated[:, 1] -= checkpoint["calibration"][
        "absolute_enhance_gain_penalty"]

    # The network predicts mean utility and byte cost for a 128x128 region.
    # Convert its outputs into a common area-integrated objective before the
    # solver so 2x2 and 8x8 do not receive different budgets by construction.
    area_scale = (tile_size / REFERENCE_TILE_SIZE) ** 2
    objective_predictions = calibrated.copy()
    objective_predictions[:, :2] *= area_scale
    reference_costs = np.maximum(
        np.rint(calibrated[:, 2]).astype(np.int64), TILE_HEADER.size + 1)
    objective_costs = np.maximum(
        np.rint(reference_costs * area_scale).astype(np.int64), 1)
    objective_predictions[:, 2] = objective_costs
    byte_budget = int(round(ENHANCE_BUDGET_RATIO * objective_costs.sum()))
    generate_budget = int(round(GENERATE_AREA_RATIO * rows * columns))
    spatial_lambda = REFERENCE_SPATIAL_LAMBDA * (
        tile_size / REFERENCE_TILE_SIZE)
    solver = solve_spatial_actions if columns <= 4 else solve_scalable_actions
    actions, diagnostics = solver(
        objective_predictions[:, 0].tolist(),
        objective_predictions[:, 1].tolist(),
        objective_costs.tolist(), byte_budget, generate_budget,
        spatial_lambda, rows, columns)
    variant_name = (
        f"area-normalized-grid-{rows}x{columns}-tile-{tile_size}-"
        "enhance-0.25-generate-area-0.25")
    return {
        "experiment": "frozen-controller grid/aspect deployment extrapolation",
        "sample": sample,
        "route_kind": "learned-controller",
        "selected_variant": variant_name,
        "configuration": {
            "tile_size": tile_size,
            "tile_grid": [rows, columns],
            "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
        },
        "raw_feature_names": list(FEATURE_NAMES),
        "base_probe_names": list(BASE_PROBE_NAMES),
        "direct_target_names": list(DIRECT_TARGET_NAMES),
        "encoder_visible_features": raw_records,
        "base_probe_features": probe.tolist(),
        "v1_indirect_predictions": base_indirect.tolist(),
        "v1_direct_anchor_predictions": base_direct.tolist(),
        "expert_predictions": {
            name: {
                key: value.tolist() for key, value in record.items()
            }
            for name, record in expert_predictions.items()
        },
        "consensus": {
            "confidence_z": checkpoint["confidence_z"],
            "correction_scale": checkpoint["correction_scale"],
            "raw_sign_agreement": agreement.tolist(),
            "uncertainty_stable": stable.tolist(),
            "active": active.tolist(),
            "active_target_count": int(np.count_nonzero(active)),
            "active_target_fraction": float(active.mean()),
            "uncalibrated_residual_correction": correction.tolist(),
        },
        "raw_128px_equivalent_direct_predictions": calibrated.tolist(),
        "direct_predictions": objective_predictions.tolist(),
        "area_normalization": {
            "reference_tile_size": REFERENCE_TILE_SIZE,
            "tile_area_scale_vs_reference": area_scale,
            "utilities_scaled_by_tile_area": True,
            "costs_scaled_by_tile_area": True,
        },
        "base_probe": {
            "qp": 16,
            "temporary_stream": str(stream_path.resolve()),
            "temporary_stream_bytes_not_charged_to_transmitted_rate": (
                qp16["rate"]["total_bytes"]),
            "reconstruction_dir": str(
                (gate_path.parent / "frames" / "qp16-base").resolve()),
            "cached_deterministic_reconstruction_reused": True,
        },
        "variants": {
            variant_name: {
                "method": "v5-consensus-area-normalized-spatial-v6",
                "actions": actions,
                "action_counts": action_counts(actions),
                "budget": {
                    "enhance_fallback_byte_ratio": ENHANCE_BUDGET_RATIO,
                    "enhance_fallback_byte_budget": byte_budget,
                    "generate_tile_budget": generate_budget,
                    "generate_area_ratio": GENERATE_AREA_RATIO,
                },
                "diagnostics": diagnostics,
            }
        },
        "scientific_boundary": {
            "controller_weights_frozen": True,
            "trained_grid": [4, 4],
            "trained_tile_size": REFERENCE_TILE_SIZE,
            "deployment_extrapolation": [rows, columns] != [4, 4],
            "ground_truth_or_teacher_targets_used_for_route": False,
            "temporary_base_stream_transmitted": False,
            "action_map_must_be_written_to_stream": True,
            "dcvc_uf_frozen": True,
            "seedvr2_lora_frozen": True,
        },
    }


def compare_reference_route(candidate: dict, reference_path: Path) -> dict:
    reference = read(reference_path)
    candidate_raw = np.asarray(candidate["encoder_visible_features"])
    reference_raw = np.asarray([
        [item[name] for name in FEATURE_NAMES]
        for item in reference["encoder_visible_features"]
    ])
    candidate_probe = np.asarray(candidate["base_probe_features"])
    reference_probe = np.asarray(reference["base_probe_features"])
    candidate_predictions = np.asarray(
        candidate["raw_128px_equivalent_direct_predictions"])
    reference_predictions = np.asarray(reference["direct_predictions"])
    prediction_max_abs = float(np.max(np.abs(
        candidate_predictions - reference_predictions)))
    if not np.array_equal(candidate_raw, reference_raw):
        raise RuntimeError("general raw features do not reproduce 4x4 reference")
    probe_max_abs = float(np.max(np.abs(candidate_probe - reference_probe)))
    if probe_max_abs > 2e-6:
        raise RuntimeError(
            "general Base-probe features do not reproduce 4x4 reference: "
            f"{probe_max_abs}")
    if prediction_max_abs > 2e-6:
        raise RuntimeError(
            f"general predictions differ from 4x4 reference: {prediction_max_abs}")
    candidate_actions = selected_actions(candidate)
    reference_actions = selected_actions(reference)
    if candidate_actions != reference_actions:
        raise RuntimeError("general solver does not reproduce 4x4 v6 actions")
    return {
        "raw_features_exact": True,
        "base_probe_feature_max_abs_difference": probe_max_abs,
        "prediction_max_abs_difference": prediction_max_abs,
        "actions_exact": True,
    }


def route_all(args: argparse.Namespace) -> None:
    plan = read(args.plan)
    if sha256_file(args.checkpoint) != plan["controller_checkpoint_sha256"]:
        raise RuntimeError("controller checkpoint differs from frozen plan")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not conservative controller v5")
    if checkpoint.get("selected_method") != "context-probe-consensus":
        raise ValueError("sensitivity requires the deployed consensus controller")
    device = torch.device(
        f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    metric = SpatialLPIPS(device, args.lpips_batch_size)
    by_sample = {
        item["sample"]["sample_id"]: item["sample"]
        for item in plan["grid_samples"] + plan["aspect_samples"]
    }
    entries = []
    regressions = []
    # First prove that the generalized feature geometry is identical at 4x4.
    for entry in plan["grid_samples"]:
        sample = entry["sample"]
        candidate = infer_route(
            checkpoint, sample, Path(entry["gate_summary"]),
            REFERENCE_TILE_SIZE, metric, device)
        regression = compare_reference_route(
            candidate, Path(entry["reference_route"]))
        regressions.append({"sample_id": sample["sample_id"], **regression})
        print(f"REGRESSION exact 4x4 {sample['sample_id']}", flush=True)
    for task in plan["tasks"]:
        route_path = Path(task["route"])
        if route_path.is_file():
            route = read(route_path)
            config = route.get("configuration", {})
            if (config.get("tile_size") == task["tile_size"]
                    and config.get("tile_grid") == task["tile_grid"]):
                print(f"SKIP valid route {task['sample_id']}/{task['variant']}",
                      flush=True)
                entries.append({
                    "sample_id": task["sample_id"],
                    "variant": task["variant"],
                    "path": str(route_path.resolve()),
                    "actions": selected_actions(route),
                })
                continue
        sample = by_sample[task["sample_id"]]
        route = infer_route(
            checkpoint, sample, Path(task["gate_summary"]),
            int(task["tile_size"]), metric, device)
        atomic_json(route_path, route)
        entries.append({
            "sample_id": task["sample_id"],
            "variant": task["variant"],
            "path": str(route_path.resolve()),
            "actions": selected_actions(route),
            "action_counts": action_counts(selected_actions(route)),
        })
        print(json.dumps(entries[-1], ensure_ascii=False), flush=True)
    value = {
        "experiment": "generalized frozen v5 routes for grid/aspect sensitivity",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "entries": entries,
        "four_by_four_regression": regressions,
        "scientific_boundary": plan["scientific_boundary"],
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "route_count": len(entries),
        "four_by_four_exact_regressions": len(regressions),
    }, indent=2))


def generic_outside_generate_unchanged(
    decoded: list[np.ndarray],
    output: list[np.ndarray],
    actions: list[int],
    rows: int,
    columns: int,
    tile_size: int,
) -> bool:
    mask = np.asarray(actions, dtype=np.int64).reshape(rows, columns) == 1
    mask = np.repeat(np.repeat(mask, tile_size, axis=0), tile_size, axis=1)
    outside = ~mask
    return all(np.array_equal(first[outside], second[outside])
               for first, second in zip(decoded, output))


def materialized_record(run_root: Path, task: dict) -> dict:
    root = run_root / "formal" / "evaluation" / task["sample_id"] / task["variant"]
    codec = root / "codec"
    backend = root / "lora050"
    if not (backend / "backend.complete").is_file():
        raise RuntimeError(f"backend is incomplete: {backend}")
    encode = read(codec / "encode_summary.json")
    decode = read(codec / "decode_summary.json")
    roi = read(root / "roi" / "manifest.json")
    batch = read(backend / "restore" / "roi_batch_metadata.json")
    evaluation = read(backend / "evaluation" / "summary.json")
    stream = Path(encode["stream"])
    if (not stream.is_file() or stream.stat().st_size != encode["stream_bytes"]
            or decode["stream_bytes"] != encode["stream_bytes"]
            or evaluation["fresh_decode_regression"].get("pixel_exact") is not True
            or batch.get("complete") is not True):
        raise RuntimeError(f"stream/decode/restore audit failed: {root}")
    route = read(Path(task["route"]))
    actions = selected_actions(route)
    roi_actions = np.asarray(roi["actions"]).reshape(-1).tolist()
    if actions != roi_actions:
        raise RuntimeError(f"route/ROI action mismatch: {root}")
    decoded = load_pngs(codec / "fresh_decode")
    output = load_pngs(Path(evaluation["output_frames"]))
    rows, columns = task["tile_grid"]
    outside_exact = generic_outside_generate_unchanged(
        decoded, output, actions, rows, columns, task["tile_size"])
    if not outside_exact:
        raise RuntimeError(f"non-Generate pixels changed: {root}")
    height, width = decoded[0].shape[:2]
    quality = evaluation["quality"]["roi-spatial-bge-stitched"]
    return {
        **{key: task[key] for key in (
            "sample_id", "dataset", "sequence", "data_role",
            "experiment_axis", "variant", "tile_size", "tile_grid", "seed")},
        "reused": False,
        "physical_source": str(backend.resolve()),
        "route": task["route"],
        "actions": actions,
        "action_counts": action_counts(actions),
        "action_fractions": {
            key: value / len(actions)
            for key, value in action_counts(actions).items()
        },
        "generate_boundary_edges": generate_boundary_edges(
            actions, rows, columns),
        "generate_component_count": len(
            generate_components(actions, rows, columns)),
        "quality": {name: quality[name] for name in QUALITY_NAMES},
        "actual_on_disk_bytes": int(encode["stream_bytes"]),
        "bpp": 8.0 * encode["stream_bytes"] / (17 * height * width),
        "runtime_seconds": evaluation["runtime"]["full_roi_pipeline_seconds"],
        "peak_cuda_allocated_bytes": batch["peak_cuda_allocated_bytes"],
        "processing_pixel_ratio_vs_full": evaluation[
            "processing_pixel_ratio_vs_full"],
        "component_count": evaluation["component_count"],
        "fresh_decode_pixel_exact": True,
        "outside_generate_pixels_unchanged": outside_exact,
        "output_frames": evaluation["output_frames"],
        "visual": evaluation["visual"],
        "frame_height": height,
        "frame_width": width,
    }


def reused_record(task: dict, reference: dict) -> dict:
    frames = load_pngs(Path(reference["output_frames"]))
    height, width = frames[0].shape[:2]
    rows, columns = task["tile_grid"]
    actions = list(map(int, reference["actions"]))
    if len(actions) != rows * columns:
        raise RuntimeError("reused reference action count differs from task")
    stream = Path(reference["stream"])
    if not stream.is_file() or stream.stat().st_size != reference["actual_on_disk_bytes"]:
        raise RuntimeError("reused reference stream byte audit failed")
    return {
        **{key: task[key] for key in (
            "sample_id", "dataset", "sequence", "data_role",
            "experiment_axis", "variant", "tile_size", "tile_grid", "seed")},
        "reused": True,
        "reuse_source": task["reuse_source"],
        "physical_source": reference["physical_source"],
        "route": task["route"],
        "actions": actions,
        "action_counts": reference["action_counts"],
        "action_fractions": {
            key: value / len(actions)
            for key, value in reference["action_counts"].items()
        },
        "generate_boundary_edges": reference["generate_boundary_edges"],
        "generate_component_count": reference["generate_component_count"],
        "quality": reference["quality"],
        "actual_on_disk_bytes": reference["actual_on_disk_bytes"],
        "bpp": 8.0 * reference["actual_on_disk_bytes"] / (17 * height * width),
        "runtime_seconds": reference["runtime_seconds"],
        "peak_cuda_allocated_bytes": reference["peak_cuda_allocated_bytes"],
        "processing_pixel_ratio_vs_full": None,
        "component_count": reference["component_count"],
        "fresh_decode_pixel_exact": reference["fresh_decode_pixel_exact"],
        "outside_generate_pixels_unchanged": reference[
            "outside_generate_pixels_unchanged"],
        "output_frames": reference["output_frames"],
        "visual": reference["visual"],
        "frame_height": height,
        "frame_width": width,
    }


def average(records: list[dict], field: str) -> float:
    return float(statistics.mean(float(item[field]) for item in records))


def panel_frame(
    frame: np.ndarray, title: str, subtitle: str, cell: tuple[int, int],
) -> Image.Image:
    cell_width, cell_height = cell
    banner = 50
    image = Image.fromarray(frame)
    image.thumbnail((cell_width, cell_height - banner), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (cell_width, cell_height), (238, 238, 238))
    output.paste(image, ((cell_width - image.width) // 2, banner))
    draw = ImageDraw.Draw(output)
    bold = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    normal = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    draw.text((6, 5), title, fill="black", font=bold)
    draw.text((6, 27), subtitle, fill="black", font=normal)
    return output


def action_image(record: dict) -> np.ndarray:
    colors = np.asarray([
        (74, 144, 226), (242, 160, 42), (70, 170, 92),
    ], dtype=np.uint8)
    rows, columns = record["tile_grid"]
    values = np.asarray(record["actions"], dtype=np.int64).reshape(rows, columns)
    return np.repeat(
        np.repeat(colors[values], record["tile_size"], axis=0),
        record["tile_size"], axis=1)


def write_visual(path: Path, records: list[dict], plan: dict) -> None:
    by_key = {(item["sample_id"], item["variant"]): item for item in records}
    sample_lookup = {
        item["sample"]["sample_id"]: item["sample"]
        for item in plan["grid_samples"] + plan["aspect_samples"]
    }
    cell = (300, 280)
    rows = []
    for sample_id in (GRID_SAMPLE_IDS[0], GRID_SAMPLE_IDS[-1]):
        sample = sample_lookup[sample_id]
        source = load_record_frames(sample)[8]
        panels = [panel_frame(source, f"{sample_id} GT", "frame 9", cell)]
        for variant, _ in GRID_VARIANTS:
            record = by_key[(sample_id, variant)]
            frame = load_pngs(Path(record["output_frames"]))[8]
            q = record["quality"]
            panels.append(panel_frame(
                frame, variant,
                f"LPIPS {q['lpips_alex']:.4f} | {record['actual_on_disk_bytes']} B",
                cell))
        rows.append(panels)
    for spec in ASPECT_SPECS:
        sample_id, variant = spec["sample_id"], spec["variant"]
        sample = sample_lookup[sample_id]
        record = by_key[(sample_id, variant)]
        source = load_record_frames(sample)[8]
        output = load_pngs(Path(record["output_frames"]))[8]
        amap = action_image(record)
        rows.append([
            panel_frame(source, f"{sample_id} GT", "frame 9", cell),
            panel_frame(output, variant,
                        f"LPIPS {record['quality']['lpips_alex']:.4f}", cell),
            panel_frame(amap, "Action map", "blue Base / orange Generate / green Enhance", cell),
            panel_frame(output, "Fresh result",
                        f"{record['frame_width']}x{record['frame_height']} | "
                        f"{record['actual_on_disk_bytes']} B", cell),
        ])
    canvas = Image.new(
        "RGB", (4 * cell[0], len(rows) * cell[1]), (220, 220, 220))
    for row_index, panels in enumerate(rows):
        for column_index, panel in enumerate(panels):
            canvas.paste(panel, (column_index * cell[0], row_index * cell[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def summarize(args: argparse.Namespace) -> None:
    plan = read(args.plan)
    route_manifest_value = read(args.route_manifest)
    if route_manifest_value["plan_sha256"] != sha256_file(args.plan):
        raise RuntimeError("route manifest and plan differ")
    references = reference_records(args.reference_summary)
    records = []
    for task in plan["logical_evaluations"]:
        if task["reuse"]:
            records.append(reused_record(task, references[task["sample_id"]]))
        else:
            records.append(materialized_record(args.run_root, task))
    grid = {}
    for variant, _ in GRID_VARIANTS:
        members = [item for item in records if item["variant"] == variant]
        grid[variant] = {
            "sample_count": len(members),
            "mean_lpips_alex": average(
                [{"value": item["quality"]["lpips_alex"]} for item in members],
                "value"),
            "mean_psnr_db": average(
                [{"value": item["quality"]["psnr_db"]} for item in members],
                "value"),
            "mean_temporal_delta_mae": average(
                [{"value": item["quality"]["temporal_delta_mae"]}
                 for item in members], "value"),
            "mean_bpp": average(members, "bpp"),
            "mean_runtime_seconds": average(members, "runtime_seconds"),
            "mean_generate_fraction": average([
                {"value": item["action_fractions"]["Generate"]}
                for item in members], "value"),
            "mean_enhance_fraction": average([
                {"value": item["action_fractions"]["Enhance"]}
                for item in members], "value"),
        }
    deltas = []
    by_key = {(item["sample_id"], item["variant"]): item for item in records}
    for sample_id in GRID_SAMPLE_IDS:
        reference = by_key[(sample_id, "grid-4x4")]
        for variant in ("grid-2x2", "grid-8x8"):
            value = by_key[(sample_id, variant)]
            deltas.append({
                "sample_id": sample_id,
                "variant": variant,
                "lpips_delta_vs_4x4": (
                    value["quality"]["lpips_alex"]
                    - reference["quality"]["lpips_alex"]),
                "psnr_delta_db_vs_4x4": (
                    value["quality"]["psnr_db"]
                    - reference["quality"]["psnr_db"]),
                "temporal_delta_mae_change_vs_4x4": (
                    value["quality"]["temporal_delta_mae"]
                    - reference["quality"]["temporal_delta_mae"]),
                "bpp_delta_vs_4x4": value["bpp"] - reference["bpp"],
                "runtime_delta_seconds_vs_4x4": (
                    value["runtime_seconds"] - reference["runtime_seconds"]),
            })
    visual = args.output.parent / "visuals" / "grid_and_aspect_comparison.png"
    write_visual(visual, records, plan)
    verification = {
        "fresh_decode_pixel_exact_count": sum(
            item["fresh_decode_pixel_exact"] for item in records),
        "outside_generate_exact_count": sum(
            item["outside_generate_pixels_unchanged"] for item in records),
        "logical_record_count": len(records),
        "materialized_record_count": sum(not item["reused"] for item in records),
        "reused_record_count": sum(item["reused"] for item in records),
        "four_by_four_regression": route_manifest_value[
            "four_by_four_regression"],
    }
    result = {
        "experiment": plan["experiment"],
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(set(item["sample_id"] for item in records)),
        "logical_evaluation_count": len(records),
        "physical_backend_task_count": plan["physical_backend_task_count"],
        "configuration": plan["configuration"],
        "grid_aggregate": grid,
        "grid_deltas_vs_4x4": deltas,
        "aspect_results": [
            item for item in records if item["experiment_axis"] == "aspect"],
        "records": records,
        "verification": verification,
        "fixed_visual": str(visual.resolve()),
        "scientific_boundary": plan["scientific_boundary"],
    }
    atomic_json(args.output.resolve(), result)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_id", "dataset", "variant", "tile_size", "tile_grid",
            "actual_on_disk_bytes", "bpp", "lpips_alex", "psnr_db",
            "temporal_delta_mae", "runtime_seconds", "generate_fraction",
            "enhance_fraction", "reused"))
        writer.writeheader()
        for item in records:
            writer.writerow({
                "sample_id": item["sample_id"],
                "dataset": item["dataset"],
                "variant": item["variant"],
                "tile_size": item["tile_size"],
                "tile_grid": "x".join(map(str, item["tile_grid"])),
                "actual_on_disk_bytes": item["actual_on_disk_bytes"],
                "bpp": item["bpp"],
                "lpips_alex": item["quality"]["lpips_alex"],
                "psnr_db": item["quality"]["psnr_db"],
                "temporal_delta_mae": item["quality"]["temporal_delta_mae"],
                "runtime_seconds": item["runtime_seconds"],
                "generate_fraction": item["action_fractions"]["Generate"],
                "enhance_fraction": item["action_fractions"]["Enhance"],
                "reused": item["reused"],
            })
    print(json.dumps({
        "summary": str(args.output.resolve()),
        "grid_aggregate": grid,
        "aspect_count": len(result["aspect_results"]),
        "verification": verification,
        "visual": str(visual.resolve()),
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    generator = np.random.default_rng(20260921)
    raw = generator.normal(size=(16, len(FEATURE_NAMES))).astype(np.float32)
    probe = generator.normal(size=(16, len(BASE_PROBE_NAMES))).astype(np.float32)
    if not np.array_equal(
            context_features_general(raw, 4, 4),
            single_sample_context_features(raw)):
        raise RuntimeError("generic context feature regression failed")
    if not np.array_equal(
            probe_neighbor_delta_general(probe, 4, 4),
            probe_neighbor_delta(probe)):
        raise RuntimeError("generic probe-neighbor regression failed")
    for kind in EXPERT_KINDS:
        if not np.array_equal(
                correction_features_general(raw, probe, kind, 4, 4),
                single_sample_features(raw, probe, kind)):
            raise RuntimeError(f"generic feature regression failed: {kind}")
    frames = [generator.integers(
        0, 256, size=(512, 512, 3), dtype=np.uint8) for _ in range(3)]
    boxes = tile_boxes(512, 512, 128)
    first = source_features(frames, boxes)
    second = source_features_general(frames, boxes, 4, 4)
    if first != second:
        raise RuntimeError("generic raw source feature regression failed")
    actions, diagnostics = solve_spatial_actions(
        [1.0] * 4, [0.5] * 4, [100] * 4, 100, 1, 0.008, 2, 2)
    if len(actions) != 4 or diagnostics["used_generate_tiles"] != 1:
        raise RuntimeError("2x2 spatial solver smoke failed")
    small_scores = generator.uniform(-0.02, 0.08, 16).tolist()
    small_enhance = generator.uniform(-0.02, 0.16, 16).tolist()
    small_costs = generator.integers(100, 1500, 16).tolist()
    small_budget = round(0.25 * sum(small_costs))
    _, exact_diagnostics = solve_spatial_actions(
        small_scores, small_enhance, small_costs, small_budget, 4, 0.004, 4, 4)
    _, milp_diagnostics = solve_scalable_actions(
        small_scores, small_enhance, small_costs, small_budget, 4, 0.004, 4, 4)
    if not math.isclose(
            exact_diagnostics["regularized_objective"],
            milp_diagnostics["regularized_objective"], abs_tol=1e-8):
        raise RuntimeError("MILP objective does not reproduce exact 4x4 solver")
    scores = generator.uniform(-0.02, 0.08, 64).tolist()
    enhance = generator.uniform(-0.02, 0.16, 64).tolist()
    costs = generator.integers(100, 1500, 64).tolist()
    large_actions, large_diagnostics = solve_scalable_actions(
        scores, enhance, costs, round(0.25 * sum(costs)), 16, 0.002, 8, 8)
    if (len(large_actions) != 64
            or large_diagnostics["predicted_used_enhance_fallback_bytes"]
            > round(0.25 * sum(costs))
            or large_diagnostics["used_generate_tiles"] > 16):
        raise RuntimeError("8x8 spatial MILP smoke failed")
    print(json.dumps({
        "status": "passed",
        "generic_4x4_features_exact": True,
        "rectangular_neighbor_count": len(neighbor_indices(0, 4, 6)),
        "two_by_two_solver_actions": actions,
        "eight_by_eight_milp_seconds": large_diagnostics["solver_seconds"],
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "plan":
        make_plan(args)
    elif args.mode == "probe":
        make_base_probes(args)
    elif args.mode == "route":
        route_all(args)
    elif args.mode == "summarize":
        summarize(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
