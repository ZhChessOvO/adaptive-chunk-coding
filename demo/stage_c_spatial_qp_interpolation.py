#!/usr/bin/env python3
"""Interpolate frozen and spatial-QP-finetuned DCVC-UF checkpoints.

The 1000-step checkpoint improves REDS rate-distortion but over-adapts away
from UVG.  This development sweep evaluates three fixed weight-space points
between the released model (alpha=0) and the trained model (alpha=1).  It uses
the already frozen 37 mixed routes and six three-QP regression clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_feather_diagnostic import boundary_metrics
from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import panel
from demo.stage_c_spatial_qp_finetune_eval import (
    ACTION_COLORS,
    BOUNDARY_KEYS,
    QUALITY_KEYS,
    UNIFORM_SAMPLE_IDS,
    VISUAL_SAMPLE_IDS,
    aggregate,
    atomic_json,
    atomic_text,
    bd_rate_percent,
    finite_mean,
    read_json,
    sha256_file,
)
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant
from src.utils.common import get_state_dict


DEFAULT_ALPHAS = (0.25, 0.50, 0.75)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def alpha_tag(alpha: float) -> str:
    value = int(round(alpha * 100))
    if not math.isclose(alpha, value / 100, abs_tol=1e-9):
        raise ValueError("alpha must have at most two decimal places")
    return f"alpha{value:03d}"


def parse_alphas(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(not 0 < item < 1 for item in values):
        raise argparse.ArgumentTypeError("alphas must be comma-separated values in (0,1)")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("alphas must be unique")
    for item in values:
        alpha_tag(item)
    return values


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--endpoint-protocol", type=Path, required=True)
    prepare.add_argument("--endpoint-summary", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--frozen-image", type=Path, required=True)
    prepare.add_argument("--frozen-video", type=Path, required=True)
    prepare.add_argument("--tuned-image", type=Path, required=True)
    prepare.add_argument("--tuned-video", type=Path, required=True)
    prepare.add_argument(
        "--alphas", type=parse_alphas,
        default=DEFAULT_ALPHAS)

    listing = subparsers.add_parser("list-tasks")
    listing.add_argument("--protocol", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--protocol", type=Path, required=True)
    summarize.add_argument("--endpoint-summary", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    subparsers.add_parser("self-test")
    return parser.parse_args(argv)


def blend_state_dicts(
    frozen: dict[str, torch.Tensor],
    tuned: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if frozen.keys() != tuned.keys():
        missing = sorted(frozen.keys() ^ tuned.keys())
        raise RuntimeError(f"checkpoint keys differ: {missing[:5]}")
    output = {}
    for name in frozen:
        left, right = frozen[name], tuned[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"checkpoint tensor differs: {name}")
        if left.is_floating_point() or left.is_complex():
            output[name] = torch.lerp(left, right, alpha)
        else:
            if not torch.equal(left, right):
                raise RuntimeError(f"non-floating checkpoint tensor changed: {name}")
            output[name] = left.clone()
    return output


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def prepare_interpolations(
    output_dir: Path,
    frozen_path: Path,
    tuned_path: Path,
    alphas: tuple[float, ...],
    kind: str,
) -> dict[str, dict]:
    frozen_sha = sha256_file(frozen_path)
    tuned_sha = sha256_file(tuned_path)
    checkpoint_root = output_dir / "checkpoints"
    expected = {}
    missing = []
    for alpha in alphas:
        tag = alpha_tag(alpha)
        target = checkpoint_root / tag / f"{kind}_model.pth.tar"
        metadata = target.with_suffix(target.suffix + ".json")
        expected[tag] = {
            "alpha": alpha,
            "path": target,
            "metadata": metadata,
        }
        valid = False
        if target.is_file() and metadata.is_file():
            record = read_json(metadata)
            valid = (
                record.get("alpha") == alpha
                and record.get("frozen_sha256") == frozen_sha
                and record.get("tuned_sha256") == tuned_sha
                and record.get("output_sha256") == sha256_file(target)
            )
        if not valid:
            missing.append(tag)

    if missing:
        frozen = get_state_dict(str(frozen_path))
        tuned = get_state_dict(str(tuned_path))
        for alpha in alphas:
            tag = alpha_tag(alpha)
            if tag not in missing:
                continue
            target = expected[tag]["path"]
            blended = blend_state_dicts(frozen, tuned, alpha)
            atomic_torch_save(target, {"state_dict": blended})
            record = {
                "created_utc": utc_now(),
                "kind": kind,
                "alpha": alpha,
                "formula": "(1-alpha)*frozen + alpha*1000-step",
                "frozen": str(frozen_path.resolve()),
                "frozen_sha256": frozen_sha,
                "tuned": str(tuned_path.resolve()),
                "tuned_sha256": tuned_sha,
                "output": str(target.resolve()),
                "output_bytes": target.stat().st_size,
                "output_sha256": sha256_file(target),
                "tensor_count": len(blended),
                "parameter_or_buffer_values": sum(
                    value.numel() for value in blended.values()),
            }
            atomic_json(expected[tag]["metadata"], record)
            del blended
        del frozen, tuned

    return {
        tag: read_json(value["metadata"])
        for tag, value in expected.items()
    }


def prepare_main(args: argparse.Namespace) -> None:
    endpoint_protocol = read_json(args.endpoint_protocol)
    endpoint_summary = read_json(args.endpoint_summary)
    if endpoint_protocol["sample_count"] != 37 or endpoint_summary["task_count"] != 110:
        raise RuntimeError("expected the completed 37-sample endpoint comparison")
    if endpoint_summary["status"] != "complete":
        raise RuntimeError("endpoint comparison is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = prepare_interpolations(
        args.output_dir, args.frozen_image, args.tuned_image, args.alphas, "image")
    video = prepare_interpolations(
        args.output_dir, args.frozen_video, args.tuned_video, args.alphas, "video_hts")

    baseline_tasks = [
        task for task in endpoint_protocol["tasks"]
        if task["model_role"] == "frozen"
    ]
    if len(baseline_tasks) != 55:
        raise RuntimeError("expected 37 mixed + 18 uniform frozen endpoint tasks")
    tasks = []
    checkpoints = {}
    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        checkpoints[tag] = {
            "alpha": alpha,
            "image": image[tag],
            "video": video[tag],
        }
        for source in baseline_tasks:
            tasks.append({
                **source,
                "task_id": f"{source['task_id'].removesuffix('--frozen')}--{tag}",
                "alpha": alpha,
                "model_role": tag,
                "image": image[tag]["output"],
                "video": video[tag]["output"],
                "image_sha256": image[tag]["output_sha256"],
                "video_sha256": video[tag]["output_sha256"],
                "checkpoint_role": "spatial-qp-interpolated",
                "output_rel": source["output_rel"].removesuffix("/frozen") + f"/{tag}",
            })
    protocol = {
        "experiment": "DCVC-UF frozen-to-1000-step weight interpolation",
        "status": "fixed-before-interpolation-evaluation",
        "created_utc": utc_now(),
        "git_commit_at_prepare": git_commit(),
        "implementation_sha256": sha256_file(Path(__file__)),
        "endpoint_protocol": str(args.endpoint_protocol.resolve()),
        "endpoint_protocol_sha256": sha256_file(args.endpoint_protocol),
        "endpoint_summary": str(args.endpoint_summary.resolve()),
        "endpoint_summary_sha256": sha256_file(args.endpoint_summary),
        "alphas": list(args.alphas),
        "candidate_tags": [alpha_tag(alpha) for alpha in args.alphas],
        "checkpoints": checkpoints,
        "sample_count": 37,
        "tasks_per_alpha": len(baseline_tasks),
        "total_task_count": len(tasks),
        "tasks": tasks,
        "selection_rule": {
            "primary": (
                "lowest combined mean LPIPS BD-rate on the six fixed uniform-QP "
                "curves among alpha 0, 0.25, 0.50, 0.75, 1"),
            "secondary": "lowest combined mean PSNR BD-rate",
            "hard_acceptance_threshold": False,
            "mixed_routes_used_for_confirmation_not_alpha_selection": True,
        },
        "scientific_boundary": {
            "development_and_cross_distribution_candidate_selection": True,
            "no_new_independent_test_claim": True,
            "routes_and_samples_reused_from_endpoint_protocol": True,
            "no_additional_gradient_updates": True,
            "actual_stream_and_fresh_decode_required": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps({
        "protocol": str((args.output_dir / "protocol.json").resolve()),
        "alphas": list(args.alphas),
        "tasks": len(tasks),
    }, indent=2))


def list_tasks_main(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    fields = (
        "task_id", "sample_id", "family", "model_role", "alpha", "gate",
        "route", "image", "video", "checkpoint_role", "output_rel",
    )
    for task in protocol["tasks"]:
        print("\t".join(str(task[field]) for field in fields))


def metric_delta_summary(reference: list[dict], candidate: list[dict]) -> dict:
    reference_by_id = {record["sample_id"]: record for record in reference}
    candidate_by_id = {record["sample_id"]: record for record in candidate}
    if reference_by_id.keys() != candidate_by_id.keys():
        raise RuntimeError("mixed-route sample sets differ")
    rows = []
    for sample_id in sorted(reference_by_id):
        frozen = reference_by_id[sample_id]
        other = candidate_by_id[sample_id]
        rows.append({
            "sample_id": sample_id,
            "dataset": frozen["dataset"],
            "stream_byte_delta": other["stream_bytes"] - frozen["stream_bytes"],
            **{
                f"{key}_delta": other["quality"][key] - frozen["quality"][key]
                for key in QUALITY_KEYS
            },
            **{
                f"boundary_{key}_delta": (
                    None if other["boundary"][key] is None else
                    other["boundary"][key] - frozen["boundary"][key])
                for key in BOUNDARY_KEYS
            },
        })
    return {
        "pair_count": len(rows),
        "mean_stream_byte_delta": finite_mean([row["stream_byte_delta"] for row in rows]),
        "mean_lpips_delta": finite_mean([row["lpips_alex_delta"] for row in rows]),
        "mean_psnr_delta_db": finite_mean([row["psnr_db_delta"] for row in rows]),
        "mean_temporal_delta_mae_delta": finite_mean([
            row["temporal_delta_mae_delta"] for row in rows]),
        "lpips_improved_count": sum(row["lpips_alex_delta"] < 0 for row in rows),
        "psnr_improved_count": sum(row["psnr_db_delta"] > 0 for row in rows),
        "temporal_improved_count": sum(
            row["temporal_delta_mae_delta"] < 0 for row in rows),
        "mean_boundary_gradient_error_delta": finite_mean([
            row["boundary_gradient_error_mae_delta"] for row in rows]),
        "per_sample": rows,
    }


def uniform_bd_summary(reference: list[dict], candidate: list[dict]) -> dict:
    def by_sample(records: list[dict]) -> dict[str, list[dict]]:
        output = defaultdict(list)
        for record in records:
            if record["family"] == "uniform-regression":
                output[record["sample_id"]].append(record)
        return output

    frozen_by_id = by_sample(reference)
    candidate_by_id = by_sample(candidate)
    if frozen_by_id.keys() != candidate_by_id.keys():
        raise RuntimeError("uniform sample sets differ")
    rows = []
    for sample_id in sorted(frozen_by_id):
        if len(frozen_by_id[sample_id]) != 3 or len(candidate_by_id[sample_id]) != 3:
            raise RuntimeError(f"incomplete three-QP curve: {sample_id}")
        rows.append({
            "sample_id": sample_id,
            "dataset": frozen_by_id[sample_id][0]["dataset"],
            "psnr_bd_rate_percent": bd_rate_percent(
                frozen_by_id[sample_id], candidate_by_id[sample_id], "psnr_db"),
            "lpips_bd_rate_percent": bd_rate_percent(
                frozen_by_id[sample_id], candidate_by_id[sample_id],
                "lpips_alex", lower_quality_is_better=True),
        })

    def summarize(selected: list[dict]) -> dict:
        result = {"sample_count": len(selected)}
        for key in ("psnr_bd_rate_percent", "lpips_bd_rate_percent"):
            values = [row[key] for row in selected if row[key] is not None]
            result[f"mean_{key}"] = finite_mean(values)
            result[f"median_{key}"] = float(np.median(values)) if values else None
            result[f"improved_count_{key}"] = sum(value < 0 for value in values)
            result[f"valid_count_{key}"] = len(values)
        return result

    return {
        "combined": summarize(rows),
        "REDS": summarize([row for row in rows if row["dataset"] == "REDS"]),
        "UVG": summarize([row for row in rows if row["dataset"] == "UVG"]),
        "per_sample": rows,
    }


def decoded_dir_from_record(record: dict) -> Path:
    return Path(record["stream"]).parent.parent / "fresh_decode"


def write_interpolation_visual(
    path: Path,
    source: list[np.ndarray],
    candidate_records: dict[str, dict],
    actions: list[int],
) -> None:
    frame_index = 8
    ordered = ("alpha000", "alpha025", "alpha050", "alpha075", "alpha100")
    labels = {
        "alpha000": "alpha 0.00 | frozen",
        "alpha025": "alpha 0.25",
        "alpha050": "alpha 0.50",
        "alpha075": "alpha 0.75",
        "alpha100": "alpha 1.00 | 1000-step",
    }
    panels = [panel(source[frame_index], "GT | frame 9", None)]
    for tag in ordered:
        record = candidate_records[tag]
        decoded = load_pngs(decoded_dir_from_record(record))
        title = f"{labels[tag]} | {record['stream_bytes']} B"
        panels.append(panel(decoded[frame_index], title, record["quality"]))
    action_array = np.asarray(actions, dtype=np.int64).reshape(4, 4)
    action_image = np.repeat(
        np.repeat(ACTION_COLORS[action_array], 128, axis=0), 128, axis=1)
    panels.append(panel(action_image, "Same B/G/E action map", None))
    width = max(item.width for item in panels)
    height = max(item.height for item in panels)
    canvas = Image.new("RGB", (4 * width, 2 * height), (230, 230, 230))
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % 4) * width, (index // 4) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def summarize_main(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    endpoint = read_json(args.endpoint_summary)
    endpoint_records = endpoint["records"]
    frozen = [
        {**record, "alpha": 0.0, "model_role": "alpha000"}
        for record in endpoint_records if record["model_role"] == "frozen"
    ]
    tuned = [
        {**record, "alpha": 1.0, "model_role": "alpha100"}
        for record in endpoint_records if record["model_role"] == "tuned"
    ]
    if len(frozen) != 55 or len(tuned) != 55:
        raise RuntimeError("endpoint records are incomplete")

    metric = LPIPSAlex(True)
    source_cache: dict[str, list[np.ndarray]] = {}
    records = []
    for task in protocol["tasks"]:
        output = args.output_dir / task["output_rel"]
        encode = read_json(output / "encode_summary.json")
        decode = read_json(output / "decode_summary.json")
        regression = read_json(output / "fresh_decode_regression.json")
        stream = Path(encode["stream"])
        if stream.stat().st_size != encode["stream_bytes"]:
            raise RuntimeError(f"stream byte mismatch: {task['task_id']}")
        if not regression["pixel_exact"]:
            raise RuntimeError(f"fresh decode mismatch: {task['task_id']}")
        if encode["model_checkpoints"]["role"] != "spatial-qp-interpolated":
            raise RuntimeError(f"checkpoint role mismatch: {task['task_id']}")
        if (Path(encode["model_checkpoints"]["image"]).resolve()
                != Path(task["image"]).resolve()
                or Path(encode["model_checkpoints"]["video"]).resolve()
                != Path(task["video"]).resolve()):
            raise RuntimeError(f"checkpoint path mismatch: {task['task_id']}")
        if decode["model_checkpoints"]["role"] != "spatial-qp-interpolated":
            raise RuntimeError(f"decoder checkpoint role mismatch: {task['task_id']}")
        sample_id = task["sample_id"]
        if sample_id not in source_cache:
            source_cache[sample_id] = load_source(read_json(Path(task["gate"])), 17)
        decoded = load_pngs(output / "fresh_decode")
        quality = evaluate_variant(source_cache[sample_id], decoded, metric)
        route = read_json(Path(task["route"]))
        variant = route["variants"][route["selected_variant"]]
        actions = variant["actions"]
        boundary = boundary_metrics(
            source_cache[sample_id], decoded,
            np.asarray(actions, dtype=np.int64).reshape(4, 4),
            int(route["configuration"]["tile_size"]),
        )
        record = {
            "task_id": task["task_id"],
            "sample_id": sample_id,
            "dataset": task["dataset"],
            "data_role": task["data_role"],
            "family": task["family"],
            "qp": task["qp"],
            "alpha": task["alpha"],
            "model_role": task["model_role"],
            "stream": str(stream.resolve()),
            "stream_bytes": encode["stream_bytes"],
            "quality": quality,
            "boundary": boundary,
            "fresh_decode_pixel_exact": True,
            "peak_cuda_allocated_bytes": max(
                encode["peak_cuda_allocated_bytes"],
                decode["peak_cuda_allocated_bytes"]),
            "actions": actions,
        }
        atomic_json(output / "quality_summary.json", record)
        records.append(record)

    all_by_tag = {"alpha000": frozen, "alpha100": tuned}
    for tag in protocol["candidate_tags"]:
        all_by_tag[tag] = [record for record in records if record["model_role"] == tag]
        if len(all_by_tag[tag]) != 55:
            raise RuntimeError(f"interpolation records are incomplete: {tag}")

    candidate_metadata = {
        "alpha000": {"alpha": 0.0, "source": "frozen endpoint"},
        **{
            tag: {"alpha": protocol["checkpoints"][tag]["alpha"], "source": "interpolated"}
            for tag in protocol["candidate_tags"]
        },
        "alpha100": {"alpha": 1.0, "source": "1000-step endpoint"},
    }
    ordered_tags = sorted(candidate_metadata, key=lambda tag: candidate_metadata[tag]["alpha"])
    aggregates = {}
    mixed_deltas = {}
    bd_rates = {}
    frozen_mixed = [record for record in frozen if record["family"] == "mixed-v6-route"]
    for tag in ordered_tags:
        values = all_by_tag[tag]
        aggregates[tag] = {}
        for family in ("mixed-v6-route", "uniform-regression"):
            family_records = [record for record in values if record["family"] == family]
            aggregates[tag][family] = {
                dataset: aggregate([
                    record for record in family_records
                    if dataset == "combined" or record["dataset"] == dataset])
                for dataset in ("combined", "REDS", "UVG")
            }
        candidate_mixed = [
            record for record in values if record["family"] == "mixed-v6-route"]
        mixed_deltas[tag] = {
            dataset: metric_delta_summary(
                [record for record in frozen_mixed
                 if dataset == "combined" or record["dataset"] == dataset],
                [record for record in candidate_mixed
                 if dataset == "combined" or record["dataset"] == dataset],
            )
            for dataset in ("combined", "REDS", "UVG")
        }
        bd_rates[tag] = uniform_bd_summary(frozen, values)

    selectable = [
        tag for tag in ordered_tags
        if bd_rates[tag]["combined"]["mean_lpips_bd_rate_percent"] is not None
    ]
    selected_tag = min(selectable, key=lambda tag: (
        bd_rates[tag]["combined"]["mean_lpips_bd_rate_percent"],
        bd_rates[tag]["combined"]["mean_psnr_bd_rate_percent"],
    ))

    endpoint_by_key = {
        (record["sample_id"], record["family"], record["qp"], record["model_role"]): record
        for record in [*frozen, *tuned]
    }
    intermediate_by_key = {
        (record["sample_id"], record["family"], record["qp"], record["model_role"]): record
        for record in records
    }
    visual_paths = []
    for sample_id in VISUAL_SAMPLE_IDS:
        visual_records = {
            "alpha000": endpoint_by_key[(
                sample_id, "mixed-v6-route", None, "alpha000")],
            "alpha100": endpoint_by_key[(
                sample_id, "mixed-v6-route", None, "alpha100")],
        }
        for tag in protocol["candidate_tags"]:
            visual_records[tag] = intermediate_by_key[(
                sample_id, "mixed-v6-route", None, tag)]
        target = args.output_dir / "visuals" / f"{sample_id}.png"
        write_interpolation_visual(
            target, source_cache[sample_id], visual_records,
            visual_records["alpha000"]["actions"])
        visual_paths.append(str(target.resolve()))

    summary = {
        "experiment": protocol["experiment"],
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit_at_summary": git_commit(),
        "implementation_sha256": sha256_file(Path(__file__)),
        "protocol": str(args.protocol.resolve()),
        "endpoint_summary": str(args.endpoint_summary.resolve()),
        "intermediate_task_count": len(records),
        "endpoint_task_count_reused": len(endpoint_records),
        "all_candidate_tags": ordered_tags,
        "candidate_metadata": candidate_metadata,
        "selection_rule": protocol["selection_rule"],
        "selected_tag": selected_tag,
        "selected_alpha": candidate_metadata[selected_tag]["alpha"],
        "uniform_bd_rate_vs_frozen": bd_rates,
        "mixed_route_delta_vs_frozen": mixed_deltas,
        "aggregates": aggregates,
        "intermediate_records": records,
        "visuals": visual_paths,
        "scientific_boundary": protocol["scientific_boundary"],
    }
    atomic_json(args.output_dir / "summary.json", summary)

    lines = [
        "# Spatial-QP checkpoint interpolation",
        "",
        "Selection uses the six fixed uniform-QP curves. Negative BD-rate is better.",
        "",
        "| Alpha | LPIPS BD-rate all | REDS | UVG | PSNR BD-rate all | Mixed bytes delta | Mixed LPIPS delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in ordered_tags:
        alpha = candidate_metadata[tag]["alpha"]
        bd = bd_rates[tag]
        mixed = mixed_deltas[tag]["combined"]
        lines.append(
            f"| {alpha:.2f} | {bd['combined']['mean_lpips_bd_rate_percent']:+.2f}% "
            f"| {bd['REDS']['mean_lpips_bd_rate_percent']:+.2f}% "
            f"| {bd['UVG']['mean_lpips_bd_rate_percent']:+.2f}% "
            f"| {bd['combined']['mean_psnr_bd_rate_percent']:+.2f}% "
            f"| {mixed['mean_stream_byte_delta']:+.1f} B "
            f"| {mixed['mean_lpips_delta']:+.6f} |")
    lines.extend(("", f"Selected alpha: **{candidate_metadata[selected_tag]['alpha']:.2f}**"))
    atomic_text(args.output_dir / "summary.md", "\n".join(lines) + "\n")

    with (args.output_dir / "intermediate_records.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample_id", "dataset", "family", "qp", "alpha", "stream_bytes",
            *QUALITY_KEYS, *BOUNDARY_KEYS,
        ))
        for record in records:
            writer.writerow((
                record["sample_id"], record["dataset"], record["family"],
                record["qp"], record["alpha"], record["stream_bytes"],
                *(record["quality"][key] for key in QUALITY_KEYS),
                *(record["boundary"][key] for key in BOUNDARY_KEYS),
            ))
    print(json.dumps({
        "summary": str((args.output_dir / "summary.json").resolve()),
        "selected_tag": selected_tag,
        "selected_alpha": candidate_metadata[selected_tag]["alpha"],
        "selected_uniform_bd_rate": bd_rates[selected_tag],
        "selected_mixed_delta": mixed_deltas[selected_tag],
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    assert alpha_tag(0.25) == "alpha025"
    assert parse_alphas("0.25,0.5,0.75") == DEFAULT_ALPHAS
    frozen = {
        "weight": torch.tensor([0.0, 2.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    tuned = {
        "weight": torch.tensor([4.0, 6.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    middle = blend_state_dicts(frozen, tuned, 0.25)
    assert torch.equal(middle["weight"], torch.tensor([1.0, 3.0]))
    assert middle["counter"].item() == 3
    print(json.dumps({
        "status": "passed",
        "alphas": list(DEFAULT_ALPHAS),
        "formula": "(1-alpha)*frozen + alpha*1000-step",
        "selection_primary": "combined mean LPIPS BD-rate",
    }, indent=2))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_main(args)
    elif args.command == "list-tasks":
        list_tasks_main(args)
    elif args.command == "summarize":
        summarize_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main(sys.argv[1:])
