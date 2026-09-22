#!/usr/bin/env python3
"""Fixed real-stream comparison for the domain-balanced codec candidates.

The expensive frozen and REDS-only endpoint streams are reused only after
their bytes, fresh-decode records, protocol, and checkpoint identities are
checked.  The two new balanced candidates are encoded and decoded under the
same six clips and uniform QP 8/16/32 routes.  This keeps the comparison small
while still exposing the REDS/UVG trade-off that motivated balanced training.
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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_feather_diagnostic import boundary_metrics
from demo.stage_c_evaluate_seedvr2_gate import load_pngs, load_source
from demo.stage_c_evaluate_spatial_quality_codec import panel
from demo.stage_c_spatial_qp_finetune_eval import (
    BOUNDARY_KEYS,
    QUALITY_KEYS,
    UNIFORM_SAMPLE_IDS,
    atomic_json,
    atomic_text,
    bd_rate_percent,
    finite_mean,
    read_json,
    resource_snapshot_main,
    sha256_file,
    utc_now,
)
from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant


FRAME_COUNT = 17
WIDTH = 512
HEIGHT = 512
QP_VALUES = (8, 16, 32)
MODEL_ROLES = ("frozen", "reds_only_v1", "balanced_25", "balanced_50")
NEW_MODEL_ROLES = ("balanced_25", "balanced_50")
VISUAL_SAMPLE_IDS = (
    "dev-s000-f00-x384-y096",
    "uvg-readysetgo-f00-center512",
    "uvg-yachtride-f00-center512",
)
LABELS = {
    "frozen": "DCVC-UF frozen",
    "reds_only_v1": "REDS-only tune",
    "balanced_25": "Balanced tune (25% UVG)",
    "balanced_50": "Balanced tune (50% UVG)",
}
COLORS = {
    "frozen": "#111827",
    "reds_only_v1": "#d1495b",
    "balanced_25": "#2f6f9f",
    "balanced_50": "#2a9d8f",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def actual_path(recorded: str | Path) -> Path:
    path = Path(recorded)
    if path.exists():
        return path.resolve()
    prefix = "/autodl-fs/data/"
    value = str(recorded)
    if value.startswith(prefix):
        translated = Path("/root/autodl-fs") / value[len(prefix):]
        if translated.exists():
            return translated.resolve()
    raise FileNotFoundError(recorded)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--endpoint-protocol", type=Path, required=True)
    prepare.add_argument("--endpoint-summary", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--balanced-25-image", type=Path, required=True)
    prepare.add_argument("--balanced-25-video", type=Path, required=True)
    prepare.add_argument("--balanced-50-image", type=Path, required=True)
    prepare.add_argument("--balanced-50-video", type=Path, required=True)

    listing = commands.add_parser("list-tasks")
    listing.add_argument("--protocol", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--protocol", type=Path, required=True)
    summarize.add_argument("--endpoint-summary", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    resource = commands.add_parser("resource-snapshot")
    resource.add_argument("--output-dir", type=Path, required=True)
    resource.add_argument("--output", type=Path)

    commands.add_parser("self-test")
    return parser.parse_args(argv)


def checkpoint_record(image: Path, video: Path, checkpoint_role: str) -> dict:
    for path in (image, video):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return {
        "image": str(image.resolve()),
        "video": str(video.resolve()),
        "checkpoint_role": checkpoint_role,
        "image_bytes": image.stat().st_size,
        "video_bytes": video.stat().st_size,
        "image_sha256": sha256_file(image),
        "video_sha256": sha256_file(video),
    }


def prepare_main(args: argparse.Namespace) -> None:
    endpoint_protocol_path = args.endpoint_protocol.resolve()
    endpoint_summary_path = args.endpoint_summary.resolve()
    endpoint_protocol = read_json(endpoint_protocol_path)
    endpoint_summary = read_json(endpoint_summary_path)
    if endpoint_summary.get("status") != "complete":
        raise RuntimeError("endpoint evaluation is not complete")
    if endpoint_protocol.get("uniform_sample_ids") != list(UNIFORM_SAMPLE_IDS):
        raise RuntimeError("endpoint uniform sample set changed")

    endpoint_tasks = [
        task for task in endpoint_protocol["tasks"]
        if task["family"] == "uniform-regression"
        and task["model_role"] == "frozen"
    ]
    if len(endpoint_tasks) != len(UNIFORM_SAMPLE_IDS) * len(QP_VALUES):
        raise RuntimeError("expected 18 frozen endpoint tasks")
    endpoint_keys = {
        (task["sample_id"], int(task["qp"])) for task in endpoint_tasks
    }
    expected_keys = {
        (sample_id, qp) for sample_id in UNIFORM_SAMPLE_IDS for qp in QP_VALUES
    }
    if endpoint_keys != expected_keys:
        raise RuntimeError("endpoint uniform task keys changed")

    checkpoints = {
        "frozen": {
            **endpoint_protocol["checkpoints"]["frozen"],
            "source": "reused endpoint evaluation",
        },
        "reds_only_v1": {
            **endpoint_protocol["checkpoints"]["tuned"],
            "source": "reused endpoint evaluation",
        },
        "balanced_25": checkpoint_record(
            args.balanced_25_image, args.balanced_25_video,
            "spatial-qp-balanced-v2-uvg025",
        ),
        "balanced_50": checkpoint_record(
            args.balanced_50_image, args.balanced_50_video,
            "spatial-qp-balanced-v2-uvg050",
        ),
    }
    tasks = []
    for source in sorted(
        endpoint_tasks, key=lambda item: (item["sample_id"], int(item["qp"])),
    ):
        for role in NEW_MODEL_ROLES:
            checkpoint = checkpoints[role]
            gate = actual_path(source["gate"])
            route = actual_path(source["route"])
            tasks.append({
                "task_id": f"{source['sample_id']}--uniform-qp{source['qp']}--{role}",
                "sample_id": source["sample_id"],
                "dataset": source["dataset"],
                "data_role": source["data_role"],
                "family": "uniform-regression",
                "qp": int(source["qp"]),
                "model_role": role,
                "gate": str(gate),
                "route": str(route),
                "output_rel": (
                    f"evaluation/{source['sample_id']}/uniform-qp{source['qp']}/{role}"
                ),
                **checkpoint,
            })

    protocol = {
        "schema_version": 1,
        "experiment": "domain-balanced spatial-QP codec fixed uniform-RD comparison",
        "status": "fixed-before-new-encoding",
        "created_utc": utc_now(),
        "git_commit_at_prepare": git_commit(),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "endpoint_protocol": str(endpoint_protocol_path),
        "endpoint_protocol_sha256": sha256_file(endpoint_protocol_path),
        "endpoint_summary": str(endpoint_summary_path),
        "endpoint_summary_sha256": sha256_file(endpoint_summary_path),
        "uniform_sample_ids": list(UNIFORM_SAMPLE_IDS),
        "qp_values": list(QP_VALUES),
        "datasets": {"REDS": 3, "UVG": 3},
        "checkpoints": checkpoints,
        "reused_task_count": 36,
        "new_task_count": len(tasks),
        "logical_task_count": 36 + len(tasks),
        "tasks": tasks,
        "selection_rule": {
            "hard_gate": False,
            "score": (
                "mean of PSNR and LPIPS mean BD-rate across the equally sized "
                "3-REDS + 3-UVG pool; lower is preferred"
            ),
            "reason": (
                "one transparent domain-balanced development ranking; final choice "
                "also inspects per-domain curves and individual clips"
            ),
        },
        "scientific_boundary": {
            "development_comparison_not_final_paper_benchmark": True,
            "same_six_clips_and_uniform_routes_for_every_model": True,
            "old_streams_reused_only_after_integrity_checks": True,
            "actual_stream_bytes_and_fresh_decode_required": True,
            "seedvr2_excluded_to_isolate_codec": True,
            "no_hard_acceptance_threshold": True,
            "single_gpu": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "protocol.json"
    if target.exists():
        existing = read_json(target)
        stable_keys = (
            "schema_version", "git_commit_at_prepare", "evaluator_sha256",
            "endpoint_protocol_sha256", "endpoint_summary_sha256", "checkpoints",
            "uniform_sample_ids", "qp_values", "tasks",
        )
        if any(existing.get(key) != protocol.get(key) for key in stable_keys):
            raise RuntimeError("refusing to change an existing evaluation protocol")
        print(json.dumps({"protocol": str(target), "reused": True}, indent=2))
        return
    atomic_json(target, protocol)
    print(json.dumps({
        "protocol": str(target.resolve()),
        "new_tasks": len(tasks),
        "reused_tasks": protocol["reused_task_count"],
    }, indent=2))


def list_tasks_main(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    fields = (
        "task_id", "sample_id", "family", "model_role", "gate", "route",
        "image", "video", "checkpoint_role", "output_rel",
    )
    for task in protocol["tasks"]:
        print("\t".join(str(task[field]) for field in fields))


def validate_reused_records(protocol: dict, endpoint_summary_path: Path) -> list[dict]:
    endpoint_summary_path = endpoint_summary_path.resolve()
    if sha256_file(endpoint_summary_path) != protocol["endpoint_summary_sha256"]:
        raise RuntimeError("endpoint summary changed after protocol freeze")
    endpoint = read_json(endpoint_summary_path)
    role_map = {"frozen": "frozen", "tuned": "reds_only_v1"}
    records = []
    for source in endpoint["records"]:
        if source["family"] != "uniform-regression":
            continue
        role = role_map.get(source["model_role"])
        if role is None:
            continue
        stream = actual_path(source["stream"])
        if stream.stat().st_size != int(source["stream_bytes"]):
            raise RuntimeError(f"reused stream byte mismatch: {source['task_id']}")
        if not source["fresh_decode_pixel_exact"]:
            raise RuntimeError(f"reused fresh decode failed: {source['task_id']}")
        output = stream.parent.parent
        regression = read_json(output / "fresh_decode_regression.json")
        decoded_dir = output / "fresh_decode"
        if not regression["pixel_exact"] or len(list(decoded_dir.glob("*.png"))) != 17:
            raise RuntimeError(f"reused decoded output incomplete: {source['task_id']}")
        records.append({
            **source,
            "task_id": source["task_id"].replace(
                f"--{source['model_role']}", f"--{role}"),
            "model_role": role,
            "stream": str(stream),
            "decoded_dir": str(decoded_dir.resolve()),
            "record_origin": "verified-reuse",
        })
    expected = 2 * len(UNIFORM_SAMPLE_IDS) * len(QP_VALUES)
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} reused records, found {len(records)}")
    return records


def collect_new_records(protocol: dict, output_dir: Path) -> tuple[list[dict], dict]:
    metric = LPIPSAlex(True)
    source_cache: dict[str, list[np.ndarray]] = {}
    decoded_cache: dict[tuple[str, int, str], list[np.ndarray]] = {}
    records = []
    for task in protocol["tasks"]:
        output = output_dir / task["output_rel"]
        encode = read_json(output / "encode_summary.json")
        decode = read_json(output / "decode_summary.json")
        regression = read_json(output / "fresh_decode_regression.json")
        stream = actual_path(encode["stream"])
        if stream.stat().st_size != int(encode["stream_bytes"]):
            raise RuntimeError(f"stream byte mismatch: {task['task_id']}")
        if not regression["pixel_exact"]:
            raise RuntimeError(f"fresh decode mismatch: {task['task_id']}")
        if encode["model_checkpoints"]["role"] != task["checkpoint_role"]:
            raise RuntimeError(f"checkpoint role mismatch: {task['task_id']}")
        if task["sample_id"] not in source_cache:
            source_cache[task["sample_id"]] = load_source(
                read_json(Path(task["gate"])), FRAME_COUNT)
        decoded_dir = output / "fresh_decode"
        decoded = load_pngs(decoded_dir)
        if len(decoded) != FRAME_COUNT:
            raise RuntimeError(f"decoded frame count mismatch: {task['task_id']}")
        quality = evaluate_variant(source_cache[task["sample_id"]], decoded, metric)
        route = read_json(Path(task["route"]))
        variant = route["variants"][route["selected_variant"]]
        actions = variant["actions"]
        boundary = boundary_metrics(
            source_cache[task["sample_id"]], decoded,
            np.asarray(actions, dtype=np.int64).reshape(4, 4),
            int(route["configuration"]["tile_size"]),
        )
        record = {
            "task_id": task["task_id"],
            "sample_id": task["sample_id"],
            "dataset": task["dataset"],
            "data_role": task["data_role"],
            "family": task["family"],
            "qp": task["qp"],
            "model_role": task["model_role"],
            "stream": str(stream),
            "stream_bytes": int(encode["stream_bytes"]),
            "quality": quality,
            "boundary": boundary,
            "fresh_decode_pixel_exact": True,
            "peak_cuda_allocated_bytes": max(
                int(encode["peak_cuda_allocated_bytes"]),
                int(decode["peak_cuda_allocated_bytes"]),
            ),
            "actions": actions,
            "decoded_dir": str(decoded_dir.resolve()),
            "record_origin": "new-encode-decode",
        }
        atomic_json(output / "quality_summary.json", record)
        records.append(record)
        decoded_cache[(task["sample_id"], int(task["qp"]), task["model_role"])] = decoded
    return records, {"sources": source_cache, "decoded": decoded_cache}


def aggregate_points(records: list[dict]) -> dict:
    groups = {
        "Combined (3 REDS + 3 UVG)": {"REDS", "UVG"},
        "REDS (3 clips)": {"REDS"},
        "UVG (3 clips)": {"UVG"},
    }
    output = {}
    for group_name, datasets in groups.items():
        output[group_name] = {}
        for role in MODEL_ROLES:
            points = []
            for qp in QP_VALUES:
                selected = [
                    record for record in records
                    if record["model_role"] == role
                    and record["dataset"] in datasets
                    and int(record["qp"]) == qp
                ]
                expected = 6 if len(datasets) == 2 else 3
                if len(selected) != expected:
                    raise RuntimeError(
                        f"{group_name}/{role}/QP{qp}: expected {expected}, "
                        f"found {len(selected)}")
                mean_bytes = finite_mean([item["stream_bytes"] for item in selected])
                points.append({
                    "qp": qp,
                    "sample_count": len(selected),
                    "mean_stream_bytes": mean_bytes,
                    "aggregate_bpp": mean_bytes * 8 / (FRAME_COUNT * WIDTH * HEIGHT),
                    "mean_lpips_alex": finite_mean([
                        item["quality"]["lpips_alex"] for item in selected]),
                    "mean_psnr_db": finite_mean([
                        item["quality"]["psnr_db"] for item in selected]),
                })
            output[group_name][role] = points
    return output


def summarize_values(rows: list[dict]) -> dict:
    output = {"sample_count": len(rows)}
    for key in ("psnr_bd_rate_percent", "lpips_bd_rate_percent"):
        values = [row[key] for row in rows if row[key] is not None]
        output[f"mean_{key}"] = finite_mean(values)
        output[f"median_{key}"] = float(np.median(values)) if values else None
        output[f"improved_count_{key}"] = sum(value < 0 for value in values)
        output[f"valid_count_{key}"] = len(values)
    return output


def bd_rate_summaries(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["sample_id"], record["model_role"])].append(record)
    output = {}
    for role in MODEL_ROLES:
        rows = []
        for sample_id in UNIFORM_SAMPLE_IDS:
            reference = grouped[(sample_id, "frozen")]
            candidate = grouped[(sample_id, role)]
            if len(reference) != 3 or len(candidate) != 3:
                raise RuntimeError(f"incomplete curve for {sample_id}/{role}")
            rows.append({
                "sample_id": sample_id,
                "dataset": reference[0]["dataset"],
                "psnr_bd_rate_percent": (
                    0.0 if role == "frozen" else
                    bd_rate_percent(reference, candidate, "psnr_db")
                ),
                "lpips_bd_rate_percent": (
                    0.0 if role == "frozen" else
                    bd_rate_percent(
                        reference, candidate, "lpips_alex",
                        lower_quality_is_better=True,
                    )
                ),
            })
        output[role] = {
            "definition": (
                f"{LABELS[role]} versus frozen DCVC-UF bitrate change at equal "
                "quality; negative is better"
            ),
            "combined": summarize_values(rows),
            "REDS": summarize_values([
                row for row in rows if row["dataset"] == "REDS"]),
            "UVG": summarize_values([
                row for row in rows if row["dataset"] == "UVG"]),
            "per_sample": rows,
        }
    return output


def selection_summary(bd_rates: dict) -> dict:
    rows = []
    for role in MODEL_ROLES:
        combined = bd_rates[role]["combined"]
        components = [
            combined["mean_psnr_bd_rate_percent"],
            combined["mean_lpips_bd_rate_percent"],
        ]
        if any(value is None or not math.isfinite(value) for value in components):
            score = None
        else:
            score = float(np.mean(components))
        rows.append({
            "model_role": role,
            "label": LABELS[role],
            "mean_psnr_bd_rate_percent": components[0],
            "mean_lpips_bd_rate_percent": components[1],
            "balanced_mean_bd_rate_score": score,
        })
    valid = [row for row in rows if row["balanced_mean_bd_rate_score"] is not None]
    if not valid:
        raise RuntimeError("no candidate has a valid selection score")
    selected = min(valid, key=lambda row: row["balanced_mean_bd_rate_score"])
    return {
        "hard_gate": False,
        "lower_is_better": True,
        "rule": (
            "arithmetic mean of combined-pool PSNR BD-rate and LPIPS BD-rate; "
            "the pool contains equal numbers of REDS and UVG clips"
        ),
        "rows": rows,
        "preliminary_selected_role": selected["model_role"],
        "preliminary_selected_label": selected["label"],
        "interpretation_required": (
            "inspect REDS and UVG curves before using this development ranking"
        ),
    }


def plot_rd(points: dict, target: Path) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True)
    for column, (group_name, curves) in enumerate(points.items()):
        for row, metric in enumerate(("mean_lpips_alex", "mean_psnr_db")):
            axis = axes[row, column]
            for role in MODEL_ROLES:
                curve = curves[role]
                axis.plot(
                    [point["aggregate_bpp"] for point in curve],
                    [point[metric] for point in curve],
                    color=COLORS[role], marker="o",
                    linewidth=2.7 if role == "frozen" else 2.0,
                    markersize=5.5 if role == "frozen" else 4.8,
                    linestyle="--" if role == "reds_only_v1" else "-",
                    label=LABELS[role], zorder=5 if role == "frozen" else 3,
                )
                for point in curve:
                    axis.annotate(
                        str(point["qp"]),
                        (point["aggregate_bpp"], point[metric]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6.8, color=COLORS[role], alpha=0.8,
                    )
            axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            axis.set_xlabel("Actual bitrate (bits / pixel)")
            if row == 0:
                axis.set_title(group_name, weight="bold")
                axis.text(0.03, 0.04, "better  ↙", transform=axis.transAxes)
            else:
                axis.text(0.03, 0.91, "better  ↖", transform=axis.transAxes)
    axes[0, 0].set_ylabel("LPIPS-Alex (lower is better)")
    axes[1, 0].set_ylabel("PSNR (dB, higher is better)")
    axes[0, 0].legend(loc="upper right", framealpha=0.95)
    figure.suptitle(
        "Fixed codec-only RD curves: frozen, REDS-only, and domain-balanced tunes",
        fontsize=14, weight="bold",
    )
    figure.text(
        0.5, -0.015,
        "Development diagnostic · uniform QP 8/16/32 · actual stream bytes · "
        "every point fresh-decoded",
        ha="center", fontsize=9, color="#4b5563",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_visuals(records: list[dict], protocol: dict, output_dir: Path) -> list[str]:
    by_key = {
        (record["sample_id"], int(record["qp"]), record["model_role"]): record
        for record in records
    }
    gate_by_sample = {}
    for task in protocol["tasks"]:
        gate_by_sample.setdefault(task["sample_id"], Path(task["gate"]))
    outputs = []
    for sample_id in VISUAL_SAMPLE_IDS:
        source = load_source(read_json(gate_by_sample[sample_id]), FRAME_COUNT)
        panels = [panel(source[8], "GT | frame 9", None)]
        for role in MODEL_ROLES:
            record = by_key[(sample_id, 16, role)]
            decoded = load_pngs(Path(record["decoded_dir"]))
            panels.append(panel(decoded[8], LABELS[role] + " | QP16", record["quality"]))
        width = max(item.width for item in panels)
        height = max(item.height for item in panels)
        canvas = Image.new("RGB", (3 * width, 2 * height), (232, 232, 232))
        for index, item in enumerate(panels):
            canvas.paste(item, ((index % 3) * width, (index // 3) * height))
        target = output_dir / "visuals" / f"{sample_id}-qp16.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, optimize=True)
        outputs.append(str(target.resolve()))
    return outputs


def summarize_main(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    if git_commit() != protocol["git_commit_at_prepare"]:
        raise RuntimeError("Git commit changed after evaluation protocol was frozen")
    reused = validate_reused_records(protocol, args.endpoint_summary)
    new, _ = collect_new_records(protocol, args.output_dir)
    records = reused + new
    expected = len(MODEL_ROLES) * len(UNIFORM_SAMPLE_IDS) * len(QP_VALUES)
    keys = {
        (record["model_role"], record["sample_id"], int(record["qp"]))
        for record in records
    }
    if len(records) != expected or len(keys) != expected:
        raise RuntimeError(f"expected {expected} unique logical records")

    points = aggregate_points(records)
    bd_rates = bd_rate_summaries(records)
    selection = selection_summary(bd_rates)
    plot_path = args.output_dir / "plots" / "balanced_spatial_qp_rd.png"
    plot_rd(points, plot_path)
    visuals = write_visuals(records, protocol, args.output_dir)

    summary = {
        "experiment": protocol["experiment"],
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit_at_summary": git_commit(),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "logical_record_count": len(records),
        "verified_reused_record_count": len(reused),
        "new_encode_decode_record_count": len(new),
        "primary_metrics": {
            "rate": "actual stream bits per pixel",
            "perceptual_distortion": "LPIPS Alex; lower is better",
            "fidelity": "PSNR dB; higher is better",
        },
        "rd_points": points,
        "bd_rate_vs_frozen": bd_rates,
        "selection": selection,
        "records": records,
        "plot": str(plot_path.resolve()),
        "plot_bytes": plot_path.stat().st_size,
        "plot_sha256": sha256_file(plot_path),
        "visuals": visuals,
        "scientific_boundary": protocol["scientific_boundary"],
    }
    atomic_json(args.output_dir / "summary.json", summary)

    with (args.output_dir / "records.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample_id", "dataset", "qp", "model_role", "record_origin",
            "stream_bytes", *QUALITY_KEYS, *BOUNDARY_KEYS,
        ))
        for record in records:
            writer.writerow((
                record["sample_id"], record["dataset"], record["qp"],
                record["model_role"], record["record_origin"],
                record["stream_bytes"],
                *(record["quality"][key] for key in QUALITY_KEYS),
                *(record["boundary"][key] for key in BOUNDARY_KEYS),
            ))

    rows = selection["rows"]
    markdown = [
        "# Domain-balanced spatial-QP codec comparison",
        "",
        "All rows use the same 3 REDS + 3 UVG clips and uniform QP 8/16/32. ",
        "Negative BD-rate is better. This is a development choice, not a hard gate.",
        "",
        "| Model | PSNR BD-rate | LPIPS BD-rate | Simple balanced score |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['label']} | {row['mean_psnr_bd_rate_percent']:+.2f}% | "
            f"{row['mean_lpips_bd_rate_percent']:+.2f}% | "
            f"{row['balanced_mean_bd_rate_score']:+.2f}% |")
    markdown.extend((
        "",
        f"Preliminary selection: **{selection['preliminary_selected_label']}**.",
        "",
        "The selection is revisited after looking at the separate REDS and UVG curves.",
    ))
    atomic_text(args.output_dir / "summary.md", "\n".join(markdown) + "\n")
    print(json.dumps({
        "summary": str((args.output_dir / "summary.json").resolve()),
        "plot": str(plot_path.resolve()),
        "selection": selection,
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    reference = [
        {"stream_bytes": rate, "quality": {"psnr_db": quality}}
        for rate, quality in ((100, 20), (200, 25), (400, 30))
    ]
    candidate = [
        {"stream_bytes": rate * 0.8, "quality": {"psnr_db": quality}}
        for rate, quality in ((100, 20), (200, 25), (400, 30))
    ]
    value = bd_rate_percent(reference, candidate, "psnr_db")
    assert math.isclose(value, -20.0, rel_tol=1e-6, abs_tol=1e-6)
    assert len(UNIFORM_SAMPLE_IDS) == 6
    assert len(MODEL_ROLES) == 4
    assert set(NEW_MODEL_ROLES).issubset(MODEL_ROLES)
    print(json.dumps({
        "status": "passed",
        "model_roles": list(MODEL_ROLES),
        "new_task_count": len(NEW_MODEL_ROLES) * 6 * 3,
        "logical_record_count": len(MODEL_ROLES) * 6 * 3,
        "synthetic_bd_rate_percent": value,
    }, indent=2))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare_main(args)
    elif args.command == "list-tasks":
        list_tasks_main(args)
    elif args.command == "summarize":
        summarize_main(args)
    elif args.command == "resource-snapshot":
        resource_snapshot_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main(sys.argv[1:])
