#!/usr/bin/env python3
"""Fixed-input SeedVR2 LoRA inference-strength sweep on one A800."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import PersistentSeedVR2, load_source
from demo.stage_c_evaluate_seedvr2_gate import load_pngs
from demo.stage_c_seedvr2_lora_eval import (
    BatchedLPIPSAlex,
    atomic_json,
    atomic_text,
    evaluate_quality,
    frame_digest,
    frame_paths,
    metric_subtitle,
    move_incomplete,
    sha256_file,
    visual_panel,
)
from demo.stage_c_seedvr2_lora_utils import set_lora_strength
from demo.stage_c_three_path_roi_probe import save_frames


FORMAT_VERSION = 1
STRENGTHS = (0.25, 0.50, 0.75)
VARIANT_BY_STRENGTH = {
    0.25: "lora-025",
    0.50: "lora-050",
    0.75: "lora-075",
}
QUALITY_KEYS = (
    "codec_qp8", "frozen", "lora_025", "lora_050", "lora_075",
    "lora_100",
)
SWEEP_KEYS = ("lora_025", "lora_050", "lora_075", "lora_100")
VISUAL_IDS = (
    "dev-s000-f00-x384-y096",
    "eval-s024-f00-x384-y096",
    "uvg-beauty-f00-center512",
    "uvg-honeybee-f00-center512",
    "uvg-jockey-f00-center512",
    "uvg-readysetgo-f00-center512",
    "uvg-shakendry-f00-center512",
    "uvg-yachtride-f00-center512",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SeedVR2 codec-artifact LoRA inference strength")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--base-plan", type=Path, required=True)
    plan.add_argument("--base-eval-root", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    restore = commands.add_parser("restore")
    restore.add_argument("--sweep-plan", type=Path, required=True)
    restore.add_argument("--output-root", type=Path, required=True)
    restore.add_argument("--upstream-root", type=Path, required=True)
    restore.add_argument("--dit-checkpoint", type=Path, required=True)
    restore.add_argument("--vae-checkpoint", type=Path, required=True)
    restore.add_argument("--positive-embedding", type=Path, required=True)
    restore.add_argument("--negative-embedding", type=Path, required=True)
    restore.add_argument("--lora-checkpoint", type=Path, required=True)
    restore.add_argument("--cuda-idx", type=int, default=0)
    restore.add_argument("--max-wall-seconds", type=int, default=3600)
    restore.add_argument("--disk-stop-percent", type=float, default=80.0)

    summary = commands.add_parser("summarize")
    summary.add_argument("--sweep-plan", type=Path, required=True)
    summary.add_argument("--output-root", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--lpips-batch-size", type=int, default=8)
    summary.add_argument("--cuda-idx", type=int, default=0)

    commands.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.command == "restore":
        if args.cuda_idx != 0:
            parser.error("the sweep is fixed to cuda:0")
        if args.max_wall_seconds < 60:
            parser.error("wall limit must be at least one minute")
        if not 0 < args.disk_stop_percent < 100:
            parser.error("disk stop percent must be inside (0,100)")
    if args.command == "summarize" and args.lpips_batch_size < 1:
        parser.error("LPIPS batch size must be positive")
    return args


def base_variant_paths(base_root: Path, sample_id: str) -> dict[str, Path]:
    return {
        variant: base_root / "outputs" / variant / sample_id
        for variant in ("frozen", "lora")
    }


def plan_main(args: argparse.Namespace) -> None:
    base_summary_path = args.base_eval_root / "formal" / "summary.json"
    required = (
        args.base_plan,
        args.lora_checkpoint,
        args.base_eval_root / "run.complete",
        base_summary_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    base_plan = json.loads(args.base_plan.read_text(encoding="utf-8"))
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    base_plan_sha = sha256_file(args.base_plan)
    if base_plan.get("sample_count") != 37:
        raise RuntimeError("the strength sweep requires the fixed 37-sample plan")
    if base_summary.get("sample_count") != 37:
        raise RuntimeError("the base comparison is incomplete")
    if len({x["sample_id"] for x in base_plan["entries"]}) != 37:
        raise RuntimeError("base sample IDs are not unique")
    if len({x["sample_id"] for x in base_summary["samples"]}) != 37:
        raise RuntimeError("base summary sample IDs are not unique")
    missing_visuals = set(VISUAL_IDS) - {
        x["sample_id"] for x in base_plan["entries"]}
    if missing_visuals:
        raise RuntimeError(f"fixed visual samples are missing: {missing_visuals}")
    if base_summary.get("plan_sha256") != base_plan_sha:
        raise RuntimeError("the base summary and plan differ")
    verification = base_summary.get("verification", {})
    if not verification.get("frozen_rerun_all_pixel_exact"):
        raise RuntimeError("the frozen baseline did not reproduce exactly")
    adapter_sha = sha256_file(args.lora_checkpoint)
    if adapter_sha != base_plan["checkpoints"]["lora_sha256"]:
        raise RuntimeError("the LoRA checkpoint differs from the base plan")

    entries = []
    for entry in base_plan["entries"]:
        item = dict(entry)
        item["base_outputs"] = {}
        for variant, sample_root in base_variant_paths(
                args.base_eval_root, entry["sample_id"]).items():
            metadata_path = sample_root / "metadata.json"
            frames_dir = sample_root / "frames"
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = frame_digest(frames_dir)
            if metadata.get("status") != "complete":
                raise RuntimeError(f"incomplete base output: {sample_root}")
            if metadata.get("output_png_sha256") != digest:
                raise RuntimeError(f"base output digest differs: {sample_root}")
            if variant == "frozen" and digest != entry["previous_frozen_png_sha256"]:
                raise RuntimeError(f"frozen output differs: {entry['sample_id']}")
            if variant == "lora" and metadata.get("lora_sha256") != adapter_sha:
                raise RuntimeError(f"LoRA output uses another adapter: {entry['sample_id']}")
            item["base_outputs"][variant] = {
                "frames_dir": str(frames_dir.resolve()),
                "png_sha256": digest,
                "metadata": str(metadata_path.resolve()),
            }
        entries.append(item)

    value = {
        "experiment": "SeedVR2 LoRA inference-strength sweep",
        "format_version": FORMAT_VERSION,
        "status": "frozen-before-strength-outputs",
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "base_plan": str(args.base_plan.resolve()),
        "base_plan_sha256": base_plan_sha,
        "base_summary": str(base_summary_path.resolve()),
        "base_summary_sha256": sha256_file(base_summary_path),
        "base_eval_root": str(args.base_eval_root.resolve()),
        "lora_checkpoint": str(args.lora_checkpoint.resolve()),
        "lora_sha256": adapter_sha,
        "checkpoints": base_plan["checkpoints"],
        "inference": base_plan["inference"],
        "sample_count": len(entries),
        "dataset_counts": dict(Counter(x["dataset"] for x in entries)),
        "strengths": [
            {"value": value, "variant": VARIANT_BY_STRENGTH[value]}
            for value in STRENGTHS
        ],
        "visual_sample_ids": list(VISUAL_IDS),
        "entries": entries,
        "analysis_protocol": {
            "same_inputs_seeds_and_adapter": True,
            "reuse_verified_strength_0_and_1_outputs": True,
            "new_strengths_fixed_before_inference": True,
            "report_REDS_UVG_and_combined": True,
            "balanced_lpips_diagnostic_weights_REDS_and_UVG_equally": True,
            "balanced_score_is_not_a_hard_gate": True,
            "visual_review_before_default_selection": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output, value)
    print(json.dumps({
        "stage": "strength-plan-complete",
        "output": str(args.output.resolve()),
        "sample_count": len(entries),
        "dataset_counts": value["dataset_counts"],
        "strengths": value["strengths"],
    }, ensure_ascii=False, indent=2))


def valid_output(
    sample_root: Path,
    *,
    entry: dict,
    strength: float,
    sweep_plan_sha: str,
    dit_sha: str,
    lora_sha: str,
) -> bool:
    metadata_path = sample_root / "metadata.json"
    frames_dir = sample_root / "frames"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_paths(frames_dir, int(entry["frame_count"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return False
    return (
        metadata.get("status") == "complete"
        and metadata.get("sample_id") == entry["sample_id"]
        and math.isclose(
            float(metadata.get("lora_strength", -1)), strength,
            rel_tol=0, abs_tol=1e-12)
        and metadata.get("seed") == int(entry["seed"])
        and metadata.get("sweep_plan_sha256") == sweep_plan_sha
        and metadata.get("input_png_sha256") == entry["fresh_decode_png_sha256"]
        and metadata.get("dit_sha256") == dit_sha
        and metadata.get("lora_sha256") == lora_sha
        and metadata.get("output_png_sha256")
        == frame_digest(frames_dir, int(entry["frame_count"]))
    )


def variant_manifest(
    *,
    plan: dict,
    plan_path: Path,
    output_root: Path,
    strength: float,
    model_load_seconds: float,
) -> dict:
    variant = VARIANT_BY_STRENGTH[strength]
    entries = []
    for entry in plan["entries"]:
        path = output_root / variant / entry["sample_id"] / "metadata.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") == "complete":
                entries.append(value)
    result = {
        "experiment": "SeedVR2 LoRA inference-strength sweep",
        "format_version": FORMAT_VERSION,
        "status": "complete" if len(entries) == len(plan["entries"]) else "in-progress",
        "updated_utc": utc_now(),
        "variant": variant,
        "lora_strength": strength,
        "sweep_plan": str(plan_path.resolve()),
        "sweep_plan_sha256": sha256_file(plan_path),
        "requested_sample_count": len(plan["entries"]),
        "completed_sample_count": len(entries),
        "model_load_seconds_this_process": model_load_seconds,
        "total_model_seconds": sum(
            float(x["runtime"]["seconds_model_load_excluded"]) for x in entries),
        "peak_cuda_allocated_bytes": max(
            (int(x["runtime"]["peak_cuda_allocated_bytes"]) for x in entries),
            default=0,
        ),
        "entries": entries,
    }
    atomic_json(output_root / variant / "manifest.json", result)
    return result


def restore_main(args: argparse.Namespace) -> None:
    for path in (
        args.sweep_plan, args.dit_checkpoint, args.vae_checkpoint,
        args.positive_embedding, args.negative_embedding, args.lora_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = json.loads(args.sweep_plan.read_text(encoding="utf-8"))
    if (plan.get("format_version") != FORMAT_VERSION
            or plan.get("status") != "frozen-before-strength-outputs"):
        raise RuntimeError("strength sweep plan is incompatible")
    expected_strengths = [x["value"] for x in plan["strengths"]]
    if expected_strengths != list(STRENGTHS):
        raise RuntimeError("strength list differs from the fixed protocol")
    if torch.cuda.device_count() != 1 or args.cuda_idx != 0:
        raise RuntimeError("the strength sweep requires one GPU at cuda:0")
    sweep_plan_sha = sha256_file(args.sweep_plan)
    dit_sha = sha256_file(args.dit_checkpoint)
    lora_sha = sha256_file(args.lora_checkpoint)
    if dit_sha != plan["checkpoints"]["dit_sha256"]:
        raise RuntimeError("DiT checkpoint differs from the sweep plan")
    if lora_sha != plan["lora_sha256"]:
        raise RuntimeError("LoRA checkpoint differs from the sweep plan")
    for path, key in (
        (args.vae_checkpoint, "vae_sha256"),
        (args.positive_embedding, "positive_embedding_sha256"),
        (args.negative_embedding, "negative_embedding_sha256"),
    ):
        if sha256_file(path) != plan["checkpoints"][key]:
            raise RuntimeError(f"checkpoint differs from the sweep plan: {path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = False
    for strength in STRENGTHS:
        variant_root = args.output_root / VARIANT_BY_STRENGTH[strength]
        variant_root.mkdir(parents=True, exist_ok=True)
        for entry in plan["entries"]:
            if not valid_output(
                variant_root / entry["sample_id"], entry=entry,
                strength=strength, sweep_plan_sha=sweep_plan_sha,
                dit_sha=dit_sha, lora_sha=lora_sha,
            ):
                pending = True
                break
    if not pending:
        for strength in STRENGTHS:
            result = variant_manifest(
                plan=plan, plan_path=args.sweep_plan,
                output_root=args.output_root, strength=strength,
                model_load_seconds=0.0)
            if result["status"] != "complete":
                raise RuntimeError(f"strength output is incomplete: {strength}")
            atomic_text(
                args.output_root / VARIANT_BY_STRENGTH[strength]
                / "variant.complete",
                "complete\n",
            )
        atomic_text(args.output_root / "restore.complete", "complete\n")
        print(json.dumps({
            "stage": "strength-restore-all-complete-resume",
            "strengths": list(STRENGTHS),
        }, indent=2))
        return

    torch.cuda.set_device(args.cuda_idx)
    bridge_args = SimpleNamespace(
        upstream_root=args.upstream_root,
        dit_checkpoint=args.dit_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        positive_embedding=args.positive_embedding,
        negative_embedding=args.negative_embedding,
        sample_steps=1,
        cfg_scale=1.0,
        dit_dtype="bfloat16",
    )
    process_started = time.perf_counter()
    model = PersistentSeedVR2(bridge_args)
    for strength in STRENGTHS:
        variant = VARIANT_BY_STRENGTH[strength]
        variant_root = args.output_root / variant
        set_lora_strength(model.runner.dit, strength)
        model.runner.lora_adapter_info["inference_strength"] = strength
        for index, entry in enumerate(plan["entries"], start=1):
            sample_root = variant_root / entry["sample_id"]
            if valid_output(
                sample_root, entry=entry, strength=strength,
                sweep_plan_sha=sweep_plan_sha, dit_sha=dit_sha,
                lora_sha=lora_sha,
            ):
                print(json.dumps({
                    "stage": "strength-resume-skip",
                    "variant": variant,
                    "sample_id": entry["sample_id"],
                }), flush=True)
                continue
            if time.perf_counter() - process_started >= args.max_wall_seconds:
                raise RuntimeError("strength sweep reached its wall limit")
            usage = shutil.disk_usage(args.output_root)
            if 100.0 * usage.used / usage.total >= args.disk_stop_percent:
                raise RuntimeError("file-store use reached the stop threshold")
            input_dir = Path(entry["fresh_decode_dir"])
            if frame_digest(input_dir) != entry["fresh_decode_png_sha256"]:
                raise RuntimeError(f"{entry['sample_id']} input PNGs changed")
            frames = load_pngs(input_dir)
            sample_started = time.perf_counter()
            restored, runtime = model.restore(frames, int(entry["seed"]))
            temporary = variant_root / (
                f".{entry['sample_id']}.tmp-{os.getpid()}")
            temporary.mkdir(parents=True, exist_ok=False)
            save_frames(temporary / "frames", restored)
            metadata = {
                "status": "complete",
                "completed_utc": utc_now(),
                "sample_id": entry["sample_id"],
                "dataset": entry["dataset"],
                "sequence": entry["sequence"],
                "variant": variant,
                "lora_strength": strength,
                "seed": int(entry["seed"]),
                "sweep_plan": str(args.sweep_plan.resolve()),
                "sweep_plan_sha256": sweep_plan_sha,
                "input_dir": str(input_dir.resolve()),
                "input_png_sha256": entry["fresh_decode_png_sha256"],
                "output_png_sha256": frame_digest(temporary / "frames"),
                "frame_count": len(restored),
                "dit_sha256": dit_sha,
                "lora_sha256": lora_sha,
                "lora_adapter": dict(model.runner.lora_adapter_info),
                "runtime": runtime,
                "sample_wall_seconds": time.perf_counter() - sample_started,
                "model_load_seconds_shared": model.model_load_seconds,
                "same_input_seed_and_adapter_across_strengths": True,
                "training_or_finetuning": False,
            }
            atomic_json(temporary / "metadata.json", metadata)
            move_incomplete(sample_root)
            os.replace(temporary, sample_root)
            print(json.dumps({
                "stage": "strength-restore-sample",
                "variant": variant,
                "strength": strength,
                "index": index,
                "requested": len(plan["entries"]),
                "sample_id": entry["sample_id"],
                "dataset": entry["dataset"],
                "seconds": runtime["seconds_model_load_excluded"],
                "peak_cuda_allocated_bytes": runtime["peak_cuda_allocated_bytes"],
            }), flush=True)
        result = variant_manifest(
            plan=plan, plan_path=args.sweep_plan, output_root=args.output_root,
            strength=strength, model_load_seconds=model.model_load_seconds)
        if result["status"] != "complete":
            raise RuntimeError(f"strength output did not complete: {strength}")
        atomic_text(variant_root / "variant.complete", "complete\n")
        print(json.dumps({
            "stage": "strength-variant-complete",
            "variant": variant,
            "strength": strength,
            "completed": result["completed_sample_count"],
            "model_seconds": result["total_model_seconds"],
            "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
        }, indent=2), flush=True)
    atomic_text(args.output_root / "restore.complete", "complete\n")


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(np.mean(values))


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}
    metrics = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
    quality = {
        key: {
            metric: mean([row["quality"][key][metric] for row in rows])
            for metric in metrics
        }
        for key in QUALITY_KEYS
    }
    delta = {
        key: {
            metric: quality[key][metric] - quality["frozen"][metric]
            for metric in metrics
        }
        for key in SWEEP_KEYS
    }
    counts = {
        key: {
            "lower_lpips": sum(
                row["quality"][key]["lpips_alex"]
                < row["quality"]["frozen"]["lpips_alex"] for row in rows),
            "higher_psnr": sum(
                row["quality"][key]["psnr_db"]
                > row["quality"]["frozen"]["psnr_db"] for row in rows),
            "lower_temporal_error": sum(
                row["quality"][key]["temporal_delta_mae"]
                < row["quality"]["frozen"]["temporal_delta_mae"] for row in rows),
        }
        for key in SWEEP_KEYS
    }
    return {
        "sample_count": len(rows),
        "quality": quality,
        "delta_vs_frozen": delta,
        "paired_counts_vs_frozen": counts,
    }


def save_strength_visual(
    path: Path,
    source: list[np.ndarray],
    variants: dict[str, list[np.ndarray]],
    quality: dict,
    frame_index: int = 8,
) -> None:
    labels = (
        ("GT", "GT"),
        ("frozen", "Frozen"),
        ("lora_025", "LoRA 0.25"),
        ("lora_050", "LoRA 0.50"),
        ("lora_075", "LoRA 0.75"),
        ("lora_100", "LoRA 1.00"),
    )
    panels = []
    for key, label in labels:
        if key == "GT":
            panels.append(visual_panel(source[frame_index], label, "fixed frame 9"))
        else:
            panels.append(visual_panel(
                variants[key][frame_index], label, metric_subtitle(quality[key])))
    canvas = Image.new(
        "RGB", (sum(x.width for x in panels), max(x.height for x in panels)),
        (230, 230, 230))
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["sample_id", "dataset", "sequence", "uvg_training_role"]
    for key in QUALITY_KEYS:
        fields.extend((
            f"{key}_lpips", f"{key}_psnr", f"{key}_temporal"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {
                name: row[name]
                for name in ("sample_id", "dataset", "sequence", "uvg_training_role")
            }
            for key in QUALITY_KEYS:
                value[f"{key}_lpips"] = row["quality"][key]["lpips_alex"]
                value[f"{key}_psnr"] = row["quality"][key]["psnr_db"]
                value[f"{key}_temporal"] = row["quality"][key]["temporal_delta_mae"]
            writer.writerow(value)
    os.replace(temporary, path)


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# SeedVR2 LoRA inference-strength sweep",
        "",
        "同一批真实 QP8 输入、同一 seed、同一 adapter；0 和 1 复用已验证输出。LPIPS 越低越好。",
        "",
        "| Strength | Combined LPIPS | Delta | REDS delta | UVG delta | Balanced relative score | LPIPS wins | PSNR delta | Temporal delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    combined = result["aggregate"]["combined"]
    reds = result["aggregate"]["REDS"]
    uvg = result["aggregate"]["UVG"]
    scores = {x["key"]: x for x in result["metric_only_ranking"]}
    for key, strength in (
        ("lora_025", 0.25), ("lora_050", 0.50),
        ("lora_075", 0.75), ("lora_100", 1.00),
    ):
        lines.append(
            f"| {strength:.2f} | {combined['quality'][key]['lpips_alex']:.6f} | "
            f"{combined['delta_vs_frozen'][key]['lpips_alex']:+.6f} | "
            f"{reds['delta_vs_frozen'][key]['lpips_alex']:+.6f} | "
            f"{uvg['delta_vs_frozen'][key]['lpips_alex']:+.6f} | "
            f"{scores[key]['balanced_relative_lpips_percent']:+.3f}% | "
            f"{combined['paired_counts_vs_frozen'][key]['lower_lpips']}/37 | "
            f"{combined['delta_vs_frozen'][key]['psnr_db']:+.3f} | "
            f"{combined['delta_vs_frozen'][key]['temporal_delta_mae']:+.3f} |")
    lines.extend([
        "",
        "Balanced relative score gives REDS and UVG equal weight and is only a diagnostic ranking, not a hard gate. Inspect the fixed visuals before selecting a default.",
    ])
    atomic_text(path, "\n".join(lines) + "\n")


def summarize_main(args: argparse.Namespace) -> None:
    if not args.sweep_plan.is_file():
        raise FileNotFoundError(args.sweep_plan)
    plan = json.loads(args.sweep_plan.read_text(encoding="utf-8"))
    sweep_plan_sha = sha256_file(args.sweep_plan)
    if sha256_file(Path(plan["base_summary"])) != plan["base_summary_sha256"]:
        raise RuntimeError("base summary changed after the sweep plan was frozen")
    for item in plan["strengths"]:
        marker = args.output_root / item["variant"] / "variant.complete"
        if not marker.is_file():
            raise FileNotFoundError(marker)
    if torch.cuda.device_count() != 1 or args.cuda_idx != 0:
        raise RuntimeError("summary expects one GPU at cuda:0")
    base_summary = json.loads(Path(plan["base_summary"]).read_text(encoding="utf-8"))
    base_rows = {row["sample_id"]: row for row in base_summary["samples"]}
    metric = BatchedLPIPSAlex(torch.device("cuda:0"), args.lpips_batch_size)
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    rows = []
    visuals = {}
    for index, entry in enumerate(plan["entries"], start=1):
        sample_id = entry["sample_id"]
        source = load_source(entry)
        base = base_rows[sample_id]
        quality = {
            "codec_qp8": base["quality"]["codec_qp8"],
            "frozen": base["quality"]["frozen"],
            "lora_100": base["quality"]["lora"],
        }
        frames = {
            "frozen": load_pngs(Path(entry["base_outputs"]["frozen"]["frames_dir"])),
            "lora_100": load_pngs(Path(entry["base_outputs"]["lora"]["frames_dir"])),
        }
        for base_variant in ("frozen", "lora"):
            base_info = entry["base_outputs"][base_variant]
            if frame_digest(Path(base_info["frames_dir"])) != base_info["png_sha256"]:
                raise RuntimeError(
                    f"base output changed after plan freeze: {sample_id} {base_variant}")
        for strength, key in (
            (0.25, "lora_025"), (0.50, "lora_050"), (0.75, "lora_075"),
        ):
            variant = VARIANT_BY_STRENGTH[strength]
            sample_root = args.output_root / variant / sample_id
            if not valid_output(
                sample_root, entry=entry, strength=strength,
                sweep_plan_sha=sweep_plan_sha,
                dit_sha=plan["checkpoints"]["dit_sha256"],
                lora_sha=plan["lora_sha256"],
            ):
                raise RuntimeError(
                    f"strength output failed validation: {sample_id} {strength}")
            output_dir = sample_root / "frames"
            value = load_pngs(output_dir)
            frames[key] = value
            quality[key] = evaluate_quality(source, value, metric)
        row = {
            "sample_id": sample_id,
            "dataset": entry["dataset"],
            "sequence": entry["sequence"],
            "uvg_training_role": entry["uvg_training_role"],
            "quality": quality,
        }
        if sample_id in plan["visual_sample_ids"]:
            visual = args.output_dir / "visuals" / f"{sample_id}.png"
            save_strength_visual(visual, source, frames, quality)
            row["visual"] = str(visual.resolve())
            visuals[sample_id] = str(visual.resolve())
        rows.append(row)
        print(json.dumps({
            "stage": "strength-summary-sample",
            "index": index,
            "requested": len(plan["entries"]),
            "sample_id": sample_id,
            "dataset": entry["dataset"],
        }), flush=True)

    groups = {
        "combined": rows,
        "REDS": [row for row in rows if row["dataset"] == "REDS"],
        "UVG": [row for row in rows if row["dataset"] == "UVG"],
        "UVG-adaptation-sequences": [
            row for row in rows if row["uvg_training_role"] == "adaptation-sequence"],
        "UVG-holdout-sequences": [
            row for row in rows
            if row["uvg_training_role"]
            == "v6-and-lora-training-holdout-sequence"],
    }
    aggregates = {name: aggregate(value) for name, value in groups.items()}
    ranking = []
    for key, strength in (
        ("lora_025", 0.25), ("lora_050", 0.50),
        ("lora_075", 0.75), ("lora_100", 1.00),
    ):
        reds = aggregates["REDS"]
        uvg = aggregates["UVG"]
        reds_relative = (
            reds["delta_vs_frozen"][key]["lpips_alex"]
            / reds["quality"]["frozen"]["lpips_alex"])
        uvg_relative = (
            uvg["delta_vs_frozen"][key]["lpips_alex"]
            / uvg["quality"]["frozen"]["lpips_alex"])
        ranking.append({
            "key": key,
            "strength": strength,
            "REDS_relative_lpips_percent": 100.0 * reds_relative,
            "UVG_relative_lpips_percent": 100.0 * uvg_relative,
            "balanced_relative_lpips_percent": 50.0 * (reds_relative + uvg_relative),
        })
    ranking.sort(key=lambda x: x["balanced_relative_lpips_percent"])
    result = {
        "experiment": "SeedVR2 LoRA inference-strength sweep",
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit": git_commit(),
        "sample_count": len(rows),
        "primary_metric": "LPIPS Alex, lower is better",
        "sweep_plan": str(args.sweep_plan.resolve()),
        "sweep_plan_sha256": sweep_plan_sha,
        "aggregate": aggregates,
        "metric_only_ranking": ranking,
        "metric_only_best_strength": ranking[0]["strength"],
        "metric_only_best_is_not_automatic_default": True,
        "samples": rows,
        "visuals": visuals,
        "verification": {
            "base_frozen_rerun_pixel_exact_count": 37,
            "strength_0_and_1_reused_from_verified_base_evaluation": True,
            "same_inputs_seeds_and_adapter": True,
            "new_output_count": len(rows) * len(STRENGTHS),
            "peak_summary_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
        },
        "analysis_protocol": plan["analysis_protocol"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", result)
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_markdown(args.output_dir / "summary.md", result)
    atomic_text(args.output_dir / "summary.complete", "complete\n")
    print(json.dumps({
        "stage": "strength-summary-complete",
        "metric_only_ranking": ranking,
        "verification": result["verification"],
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    from demo.stage_c_seedvr2_lora_utils import self_test as lora_self_test

    value = lora_self_test()
    if not value["zero_strength_exact"] or not value["half_strength_linear"]:
        raise AssertionError("LoRA strength utility self-test failed")
    rows = []
    for amount in (0.0, 0.1):
        quality = {}
        for key in QUALITY_KEYS:
            quality[key] = {
                "lpips_alex": 1.0 + amount,
                "psnr_db": 2.0 - amount,
                "temporal_delta_mae": 3.0 + amount,
                "rgb_mse": 4.0 + amount,
            }
        rows.append({"quality": quality})
    result = aggregate(rows)
    if result["sample_count"] != 2:
        raise AssertionError("strength aggregate self-test failed")
    print(json.dumps({
        "status": "passed",
        "strengths": list(STRENGTHS),
        "lora_utility": value,
        "visual_count": len(VISUAL_IDS),
    }, indent=2))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            plan_main(args)
        elif args.command == "restore":
            restore_main(args)
        elif args.command == "summarize":
            summarize_main(args)
        else:
            self_test()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
