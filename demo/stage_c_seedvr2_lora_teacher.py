#!/usr/bin/env python3
"""Rebuild only the Generate part of the regional teacher with SeedVR2 LoRA.

The original 560-sample teacher already contains encoder-visible features,
Base/Enhance quality, real byte costs, and measured ROI costs.  None of those
quantities changes when only the frozen inference-time SeedVR2 LoRA strength
changes.  This stage therefore reuses them byte-for-byte and recomputes only
the Generate reconstruction and its dependent targets.

Each sample re-encodes only the original scalar QP8 Generate input and reads
the materialized stream back for a fresh decode.  This keeps the input exactly
aligned with the old teacher protocol without repeating QP16, QP32, regional
Enhance streams, or ROI timing.  A small frozen-strength replay is required
before the long run proceeds and must reproduce the old teacher metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import (
    FEATURE_NAMES,
    PersistentSeedVR2,
    SpatialLPIPS,
    load_source,
    local_quality,
    stream_roundtrip,
)
from demo.stage_c_seedvr2_lora_utils import set_lora_strength
from demo.stage_c_three_path_roi_probe import (
    encode_dcvc_stream,
    load_codecs,
    tile_boxes,
)
from src.utils.common import set_torch_env


PROTOCOL_VERSION = 2
GENERATE_TARGETS = (
    "generate_lpips_gain_vs_base",
    "generate_psnr_delta_db_vs_base",
    "generate_temporal_risk_vs_base",
)
REUSED_TARGETS = (
    "enhance_lpips_gain_vs_base",
    "enhance_psnr_delta_db_vs_base",
    "enhance_temporal_gain_vs_base",
    "enhance_fallback_extra_on_disk_bytes",
    "generate_region_area_pixels",
    "generate_region_area_fraction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def disk_percent(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total


def dataset_for_sample(sample: dict) -> str:
    split = sample["split"]
    if split == "train":
        return "REDS"
    if split == "v6_adaptation_train":
        return "UVG"
    raise RuntimeError(f"unexpected teacher split: {split}")


def validate_manifest(path: Path, expected: int | None = None) -> dict:
    value = read_json(path)
    complete = value.get("complete", value.get("status") == "complete")
    if not complete:
        raise RuntimeError(f"manifest is incomplete: {path}")
    entries = value.get("entries", [])
    completed = value.get("completed_sample_count", len(entries))
    if completed != len(entries):
        raise RuntimeError(f"manifest count differs from entries: {path}")
    if expected is not None and len(entries) != expected:
        raise RuntimeError(
            f"expected {expected} entries in {path}, found {len(entries)}")
    ids = [entry["sample_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate sample IDs in {path}")
    return value


def diagnostic_ids(entries: list[dict]) -> list[str]:
    selected = []
    seen_uvg = set()
    for entry in entries:
        if entry["dataset"] == "REDS" and not selected:
            selected.append(entry["sample_id"])
        elif entry["dataset"] == "UVG" and entry["sequence"] not in seen_uvg:
            selected.append(entry["sample_id"])
            seen_uvg.add(entry["sequence"])
    return selected


def plan_main(args: argparse.Namespace) -> None:
    source_path = args.source_quality_manifest.resolve()
    roi_path = args.roi_cost_manifest.resolve()
    adapter_path = args.lora_checkpoint.resolve()
    model_paths = {
        "codec_image": args.model_path_i.resolve(),
        "codec_video": args.model_path_p.resolve(),
        "base_dit": args.dit_checkpoint.resolve(),
        "vae": args.vae_checkpoint.resolve(),
        "positive_embedding": args.positive_embedding.resolve(),
        "negative_embedding": args.negative_embedding.resolve(),
    }
    for path in (source_path, roi_path, adapter_path, *model_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = validate_manifest(source_path, 560)
    roi = validate_manifest(roi_path, 560)
    source_by_id = {entry["sample_id"]: entry for entry in source["entries"]}
    roi_ids = [entry["sample_id"] for entry in roi["entries"]]
    source_ids = list(source_by_id)
    if set(source_ids) != set(roi_ids):
        raise RuntimeError("quality and ROI-cost sample IDs differ")
    entries = []
    counts = Counter()
    for sample_id in source_ids:
        source_entry = source_by_id[sample_id]
        source_record = Path(source_entry["path"]).resolve()
        if not source_record.is_file():
            raise FileNotFoundError(source_record)
        teacher = read_json(source_record)
        sample = teacher["sample"]
        dataset = dataset_for_sample(sample)
        counts[dataset] += 1
        entries.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "split": sample["split"],
            "sequence": sample["sequence"],
            "source_teacher_record": str(source_record),
            "source_teacher_record_sha256": sha256_file(source_record),
            "expected_scalar_qp8_stream_bytes": source_entry[
                "uniform_stream_bytes"]["Generate"],
        })
    if counts != Counter({"REDS": 500, "UVG": 60}):
        raise RuntimeError(f"unexpected dataset counts: {counts}")
    value = {
        "experiment": "SeedVR2 LoRA 0.50 Generate-teacher rebuild",
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen-before-output-generation",
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "lora_strength": args.lora_strength,
        "sample_count": len(entries),
        "region_count": len(entries) * 16,
        "dataset_sample_counts": dict(counts),
        "entries": entries,
        "frozen_replay_diagnostic_ids": diagnostic_ids(entries),
        "inputs": {
            "source_quality_manifest": str(source_path),
            "source_quality_manifest_sha256": sha256_file(source_path),
            "roi_cost_manifest_reused": str(roi_path),
            "roi_cost_manifest_sha256": sha256_file(roi_path),
            "lora_checkpoint": str(adapter_path),
            "lora_checkpoint_sha256": sha256_file(adapter_path),
            "models": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in model_paths.items()
            },
        },
        "reuse_policy": {
            "recompute": [
                "scalar QP8 stream and fresh decode",
                "Generate reconstruction",
                "Generate regional LPIPS/PSNR/temporal metrics",
                *GENERATE_TARGETS,
            ],
            "reuse_exactly": [
                "encoder-visible features",
                "Base quality",
                "Enhance quality",
                "uniform/fallback byte labels",
                "measured ROI-cost teacher",
            ],
            "reason": (
                "only the frozen inference-time SeedVR2 LoRA multiplier changed"
            ),
        },
        "scientific_boundary": {
            "single_gpu_only": True,
            "dcvc_uf_checkpoint_frozen": True,
            "seedvr2_base_dit_and_vae_frozen": True,
            "seedvr2_lora_adapter_frozen_during_teacher_rebuild": True,
            "seedvr2_lora_strength": args.lora_strength,
            "source_rgb_used_for_teacher_metrics_only": True,
            "generate_input_is_reencoded_scalar_qp8_and_fresh_decoded": True,
            "base_enhance_and_roi_costs_not_recomputed": True,
            "hard_promotion_gate": False,
        },
    }
    if args.output.is_file():
        old = read_json(args.output)
        comparable_old = copy.deepcopy(old)
        comparable_new = copy.deepcopy(value)
        for item in (comparable_old, comparable_new):
            item.pop("created_utc", None)
        if comparable_old != comparable_new:
            raise RuntimeError("existing immutable teacher plan differs")
        print(json.dumps({"stage": "plan-reused", "path": str(args.output)}, indent=2))
        return
    atomic_json(args.output, value)
    print(json.dumps({
        "stage": "plan-created",
        "path": str(args.output.resolve()),
        "sample_count": len(entries),
        "dataset_sample_counts": dict(counts),
        "diagnostic_ids": value["frozen_replay_diagnostic_ids"],
    }, ensure_ascii=False, indent=2))


def seedvr2_args(args: argparse.Namespace, plan: dict) -> SimpleNamespace:
    inputs = plan["inputs"]
    return SimpleNamespace(
        upstream_root=args.upstream_root.resolve(),
        dit_checkpoint=Path(inputs["models"]["base_dit"]["path"]),
        lora_checkpoint=Path(inputs["lora_checkpoint"]),
        lora_strength=float(plan["lora_strength"]),
        vae_checkpoint=Path(inputs["models"]["vae"]["path"]),
        positive_embedding=Path(inputs["models"]["positive_embedding"]["path"]),
        negative_embedding=Path(inputs["models"]["negative_embedding"]["path"]),
        sample_steps=1,
        cfg_scale=1.0,
        dit_dtype="bfloat16",
    )


def regional_quality(
    source: list[np.ndarray], output: list[np.ndarray], metric: SpatialLPIPS,
) -> list[dict[str, float]]:
    boxes = tile_boxes(512, 512, 128)
    maps = metric.maps(source, output)
    return [local_quality(source, output, maps, box) for box in boxes]


def maximum_quality_difference(first: list[dict], second: list[dict]) -> dict:
    names = ("lpips_alex", "psnr_db", "rgb_mse", "temporal_delta_mae")
    return {
        name: max(abs(float(a[name]) - float(b[name])) for a, b in zip(first, second))
        for name in names
    }


def update_generate_teacher(
    teacher: dict,
    qualities: list[dict[str, float]],
) -> dict:
    if len(qualities) != 16 or len(teacher["regions"]) != 16:
        raise RuntimeError("teacher must contain exactly 16 regions")
    output = copy.deepcopy(teacher)
    for region, quality in zip(output["regions"], qualities):
        base = region["candidates"]["Base"]
        region["candidates"]["Generate"] = quality
        region["targets"]["generate_lpips_gain_vs_base"] = (
            float(base["lpips_alex"]) - float(quality["lpips_alex"]))
        region["targets"]["generate_psnr_delta_db_vs_base"] = (
            float(quality["psnr_db"]) - float(base["psnr_db"]))
        region["targets"]["generate_temporal_risk_vs_base"] = max(
            0.0,
            float(quality["temporal_delta_mae"])
            - float(base["temporal_delta_mae"]),
        )
    return output


def save_visual(
    path: Path,
    source: list[np.ndarray],
    frozen: list[np.ndarray],
    adapted: list[np.ndarray],
) -> None:
    frame = 8
    panels = [source[frame], frozen[frame], adapted[frame]]
    labels = ("GT", "Frozen SeedVR2", "LoRA strength 0.50")
    title_height = 30
    canvas = Image.new("RGB", (512 * 3, 512 + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(array), (index * 512, title_height))
        draw.text((index * 512 + 8, 8), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def selected_entries(plan: dict, limit_per_dataset: int | None) -> list[dict]:
    entries = list(plan["entries"])
    if limit_per_dataset is None:
        return entries
    counts = Counter()
    selected = []
    for entry in entries:
        dataset = entry["dataset"]
        if counts[dataset] < limit_per_dataset:
            selected.append(entry)
            counts[dataset] += 1
    if counts != Counter({"REDS": limit_per_dataset, "UVG": limit_per_dataset}):
        raise RuntimeError(f"balanced smoke selection failed: {counts}")
    return selected


def validate_resumed_sample(path: Path, entry: dict, plan_hash: str, plan: dict) -> dict:
    value = read_json(path)
    provenance = value.get("provenance", {})
    if (value.get("protocol_version") != PROTOCOL_VERSION
            or value.get("sample", {}).get("sample_id") != entry["sample_id"]
            or provenance.get("experiment_plan_sha256") != plan_hash
            or provenance.get("lora_checkpoint_sha256")
            != plan["inputs"]["lora_checkpoint_sha256"]
            or provenance.get("lora_strength") != plan["lora_strength"]):
        raise RuntimeError(f"incompatible resumed teacher sample: {path}")
    return value


def write_manifest(
    output_dir: Path,
    entries: list[dict],
    plan: dict,
    plan_path: Path,
    started: float,
    model_load_seconds: float,
) -> dict:
    plan_hash = sha256_file(plan_path)
    rows = []
    for entry in entries:
        path = output_dir / "samples" / f"{entry['sample_id']}.json"
        if not path.is_file():
            continue
        sample = validate_resumed_sample(path, entry, plan_hash, plan)
        runtime = sample["runtime"]
        rows.append({
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            "split": entry["split"],
            "sequence": entry["sequence"],
            "path": str(path.resolve()),
            "seconds": runtime["sample_total_seconds"],
            "seedvr2_seconds": runtime["seedvr2"]["seconds_model_load_excluded"],
            "frozen_replay_checked": runtime.get("frozen_protocol_regression") is not None,
        })
    manifest = {
        "experiment": "SeedVR2 LoRA 0.50 Generate-teacher labels",
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete" if len(rows) == len(entries) else "in-progress",
        "complete": len(rows) == len(entries),
        "updated_utc": utc_now(),
        "requested_sample_count": len(entries),
        "completed_sample_count": len(rows),
        "dataset_sample_counts": dict(Counter(row["dataset"] for row in rows)),
        "entries": rows,
        "feature_names": list(FEATURE_NAMES),
        "experiment_plan": str(plan_path.resolve()),
        "experiment_plan_sha256": plan_hash,
        "lora_checkpoint": plan["inputs"]["lora_checkpoint"],
        "lora_checkpoint_sha256": plan["inputs"]["lora_checkpoint_sha256"],
        "lora_strength": plan["lora_strength"],
        "roi_cost_manifest_reused": plan["inputs"]["roi_cost_manifest_reused"],
        "model_load_seconds": model_load_seconds,
        "current_process_elapsed_seconds": time.perf_counter() - started,
        "scientific_boundary": plan["scientific_boundary"],
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


@torch.inference_mode()
def relabel_main(args: argparse.Namespace) -> None:
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    if plan.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported teacher rebuild plan")
    if plan.get("git_commit") != git_commit():
        raise RuntimeError("repository commit differs from frozen teacher plan")
    entries = selected_entries(plan, args.limit_per_dataset)
    plan_hash = sha256_file(plan_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        args.output_dir / "samples" / f"{entry['sample_id']}.json"
        for entry in entries
    ]
    for entry, path in zip(entries, existing):
        if path.is_file():
            validate_resumed_sample(path, entry, plan_hash, plan)
    if existing and all(path.is_file() for path in existing):
        previous = args.output_dir / "manifest.json"
        old_load = read_json(previous).get("model_load_seconds", 0.0) if previous.is_file() else 0.0
        manifest = write_manifest(
            args.output_dir, entries, plan, plan_path, time.perf_counter(), old_load)
        print(json.dumps({
            "stage": "teacher-relabel-resume-all-complete",
            "completed": manifest["completed_sample_count"],
        }, indent=2))
        return
    set_torch_env()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("teacher rebuild supports one GPU only")
    if torch.cuda.device_count() != 1 or args.cuda_idx != 0:
        raise RuntimeError("teacher rebuild requires one visible GPU at cuda:0")
    torch.cuda.set_device(0)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    codec_args = SimpleNamespace(
        model_path_i=Path(plan["inputs"]["models"]["codec_image"]["path"]),
        model_path_p=Path(plan["inputs"]["models"]["codec_video"]["path"]),
        skip_thres=0.0,
    )
    codec_load_started = time.perf_counter()
    i_net, p_net = load_codecs(codec_args, device)
    codec_load_seconds = time.perf_counter() - codec_load_started
    codec_stream = torch.cuda.Stream(device=device)
    seedvr2 = PersistentSeedVR2(seedvr2_args(args, plan))
    metric = SpatialLPIPS(seedvr2.device, args.lpips_batch_size)
    diagnostic = set(plan["frozen_replay_diagnostic_ids"])
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    completed_this_process = 0
    for index, entry in enumerate(entries, start=1):
        output_path = args.output_dir / "samples" / f"{entry['sample_id']}.json"
        if output_path.is_file():
            print(json.dumps({
                "stage": "teacher-relabel-resume-skip",
                "index": index,
                "requested": len(entries),
                "sample_id": entry["sample_id"],
            }), flush=True)
            continue
        if time.perf_counter() - started >= args.max_wall_seconds:
            break
        usage = {
            name: disk_percent(Path(name))
            for name in ("/root", "/root/autodl-tmp", "/root/autodl-fs")
        }
        if max(usage.values()) >= args.disk_stop_percent:
            raise RuntimeError(f"disk stop threshold reached: {usage}")
        sample_started = time.perf_counter()
        teacher_path = Path(entry["source_teacher_record"])
        teacher = read_json(teacher_path)
        if sha256_file(teacher_path) != entry["source_teacher_record_sha256"]:
            raise RuntimeError(f"source teacher changed: {teacher_path}")
        source = load_source(teacher["sample"])
        sample_scratch = args.scratch_dir / entry["sample_id"]
        if sample_scratch.exists():
            shutil.rmtree(sample_scratch)
        sample_scratch.mkdir(parents=True, exist_ok=False)
        with torch.cuda.stream(codec_stream):
            stream, encode_runtime = encode_dcvc_stream(
                source, 8, 8, i_net, p_net, device, 32)
            stream_sha256 = hashlib.sha256(stream).hexdigest()
            decoded, stream_bytes, stream_breakdown = stream_roundtrip(
                stream=stream,
                path=sample_scratch / "uniform_qp8.dcvc",
                frame_count=17,
                i_net=i_net,
                p_net=p_net,
                device=device,
            )
        if stream_bytes != entry["expected_scalar_qp8_stream_bytes"]:
            raise RuntimeError(
                f"scalar QP8 bytes changed for {entry['sample_id']}: "
                f"{stream_bytes} != {entry['expected_scalar_qp8_stream_bytes']}")
        if stream_breakdown != teacher["rate"]["uniform_stream_breakdown"]["Generate"]:
            raise RuntimeError(
                f"scalar QP8 stream breakdown changed for {entry['sample_id']}")
        seed = int(teacher["sample"]["seed"])
        set_lora_strength(seedvr2.runner.dit, plan["lora_strength"])
        adapted, adapted_runtime = seedvr2.restore(decoded, seed)
        adapted_runtime["lora_strength"] = plan["lora_strength"]
        adapted_runtime["condition_source"] = "fresh-decoded scalar QP8 stream"
        metric_started = time.perf_counter()
        adapted_quality = regional_quality(source, adapted, metric)
        metric_seconds = time.perf_counter() - metric_started
        frozen_regression = None
        if entry["sample_id"] in diagnostic:
            set_lora_strength(seedvr2.runner.dit, 0.0)
            frozen, frozen_runtime = seedvr2.restore(decoded, seed)
            frozen_runtime["lora_strength"] = 0.0
            frozen_runtime["condition_source"] = "fresh-decoded scalar QP8 stream"
            frozen_quality = regional_quality(source, frozen, metric)
            old_quality = [
                region["candidates"]["Generate"] for region in teacher["regions"]
            ]
            differences = maximum_quality_difference(frozen_quality, old_quality)
            frozen_regression = {
                "strength": 0.0,
                "regional_metric_max_abs_difference_vs_source_teacher": differences,
                "tolerance": args.frozen_metric_tolerance,
                "within_tolerance": max(differences.values()) <= args.frozen_metric_tolerance,
                "runtime": frozen_runtime,
            }
            if not frozen_regression["within_tolerance"]:
                raise RuntimeError(
                    f"fresh scalar QP8 replay does not reproduce frozen teacher for "
                    f"{entry['sample_id']}: {differences}")
            save_visual(
                args.output_dir / "visuals" / f"{entry['sample_id']}.png",
                source, frozen, adapted)
            set_lora_strength(seedvr2.runner.dit, plan["lora_strength"])
        result = update_generate_teacher(teacher, adapted_quality)
        result["experiment"] = "A800 regional SeedVR2-LoRA teacher refresh"
        result["protocol_version"] = PROTOCOL_VERSION
        result["configuration"]["seedvr2_lora_checkpoint"] = plan["inputs"]["lora_checkpoint"]
        result["configuration"]["seedvr2_lora_checkpoint_sha256"] = plan["inputs"]["lora_checkpoint_sha256"]
        result["configuration"]["seedvr2_lora_strength"] = plan["lora_strength"]
        result["runtime"] = {
            "sample_total_seconds": time.perf_counter() - sample_started,
            "codec_model_load_seconds_process_level": codec_load_seconds,
            "generate_qp8_encode": encode_runtime,
            "generate_qp8_stream_bytes": stream_bytes,
            "generate_qp8_stream_sha256": stream_sha256,
            "generate_qp8_fresh_decode": True,
            "seedvr2_model_load_seconds_process_level": seedvr2.model_load_seconds,
            "seedvr2": adapted_runtime,
            "generate_metric_seconds": metric_seconds,
            "frozen_protocol_regression": frozen_regression,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(seedvr2.device)),
            "disk_percent_before_sample": usage,
            "source_teacher_sample_total_seconds_not_repeated": teacher["runtime"]["sample_total_seconds"],
        }
        result["provenance"] = {
            "experiment_plan": str(plan_path),
            "experiment_plan_sha256": plan_hash,
            "source_teacher_record": str(teacher_path),
            "source_teacher_record_sha256": entry["source_teacher_record_sha256"],
            "scalar_qp8_stream_bytes": stream_bytes,
            "scalar_qp8_stream_sha256": stream_sha256,
            "scalar_qp8_stream_materialized_and_fresh_decoded": True,
            "lora_checkpoint": plan["inputs"]["lora_checkpoint"],
            "lora_checkpoint_sha256": plan["inputs"]["lora_checkpoint_sha256"],
            "lora_strength": plan["lora_strength"],
        }
        result["scientific_boundary"] = {
            **plan["scientific_boundary"],
            "teacher_rebuild_training_or_finetuning": False,
            "generate_quality_recomputed": True,
            "base_enhance_features_rates_reused_exactly": True,
            "frozen_protocol_regression_checked_for_this_sample": frozen_regression is not None,
            "true_fill_used": False,
        }
        for old_region, new_region in zip(teacher["regions"], result["regions"]):
            if old_region["features"] != new_region["features"]:
                raise RuntimeError("encoder features changed during Generate relabel")
            for action in ("Base", "Enhance"):
                if old_region["candidates"][action] != new_region["candidates"][action]:
                    raise RuntimeError(f"{action} quality changed during Generate relabel")
            for target in REUSED_TARGETS:
                if old_region["targets"][target] != new_region["targets"][target]:
                    raise RuntimeError(f"reused target changed: {target}")
        if teacher["rate"] != result["rate"]:
            raise RuntimeError("rate labels changed during Generate relabel")
        atomic_json(output_path, result)
        if any(sample_scratch.iterdir()):
            raise RuntimeError(f"scratch files remain after {entry['sample_id']}")
        sample_scratch.rmdir()
        completed_this_process += 1
        print(json.dumps({
            "stage": "teacher-relabel-sample",
            "index": index,
            "requested": len(entries),
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            "sample_seconds": result["runtime"]["sample_total_seconds"],
            "seedvr2_seconds": adapted_runtime["seconds_model_load_excluded"],
            "frozen_regression": frozen_regression is not None,
            "peak_cuda_mib": result["runtime"]["peak_cuda_allocated_bytes"] / 1048576,
            "disk_percent": usage,
        }, ensure_ascii=False), flush=True)
        if completed_this_process == 1 or completed_this_process % 10 == 0:
            write_manifest(
                args.output_dir, entries, plan, plan_path, started,
                seedvr2.model_load_seconds)
    manifest = write_manifest(
        args.output_dir, entries, plan, plan_path, started,
        seedvr2.model_load_seconds)
    print(json.dumps({
        "stage": "teacher-relabel-finished",
        "manifest": str((args.output_dir / "manifest.json").resolve()),
        "requested": manifest["requested_sample_count"],
        "completed": manifest["completed_sample_count"],
        "complete": manifest["complete"],
        "elapsed_seconds": time.perf_counter() - started,
    }, ensure_ascii=False, indent=2))


def aggregate_rows(rows: list[dict]) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in rows]))

    return {
        "region_count": len(rows),
        "sample_count": len({row["sample_id"] for row in rows}),
        "old_generate_lpips_mean": mean("old_lpips"),
        "new_generate_lpips_mean": mean("new_lpips"),
        "generate_lpips_delta_mean": mean("lpips_delta"),
        "generate_lpips_improved_regions": sum(row["lpips_delta"] < 0 for row in rows),
        "old_generate_psnr_db_mean": mean("old_psnr"),
        "new_generate_psnr_db_mean": mean("new_psnr"),
        "generate_psnr_delta_db_mean": mean("psnr_delta"),
        "old_generate_temporal_delta_mae_mean": mean("old_temporal"),
        "new_generate_temporal_delta_mae_mean": mean("new_temporal"),
        "generate_temporal_delta_mean": mean("temporal_delta"),
        "old_generate_gain_vs_base_mean": mean("old_gain"),
        "new_generate_gain_vs_base_mean": mean("new_gain"),
        "generate_gain_change_mean": mean("gain_delta"),
        "old_beneficial_generate_regions": sum(row["old_gain"] > 0 for row in rows),
        "new_beneficial_generate_regions": sum(row["new_gain"] > 0 for row in rows),
        "old_lpips_oracle_action_counts": dict(Counter(row["old_oracle"] for row in rows)),
        "new_lpips_oracle_action_counts": dict(Counter(row["new_oracle"] for row in rows)),
    }


def summarize_main(args: argparse.Namespace) -> None:
    plan = read_json(args.plan)
    manifest = validate_manifest(args.teacher_manifest, 560)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unexpected refreshed teacher protocol")
    if manifest.get("experiment_plan_sha256") != sha256_file(args.plan):
        raise RuntimeError("teacher manifest does not match experiment plan")
    output_by_id = {entry["sample_id"]: entry for entry in manifest["entries"]}
    rows = []
    frozen_checks = []
    unchanged = Counter()
    for entry in plan["entries"]:
        old = read_json(Path(entry["source_teacher_record"]))
        new = read_json(Path(output_by_id[entry["sample_id"]]["path"]))
        if old["sample"] != new["sample"] or old["rate"] != new["rate"]:
            raise RuntimeError(f"sample identity or rate changed: {entry['sample_id']}")
        check = new["runtime"].get("frozen_protocol_regression")
        if check is not None:
            if not check["within_tolerance"]:
                raise RuntimeError("a frozen protocol regression failed")
            frozen_checks.append({"sample_id": entry["sample_id"], **check})
        for old_region, new_region in zip(old["regions"], new["regions"]):
            if old_region["features"] != new_region["features"]:
                raise RuntimeError("features changed")
            for action in ("Base", "Enhance"):
                if old_region["candidates"][action] != new_region["candidates"][action]:
                    raise RuntimeError(f"{action} candidate changed")
                unchanged[f"{action}_candidate"] += 1
            for target in REUSED_TARGETS:
                if old_region["targets"][target] != new_region["targets"][target]:
                    raise RuntimeError(f"reused target changed: {target}")
                unchanged[target] += 1
            old_generate = old_region["candidates"]["Generate"]
            new_generate = new_region["candidates"]["Generate"]
            base = old_region["candidates"]["Base"]
            enhance = old_region["candidates"]["Enhance"]
            old_gain = float(base["lpips_alex"] - old_generate["lpips_alex"])
            new_gain = float(base["lpips_alex"] - new_generate["lpips_alex"])
            old_candidates = {
                "Generate": old_generate["lpips_alex"],
                "Base": base["lpips_alex"],
                "Enhance": enhance["lpips_alex"],
            }
            new_candidates = {**old_candidates, "Generate": new_generate["lpips_alex"]}
            rows.append({
                "sample_id": entry["sample_id"],
                "dataset": entry["dataset"],
                "old_lpips": float(old_generate["lpips_alex"]),
                "new_lpips": float(new_generate["lpips_alex"]),
                "lpips_delta": float(new_generate["lpips_alex"] - old_generate["lpips_alex"]),
                "old_psnr": float(old_generate["psnr_db"]),
                "new_psnr": float(new_generate["psnr_db"]),
                "psnr_delta": float(new_generate["psnr_db"] - old_generate["psnr_db"]),
                "old_temporal": float(old_generate["temporal_delta_mae"]),
                "new_temporal": float(new_generate["temporal_delta_mae"]),
                "temporal_delta": float(new_generate["temporal_delta_mae"] - old_generate["temporal_delta_mae"]),
                "old_gain": old_gain,
                "new_gain": new_gain,
                "gain_delta": new_gain - old_gain,
                "old_oracle": min(old_candidates, key=old_candidates.get),
                "new_oracle": min(new_candidates, key=new_candidates.get),
            })
    groups = {
        "combined": aggregate_rows(rows),
        "REDS": aggregate_rows([row for row in rows if row["dataset"] == "REDS"]),
        "UVG": aggregate_rows([row for row in rows if row["dataset"] == "UVG"]),
    }
    result = {
        "experiment": "SeedVR2 LoRA 0.50 Generate-teacher rebuild summary",
        "status": "complete",
        "completed_utc": utc_now(),
        "sample_count": 560,
        "region_count": len(rows),
        "lora_strength": plan["lora_strength"],
        "adapter_sha256": plan["inputs"]["lora_checkpoint_sha256"],
        "groups": groups,
        "frozen_protocol_regression": {
            "checked_sample_count": len(frozen_checks),
            "expected_ids": plan["frozen_replay_diagnostic_ids"],
            "checked_ids": [item["sample_id"] for item in frozen_checks],
            "all_within_tolerance": all(item["within_tolerance"] for item in frozen_checks),
            "checks": frozen_checks,
        },
        "reuse_integrity": {
            "unchanged_counts": dict(unchanged),
            "expected_per_field": len(rows),
            "source_sample_and_rate_exact": True,
            "features_base_enhance_and_non_generate_targets_exact": True,
            "roi_cost_manifest_reused": plan["inputs"]["roi_cost_manifest_reused"],
            "roi_cost_manifest_sha256": plan["inputs"]["roi_cost_manifest_sha256"],
        },
        "runtime": {
            "model_load_seconds": manifest["model_load_seconds"],
            "seedvr2_inference_seconds_sum": float(sum(
                entry["seedvr2_seconds"] for entry in manifest["entries"])),
            "sample_wall_seconds_sum": float(sum(
                entry["seconds"] for entry in manifest["entries"])),
        },
        "next_step": (
            "retrain the fixed conservative residual router on the refreshed "
            "quality manifest while reusing the measured ROI-cost manifest"
        ),
        "scientific_boundary": plan["scientific_boundary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", result)
    combined = groups["combined"]
    markdown = f"""# SeedVR2 LoRA 0.50 Generate teacher refresh

- Samples: 560 (REDS 500 + UVG 60); regions: {len(rows)}.
- Mean regional Generate LPIPS: {combined['old_generate_lpips_mean']:.6f} -> {combined['new_generate_lpips_mean']:.6f} ({combined['generate_lpips_delta_mean']:+.6f}).
- Improved Generate regions: {combined['generate_lpips_improved_regions']}/{len(rows)}.
- Beneficial Generate regions versus Base: {combined['old_beneficial_generate_regions']} -> {combined['new_beneficial_generate_regions']}.
- Mean Generate PSNR change: {combined['generate_psnr_delta_db_mean']:+.4f} dB.
- Mean Generate temporal-error change: {combined['generate_temporal_delta_mean']:+.4f}.
- Frozen scalar-QP8 replay: {len(frozen_checks)}/{len(plan['frozen_replay_diagnostic_ids'])} diagnostic samples within tolerance.
- Base, Enhance, features, rates and reused ROI costs were not recomputed and passed exact JSON equality checks.

Next: retrain the conservative router weights with these Generate labels, then apply the unchanged spatial-consistency term and run the fixed 37-video comparison.
"""
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def selection_main(args: argparse.Namespace) -> None:
    source = read_json(args.source_selection)
    if source.get("selection_policy", {}).get("hard_promotion_gate") is not False:
        raise RuntimeError("source router selection unexpectedly has a hard gate")
    plan = read_json(args.plan)
    value = copy.deepcopy(source)
    value["experiment"] = "LoRA-teacher router retraining with fixed v5 policy"
    value["status"] = "fixed-before-LoRA-teacher router retraining"
    value["created_utc"] = utc_now()
    value["source_selection"] = {
        "path": str(args.source_selection.resolve()),
        "sha256": sha256_file(args.source_selection),
    }
    value["teacher_rebuild_plan"] = {
        "path": str(args.plan.resolve()),
        "sha256": sha256_file(args.plan),
        "lora_strength": plan["lora_strength"],
        "adapter_sha256": plan["inputs"]["lora_checkpoint_sha256"],
    }
    value["selection_policy"]["hyperparameters_reused_to_isolate_teacher_change"] = True
    boundary = value.setdefault("scientific_boundary", {})
    boundary.update({
        "seedvr2_frozen": False,
        "seedvr2_base_weights_frozen": True,
        "seedvr2_lora_adapter_frozen_at_strength_0_50": True,
        "router_weights_retrained": True,
        "selection_hyperparameters_changed": False,
        "roi_cost_teacher_reused": True,
        "hard_promotion_gate": False,
    })
    atomic_json(args.output, value)
    print(json.dumps({
        "stage": "router-selection-created",
        "path": str(args.output.resolve()),
        "selected_candidate": value["selected_candidate"],
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    teacher = {
        "regions": [{
            "candidates": {
                "Generate": {"lpips_alex": 0.5, "psnr_db": 20.0, "temporal_delta_mae": 3.0},
                "Base": {"lpips_alex": 0.4, "psnr_db": 22.0, "temporal_delta_mae": 2.0},
                "Enhance": {"lpips_alex": 0.3, "psnr_db": 24.0, "temporal_delta_mae": 1.0},
            },
            "targets": {
                "generate_lpips_gain_vs_base": -0.1,
                "enhance_lpips_gain_vs_base": 0.1,
                "generate_psnr_delta_db_vs_base": -2.0,
                "enhance_psnr_delta_db_vs_base": 2.0,
                "generate_temporal_risk_vs_base": 1.0,
                "enhance_temporal_gain_vs_base": 1.0,
                "enhance_fallback_extra_on_disk_bytes": 100,
                "generate_region_area_pixels": 16,
                "generate_region_area_fraction": 0.0625,
            },
        } for _ in range(16)]
    }
    quality = [{
        "lpips_alex": 0.35,
        "psnr_db": 23.0,
        "rgb_mse": 10.0,
        "temporal_delta_mae": 1.5,
    } for _ in range(16)]
    output = update_generate_teacher(teacher, quality)
    assert math.isclose(
        output["regions"][0]["targets"]["generate_lpips_gain_vs_base"],
        0.05,
    )
    assert output["regions"][0]["targets"]["generate_psnr_delta_db_vs_base"] == 1.0
    assert output["regions"][0]["targets"]["generate_temporal_risk_vs_base"] == 0.0
    assert output["regions"][0]["candidates"]["Base"] == teacher["regions"][0]["candidates"]["Base"]
    assert teacher["regions"][0]["candidates"]["Generate"]["lpips_alex"] == 0.5
    print(json.dumps({"stage": "self-test", "status": "passed"}))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Generate teacher values with SeedVR2 LoRA")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--source-quality-manifest", type=Path, required=True)
    plan.add_argument("--roi-cost-manifest", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--lora-strength", type=float, default=0.5)
    plan.add_argument("--model-path-i", type=Path, required=True)
    plan.add_argument("--model-path-p", type=Path, required=True)
    plan.add_argument("--dit-checkpoint", type=Path, required=True)
    plan.add_argument("--vae-checkpoint", type=Path, required=True)
    plan.add_argument("--positive-embedding", type=Path, required=True)
    plan.add_argument("--negative-embedding", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    relabel = commands.add_parser("relabel")
    relabel.add_argument("--plan", type=Path, required=True)
    relabel.add_argument("--output-dir", type=Path, required=True)
    relabel.add_argument(
        "--scratch-dir", type=Path,
        default=Path("/root/autodl-tmp/DCVC/tmp/a800_lora_teacher"))
    relabel.add_argument(
        "--upstream-root", type=Path,
        default=REPO_ROOT / "third_party" / "SeedVR2")
    relabel.add_argument("--limit-per-dataset", type=int)
    relabel.add_argument("--lpips-batch-size", type=int, default=4)
    relabel.add_argument("--frozen-metric-tolerance", type=float, default=1e-5)
    relabel.add_argument("--max-wall-seconds", type=int, default=12 * 60 * 60)
    relabel.add_argument("--disk-stop-percent", type=float, default=80.0)
    relabel.add_argument("--cuda-idx", type=int, default=0)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--teacher-manifest", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    selection = commands.add_parser("make-router-selection")
    selection.add_argument("--source-selection", type=Path, required=True)
    selection.add_argument("--plan", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)

    commands.add_parser("self-test")
    args = parser.parse_args(argv)
    if hasattr(args, "lora_strength") and (
            not math.isfinite(args.lora_strength) or args.lora_strength < 0):
        parser.error("LoRA strength must be finite and nonnegative")
    if getattr(args, "limit_per_dataset", None) is not None and args.limit_per_dataset < 1:
        parser.error("limit per dataset must be positive")
    if getattr(args, "lpips_batch_size", 1) < 1:
        parser.error("LPIPS batch size must be positive")
    if getattr(args, "frozen_metric_tolerance", 0.0) < 0:
        parser.error("frozen metric tolerance must be nonnegative")
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.command == "plan":
        plan_main(args)
    elif args.command == "relabel":
        relabel_main(args)
    elif args.command == "summarize":
        summarize_main(args)
    elif args.command == "make-router-selection":
        selection_main(args)
    else:
        self_test()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
