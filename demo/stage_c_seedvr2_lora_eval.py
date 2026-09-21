#!/usr/bin/env python3
"""Fixed-input frozen-vs-LoRA evaluation for SeedVR2-3B.

The experiment reuses the 37 real all-Generate spatial-QP streams and their
pixel-exact fresh decodes from the frozen REDS+UVG evaluation.  A new frozen
rerun and the trained LoRA receive identical decoded PNGs and per-sample noise
seeds.  This isolates the restoration backend from codec and router changes.
"""

from __future__ import annotations

import argparse
import csv
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
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import PersistentSeedVR2, load_source
from demo.stage_c_evaluate_seedvr2_gate import load_pngs
from demo.stage_c_three_path_roi_probe import (
    psnr_from_mse,
    rgb_mse,
    save_frames,
    temporal_delta_mae,
)


FORMAT_VERSION = 1
UVG_ADAPTATION_SEQUENCES = {
    "Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry",
}
UVG_HOLDOUT_SEQUENCES = {"ReadySetGo", "YachtRide"}
FIXED_VISUAL_IDS = (
    "dev-s000-f00-x384-y096",
    "eval-s024-f00-x384-y096",
    "uvg-beauty-f00-center512",
    "uvg-jockey-f00-center512",
    "uvg-readysetgo-f00-center512",
    "uvg-yachtride-f00-center512",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_paths(path: Path, expected: int = 17) -> list[Path]:
    paths = sorted(path.glob("*.png"))
    if len(paths) != expected:
        raise RuntimeError(
            f"expected {expected} PNGs in {path}, found {len(paths)}")
    return paths


def frame_digest(path: Path, expected: int = 17) -> str:
    digest = hashlib.sha256()
    for item in frame_paths(path, expected):
        digest.update(item.name.encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SeedVR2 against codec-artifact LoRA")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--joint-root", type=Path, required=True)
    plan.add_argument("--sample-manifest", type=Path, required=True)
    plan.add_argument("--dit-checkpoint", type=Path, required=True)
    plan.add_argument("--vae-checkpoint", type=Path, required=True)
    plan.add_argument("--positive-embedding", type=Path, required=True)
    plan.add_argument("--negative-embedding", type=Path, required=True)
    plan.add_argument("--lora-checkpoint", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    restore = commands.add_parser("restore")
    restore.add_argument("--plan", type=Path, required=True)
    restore.add_argument("--output-root", type=Path, required=True)
    restore.add_argument("--variant", choices=("frozen", "lora"), required=True)
    restore.add_argument("--upstream-root", type=Path, required=True)
    restore.add_argument("--dit-checkpoint", type=Path, required=True)
    restore.add_argument("--vae-checkpoint", type=Path, required=True)
    restore.add_argument("--positive-embedding", type=Path, required=True)
    restore.add_argument("--negative-embedding", type=Path, required=True)
    restore.add_argument("--lora-checkpoint", type=Path)
    restore.add_argument("--sample-steps", type=int, default=1)
    restore.add_argument("--cfg-scale", type=float, default=1.0)
    restore.add_argument(
        "--dit-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    restore.add_argument("--cuda-idx", type=int, default=0)
    restore.add_argument("--max-wall-seconds", type=int, default=7200)
    restore.add_argument("--disk-stop-percent", type=float, default=80.0)

    summary = commands.add_parser("summarize")
    summary.add_argument("--plan", type=Path, required=True)
    summary.add_argument("--output-root", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--lpips-batch-size", type=int, default=8)
    summary.add_argument("--cuda-idx", type=int, default=0)

    commands.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.command == "restore":
        if args.variant == "lora" and args.lora_checkpoint is None:
            parser.error("LoRA restore requires --lora-checkpoint")
        if args.variant == "frozen" and args.lora_checkpoint is not None:
            parser.error("frozen restore must not receive --lora-checkpoint")
        if args.sample_steps != 1 or args.cfg_scale != 1.0:
            parser.error("the fixed evaluation requires one step and CFG 1.0")
        if args.max_wall_seconds < 60:
            parser.error("wall limit must be at least one minute")
        if not 0 < args.disk_stop_percent < 100:
            parser.error("disk stop percent must be inside (0,100)")
    if args.command == "summarize" and args.lpips_batch_size < 1:
        parser.error("LPIPS batch size must be positive")
    return args


def plan_main(args: argparse.Namespace) -> None:
    required = [
        args.sample_manifest,
        args.dit_checkpoint,
        args.vae_checkpoint,
        args.positive_embedding,
        args.negative_embedding,
        args.lora_checkpoint,
        args.joint_root / "formal_evaluation.complete",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = read_jsonl(args.sample_manifest)
    if len(records) != 37 or Counter(x["dataset"] for x in records) != {
            "REDS": 30, "UVG": 7}:
        raise RuntimeError("expected the frozen 30 REDS + 7 UVG manifest")
    if len({record["sample_id"] for record in records}) != len(records):
        raise RuntimeError("sample IDs are not unique")

    entries = []
    for record in records:
        sample_id = record["sample_id"]
        root = (
            args.joint_root / "formal" / "evaluation" / sample_id /
            "spatial" / "all-generate")
        stream = root / "codec" / "stream.dcvc-sq"
        input_dir = root / "codec" / "fresh_decode"
        previous_output = root / "seedvr2"
        evaluation_path = root / "evaluation" / "evaluation.json"
        decode_path = root / "codec" / "decode_summary.json"
        metadata_path = previous_output / "seedvr2_metadata.json"
        for path in (stream, evaluation_path, decode_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        decode = json.loads(decode_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if evaluation["fresh_decode_regression"]["pixel_exact"] is not True:
            raise RuntimeError(f"{sample_id} lacks pixel-exact fresh decode")
        if decode.get("source_rgb_read_by_decoder") is not False:
            raise RuntimeError(f"{sample_id} decoder source boundary differs")
        if stream.stat().st_size != evaluation["stream"]["actual_on_disk_bytes"]:
            raise RuntimeError(f"{sample_id} stream byte accounting differs")
        frame_paths(input_dir)
        frame_paths(previous_output)
        sequence = record["sequence"]
        if record["dataset"] == "UVG":
            if sequence in UVG_ADAPTATION_SEQUENCES:
                uvg_role = "adaptation-sequence"
            elif sequence in UVG_HOLDOUT_SEQUENCES:
                uvg_role = "v6-and-lora-training-holdout-sequence"
            else:
                raise RuntimeError(f"unknown UVG sequence role: {sequence}")
        else:
            uvg_role = None
        entries.append({
            "sample_id": sample_id,
            "dataset": record["dataset"],
            "sequence": sequence,
            "data_role": record["data_role"],
            "source_role": record["source_role"],
            "uvg_training_role": uvg_role,
            "source_files": record["source_files"],
            "crop": record["crop"],
            "selected_source_sha256": record["selected_source_sha256"],
            "frame_count": 17,
            "seed": int(metadata["seed"]),
            "stream": str(stream.resolve()),
            "stream_bytes": stream.stat().st_size,
            "stream_sha256": sha256_file(stream),
            "fresh_decode_dir": str(input_dir.resolve()),
            "fresh_decode_png_sha256": frame_digest(input_dir),
            "previous_frozen_output_dir": str(previous_output.resolve()),
            "previous_frozen_png_sha256": frame_digest(previous_output),
            "previous_frozen_quality": evaluation["quality"],
            "fresh_decode_pixel_exact": True,
            "decoder_reads_source_rgb": False,
        })

    value = {
        "experiment": "SeedVR2 frozen-vs-LoRA fixed-input evaluation",
        "format_version": FORMAT_VERSION,
        "status": "frozen-before-new-inference",
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "sample_manifest": str(args.sample_manifest.resolve()),
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "source_joint_root": str(args.joint_root.resolve()),
        "sample_count": len(entries),
        "dataset_counts": dict(Counter(x["dataset"] for x in entries)),
        "uvg_role_counts": dict(Counter(
            x["uvg_training_role"] for x in entries
            if x["uvg_training_role"] is not None)),
        "checkpoints": {
            "dit": str(args.dit_checkpoint.resolve()),
            "dit_sha256": sha256_file(args.dit_checkpoint),
            "vae": str(args.vae_checkpoint.resolve()),
            "vae_sha256": sha256_file(args.vae_checkpoint),
            "positive_embedding": str(args.positive_embedding.resolve()),
            "positive_embedding_sha256": sha256_file(args.positive_embedding),
            "negative_embedding": str(args.negative_embedding.resolve()),
            "negative_embedding_sha256": sha256_file(args.negative_embedding),
            "lora": str(args.lora_checkpoint.resolve()),
            "lora_sha256": sha256_file(args.lora_checkpoint),
        },
        "inference": {
            "sample_steps": 1,
            "cfg_scale": 1.0,
            "dit_dtype": "bfloat16",
            "paired_seed_per_sample": True,
            "processing_shape": [17, 512, 512],
        },
        "fixed_visual_sample_ids": list(FIXED_VISUAL_IDS),
        "entries": entries,
        "scientific_boundary": {
            "all_37_samples_were_used_by_prior_v6_analysis": True,
            "not_new_independent_evidence": True,
            "plan_frozen_before_lora_outputs": True,
            "same_codec_stream_decode_and_noise_seed_for_both_variants": True,
            "real_spatial_qp_streams_reused_without_reencoding": True,
            "stream_bytes_unchanged_by_restoration_backend": True,
            "full_frame_generate_backend_test_before_roi_reintegration": True,
            "hard_promotion_gate": False,
            "single_gpu": True,
        },
    }
    atomic_json(args.output, value)
    print(json.dumps({
        "stage": "plan-complete",
        "output": str(args.output.resolve()),
        "sample_count": len(entries),
        "dataset_counts": value["dataset_counts"],
        "uvg_role_counts": value["uvg_role_counts"],
    }, ensure_ascii=False, indent=2))


def valid_restoration(
    sample_root: Path,
    *,
    entry: dict,
    variant: str,
    plan_sha256: str,
    dit_sha256: str,
    lora_sha256: str | None,
) -> bool:
    metadata_path = sample_root / "metadata.json"
    frames_dir = sample_root / "frames"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = frame_paths(frames_dir, int(entry["frame_count"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return False
    return (
        len(paths) == int(entry["frame_count"])
        and metadata.get("status") == "complete"
        and metadata.get("sample_id") == entry["sample_id"]
        and metadata.get("variant") == variant
        and metadata.get("seed") == int(entry["seed"])
        and metadata.get("plan_sha256") == plan_sha256
        and metadata.get("input_png_sha256")
        == entry["fresh_decode_png_sha256"]
        and metadata.get("dit_sha256") == dit_sha256
        and metadata.get("lora_sha256") == lora_sha256
        and metadata.get("output_png_sha256")
        == frame_digest(frames_dir, int(entry["frame_count"]))
    )


def move_incomplete(path: Path) -> None:
    if not path.exists():
        return
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(path.name + f".incomplete-{suffix}")
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(path.name + f".incomplete-{suffix}-{counter}")
    os.replace(path, target)


def build_variant_manifest(
    *,
    plan: dict,
    plan_path: Path,
    output_root: Path,
    variant: str,
    model_load_seconds: float,
) -> dict:
    entries = []
    for entry in plan["entries"]:
        path = output_root / variant / entry["sample_id"] / "metadata.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") == "complete":
                entries.append(value)
    result = {
        "experiment": "SeedVR2 fixed-input paired restoration",
        "format_version": FORMAT_VERSION,
        "status": "complete" if len(entries) == len(plan["entries"]) else "in-progress",
        "updated_utc": utc_now(),
        "variant": variant,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
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
        args.plan, args.dit_checkpoint, args.vae_checkpoint,
        args.positive_embedding, args.negative_embedding,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.lora_checkpoint is not None and not args.lora_checkpoint.is_file():
        raise FileNotFoundError(args.lora_checkpoint)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (plan.get("format_version") != FORMAT_VERSION
            or plan.get("status") != "frozen-before-new-inference"):
        raise RuntimeError("evaluation plan is incompatible")
    plan_sha = sha256_file(args.plan)
    dit_sha = sha256_file(args.dit_checkpoint)
    if dit_sha != plan["checkpoints"]["dit_sha256"]:
        raise RuntimeError("DiT checkpoint differs from the frozen plan")
    lora_sha = (
        sha256_file(args.lora_checkpoint)
        if args.lora_checkpoint is not None else None)
    if args.variant == "lora" and lora_sha != plan["checkpoints"]["lora_sha256"]:
        raise RuntimeError("LoRA checkpoint differs from the frozen plan")
    args.output_root.mkdir(parents=True, exist_ok=True)
    variant_root = args.output_root / args.variant
    variant_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for entry in plan["entries"]:
        sample_root = variant_root / entry["sample_id"]
        if valid_restoration(
            sample_root,
            entry=entry,
            variant=args.variant,
            plan_sha256=plan_sha,
            dit_sha256=dit_sha,
            lora_sha256=lora_sha,
        ):
            print(json.dumps({
                "stage": "restore-resume-skip",
                "variant": args.variant,
                "sample_id": entry["sample_id"],
            }), flush=True)
        else:
            pending.append(entry)
    if not pending:
        result = build_variant_manifest(
            plan=plan, plan_path=args.plan, output_root=args.output_root,
            variant=args.variant, model_load_seconds=0.0)
        if result["status"] != "complete":
            raise RuntimeError("restoration outputs are incomplete")
        atomic_text(variant_root / "variant.complete", "complete\n")
        print(json.dumps({
            "stage": "restore-all-complete-resume",
            "variant": args.variant,
            "completed": result["completed_sample_count"],
        }, indent=2))
        return
    if torch.cuda.device_count() != 1 or args.cuda_idx != 0:
        raise RuntimeError("the fixed evaluation requires one GPU at cuda:0")
    torch.cuda.set_device(args.cuda_idx)
    bridge_args = SimpleNamespace(
        upstream_root=args.upstream_root,
        dit_checkpoint=args.dit_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        positive_embedding=args.positive_embedding,
        negative_embedding=args.negative_embedding,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        dit_dtype=args.dit_dtype,
    )
    process_started = time.perf_counter()
    model = PersistentSeedVR2(bridge_args)
    for index, entry in enumerate(plan["entries"], start=1):
        sample_root = variant_root / entry["sample_id"]
        if valid_restoration(
            sample_root,
            entry=entry,
            variant=args.variant,
            plan_sha256=plan_sha,
            dit_sha256=dit_sha,
            lora_sha256=lora_sha,
        ):
            continue
        if time.perf_counter() - process_started >= args.max_wall_seconds:
            raise RuntimeError("restoration process reached its wall limit")
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
        output_digest = frame_digest(temporary / "frames")
        metadata = {
            "status": "complete",
            "completed_utc": utc_now(),
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            "sequence": entry["sequence"],
            "variant": args.variant,
            "seed": int(entry["seed"]),
            "plan": str(args.plan.resolve()),
            "plan_sha256": plan_sha,
            "input_dir": str(input_dir.resolve()),
            "input_png_sha256": entry["fresh_decode_png_sha256"],
            "output_dir": str((sample_root / "frames").resolve()),
            "output_png_sha256": output_digest,
            "frame_count": len(restored),
            "dit": str(args.dit_checkpoint.resolve()),
            "dit_sha256": dit_sha,
            "lora": (
                str(args.lora_checkpoint.resolve())
                if args.lora_checkpoint is not None else None),
            "lora_sha256": lora_sha,
            "lora_adapter": model.runner.lora_adapter_info,
            "runtime": runtime,
            "sample_wall_seconds": time.perf_counter() - sample_started,
            "model_load_seconds_shared": model.model_load_seconds,
            "same_input_and_seed_as_pair": True,
            "training_or_finetuning": False,
        }
        atomic_json(temporary / "metadata.json", metadata)
        move_incomplete(sample_root)
        os.replace(temporary, sample_root)
        print(json.dumps({
            "stage": "restore-sample",
            "variant": args.variant,
            "index": index,
            "requested": len(plan["entries"]),
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            "seconds": runtime["seconds_model_load_excluded"],
            "peak_cuda_allocated_bytes": runtime["peak_cuda_allocated_bytes"],
        }), flush=True)
        build_variant_manifest(
            plan=plan, plan_path=args.plan, output_root=args.output_root,
            variant=args.variant, model_load_seconds=model.model_load_seconds)
    result = build_variant_manifest(
        plan=plan, plan_path=args.plan, output_root=args.output_root,
        variant=args.variant, model_load_seconds=model.model_load_seconds)
    if result["status"] != "complete":
        raise RuntimeError("restoration did not complete")
    atomic_text(variant_root / "variant.complete", "complete\n")
    print(json.dumps({
        "stage": "restore-complete",
        "variant": args.variant,
        "completed": result["completed_sample_count"],
        "model_load_seconds": model.model_load_seconds,
        "model_seconds": result["total_model_seconds"],
        "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
    }, indent=2), flush=True)


class BatchedLPIPSAlex:
    def __init__(self, device: torch.device, batch_size: int) -> None:
        import lpips

        self.model = lpips.LPIPS(
            net="alex", verbose=False).eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def evaluate(
        self, reference: list[np.ndarray], reconstruction: list[np.ndarray],
    ) -> float:
        if len(reference) != len(reconstruction):
            raise ValueError("LPIPS frame counts differ")
        values = []
        for start in range(0, len(reference), self.batch_size):
            first = np.stack(reference[start:start + self.batch_size])
            second = np.stack(reconstruction[start:start + self.batch_size])
            first_tensor = torch.from_numpy(first.copy()).permute(0, 3, 1, 2)
            second_tensor = torch.from_numpy(second.copy()).permute(0, 3, 1, 2)
            first_tensor = first_tensor.to(
                self.device, dtype=torch.float32).div_(127.5).sub_(1)
            second_tensor = second_tensor.to(
                self.device, dtype=torch.float32).div_(127.5).sub_(1)
            values.extend(
                self.model(first_tensor, second_tensor).flatten().cpu().tolist())
        return float(np.mean(values))


def evaluate_quality(
    reference: list[np.ndarray],
    reconstruction: list[np.ndarray],
    metric: BatchedLPIPSAlex,
) -> dict:
    mse = rgb_mse(reference, reconstruction)
    return {
        "rgb_mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "lpips_alex": metric.evaluate(reference, reconstruction),
        "temporal_delta_mae": temporal_delta_mae(reference, reconstruction),
    }


def exact_frames(
    first: list[np.ndarray], second: list[np.ndarray],
) -> dict:
    if len(first) != len(second):
        return {"frame_count_equal": False, "pixel_exact": False}
    maximum = max(
        int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        for a, b in zip(first, second)
    )
    return {
        "frame_count_equal": True,
        "max_abs_pixel_error": maximum,
        "pixel_exact": maximum == 0,
    }


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(np.mean(values))


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}
    variants = ("codec_qp8", "frozen", "lora")
    metrics = ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
    result = {
        "sample_count": len(rows),
        "quality": {
            variant: {
                metric: mean([row["quality"][variant][metric] for row in rows])
                for metric in metrics
            }
            for variant in variants
        },
    }
    result["delta_lora_minus_frozen"] = {
        metric: (
            result["quality"]["lora"][metric]
            - result["quality"]["frozen"][metric])
        for metric in metrics
    }
    result["paired_counts"] = {
        "lora_lower_lpips": sum(
            row["delta_lora_minus_frozen"]["lpips_alex"] < 0 for row in rows),
        "equal_lpips": sum(
            row["delta_lora_minus_frozen"]["lpips_alex"] == 0 for row in rows),
        "lora_higher_psnr": sum(
            row["delta_lora_minus_frozen"]["psnr_db"] > 0 for row in rows),
        "lora_lower_temporal_error": sum(
            row["delta_lora_minus_frozen"]["temporal_delta_mae"] < 0
            for row in rows),
    }
    result["mean_abs_pixel_change_lora_vs_frozen"] = mean([
        row["lora_vs_frozen"]["mean_abs_pixel_change"] for row in rows])
    return result


def visual_panel(frame: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(frame)
    banner = 68
    result = Image.new("RGB", (image.width, image.height + banner), "white")
    result.paste(image, (0, banner))
    draw = ImageDraw.Draw(result)
    bold = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    normal = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((8, 7), title, fill="black", font=bold)
    draw.text((8, 38), subtitle, fill="black", font=normal)
    return result


def metric_subtitle(quality: dict) -> str:
    return (
        f"LPIPS {quality['lpips_alex']:.4f} | "
        f"PSNR {quality['psnr_db']:.2f} | T {quality['temporal_delta_mae']:.2f}")


def save_visual(
    path: Path,
    reference: list[np.ndarray],
    codec: list[np.ndarray],
    frozen: list[np.ndarray],
    lora: list[np.ndarray],
    quality: dict,
    frame_index: int = 8,
) -> None:
    difference = np.abs(
        lora[frame_index].astype(np.int16)
        - frozen[frame_index].astype(np.int16)).astype(np.float32)
    difference = np.clip(difference * 6.0, 0, 255).round().astype(np.uint8)
    panels = [
        visual_panel(reference[frame_index], "GT", "fixed frame 9"),
        visual_panel(
            codec[frame_index], "QP8 codec input",
            metric_subtitle(quality["codec_qp8"])),
        visual_panel(
            frozen[frame_index], "Frozen SeedVR2",
            metric_subtitle(quality["frozen"])),
        visual_panel(
            lora[frame_index], "SeedVR2 + LoRA",
            metric_subtitle(quality["lora"])),
        visual_panel(
            difference, "|LoRA - frozen| x6",
            "display-only difference map"),
    ]
    canvas = Image.new(
        "RGB", (sum(panel.width for panel in panels), max(
            panel.height for panel in panels)), (230, 230, 230))
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id", "dataset", "sequence", "data_role", "uvg_training_role",
        "frozen_lpips", "lora_lpips", "delta_lpips",
        "frozen_psnr", "lora_psnr", "delta_psnr",
        "frozen_temporal", "lora_temporal", "delta_temporal",
        "mean_abs_pixel_change", "frozen_rerun_pixel_exact",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                "data_role": row["data_role"],
                "uvg_training_role": row["uvg_training_role"],
                "frozen_lpips": row["quality"]["frozen"]["lpips_alex"],
                "lora_lpips": row["quality"]["lora"]["lpips_alex"],
                "delta_lpips": row["delta_lora_minus_frozen"]["lpips_alex"],
                "frozen_psnr": row["quality"]["frozen"]["psnr_db"],
                "lora_psnr": row["quality"]["lora"]["psnr_db"],
                "delta_psnr": row["delta_lora_minus_frozen"]["psnr_db"],
                "frozen_temporal": row["quality"]["frozen"]["temporal_delta_mae"],
                "lora_temporal": row["quality"]["lora"]["temporal_delta_mae"],
                "delta_temporal": row["delta_lora_minus_frozen"]["temporal_delta_mae"],
                "mean_abs_pixel_change": row["lora_vs_frozen"]["mean_abs_pixel_change"],
                "frozen_rerun_pixel_exact": row["frozen_rerun_vs_previous"]["pixel_exact"],
            })
    os.replace(temporary, path)


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# SeedVR2 frozen vs LoRA fixed-input evaluation",
        "",
        "同一批真实 QP8 spatial-QP fresh decode、同一随机种子；LPIPS 和时序误差越低越好，PSNR 越高越好。",
        "",
        "| Group | N | Frozen LPIPS | LoRA LPIPS | Delta | LPIPS better | PSNR delta | Temporal delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["aggregate"].items():
        if value["sample_count"] == 0:
            continue
        lines.append(
            f"| {name} | {value['sample_count']} | "
            f"{value['quality']['frozen']['lpips_alex']:.6f} | "
            f"{value['quality']['lora']['lpips_alex']:.6f} | "
            f"{value['delta_lora_minus_frozen']['lpips_alex']:+.6f} | "
            f"{value['paired_counts']['lora_lower_lpips']}/{value['sample_count']} | "
            f"{value['delta_lora_minus_frozen']['psnr_db']:+.3f} | "
            f"{value['delta_lora_minus_frozen']['temporal_delta_mae']:+.3f} |")
    lines.extend([
        "",
        f"Frozen rerun pixel-exact with prior frozen output: {result['verification']['frozen_rerun_pixel_exact_count']}/{result['sample_count']}.",
        "",
        "This comparison is development/error analysis on the already-used 37 samples, not new independent evidence. No hard promotion threshold was applied.",
    ])
    atomic_text(path, "\n".join(lines) + "\n")


def summarize_main(args: argparse.Namespace) -> None:
    for path in (
        args.plan,
        args.output_root / "frozen" / "variant.complete",
        args.output_root / "lora" / "variant.complete",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if torch.cuda.device_count() != 1 or args.cuda_idx != 0:
        raise RuntimeError("summary expects one GPU at cuda:0")
    device = torch.device("cuda:0")
    metric = BatchedLPIPSAlex(device, args.lpips_batch_size)
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    visual_paths = {}
    for index, entry in enumerate(plan["entries"], start=1):
        source = load_source(entry)
        codec = load_pngs(Path(entry["fresh_decode_dir"]))
        previous_frozen = load_pngs(Path(entry["previous_frozen_output_dir"]))
        frozen_dir = args.output_root / "frozen" / entry["sample_id"] / "frames"
        lora_dir = args.output_root / "lora" / entry["sample_id"] / "frames"
        frozen = load_pngs(frozen_dir)
        lora = load_pngs(lora_dir)
        regression = exact_frames(previous_frozen, frozen)
        quality = {
            "codec_qp8": evaluate_quality(source, codec, metric),
            "frozen": evaluate_quality(source, frozen, metric),
            "lora": evaluate_quality(source, lora, metric),
        }
        delta = {
            name: quality["lora"][name] - quality["frozen"][name]
            for name in ("lpips_alex", "psnr_db", "temporal_delta_mae", "rgb_mse")
        }
        abs_change = float(np.mean(np.abs(
            np.stack(lora).astype(np.float32)
            - np.stack(frozen).astype(np.float32))))
        row = {
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            "sequence": entry["sequence"],
            "data_role": entry["data_role"],
            "uvg_training_role": entry["uvg_training_role"],
            "stream_bytes": entry["stream_bytes"],
            "quality": quality,
            "delta_lora_minus_frozen": delta,
            "lora_vs_frozen": {
                "mean_abs_pixel_change": abs_change,
            },
            "frozen_rerun_vs_previous": regression,
        }
        if entry["sample_id"] in plan["fixed_visual_sample_ids"]:
            visual = args.output_dir / "visuals" / f"{entry['sample_id']}.png"
            save_visual(visual, source, codec, frozen, lora, quality)
            row["fixed_visual"] = str(visual.resolve())
            visual_paths[entry["sample_id"]] = str(visual.resolve())
        rows.append(row)
        print(json.dumps({
            "stage": "summary-sample",
            "index": index,
            "requested": len(plan["entries"]),
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            "lpips_delta": delta["lpips_alex"],
            "frozen_pixel_exact": regression["pixel_exact"],
        }), flush=True)
    groups = {
        "combined": rows,
        "REDS": [row for row in rows if row["dataset"] == "REDS"],
        "UVG": [row for row in rows if row["dataset"] == "UVG"],
        "UVG-adaptation-sequences": [
            row for row in rows
            if row["uvg_training_role"] == "adaptation-sequence"],
        "UVG-holdout-sequences": [
            row for row in rows
            if row["uvg_training_role"]
            == "v6-and-lora-training-holdout-sequence"],
    }
    frozen_exact = sum(
        row["frozen_rerun_vs_previous"]["pixel_exact"] for row in rows)
    result = {
        "experiment": "SeedVR2 frozen-vs-LoRA fixed-input evaluation",
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit": git_commit(),
        "sample_count": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "primary_metric": "LPIPS Alex, lower is better",
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "aggregate": {name: aggregate(value) for name, value in groups.items()},
        "samples": rows,
        "visuals": visual_paths,
        "verification": {
            "same_input_and_seed_for_both_variants": True,
            "all_input_streams_have_prior_pixel_exact_fresh_decode": all(
                entry["fresh_decode_pixel_exact"] for entry in plan["entries"]),
            "frozen_rerun_pixel_exact_count": frozen_exact,
            "frozen_rerun_all_pixel_exact": frozen_exact == len(rows),
            "peak_summary_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
        },
        "scientific_boundary": plan["scientific_boundary"],
    }
    if not result["verification"]["frozen_rerun_all_pixel_exact"]:
        raise RuntimeError(
            "new frozen runner differs from the prior frozen output; "
            "do not interpret the LoRA pair")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", result)
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_markdown(args.output_dir / "summary.md", result)
    atomic_text(args.output_dir / "summary.complete", "complete\n")
    print(json.dumps({
        "stage": "summary-complete",
        "sample_count": len(rows),
        "aggregate": result["aggregate"],
        "verification": result["verification"],
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    reference = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    same = [frame.copy() for frame in reference]
    changed = [np.full((4, 4, 3), 2, dtype=np.uint8) for _ in range(2)]
    if exact_frames(reference, same)["pixel_exact"] is not True:
        raise AssertionError("exact-frame self-test failed")
    if exact_frames(reference, changed)["max_abs_pixel_error"] != 2:
        raise AssertionError("pixel-difference self-test failed")
    rows = [{
        "quality": {
            "codec_qp8": {"lpips_alex": 2.0, "psnr_db": 1.0,
                          "temporal_delta_mae": 3.0, "rgb_mse": 4.0},
            "frozen": {"lpips_alex": 1.0, "psnr_db": 2.0,
                       "temporal_delta_mae": 3.0, "rgb_mse": 4.0},
            "lora": {"lpips_alex": 0.5, "psnr_db": 2.5,
                     "temporal_delta_mae": 2.5, "rgb_mse": 3.5},
        },
        "delta_lora_minus_frozen": {
            "lpips_alex": -0.5, "psnr_db": 0.5,
            "temporal_delta_mae": -0.5, "rgb_mse": -0.5,
        },
        "lora_vs_frozen": {"mean_abs_pixel_change": 2.0},
    }]
    value = aggregate(rows)
    if value["delta_lora_minus_frozen"]["lpips_alex"] != -0.5:
        raise AssertionError("aggregate self-test failed")
    print(json.dumps({
        "status": "passed",
        "format_version": FORMAT_VERSION,
        "fixed_visual_count": len(FIXED_VISUAL_IDS),
        "paired_metrics": ["LPIPS Alex", "PSNR", "temporal delta MAE"],
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
