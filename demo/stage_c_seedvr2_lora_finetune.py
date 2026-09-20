#!/usr/bin/env python3
"""Single-A800 SeedVR2 codec-artifact adaptation.

Two resumable stages are provided:

* ``cache`` materializes real QP8 DCVC-UF round trips and stores deterministic
  VAE latents for the decoded input and clean target.
* ``train`` keeps the released SeedVR2-3B DiT frozen and optimizes a compact
  LoRA adapter in its last eight blocks for the actual one-step inference path.

The cache may use a selected spatial-QP-aware codec checkpoint, while the
SeedVR2 base checkpoint and VAE always remain frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "SeedVR2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_teacher import load_source, stream_roundtrip
from demo.stage_c_seedvr2_bridge import (
    configure_runner,
    install_flash_attention_fallback,
    pad_temporal,
    resize_and_normalize,
)
from demo.stage_c_seedvr2_lora_utils import (
    DEFAULT_ALPHA,
    DEFAULT_LAST_N_BLOCKS,
    DEFAULT_RANK,
    adapter_payload,
    atomic_torch_save,
    inject_seedvr2_lora,
    load_lora_state_dict,
    save_lora_adapter,
    self_test as lora_self_test,
    trainable_lora_parameters,
)
from demo.stage_c_three_path_roi_probe import encode_dcvc_stream, load_codecs
from src.utils.common import set_torch_env


CACHE_FORMAT_VERSION = 1
TRAINING_FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-A800 SeedVR2 LoRA adaptation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--sample-manifest", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--scratch-dir", type=Path, required=True)
    cache.add_argument("--model-path-i", type=Path, required=True)
    cache.add_argument("--model-path-p", type=Path, required=True)
    cache.add_argument("--checkpoint-role", required=True)
    cache.add_argument("--vae-checkpoint", type=Path, required=True)
    cache.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT)
    cache.add_argument("--generate-qp", type=int, default=8)
    cache.add_argument("--reset-interval", type=int, default=32)
    cache.add_argument("--skip-thres", type=float, default=0.0)
    cache.add_argument("--limit", type=int)
    cache.add_argument(
        "--limit-per-dataset", type=int,
        help="Optional balanced REDS/UVG cap used by the smoke run")
    cache.add_argument("--disk-stop-percent", type=float, default=85.0)
    cache.add_argument("--cuda-idx", type=int, default=0)

    train = subparsers.add_parser("train")
    train.add_argument("--cache-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT)
    train.add_argument("--dit-checkpoint", type=Path, required=True)
    train.add_argument("--vae-checkpoint", type=Path, required=True)
    train.add_argument("--positive-embedding", type=Path, required=True)
    train.add_argument("--negative-embedding", type=Path, required=True)
    train.add_argument("--max-steps", type=int, default=1000)
    train.add_argument("--patch-size", type=int, default=256)
    train.add_argument("--uvg-probability", type=float, default=0.25)
    train.add_argument("--learning-rate", type=float, default=1.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--rank", type=int, default=DEFAULT_RANK)
    train.add_argument("--lora-alpha", type=float, default=DEFAULT_ALPHA)
    train.add_argument(
        "--last-n-blocks", type=int, default=DEFAULT_LAST_N_BLOCKS)
    train.add_argument("--save-every", type=int, default=25)
    train.add_argument("--seed", type=int, default=20260920)
    train.add_argument("--cuda-idx", type=int, default=0)

    subparsers.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.command == "cache":
        if ((args.limit is not None and args.limit < 1)
                or (args.limit_per_dataset is not None
                    and args.limit_per_dataset < 1)):
            parser.error("cache limits must be positive")
        if args.limit is not None and args.limit_per_dataset is not None:
            parser.error("use only one cache limit")
        if not 0 <= args.generate_qp < 64:
            parser.error("--generate-qp must be inside [0,63]")
        if not 0 < args.disk_stop_percent < 100:
            parser.error("--disk-stop-percent must be inside (0,100)")
    if args.command == "train":
        if args.max_steps < 1 or args.save_every < 1:
            parser.error("steps and save interval must be positive")
        if args.patch_size < 64 or args.patch_size % 16:
            parser.error("--patch-size must be >=64 and divisible by 16")
        if not 0 <= args.uvg_probability <= 1:
            parser.error("--uvg-probability must be inside [0,1]")
        if args.learning_rate <= 0 or args.grad_clip <= 0:
            parser.error("learning rate and gradient clip must be positive")
    return args


def initialize_single_gpu(cuda_idx: int) -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("SeedVR2 adaptation is authorized for one GPU only")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    if cuda_idx != 0:
        raise RuntimeError("the single visible GPU must use cuda index 0")
    set_torch_env()
    install_flash_attention_fallback()
    upstream = str(UPSTREAM_ROOT.resolve())
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from common.distributed import get_device, init_torch

    if not torch.distributed.is_initialized():
        init_torch(cudnn_benchmark=False)
    return get_device()


def load_combined_records(
    path: Path,
    limit: int | None,
    limit_per_dataset: int | None,
) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    records = []
    for entry in entries:
        teacher = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        sample = teacher["sample"]
        if sample["frame_count"] != 17:
            raise RuntimeError("SeedVR2 cache requires 17-frame samples")
        if sample["crop"]["width"] != 512 or sample["crop"]["height"] != 512:
            raise RuntimeError("SeedVR2 cache requires 512x512 source crops")
        split = sample["split"]
        if split not in ("train", "v6_adaptation_train"):
            raise RuntimeError(f"unexpected adaptation split: {split}")
        dataset = "UVG" if split == "v6_adaptation_train" else "REDS"
        records.append({
            **sample,
            "dataset": dataset,
            "teacher_record": str(Path(entry["path"]).resolve()),
        })
    if limit_per_dataset is not None:
        balanced = []
        counts = Counter()
        for record in records:
            dataset = record["dataset"]
            if counts[dataset] < limit_per_dataset:
                balanced.append(record)
                counts[dataset] += 1
        records = balanced
    elif limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError("empty SeedVR2 adaptation manifest")
    return records


def configure_frozen_vae(args: argparse.Namespace, device: torch.device):
    from common.config import create_object, load_config
    from omegaconf import OmegaConf

    previous = Path.cwd()
    try:
        os.chdir(args.upstream_root.resolve())
        config = load_config("configs_3b/main.yaml")
    finally:
        os.chdir(previous)
    OmegaConf.set_readonly(config, False)
    config.vae.checkpoint = str(args.vae_checkpoint.resolve())
    config.vae.slicing.memory_device = "cpu"
    vae = create_object(config.vae.model)
    vae.requires_grad_(False).eval().to(
        device=device, dtype=getattr(torch, config.vae.dtype))
    state = torch.load(
        args.vae_checkpoint, map_location=device, mmap=True, weights_only=True)
    vae.load_state_dict(state)
    if hasattr(vae, "set_causal_slicing"):
        vae.set_causal_slicing(**config.vae.slicing)
    if hasattr(vae, "set_memory_limit"):
        vae.set_memory_limit(**config.vae.memory_limit)
    return vae, config


@torch.inference_mode()
def deterministic_vae_latent(
    frames: list[np.ndarray], vae, config, device: torch.device,
) -> torch.Tensor:
    values = np.stack(frames)
    tensor = torch.from_numpy(values.copy()).permute(0, 3, 1, 2)
    tensor = tensor.float().div_(255.0)
    sample, original_length = pad_temporal(
        resize_and_normalize(tensor, 512, 512, device))
    if original_length != 17:
        raise RuntimeError("unexpected temporal padding input")
    sample = sample.unsqueeze(0).to(device, dtype=getattr(torch, config.vae.dtype))
    if hasattr(vae, "preprocess"):
        sample = vae.preprocess(sample)
    encoded = vae.encode(sample)
    latent = (
        encoded.posterior.mode()
        if encoded.posterior is not None else encoded.latent)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    latent = latent.permute(0, 2, 3, 4, 1).squeeze(0)
    latent = latent.mul(float(config.vae.scaling_factor))
    return latent.to(device="cpu", dtype=torch.bfloat16).contiguous()


def validate_cache_file(path: Path, sample_id: str) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("format_version") != CACHE_FORMAT_VERSION
            or value.get("sample_id") != sample_id):
        raise RuntimeError(f"invalid latent cache file: {path}")
    clean = value["clean_latent"]
    degraded = value["degraded_latent"]
    if clean.shape != degraded.shape or clean.ndim != 4 or clean.shape[-1] != 16:
        raise RuntimeError(f"invalid latent cache tensor shape: {path}")
    if clean.dtype != torch.bfloat16 or degraded.dtype != torch.bfloat16:
        raise RuntimeError(f"invalid latent cache dtype: {path}")
    return value


def write_cache_manifest(
    args: argparse.Namespace,
    records: list[dict],
    codec: dict,
    vae_sha256: str,
    started: float,
) -> dict:
    entries = []
    for record in records:
        path = args.output_dir / "samples" / f"{record['sample_id']}.pt"
        if not path.is_file():
            continue
        value = validate_cache_file(path, record["sample_id"])
        entries.append({
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record["split"],
            "sequence": record["sequence"],
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "stream_bytes": value["stream_bytes"],
            "latent_shape": list(value["clean_latent"].shape),
        })
    counts = Counter(entry["dataset"] for entry in entries)
    manifest = {
        "experiment": "SeedVR2 codec-artifact deterministic latent cache",
        "format_version": CACHE_FORMAT_VERSION,
        "status": "complete" if len(entries) == len(records) else "in-progress",
        "updated_utc": utc_now(),
        "git_commit": git_commit(),
        "source_manifest": str(args.sample_manifest.resolve()),
        "requested_sample_count": len(records),
        "completed_sample_count": len(entries),
        "dataset_counts": dict(counts),
        "entries": entries,
        "codec": codec,
        "generate_qp": args.generate_qp,
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_sha256": vae_sha256,
        "latent_encoding": {
            "posterior_mode_not_sample": True,
            "clean_and_real_qp8_decode": True,
            "full_source_shape": [17, 512, 512],
            "dtype": "bfloat16",
        },
        "current_process_elapsed_seconds": time.perf_counter() - started,
        "scientific_boundary": {
            "training_and_adaptation_data_only": True,
            "reds_train_and_uvg_adaptation": True,
            "codec_frozen_during_cache": True,
            "seedvr2_vae_frozen": True,
            "seedvr2_dit_not_loaded": True,
            "real_codec_stream_materialized_and_read_back": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    return manifest


def cache_main(args: argparse.Namespace) -> None:
    for path in (
        args.sample_manifest, args.model_path_i, args.model_path_p,
        args.vae_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    records = load_combined_records(
        args.sample_manifest, args.limit, args.limit_per_dataset)
    codec = {
        "role": args.checkpoint_role,
        "image": str(args.model_path_i.resolve()),
        "image_sha256": sha256_file(args.model_path_i),
        "video": str(args.model_path_p.resolve()),
        "video_sha256": sha256_file(args.model_path_p),
    }
    vae_sha256 = sha256_file(args.vae_checkpoint)
    started = time.perf_counter()
    existing = [
        args.output_dir / "samples" / f"{record['sample_id']}.pt"
        for record in records
    ]
    if existing and all(path.is_file() for path in existing):
        manifest = write_cache_manifest(
            args, records, codec, vae_sha256, started)
        if manifest["status"] != "complete":
            raise RuntimeError("latent cache files exist but manifest is incomplete")
        print(json.dumps({
            "stage": "cache-resume-all-complete",
            "completed": len(existing),
        }, indent=2))
        return

    device = initialize_single_gpu(args.cuda_idx)
    codec_args = SimpleNamespace(
        model_path_i=args.model_path_i,
        model_path_p=args.model_path_p,
        skip_thres=args.skip_thres,
    )
    i_net, p_net = load_codecs(codec_args, device)
    vae, config = configure_frozen_vae(args, device)
    torch.cuda.reset_peak_memory_stats(device)
    for index, record in enumerate(records, start=1):
        output = args.output_dir / "samples" / f"{record['sample_id']}.pt"
        if output.is_file():
            validate_cache_file(output, record["sample_id"])
            print(json.dumps({
                "stage": "cache-resume-skip", "index": index,
                "requested": len(records), "sample_id": record["sample_id"],
            }), flush=True)
            continue
        usage = shutil.disk_usage(args.output_dir)
        used_percent = 100.0 * usage.used / usage.total
        if used_percent >= args.disk_stop_percent:
            raise RuntimeError(
                f"file-store use {used_percent:.2f}% reached stop threshold")
        sample_started = time.perf_counter()
        source = load_source(record)
        stream, encode = encode_dcvc_stream(
            source, args.generate_qp, args.generate_qp,
            i_net, p_net, device, args.reset_interval)
        degraded, stream_bytes, _ = stream_roundtrip(
            stream=stream,
            path=args.scratch_dir / f"{record['sample_id']}.bin",
            frame_count=17,
            i_net=i_net,
            p_net=p_net,
            device=device,
        )
        clean_latent = deterministic_vae_latent(source, vae, config, device)
        degraded_latent = deterministic_vae_latent(
            degraded, vae, config, device)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record["split"],
            "sequence": record["sequence"],
            "source_files": record["source_files"],
            "crop": record["crop"],
            "generate_qp": args.generate_qp,
            "stream_bytes": stream_bytes,
            "codec_role": args.checkpoint_role,
            "codec_image_sha256": codec["image_sha256"],
            "codec_video_sha256": codec["video_sha256"],
            "vae_sha256": vae_sha256,
            "clean_latent": clean_latent,
            "degraded_latent": degraded_latent,
        }
        atomic_torch_save(output, payload)
        del source, degraded, stream, clean_latent, degraded_latent, payload
        elapsed = time.perf_counter() - sample_started
        print(json.dumps({
            "stage": "cache-sample", "index": index,
            "requested": len(records), "sample_id": record["sample_id"],
            "dataset": record["dataset"], "stream_bytes": stream_bytes,
            "encode_seconds": encode["seconds"],
            "sample_seconds": elapsed,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
        }), flush=True)
        if index % 10 == 0:
            write_cache_manifest(args, records, codec, vae_sha256, started)
    manifest = write_cache_manifest(args, records, codec, vae_sha256, started)
    if manifest["status"] != "complete":
        raise RuntimeError("latent cache did not complete")
    print(json.dumps({
        "stage": "cache-complete",
        "manifest": str((args.output_dir / "manifest.json").resolve()),
        "completed": manifest["completed_sample_count"],
        "dataset_counts": manifest["dataset_counts"],
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }, indent=2), flush=True)


def immutable_training_config(args: argparse.Namespace, manifest: dict) -> dict:
    return {
        "format_version": TRAINING_FORMAT_VERSION,
        "cache_manifest": str(args.cache_manifest.resolve()),
        "cache_manifest_sha256_at_start": sha256_file(args.cache_manifest),
        "cache_sample_count": manifest["completed_sample_count"],
        "base_dit": str(args.dit_checkpoint.resolve()),
        "base_dit_sha256": sha256_file(args.dit_checkpoint),
        "positive_embedding": str(args.positive_embedding.resolve()),
        "positive_embedding_sha256": sha256_file(args.positive_embedding),
        "max_steps": args.max_steps,
        "patch_size": args.patch_size,
        "uvg_probability": args.uvg_probability,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "last_n_blocks": args.last_n_blocks,
        "seed": args.seed,
        "one_step_timestep": 1000.0,
        "loss": {
            "velocity_mse": 1.0,
            "clean_latent_l1": 0.1,
            "temporal_delta_mse": 0.2,
        },
    }


def select_training_entry(
    entries: list[dict], step: int, seed: int, uvg_probability: float,
) -> tuple[dict, random.Random]:
    rng = random.Random(seed + step * 1_000_003)
    by_dataset = {
        dataset: [entry for entry in entries if entry["dataset"] == dataset]
        for dataset in ("REDS", "UVG")
    }
    if not by_dataset["REDS"] or not by_dataset["UVG"]:
        raise RuntimeError("training cache must contain both REDS and UVG")
    dataset = "UVG" if rng.random() < uvg_probability else "REDS"
    return rng.choice(by_dataset[dataset]), rng


def crop_latents(
    clean: torch.Tensor,
    degraded: torch.Tensor,
    patch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if clean.shape != degraded.shape or clean.ndim != 4:
        raise RuntimeError("cached latent tensors differ")
    latent_patch = patch_size // 8
    _, height, width, _ = clean.shape
    if latent_patch > height or latent_patch > width:
        raise RuntimeError("training patch is larger than the cached latent")
    top = rng.randrange(height - latent_patch + 1)
    left = rng.randrange(width - latent_patch + 1)
    clean = clean[:, top:top + latent_patch, left:left + latent_patch]
    degraded = degraded[:, top:top + latent_patch, left:left + latent_patch]
    flip = rng.random() < 0.5
    if flip:
        clean = clean.flip(2)
        degraded = degraded.flip(2)
    return clean.contiguous(), degraded.contiguous(), {
        "latent_top": top,
        "latent_left": left,
        "latent_size": latent_patch,
        "horizontal_flip": flip,
    }


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_step_log(path: Path, completed_steps: int) -> list[dict]:
    """Drop records newer than the last atomic resume checkpoint."""

    if not path.is_file():
        return []
    by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["step"])
        if step <= completed_steps:
            by_step[step] = record
    records = [by_step[step] for step in sorted(by_step)]
    if records and [record["step"] for record in records] != list(
            range(1, completed_steps + 1)):
        raise RuntimeError("SeedVR2 step log is incomplete before resume point")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8")
    os.replace(temporary, path)
    return records


def save_resume_checkpoint(
    path: Path,
    model,
    optimizer,
    completed_steps: int,
    config: dict,
    records: list[dict],
) -> None:
    payload = {
        "format_version": TRAINING_FORMAT_VERSION,
        "completed_steps": completed_steps,
        "config": config,
        "adapter": adapter_payload(
            model,
            rank=config["rank"],
            alpha=config["lora_alpha"],
            last_n_blocks=config["last_n_blocks"],
            metadata={"completed_steps": completed_steps},
        ),
        "optimizer": optimizer.state_dict(),
        "recent_records": records[-25:],
    }
    atomic_torch_save(path, payload)


def train_main(args: argparse.Namespace) -> None:
    for path in (
        args.cache_manifest, args.dit_checkpoint, args.vae_checkpoint,
        args.positive_embedding, args.negative_embedding,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("SeedVR2 latent cache is incomplete")
    entries = manifest["entries"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = immutable_training_config(args, manifest)
    config_path = args.output_dir / "training_config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError("immutable SeedVR2 training configuration changed")
    else:
        atomic_json(config_path, config)

    bridge_args = SimpleNamespace(
        upstream_root=args.upstream_root,
        dit_checkpoint=args.dit_checkpoint,
        lora_checkpoint=None,
        vae_checkpoint=args.vae_checkpoint,
        positive_embedding=args.positive_embedding,
        negative_embedding=args.negative_embedding,
        sample_steps=1,
        cfg_scale=1.0,
        dit_dtype="bfloat16",
    )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("SeedVR2 adaptation is authorized for one GPU only")
    if torch.cuda.device_count() != 1 or args.cuda_idx != 0:
        raise RuntimeError("expected one visible GPU at cuda:0")
    runner, device = configure_runner(bridge_args)
    runner.vae = None
    gc.collect()
    torch.cuda.empty_cache()
    runner.dit.to(device=device, dtype=torch.bfloat16)
    targets = inject_seedvr2_lora(
        runner.dit,
        rank=args.rank,
        alpha=args.lora_alpha,
        last_n_blocks=args.last_n_blocks,
    )
    parameters = trainable_lora_parameters(runner.dit)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    resume_path = args.output_dir / "resume.pt"
    completed_steps = 0
    records = []
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (resume.get("format_version") != TRAINING_FORMAT_VERSION
                or resume.get("config") != config):
            raise RuntimeError("SeedVR2 resume checkpoint configuration differs")
        load_lora_state_dict(runner.dit, resume["adapter"]["state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        completed_steps = int(resume["completed_steps"])
        records = list(resume.get("recent_records", []))
    if completed_steps > args.max_steps:
        raise RuntimeError("resume step exceeds requested maximum")

    from models.dit_v2 import na

    positive = torch.load(
        args.positive_embedding, map_location=device, weights_only=True)
    positive = positive.to(device=device, dtype=torch.bfloat16)
    text, text_shape = na.flatten([positive])
    runner.dit.train()
    for module in runner.dit.modules():
        if hasattr(module, "base") and isinstance(module.base, torch.nn.Linear):
            module.base.eval()
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    loss_path = args.output_dir / "steps.jsonl"
    logged_records = truncate_step_log(loss_path, completed_steps)
    if completed_steps and len(logged_records) != completed_steps:
        raise RuntimeError("SeedVR2 step log and resume checkpoint differ")
    for step in range(completed_steps + 1, args.max_steps + 1):
        entry, rng = select_training_entry(
            entries, step, args.seed, args.uvg_probability)
        cached = torch.load(entry["path"], map_location="cpu", weights_only=True)
        clean, degraded, crop = crop_latents(
            cached["clean_latent"], cached["degraded_latent"],
            args.patch_size, rng)
        clean = clean.to(device=device, dtype=torch.bfloat16)
        degraded = degraded.to(device=device, dtype=torch.bfloat16)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + step * 97_409)
        noise = torch.randn(
            clean.shape, device=device, dtype=torch.bfloat16,
            generator=generator)
        condition = torch.empty(
            (*degraded.shape[:-1], degraded.shape[-1] + 1),
            device=device, dtype=torch.bfloat16)
        condition[..., :-1] = degraded
        condition[..., -1] = 1
        flattened_noise, latent_shape = na.flatten([noise])
        flattened_condition, condition_shape = na.flatten([condition])
        if not torch.equal(latent_shape, condition_shape):
            raise RuntimeError("noise and condition latent shapes differ")
        target_velocity = (noise - clean).reshape(-1, clean.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = runner.dit(
                vid=torch.cat([flattened_noise, flattened_condition], dim=-1),
                txt=text,
                vid_shape=latent_shape,
                txt_shape=text_shape,
                timestep=torch.full(
                    (1,), 1000.0, device=device, dtype=torch.float32),
                disable_cache=True,
            ).vid_sample
            velocity_mse = F.mse_loss(
                prediction.float(), target_velocity.float())
            predicted_clean = (
                flattened_noise.float() - prediction.float()).reshape(
                    clean.shape)
            clean_float = clean.float()
            clean_l1 = F.l1_loss(predicted_clean, clean_float)
            temporal_mse = F.mse_loss(
                predicted_clean[1:] - predicted_clean[:-1],
                clean_float[1:] - clean_float[:-1],
            )
            loss = velocity_mse + 0.1 * clean_l1 + 0.2 * temporal_mse
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite SeedVR2 loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite SeedVR2 gradient at step {step}")
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        record = {
            "step": step,
            "sample_id": entry["sample_id"],
            "dataset": entry["dataset"],
            **crop,
            "loss": float(loss.detach()),
            "velocity_mse": float(velocity_mse.detach()),
            "clean_latent_l1": float(clean_l1.detach()),
            "temporal_delta_mse": float(temporal_mse.detach()),
            "grad_norm_before_clip": float(grad_norm.detach()),
            "seconds": step_seconds,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
            "utc": utc_now(),
        }
        append_jsonl(loss_path, record)
        logged_records.append(record)
        records.append(record)
        print(json.dumps({"stage": "train-step", **record}), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_resume_checkpoint(
                resume_path, runner.dit, optimizer, step, config, records)
            if step in {100, 250, 500, 1000, args.max_steps}:
                save_lora_adapter(
                    args.output_dir / "checkpoints" / f"adapter_step_{step:06d}.pt",
                    runner.dit,
                    rank=args.rank,
                    alpha=args.lora_alpha,
                    last_n_blocks=args.last_n_blocks,
                    metadata={
                        "completed_steps": step,
                        "base_dit_sha256": config["base_dit_sha256"],
                        "cache_manifest": config["cache_manifest"],
                    },
                )
        del cached, clean, degraded, noise, condition, prediction
        del target_velocity, predicted_clean, clean_float, loss

    final_path = args.output_dir / "seedvr2_codec_lora.pt"
    adapter_info = save_lora_adapter(
        final_path,
        runner.dit,
        rank=args.rank,
        alpha=args.lora_alpha,
        last_n_blocks=args.last_n_blocks,
        metadata={
            "completed_steps": args.max_steps,
            "base_dit_sha256": config["base_dit_sha256"],
            "cache_manifest": config["cache_manifest"],
            "training_config": str(config_path.resolve()),
        },
    )
    all_records = [
        json.loads(line) for line in loss_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "experiment": "single-A800 SeedVR2 codec-artifact LoRA adaptation",
        "status": "complete",
        "completed_utc": utc_now(),
        "git_commit": git_commit(),
        "completed_steps": args.max_steps,
        "seen_dataset_steps": dict(Counter(
            record["dataset"] for record in all_records)),
        "finite_loss_count": sum(math.isfinite(record["loss"]) for record in all_records),
        "mean_step_seconds": float(np.mean([
            record["seconds"] for record in all_records])),
        "last_25_mean_loss": float(np.mean([
            record["loss"] for record in all_records[-25:]])),
        "peak_cuda_allocated_bytes": max(
            record["peak_cuda_allocated_bytes"] for record in all_records),
        "adapter": str(final_path.resolve()),
        "adapter_bytes": final_path.stat().st_size,
        "adapter_info": adapter_info,
        "target_modules": targets,
        "base_dit_frozen": True,
        "vae_not_used_during_training": True,
        "one_step_deployment_aligned": True,
        "training_config": config,
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def self_test() -> None:
    result = lora_self_test()
    clean = torch.arange(5 * 8 * 8 * 16, dtype=torch.bfloat16).reshape(5, 8, 8, 16)
    degraded = clean.clone()
    a, b, crop = crop_latents(clean, degraded, 64, random.Random(7))
    if a.shape != (5, 8, 8, 16) or not torch.equal(a, b):
        raise AssertionError("latent crop self-test failed")
    entries = [
        {"dataset": "REDS", "sample_id": "r"},
        {"dataset": "UVG", "sample_id": "u"},
    ]
    selected, _ = select_training_entry(entries, 1, 5, 1.0)
    if selected["dataset"] != "UVG":
        raise AssertionError("dataset selection self-test failed")
    print(json.dumps({
        **result,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "training_format_version": TRAINING_FORMAT_VERSION,
        "latent_crop": crop,
        "loss": "velocity MSE + 0.1 clean L1 + 0.2 temporal-delta MSE",
    }, indent=2))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    try:
        if args.command == "cache":
            cache_main(args)
        elif args.command == "train":
            train_main(args)
        else:
            self_test()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
