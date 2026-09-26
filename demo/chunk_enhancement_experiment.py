#!/usr/bin/env python3
"""Resumable UF-feature extraction, single-layer training and real-stream evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.chunk_enhancement_codec import (
    atomic_torch, configure_torch, decode_features, decode_enhancement,
    encode_enhancement, load_model, pack_region, region_features,
)
from demo.chunk_enhancement_model import ChunkEnhancement
from demo.scalable_codec import BaseCodec, atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_experiment import resources, check_space, now, load_source
from demo.scalable_format import frame_hash, parse

RUNS = Path("/root/autodl-fs/DCVC/runs")
CACHE = RUNS / "a800_scalable_cache_20260926"
MECHANISM = RUNS / "a800_scalable_mechanism_20260926"
DEFAULT_ROOT = RUNS / "a800_chunk_enhancement_20260926"
CODE_FILES = ["demo/chunk_enhancement_model.py", "demo/chunk_enhancement_codec.py",
              "demo/scalable_format.py", "demo/compact_enhancement_format.py",
              "demo/feature_head_enhancement.py", "src/models/video_model_ht.py",
              "src/layers/layers.py", "src/utils/transforms.py",
              "demo/chunk_enhancement_experiment.py", "demo/run_chunk_enhancement.sh",
              "demo/stage_c_three_path_roi_probe.py", "src/layers/extensions/inference/dmc_hts_proxy.cpp",
              "src/layers/extensions/inference/dmc_hts_proxy.h", "src/layers/extensions/inference/bind.cpp"]


def read(path):
    return json.loads(Path(path).read_text())


def codec():
    return BaseCodec(REPO / "checkpoints/cvpr2026_image.pth.tar",
                     REPO / "checkpoints/cvpr2026_video_hts.pth.tar")


class Run:
    def __init__(self, args):
        self.root = args.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.max_seconds = args.max_hours * 3600
        self.stop = threading.Event()
        self.progress = {"mode": args.command, "completed": 0}
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.lock = (self.root / f"{args.command}.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop.set())

    def monitor(self):
        while not self.stop.wait(30):
            self.log_resources()

    def log_resources(self):
        event = dict(self.progress, **resources(), elapsed_seconds=time.monotonic()-self.started)
        with (self.root / "heartbeat.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        print(json.dumps({"heartbeat": event}), flush=True)

    def update(self, **kw):
        self.progress.update(kw)
        atomic_json(self.root / f"{self.progress['mode']}.progress.json", dict(self.progress, utc=now()))

    def check(self):
        check_space()
        if self.stop.is_set() or time.monotonic()-self.started > self.max_seconds:
            raise InterruptedError("checkpoint boundary reached; resume the same command")


def features_main(args, run):
    summary = read(CACHE / "summary.json")
    if not summary["complete"] or summary["sample_count"] != 120:
        raise RuntimeError("expected completed mixed 120-sample base cache")
    entries = summary["results"][:args.limit or None]
    base_codec = codec()
    out = args.output / "features"
    out.mkdir(exist_ok=True)
    records = []
    for i, entry in enumerate(entries):
        run.check()
        sid = entry["sample"]["sample_id"]
        original = CACHE / "samples" / sid
        target, record = out / f"{sid}.pt", out / f"{sid}.json"
        if record.exists():
            item = read(record)
            if (item["base_file_hash"] != file_hash(original / "base.acse")
                    or item["feature_hash"] != file_hash(target)):
                raise RuntimeError("feature resume validation failed")
        else:
            started = time.monotonic()
            if file_hash(original / "base.acse") != entry["artifacts"]["base.acse"]:
                raise RuntimeError("original base cache integrity failure")
            base, chunks = decode_features(base_codec, (original / "base.acse").read_bytes(),
                                          mutate_copies=True)
            if frame_hash(base) != entry["base_rgb_sha256"]:
                raise RuntimeError("cached base changed")
            atomic_torch(target, {"format": "uf_hts_F_ctx_v1", "chunks": chunks,
                                  "base_rgb_sha256": frame_hash(base)})
            item = {"sample_id": sid, "sample": entry["sample"],
                    "pair_path": str(original / "pair.npz"), "pair_hash": entry["artifacts"]["pair.npz"],
                    "base_path": str(original / "base.acse"), "base_file_hash": file_hash(original / "base.acse"),
                    "feature_path": str(target), "feature_hash": file_hash(target),
                    "feature_bytes": target.stat().st_size, "chunks": len(chunks),
                    "base_exact": True, "mutating_feature_copies_does_not_change_base": True,
                    "seconds": time.monotonic()-started}
            atomic_json(record, item)
        records.append(item)
        run.update(completed=i+1, total=len(entries), sample=sid)
        print(f"FEATURES {i+1}/{len(entries)} {sid}", flush=True)
    atomic_json(args.output / "features.json", {"samples": records,
                "dataset_counts": dict(Counter(r["sample"]["dataset"] for r in records)),
                "total_feature_bytes": sum(r["feature_bytes"] for r in records), "utc": now()})


def checkpoint(model, step, **extra):
    return {"format": model.FORMAT, "model_config": model.config,
            "model": model.export_state() if hasattr(model, "export_state") else model.state_dict(),
            "step": step, **extra}


def new_model(architecture):
    if architecture == "uf_head_pad16":
        from demo.feature_head_enhancement import Pad16FeatureHeadEnhancement
        return Pad16FeatureHeadEnhancement().cuda().train()
    if architecture == "uf_head":
        from demo.feature_head_enhancement import FeatureHeadEnhancement
        return FeatureHeadEnhancement().cuda().train()
    return ChunkEnhancement().cuda().train()


def decode_main(args, run):
    configure_torch()
    model = load_model(args.checkpoint)
    base_codec = codec()
    output, report = decode_enhancement(model, args.checkpoint, base_codec,
                                        args.stream.read_bytes(), allow_incomplete_tail=args.allow_incomplete_tail)
    atomic_npz(args.output / "reconstruction.npz", reconstruction=output)
    report.update(pid=os.getpid(), seconds=time.monotonic()-run.started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(args.output / "decode.json", report)


def fresh_decode(stream, weights, output, *, partial=False):
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__)), "decode", "--stream", str(stream),
               "--checkpoint", str(weights), "--output", str(output)]
    if partial:
        command.append("--allow-incomplete-tail")
    with (output / "process.log").open("w") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    with np.load(output / "reconstruction.npz") as data:
        frames = data["reconstruction"].copy()
    return frames, read(output / "decode.json")


def smoke_main(args, run):
    configure_torch()
    torch.manual_seed(260926)
    base_codec = codec()
    # Previously used mechanism-development data; no training data role change.
    entry = read(MECHANISM / "summary.json")["results"][2]
    sample = entry["sample"]
    original = (MECHANISM / "samples" / sample["sample_id"] / "base.acse").read_bytes()
    base, chunks = decode_features(base_codec, original, mutate_copies=True)
    source = load_source(sample)
    model = new_model(args.architecture)
    # Exercise nonzero symbols/corrections, not a vacuous zero-output roundtrip.
    # This diagnostic initialization is NOT used by formal training.
    with torch.no_grad():
        model.analysis[-1].weight.mul_(8)
        model.synthesis[-1].weight.mul_(8)
        if hasattr(model, "feature_synthesis"):
            model.feature_synthesis[-1].weight.mul_(8)
    roi = [96, 64, 96, 64]
    chunk = chunks[1]
    source_t = pack_region(source, chunk["start"], 8, roi, "cuda")
    base_t = pack_region(base, chunk["start"], 8, roi, "cuda")
    feat = region_features(chunk, roi, "cuda", getattr(model, "feature_halo", 0))
    feature_before = feat.clone()
    head_checks = {}
    if hasattr(model, "head"):
        with torch.no_grad():
            zero = torch.zeros_like(model.core(feat)[:, :512])
            np.testing.assert_array_equal(model.apply_feature_delta(base_t, feat, zero).cpu(), base_t.cpu())
        head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
        head_checks["zero_correction_exact"] = True
        with torch.no_grad():
            rendered = model.render(chunk["features"][:, :512].cuda().float()).clamp(0, 1)
            rh, rw = base.shape[1:3]
            native = pack_region(base, chunk["start"], chunk["count"], [0, 0, rw, rh], "cuda")
            error = (rendered[..., :rh, :rw]-native[..., :rh, :rw]).abs()*255
            head_checks.update(native_renderer_mean_abs_u8=error.mean().item(),
                               native_renderer_max_abs_u8=error.max().item())
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    for _ in range(32):
        prediction = model(source_t, base_t, feat)
        loss = 128 * (prediction["reconstruction"]-source_t).square().mean() + prediction["bits"]/(8*96*64)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        active = [(n, p) for n, p in model.named_parameters() if p.requires_grad
                  and not (hasattr(model, "head") and n.startswith("synthesis."))]
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in active):
            raise RuntimeError("non-finite/missing training gradient")
        optimizer.step()
    torch.testing.assert_close(feat, feature_before, rtol=0, atol=0)
    if hasattr(model, "head"):
        for k, value in model.head.state_dict().items():
            torch.testing.assert_close(value, head_before[k], rtol=0, atol=0)
        assert all(p.grad is None and not p.requires_grad for p in model.head.parameters())
        head_checks.update(frozen_head_exact=True, decoded_features_exact=True,
                           feature_synthesis_gradient_nonzero=bool(model.feature_synthesis[-1].weight.grad.abs().sum()))
    weights = args.output / "smoke.pt"
    atomic_torch(weights, checkpoint(model, 32, diagnostic_only=True))
    model = load_model(weights)
    prefix, wires, expected, costs = encode_enhancement(model, weights, original, source,
                                                       base, chunks, [roi], 1.0)
    if np.array_equal(expected, base):
        raise RuntimeError("vacuous smoke: enhancement changes no display pixels")
    payload_original, _, _ = model.compress(source_t, base_t, feat)
    payload_altered, _, _ = model.compress(1-source_t, base_t, feat)
    if payload_original == payload_altered:
        raise RuntimeError("vacuous smoke: payload does not depend on source")
    results = {}
    for name, data, target, partial in (
        ("base", prefix, base, False),
        ("complete", prefix + b"".join(wires), expected, False),
        ("missing_first", prefix + b"".join(wires[1:]), None, False),
        ("partial_last", prefix + b"".join(wires[:-1]) + wires[-1][:-7], None, True),
    ):
        stream = args.output / f"{name}.acse"
        atomic_bytes(stream, data)
        decoded, report = fresh_decode(stream, weights, args.output / name, partial=partial)
        if target is None:
            target, _ = decode_enhancement(model, weights, base_codec, data, allow_incomplete_tail=partial)
        np.testing.assert_array_equal(decoded, target)
        if name == "missing_first":
            np.testing.assert_array_equal(decoded[0], base[0])
        results[name] = report
    atomic_json(args.output / "smoke.json", {"passed": True, "architecture": args.architecture,
                "head_checks": head_checks, "data_role": "previously used development",
                "nonzero_correction": True, "source_changes_payload": True,
                "sample": sample, "feature_copy_isolation": True, "fresh_decodes": results,
                "packet_costs": costs, "parameters": sum(p.numel() for p in model.parameters()),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "seconds": time.monotonic()-run.started})


def train_main(args, run):
    configure_torch()
    manifest_path = args.feature_manifest or DEFAULT_ROOT / "features.json"
    manifest = read(manifest_path)
    if args.limit:
        # Small resumability test retains both domains; full training uses all 120.
        manifest["samples"] = [r for d in ("REDS", "UVG") for r in
                               [v for v in manifest["samples"] if v["sample"]["dataset"] == d][:args.limit]]
    config = {"seed": 260926, "manifest_sha256": file_hash(manifest_path),
              "limit_per_domain": args.limit,
              "crop_sizes": [128, 192, 256], "qsteps": [0.5, 1.0, 2.0],
              "lambdas": [256.0, 128.0, 64.0], "temporal_weight": 8.0,
              "learning_rate": args.learning_rate, "base_frozen": True, "single_enhancement_layer": True,
              "mixed_rectangles": args.mixed_rectangles,
              "initialize_sha256": file_hash(args.initialize) if args.initialize else None,
              "uvg_sampling_probability": 0.25,
              "warmup_steps": args.warmup_steps, "rate_ramp_steps": args.rate_ramp_steps,
              "initial_gain": args.initial_gain, "lambda_scale": args.lambda_scale,
              "architecture": args.architecture,
              "code_hashes": {p: file_hash(REPO / p) for p in CODE_FILES}}
    if min(args.warmup_steps, args.rate_ramp_steps) < 0 or min(args.initial_gain, args.lambda_scale, args.learning_rate) <= 0:
        raise ValueError("invalid warmup, rate ramp or initialization")
    config_path = args.output / "config.json"
    if config_path.exists() and read(config_path) != config:
        raise RuntimeError("training code/config changed; use a new run directory")
    atomic_json(config_path, config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = new_model(args.architecture)
    if args.initialize:
        if args.initial_gain != 1.0:
            raise ValueError("warm-start must not rescale pretrained parameters")
        initial = load_model(args.initialize, "cpu")
        if initial.config != model.config:
            raise ValueError("warm-start parameter architecture/head mismatch")
        model.load_export_state(initial.export_state())
        del initial
    with torch.no_grad():
        model.analysis[-1].weight.mul_(args.initial_gain)
        model.synthesis[-1].weight.mul_(args.initial_gain)
        if hasattr(model, "feature_synthesis"):
            model.feature_synthesis[-1].weight.mul_(args.initial_gain)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"])
    rng = random.Random(config["seed"])
    step = 0
    resume = args.output / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise RuntimeError("checkpoint config mismatch")
        if saved["model_config"] != model.config:
            raise RuntimeError("model architecture/frozen head changed on resume")
        if hasattr(model, "load_export_state"):
            model.load_export_state(saved["model"])
        else:
            model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["random_state"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step = saved["step"]
    def save():
        atomic_torch(resume, checkpoint(model, step, optimizer=optimizer.state_dict(), config=config,
                     random_state=rng.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        atomic_torch(args.output / f"step_{step:06d}.pt", checkpoint(model, step, config=config))
    samples = {"REDS": [], "UVG": []}
    # Approximately 6 GiB host RAM for this bounded 120-sample pilot; no dataset copy to fast disk.
    for i, record in enumerate(manifest["samples"]):
        run.check()
        if file_hash(Path(record["pair_path"])) != record["pair_hash"] or file_hash(Path(record["feature_path"])) != record["feature_hash"]:
            raise RuntimeError("training input hash mismatch")
        with np.load(record["pair_path"]) as pair:
            source, base = pair["source"].copy(), pair["base"].copy()
        features = torch.load(record["feature_path"], map_location="cpu", weights_only=False)
        samples[record["sample"]["dataset"]].append((record["sample_id"], source, base, features["chunks"]))
        run.update(phase="load_training_inputs", completed=i+1, total=len(manifest["samples"]))
    if not all(samples.values()):
        raise RuntimeError("both REDS and UVG required")
    atomic_json(args.output / "training_manifest.json", {"data_role": "training; original REDS train and approved UVG adaptation",
                "samples": manifest["samples"], "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "model_config": model.config})
    trained_steps = step
    try:
        while step < args.steps:
            run.check()
            t0 = time.monotonic()
            dataset = "UVG" if rng.random() < 0.25 else "REDS"
            sid, source, base, chunks = rng.choice(samples[dataset])
            chunk = rng.choice(chunks)
            size = rng.choice(config["crop_sizes"])
            height, width = base.shape[1:3]
            size = min(size, height//64*64, width//64*64)
            x, y = rng.randrange((width-size)//8+1)*8, rng.randrange((height-size)//8+1)*8
            rw, rh = size, size
            if config["mixed_rectangles"]:
                rw, rh = rng.choice([(128, 128), (192, 192), (256, 256),
                                      (96, 64), (64, 96), (160, 96), (96, 160)])
                rw, rh = min(rw, width//16*16), min(rh, height//16*16)
                x, y = rng.randrange((width-rw)//8+1)*8, rng.randrange((height-rh)//8+1)*8
            roi = [x, y, rw, rh]
            start, count = chunk["start"], chunk["count"]
            target = pack_region(source, start, count, roi, "cuda", model.spatial_alignment)
            bottom = pack_region(base, start, count, roi, "cuda", model.spatial_alignment)
            feat = region_features(chunk, roi, "cuda", getattr(model, "feature_halo", 0), model.spatial_alignment)
            quality = rng.randrange(3)
            qstep, weight = config["qsteps"][quality], config["lambdas"][quality]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(target, bottom, feat, qstep, count)
            ph, pw = bottom.shape[-2:]
            pred = prediction["reconstruction"].reshape(8, 3, ph, pw)[:count, :, :rh, :rw]
            truth = target.reshape(8, 3, ph, pw)[:count, :, :rh, :rw]
            mse = (pred-truth).square().mean()
            temporal = ((pred[1:]-pred[:-1])-(truth[1:]-truth[:-1])).square().mean() if count > 1 else mse*0
            bpp = prediction["bits"] / (count*rh*rw)
            rate_weight = min(1.0, max(0.0, (step-config["warmup_steps"])/max(1, config["rate_ramp_steps"])))
            if config["warmup_steps"] == 0 and config["rate_ramp_steps"] == 0:
                rate_weight = 1.0
            loss = rate_weight*bpp + weight*config["lambda_scale"]*mse + config["temporal_weight"]*temporal
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(loss) or prediction["saturated"].item():
                raise RuntimeError("invalid loss or saturated entropy symbols")
            optimizer.step()
            step += 1
            torch.cuda.synchronize()
            event = {"step": step, "dataset": dataset, "sample_id": sid, "roi": roi,
                     "start": start, "count": count, "qstep": qstep, "loss": loss.item(),
                     "bpp_estimate": bpp.item(), "psnr": -10*math.log10(max(mse.item(), 1e-12)),
                     "rate_weight": rate_weight,
                     "y_nonzero_fraction": (prediction["y_symbols"] != 0).float().mean().item(),
                     "z_nonzero_fraction": (prediction["z_symbols"] != 0).float().mean().item(),
                     "base_psnr": -10*math.log10(max((bottom.reshape(8,3,ph,pw)[:count,:,:rh,:rw]-truth).square().mean().item(), 1e-12)),
                     "temporal_mse": temporal.item(), "grad_norm": grad.item(),
                     "seconds": time.monotonic()-t0, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(event) + "\n")
            if step % 25 == 0:
                run.update(phase="train", completed=step, total=args.steps, last=event)
                print(json.dumps(event), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save()
    except BaseException:
        if step > trained_steps:
            save()
        raise
    atomic_torch(args.output / "final.pt", checkpoint(model, step, config=config))
    atomic_json(args.output / "train_summary.json", {"steps": step, "parameters": sum(p.numel() for p in model.parameters()),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "resources": resources(),
                "elapsed_seconds_this_invocation": time.monotonic()-run.started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "smoke", "train", "decode"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--rate-ramp-steps", type=int, default=0)
    parser.add_argument("--initial-gain", type=float, default=1.0)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--architecture", choices=("rgb", "uf_head", "uf_head_pad16"), default="rgb")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mixed-rectangles", action="store_true")
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--allow-incomplete-tail", action="store_true")
    args = parser.parse_args()
    run = Run(args)
    configure_torch()
    atomic_json(args.output / f"{args.command}.preflight.json", dict(resources(), torch=torch.__version__, argv=sys.argv))
    run.thread.start()
    try:
        {"features": features_main, "smoke": smoke_main, "train": train_main, "decode": decode_main}[args.command](args, run)
        run.log_resources()
        atomic_json(args.output / f"{args.command}.complete", {"utc": now(), "seconds": time.monotonic()-run.started})
    except BaseException as exc:
        atomic_json(args.output / f"{args.command}.failure.json", {"utc": now(), "error": repr(exc)})
        raise
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
