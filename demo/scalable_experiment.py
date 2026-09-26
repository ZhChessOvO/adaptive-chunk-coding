#!/usr/bin/env python3
"""Resumable real-stream mechanism evaluation and mixed-domain base cache.

No diffusion, router, or optimizer is run. Cache pairs use QP8 bases decoded in
separate processes and original RGB targets for later enhancement training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.scalable_codec import BaseCodec, atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import (
    apply_packets, base_container, canonical_json, frame_hash, make_packet, parse, sha256,
)

PERSIST = Path("/root/autodl-fs/DCVC")
REDS_LEDGER = PERSIST / "runs/a800_pilot_20260917/pilot/sample_ledger/train_samples.jsonl"
DEV_LEDGER = PERSIST / "runs/a800_pilot_20260917/pilot/sample_ledger/development_samples.jsonl"
UVG_LEDGER = PERSIST / "data/UVG_adaptation/uvg_adaptation_samples.jsonl"
JOINT_LEDGER = PERSIST / "runs/a800_joint_evaluation_20260918/manifests/joint_samples.jsonl"
CODE_FILES = ("demo/scalable_format.py", "demo/scalable_codec.py",
              "demo/scalable_experiment.py", "demo/run_scalable_stage1.sh")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def code_fingerprint() -> dict:
    return {name: file_hash(REPO / name) for name in CODE_FILES}


def resources() -> dict:
    disks = {}
    for path in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
        usage = shutil.disk_usage(path)
        disks[path] = {"total_bytes": usage.total, "used_bytes": usage.used,
                       "free_bytes": usage.free, "used_percent": 100 * usage.used / usage.total}
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip()
    return {"utc": now(), "disks": disks, "gpu": gpu}


def check_space() -> None:
    for path in ("/root", "/root/autodl-tmp", "/root/autodl-fs"):
        usage = shutil.disk_usage(path)
        if usage.used / usage.total >= 0.80:
            raise RuntimeError(f"80% disk stop: {path}")


def load_source(sample: dict) -> np.ndarray:
    crop = sample["crop"]
    frames = []
    for path in sample["source_files"]:
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        frame = frame[crop["y"]:crop["y"] + crop["height"],
                      crop["x"]:crop["x"] + crop["width"]]
        if frame.shape != (crop["height"], crop["width"], 3):
            raise ValueError(f"crop outside source: {path}")
        frames.append(frame.copy())
    if len(frames) != sample["frame_count"]:
        raise ValueError("source frame count mismatch")
    return np.stack(frames)


def plan_main(args) -> None:
    if args.plan.exists():
        if read_json(args.plan)["code_files"] != code_fingerprint():
            raise RuntimeError("existing plan has different code; use a new run root")
        print(f"REUSE_PLAN {args.plan}", flush=True)
        return
    dev = rows(DEV_LEDGER)
    uvg = [r for r in rows(JOINT_LEDGER) if r.get("dataset") == "UVG"]
    if len(uvg) != 7:
        raise ValueError("expected seven existing UVG evaluation samples")
    selected = [copy.deepcopy(dev[0]), copy.deepcopy(uvg[0]),
                copy.deepcopy(dev[1]), copy.deepcopy(uvg[3])]
    for index, sample in enumerate(selected):
        sample["dataset"] = "REDS" if index in (0, 2) else "UVG"
        sample["source_role"] = "mechanism development diagnostic; previously used content"
        sample["sample_id"] = f"mechanism-{index:02d}-{sample['dataset'].lower()}"
        if index == 2:
            paths = sorted(Path(sample["source_dir"]).glob("*.png"))
            sample["source_files"] = [str(p) for p in paths[:33]]
            sample["frame_count"] = 33
        if index >= 2:
            sample["crop"]["width"] = 384
            sample["crop"]["height"] = 256
        # Old source digests refer to old crops/windows, not this diagnostic.
        sample.pop("selected_source_sha256", None)
    available_reds = rows(REDS_LEDGER)[:240]
    # Uniformly distributed original sequence IDs, not quality-based selection.
    train_reds = [copy.deepcopy(available_reds[int(i)])
                  for i in np.linspace(0, 239, 90, dtype=int)]
    by_sequence = {}
    for row in rows(UVG_LEDGER):
        by_sequence.setdefault(row["sequence"], []).append(row)
    if set(by_sequence) != {"Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry"}:
        raise ValueError("UVG ledger differs from approved training sequences")
    train_uvg = []
    for seq in sorted(by_sequence):
        group = by_sequence[seq]
        train_uvg.extend(copy.deepcopy(group[int(i)])
                         for i in np.linspace(0, len(group) - 1, 6, dtype=int))
    for row in train_reds:
        row["dataset"] = "REDS"
        row["source_role"] = "enhancement training cache; original REDS train"
    for row in train_uvg:
        row["dataset"] = "UVG"
        row["source_role"] = "enhancement training cache; existing UVG adaptation training"
    training = train_reds + train_uvg
    if len({r["sample_id"] for r in training}) != 120:
        raise ValueError("duplicate training sample IDs")
    plan = {
        "version": 1, "created_utc": now(), "code_files": code_fingerprint(),
        "git_commit_at_plan": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "base_qp": 8, "mechanism_qsteps": [24, 8], "haar_levels": 2,
        "mechanism_samples": selected, "cache_samples": training,
        "cache_dataset_counts": dict(Counter(r["dataset"] for r in training)),
        "source_ledgers": {str(p): file_hash(p) for p in
                           (REDS_LEDGER, DEV_LEDGER, UVG_LEDGER, JOINT_LEDGER)},
        "neural_training_run": False,
        "cache_purpose": "paired RGB for later training; no optimizer in this stage",
        "generation_enabled": False, "router_enabled": False,
    }
    atomic_json(args.plan, plan)
    print(json.dumps({"plan": str(args.plan), "mechanism_samples": 4,
                      "cache_samples": 120, "datasets": plan["cache_dataset_counts"]}), flush=True)


def fresh_decode(stream: Path, output: Path, report: Path, log: Path, *,
                 incomplete: bool = False) -> dict:
    command = [sys.executable, str(REPO / "demo/scalable_codec.py"), "decode",
               "--stream", str(stream), "--output", str(output), "--report", str(report)]
    if incomplete:
        command.append("--allow-incomplete-tail")
    with log.open("w") as handle:
        subprocess.run(command, cwd=REPO, check=True, stdout=handle,
                       stderr=subprocess.STDOUT, timeout=900)
    return read_json(report)


def load_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        return loaded[key].copy()


def quality(source: np.ndarray, output: np.ndarray, metric) -> dict:
    from demo.stage_c_three_path_roi_probe import evaluate_variant
    result = evaluate_variant(list(source), list(output), metric)
    if not math.isfinite(result["psnr_db"]):
        result["psnr_db"] = None
    return result


def make_visual(path: Path, source: np.ndarray, variants: dict, results: dict) -> None:
    index = min(8, source.shape[0] - 1)
    h, w = source.shape[1:3]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    panels = [("Source (development)", source[index])] + [
        (name, variants[name][index]) for name in ("base", "a1", "a1_b1", "two_levels", "uf32")]
    canvas = Image.new("RGB", (w * 3, (h + 72) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, pixels) in enumerate(panels):
        x, y = (i % 3) * w, (i // 3) * (h + 72)
        canvas.paste(Image.fromarray(pixels), (x, y + 72))
        draw.text((x + 6, y + 5), name, font=font, fill="black")
        if name in results:
            row = results[name]
            draw.text((x + 6, y + 27),
                      f"{row['bytes']} B | PSNR {row['quality']['psnr_db']:.3f}",
                      font=font, fill="black")
            draw.text((x + 6, y + 48), f"LPIPS {row['quality']['lpips_alex']:.4f}",
                      font=font, fill="black")
    canvas.save(path)


def validate_resumed(root: Path, entry_hash: str) -> dict | None:
    path = root / "complete.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result["entry_hash"] != entry_hash:
        raise RuntimeError(f"resume identity mismatch: {root}")
    for name, expected in result["artifacts"].items():
        artifact = root / name
        if not artifact.is_file() or file_hash(artifact) != expected:
            raise RuntimeError(f"resume artifact mismatch: {artifact}")
    return result


def mechanism_sample(sample: dict, plan: dict, root: Path, codec: BaseCodec, metric) -> dict:
    entry_hash = sha256(canonical_json({"sample": sample, "plan": plan}))
    resumed = validate_resumed(root, entry_hash)
    if resumed:
        print(f"SKIP {sample['sample_id']}", flush=True)
        return resumed
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    source = load_source(sample)
    base_bytes, encoding = codec.encode(source)
    atomic_bytes(root / "base.dcvc", base_bytes)
    base = codec.decode((root / "base.dcvc").read_bytes(), len(source))
    bottom = base_container(base_bytes, codec.metadata(source, base))
    _, height, width, _ = source.shape
    rois = [[width // 4, height // 4, width // 4, height // 4],
            [width // 2, height // 2, width // 4, height // 4]]
    count = min(17, len(source))
    q1, q2 = plan["mechanism_qsteps"]
    specs = [(0, 1, 0, q1), (1, 1, 0, q1), (0, 2, 1, q2), (1, 2, 2, q2)]
    wires, expected = [], [base]
    for packet_id, (region_id, layer, parent, qstep) in enumerate(specs, 1):
        wires.append(make_packet(source, expected[-1], packet_id=packet_id, start=0,
                                 count=count, roi=rois[region_id], layer=layer,
                                 parent_id=parent, qstep=qstep, levels=plan["haar_levels"]))
        output, _ = apply_packets(base, parse(bottom + b"".join(wires)).packets)
        expected.append(output)
    streams = [bottom + b"".join(wires[:n]) for n in range(5)]
    names = ["base", "a1", "a1_b1", "a2_b1", "two_levels"]
    variants, results, reports = {}, {}, {}
    mask = np.zeros(source.shape[:3], bool)
    for x, y, rw, rh in rois:
        mask[:count, y:y + rh, x:x + rw] = True
    for index, (name, data) in enumerate(zip(names, streams)):
        path = root / f"{name}.acse"
        atomic_bytes(path, data)
        report = fresh_decode(path, root / f"{name}.npz", root / f"{name}.decode.json",
                              root / f"{name}.decode.log")
        actual = load_npz(root / f"{name}.npz", "reconstruction")
        np.testing.assert_array_equal(actual, expected[index])
        np.testing.assert_array_equal(load_npz(root / f"{name}.npz", "base"), base)
        if index and not data.startswith(streams[index - 1]):
            raise RuntimeError("not an append-only prefix")
        np.testing.assert_array_equal(actual[~mask], base[~mask])
        if source.shape[0] > count:
            np.testing.assert_array_equal(actual[count:], base[count:])
        parsed = parse(data)
        row = {"bytes": path.stat().st_size,
               "bpp": path.stat().st_size * 8 / (len(source) * height * width),
               "quality": quality(source, actual, metric),
               "roi_mse": float(np.mean((source[mask].astype(np.float64)
                                        - actual[mask].astype(np.float64)) ** 2)),
               "base_stream_bytes": len(parsed.base),
               "global_header_bytes": parsed.base_end - len(parsed.base),
               "packet_header_bytes": sum(len(p.wire) - len(p.payload) for p in parsed.packets),
               "enhancement_payload_bytes": sum(len(p.payload) for p in parsed.packets),
               "complete_fresh_decode_seconds": report["complete_seconds"]}
        variants[name], results[name], reports[name] = actual, row, report
        print(json.dumps({"sample": sample["sample_id"], "prefix": name, **row}), flush=True)
    high_stream, _ = codec.encode(source, 32)
    atomic_bytes(root / "uf32.dcvc", high_stream)
    high_dir = root / "uf32_fresh"
    command = [sys.executable, str(REPO / "demo/stage_c_a800_scalar_fresh_decode.py"),
               "--stream", str(root / "uf32.dcvc"), "--frame-count", str(len(source)),
               "--decode-repeats", "1", "--output", str(root / "uf32.decode.json"),
               "--output-frames-dir", str(high_dir)]
    with (root / "uf32.decode.log").open("w") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=900)
    high = np.stack([np.asarray(Image.open(p).convert("RGB"))
                     for p in sorted(high_dir.glob("*.png"))])
    variants["uf32"] = high
    results["uf32"] = {"bytes": len(high_stream), "quality": quality(source, high, metric),
                       "bpp": len(high_stream) * 8 / (len(source) * height * width)}
    orphan = bottom + wires[1] + wires[2] + wires[3]
    partial = streams[2] + wires[2][:-7]
    for name, data, allow_tail in (("missing_a1", orphan, False), ("partial_a2", partial, True)):
        path = root / f"{name}.acse"
        atomic_bytes(path, data)
        report = fresh_decode(path, root / f"{name}.npz", root / f"{name}.decode.json",
                              root / f"{name}.decode.log", incomplete=allow_tail)
        actual = load_npz(root / f"{name}.npz", "reconstruction")
        target, status = apply_packets(base, parse(data, allow_incomplete_tail=allow_tail).packets)
        np.testing.assert_array_equal(actual, target)
        np.testing.assert_array_equal(load_npz(root / f"{name}.npz", "base"), base)
        if name == "partial_a2":
            np.testing.assert_array_equal(actual, expected[2])
        else:
            if status["applied_packet_ids"] != [2, 4]:
                raise RuntimeError("missing-parent handling lost an independent region")
            x, y, rw, rh = rois[0]
            np.testing.assert_array_equal(actual[:count, y:y + rh, x:x + rw],
                                          base[:count, y:y + rh, x:x + rw])
        reports[name] = report
    make_visual(root / "fixed_frame.png", source, variants, results)
    import torch
    result = {"sample": sample, "entry_hash": entry_hash, "variants": results,
              "rois": rois, "enhanced_frame_count": count, "encoding": encoding,
              "source_rgb_sha256": frame_hash(source), "decode_reports": reports,
              "prefix_and_fresh_decode_passed": True, "non_enhanced_pixels_exact": True,
              "base_reference_chain_unchanged": True, "missing_and_partial_packets_passed": True,
              "seconds": time.perf_counter() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "resources": resources(),
              "artifacts": {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*")
                            if p.is_file() and not p.name.endswith(".tmp") and p.name != "complete.json"}}
    atomic_json(root / "complete.json", result)
    return result


def cache_sample(sample: dict, plan: dict, root: Path, codec: BaseCodec) -> dict:
    entry_hash = sha256(canonical_json({"sample": sample, "plan": plan}))
    resumed = validate_resumed(root, entry_hash)
    if resumed:
        print(f"SKIP {sample['sample_id']}", flush=True)
        return resumed
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = load_source(sample)
    stream, encoding = codec.encode(source, 8)
    base = codec.decode(stream, len(source))
    path = root / "base.acse"
    atomic_bytes(path, base_container(stream, codec.metadata(source, base)))
    report = fresh_decode(path, root / "fresh.npz", root / "decode.json", root / "decode.log")
    fresh = load_npz(root / "fresh.npz", "base")
    np.testing.assert_array_equal(fresh, base)
    atomic_npz(root / "pair.npz", source=source, base=fresh)
    np.testing.assert_array_equal(load_npz(root / "pair.npz", "base"), fresh)
    np.testing.assert_array_equal(load_npz(root / "pair.npz", "source"), source)
    # Only remove this redundant reproducible output, after verifying pair.npz.
    (root / "fresh.npz").unlink()
    result = {"sample": sample, "entry_hash": entry_hash, "encoding": encoding,
              "source_rgb_sha256": frame_hash(source), "base_rgb_sha256": frame_hash(fresh),
              "shape": list(source.shape), "decode_report": report,
              "seconds": time.perf_counter() - started, "cache_bytes": (root / "pair.npz").stat().st_size,
              "artifacts": {name: file_hash(root / name)
                            for name in ("pair.npz", "base.acse", "decode.json")}}
    atomic_json(root / "complete.json", result)
    return result


def run_main(args) -> None:
    plan = read_json(args.plan)
    if plan["code_files"] != code_fingerprint():
        raise RuntimeError("code differs from recorded plan")
    check_space()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch
    atomic_json(output / "preflight.json", {**resources(), "torch": torch.__version__,
                                           "numpy": np.__version__, "python": sys.version})
    stopped = threading.Event()
    started = time.perf_counter()
    progress = {"completed": 0, "total": 0}

    def monitor():
        while not stopped.wait(30):
            event = {**resources(), **progress, "elapsed_seconds": time.perf_counter() - started}
            with (output / "heartbeat.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps({"heartbeat": event}), flush=True)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        codec = BaseCodec(REPO / "checkpoints/cvpr2026_image.pth.tar",
                          REPO / "checkpoints/cvpr2026_video_hts.pth.tar")
        samples = plan["mechanism_samples"] if args.mode == "mechanism" else plan["cache_samples"]
        if args.limit:
            samples = samples[:args.limit]
        progress["total"] = len(samples)
        metric = None
        if args.mode == "mechanism":
            from demo.stage_c_three_path_roi_probe import LPIPSAlex
            metric = LPIPSAlex(True)
        results = []
        for sample in samples:
            check_space()
            if time.perf_counter() - started >= args.max_hours * 3600:
                raise RuntimeError("wall-time limit reached; resume the same command")
            root = output / "samples" / sample["sample_id"]
            print(f"START {args.mode} {sample['sample_id']}", flush=True)
            if args.mode == "mechanism":
                result = mechanism_sample(sample, plan, root, codec, metric)
            else:
                result = cache_sample(sample, plan, root, codec)
            results.append(result)
            progress["completed"] += 1
            atomic_json(output / "progress.json", {**progress, "utc": now(),
                                                  "last_sample": sample["sample_id"]})
            print(json.dumps({"complete": sample["sample_id"], **progress,
                              "sample_seconds": result["seconds"]}), flush=True)
        summary = {"mode": args.mode, "complete": True, "sample_count": len(samples),
                   "neural_training_run": False, "plan_sha256": file_hash(args.plan),
                   "elapsed_seconds_this_invocation": time.perf_counter() - started,
                   "results": results, "resources": resources(),
                   "actual_output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "run.complete", {"utc": now(), "samples": len(samples)})
        print(f"COMPLETE {args.mode} {len(samples)}", flush=True)
    except BaseException as exc:
        atomic_json(output / "last_failure.json", {"utc": now(), "type": type(exc).__name__,
                                                 "message": str(exc)})
        raise
    finally:
        stopped.set()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--mode", choices=("mechanism", "cache"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--max-hours", type=float, default=8)
    args = parser.parse_args()
    if args.command == "plan":
        plan_main(args)
    else:
        run_main(args)


if __name__ == "__main__":
    main()
