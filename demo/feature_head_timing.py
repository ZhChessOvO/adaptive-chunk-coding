"""Warm end-to-end receiver timings on already measured actual streams.

Includes base decoding, feature export, entropy decoding and display assembly.
Excludes model/process setup, source access, metrics and output-file compression.
Do not add these values to the independent fresh-decode process wall times.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.chunk_enhancement_codec import configure_torch, decode_enhancement, load_model
from demo.chunk_enhancement_experiment import codec, Run
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash, parse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path(
        "/root/autodl-fs/DCVC/runs/a800_chunk_enhancement_20260926"))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-hours", type=float, default=2)
    args = p.parse_args()
    if args.repeats < 3:
        raise ValueError("at least three measured repetitions required")
    args.output, args.command = args.root / "timing", "timing"
    run = Run(args)
    run.thread.start()
    try:
        configure_torch()
        base_codec = codec()
        rows = json.loads((args.root / "evaluation/summary.json").read_text())["results"]
        for row in rows:
            sid = row["sample"]["sample_id"]
            record = args.output / f"{sid}.json"
            paths = {"rgb": (args.reference / "train_warmup/final.pt", args.reference / "evaluation_warmup" / sid / "q1.acse"),
                     "head": (args.root / "train/final.pt", args.root / "evaluation" / sid / "q1.acse")}
            identities = {name: {str(p): file_hash(p) for p in pair} for name, pair in paths.items()}
            if record.exists():
                saved = json.loads(record.read_text())
                if saved["identities"] != identities or saved["repeats"] != args.repeats:
                    raise ValueError("timing resume inputs changed")
                continue
            results = {}
            stream = paths["head"][1].read_bytes()
            parsed = parse(stream)
            # Each mode gets fresh warm-up; only one enhancement model is kept
            # on the GPU at a time so the memory measurements are interpretable.
            for mode in ("base", "rgb", "head"):
                run.check()
                if mode == "base":
                    operation = lambda: base_codec.decode(parsed.base, parsed.meta["frame_count"])
                    expected = parsed.meta["base_rgb_sha256"]
                else:
                    weights, stream_path = paths[mode]
                    model = load_model(weights)
                    stream = stream_path.read_bytes()
                    operation = lambda: decode_enhancement(model, weights, base_codec, stream)[0]
                    result_path = stream_path.parent / "q1/decode.json"
                    expected = json.loads(result_path.read_text())["output_hash"]
                for _ in range(2):
                    frame_hash(operation())
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                durations = []
                for _ in range(args.repeats):
                    run.check()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = operation()
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter()-start)
                    if frame_hash(output) != expected:
                        raise RuntimeError("timed decoding changed reconstruction")
                results[mode] = {"seconds": durations, "median_seconds": statistics.median(durations),
                                 "min_seconds": min(durations), "max_seconds": max(durations),
                                 "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "frames": parsed.meta["frame_count"], "enhance_first_frames": 17 if mode != "base" else 0}
                if mode != "base":
                    del model
                    torch.cuda.empty_cache()
                run.update(sample=sid, implementation=mode)
            atomic_json(record, {"sample": sid, "identities": identities, "repeats": args.repeats,
                                "includes_setup": False, "source_access": False, "results": results})
            print(json.dumps({"sample": sid, "median_seconds": {k: v["median_seconds"] for k, v in results.items()}}), flush=True)
        atomic_json(args.output / "summary.json", {
            "definition": __doc__, "results": [json.loads((args.output / f"{r['sample']['sample_id']}.json").read_text()) for r in rows],
            "elapsed_seconds": time.monotonic()-run.started})
        run.log_resources()
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
