"""Warm receiver timing after both paired training processes have completed."""
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
    p.add_argument("--max-hours", type=float, default=2)
    args = p.parse_args()
    root = args.root
    args.output, args.command = root / "timing", "timing"
    run = Run(args)
    run.thread.start()
    configure_torch()
    reference = root.parent / "a800_feature_head_20260926"
    models = {"original": (reference / "train/final.pt", reference / "evaluation"),
              "compact": (reference / "train/final.pt", root / "compact"),
              "pad16": (root / "pad16_initial.pt", root / "padding_evaluation"),
              "train_l4": (root / "train_l4/final.pt", root / "evaluation_l4"),
              "train_l2": (root / "train_l2/final.pt", root / "evaluation_l2")}
    try:
        base_codec = codec()
        samples = json.loads((root / "comparison.json").read_text())["results"]
        rows = []
        for sample in samples:
            sid = sample["sample"]["sample_id"]
            record = args.output / f"{sid}.json"
            identities = {name: {str(p):file_hash(p) for p in (weights, directory / sid / "q1.acse")}
                          for name,(weights,directory) in models.items()}
            if record.exists():
                saved = json.loads(record.read_text())
                if saved["identities"] != identities:
                    raise RuntimeError("timing inputs changed")
                rows.append(saved)
                continue
            results = {}
            for name in ("base", *models):
                run.check()
                weights, directory = models["original" if name == "base" else name]
                stream = (directory / sid / "q1.acse").read_bytes()
                parsed = parse(stream)
                if name == "base":
                    operation = lambda:base_codec.decode(parsed.base, parsed.meta["frame_count"])
                    expected = parsed.meta["base_rgb_sha256"]
                else:
                    model = load_model(weights)
                    operation = lambda:decode_enhancement(model, weights, base_codec, stream)[0]
                    expected = json.loads((directory / sid / "q1/decode.json").read_text())["output_hash"]
                for _ in range(2):
                    operation()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                seconds = []
                for _ in range(5):
                    t = time.perf_counter()
                    output = operation()
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter()-t)
                    if frame_hash(output) != expected:
                        raise RuntimeError("timed decode changed pixels")
                results[name] = dict(seconds=seconds, median_seconds=statistics.median(seconds),
                                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                if name != "base":
                    del model
                    torch.cuda.empty_cache()
                run.update(sample=sid, implementation=name)
            row = dict(sample=sid, identities=identities, results=results)
            atomic_json(record, row)
            rows.append(row)
        atomic_json(args.output / "summary.json", dict(results=rows, seconds=time.monotonic()-run.started,
                    definition="2 warmups, 5 repetitions, base+feature+entropy+display, no loading/metrics/saving"))
        run.log_resources()
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
