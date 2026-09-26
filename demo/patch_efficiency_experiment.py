"""Lossless framing audit and parameter-identical pad16 checkpoint migration."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.chunk_enhancement_codec import atomic_torch, configure_torch, load_model
from demo.chunk_enhancement_experiment import Run, checkpoint, fresh_decode, read
from demo.compact_enhancement_format import repack
from demo.feature_head_enhancement import Pad16FeatureHeadEnhancement
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import parse
from demo.scalable_experiment import resources


def prepare(args, run):
    original = load_model(args.reference / "train/final.pt", "cpu")
    model = Pad16FeatureHeadEnhancement(**original.config).eval()
    model.load_export_state(original.export_state())
    torch.testing.assert_close(model.export_state(), original.export_state(), rtol=0, atol=0)
    atomic_torch(args.output / "pad16_initial.pt", checkpoint(model, 20000,
                 initialization=str(args.reference / "train/final.pt"),
                 initialization_sha256=file_hash(args.reference / "train/final.pt"),
                 note="parameter-identical migration; no optimizer update"))
    atomic_json(args.output / "prepare.json", {"weights_identical": True,
                "source_sha256": file_hash(args.reference / "train/final.pt"), "resources": resources()})


def audit(args, run):
    summary = read(args.reference / "evaluation/summary.json")
    weights = args.reference / "train/final.pt"
    protocol = {"source_summary_sha256": file_hash(args.reference / "evaluation/summary.json"),
                "checkpoint_sha256": file_hash(weights), "command": "lossless-repack",
                "data_role": "previously used development", "source_pixels_read": False,
                "code_hashes": {n:file_hash(Path(__file__).parent/n) for n in
                    ("patch_efficiency_experiment.py", "compact_enhancement_format.py", "scalable_format.py")}}
    if (args.output / "protocol.json").exists() and read(args.output / "protocol.json") != protocol:
        raise RuntimeError("repack protocol changed")
    atomic_json(args.output / "protocol.json", protocol)
    rows = []
    for i, result in enumerate(summary["results"]):
        sid = result["sample"]["sample_id"]
        root = args.output / sid
        root.mkdir(exist_ok=True)
        for key in ("q2", "q1", "q0.5"):
            run.check()
            record = root / f"{key}.json"
            if record.exists():
                saved = read(record)
                if file_hash(root / f"{key}.acse") != saved["stream_sha256"]:
                    raise RuntimeError("repacked stream changed")
                rows.append(saved)
                continue
            old_path = args.reference / "evaluation" / sid / f"{key}.acse"
            old = old_path.read_bytes()
            new = repack(old)
            a, b = parse(old), parse(new)
            assert a.meta == b.meta and a.base == b.base
            assert [(p.meta,p.payload) for p in a.packets] == [(p.meta,p.payload) for p in b.packets]
            path = root / f"{key}.acse"
            atomic_bytes(path, new)
            actual, report = fresh_decode(path, weights, root / key)
            with np.load(args.reference / "evaluation" / sid / key / "reconstruction.npz") as f:
                np.testing.assert_array_equal(actual, f["reconstruction"])
            if key == "q1":
                # Dropping a complete packet, or truncating the last one, cannot
                # affect the base reference chain or other received packets.
                drop = new[:b.base_end]+b"".join(p.wire for p in b.packets[:-1])
                cut = new[:-7]
                drop_path, cut_path = root / "drop_last.acse", root / "truncated.acse"
                atomic_bytes(drop_path, drop)
                atomic_bytes(cut_path, cut)
                dropped, _ = fresh_decode(drop_path, weights, root / "drop_last")
                truncated, _ = fresh_decode(cut_path, weights, root / "truncated", partial=True)
                np.testing.assert_array_equal(dropped, truncated)
            row = dict(sample_id=sid, quality=key, old_bytes=len(old), new_bytes=path.stat().st_size,
                       saved_bytes=len(old)-len(new), base_bytes=len(b.base),
                       old_framing_bytes=len(old)-len(a.base)-sum(len(p.payload) for p in a.packets),
                       new_framing_bytes=len(new)-len(b.base)-sum(len(p.payload) for p in b.packets),
                       entropy_payload_bytes=sum(len(p.payload) for p in b.packets),
                       stream_sha256=file_hash(path), old_stream_sha256=file_hash(old_path),
                       fresh_decode=report, pixels_identical=True, payloads_identical=True,
                       roi_psnr=result["points"][key]["roi_psnr"],
                       lpips=result["points"][key]["quality"]["lpips_alex"])
            atomic_json(record, row)
            rows.append(row)
            run.update(sample=sid, quality=key, completed=len(rows), total=12)
            print(json.dumps(row), flush=True)
    atomic_json(args.output / "summary.json", dict(protocol=protocol, results=rows,
                seconds=time.monotonic()-run.started, resources=resources()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "audit"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("/root/autodl-fs/DCVC/runs/a800_feature_head_20260926"))
    p.add_argument("--max-hours", type=float, default=4)
    args = p.parse_args()
    configure_torch()
    run = Run(args)
    run.thread.start()
    try:
        {"prepare": prepare, "audit": audit}[args.command](args, run)
        atomic_json(args.output / f"{args.command}.complete", {"resources": resources()})
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
