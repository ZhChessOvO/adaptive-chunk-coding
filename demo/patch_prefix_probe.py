"""Audit paired training, then measure every boundary of one unchanged stream.

This is a four-clip development probe, not a trained router. Ground truth is
used only by the evaluator to label packet benefit, never by fresh decoders.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.audit_chunk_resume import same
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation, roi_psnr
from demo.chunk_enhancement_experiment import Run, fresh_decode, read, MECHANISM
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_experiment import load_source, now, quality, resources
from demo.scalable_format import frame_hash, parse
from demo.stage_c_three_path_roi_probe import LPIPSAlex


def load_frames(path, key="reconstruction"):
    with np.load(path, allow_pickle=False) as data:
        return data[key].copy()


def verify_artifacts(root, artifacts):
    for relative, digest in artifacts.items():
        if file_hash(root / relative) != digest:
            raise RuntimeError(f"artifact changed: {root / relative}")


def audit_training(root):
    """Check actual exports/resume tensors and hashes, not just exit markers."""
    if not (root / "pipeline.complete").exists():
        raise RuntimeError("paired pipeline not complete")
    comparison = read(root / "comparison.json")
    for path, digest in comparison["sources"].items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError("comparison input changed")
    summaries, schedules = {}, []
    for weight in (4, 2):
        name = f"train_l{weight}"
        directory = root / name
        if not (directory / "train.complete").exists():
            raise RuntimeError("training not complete")
        states = [torch.load(directory / f, map_location="cpu", weights_only=False)
                  for f in ("final.pt", "step_020000.pt", "resume.pt")]
        a, b, resumed = states
        if not all(c["step"] == 20000 for c in states):
            raise RuntimeError("wrong final step")
        for field in ("model", "config", "format", "model_config"):
            if not same(a[field], b[field]) or not same(a[field], resumed[field]):
                raise RuntimeError(f"export/resume mismatch: {field}")
        for field in ("optimizer", "random_state", "torch_rng", "cuda_rng"):
            if field not in resumed:
                raise RuntimeError("incomplete resume state")
        records = {r["step"]: r for r in
                   (json.loads(s) for s in (directory / "metrics.jsonl").read_text().splitlines())}
        if set(records) != set(range(1, 20001)):
            raise RuntimeError("missing training steps")
        schedules.append(records)
        evaluation = root / f"evaluation_l{weight}"
        if not (evaluation / "evaluate.complete").exists():
            raise RuntimeError("evaluation not complete")
        summary = read(evaluation / "summary.json")
        if summary["protocol"]["checkpoint_sha256"] != file_hash(directory / "final.pt"):
            raise RuntimeError("evaluated a different checkpoint")
        checked = 0
        for row in summary["results"]:
            verify_artifacts(evaluation / row["sample"]["sample_id"], row["artifacts"])
            checked += len(row["artifacts"])
            if not row["fresh_decode_exact"]:
                raise RuntimeError("fresh decode failed")
            for key in ("q0.5", "q1", "q2", "prefix_first_packet"):
                p = row["points"][key]
                wire = (evaluation / row["sample"]["sample_id"] / f"{key}.acse").read_bytes()
                decoded = p["fresh_decode"]
                if (len(wire) != p["bytes"] or decoded["total_bytes"] != len(wire)
                        or decoded["source_frames_read"] or not decoded["non_enhanced_exact"]):
                    raise RuntimeError("invalid byte/source/reference accounting")
        summaries[name] = dict(steps=20000, export_resume_exact=True, checked_artifacts=checked,
            fresh_decodes=16, checkpoint_sha256=file_hash(directory / "final.pt"),
            training=read(directory / "train_summary.json"),
            evaluation_active_seconds=summary["active_seconds"])
        del states, a, b, resumed
    fields = ("step", "dataset", "sample_id", "roi", "start", "count", "qstep")
    if not all(all(schedules[0][i][k] == schedules[1][i][k] for k in fields) for i in schedules[0]):
        raise RuntimeError("paired sample/crop/quality schedules differ")
    timings = read(root / "timing/summary.json")
    for row in timings["results"]:
        for identities in row["identities"].values():
            for path, digest in identities.items():
                if file_hash(Path(path)) != digest:
                    raise RuntimeError("timing input changed")
    return dict(passed=True, utc=now(), training=summaries, same_20000_step_schedule=True,
                timing_inputs_verified=True, resources=resources())


def packet_region(packet):
    start, count = packet.meta["start"], packet.meta["count"]
    x, y, w, h = packet.meta["roi"]
    return np.s_[start:start+count, y:y+h, x:x+w]


def mse(source, output):
    return float(np.mean((source.astype(np.float64)-output.astype(np.float64))**2))


def diagnostic_label(source, base, full, packet):
    region = packet_region(packet)
    before, after = mse(source[region], base[region]), mse(source[region], full[region])
    size = source[region].size
    return dict(packet=packet.meta, packet_bytes=len(packet.wire),
                mse_before=before, mse_after=after,
                local_psnr_gain_db=10*math.log10(max(before, 1e-12)/max(after, 1e-12)),
                squared_error_reduction=(before-after)*size,
                squared_error_reduction_per_byte=(before-after)*size/len(packet.wire),
                label_requires_source=True)


def probe(args, run):
    configure_torch()
    checkpoint = args.root / "train_l4/final.pt"
    evaluation = args.root / "evaluation_l4"
    previous = read(evaluation / "summary.json")
    identities = {str(p): file_hash(p) for p in (checkpoint, evaluation / "summary.json", Path(__file__))}
    protocol = dict(inputs=identities, data_role="previously used development, four fixed clips",
        checkpoint_selection="A: same RD weight, compact pad16; B retained as rate-focused comparison",
        samples=[r["sample"] for r in previous["results"]],
        qstep=1.0, order="unchanged original stream: time chunk, then fixed ROI order",
        generation=False, learned_router=False, source_used_only_for_metrics=True,
        every_point_is_literal_prefix=True, include_base_container_header=True)
    manifest = args.output / "protocol.json"
    if manifest.exists() and read(manifest) != protocol:
        raise RuntimeError("protocol changed; use a new output directory")
    atomic_json(manifest, protocol)
    atomic_json(args.output / "completion_audit.json", audit_training(args.root))
    metric = LPIPSAlex(True)
    rows = []
    for i, reference in enumerate(previous["results"]):
        run.check()
        sid = reference["sample"]["sample_id"]
        root = args.output / sid
        root.mkdir(exist_ok=True)
        wire = (evaluation / sid / "q1.acse").read_bytes()
        parsed = parse(wire)
        source = load_source(reference["sample"])
        base = load_frames(MECHANISM / "samples" / sid / "base.npz", "base")
        full = load_frames(evaluation / sid / "q1/reconstruction.npz")
        if frame_hash(base) != parsed.meta["base_rgb_sha256"]:
            raise RuntimeError("cached base mismatch")
        if frame_hash(full) != reference["points"]["q1"]["fresh_decode"]["output_hash"]:
            raise RuntimeError("full output mismatch")
        expected = base.copy()
        boundaries = [parsed.base_end] + [p.end_offset for p in parsed.packets]
        points, packet_labels = [], []
        for number, end in enumerate(boundaries):
            run.check()
            if number:
                packet = parsed.packets[number-1]
                region = packet_region(packet)
                expected[region] = full[region]
                packet_labels.append(diagnostic_label(source, base, full, packet))
            stream = root / f"prefix_{number}.acse"
            prefix = wire[:end]
            if stream.exists() and stream.read_bytes() != prefix:
                raise RuntimeError("saved prefix changed")
            atomic_bytes(stream, prefix)
            destination = root / f"prefix_{number}"
            record = destination / "point.json"
            if record.exists():
                point = read(record)
                verify_artifacts(destination, point["artifacts"])
                actual = load_frames(destination / "reconstruction.npz")
                report = read(destination / "decode.json")
            else:
                actual, report = fresh_decode(stream, checkpoint, destination)
                point = dict(packet_count=number, bytes=end, roi_psnr=roi_psnr(source, actual, reference["rois"]),
                    quality=quality(source, actual, metric), fresh_decode=report,
                    artifacts={str(p.relative_to(destination)): file_hash(p)
                               for p in destination.rglob("*") if p.is_file()})
            np.testing.assert_array_equal(actual, expected)
            if (report["source_frames_read"] or report["base_hash"] != frame_hash(base)
                    or report["output_hash"] != frame_hash(actual) or report["total_bytes"] != end
                    or report["applied_packets"] != [p.meta["packet_id"] for p in parsed.packets[:number]]):
                raise RuntimeError("prefix fresh decode validation failed")
            point["literal_prefix_sha256"] = file_hash(stream)
            point["matches_base_plus_exact_received_regions"] = True
            atomic_json(record, point)
            points.append(point)
            run.update(completed=i*len(boundaries)+number+1, total=4*len(boundaries), sample=sid, packets=number)
            print(json.dumps(dict(sample=sid, packets=number, bytes=end, roi_psnr=point["roi_psnr"])), flush=True)
        rows.append(dict(sample=reference["sample"], rois=reference["rois"], points=points,
                         packet_labels=packet_labels, full_stream_sha256=file_hash(evaluation / sid / "q1.acse")))
    for field, ylabel, name in (("roi_psnr", "Fixed 2-ROI PSNR (dB)", "prefix_roi_rd.png"),
                                 ("lpips_alex", "Whole-frame LPIPS (lower is better)", "prefix_lpips_rd.png")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, row in zip(axes.flat, rows):
            pts = row["points"]
            values = [p[field] if field == "roi_psnr" else p["quality"][field] for p in pts]
            ax.plot([p["bytes"]/1000 for p in pts], values, "o-")
            for p, value in zip(pts, values):
                ax.annotate(str(p["packet_count"]), (p["bytes"]/1000, value), xytext=(4, 5), textcoords="offset points")
            ax.set(title=row["sample"]["sample_id"], xlabel="Complete prefix kB, including 190 B header", ylabel=ylabel)
            ax.grid(alpha=.25)
        fig.suptitle("One q=1 encoding, 0..6 appended packets; A; fixed order, not a trained router")
        fig.tight_layout()
        fig.savefig(args.output / name, dpi=150)
        plt.close(fig)
    summary = dict(protocol=protocol, results=rows, seconds=time.monotonic()-run.started,
        all_28_prefix_decodes_exact=True, resources=resources(),
        packet_benefits_are_diagnostic_labels_not_deployed_predictions=True)
    atomic_json(args.output / "summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/root/autodl-fs/DCVC/runs/a800_patch_efficiency_20260926"))
    p.add_argument("--output", type=Path, default=Path("/root/autodl-fs/DCVC/runs/a800_patch_prefix_20260926"))
    p.add_argument("--max-hours", type=float, default=2)
    args = p.parse_args()
    args.command = "prefix"
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            probe(args, run)
        run.log_resources()
        atomic_json(args.output / "prefix.complete", dict(utc=now()))
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
