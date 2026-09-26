#!/usr/bin/env python3
"""Fixed development comparison; not an independent test or a BD-rate claim."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.chunk_enhancement_codec import (
    configure_torch, decode_features, encode_enhancement, load_model,
    pack_region, region_features, unpack_region,
)
from demo.chunk_enhancement_experiment import Run, MECHANISM, codec, fresh_decode, read
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_experiment import quality, load_source, now
from demo.scalable_format import parse
from demo.stage_c_three_path_roi_probe import LPIPSAlex


def roi_psnr(source, output, rois, count=17):
    squared, n = 0.0, 0
    for x, y, w, h in rois:
        diff = (source[:count, y:y+h, x:x+w].astype(np.float64)
                -output[:count, y:y+h, x:x+w].astype(np.float64))
        squared += np.square(diff).sum()
        n += diff.size
    return 10 * math.log10(255**2 / max(squared/n, 1e-12))


@torch.no_grad()
def information_diagnostic(model, source, base, chunks, rois):
    """No-source restoration diagnostic, NOT a separately trained RD baseline.

    Zero discrete y/z symbols, retain the same learned conditional mean/synthesis.
    This exposes improvements already predictable from the base. The real packet
    reconstruction is measured separately. No zero-byte claim is made for control.
    """
    output = base.copy()
    counts = {"y_symbols": 0, "nonzero_y": 0, "z_symbols": 0, "nonzero_z": 0}
    for chunk in chunks:
        start, count = chunk["start"], chunk["count"]
        for roi in rois:
            bottom = pack_region(base, start, count, roi, "cuda")
            target = pack_region(source, start, count, roi, "cuda")
            features = region_features(chunk, roi, "cuda")
            result = model(target, bottom, features, 1.0, count)
            for kind in ("y", "z"):
                symbols = result[f"{kind}_symbols"]
                counts[f"{kind}_symbols"] += symbols.numel()
                counts[f"nonzero_{kind}"] += torch.count_nonzero(symbols).item()
            c = model.condition(bottom, features, 1.0, count)
            z = torch.zeros_like(model.scales_z(c))
            mean, _ = model.prior_y(c, z)
            reconstructed = model.reconstruct(bottom, c, mean, 1.0)
            x, y, w, h = roi
            output[start:start+count, y:y+h, x:x+w] = unpack_region(reconstructed, count, roi)
    return output, counts


def visualize(path, source, images, points, rois):
    names = ["source", "base", "q0.5", "q1", "q2", "uf32"]
    index = min(8, len(source)-1)
    h, w = source.shape[1:3]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    canvas = Image.new("RGB", (w*3, (h+66)*2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(names):
        x, y = (i%3)*w, (i//3)*(h+66)
        frame = source[index] if name == "source" else images[name][index]
        canvas.paste(Image.fromarray(frame), (x, y+66))
        draw.text((x+6, y+4), name, font=font, fill="black")
        if name in points:
            p = points[name]
            draw.text((x+6, y+24), f"{p['bytes']} B; ROI PSNR {p['roi_psnr']:.2f}", font=font, fill="black")
            draw.text((x+6, y+44), f"LPIPS {p['quality']['lpips_alex']:.4f}", font=font, fill="black")
        for rx, ry, rw, rh in rois:
            draw.rectangle((x+rx, y+66+ry, x+rx+rw-1, y+66+ry+rh-1), outline="#00c070", width=1)
    canvas.save(path)


def evaluate(args, run):
    configure_torch()
    model = load_model(args.checkpoint)
    base_codec, metric = codec(), LPIPSAlex(True)
    previous = read(MECHANISM / "summary.json")["results"]
    protocol = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_hash(args.checkpoint),
                "evaluation_code_sha256": file_hash(Path(__file__)), "qsteps": [0.5, 1.0, 2.0],
                "samples": [r["sample"] for r in previous], "data_role": "previously used mechanism development",
                "enhance_first_frames": 17, "generate": False, "router": False,
                "same_rois_as_haar": True, "separate_q_encodings_are_not_prefixes": True}
    if (args.output / "protocol.json").exists() and read(args.output / "protocol.json") != protocol:
        raise RuntimeError("evaluation protocol changed; use another output directory")
    atomic_json(args.output / "protocol.json", protocol)
    rows = []
    for number, entry in enumerate(previous):
        run.check()
        sample = entry["sample"]
        sid = sample["sample_id"]
        root = args.output / sid
        root.mkdir(exist_ok=True)
        if (root / "result.json").exists():
            result = read(root / "result.json")
            if result["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
                raise RuntimeError("resume checkpoint changed")
            for relative, digest in result["artifacts"].items():
                if file_hash(root / relative) != digest:
                    raise RuntimeError("evaluation resume artifact mismatch")
            rows.append(result)
            continue
        historical = MECHANISM / "samples" / sid
        original = (historical / "base.acse").read_bytes()
        source = load_source(sample)
        base, chunks = decode_features(base_codec, original)
        chunks = [c for c in chunks if c["start"] < 17]
        rois = entry["rois"]
        uf32 = np.stack([np.asarray(Image.open(p).convert("RGB"))
                         for p in sorted((historical / "uf32_fresh").glob("*.png"))])
        points = {"base": {"bytes": len(parse(original).base), "roi_psnr": roi_psnr(source, base, rois),
                            "quality": quality(source, base, metric)},
                  "uf32": {"bytes": entry["variants"]["uf32"]["bytes"],
                            "roi_psnr": roi_psnr(source, uf32, rois), "quality": quality(source, uf32, metric)}}
        images = {"base": base, "uf32": uf32}
        for name in ("a1_b1", "two_levels"):
            with np.load(historical / f"{name}.npz") as stored:
                pixels = stored["reconstruction"].copy()
            points[name] = {"bytes": entry["variants"][name]["bytes"],
                            "roi_psnr": roi_psnr(source, pixels, rois), "quality": entry["variants"][name]["quality"]}
        for q in protocol["qsteps"]:
            name = f"q{q:g}"
            start = time.monotonic()
            prefix, wires, expected, details = encode_enhancement(
                model, args.checkpoint, original, source, base, chunks, rois, q)
            torch.cuda.synchronize()
            encode_seconds = time.monotonic()-start
            stream = root / f"{name}.acse"
            atomic_bytes(stream, prefix+b"".join(wires))
            actual, report = fresh_decode(stream, args.checkpoint, root / name)
            np.testing.assert_array_equal(actual, expected)
            points[name] = {"bytes": stream.stat().st_size, "enhancement_bytes": sum(len(w) for w in wires),
                            "roi_psnr": roi_psnr(source, actual, rois), "quality": quality(source, actual, metric),
                            "packet_details": details, "fresh_decode": report, "encoding_seconds_excluding_base": encode_seconds}
            images[name] = actual
            if q == 1.0:
                # One fixed encoding, base prefix -> a subset -> all selected regions.
                subset = root / "prefix_first_packet.acse"
                atomic_bytes(subset, prefix+wires[0])
                sub_frames, sub_report = fresh_decode(subset, args.checkpoint, root / "prefix_first_packet")
                points["prefix_first_packet"] = {"bytes": subset.stat().st_size,
                    "roi_psnr": roi_psnr(source, sub_frames, rois), "quality": quality(source, sub_frames, metric),
                    "fresh_decode": sub_report, "prefix_of": "q1"}
                if not stream.read_bytes().startswith(subset.read_bytes()):
                    raise RuntimeError("not a true byte prefix")
            print(json.dumps({"sample": sid, "quality": name, "bytes": points[name]["bytes"],
                              "roi_psnr": points[name]["roi_psnr"]}), flush=True)
        prior_only, symbols = information_diagnostic(model, source, base, chunks, rois)
        diagnostic = {"kind": "zero_discrete_symbols_same_trained_decoder",
                      "not_separately_trained_ablation": True, "not_an_rd_point": True,
                      "roi_psnr": roi_psnr(source, prior_only, rois),
                      "quality": quality(source, prior_only, metric), "symbols": symbols,
                      "q1_vs_prior_roi_psnr": points["q1"]["roi_psnr"]-roi_psnr(source, prior_only, rois),
                      "q1_pixels_different_from_prior": int(np.count_nonzero(images["q1"] != prior_only))}
        visualize(root / "fixed_frame.png", source, images, points, rois)
        result = {"sample": sample, "rois": rois, "points": points,
                  "information_diagnostic": diagnostic,
                  "checkpoint_sha256": protocol["checkpoint_sha256"], "fresh_decode_exact": True,
                  "artifacts": {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}}
        atomic_json(root / "result.json", result)
        rows.append(result)
        run.update(completed=number+1, total=4, sample=sid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, row in zip(axes.ravel(), rows):
        p = row["points"]
        for label, keys, marker in (("UF native (2 references)", ["base", "uf32"], "o"),
                                    ("Haar prototype (prefixes)", ["a1_b1", "two_levels"], "s"),
                                    ("Single-layer neural (3 encodings)", ["q2", "q1", "q0.5"], "^")):
            pts = sorted([p[k] for k in keys], key=lambda x: x["bytes"])
            ax.plot([v["bytes"]/1000 for v in pts], [v["roi_psnr"] for v in pts], marker=marker, label=label)
        ax.set_xscale("log")
        ax.set_title(row["sample"]["sample_id"])
        ax.set_xlabel("Total on-disk kB (log scale)")
        ax.set_ylabel("Selected-region PSNR (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Development diagnostic: same ROIs / first 17 frames, not matched whole-frame quality")
    fig.tight_layout()
    fig.savefig(args.output / "rd_diagnostic.png", dpi=160)
    atomic_json(args.output / "summary.json", {"protocol": protocol, "results": rows,
                "seconds": time.monotonic()-run.started, "utc": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    args.command = "evaluate"
    run = Run(args)
    run.thread.start()
    try:
        evaluate(args, run)
        run.log_resources()
        atomic_json(args.output / "evaluate.complete", {"utc": now()})
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.lock.close()


if __name__ == "__main__":
    main()
