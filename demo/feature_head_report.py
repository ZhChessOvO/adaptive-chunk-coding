"""Compare measured RGB/UF-head enhancement candidates on identical dev clips."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import load_source


def read(path):
    return json.loads(path.read_text())


def temporal_parts(source, decoded, rois):
    """Separate the retained I-frame branch from the new P8 feature branch."""
    out = {}
    for name, frames in (("I", slice(0, 1)), ("P8x2", slice(1, 17))):
        errors = []
        for x, y, w, h in rois:
            a, b = source[frames, y:y+h, x:x+w], decoded[frames, y:y+h, x:x+w]
            errors.append((a.astype(np.float64)-b.astype(np.float64)).reshape(-1))
        mse = np.mean(np.concatenate(errors)**2)
        out[name] = -10*np.log10(max(mse, 1e-12)/255**2)
    return out


def boundary_panel(path, frames, roi):
    """Include unmodified neighbors; do not draw over the actual patch seam."""
    x, y, w, h = roi
    margin, zoom = 16, 3
    height, width = next(iter(frames.values())).shape[:2]
    box = max(0, x-margin), max(0, y-margin), min(width, x+w+margin), min(height, y+h+margin)
    tile_w = max((box[2]-box[0])*zoom, 280)
    tile_h = (box[3]-box[1])*zoom
    canvas = Image.new("RGB", (tile_w*len(frames), tile_h+52), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for i, (name, frame) in enumerate(frames.items()):
        crop = Image.fromarray(frame).crop(box)
        canvas.paste(crop.resize((crop.width*zoom, crop.height*zoom), Image.Resampling.NEAREST), (i*tile_w, 52))
        draw.text((i*tile_w+5, 5), name, fill="black", font=font)
        draw.text((i*tile_w+5, 27), "Frame 8; +16px context; no overlay", fill="black", font=font)
    canvas.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path(
        "/root/autodl-fs/DCVC/runs/a800_chunk_enhancement_20260926"))
    args = p.parse_args()
    old_config = read(args.reference / "train_warmup/config.json")
    new_config = read(args.root / "train/config.json")
    recipe_fields = [k for k in old_config if k != "code_hashes"]
    if any(old_config[k] != new_config[k] for k in recipe_fields):
        raise ValueError("training recipes differ; label the comparison explicitly")
    old_steps = [json.loads(v) for v in (args.reference / "train_warmup/metrics.jsonl").read_text().splitlines()]
    new_steps = [json.loads(v) for v in (args.root / "train/metrics.jsonl").read_text().splitlines()]
    choices = ("step", "dataset", "sample_id", "roi", "start", "count", "qstep")
    same_choices = len(old_steps) == len(new_steps) and all(
        all(a[k] == b[k] for k in choices) for a, b in zip(old_steps, new_steps))
    if not same_choices:
        raise ValueError("training sample/crop/quality schedules differ")
    new_path, old_path = args.root / "evaluation/summary.json", args.reference / "evaluation_warmup/summary.json"
    new, old = read(new_path), read(old_path)
    extra_path = args.reference / "uf_intermediate/summary.json"
    extras = {r["sample"]["sample_id"]: r for r in read(extra_path)["results"]}
    old_by_id = {r["sample"]["sample_id"]: r for r in old["results"]}
    qkeys = ["q2", "q1", "q0.5"]
    rows = []
    for n in new["results"]:
        sid = n["sample"]["sample_id"]
        o, extra = old_by_id[sid], extras[sid]
        assert n["sample"] == o["sample"] == extra["sample"] and n["rois"] == o["rois"]
        for key in qkeys:
            assert n["points"][key]["fresh_decode"]["base_hash"] == o["points"][key]["fresh_decode"]["base_hash"]
        uf = {k: n["points"][k] for k in ("base", "uf32")}
        uf.update(extra["points"])
        source = load_source(n["sample"])
        parts = {}
        frames = {"Source": source[8]}
        historical = Path("/root/autodl-fs/DCVC/runs/a800_scalable_mechanism_20260926/samples") / sid
        with np.load(historical / "base.npz") as cache:
            frames["UF8 base"] = cache["base"][8]
        for name, root, result in (("rgb", args.reference / "evaluation_warmup", o),
                                   ("head", args.root / "evaluation", n)):
            with np.load(root / sid / "q1/reconstruction.npz") as cache:
                decoded = cache["reconstruction"]
                parts[name] = {"roi_psnr": temporal_parts(source, decoded, n["rois"])}
                frames[{"rgb": "RGB patch q=1", "head": "Feature patch q=1"}[name]] = decoded[8].copy()
            packets = result["points"]["q1"]["packet_details"]
            parts[name]["packet_bytes_excluding_global_header"] = {
                label: sum(v["packet_bytes"] for v in packets if (v["start"] == 0) == is_i)
                for label, is_i in (("I", True), ("P8x2", False))}
        frames["UF32 reference"] = np.asarray(Image.open(sorted((historical / "uf32_fresh").glob("*.png"))[8]).convert("RGB"))
        boundary_panel(args.root / f"boundary_{sid}.png", frames, n["rois"][0])
        rows.append({"sample": n["sample"], "rois": n["rois"], "uf": uf,
                     "q1_temporal_parts": parts,
                     "rgb": {k: o["points"][k] for k in qkeys},
                     "head": {k: n["points"][k] for k in qkeys},
                     "head_information_diagnostic": n["information_diagnostic"],
                     "rgb_information_diagnostic": o["information_diagnostic"]})
    for metric, ylabel, filename in (
            (lambda v: v["roi_psnr"], "Selected-region PSNR (dB), higher is better", "comparison_roi_rd.png"),
            (lambda v: v["quality"]["lpips_alex"], "Whole-frame LPIPS, lower is better", "comparison_lpips_rd.png")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, row in zip(axes.ravel(), rows):
            for key, label, marker in (("uf", "UF (whole frame)", "o"),
                                        ("rgb", "RGB patch (2 regions)", "^"),
                                        ("head", "UF feature patch (2 regions)", "s")):
                values = sorted(row[key].values(), key=lambda v: v["bytes"])
                ax.plot([v["bytes"]/1000 for v in values], [metric(v) for v in values], marker=marker, label=label)
            ax.set_title(row["sample"]["sample_id"])
            ax.set_xlabel("Total on-disk kB, including all headers")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle("Previously used development clips; fixed 2 ROIs / first 17 frames; no BD-rate claim")
        fig.tight_layout()
        fig.savefig(args.root / filename, dpi=160)
        plt.close(fig)
    # Reuse already-rendered fixed contact sheets. Same frame (index 8), same
    # ROI, nearest-neighbor zoom for every method; no generative image editing.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for row in rows:
        sid = row["sample"]["sample_id"]
        a = Image.open(args.reference / "evaluation_warmup" / sid / "fixed_frame.png")
        b = Image.open(args.root / "evaluation" / sid / "fixed_frame.png")
        assert a.size == b.size
        w, h = a.width//3, a.height//2-66
        x, y, rw, rh = row["rois"][0]
        tile_w, tile_h = max(rw*3, 280), rh*3
        canvas = Image.new("RGB", (tile_w*5, tile_h+66), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (name, sheet, col, line, value) in enumerate((
            ("Source", a, 0, 0, None), ("UF8 base", a, 1, 0, row["uf"]["base"]),
            ("RGB patch q=1", a, 0, 1, row["rgb"]["q1"]),
            ("Feature patch q=1", b, 0, 1, row["head"]["q1"]),
            ("UF32 reference", a, 2, 1, row["uf"]["uf32"]))):
            left, top = col*w+x, line*(h+66)+66+y
            crop = sheet.crop((left+1, top+1, left+rw-1, top+rh-1))
            canvas.paste(crop.resize((rw*3, rh*3), Image.Resampling.NEAREST), (i*tile_w, 66))
            draw.text((i*tile_w+5, 5), name, fill="black", font=font)
            if value:
                draw.text((i*tile_w+5, 25), f"total {value['bytes']/1000:.2f} kB", fill="black", font=font)
                draw.text((i*tile_w+5, 45), f"2-ROI PSNR {value['roi_psnr']:.2f} dB", fill="black", font=font)
        canvas.save(args.root / f"zoom_{sid}.png")
    atomic_json(args.root / "comparison.json", {
        "data_role": "previously used development", "same_base_and_rois": True,
        "training_recipe_and_sample_schedule_equal": same_choices, "training_steps": len(new_steps),
        "sources": {str(p): file_hash(p) for p in (new_path, old_path, extra_path)}, "results": rows})
    for row in rows:
        print(json.dumps({"sample": row["sample"]["sample_id"], "q1": {
            k: {"bytes": row[k]["q1"]["bytes"], "roi_psnr": row[k]["q1"]["roi_psnr"],
                "lpips": row[k]["q1"]["quality"]["lpips_alex"]} for k in ("rgb", "head")}}))


if __name__ == "__main__":
    main()
