"""Plot measured single-layer development results, keeping failed recipes visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    first = json.loads((args.root / "evaluation/summary.json").read_text())
    second = json.loads((args.root / "evaluation_warmup/summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, a, b in zip(axes.ravel(), first["results"], second["results"]):
        assert a["sample"] == b["sample"] and a["rois"] == b["rois"]
        for label, data, keys, style in (
                ("UF (full-frame reference)", a, ["base", "uf32"], "o-"),
                ("Haar (1 / 2 level prefixes)", a, ["a1_b1", "two_levels"], "s-"),
                ("Neural: collapsed first recipe", a, ["q2", "q1", "q0.5"], "x:"),
                ("Neural: revised training", b, ["q2", "q1", "q0.5"], "^-")):
            values = sorted([data["points"][k] for k in keys], key=lambda v: v["bytes"])
            ax.plot([v["bytes"]/1000 for v in values], [v["roi_psnr"] for v in values], style, label=label)
        ax.set_xscale("log")
        ax.set_title(b["sample"]["sample_id"])
        ax.set_xlabel("Total on-disk kB (log scale)")
        ax.set_ylabel("Selected-region PSNR (dB)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle("Development diagnostic: same 2 ROIs / first 17 frames; not matched whole-frame quality")
    fig.tight_layout()
    fig.savefig(args.root / "comparison_rd.png", dpi=160)
    plt.close(fig)

    # Magnified fixed frame: source / base / proposed q=1 / native UF32.
    # Reading the already rendered contact sheet avoids fresh metrics or new data.
    for row in second["results"]:
        sid = row["sample"]["sample_id"]
        sheet = Image.open(args.root / "evaluation_warmup" / sid / "fixed_frame.png")
        w, h = sheet.width//3, sheet.height//2-66
        roi = row["rois"][0]
        x, y, rw, rh = roi
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        tile_w, tile_h = max(rw*3, 280), rh*3
        canvas = Image.new("RGB", (tile_w*4, tile_h+62), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (name, column, line, key) in enumerate((
                ("Source", 0, 0, None), ("UF8 base", 1, 0, "base"),
                ("Single-layer q=1", 0, 1, "q1"), ("UF32 reference", 2, 1, "uf32"))):
            left, top = column*w+x, line*(h+66)+66+y
            # Exclude the one-pixel drawn green perimeter of the fixed ROI.
            crop = sheet.crop((left+1, top+1, left+rw-1, top+rh-1))
            crop = crop.resize((rw*3, rh*3), Image.Resampling.NEAREST)
            canvas.paste(crop, (i*tile_w, 62))
            draw.text((i*tile_w+5, 5), name, fill="black", font=font)
            if key:
                point = row["points"][key]
                draw.text((i*tile_w+5, 25), f"total {point['bytes']/1000:.2f} kB", fill="black", font=font)
                draw.text((i*tile_w+5, 43), f"2-ROI PSNR {point['roi_psnr']:.2f} dB", fill="black", font=font)
        canvas.save(args.root / f"zoom_{sid}.png")
    print(str(args.root / "comparison_rd.png"))


if __name__ == "__main__":
    main()
