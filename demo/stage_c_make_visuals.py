#!/usr/bin/env python3
"""Create compact E19 comparison images for experiment review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "e19_visualizations"
BASIC = ROOT / "output" / "e19_three_path_jockey_basicvsrpp_ctxfix"
SDXL = ROOT / "output" / "e19_diffusion_ablation_jockey_sdxl_ctxfix"
BUDGETS = [
    ("0.3x Base extra budget", ROOT / "output" / "e19_three_path_jockey_basicvsrpp_b030"),
    ("0.6x Base extra budget", ROOT / "output" / "e19_three_path_jockey_basicvsrpp_b060"),
    ("1.0x Base extra budget", BASIC),
]
FRAME = "im00009.png"
COLORS = {0: (45, 105, 210), 1: (245, 145, 35), 2: (40, 170, 90)}
NAMES = {0: "B", 1: "G", 2: "E"}


def font(size: int, bold: bool = False):
    suffix = "-Bold" if bold else ""
    path = Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf")
    return ImageFont.truetype(str(path), size=size)


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def labeled_panel(image: Image.Image, title: str, subtitle: str, size: int = 320):
    thumb = image.resize((size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size + 58), "white")
    panel.paste(thumb, (0, 58))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), title, fill="black", font=font(18, True))
    draw.text((8, 31), subtitle, fill=(60, 60, 60), font=font(13))
    return panel


def quality(summary: dict, name: str) -> str:
    q = summary["variants"][name]["quality"]
    return f"PSNR {q['psnr_db']:.3f} dB | LPIPS {q['lpips_alex']:.3f}"


def comparison() -> None:
    basic_summary = json.loads((BASIC / "summary.json").read_text())
    sdxl_summary = json.loads((SDXL / "summary.json").read_text())
    panels = [
        labeled_panel(load(BASIC / "frames/original" / FRAME), "GT", "source reference"),
        labeled_panel(load(BASIC / "frames/base-only" / FRAME), "Base QP32", quality(basic_summary, "base-only")),
        labeled_panel(load(BASIC / "frames/base+restorer-all" / FRAME), "BasicVSR++ all", quality(basic_summary, "base+restorer-all")),
        labeled_panel(load(BASIC / "frames/enhance-only-oracle" / FRAME), "Enhance only", quality(basic_summary, "enhance-only-oracle")),
        labeled_panel(load(BASIC / "frames/joint-three-path-oracle" / FRAME), "Joint B/G/E", quality(basic_summary, "joint-three-path-oracle")),
        labeled_panel(load(SDXL / "frames/base+restorer-all" / FRAME), "SDXL all", quality(sdxl_summary, "base+restorer-all")),
    ]
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(OUT / "jockey_frame00009_main_comparison.png", optimize=True)


def action_maps() -> None:
    base = load(BASIC / "frames/original" / FRAME)
    panels = []
    for title, root in BUDGETS:
        summary = json.loads((root / "summary.json").read_text())
        record = summary["variants"]["joint-three-path-oracle"]
        actions = record["actions"]
        overlay = base.convert("RGBA")
        layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for index, action in enumerate(actions):
            x = (index % 2) * 256
            y = (index // 2) * 256
            color = COLORS[action]
            draw.rectangle((x, y, x + 255, y + 255), fill=(*color, 52), outline=(*color, 255), width=8)
            draw.rounded_rectangle((x + 12, y + 12, x + 62, y + 62), radius=8, fill=(*color, 235))
            draw.text((x + 27, y + 17), NAMES[action], fill="white", font=font(28, True), anchor="ma")
        merged = Image.alpha_composite(overlay, layer).convert("RGB")
        subtitle = (
            f"{record['rate']['total_bytes']} B | {record['quality']['psnr_db']:.3f} dB | "
            f"route {''.join(NAMES[x] for x in actions)}")
        panels.append(labeled_panel(merged, title, subtitle, 420))
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height + 42), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    draw.text((12, panels[0].height + 10), "Blue=B Base   Orange=G Generate   Green=E Enhance", fill="black", font=font(18, True))
    canvas.save(OUT / "jockey_budget_action_maps.png", optimize=True)


def zoom() -> None:
    basic_summary = json.loads((BASIC / "summary.json").read_text())
    sdxl_summary = json.loads((SDXL / "summary.json").read_text())
    sources = [
        (BASIC / "frames/original" / FRAME, "GT", "top-left ROI"),
        (BASIC / "frames/base-only" / FRAME, "Base QP32", quality(basic_summary, "base-only")),
        (BASIC / "frames/base+restorer-all" / FRAME, "BasicVSR++", quality(basic_summary, "base+restorer-all")),
        (BASIC / "frames/joint-three-path-oracle" / FRAME, "Joint", quality(basic_summary, "joint-three-path-oracle")),
        (SDXL / "frames/base+restorer-all" / FRAME, "SDXL", quality(sdxl_summary, "base+restorer-all")),
    ]
    panels = [labeled_panel(load(path).crop((0, 0, 256, 256)), title, sub, 300) for path, title, sub in sources]
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(OUT / "jockey_frame00009_top_left_zoom.png", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison()
    action_maps()
    zoom()
    print(OUT)


if __name__ == "__main__":
    main()
