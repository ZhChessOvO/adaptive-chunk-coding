#!/usr/bin/env python3
"""Build a paper-facing evidence package from the frozen v6 evaluation.

This script never reruns the codec or restoration models.  It validates the
completed machine-readable summaries and derives small tables, diagrams, and
claim-to-evidence notes from those frozen results.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


VARIANTS = (
    "baseline-v5-feather16",
    "adaptation-only",
    "spatial-only",
    "combined",
)
GROUPS = ("combined", "REDS", "UVG")
VARIANT_LABELS = {
    "baseline-v5-feather16": "v5 + feather16",
    "adaptation-only": "Adaptation only",
    "spatial-only": "Spatial only",
    "combined": "Combined (v6)",
}
COLORS = {
    "ink": "#183044",
    "muted": "#5f7180",
    "paper": "#f7f8f5",
    "panel": "#ffffff",
    "line": "#cbd5dc",
    "blue": "#4a90e2",
    "orange": "#f2a02a",
    "green": "#46aa5c",
    "purple": "#7a66cc",
    "teal": "#268b8b",
    "red": "#c95656",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--resource-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=REPO_ROOT, text=True,
    ).strip()
    return {"commit": commit, "worktree_dirty": bool(status)}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def validate(summary: dict, frozen: dict, resources: dict) -> None:
    if summary.get("status") != "complete" or summary.get("sample_count") != 37:
        raise RuntimeError("v6 summary is not the completed 37-sample result")
    if tuple(summary.get("logical_variants", [])) != VARIANTS:
        raise RuntimeError("v6 logical variants differ from the frozen 2x2")
    if summary.get("dataset_counts") != {"REDS": 30, "UVG": 7}:
        raise RuntimeError("v6 dataset counts differ")
    if summary.get("feather_pixels") != 16:
        raise RuntimeError("v6 feather width differs")
    for group in GROUPS:
        if set(summary["aggregate"].get(group, {})) != set(VARIANTS):
            raise RuntimeError(f"missing aggregate variants for {group}")

    verification = summary.get("verification", {})
    required_verification = (
        "all_new_stream_sizes_rechecked_against_files",
        "all_new_spatial_fresh_decodes_pixel_exact",
        "fixed_visuals_exist",
        "exact_action_reuse_only",
    )
    if not all(verification.get(key) is True for key in required_verification):
        raise RuntimeError("v6 verification flags are incomplete")
    if verification.get("new_real_evaluation_tasks") != 58:
        raise RuntimeError("unexpected v6 real-evaluation task count")
    if verification.get("logical_target_reuse_count") != 53:
        raise RuntimeError("unexpected v6 exact-action reuse count")

    boundary = summary.get("scientific_boundary", {})
    required_boundary = (
        "all_rate_values_are_real_final_file_bytes",
        "source_rgb_not_read_by_decoder",
        "dcvc_uf_frozen",
        "seedvr2_frozen",
        "spatial_qp_codec_frozen",
        "single_gpu",
    )
    if not all(boundary.get(key) is True for key in required_boundary):
        raise RuntimeError("scientific-boundary flags differ")
    if boundary.get("training_or_finetuning_during_evaluation") is not False:
        raise RuntimeError("evaluation unexpectedly reports training")

    if frozen.get("status") != "frozen-before-quality-evaluation":
        raise RuntimeError("source was not frozen before evaluation")
    if frozen.get("spatial_lambda") != 0.004:
        raise RuntimeError("spatial lambda differs from the selected v6 value")
    if frozen.get("feather_pixels") != 16:
        raise RuntimeError("frozen feather width differs")
    if frozen.get("models_and_codec_frozen") is not True:
        raise RuntimeError("models or codec were not frozen")

    if resources.get("status") != "complete":
        raise RuntimeError("resource snapshot is incomplete")
    if resources.get("single_gpu_indices_observed") != [0]:
        raise RuntimeError("resource snapshot is not single-GPU")
    if resources.get("sample_count") != 37:
        raise RuntimeError("resource snapshot sample count differs")
    if resources.get("verification") != verification:
        raise RuntimeError("resource and quality verification records differ")

    for label, value in summary.get("notion_visual_candidates", {}).items():
        if not Path(value).is_file():
            raise RuntimeError(f"missing qualitative visual {label}: {value}")


def aggregate(summary: dict, group: str, variant: str) -> dict:
    return summary["aggregate"][group][variant]


def effect(summary: dict, group: str, name: str) -> dict:
    return summary["comparisons"][group]["factorial_effects"][name]


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def ablation_table(summary: dict) -> str:
    rows = []
    for name in VARIANTS:
        value = aggregate(summary, "combined", name)
        rows.append(
            f"{latex_escape(VARIANT_LABELS[name])} & "
            f"{value['actual_on_disk_bytes_mean']:.1f} & "
            f"{value['quality_mean']['lpips_alex']:.6f} & "
            f"{value['quality_mean']['psnr_db']:.3f} & "
            f"{value['runtime_seconds_mean']:.2f} & "
            f"{value['generate_boundary_edges_total']} & "
            f"{value['generate_components_total']} \\\\"
        )
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Factorial ablation on 30 REDS and 7 UVG clips. Rate is the complete on-disk stream size for 17 frames; lower LPIPS is better.}",
        r"\label{tab:v6-factorial}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Bytes & LPIPS$\downarrow$ & PSNR$\uparrow$ & Time (s)$\downarrow$ & G edges$\downarrow$ & G comps.$\downarrow$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def cross_dataset_table(summary: dict) -> str:
    rows = []
    titles = {"combined": "All (37)", "REDS": "REDS (30)", "UVG": "UVG (7)"}
    for group in GROUPS:
        baseline = aggregate(summary, group, "baseline-v5-feather16")
        combined = aggregate(summary, group, "combined")
        comparison = effect(summary, group, "combined_vs_baseline")
        byte_percent = (
            100.0 * comparison["mean_byte_delta"]
            / baseline["actual_on_disk_bytes_mean"]
        )
        rows.append(
            f"{titles[group]} & {baseline['quality_mean']['lpips_alex']:.6f} & "
            f"{combined['quality_mean']['lpips_alex']:.6f} & "
            f"{comparison['mean_lpips_delta']:+.6f} & "
            f"{comparison['mean_byte_delta']:+.1f} ({byte_percent:+.2f}\\%) & "
            f"{baseline['generate_components_total']} $\\rightarrow$ "
            f"{combined['generate_components_total']} \\\\"
        )
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen v6 versus v5 by dataset. Negative $\Delta$LPIPS is better.}",
        r"\label{tab:v6-cross-dataset}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & v5 LPIPS & v6 LPIPS & $\Delta$LPIPS & $\Delta$bytes & G comps. \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main_results_table(summary: dict) -> str:
    baseline = aggregate(summary, "combined", "baseline-v5-feather16")
    combined = aggregate(summary, "combined", "combined")
    scalar = summary["comparisons"]["combined"]["targets_vs_baseline"]["combined"]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Headline result on the 37-clip development and error-analysis pool.}",
        r"\label{tab:v6-main}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Metric & v5 + feather16 & Combined v6 \\",
        r"\midrule",
        f"On-disk bytes / 17 frames & {baseline['actual_on_disk_bytes_mean']:.1f} & {combined['actual_on_disk_bytes_mean']:.1f} \\\\ ",
        f"LPIPS $\\downarrow$ & {baseline['quality_mean']['lpips_alex']:.6f} & \\textbf{{{combined['quality_mean']['lpips_alex']:.6f}}} \\\\ ",
        f"PSNR (dB) $\\uparrow$ & {baseline['quality_mean']['psnr_db']:.3f} & \\textbf{{{combined['quality_mean']['psnr_db']:.3f}}} \\\\ ",
        f"Full time (s) $\\downarrow$ & {baseline['runtime_seconds_mean']:.2f} & \\textbf{{{combined['runtime_seconds_mean']:.2f}}} \\\\ ",
        f"Generate boundary edges $\\downarrow$ & {baseline['generate_boundary_edges_total']} & \\textbf{{{combined['generate_boundary_edges_total']}}} \\\\ ",
        f"Generate components $\\downarrow$ & {baseline['generate_components_total']} & \\textbf{{{combined['generate_components_total']}}} \\\\ ",
        r"\midrule",
        f"Better / equal vs v5 & -- & {scalar['target_better_lpips_than_baseline_count']} / {scalar['target_equal_lpips_to_baseline_count']} of 37 \\\\ ",
        f"Mean $\\Delta$LPIPS vs nearest scalar QP & -- & {scalar['mean_target_minus_nearest_scalar_lpips']:+.6f} \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def claim_matrix(summary: dict, resources: dict, snapshot_hashes: dict) -> str:
    combined = aggregate(summary, "combined", "combined")
    baseline = aggregate(summary, "combined", "baseline-v5-feather16")
    all_effect = effect(summary, "combined", "combined_vs_baseline")
    uvg_effect = effect(summary, "UVG", "combined_vs_baseline")
    reds_effect = effect(summary, "REDS", "combined_vs_baseline")
    adapt_effect = effect(summary, "combined", "adaptation_without_spatial")
    spatial_effect = effect(summary, "combined", "spatial_without_adaptation")
    spatial_after_adapt = effect(summary, "combined", "spatial_with_adaptation")
    return f"""# v6 claim-to-evidence matrix

This page separates what the current evidence supports from what the paper should not overclaim. Values are generated from the frozen summary, not copied by hand.

| Candidate claim | Direct evidence | Safe wording | Do not claim |
|---|---|---|---|
| Self-contained spatial-QP stream | 58 new action maps were written to disk; every byte count was rechecked; every fresh decode was pixel-exact without source RGB. | A single DCVC-UF stream carries the action map and spatial quality field and remains independently decodable. | Standard compliance, production readiness, or zero syntax overhead. |
| Restoration-aware three-way allocation | Every tile chooses Base, Generate, or Enhance under byte and Generate-call budgets; the final stream contains all syntax and payload bytes. | The method jointly allocates transmitted information and restoration compute by region. | Generate is free, or Enhance gains are generative gains. |
| Cross-domain adaptation helps the intended failure mode | Adaptation-only changes mean LPIPS by {adapt_effect['mean_lpips_delta']:+.6f} over 37 clips; combined changes UVG by {uvg_effect['mean_lpips_delta']:+.6f} with {uvg_effect['better_count']}/7 improved. | A small UVG label mixture corrects content-dependent Generate mistakes, especially on UVG. | General domain adaptation, unseen-domain SOTA, or a fully blind test. |
| Exact spatial optimization reduces fragmentation | Spatial-only changes G boundaries {baseline['generate_boundary_edges_total']} to {aggregate(summary, 'combined', 'spatial-only')['generate_boundary_edges_total']} and components {baseline['generate_components_total']} to {aggregate(summary, 'combined', 'spatial-only')['generate_components_total']}. | The exact objective produces fewer Generate boundaries and connected ROIs at nearly unchanged rate. | It removes every seam or improves every boundary-severity metric. |
| The two parts are complementary | Combined has LPIPS {combined['quality_mean']['lpips_alex']:.6f}, {combined['generate_components_total']} G components, and {combined['runtime_seconds_mean']:.2f} s mean full runtime; adding spatial consistency after adaptation changes LPIPS by {spatial_after_adapt['mean_lpips_delta']:+.6f}. | Combined gives the best aggregate balance among the frozen four variants. | Every video improves, or either factor alone dominates all metrics. |
| Single-card feasibility | One A800 was observed; wall time was {resources['wall_seconds']} s and peak sampled memory was {resources['peak_nvidia_smi_used_memory_mib']} MiB. | The complete 58-task v6 evaluation fits comfortably on one A800 80GB. | The same throughput on other GPUs or at production scale. |

## Headline numbers

- Combined vs v5: {all_effect['mean_lpips_delta']:+.6f} LPIPS, {all_effect['mean_byte_delta']:+.1f} B, {baseline['generate_boundary_edges_total']} to {combined['generate_boundary_edges_total']} G boundaries, and {baseline['generate_components_total']} to {combined['generate_components_total']} components.
- REDS: {reds_effect['mean_lpips_delta']:+.6f} LPIPS, effectively flat.
- UVG: {uvg_effect['mean_lpips_delta']:+.6f} LPIPS, {uvg_effect['better_count']}/7 improved and {uvg_effect['equal_count']}/7 equal.
- Spatial-only: {spatial_effect['mean_lpips_delta']:+.6f} LPIPS; its contribution is fragmentation and execution structure, not standalone quality.

## Provenance

- v6 summary SHA-256: `{snapshot_hashes['summary']}`
- frozen source SHA-256: `{snapshot_hashes['frozen_source']}`
- resource snapshot SHA-256: `{snapshot_hashes['resources']}`
"""


def qualitative_manifest(summary: dict) -> str:
    visuals = summary["notion_visual_candidates"]
    captions = {
        "Beauty": (
            "Shows the full 2x2 comparison on a face. Adaptation and spatial "
            "optimization alter the action layout; combined improves LPIPS."
        ),
        "Jockey": (
            "Shows a hard motion/background case. Spatial optimization turns "
            "several Generate fragments into a more coherent ROI and improves "
            "runtime, while remaining texture limitations stay visible."
        ),
        "YachtRide": (
            "A sequence not used for v6 training. Adaptation changes LPIPS while "
            "the spatial action map is unchanged, so it cleanly illustrates the "
            "cross-domain controller effect without claiming a blind test."
        ),
    }
    lines = [
        "# Qualitative figure manifest",
        "",
        "Each source figure contains GT, the four frozen variants, and their action maps. Blue is Base, orange is Generate, and green is Enhance.",
        "",
        "| Sequence | Formal source | Why it is useful |",
        "|---|---|---|",
    ]
    for name in ("Beauty", "Jockey", "YachtRide"):
        lines.append(f"| {name} | `{visuals[name]}` | {captions[name]} |")
    lines.extend([
        "",
        "These examples are fixed error-analysis illustrations. They do not replace the 37-clip aggregate statistics.",
        "",
    ])
    return "\n".join(lines)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    title_size: int = 32,
    body_size: int = 24,
    title_color: str = COLORS["ink"],
    body_color: str = COLORS["muted"],
) -> None:
    x0, y0, x1, y1 = box
    title_font = font(title_size, bold=True)
    body_font = font(body_size)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((x0 + x1 - title_width) / 2, y0 + 26), title,
              font=title_font, fill=title_color)
    wrap = max(12, int((x1 - x0) / (body_size * 0.58)))
    lines = textwrap.wrap(body, width=wrap)
    line_height = body_size + 8
    start_y = y0 + 80
    for index, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=body_font)
        width = bbox[2] - bbox[0]
        draw.text(((x0 + x1 - width) / 2, start_y + index * line_height),
                  line, font=body_font, fill=body_color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = COLORS["muted"],
    width: int = 6,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    spread = 0.55
    points = [
        end,
        (
            end[0] - length * math.cos(angle - spread),
            end[1] - length * math.sin(angle - spread),
        ),
        (
            end[0] - length * math.cos(angle + spread),
            end[1] - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(points, fill=color)


def method_png(path: Path) -> None:
    image = Image.new("RGB", (2400, 1280), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((100, 54), "Budget-Adaptive Regional Coding and Restoration",
              font=font(52, bold=True), fill=COLORS["ink"])
    draw.text((102, 120),
              "Frozen codec and restorer; only lightweight routing experts are learned",
              font=font(28), fill=COLORS["muted"])

    draw.rounded_rectangle((70, 205, 2330, 665), radius=30,
                           fill="#eef5fb", outline="#a8c8e5", width=4)
    draw.text((105, 228), "ENCODER: decide where bits and restoration compute go",
              font=font(28, bold=True), fill="#296397")
    encoder_boxes = [
        ((110, 315, 460, 575), "Video + budgets",
         "17 frames, byte budget, Generate-call budget", COLORS["panel"]),
        ((535, 315, 885, 575), "Two-view evidence",
         "Source/context features + temporary all-Base probe", COLORS["panel"]),
        ((960, 315, 1310, 575), "Small controller",
         "v1 anchor + context/probe residual experts + 10.7% UVG adaptation", "#f5f0ff"),
        ((1385, 315, 1735, 575), "Exact spatial solver",
         "Maximize tile utility - 0.004 x Generate boundaries", "#fff4df"),
        ((1810, 315, 2290, 575), "Self-contained spatial-QP stream",
         "4x4 G/B/E decisions, 2-bit map, headers and entropy payloads", "#eef8ef"),
    ]
    for box, title, body, fill in encoder_boxes:
        draw.rounded_rectangle(box, radius=24, fill=fill,
                               outline=COLORS["line"], width=4)
        draw_centered_text(draw, box, title, body)
    for left, right in zip(encoder_boxes, encoder_boxes[1:]):
        draw_arrow(draw, (left[0][2] + 10, 445), (right[0][0] - 10, 445))

    draw.rounded_rectangle((70, 710, 2330, 1180), radius=30,
                           fill="#f1f8f1", outline="#abd2b4", width=4)
    draw.text((105, 733), "DECODER: preserve a valid codec path, restore only connected G regions",
              font=font(28, bold=True), fill="#2d7540")
    decoder_boxes = [
        ((110, 830, 470, 1090), "Fresh DCVC-UF decode",
         "One latent field and one temporal reference loop", COLORS["panel"]),
        ((560, 830, 960, 1090), "Regional actions",
         "B: normal decode   E: more real bits   G: low-rate context", COLORS["panel"]),
        ((1050, 830, 1450, 1090), "Connected G ROIs",
         "Merge adjacent tiles; one resident frozen SeedVR2", "#fff4df"),
        ((1540, 830, 1900, 1090), "16 px feather",
         "Blend restored ROIs into the codec reconstruction", "#f5f0ff"),
        ((1990, 830, 2290, 1090), "Output video",
         "Quality, bytes and full runtime are measured", COLORS["panel"]),
    ]
    for box, title, body, fill in decoder_boxes:
        draw.rounded_rectangle(box, radius=24, fill=fill,
                               outline=COLORS["line"], width=4)
        draw_centered_text(draw, box, title, body)
    for left, right in zip(decoder_boxes, decoder_boxes[1:]):
        draw_arrow(draw, (left[0][2] + 10, 960), (right[0][0] - 10, 960))
    draw.line(((2050, 585), (2050, 687), (82, 687), (82, 960)),
              fill=COLORS["teal"], width=7)
    draw_arrow(draw, (82, 960), (100, 960),
               color=COLORS["teal"], width=7)

    legend_y = 1215
    draw.text((105, legend_y), "Action map:", font=font(24, bold=True),
              fill=COLORS["ink"])
    for index, (name, color_value) in enumerate((
        ("Base", COLORS["blue"]),
        ("Generate", COLORS["orange"]),
        ("Enhance", COLORS["green"]),
    )):
        x = 290 + index * 250
        draw.rounded_rectangle((x, legend_y - 2, x + 42, legend_y + 40),
                               radius=8, fill=color_value)
        draw.text((x + 55, legend_y), name, font=font(23), fill=COLORS["ink"])
    draw.text((1350, legend_y), "Frozen: DCVC-UF, SeedVR2, spatial-QP format",
              font=font(23), fill=COLORS["muted"])
    atomic_png(image, path)


def method_svg(path: Path) -> None:
    boxes = [
        (80, 180, 290, 145, "Video + budgets", "frames, bytes, compute", "#ffffff"),
        (410, 180, 310, 145, "Two-view evidence", "context + Base probe", "#ffffff"),
        (760, 180, 330, 145, "Small controller", "anchor + residual experts", "#f5f0ff"),
        (1130, 180, 330, 145, "Exact spatial solver", "utility - 0.004 boundaries", "#fff4df"),
        (1500, 180, 420, 145, "Spatial-QP stream", "2-bit map + all payload bytes", "#eef8ef"),
        (80, 520, 330, 145, "Fresh DCVC-UF decode", "single stream and reference loop", "#ffffff"),
        (500, 520, 330, 145, "B / G / E actions", "normal / restore / more bits", "#ffffff"),
        (920, 520, 330, 145, "Connected G ROIs", "frozen SeedVR2", "#fff4df"),
        (1340, 520, 280, 145, "16 px feather", "smooth ROI blending", "#f5f0ff"),
        (1710, 520, 210, 145, "Output", "measure all costs", "#ffffff"),
    ]
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="790" viewBox="0 0 2000 790">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#5f7180"/></marker></defs>',
        '<rect width="2000" height="790" fill="#f7f8f5"/>',
        '<text x="70" y="62" font-family="DejaVu Sans,Arial" font-size="36" font-weight="700" fill="#183044">Budget-Adaptive Regional Coding and Restoration</text>',
        '<text x="72" y="102" font-family="DejaVu Sans,Arial" font-size="20" fill="#5f7180">Frozen codec and restoration backbone; learned regional allocation</text>',
        '<rect x="45" y="125" width="1910" height="245" rx="24" fill="#eef5fb" stroke="#a8c8e5" stroke-width="3"/>',
        '<text x="75" y="158" font-family="DejaVu Sans,Arial" font-size="20" font-weight="700" fill="#296397">ENCODER</text>',
        '<rect x="45" y="465" width="1910" height="245" rx="24" fill="#f1f8f1" stroke="#abd2b4" stroke-width="3"/>',
        '<text x="75" y="498" font-family="DejaVu Sans,Arial" font-size="20" font-weight="700" fill="#2d7540">DECODER</text>',
    ]
    for x, y, width, height, title, body, fill in boxes:
        elements.extend([
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{fill}" stroke="#cbd5dc" stroke-width="3"/>',
            f'<text x="{x + width / 2}" y="{y + 55}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="21" font-weight="700" fill="#183044">{html.escape(title)}</text>',
            f'<text x="{x + width / 2}" y="{y + 96}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="17" fill="#5f7180">{html.escape(body)}</text>',
        ])
    for row in ((0, 5), (5, 10)):
        for index in range(row[0], row[1] - 1):
            current = boxes[index]
            following = boxes[index + 1]
            elements.append(
                f'<line x1="{current[0] + current[2] + 10}" y1="{current[1] + current[3] / 2}" x2="{following[0] - 12}" y2="{following[1] + following[3] / 2}" stroke="#5f7180" stroke-width="4" marker-end="url(#arrow)"/>'
            )
    elements.append('<path d="M1710 325 L1710 415 L60 415 L60 592 L70 592" fill="none" stroke="#268b8b" stroke-width="5" marker-end="url(#arrow)"/>')
    elements.extend([
        '<rect x="80" y="735" width="22" height="22" rx="4" fill="#4a90e2"/><text x="112" y="753" font-family="DejaVu Sans,Arial" font-size="18" fill="#183044">Base</text>',
        '<rect x="210" y="735" width="22" height="22" rx="4" fill="#f2a02a"/><text x="242" y="753" font-family="DejaVu Sans,Arial" font-size="18" fill="#183044">Generate</text>',
        '<rect x="380" y="735" width="22" height="22" rx="4" fill="#46aa5c"/><text x="412" y="753" font-family="DejaVu Sans,Arial" font-size="18" fill="#183044">Enhance</text>',
        '<text x="1100" y="753" font-family="DejaVu Sans,Arial" font-size="18" fill="#5f7180">Frozen: DCVC-UF, SeedVR2, spatial-QP format</text>',
        '</svg>',
    ])
    atomic_text(path, "\n".join(elements) + "\n")


def tradeoff_png(summary: dict, path: Path) -> None:
    image = Image.new("RGB", (1500, 980), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), "v6 factorial trade-off: quality and ROI fragmentation",
              font=font(40, bold=True), fill=COLORS["ink"])
    draw.text((72, 105), "Lower-left is better; all points use the same 37 clips",
              font=font(24), fill=COLORS["muted"])
    left, top, right, bottom = 170, 190, 1390, 830
    draw.rectangle((left, top, right, bottom), fill="#ffffff",
                   outline=COLORS["line"], width=3)
    values = {
        name: (
            aggregate(summary, "combined", name)["generate_components_total"],
            aggregate(summary, "combined", name)["quality_mean"]["lpips_alex"],
        )
        for name in VARIANTS
    }
    xs = [value[0] for value in values.values()]
    ys = [value[1] for value in values.values()]
    x_min, x_max = min(xs) - 4, max(xs) + 4
    y_min, y_max = min(ys) - 0.0006, max(ys) + 0.0006

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    for step in range(6):
        value = x_min + step * (x_max - x_min) / 5
        x, _ = point(value, y_min)
        draw.line((x, top, x, bottom), fill="#e4e9ed", width=2)
        label = f"{value:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font(20))
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 14), label,
                  font=font(20), fill=COLORS["muted"])
    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        _, y = point(x_min, value)
        draw.line((left, y, right, y), fill="#e4e9ed", width=2)
        label = f"{value:.4f}"
        bbox = draw.textbbox((0, 0), label, font=font(20))
        draw.text((left - (bbox[2] - bbox[0]) - 15, y - 12), label,
                  font=font(20), fill=COLORS["muted"])
    draw.text((600, 900), "Generate connected components (total)",
              font=font(25, bold=True), fill=COLORS["ink"])
    draw.text((24, 505), "LPIPS", font=font(25, bold=True),
              fill=COLORS["ink"])

    palette = {
        "baseline-v5-feather16": COLORS["blue"],
        "adaptation-only": COLORS["purple"],
        "spatial-only": COLORS["orange"],
        "combined": COLORS["green"],
    }
    offsets = {
        "baseline-v5-feather16": (-250, -65),
        "adaptation-only": (25, -60),
        "spatial-only": (25, 15),
        "combined": (25, 15),
    }
    for name, (components, lpips) in values.items():
        x, y = point(components, lpips)
        radius = 22 if name != "combined" else 30
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=palette[name], outline="#ffffff", width=5)
        dx, dy = offsets[name]
        label = f"{VARIANT_LABELS[name]}\nLPIPS {lpips:.6f}; comps {components}"
        draw.multiline_text((x + dx, y + dy), label, font=font(22, bold=True),
                            fill=COLORS["ink"], spacing=6)
    atomic_png(image, path)


def dataset_effect_png(summary: dict, path: Path) -> None:
    image = Image.new("RGB", (1500, 900), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), "Where does combined v6 help?",
              font=font(42, bold=True), fill=COLORS["ink"])
    draw.text((72, 110), "LPIPS change versus v5 + feather16; negative is better",
              font=font(24), fill=COLORS["muted"])
    groups = (
        ("All 37", "combined", COLORS["teal"]),
        ("REDS 30", "REDS", COLORS["blue"]),
        ("UVG 7", "UVG", COLORS["green"]),
    )
    deltas = [effect(summary, key, "combined_vs_baseline")["mean_lpips_delta"]
              for _, key, _ in groups]
    scale = max(abs(value) for value in deltas) * 1.25
    axis_x = 900
    top = 235
    row_height = 175
    draw.line((axis_x, top - 35, axis_x, top + row_height * 3 - 45),
              fill=COLORS["ink"], width=4)
    draw.text((axis_x - 12, 730), "0", font=font(22), fill=COLORS["muted"])
    for index, (label, key, color_value) in enumerate(groups):
        value = deltas[index]
        y = top + index * row_height
        draw.text((90, y + 22), label, font=font(31, bold=True),
                  fill=COLORS["ink"])
        width = abs(value) / scale * 650
        if value < 0:
            box = (axis_x - width, y, axis_x, y + 80)
            if width > 300:
                value_x = axis_x - width + 25
                value_color = "#ffffff"
            else:
                value_x = axis_x - width - 230
                value_color = COLORS["ink"]
        else:
            box = (axis_x, y, axis_x + width, y + 80)
            value_x = axis_x + width + 30
            value_color = COLORS["ink"]
        draw.rounded_rectangle(box, radius=18, fill=color_value)
        draw.text((value_x, y + 20), f"{value:+.6f}",
                  font=font(27, bold=True), fill=value_color)
        comparison = effect(summary, key, "combined_vs_baseline")
        draw.text((90, y + 68),
                  f"{comparison['better_count']} better, {comparison['equal_count']} equal",
                  font=font(21), fill=COLORS["muted"])
    draw.text((72, 820),
              "Interpretation: the aggregate gain is driven by UVG; REDS is effectively flat.",
              font=font(25, bold=True), fill=COLORS["ink"])
    atomic_png(image, path)


def package_index(
    summary: dict, frozen: dict, resources: dict, packaging_git: dict,
) -> str:
    baseline = aggregate(summary, "combined", "baseline-v5-feather16")
    combined = aggregate(summary, "combined", "combined")
    comparison = effect(summary, "combined", "combined_vs_baseline")
    return f"""# v6 paper evidence package

This directory is a read-only derivative of the completed single-A800 evaluation. It does not contain new training, codec runs, or SeedVR2 runs.

## Frozen main method

- Controller: v5 anchor plus context and Base-probe residual experts, adapted with 60 UVG windows (560 total windows; 10.7% UVG).
- Spatial objective: tile utility minus `0.004` times the Generate/non-Generate 4-neighbor boundary count.
- Stream: self-contained spatial-QP DCVC-UF with Generate/Base/Enhance quality indices 8/16/32.
- Decoder: connected Generate ROIs, frozen SeedVR2, 64-pixel context, 16-pixel feathering.
- Frozen backbones: DCVC-UF and SeedVR2; no backbone fine-tuning.

## Headline result

Across 30 REDS and 7 UVG clips, combined v6 changes mean LPIPS from {baseline['quality_mean']['lpips_alex']:.6f} to {combined['quality_mean']['lpips_alex']:.6f} ({comparison['mean_lpips_delta']:+.6f}), mean stream size by {comparison['mean_byte_delta']:+.1f} B, Generate boundaries from {baseline['generate_boundary_edges_total']} to {combined['generate_boundary_edges_total']}, and connected components from {baseline['generate_components_total']} to {combined['generate_components_total']}. It is the selected balance, not a claim that every clip improves.

## Files

- `method_overview.png` / `.svg`: paper method diagram.
- `v6_main_results.tex`: compact headline table.
- `v6_factorial_ablation.tex`: four-way 2x2 ablation.
- `v6_cross_dataset.tex`: REDS/UVG split and deltas.
- `v6_tradeoff.png`: LPIPS versus Generate fragmentation.
- `v6_dataset_effect.png`: dataset-specific LPIPS effect.
- `claim_evidence_matrix.md`: supported claims, exact evidence, and overclaim boundaries.
- `qualitative_manifest.md`: fixed formal figures and what each demonstrates.
- `paper_snapshot.json`: compact provenance and resource record.

## Reproducibility boundary

- Frozen evaluator commit: `{frozen['git_commit']}`.
- Paper-packaging commit: `{packaging_git['commit']}`; worktree dirty while generating: `{str(packaging_git['worktree_dirty']).lower()}`.
- Formal wall time: {resources['wall_seconds']} s on one A800; peak sampled memory {resources['peak_nvidia_smi_used_memory_mib']} MiB.
- All 58 new streams passed byte recheck and pixel-exact fresh decode; 53 logical results were reused only when all 16 actions matched exactly.
- The 37 clips are development/error-analysis evidence after method selection, not a newly blind benchmark.
"""


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    frozen_path = args.frozen_source.resolve()
    resources_path = args.resource_snapshot.resolve()
    output_dir = args.output_dir.resolve()
    summary = read_json(summary_path)
    frozen = read_json(frozen_path)
    resources = read_json(resources_path)
    validate(summary, frozen, resources)
    output_dir.mkdir(parents=True, exist_ok=True)

    hashes = {
        "summary": sha256(summary_path),
        "frozen_source": sha256(frozen_path),
        "resources": sha256(resources_path),
    }
    packaging_git = git_state()
    baseline = aggregate(summary, "combined", "baseline-v5-feather16")
    combined = aggregate(summary, "combined", "combined")
    combined_effect = effect(summary, "combined", "combined_vs_baseline")
    snapshot = {
        "experiment": "v6 paper evidence package",
        "status": "complete",
        "derivative_only_no_new_model_execution": True,
        "sources": {
            "summary": {"path": str(summary_path), "sha256": hashes["summary"]},
            "frozen_source": {
                "path": str(frozen_path),
                "sha256": hashes["frozen_source"],
            },
            "resource_snapshot": {
                "path": str(resources_path),
                "sha256": hashes["resources"],
            },
            "frozen_git_commit": frozen["git_commit"],
            "paper_packaging_git": packaging_git,
        },
        "frozen_method": {
            "selected_variant": "combined",
            "spatial_lambda": frozen["spatial_lambda"],
            "feather_pixels": frozen["feather_pixels"],
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
        "headline": {
            "sample_count": summary["sample_count"],
            "dataset_counts": summary["dataset_counts"],
            "baseline": baseline,
            "combined": combined,
            "combined_vs_baseline": combined_effect,
        },
        "verification": summary["verification"],
        "resources": {
            "wall_seconds": resources["wall_seconds"],
            "peak_nvidia_smi_used_memory_mib": resources[
                "peak_nvidia_smi_used_memory_mib"],
            "single_gpu_indices_observed": resources[
                "single_gpu_indices_observed"],
            "ordinary_file_bytes": resources["ordinary_file_bytes"],
        },
        "scientific_boundary": summary["scientific_boundary"],
    }

    atomic_json(output_dir / "paper_snapshot.json", snapshot)
    atomic_text(
        output_dir / "README.md",
        package_index(summary, frozen, resources, packaging_git),
    )
    atomic_text(output_dir / "v6_main_results.tex", main_results_table(summary))
    atomic_text(output_dir / "v6_factorial_ablation.tex", ablation_table(summary))
    atomic_text(output_dir / "v6_cross_dataset.tex", cross_dataset_table(summary))
    atomic_text(output_dir / "claim_evidence_matrix.md",
                claim_matrix(summary, resources, hashes))
    atomic_text(output_dir / "qualitative_manifest.md",
                qualitative_manifest(summary))
    method_png(output_dir / "method_overview.png")
    method_svg(output_dir / "method_overview.svg")
    tradeoff_png(summary, output_dir / "v6_tradeoff.png")
    dataset_effect_png(summary, output_dir / "v6_dataset_effect.png")

    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {
            "artifact_manifest.json", "paper_package.complete",
        }
    }
    atomic_json(output_dir / "artifact_manifest.json", manifest)
    written_manifest = read_json(output_dir / "artifact_manifest.json")
    if written_manifest != manifest:
        raise RuntimeError("artifact manifest changed while being written")
    for name, record in written_manifest.items():
        artifact = output_dir / name
        if (
            not artifact.is_file()
            or artifact.stat().st_size != record["bytes"]
            or sha256(artifact) != record["sha256"]
        ):
            raise RuntimeError(f"paper artifact verification failed: {artifact}")
    completion = {
        "status": "complete",
        "artifact_count": len(manifest) + 2,
        "artifact_manifest_sha256": sha256(output_dir / "artifact_manifest.json"),
    }
    atomic_json(output_dir / "paper_package.complete", completion)
    print(json.dumps({
        "output_dir": str(output_dir),
        "artifact_count": completion["artifact_count"],
        "headline_lpips_delta": combined_effect["mean_lpips_delta"],
        "headline_byte_delta": combined_effect["mean_byte_delta"],
        "summary_sha256": hashes["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
