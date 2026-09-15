#!/usr/bin/env python3
"""Aggregate E21 budget points and create review-ready visualizations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ACTION_NAMES = ("Base", "Generate", "Enhance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment") != "E21 perceptual Base/Generate/Enhance response probe":
        raise ValueError(f"not an E21 summary: {path}")
    value["_path"] = path
    return value


def transition_counts(first: list[int], second: list[int]) -> dict:
    result = {}
    for before, after in zip(first, second):
        key = f"{ACTION_NAMES[before]}->{ACTION_NAMES[after]}"
        result[key] = result.get(key, 0) + 1
    return result


def dense_baselines(summaries: list[dict]) -> dict:
    return max(
        summaries,
        key=lambda item: len(item["ordinary_scalar_qp_baselines"]),
    )["ordinary_scalar_qp_baselines"]


def nearest_scalar(baselines: dict, target_bytes: int) -> tuple[str, dict]:
    return min(
        baselines.items(),
        key=lambda item: abs(item[1]["rate"]["total_bytes"] - target_bytes),
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def action_map_sheet(path: Path, rows: list[dict]) -> None:
    panels = []
    for row in rows:
        image = Image.open(row["action_visual"]).convert("RGB")
        banner = 58
        panel = Image.new("RGB", (image.width, image.height + banner), "white")
        panel.paste(image, (0, banner))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (8, 5), f"Enhance budget = {row['budget_ratio']:.2f}x Base",
            fill="black", font=font(18, True))
        draw.text(
            (8, 32),
            f"{row['bytes']} B | LPIPS {row['lpips']:.4f} | "
            f"B/G/E {row['base_tiles']}/{row['generate_tiles']}/{row['enhance_tiles']}",
            fill=(55, 55, 55), font=font(13))
        panels.append(panel)
    canvas = Image.new(
        "RGB", (sum(item.width for item in panels), max(item.height for item in panels)),
        (230, 230, 230))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def plot_curves(path: Path, rows: list[dict], baselines: dict) -> None:
    scalar = sorted(
        (
            int(name.rsplit("qp", 1)[1]),
            record["rate"]["total_bytes"],
            record["quality"]["lpips_alex"],
            record["quality"]["temporal_delta_mae"],
        )
        for name, record in baselines.items()
    )
    joint_x = [row["bytes"] for row in rows]
    joint_lpips = [row["lpips"] for row in rows]
    joint_temporal = [row["temporal_delta_mae"] for row in rows]
    enhance_x = [row["enhance_only_bytes"] for row in rows]
    enhance_lpips = [row["enhance_only_lpips"] for row in rows]
    enhance_temporal = [row["enhance_only_temporal_delta_mae"] for row in rows]
    base = rows[0]["base"]
    all_generate = rows[0]["all_generate"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for axis, scalar_index, joint_y, enhance_y, ylabel in (
        (axes[0], 2, joint_lpips, enhance_lpips, "LPIPS (lower is better)"),
        (axes[1], 3, joint_temporal, enhance_temporal,
         "Temporal delta MAE (lower is better)"),
    ):
        axis.plot(
            [item[1] for item in scalar], [item[scalar_index] for item in scalar],
            "o-", color="#2d69d2", label="Ordinary scalar-QP DCVC-UF")
        axis.plot(joint_x, joint_y, "o-", color="#d57916", linewidth=2.4,
                  label="Joint B/G/E Oracle")
        axis.plot(enhance_x, enhance_y, "s--", color="#28aa5a",
                  label="Enhance-only Oracle")
        metric_key = "lpips_alex" if scalar_index == 2 else "temporal_delta_mae"
        axis.scatter(
            [base["bytes"]], [base[metric_key]], marker="D", s=62,
            color="#555555", label="Base QP16")
        axis.scatter(
            [all_generate["bytes"]], [all_generate[metric_key]], marker="*", s=125,
            color="#b83a9a", label="All Generate")
        for qp, byte_count, metric, _temporal in scalar:
            if qp in (16, 18, 20, 21, 22, 23, 24, 25, 27, 29, 32):
                shown = metric if scalar_index == 2 else _temporal
                axis.annotate(
                    f"Q{qp}", (byte_count, shown), xytext=(2, 5),
                    textcoords="offset points", fontsize=7, color="#234f9e")
        for row, x_value, y_value in zip(rows, joint_x, joint_y):
            axis.annotate(
                f"{row['budget_ratio']:.2f}x", (x_value, y_value),
                xytext=(5, -12), textcoords="offset points", fontsize=8,
                color="#9a5108")
        axis.set_xlabel("Actual on-disk bytes (17 x 512 x 512 clip)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].set_title("Perceptual rate-quality")
    axes[1].set_title("Temporal diagnostic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in rows[0] if key not in ("base", "all_generate")],
        )
        writer.writeheader()
        writer.writerows({
            key: value for key, value in row.items()
            if key not in ("base", "all_generate")
        } for row in rows)


def main() -> None:
    args = parse_args()
    summaries = [load(path) for path in args.summaries]
    summaries.sort(key=lambda item: item["configuration"]["enhance_budget_ratio_of_base_stream"])
    signature = (
        summaries[0]["sequence"], summaries[0]["frames"],
        summaries[0]["configuration"]["base_qp"],
        summaries[0]["configuration"]["enhance_qp"],
        summaries[0]["configuration"]["tile_size"],
        summaries[0]["configuration"]["max_generate_tiles"],
        summaries[0]["configuration"]["fixed_decoder_feather_pixels"],
    )
    for item in summaries[1:]:
        current = (
            item["sequence"], item["frames"],
            item["configuration"]["base_qp"],
            item["configuration"]["enhance_qp"],
            item["configuration"]["tile_size"],
            item["configuration"]["max_generate_tiles"],
            item["configuration"]["fixed_decoder_feather_pixels"],
        )
        if current != signature:
            raise ValueError("E21 summaries do not share one protocol")

    baselines = dense_baselines(summaries)
    rows = []
    for item in summaries:
        joint = item["variants"]["joint-three-path-oracle"]
        enhance = item["variants"]["enhance-only-oracle"]
        base = item["variants"]["base-only"]
        all_generate = item["variants"]["all-generate"]
        scalar_name, scalar = nearest_scalar(
            baselines, joint["rate"]["total_bytes"])
        counts = joint["action_counts"]
        rows.append({
            "budget_ratio": item["configuration"]["enhance_budget_ratio_of_base_stream"],
            "bytes": joint["rate"]["total_bytes"],
            "bpp": 8.0 * joint["rate"]["total_bytes"] / (
                item["frames"] * item["crop"]["width"] * item["crop"]["height"]),
            "psnr_db": joint["quality"]["psnr_db"],
            "lpips": joint["quality"]["lpips_alex"],
            "temporal_delta_mae": joint["quality"]["temporal_delta_mae"],
            "base_tiles": counts["Base"],
            "generate_tiles": counts["Generate"],
            "enhance_tiles": counts["Enhance"],
            "actions": "".join(ACTION_NAMES[value][0] for value in joint["actions"]),
            "nearest_scalar": scalar_name,
            "nearest_scalar_bytes": scalar["rate"]["total_bytes"],
            "bytes_vs_scalar": joint["rate"]["total_bytes"] - scalar["rate"]["total_bytes"],
            "lpips_delta_vs_scalar": (
                joint["quality"]["lpips_alex"] - scalar["quality"]["lpips_alex"]),
            "psnr_delta_vs_scalar": (
                joint["quality"]["psnr_db"] - scalar["quality"]["psnr_db"]),
            "temporal_delta_vs_scalar": (
                joint["quality"]["temporal_delta_mae"]
                - scalar["quality"]["temporal_delta_mae"]),
            "enhance_only_bytes": enhance["rate"]["total_bytes"],
            "enhance_only_lpips": enhance["quality"]["lpips_alex"],
            "enhance_only_temporal_delta_mae": enhance["quality"]["temporal_delta_mae"],
            "generate_lpips_contribution_fixed_route": item["comparisons"][
                "generate_contribution_on_fixed_joint_route"]["lpips_delta"],
            "enhance_lpips_contribution_fixed_route": item["comparisons"][
                "enhancement_contribution_on_fixed_joint_route"]["lpips_delta"],
            "full_decode_seconds": joint["runtime"]["full_decode_component_sum_seconds"],
            "peak_cuda_allocated_bytes": joint["runtime"][
                "full_decode_peak_cuda_allocated_bytes"],
            "action_visual": str(item["_path"].parent / "visuals" / "joint_action_map.png"),
            "base": {
                "bytes": base["rate"]["total_bytes"], **base["quality"]},
            "all_generate": {
                "bytes": all_generate["rate"]["total_bytes"],
                **all_generate["quality"],
            },
        })

    transitions = []
    for first, second in zip(summaries, summaries[1:]):
        before = first["variants"]["joint-three-path-oracle"]["actions"]
        after = second["variants"]["joint-three-path-oracle"]["actions"]
        transitions.append({
            "from_budget_ratio": first["configuration"][
                "enhance_budget_ratio_of_base_stream"],
            "to_budget_ratio": second["configuration"][
                "enhance_budget_ratio_of_base_stream"],
            "changed_tiles": sum(a != b for a, b in zip(before, after)),
            "transitions": transition_counts(before, after),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_visual = args.output_dir / "visuals" / "budget_action_maps.png"
    curve_visual = args.output_dir / "visuals" / "perceptual_rd_curve.png"
    action_map_sheet(action_visual, rows)
    plot_curves(curve_visual, rows, baselines)
    write_csv(args.output_dir / "budget_points.csv", rows)
    result = {
        "experiment": "E21 budget-aware three-path summary",
        "status": "no-training-oracle-budget-sweep-complete",
        "protocol": {
            "sequence": signature[0],
            "frames": signature[1],
            "base_qp": signature[2],
            "enhance_qp": signature[3],
            "tile_size": signature[4],
            "max_generate_tiles": signature[5],
            "fixed_decoder_feather_pixels": signature[6],
            "metric_priority": "LPIPS first; temporal diagnostic; PSNR report only",
        },
        "budget_points": [
            {key: value for key, value in row.items()
             if key not in ("base", "all_generate", "action_visual")}
            for row in rows
        ],
        "action_transitions": transitions,
        "dense_scalar_qp_baselines": baselines,
        "visuals": {
            "budget_action_maps": str(action_visual),
            "perceptual_rd_curve": str(curve_visual),
        },
        "interpretation": (
            "The legal independent-tile fallback has a low-rate perceptual "
            "window, but loses to ordinary scalar-QP coding at the highest "
            "tested enhancement budget.  This motivates one-shot spatial QP "
            "syntax instead of more fallback-tile tuning."),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    report = [
        "# E21 预算自适应三路小验证", "",
        "本页只汇总已经完成的合法码流结果，不读取源图。LPIPS 为主指标，时序误差为诊断，PSNR 仅报告。", "",
        "| 增强预算 | 实际字节 | B/G/E | LPIPS | 最近普通QP | 字节差 | LPIPS差 |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['budget_ratio']:.2f}× | {row['bytes']} | "
            f"{row['base_tiles']}/{row['generate_tiles']}/{row['enhance_tiles']} | "
            f"{row['lpips']:.6f} | {row['nearest_scalar']} | "
            f"{row['bytes_vs_scalar']:+d} | {row['lpips_delta_vs_scalar']:+.6f} |")
    report += [
        "", "负的 LPIPS 差表示三路更好。低预算窗口成立，但最高预算点被普通 QP29 反超。",
        "这说明下一步应减少独立 tile 重复开销并实现真正的一次编码空间质量图，而不是继续调 tile fallback。",
    ]
    (args.output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(args.output_dir / "summary.json"),
        "action_visual": str(action_visual),
        "curve_visual": str(curve_visual),
        "budget_points": result["budget_points"],
        "transitions": transitions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
