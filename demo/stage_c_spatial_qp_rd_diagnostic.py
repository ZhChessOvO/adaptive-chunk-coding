#!/usr/bin/env python3
"""Plot existing codec-only LPIPS rate-distortion curves.

The plot is intentionally a development diagnostic.  It reuses the actual
streams from the frozen/finetuned endpoint comparison and the three weight
interpolation runs; it does not encode, decode, or evaluate any new video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FRAME_COUNT = 17
WIDTH = 512
HEIGHT = 512
TAGS = ("alpha000", "alpha025", "alpha050", "alpha075", "alpha100")
LABELS = {
    "alpha000": "DCVC-UF frozen (alpha=0)",
    "alpha025": "v1 interpolation alpha=0.25",
    "alpha050": "v1 interpolation alpha=0.50",
    "alpha075": "v1 interpolation alpha=0.75",
    "alpha100": "v1 finetuned (alpha=1)",
}
COLORS = {
    "alpha000": "#111827",
    "alpha025": "#2f6f9f",
    "alpha050": "#2a9d8f",
    "alpha075": "#e9a03b",
    "alpha100": "#d1495b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-summary", type=Path, required=True)
    parser.add_argument("--interpolation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actual_path(recorded: str) -> Path:
    path = Path(recorded)
    if path.exists():
        return path
    prefix = "/autodl-fs/data/"
    if recorded.startswith(prefix):
        translated = Path("/root/autodl-fs") / recorded[len(prefix):]
        if translated.exists():
            return translated
    raise FileNotFoundError(recorded)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def collect_records(endpoint: dict, interpolation: dict) -> list[dict]:
    records = []
    for record in endpoint["records"]:
        if record["family"] != "uniform-regression":
            continue
        tag = {"frozen": "alpha000", "tuned": "alpha100"}.get(
            record["model_role"])
        if tag is None:
            raise ValueError(f"unknown endpoint model role: {record['model_role']}")
        records.append({**record, "tag": tag})
    for record in interpolation["intermediate_records"]:
        if record["family"] == "uniform-regression":
            records.append({**record, "tag": record["model_role"]})

    expected = 5 * 6 * 3
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} uniform records, found {len(records)}")
    keys = {
        (record["tag"], record["sample_id"], int(record["qp"]))
        for record in records
    }
    if len(keys) != expected:
        raise RuntimeError("uniform record keys are not unique")
    for record in records:
        if not record["fresh_decode_pixel_exact"]:
            raise RuntimeError(f"fresh decode failed for {record['task_id']}")
        stream = actual_path(record["stream"])
        if stream.stat().st_size != int(record["stream_bytes"]):
            raise RuntimeError(f"stream byte mismatch for {record['task_id']}")
    return records


def aggregate_points(records: list[dict]) -> dict[str, dict[str, list[dict]]]:
    output: dict[str, dict[str, list[dict]]] = {}
    groups = {
        "Combined (3 REDS + 3 UVG)": {"REDS", "UVG"},
        "REDS (3 clips)": {"REDS"},
        "UVG (3 clips)": {"UVG"},
    }
    for group_name, datasets in groups.items():
        output[group_name] = {}
        for tag in TAGS:
            points = []
            for qp in (8, 16, 32):
                selected = [
                    record for record in records
                    if record["tag"] == tag
                    and record["dataset"] in datasets
                    and int(record["qp"]) == qp
                ]
                expected_count = 6 if len(datasets) == 2 else 3
                if len(selected) != expected_count:
                    raise RuntimeError(
                        f"{group_name}/{tag}/QP{qp}: expected "
                        f"{expected_count}, found {len(selected)}")
                mean_bytes = sum(item["stream_bytes"] for item in selected) / len(selected)
                mean_lpips = sum(
                    item["quality"]["lpips_alex"] for item in selected
                ) / len(selected)
                points.append({
                    "qp": qp,
                    "sample_count": len(selected),
                    "mean_stream_bytes": mean_bytes,
                    "aggregate_bpp": mean_bytes * 8 / (FRAME_COUNT * WIDTH * HEIGHT),
                    "mean_lpips_alex": mean_lpips,
                })
            output[group_name][tag] = points
    return output


def plot(points: dict[str, dict[str, list[dict]]], output: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
    })
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for axis, (group_name, curves) in zip(axes, points.items()):
        for tag in TAGS:
            curve = curves[tag]
            axis.plot(
                [point["aggregate_bpp"] for point in curve],
                [point["mean_lpips_alex"] for point in curve],
                color=COLORS[tag],
                marker="o",
                markersize=5.5 if tag == "alpha000" else 4.5,
                linewidth=2.7 if tag == "alpha000" else 1.7,
                alpha=1.0 if tag in ("alpha000", "alpha100") else 0.9,
                label=LABELS[tag],
                zorder=5 if tag == "alpha000" else 3,
            )
            for point in curve:
                axis.annotate(
                    f"{point['qp']}",
                    (point["aggregate_bpp"], point["mean_lpips_alex"]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color=COLORS[tag], alpha=0.82,
                )
        axis.set_title(group_name, weight="bold")
        axis.set_xlabel("Actual bitrate (bits / pixel)")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        axis.text(
            0.03, 0.04, "better  ↙", transform=axis.transAxes,
            fontsize=9, color="#374151",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db"},
        )
    axes[0].set_ylabel("LPIPS-Alex (lower is better)")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.95)
    figure.suptitle(
        "Existing codec-only RD curves: frozen DCVC-UF vs spatial-QP v1 variants",
        fontsize=15, weight="bold",
    )
    figure.text(
        0.5, -0.035,
        "Development diagnostic only · uniform QP 8/16/32 · actual stream bytes "
        "· every point fresh-decoded",
        ha="center", fontsize=9, color="#4b5563",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    endpoint_path = args.endpoint_summary.resolve()
    interpolation_path = args.interpolation_summary.resolve()
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    interpolation = json.loads(interpolation_path.read_text(encoding="utf-8"))
    if endpoint.get("status") != "complete" or interpolation.get("status") != "complete":
        raise RuntimeError("source evaluations are not complete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_records(endpoint, interpolation)
    points = aggregate_points(records)
    image_path = args.output_dir / "spatial_qp_existing_rd_diagnostic.png"
    plot(points, image_path)
    summary = {
        "experiment": "existing spatial-QP codec RD diagnostic",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development visualization; not a paper figure or new evaluation",
        "sources": {
            "endpoint_summary": str(endpoint_path),
            "endpoint_summary_sha256": sha256(endpoint_path),
            "interpolation_summary": str(interpolation_path),
            "interpolation_summary_sha256": sha256(interpolation_path),
        },
        "protocol": {
            "new_encoding_or_decoding": False,
            "uniform_qp_values": [8, 16, 32],
            "frame_count": FRAME_COUNT,
            "resolution": [WIDTH, HEIGHT],
            "rate": "actual stream bytes converted to bits per pixel",
            "distortion": "mean LPIPS-Alex; lower is better",
            "sample_pools": {
                "combined": "the same 3 REDS and 3 UVG clips used by v1 interpolation",
                "per_dataset": "3 clips",
            },
        },
        "curves": points,
        "verification": {
            "record_count": len(records),
            "all_recorded_stream_sizes_match_files": True,
            "all_fresh_decodes_pixel_exact": True,
        },
        "image": str(image_path.resolve()),
        "image_bytes": image_path.stat().st_size,
        "image_sha256": sha256(image_path),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
