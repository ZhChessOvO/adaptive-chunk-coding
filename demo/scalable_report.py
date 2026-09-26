#!/usr/bin/env python3
"""Summarize the integer-transform mechanism baseline, never a learned result."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.scalable_codec import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    run = json.loads((root / "summary.json").read_text())
    if not run["complete"] or run["mode"] != "mechanism" or run["sample_count"] != 4:
        raise ValueError("expected the complete four-sample mechanism run")
    records = []
    details = []
    names = ["base", "a1", "a1_b1", "a2_b1", "two_levels"]
    for item in run["results"]:
        sample = item["sample"]
        for name, values in item["variants"].items():
            records.append({"sample": sample["sample_id"], "dataset": sample["dataset"],
                            "variant": name, "bytes": values["bytes"], "bpp": values["bpp"],
                            **values["quality"], "roi_mse": values.get("roi_mse")})
        vs = item["variants"]
        details.append({
            "sample": sample["sample_id"], "dataset": sample["dataset"],
            "sequence": sample["sequence"], "frames": sample["frame_count"],
            "width": sample["crop"]["width"], "height": sample["crop"]["height"],
            "base_bytes": vs["base"]["bytes"], "base_native_bytes": vs["base"]["base_stream_bytes"],
            "level1_bytes": vs["a1_b1"]["bytes"], "level2_bytes": vs["two_levels"]["bytes"],
            "uf32_bytes": vs["uf32"]["bytes"],
            "level2_to_uf32_byte_ratio": vs["two_levels"]["bytes"] / vs["uf32"]["bytes"],
            "roi_psnr": [10 * math.log10(255**2 / vs[k]["roi_mse"])
                         for k in ("base", "a1_b1", "two_levels")],
            "global_psnr": [vs[k]["quality"]["psnr_db"]
                            for k in ("base", "a1_b1", "two_levels", "uf32")],
            "lpips": [vs[k]["quality"]["lpips_alex"]
                      for k in ("base", "a1_b1", "two_levels", "uf32")],
            "all_prefixes_roi_mse_decreased": all(
                vs[b]["roi_mse"] < vs[a]["roi_mse"] for a, b in zip(names, names[1:])),
            "seconds": item["seconds"],
        })
    output = root / "report"
    output.mkdir(exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Two native QP points are reference markers, not a matched-rate BD result.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for col, item in enumerate(run["results"]):
        vs, sample = item["variants"], item["sample"]
        bpp = [vs[k]["bpp"] for k in names]
        native_x = [vs["base"]["base_stream_bytes"] * 8 /
                    (sample["frame_count"] * sample["crop"]["width"] * sample["crop"]["height"]),
                    vs["uf32"]["bpp"]]
        for row, key, ylabel in ((0, "psnr_db", "Global PSNR (higher is better)"),
                                 (1, "lpips_alex", "LPIPS (lower is better)")):
            axis = axes[row, col]
            axis.plot(bpp, [vs[k]["quality"][key] for k in names],
                      "o-", label="Same-stream residual prefixes")
            axis.scatter(native_x, [vs["base"]["quality"][key], vs["uf32"]["quality"][key]],
                         marker="x", s=75, color="tab:red", label="Native UF QP8 / QP32 references")
            axis.set_xscale("log")
            axis.set_xlabel("Actual bpp (log scale)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.set_title(f"{sample['dataset']} {sample['sequence']} | "
                           f"{sample['frame_count']} frames | "
                           f"{sample['crop']['width']}x{sample['crop']['height']}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Mechanism baseline only: fixed ROIs, integer residuals; NOT a learned-codec result",
                 fontsize=13)
    fig.savefig(output / "prefix_diagnostic.png", dpi=140)
    plt.close(fig)
    summary = {
        "status": "mechanism demonstrated; residual efficiency not competitive",
        "sample_count": 4, "fresh_decode_variants_per_sample": 8,
        "assertions_all_passed": all(
            r["prefix_and_fresh_decode_passed"] and r["non_enhanced_pixels_exact"]
            and r["missing_and_partial_packets_passed"] for r in run["results"]),
        "learned_enhancement_trained": False, "router_used": False, "generation_used": False,
        "caveat": "Fixed 2/16 regions, first 17 frames only; QP32 is full-frame. "
                  "Not an equal-rate comparison, not a final method, not independent evaluation.",
        "details": details, "resource_snapshot": run["resources"],
        "measured_sample_seconds_sum": sum(r["seconds"] for r in run["results"]),
        "peak_fresh_decoder_cuda_allocated_bytes": max(
            report["peak_cuda_allocated_bytes"] for r in run["results"]
            for report in r["decode_reports"].values()),
    }
    atomic_json(output / "mechanism_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
