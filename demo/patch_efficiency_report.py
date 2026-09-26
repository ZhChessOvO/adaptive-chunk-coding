"""Separate lossless framing savings, padding changes, and learned RD changes."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.feature_head_report import boundary_panel
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import load_source, resources
from demo.scalable_format import parse

ROOT = Path("/root/autodl-fs/DCVC/runs")


def read(p):
    return json.loads(p.read_text())


def accounting(path, point):
    data = path.read_bytes()
    c = parse(data)
    if len(data) != point["bytes"] or point["fresh_decode"]["output_hash"] != read(path.with_suffix("") / "decode.json")["output_hash"]:
        raise RuntimeError("stream/result mismatch")
    import struct
    z = sum(struct.unpack_from("<I", p.payload)[0] for p in c.packets)
    payload = sum(len(p.payload) for p in c.packets)
    fields = dict(base=len(c.base), framing=len(data)-len(c.base)-payload+4*len(c.packets),
                  hyper=z, main=payload-z-4*len(c.packets))
    assert sum(fields.values()) == len(data)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "a800_patch_efficiency_20260926")
    args = parser.parse_args()
    root = args.root
    reference = ROOT / "a800_feature_head_20260926"
    inputs = {"original": reference / "evaluation/summary.json",
              "pad16": root / "padding_evaluation/summary.json"}
    for value in (4, 2):
        p = root / f"evaluation_l{value}/summary.json"
        if p.exists():
            inputs[f"train_l{value}"] = p
    data = {k:read(p)["results"] for k,p in inputs.items()}
    compact = read(root / "compact/summary.json")["results"]
    native = read(reference / "comparison.json")["results"]
    keys = ["q2", "q1", "q0.5"]
    labels = {"uf":"Native UF (whole frame)", "original":"Feature patch: original",
              "compact":"Same pixels, compact header", "pad16":"Compact + pad16, no training",
              "train_l4":"Pad16 adapted, loss weight 4", "train_l2":"Pad16 adapted, loss weight 2"}
    rows = []
    for i, original in enumerate(data["original"]):
        sid = original["sample"]["sample_id"]
        row = dict(sample=original["sample"], rois=original["rois"], uf=native[i]["uf"])
        costs = {}
        for name, series in data.items():
            result = series[i]
            assert result["sample"] == original["sample"] and result["rois"] == original["rois"]
            row[name] = {q:result["points"][q] for q in keys}
            for q in keys:
                actual = accounting(inputs[name].parent / sid / f"{q}.acse", row[name][q])
                assert row[name][q]["fresh_decode"]["base_hash"] == original["points"][q]["fresh_decode"]["base_hash"]
                if q == "q1":
                    costs[name] = actual
        row["compact"] = {}
        for q in keys:
            record = next(r for r in compact if r["sample_id"] == sid and r["quality"] == q)
            assert record["pixels_identical"] and record["payloads_identical"]
            row["compact"][q] = dict(row["original"][q], bytes=record["new_bytes"], fresh_decode=record["fresh_decode"])
        costs["compact"] = accounting(root / "compact" / sid / "q1.acse", row["compact"]["q1"])
        row["q1_byte_breakdown"] = costs
        source = load_source(original["sample"])
        frames = {"Source":source[8]}
        for name in data:
            with np.load(inputs[name].parent / sid / "q1/reconstruction.npz") as f:
                frames[labels[name]] = f["reconstruction"][8].copy()
        historical = ROOT / "a800_scalable_mechanism_20260926/samples" / sid
        frames["Native UF32"] = np.asarray(Image.open(sorted((historical / "uf32_fresh").glob("*.png"))[8]).convert("RGB"))
        boundary_panel(root / f"boundary_{sid}.png", frames, original["rois"][0])
        rows.append(row)
        print(json.dumps({"sample":sid, "q1":{k:{"bytes":row[k]["q1"]["bytes"],
                 "roi_psnr":row[k]["q1"]["roi_psnr"], "lpips":row[k]["q1"]["quality"]["lpips_alex"]}
                 for k in labels if k != "uf" and k in row}}))
    ordered = [k for k in labels if k in rows[0]]
    for metric, axis_label, filename in ((lambda p:p["roi_psnr"], "Selected ROI PSNR (dB)", "roi_rd.png"),
             (lambda p:p["quality"]["lpips_alex"], "Whole-frame LPIPS (lower is better)", "lpips_rd.png")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax,row in zip(axes.flat, rows):
            for k in ordered:
                pts = sorted(row[k].values(), key=lambda p:p["bytes"])
                ax.plot([p["bytes"]/1000 for p in pts], [metric(p) for p in pts],
                        marker="o" if k == "uf" else ".", linestyle="--" if k in ("original","compact") else "-", label=labels[k])
            ax.set(title=row["sample"]["sample_id"], xlabel="All transmitted kB", ylabel=axis_label)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle("Fixed 2 ROIs / first 17 frames; previously used dev clips; no whole-frame superiority claim")
        fig.tight_layout()
        fig.savefig(root / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax,row in zip(axes.flat, rows):
        names = [k for k in ordered if k != "uf"]
        bottom = np.zeros(len(names))
        for field in ("base", "framing", "hyper", "main"):
            values = np.array([row["q1_byte_breakdown"][k][field]/1000 for k in names])
            ax.bar(names, values, bottom=bottom, label=field)
            bottom += values
        ax.set(title=row["sample"]["sample_id"], ylabel="Actual kB, q=1")
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "byte_breakdown.png", dpi=150)
    plt.close(fig)
    schedule_equal = None
    if "train_l4" in data and "train_l2" in data:
        schedules = [[json.loads(s) for s in (root / f"train_l{w}/metrics.jsonl").read_text().splitlines()] for w in (4,2)]
        # Training logs can contain abandoned steps after a hard reboot. Use
        # the last observation per step, corresponding to resumed replay.
        schedules = [dict((r["step"],r) for r in s) for s in schedules]
        fields = ("step", "dataset", "sample_id", "roi", "start", "count", "qstep")
        schedule_equal = schedules[0].keys() == schedules[1].keys() and all(
            all(schedules[0][i][k] == schedules[1][i][k] for k in fields) for i in schedules[0])
        if not schedule_equal:
            raise RuntimeError("paired training schedules differ")
    atomic_json(root / "comparison.json", dict(results=rows, training_schedules_equal=schedule_equal,
                sources={str(p):file_hash(p) for p in inputs.values()}, resources=resources(),
                data_role="previously used development", labels=labels))


if __name__ == "__main__":
    main()
