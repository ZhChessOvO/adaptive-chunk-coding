#!/usr/bin/env python3
"""Rate-match a routed Stage-1 point against all-Base QP points.

Interpolation is linear in PSNR and logarithmic in actual total bpp.  This is
the minimum fair check needed before interpreting a same-QP byte reduction as
an RD improvement.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", action="append", required=True, metavar="QP=ROOT",
        help="Root produced by stage1_multichunk_oracle.py at an all-Base QP.")
    parser.add_argument("--routed-root", required=True)
    parser.add_argument("--drop-target", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_baselines(items):
    baselines = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"baseline must be QP=ROOT, got {item!r}")
        qp_text, root_text = item.split("=", 1)
        qp = int(qp_text)
        root = Path(root_text)
        if qp in baselines:
            raise ValueError(f"duplicate QP {qp}")
        if not (root / "summary.json").is_file():
            raise FileNotFoundError(root / "summary.json")
        baselines[qp] = root
    if len(baselines) < 2:
        raise ValueError("at least two all-Base QP points are required")
    return baselines


def load_sequence(root, name):
    return json.loads((root / name / "summary.json").read_text(encoding="utf-8"))


def interpolate_log_rate(points, quality):
    points = sorted(points, key=lambda point: point[1])
    for lower, upper in zip(points, points[1:]):
        if lower[1] <= quality <= upper[1]:
            span = upper[1] - lower[1]
            if span <= 0:
                raise ValueError("baseline quality points must be strictly ordered")
            weight = (quality - lower[1]) / span
            log_rate = (
                math.log(lower[2])
                + weight * (math.log(upper[2]) - math.log(lower[2])))
            return math.exp(log_rate), lower[0], upper[0], weight
    return None


def main():
    args = parse_args()
    baselines = parse_baselines(args.baseline)
    routed_root = Path(args.routed_root)
    routed_index = json.loads(
        (routed_root / "summary.json").read_text(encoding="utf-8"))
    names = [item["sequence"] for item in routed_index["sequences"]]
    rows = []
    for name in names:
        routed = load_sequence(routed_root, name)
        policy = next(
            item for item in routed["policies"]
            if math.isclose(item["drop_target_db"], args.drop_target, abs_tol=1e-9))
        for metric in ("mean_frame_psnr", "aggregate_psnr"):
            base_points = []
            for qp, root in baselines.items():
                data = load_sequence(root, name)
                base_points.append((
                    qp,
                    data["baseline"]["metrics"][metric],
                    data["baseline"]["stream"]["total_bpp"],
                ))
            quality = policy["metrics"][metric]
            routed_bpp = policy["stream"]["total_bpp"]
            interpolation = interpolate_log_rate(base_points, quality)
            if interpolation is None:
                rows.append({
                    "sequence": name,
                    "metric": metric,
                    "routed_quality": quality,
                    "routed_bpp": routed_bpp,
                    "matched_all_base_bpp": "",
                    "matched_rate_change_percent": "",
                    "lower_qp": "",
                    "upper_qp": "",
                    "interpolation_weight": "",
                    "status": "outside_baseline_quality_range",
                })
                continue
            matched_bpp, lower_qp, upper_qp, weight = interpolation
            rows.append({
                "sequence": name,
                "metric": metric,
                "routed_quality": quality,
                "routed_bpp": routed_bpp,
                "matched_all_base_bpp": matched_bpp,
                "matched_rate_change_percent": 100.0 * (routed_bpp / matched_bpp - 1.0),
                "lower_qp": lower_qp,
                "upper_qp": upper_qp,
                "interpolation_weight": weight,
                "status": "interpolated",
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rd_matched.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "linear quality / logarithmic actual-total-bpp interpolation",
        "drop_target_db": args.drop_target,
        "baseline_roots": {str(qp): str(root) for qp, root in baselines.items()},
        "routed_root": str(routed_root),
        "same_qp_reduction_is_not_used_as_rd_evidence": True,
        "rows": rows,
    }
    (output_dir / "rd_matched.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        if row["metric"] == "mean_frame_psnr":
            print(
                f"{row['sequence']}: matched rate change "
                f"{row['matched_rate_change_percent']:+.3f}%")


if __name__ == "__main__":
    main()
