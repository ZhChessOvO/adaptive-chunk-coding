#!/usr/bin/env python3
"""Describe existing v1/v3 low-budget route disagreements for v4 design."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_a800_controller import atomic_json, read_teacher_rows
from demo.stage_c_a800_low_budget_v2 import grouped_indices
from demo.stage_c_a800_low_budget_v3 import direct_route_utility, direct_targets
from demo.stage_c_three_path_roi_probe import ACTION_ENHANCE, TILE_HEADER


ACTION_NAMES = {0: "Base", 1: "Generate", 2: "Enhance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-teacher-manifest", type=Path, required=True)
    parser.add_argument("--development-roi-cost-manifest", type=Path, required=True)
    parser.add_argument("--v1-routes-root", type=Path, required=True)
    parser.add_argument("--v3-routes-root", type=Path, required=True)
    parser.add_argument("--v1-formal-summary", type=Path, required=True)
    parser.add_argument("--v3-formal-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def route(path: Path) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    variant = value["variants"][value["selected_variant"]]
    actions = np.asarray(variant["actions"], dtype=np.uint8)
    if actions.shape != (16,):
        raise ValueError(f"unexpected route shape in {path}: {actions.shape}")
    return actions


def formal_by_sample(path: Path, variant: str) -> dict[str, dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["sample_id"]: item["variants"][variant]
        for item in value["samples"]
    }


def main() -> None:
    args = parse_args()
    _, indirect, metadata = read_teacher_rows(
        args.development_teacher_manifest,
        args.development_roi_cost_manifest,
    )
    targets = direct_targets(indirect)
    v1_formal = formal_by_sample(args.v1_formal_summary, "low-joint")
    v3_formal = formal_by_sample(args.v3_formal_summary, "low-v3")
    transitions = Counter()
    region_rows = []
    sample_rows = []
    for indices in grouped_indices(metadata):
        sample_id = metadata[int(indices[0])]["sample_id"]
        v1 = route(args.v1_routes_root / f"{sample_id}.json")
        v3 = route(args.v3_routes_root / f"{sample_id}.json")
        sample_targets = targets[indices]
        costs = np.maximum(
            np.rint(sample_targets[:, 2]).astype(np.int64),
            TILE_HEADER.size + 1,
        )
        disagreement_indices = np.flatnonzero(v1 != v3)
        v1_utility = direct_route_utility(v1, sample_targets)
        v3_utility = direct_route_utility(v3, sample_targets)
        sample_rows.append({
            "sample_id": sample_id,
            "disagreement_count": int(len(disagreement_indices)),
            "v1_teacher_utility": v1_utility,
            "v3_teacher_utility": v3_utility,
            "v3_minus_v1_teacher_utility": v3_utility - v1_utility,
            "v1_true_enhance_fallback_bytes": int(
                costs[v1 == ACTION_ENHANCE].sum()),
            "v3_true_enhance_fallback_bytes": int(
                costs[v3 == ACTION_ENHANCE].sum()),
            "v1_actual_stream_bytes": v1_formal[sample_id][
                "actual_on_disk_bytes"],
            "v3_actual_stream_bytes": v3_formal[sample_id][
                "actual_on_disk_bytes"],
            "v1_lpips": v1_formal[sample_id]["quality"]["lpips_alex"],
            "v3_lpips": v3_formal[sample_id]["quality"]["lpips_alex"],
            "v3_minus_v1_lpips": (
                v3_formal[sample_id]["quality"]["lpips_alex"]
                - v1_formal[sample_id]["quality"]["lpips_alex"]),
        })
        for region, (v1_action, v3_action) in enumerate(zip(v1, v3)):
            transitions[(int(v1_action), int(v3_action))] += 1
            if v1_action == v3_action:
                continue
            generate_utility = float(sample_targets[region, 0])
            enhance_utility = float(sample_targets[region, 1])
            utility_by_action = {
                0: 0.0,
                1: generate_utility,
                2: enhance_utility,
            }
            region_rows.append({
                "sample_id": sample_id,
                "region": region,
                "v1_action": ACTION_NAMES[int(v1_action)],
                "v3_action": ACTION_NAMES[int(v3_action)],
                "true_generate_utility": generate_utility,
                "true_enhance_utility": enhance_utility,
                "true_enhance_fallback_bytes": int(costs[region]),
                "v3_minus_v1_teacher_utility": (
                    utility_by_action[int(v3_action)]
                    - utility_by_action[int(v1_action)]),
            })
    effects = np.asarray([
        item["v3_minus_v1_teacher_utility"] for item in region_rows
    ])
    output = {
        "experiment": "existing v1-v3 route disagreement diagnosis for v4",
        "data_role": "REDS val/000..005 development; descriptive only",
        "sample_count": len(sample_rows),
        "region_count": len(metadata),
        "disagreement_count": len(region_rows),
        "transition_counts": {
            f"{ACTION_NAMES[source]}_to_{ACTION_NAMES[target]}": int(count)
            for (source, target), count in sorted(transitions.items())
        },
        "disagreement_effects": {
            "beneficial_v3_count": int(np.count_nonzero(effects > 0)),
            "harmful_v3_count": int(np.count_nonzero(effects < 0)),
            "zero_count": int(np.count_nonzero(effects == 0)),
            "sum_v3_minus_v1_teacher_utility": float(effects.sum()),
        },
        "samples": sample_rows,
        "disagreements": region_rows,
        "interpretation": (
            "Base-probe changes are mixed rather than uniformly harmful; use v1 "
            "as the anchor and learn when context/probe corrections are reliable."),
        "scientific_boundary": {
            "development_quality_used_for_description_only": True,
            "development_quality_not_used_to_fit_v4": True,
            "independent_test_012_023_not_read": True,
            "sealed_024_029_not_read": True,
        },
    }
    atomic_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "disagreement_count": len(region_rows),
        "effects": output["disagreement_effects"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
