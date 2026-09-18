#!/usr/bin/env python3
"""Prepare combined REDS+UVG labels and compare domain-adapted routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ACTION_NAMES = ("Base", "Generate", "Enhance")
UVG_TRAIN_SEQUENCES = {
    "Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry",
}
UVG_HOLDOUT_SEQUENCES = {"ReadySetGo", "YachtRide"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    merge = commands.add_parser("merge-labels")
    merge.add_argument("--reds-quality", type=Path, required=True)
    merge.add_argument("--reds-roi-cost", type=Path, required=True)
    merge.add_argument("--uvg-quality", type=Path, required=True)
    merge.add_argument("--uvg-roi-cost", type=Path, required=True)
    merge.add_argument("--v5-selection", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)

    compare = commands.add_parser("compare-routes")
    compare.add_argument("--sample-manifest", type=Path, required=True)
    compare.add_argument("--baseline-routes", type=Path, required=True)
    compare.add_argument("--candidate-routes", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_manifest(path: Path, expected_count: int, kind: str) -> dict:
    value = read(path)
    if not value.get("complete"):
        raise RuntimeError(f"{kind} manifest is incomplete: {path}")
    if value.get("completed_sample_count") != expected_count:
        raise RuntimeError(
            f"{kind} count differs: {value.get('completed_sample_count')}")
    entries = value.get("entries", [])
    if len(entries) != expected_count:
        raise RuntimeError(f"{kind} entry count differs")
    ids = [item["sample_id"] for item in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{kind} has duplicate sample IDs")
    for item in entries:
        sample_path = Path(item["path"])
        if not sample_path.is_file() or sample_path.stat().st_size == 0:
            raise RuntimeError(f"{kind} sample is missing: {sample_path}")
        sample = read(sample_path)
        if sample["sample"]["sample_id"] != item["sample_id"]:
            raise RuntimeError(f"{kind} sample identity differs")
    return value


def sequence_for_entry(entry: dict) -> str:
    return str(entry["sequence"])


def merge_labels(args: argparse.Namespace) -> None:
    reds_quality = validate_manifest(args.reds_quality.resolve(), 500, "REDS quality")
    reds_roi = validate_manifest(args.reds_roi_cost.resolve(), 500, "REDS ROI")
    uvg_quality = validate_manifest(args.uvg_quality.resolve(), 60, "UVG quality")
    uvg_roi = validate_manifest(args.uvg_roi_cost.resolve(), 60, "UVG ROI")

    reds_quality_ids = {item["sample_id"] for item in reds_quality["entries"]}
    reds_roi_ids = {item["sample_id"] for item in reds_roi["entries"]}
    uvg_quality_ids = {item["sample_id"] for item in uvg_quality["entries"]}
    uvg_roi_ids = {item["sample_id"] for item in uvg_roi["entries"]}
    if reds_quality_ids != reds_roi_ids:
        raise RuntimeError("REDS quality and ROI sample IDs differ")
    if uvg_quality_ids != uvg_roi_ids:
        raise RuntimeError("UVG quality and ROI sample IDs differ")
    if reds_quality_ids & uvg_quality_ids:
        raise RuntimeError("REDS and UVG sample IDs overlap")

    uvg_sequences = {
        sequence_for_entry(item) for item in uvg_quality["entries"]
    }
    if uvg_sequences != UVG_TRAIN_SEQUENCES:
        raise RuntimeError(f"UVG training sequence set differs: {uvg_sequences}")
    if uvg_sequences & UVG_HOLDOUT_SEQUENCES:
        raise RuntimeError("UVG hold-out entered training labels")
    uvg_counts = Counter(
        sequence_for_entry(item) for item in uvg_quality["entries"])
    if uvg_counts != Counter({name: 12 for name in UVG_TRAIN_SEQUENCES}):
        raise RuntimeError(f"UVG per-sequence label counts differ: {uvg_counts}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_quality_path = output_dir / "combined_quality_manifest.json"
    combined_roi_path = output_dir / "combined_roi_cost_manifest.json"
    fixed_selection_path = output_dir / "fixed_v5_selection.json"

    quality_entries = reds_quality["entries"] + uvg_quality["entries"]
    roi_entries = reds_roi["entries"] + uvg_roi["entries"]
    quality = {
        "experiment": "A800 REDS plus UVG adaptation quality labels",
        "protocol_version": reds_quality["protocol_version"],
        "requested_sample_count": 560,
        "completed_sample_count": 560,
        "complete": True,
        "feature_names": reds_quality["feature_names"],
        "entries": quality_entries,
        "sources": {
            "REDS": str(args.reds_quality.resolve()),
            "UVG": str(args.uvg_quality.resolve()),
        },
        "dataset_sample_counts": {"REDS": 500, "UVG": 60},
        "scientific_boundary": {
            "manifests_merged_without_copying_labels": True,
            "uvg_is_cross_domain_adaptation_not_independent_evidence": True,
            "readysetgo_and_yachtride_excluded": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
    }
    roi = {
        "experiment": "A800 REDS plus UVG adaptation ROI-cost labels",
        "protocol_version": reds_roi["protocol_version"],
        "requested_sample_count": 560,
        "completed_sample_count": 560,
        "complete": True,
        "measurement_classes": reds_roi["measurement_classes"],
        "entries": roi_entries,
        "sources": {
            "REDS": str(args.reds_roi_cost.resolve()),
            "UVG": str(args.uvg_roi_cost.resolve()),
        },
        "dataset_sample_counts": {"REDS": 500, "UVG": 60},
        "scientific_boundary": quality["scientific_boundary"],
    }
    atomic_json(combined_quality_path, quality)
    atomic_json(combined_roi_path, roi)

    old_selection = read(args.v5_selection.resolve())
    if old_selection["selection_policy"].get("hard_promotion_gate") is not False:
        raise RuntimeError("source v5 selection unexpectedly has a hard gate")
    fixed_selection = {
        "experiment": "A800 UVG adaptation with fixed v5 selection",
        "status": "fixed-before-domain-adapted-expert-training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": {
            "hard_promotion_gate": False,
            "method_and_hyperparameters_reused_from_frozen_v5": True,
            "uvg_labels_used_for_selection": False,
        },
        "selected_candidate": old_selection["selected_candidate"],
        "source_v5_selection": {
            "path": str(args.v5_selection.resolve()),
            "sha256": sha256(args.v5_selection.resolve()),
        },
        "scientific_boundary": {
            **old_selection["scientific_boundary"],
            "uvg_adaptation_changes_training_data_and_residual_experts_only": True,
            "v1_anchor_remains_the_frozen_reds_checkpoint": True,
            "v5_confidence_calibration_and_budgets_remain_fixed": True,
            "readysetgo_and_yachtride_excluded_from_training": True,
        },
    }
    atomic_json(fixed_selection_path, fixed_selection)

    summary = {
        "experiment": "A800 REDS plus UVG adaptation label merge",
        "status": "complete",
        "sample_count": 560,
        "region_count": 560 * 16,
        "dataset_sample_counts": {"REDS": 500, "UVG": 60},
        "uvg_training_sequences": sorted(UVG_TRAIN_SEQUENCES),
        "uvg_holdout_sequences": sorted(UVG_HOLDOUT_SEQUENCES),
        "uvg_fraction_of_samples": 60 / 560,
        "combined_quality_manifest": str(combined_quality_path),
        "combined_quality_manifest_sha256": sha256(combined_quality_path),
        "combined_roi_cost_manifest": str(combined_roi_path),
        "combined_roi_cost_manifest_sha256": sha256(combined_roi_path),
        "fixed_selection": str(fixed_selection_path),
        "fixed_selection_sha256": sha256(fixed_selection_path),
        "scientific_boundary": fixed_selection["scientific_boundary"],
    }
    atomic_json(output_dir / "merge_summary.json", summary)
    print(json.dumps(summary, indent=2))


def route_map(manifest_path: Path) -> dict[str, list[int]]:
    manifest = read(manifest_path)
    if manifest.get("sample_count") != len(manifest.get("entries", [])):
        raise RuntimeError(f"route manifest count differs: {manifest_path}")
    result = {}
    for entry in manifest["entries"]:
        value = read(Path(entry["path"]))
        variant = value["selected_variant"]
        actions = value["variants"][variant]["actions"]
        if len(actions) != 16 or any(action not in (0, 1, 2) for action in actions):
            raise RuntimeError(f"invalid route actions: {entry['sample_id']}")
        result[entry["sample_id"]] = actions
    return result


def generate_boundary_edges(actions: list[int]) -> int:
    value = 0
    for index, action in enumerate(actions):
        row, column = divmod(index, 4)
        if column < 3:
            neighbor = actions[index + 1]
            value += int((action == 1) != (neighbor == 1))
        if row < 3:
            neighbor = actions[index + 4]
            value += int((action == 1) != (neighbor == 1))
    return value


def summarize_route_rows(rows: list[dict]) -> dict:
    action_counts_before = Counter()
    action_counts_after = Counter()
    for row in rows:
        action_counts_before.update(row["baseline_action_names"])
        action_counts_after.update(row["candidate_action_names"])
    return {
        "sample_count": len(rows),
        "samples_with_any_action_change": sum(
            row["changed_tile_count"] > 0 for row in rows),
        "changed_tile_count": sum(row["changed_tile_count"] for row in rows),
        "generate_tiles_added": sum(row["generate_tiles_added"] for row in rows),
        "generate_tiles_removed": sum(
            row["generate_tiles_removed"] for row in rows),
        "generate_boundary_edges_before": sum(
            row["generate_boundary_edges_before"] for row in rows),
        "generate_boundary_edges_after": sum(
            row["generate_boundary_edges_after"] for row in rows),
        "action_counts_before": {
            name: action_counts_before[name] for name in ACTION_NAMES
        },
        "action_counts_after": {
            name: action_counts_after[name] for name in ACTION_NAMES
        },
    }


def compare_routes(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.sample_manifest.resolve())
    sample_by_id = {item["sample_id"]: item for item in samples}
    baseline = route_map(args.baseline_routes.resolve())
    candidate = route_map(args.candidate_routes.resolve())
    if set(sample_by_id) != set(baseline) or set(sample_by_id) != set(candidate):
        raise RuntimeError("sample and route ID sets differ")

    rows = []
    for sample_id in sample_by_id:
        sample = sample_by_id[sample_id]
        before = baseline[sample_id]
        after = candidate[sample_id]
        removed = sum(a == 1 and b != 1 for a, b in zip(before, after))
        added = sum(a != 1 and b == 1 for a, b in zip(before, after))
        rows.append({
            "sample_id": sample_id,
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "data_role": sample["data_role"],
            "changed_tile_count": sum(a != b for a, b in zip(before, after)),
            "generate_tiles_added": added,
            "generate_tiles_removed": removed,
            "generate_boundary_edges_before": generate_boundary_edges(before),
            "generate_boundary_edges_after": generate_boundary_edges(after),
            "baseline_actions": before,
            "candidate_actions": after,
            "baseline_action_names": [ACTION_NAMES[action] for action in before],
            "candidate_action_names": [ACTION_NAMES[action] for action in after],
        })

    groups = {
        "combined": rows,
        "REDS": [row for row in rows if row["dataset"] == "REDS"],
        "UVG": [row for row in rows if row["dataset"] == "UVG"],
        "UVG-adaptation-sequences": [
            row for row in rows
            if row["dataset"] == "UVG" and row["sequence"] in UVG_TRAIN_SEQUENCES
        ],
        "UVG-v6-holdout-sequences": [
            row for row in rows
            if row["dataset"] == "UVG" and row["sequence"] in UVG_HOLDOUT_SEQUENCES
        ],
    }
    value = {
        "experiment": "A800 route-only effect of UVG-adapted residual experts",
        "status": "complete",
        "primary_warning": (
            "Route-only comparison is diagnostic; it is not a quality result. "
            "Changed routes require real spatial-QP and SeedVR2 evaluation."
        ),
        "groups": {
            name: summarize_route_rows(members) for name, members in groups.items()
        },
        "samples": rows,
        "scientific_boundary": {
            "readysetgo_and_yachtride_not_used_for_v6_training": True,
            "all_uvg_first_windows_were_previously_seen_in_v5_evaluation": True,
            "no_quality_claim_from_route_only_comparison": True,
        },
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps(value["groups"], indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "merge-labels":
        merge_labels(args)
    else:
        compare_routes(args)


if __name__ == "__main__":
    main()
