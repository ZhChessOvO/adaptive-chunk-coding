#!/usr/bin/env python3
"""Materialize fixed controls and same-route ablations for the A800 pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_NAMES = {
    ACTION_BASE: "Base",
    ACTION_GENERATE: "Generate",
    ACTION_ENHANCE: "Enhance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build learned-route controls without using ground truth")
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-variant",
        help=("Use one named budget variant instead of route-summary's "
              "selected_variant.  This projects an already-computed route; "
              "it does not consult ground-truth metrics."))
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def counts(actions: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(actions == action))
        for action, name in ACTION_NAMES.items()
    }


def route_document(
    source: dict, source_path: Path, name: str, actions: np.ndarray,
    route_kind: str, description: str, source_variant: str,
) -> dict:
    source_selected = source["variants"][source_variant]
    return {
        "experiment": "A800 formal route control",
        "sample": source.get("sample"),
        "route_kind": route_kind,
        "selected_variant": name,
        "configuration": source["configuration"],
        "variants": {
            name: {
                "actions": actions.astype(int).tolist(),
                "action_counts": counts(actions),
                "description": description,
                "source_selected_variant_metadata": {
                    key: value for key, value in source_selected.items()
                    if key != "actions"
                },
            }
        },
        "provenance": {
            "source_route_summary": str(source_path.resolve()),
            "source_selected_variant": source_variant,
            "ground_truth_metrics_used_to_construct_control": False,
        },
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.route_summary.read_text(encoding="utf-8"))
    selected = args.source_variant or source["selected_variant"]
    if selected not in source["variants"]:
        raise ValueError(f"route variant does not exist: {selected}")
    learned = np.asarray(
        source["variants"][selected]["actions"], dtype=np.uint8)
    if learned.shape != (16,) or np.any(learned > ACTION_ENHANCE):
        raise ValueError("expected exactly 16 actions inside {Base, Generate, Enhance}")

    variants = {
        "learned-joint": (
            learned,
            "learned-controller",
            "Unchanged selected controller route.",
        ),
        "same-route-no-generate": (
            np.where(learned == ACTION_GENERATE, ACTION_BASE, learned),
            "learned-controller-ablation",
            "Same learned route, with Generate cells changed to Base.",
        ),
        "same-route-no-enhance": (
            np.where(learned == ACTION_ENHANCE, ACTION_BASE, learned),
            "learned-controller-ablation",
            "Same learned route, with Enhance cells changed to Base.",
        ),
        "all-generate": (
            np.full(16, ACTION_GENERATE, dtype=np.uint8),
            "fixed-control",
            "All 16 cells use Generate.",
        ),
        "enhance-only": (
            np.full(16, ACTION_ENHANCE, dtype=np.uint8),
            "fixed-control",
            "All 16 cells use Enhance; no restoration is run.",
        ),
        "all-base": (
            np.full(16, ACTION_BASE, dtype=np.uint8),
            "fixed-control",
            "All 16 cells use Base.",
        ),
    }
    manifest_entries = []
    for name, (actions, kind, description) in variants.items():
        path = args.output_dir / f"{name}.json"
        document = route_document(
            source, args.route_summary, name, actions, kind, description,
            selected)
        atomic_json(path, document)
        manifest_entries.append({
            "name": name,
            "path": str(path.resolve()),
            "action_counts": counts(actions),
        })

    manifest = {
        "experiment": "A800 formal route controls and ablations",
        "source_route_summary": str(args.route_summary.resolve()),
        "source_selected_variant": selected,
        "entries": manifest_entries,
        "scientific_boundary": {
            "learned_route_changed_only_by_declared_ablation": True,
            "ground_truth_used": False,
            "same_quality_profile": True,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
