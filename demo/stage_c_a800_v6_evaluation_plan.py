#!/usr/bin/env python3
"""Freeze the minimal real-evaluation task set for the four v6 ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LOGICAL_VARIANTS = (
    "baseline-v5-feather16",
    "adaptation-only",
    "spatial-only",
    "combined",
)
TARGET_VARIANTS = LOGICAL_VARIANTS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--baseline-routes", type=Path, required=True)
    parser.add_argument("--adaptation-routes", type=Path, required=True)
    parser.add_argument("--spatial-routes", type=Path, required=True)
    parser.add_argument("--combined-routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def load_routes(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    manifest = read(path.resolve())
    entries = manifest.get("entries", [])
    if manifest.get("sample_count") != 37 or len(entries) != 37:
        raise RuntimeError(f"route manifest is incomplete: {path}")
    output = {}
    for entry in entries:
        route_path = Path(entry["path"]).resolve()
        route = read(route_path)
        sample_id = route["sample"]["sample_id"]
        configuration = route["configuration"]
        if (
            configuration.get("tile_size") != 128
            or configuration.get("tile_grid") != [4, 4]
            or configuration.get("quality_profile")
            != {"Generate": 8, "Base": 16, "Enhance": 32}
        ):
            raise RuntimeError(
                f"route configuration differs for {sample_id} in {path}")
        selected_name = route["selected_variant"]
        selected = route["variants"][selected_name]
        actions = tuple(map(int, selected["actions"]))
        if len(actions) != 16 or any(action not in (0, 1, 2) for action in actions):
            raise RuntimeError(f"invalid actions for {sample_id} in {path}")
        if sample_id in output:
            raise RuntimeError(f"duplicate route sample: {sample_id}")
        output[sample_id] = {
            "route": str(route_path),
            "selected_variant": selected_name,
            "actions": list(actions),
            "action_key": "".join(map(str, actions)),
            "configuration": configuration,
        }
    if set(output) != expected_ids:
        raise RuntimeError(f"route and sample IDs differ: {path}")
    return output


def main() -> None:
    args = parse_args()
    sample_path = args.sample_manifest.resolve()
    samples = read_jsonl(sample_path)
    if len(samples) != 37:
        raise RuntimeError(f"expected 37 samples, found {len(samples)}")
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("sample manifest contains duplicate IDs")
    expected_ids = set(sample_ids)

    manifest_paths = {
        "baseline-v5-feather16": args.baseline_routes.resolve(),
        "adaptation-only": args.adaptation_routes.resolve(),
        "spatial-only": args.spatial_routes.resolve(),
        "combined": args.combined_routes.resolve(),
    }
    routes = {
        name: load_routes(path, expected_ids)
        for name, path in manifest_paths.items()
    }

    records = []
    tasks = []
    canonical_counts = Counter()
    for sample in samples:
        sample_id = sample["sample_id"]
        baseline = routes["baseline-v5-feather16"][sample_id]
        action_owner = {
            baseline["action_key"]: "baseline-v5-feather16",
        }
        variants = {
            "baseline-v5-feather16": {
                **baseline,
                "canonical_variant": "baseline-v5-feather16",
                "requires_new_codec_or_seedvr2": False,
            },
        }
        for name in TARGET_VARIANTS:
            current = routes[name][sample_id]
            canonical = action_owner.get(current["action_key"])
            if canonical is None:
                canonical = name
                action_owner[current["action_key"]] = name
                tasks.append({
                    "sample_id": sample_id,
                    "dataset": sample["dataset"],
                    "sequence": sample["sequence"],
                    "variant": name,
                    "route": current["route"],
                    "seed": int(sample["seed"]),
                    "action_key": current["action_key"],
                })
            variants[name] = {
                **current,
                "canonical_variant": canonical,
                "requires_new_codec_or_seedvr2": canonical == name,
            }
            canonical_counts[canonical] += 1
        records.append({
            "sample_id": sample_id,
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "data_role": sample["data_role"],
            "seed": int(sample["seed"]),
            "variants": variants,
            "unique_action_map_count_including_baseline": len(action_owner),
        })

    value = {
        "experiment": "A800 v6 minimal exact-action quality evaluation plan",
        "status": "frozen-before-quality-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_manifest": str(sample_path),
        "sample_manifest_sha256": sha256(sample_path),
        "route_manifests": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in manifest_paths.items()
        },
        "logical_variants": list(LOGICAL_VARIANTS),
        "sample_count": len(records),
        "logical_target_evaluations": len(records) * len(TARGET_VARIANTS),
        "new_real_evaluation_task_count": len(tasks),
        "reused_target_evaluation_count": (
            len(records) * len(TARGET_VARIANTS) - len(tasks)),
        "reuse_rule": (
            "Within the same sample and SeedVR2 seed, routes with identical "
            "16-tile actions have the same spatial-QP stream, connected ROIs, "
            "and restored output. Reuse is allowed only for an exact action "
            "key match; controller scores alone never trigger reuse."),
        "canonical_reuse_counts": dict(canonical_counts),
        "samples": records,
        "tasks": tasks,
        "scientific_boundary": {
            "result_driven_task_exclusion": False,
            "exact_action_match_required_for_reuse": True,
            "same_sample_and_seed_required_for_reuse": True,
            "baseline_feather16_is_frozen_existing_formal_output": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
            "single_gpu": True,
        },
    }
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "sample_count": value["sample_count"],
        "logical_target_evaluations": value["logical_target_evaluations"],
        "new_real_evaluation_task_count": value[
            "new_real_evaluation_task_count"],
        "reused_target_evaluation_count": value[
            "reused_target_evaluation_count"],
        "canonical_reuse_counts": value["canonical_reuse_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
