#!/usr/bin/env python3
"""Safely reuse compatible formal outputs in the joint evaluation tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-kind", choices=("v1", "v5"), required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--current-route", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--v1-followup-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def actions(path: Path) -> list[int]:
    value = read(path)
    selected = value["selected_variant"]
    return value["variants"][selected]["actions"]


def ensure_link(target: Path, link: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"reuse link points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing reuse path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    args = parse_args()
    sample_id = args.sample_id
    sample_root = args.sample_root.resolve()
    sample_root.mkdir(parents=True, exist_ok=True)
    reused: dict[str, str] = {}
    external_route = None

    if sample_id.startswith("dev-s"):
        baseline_sample = (
            args.baseline_root.resolve() / "formal" / "development" / sample_id)
        controls = baseline_sample / "spatial"
        if args.controller_kind == "v1":
            final_sample = (
                args.v1_followup_root.resolve() / "formal" /
                "development" / sample_id)
            external_route = final_sample / "routes" / "learned-joint.json"
        else:
            final_sample = (
                args.v5_root.resolve() / "formal" / "development" / sample_id)
            external_route = final_sample / "routes" / "learned-joint.json"
        final_spatial = final_sample / "spatial"
    elif sample_id.startswith("test-s"):
        previous_sample = (
            args.independent_root.resolve() / "formal" / "test" / sample_id)
        controls = previous_sample / "spatial"
        final_spatial = previous_sample / "spatial" if (
            args.controller_kind == "v1") else None
        if final_spatial is not None:
            external_route = (
                previous_sample / "routes" / "low" / "learned-joint.json")
    else:
        controls = None
        final_spatial = None

    if controls is not None:
        for name in ("all-generate", "enhance-only"):
            target = controls / name
            ensure_link(target, sample_root / "spatial" / name)
            reused[name] = str(target.resolve())

    if final_spatial is not None:
        mapping = {
            "final-joint": "low-joint",
            "final-no-generate": "low-no-generate",
            "final-no-enhance": "low-no-enhance",
        }
        for destination, source in mapping.items():
            target = final_spatial / source
            ensure_link(target, sample_root / "spatial" / destination)
            reused[destination] = str(target.resolve())
        assert external_route is not None
        if actions(args.current_route) != actions(external_route):
            raise RuntimeError(
                f"frozen route differs from reusable output for {sample_id}")

    result = {
        "sample_id": sample_id,
        "controller_kind": args.controller_kind,
        "reused": bool(reused),
        "reused_variants": reused,
        "external_route_verified_equal": external_route is not None,
        "external_route": (
            str(external_route.resolve()) if external_route is not None else None),
    }
    path = sample_root / "reuse.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
