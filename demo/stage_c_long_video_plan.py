#!/usr/bin/env python3
"""Build a continuous spatial-QP plan from overlapping 17-frame routes.

The router still sees the same 17-frame problem used during development.  A
long video is covered by router windows starting at 0, 16, 32, ... .  The
first route controls the leading I frame and the first two P8 units; every
later route controls the next two P8 units.  DCVC-UF is then encoded once, so
its reference state remains continuous across all of those units.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.stage_c_spatial_quality_codec import (
    load_action_units,
    selected_route_variant,
)


P_CHUNK = 8
ROUTER_WINDOW = 17
ROUTER_STRIDE = 16
TEMPORAL_VARIANT = "temporal-17-frame-route-v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def parse_route_argument(value: str) -> tuple[int, Path]:
    try:
        start_text, path_text = value.split("=", 1)
        start = int(start_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "window routes must use START=/path/to/route.json") from error
    if start < 0 or start % ROUTER_STRIDE:
        raise argparse.ArgumentTypeError(
            f"route start must be a nonnegative multiple of {ROUTER_STRIDE}")
    path = Path(path_text).expanduser()
    return start, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a continuous long-video route and source gate")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--source-start", type=int, default=0)
    build.add_argument("--frame-count", type=int, required=True)
    build.add_argument("--sequence-name", required=True)
    build.add_argument("--source-role", required=True)
    build.add_argument("--crop-x", type=int, required=True)
    build.add_argument("--crop-y", type=int, required=True)
    build.add_argument("--width", type=int, required=True)
    build.add_argument("--height", type=int, required=True)
    build.add_argument(
        "--window-route", action="append", type=parse_route_argument,
        required=True, metavar="START=PATH",
        help="One 17-frame router result for each start 0,16,32,...")
    build.add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "build":
        if args.source_start < 0:
            parser.error("--source-start must be nonnegative")
        if args.frame_count < 1 or (args.frame_count - 1) % P_CHUNK:
            parser.error("--frame-count must be 1 plus an integer number of P8 units")
        if min(args.width, args.height) <= 0:
            parser.error("crop dimensions must be positive")
    return args


def required_route_starts(frame_count: int) -> list[int]:
    p_units = (frame_count - 1) // P_CHUNK
    return [index * ROUTER_STRIDE for index in range((p_units + 1) // 2)]


def build_unit_routes(
    frame_count: int,
    window_actions: dict[int, list[int]],
    window_sources: dict[int, dict],
) -> list[dict]:
    required = required_route_starts(frame_count)
    if sorted(window_actions) != required:
        raise ValueError(
            f"router starts must be exactly {required}, got {sorted(window_actions)}")
    units = [{
        "unit_index": 0,
        "type": "I",
        "frame_start": 0,
        "frame_count": 1,
        "router_window_start": 0,
        "router_window_frame_count": ROUTER_WINDOW,
        "actions": window_actions[0],
        **window_sources[0],
    }]
    p_units = (frame_count - 1) // P_CHUNK
    for p_index in range(p_units):
        route_start = (p_index // 2) * ROUTER_STRIDE
        units.append({
            "unit_index": p_index + 1,
            "type": "P8",
            "frame_start": 1 + p_index * P_CHUNK,
            "frame_count": P_CHUNK,
            "router_window_start": route_start,
            "router_window_frame_count": ROUTER_WINDOW,
            "actions": window_actions[route_start],
            **window_sources[route_start],
        })
    return units


def action_counts(actions: list[int]) -> dict[str, int]:
    return {
        name: actions.count(index)
        for index, name in enumerate(("Base", "Generate", "Enhance"))
    }


def read_source_frames(args: argparse.Namespace) -> list[Path]:
    candidates = sorted(args.source_dir.expanduser().resolve().glob("*.png"))
    selected = candidates[
        args.source_start:args.source_start + args.frame_count]
    if len(selected) != args.frame_count:
        raise ValueError(
            f"{args.source_dir} provides {len(selected)} of "
            f"{args.frame_count} requested frames")
    for path in selected:
        with Image.open(path) as image:
            width, height = image.size
        if (args.crop_x + args.width > width
                or args.crop_y + args.height > height):
            raise ValueError(f"crop falls outside source frame {path}")
    return selected


def build_main(args: argparse.Namespace) -> None:
    route_arguments = dict(args.window_route)
    if len(route_arguments) != len(args.window_route):
        raise ValueError("duplicate --window-route start")
    required = required_route_starts(args.frame_count)
    if sorted(route_arguments) != required:
        raise ValueError(
            f"expected router windows at {required}, got {sorted(route_arguments)}")

    routes = {}
    window_actions = {}
    window_sources = {}
    common_configuration = None
    route_windows = []
    for relative_start in required:
        path = route_arguments[relative_start].expanduser().resolve()
        route = json.loads(path.read_text(encoding="utf-8"))
        configuration = route["configuration"]
        if common_configuration is None:
            common_configuration = configuration
        elif configuration != common_configuration:
            raise ValueError("all window routes must use the same configuration")
        variant_name, variant = selected_route_variant(route)
        actions = list(map(int, variant["actions"]))
        rows, columns = map(int, configuration["tile_grid"])
        if len(actions) != rows * columns:
            raise ValueError(f"route {path} action count differs from its grid")
        sample = route.get("sample", {})
        recorded_start = sample.get("frame_start")
        expected_absolute_start = args.source_start + relative_start
        if (recorded_start is not None
                and int(recorded_start) != expected_absolute_start):
            raise ValueError(
                f"route {path} starts at {recorded_start}, expected "
                f"{expected_absolute_start}")
        source = {
            "source_route": str(path),
            "source_route_kind": route.get("route_kind"),
            "source_selected_variant": variant_name,
        }
        routes[relative_start] = route
        window_actions[relative_start] = actions
        window_sources[relative_start] = source
        route_windows.append({
            "relative_frame_start": relative_start,
            "absolute_source_frame_start": expected_absolute_start,
            "frame_count": ROUTER_WINDOW,
            "route_summary": str(path),
            "route_kind": route.get("route_kind"),
            "selected_variant": variant_name,
            "actions": actions,
            "action_counts": action_counts(actions),
        })

    assert common_configuration is not None
    source_files = read_source_frames(args)
    units = build_unit_routes(
        args.frame_count, window_actions, window_sources)
    first_actions = units[0]["actions"]
    transition_count = sum(
        first["actions"] != second["actions"]
        for first, second in zip(units, units[1:]))
    sample = {
        "sample_id": f"{args.sequence_name}-long-s{args.source_start:04d}",
        "dataset": routes[0].get("sample", {}).get("dataset", "unspecified"),
        "split": routes[0].get("sample", {}).get("split", "unspecified"),
        "data_role": "development-mechanism-smoke",
        "source_role": args.source_role,
        "sequence": args.sequence_name,
        "source_dir": str(args.source_dir.expanduser().resolve()),
        "frame_start": args.source_start,
        "frame_count": args.frame_count,
        "source_files": [str(path) for path in source_files],
        "crop": {
            "x": args.crop_x,
            "y": args.crop_y,
            "width": args.width,
            "height": args.height,
        },
    }
    route_summary = {
        "experiment": "continuous long-video 17-frame router assembly",
        "route_kind": "time-varying-coding-unit-route-from-overlapping-windows",
        "selected_variant": TEMPORAL_VARIANT,
        "configuration": common_configuration,
        "sample": sample,
        "variants": {
            TEMPORAL_VARIANT: {
                "method": "17-frame-router-window-to-continuous-P8-plan",
                "actions": first_actions,
                "action_counts": action_counts(first_actions),
            },
        },
        "router_windows": route_windows,
        "coding_unit_routes": units,
        "time_varying_action_maps": transition_count > 0,
        "coding_unit_action_transition_count": transition_count,
        "scientific_boundary": {
            "router_is_run_on_17_frame_windows": True,
            "router_window_stride_frames": ROUTER_STRIDE,
            "codec_reference_state_resets_between_windows": False,
            "one_continuous_codec_stream": True,
            "source_rgb_sent_to_decoder": False,
            "training_or_finetuning": False,
        },
    }
    gate_summary = {
        "experiment": "continuous long-video source gate",
        "sequence": args.sequence_name,
        "frames": args.frame_count,
        "source_role": args.source_role,
        "source_files": sample["source_files"],
        "crop": sample["crop"],
        "configuration": {
            "qps": sorted(common_configuration["quality_profile"].values()),
            "continuous_codec_reference": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route_path = args.output_dir / "long_route.json"
    gate_path = args.output_dir / "long_gate.json"
    atomic_json(route_path, route_summary)
    atomic_json(gate_path, gate_summary)
    plan = {
        "experiment": "continuous long-video plan",
        "status": "ready-for-codec",
        "route_summary": str(route_path.resolve()),
        "gate_summary": str(gate_path.resolve()),
        "frame_count": args.frame_count,
        "router_window_count": len(route_windows),
        "coding_unit_count": len(units),
        "coding_unit_action_transition_count": transition_count,
        "source_first": str(source_files[0]),
        "source_last": str(source_files[-1]),
    }
    atomic_json(args.output_dir / "plan_summary.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def self_test() -> None:
    window_actions = {
        0: [0, 1, 2, 0],
        16: [2, 1, 0, 2],
    }
    sources = {
        start: {"source_route": f"route-{start}.json"}
        for start in window_actions
    }
    units = build_unit_routes(33, window_actions, sources)
    assert [unit["frame_start"] for unit in units] == [0, 1, 9, 17, 25]
    assert [unit["router_window_start"] for unit in units] == [0, 0, 0, 16, 16]
    assert units[2]["actions"] == window_actions[0]
    assert units[3]["actions"] == window_actions[16]
    assert required_route_starts(17) == [0]
    assert required_route_starts(25) == [0, 16]
    route = {
        "configuration": {
            "tile_size": 64,
            "tile_grid": [2, 2],
            "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
        },
        "selected_variant": TEMPORAL_VARIANT,
        "variants": {
            TEMPORAL_VARIANT: {"actions": window_actions[0]},
        },
        "coding_unit_routes": units,
    }
    loaded, _, _, metadata = load_action_units(
        route, 128, 128, 64, 33)
    assert len(loaded) == len(metadata) == 5
    assert loaded[2].tolist() == [[0, 1], [2, 0]]
    assert loaded[3].tolist() == [[2, 1], [0, 2]]
    print(json.dumps({
        "status": "passed",
        "33_frame_unit_count": len(units),
        "33_frame_router_starts": required_route_starts(33),
        "25_frame_tail_supported": True,
        "codec_temporal_route_loader": True,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
