#!/usr/bin/env python3
"""Exact spatially regularized routing for a rectangular three-path action map.

The existing low-budget controller predicts an independent utility for Base,
Generate, and Enhance at every tile.  This module keeps those predictions and
all byte/Generate budgets frozen, but solves one joint objective:

    predicted utility - lambda * Generate/non-Generate boundary edges.

Only the Generate boundary is regularized because that is where a generative
ROI is pasted back into a codec reconstruction.  A frontier dynamic program
finds the exact optimum on the declared grid; no greedy smoothing is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


ACTION_BASE = 0
ACTION_GENERATE = 1
ACTION_ENHANCE = 2
ACTION_NAMES = ("Base", "Generate", "Enhance")
DEFAULT_SPATIAL_LAMBDA = 0.004
DEFAULT_GENERATE_BUDGET = 4
# spatial-QP writes a six-byte tile header and at least one payload byte.
MIN_ENHANCE_BYTES = 7
UVG_ADAPTATION_SEQUENCES = {
    "Beauty", "Bosphorus", "HoneyBee", "Jockey", "ShakeNDry",
}
UVG_HOLDOUT_SEQUENCES = {"ReadySetGo", "YachtRide"}
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add an exact Generate-boundary term to frozen routes")
    commands = parser.add_subparsers(dest="command", required=True)

    route = commands.add_parser("route")
    route.add_argument("--input-manifest", type=Path, required=True)
    route.add_argument("--output-dir", type=Path, required=True)
    route.add_argument(
        "--spatial-lambda", type=float, default=DEFAULT_SPATIAL_LAMBDA)
    route.add_argument(
        "--enhance-budget-ratio", type=float,
        help=("Override the selected route's Enhance-byte budget with this "
              "fraction of the predicted all-Enhance fallback cost."))
    route.add_argument(
        "--generate-tile-budget", type=int,
        help="Override the selected route's maximum number of Generate tiles.")
    route.add_argument("--expected-sample-count", type=int)

    sweep = commands.add_parser("sweep")
    sweep.add_argument("--input-manifest", type=Path, required=True)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument(
        "--lambdas", type=float, nargs="+",
        default=(0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008))
    sweep.add_argument(
        "--selected-lambda", type=float, default=DEFAULT_SPATIAL_LAMBDA)
    sweep.add_argument("--expected-sample-count", type=int)

    commands.add_parser("self-test")
    args = parser.parse_args()
    if getattr(args, "spatial_lambda", 0.0) < 0:
        parser.error("spatial lambda must be nonnegative")
    if (
        getattr(args, "enhance_budget_ratio", None) is not None
        and not 0.0 <= args.enhance_budget_ratio <= 1.0
    ):
        parser.error("Enhance budget ratio must be between zero and one")
    if (
        getattr(args, "generate_tile_budget", None) is not None
        and not 0 <= args.generate_tile_budget <= 16
    ):
        parser.error("Generate tile budget must be between zero and 16")
    if args.command == "sweep":
        if any(value < 0 for value in args.lambdas):
            parser.error("sweep lambdas must be nonnegative")
        if args.selected_lambda not in args.lambdas:
            parser.error("selected lambda must be present in the sweep")
        if len(set(args.lambdas)) != len(args.lambdas):
            parser.error("sweep lambdas must be unique")
    return args


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def selected_variant(route: dict) -> tuple[str, dict]:
    name = route["selected_variant"]
    return name, route["variants"][name]


def action_counts(actions: list[int] | tuple[int, ...]) -> dict[str, int]:
    counts = Counter(actions)
    return {name: counts[index] for index, name in enumerate(ACTION_NAMES)}


def generate_boundary_edges(
    actions: list[int] | tuple[int, ...], rows: int = 4, columns: int = 4,
) -> int:
    if len(actions) != rows * columns:
        raise ValueError("action count differs from grid geometry")
    edges = 0
    for index, action in enumerate(actions):
        row, column = divmod(index, columns)
        if column + 1 < columns:
            edges += int(
                (action == ACTION_GENERATE)
                != (actions[index + 1] == ACTION_GENERATE))
        if row + 1 < rows:
            edges += int(
                (action == ACTION_GENERATE)
                != (actions[index + columns] == ACTION_GENERATE))
    return edges


def generate_components(
    actions: list[int] | tuple[int, ...], rows: int = 4, columns: int = 4,
) -> list[list[int]]:
    remaining = {
        index for index, action in enumerate(actions)
        if action == ACTION_GENERATE
    }
    output = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        component = []
        while queue:
            index = queue.popleft()
            component.append(index)
            row, column = divmod(index, columns)
            for next_row, next_column in (
                (row - 1, column), (row + 1, column),
                (row, column - 1), (row, column + 1),
            ):
                neighbor = next_row * columns + next_column
                if (
                    0 <= next_row < rows and 0 <= next_column < columns
                    and neighbor in remaining
                ):
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        output.append(sorted(component))
    return output


def direct_values(route: dict) -> tuple[list[float], list[float], list[int]]:
    predictions = route["direct_predictions"]
    rows, columns = map(int, route["configuration"]["tile_grid"])
    expected = rows * columns
    if len(predictions) != expected or any(len(row) != 3 for row in predictions):
        raise ValueError(
            f"expected {expected} three-target direct predictions")
    generate = [float(row[0]) for row in predictions]
    enhance = [float(row[1]) for row in predictions]
    costs = [max(int(round(float(row[2]))), MIN_ENHANCE_BYTES)
             for row in predictions]
    return generate, enhance, costs


def unary_utility(
    actions: list[int] | tuple[int, ...],
    generate_scores: list[float], enhance_scores: list[float],
) -> float:
    return float(sum(
        generate_scores[index] if action == ACTION_GENERATE
        else enhance_scores[index] if action == ACTION_ENHANCE
        else 0.0
        for index, action in enumerate(actions)
    ))


def better_state(
    candidate: tuple[float, float, int, tuple[int, ...]],
    incumbent: tuple[float, float, int, tuple[int, ...]] | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate[0] > incumbent[0] + EPSILON:
        return True
    if abs(candidate[0] - incumbent[0]) <= EPSILON:
        return candidate[3] < incumbent[3]
    return False


def solve_spatial_actions(
    generate_scores: list[float],
    enhance_scores: list[float],
    enhance_costs: list[int],
    byte_budget: int,
    generate_budget: int,
    spatial_lambda: float,
    rows: int = 4,
    columns: int = 4,
) -> tuple[list[int], dict]:
    """Solve the unary utility plus Generate-boundary objective exactly.

    The frontier mask stores the Generate status of the last ``columns``
    processed cells.  For each mask and Generate count, byte states dominated
    by a cheaper state with at least the same score are removed.  This pruning
    is exact because all remaining byte costs are nonnegative.
    """
    count = rows * columns
    if not (
        len(generate_scores) == len(enhance_scores)
        == len(enhance_costs) == count
    ):
        raise ValueError("score/cost count differs from grid geometry")
    if byte_budget < 0 or generate_budget < 0 or spatial_lambda < 0:
        raise ValueError("budgets and spatial lambda must be nonnegative")
    if any(cost < 0 for cost in enhance_costs):
        raise ValueError("Enhance costs must be nonnegative")

    # (frontier mask, used bytes, used Generate) ->
    # (regularized objective, unary utility, boundary count, actions)
    states: dict[
        tuple[int, int, int], tuple[float, float, int, tuple[int, ...]]
    ] = {(0, 0, 0): (0.0, 0.0, 0, tuple())}
    mask_limit = (1 << columns) - 1

    for index in range(count):
        row, column = divmod(index, columns)
        updated: dict[
            tuple[int, int, int], tuple[float, float, int, tuple[int, ...]]
        ] = {}
        for (mask, used_bytes, used_generate), state in states.items():
            objective, unary, boundaries, actions = state
            choices = [(ACTION_BASE, 0, 0, 0.0)]
            # Preserve the controller's safety semantics: smoothing may move or
            # connect beneficial Generate tiles, but cannot turn a predicted
            # harmful tile into Generate merely to make a prettier shape.
            if (
                used_generate < generate_budget
                and generate_scores[index] > 0
            ):
                choices.append((
                    ACTION_GENERATE, 0, 1, generate_scores[index]))
            if (
                enhance_scores[index] > 0
                and used_bytes + enhance_costs[index] <= byte_budget
            ):
                choices.append((
                    ACTION_ENHANCE, enhance_costs[index], 0,
                    enhance_scores[index]))

            for action, byte_cost, generate_cost, gain in choices:
                is_generate = int(action == ACTION_GENERATE)
                new_edges = 0
                if column > 0:
                    new_edges += int(is_generate != bool(mask & 1))
                if row > 0:
                    new_edges += int(
                        is_generate != bool(mask & (1 << (columns - 1))))
                key = (
                    ((mask << 1) & mask_limit) | is_generate,
                    used_bytes + byte_cost,
                    used_generate + generate_cost,
                )
                candidate = (
                    objective + gain - spatial_lambda * new_edges,
                    unary + gain,
                    boundaries + new_edges,
                    actions + (action,),
                )
                if better_state(candidate, updated.get(key)):
                    updated[key] = candidate

        grouped: dict[
            tuple[int, int],
            list[tuple[int, tuple[float, float, int, tuple[int, ...]]]],
        ] = defaultdict(list)
        for (mask, used_bytes, used_generate), state in updated.items():
            grouped[(mask, used_generate)].append((used_bytes, state))

        states = {}
        for (mask, used_generate), candidates in grouped.items():
            best_objective = -math.inf
            for used_bytes, state in sorted(candidates, key=lambda item: item[0]):
                if state[0] > best_objective + EPSILON:
                    states[(mask, used_bytes, used_generate)] = state
                    best_objective = state[0]

    selected_key, selected = max(
        states.items(),
        key=lambda item: (
            item[1][0], -item[0][1], -item[0][2],
            tuple(-action for action in item[1][3]),
        ),
    )
    actions = list(selected[3])
    components = generate_components(actions, rows, columns)
    diagnostics = {
        "spatial_lambda_per_generate_boundary": spatial_lambda,
        "predicted_unary_utility": selected[1],
        "regularized_objective": selected[0],
        "generate_boundary_edges": selected[2],
        "generate_component_count": len(components),
        "generate_components": components,
        "predicted_used_enhance_fallback_bytes": selected_key[1],
        "used_generate_tiles": selected_key[2],
        "enhance_fallback_byte_budget": byte_budget,
        "generate_tile_budget": generate_budget,
        "exact_frontier_dynamic_program": True,
        "nonpositive_generate_tiles_forbidden": True,
        "nonpositive_enhance_tiles_forbidden": True,
    }
    return actions, diagnostics


def reroute_record(
    route: dict,
    spatial_lambda: float,
    enhance_budget_ratio: float | None = None,
    generate_tile_budget: int | None = None,
) -> tuple[dict, dict]:
    source_name, source = selected_variant(route)
    rows, columns = map(int, route["configuration"]["tile_grid"])
    old_actions = list(map(int, source["actions"]))
    generate_scores, enhance_scores, enhance_costs = direct_values(route)
    budget = source["budget"]
    source_byte_budget = int(budget["enhance_fallback_byte_budget"])
    source_generate_budget = int(budget.get(
        "generate_tile_budget", DEFAULT_GENERATE_BUDGET))
    byte_budget = (
        source_byte_budget if enhance_budget_ratio is None
        else int(round(enhance_budget_ratio * sum(enhance_costs)))
    )
    generate_budget = (
        source_generate_budget if generate_tile_budget is None
        else int(generate_tile_budget)
    )
    new_actions, diagnostics = solve_spatial_actions(
        generate_scores, enhance_scores, enhance_costs, byte_budget,
        generate_budget, spatial_lambda, rows, columns)

    old_unary = unary_utility(old_actions, generate_scores, enhance_scores)
    old_edges = generate_boundary_edges(old_actions, rows, columns)
    old_components = generate_components(old_actions, rows, columns)
    if enhance_budget_ratio is None and generate_tile_budget is None:
        new_name = f"spatial-consistent-lambda-{spatial_lambda:.4f}-v6"
    else:
        ratio_label = (
            "source" if enhance_budget_ratio is None
            else f"{enhance_budget_ratio:.3f}")
        generate_label = (
            "source" if generate_tile_budget is None
            else str(generate_tile_budget))
        new_name = (
            f"spatial-consistent-lambda-{spatial_lambda:.4f}-"
            f"enhance-{ratio_label}-generate-{generate_label}-budget-curve")
    diagnostics.update({
        "source_variant": source_name,
        "source_actions": old_actions,
        "source_predicted_unary_utility": old_unary,
        "source_generate_boundary_edges": old_edges,
        "source_generate_component_count": len(old_components),
        "source_generate_components": old_components,
        "changed_tile_count": sum(
            before != after for before, after in zip(old_actions, new_actions)),
        "predicted_unary_utility_delta_vs_source": (
            diagnostics["predicted_unary_utility"] - old_unary),
        "generate_boundary_edge_delta_vs_source": (
            diagnostics["generate_boundary_edges"] - old_edges),
        "source_enhance_fallback_byte_budget": source_byte_budget,
        "source_generate_tile_budget": source_generate_budget,
        "enhance_budget_ratio_override": enhance_budget_ratio,
        "generate_tile_budget_override": generate_tile_budget,
    })

    output = copy.deepcopy(route)
    output["experiment"] = (
        "A800 v6 exact spatially consistent three-path route")
    output["route_kind"] = (
        "frozen-controller-plus-exact-generate-boundary-regularization")
    output["input_selected_variant"] = source_name
    output["selected_variant"] = new_name
    output["variants"][new_name] = {
        "method": "exact-spatially-regularized-three-path-routing-v6",
        "actions": new_actions,
        "action_counts": action_counts(new_actions),
        "budget": {
            **copy.deepcopy(budget),
            "enhance_fallback_byte_budget": byte_budget,
            "generate_tile_budget": generate_budget,
            "enhance_fallback_byte_ratio": (
                enhance_budget_ratio
                if enhance_budget_ratio is not None
                else budget.get("enhance_fallback_byte_ratio")),
        },
        "diagnostics": diagnostics,
    }
    output["spatial_consistency"] = {
        "objective": (
            "sum predicted per-tile utility minus spatial_lambda times the "
            "number of Generate/non-Generate four-neighbor edges"),
        "spatial_lambda": spatial_lambda,
        "enhance_budget_ratio_override": enhance_budget_ratio,
        "generate_tile_budget_override": generate_tile_budget,
        "solver": "exact frontier dynamic program with safe Pareto pruning",
        "regularized_action": "Generate versus non-Generate only",
    }
    output.setdefault("scientific_boundary", {}).update({
        "controller_predictions_unchanged_by_spatial_solver": True,
        "byte_and_generate_budgets_unchanged": True,
        "ground_truth_metrics_used_for_spatial_route": False,
        "dcvc_uf_frozen": True,
        "seedvr2_frozen": True,
        "spatial_qp_codec_frozen": True,
    })
    comparison = {
        "sample_id": route["sample"]["sample_id"],
        "dataset": route["sample"]["dataset"],
        "sequence": route["sample"]["sequence"],
        "source_variant": source_name,
        "candidate_variant": new_name,
        "source_actions": old_actions,
        "candidate_actions": new_actions,
        "source_action_counts": action_counts(old_actions),
        "candidate_action_counts": action_counts(new_actions),
        **diagnostics,
    }
    return output, comparison


def validate_manifest(path: Path, expected_count: int | None) -> dict:
    manifest = read(path.resolve())
    entries = manifest.get("entries", [])
    if manifest.get("sample_count") != len(entries):
        raise RuntimeError("route manifest sample count differs from entries")
    if expected_count is not None and len(entries) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} routes, found {len(entries)}")
    ids = [entry["sample_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("route manifest contains duplicate sample IDs")
    return manifest


def route_main(args: argparse.Namespace) -> None:
    input_path = args.input_manifest.resolve()
    manifest = validate_manifest(input_path, args.expected_sample_count)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    comparisons = []
    for entry in manifest["entries"]:
        route = read(Path(entry["path"]).resolve())
        output, comparison = reroute_record(
            route, args.spatial_lambda, args.enhance_budget_ratio,
            args.generate_tile_budget)
        sample_id = comparison["sample_id"]
        path = output_dir / f"{sample_id}.json"
        atomic_json(path, output)
        entries.append({
            "sample_id": sample_id,
            "path": str(path),
            "selected_variant": output["selected_variant"],
            "action_counts": action_counts(
                comparison["candidate_actions"]),
            "changed_tile_count": comparison["changed_tile_count"],
            "generate_boundary_edges": comparison[
                "generate_boundary_edges"],
            "generate_component_count": comparison[
                "generate_component_count"],
        })
        comparisons.append(comparison)
        print(json.dumps({
            "stage": "spatial-consistency-route",
            "sample_id": sample_id,
            "changed_tiles": comparison["changed_tile_count"],
            "generate_edges": [
                comparison["source_generate_boundary_edges"],
                comparison["generate_boundary_edges"],
            ],
            "generate_components": [
                comparison["source_generate_component_count"],
                comparison["generate_component_count"],
            ],
        }, ensure_ascii=False), flush=True)

    output_manifest = {
        "experiment": "A800 v6 exact spatially consistent route manifest",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_path),
        "input_manifest_sha256": sha256(input_path),
        "spatial_lambda": args.spatial_lambda,
        "enhance_budget_ratio_override": args.enhance_budget_ratio,
        "generate_tile_budget_override": args.generate_tile_budget,
        "sample_count": len(entries),
        "entries": entries,
        "comparisons": comparisons,
        "scientific_boundary": {
            "controller_predictions_unchanged": True,
            "ground_truth_metrics_used_for_routes": False,
            "single_gpu_required": False,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
    }
    atomic_json(output_dir / "manifest.json", output_manifest)


def group_name(route: dict) -> list[str]:
    sample = route["sample"]
    dataset = sample["dataset"]
    sequence = sample["sequence"]
    groups = ["combined", dataset]
    if dataset == "UVG" and sequence in UVG_ADAPTATION_SEQUENCES:
        groups.append("UVG-adaptation-sequences")
    if dataset == "UVG" and sequence in UVG_HOLDOUT_SEQUENCES:
        groups.append("UVG-v6-holdout-sequences")
    return groups


def summarize_comparisons(rows: list[dict]) -> dict:
    before_counts = Counter()
    after_counts = Counter()
    for row in rows:
        before_counts.update(row["source_actions"])
        after_counts.update(row["candidate_actions"])
    before_unary = sum(row["source_predicted_unary_utility"] for row in rows)
    after_unary = sum(row["predicted_unary_utility"] for row in rows)
    before_edges = sum(row["source_generate_boundary_edges"] for row in rows)
    after_edges = sum(row["generate_boundary_edges"] for row in rows)
    before_components = sum(
        row["source_generate_component_count"] for row in rows)
    after_components = sum(row["generate_component_count"] for row in rows)
    return {
        "sample_count": len(rows),
        "samples_with_action_change": sum(
            row["changed_tile_count"] > 0 for row in rows),
        "changed_tile_count": sum(row["changed_tile_count"] for row in rows),
        "action_counts_before": {
            name: before_counts[index]
            for index, name in enumerate(ACTION_NAMES)
        },
        "action_counts_after": {
            name: after_counts[index]
            for index, name in enumerate(ACTION_NAMES)
        },
        "generate_boundary_edges_before": before_edges,
        "generate_boundary_edges_after": after_edges,
        "generate_boundary_edge_reduction_fraction": (
            (before_edges - after_edges) / before_edges
            if before_edges else 0.0),
        "generate_components_before": before_components,
        "generate_components_after": after_components,
        "predicted_unary_utility_before": before_unary,
        "predicted_unary_utility_after": after_unary,
        "predicted_unary_utility_delta": after_unary - before_unary,
        "predicted_unary_utility_loss_fraction": (
            (before_unary - after_unary) / abs(before_unary)
            if before_unary else 0.0),
    }


def sweep_main(args: argparse.Namespace) -> None:
    input_path = args.input_manifest.resolve()
    manifest = validate_manifest(input_path, args.expected_sample_count)
    routes = [read(Path(entry["path"]).resolve())
              for entry in manifest["entries"]]
    candidates = []
    zero_regression = None
    for spatial_lambda in args.lambdas:
        grouped: dict[str, list[dict]] = defaultdict(list)
        sample_rows = []
        for route in routes:
            _, comparison = reroute_record(route, spatial_lambda)
            sample_rows.append(comparison)
            for name in group_name(route):
                grouped[name].append(comparison)
        if spatial_lambda == 0:
            zero_regression = all(
                row["source_actions"] == row["candidate_actions"]
                for row in sample_rows)
            if not zero_regression:
                raise RuntimeError(
                    "lambda=0 did not reproduce the frozen input action maps")
        candidates.append({
            "spatial_lambda": spatial_lambda,
            "groups": {
                name: summarize_comparisons(rows)
                for name, rows in grouped.items()
            },
            "samples": sample_rows,
        })

    selected = next(
        item for item in candidates
        if item["spatial_lambda"] == args.selected_lambda)
    summary = {
        "experiment": "A800 v6 Generate-boundary regularization sweep",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_path),
        "input_manifest_sha256": sha256(input_path),
        "candidate_lambdas": list(args.lambdas),
        "selected_lambda": args.selected_lambda,
        "selection_policy": {
            "hard_promotion_gate": False,
            "reason": (
                "Reuse the existing 0.004 low-confidence fragment scale as "
                "an explicit per-boundary utility cost; the sweep is a soft "
                "sensitivity check rather than a pass/fail gate."),
        },
        "lambda_zero_exact_action_regression": zero_regression,
        "selected_summary": selected["groups"],
        "candidates": candidates,
        "primary_warning": (
            "Predicted utility and action geometry are diagnostics. Quality "
            "claims require real spatial-QP bytes, fresh decode, and SeedVR2."),
        "scientific_boundary": {
            "ground_truth_quality_not_used_for_lambda_choice": True,
            "controller_predictions_unchanged": True,
            "budgets_unchanged": True,
            "dcvc_uf_frozen": True,
            "seedvr2_frozen": True,
            "spatial_qp_codec_frozen": True,
        },
    }
    atomic_json(args.output.resolve(), summary)
    print(json.dumps({
        "selected_lambda": args.selected_lambda,
        "selected_summary": summary["selected_summary"],
        "lambda_zero_exact_action_regression": zero_regression,
    }, ensure_ascii=False, indent=2))


def brute_force(
    generate: list[float], enhance: list[float], costs: list[int],
    byte_budget: int, generate_budget: int, spatial_lambda: float,
    rows: int, columns: int,
) -> tuple[float, tuple[int, ...]]:
    best: tuple[float, tuple[int, ...]] | None = None
    for actions in itertools.product(range(3), repeat=rows * columns):
        if actions.count(ACTION_GENERATE) > generate_budget:
            continue
        if any(
            action == ACTION_GENERATE and generate[index] <= 0
            for index, action in enumerate(actions)
        ):
            continue
        if any(
            action == ACTION_ENHANCE and enhance[index] <= 0
            for index, action in enumerate(actions)
        ):
            continue
        used_bytes = sum(
            costs[index] for index, action in enumerate(actions)
            if action == ACTION_ENHANCE)
        if used_bytes > byte_budget:
            continue
        objective = (
            unary_utility(actions, generate, enhance)
            - spatial_lambda * generate_boundary_edges(
                actions, rows, columns))
        candidate = (objective, actions)
        if (
            best is None or candidate[0] > best[0] + EPSILON
            or (
                abs(candidate[0] - best[0]) <= EPSILON
                and candidate[1] < best[1]
            )
        ):
            best = candidate
    assert best is not None
    return best


def self_test() -> None:
    randomizer = random.Random(20260920)
    checked = 0
    for rows, columns in ((2, 2), (2, 3)):
        count = rows * columns
        for _ in range(20):
            generate = [randomizer.uniform(-0.02, 0.10) for _ in range(count)]
            enhance = [randomizer.uniform(-0.02, 0.08) for _ in range(count)]
            costs = [randomizer.randint(7, 19) for _ in range(count)]
            budget = randomizer.randint(7, 35)
            generate_budget = randomizer.randint(0, min(3, count))
            spatial_lambda = randomizer.choice((0.0, 0.001, 0.004, 0.01))
            actions, diagnostics = solve_spatial_actions(
                generate, enhance, costs, budget, generate_budget,
                spatial_lambda, rows, columns)
            reference_objective, _ = brute_force(
                generate, enhance, costs, budget, generate_budget,
                spatial_lambda, rows, columns)
            if not math.isclose(
                diagnostics["regularized_objective"], reference_objective,
                rel_tol=0.0, abs_tol=1e-10,
            ):
                raise AssertionError("frontier DP differs from brute force")
            if actions.count(ACTION_GENERATE) > generate_budget:
                raise AssertionError("Generate budget exceeded")
            used_bytes = sum(
                costs[index] for index, action in enumerate(actions)
                if action == ACTION_ENHANCE)
            if used_bytes > budget:
                raise AssertionError("Enhance byte budget exceeded")
            checked += 1
    print(json.dumps({
        "status": "passed",
        "random_bruteforce_cases": checked,
        "solver": "exact frontier dynamic program",
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "route":
        route_main(args)
    elif args.command == "sweep":
        sweep_main(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
