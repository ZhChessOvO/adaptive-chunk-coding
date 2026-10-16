"""Sender helpers. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
import random
import numpy as np
import torch
import routervc.routers.sender as sender
from routervc.utils.precision import codec_precision

ARMS = ("source", "zero_source")
STATES = ("empty", "partial")


def semantic_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple((cpu_tree(item) for item in value))
    return value


def capture_rng(device="cpu"):
    numpy = np.random.get_state()
    return dict(
        python=random.getstate(),
        torch=torch.get_rng_state().clone(),
        numpy=dict(
            name=numpy[0],
            keys=torch.from_numpy(numpy[1].astype(np.int64)),
            position=numpy[2],
            has_gauss=numpy[3],
            cached_gauss=numpy[4],
        ),
        cuda=(
            [v.cpu().clone() for v in torch.cuda.get_rng_state_all()]
            if str(device).startswith("cuda")
            else []
        ),
    )


def restore_rng(value, device="cpu"):
    random.setstate(value["python"])
    numpy = value["numpy"]
    np.random.set_state(
        (
            numpy["name"],
            numpy["keys"].cpu().numpy().astype(np.uint32),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gauss"],
        )
    )
    torch.set_rng_state(value["torch"].cpu())
    if str(device).startswith("cuda"):
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device topology changed")
        torch.cuda.set_rng_state_all([v.cpu() for v in value["cuda"]])
    elif value["cuda"]:
        raise ValueError("cannot exactly resume a CUDA training checkpoint on CPU")


@contextmanager
def deterministic_training(seed, device):
    """Use deterministic FP32 for CPU AND GPU; restore the caller's policy/RNG."""
    if str(device).startswith("cuda") and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in (":4096:8", ":16:8"):
        raise ValueError(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the worker before CUDA initialization"
        )
    previous = capture_rng(device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        with codec_precision():
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        restore_rng(previous, device)


def split_rows(protocol):
    rows = protocol["rows"]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("unique nonempty sender sample ledger required")
    if any(
        (
            r["dataset"] not in ("REDS", "UVG")
            or r["router_split"] not in ("train", "validation")
            for r in rows
        )
    ):
        raise ValueError("unknown sender dataset/split")
    train = [r for r in rows if r["router_split"] == "train"]
    valid = (
        rows
        if protocol["smoke"]
        else [r for r in rows if r["router_split"] == "validation"]
    )
    if not train or not valid:
        raise ValueError("nonempty sender train/validation sets required")
    return (train, valid)


def arm_batch(data, arm, device):
    if arm not in ARMS or set(data["inputs"]) != sender.INPUT_KEYS:
        raise ValueError("unknown sender arm or inputs")
    target = data["targets"]
    if (
        set(target) != {"value", "weight", "label_scope"}
        or target["label_scope"] != sender.LABEL_SCOPE
        or target["value"].shape != (2, 16)
        or (target["weight"].shape != (2, 16))
    ):
        raise ValueError("two states of scalar FINAL marginal targets are required")
    value, weight = (target["value"], target["weight"])
    known = weight > 0
    if (
        not torch.isfinite(weight).all()
        or torch.any(weight < 0)
        or (not torch.isfinite(value[known]).all())
        or (not torch.all(known.sum(1) == 3))
    ):
        raise ValueError(
            "exactly three finite measured candidates per state are required"
        )
    costs = data["packet_bytes"]
    if (
        costs.dtype != torch.int64
        or costs.shape != (2, 16)
        or torch.any(costs <= 0)
        or (not torch.equal(costs, data["inputs"]["packet_bytes"].squeeze(-1)))
    ):
        raise ValueError("candidate costs must be actual positive integer packet bytes")
    coverage = data["inputs"]["coverage"]
    if coverage.shape != (2, 16, 1) or torch.any(coverage.squeeze(-1)[known] != 0):
        raise ValueError("measured additions must be unreceived candidates")
    plans = data["plans"]
    if len(plans) != 2 or plans[0]["selected"] or (not plans[1]["selected"]):
        raise ValueError("expected empty and nonempty fixed teacher states")
    for state, plan in enumerate(plans):
        if sorted(plan["candidates"]) != torch.where(known[state])[0].tolist():
            raise ValueError("known candidate mask differs from measured teacher plans")
        if (
            sorted(plan["selected"])
            != torch.where(coverage[state, :, 0] == 1)[0].tolist()
        ):
            raise ValueError(
                "received input coverage differs from measured teacher state"
            )
    inputs = {key: tensor.to(device) for key, tensor in data["inputs"].items()}
    if arm == "zero_source":
        inputs["global_video"] = inputs["global_video"].clone()
        inputs["local_video"] = inputs["local_video"].clone()
        inputs["global_video"][:, :3] = 0
        inputs["global_video"][:, 12:18] = 0
        inputs["local_video"][:, :, :3] = 0
        inputs["local_video"][:, :, 12:18] = 0
    return dict(
        inputs=inputs,
        targets=dict(
            value=value.to(device),
            weight=weight.to(device),
            label_scope=sender.LABEL_SCOPE,
        ),
        packet_bytes=costs.to(device),
    )


def fit_scale(protocol, get, check=lambda: None):
    """Never inspect validation labels when fitting the common final-Delta scale."""
    train, _ = split_rows(protocol)
    observations, ledger = ([], [])
    for row in train:
        check()
        batch = arm_batch(get(row), "source", "cpu")
        known = batch["targets"]["weight"] > 0
        values = batch["targets"]["value"][known].double().tolist()
        observations.extend(values)
        ledger.append(
            dict(
                sample_id=row["sample_id"],
                values=values,
                known=known.nonzero().tolist(),
            )
        )
    values = np.asarray(observations, dtype=np.float64)
    raw = float(np.quantile(np.abs(values), 0.75, method="linear"))
    return dict(
        format="sender_train_final_delta_scale_v1",
        value=max(raw, 1e-06),
        statistic="linear 0.75 quantile of abs(measured signed FINAL LPIPS delta)",
        raw_quantile=raw,
        floor=1e-06,
        floor_applied=raw < 1e-06,
        train_samples=[r["sample_id"] for r in train],
        measured_marginals=len(values),
        observations_sha256=semantic_hash(ledger),
        label_scope=sender.LABEL_SCOPE,
        negative=int((values < 0).sum()),
        positive=int((values > 0).sum()),
        zero=int((values == 0).sum()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        loss_only_normalization=True,
        validation_used=False,
        receiver_scale_reused=False,
    )


def measured_regret(prediction, value, weight, packet_bytes):
    """Positive top1 or no-add among known candidates; density uses mean-byte units."""
    prediction, value, weight, costs = map(
        np.asarray, (prediction, value, weight, packet_bytes)
    )
    if (
        any((v.shape != (16,) for v in (prediction, value, weight, costs)))
        or not np.isfinite(prediction).all()
        or (not np.isfinite(weight).all())
        or np.any(weight < 0)
        or (costs.dtype.kind not in "iu")
        or np.any(costs[weight > 0] <= 0)
        or (not np.isfinite(value[weight > 0]).all())
        or ((weight > 0).sum() != 3)
    ):
        raise ValueError(
            "three measured candidates with finite predictions and actual byte costs required"
        )
    known = np.where(weight > 0)[0].tolist()
    reference = float(costs[known].mean())
    true = {i: float(value[i]) * reference / int(costs[i]) for i in known}
    predicted = {i: float(prediction[i]) * reference / int(costs[i]) for i in known}
    positive = [i for i in known if predicted[i] > 0]
    chosen = min(positive, key=lambda i: (-predicted[i], i)) if positive else None
    positives_true = [i for i in known if true[i] > 0]
    best = min(positives_true, key=lambda i: (-true[i], i)) if positives_true else None
    picked = true[chosen] if chosen is not None else 0.0
    reference_gain = true[best] if best is not None else 0.0
    pairs = [
        (i, j)
        for n, i in enumerate(known)
        for j in known[n + 1 :]
        if true[i] != true[j]
    ]
    concordant = sum(
        (
            (
                1.0
                if (predicted[i] - predicted[j]) * (true[i] - true[j]) > 0
                else 0.5 if predicted[i] == predicted[j] else 0.0
            )
            for i, j in pairs
        )
    )
    return dict(
        regret=reference_gain - picked,
        density_regret_per_byte=(reference_gain - picked) / reference,
        selected_final_lpips_gain=float(value[chosen]) if chosen is not None else 0.0,
        best_density_candidate_final_lpips_gain=(
            float(value[best]) if best is not None else 0.0
        ),
        selected_normalized_density=picked,
        best_measured_normalized_density=reference_gain,
        pair_accuracy=concordant / len(pairs) if pairs else 1.0,
        ranked_pairs=len(pairs),
        no_add=float(chosen is None),
        harmful_add=float(chosen is not None and value[chosen] < 0),
        selected_region=chosen,
        best_measured_region=best,
        measured_candidates=3,
    )


SCORE_KEYS = (
    "loss",
    "regret",
    "density_regret_per_byte",
    "selected_final_lpips_gain",
    "best_density_candidate_final_lpips_gain",
    "selected_normalized_density",
    "best_measured_normalized_density",
    "pair_accuracy",
    "ranked_pairs",
    "no_add",
    "harmful_add",
)


@torch.no_grad()
def validate(
    model, arm, rows, get, scale, device, check=lambda: None, ranking_weight=0.1
):
    model.eval()
    groups = {}
    for row in rows:
        check()
        batch = arm_batch(get(row), arm, device)
        with codec_precision():
            predicted = model(batch["inputs"])
            states = {}
            for index, name in enumerate(STATES):
                target = {
                    key: value[index : index + 1] if torch.is_tensor(value) else value
                    for key, value in batch["targets"].items()
                }
                loss, _ = sender.training_loss(
                    predicted[index : index + 1],
                    target,
                    batch["packet_bytes"][index : index + 1],
                    gain_scale=scale,
                    ranking_weight=ranking_weight,
                )
                scores = measured_regret(
                    predicted[index].cpu().numpy(),
                    target["value"][0].cpu().numpy(),
                    target["weight"][0].cpu().numpy(),
                    batch["packet_bytes"][index].cpu().numpy(),
                )
                states[name] = dict(loss=float(loss), **scores)
        groups.setdefault(row["dataset"], []).append(states)
    if not groups:
        raise ValueError("empty sender validation")
    by_dataset = {}
    for dataset, samples in groups.items():
        by_state = {
            name: {
                key: float(np.mean([s[name][key] for s in samples]))
                for key in SCORE_KEYS
            }
            for name in STATES
        }
        by_dataset[dataset] = dict(
            samples=len(samples),
            by_state=by_state,
            **{
                key: float(np.mean([v[key] for v in by_state.values()]))
                for key in SCORE_KEYS
            },
        )
    by_state = {
        name: {
            key: float(np.mean([v["by_state"][name][key] for v in by_dataset.values()]))
            for key in SCORE_KEYS
        }
        for name in STATES
    }
    return dict(
        groups=by_dataset,
        by_state=by_state,
        **{
            key: float(np.mean([v[key] for v in by_state.values()]))
            for key in SCORE_KEYS
        },
        scope="dataset-equal and state-equal top1/no-add among three measured candidates; NOT full oracle or RD",
    )
