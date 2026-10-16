"""Measure new conditional final-picture labels using frozen partner models.

Inputs are an explicit train/validation frame manifest. Prepared windows are
atomic and independently restartable. Pixel/packet scratch is not an archive.
The compact labels can be used after their source datasets have been unmounted.
"""

import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from routervc.assets import Assets, REPO, code_identity
from routervc.codec import precision, format as packets
from routervc.codec.uf import LPIPSAlex, evaluate_variant
from routervc.data.source import source
from routervc.routers import sender, receiver, routing
from routervc.pipeline import ordering
from routervc.fusion.render import render
from routervc.utils.io import atomic_json, atomic_npz, file_hash
from routervc.utils.records import immutable
from routervc.utils.precision import codec_precision


def proposals(selected, prediction, x, y, costs):
    remaining = [i for i in range(16) if i not in selected]
    if len(remaining) < 3:
        raise ValueError("need three unreceived candidate regions")
    first = min(remaining, key=lambda i: (-float(prediction[i]) / int(costs[i]), i))
    error = []
    for rx, ry, w, h in sender.grid_rois(*x.shape[1:3]):
        delta = (
            x[:, ry : ry + h, rx : rx + w].astype(np.float32)
            - y[:, ry : ry + h, rx : rx + w]
        )
        error.append(float(np.mean(delta * delta)))
    second = min(
        (i for i in remaining if i != first),
        key=lambda i: (-error[i] / int(costs[i]), i),
    )
    price = np.log(costs.astype(float))
    span = max(float(np.ptp(price)), 1e-12)
    third = max(
        (i for i in remaining if i not in (first, second)),
        key=lambda i: (
            min(
                (abs(i // 4 - j // 4) + abs(i % 4 - j % 4)) / 6
                + abs(price[i] - price[j]) / span
                for j in (first, second)
            ),
            -i,
        ),
    )
    return dict(selected=selected.copy(), candidates=[first, second, third])


def prepare(args):
    assets = Assets()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    supplied = json.loads(args.manifest.read_text())
    rows = supplied["rows"]
    # Portable paths in input manifests are relative to that manifest.
    rows = [
        dict(r, files=[str((args.manifest.parent / p).resolve()) for p in r["files"]])
        for r in rows
    ]
    if any(r["router_split"] not in ("train", "validation") for r in rows):
        raise ValueError("no test data in teacher")
    binding = dict(
        source_manifest=file_hash(args.manifest),
        kind=args.kind,
        width=args.width,
        G_variant=args.g_variant,
        sender=assets.sha("sender"),
        receiver=assets.sha("receiver"),
        generator=assets.generator_identity(args.g_variant),
        code=code_identity("pipeline"),
    )
    immutable(out / "binding.json", binding)
    rs, _ = sender.load_model(assets.path("sender"))
    rg, _ = receiver.load(assets.path("receiver"))
    metric = None
    generator = None
    prepared = []
    for index, row in enumerate(rows):
        folder = out / row["sample_id"]
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            done = json.loads((folder / "complete.json").read_text())
            if (
                done["binding"] != binding
                or file_hash(out / done["row"]["npz"]) != done["row"]["sha256"]
            ):
                raise ValueError("completed teacher input changed")
            prepared.append(done["row"])
            continue
        x = source(row)
        with codec_precision():
            codec = precision.PrecisionCodec()
            bank, b, full, _ = codec.encode(x, args.width)
        del codec
        gc.collect()
        torch.cuda.empty_cache()
        order = ordering(x, bank, rs)
        costs = np.asarray(order["bundle_bytes"], np.int64)
        cap = row.get("sender_parent_cap", row.get("E_cap", (0.25, 0.5)[index % 2]))
        partial = sender.prefix_under_budget(
            order["order"], costs, int(costs.sum() * cap)
        )["indices"]
        if args.kind == "generator":
            if row["router_split"] != "train":
                continue
            _, y, _ = precision.decode(packets.subset(bank, partial))
            atomic_npz(folder / "received.npz", enhanced=y)
            r = dict(
                row,
                E_cap=cap,
                received=str(folder / "received.npz"),
                received_sha256=file_hash(folder / "received.npz"),
            )
            r["npz"] = str((folder / "received.npz").relative_to(out))
            r["sha256"] = r["received_sha256"]
        else:
            if metric is None:
                metric = LPIPSAlex(True)
            if generator is None:
                from routervc.generation.engine import Generator

                generator = Generator(assets, args.g_variant)

            def measure(y, indices, patches=None):
                patches = (
                    patches
                    if patches is not None
                    else [generator(y, i)[0] for i in indices]
                )
                final = render(y, patches, indices) if indices else y.copy()
                return evaluate_variant(x, final, metric)

            def receiver_score(indices):
                _, y, _ = precision.decode(packets.subset(bank, indices))
                selected = receiver.select(
                    rg,
                    b,
                    y,
                    routing.coverage(packets.parse(packets.subset(bank, indices))),
                )["indices"]
                return measure(y, selected)

            if args.kind == "sender":
                if not partial or len(partial) > 13:
                    raise ValueError(
                        "teacher prefix needs 1..13 regions; choose another byte cap"
                    )
                inputs = []
                plans = []
                value = np.zeros((2, 16), np.float32)
                weight = np.zeros_like(value)
                for state, selected in enumerate(([], partial)):
                    inner = packets.subset(bank, selected)
                    _, y, _ = precision.decode(inner)
                    cov = routing.coverage(packets.parse(inner))
                    pred = sender.predict(rs, x, b, y, full, cov, costs, 8)[0].numpy()
                    plan = proposals(selected, pred, x, y, costs)
                    plans.append(plan)
                    inputs.append(
                        sender.build_inputs(x, b, y, full, cov, costs, 8, rs.config)
                    )
                    before = receiver_score(selected)
                    for i in plan["candidates"]:
                        after = receiver_score([*selected, i])
                        value[state, i] = before["lpips_alex"] - after["lpips_alex"]
                        weight[state, i] = 1
                arrays = {
                    f"input_{k}": torch.cat([s[k] for s in inputs]).numpy()
                    for k in inputs[0]
                }
                arrays.update(value=value, weight=weight)
            else:
                inner = packets.subset(bank, partial)
                _, y, _ = precision.decode(inner)
                cov = routing.coverage(packets.parse(inner))
                selection = receiver.select(rg, b, y, cov)
                ranking = selection["indices"]
                rng = np.random.default_rng(
                    int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16], 16)
                )
                extra = [int(i) for i in rng.permutation(16) if i not in ranking][:2]
                pool = sorted(set(ranking + extra))
                states = list(
                    dict.fromkeys(
                        tuple(sorted(s))
                        for s in ([], ranking[:3], ranking[:6], (extra + ranking)[:3])
                    )
                )
                patches = [generator(y, i)[0] for i in pool]
                value = np.zeros((len(states), 16, 3), np.float32)
                weight = np.zeros_like(value)
                selected = np.zeros((len(states), 16, 1), np.float32)
                for s, indices in enumerate(states):
                    before = measure(y, indices, patches)
                    selected[s, list(indices), 0] = 1
                    for i in pool:
                        if i in indices:
                            continue
                        after = measure(y, [*indices, i], patches)
                        value[s, i] = (
                            before["lpips_alex"] - after["lpips_alex"],
                            after["psnr_db"] - before["psnr_db"],
                            before["temporal_delta_mae"] - after["temporal_delta_mae"],
                        )
                        weight[s, i] = 1
                arrays = {
                    k: v.numpy()
                    for k, v in receiver.old.build_inputs(b, y, cov).items()
                }
                arrays.update(selected=selected, value=value, weight=weight)
                plans = None
            atomic_npz(folder / "inputs.npz", **arrays)
            r = {
                k: row[k] for k in ("sample_id", "dataset", "sequence", "router_split")
            }
            r.update(
                npz=str((folder / "inputs.npz").relative_to(out)),
                sha256=file_hash(folder / "inputs.npz"),
            )
            if plans is not None:
                r["plans"] = plans
        prepared.append(r)
        atomic_json(folder / "complete.json", dict(binding=binding, row=r))
        print("PREPARED", index + 1, len(rows), row["sample_id"], flush=True)
        from routervc.utils.resources import snapshot

        atomic_json(
            out / "progress.json",
            dict(completed=index + 1, total=len(rows), resources=snapshot()),
        )
    if args.kind == "generator":
        names = ("dit", "vae", "positive", "negative")
        config = dict(
            version="routervc_g_continuation_release_v1",
            rows=prepared,
            seed=20261016,
            steps=args.steps,
            initial=dict(
                path=str(assets.path("g_" + args.g_variant)),
                sha256=assets.sha("g_" + args.g_variant),
            ),
            assets={
                k: dict(path=str(assets.path(k)), sha256=assets.sha(k)) for k in names
            },
            code={
                str(p.relative_to(REPO)): file_hash(p)
                for p in sorted((REPO / "routervc/training").glob("*.py"))
            },
            loss=dict(
                velocity=0.25,
                clean_latent_l1=0.025,
                latent_temporal=0.05,
                lpips=0.5,
                low_frequency=0.1,
                edge=0.05,
                rgb_temporal=0.1,
            ),
            phase_weight=1.0,
            learning_rate=dict(peak=5e-6, end=2e-6, warmup=40),
            provenance=binding,
            scope="new continuation from retained weights; not replay of historical initialization",
        )
        atomic_json(out / "protocol.json", config)
    else:
        atomic_json(
            out / "labels.json",
            dict(
                format="routervc_labels_v1",
                kind=args.kind,
                rows=prepared,
                teacher=binding,
            ),
        )
