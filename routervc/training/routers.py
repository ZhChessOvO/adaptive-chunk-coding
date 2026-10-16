"""Independent-router continuation on measured final-picture marginal labels.

Alternation is explicit: freeze the other router/G/fusion, prepare labels, then
fit this router. This is not differentiable joint training through entropy or G.
The optimizer, shuffle cursor, RNG and best weights are atomically resumable.
"""

from pathlib import Path
import signal
import time
import numpy as np
import torch
from routervc.assets import Assets
from routervc.data.labels import Labels
from routervc.routers import sender, receiver
from routervc.training import sender_helpers as helpers
from routervc.utils.io import atomic_json, file_hash
from routervc.utils.records import immutable
from routervc.utils.torch import atomic_torch
from routervc.utils.precision import codec_precision


def receiver_scale(rows, get):
    square = torch.zeros(3, dtype=torch.float64)
    count = square.clone()
    for r in rows:
        t = get(r)["targets"]
        w = t["weight"].double()
        v = torch.where(w > 0, t["value"].double(), 0.0)
        square += (v.square() * w).sum((0, 1))
        count += w.sum((0, 1))
    if not torch.all(count > 0):
        raise ValueError("missing receiver supervision")
    return (square / count).sqrt().clamp_min(1e-5).float()


def batch(data, kind, device):
    if kind == "sender":
        return helpers.arm_batch(data, "source", device)
    return {
        p: {k: v.to(device) for k, v in data[p].items()} for p in ("inputs", "targets")
    }


def loss(model, b, kind, scale):
    pred = model(b["inputs"])
    if kind == "sender":
        return sender.training_loss(
            pred, b["targets"], b["packet_bytes"], gain_scale=scale, ranking_weight=0.1
        )[0]
    return receiver.old.training_loss(
        pred, b["targets"], gain_scale=scale, ranking_weight=0.1
    )[0]


@torch.no_grad()
def validate(model, rows, get, kind, scale, device):
    if kind == "sender":
        return helpers.validate(model, "source", rows, get, scale, device)
    groups = {}
    model.eval()
    for row in rows:
        b = batch(get(row), kind, device)
        with codec_precision():
            pred = model(b["inputs"])
            current = receiver.old.training_loss(
                pred, b["targets"], gain_scale=scale, ranking_weight=0.1
            )[0]
        truth = b["targets"]["value"][..., 0].cpu().numpy()
        known = b["targets"]["weight"][..., 0].cpu().numpy() > 0
        values = pred[..., 0].cpu().numpy()
        regrets = []
        for t, k, v in zip(truth, known, values):
            ids = np.flatnonzero(k)
            if not len(ids):
                continue
            chosen = max(ids, key=lambda i: (float(v[i]), -int(i)))
            regrets.append(
                max(0.0, float(t[ids].max()))
                - (float(t[chosen]) if v[chosen] > 0 else 0.0)
            )
        groups.setdefault(row["dataset"], []).append(
            dict(loss=float(current), regret=float(np.mean(regrets)))
        )
    group = {
        g: {k: float(np.mean([r[k] for r in rows])) for k in ("loss", "regret")}
        for g, rows in groups.items()
    }
    return dict(
        groups=group,
        **{
            k: float(np.mean([v[k] for v in group.values()]))
            for k in ("loss", "regret")
        }
    )


def train(args):
    data = Labels(args.manifest, args.kind)
    assets = Assets()
    train_rows = [r for r in data.rows if r["router_split"] == "train"]
    valid = [r for r in data.rows if r["router_split"] == "validation"]
    if not train_rows or not valid:
        raise ValueError("both train and validation are required")
    device = args.device
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    binding = dict(
        manifest=file_hash(data.path),
        kind=args.kind,
        initial=assets.sha(args.kind),
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    immutable(out / "binding.json", binding)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    with helpers.deterministic_training(args.seed, device):
        model, payload = (
            sender.load_model(assets.path("sender"))
            if args.kind == "sender"
            else receiver.load(assets.path("receiver"))
        )
        model = model.to(device).requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        scale = (
            helpers.fit_scale(dict(rows=data.rows, smoke=False), data)["value"]
            if args.kind == "sender"
            else receiver_scale(train_rows, data).to(device)
        )
        state = dict(epoch=0, cursor=0, updates=0, best=None, history=[])
        if (out / "resume.pt").exists():
            saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=True)
            if saved["binding"] != binding:
                raise ValueError("resume recipe changed")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
            helpers.restore_rng(saved["rng"], device)

        def save():
            atomic_torch(
                out / "resume.pt",
                dict(
                    binding=binding,
                    model=helpers.cpu_tree(model.state_dict()),
                    optimizer=helpers.cpu_tree(optimizer.state_dict()),
                    state=state,
                    rng=helpers.capture_rng(device),
                ),
            )

        began = time.monotonic()
        for epoch in range(state["epoch"], args.epochs):
            order = torch.randperm(
                len(train_rows),
                generator=torch.Generator().manual_seed(args.seed + epoch),
            ).tolist()
            for pos in range(state["cursor"], len(order)):
                if stop[0]:
                    save()
                    return
                b = batch(data(train_rows[order[pos]]), args.kind, device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with codec_precision():
                    current = loss(model, b, args.kind, scale)
                if not torch.isfinite(current):
                    raise RuntimeError("nonfinite loss")
                current.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                state.update(cursor=pos + 1, updates=state["updates"] + 1)
                save()
                from routervc.utils.resources import snapshot

                atomic_json(
                    out / "progress.json",
                    dict(
                        epoch=epoch + 1,
                        cursor=pos + 1,
                        updates=state["updates"],
                        loss=float(current.detach()),
                        seconds=time.monotonic() - began,
                        resources=snapshot(),
                    ),
                )
                print("UPDATE", state["updates"], float(current.detach()), flush=True)
                if args.stop_after and state["updates"] >= args.stop_after:
                    return
            score = validate(model, valid, data, args.kind, scale, device)
            if state["best"] is None or (score["regret"], score["loss"]) < (
                state["best"]["score"]["regret"],
                state["best"]["score"]["loss"],
            ):
                state["best"] = dict(epoch=epoch + 1, score=score)
                exported = (
                    sender.export_payload(model, binding, state["updates"], score)
                    if args.kind == "sender"
                    else receiver.payload(model, binding, epoch + 1, score)
                )
                atomic_torch(out / "best.pt", exported)
            state["history"].append(dict(epoch=epoch + 1, validation=score))
            state.update(epoch=epoch + 1, cursor=0)
            save()
            atomic_json(out / "history.json", state["history"])
        atomic_json(
            out / "complete.json",
            dict(
                complete=True,
                best=state["best"],
                updates=state["updates"],
                best_sha256=file_hash(out / "best.pt"),
                scope="warm-start continuation; fixed offline final-picture labels",
            ),
        )
