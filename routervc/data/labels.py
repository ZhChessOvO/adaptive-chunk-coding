"""Compact, hash-bound offline labels; no latent/rate estimates masquerade as bytes."""

from pathlib import Path
import json
import numpy as np
import torch
from routervc.utils.io import file_hash
from routervc.routers import sender


class Labels:
    def __init__(self, manifest, kind):
        self.path = Path(manifest).resolve()
        self.config = json.loads(self.path.read_text())
        if self.config["kind"] != kind or self.config["format"] != "routervc_labels_v1":
            raise ValueError("incorrect label manifest")
        self.kind = kind
        self.rows = self.config["rows"]
        if len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate sample ID")
        groups = {}
        for r in self.rows:
            if r["router_split"] not in ("train", "validation"):
                raise ValueError("training manifest must not include test samples")
            key = (r["dataset"], r["sequence"])
            if groups.setdefault(key, r["router_split"]) != r["router_split"]:
                raise ValueError("sequence crosses train/validation")

    def __call__(self, row):
        path = (self.path.parent / row["npz"]).resolve()
        if file_hash(path) != row["sha256"]:
            raise ValueError("label input changed")
        with np.load(path, allow_pickle=False) as z:
            targets = {k: torch.from_numpy(z[k].copy()) for k in ("value", "weight")}
            if self.kind == "sender":
                inputs = {
                    k[6:]: torch.from_numpy(z[k].copy())
                    for k in z.files
                    if k.startswith("input_")
                }
                targets["label_scope"] = sender.LABEL_SCOPE
                return dict(
                    inputs=inputs,
                    targets=targets,
                    plans=row["plans"],
                    packet_bytes=inputs["packet_bytes"].squeeze(-1),
                )
            n = len(z["selected"])
            inputs = {
                k: torch.from_numpy(z[k].copy()).expand(n, *z[k].shape[1:])
                for k in ("local_pairs", "global_pairs", "geometry", "coverage")
            }
            inputs["selected"] = torch.from_numpy(z["selected"].copy())
            return dict(inputs=inputs, targets=targets)
