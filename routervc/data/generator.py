"""Generator. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from pathlib import Path
import random
import numpy as np
from routervc.utils.records import digest
from routervc.data.source import source

REPO = Path(__file__).resolve().parents[2]
SEED = 20261014


def choose(rows, step, seed=SEED):
    rng = random.Random(seed + step * 1000033)
    dataset = "UVG" if (step - 1) % 4 == 3 else "REDS"
    row = rng.choice([r for r in rows if r["dataset"] == dataset])
    _, h, w, _ = row["shape"]
    crop = [
        16 * rng.randrange((w - 256) // 16 + 1),
        16 * rng.randrange((h - 256) // 16 + 1),
        256,
        256,
    ]
    return (row, crop)


def preload(rows, progress):
    memory = {}
    for i, row in enumerate(rows):
        if digest(row["received"]) != row["received_sha256"]:
            raise ValueError("received pixels changed")
        with np.load(row["received"]) as z:
            received = z["enhanced"].copy()
        original = source(row)
        if original.shape != received.shape or list(original.shape) != row["shape"]:
            raise ValueError("source/receiver geometry differs")
        memory[row["sample_id"]] = (original, received)
        progress(
            phase="loading_verified_training_inputs", loaded=i + 1, total=len(rows)
        )
    return memory
