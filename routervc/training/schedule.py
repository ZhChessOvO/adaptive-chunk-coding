"""Schedule. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import math


def learning_rate(step, config):
    spec = config["learning_rate"]
    warm = spec["warmup"]
    if step <= warm:
        return spec["peak"] * step / warm
    t = (step - warm) / max(1, config["steps"] - warm)
    return spec["end"] + 0.5 * (spec["peak"] - spec["end"]) * (
        1 + math.cos(math.pi * t)
    )
