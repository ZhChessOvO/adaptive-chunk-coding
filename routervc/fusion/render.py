"""Render. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import routervc.fusion.composition as fmt
import routervc.routers.routing as routing
from routervc.fusion.multiband import fuse


def current(received, patches, indices):
    result = received.copy()
    by_region = {p["region"]: p for p in patches}
    for i in indices:
        p = by_region[i]
        x0, y0, cw, ch = p["crop"]
        x, y, w, h = p["core"]
        raw = received.copy()
        raw[:, y : y + h, x : x + w] = p["pixels"][
            :, y - y0 : y - y0 + h, x - x0 : x - x0 + w
        ]
        alpha = fmt.weights(received.shape, routing.control(received.shape, {}, i))
        pixels = fmt.combine(received, raw, alpha)
        result[alpha > 0] = pixels[alpha > 0]
    return result


def render(received, patches, indices):
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate G")
    used = [next((p for p in patches if p["region"] == i)) for i in sorted(indices)]
    return fuse(
        received, current(received, used, sorted(indices)), used, sorted(indices)
    )["multiband"]
