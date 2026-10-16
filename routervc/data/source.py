"""Source. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np
from routervc.utils.records import digest


def source(row):
    from PIL import Image

    frames = []
    for path, expected in zip(row["files"], row["hashes"]):
        if digest(path) != expected:
            raise ValueError("source pixels changed: " + path)
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    result = np.stack(frames)
    if list(result.shape) != row["shape"]:
        raise ValueError("source geometry changed")
    return result
