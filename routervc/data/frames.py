"""Explicit frame-directory input; no implicit cropping, scaling or downloads."""

from pathlib import Path
import numpy as np
from PIL import Image


def read_frames(folder, start=0, frames=17):
    paths = sorted(
        p
        for p in Path(folder).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    paths = paths[start : start + frames]
    if len(paths) != frames:
        raise ValueError(f"expected {frames} frames, found {len(paths)}")
    values = []
    for path in paths:
        with Image.open(path) as image:
            values.append(np.asarray(image.convert("RGB")))
    return np.stack(values)


def save_frames(folder, pixels):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(pixels):
        Image.fromarray(frame).save(folder / f"{i:08d}.png")
