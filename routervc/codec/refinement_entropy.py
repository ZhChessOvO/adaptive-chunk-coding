"""Refinement entropy. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import routervc.codec.entropy as entropy
from routervc.codec.split import restore_symbols

WIDTHS = (3, 5, 7, 9)


def validate(width):
    if type(width) is not int or width not in WIDTHS:
        raise ValueError("this profile supports only width3/5/7/9; no teens")


def bounds(width):
    validate(width)
    return ((-128 + width // 2) // width, (127 + width // 2) // width)


@lru_cache(maxsize=1)
def backend():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("compact_rans.cpp")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return load(
        name="routervc_precision9_rans_" + identity,
        sources=[str(source)],
        extra_cflags=["-O2"],
        verbose=False,
    )


@lru_cache(maxsize=4)
def cdfs(width):
    low, high = bounds(width)
    return np.concatenate([entropy.tables(width, c)[0] for c in range(low, high + 1)])


def indexes(coarse, scales, width):
    low, high = bounds(width)
    c = np.asarray(coarse)
    if c.dtype.kind not in "iu" or not c.size or c.min() < low or (c.max() > high):
        raise ValueError("invalid coarse bin")
    if c.shape != np.asarray(scales).shape:
        raise ValueError("scale/coarse shape mismatch")
    return ((c.astype(np.int32) - low) * 128 + entropy.scale_indexes(scales)).ravel()


def encode(coarse, fine, scales, width):
    validate(width)
    restore_symbols(coarse, fine, width)
    return bytes(
        backend().encode(
            np.asarray(fine, dtype=np.int16).ravel(),
            indexes(coarse, scales, width),
            cdfs(width),
        )
    )


def decode(data, coarse, scales, width):
    ids = indexes(coarse, scales, width)
    fine = backend().decode(data, ids, cdfs(width)).reshape(coarse.shape)
    restore_symbols(coarse, fine, width)
    return fine
