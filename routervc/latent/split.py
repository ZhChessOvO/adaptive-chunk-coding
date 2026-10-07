"""Signed, zero-centred nested bins of the native HT-S int8 residual symbols.

Odd widths retain an exact zero representative. Floor division is intentional
for negative symbols. These are symbol coordinates, NOT decoded UF features.
"""
from __future__ import annotations

import numpy as np


def validate_width(width):
    if type(width) is not int or width < 1 or width > 127 or width % 2 != 1:
        raise ValueError('bin width must be an odd integer in [1,127]')


def split_symbols(symbols, width):
    validate_width(width)
    values = np.asarray(symbols)
    if values.dtype.kind not in 'iu' or values.size == 0:
        raise ValueError('nonempty integer symbols required')
    if values.min() < -128 or values.max() > 127:
        raise ValueError('native int8 symbol out of range; never silently wrap')
    values = values.astype(np.int32)
    coarse = (values + width // 2) // width
    fine = values - coarse * width
    return coarse.astype(np.int16), fine.astype(np.int16)


def restore_symbols(coarse, fine, width):
    validate_width(width)
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    if coarse.shape != fine.shape or any(x.dtype.kind not in 'iu' for x in (coarse, fine)):
        raise ValueError('matching integer layers required')
    if np.any(np.abs(fine.astype(np.int64)) > width // 2):
        raise ValueError('invalid refinement coordinate')
    values = coarse.astype(np.int64) * width + fine.astype(np.int64)
    if np.any((values < -128) | (values > 127)):
        raise ValueError('refinement outside native alphabet')
    return values.astype(np.int16)


def coarse_representatives(coarse, width):
    validate_width(width)
    values = np.asarray(coarse)
    if values.dtype.kind not in 'iu':
        raise ValueError('integer coarse coordinates required')
    # Terminal bins are truncated by UF's signed int8 alphabet.
    lower = np.maximum(values.astype(np.int64) * width - width // 2, -128)
    upper = np.minimum(values.astype(np.int64) * width + width // 2, 127)
    if np.any(lower > upper):
        raise ValueError('empty coarse bin')
    return ((lower + upper) / 2).astype(np.float32)
