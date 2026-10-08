"""One rANS state per ROI, EXACT same conditional CDFs as the October 7 E.

Only stream packing changes. Arbitrary 32-bit CDF indexes remove the old need
to restart rANS separately for every coarse bin. No probabilities are learned,
quantized further, sent privately, or chosen using an unreceived E symbol.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from routervc.latent import entropy
from routervc.latent.split import restore_symbols


@lru_cache(maxsize=1)
def backend():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('compact_rans.cpp')
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return load(name='routervc_compact_rans_'+identity, sources=[str(source)],
                extra_cflags=['-O2'], verbose=False)


@lru_cache(maxsize=1)
def cdfs():
    return np.concatenate([entropy.tables(3, c)[0] for c in range(-43, 43)])


def indexes(coarse, scales):
    c = np.asarray(coarse)
    if c.dtype.kind not in 'iu' or not c.size or c.min() < -43 or c.max() > 42:
        raise ValueError('invalid width3 coarse bin')
    if c.shape != np.asarray(scales).shape:
        raise ValueError('scale/coarse shape mismatch')
    return ((c.astype(np.int32)+43)*128+entropy.scale_indexes(scales)).ravel()


def encode(coarse, fine, scales):
    restore_symbols(coarse, fine, 3)
    ids = indexes(coarse, scales)
    return bytes(backend().encode(np.asarray(fine, dtype=np.int16).ravel(), ids, cdfs()))


def decode(data, coarse, scales):
    ids = indexes(coarse, scales)
    r = backend().decode(data, ids, cdfs()).reshape(coarse.shape)
    restore_symbols(coarse, r, 3)
    return r


def old_group_overhead(coarse):
    """Minimum bytes, excluding entropy-coded bits: length + final state/group."""
    groups = len(np.unique(coarse))
    return dict(groups=groups, length_bytes=4*groups, rans_state_bytes=4*groups,
                compact_rans_state_bytes=4)
