"""Lossless pixel identity used by encoder/receiver verification."""
import hashlib
import numpy as np


def frame_hash(frames):
    values = np.ascontiguousarray(frames)
    return hashlib.sha256(values.tobytes()).hexdigest()
