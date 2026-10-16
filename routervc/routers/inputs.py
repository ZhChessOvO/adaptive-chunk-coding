"""Inputs. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

INPUT_KEYS = ("global_pairs", "local_pairs", "geometry", "coverage")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class VisualRouterConfig:
    use_global: bool = True
    channels: int = 32
    hidden: int = 64
    temporal_frames: int = 3
    local_size: int = 64
    global_size: int = 96

    def __post_init__(self):
        require(type(self.use_global) is bool, "use_global must be boolean")
        require(
            all(
                (
                    type(v) is int and v > 0
                    for v in (
                        self.channels,
                        self.hidden,
                        self.temporal_frames,
                        self.local_size,
                        self.global_size,
                    )
                )
            ),
            "model dimensions must be positive integers",
        )
        require(
            self.local_size >= 16 and self.global_size >= 16,
            "image sizes must be at least 16",
        )


def grid_rois(height, width):
    """Same 8-aligned 4x4 topology as RTVC, without importing its pinned models."""
    require(
        all((type(v) is int and v >= 64 and (v % 8 == 0) for v in (height, width))),
        "coded geometry must be >=64 and divisible by eight",
    )
    ys = [8 * (i * (height // 8) // 4) for i in range(5)]
    xs = [8 * (i * (width // 8) // 4) for i in range(5)]
    return [
        [xs[x], ys[y], xs[x + 1] - xs[x], ys[y + 1] - ys[y]]
        for y in range(4)
        for x in range(4)
    ]


def validated_pixels(base, received, rois):
    base, received = (np.asarray(base), np.asarray(received))
    require(
        base.dtype == np.uint8 and received.dtype == np.uint8,
        "B/Y must be entropy-decoded uint8 RGB, not source features",
    )
    require(
        base.ndim == 4
        and base.shape[-1] == 3
        and (base.shape[0] >= 2)
        and (received.shape == base.shape),
        "expected paired [T>=2,H,W,3] B/Y",
    )
    expected = grid_rois(base.shape[1], base.shape[2])
    rois = expected if rois is None else rois
    require(
        np.array_equal(np.asarray(rois), np.asarray(expected)),
        "ROIs must match the dynamic receiver raster 4x4 grid",
    )
    return (base, received, expected)


def _letterbox(pairs, size):
    """CPU RGB-only preprocessing; preserve aspect rather than stretch a face."""
    values = torch.from_numpy(np.ascontiguousarray(pairs)).permute(0, 3, 1, 2).float()
    height, width = values.shape[-2:]
    ratio = size / max(height, width)
    h, w = (max(1, round(height * ratio)), max(1, round(width * ratio)))
    if (height, width) != (h, w):
        values = F.interpolate(
            values, size=(h, w), mode="bilinear", align_corners=False, antialias=True
        )
    left, top = ((size - w) // 2, (size - h) // 2)
    values = F.pad(
        values, (left, size - w - left, top, size - h - top), mode="replicate"
    )
    return values / 127.5 - 1.0


def sampled_frame_indices(frame_count, temporal_frames=3):
    require(
        type(frame_count) is int
        and frame_count >= 2
        and (type(temporal_frames) is int)
        and (temporal_frames > 0),
        "invalid temporal sample dimensions",
    )
    return (
        np.rint(np.linspace(0, frame_count - 1, temporal_frames)).astype(int).tolist()
    )


def build_receiver_inputs(
    base, received, coverage, rois=None, *, config=None, region_indices=None
):
    """Return batch-one tensors using only B, actual Y and received-E coverage.

    No source, labels, packet benefit estimates or semantic masks are accepted.
    The whole supplied field of view is retained; a historical UVG crop stays a
    crop and is NOT called the original full UVG frame by this function.
    """
    config = config or VisualRouterConfig()
    base, received, rois = validated_pixels(base, received, rois)
    indices = list(range(16)) if region_indices is None else list(region_indices)
    require(
        bool(indices)
        and len(set(indices)) == len(indices)
        and all((type(i) is int and 0 <= i < 16 for i in indices)),
        "invalid selected region indices",
    )
    coverage = np.asarray(coverage)
    require(
        coverage.shape == (16,)
        and coverage.dtype.kind in "biuf"
        and np.isfinite(coverage).all()
        and np.all((coverage >= 0) & (coverage <= 1)),
        "coverage must have 16 finite fractions in [0,1]",
    )
    ids = sampled_frame_indices(len(base), config.temporal_frames)
    pairs = np.concatenate((base[ids], received[ids]), axis=-1)
    global_pairs = _letterbox(pairs, config.global_size)
    selected_rois = [rois[i] for i in indices]
    local_pairs = torch.stack(
        [
            _letterbox(pairs[:, y : y + h, x : x + w], config.local_size)
            for x, y, w, h in selected_rois
        ]
    )
    height, width = base.shape[1:3]
    geometry = [
        [
            (x + w / 2) / width,
            (y + h / 2) / height,
            w / width,
            h / height,
            w / 512.0,
            h / 512.0,
        ]
        for x, y, w, h in selected_rois
    ]
    return dict(
        global_pairs=global_pairs.unsqueeze(0),
        local_pairs=local_pairs.unsqueeze(0),
        geometry=torch.tensor(geometry, dtype=torch.float32).unsqueeze(0),
        coverage=torch.tensor(coverage[indices], dtype=torch.float32).reshape(
            1, len(indices), 1
        ),
    )
