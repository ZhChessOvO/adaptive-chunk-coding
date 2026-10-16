"""Texture. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def phase_bias_loss(pred, target, periods=(8, 16)):
    """RMS of stationary, high-pass *error* folded over 8/16px phases.

    Averaging cells/time rejects much nonstationary detail; subtracting the
    source rejects correctly reproduced periodic content. Finite scene bias
    remains possible, so this is kept separate from visual/LPIPS evaluation.
    Values use the training RGB range [-1,1]; no clipping hides overshoots.
    """
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("expected paired TCHW tensors")
    error = (pred - target).float()
    coordinate = torch.arange(-9, 10, device=error.device, dtype=error.dtype)
    kernel = torch.exp(-coordinate.square() / 18)
    kernel = kernel / kernel.sum()
    c = error.shape[1]
    blur = F.conv2d(
        F.pad(error, (9, 9, 0, 0), mode="reflect"),
        kernel.reshape(1, 1, 1, -1).expand(c, 1, 1, -1),
        groups=c,
    )
    blur = F.conv2d(
        F.pad(blur, (0, 0, 9, 9), mode="reflect"),
        kernel.reshape(1, 1, -1, 1).expand(c, 1, -1, 1),
        groups=c,
    )
    high = error - blur
    terms = []
    for p in periods:
        h, w = (high.shape[-2] // p * p, high.shape[-1] // p * p)
        if min(h, w) < p * 2:
            raise ValueError("too few complete phase cells")
        folded = (
            high[..., :h, :w]
            .reshape(high.shape[0], c, h // p, p, w // p, p)
            .mean((0, 2, 4))
        )
        folded = folded - folded.mean((-2, -1), keepdim=True)
        terms.append(torch.sqrt(folded.square().mean() + 1e-12) - 1e-06)
    return torch.stack(terms).mean()
