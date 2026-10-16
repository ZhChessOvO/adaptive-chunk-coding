"""Losses. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import torch.nn.functional as F


def image_terms(pred, target, metric, offset=0):
    pred, target = (pred[..., 16:-16, 16:-16], target[..., 16:-16, 16:-16])
    indices = list(range(offset % 4, pred.shape[0], 4))
    perceptual = metric(pred[indices].clamp(-1, 1), target[indices]).mean()
    low = F.l1_loss(F.avg_pool2d(pred, 8), F.avg_pool2d(target, 8))
    edge = (
        F.l1_loss(
            pred[..., 1:, :] - pred[..., :-1, :],
            target[..., 1:, :] - target[..., :-1, :],
        )
        + F.l1_loss(
            pred[..., :, 1:] - pred[..., :, :-1],
            target[..., :, 1:] - target[..., :, :-1],
        )
    ) / 2
    temporal = F.l1_loss(pred[1:] - pred[:-1], target[1:] - target[:-1])
    return dict(lpips=perceptual, low_frequency=low, edge=edge, rgb_temporal=temporal)
