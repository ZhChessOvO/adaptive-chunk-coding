"""Latents. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np
import torch
from routervc.generation.runner import resize_and_normalize
from routervc.generation.runner import pad_temporal


def crop_rgb(frames, crop):
    x, y, w, h = crop
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("expected uint8 THWC receiver pixels")
    if (
        min(x, y) < 0
        or min(w, h) <= 0
        or x + w > frames.shape[2]
        or (y + h > frames.shape[1])
    ):
        raise ValueError("invalid RGB crop")
    if x % 8 or y % 8 or w % 16 or h % 16:
        raise ValueError("unaligned RGB crop")
    return frames[:, y : y + h, x : x + w].copy()


@torch.no_grad()
def encode_roi(runner, frames, crop, device):
    """Same preprocessing, encoder slicing, posterior and dtype as receiver.

    Decoder slicing stays disabled during differentiable image supervision.
    VAE posterior sampling inside encode must not perturb the training RNG.
    """
    values = crop_rgb(frames, crop)
    tensor = torch.from_numpy(values).permute(0, 3, 1, 2).float().div_(255.0)
    sample, length = pad_temporal(
        resize_and_normalize(tensor, crop[3], crop[2], device)
    )
    if length != 17:
        raise ValueError("training/probe encode operates on actual 17-frame windows")
    runner.config.vae.use_sample = False
    runner.vae.set_causal_slicing(**runner.config.vae.slicing)
    try:
        with torch.random.fork_rng(
            devices=[torch.cuda.current_device()]
        ), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = runner.vae_encode([sample])[0]
        assert latent.dtype == torch.bfloat16 and torch.isfinite(latent).all()
        return latent.detach().cpu().contiguous()
    finally:
        runner.vae.set_causal_slicing(split_size=None, memory_device="same")
