"""Geometry. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import torch


def window(shape, core, size):
    _, height, width, _ = shape
    x, y, w, h = core
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or (y + h > height):
        raise ValueError("core outside input")
    if size % 16 or size < max(w, h) or size > min(width, height):
        raise ValueError("unsupported native-scale window")
    left = min(max(0, (x + w // 2 - size // 2) // 8 * 8), width - size)
    top = min(max(0, (y + h // 2 - size // 2) // 8 * 8), height - size)
    if (
        left % 8
        or top % 8
        or left > x
        or (top > y)
        or (left + size < x + w)
        or (top + size < y + h)
    ):
        raise ValueError("unaligned or uncovered core")
    return [left, top, size, size]


def master_noise(seed, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((5, 64, 64, 16), generator=generator, dtype=torch.bfloat16).to(
        device
    )


def noise_crop(master, crop, outer):
    x, y, w, h = crop
    dx, dy = (x - outer[0], y - outer[1])
    if min(dx, dy) < 0 or any((v % 8 for v in (dx, dy, w, h))):
        raise ValueError("unaligned noise coordinates")
    result = master[:, dy // 8 : (dy + h) // 8, dx // 8 : (dx + w) // 8, :].contiguous()
    if result.shape != (5, h // 8, w // 8, 16):
        raise ValueError("noise crop outside field")
    return result
