"""Checks. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations


def context_exact(bridge, proxy, expected):
    import torch

    current = bridge.context(proxy)
    return all(
        (
            torch.equal(
                current[k].contiguous().reshape(-1).view(torch.uint8),
                expected[k].contiguous().reshape(-1).view(torch.uint8),
            )
            for k in current
        )
    )
