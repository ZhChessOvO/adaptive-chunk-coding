"""Frozen. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from routervc.generation.lora import trainable_lora_parameters


def assert_frozen(runner, initial):
    allowed = {id(p) for p in trainable_lora_parameters(runner.dit)}
    assert (
        sum((p.numel() for p in runner.dit.parameters() if p.requires_grad)) == 2822656
    )
    for p in runner.dit.parameters():
        assert p.requires_grad == (id(p) in allowed)
        if id(p) not in allowed:
            assert p.grad is None
    assert not any(
        (p.requires_grad or p.grad is not None for p in runner.vae.parameters())
    )
