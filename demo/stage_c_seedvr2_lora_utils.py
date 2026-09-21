#!/usr/bin/env python3
"""Small, explicit LoRA adapter support for the released SeedVR2-3B DiT.

The upstream SeedVR2 repository publishes inference code but no runnable
training entrypoint.  This module leaves the released BF16 checkpoint frozen
and adds trainable low-rank updates only to the last DiT blocks and output
projection.  Adapters are stored separately from the multi-gigabyte base
checkpoint and can be applied by the inference bridge.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


FORMAT_VERSION = 1
DEFAULT_RANK = 8
DEFAULT_ALPHA = 8.0
DEFAULT_LAST_N_BLOCKS = 8


class LoRALinear(nn.Module):
    """Frozen linear layer plus a float32 low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.adapter_strength = 1.0
        self.lora_a = nn.Parameter(torch.empty(
            self.rank, base.in_features, device=base.weight.device,
            dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, self.rank, device=base.weight.device,
            dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        frozen = self.base(value)
        residual = F.linear(F.linear(value.float(), self.lora_a), self.lora_b)
        return frozen + residual.to(frozen.dtype).mul(
            self.scale * self.adapter_strength)


def is_target_linear(
    name: str,
    *,
    block_count: int,
    last_n_blocks: int,
) -> bool:
    """Return whether a SeedVR2 linear is part of the compact adapter."""

    if name == "vid_out.proj":
        return True
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blocks" or not parts[1].isdigit():
        return False
    block_index = int(parts[1])
    if block_index < block_count - last_n_blocks:
        return False
    suffixes = (
        ".attn.proj_qkv.all",
        ".attn.proj_qkv.vid",
        ".attn.proj_out.all",
        ".attn.proj_out.vid",
        ".mlp.all.proj_in_gate",
        ".mlp.vid.proj_in_gate",
        ".mlp.all.proj_in",
        ".mlp.vid.proj_in",
        ".mlp.all.proj_out",
        ".mlp.vid.proj_out",
    )
    return name.endswith(suffixes)


def _replace_submodule(root: nn.Module, name: str, value: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, value)


def inject_seedvr2_lora(
    model: nn.Module,
    *,
    rank: int = DEFAULT_RANK,
    alpha: float = DEFAULT_ALPHA,
    last_n_blocks: int = DEFAULT_LAST_N_BLOCKS,
) -> list[str]:
    """Freeze ``model`` and inject LoRA modules at fixed architecture points."""

    if not hasattr(model, "blocks"):
        raise TypeError("SeedVR2 DiT is expected to expose a blocks ModuleList")
    block_count = len(model.blocks)
    if not 1 <= last_n_blocks <= block_count:
        raise ValueError("last_n_blocks is outside the DiT depth")
    model.requires_grad_(False)
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and is_target_linear(
            name, block_count=block_count, last_n_blocks=last_n_blocks)
    ]
    expected = last_n_blocks * 5 + 1
    if len(targets) != expected:
        raise RuntimeError(
            f"expected {expected} SeedVR2 LoRA targets, found {len(targets)}: "
            f"{targets[:5]}")
    for name in targets:
        base = model.get_submodule(name)
        _replace_submodule(model, name, LoRALinear(base, rank, alpha))
    return targets


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {
        name: module for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def trainable_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    modules = lora_modules(model)
    parameters = []
    for module in modules.values():
        parameters.extend((module.lora_a, module.lora_b))
    if not parameters:
        raise RuntimeError("model has no injected LoRA modules")
    return parameters


def set_lora_strength(model: nn.Module, strength: float) -> None:
    """Set one inference-time multiplier for every injected LoRA residual."""

    strength = float(strength)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("LoRA strength must be finite and non-negative")
    modules = lora_modules(model)
    if not modules:
        raise RuntimeError("model has no injected LoRA modules")
    for module in modules.values():
        module.adapter_strength = strength


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.lora_a"] = module.lora_a.detach().cpu()
        state[f"{name}.lora_b"] = module.lora_b.detach().cpu()
    if not state:
        raise RuntimeError("model has no injected LoRA modules")
    return state


def load_lora_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    modules = lora_modules(model)
    expected = {
        f"{name}.{parameter}"
        for name in modules for parameter in ("lora_a", "lora_b")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise RuntimeError(
            f"LoRA state keys differ; missing={missing[:3]} extra={extra[:3]}")
    with torch.no_grad():
        for name, module in modules.items():
            for parameter in ("lora_a", "lora_b"):
                source = state[f"{name}.{parameter}"]
                target = getattr(module, parameter)
                if source.shape != target.shape:
                    raise RuntimeError(f"LoRA tensor shape differs: {name}.{parameter}")
                target.copy_(source.to(device=target.device, dtype=target.dtype))


def adapter_payload(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    last_n_blocks: int,
    metadata: dict | None = None,
) -> dict:
    modules = lora_modules(model)
    return {
        "format_version": FORMAT_VERSION,
        "adapter_type": "SeedVR2-3B last-block LoRA",
        "rank": int(rank),
        "alpha": float(alpha),
        "last_n_blocks": int(last_n_blocks),
        "target_modules": list(modules),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_lora_parameters(model)),
        "state_dict": lora_state_dict(model),
        "metadata": metadata or {},
    }


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_lora_adapter(
    path: Path,
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    last_n_blocks: int,
    metadata: dict | None = None,
) -> dict:
    payload = adapter_payload(
        model,
        rank=rank,
        alpha=alpha,
        last_n_blocks=last_n_blocks,
        metadata=metadata,
    )
    atomic_torch_save(path, payload)
    return {
        key: payload[key]
        for key in (
            "format_version", "adapter_type", "rank", "alpha",
            "last_n_blocks", "target_modules", "trainable_parameter_count",
            "metadata",
        )
    }


def load_lora_adapter(
    model: nn.Module,
    path: Path,
    *,
    trainable: bool = False,
    strength: float = 1.0,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("unsupported SeedVR2 LoRA checkpoint format")
    targets = inject_seedvr2_lora(
        model,
        rank=int(payload["rank"]),
        alpha=float(payload["alpha"]),
        last_n_blocks=int(payload["last_n_blocks"]),
    )
    if targets != payload["target_modules"]:
        raise RuntimeError("LoRA target list differs from the saved adapter")
    load_lora_state_dict(model, payload["state_dict"])
    set_lora_strength(model, strength)
    if not trainable:
        model.requires_grad_(False)
    result = {
        key: payload[key]
        for key in (
            "format_version", "adapter_type", "rank", "alpha",
            "last_n_blocks", "target_modules", "trainable_parameter_count",
            "metadata",
        )
    }
    result["inference_strength"] = float(strength)
    return result


def self_test() -> dict:
    base = nn.Linear(5, 7)
    wrapper = LoRALinear(base, rank=2, alpha=2)
    value = torch.randn(3, 5)
    expected = base(value)
    actual = wrapper(value)
    if not torch.equal(expected, actual):
        raise AssertionError("zero-initialized LoRA changed the base output")
    with torch.no_grad():
        wrapper.lora_b.fill_(0.25)
    full = wrapper(value)
    if torch.equal(expected, full):
        raise AssertionError("nonzero LoRA did not change the output")
    wrapper.adapter_strength = 0.0
    if not torch.equal(expected, wrapper(value)):
        raise AssertionError("zero LoRA strength did not recover the base output")
    wrapper.adapter_strength = 0.5
    halfway = wrapper(value)
    if not torch.allclose(halfway, expected + (full - expected) * 0.5):
        raise AssertionError("LoRA strength interpolation is inconsistent")
    return {
        "status": "passed",
        "zero_initialization_exact": True,
        "zero_strength_exact": True,
        "half_strength_linear": True,
        "default_rank": DEFAULT_RANK,
        "default_last_n_blocks": DEFAULT_LAST_N_BLOCKS,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
