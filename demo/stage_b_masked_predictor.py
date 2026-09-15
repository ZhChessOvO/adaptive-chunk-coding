#!/usr/bin/env python3
"""Active-only masked latent predictor for Adaptive Chunk Coding Stage B.

The model is deliberately separate from DCVC-UF.  Its inference API accepts
only decoder-available tensors and a transmitted route mask.  Training labels
must stay outside this module.

For every skipped 2x2 latent block, the predictor gathers a 3x3 neighborhood
of decoded Base residual blocks.  Learned projections and the MLP run only on
those gathered queries; no dense learned network is evaluated over the full
latent grid.  The output is a dequantized final-latent correction scattered
only into skipped positions after the four-stage entropy prior has completed.
Consequently, already decoded Base latent values are invariant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PredictorConfig:
    block_size: int = 2
    radius: int = 1
    q_width: int = 32
    common_width: int = 64
    mean_width: int = 32
    hidden_width: int = 128


def tensor_to_blocks(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Convert one NCHW tensor to raster-ordered flattened spatial blocks."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("expected one NCHW tensor")
    _, channels, height, width = tensor.shape
    if height % block_size or width % block_size:
        raise ValueError("tensor dimensions must be divisible by block_size")
    grid_h, grid_w = height // block_size, width // block_size
    return tensor[0].reshape(
        channels, grid_h, block_size, grid_w, block_size
    ).permute(1, 3, 0, 2, 4).reshape(
        grid_h * grid_w, channels * block_size * block_size)


def blocks_to_tensor(
        blocks: torch.Tensor, shape: tuple[int, ...], block_size: int) -> torch.Tensor:
    """Inverse of :func:`tensor_to_blocks`."""
    batch, channels, height, width = shape
    if batch != 1 or height % block_size or width % block_size:
        raise ValueError("invalid output shape for block conversion")
    grid_h, grid_w = height // block_size, width // block_size
    expected = (grid_h * grid_w, channels * block_size * block_size)
    if tuple(blocks.shape) != expected:
        raise ValueError(f"block tensor has shape {tuple(blocks.shape)}, expected {expected}")
    return blocks.reshape(
        grid_h, grid_w, channels, block_size, block_size
    ).permute(2, 0, 3, 1, 4).reshape(shape)


def _route_tensor(skip_blocks, device: torch.device) -> torch.Tensor:
    if isinstance(skip_blocks, np.ndarray):
        route = torch.from_numpy(skip_blocks)
    elif torch.is_tensor(skip_blocks):
        route = skip_blocks
    else:
        raise TypeError("skip_blocks must be a NumPy array or Torch tensor")
    if route.ndim != 2:
        raise ValueError("skip_blocks must have shape [grid_h, grid_w]")
    return route.to(device=device, dtype=torch.bool)


def _neighbor_table(
        query_ids: torch.Tensor, grid_h: int, grid_w: int, radius: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(-radius, radius + 1, device=query_ids.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dy = dy.reshape(1, -1)
    dx = dx.reshape(1, -1)
    rows = torch.div(query_ids, grid_w, rounding_mode="floor")[:, None] + dy
    cols = torch.remainder(query_ids, grid_w)[:, None] + dx
    valid = (rows >= 0) & (rows < grid_h) & (cols >= 0) & (cols < grid_w)
    rows = rows.clamp(0, grid_h - 1)
    cols = cols.clamp(0, grid_w - 1)
    return rows * grid_w + cols, valid


def _projector(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.GELU(),
    )


class SparseMaskedLatentPredictor(nn.Module):
    """Predict final-y corrections for only the routed Generate blocks."""

    def __init__(self, config: PredictorConfig | None = None):
        super().__init__()
        self.config = config or PredictorConfig()
        cfg = self.config
        block_area = cfg.block_size * cfg.block_size
        q_features = 256 * block_area
        mean_features = 256 * block_area
        common_features = 768 * block_area
        neighbor_slots = (2 * cfg.radius + 1) ** 2

        self.q_projector = _projector(q_features, cfg.q_width)
        self.common_projector = _projector(common_features, cfg.common_width)
        self.mean_projector = _projector(mean_features, cfg.mean_width)
        trunk_features = (
            neighbor_slots * cfg.q_width + neighbor_slots
            + cfg.common_width + cfg.mean_width)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_features, cfg.hidden_width),
            nn.LayerNorm(cfg.hidden_width),
            nn.GELU(),
            nn.Linear(cfg.hidden_width, cfg.hidden_width),
            nn.GELU(),
            nn.Linear(cfg.hidden_width, mean_features),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    @property
    def architecture(self) -> dict:
        return asdict(self.config)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def predicted_macs(self, skipped_blocks: int) -> int:
        """Count Linear MACs; normalization, GELU, gather and scatter are excluded."""
        cfg = self.config
        block_area = cfg.block_size * cfg.block_size
        q_features = 256 * block_area
        mean_features = 256 * block_area
        common_features = 768 * block_area
        neighbors = (2 * cfg.radius + 1) ** 2
        trunk_features = (
            neighbors * cfg.q_width + neighbors
            + cfg.common_width + cfg.mean_width)
        per_query = (
            neighbors * q_features * cfg.q_width
            + common_features * cfg.common_width
            + mean_features * cfg.mean_width
            + trunk_features * cfg.hidden_width
            + cfg.hidden_width * cfg.hidden_width
            + cfg.hidden_width * mean_features)
        return int(skipped_blocks * per_query)

    def predict_blocks(
            self, decoded_q: torch.Tensor, mean_y: torch.Tensor,
            common_params: torch.Tensor, skip_blocks
            ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return correction blocks, their raster query ids, and work counters."""
        if decoded_q.shape != mean_y.shape:
            raise ValueError("decoded_q and mean_y must have identical shapes")
        if decoded_q.shape[0] != 1 or decoded_q.shape[1] != 256:
            raise ValueError("expected one 256-channel y tensor")
        if common_params.shape[0] != 1 or common_params.shape[1] != 768:
            raise ValueError("expected one 768-channel common_params tensor")
        if common_params.shape[-2:] != decoded_q.shape[-2:]:
            raise ValueError("common_params and y spatial shapes must match")

        cfg = self.config
        route = _route_tensor(skip_blocks, decoded_q.device)
        grid_h = decoded_q.shape[-2] // cfg.block_size
        grid_w = decoded_q.shape[-1] // cfg.block_size
        if tuple(route.shape) != (grid_h, grid_w):
            raise ValueError(
                f"route shape {tuple(route.shape)} does not match {(grid_h, grid_w)}")
        query_ids = torch.nonzero(route.reshape(-1), as_tuple=False).reshape(-1)
        block_width = decoded_q.shape[1] * cfg.block_size * cfg.block_size
        if query_ids.numel() == 0:
            empty = decoded_q.new_empty((0, block_width), dtype=torch.float32)
            return empty, query_ids, {
                "skipped_blocks": 0,
                "neighbor_slots_per_query": (2 * cfg.radius + 1) ** 2,
                "observed_neighbor_block_reads": 0,
                "predicted_coefficients": 0,
                "linear_macs": 0,
                "whole_latent_learned_activation": False,
            }

        neighbor_ids, valid = _neighbor_table(
            query_ids, grid_h, grid_w, cfg.radius)
        flat_route = route.reshape(-1)
        observed = valid & ~flat_route[neighbor_ids]

        q_blocks = tensor_to_blocks(decoded_q, cfg.block_size).float()
        mean_blocks = tensor_to_blocks(mean_y, cfg.block_size).float()
        common_blocks = tensor_to_blocks(common_params, cfg.block_size).float()
        q_neighbors = q_blocks[neighbor_ids] * observed[:, :, None]
        q_projector_input = q_neighbors / 8.0
        common_projector_input = common_blocks[query_ids]
        mean_projector_input = mean_blocks[query_ids]
        q_embedding = self.q_projector(q_projector_input).reshape(
            query_ids.numel(), -1)
        common_embedding = self.common_projector(common_projector_input)
        mean_embedding = self.mean_projector(mean_projector_input)
        features = torch.cat((
            q_embedding,
            observed.to(dtype=q_embedding.dtype),
            common_embedding,
            mean_embedding,
        ), dim=1)
        prediction = self.trunk(features)
        profile = {
            "skipped_blocks": int(query_ids.numel()),
            "neighbor_slots_per_query": int(neighbor_ids.shape[1]),
            "observed_neighbor_block_reads": int(observed.sum().item()),
            "predicted_coefficients": int(prediction.numel()),
            "linear_macs": self.predicted_macs(int(query_ids.numel())),
            "whole_latent_learned_activation": False,
            "learned_module_input_shapes": {
                "q_projector": list(q_projector_input.shape),
                "common_projector": list(common_projector_input.shape),
                "mean_projector": list(mean_projector_input.shape),
                "trunk": list(features.shape),
            },
            "learned_module_output_shape": list(prediction.shape),
        }
        return prediction, query_ids, profile

    def forward(
            self, decoded_q: torch.Tensor, mean_y: torch.Tensor,
            common_params: torch.Tensor, skip_blocks) -> torch.Tensor:
        prediction, _, _ = self.predict_blocks(
            decoded_q, mean_y, common_params, skip_blocks)
        return prediction

    def apply(
            self, decoded_q: torch.Tensor, mean_y: torch.Tensor,
            common_params: torch.Tensor, skip_blocks
            ) -> tuple[torch.Tensor, dict]:
        """Scatter prediction only to Generate positions and preserve Base exactly."""
        prediction, query_ids, profile = self.predict_blocks(
            decoded_q, mean_y, common_params, skip_blocks)
        if query_ids.numel() == 0:
            return mean_y, profile
        delta_blocks = mean_y.new_zeros(
            tensor_to_blocks(mean_y, self.config.block_size).shape)
        delta_blocks[query_ids] = prediction.to(dtype=mean_y.dtype)
        delta_y = blocks_to_tensor(
            delta_blocks, tuple(mean_y.shape), self.config.block_size)
        return mean_y + delta_y, profile


def skipped_target_blocks(
        target_delta: torch.Tensor, skip_blocks, block_size: int = 2
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only helper kept separate from the predictor inference API."""
    route = _route_tensor(skip_blocks, target_delta.device)
    query_ids = torch.nonzero(route.reshape(-1), as_tuple=False).reshape(-1)
    return tensor_to_blocks(target_delta, block_size)[query_ids].float(), query_ids
