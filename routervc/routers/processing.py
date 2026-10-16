"""Processing. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np
import torch
import routervc.routers.inputs as visual


def processing_roi(roi, height, width, halo):
    visual.require(
        type(halo) is int and halo in (0, 64), "only core/G-halo profiles supported"
    )
    x, y, w, h = roi
    x0, y0 = (max(0, x - halo), max(0, y - halo))
    return [x0, y0, min(width, x + w + halo) - x0, min(height, y + h + halo) - y0]


def build_inputs(base, received, coverage, *, halo=0, config=None):
    config = config or visual.VisualRouterConfig()
    inputs = visual.build_receiver_inputs(base, received, coverage, config=config)
    if halo:
        rois = visual.grid_rois(*base.shape[1:3])
        ids = visual.sampled_frame_indices(len(base), config.temporal_frames)
        pairs = np.concatenate((base[ids], received[ids]), -1)
        expanded = [processing_roi(r, *base.shape[1:3], halo) for r in rois]
        inputs["local_pairs"] = torch.stack(
            [
                visual._letterbox(pairs[:, y : y + h, x : x + w], config.local_size)
                for x, y, w, h in expanded
            ]
        ).unsqueeze(0)
    return inputs
