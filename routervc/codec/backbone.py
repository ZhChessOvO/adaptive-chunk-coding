"""Backbone. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
import numpy as np
from routervc.codec.uf import BaseCodec


def profile_hash():
    from routervc.assets import code_identity

    return code_identity("codec")


class LatentCodec:

    def __init__(self):
        from routervc.utils.torch import configure_torch
        from routervc.codec.native import load_bridge

        self.bridge, self.bindings = load_bridge()
        from routervc.assets import Assets

        assets = Assets()
        self.codec = BaseCodec(assets.path("uf_i"), assets.path("uf_p"))
        configure_torch()
        self.identities = (
            self.codec.models["model_i_sha256"],
            self.codec.models["model_p_sha256"],
            self.bindings["native_library_sha256"],
            profile_hash(),
        )

    def rgb(self, result, h, w):
        from routervc.codec.uf import rgb_from_tensor

        return np.stack([rgb_from_tensor(result[f"frame{i}"], h, w) for i in range(8)])

    def tensor(self, array):
        import torch

        return (
            torch.from_numpy(np.asarray(array).copy())
            .to(device=self.codec.device, dtype=torch.float16)
            .contiguous(memory_format=torch.channels_last)
        )
