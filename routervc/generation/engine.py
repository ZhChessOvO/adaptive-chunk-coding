"""Native-scale, posterior-mean BF16 generation of one addressed region.

All crops condition on the same ungenerated received picture. The seed depends
only on the region, so selection order cannot change a generated patch. There
is no active feature-delta or internal-condition branch in the final model.
"""

from types import SimpleNamespace
import numpy as np

from routervc.assets import Assets
from routervc.generation.runtime import PersistentSeedVR2
from routervc.routers.routing import control


class Generator:
    def __init__(self, assets=None, variant="formal"):
        self.assets = assets or Assets()
        self.variant = variant
        self.identity = self.assets.generator_identity(variant)
        self.model = PersistentSeedVR2(
            SimpleNamespace(
                upstream_root=self.assets.upstream(),
                dit_checkpoint=self.assets.path("dit"),
                vae_checkpoint=self.assets.path("vae"),
                positive_embedding=self.assets.path("positive"),
                negative_embedding=self.assets.path("negative"),
                lora_checkpoint=self.assets.path("g_" + variant),
                lora_strength=1.0,
                sample_steps=1,
                cfg_scale=1.0,
                dit_dtype="bfloat16",
            )
        )
        self.model.runner.config.vae.use_sample = False

    def __call__(self, received, region, seed=20261008):
        settings = control(received.shape, {}, region, seed)
        _, n, x, y, w, h = settings["generate"][0]
        x0, y0 = max(0, x - 64), max(0, y - 64)
        x1, y1 = min(received.shape[2], x + w + 64), min(received.shape[1], y + h + 64)
        pixels, runtime = self.model.restore(
            list(received[:, y0:y1, x0:x1]),
            seed=settings["seed"],
            processing_height=y1 - y0,
            processing_width=x1 - x0,
            output_height=y1 - y0,
            output_width=x1 - x0,
        )
        raw = np.stack(pixels)
        if raw.shape != (n, y1 - y0, x1 - x0, 3):
            raise ValueError("unexpected generated geometry")
        return (
            dict(
                region=region,
                pixels=raw,
                crop=[x0, y0, x1 - x0, y1 - y0],
                core=[x, y, w, h],
            ),
            runtime,
        )
