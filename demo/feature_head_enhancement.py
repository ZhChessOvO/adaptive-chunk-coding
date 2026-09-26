"""Display-only UF feature correction with an immutable pretrained renderer.

The packet still codes one joint P8 latent, not a replacement high-QP stream.
The receiver adds H(F+dF)-H(F) to the exact native base RGB. This anchoring
avoids changing the base because of FP32/FP16 or cropped-boundary differences.
The I frame uses the existing learned RGB branch; no P8 feature is fabricated.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from demo.chunk_enhancement_model import ChunkEnhancement, Residual
from demo.scalable_codec import file_hash
from src.models.video_model_ht import ReconHead
from src.utils.common import get_state_dict
from src.utils.transforms import ycbcr2rgb


UF_WEIGHTS = Path(__file__).resolve().parents[1] / "checkpoints/cvpr2026_video_hts.pth.tar"


class FeatureHeadEnhancement(ChunkEnhancement):
    FORMAT = "uf_feature_head_single_v1"
    # Four spatial 3x3 depthwise convolutions in each UF renderer path.
    feature_halo = 4

    def __init__(self, width=96, latent=64, hyper=24, head_sha256=None):
        super().__init__(width, latent, hyper)
        actual_hash = file_hash(UF_WEIGHTS)
        if head_sha256 is not None and head_sha256 != actual_hash:
            raise ValueError("frozen UF reconstruction checkpoint hash mismatch")
        self.config["head_sha256"] = actual_hash
        self.feature_synthesis = nn.Sequential(
            nn.Conv2d(latent + width, width, 3, padding=1), nn.GELU(),
            # Match the previous RGB branch's depth and entropy bottleneck.
            Residual(width), Residual(width), nn.Conv2d(width, 512, 1))
        nn.init.normal_(self.feature_synthesis[-1].weight, std=0.001)
        nn.init.zeros_(self.feature_synthesis[-1].bias)
        self.head = ReconHead(is_hts=True)
        weights = get_state_dict(str(UF_WEIGHTS))
        self.head.load_state_dict({k.removeprefix("recon_head."): v for k, v in weights.items()
                                   if k.startswith("recon_head.")}, strict=True)
        self.head.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.head.eval()
        return self

    def export_state(self):
        # The frozen official weights are already installed and hash-pinned.
        return {k: v for k, v in self.state_dict().items() if not k.startswith("head.")}

    def load_export_state(self, state):
        result = self.load_state_dict(state, strict=False)
        expected = {"head." + k for k in self.head.state_dict()}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f"invalid trainable enhancement state: {result}")

    def core(self, features):
        h = self.feature_halo
        return features[..., h:-h, h:-h]

    def condition(self, base, features, qstep, valid_count):
        return super().condition(base, self.core(features), qstep, valid_count)

    def render(self, feature):
        # UF's head returns centered YCbCr, not RGB. Keep floating values until
        # the anchored correction is formed, then clamp only the final display.
        return torch.cat([ycbcr2rgb(frame + 0.5, clamp=False)
                          for frame in self.head(feature)], dim=1)

    def apply_feature_delta(self, base, features, delta):
        h = self.feature_halo
        feature = features[:, :512].float()
        # Read-only halo costs no transmitted bits; it comes from the base.
        padded_delta = F.pad(delta, (h, h, h, h))
        with torch.no_grad():
            anchor = self.render(feature)
        corrected = self.render(feature + padded_delta)
        margin = h * 8
        difference = (corrected-anchor)[..., margin:-margin, margin:-margin]
        return (base + difference).clamp(0, 1)

    def reconstruct(self, base, c, y, qstep, features=None, valid_count=8):
        if features is None:
            raise ValueError("feature-head reconstruction requires decoded UF features")
        # The first chunk is explicitly zero-filled by decode_features. This
        # also supports a one-frame P tail without mistaking it for an I frame.
        if not torch.count_nonzero(self.core(features)).item():
            return super().reconstruct(base, c, y, qstep, features, valid_count)
        y = F.interpolate(y*qstep, size=c.shape[-2:], mode="nearest")
        delta = self.feature_synthesis(torch.cat((y, c), dim=1))
        return self.apply_feature_delta(base, features, delta)
