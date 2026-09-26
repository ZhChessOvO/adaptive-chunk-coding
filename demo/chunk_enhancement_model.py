"""Single-layer, UF-feature-conditioned joint 8-frame enhancement codec.

Native UF rANS is reused, but the enhancement transforms and probability
predictors are newly trained. No UF parameters or reference buffers are changed.
"""
from __future__ import annotations

import math
import struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.entropy_models import EntropyCoder, GaussianEncoder


def quantize(x):
    rounded = x.round().clamp(-127, 127)
    return x + (rounded - x).detach()


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x) * 0.1


class GaussianStreams:
    """Independent z/y streams; signed 8-bit symbols with native escape coding."""
    def __init__(self):
        self.coder = EntropyCoder()
        self.gaussian = GaussianEncoder()
        self.gaussian.update(self.coder, skip_thres=0.0)
        self.coder.set_entropy_coder_parallel(1)

    @staticmethod
    def indexes(scale):
        step = (math.log(16.0) - math.log(0.11)) / 127
        return ((scale.detach().float().clamp(0.11, 16).log() - math.log(0.11))
                / step).round().clamp(0, 127).byte().cpu().numpy().reshape(-1)

    def encode(self, symbols, scales):
        values = symbols.detach().round().cpu().numpy().reshape(-1)
        if not np.isfinite(values).all() or values.min() < -127 or values.max() > 127:
            raise ValueError("enhancement symbol outside signed 8-bit coding range")
        indexes = self.indexes(scales)
        if len(indexes) != len(values):
            raise ValueError("entropy shape mismatch")
        combined = (values.astype(np.int16) * 256 + indexes.astype(np.int16)).astype(np.int16)
        self.coder.encoder.reset()
        self.coder.encoder.encode_y(combined)
        self.coder.encoder.flush()
        return self.coder.encoder.get_encoded_stream().tobytes()

    def decode(self, data, scales):
        if len(data) < 4:
            raise ValueError("truncated entropy stream")
        self.coder.decoder.set_stream(np.frombuffer(data, dtype=np.uint8).copy())
        self.coder.decoder.decode_y(self.indexes(scales))
        values = self.coder.decoder.get_decoded_tensor().copy()
        return torch.from_numpy(values).to(scales.device, torch.float32).reshape(scales.shape)


class ChunkEnhancement(nn.Module):
    """Joint temporal representation at H/16, with an H/64 hyperprior.

    Input is one 8-frame RGB chunk [B,24,H,W] and real decoded UF
    feature/context [B,1024,H/8,W/8]. Regions are padded to multiples of 64.
    I frames and short tails repeat the final frame; valid_count is signaled.
    """
    FORMAT = "uf_chunk_single_v1"
    spatial_alignment = 64

    def __init__(self, width=96, latent=64, hyper=24):
        super().__init__()
        self.config = dict(width=width, latent=latent, hyper=hyper)
        self.feature_condition = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.GELU(), Residual(64))
        self.rgb_condition = nn.Sequential(nn.Conv2d(24 * 64, 32, 1), nn.GELU())
        self.condition_mix = nn.Sequential(nn.Conv2d(98, width, 1), nn.GELU(), Residual(width))
        self.analysis = nn.Sequential(nn.Conv2d(24 * 64 + width, width, 1), nn.GELU(),
                                      Residual(width), Residual(width),
                                      nn.Conv2d(width, latent, 3, stride=2, padding=1))
        self.hyper_analysis = nn.Sequential(nn.Conv2d(latent, hyper, 3, stride=2, padding=1),
                                            nn.GELU(), nn.Conv2d(hyper, hyper, 3, stride=2, padding=1))
        self.z_prior = nn.Conv2d(width, hyper, 1)
        self.y_prior = nn.Sequential(nn.Conv2d(width + hyper, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, latent * 2, 1))
        self.synthesis = nn.Sequential(nn.Conv2d(latent + width, width, 3, padding=1), nn.GELU(),
                                       Residual(width), Residual(width),
                                       nn.Conv2d(width, 24 * 64, 1))
        nn.init.normal_(self.synthesis[-1].weight, std=0.001)
        nn.init.zeros_(self.synthesis[-1].bias)
        self.entropy = None

    def condition(self, base, features, qstep, valid_count):
        if base.ndim != 4 or base.shape[1] != 24 or any(v % self.spatial_alignment for v in base.shape[-2:]):
            raise ValueError(f"expected RGB padded to {self.spatial_alignment} pixels")
        expected = (base.shape[0], 1024, base.shape[2] // 8, base.shape[3] // 8)
        if tuple(features.shape) != expected or not 1 <= valid_count <= 8 or qstep <= 0:
            raise ValueError("invalid decoded feature shape, valid_count or qstep")
        feat = self.feature_condition(torch.asinh(features.float()))
        rgb = self.rgb_condition(F.pixel_unshuffle(base - 0.5, 8))
        flags = base.new_ones((base.shape[0], 2, *feat.shape[-2:]))
        flags[:, 0] *= math.log2(qstep)
        flags[:, 1] *= valid_count / 8
        return self.condition_mix(torch.cat((feat, rgb, flags), 1))

    def scales_z(self, c):
        return F.softplus(self.z_prior(F.avg_pool2d(c, 8))) + 0.11

    def prior_y(self, c, z):
        size = (c.shape[-2] // 2, c.shape[-1] // 2)
        params = self.y_prior(torch.cat((F.avg_pool2d(c, 2),
                                         F.interpolate(z, size=size, mode="nearest")), 1))
        means, raw_scale = params.chunk(2, 1)
        return means, F.softplus(raw_scale) + 0.11

    def reconstruct(self, base, c, y, qstep, features=None, valid_count=8):
        y = F.interpolate(y * qstep, size=c.shape[-2:], mode="nearest")
        delta = F.pixel_shuffle(self.synthesis(torch.cat((y, c), 1)), 8)
        return (base + delta).clamp(0, 1)

    def forward(self, source, base, features, qstep=1.0, valid_count=8):
        c = self.condition(base, features, qstep, valid_count)
        y = self.analysis(torch.cat((F.pixel_unshuffle(source - base, 8), c), 1)) / qstep
        z = self.hyper_analysis(y)
        z_hat = quantize(z)
        mean, scale = self.prior_y(c, z_hat)
        residual = y - mean
        symbols = quantize(residual)
        y_hat = symbols + mean
        # Smooth likelihood approximation; actual byte cost is separately measured.
        if self.training:
            z_rate = z + torch.empty_like(z).uniform_(-0.5, 0.5)
            y_rate = residual + torch.empty_like(residual).uniform_(-0.5, 0.5)
        else:
            z_rate, y_rate = z_hat, symbols
        bits = (-torch.log2(GaussianEncoder.get_prob_train(z_rate, self.scales_z(c))).sum()
                -torch.log2(GaussianEncoder.get_prob_train(y_rate, scale)).sum())
        return {"reconstruction": self.reconstruct(base, c, y_hat, qstep, features, valid_count), "bits": bits,
                "y_symbols": symbols, "z_symbols": z_hat,
                "saturated": ((residual.abs() > 127).sum() + (z.abs() > 127).sum()).detach()}

    def _entropy(self):
        if self.entropy is None:
            self.entropy = GaussianStreams()
        return self.entropy

    @torch.no_grad()
    def compress(self, source, base, features, qstep=1.0, valid_count=8):
        if self.training or source.shape[0] != 1:
            raise ValueError("real coding requires eval mode, one region per packet")
        c = self.condition(base, features, qstep, valid_count)
        y = self.analysis(torch.cat((F.pixel_unshuffle(source - base, 8), c), 1)) / qstep
        z = self.hyper_analysis(y).round()
        if z.abs().max() > 127:
            raise ValueError("hyperlatent saturation; train/adjust qstep before coding")
        z_stream = self._entropy().encode(z, self.scales_z(c))
        mean, scale = self.prior_y(c, z)
        symbols = (y - mean).round()
        y_stream = self._entropy().encode(symbols, scale)
        payload = struct.pack("<I", len(z_stream)) + z_stream + y_stream
        return payload, self.reconstruct(base, c, symbols + mean, qstep, features, valid_count), {
            "z_bytes": len(z_stream), "y_bytes": len(y_stream), "payload_header_bytes": 4}

    @torch.no_grad()
    def decompress(self, payload, base, features, qstep=1.0, valid_count=8):
        if self.training or len(payload) < 12:
            raise ValueError("invalid decode mode or payload")
        nz, = struct.unpack_from("<I", payload)
        if nz < 4 or nz > len(payload) - 8:
            raise ValueError("invalid hyperprior length")
        c = self.condition(base, features, qstep, valid_count)
        z = self._entropy().decode(payload[4:4+nz], self.scales_z(c))
        mean, scale = self.prior_y(c, z)
        y = self._entropy().decode(payload[4+nz:], scale) + mean
        return self.reconstruct(base, c, y, qstep, features, valid_count)
