# Copyright (c) OpenMMLab. All rights reserved.
# Adapted and modified for this project under the Apache License 2.0.

"""PyTorch-only BasicVSR++ adapter for the released MMagic checkpoint.

This file mirrors the module names and forward equations used by MMagic's
BasicVSRPlusPlusNet, but replaces mmcv's compiled modulated deformable
convolution with torchvision.ops.deform_conv2d.  The adapter is tracked here
so a clean server checkout can run the deterministic restoration control
without compiling MMCV.  The upstream MMagic project is licensed under
Apache-2.0; attribution is recorded in ``NOTICE.txt``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


def flow_warp(x, flow, padding_mode="zeros"):
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError((x.shape, flow.shape))
    _, _, height, width = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=x.dtype),
        torch.arange(width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1) + flow
    grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(
        x,
        torch.stack((grid_x, grid_y), dim=-1),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


class ConvModule(nn.Module):
    """Minimal mmcv ConvModule name-compatible subset used by SPyNet."""

    def __init__(self, in_channels, out_channels, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3)
        self.activate = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x):
        return self.activate(self.conv(x))


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        channels = ((8, 32), (32, 64), (64, 32), (32, 16), (16, 2))
        self.basic_module = nn.Sequential(*[
            ConvModule(in_ch, out_ch, activate=index < 4)
            for index, (in_ch, out_ch) in enumerate(channels)
        ])

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        batch, _, height, width = ref.shape
        refs = [(ref - self.mean) / self.std]
        supps = [(supp - self.mean) / self.std]
        for _ in range(5):
            refs.append(F.avg_pool2d(refs[-1], 2, 2, count_include_pad=False))
            supps.append(F.avg_pool2d(supps[-1], 2, 2, count_include_pad=False))
        refs.reverse()
        supps.reverse()
        flow = ref.new_zeros(batch, 2, height // 32, width // 32)
        for level in range(6):
            if level:
                flow_up = F.interpolate(
                    flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            else:
                flow_up = flow
            warped = flow_warp(
                supps[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + self.basic_module[level](
                torch.cat((refs[level], warped, flow_up), dim=1))
        return flow

    def forward(self, ref, supp):
        height, width = ref.shape[-2:]
        up_height = math.ceil(height / 32) * 32
        up_width = math.ceil(width / 32) * 32
        ref_up = F.interpolate(
            ref, size=(up_height, up_width), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(
            supp, size=(up_height, up_width), mode="bilinear", align_corners=False)
        flow = F.interpolate(
            self.compute_flow(ref_up, supp_up),
            size=(height, width), mode="bilinear", align_corners=False)
        flow[:, 0] *= width / up_width
        flow[:, 1] *= height / up_height
        return flow


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        for layer in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
            layer.weight.data.mul_(0.1)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Sequential(*[
                ResidualBlockNoBN(out_channels) for _ in range(num_blocks)
            ]),
        )

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels, out_channels * scale_factor ** 2, 3, 1, 1)
        self.scale_factor = scale_factor
        nn.init.kaiming_normal_(self.upsample_conv.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.upsample_conv.bias)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class SecondOrderDeformableAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, deform_groups=16,
                 max_residue_magnitude=10):
        super().__init__()
        self.out_channels = out_channels
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * 9
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)

    def forward(self, x, extra_feat, flow_1, flow_2):
        offset_mask = self.conv_offset(
            torch.cat((extra_feat, flow_1, flow_2), dim=1))
        offset_1, offset_2, mask = torch.chunk(offset_mask, 3, dim=1)
        offsets = self.max_residue_magnitude * torch.tanh(
            torch.cat((offset_1, offset_2), dim=1))
        offset_1, offset_2 = torch.chunk(offsets, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(
            1, offset_1.shape[1] // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(
            1, offset_2.shape[1] // 2, 1, 1)
        return deform_conv2d(
            x,
            torch.cat((offset_1, offset_2), dim=1),
            self.weight,
            self.bias,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            mask=torch.sigmoid(mask),
        )


class BasicVSRPlusPlusFeatureBackbone(nn.Module):
    """Track-1 BasicVSR++ with a 64-channel, 32x32 feature output."""

    branch_names = ("backward_1", "forward_1", "backward_2", "forward_2")

    def __init__(self, mid_channels=128, num_blocks=25):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
        )
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for index, name in enumerate(self.branch_names):
            self.deform_align[name] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels)
            self.backbone[name] = ResidualBlocksWithInputConv(
                (2 + index) * mid_channels, mid_channels, num_blocks)
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def compute_flow(self, frames):
        batch, time, channels, height, width = frames.shape
        first = frames[:, :-1].reshape(-1, channels, height, width)
        second = frames[:, 1:].reshape(-1, channels, height, width)
        backward = self.spynet(first, second).view(
            batch, time - 1, 2, height, width)
        forward = self.spynet(second, first).view(
            batch, time - 1, 2, height, width)
        return forward, backward

    def propagate(self, feats, flows, name):
        batch, time_minus_one, _, height, width = flows.shape
        frame_indices = list(range(time_minus_one + 1))
        flow_indices = list(range(-1, time_minus_one))
        mapping = list(range(len(feats["spatial"])))
        mapping += mapping[::-1]
        if "backward" in name:
            frame_indices.reverse()
            flow_indices = frame_indices
        feat_prop = flows.new_zeros(batch, self.mid_channels, height, width)
        for iteration, index in enumerate(frame_indices):
            current = feats["spatial"][mapping[index]]
            if iteration > 0:
                flow_1 = flows[:, flow_indices[iteration]]
                cond_1 = flow_warp(feat_prop, flow_1.permute(0, 2, 3, 1))
                feat_2 = torch.zeros_like(feat_prop)
                flow_2 = torch.zeros_like(flow_1)
                cond_2 = torch.zeros_like(cond_1)
                if iteration > 1:
                    feat_2 = feats[name][-2]
                    flow_2 = flows[:, flow_indices[iteration - 1]]
                    flow_2 = flow_1 + flow_warp(
                        flow_2, flow_1.permute(0, 2, 3, 1))
                    cond_2 = flow_warp(feat_2, flow_2.permute(0, 2, 3, 1))
                condition = torch.cat((cond_1, current, cond_2), dim=1)
                feat_prop = self.deform_align[name](
                    torch.cat((feat_prop, feat_2), dim=1),
                    condition, flow_1, flow_2)
            merged = [current]
            merged.extend(
                feats[key][index]
                for key in feats if key not in ("spatial", name))
            merged.append(feat_prop)
            feat_prop = feat_prop + self.backbone[name](torch.cat(merged, dim=1))
            feats[name].append(feat_prop)
        if "backward" in name:
            feats[name].reverse()

    def forward_features(self, frames):
        """Return [B,T,64,32,32] decoder-side features for 256x256 RGB."""
        if frames.ndim != 5 or frames.shape[2:] != (3, 256, 256):
            raise ValueError(f"expected [B,T,3,256,256], got {tuple(frames.shape)}")
        batch, time = frames.shape[:2]
        low = F.interpolate(
            frames.reshape(-1, 3, 256, 256),
            scale_factor=0.25,
            mode="bicubic",
            align_corners=False,
        ).reshape(batch, time, 3, 64, 64)
        spatial = self.feat_extract(
            frames.reshape(-1, 3, 256, 256)).reshape(
                batch, time, self.mid_channels, 64, 64)
        feats = {"spatial": [spatial[:, index] for index in range(time)]}
        forward, backward = self.compute_flow(low)
        for name in self.branch_names:
            feats[name] = []
            self.propagate(feats, backward if "backward" in name else forward, name)
        outputs = []
        for index in range(time):
            merged = [feats["spatial"][index]]
            merged.extend(feats[name][index] for name in self.branch_names)
            feature = self.reconstruction(torch.cat(merged, dim=1))
            # Fixed, parameter-free fold keeps the E12 head width unchanged.
            feature = (feature[:, :64] + feature[:, 64:]) / math.sqrt(2.0)
            outputs.append(F.avg_pool2d(feature, 2))
        return torch.stack(outputs, dim=1)

    def forward(self, frames):
        return self.forward_features(frames)

    def load_released_checkpoint(self, checkpoint):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        state = {
            key.removeprefix("generator."): value
            for key, value in payload["state_dict"].items()
            if key.startswith("generator.")
        }
        incompatible = self.load_state_dict(state, strict=True)
        return {
            "checkpoint_tensors": len(state),
            "model_tensors": len(self.state_dict()),
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }


def make_random_backbone(seed):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        return BasicVSRPlusPlusFeatureBackbone()
    finally:
        torch.random.set_rng_state(state)
