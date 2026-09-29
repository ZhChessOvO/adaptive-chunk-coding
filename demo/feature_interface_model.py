"""Frozen-generator interface study; explicit, receiver-reproducible ablations."""
import torch

from demo.feature_condition_model import FeatureCondition, FORMAT as FEATURE_FORMAT

FORMAT = 'frozen_rgb_lora_feature_interface_v1'
MODES = ('actual', 'zero', 'shuffle', 'off')


def packet_view(packets, mode):
    """Only transform received features; preserve their coverage and metadata.

    Shuffle uses a fixed spatial permutation within each received packet. It
    retains the channel vector at each location and requires no source/cache.
    This inference perturbation is not a replacement for a trained control.
    """
    if mode not in MODES:
        raise ValueError('unknown interface mode')
    if mode == 'off':
        return []
    result = []
    for packet in packets:
        delta = packet['delta']
        if delta is not None and mode == 'zero':
            delta = torch.zeros_like(delta)
        elif delta is not None and mode == 'shuffle':
            h, w = delta.shape[-2:]
            generator = torch.Generator(device='cpu').manual_seed(260930 + packet['packet_id'])
            order = torch.randperm(h*w, generator=generator).to(delta.device)
            delta = delta.flatten(-2).index_select(-1, order).reshape_as(delta)
        result.append(dict(packet, delta=delta))
    return result


class FeatureInterface(FeatureCondition):
    def __init__(self, mode='actual'):
        super().__init__()
        if mode not in MODES:
            raise ValueError('unknown interface mode')
        self.mode = mode

    def forward(self, condition, packets, **kwargs):
        # FP32 adapter arithmetic in both training and inference, followed by
        # the same explicit cast/add as v1. Do not silently change old profiles.
        with torch.autocast(condition.device.type, enabled=False):
            return super().forward(condition, packet_view(packets, self.mode), **kwargs)


@torch.no_grad()
def condition_statistics(raw, effective, side, coverage):
    covered = coverage.permute(0, 2, 3, 1).expand_as(raw) > 0
    delta = effective.float() - raw.float()
    values = delta[covered]
    proposed = side.float()[covered]
    return dict(condition_dtype=str(raw.dtype), coverage=float(coverage.mean()),
        side_rms=float(side.float().square().mean().sqrt()),
        effective_rms=float(delta.square().mean().sqrt()),
        covered_changed_fraction=float((values != 0).float().mean()) if values.numel() else 0.,
        covered_side_rms=float(proposed.square().mean().sqrt()) if proposed.numel() else 0.,
        rounded_away_fraction=float(((values == 0) & (proposed != 0)).float().mean()) if values.numel() else 0.,
        condition_rms=float(raw.float().square().mean().sqrt()),
        side_max=float(side.float().abs().max()),
        outside_coverage_exact=bool(torch.equal(effective[~covered], raw[~covered])))


def validate_bundle(bundle):
    mode = bundle.get('interface_mode')
    if (bundle.get('interface_format') != FORMAT or mode not in MODES or
            bundle.get('feature_format') != FEATURE_FORMAT or
            bundle.get('feature_enabled') != (mode != 'off')):
        raise ValueError('invalid frozen-generator interface bundle')
    return mode
