"""Receiver base. Extracted from the validated RouterVC research implementation."""

from __future__ import annotations
from dataclasses import asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
import routervc.routers.inputs as visual
import routervc.routers.processing as mixed

FORMAT = "routervc_receiver_g_only_v1"
Config = visual.VisualRouterConfig
GAIN_NAMES = ("conditional_G_lpips", "conditional_G_psnr", "conditional_G_temporal")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def build_inputs(base, received, coverage, *, halo=0, config=None):
    """Only received uint8 [T,H,W,3] B/Y and packet-derived coverage are accepted."""
    _require(type(halo) is int and halo in (0, 64), "receiver halo must be 0 or 64")
    return mixed.build_inputs(
        base, received, coverage, halo=halo, config=config or Config()
    )


class ReceiverGUtilityRouter(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        _require(type(self.config) is Config, "expected receiver Config")
        c, hidden = (self.config.channels, self.config.hidden)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, c, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.body = nn.Sequential(
            nn.Linear(4 * c + 7, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden, 3)

    def _encode_time(self, frames):
        shape = frames.shape
        values = self.encoder(frames.reshape(-1, *shape[-3:])).flatten(1)
        values = values.reshape(*shape[:-3], self.config.channels)
        return torch.cat(
            (values.mean(-2), (values[..., -1, :] - values[..., 0, :]).abs()), dim=-1
        )

    def forward(self, inputs):
        _require(
            type(inputs) is dict and set(inputs) == set(visual.INPUT_KEYS),
            "receiver accepts only global_pairs, local_pairs, geometry and coverage",
        )
        _require(
            all(
                (
                    torch.is_tensor(v)
                    and torch.is_floating_point(v)
                    and torch.isfinite(v).all()
                    for v in inputs.values()
                )
            ),
            "receiver input tensors must be finite floating point",
        )
        local, global_ = (inputs["local_pairs"], inputs["global_pairs"])
        coverage, geometry = (inputs["coverage"], inputs["geometry"])
        k = self.config.temporal_frames
        _require(
            local.ndim == 6
            and local.shape[0] > 0
            and (1 <= local.shape[1] <= 16)
            and (local.shape[2:4] == (k, 6))
            and (local.shape[-2:] == (self.config.local_size,) * 2),
            "invalid local receiver tensor",
        )
        batch, regions = local.shape[:2]
        _require(
            global_.shape
            == (batch, k, 6, self.config.global_size, self.config.global_size)
            and coverage.shape == (batch, regions, 1)
            and (geometry.shape == (batch, regions, 6)),
            "invalid receiver global/coverage/geometry shapes",
        )
        _require(
            torch.all((coverage >= 0) & (coverage <= 1)), "coverage must be in [0,1]"
        )
        local_embedding = self._encode_time(local)
        global_embedding = (
            self._encode_time(global_)
            if self.config.use_global
            else local_embedding.new_zeros((batch, self.config.channels * 2))
        )
        fused = torch.cat(
            (
                local_embedding,
                global_embedding[:, None].expand(-1, regions, -1),
                geometry,
                coverage,
            ),
            dim=-1,
        )
        return self.output(self.body(fused))

    def metadata(self):
        return dict(
            format=FORMAT,
            architecture=asdict(self.config),
            parameters=sum((p.numel() for p in self.parameters())),
            experts=0,
            shared_encoder=True,
            gain_names=list(GAIN_NAMES),
            input_source="decoded_B_actual_Y_packet_coverage_only",
            sender_required=False,
            source_frames_used=False,
            direct_E_outputs=False,
            semantic_heads=False,
            transmitted_E_G_protection_mask=False,
        )


def training_loss(prediction, targets, *, gain_scale=None, ranking_weight=0.1):
    """Masked normalized Huber plus within-view LPIPS ordering (not across views).

    For valid pairs i<j with different measured LPIPS gains, the ranking term is
    softplus(-sign(t_i-t_j)*(p_i-p_j)/scale_lpips), weighted by
    min(abs(t_i-t_j)/scale_lpips,1)*min(w_i,w_j). Normalize per view, then average
    views with a known pair. Huber calibrates absolute gains, including harmful
    negative G gains, so ranking is not a requirement to use every G call.
    Scales affect the LOSS ONLY; predictions remain in the measured units.
    """
    _require(
        torch.is_tensor(prediction)
        and prediction.ndim == 3
        and (prediction.shape[-1] == 3)
        and (prediction.shape[0] > 0)
        and (1 <= prediction.shape[1] <= 16)
        and torch.is_floating_point(prediction)
        and torch.isfinite(prediction).all(),
        "expected finite [views,regions,3] G predictions",
    )
    _require(
        type(targets) is dict and set(targets) == {"value", "weight"},
        "invalid G target fields",
    )
    _require(
        all((torch.is_tensor(targets[k]) for k in targets)), "G targets must be tensors"
    )
    target = targets["value"].to(device=prediction.device, dtype=prediction.dtype)
    weight = targets["weight"].to(device=prediction.device, dtype=prediction.dtype)
    _require(
        target.shape == weight.shape == prediction.shape
        and torch.isfinite(weight).all()
        and torch.all(weight >= 0),
        "invalid G target shapes or weights",
    )
    known = weight > 0
    _require(torch.isfinite(target[known]).all(), "nonfinite known G target")
    scales = (
        prediction.new_ones(3)
        if gain_scale is None
        else torch.as_tensor(
            gain_scale, dtype=prediction.dtype, device=prediction.device
        )
    )
    _require(
        scales.shape == (3,) and torch.isfinite(scales).all() and torch.all(scales > 0),
        "G loss requires three positive train-only scales",
    )
    _require(
        type(ranking_weight) in (int, float)
        and math.isfinite(ranking_weight)
        and (ranking_weight >= 0),
        "invalid ranking coefficient",
    )
    zero = prediction.sum() * 0.0
    safe_target = torch.where(known, target, torch.zeros_like(target))
    metric_weight = prediction.new_tensor([1.0, 0.1, 0.1])
    weighted = weight * metric_weight
    element_loss = F.smooth_l1_loss(
        prediction / scales, safe_target / scales, reduction="none"
    )
    view_denominator = weighted.sum((1, 2))
    supervised_views = view_denominator > 0
    huber = (
        (
            (element_loss * weighted).sum((1, 2))[supervised_views]
            / view_denominator[supervised_views]
        ).mean()
        if supervised_views.any()
        else zero
    )
    truth = safe_target[..., 0]
    delta_truth = truth[:, :, None] - truth[:, None, :]
    lpips_prediction, lpips_known, lpips_weight = (
        prediction[..., 0],
        known[..., 0],
        weight[..., 0],
    )
    delta_prediction = (
        lpips_prediction[:, :, None] - lpips_prediction[:, None, :]
    ) / scales[0]
    valid_pairs = (
        lpips_known[:, :, None]
        & lpips_known[:, None, :]
        & torch.triu(
            torch.ones(
                (prediction.shape[1], prediction.shape[1]),
                dtype=torch.bool,
                device=prediction.device,
            ),
            diagonal=1,
        )[None]
        & (delta_truth != 0)
    )
    pair_weight = torch.minimum(lpips_weight[:, :, None], lpips_weight[:, None, :])
    pair_weight = (
        pair_weight * (delta_truth.abs() / scales[0]).clamp(max=1.0) * valid_pairs
    )
    pair_losses = F.softplus(-delta_truth.sign() * delta_prediction)
    pair_denominator = pair_weight.sum((1, 2))
    ranked_views = pair_denominator > 0
    ranking = (
        (
            (pair_losses * pair_weight).sum((1, 2))[ranked_views]
            / pair_denominator[ranked_views]
        ).mean()
        if ranked_views.any()
        else zero
    )
    return (huber + float(ranking_weight) * ranking, dict(huber=huber, ranking=ranking))
