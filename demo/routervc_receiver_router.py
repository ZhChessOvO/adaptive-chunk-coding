"""Independent receiver R_g: decoded B/Y/coverage -> three conditional G gains.

This is not the sender Router with a different input switch. It has no E output,
no semantic heads and no sender/source/teacher dependency at inference. The
small 2D global/local encoder is initialized from the historical mixed Router;
core and halo are independently trained, equal-capacity input profiles, not
experts. The sender will use a separate spatiotemporal architecture and weights.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from demo import routervc_visual_router as visual
from demo import routervc_mixed_router as mixed

FORMAT = 'routervc_receiver_g_only_v1'
POLICY_FORMAT = 'routervc_receiver_g_policy_v1'
Config = visual.VisualRouterConfig
ARMS = ('core', 'halo')
INPUT_HALOS = {'core': 0, 'halo': 64}
GAIN_NAMES = ('conditional_G_lpips', 'conditional_G_psnr', 'conditional_G_temporal')
G_INDICES = (1, 3, 5)
CPU_THREADS = 4
CODE = ('routervc_receiver_router.py', 'routervc_visual_router.py', 'routervc_mixed_router.py')
PAYLOAD_KEYS = {'format', 'architecture', 'arm', 'input_halo', 'state_dict', 'binding',
                'epoch', 'selection', 'semantic_supervision', 'content_heads_usable',
                'metadata', 'code'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def code_identity():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in CODE}


def policy_identity():
    binding = dict(format=POLICY_FORMAT, code=code_identity(), gain_names=list(GAIN_NAMES),
                   cpu_threads=CPU_THREADS, sender_required=False)
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_inputs(base, received, coverage, *, halo=0, config=None):
    """Only received uint8 [T,H,W,3] B/Y and packet-derived coverage are accepted."""
    _require(type(halo) is int and halo in (0, 64), 'receiver halo must be 0 or 64')
    return mixed.build_inputs(base, received, coverage, halo=halo, config=config or Config())


class ReceiverGUtilityRouter(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        _require(type(self.config) is Config, 'expected receiver Config')
        c, hidden = self.config.channels, self.config.hidden
        self.encoder = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, c, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.body = nn.Sequential(nn.Linear(4*c + 7, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.output = nn.Linear(hidden, 3)

    def _encode_time(self, frames):
        shape = frames.shape
        values = self.encoder(frames.reshape(-1, *shape[-3:])).flatten(1)
        values = values.reshape(*shape[:-3], self.config.channels)
        return torch.cat((values.mean(-2), (values[..., -1, :] - values[..., 0, :]).abs()), dim=-1)

    def forward(self, inputs):
        _require(type(inputs) is dict and set(inputs) == set(visual.INPUT_KEYS),
                 'receiver accepts only global_pairs, local_pairs, geometry and coverage')
        _require(all(torch.is_tensor(v) and torch.is_floating_point(v) and torch.isfinite(v).all()
                     for v in inputs.values()), 'receiver input tensors must be finite floating point')
        local, global_ = inputs['local_pairs'], inputs['global_pairs']
        coverage, geometry = inputs['coverage'], inputs['geometry']
        k = self.config.temporal_frames
        _require(local.ndim == 6 and local.shape[0] > 0 and 1 <= local.shape[1] <= 16
                 and local.shape[2:4] == (k, 6)
                 and local.shape[-2:] == (self.config.local_size,)*2, 'invalid local receiver tensor')
        batch, regions = local.shape[:2]
        _require(global_.shape == (batch, k, 6, self.config.global_size, self.config.global_size)
                 and coverage.shape == (batch, regions, 1) and geometry.shape == (batch, regions, 6),
                 'invalid receiver global/coverage/geometry shapes')
        _require(torch.all((coverage >= 0) & (coverage <= 1)), 'coverage must be in [0,1]')
        local_embedding = self._encode_time(local)
        global_embedding = (self._encode_time(global_) if self.config.use_global else
                            local_embedding.new_zeros((batch, self.config.channels*2)))
        fused = torch.cat((local_embedding, global_embedding[:, None].expand(-1, regions, -1),
                           geometry, coverage), dim=-1)
        # G may help without E. Never gate these gains by enhancement coverage.
        return self.output(self.body(fused))

    def metadata(self):
        return dict(format=FORMAT, architecture=asdict(self.config),
            parameters=sum(p.numel() for p in self.parameters()), experts=0, shared_encoder=True,
            gain_names=list(GAIN_NAMES), input_source='decoded_B_actual_Y_packet_coverage_only',
            sender_required=False, source_frames_used=False, direct_E_outputs=False,
            semantic_heads=False, transmitted_E_G_protection_mask=False)


def initialize_from_mixed(model, legacy_state):
    """Transfer encoder/body and historical G rows [1,3,5], not E/semantic heads."""
    _require(type(model) is ReceiverGUtilityRouter and isinstance(legacy_state, dict),
             'expected new receiver model and historical state dictionary')
    state, expected = {}, model.state_dict()
    _require(set(legacy_state) == set(expected), 'historical state keys differ')
    for key, template in expected.items():
        value = legacy_state[key]
        _require(torch.is_tensor(value) and value.dtype == template.dtype and torch.isfinite(value).all(),
                 f'invalid historical parameter: {key}')
        if key in ('output.weight', 'output.bias'):
            _require(value.shape == (18, *template.shape[1:]), 'expected historical 18-output head')
            value = value[list(G_INDICES)]
        _require(value.shape == template.shape, f'historical architecture differs: {key}')
        state[key] = value.detach().clone()
    model.load_state_dict(state, strict=True)
    return model


def g_only_targets(old_targets):
    """Extract measured G|actual-Y columns; unknowns and negative gains survive."""
    source = old_targets.get('gains', old_targets)
    _require(isinstance(source, dict) and set(source) == {'value', 'weight'}, 'invalid gain target fields')
    value, weight = source['value'], source['weight']
    _require(torch.is_tensor(value) and torch.is_tensor(weight) and value.shape == weight.shape
             and value.ndim == 3 and value.shape[-1] == 6, 'expected old [views,regions,6] gain targets')
    return {name: tensor[..., list(G_INDICES)].clone() for name, tensor in source.items()}


def training_loss(prediction, targets, *, gain_scale=None, ranking_weight=.1):
    """Masked normalized Huber plus within-view LPIPS ordering (not across views).

    For valid pairs i<j with different measured LPIPS gains, the ranking term is
    softplus(-sign(t_i-t_j)*(p_i-p_j)/scale_lpips), weighted by
    min(abs(t_i-t_j)/scale_lpips,1)*min(w_i,w_j). Normalize per view, then average
    views with a known pair. Huber calibrates absolute gains, including harmful
    negative G gains, so ranking is not a requirement to use every G call.
    Scales affect the LOSS ONLY; predictions remain in the measured units.
    """
    _require(torch.is_tensor(prediction) and prediction.ndim == 3 and prediction.shape[-1] == 3
             and prediction.shape[0] > 0 and 1 <= prediction.shape[1] <= 16
             and torch.is_floating_point(prediction) and torch.isfinite(prediction).all(),
             'expected finite [views,regions,3] G predictions')
    _require(type(targets) is dict and set(targets) == {'value', 'weight'}, 'invalid G target fields')
    _require(all(torch.is_tensor(targets[k]) for k in targets), 'G targets must be tensors')
    target = targets['value'].to(device=prediction.device, dtype=prediction.dtype)
    weight = targets['weight'].to(device=prediction.device, dtype=prediction.dtype)
    _require(target.shape == weight.shape == prediction.shape and torch.isfinite(weight).all()
             and torch.all(weight >= 0), 'invalid G target shapes or weights')
    known = weight > 0
    _require(torch.isfinite(target[known]).all(), 'nonfinite known G target')
    scales = prediction.new_ones(3) if gain_scale is None else torch.as_tensor(
        gain_scale, dtype=prediction.dtype, device=prediction.device)
    _require(scales.shape == (3,) and torch.isfinite(scales).all() and torch.all(scales > 0),
             'G loss requires three positive train-only scales')
    _require(type(ranking_weight) in (int, float) and math.isfinite(ranking_weight)
             and ranking_weight >= 0, 'invalid ranking coefficient')
    zero = prediction.sum()*0.
    safe_target = torch.where(known, target, torch.zeros_like(target))
    metric_weight = prediction.new_tensor([1., .1, .1])
    weighted = weight*metric_weight
    element_loss = F.smooth_l1_loss(prediction/scales, safe_target/scales, reduction='none')
    view_denominator = weighted.sum((1, 2))
    supervised_views = view_denominator > 0
    huber = ((element_loss*weighted).sum((1, 2))[supervised_views]
             / view_denominator[supervised_views]).mean() if supervised_views.any() else zero

    truth = safe_target[..., 0]
    delta_truth = truth[:, :, None] - truth[:, None, :]
    lpips_prediction, lpips_known, lpips_weight = prediction[..., 0], known[..., 0], weight[..., 0]
    delta_prediction = (lpips_prediction[:, :, None] - lpips_prediction[:, None, :])/scales[0]
    valid_pairs = (lpips_known[:, :, None] & lpips_known[:, None, :]
        & torch.triu(torch.ones((prediction.shape[1], prediction.shape[1]), dtype=torch.bool,
                               device=prediction.device), diagonal=1)[None]
        & (delta_truth != 0))
    pair_weight = torch.minimum(lpips_weight[:, :, None], lpips_weight[:, None, :])
    pair_weight = pair_weight * (delta_truth.abs()/scales[0]).clamp(max=1.) * valid_pairs
    pair_losses = F.softplus(-delta_truth.sign()*delta_prediction)
    pair_denominator = pair_weight.sum((1, 2))
    ranked_views = pair_denominator > 0
    ranking = ((pair_losses*pair_weight).sum((1, 2))[ranked_views]
               / pair_denominator[ranked_views]).mean() if ranked_views.any() else zero
    return huber + float(ranking_weight)*ranking, dict(huber=huber, ranking=ranking)


def regret(prediction, target, caps=(4, 8)):
    _require(isinstance(caps, (list, tuple)) and len(caps) > 0
             and all(type(v) is int and 0 <= v <= 16 for v in caps), 'invalid receiver G caps')
    return mixed.regret(prediction, target, caps)


def export_payload(model, arm, binding, epoch, selection_metric):
    _require(type(model) is ReceiverGUtilityRouter and arm in ARMS, 'invalid receiver model/profile')
    _require(type(binding) is dict and type(selection_metric) is dict, 'binding and selection must be dictionaries')
    _require(type(epoch) is int and epoch > 0, 'invalid receiver epoch')
    _require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'nonfinite receiver weights')
    return dict(format=FORMAT, architecture=asdict(model.config), arm=arm, input_halo=INPUT_HALOS[arm],
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        binding=binding, epoch=epoch, selection=selection_metric, semantic_supervision=False,
        content_heads_usable=False, metadata=model.metadata(), code=code_identity())


def load_model(path, *, expected_sha256=None):
    """Authenticate CPU-only R_g. Never open a sender, source or teacher asset."""
    _require(expected_sha256 is None or _valid_hash(expected_sha256), 'invalid expected receiver hash')
    wire = Path(path).read_bytes()
    actual = hashlib.sha256(wire).hexdigest()
    _require(expected_sha256 is None or actual == expected_sha256, 'receiver checkpoint hash mismatch')
    try:
        payload = torch.load(io.BytesIO(wire), weights_only=True, map_location='cpu')
    except Exception as error:
        raise ValueError('invalid receiver checkpoint') from error
    _require(type(payload) is dict and set(payload) == PAYLOAD_KEYS, 'invalid receiver checkpoint fields')
    _require(payload['format'] == FORMAT and type(payload['arm']) is str and payload['arm'] in ARMS,
             'not a receiver G-only checkpoint')
    _require(type(payload['input_halo']) is int and payload['input_halo'] == INPUT_HALOS[payload['arm']],
             'receiver arm/halo mismatch')
    _require(payload['semantic_supervision'] is False and payload['content_heads_usable'] is False,
             'receiver must not claim semantic supervision')
    _require(payload['code'] == code_identity(), 'receiver preprocessing/source identity changed')
    architecture = payload['architecture']
    _require(type(architecture) is dict and set(architecture) == {f.name for f in fields(Config)},
             'invalid receiver architecture fields')
    try:
        config = Config(**architecture)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid receiver architecture') from error
    _require(type(payload['binding']) is dict and type(payload['selection']) is dict
             and type(payload['epoch']) is int and payload['epoch'] > 0, 'invalid training provenance')
    with torch.random.fork_rng(devices=[]):
        model = ReceiverGUtilityRouter(config)
    _require(payload['metadata'] == model.metadata(), 'receiver metadata/architecture mismatch')
    expected, state = model.state_dict(), payload['state_dict']
    _require(isinstance(state, dict) and set(state) == set(expected), 'invalid receiver parameter keys')
    for name, template in expected.items():
        value = state[name]
        _require(torch.is_tensor(value) and value.device.type == 'cpu' and value.layout == torch.strided
                 and value.dtype == template.dtype and value.shape == template.shape
                 and torch.isfinite(value).all(), f'invalid/nonfinite receiver parameter: {name}')
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    model._receiver_model_sha256 = actual
    model._receiver_input_halo = payload['input_halo']
    model._receiver_architecture = architecture.copy()
    torch.set_num_threads(CPU_THREADS)
    return model, payload


@torch.no_grad()
def predict(model, base, received, coverage):
    _require(type(model) is ReceiverGUtilityRouter
             and _valid_hash(getattr(model, '_receiver_model_sha256', None))
             and type(getattr(model, '_receiver_input_halo', None)) is int
             and getattr(model, '_receiver_input_halo', None) in (0, 64)
             and getattr(model, '_receiver_architecture', None) == asdict(model.config),
             'predict requires an authenticated receiver checkpoint')
    _require(all(v.device.type == 'cpu' and v.dtype == torch.float32 and torch.isfinite(v).all()
                 for v in (*model.parameters(), *model.buffers())), 'receiver requires finite CPU float32 weights')
    torch.set_num_threads(CPU_THREADS)
    model.eval()
    result = model(build_inputs(base, received, coverage, halo=model._receiver_input_halo, config=model.config))
    _require(result.shape == (1, 16, 3) and torch.isfinite(result).all(), 'invalid receiver G prediction')
    return result.detach()
