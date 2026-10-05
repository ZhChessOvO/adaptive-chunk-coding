"""Independent source-aware R_s for conditional final-quality E allocation.

Every frame of the current window enters a multiscale 3D global encoder. Local
sampled-frame detail and error crops, actual candidate bundle bytes, received
coverage and the receiver G-call budget condition a 4x4 grid-interaction body.
One scalar per region predicts the FINAL LPIPS reduction after the fixed R_g/G
pipeline, not direct E gain. Measuring those labels is a separate future stage;
this module does not pretend that untrained outputs or direct-E labels are it.

The source-aware architecture and weights are independent of R_g. No source,
scores, E/G/protection mask or sender checkpoint is part of the receiving API.
Full view means the entire supplied REDS view or existing UVG crop, not missing
original UVG frames. Full-resolution candidate preparation is a separate cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'routervc_source_aware_sender_v1'
LABEL_SCOPE = 'final_lpips_marginal_after_fixed_receiver_and_generation'
REGIONS = 16
CPU_THREADS = 4
INPUT_KEYS = {'global_video', 'global_extent', 'local_video', 'geometry', 'coverage', 'packet_bytes', 'max_g'}
PAYLOAD_KEYS = {'format', 'architecture', 'state_dict', 'binding', 'step', 'selection',
                'metadata', 'code'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class SenderConfig:
    frame_count: int = 17
    global_size: int = 96
    local_size: int = 32
    detail_frames: int = 3
    channels: int = 24
    detail_channels: int = 16
    hidden: int = 64
    temporal_bins: int = 3
    zero_source: bool = False

    def __post_init__(self):
        require(type(self.zero_source) is bool, 'zero_source must be boolean')
        for field in fields(self):
            if field.name != 'zero_source':
                value = getattr(self, field.name)
                require(type(value) is int and value > 0, 'sender dimensions must be positive integers')
        require(self.frame_count >= 17 and (self.frame_count-1) % 8 == 0,
                'sender windows require I plus complete P8 chunks, at least 17 frames')
        require(self.global_size >= 32 and self.global_size % 2 == 0 and self.local_size >= 16,
                'sender global/local image sizes are too small or invalid')
        require(self.detail_frames <= self.frame_count and self.temporal_bins <= self.frame_count,
                'sender temporal sampling exceeds the window')


Config = SenderConfig


def grid_rois(height, width):
    require(all(type(v) is int and v >= 64 and v % 8 == 0 for v in (height, width)),
            'coded geometry must be >=64 and divisible by eight')
    xs = [8*(i*(width//8)//4) for i in range(5)]
    ys = [8*(i*(height//8)//4) for i in range(5)]
    return [[xs[x], ys[y], xs[x+1]-xs[x], ys[y+1]-ys[y]]
            for y in range(4) for x in range(4)]


def _resized(frames, size):
    """Aspect-preserving deterministic uint8 RGB -> [T,3,S,S] in [0,1]."""
    pixels = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()/255.
    height, width = pixels.shape[-2:]
    ratio = size/max(height, width)
    h, w = max(1, round(height*ratio)), max(1, round(width*ratio))
    if (h, w) != (height, width):
        pixels = F.interpolate(pixels, (h, w), mode='bilinear', align_corners=False, antialias=True)
    left, top = (size-w)//2, (size-h)//2
    return F.pad(pixels, (left, size-w-left, top, size-h-top), mode='replicate')


def _video_fields(x, b, y, candidate, *, zero_source, local):
    # Ablation removes X and BOTH source residuals, not just the source image.
    xb, xy = x-b, x-y
    if zero_source:
        x, xb, xy = torch.zeros_like(x), torch.zeros_like(xb), torch.zeros_like(xy)
    fields_ = [x, b, y, candidate, xb, xy]
    if local:
        fields_.append(candidate-y)
    return torch.cat(fields_, dim=1).permute(1, 0, 2, 3).contiguous()


def _temporal_average(features, bins):
    """Adaptive temporal averages expressed as fixed dense weights, not scatter.

CUDA AdaptiveAvgPool3d backward is documented as nondeterministic. Fixed
start/end bin weights with multiply/reduce preserve its forward definition.
The final GPU interrupted/resumed test is still required for the whole model.
"""
    frames = features.shape[2]
    positions = torch.arange(frames, device=features.device)
    starts = torch.tensor([i*frames//bins for i in range(bins)], device=features.device)
    ends = torch.tensor([((i+1)*frames+bins-1)//bins for i in range(bins)], device=features.device)
    mask = (positions[None] >= starts[:, None]) & (positions[None] < ends[:, None])
    weights = mask.to(features.dtype)/(ends-starts)[:, None].to(features.dtype)
    return (features[:, :, None]*weights[None, None, :, :, None, None]).sum(3)


def _aligned_picture(features, extent, samples=8):
    """Bilinear border sampling via separable fixed weights, no grid_sample.

Extent is non-learned letterbox metadata. We use the same align_corners=False
pixel-center mapping, then differentiable dense multiply/reductions for image
features. No CUDA scatter-style sampling backward or indexed accumulation.
"""
    h, w = features.shape[-2:]
    centers = (torch.arange(samples, device=features.device, dtype=features.dtype)+.5)/samples
    extent = extent.detach().to(features.dtype)
    x = ((extent[:, 0, None]+extent[:, 2, None]*centers[None])*w-.5).clamp(0, w-1)
    y = ((extent[:, 1, None]+extent[:, 3, None]*centers[None])*h-.5).clamp(0, h-1)
    wx = (1-(torch.arange(w, device=features.device, dtype=features.dtype)[None, :, None]-x[:, None]).abs()).clamp_min(0)
    wy = (1-(torch.arange(h, device=features.device, dtype=features.dtype)[None, :, None]-y[:, None]).abs()).clamp_min(0)
    across_x = (features[..., None]*wx[:, None, None, None]).sum(-2)
    return (across_x[..., None, :]*wy[:, None, None, :, :, None]).sum(-3)


class _MeanVolume(nn.Module):
    """Global detail average without CUDA adaptive-pooling backward."""
    def forward(self, features):
        return features.mean((-3, -2, -1), keepdim=True)


def build_inputs(source, base, received, candidate_enhanced, coverage, packet_bytes, max_g, config=None):
    """Return CPU inputs for one source-aware sender decision, without mutation.

candidate_enhanced is the actual entropy-decoded full candidate E image, not a
target/source substitute. packet_bytes contains the ACTUAL additional bytes to
complete each candidate region (including required addressing), never estimated
entropy or image sizes. For an already complete region the value may be zero;
the planner excludes its coverage=1 entry. No labels enter this function.
"""
    config = config or Config()
    require(type(config) is Config, 'expected SenderConfig')
    images = [np.asarray(v) for v in (source, base, received, candidate_enhanced)]
    shape = images[0].shape
    require(len(shape) == 4 and shape[0] == config.frame_count and shape[-1] == 3
            and all(v.shape == shape and v.dtype == np.uint8 for v in images),
            'sender requires matched uint8 [all T,H,W,3] X/B/Y/candidate windows')
    rois = grid_rois(shape[1], shape[2])
    coverage = np.asarray(coverage)
    packet_bytes = np.asarray(packet_bytes)
    require(coverage.shape == (REGIONS,) and coverage.dtype.kind in 'biuf'
            and np.isfinite(coverage).all() and np.all((coverage >= 0) & (coverage <= 1)),
            'coverage requires sixteen finite received fractions')
    require(packet_bytes.shape == (REGIONS,) and packet_bytes.dtype.kind in 'iu'
            and np.all(packet_bytes >= 0) and np.all(packet_bytes[coverage < 1] > 0)
            and np.all(packet_bytes < 2**53), 'actual candidate byte costs must be valid positive integers')
    require(type(max_g) is int and 0 <= max_g <= REGIONS, 'invalid receiver G call budget')
    global_video = _video_fields(*[_resized(v, config.global_size) for v in images],
                                zero_source=config.zero_source, local=False)
    sampled = np.rint(np.linspace(0, shape[0]-1, config.detail_frames)).astype(int)
    local = []
    for x, y, w, h in rois:
        fields_ = [_resized(v[sampled, y:y+h, x:x+w], config.local_size) for v in images]
        local.append(_video_fields(*fields_, zero_source=config.zero_source, local=True))
    height, width = shape[1:3]
    geometry = [[(x+w/2)/width, (y+h/2)/height, w/width, h/height, w/512., h/512.]
                for x, y, w, h in rois]
    scaled_h = max(1, round(height*config.global_size/max(height, width)))
    scaled_w = max(1, round(width*config.global_size/max(height, width)))
    left, top = (config.global_size-scaled_w)//2, (config.global_size-scaled_h)//2
    # Keep padded thumbnail coordinates distinct from the original region grid.
    extent = [left/config.global_size, top/config.global_size,
              scaled_w/config.global_size, scaled_h/config.global_size]
    return dict(global_video=global_video.unsqueeze(0), local_video=torch.stack(local).unsqueeze(0),
        global_extent=torch.tensor([extent], dtype=torch.float32),
        geometry=torch.tensor(geometry, dtype=torch.float32).unsqueeze(0),
        coverage=torch.tensor(coverage, dtype=torch.float32).reshape(1, REGIONS, 1),
        packet_bytes=torch.tensor(packet_bytes.astype(np.int64), dtype=torch.int64).reshape(1, REGIONS, 1),
        max_g=torch.tensor([[max_g]], dtype=torch.int64))


class SenderUtilityRouter(nn.Module):
    """Multiscale 3D -> spatial tokens -> two grid mixing layers -> signed delta."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        require(type(self.config) is Config, 'expected SenderConfig')
        c, d, hidden = self.config.channels, self.config.detail_channels, self.config.hidden
        self.video_encoder = nn.Sequential(
            nn.Conv3d(18, 16, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(16, c, 3, stride=(2, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(c, c, 3, stride=(2, 2, 2), padding=1), nn.GELU())
        self.detail_encoder = nn.Sequential(
            nn.Conv3d(21, d, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(d, d, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
            _MeanVolume())
        # +10 = core geometry(6), coverage(1), logbytes/byte share(2), G budget(1).
        self.token_projection = nn.Sequential(nn.Linear(2*c*self.config.temporal_bins+d+10, hidden), nn.GELU())
        self.grid_context = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.final_gain = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _global_tokens(self, video, extent):
        features = self.video_encoder(video)
        n, c, _, h, w = features.shape
        time = _temporal_average(features, self.config.temporal_bins)
        # Align 4x4 region tokens to the actual picture, not letterbox padding.
        # Four feature samples per cell retain low-resolution global context.
        aligned = _aligned_picture(time, extent)
        pooled = aligned.reshape(n, c, self.config.temporal_bins, 4, 2, 4, 2).mean(-1).mean(-2)
        return pooled.permute(0, 3, 4, 1, 2).reshape(video.shape[0], REGIONS, -1)

    def forward(self, inputs):
        require(type(inputs) is dict and set(inputs) == INPUT_KEYS, 'unexpected source-aware sender input fields')
        floats = ('global_video', 'global_extent', 'local_video', 'geometry', 'coverage')
        require(all(torch.is_tensor(inputs[k]) and torch.is_floating_point(inputs[k])
                    and torch.isfinite(inputs[k]).all() for k in floats), 'sender image/features must be finite floats')
        video, local = inputs['global_video'], inputs['local_video']
        geometry, coverage = inputs['geometry'], inputs['coverage']
        extent = inputs['global_extent']
        costs, max_g = inputs['packet_bytes'], inputs['max_g']
        require(video.ndim == 5 and video.shape[0] > 0, 'invalid sender video batch')
        n = video.shape[0]
        cfg = self.config
        require(video.shape[1:] == (18, cfg.frame_count, cfg.global_size, cfg.global_size)
                and local.shape == (n, REGIONS, 21, cfg.detail_frames, cfg.local_size, cfg.local_size)
                and geometry.shape == (n, REGIONS, 6) and coverage.shape == (n, REGIONS, 1),
                'sender tensor geometry differs from architecture')
        require(extent.shape == (n, 4) and torch.all(extent[:, :2] >= 0)
                and torch.all(extent[:, 2:] > 0) and torch.all(extent[:, :2]+extent[:, 2:] <= 1),
                'invalid original-picture extent in global letterbox')
        require(torch.is_tensor(costs) and costs.dtype == torch.int64 and costs.shape == (n, REGIONS, 1)
                and torch.all(costs >= 0) and torch.all(costs[coverage < 1] > 0)
                and torch.all(costs < 2**53), 'sender requires actual integer packet costs')
        require(torch.is_tensor(max_g) and max_g.dtype == torch.int64 and max_g.shape == (n, 1)
                and torch.all((max_g >= 0) & (max_g <= REGIONS)), 'invalid sender G budget')
        require(torch.all((coverage >= 0) & (coverage <= 1)), 'invalid sender received coverage')
        require(len({v.device for v in inputs.values()}) == 1, 'sender input devices differ')
        if cfg.zero_source:
            require(torch.count_nonzero(video[:, [*range(3), *range(12, 18)]]) == 0
                    and torch.count_nonzero(local[:, :, [*range(3), *range(12, 18)]]) == 0,
                    'no-source ablation leaked source or residual channels')
        # Exact factor2 spatial resize is a 2x2 average; no temporal subsampling
        # and no differentiable CUDA interpolation operation are needed.
        small = video.reshape(n, 18, cfg.frame_count, cfg.global_size//2, 2,
                              cfg.global_size//2, 2).mean(-1).mean(-2)
        high_tokens, low_tokens = self._global_tokens(video, extent), self._global_tokens(small, extent)
        detail = self.detail_encoder(local.reshape(n*REGIONS, *local.shape[2:])).reshape(n, REGIONS, -1)
        float_costs = costs.to(video.dtype)
        costs_features = torch.cat((torch.log1p(float_costs)/math.log1p(2**20),
                                    float_costs/float_costs.sum(1, keepdim=True).clamp_min(1)), dim=-1)
        g_features = (max_g.to(video.dtype)/REGIONS)[:, None].expand(-1, REGIONS, -1)
        tokens = self.token_projection(torch.cat((high_tokens, low_tokens, detail,
                                                  geometry, coverage, costs_features, g_features), dim=-1))
        grid = tokens.transpose(1, 2).reshape(n, cfg.hidden, 4, 4)
        context = self.grid_context(grid).flatten(2).transpose(1, 2)
        # Signed, unbounded output: negative final benefit must remain representable.
        return self.final_gain(torch.cat((tokens, context), dim=-1)).squeeze(-1)

    def metadata(self):
        return dict(format=FORMAT, architecture=asdict(self.config),
            parameters=sum(p.numel() for p in self.parameters()),
            architecture_family='multiscale_3D_plus_local_details_plus_4x4_grid_interaction',
            output='signed_conditional_final_lpips_gain_per_E_bundle', label_scope=LABEL_SCOPE,
            global_frame_indices=list(range(self.config.frame_count)),
            detail_frame_indices=np.rint(np.linspace(0, self.config.frame_count-1,
                                                     self.config.detail_frames)).astype(int).tolist(),
            source_used=not self.config.zero_source, source_residuals_used=not self.config.zero_source,
            current_view_only=True, candidate_preparation_cost_included=False,
            diffusion_executed=False, receiver_weights_shared=False,
            sender_model_transmitted=False, mask_transmitted=False, semantic_protection_trained=False)


def training_loss(prediction, targets, packet_bytes, *, gain_scale=1., ranking_weight=.1):
    """Masked final-gain Huber plus within-state true-gain/byte weighted ranking.

The label_scope field must explicitly attest fixed R_g/G final-output labels.
This is a data contract, not proof those labels have already been measured.
Unknown labels have weight zero; negative benefits are retained, not clamped.
Scales normalize only the loss, not the deployed model's LPIPS-unit output.
"""
    require(torch.is_tensor(prediction) and prediction.ndim == 2 and prediction.shape[1] == REGIONS
            and prediction.shape[0] > 0 and torch.is_floating_point(prediction)
            and torch.isfinite(prediction).all(), 'expected finite [batch,16] sender prediction')
    require(type(targets) is dict and set(targets) == {'value', 'weight', 'label_scope'}
            and targets['label_scope'] == LABEL_SCOPE, 'sender requires measured FINAL fixed-receiver labels')
    require(torch.is_tensor(targets['value']) and torch.is_tensor(targets['weight']), 'targets must be tensors')
    value = targets['value'].to(prediction)
    weight = targets['weight'].to(prediction)
    require(value.shape == weight.shape == prediction.shape and torch.isfinite(weight).all()
            and torch.all(weight >= 0), 'invalid sender target shapes or weights')
    known = weight > 0
    require(torch.isfinite(value[known]).all(), 'nonfinite known sender final gain')
    require(torch.is_tensor(packet_bytes) and packet_bytes.dtype == torch.int64,
            'loss requires actual integer packet costs')
    costs = packet_bytes.to(prediction.device)
    if costs.shape == (*prediction.shape, 1):
        costs = costs.squeeze(-1)
    require(costs.shape == prediction.shape and torch.all(costs >= 0)
            and torch.all(costs[known] > 0), 'measured candidates require positive actual byte costs')
    require(type(gain_scale) in (int, float) and math.isfinite(gain_scale) and gain_scale > 0,
            'gain_scale must be positive train-only final-LPIPS scale')
    require(type(ranking_weight) in (int, float) and math.isfinite(ranking_weight) and ranking_weight >= 0,
            'invalid sender ranking coefficient')
    truth = torch.where(known, value, torch.zeros_like(value))
    scale = float(gain_scale)
    losses = F.smooth_l1_loss(prediction/scale, truth/scale, reduction='none')
    mass = weight.sum(1)
    supervised = mass > 0
    zero = prediction.sum()*0.
    huber = ((losses*weight).sum(1)[supervised]/mass[supervised]).mean() if supervised.any() else zero
    # A per-state common reference byte scale avoids tiny gradient magnitudes
    # while preserving the exact ordering of final gain / actual bytes.
    reference = (costs.to(prediction.dtype)*known).sum(1, keepdim=True)/known.sum(1, keepdim=True).clamp_min(1)
    normalized_cost = costs.to(prediction.dtype).clamp_min(1)/reference.clamp_min(1)
    utility_true = truth/scale/normalized_cost
    utility_prediction = prediction/scale/normalized_cost
    delta_true = utility_true[:, :, None]-utility_true[:, None, :]
    delta_prediction = utility_prediction[:, :, None]-utility_prediction[:, None, :]
    valid_pairs = known[:, :, None] & known[:, None, :] & (delta_true != 0)
    valid_pairs &= torch.triu(torch.ones((REGIONS, REGIONS), dtype=torch.bool,
                                         device=prediction.device), diagonal=1)[None]
    pair_weight = torch.minimum(weight[:, :, None], weight[:, None, :])*delta_true.abs().clamp(max=1)*valid_pairs
    pair_loss = F.softplus(-delta_true.sign()*delta_prediction)
    pair_mass = pair_weight.sum((1, 2))
    ranked = pair_mass > 0
    ranking = ((pair_loss*pair_weight).sum((1, 2))[ranked]/pair_mass[ranked]).mean() if ranked.any() else zero
    return huber+float(ranking_weight)*ranking, dict(huber=huber, ranking=ranking)


def rank_candidates(prediction, packet_bytes, coverage):
    """Positive remaining E candidates ranked by gain/actual-byte, stable ties.

This only ranks the current state. A conditional planner must update actual Y
and coverage and call R_s again after appending a bundle. Do not mislabel one
static rank as a joint optimum or as measured full-video quality.
"""
    gains = np.asarray(prediction)
    costs, received = np.asarray(packet_bytes), np.asarray(coverage)
    require(gains.shape == costs.shape == received.shape == (REGIONS,)
            and gains.dtype.kind in 'iuf' and np.isfinite(gains).all()
            and costs.dtype.kind in 'iu' and np.all(costs >= 0)
            and received.dtype.kind in 'biuf' and np.isfinite(received).all()
            and np.all((received >= 0) & (received <= 1))
            and np.all(costs[received < 1] > 0), 'invalid sender planning candidates')
    usable = [i for i in range(REGIONS) if received[i] < 1 and gains[i] > 0]
    return sorted(usable, key=lambda i: (-float(gains[i])/int(costs[i]), i))


def prefix_under_budget(order, packet_bytes, budget):
    """Take a literal complete-bundle prefix; never skip an unaffordable bundle."""
    require(type(order) in (list, tuple) and len(order) == len(set(order))
            and all(type(i) is int and 0 <= i < REGIONS for i in order), 'invalid ordered candidate prefix')
    costs = np.asarray(packet_bytes)
    require(costs.shape == (REGIONS,) and costs.dtype.kind in 'iu'
            and np.all(costs >= 0) and all(costs[i] > 0 for i in order), 'invalid prefix byte costs')
    require(type(budget) is int and budget >= 0, 'invalid E byte budget')
    chosen, used = [], 0
    for index in order:
        cost = int(costs[index])
        if used+cost > budget:
            break
        chosen.append(index)
        used += cost
    return dict(indices=chosen, packet_bytes=used, unused_bytes=budget-used,
                sender_mask_transmitted=False, scope='given-order literal complete-bundle prefix')


def code_identity():
    return {'routervc_sender_router.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def export_payload(model, binding, step, selection_metric):
    require(type(model) is SenderUtilityRouter and type(binding) is dict and type(selection_metric) is dict,
            'invalid sender export metadata')
    require(type(step) is int and step >= 0, 'invalid sender update count')
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'nonfinite sender parameters')
    return dict(format=FORMAT, architecture=asdict(model.config),
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        binding=binding, step=step, selection=selection_metric, metadata=model.metadata(), code=code_identity())


def load_model(path, *, expected_sha256=None):
    """Load independently authenticated R_s; never load R_g or teacher weights."""
    require(expected_sha256 is None or _valid_hash(expected_sha256), 'invalid expected sender hash')
    wire = Path(path).read_bytes()
    digest = hashlib.sha256(wire).hexdigest()
    require(expected_sha256 is None or digest == expected_sha256, 'sender checkpoint hash mismatch')
    try:
        payload = torch.load(io.BytesIO(wire), weights_only=True, map_location='cpu')
    except Exception as error:
        raise ValueError('invalid sender checkpoint') from error
    require(type(payload) is dict and set(payload) == PAYLOAD_KEYS and payload['format'] == FORMAT,
            'not an independent sender checkpoint')
    architecture = payload['architecture']
    require(type(architecture) is dict and set(architecture) == {f.name for f in fields(Config)},
            'invalid sender architecture')
    config = Config(**architecture)
    require(payload['code'] == code_identity(), 'sender code identity changed')
    require(type(payload['binding']) is dict and type(payload['selection']) is dict
            and type(payload['step']) is int and payload['step'] >= 0, 'invalid sender provenance')
    with torch.random.fork_rng(devices=[]):
        model = SenderUtilityRouter(config)
    require(payload['metadata'] == model.metadata(), 'sender metadata differs')
    expected, state = model.state_dict(), payload['state_dict']
    require(isinstance(state, dict) and set(state) == set(expected), 'invalid sender parameter keys')
    for key, template in expected.items():
        value = state[key]
        require(torch.is_tensor(value) and value.device.type == 'cpu' and value.layout == torch.strided
                and value.dtype == template.dtype and value.shape == template.shape
                and torch.isfinite(value).all(), f'invalid sender parameter: {key}')
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    model._sender_model_sha256 = digest
    model._sender_architecture = architecture.copy()
    torch.set_num_threads(CPU_THREADS)
    return model, payload


@torch.no_grad()
def predict(model, source, base, received, candidate_enhanced, coverage, packet_bytes, max_g):
    require(type(model) is SenderUtilityRouter
            and getattr(model, '_sender_architecture', None) == asdict(model.config)
            and _valid_hash(getattr(model, '_sender_model_sha256', None)),
            'sender inference requires an authenticated independent checkpoint')
    require(all(v.device.type == 'cpu' and v.dtype == torch.float32 and torch.isfinite(v).all()
                for v in (*model.parameters(), *model.buffers())), 'sender inference uses finite CPU float32')
    torch.set_num_threads(CPU_THREADS)
    model.eval()
    result = model(build_inputs(source, base, received, candidate_enhanced,
                                coverage, packet_bytes, max_g, model.config))
    require(result.shape == (1, REGIONS) and torch.isfinite(result).all(), 'invalid sender final-gain predictions')
    return result.detach()
