"""Receiver-only global/local visual Router and offline teacher adapters.

One shared small CNN encodes B/Y pairs at two scales; one fused body predicts
direct E and conditional G(Y) gains. Global/local are inputs, not two experts.
No pretrained downloads, codec changes, protection metadata, or training runner.
An isolated no-E view supervises G|B; an isolated E view supervises G|E. Their
labels are not an additive oracle for arbitrary mixed-neighbor reconstructions.

Content/importance heads are reserved for REAL offline labels. Pure perceptual
labels leave every semantic supervision weight at zero. Untrained semantic
outputs must not be used by a deployment policy or called hallucination safety.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'routervc_visual_conditional_v1'
STATES = ('B', 'E', 'G', 'EG')
METRICS = (('lpips_alex', -1.), ('psnr_db', 1.), ('temporal_delta_mae', -1.))
GAIN_NAMES = ('direct_E_lpips', 'conditional_G_lpips', 'direct_E_psnr',
              'conditional_G_psnr', 'direct_E_temporal', 'conditional_G_temporal')
CATEGORIES = ('text_digits', 'face', 'key_structure')
INPUT_KEYS = ('global_pairs', 'local_pairs', 'geometry', 'coverage')
TARGET_KEYS = ('gains', 'content_gain', 'generation_harm', 'importance')


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class VisualRouterConfig:
    use_global: bool = True
    channels: int = 32
    hidden: int = 64
    temporal_frames: int = 3
    local_size: int = 64
    global_size: int = 96

    def __post_init__(self):
        require(type(self.use_global) is bool, 'use_global must be boolean')
        require(all(type(v) is int and v > 0 for v in
                    (self.channels, self.hidden, self.temporal_frames, self.local_size, self.global_size)),
                'model dimensions must be positive integers')
        require(self.local_size >= 16 and self.global_size >= 16, 'image sizes must be at least 16')


def grid_rois(height, width):
    """Same 8-aligned 4x4 topology as RTVC, without importing its pinned models."""
    require(all(type(v) is int and v >= 64 and v % 8 == 0 for v in (height, width)),
            'coded geometry must be >=64 and divisible by eight')
    ys = [8 * (i * (height // 8) // 4) for i in range(5)]
    xs = [8 * (i * (width // 8) // 4) for i in range(5)]
    return [[xs[x], ys[y], xs[x + 1] - xs[x], ys[y + 1] - ys[y]]
            for y in range(4) for x in range(4)]


def validated_pixels(base, received, rois):
    base, received = np.asarray(base), np.asarray(received)
    require(base.dtype == np.uint8 and received.dtype == np.uint8,
            'B/Y must be entropy-decoded uint8 RGB, not source features')
    require(base.ndim == 4 and base.shape[-1] == 3 and base.shape[0] >= 2
            and received.shape == base.shape, 'expected paired [T>=2,H,W,3] B/Y')
    expected = grid_rois(base.shape[1], base.shape[2])
    rois = expected if rois is None else rois
    require(np.array_equal(np.asarray(rois), np.asarray(expected)),
            'ROIs must match the dynamic receiver raster 4x4 grid')
    return base, received, expected


def _letterbox(pairs, size):
    """CPU RGB-only preprocessing; preserve aspect rather than stretch a face."""
    values = torch.from_numpy(np.ascontiguousarray(pairs)).permute(0, 3, 1, 2).float()
    height, width = values.shape[-2:]
    ratio = size / max(height, width)
    h, w = max(1, round(height * ratio)), max(1, round(width * ratio))
    if (height, width) != (h, w):
        values = F.interpolate(values, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
    left, top = (size - w) // 2, (size - h) // 2
    values = F.pad(values, (left, size - w - left, top, size - h - top), mode='replicate')
    return values / 127.5 - 1.


def sampled_frame_indices(frame_count, temporal_frames=3):
    require(type(frame_count) is int and frame_count >= 2 and type(temporal_frames) is int
            and temporal_frames > 0, 'invalid temporal sample dimensions')
    return np.rint(np.linspace(0, frame_count - 1, temporal_frames)).astype(int).tolist()


def build_receiver_inputs(base, received, coverage, rois=None, *, config=None, region_indices=None):
    """Return batch-one tensors using only B, actual Y and received-E coverage.

    No source, labels, packet benefit estimates or semantic masks are accepted.
    The whole supplied field of view is retained; a historical UVG crop stays a
    crop and is NOT called the original full UVG frame by this function.
    """
    config = config or VisualRouterConfig()
    base, received, rois = validated_pixels(base, received, rois)
    indices = list(range(16)) if region_indices is None else list(region_indices)
    require(bool(indices) and len(set(indices)) == len(indices)
            and all(type(i) is int and 0 <= i < 16 for i in indices), 'invalid selected region indices')
    coverage = np.asarray(coverage)
    require(coverage.shape == (16,) and coverage.dtype.kind in 'biuf'
            and np.isfinite(coverage).all() and np.all((coverage >= 0) & (coverage <= 1)),
            'coverage must have 16 finite fractions in [0,1]')
    ids = sampled_frame_indices(len(base), config.temporal_frames)
    pairs = np.concatenate((base[ids], received[ids]), axis=-1)
    global_pairs = _letterbox(pairs, config.global_size)
    selected_rois = [rois[i] for i in indices]
    local_pairs = torch.stack([_letterbox(pairs[:, y:y+h, x:x+w], config.local_size)
                               for x, y, w, h in selected_rois])
    height, width = base.shape[1:3]
    geometry = [[(x+w/2)/width, (y+h/2)/height, w/width, h/height,
                 w/512., h/512.] for x, y, w, h in selected_rois]
    return dict(global_pairs=global_pairs.unsqueeze(0), local_pairs=local_pairs.unsqueeze(0),
                geometry=torch.tensor(geometry, dtype=torch.float32).unsqueeze(0),
                coverage=torch.tensor(coverage[indices], dtype=torch.float32).reshape(1, len(indices), 1))


class VisualUtilityRouter(nn.Module):
    """A shared encoder, one fused body, and jointly supervised output channels."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or VisualRouterConfig()
        c, hidden = self.config.channels, self.config.hidden
        self.encoder = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, c, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        # Mean appearance + first/last feature difference at each spatial scale.
        self.body = nn.Sequential(nn.Linear(4*c + 7, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        # 6 ordinary gains + 2x3 content gains + 3 G-harms + 3 importance logits.
        self.output = nn.Linear(hidden, 18)

    def _encode_time(self, frames):
        shape = frames.shape
        values = self.encoder(frames.reshape(-1, *shape[-3:])).flatten(1)
        values = values.reshape(*shape[:-3], self.config.channels)
        return torch.cat((values.mean(-2), (values[..., -1, :] - values[..., 0, :]).abs()), dim=-1)

    def forward(self, inputs):
        require(set(inputs) == set(INPUT_KEYS), 'unexpected/missing receiver input fields')
        local, global_ = inputs['local_pairs'], inputs['global_pairs']
        coverage, geometry = inputs['coverage'], inputs['geometry']
        k = self.config.temporal_frames
        require(local.ndim == 6 and 1 <= local.shape[1] <= 16 and local.shape[2:4] == (k, 6)
                and local.shape[-2:] == (self.config.local_size,)*2, 'invalid local image tensor')
        batch, regions = local.shape[:2]
        require(global_.shape == (batch, k, 6, self.config.global_size, self.config.global_size)
                and coverage.shape == (batch, regions, 1) and geometry.shape == (batch, regions, 6),
                'invalid receiver image/geometry/coverage shapes')
        require(all(torch.is_floating_point(v) and torch.isfinite(v).all() for v in inputs.values()),
                'receiver input tensors must be finite floating point')
        require(torch.all((coverage >= 0) & (coverage <= 1)), 'invalid E coverage')
        local_embedding = self._encode_time(local)
        # Equal architecture/parameter count; no global input dependence in ablation.
        global_embedding = (self._encode_time(global_) if self.config.use_global else
                            local_embedding.new_zeros((batch, self.config.channels*2)))
        fused = torch.cat((local_embedding, global_embedding[:, None].expand(-1, regions, -1),
                           geometry, coverage), dim=-1)
        raw = self.output(self.body(fused))
        gate = torch.cat((coverage, torch.ones_like(coverage)), dim=-1)
        gains = raw[..., :6] * gate.repeat(1, 1, 3)
        content_gain = raw[..., 6:12].reshape(batch, regions, 2, 3) * gate[..., None]
        return dict(gains=gains, content_gain=content_gain,
                    generation_harm=F.softplus(raw[..., 12:15]), importance_logits=raw[..., 15:18])

    def metadata(self):
        return dict(format=FORMAT, architecture=asdict(self.config), shared_encoder=True,
                    parameters=sum(p.numel() for p in self.parameters()), experts=0,
                    input_source='received_B_Y_E_coverage_only', pretrained_weights=False,
                    transmitted_E_G_protection_mask=False, gain_names=list(GAIN_NAMES),
                    categories=list(CATEGORIES), semantic_supervision='not inferred from architecture')


def _empty_target(shape):
    return dict(value=torch.zeros(shape, dtype=torch.float32), weight=torch.zeros(shape, dtype=torch.float32))


def build_view_targets(qualities, view_index, *, content_teacher=None):
    """Adapt measured per-region four-state scores to one source-free view.

    view 0 = Y=B, all cells supervised. view i+1 = isolated E at i, only i is
    supervised. Missing metrics/content labels have zero weights, not safe labels.
    content_teacher optionally comes from routervc_content_objective and must
    retain every value/weight mask. No semantic detector or annotations are made.
    """
    require(len(qualities) == 16 and type(view_index) is int and 0 <= view_index <= 16,
            'expected sixteen regions and view index in 0..16')
    result = dict(gains=_empty_target((1, 16, 6)), content_gain=_empty_target((1, 16, 2, 3)),
                  generation_harm=_empty_target((1, 16, 3)), importance=_empty_target((1, 16, 3)))
    selected = range(16) if view_index == 0 else [view_index - 1]
    before, after = ('B', 'G') if view_index == 0 else ('E', 'EG')
    for region in selected:
        quality = qualities[region]
        require(set(quality) == set(STATES), 'every region needs B/E/G/EG quality records')
        for metric_index, (metric, direction) in enumerate(METRICS):
            values = [quality[s].get(metric, np.nan) for s in STATES]
            require(not np.isinf(np.asarray(values, dtype=float)).any(), 'infinite measured metric')
            for gain_index, (parent, child) in enumerate((('B', before), (before, after))):
                a, b = quality[parent].get(metric, np.nan), quality[child].get(metric, np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    index = 2*metric_index + gain_index
                    result['gains']['value'][0, region, index] = direction * (b-a)
                    result['gains']['weight'][0, region, index] = 1.
        require(all(np.isfinite(quality[s].get('lpips_alex', np.nan))
                    and quality[s]['lpips_alex'] >= 0 for s in STATES), 'measured LPIPS is required')
    if content_teacher is not None:
        from demo.routervc_content_objective import SCHEMA
        require(content_teacher.get('schema') == SCHEMA
                and content_teacher.get('offline_supervision_only') is True,
                'content labels must come from the offline teacher contract')
        require(tuple(content_teacher.get('categories', ())) == CATEGORIES,
                'content category order differs')
        targets = content_teacher['targets']
        def copy_target(destination, source, destination_index, source_index):
            item = targets[source]
            value = np.asarray(item['value'])[source_index]
            weight = np.asarray(item['weight'])[source_index]
            require(np.shape(value) == np.shape(weight) and np.isfinite(weight).all()
                    and np.all(weight >= 0) and np.isfinite(value[weight > 0]).all(),
                    'invalid masked content target')
            # Unknowns stay weight zero. Their finite filler is NOT a negative label.
            result[destination]['value'][destination_index] = torch.as_tensor(np.where(weight > 0, value, 0.), dtype=torch.float32)
            result[destination]['weight'][destination_index] = torch.as_tensor(weight, dtype=torch.float32)
        for region in selected:
            if view_index:
                copy_target('content_gain', 'category_content_gain', (0, region, 0), (region, 0))
            copy_target('content_gain', 'category_content_gain', (0, region, 1), (region, 1 if view_index == 0 else 2))
            copy_target('generation_harm', 'category_generation_harm', (0, region), (region, 0 if view_index == 0 else 1))
            if 'importance' in targets:
                copy_target('importance', 'importance', (0, region), region)
    return result


def build_training_view(base, enhanced, qualities, view_index, rois=None, *, config=None,
                        content_teacher=None, target_region=None):
    """Lazy one-view adapter; callers should not materialize all 17 RGB videos."""
    base, enhanced, rois = validated_pixels(base, enhanced, rois)
    require(type(view_index) is int and 0 <= view_index <= 16, 'invalid isolated view index')
    config = config or VisualRouterConfig()
    if target_region is not None:
        require(type(target_region) is int and 0 <= target_region < 16
                and (view_index == 0 or view_index == target_region + 1),
                'target region must be the isolated enhanced cell or a no-E cell')
    coverage = np.zeros(16, dtype=np.float32)
    received = base
    if view_index:
        received = base.copy()
        index = view_index - 1
        x, y, w, h = rois[index]
        received[:, y:y+h, x:x+w] = enhanced[:, y:y+h, x:x+w]
        coverage[index] = 1.
    indices = list(range(16)) if target_region is None else [target_region]
    targets = build_view_targets(qualities, view_index, content_teacher=content_teacher)
    targets = {name: {field: value[:, indices] for field, value in item.items()}
               for name, item in targets.items()}
    return dict(inputs=build_receiver_inputs(base, received, coverage, rois, config=config,
                                             region_indices=indices), targets=targets,
                receiver_metadata=dict(frame_indices=sampled_frame_indices(len(base), config.temporal_frames),
                    region_indices=indices, rois=[rois[i] for i in indices],
                    decoded_shape=list(base.shape), global_letterbox_size=config.global_size,
                    local_letterbox_size=config.local_size,
                    temporal_fusion='encode each sampled frame; feature mean and first/last difference',
                    global_input='whole supplied received field of view; original versus crop belongs in data manifest'),
                scope='isolated-cell teacher, not exact mixed-neighbor oracle')


def build_training_case(base, enhanced, qualities, region, with_e, rois=None, *, config=None, content_teacher=None):
    """One target cell, only one local crop; full global Y is the true E view.

    This is the efficient runner entrypoint. Enumerate (region=0..15, with_e
    false/true), not 17 full 16-cell CNN views with almost all labels masked.
    """
    require(type(with_e) is bool, 'with_e must be boolean')
    require(type(region) is int and 0 <= region < 16, 'invalid target region')
    return build_training_view(base, enhanced, qualities, region + 1 if with_e else 0,
                               rois, config=config, content_teacher=content_teacher, target_region=region)


def load_measured_sample(label_path, received_path):
    """Read old/new measurement records, but not source RGB/source statistics.

    Adapter expects regions[{roi,quality:{B,E,G,EG}}] and an NPZ containing
    decoded base plus enhanced (old teacher) or all_E (public candidate bank).
    The trainer must verify manifest SHA256s before calling this small adapter.
    """
    labels = json.loads(Path(label_path).read_text())
    regions = labels['regions']
    require(len(regions) == 16, 'teacher must have 16 raster-ordered regions')
    with np.load(received_path, allow_pickle=False) as cache:
        base = cache['base'].copy()
        enhanced = cache['enhanced' if 'enhanced' in cache else 'all_E'].copy()
    rois = [row['roi'] for row in regions]
    validated_pixels(base, enhanced, rois)
    return dict(base=base, enhanced=enhanced, qualities=[row['quality'] for row in regions], rois=rois)


def batch_examples(examples):
    require(bool(examples), 'empty training batch')
    return dict(inputs={k: torch.cat([r['inputs'][k] for r in examples]) for k in INPUT_KEYS},
                targets={k: {field: torch.cat([r['targets'][k][field] for r in examples])
                             for field in ('value', 'weight')} for k in TARGET_KEYS})


def supervision_metadata(targets):
    observed = {k: int((targets[k]['weight'] > 0).sum()) for k in TARGET_KEYS}
    content = observed['content_gain'] + observed['generation_harm'] > 0
    return dict(observed_target_elements=observed,
                semantic_supervision=bool(content or observed['importance']),
                content_fidelity_supervision=bool(content),
                importance_supervision=bool(observed['importance']),
                untrained_semantic_outputs_must_not_route=not content,
                guarantees_semantic_correctness=False,
                missing_labels='unknown with zero loss weight; never assumed safe')


def masked_training_loss(outputs, targets, *, gain_scale=None, content_weight=1.,
                         harm_weight=1., importance_weight=.1):
    """Masked Huber gains/content/harm and optional soft importance BCE.

    gain_scale is six positive TRAIN-set scales; defaults to one for smoke.
    Pixel-derived PSNR/temporal outputs remain auxiliary at weight 0.1. Runtime
    allocations must explicitly choose which measured objectives they deploy.
    """
    prediction = outputs['gains']
    scales = prediction.new_ones(6) if gain_scale is None else torch.as_tensor(gain_scale, device=prediction.device, dtype=prediction.dtype)
    require(scales.shape == (6,) and torch.isfinite(scales).all() and torch.all(scales > 0), 'invalid gain scales')
    coefficients = dict(gains=1., content_gain=content_weight, generation_harm=harm_weight, importance=importance_weight)
    require(all(isinstance(v, (float, int)) and not isinstance(v, bool)
                and np.isfinite(v) and v >= 0 for v in coefficients.values()), 'invalid loss coefficients')
    metrics, terms = prediction.new_tensor([1., 1., .1, .1, .1, .1]), {}
    for name in TARGET_KEYS:
        pred = outputs['importance_logits' if name == 'importance' else name]
        target = targets[name]['value'].to(pred.device)
        weight = targets[name]['weight'].to(pred.device)
        require(target.shape == weight.shape == pred.shape and torch.isfinite(weight).all()
                and torch.all(weight >= 0), 'invalid supervision shape/weights')
        require(torch.isfinite(pred).all(), 'nonfinite model prediction')
        active = weight > 0
        require(torch.isfinite(target[active]).all(), 'unknown target has nonzero supervision weight')
        if not torch.any(active):
            terms[name] = pred.sum() * 0.
            continue
        if name == 'gains':
            pred, target, weight = pred/scales, target/scales, weight*metrics
        if name == 'importance':
            require(torch.all((target[active] >= 0) & (target[active] <= 1)), 'importance target outside [0,1]')
            losses = F.binary_cross_entropy_with_logits(pred[active], target[active], reduction='none')
        else:
            losses = F.smooth_l1_loss(pred[active], target[active], reduction='none')
        terms[name] = (losses * weight[active]).sum() / weight[active].sum()
    total = sum(coefficients[name]*value for name, value in terms.items())
    return total, terms
