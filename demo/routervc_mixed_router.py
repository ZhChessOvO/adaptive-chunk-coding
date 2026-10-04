"""Mixed-reconstruction Router inputs; old visual profiles remain untouched.

The optional local halo is exactly G's clipped 64px processing neighborhood,
letterboxed to the same 64px CNN input. This changes field of view, not capacity;
the context/detail tradeoff is part of the ablation. No semantic labels or masks
are sent. Saved models have a separate format and cannot enter old receivers.
"""
from dataclasses import asdict
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from demo import routervc_visual_router as visual

FORMAT = 'routervc_mixed_reconstruction_v1'
ARMS = ('isolated_core', 'mixed_core', 'mixed_halo')


def selection(sample_id, index):
    """Two fixed spatial configurations; independent of GT and model predictions.

    4/8 are region counts, NOT E-byte caps. Shared random ordering yields a true
    nested subset, while the 120 different orders cover varied neighborhoods.
    """
    visual.require(index in (0, 1), 'unknown mixture index')
    seed = int(hashlib.sha256(('mixed-v1/'+sample_id).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).permutation(16)[:(4, 8)[index]].tolist()


def processing_roi(roi, height, width, halo):
    visual.require(type(halo) is int and halo in (0, 64), 'only core/G-halo profiles supported')
    x, y, w, h = roi
    x0, y0 = max(0, x-halo), max(0, y-halo)
    return [x0, y0, min(width, x+w+halo)-x0, min(height, y+h+halo)-y0]


def build_inputs(base, received, coverage, *, halo=0, config=None):
    config = config or visual.VisualRouterConfig()
    inputs = visual.build_receiver_inputs(base, received, coverage, config=config)
    if halo:
        rois = visual.grid_rois(*base.shape[1:3])
        ids = visual.sampled_frame_indices(len(base), config.temporal_frames)
        pairs = np.concatenate((base[ids], received[ids]), -1)
        expanded = [processing_roi(r, *base.shape[1:3], halo) for r in rois]
        inputs['local_pairs'] = torch.stack([
            visual._letterbox(pairs[:, y:y+h, x:x+w], config.local_size)
            for x, y, w, h in expanded]).unsqueeze(0)
    # Geometry always describes the decision/writeback core, not the halo.
    return inputs


def mixed_targets(qualities, selected, generated_scores):
    visual.require(len(generated_scores) == len(qualities) == 16, 'sixteen measured cells required')
    result = visual.build_view_targets(qualities, 0)
    for i in range(16):
        before = qualities[i]['E' if i in selected else 'B']
        for j, (metric, direction) in enumerate(visual.METRICS):
            baseline = qualities[i]['B'].get(metric)
            parent, child = before.get(metric), generated_scores[i].get(metric)
            for k, a, b in ((2*j, baseline, parent), (2*j+1, parent, child)):
                known = a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
                result['gains']['value'][0, i, k] = direction*(b-a) if known else 0.
                result['gains']['weight'][0, i, k] = float(known)
        visual.require(result['gains']['weight'][0, i, :2].all(), 'measured LPIPS required')
    return result


def best_g_indices(gains, cap):
    values = np.asarray(gains, dtype=float)
    visual.require(values.shape == (16,) and np.isfinite(values).all(), 'invalid G utility')
    return [int(i) for i in np.argsort(-values, kind='stable') if values[i] > 0][:cap]


def regret(prediction, target, caps=(4, 8)):
    """Conditional local gain regret, NOT whole-video LPIPS or oracle E routing."""
    values = []
    for cap in caps:
        optimal, chosen = best_g_indices(target, cap), best_g_indices(prediction, cap)
        values.append(float((np.sum(np.asarray(target)[optimal])-
                             np.sum(np.asarray(target)[chosen]))/16))
    return float(np.mean(values))


def export_payload(model, arm, binding, epoch, selection_metric):
    visual.require(arm in ARMS, 'unknown arm')
    return dict(format=FORMAT, architecture=asdict(model.config), arm=arm,
        input_halo=64 if arm == 'mixed_halo' else 0,
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        binding=binding, epoch=epoch, selection=selection_metric,
        semantic_supervision=False, content_heads_usable=False,
        preprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def load_model(path):
    """Public CPU predictor for the new input contract; no teacher/GT loaded."""
    wire = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(wire), weights_only=True, map_location='cpu')
    visual.require(payload.get('format') == FORMAT and payload.get('arm') in ARMS,
                   'not a mixed-reconstruction checkpoint')
    visual.require(payload.get('semantic_supervision') is False and
                   payload.get('content_heads_usable') is False, 'semantic outputs are not trained')
    visual.require(payload['preprocessing_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'Router preprocessing differs from checkpoint')
    visual.require(payload['input_halo'] == (64 if payload['arm'] == 'mixed_halo' else 0),
                   'halo/arm mismatch')
    with torch.random.fork_rng(devices=[]):
        model = visual.VisualUtilityRouter(visual.VisualRouterConfig(**payload['architecture']))
    model.load_state_dict(payload['state_dict'], strict=True)
    visual.require(all(torch.isfinite(v).all() for v in model.parameters()), 'nonfinite weights')
    model.requires_grad_(False).eval()
    return model, payload


@torch.no_grad()
def predict(model, payload, base, received, coverage):
    return model(build_inputs(base, received, coverage, halo=payload['input_halo'],
                              config=model.config))['gains']
