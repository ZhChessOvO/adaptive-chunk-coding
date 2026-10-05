"""Source-free Rg policy: choose G only, never allocate E or load sender Rs.

The three outputs are conditional LPIPS/PSNR/temporal G gains on actual mixed Y.
LPIPS chooses regions with the existing optional boundary penalty; the other
two predictions remain diagnostics, not untrained protection indicators.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from demo import routervc_receiver_router as receiver
from demo.routervc_visual_router import grid_rois, sampled_frame_indices
from demo.routervc_visual_policy import coverage_from_packets
from demo.routervc_policy import select_generate

PROFILE = 'routervc_receiver_G_only_policy_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def policy_identity():
    folder = Path(__file__).resolve().parent
    files = ('routervc_receiver_policy.py', 'routervc_visual_policy.py', 'routervc_policy.py')
    binding = dict(profile=PROFILE, receiver=receiver.policy_identity(),
        code={name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files},
        selection_metric='conditional_G_lpips', sender_model_required=False)
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def route(base, enhanced, inner, config, router, *, expected_policy=None):
    """Choose local G from B/Y and packet-derived coverage; source has no API."""
    expected_policy = policy_identity() if expected_policy is None else expected_policy
    _require(type(config) is dict and _valid_hash(config.get('receiver_router'))
             and _valid_hash(expected_policy) and config.get('policy') == expected_policy,
             'receiver policy identity mismatch')
    _require(type(config.get('max_g')) is int and 0 <= config['max_g'] <= 16,
             'invalid G call budget')
    boundary = config.get('boundary_lambda')
    _require(type(boundary) in (int, float) and np.isfinite(boundary) and 0 <= boundary <= 1,
             'invalid G boundary penalty')
    shape = (inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3)
    _require(np.shape(base) == shape and np.shape(enhanced) == shape,
             'received pixels differ from stream geometry')
    if isinstance(router, (str, Path)):
        model, _ = receiver.load_model(router, expected_sha256=config['receiver_router'])
    else:
        model = router
    _require(getattr(model, '_receiver_model_sha256', None) == config['receiver_router'],
             'receiver Router model hash mismatch')
    rois = grid_rois(shape[1], shape[2])
    coverage = coverage_from_packets(inner, rois)
    gains = receiver.predict(model, base, enhanced, coverage)[0].detach().cpu().numpy()
    _require(gains.shape == (16, 3) and np.isfinite(gains).all(), 'invalid receiver predictions')
    result = select_generate(gains[:, 0], config['max_g'], boundary)
    result.update(coverage=coverage.tolist(), predictions=gains.tolist(), rois=rois,
        states=[('EG' if coverage[i] > 0 else 'G') if i in result['indices']
                else ('E' if coverage[i] > 0 else 'B') for i in range(16)],
        sampled_frame_indices=sampled_frame_indices(len(base), model.config.temporal_frames),
        receiver_router_sha256=model._receiver_model_sha256,
        input_halo=model._receiver_input_halo,
        input_scope='decoded B, actual received mixed Y and received packet coverage only',
        gain_names=['conditional_G_lpips', 'conditional_G_psnr', 'conditional_G_temporal'],
        source_frames_used_by_router=False, sender_router_loaded=False,
        unreceived_candidates_used=False, generation_mask_transmitted=False,
        semantic_supervision=False, semantic_heads_used=False,
        explicit_E_mask_bytes=0, explicit_G_map_bytes=0, protection_mask_bytes=0,
        mask_removal_savings_bytes=0)
    return result
