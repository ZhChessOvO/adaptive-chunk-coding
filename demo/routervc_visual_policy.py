"""CPU shared policy for the visual Router perceptual baseline.

Only the six trained utility outputs participate. The reserved content heads
are not protection models. Sender utility is predicted on seventeen ACTUAL
pixel views: B, then B with one decoded E cell inserted. Every isolated view
rebuilds its global thumbnail; replacing a row of old statistical features is
not equivalent. The receiver predicts again on its actual mixed Y.

This module does not serialize masks or change any codec/header. The enclosing
new stream profile must charge its header and all packet addressing bytes.
Shared-source identity covers preprocessing and reused allocation algorithms.
"""
from dataclasses import asdict, fields
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from demo import routervc_visual_router as visual
from demo.routervc_policy import select_generate

CPU_THREADS = 4
REGIONS = 16
TRAIN_FORMAT = 'routervc_visual_perceptual_train_v1'
POLICY_FORMAT = 'routervc_visual_perceptual_policy_v1'
IDENTITY_FILES = ('routervc_visual_policy.py', 'routervc_visual_router.py',
                  'routervc_policy.py', 'routervc_encode.py',
                  'four_state_router_evaluate.py', 'scalable_format.py',
                  'compact_enhancement_format.py')
PAYLOAD_KEYS = {'format', 'architecture', 'state_dict', 'training_binding',
                'semantic_supervision', 'content_heads_usable', 'metadata'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _valid_hash(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in IDENTITY_FILES}


def policy_identity():
    """Aggregate policy/preprocessing digest for the enclosing stream profile."""
    binding = dict(format=POLICY_FORMAT, cpu_threads=CPU_THREADS,
                   gain_names=list(visual.GAIN_NAMES), code=code_hashes())
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_model(path, *, expected_sha256=None):
    """Authenticate and strictly load a visual_train model.pt on CPU only.

    The hash is computed over the same in-memory bytes passed to torch.load.
    No teacher/source data, optimizer, gain scaling or semantic predictor is
    loaded. The training gain_scale weights a loss; it is not an output scale.
    """
    _require(expected_sha256 is None or _valid_hash(expected_sha256), 'invalid expected Router hash')
    wire = Path(path).read_bytes()
    actual_hash = hashlib.sha256(wire).hexdigest()
    _require(expected_sha256 is None or actual_hash == expected_sha256, 'shared visual Router model hash mismatch')
    try:
        payload = torch.load(io.BytesIO(wire), weights_only=True, map_location='cpu')
    except Exception as error:
        raise ValueError('invalid visual Router checkpoint') from error
    _require(type(payload) is dict and set(payload) == PAYLOAD_KEYS, 'unsupported visual Router checkpoint fields')
    _require(payload['format'] == visual.FORMAT, 'unsupported visual Router format')
    _require(payload['semantic_supervision'] is False and payload['content_heads_usable'] is False,
             'this perceptual policy requires semantic_supervision=false and unusable content heads')
    architecture = payload['architecture']
    _require(type(architecture) is dict and set(architecture) == {f.name for f in fields(visual.VisualRouterConfig)},
             'invalid visual Router architecture fields')
    try:
        config = visual.VisualRouterConfig(**architecture)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid visual Router architecture') from error
    training = payload['training_binding']
    _require(type(training) is dict and training.get('format') == TRAIN_FORMAT
             and training.get('semantic_supervision') is False
             and training.get('model') == asdict(config), 'training configuration/architecture mismatch')
    code = training.get('code')
    _require(type(code) is dict and _valid_hash(code.get('routervc_visual_train.py'))
             and code.get('routervc_visual_router.py') == hashlib.sha256(
                 Path(visual.__file__).read_bytes()).hexdigest(),
             'checkpoint preprocessing code differs from shared visual Router')
    torch.set_num_threads(CPU_THREADS)
    # A CPU policy load must not perturb the separate generator's random state.
    rng_state = torch.random.get_rng_state()
    try:
        model = visual.VisualUtilityRouter(config)
    finally:
        torch.random.set_rng_state(rng_state)
    _require(payload['metadata'] == model.metadata(), 'checkpoint architecture metadata mismatch')
    state = payload['state_dict']
    expected = model.state_dict()
    _require(isinstance(state, dict) and set(state) == set(expected), 'invalid visual Router state_dict keys')
    for name, template in expected.items():
        value = state[name]
        _require(torch.is_tensor(value) and value.device.type == 'cpu'
                 and value.layout == torch.strided and value.dtype == template.dtype
                 and value.shape == template.shape and torch.isfinite(value).all(),
                 f'invalid/nonfinite visual Router parameter: {name}')
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    model._visual_policy_model_sha256 = actual_hash
    model._visual_policy_architecture = asdict(config)
    model._visual_policy_semantic_supervision = False
    return model


def _model(router):
    model = load_model(router) if isinstance(router, (str, Path)) else router
    _require(type(model) is visual.VisualUtilityRouter
             and _valid_hash(getattr(model, '_visual_policy_model_sha256', None))
             and getattr(model, '_visual_policy_semantic_supervision', None) is False,
             'use a checked visual-policy checkpoint, not an old/statistical or unbound model')
    _require(asdict(model.config) == getattr(model, '_visual_policy_architecture', None),
             'loaded visual Router configuration changed')
    _require(all(v.device.type == 'cpu' and v.dtype == torch.float32 and torch.isfinite(v).all()
                 for v in (*model.parameters(), *model.buffers())),
             'visual policy parameters must be finite CPU float32')
    model.eval()
    return model


@torch.no_grad()
def predict(model, base, received, coverage, rois=None):
    """Return unscaled [1,16,6] gains from actual B/Y/coverage only."""
    model = _model(model)
    if torch.get_num_threads() != CPU_THREADS:
        torch.set_num_threads(CPU_THREADS)
    inputs = visual.build_receiver_inputs(base, received, coverage, rois, config=model.config)
    outputs = model(inputs)
    _require(isinstance(outputs, dict) and 'gains' in outputs, 'missing trained visual gains')
    result = outputs['gains']
    _require(torch.is_tensor(result) and result.shape == (1, REGIONS, 6)
             and result.dtype == torch.float32 and result.device.type == 'cpu'
             and torch.isfinite(result).all(), 'invalid visual Router predictions')
    # Do not even inspect values from untrained importance/content/harm heads.
    return result.detach()


@torch.no_grad()
def predict_utility(bank, base, all_e, router):
    """Isolated B/E/G/EG table; NOT an oracle for a mixed-neighbor video.

    For each E cell, the complete global B/Y image pair is rebuilt from actual
    decoded pixels. All sixteen local outputs are returned for audit, although
    allocation uses that view's target-cell direct E and conditional G only.
    """
    from demo.routervc_encode import _validate_candidates, compose_candidates

    info = _validate_candidates(bank, base, all_e)
    model = _model(router)
    coverage = np.zeros(REGIONS, np.float32)
    predictions = [predict(model, base, base, coverage, info['rois'])[0]]
    for index in range(REGIONS):
        mixed = compose_candidates(base, all_e, [index], info['rois'])
        coverage = np.zeros(REGIONS, np.float32)
        coverage[index] = 1.
        predictions.append(predict(model, base, mixed, coverage, info['rois'])[0])
    result = torch.stack(predictions)
    indices = torch.arange(REGIONS)
    direct = result[indices + 1, indices, 0]
    utility = torch.stack((torch.zeros_like(direct), direct, result[0, :, 1],
                           direct + result[indices + 1, indices, 1]), dim=1)
    return utility.numpy(), result.numpy()


def allocate(utility, e_bytes, extra_e_budget, max_g_calls, *, mode='prefix'):
    """Reuse fixed legacy solvers, with only measured E packet bytes budgeted.

    G states here are provisional sender predictions and are not transmitted.
    Independent plans need not be nested; prefix mode uses one budget-independent
    rank and stops at the first bundle that does not fit.
    """
    from demo.four_state_router_evaluate import frontier, solve
    from demo.routervc_encode import _table_score, prefix_order

    utility, e_bytes = np.asarray(utility), np.asarray(e_bytes)
    _require(utility.shape == (REGIONS, 4) and utility.dtype.kind in 'iuf'
             and np.isfinite(utility).all(), 'expected finite [16,4] four-state utility')
    _require(e_bytes.shape == (REGIONS,) and e_bytes.dtype.kind in 'iu'
             and np.all(e_bytes > 0), 'E packet costs must be sixteen positive integers')
    _require(type(extra_e_budget) is int and extra_e_budget >= 0
             and type(max_g_calls) is int and 0 <= max_g_calls <= REGIONS,
             'invalid E-byte/G-call budget')
    _require(mode in ('independent', 'prefix'), 'invalid allocation mode')
    rank, details = [], []
    if mode == 'independent':
        plan = solve(frontier(utility, e_bytes, 0, 0), extra_e_budget, max_g_calls)
        states, score = plan['states'], plan['predicted_utility']
        indices = [i for i, state in enumerate(states) if state in (1, 3)]
    else:
        rank, details = prefix_order(utility, e_bytes, max_g_calls)
        indices, used = [], 0
        for i in rank:
            if used + int(e_bytes[i]) > extra_e_budget:
                break
            indices.append(i)
            used += int(e_bytes[i])
        score, states = _table_score(utility, indices, max_g_calls)
    used = sum(int(e_bytes[i]) for i in indices)
    _require(used <= extra_e_budget, 'E allocation exceeded packet-byte budget')
    return dict(mode=mode, selected_indices=indices, e_packet_bytes=used,
                budget_e_packet_bytes=extra_e_budget, unused_e_budget_bytes=extra_e_budget-used,
                max_g_calls=max_g_calls, provisional_states=states, predicted_utility=score,
                prefix_order=rank, prefix_ranking=details,
                generation_mask_transmitted=False, semantic_heads_used=False,
                allocation_scope='isolated-table exact independent allocation' if mode == 'independent'
                else 'greedy marginal gain/byte; literal complete-region prefix',
                mixed_video_oracle=False,
                budget_scope='E packets only; all base/shared/profile/addressing bytes remain charged')


def coverage_from_packets(inner, rois):
    """Coverage comes only from packet coordinates/time, not a new mask field."""
    count = inner.meta['frame_count']
    _require(type(count) is int and count >= 2, 'invalid packet frame count')
    expected = visual.grid_rois(inner.meta['height'], inner.meta['width'])
    _require(rois == expected, 'packet coverage requires the receiver raster grid')
    covered = np.zeros((count, REGIONS), bool)
    for packet in inner.packets:
        meta = packet.meta
        _require(meta['roi'] in rois, 'E packet is outside the visual Router grid')
        index, start, frames = rois.index(meta['roi']), meta['start'], meta['count']
        _require(type(start) is int and type(frames) is int and start >= 0 and frames >= 1
                 and start + frames <= count and not covered[start:start+frames, index].any(),
                 'invalid or duplicate E packet coverage')
        covered[start:start+frames, index] = True
    return covered.mean(0).astype(np.float32)


def route(base, enhanced, inner, config, router, *, expected_policy=None):
    """Derive G again on actual mixed Y; no sender table or semantic head used.

    A new enclosing wire profile passes its aggregate ``expected_policy`` hash;
    otherwise this standalone helper expects ``policy_identity()``. This avoids
    a circular dependency on that format/decoder's own source-identity function.
    """
    expected_policy = policy_identity() if expected_policy is None else expected_policy
    _require(isinstance(config, dict) and _valid_hash(config.get('router'))
             and _valid_hash(expected_policy) and config.get('policy') == expected_policy,
             'shared visual policy identity mismatch')
    _require(type(config.get('max_g')) is int and 0 <= config['max_g'] <= REGIONS,
             'invalid G call budget')
    boundary = config.get('boundary_lambda')
    _require(type(boundary) in (int, float) and np.isfinite(boundary) and boundary >= 0,
             'invalid boundary penalty')
    expected_shape = (inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3)
    _require(np.shape(base) == expected_shape and np.shape(enhanced) == expected_shape,
             'received pixels differ from stream geometry')
    model = load_model(router, expected_sha256=config['router']) if isinstance(router, (str, Path)) else _model(router)
    _require(model._visual_policy_model_sha256 == config['router'], 'shared visual Router model hash mismatch')
    rois = visual.grid_rois(base.shape[1], base.shape[2])
    coverage = coverage_from_packets(inner, rois)
    gains = predict(model, base, enhanced, coverage, rois)[0].numpy()
    selection = select_generate(gains[:, 1], config['max_g'], boundary)
    selection.update(coverage=coverage.tolist(), predictions=gains.tolist(), rois=rois,
        states=[('EG' if coverage[i] > 0 else 'G') if i in selection['indices']
                else ('E' if coverage[i] > 0 else 'B') for i in range(REGIONS)],
        sampled_frame_indices=visual.sampled_frame_indices(len(base), model.config.temporal_frames),
        router_sha256=model._visual_policy_model_sha256,
        preprocessing_sha256=hashlib.sha256(Path(visual.__file__).read_bytes()).hexdigest(),
        input_scope='decoded B, actual mixed Y and received E packet coverage',
        semantic_supervision=False, semantic_heads_used=False,
        source_frames_used_by_router=False, generation_mask_transmitted=False,
        explicit_G_map_bytes=0, mask_removal_savings_bytes=0,
        calibration_scope='perceptual baseline; partial coverage/other geometries mechanically supported, not guaranteed calibrated')
    return selection
