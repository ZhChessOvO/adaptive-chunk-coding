"""Conditional source-aware sender over authenticated, unchanged qE=2 packets.

This is a CACHE-REUSE allocation path, not a source-to-bitstream timing claim.
The caller authenticates its source file; independent E receive records bind
the complete candidate reconstruction. R_s predicts signed FINAL LPIPS gains
after fixed R_g/G. It never runs diffusion to make a deployment decision.

Build one budget-independent order, updating actual received Y after every
selected region, then emit literal prefixes. Only the RVRC receiver profile,
base stream and original entropy packets go on the wire. Source pixels, full
candidate pixels, R_s weights/predictions and masks remain sender-local.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from demo import routervc_receiver_format as fmt
from demo import routervc_sender_router as sender
from demo.routervc_encode import compose_candidates, validate_frames
from demo.routervc_light_packets import bank_info, subset_bank
from demo.scalable_codec import file_hash
from demo.scalable_format import frame_hash

FORMAT = 'routervc_conditional_sender_prefix_v1'
ROLE = 'source-aware sender final-marginal prediction'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_hash(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def _valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _record(path, expected_sha256):
    _require(_valid_hash(expected_sha256), 'explicit record SHA256 required')
    path = Path(path).resolve()
    raw = path.read_bytes()
    _require(_sha(raw) == expected_sha256, 'authenticated record changed')
    value = json.loads(raw)
    _require(type(value) is dict, 'record must be a JSON object')
    return path, value


@dataclass(frozen=True)
class CandidateCache:
    bank: bytes
    source: np.ndarray
    base: np.ndarray
    all_e: np.ndarray
    binding: dict
    authentication_seconds: float


@dataclass(frozen=True)
class SenderContext:
    model: object
    payload: dict
    config: dict
    binding: dict


def authenticate_candidates(bank, source, base, all_e, *, receive_record,
                            expected_receive_sha256, expected_bank_sha256,
                            expected_source_rgb_sha256):
    """Authenticate cached independently decoded candidates, never regenerate E.

expected_source_rgb_sha256 comes from the caller's authenticated source file
or original encoding manifest, NOT the receiver record (receivers do not read
source). receive_record is the existing source-free received/<sample>/E.json.
We verify its actual NPZ artifact as well as the supplied RGB arrays. This API
does not claim to independently entropy-decode on each allocation call.
"""
    began = time.monotonic()
    _require(type(bank) is bytes and _valid_hash(expected_bank_sha256)
             and _sha(bank) == expected_bank_sha256, 'candidate bank changed')
    _require(_valid_hash(expected_source_rgb_sha256), 'explicit source RGB hash required')
    shape = validate_frames(source)
    _require(shape[0] == 17 and validate_frames(base) == shape and validate_frames(all_e) == shape,
             'sender requires matching complete 17-frame source/base/E windows')
    info = bank_info(bank)
    meta = info['parsed'].meta
    _require(shape == tuple(meta[k] for k in ('frame_count', 'height', 'width')),
             'candidate geometry differs from bank')
    path, record = _record(receive_record, expected_receive_sha256)
    source_hash, base_hash, enhanced_hash = map(frame_hash, (source, base, all_e))
    _require(record.get('complete') is True and record.get('source_frames_read') is False
             and record.get('sender_candidate_pixels_read') is False
             and record.get('base_reference_unchanged') is True,
             'candidate cache lacks independent source-free entropy receive proof')
    _require(record.get('bank_sha256') == expected_bank_sha256
             and source_hash == expected_source_rgb_sha256
             and base_hash == record.get('base_hash') == meta['base_rgb_sha256']
             and enhanced_hash == record.get('enhanced_hash'),
             'source/base/full candidate RGB does not match authenticated provenance')
    artifacts = record.get('artifacts', {})
    _require(type(artifacts) is dict and _valid_hash(artifacts.get('received_E.npz')),
             'independent receive NPZ artifact identity is required')
    _require(file_hash(path.parent/'received_E.npz') == artifacts['received_E.npz'],
             'independent candidate reconstruction artifact changed')
    binding = dict(bank_sha256=expected_bank_sha256, source_rgb_sha256=source_hash,
        base_rgb_sha256=base_hash, candidate_rgb_sha256=enhanced_hash,
        shape=list(source.shape), receive_record=str(path), receive_record_sha256=expected_receive_sha256,
        candidate_artifact_sha256=artifacts['received_E.npz'],
        candidate_cache_authenticated=True, candidate_encode_measured_here=False,
        candidate_decode_measured_here=False, candidate_encode_seconds=None)
    return CandidateCache(bank, source, base, all_e, binding, time.monotonic()-began)


def load_sender(checkpoint, config, *, expected_sha256, completion_record=None,
                expected_completion_sha256=None, allow_smoke=False):
    """Load R_s with its exact fixed R_g/G training policy and completion proof.

allow_smoke is explicit test authorization, never evidence of trained quality.
Even smoke checkpoints must bind the entire receiver configuration; when a
completion record is supplied it must authenticate the selected artifact.
No receiver model, source video or diffusion model is loaded by this function.
"""
    _require(type(allow_smoke) is bool and _valid_hash(expected_sha256),
             'explicit sender hash and boolean smoke permission required')
    fmt.validate(config)
    path = Path(checkpoint).resolve()
    model, payload = sender.load_model(path, expected_sha256=expected_sha256)
    binding = payload['binding']
    protocol = binding.get('protocol', {})
    teacher = protocol.get('teacher', {})
    smoke = protocol.get('smoke')
    _require(protocol.get('format') == 'routervc_sender_training_v1' and type(smoke) is bool
             and teacher.get('format') == 'routervc_sender_final_teacher_v1'
             and teacher.get('profile') == fmt.PROFILE
             and teacher.get('label_scope') == sender.LABEL_SCOPE,
             'sender lacks measured fixed-receiver FINAL-gain training provenance')
    _require(teacher.get('wire_config') == config
             and teacher.get('receiver', {}).get('sha256') == config['receiver_router']
             and teacher.get('max_g') == config['max_g']
             and teacher.get('boundary_lambda') == config['boundary_lambda'],
             'sender training R_g/G profile differs from requested receiver')
    _require(binding.get('architecture') == payload['architecture']
             and binding.get('arm') == ('zero_source' if model.config.zero_source else 'source'),
             'sender arm/architecture binding differs from weights')
    enhancement = teacher.get('enhancement', {}).get('sha256')
    _require(_valid_hash(enhancement), 'sender training enhancement model hash missing')
    if not allow_smoke:
        _require(not smoke and payload['step'] > 0
                 and teacher.get('receiver', {}).get('smoke_weights') is False,
                 'formal sender deployment requires trained non-smoke R_s and R_g')
        _require(completion_record is not None, 'formal sender completion record required')
    completed = None
    if completion_record is not None:
        done, result = _record(completion_record, expected_completion_sha256)
        try:
            relative = str(path.relative_to(done.parent))
        except ValueError as error:
            raise ValueError('sender model does not belong to completed training directory') from error
        _require(result.get('complete') is True and result.get('role') == ROLE
                 and result.get('artifacts', {}).get(relative) == expected_sha256,
                 'sender completed artifact is missing or changed')
        _, training = _record(done.parent/'config.json', result.get('config'))
        _require(training.get('protocol') == protocol
                 and training.get('architectures', {}).get(binding['arm']) == payload['architecture']
                 and training.get('labels') == binding.get('labels')
                 and training.get('scale') == binding.get('scale'),
                 'sender completion training binding differs from checkpoint')
        completed = dict(path=str(done), sha256=expected_completion_sha256, artifact=relative)
    else:
        _require(expected_completion_sha256 is None, 'completion hash without completion record')
    context_binding = dict(checkpoint=str(path), sha256=expected_sha256, arm=binding['arm'],
        protocol_sha256=_json_hash(protocol), payload_binding_sha256=_json_hash(binding),
        completion=completed, smoke_weights=smoke, trained_updates=payload['step'],
        formal_weights=(not smoke and payload['step'] > 0 and completed is not None
                        and teacher.get('receiver', {}).get('smoke_weights') is False),
        enhancement_sha256=enhancement, allow_smoke=allow_smoke)
    # A JSON copy avoids a caller subsequently mutating its training config.
    return SenderContext(model, payload, json.loads(json.dumps(config)), context_binding)


def _validate_candidates(candidates, *, pixels):
    _require(type(candidates) is CandidateCache, 'authenticated CandidateCache required')
    b = candidates.binding
    _require(b.get('candidate_cache_authenticated') is True
             and _sha(candidates.bank) == b.get('bank_sha256'), 'candidate bank changed after authentication')
    info = bank_info(candidates.bank)
    if pixels:
        for array, key in ((candidates.source, 'source_rgb_sha256'), (candidates.base, 'base_rgb_sha256'),
                           (candidates.all_e, 'candidate_rgb_sha256')):
            _require(list(array.shape) == b['shape'] and frame_hash(array) == b[key],
                     'candidate pixels changed after authentication')
    return info


def conditional_order(candidates, context):
    """One full greedy conditional order, independent of any future E budget.

Negative/zero remaining predictions terminate the sequence. Gains are model
predictions, not measured quality promises or a globally optimal allocation.
Only the tiny CPU R_s is called, at most once per candidate addition (16).
"""
    began = time.monotonic()
    info = _validate_candidates(candidates, pixels=True)
    _require(type(context) is SenderContext, 'authenticated SenderContext required')
    fmt.validate(context.config)
    _require(context.payload['binding']['protocol']['teacher']['wire_config'] == context.config
             and _json_hash(context.payload['binding']) == context.binding['payload_binding_sha256']
             and getattr(context.model, '_sender_model_sha256', None) == context.binding['sha256']
             and asdict(context.model.config) == context.payload['architecture'],
             'sender authentication/training policy changed after loading')
    _require(all(value.equal(context.payload['state_dict'][key])
                 for key, value in context.model.state_dict().items()),
             'sender weights changed after checkpoint authentication')
    _require(info['parsed'].meta['enhancement_model_sha256'] == context.binding['enhancement_sha256'],
             'candidate enhancement differs from sender training teacher')
    costs = np.asarray(info['e_bytes'], dtype=np.int64)
    order, trace = [], []
    reason = 'all_regions_selected'
    for _ in range(16):
        coverage = np.zeros(16, np.float32)
        coverage[order] = 1
        received = compose_candidates(candidates.base, candidates.all_e, order, info['rois'])
        values = sender.predict(context.model, candidates.source, candidates.base, received,
            candidates.all_e, coverage, costs, context.config['max_g'])
        prediction = values.detach().cpu().numpy()
        _require(prediction.shape == (1, 16), 'invalid conditional sender output geometry')
        prediction = prediction[0]
        ranked = sender.rank_candidates(prediction, costs, coverage)
        chosen = ranked[0] if ranked else None
        trace.append(dict(selected_before=list(order), received_hash=frame_hash(received),
            predictions=prediction.astype(float).tolist(), chosen=chosen,
            gain=None if chosen is None else float(prediction[chosen]),
            gain_per_byte=None if chosen is None else float(prediction[chosen])/int(costs[chosen]),
            incremental_bytes=0 if chosen is None else int(costs[chosen])))
        if chosen is None:
            reason = 'no_positive_predicted_remaining_gain'
            break
        order.append(chosen)
    result = dict(format=FORMAT, order=order, packet_bytes=costs.tolist(), trace=trace,
        stop_reason=reason, config=context.config, sender=context.binding,
        candidates=candidates.binding, budget_independent=True, generator_calls=0,
        source_visible_only_at_sender=True, candidate_pixels_transmitted=False,
        sender_router_is_receiver_dependency=False, mask_bytes=0,
        scope='conditional predicted FINAL LPIPS gain per actual complete-region byte; not measured RD',
        allocation_seconds=time.monotonic()-began)
    # Accidental alteration of an order or binding must not silently emit a
    # different stream under the recorded plan. This is not a digital signature.
    result['plan_sha256'] = _json_hash(result)
    return result


def encode_prefix(candidates, plan, e_budget):
    """Return (RVRC bytes, ledger) for an E-packet byte cap; base/header are extra.

The first unaffordable bundle stops the prefix, even when a later one fits.
No E payload is regenerated/refined and no budget-dependent R_s rerun occurs.
"""
    info = _validate_candidates(candidates, pixels=False)
    _require(type(plan) is dict and plan.get('format') == FORMAT
             and plan.get('plan_sha256') == _json_hash({k: v for k, v in plan.items() if k != 'plan_sha256'}),
             'sender prefix plan changed')
    _require(plan['candidates'] == candidates.binding and plan['packet_bytes'] == info['e_bytes']
             and plan.get('budget_independent') is True and plan.get('generator_calls') == 0,
             'prefix plan belongs to different candidates')
    selected = sender.prefix_under_budget(plan['order'], np.asarray(info['e_bytes'], dtype=np.int64), e_budget)
    raw = subset_bank(candidates.bank, selected['indices'])
    wire = fmt.wrap(raw, plan['config'])
    config, inner, parsed, header = fmt.parse(wire)
    packet_bytes = sum(len(p.wire) for p in parsed.packets)
    expected = b''.join(p.wire for index in selected['indices'] for p in info['groups'][index])
    _require(config == plan['config'] and inner[:parsed.base_end] == candidates.bank[:info['parsed'].base_end]
             and inner[parsed.base_end:] == expected and packet_bytes == selected['packet_bytes']
             and len(wire) == header+parsed.base_end+packet_bytes,
             'literal entropy packet prefix byte accounting failed')
    ledger = dict(indices=selected['indices'], packet_bytes=packet_bytes, header_bytes=header,
        base_container_bytes=parsed.base_end, base_codec_payload_bytes=len(parsed.base), total_bytes=len(wire),
        packet_count=len(parsed.packets), e_budget=e_budget, unused_e_budget=selected['unused_bytes'],
        bpp=8*len(wire)/np.prod(candidates.base.shape[:3]).item(),
        explicit_E_mask_bytes=0, explicit_G_map_bytes=0, protection_mask_bytes=0,
        sender_router_bytes=0, source_frames_transmitted=False, candidate_pixels_transmitted=False,
        literal_prefix=True, payloads_regenerated=False, generator_calls=0,
        plan_sha256=plan['plan_sha256'], wire_sha256=_sha(wire), profile=fmt.PROFILE,
        candidate_cache_reused=True, candidate_encode_measured_here=False,
        candidate_decode_measured_here=False, candidate_encode_seconds=None,
        allocation_seconds=plan['allocation_seconds'], smoke_weights=plan['sender']['smoke_weights'],
        formal_weights=plan['sender']['formal_weights'])
    return wire, ledger


def plan_and_encode(candidates, context, budgets):
    """Allocate once and return (plan, [(wire, ledger), ...]) for all E caps."""
    budgets = list(budgets)
    _require(all(type(b) is int and b >= 0 for b in budgets), 'nonnegative integer E byte caps required')
    plan = conditional_order(candidates, context)
    return plan, [encode_prefix(candidates, plan, budget) for budget in budgets]
