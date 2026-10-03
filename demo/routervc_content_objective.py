"""Offline content-aware supervision contract, not a trained semantic Router.

All source-derived annotations are TRAIN/EVALUATION ONLY. A future shared
receiver student must predict these targets from received reconstructions and
coverage, never from this module's source annotations. This module creates no
detectors, routes, protection maps, or packets, and leaves B/E/G/EG available.

Inputs use states (B, E, G, EG) and categories (text_digits, face, key_structure).
External metric providers must document what they actually measure; ordinary
PSNR, edges, or LPIPS alone do not establish identity/content preservation.
Error scales must be fixed using training data or an explicit metric range,
not fitted separately to the four candidates or to evaluation data.

For region i, state s, category k, externally supplied importance w in [0,1],
error e >= 0 and fixed positive scale sigma:
    C_i(s) = sum_k w_ik * e_isk / sigma_k / K
    H_i(G) = sum_k w_ik * max(e_iGk - e_iBk, 0) / sigma_k / K
    H_i(EG) = sum_k w_ik * max(e_iEGk - e_iEk, 0) / sigma_k / K
    J_i(s) = LPIPS_i(s) + content_weight*C_i(s) + harm_weight*H_i(s)
H(B)=H(E)=0 structurally; these are not negative G-harm training examples.
K is always three, not the observed count or importance sum, so small weights
stay small. Harm is clipped per category before summing: improving a face must
not cancel a changed digit. These are measured proxy objectives, not guarantees.

Unknown annotations/errors remain NaN with zero supervision weight. Aggregate
content targets require all category contributions to be known; per-category
targets can still use partial annotations. An explicitly assessed zero weight
has zero contribution even when that category's errors were not measured.
Pure LPIPS gains always remain available as a separate ablation target.
"""

from copy import deepcopy
import math
from numbers import Real

import numpy as np


SCHEMA = 'routervc-content-teacher-v1'
STATES = ('B', 'E', 'G', 'EG')
CATEGORIES = ('text_digits', 'face', 'key_structure')
GAINS = ('E_over_B', 'G_over_B', 'EG_over_E')
G_PARENTS = ('G_over_B', 'EG_over_E')
SCOPES = dict(text_digits='recognized_text', face='face_identity',
              key_structure='annotated_structure')
# A five-landmark geometry provider is useful but must never be mislabeled as
# identity preservation. The metric's exact scope stays in exported metadata.
ALLOWED_SCOPES = {key: {scope} for key, scope in SCOPES.items()}
ALLOWED_SCOPES['face'].add('face_landmarks')


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def _metadata(metadata):
    """Validate declared provenance, not the scientific validity of annotations."""
    if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA:
        raise ValueError('missing or unsupported offline teacher schema')
    if metadata.get('scope') != 'offline_train_evaluation':
        raise ValueError('source annotations are restricted to offline train/evaluation')
    if metadata.get('source_role') not in ('train', 'evaluation'):
        raise ValueError('source_role must explicitly be train or evaluation')
    for key in ('sample_id', 'annotation_provenance', 'importance_definition', 'scale_provenance'):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f'missing {key}')
    digest = metadata.get('source_sha256', '')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('source_sha256 must identify the source used by the teacher')
    if tuple(metadata.get('state_order', ())) != STATES:
        raise ValueError('state_order must be B, E, G, EG')
    if tuple(metadata.get('category_order', ())) != CATEGORIES:
        raise ValueError('category_order differs from the teacher contract')
    metrics = metadata.get('metrics')
    if not isinstance(metrics, dict) or set(metrics) != set(CATEGORIES):
        raise ValueError('all three external content metrics must be described')
    scales = []
    for category in CATEGORIES:
        spec = metrics[category]
        if not isinstance(spec, dict) or spec.get('scope') not in ALLOWED_SCOPES[category]:
            raise ValueError(f'{category} requires an explicit content-specific metric scope')
        if spec.get('direction') != 'lower_is_better':
            raise ValueError('content errors must be lower-is-better quantities')
        for key in ('name', 'definition'):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f'missing {category} metric {key}')
        scales.append(_number(spec.get('scale'), f'{category} scale', positive=True))
    return np.asarray(scales, dtype=np.float64)


def _target(values):
    """Finite fillers are safe only together with the returned loss weights."""
    known = np.isfinite(values)
    return dict(value=np.where(known, values, 0.), weight=known.astype(np.float64))


def _gains(cost):
    return np.stack((cost[:, 0] - cost[:, 1], cost[:, 0] - cost[:, 2],
                     cost[:, 1] - cost[:, 3]), axis=1)


def build_teacher_targets(lpips, importance, content_errors, annotation_status,
                          metadata, *, content_weight=1., harm_weight=1.,
                          include_importance_target=True):
    """Build one sample's offline labels; accepts any positive number of regions.

    lpips: [R,4], finite nonnegative measured errors.
    importance: [R,3], finite [0,1] for assessed categories, NaN for unknown.
    content_errors: [R,4,3], nonnegative external errors, NaN if unmeasured.
    annotation_status: [R,3] strings present/absent/unknown. Absent means an
      actual assessment found no such content, not merely no detector output.
      It requires weight zero and zero/NaN errors. Unknown requires NaNs in
      both importance and all four errors. Missing state measurements for
      present content stay unknown; they do not become no-harm labels.
    metadata: SCHEMA, offline scope, train/evaluation source_role, sample_id,
      source_sha256, annotation_provenance, importance_definition,
      scale_provenance, state_order, category_order, region_ids, and metrics.
      Each category metric has name, definition, scope (SCOPES), positive
      scale and direction='lower_is_better'. Scales/proxies need validation
      outside this contract; provenance strings are not proof of accuracy.

    Returned costs retain NaN for unavailable content-aware objectives. Every
    regression target contains BOTH value and weight; never train on its
    finite fillers without the weights. Positive gains mean improvement.
    No byte or compute term is added here; those remain allocator constraints.
    """
    scales = _metadata(metadata)
    cw = _number(content_weight, 'content_weight')
    hw = _number(harm_weight, 'harm_weight')
    p = np.asarray(lpips, dtype=np.float64)
    w = np.asarray(importance, dtype=np.float64)
    e = np.asarray(content_errors, dtype=np.float64)
    status = np.asarray(annotation_status)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] != len(STATES):
        raise ValueError('lpips must have shape [positive_regions,4]')
    regions = p.shape[0]
    if (w.shape != (regions, 3) or e.shape != (regions, 4, 3)
            or status.shape != (regions, 3)):
        raise ValueError('importance/status must be [R,3], content_errors [R,4,3]')
    ids = metadata.get('region_ids')
    if (not isinstance(ids, (list, tuple)) or len(ids) != regions
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != regions):
        raise ValueError('region_ids must uniquely identify all supplied regions')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('lpips must be finite and nonnegative')
    if (np.isinf(w).any() or np.isinf(e).any()
            or np.any(w < 0) or np.any(w > 1) or np.any(e < 0)):
        raise ValueError('importance is [0,1]; errors are nonnegative; infinity is invalid')
    if not np.isin(status, ('present', 'absent', 'unknown')).all():
        raise ValueError('annotation status must be present, absent, or unknown')
    unknown = status == 'unknown'
    if (np.any(~np.isnan(w[unknown]))
            or np.any(~np.isnan(e.transpose(0, 2, 1)[unknown]))):
        raise ValueError('unknown annotations must not contain invented weights or errors')
    if np.any(~np.isfinite(w[~unknown])):
        raise ValueError('assessed content needs explicit importance, including absent zero')
    absent = status == 'absent'
    absent_errors = e.transpose(0, 2, 1)[absent]
    if np.any(w[absent] != 0) or np.any(np.isfinite(absent_errors) & (absent_errors != 0)):
        raise ValueError('assessed-absent content requires zero importance and zero/NaN errors')

    # NaN * 0 must not invalidate an explicitly assessed zero contribution.
    zero = (~unknown & (w == 0))[:, None, :]
    category_cost = np.where(zero, 0., w[:, None, :] * e / scales)
    content_cost = category_cost.mean(axis=2)  # any unknown stays unknown
    category_harm = np.maximum(np.stack((category_cost[:, 2] - category_cost[:, 0],
                                        category_cost[:, 3] - category_cost[:, 1]), axis=1), 0.)
    g_harm = category_harm.mean(axis=2)
    state_harm = np.zeros_like(p)
    state_harm[:, 2:] = g_harm
    objective = p.copy()
    if cw:
        objective += cw * content_cost
    if hw:
        objective += hw * state_harm
    targets = dict(lpips_gain=_target(_gains(p)),
                   content_gain=_target(_gains(content_cost)),
                   category_content_gain=_target(_gains(category_cost)),
                   generation_harm=_target(g_harm),
                   category_generation_harm=_target(category_harm),
                   objective_gain=_target(_gains(objective)))
    if include_importance_target:
        targets['importance'] = _target(w)
    return dict(schema=SCHEMA, metadata=deepcopy(metadata), states=STATES,
                gains=GAINS, generation_parents=G_PARENTS, categories=CATEGORIES,
                coefficients=dict(content=cw, harm=hw), category_divisor=3,
                category_content_cost=category_cost, content_cost=content_cost,
                generation_harm=g_harm, objective_cost=objective,
                objective_known=np.isfinite(objective), targets=targets,
                annotation_status=status.copy(), semantic_guarantee=False,
                receiver_inputs=False, offline_supervision_only=True)
