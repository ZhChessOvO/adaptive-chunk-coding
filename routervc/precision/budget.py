"""Tiny calibration-table selector; evaluation final scores are never inputs.

This bounded prototype searches four measured E-cap grid points per width,
not all possible prefixes. Candidate bytes come from actual sender encodings.
"""
import numpy as np


def features(source, base, base_bytes, full_e_bytes):
    if source.shape != base.shape or source.dtype != np.uint8: raise ValueError('invalid source/B')
    pixels = int(np.prod(source.shape[:3]))
    x = source.astype(np.float32)/255; b = base.astype(np.float32)/255
    mse = float(np.mean((x-b)**2))
    edge = float(np.mean(np.abs(np.diff(x,axis=2)))+np.mean(np.abs(np.diff(x,axis=1))))
    return [float(np.log(max(1e-8,v))) for v in
            (base_bytes*8/pixels, full_e_bytes*8/pixels, mse+1e-6, edge+1e-6)]


def fit(points):
    if not points or any(p['cohort'] != 'calibration' for p in points):
        raise ValueError('selector calibration must contain training windows only')
    vectors = np.asarray([p['sender_features'] for p in points],float)
    scale = np.maximum(vectors.std(0), .1)
    return dict(version='three_neighbour_table_v1', scale=scale.tolist(), neighbours=3,
        rows=[dict(width=p['width'], cap=p['cap'], sample_id=p['sample_id'],
            features=p['sender_features'], lpips=p['quality']['lpips_alex']) for p in points],
        evaluation_quality_used=False)


def predict(table, candidate):
    # Deliberately accept only sender-side data, not an evaluated point object.
    if set(candidate) != {'width','cap','sender_features','actual_bytes','base_bytes','key'}:
        raise ValueError('unexpected selector inputs (possible evaluation score leakage)')
    rows = [r for r in table['rows'] if r['width']==candidate['width'] and r['cap']==candidate['cap']]
    if not rows: raise ValueError('missing calibration group')
    distances = np.linalg.norm((np.asarray([r['features'] for r in rows])-
        np.asarray(candidate['sender_features']))/np.asarray(table['scale']),axis=1)
    chosen = np.argsort(distances,kind='stable')[:table['neighbours']]
    weights = 1/(distances[chosen]+.1)
    return float(np.average([rows[i]['lpips'] for i in chosen],weights=weights))


def choose(table, candidates, byte_budget):
    if type(byte_budget) is not int or byte_budget < 0: raise ValueError('invalid byte budget')
    feasible = [c for c in candidates if c['actual_bytes'] <= byte_budget]
    if not feasible: return dict(key=None, feasible=False, reason='B does not fit')
    scores = {c['key']:predict(table,c) for c in feasible}
    chosen = min(feasible,key=lambda c:(scores[c['key']],c['actual_bytes'],c['width'],c['cap']))
    return dict(key=chosen['key'], feasible=True, predicted_lpips=scores[chosen['key']],
                actual_bytes=chosen['actual_bytes'], width=chosen['width'], candidate_predictions=scores)


def fixed(candidates, byte_budget, width):
    feasible = [c for c in candidates if c['width']==width and c['actual_bytes'] <= byte_budget]
    return max(feasible,key=lambda c:(c['actual_bytes'],c['cap']))['key'] if feasible else None


def base_fraction(candidates, byte_budget, fraction=.7):
    feasible_widths = sorted({c['width'] for c in candidates if c['base_bytes'] <= byte_budget})
    if not feasible_widths: return None
    fits = [w for w in feasible_widths if next(c['base_bytes'] for c in candidates if c['width']==w) <= fraction*byte_budget]
    # Finest base that meets the target, otherwise the cheapest available base.
    width = min(fits) if fits else min(feasible_widths,key=lambda w:min(
        c['base_bytes'] for c in candidates if c['width']==w))
    return fixed(candidates,byte_budget,width)
