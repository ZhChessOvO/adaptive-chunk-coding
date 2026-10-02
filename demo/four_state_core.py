"""Four-state counterfactual protocol; all bytes are serialized, not estimated.

The table isolates one spatial cell, with Base in its neighboring context.
It is NOT an additive full-video quality model or a receiver-derived G policy.
"""
from pathlib import Path
import hashlib

import numpy as np

from demo import scalable_cooperation_format as fmt
from demo.scalable_format import parse
from demo.chunk_enhancement_experiment import read
from demo.conditioned_generation_cache import FEATURES
from demo.scalable_codec import file_hash, atomic_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_four_state_20261002')
MODELS = DEFAULT.parent/'a800_online_eg_20261002/joint'
ENHANCEMENT = MODELS/'enhancement.pt'
ADAPTER = MODELS/'adapter.pt'
STATES = ('B', 'E', 'G', 'EG')
ROIS = [[x, y, 128, 128] for y in range(0, 512, 128) for x in range(0, 512, 128)]
CODE = ('four_state_core.py', 'four_state_prepare.py', 'four_state_receive.py',
        'four_state_labels.py', 'four_state_pipeline.py', 'run_four_state.sh')


def rows(smoke=False):
    values = read(FEATURES)['samples']
    reds = [r for r in values if r['sample']['dataset'] == 'REDS']
    uvg = [r for r in values if r['sample']['dataset'] == 'UVG']
    assert len(reds) == 90 and len(uvg) == 30
    # Interleave domains so partial progress is not REDS-only.
    mixed = [r for i in range(30) for r in (*reds[3*i:3*i+3], uvg[i])]
    return [reds[0], uvg[0]] if smoke else mixed


def seed_for(sid, region):
    return int(hashlib.sha256(f'four-state-v1/{sid}/{region}'.encode()).hexdigest()[:12], 16)


def control(profile, sid, region):
    return dict(profile, seed=seed_for(sid, region), strength=1., window=17, stride=8,
                context=64, feather=16, processing_scale=1, blend=1., protect=[],
                generate=[[0, 17, *ROIS[region]]])


def subset(bank, indices):
    """Re-use complete, independent packets; do not re-encode at another quality."""
    p = parse(bank)
    if len(set(indices)) != len(indices) or any(i not in range(16) for i in indices):
        raise ValueError('invalid or duplicate region selection')
    ordered = []
    for i in indices:
        packets = [v for v in p.packets if v.meta['roi'] == ROIS[i]]
        if [(v.meta['start'], v.meta['count']) for v in packets] != [(0, 1), (1, 8), (9, 8)]:
            raise ValueError('each 17-frame E option must contain I + P8 + P8')
        ordered += [v.wire for v in packets]
    return bank[:p.base_end]+b''.join(ordered)


def isolation(base, enhanced, index):
    result = base.copy()
    x, y, w, h = ROIS[index]
    result[:, y:y+h, x:x+w] = enhanced[:, y:y+h, x:x+w]
    return result


def crop(frames, index):
    x, y, w, h = ROIS[index]
    return frames[:, y:y+h, x:x+w].copy()


def costs(bank, profile, sid, index):
    b, e = subset(bank, []), subset(bank, [index])
    c = control(profile, sid, index)
    g, eg = fmt.wrap(b, c), fmt.wrap(e, c)
    overhead = len(g)-len(b)
    return dict(base_container_bytes=len(b), e_packet_bytes=len(e)-len(b),
                g_region_bytes=fmt.REGION.size,
                g_shared_bytes=overhead-fmt.REGION.size,
                individual_stream_bytes=dict(B=len(b), E=len(e), G=len(g), EG=len(eg)))


def aggregate_bytes(base_bytes, e_bytes, states, shared_g_bytes, region_bytes):
    """A single shared G envelope, not N complete per-cell files added together."""
    if len(e_bytes) != len(states) or any(s not in STATES for s in states):
        raise ValueError('bad state map')
    ng = sum(s in ('G', 'EG') for s in states)
    return base_bytes + sum(b for b, s in zip(e_bytes, states) if s in ('E', 'EG')) + (
        shared_g_bytes+ng*region_bytes if ng else 0)


def verify(folder, artifacts):
    for filename, digest in artifacts.items():
        if file_hash(folder/filename) != digest:
            raise RuntimeError(f'changed artifact: {folder/filename}')


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'configuration drift: {path}')
    else:
        atomic_json(path, value)


def protocol(root, smoke):
    from demo.online_eg_decode import identities
    completed = read(MODELS/'complete.json')
    assert completed['steps'] == 3000 and completed['enhancement_changed']
    for key in ('enhancement', 'adapter'):
        assert completed[key+'_sha256'] == file_hash(MODELS/f'{key}.pt')
    value = dict(version=1, code={n:file_hash(REPO/'demo'/n) for n in CODE},
        enhancement=file_hash(ENHANCEMENT), adapter=file_hash(ADAPTER),
        profile=identities(ADAPTER), feature_manifest=file_hash(FEATURES),
        samples=[r['sample_id'] for r in rows(smoke)], smoke=smoke,
        states=list(STATES), rois=ROIS, qstep=1., neighbor_E='none in isolated labels',
        seed_policy='sha256 sample/cell, paired across G and EG',
        role='mixed existing component-training windows; not independent evaluation',
        signal='explicit G control charged; receiver-derived policy not implemented')
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root/'protocol.json', value)
    return value
