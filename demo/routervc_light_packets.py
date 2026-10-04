"""qE=2 complete-region banks; historical q1 sender remains unchanged.

Only sender-side candidate validation changes. The existing visual receiver
already reads qE from each entropy packet; no mask or new header is introduced.
"""
import numpy as np
import torch

from demo.scalable_format import parse, frame_hash
from demo.routervc_visual_router import grid_rois
from demo.routervc_encode import compose_candidates, validate_frames

QSTEP = 2.


def bank_info(bank):
    parsed = parse(bank)
    t, h, w = (parsed.meta[k] for k in ('frame_count', 'height', 'width'))
    if t < 17 or (t - 1) % 8:
        raise ValueError('expected I + complete P8 chunks')
    rois = grid_rois(h, w)
    expected = [(0, 1)] + [(s, 8) for s in range(1, t, 8)]
    groups = [[] for _ in rois]
    for packet in parsed.packets:
        m = packet.meta
        if m['roi'] not in rois or m['qstep'] != QSTEP:
            raise ValueError('expected raster-grid qE=2 bank')
        groups[rois.index(m['roi'])].append(packet)
    if any([(p.meta['start'], p.meta['count']) for p in g] != expected for g in groups):
        raise ValueError('each region must have complete ordered I/P8 packets')
    return dict(parsed=parsed, rois=rois, groups=groups,
                e_bytes=[sum(len(p.wire) for p in g) for g in groups])


def subset_bank(bank, indices):
    info = bank_info(bank)
    indices = list(indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or i not in range(16) for i in indices):
        raise ValueError('invalid region indices')
    return bank[:info['parsed'].base_end] + b''.join(p.wire for i in indices for p in info['groups'][i])


@torch.no_grad()
def predict_utility(bank, base, all_e, router):
    from demo.routervc_visual_policy import _model, predict
    validate_frames(base); validate_frames(all_e)
    info = bank_info(bank); m = info['parsed'].meta
    if (base.shape != all_e.shape or base.shape != (m['frame_count'], m['height'], m['width'], 3)
            or frame_hash(base) != m['base_rgb_sha256']):
        raise ValueError('candidate pixels differ from bank')
    model = _model(router)
    predictions = [predict(model, base, base, np.zeros(16, np.float32), info['rois'])[0]]
    for i in range(16):
        coverage = np.zeros(16, np.float32); coverage[i] = 1
        mixed = compose_candidates(base, all_e, [i], info['rois'])
        predictions.append(predict(model, base, mixed, coverage, info['rois'])[0])
    result = torch.stack(predictions); ids = torch.arange(16)
    direct = result[ids + 1, ids, 0]
    utility = torch.stack((torch.zeros_like(direct), direct, result[0, :, 1],
                           direct + result[ids + 1, ids, 1]), 1)
    return utility.numpy(), result.numpy()
