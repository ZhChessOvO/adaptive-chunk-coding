"""Immutable, train-only real width3 B/E receiver inputs for paired repair."""
from collections import Counter
from pathlib import Path
import random

import numpy as np

from demo.routervc_fullview_probe import read, immutable, digest
from routervc.latent.router_data import source
from demo.scalable_generation_decode import ASSETS

REPO = Path(__file__).resolve().parents[2]
RUNS = Path('/root/autodl-fs/DCVC/runs')
PREVIOUS = RUNS/'routervc_fusion_20261010/p1_data'
CURRENT = RUNS/'a800_online_eg_20261002/joint/adapter.pt'
DEFAULT = RUNS/'routervc_texture_repair_20261014'
SEED = 20261014
CODE = ('routervc/generation/texture_data.py', 'routervc/generation/texture_losses.py',
        'routervc/generation/texture_training.py', 'tools/texture_repair.py',
        'demo/roi_condition_data.py', 'demo/conditioned_generation_train.py',
        'demo/stage_c_seedvr2_bridge.py', 'demo/stage_c_seedvr2_lora_utils.py')


def protocol(root, steps, smoke=False):
    previous = read(PREVIOUS/'protocol.json')
    done = read(PREVIOUS/'complete.json')
    if not done['complete'] or done['protocol_sha256'] != digest(PREVIOUS/'protocol.json'):
        raise ValueError('previous input collection incomplete/changed')
    index = {r['sample_id']:r for r in done['samples']}
    rows = []
    for row in previous['rows']:
        if row['router_split'] != 'train':
            continue
        item = index[row['sample_id']]
        capture = Path(item['capture'])
        if digest(capture/'complete.json') != item['receipt_sha256']:
            raise ValueError('capture receipt changed')
        receipt = read(capture/'complete.json')
        rows.append(dict(row, received=str(capture/'received.npz'),
                         received_sha256=receipt['artifacts']['received.npz'],
                         capture_receipt=item['receipt_sha256']))
    if Counter(r['dataset'] for r in rows) != dict(REDS=36, UVG=12):
        raise ValueError('expected the existing grouped 48-window training set')
    if smoke:
        rows = [next(r for r in rows if r['dataset'] == d) for d in ('REDS', 'UVG')]
    result = dict(format='routervc_g_texture_repair_v1', steps=steps, smoke=smoke,
        rows=rows, seed=SEED, initial=str(CURRENT), initial_sha256=digest(CURRENT),
        prior_protocol=digest(PREVIOUS/'protocol.json'),
        assets={k:dict(path=str(v), sha256=digest(v)) for k,v in ASSETS.items()
                if k in ('dit','vae','positive','negative')},
        code={n:digest(REPO/n) for n in CODE},
        crop=256, halo=64, core=128, frame_count=17,
        sampling='REDS:UVG=3:1; stateless paired random sample and 16px-aligned RGB crop',
        conditioning='actual frozen width3 received Y, posterior mean BF16; no GT condition',
        transform='LPIPS view only: 112/120/128/136/144px crop, +/-8px shift, resized128',
        phase='source-referenced RGB error, Gaussian sigma3 highpass, temporal/spatial phase mean, periods8/16 RMS',
        loss=dict(velocity=.25, clean_latent_l1=.025, latent_temporal=.05,
                  lpips=.5, low_frequency=.1, edge=.05, rgb_temporal=.1),
        phase_weight=1., learning_rate=dict(peak=1e-5, end=2e-6, warmup=min(100, steps)),
        checkpoint_every=25, milestone_every=100, frozen=['UF','B/E','R_s','R_g','fusion','base_DiT','VAE'],
        arms=['control','repair'], promotion=False,
        control_scope='original G loss weights on same frozen CURRENT latent B/E inputs; not a rerun of old joint E training')
    immutable(Path(root)/'protocol.json', result)
    return result


def choose(rows, step, seed=SEED):
    rng = random.Random(seed+step*1000033)
    dataset = 'UVG' if (step-1) % 4 == 3 else 'REDS'
    row = rng.choice([r for r in rows if r['dataset'] == dataset])
    _, h, w, _ = row['shape']
    crop = [16*rng.randrange((w-256)//16+1), 16*rng.randrange((h-256)//16+1), 256, 256]
    return row, crop


def preload(rows, progress):
    memory = {}
    for i, row in enumerate(rows):
        if digest(row['received']) != row['received_sha256']:
            raise ValueError('received pixels changed')
        with np.load(row['received']) as z:
            received = z['enhanced'].copy()
        original = source(row)
        if original.shape != received.shape or list(original.shape) != row['shape']:
            raise ValueError('source/receiver geometry differs')
        memory[row['sample_id']] = original, received
        progress(phase='loading_verified_training_inputs', loaded=i+1, total=len(rows))
    return memory
