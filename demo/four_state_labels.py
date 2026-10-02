"""Evaluation/teacher side only: source supervision, not receiver information.

Local counterfactual patches must not be stitched into a purported decoded
mixed stream: neighboring E changes the G condition. Store that scope explicitly.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import rows, ROIS, STATES, crop, verify, immutable_json
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import read
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import quality
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from demo.stage_c_a800_teacher import source_features


def labels(root):
    configure_torch()
    p = read(root/'protocol.json')
    assert read(root/'received/complete.json')['complete']
    metric, results = None, []
    dest = root/'labels'
    dest.mkdir(exist_ok=True)
    for row in rows(p['smoke']):
        sid = row['sample_id']
        received = root/'received'/sid
        encoded = root/'encoded'/sid
        dependencies = dict(protocol=file_hash(root/'protocol.json'), pair=row['pair_hash'],
            encoded=file_hash(encoded/'complete.json'), received=file_hash(received/'complete.json'))
        path = dest/f'{sid}.json'
        if path.exists():
            r = read(path)
            assert r['dependencies'] == dependencies
        else:
            started = time.monotonic()
            assert file_hash(Path(row['pair_path'])) == row['pair_hash']
            with np.load(row['pair_path'], allow_pickle=False) as cache:
                source = cache['source'].copy()
            e_report = read(received/'E.json')
            verify(received, e_report['artifacts'])
            with np.load(received/'received_E.npz', allow_pickle=False) as cache:
                base, enhanced = cache['base'].copy(), cache['enhanced'].copy()
            cost = read(encoded/'complete.json')['per_region_cost']
            if metric is None:
                metric = LPIPSAlex(True)
            regions = []
            source_stats = source_features(list(source), ROIS)
            base_stats = source_features(list(base), ROIS)
            enhanced_stats = source_features(list(enhanced), ROIS)
            for i in range(16):
                folder = received/f'cell_{i:02d}'
                record = read(folder/'result.json')
                verify(folder, record['artifacts'])
                with np.load(folder/'outputs.npz', allow_pickle=False) as outputs:
                    variants = dict(B=crop(base, i), E=crop(enhanced, i),
                                    G=outputs['G'].copy(), EG=outputs['EG'].copy())
                scores = {s:quality(crop(source,i), v, metric) for s,v in variants.items()}
                gain = {s:scores['B']['lpips_alex']-scores[s]['lpips_alex'] for s in STATES}
                times = {s:record['reports'][s]['runtime']['seconds_model_load_excluded'] for s in ('G','EG')}
                regions.append(dict(region=i, roi=ROIS[i], quality=scores, lpips_gain=gain,
                    # Positive means E adds more perceptual benefit with G than without.
                    interaction_gain=gain['EG']-gain['E']-gain['G'],
                    costs=cost[i], e_seconds=e_report['per_region_e_seconds'][i],
                    g_seconds=times, source_stats=source_stats[i], base_stats=base_stats[i],
                    received_E_stats=enhanced_stats[i],
                    sender_base_error_mse=scores['B']['rgb_mse']))
            r = dict(sample_id=sid, dataset=row['sample']['dataset'], sequence=row['sample']['sequence'],
                source_role=row['sample'].get('source_role'), dependencies=dependencies,
                seconds=time.monotonic()-started, regions=regions,
                receiver_uses_source=False, quality_scope='17-frame isolated 128px ROI after feather',
                source_stats_role='encoder-only features; never a free receiver policy')
            atomic_json(path, r)
        results.append(dict(sample_id=sid, dataset=r['dataset'], path=str(path), sha256=file_hash(path)))
        print(f'LABELS {len(results)}/{len(p["samples"])} {sid}', flush=True)
    immutable_json(root/'labels.json', dict(complete=True, samples=results,
        region_count=16*len(results), states=list(STATES),
        protocol=file_hash(root/'protocol.json'), datasets=dict(Counter(r['dataset'] for r in results))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    labels(p.parse_args().root)
