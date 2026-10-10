"""Lazy final-picture conditional labels on real, frozen sender prefixes."""
from collections import OrderedDict, Counter
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_npz
from demo.scalable_format import frame_hash
from routervc.latent import router_data, routing, generation
from routervc.cooperation import receiver, stream
from routervc.cooperation.render import current, render

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_cooperation_20261010')
PREVIOUS = Path('/root/autodl-fs/DCVC/runs/routervc_fusion_20261010/p1_data')
REPO = Path(__file__).resolve().parents[2]
CODE = ('routervc/cooperation/receiver.py', 'routervc/cooperation/render.py',
        'routervc/cooperation/stream.py', 'routervc/cooperation/receive.py',
        'routervc/cooperation/data.py', 'routervc/cooperation/training.py',
        'tools/cooperative_receiver.py', 'tools/cooperative_worker.py',
        'tools/test_cooperative_receiver.py', 'routervc/fusion/blend.py',
        'routervc/fusion/capture.py', 'demo/routervc_receiver_train.py')


def protocol():
    previous = read(PREVIOUS/'protocol.json'); done = read(PREVIOUS/'complete.json')
    if not done['complete'] or done['protocol_sha256'] != digest(PREVIOUS/'protocol.json'):
        raise ValueError('actual sender data incomplete/changed')
    index = {r['sample_id']:r for r in done['samples']}
    rows = [dict(r, capture=index[r['sample_id']]['capture'],
                 stream=index[r['sample_id']]['stream'],
                 capture_receipt=index[r['sample_id']]['receipt_sha256']) for r in previous['rows']]
    if Counter(r['router_split'] for r in rows) != dict(train=48, validation=12):
        raise ValueError('preserve 48/12 grouped pilot')
    groups = {}
    for r in rows:
        if groups.setdefault((r['dataset'], r['sequence']), r['router_split']) != r['router_split']:
            raise ValueError('sequence crosses split')
    return dict(format='cooperative_receiver_p2_v1', rows=rows, epochs=120,
        seed=20261010, learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1,
        initial_path=previous['receiver'], initial_sha256=previous['receiver_sha256'],
        frozen_sender=previous['sender'], frozen_sender_sha256=previous['sender_sha256'],
        prior_protocol=digest(PREVIOUS/'protocol.json'), prior_complete=digest(PREVIOUS/'complete.json'),
        G_assets=generation.assets(), receiver_profile=stream.identity(),
        code={n:digest(REPO/n) for n in CODE}, extra_candidates=2, max_g=8,
        scales='RMS of two TRAIN calibration windows (one per dataset); fixed before formal fit',
        labels='whole 17-frame final-multiband marginal gains; unknown candidates masked',
        states='empty, old-policy top3, top6, deterministic exploratory triple',
        policy_scope='actual frozen R_s prefixes; frozen old R_g plus exploration states, not new R_g on-policy',
        freeze=['UF', 'E_width3', 'R_s', 'G', 'multiband_F'], masks_sent=False,
        semantic_supervision=False)


def candidates(row, selection, extra=2):
    old = selection['indices']
    ranking = sorted(old, key=lambda i:(-selection['predictions'][i][0], i))
    seed = int(hashlib.sha256(('p2/'+row['sample_id']).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    explore = [int(i) for i in rng.permutation(16) if i not in old][:extra]
    pool = sorted(set(old+explore))
    states = [[], ranking[:3], ranking[:6], sorted((explore+ranking)[:3])]
    states = list(dict.fromkeys(tuple(sorted(s)) for s in states))
    return pool, [list(s) for s in states], explore


class Samples:
    def __init__(self, root, protocol, run, launch):
        self.root, self.protocol, self.run, self.launch = Path(root), protocol, run, launch
        self.memory = OrderedDict(); self.metric = None

    def prepare(self, row):
        folder = self.root/'samples'/row['sample_id']; folder.mkdir(parents=True, exist_ok=True)
        binding = dict(protocol=digest(self.root/'protocol.json'), row=row)
        immutable(folder/'binding.json', binding)
        if (folder/'complete.json').exists():
            done = read(folder/'complete.json')
            if done['binding'] != digest(folder/'binding.json'): raise ValueError('labels changed')
            verify_artifacts(folder, done['artifacts']); return done
        capture = Path(row['capture'])
        if digest(capture/'complete.json') != row['capture_receipt']: raise ValueError('old capture changed')
        receipt = read(capture/'complete.json'); verify_artifacts(capture, receipt['artifacts'])
        if digest(row['stream']) != receipt['stream_sha256']: raise ValueError('stream changed')
        selection = read(capture/'selection.json')
        pool, states, extra = candidates(row, selection, self.protocol['extra_candidates'])
        if extra:
            self.run.update(phase='teacher_missing_G', sample=row['sample_id'], extra=extra)
            self.launch('capture', ['--stream', row['stream'], '--regions', *map(str, extra)], folder/'extra')
        with np.load(capture/'received.npz') as z: base, received = z['base'], z['enhanced']
        patches = []
        for i in pool:
            parent = folder/'extra' if i in extra else capture
            dest = parent/f'g{i:02d}'; report = read(dest/'result.json')
            verify_artifacts(dest, report['artifacts'])
            with np.load(dest/'raw.npz') as z: raw = z['pixels']
            patches.append(dict(region=i, pixels=raw, core=report['core'], crop=report['crop']))
        # Cached reuse is authenticated and the old reconstruction must replay exactly.
        replay = current(received, patches, selection['indices'])
        if frame_hash(replay) != receipt['output_hash']: raise ValueError('old G replay differs')
        del replay
        pixels = router_data.source(row)
        inner, config, parsed = routing.parse(Path(row['stream']).read_bytes())
        inputs = receiver.old.build_inputs(base, received, routing.coverage(parsed))
        values = np.zeros((len(states), 16, 3), np.float32)
        weights = np.zeros_like(values); selected = np.zeros((len(states), 16, 1), np.float32)
        if self.metric is None:
            from demo.stage_c_three_path_roi_probe import LPIPSAlex
            with torch.random.fork_rng(devices=[0]): self.metric = LPIPSAlex(True)
        from demo.scalable_experiment import quality
        from demo.four_state_receive import codec_precision
        metrics = folder/'measurements.json'
        measured = read(metrics) if metrics.exists() else dict(binding=binding, sets={})
        if measured['binding'] != binding: raise ValueError('measurement journal changed')
        def score(indices):
            key = ','.join(map(str, sorted(indices))) or 'empty'
            if key not in measured['sets']:
                self.run.check()
                self.run.update(phase='teacher_final_fused_gains', sample=row['sample_id'],
                    G_set=list(indices), measured_sets=len(measured['sets']))
                began = time.monotonic()
                output = render(received, patches, indices)
                with torch.no_grad(), codec_precision(): q = quality(pixels, output, self.metric)
                if not all(q[k] is not None and np.isfinite(q[k]) for k in
                           ('lpips_alex', 'psnr_db', 'temporal_delta_mae')):
                    raise ValueError('nonfinite final-picture metric')
                measured['sets'][key] = dict(quality=q, output_hash=frame_hash(output),
                                             seconds=time.monotonic()-began)
                save(metrics, measured)
            return measured['sets'][key]['quality']
        for s, indices in enumerate(states):
            before = score(indices); selected[s, indices, 0] = 1
            for i in pool:
                if i in indices: continue
                after = score([*indices, i])
                values[s, i] = (before['lpips_alex']-after['lpips_alex'],
                    after['psnr_db']-before['psnr_db'],
                    before['temporal_delta_mae']-after['temporal_delta_mae'])
                weights[s, i] = 1
        atomic_npz(folder/'inputs.npz', **{k:v.numpy() for k,v in inputs.items()},
                   selected=selected, value=values, weight=weights)
        names = ['binding.json', 'inputs.npz', 'measurements.json']
        done = dict(complete=True, binding=digest(folder/'binding.json'), states=states,
            candidates=pool, new_G_calls=len(extra), reused_G_calls=len(selection['indices']),
            known_marginals=int(weights[..., 0].sum()), final_picture=True,
            masks_sent=False, extra_receipt=digest(folder/'extra/complete.json') if extra else None,
            artifacts={n:digest(folder/n) for n in names})
        save(folder/'complete.json', done)
        print('COOPERATIVE_LABELS_COMPLETE', row['sample_id'], done['known_marginals'], flush=True)
        return done

    def get(self, row):
        sid = row['sample_id']
        if sid not in self.memory:
            self.prepare(row)
            with np.load(self.root/'samples'/sid/'inputs.npz') as z:
                n = len(z['selected'])
                inputs = {k:torch.from_numpy(z[k].copy()).expand(n, *z[k].shape[1:])
                          for k in ('local_pairs', 'global_pairs', 'geometry', 'coverage')}
                inputs['selected'] = torch.from_numpy(z['selected'].copy())
                result = dict(inputs=inputs, targets={k:torch.from_numpy(z[k].copy()) for k in ('value', 'weight')})
            self.memory[sid] = result
        self.memory.move_to_end(sid)
        while len(self.memory) > 2: self.memory.popitem(last=False)
        return self.memory[sid]

    def release(self):
        self.memory.clear(); self.metric = None; gc.collect(); torch.cuda.empty_cache()
