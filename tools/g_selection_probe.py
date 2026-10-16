"""Small offline GT-assisted selection diagnostic, NOT a deployable Router.

Reuse the six completed E50 reconstructions and selected G halos. Generate only
the missing candidate halos, then compare a finite set of at-most-eight-region
choices. No fitting, packet changes, or automatic model promotion.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from demo.scalable_format import frame_hash
from demo.scalable_experiment import now, quality
from routervc.cooperation.render import render
from routervc.generation import integration
from routervc.generation.texture_data import REPO
from routervc.latent import routing, generation
from tools.cooperative_review import source_rgb
from tools.g_integration import DEFAULT as PRIOR, check_protocol, conditions
from tools.texture_clean import CleanRun
from tools.texture_repair import execute

DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_g_selection_20261016')
FRAMES = [0, 4, 8, 12, 16]
CODE = ('tools/g_selection_probe.py', 'tools/test_g_selection_probe.py',
        'tools/run_g_selection_probe.sh')


def choices(gains, old):
    """Deterministic finite diagnostic choices, not exhaustive optimization."""
    if len(gains) != 16 or not np.isfinite(gains).all():
        raise ValueError('sixteen finite marginal gains required')
    if len(old) > 8 or len(set(old)) != len(old) or any(i not in range(16) for i in old):
        raise ValueError('invalid prior choice')
    ranked = sorted((i for i in range(16) if gains[i] > 0), key=lambda i: (-gains[i], i))
    return dict(old=list(old), off=[], positive_old=[i for i in old if gains[i] > 0],
                top4=ranked[:4], top6=ranked[:6], top8=ranked[:8])


def prepare(root):
    if (root/'protocol.json').exists():
        p = read(root/'protocol.json')
        for name, sha in p['code'].items():
            if digest(REPO/name) != sha: raise ValueError('probe source changed: '+name)
        verify_artifacts(root, p['artifacts'])
        if digest(PRIOR/'protocol.json') != p['prior_protocol']: raise ValueError('prior changed')
        return p
    old = check_protocol(PRIOR)
    verify_artifacts(PRIOR, read(PRIOR/'complete.json')['artifacts'])
    model = old['models']['control']; cases = []; files = []
    for job in old['jobs']:
        sid = job['row']['sample_id']; src = PRIOR/'samples'/sid/'control/main'
        done = read(src/'complete.json'); verify_artifacts(src, done['artifacts'])
        dest = root/'generation'/sid
        with np.load(src/'pixels.npz', allow_pickle=False) as z: received = z['enhanced'].copy()
        if frame_hash(received) != done['enhanced_hash']: raise ValueError('received hash differs')
        atomic_npz(dest/'received.npz', pixels=received); files.append(dest/'received.npz')
        for i in done['generated']:
            folder = dest/f'g{i:02d}'; folder.mkdir(parents=True, exist_ok=True)
            for name in ('raw.npz', 'complete.json'):
                shutil.copy2(src/f'g{i:02d}'/name, folder/name); files.append(folder/name)
        cases.append(dict(sample_id=sid, old=done['generated'], received_hash=done['enhanced_hash'],
                          cached_pixels_sha256=digest(src/'pixels.npz')))
    request = dict(model=model, seed=routing.SEED, cases=cases,
                   code={n:digest(REPO/n) for n in CODE})
    immutable(root/'generation/request.json', request); files.append(root/'generation/request.json')
    p = dict(created=now(), prior_protocol=digest(PRIOR/'protocol.json'), code=request['code'],
             model=model, jobs=old['jobs'], cases=cases, selection_frames=FRAMES,
             scope='six previously used E50 cases; offline GT-assisted finite choices, not an optimum',
             no_training=True, no_model_promotion=True, bytes_unchanged=True,
             artifacts={str(f.relative_to(root)):digest(f) for f in files})
    immutable(root/'protocol.json', p)
    return p


@torch.no_grad()
def generate(args):
    from tools.latent_router_worker import receiver_guard
    from tools.cooperative_sender_decode import deny_sender_checkpoint
    receiver_guard(args.output, args.output/'request.json'); sys.addaudithook(deny_sender_checkpoint)
    cfg = read(args.output/'request.json'); model = cfg['model']
    for name, sha in cfg['code'].items():
        if digest(REPO/name) != sha: raise ValueError('worker changed')
    if digest(model['path']) != model['sha256'] or generation.assets(Path(model['path'])) != model['assets']:
        raise ValueError('model/assets changed')
    run = CleanRun(SimpleNamespace(output=args.output, command='generation', max_hours=1))
    run.thread.start(); run.log_resources(); generator = None; completed = 0
    try:
        for case in cfg['cases']:
            dest = args.output/case['sample_id']
            with np.load(dest/'received.npz', allow_pickle=False) as z: received = z['pixels'].copy()
            if frame_hash(received) != case['received_hash']: raise ValueError('received changed')
            # Actual reload replay on a cached selected halo in EVERY case.
            if not (dest/'replay.json').exists():
                i = case['old'][0]; folder = dest/f'g{i:02d}'
                if generator is None: generator = integration.ExplicitCapture(Path(model['path']))
                raw, report = generator(received, routing.control(received.shape, model['assets'], i, cfg['seed']))
                prior = read(folder/'complete.json'); verify_artifacts(folder, prior['artifacts'])
                with np.load(folder/'raw.npz', allow_pickle=False) as z: np.testing.assert_array_equal(raw, z['pixels'])
                if conditions({'generation_runtime':[report]}) != conditions({'generation_runtime':[prior]}):
                    raise ValueError('replay geometry/condition/noise differs')
                atomic_json(dest/'replay.json', dict(exact=True, region=i, pixels=frame_hash(raw)))
            for i in range(16):
                run.check(); folder = dest/f'g{i:02d}'
                if (folder/'complete.json').exists():
                    verify_artifacts(folder, read(folder/'complete.json')['artifacts']); continue
                if generator is None: generator = integration.ExplicitCapture(Path(model['path']))
                raw, report = generator(received, routing.control(received.shape, model['assets'], i, cfg['seed']))
                atomic_npz(folder/'raw.npz', pixels=raw)
                report['artifacts'] = {'raw.npz':digest(folder/'raw.npz')}
                atomic_json(folder/'complete.json', report); completed += 1
                run.update(sample=case['sample_id'], region=i, newly_generated=completed)
                if args.stop_after and completed >= args.stop_after:
                    atomic_json(args.output/'intentional_stop.json', dict(new_regions=completed)); return
        files = [p for c in cfg['cases'] for p in (args.output/c['sample_id']).rglob('*') if p.is_file()]
        atomic_json(args.output/'complete.json', dict(complete=True, source_access_guard=True,
            sender_access_guard=True, actual_replay_cases=len(cfg['cases']),
            artifacts={str(p.relative_to(args.output)):digest(p) for p in files}))
    finally:
        del generator; gc.collect(); torch.cuda.empty_cache()
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


class FastMetric:
    def __init__(self):
        import lpips
        self.model = lpips.LPIPS(net='alex', verbose=False).cuda().eval()

    @torch.inference_mode()
    def evaluate(self, reference, reconstruction):
        values = []
        for a,b in zip(reference,reconstruction,strict=True):
            a = torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/127.5-1
            b = torch.from_numpy(b.copy()).permute(2,0,1)[None].cuda().float()/127.5-1
            values.append(float(self.model(a,b)))
        return float(np.mean(values))


def score(run, root, cfg):
    metric = FastMetric(); results = []
    for job, case in zip(cfg['jobs'],cfg['cases'],strict=True):
        sid = case['sample_id']; target = root/'scores'/f'{sid}.json'
        if target.exists(): results.append(read(target)); continue
        source = source_rgb(job['row']); dest = root/'generation'/sid
        with np.load(dest/'received.npz', allow_pickle=False) as z: received = z['pixels'].copy()
        patches = []
        for i in range(16):
            folder = dest/f'g{i:02d}'; r = read(folder/'complete.json'); verify_artifacts(folder,r['artifacts'])
            with np.load(folder/'raw.npz', allow_pickle=False) as z:
                patches.append(dict(region=i,pixels=z['pixels'].copy(),crop=r['crop'],core=r['core']))
        old = render(received,patches,case['old'])
        prior = PRIOR/'samples'/sid
        with np.load(prior/'control/main/pixels.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(old,z['reconstruction'])
        reference = read(prior/'scores.json'); old_quality = quality(source,old,metric)
        cpu_score = reference['metrics']['control']['quality']['lpips_alex']
        if abs(old_quality['lpips_alex']-cpu_score) > 1e-5:
            raise ValueError('GPU/CPU metric calibration differs')
        baseline = metric.evaluate(source[FRAMES],received[FRAMES]); gains = []
        for i in range(16):
            run.check(); output = render(received,patches,[i])
            gains.append(baseline-metric.evaluate(source[FRAMES],output[FRAMES]))
            run.update(phase='marginal_scoring',sample=sid,region=i)
        policies = choices(gains,case['old']); scores = {}; outputs = {}
        for name, indices in policies.items():
            run.check(); output = old if name=='old' else render(received,patches,indices)
            scores[name] = quality(source,output,metric); outputs[name] = output
        best = min(scores,key=lambda k:scores[k]['lpips_alex'])
        # Store only a representative complete picture, not all decoded videos.
        from tools.g_integration_report import canvas
        with np.load(prior/'legacy/main/pixels.npz',allow_pickle=False) as z: legacy=z['reconstruction'].copy()
        canvas([('Original',source),('Legacy / old regions',legacy),('Candidate / old regions',old),
                ('Candidate / GT-assisted '+best,outputs[best])],8,width=768).save(root/(sid+'.png'))
        point = dict(sample_id=sid,cohort=job['cohort'],dataset=job['row']['dataset'],
            actual_bytes=job['point']['actual_bytes'],marginal_frames=FRAMES,single_region_gains=gains,
            choices=policies,quality=scores,best_finite_choice=best,
            legacy_quality=reference['metrics']['legacy']['quality'],
            gpu_cpu_lpips_difference=old_quality['lpips_alex']-cpu_score,
            exact_prior_render=True,offline_uses_GT=True,not_exhaustive_optimum=True)
        atomic_json(target,point); results.append(point)
        run.update(phase='case_scored',completed=len(results),total=len(cfg['cases']),sample=sid)
        print(json.dumps(point),flush=True)
    atomic_json(root/'summary.json',dict(cases=results,no_training=True,no_promotion=True))


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch()
    if args.worker: return generate(args)
    root=args.output
    if (root/'complete.json').exists():
        prepare(root); verify_artifacts(root,read(root/'complete.json')['artifacts']); return
    run=CleanRun(SimpleNamespace(output=root,command='selection_probe',max_hours=1))
    run.thread.start(); run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            cfg=prepare(root); dest=root/'generation'
            if not (dest/'complete.json').exists():
                if not (dest/'intentional_stop.json').exists():
                    execute(run,'partial_generation','tools.g_selection_probe',
                            ['--worker','--output',dest,'--stop-after','8'])
                preserved={str(p.relative_to(root)):digest(p) for p in dest.glob('*/g*/complete.json')}
                execute(run,'resume_generation','tools.g_selection_probe',['--worker','--output',dest])
                verify_artifacts(root,preserved)
                atomic_json(root/'restart.json',dict(preserved=preserved,exact=True))
            verify_artifacts(dest,read(dest/'complete.json')['artifacts']); score(run,root,cfg)
            files=[root/'protocol.json',root/'summary.json',root/'restart.json',dest/'complete.json']
            files+=list((root/'scores').glob('*.json'))+list(root.glob('*.png'))
            atomic_json(root/'complete.json',dict(complete=True,utc=now(),no_training=True,
                offline_GT_diagnostic=True,artifacts={str(p.relative_to(root)):digest(p) for p in files}))
            run.update(phase='complete_review_required')
    except BaseException as error:
        atomic_json(root/'last_failure.json',dict(utc=now(),error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--worker',action='store_true');p.add_argument('--stop-after',type=int,default=0)
    main(p.parse_args())
