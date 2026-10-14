"""Frozen G field-of-view comparison: same core, native scale, paired noise.

No fitting, codec changes or source-conditioned inference. Historical ROIs
are replayed first; new 256/512 comparisons use a spatial master-noise field.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from demo.chunk_enhancement_codec import configure_torch
from demo.condition_path_decode import tensor_identity
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from demo.scalable_experiment import quality, now
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_seedvr2_lora_utils import load_lora_state_dict, lora_state_dict
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.generation.texture_clean import DEFAULT as CLEAN, verify_bindings
from routervc.generation.texture_data import REPO, CURRENT
from routervc.generation.context_geometry import geometry, rgb_crop, local_core, master_noise, noise_crop
from routervc.latent.router_data import source as load_source
from tools.texture_diagnostic import run_one, phase_diagnostic
from tools.texture_repair_review import temporal_phase
from tools.texture_clean import CleanRun

DEFAULT = CLEAN.parent/'routervc_texture_context_20261015'
CODE = ('routervc/generation/context_geometry.py', 'tools/texture_context_probe.py',
        'tools/texture_context_report.py', 'tools/texture_context.py',
        'tools/run_texture_context.sh')


def make_protocol(root):
    previous = read(CLEAN/'formal/protocol.json'); verify_bindings(previous)
    old_eval = CLEAN/'evaluation/stages/step_002000'
    verify_artifacts(old_eval, read(old_eval/'complete.json')['artifacts'])
    cases = read(CLEAN/'evaluation/protocol.json')['cases']
    paths = dict(legacy=CURRENT, clean=CLEAN/'initial/adapter.pt',
                 phase2000=CLEAN/'formal/phase_only/adapter.pt')
    result = dict(version=1, scope='frozen raw G, not RD or full-video benchmark', cases=cases,
        models={n:dict(path=str(p),sha256=digest(p)) for n,p in paths.items()},
        assets=previous['assets'], prior_code=previous['code'],
        code={n:digest(REPO/n) for n in CODE}, previous_summary=str(old_eval/'summary.json'),
        previous_summary_sha256=digest(old_eval/'summary.json'),
        seed=2026101501, second_seed_known_cases_only=True,
        comparison='same central 128 square, native 256 vs 512 contexts; no resizing',
        noise='CPU BF16 master [5,64,64,16], coordinate crops; VAE conditions may change with context',
        roles='3 known failures + 12 previously used internal validation; developmental diagnostic',
        promotion=False, training=False)
    immutable(root/'protocol.json', result)
    return result


def full_inputs(case):
    folder = Path(case['folder']); path = folder/'received.npz'
    if digest(path) != case['received_sha256']: raise ValueError('received input changed')
    with np.load(path) as z: received = z['enhanced'].copy()
    if case.get('source_path'):
        if digest(case['source_path']) != case['source_sha256']: raise ValueError('source changed')
        with np.load(case['source_path']) as z: source = z['source'].copy()
    else: source = load_source(case)
    if source.shape != received.shape or source.shape[0] != 17:
        raise ValueError('expected paired 17-frame full inputs')
    return source, received


def frozen_restore(model, received, noise, seed):
    """Inject only independently generated noise, never source pixels."""
    runner = model.runner; runner.config.vae.use_sample = False
    original = runner.get_condition
    observation = {}
    expected = tensor_identity(noise)
    def get_condition(latent, latent_blur, task):
        if latent.shape != noise.shape: raise ValueError('unexpected latent geometry')
        latent.copy_(noise.to(device=latent.device,dtype=latent.dtype))
        observation.update(noise=tensor_identity(latent), condition=tensor_identity(latent_blur))
        if observation['noise'] != expected: raise ValueError('noise injection differs')
        return original(latent=latent,latent_blur=latent_blur,task=task)
    with patch.object(runner,'get_condition',get_condition), torch.no_grad():
        pixels,runtime = model.restore(list(received),seed=seed)
    return np.stack(pixels), dict(runtime=runtime, **observation)


def verify(root):
    config = read(root/'protocol.json')
    for name,value in {**config['prior_code'],**config['code']}.items():
        if digest(REPO/name) != value: raise ValueError('bound code changed: '+name)
    for spec in (*config['models'].values(), *config['assets'].values()):
        if digest(spec['path']) != spec['sha256']: raise ValueError('asset changed')
    done = read(root/'complete.json'); verify_artifacts(root,done['artifacts'])
    summary = read(root/'summary.json')
    for item in summary['cases']:
        for row in item['results']:
            dest = root/row['relative_dir']; verify_artifacts(dest,row['artifacts'])
            if read(dest/'result.json') != row: raise ValueError('point journal mismatch')
    if done['new_points'] != sum(len(i['results']) for i in summary['cases']):
        raise ValueError('point count mismatch')
    return done


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if (args.output/'complete.json').exists():
        print(json.dumps(verify(args.output)),flush=True);return
    configure_torch(); torch.use_deterministic_algorithms(True)
    run = CleanRun(SimpleNamespace(output=args.output,command='context_probe',max_hours=4))
    run.thread.start(); run.log_resources()
    try:
        config = make_protocol(args.output)
        old_items = {i['case']['sample_id']:i for i in read(config['previous_summary'])['cases']}
        # Generate location overviews before loading the model.
        from tools.texture_context_report import overview, comparisons, summary_figure
        for case in config['cases']:
            source,received = full_inputs(case)
            dest = args.output/case['sample_id']; dest.mkdir(exist_ok=True)
            geo = geometry(source.shape,case['core'])
            if not (dest/'overview.png').exists(): overview(dest,case,source,received,geo)
        assets = {k:Path(v['path']) for k,v in config['assets'].items()}
        model = PersistentSeedVR2(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
            dit_checkpoint=assets['dit'],vae_checkpoint=assets['vae'],positive_embedding=assets['positive'],
            negative_embedding=assets['negative'],lora_checkpoint=Path(config['models']['legacy']['path']),
            lora_strength=1.,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
        states = {n:torch.load(v['path'],map_location='cpu',weights_only=True)['state_dict']
                  for n,v in config['models'].items()}
        metric = LPIPSAlex(True); all_items = []; completed = 0
        for case_index,case in enumerate(config['cases']):
            run.check(); source,received = full_inputs(case); geo = geometry(source.shape,case['core'])
            folder = args.output/case['sample_id']; results = []
            replay_path = folder/'historical_replay.json'
            if case_index in (0,2) and not replay_path.exists():
                old_case = case
                values,_,observed = run_one(model,rgb_crop(received,old_case['crop']),old_case,
                    ('legacy','legacy',1.,'all'),states)
                old = next(r for r in old_items[case['sample_id']]['results'] if r['name']=='legacy')
                with np.load(old['output_path']) as z: np.testing.assert_array_equal(values,z['pixels'])
                for k in ('condition','noise'):
                    if observed[k] != old['observation'][k]: raise ValueError('historical pairing changed')
                immutable(replay_path,dict(complete=True,pixels_condition_noise_exact=True,
                    previous_output=digest(old['output_path'])))
            seeds = [config['seed']+case_index*1000033]
            if case['evaluation_group']=='known_artifact': seeds += [seeds[0]+7907]
            expected_conditions = {}
            for seed_index,seed in enumerate(seeds):
                master = master_noise(seed)
                for name,state in states.items():
                    load_lora_state_dict(model.runner.dit,state)
                    for size,label in ((256,'small'),(512,'large')):
                        run.check(); crop = geo[label]; noise = noise_crop(master,crop,geo['large'])
                        rel = f'{case["sample_id"]}/seed{seed_index}/{name}_{size}'
                        dest = args.output/rel; dest.mkdir(parents=True,exist_ok=True)
                        binding = dict(protocol=digest(args.output/'protocol.json'),sample=case['sample_id'],
                            seed=seed,model=name,crop=crop,core=geo['core'])
                        if (dest/'result.json').exists():
                            row = read(dest/'result.json'); verify_artifacts(dest,row['artifacts'])
                            if row['binding'] != binding: raise ValueError('point binding changed')
                        else:
                            began = time.monotonic()
                            pixels,observation = frozen_restore(model,rgb_crop(received,crop),noise,seed)
                            target = rgb_crop(source,geo['core']); output = local_core(pixels,crop,geo['core'])
                            q = quality(target,output,metric)
                            phase,_ = phase_diagnostic(output,target)
                            temporal = temporal_phase(output,target)
                            extra = None
                            if case_index==0 and name=='phase2000' and seed_index==0:
                                again,obs = frozen_restore(model,rgb_crop(received,crop),noise,seed)
                                np.testing.assert_array_equal(pixels,again)
                                if obs['noise'] != observation['noise']: raise ValueError('repeat noise differs')
                                extra = 'exact pixels and noise repeated'
                            atomic_npz(dest/'output.npz',pixels=pixels)
                            row = dict(binding=binding,relative_dir=rel,name=name,size=size,seed_index=seed_index,
                                quality=q,phase_error=phase,temporal_phase=temporal,observation=observation,
                                seconds=time.monotonic()-began,repeat=extra,
                                artifacts={'output.npz':digest(dest/'output.npz')})
                            atomic_json(dest/'result.json',row)
                        key = (seed_index,size)
                        pair = {k:row['observation'][k] for k in ('noise','condition')}
                        if key in expected_conditions and pair != expected_conditions[key]:
                            raise ValueError('same-FOV model comparison has different conditions/noise')
                        expected_conditions[key]=pair
                        if row['observation']['noise'] != tensor_identity(noise):
                            raise ValueError('cross-FOV master noise pairing failed')
                        results.append(row); completed += 1
                        run.update(phase='frozen_evaluation',completed=completed,total=108,
                            sample=case['sample_id'],model=name,size=size)
                        print(json.dumps(dict(completed=completed,sample=case['sample_id'],model=name,
                            size=size,quality=row['quality'],phase=row['phase_error'])),flush=True)
            for k,v in lora_state_dict(model.runner.dit).items():
                # Inference wrapper is BF16; compare to its initial cast, not training FP32.
                torch.testing.assert_close(v,states['phase2000'][k].to(v.dtype),rtol=0,atol=0)
            comparisons(folder,case,source,received,geo,results,args.output)
            all_items.append(dict(case=case,geometry=geo,results=results))
            atomic_json(args.output/'partial_summary.json',dict(cases=all_items))
        summary = dict(cases=all_items,scope=config['scope'],paired_noise=True)
        atomic_json(args.output/'summary.json',summary)
        aggregate = summary_figure(summary,args.output)
        atomic_json(args.output/'aggregate.json',aggregate)
        names = ['protocol.json','summary.json','aggregate.json','summary.png']
        names += [str(p.relative_to(args.output)) for p in args.output.rglob('*.png')
                  if p.name!='summary.png']
        names += [str(p.relative_to(args.output)) for p in args.output.rglob('historical_replay.json')]
        atomic_json(args.output/'complete.json',dict(complete=True,new_points=completed,
            frozen_models=True,exact_historical_replays=2,exact_repeats=2,
            artifacts={n:digest(args.output/n) for n in names},utc=now()))
        verify(args.output); run.update(phase='complete_no_training')
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),utc=now(),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join();run.lock.close()
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT/'probe')
    main(p.parse_args())
