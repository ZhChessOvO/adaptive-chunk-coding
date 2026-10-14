"""Milestone raw-G evaluations; exact zero-init/reload checks and native pixels.

Executed by the tmux parent while it holds the global GPU lock. Source RGB
is used only by the evaluator, never passed to the SeedVR2 inference call.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from demo.scalable_experiment import quality
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.generation.texture_clean import ARMS, MILESTONES, REPO, verify_bindings
from tools.texture_repair import RepairRun
from tools.texture_repair_evaluate import aggregate
from tools.texture_diagnostic import core, run_one, phase_diagnostic
from tools.texture_repair_review import temporal_phase
from tools.texture_clean_report import draw, trajectory


def model_paths(stage, config, step, smoke):
    result = {'clean':Path(config['initial'])}
    steps = [6] if smoke else [s for s in MILESTONES if s <= step]
    for s in steps:
        for arm in ARMS:
            folder = stage/'direct'/arm if smoke else stage/arm
            result[f'{arm}_{s:06d}'] = folder/'checkpoints'/f'adapter_{s:06d}.pt'
    return result


def verify_stage(output, step):
    root = output/'stages'/f'step_{step:06d}'
    done = read(root/'complete.json'); verify_artifacts(root,done['artifacts'])
    config = read(root/'protocol.json')
    if digest(output/'protocol.json') != config['evaluation_protocol']:
        raise ValueError('evaluation protocol changed')
    for model in config['models'].values():
        if digest(model['path']) != model['sha256']:
            raise ValueError('evaluation model changed')
    for item in read(root/'summary.json')['cases']:
        if digest(item['reference_inputs']) != item['reference_inputs_sha256']:
            raise ValueError('evaluation source/received changed')
        for row in item['results']:
            if digest(row['output_path']) != row['artifacts']['output.npz']:
                raise ValueError('evaluation output changed')
            if row['name'] != 'legacy' and read(Path(row['output_path']).parent/'result.json') != row:
                raise ValueError('point journal differs from summary')
    if done['logical_points'] != done['new_candidate_points']+done['historical_reference_points']:
        raise ValueError('point counts differ')
    return done


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    config = read(args.stage/'protocol.json'); verify_bindings(config)
    root = args.output/'stages'/f'step_{args.step:06d}'
    if (root/'complete.json').exists():
        print(json.dumps(verify_stage(args.output,args.step)),flush=True); return
    run = RepairRun(SimpleNamespace(output=args.output,command='evaluate',max_hours=3))
    run.thread.start(); run.log_resources()
    model = metric = None
    try:
        old = Path(config['reference_evaluation'])
        if digest(old/'complete.json') != config['reference_evaluation_sha256']:
            raise ValueError('previous receipt changed')
        verify_artifacts(old,read(old/'complete.json')['artifacts'])
        old_items = read(old/'summary.json')['cases']
        if args.smoke:
            old_items = [next(i for i in old_items if i['case']['dataset']==d
                         and i['case']['evaluation_group']=='known_artifact') for d in ('REDS','UVG')]
        chosen = [i['case'] for i in old_items]
        base = dict(version=1,cases=chosen,training_protocol=digest(args.stage/'protocol.json'),
            old_evaluation_receipt=config['reference_evaluation_sha256'],smoke=args.smoke,
            scope='fixed local raw G before fusion; no RD or independent-test claim',
            code={k:v for k,v in config['code'].items()})
        immutable(args.output/'protocol.json',base)
        models = model_paths(args.stage,config,args.step,args.smoke)
        stage_protocol = dict(step=args.step,evaluation_protocol=digest(args.output/'protocol.json'),
            models={n:dict(path=str(p),sha256=digest(p)) for n,p in models.items()})
        immutable(root/'protocol.json',stage_protocol)
        states = {n:torch.load(p,map_location='cpu',weights_only=True)['state_dict'] for n,p in models.items()}
        assets = {k:Path(v['path']) for k,v in config['assets'].items()}
        all_results = []; temporal = []; artifacts = {}
        for old_item in old_items:
            case = old_item['case']; sid = case['sample_id']
            inputs = old/sid/'inputs.npz'
            with np.load(inputs) as z: source=z['source'].copy(); received=z['received'].copy()
            original = next(r for r in old_item['results'] if r['name']=='initial')
            old_dest = old/sid/'initial'; verify_artifacts(old_dest,original['artifacts'])
            if read(old_dest/'result.json') != original: raise ValueError('old reference differs')
            legacy = dict(original,name='legacy',output_path=str(old_dest/'output.npz'),
                          provenance='authenticated_old_LoRA_reference_not_paired_training')
            expected = {k:legacy['observation'][k] for k in ('condition','noise')}
            rows = [legacy]
            for name,path in models.items():
                run.check(); dest=args.output/'points'/sid/name; dest.mkdir(parents=True,exist_ok=True)
                binding = dict(evaluation_protocol=digest(args.output/'protocol.json'),
                    model_sha256=digest(path), sample=sid, name=name)
                if (dest/'result.json').exists():
                    row = read(dest/'result.json'); verify_artifacts(dest,row['artifacts'])
                    if row['binding'] != binding: raise ValueError('point binding changed')
                else:
                    if model is None:
                        model = PersistentSeedVR2(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
                            dit_checkpoint=assets['dit'],vae_checkpoint=assets['vae'],
                            positive_embedding=assets['positive'],negative_embedding=assets['negative'],
                            lora_checkpoint=Path(config['initial']),lora_strength=1.,
                            sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
                        metric = LPIPSAlex(True)
                    began = time.monotonic()
                    pixels,_,observation = run_one(model,received,case,(name,name,1.,'all'),states)
                    q = quality(core(source,case),core(pixels,case),metric)
                    phase,folded = phase_diagnostic(core(pixels,case))
                    error,_ = phase_diagnostic(core(pixels,case),core(source,case))
                    error8,_ = phase_diagnostic(core(pixels,case),core(source,case),period=8)
                    temporal_info = temporal_phase(core(pixels,case),core(source,case))
                    checks = {}
                    if args.smoke and name == 'clean':
                        bypass,_,observed = run_one(model,received,case,('bypass','clean',0.,'bypass'),states)
                        np.testing.assert_array_equal(pixels,bypass)
                        if any(observed[k]!=observation[k] for k in expected):
                            raise ValueError('zero-init original pairing differs')
                        checks['zero_init_equals_original_bypass'] = True
                    if args.smoke and name.startswith('phase_only_'):
                        # Swap away, then reload the candidate to prove inference state is not leaked.
                        run_one(model,received,case,('swap','clean',1.,'all'),states)
                        again,_,observed = run_one(model,received,case,(name,name,1.,'all'),states)
                        np.testing.assert_array_equal(pixels,again)
                        if any(observed[k]!=observation[k] for k in expected):
                            raise ValueError('reload pairing differs')
                        checks['candidate_reload_exact'] = True
                    atomic_npz(dest/'output.npz',pixels=pixels,phase16=folded)
                    row = dict(name=name,binding=binding,quality=q,phase16=phase,
                        error_phase16=error,error_phase8=error8,temporal_phase=temporal_info,
                        observation=observation,checks=checks,seconds=time.monotonic()-began,
                        output_path=str(dest/'output.npz'),provenance='new_decode',
                        artifacts={'output.npz':digest(dest/'output.npz')})
                    atomic_json(dest/'result.json',row)
                if {k:row['observation'][k] for k in expected} != expected:
                    raise ValueError('condition/noise not paired with original reference')
                rows.append(row)
                run.update(phase='paired_review',step=args.step,sample=sid,model=name)
                print(json.dumps(dict(sample=sid,model=name,quality=row['quality'])),flush=True)
            for row in rows:
                if 'temporal_phase' in row: phase = row['temporal_phase']
                else:
                    with np.load(row['output_path']) as z: pixels = core(z['pixels'],case)
                    phase = temporal_phase(pixels,core(source,case))
                if not np.isclose(phase['stationary_rms'],row['error_phase16']['phase_rms_u8'],rtol=1e-5):
                    raise ValueError('per-frame phase audit inconsistent')
                temporal.append(dict(sample=sid,name=row['name'],dataset=case['dataset'],
                                     group=case['evaluation_group'],**phase))
            folder = root/sid; folder.mkdir(exist_ok=True)
            names = draw(folder,case,source,received,{r['name']:r for r in rows},args.step)
            artifacts.update({sid+'/'+n:digest(folder/n) for n in names})
            all_results.append(dict(case=case,results=rows,paired_conditions_noise=True,
                reference_inputs=str(inputs),reference_inputs_sha256=digest(inputs)))
        summary = dict(cases=all_results,aggregate=aggregate(all_results),scope=base['scope'],
                       step=args.step,production_model_changed=False)
        atomic_json(root/'summary.json',summary)
        atomic_json(root/'temporal_audit.json',dict(complete=True,rows=temporal))
        trajectory(summary,root)
        artifacts.update({n:digest(root/n) for n in ('protocol.json','summary.json',
                                                   'temporal_audit.json','trajectory.png')})
        atomic_json(root/'complete.json',dict(complete=True,step=args.step,
            logical_points=len(chosen)*(len(models)+1),new_candidate_points=len(chosen)*len(models),
            historical_reference_points=len(chosen),counts_are_cumulative_within_stage=True,
            seconds_this_invocation=time.monotonic()-run.started,artifacts=artifacts))
        verify_stage(args.output,args.step); run.update(phase='complete'); run.log_resources()
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
        run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args())
