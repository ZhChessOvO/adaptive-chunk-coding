"""One-factor paired G continuation after measured objective diagnostics.

Reuse the immutable mixed-view trainer with two explicit protocols. Only the
RGB temporal weight differs; neither the periodic term nor other assets change.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now
from routervc.generation.texture_data import REPO
from tools.g_objective_diagnostic import DEFAULT as DIAGNOSTIC
from tools.texture_context_probe import DEFAULT as CONTEXT
from tools.texture_clean import CleanRun
from tools.texture_repair import execute

DEFAULT=DIAGNOSTIC/'trial'
ARMS=('control','relaxed_rgb_temporal')
CODE=('tools/g_objective_trial.py','tools/g_objective_trial_evaluate.py','tools/run_g_objective_trial.sh',
      'tools/context_training_report.py')


def recipe(old, arm, smoke=False):
    if arm not in ARMS:raise ValueError('unknown trial arm')
    if old['loss']['rgb_temporal']!=.1 or old['phase_weight']!=1.:
        raise ValueError('unexpected starting objective recipe')
    config=copy.deepcopy(old)
    config.update(version=2,variant=arm,steps=6 if smoke else 400,seed=2026101503,
        learning_rate=dict(peak=5e-6,end=2e-6,warmup=6 if smoke else 40),
        arms=['mixed'],scope='same mixed-view recipe, only RGB temporal coefficient differs',
        production_changed=False)
    config['loss']['rgb_temporal']=.1 if arm=='control' else .025
    config['phase_weight']=1.
    if smoke:config['rows']=[next(r for r in config['rows'] if r['dataset']==d) for d in ('REDS','UVG')]
    return config


def protocol(stage,arm,smoke=False):
    old=read(CONTEXT/'training/formal/protocol.json')
    diagnosis=read(DIAGNOSTIC/'complete.json');verify_artifacts(DIAGNOSTIC,diagnosis['artifacts'])
    initial=CONTEXT/'training/formal/mixed/adapter.pt'
    prior=read(initial.parent/'complete.json')
    if not prior['complete'] or digest(initial)!=prior['adapter_sha256']:raise ValueError('initial checkpoint changed')
    config=recipe(old,arm,smoke)
    config['initial']=dict(path=str(initial),sha256=digest(initial))
    config['diagnostic_complete']=digest(DIAGNOSTIC/'complete.json')
    config['code'].update({n:digest(REPO/n) for n in CODE})
    for n,s in config['code'].items():
        if digest(REPO/n)!=s:raise ValueError('historical source changed: '+n)
    immutable(stage/arm/'protocol.json',config)
    return config


def fit(run,stage,arm,dest,stop=0):
    name=str(dest.relative_to(run.root)).replace('/','_')+f'_to{stop}'
    execute(run,name,'routervc.generation.context_training',
        ['--protocol',stage/arm/'protocol.json','--output',dest,'--arm','mixed','--stop-after',stop])


def verify_cursor(stage,arm,path,steps):
    saved=torch.load(path/'resume.pt',weights_only=True,map_location='cpu')
    if saved['step']!=steps or saved['binding']!=dict(protocol=digest(stage/arm/'protocol.json'),arm='mixed'):
        raise ValueError('wrong cursor/binding')
    state=saved['adapter']['state_dict']
    if len(state)!=82 or not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('invalid LoRA state')
    states=saved['optimizer']['state']
    if len(states)!=82 or not all(int(s['step'])==steps for s in states.values()):raise ValueError('wrong optimizer step')
    if not all(torch.isfinite(v).all() for s in states.values() for v in s.values() if isinstance(v,torch.Tensor)):
        raise ValueError('nonfinite optimizer')
    if not all(k in saved for k in ('rng_cpu','rng_cuda','rng_python')):raise ValueError('missing RNG')
    initial=read(stage/arm/'protocol.json')['initial']
    old=torch.load(initial['path'],weights_only=True,map_location='cpu')['state_dict']
    changed=sum(not torch.equal(v,old[k]) for k,v in state.items())
    if changed!=82:raise ValueError('expected every LoRA tensor updated')
    records=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
    if [r['step'] for r in records]!=list(range(1,steps+1)):raise ValueError('missing journal step')
    return records


def paired_audit(stage,paths,steps):
    records={a:verify_cursor(stage,a,paths[a],steps) for a in ARMS}
    for left,right in zip(*(records[a] for a in ARMS)):
        for key in ('step','sample','dataset','E_cap','crop','paired_small','paired_noise','condition','noise','learning_rate'):
            if left[key]!=right[key]:raise ValueError('unpaired field: '+key)
    if steps==400:
        for values in records.values():
            if Counter(r['dataset'] for r in values)!={'REDS':300,'UVG':100}:raise ValueError('dataset schedule changed')
            if Counter(r['crop'][2] for r in values)!={256:100,512:300}:raise ValueError('mixed-view schedule changed')
    return dict(complete=True,steps=steps,identical_samples_crops_conditions_noise_learning_rate=True,
                only_coefficient_changed='rgb_temporal: 0.1 vs 0.025',phase_weight_unchanged=1.)


def verify(root):
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    for stage_name,steps,sub in (('smoke',6,'direct'),('formal',400,'')):
        stage=root/stage_name
        for a in ARMS:
            cfg=read(stage/a/'protocol.json')
            for n,s in cfg['code'].items():
                if digest(REPO/n)!=s:raise ValueError('bound trial source changed: '+n)
            for spec in (*cfg['assets'].values(),cfg['initial']):
                if digest(spec['path'])!=spec['sha256']:raise ValueError('asset changed')
            path=stage/a/sub
            saved=torch.load(path/'resume.pt',weights_only=True,map_location='cpu')
            if stage_name=='smoke':
                exact(saved,torch.load(stage/a/'resumed/resume.pt',weights_only=True,map_location='cpu'))
            receipt=read(path/'complete.json')
            if digest(path/'adapter.pt')!=receipt['adapter_sha256']:raise ValueError('training receipt changed')
            exact(saved['adapter'],torch.load(path/'adapter.pt',weights_only=True,map_location='cpu'))
            exact(saved['adapter'],torch.load(path/'checkpoints'/f'adapter_{steps:06d}.pt',weights_only=True,map_location='cpu'))
        paired_audit(stage,{a:stage/a/sub for a in ARMS},steps)
    from tools.g_objective_trial_evaluate import verify as verify_eval
    verify_eval(root/'smoke/evaluation');verify_eval(root/'evaluation')
    return done


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    root=args.output
    if args.verify_only or (root/'complete.json').exists():
        print(json.dumps(verify(root)),flush=True);return
    run=CleanRun(SimpleNamespace(output=root,command='objective_trial',max_hours=8))
    run.thread.start();run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            smoke=root/'smoke'
            for arm in ARMS:protocol(smoke,arm,True)
            if not (smoke/'complete.json').exists():
                for arm in ARMS:
                    direct,resumed=smoke/arm/'direct',smoke/arm/'resumed'
                    fit(run,smoke,arm,direct)
                    if not (resumed/'resume.pt').exists():fit(run,smoke,arm,resumed,3)
                    fit(run,smoke,arm,resumed)
                    exact(torch.load(direct/'resume.pt',weights_only=True,map_location='cpu'),
                          torch.load(resumed/'resume.pt',weights_only=True,map_location='cpu'))
                paired_audit(smoke,{a:smoke/a/'direct' for a in ARMS},6)
                execute(run,'smoke_review','tools.g_objective_trial_evaluate',['--root',root,'--smoke'])
                names=[f'{a}/{m}/{n}' for a in ARMS for m in ('direct','resumed')
                       for n in ('resume.pt','adapter.pt','steps.jsonl','complete.json')]+['evaluation/complete.json']
                atomic_json(smoke/'complete.json',dict(complete=True,exact_resume_both=True,
                    actual_reload_repeat_checked=True,artifacts={n:digest(smoke/n) for n in names}))
            verify_artifacts(smoke,read(smoke/'complete.json')['artifacts'])
            if args.smoke_only:return
            formal=root/'formal'
            for arm in ARMS:protocol(formal,arm)
            for arm in reversed(ARMS):
                if not (root/'formal_started.json').exists():
                    fit(run,formal,arm,formal/arm,25)
                    verify_cursor(formal,arm,formal/arm,25)
                    immutable(root/'formal_started.json',dict(utc=now(),arm=arm,verified_step=25,
                        steps_per_arm=400,checkpoint_sha256=digest(formal/arm/'resume.pt'),
                        exact_smoke=digest(smoke/'complete.json')))
                fit(run,formal,arm,formal/arm)
            atomic_json(formal/'paired_audit.json',paired_audit(formal,{a:formal/a for a in ARMS},400))
            atomic_json(root/'training_complete.json',dict(complete=True,steps_per_arm=400,production_changed=False))
            execute(run,'final_review','tools.g_objective_trial_evaluate',['--root',root])
            names=['smoke/complete.json','formal/paired_audit.json','training_complete.json','evaluation/complete.json']
            names += [f'formal/{a}/{n}' for a in ARMS for n in ('protocol.json','resume.pt','adapter.pt','steps.jsonl','complete.json')]
            atomic_json(root/'complete.json',dict(complete=True,production_changed=False,
                artifacts={n:digest(root/n) for n in names},utc=now()))
            verify(root);run.update(phase='complete_review_required')
    except BaseException as error:
        atomic_json(root/'last_failure.json',dict(utc=now(),error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--verify-only',action='store_true');main(p.parse_args())
