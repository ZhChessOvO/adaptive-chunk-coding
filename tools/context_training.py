"""Paired short G view adaptation, smoke/restart first and automatic final review."""
import argparse
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
from routervc.generation.texture_clean import DEFAULT as CLEAN, verify_bindings
from routervc.generation.texture_data import REPO
from routervc.generation.context_training import ARMS
from tools.texture_clean import CleanRun
from tools.texture_context_probe import DEFAULT, verify as verify_probe
from tools.texture_repair import execute

CODE=('routervc/generation/context_training.py','tools/context_training.py',
      'tools/context_training_evaluate.py','tools/run_context_training.sh')


def protocol(folder,smoke=False):
    old=read(CLEAN/'formal/protocol.json');verify_bindings(old)
    probe=read(DEFAULT/'probe/protocol.json')
    initial=CLEAN/'formal/phase_only/adapter.pt'
    rows=copy.deepcopy(old['rows'])
    if smoke:rows=[next(r for r in rows if r['dataset']==d) for d in ('REDS','UVG')]
    result=dict(version=1,rows=rows,seed=2026101502,steps=6 if smoke else 400,
        initial=dict(path=str(initial),sha256=digest(initial)),assets=old['assets'],
        code={**old['code'],**probe['code'],**{n:digest(REPO/n) for n in CODE}},
        probe_complete=digest(DEFAULT/'probe/complete.json'),loss=old['loss'],phase_weight=1.,
        learning_rate=dict(peak=5e-6,end=2e-6,warmup=6 if smoke else 40),
        arms=list(ARMS),sampler='paired samples, original small crops, global spatial noise; REDS:UVG 3:1',
        local='256 input / 128 supervised interior',mixed='75 percent 512 input / 384 interior; 25 percent original local',
        scope='input AND supervision field-of-view recipe; not a pure context-only intervention',
        memory='unsliced differentiable VAE, saved tensors CPU offload; identical in both arms',
        frozen=['base_DiT','VAE','UF','B/E','R_s','R_g','fusion'],promotion=False)
    immutable(folder/'protocol.json',result);return result


def fit(run,stage,arm,dest,stop=0):
    name=str(dest.relative_to(run.root)).replace('/','_')+f'_to{stop}'
    execute(run,name,'routervc.generation.context_training',
        ['--protocol',stage/'protocol.json','--output',dest,'--arm',arm,'--stop-after',stop])


def paired_audit(stage,paths,expected):
    records={}
    for arm,path in paths.items():
        saved=torch.load(path/'resume.pt',map_location='cpu',weights_only=True)
        if saved['step']!=expected or saved['binding']!=dict(protocol=digest(stage/'protocol.json'),arm=arm):
            raise ValueError('incorrect formal cursor/binding')
        if not all(torch.isfinite(t).all() for t in saved['adapter']['state_dict'].values()):raise ValueError('invalid weights')
        if len(saved['optimizer']['state'])!=82 or not all(int(s['step'])==expected for s in saved['optimizer']['state'].values()):
            raise ValueError('optimizer not updated')
        records[arm]=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
        if [r['step'] for r in records[arm]]!=list(range(1,expected+1)):raise ValueError('incomplete journal')
    for a,b in zip(records['local'],records['mixed']):
        for key in ('step','sample','dataset','E_cap','paired_small','paired_noise','learning_rate'):
            if a[key]!=b[key]:raise ValueError('pair differs: '+key)
        if a['crop']==b['crop'] and (a['condition']!=b['condition'] or a['noise']!=b['noise']):
            raise ValueError('same-FOV conditions differ')
    return dict(complete=True,steps=expected,paired_samples_and_overlap_noise=True)


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    root=DEFAULT/'training'
    run=CleanRun(SimpleNamespace(output=root,command='context_training_queue',max_hours=12))
    run.thread.start();run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            verify_probe(DEFAULT/'probe')
            smoke=root/'smoke';protocol(smoke,True)
            if not (smoke/'complete.json').exists():
                for arm in ARMS:
                    direct,resumed=smoke/'direct'/arm,smoke/'resumed'/arm
                    fit(run,smoke,arm,direct)
                    if not (resumed/'resume.pt').exists():fit(run,smoke,arm,resumed,3)
                    fit(run,smoke,arm,resumed)
                    exact(torch.load(direct/'resume.pt',map_location='cpu',weights_only=True),
                          torch.load(resumed/'resume.pt',map_location='cpu',weights_only=True))
                paired_audit(smoke,{a:smoke/'direct'/a for a in ARMS},6)
                execute(run,'smoke_evaluation','tools.context_training_evaluate',['--root',root,'--smoke'])
                names=[f'{mode}/{arm}/{n}' for mode in ('direct','resumed') for arm in ARMS
                       for n in ('adapter.pt','resume.pt','steps.jsonl','complete.json')]
                names+=['evaluation/complete.json']
                atomic_json(smoke/'complete.json',dict(complete=True,exact_restart_both_arms=True,
                    actual_wide_image_gradients=True,artifacts={n:digest(smoke/n) for n in names}))
            verify_artifacts(smoke,read(smoke/'complete.json')['artifacts'])
            if args.smoke_only:return
            formal=root/'formal';cfg=protocol(formal)
            for arm in ('mixed','local'):
                if arm=='mixed' and not (root/'formal_started.json').exists():
                    fit(run,formal,arm,formal/arm,25)
                    saved=torch.load(formal/arm/'resume.pt',map_location='cpu',weights_only=True)
                    initial=torch.load(cfg['initial']['path'],map_location='cpu',weights_only=True)['state_dict']
                    if saved['step']<25 or not any(not torch.equal(v,initial[k]) for k,v in saved['adapter']['state_dict'].items()):
                        raise ValueError('no genuine formal updates')
                    if len(saved['optimizer']['state'])!=82 or not all(int(s['step'])==saved['step'] for s in saved['optimizer']['state'].values()):
                        raise ValueError('formal optimizer missing')
                    immutable(root/'formal_started.json',dict(utc=now(),arm=arm,steps=saved['step'],planned_steps_per_arm=400,
                        checkpoint_sha256=digest(formal/arm/'resume.pt'),exact_smoke=digest(smoke/'complete.json')))
                fit(run,formal,arm,formal/arm)
            atomic_json(formal/'paired_audit.json',paired_audit(formal,{a:formal/a for a in ARMS},400))
            atomic_json(root/'training_complete.json',dict(complete=True,steps_per_arm=400,production_changed=False))
            execute(run,'final_evaluation','tools.context_training_evaluate',['--root',root])
            run.update(phase='complete_review_required')
            names=['smoke/complete.json','formal/protocol.json','formal/paired_audit.json','training_complete.json','evaluation/complete.json']
            atomic_json(root/'complete.json',dict(complete=True,artifacts={n:digest(root/n) for n in names},production_changed=False))
    except BaseException as e:
        atomic_json(root/'last_failure.json',dict(error=repr(e),utc=now(),progress=run.progress));raise
    finally:run.log_resources();run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke-only',action='store_true');main(p.parse_args())
