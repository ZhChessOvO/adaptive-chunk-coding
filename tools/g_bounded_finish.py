"""Single bounded same-recipe G fine-tune, with 400/800 frozen-chain reviews.

No Router/UF/fusion fitting or automatic model promotion. The user-specified
server window includes backup; an absolute deadline survives queue restarts.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from demo.chunk_enhancement_codec import atomic_torch, configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now, quality
from demo.routervc_policy import grid_rois
from routervc.generation.integration import compatible_bundle
from routervc.generation.texture_data import REPO
from routervc.latent import generation
from tools.g_integration import DEFAULT as CHAIN, fresh, check_protocol
from tools.g_integration_report import canvas
from tools.g_objective_trial import verify_cursor
from tools.g_selection_probe import DEFAULT as PROBE, FastMetric
from tools.cooperative_review import source_rgb
from tools.texture_clean import CleanRun
from tools.texture_repair import execute
from tools.texture_repair_review import temporal_phase

DEFAULT=Path('/root/autodl-fs/DCVC/runs/routervc_g_finish_20261016')
PRIOR=Path('/root/autodl-fs/DCVC/runs/routervc_g_objectives_20261015/trial/formal/control')
CODE=('tools/g_bounded_finish.py','tools/run_g_bounded_finish.sh','tools/test_g_bounded_finish.py',
      'tools/g_integration_report.py','tools/g_selection_probe.py')


def deadline_seconds(value):
    stamp=datetime.fromisoformat(value)
    if stamp.tzinfo is None: raise ValueError('explicit timezone required')
    return stamp.timestamp()


def recipe(old, initial, smoke=False):
    cfg=copy.deepcopy(old)
    if cfg['loss']['rgb_temporal']!=.1 or cfg['phase_weight']!=1.:
        raise ValueError('unexpected accepted objective')
    cfg.update(variant='bounded_same_recipe',steps=6 if smoke else 800,seed=2026101601,
        initial=initial,arms=['mixed'],learning_rate=dict(peak=5e-6,end=2e-6,warmup=6 if smoke else 40),
        scope='one G-only same-objective warm-start round; fresh AdamW; no sweep or Router fit',
        production_changed=False)
    if smoke: cfg['rows']=[next(r for r in cfg['rows'] if r['dataset']==d) for d in ('REDS','UVG')]
    return cfg


def needs_fit(completed,total,stop,finished):
    # A stopped trainer does not itself skip a stop_after cursor already reached.
    # Re-enter a final cursor without complete.json to finalize its adapter.
    return not finished and (not stop or stop>=total or completed<stop)


class BoundedRun(CleanRun):
    def __init__(self,args,deadline):
        self.deadline=deadline_seconds(deadline)
        super().__init__(args)

    def check(self):
        super().check()
        if datetime.now(timezone.utc).timestamp()>=self.deadline:
            raise InterruptedError('absolute backup-reserving deadline reached; save and stop')


def prepare(root,deadline):
    if (root/'protocol.json').exists():
        cfg=read(root/'protocol.json')
        if cfg['deadline_utc']!=deadline: raise ValueError('deadline is bound; do not silently extend')
        for name,sha in cfg['code'].items():
            if digest(REPO/name)!=sha: raise ValueError('bound source changed: '+name)
        for path,sha in cfg['inputs'].items():
            if digest(path)!=sha: raise ValueError('bound input changed: '+path)
        return cfg
    done=read(PROBE/'complete.json');verify_artifacts(PROBE,done['artifacts'])
    verify_artifacts(PROBE/'generation',read(PROBE/'generation/complete.json')['artifacts'])
    chain=check_protocol(CHAIN)
    initial=dict(path=str(PRIOR/'adapter.pt'),sha256=digest(PRIOR/'adapter.pt'))
    if initial['sha256']!=read(PRIOR/'complete.json')['adapter_sha256']: raise ValueError('initial changed')
    old=read(PRIOR/'protocol.json')
    code={**old['code'],**chain['code'],**read(PROBE/'protocol.json')['code'],
          **{n:digest(REPO/n) for n in CODE}}
    for name,sha in code.items():
        if digest(REPO/name)!=sha: raise ValueError('historical code changed')
    for stage in ('smoke','formal'):
        cfg=recipe(old,initial,stage=='smoke');cfg['code']=code
        immutable(root/stage/'continuation/protocol.json',cfg)
    cfg=dict(created=now(),deadline_utc=deadline,code=code,initial=initial,chain=chain,
        inputs={str(p):digest(p) for p in (PROBE/'complete.json',PROBE/'summary.json',
            CHAIN/'protocol.json',PRIOR/'protocol.json',PRIOR/'adapter.pt')},
        steps=800,milestones=[400,800],same_objective=True,no_router_training=True,no_promotion=True)
    immutable(root/'protocol.json',cfg); return cfg


def fit(run,root,stage,dest,stop=0):
    cfg=read(root/stage/'continuation/protocol.json')
    cursor=0
    if (dest/'resume.pt').exists():
        saved=torch.load(dest/'resume.pt',map_location='cpu',weights_only=True)
        if saved['binding']!=dict(protocol=digest(root/stage/'continuation/protocol.json'),arm='mixed'):
            raise ValueError('resume protocol changed')
        cursor=saved['step']
    if not needs_fit(cursor,cfg['steps'],stop,(dest/'complete.json').exists()):return
    execute(run,stage+'_'+dest.name+'_to'+str(stop),'routervc.generation.context_training',
        ['--protocol',root/stage/'continuation/protocol.json','--output',dest,'--arm','mixed','--stop-after',stop])


def evaluate(run,root,cfg,step):
    stage=root/'review'/f'step_{step:06d}'
    if (stage/'complete.json').exists():
        verify_artifacts(stage,read(stage/'complete.json')['artifacts']); return
    source=root/'formal/continuation/checkpoints'/f'adapter_{step:06d}.pt'
    candidate=torch.load(source,map_location='cpu',weights_only=True)
    legacy=torch.load(cfg['chain']['models']['legacy']['path'],map_location='cpu',weights_only=True)
    exported=stage/'model/adapter.pt'; bundle=compatible_bundle(candidate,legacy)
    if not exported.exists(): atomic_torch(exported,bundle)
    exact(torch.load(exported,map_location='cpu',weights_only=True),bundle)
    model=dict(path=str(exported),sha256=digest(exported),original_path=str(source),
               original_sha256=digest(source),assets=generation.assets(exported))
    arm=f'continuation{step}'; protocol=dict(teacher=cfg['chain']['teacher'],models={arm:model})
    immutable(stage/'protocol.json',dict(step=step,model=model,queue_protocol=digest(root/'protocol.json')))
    destinations=[]
    for j,job in enumerate(cfg['chain']['jobs']):
        run.check(); dest=fresh(run,stage,protocol,job,arm); destinations.append(dest)
        if step==400 and j<2:
            repeat=fresh(run,stage,protocol,job,arm,'repeat')
            if read(repeat/'complete.json')['output_hash']!=read(dest/'complete.json')['output_hash']:
                raise ValueError('actual model reload differs')
    metric=FastMetric(); points=[]
    for job,dest in zip(cfg['chain']['jobs'],destinations,strict=True):
        run.check(); sid=job['row']['sample_id']; target=stage/'scores'/f'{sid}.json'
        if target.exists(): points.append(read(target)); continue
        source_rgb_array=source_rgb(job['row'])
        with np.load(dest/'pixels.npz',allow_pickle=False) as z:
            pixels=z['reconstruction'].copy(); received=z['enhanced'].copy()
        before=CHAIN/'samples'/sid
        for prior_arm in ('control','legacy'):
            prior_dest=before/prior_arm/'main'
            verify_artifacts(prior_dest,read(prior_dest/'complete.json')['artifacts'])
        with np.load(before/'control/main/pixels.npz',allow_pickle=False) as z: starting=z['reconstruction'].copy()
        with np.load(before/'legacy/main/pixels.npz',allow_pickle=False) as z: legacy_rgb=z['reconstruction'].copy()
        prior=read(before/'scores.json')
        calibration=metric.evaluate(source_rgb_array,starting)-prior['metrics']['control']['quality']['lpips_alex']
        if abs(calibration)>1e-5: raise ValueError('CPU/GPU metric calibration differs')
        phase=[]
        for i in job['receipt']['generated']:
            x,y,w,h=grid_rois(*pixels.shape[1:3])[i]
            phase.append(temporal_phase(pixels[:,y:y+h,x:x+w],source_rgb_array[:,y:y+h,x:x+w]))
        report=read(dest/'complete.json')
        point=dict(sample_id=sid,cohort=job['cohort'],dataset=job['row']['dataset'],step=step,
            actual_bytes=report['actual_bytes'],bpp=report['bpp'],quality=quality(source_rgb_array,pixels,metric),
            G_core_phase={k:float(np.mean([r[k] for r in phase])) for k in phase[0] if k!='period'},
            references={a:prior['metrics'][a] for a in ('legacy','control','received')},
            resources={k:report[k] for k in ('seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')},
            gpu_cpu_calibration=calibration,regions=report['generated'])
        panels=[('Original',source_rgb_array),('Received Y',received),('Legacy production',legacy_rgb),
                ('Starting repaired G',starting),(f'Same-recipe +{step}',pixels)]
        canvas(panels,8,width=768).save(stage/(sid+'.png'))
        atomic_json(target,point);points.append(point)
        run.update(phase='review',step=step,sample=sid,completed=len(points),total=6)
    del metric;torch.cuda.empty_cache()
    atomic_json(stage/'summary.json',dict(step=step,cases=points,no_promotion=True))
    files=[stage/'protocol.json',stage/'summary.json',exported]
    files+=list((stage/'scores').glob('*.json'))+list(stage.glob('*.png'))
    for dest in destinations:
        files += [p for p in dest.rglob('*') if p.is_file() and p.suffix in ('.npz','.json','.rvlg')]
    atomic_json(stage/'complete.json',dict(complete=True,step=step,no_promotion=True,
        artifacts={str(p.relative_to(stage)):digest(p) for p in files}))


def verify(root):
    cfg=read(root/'protocol.json');prepare(root,cfg['deadline_utc'])
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    verify_cursor(root/'formal','continuation',root/'formal/continuation',800)
    for step in cfg['milestones']:
        stage=root/'review'/f'step_{step:06d}';verify_artifacts(stage,read(stage/'complete.json')['artifacts'])
    return done


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch()
    root=args.output
    if args.verify_only or (root/'complete.json').exists(): print(json.dumps(verify(root)));return
    # Check before writing protocols or loading any model.
    if datetime.now(timezone.utc).timestamp()>=deadline_seconds(args.deadline_utc):
        raise RuntimeError('deadline already passed; no automatic extension')
    run=BoundedRun(SimpleNamespace(output=root,command='bounded_finish',max_hours=4),args.deadline_utc)
    run.thread.start();run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            cfg=prepare(root,args.deadline_utc);run.check()
            smoke=root/'smoke/continuation';direct=smoke/'direct';resumed=smoke/'resumed'
            if not (root/'smoke/complete.json').exists():
                fit(run,root,'smoke',direct)
                if not (resumed/'resume.pt').exists():fit(run,root,'smoke',resumed,3)
                fit(run,root,'smoke',resumed)
                exact(torch.load(direct/'resume.pt',map_location='cpu',weights_only=True),
                      torch.load(resumed/'resume.pt',map_location='cpu',weights_only=True))
                for d in (direct,resumed):verify_cursor(root/'smoke','continuation',d,6)
                names=[f'continuation/{m}/{n}' for m in ('direct','resumed')
                       for n in ('resume.pt','adapter.pt','steps.jsonl','complete.json')]
                atomic_json(root/'smoke/complete.json',dict(exact_restart=True,
                    artifacts={n:digest(root/'smoke'/n) for n in names}))
            verify_artifacts(root/'smoke',read(root/'smoke/complete.json')['artifacts'])
            dest=root/'formal/continuation'
            if not (root/'formal_started.json').exists():
                fit(run,root,'formal',dest,25);verify_cursor(root/'formal','continuation',dest,25)
                immutable(root/'formal_started.json',dict(utc=now(),verified_step=25,finite_updated_tensors=82,
                    optimizer_states=82,planned_steps=800,checkpoint_sha256=digest(dest/'resume.pt'),
                    deadline_utc=args.deadline_utc,smoke=digest(root/'smoke/complete.json')))
            for step in cfg['milestones']:
                stage=root/'review'/f'step_{step:06d}'
                if not (stage/'complete.json').exists():
                    fit(run,root,'formal',dest,step);verify_cursor(root/'formal','continuation',dest,step)
                    evaluate(run,root,cfg,step)
            atomic_json(root/'summary.json',dict(complete=True,steps=800,
                training=read(dest/'complete.json'),initial=cfg['initial'],
                reviews=[read(root/'review'/f'step_{s:06d}/summary.json') for s in cfg['milestones']],
                no_model_promotion=True,no_more_training_queued=True,deadline_utc=args.deadline_utc))
            files=[root/'protocol.json',root/'summary.json',root/'smoke/complete.json',root/'formal_started.json']
            files += [dest/n for n in ('protocol.json','adapter.pt','resume.pt','steps.jsonl','complete.json')]
            files += [root/'review'/f'step_{s:06d}/complete.json' for s in cfg['milestones']]
            atomic_json(root/'complete.json',dict(complete=True,utc=now(),steps=800,no_promotion=True,
                artifacts={str(p.relative_to(root)):digest(p) for p in files}))
            verify(root);run.update(phase='complete_review_required_no_more_training')
    except BaseException as error:
        atomic_json(root/'last_failure.json',dict(utc=now(),error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--deadline-utc',default='2026-10-16T08:30:00+00:00')
    p.add_argument('--verify-only',action='store_true');main(p.parse_args())
