"""Bounded 2x2 loss ablation, authenticated old arms, exact-resume new arms."""
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
from routervc.generation.texture_data import DEFAULT as PREVIOUS, REPO, RUNS
from tools.texture_repair import RepairRun, execute

DEFAULT = RUNS/'routervc_texture_loss_ablation_20261014'
NEW_ARMS = ('augmentation_only', 'phase_only')
CODE = ('routervc/generation/texture_ablation.py', 'tools/texture_ablation.py',
        'tools/texture_ablation_evaluate.py', 'tools/texture_ablation_report.py',
        'tools/texture_repair_evaluate.py', 'tools/texture_repair_review.py',
        'tools/texture_diagnostic.py', 'tools/run_texture_ablation.sh')
PAIRED_KEYS = ('step', 'sample', 'dataset', 'crop', 'E_cap', 'condition', 'noise', 'learning_rate')


def make_protocol(root, smoke=False):
    original = PREVIOUS/('smoke_v2' if smoke else 'formal')/'protocol.json'
    config = copy.deepcopy(read(original))
    for name, value in config['code'].items():
        if digest(REPO/name) != value: raise ValueError('historical source changed: '+name)
    if not read(PREVIOUS/'training_complete.json')['complete']:
        raise ValueError('historical training incomplete')
    config.update(format='routervc_g_texture_loss_ablation_v1', arms=list(NEW_ARMS),
        reference_root=str(PREVIOUS), reference_protocol=digest(original),
        reference_evaluation=digest(PREVIOUS/'evaluation/complete.json'),
        factors={'control':[False,False], 'repair':[True,True],
                 'augmentation_only':[True,False], 'phase_only':[False,True]},
        compatibility='replay both old 6-step arms exactly before reusing old 400-step arms')
    config['code'].update({name:digest(REPO/name) for name in CODE})
    immutable(root/'protocol.json', config)
    return config


def fit(run, root, arm, dest, stop=0):
    execute(run, str(dest.relative_to(run.root)).replace('/','_')+f'_to{stop}',
        'routervc.generation.texture_ablation',
        ['--protocol',root/'protocol.json','--output',dest,'--arm',arm,'--stop-after',stop])


def numeric_state(value):
    # Experiment provenance intentionally differs, all numeric training state must not.
    return {**{k:value[k] for k in ('step','optimizer','rng_cpu','rng_cuda','rng_python')},
            'parameters':value['adapter']['state_dict']}


def rows(folder):
    done = read(folder/'complete.json')
    if not done['complete'] or digest(folder/'adapter.pt') != done['adapter_sha256']:
        raise ValueError('completed adapter changed: '+str(folder))
    result = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    if len(result) != done['steps'] or [r['step'] for r in result] != list(range(1,done['steps']+1)):
        raise ValueError('incomplete/non-contiguous step journal')
    return result


def verify_pair(folders):
    records = {name:rows(folder) for name,folder in folders.items()}
    baseline = next(iter(records.values()))
    for name, records_arm in records.items():
        if len(records_arm) != len(baseline): raise ValueError('arm lengths differ')
        for a,b in zip(baseline,records_arm):
            for key in PAIRED_KEYS:
                if a[key] != b[key]: raise ValueError(f'{name}: paired {key} differs')
    return dict(complete=True, arms=list(folders), steps=len(baseline),
                identical_conditions_noise_schedule=True,
                journals={name:digest(folder/'steps.jsonl') for name,folder in folders.items()})


def verify_completed(root):
    done = read(root/'complete.json'); verify_artifacts(root,done['artifacts'])
    protocol = read(root/'formal/protocol.json')
    for name,value in protocol['code'].items():
        if digest(REPO/name) != value: raise ValueError('bound source changed: '+name)
    if digest(protocol['initial']) != protocol['initial_sha256']:
        raise ValueError('production LoRA changed')
    folders={a:PREVIOUS/'formal'/a for a in ('control','repair')}
    folders.update({a:root/'formal'/a for a in NEW_ARMS})
    if verify_pair(folders) != read(root/'formal/paired_audit.json'):
        raise ValueError('paired audit differs')
    from tools.texture_ablation_evaluate import verify_evaluation
    verify_evaluation(root/'evaluation')
    return done


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if args.verify_only or (args.output/'complete.json').exists():
        print(json.dumps(verify_completed(args.output)),flush=True);return
    run = RepairRun(SimpleNamespace(output=args.output,command='ablation_queue',max_hours=6))
    run.thread.start();run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            smoke=args.output/'smoke';make_protocol(smoke,True)
            for arm in ('control','repair'):
                dest=smoke/'compatibility'/arm;fit(run,smoke,arm,dest)
                old=PREVIOUS/'smoke_v2/direct'/arm
                a=torch.load(dest/'resume.pt',weights_only=True,map_location='cpu')
                b=torch.load(old/'resume.pt',weights_only=True,map_location='cpu')
                exact(numeric_state(a),numeric_state(b))
                paired=verify_pair({'new':dest,'historical':old})
                for x,y in zip(rows(dest),rows(old)):
                    exact({k:x[k] for k in ('loss','terms','phase_error','transform','gradient_norm')},
                          {k:y[k] for k in ('loss','terms','phase_error','transform','gradient_norm')})
                atomic_json(dest/'compatibility.json',dict(paired,exact_parameters_optimizer_rng=True))
            for arm in NEW_ARMS:
                direct,resumed=smoke/'direct'/arm,smoke/'resumed'/arm
                fit(run,smoke,arm,direct)
                if not (resumed/'resume.pt').exists(): fit(run,smoke,arm,resumed,3)
                fit(run,smoke,arm,resumed)
                exact(torch.load(direct/'resume.pt',weights_only=True,map_location='cpu'),
                      torch.load(resumed/'resume.pt',weights_only=True,map_location='cpu'))
            verify_pair({a:smoke/'direct'/a for a in NEW_ARMS})
            atomic_json(smoke/'complete.json',dict(complete=True,exact_resume_new_arms=True,
                exact_historical_control_and_repair_replay=True,steps=6))
            if args.smoke_only:return
            formal=args.output/'formal';make_protocol(formal)
            for arm in NEW_ARMS:fit(run,formal,arm,formal/arm)
            folders={a:PREVIOUS/'formal'/a for a in ('control','repair')}
            folders.update({a:formal/a for a in NEW_ARMS})
            atomic_json(formal/'paired_audit.json',verify_pair(folders))
            atomic_json(args.output/'training_complete.json',dict(complete=True,steps_per_arm=400,
                new_arms=list(NEW_ARMS),reused_arms=['control','repair'],no_model_promotion=True))
            execute(run,'evaluation','tools.texture_ablation_evaluate',
                ['--root',args.output,'--output',args.output/'evaluation'])
            names=['smoke/complete.json','formal/paired_audit.json','training_complete.json',
                   'formal/protocol.json','evaluation/complete.json']
            atomic_json(args.output/'complete.json',dict(complete=True,production_model_changed=False,
                artifacts={n:digest(args.output/n) for n in names}))
            verify_completed(args.output)
            run.update(phase='complete_no_model_promotion')
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--verify-only',action='store_true')
    main(p.parse_args())
