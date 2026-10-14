"""Fresh LoRA: exact restart smoke, paired 2000-step fits, automatic reviews."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now
from routervc.generation.texture_clean import (
    ARMS, DEFAULT, MILESTONES, initialize, protocol, verify_bindings)
from tools.texture_ablation import PAIRED_KEYS, verify_pair
from tools.texture_repair import RepairRun, execute


class CleanRun(RepairRun):
    def check(self):
        super().check()
        if time.monotonic() - getattr(self, '_last_size_check', 0) >= 30:
            self._last_size_check = time.monotonic()
            size = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            if size >= 8*1024**3:
                raise InterruptedError('8 GiB experiment output guard; checkpoint and review storage')


def fit(run, stage, arm, dest, stop=0):
    execute(run, str(dest.relative_to(run.root)).replace('/', '_') + f'_to{stop}',
        'routervc.generation.texture_ablation',
        ['--protocol', stage/'protocol.json', '--output', dest, '--arm', arm, '--stop-after', stop])


def partial_audit(stage, step):
    records = {}
    for arm in ARMS:
        folder = stage / arm
        saved = torch.load(folder/'resume.pt', map_location='cpu', weights_only=True)
        if saved['step'] != step or saved['binding'] != dict(protocol=digest(stage/'protocol.json'), arm=arm):
            raise ValueError('incorrect optimization cursor/binding')
        state = saved['adapter']['state_dict']
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError('nonfinite adapter')
        if not any(bool(t.any()) for k,t in state.items() if k.endswith('lora_b')):
            raise ValueError('fresh LoRA did not update')
        if not saved['optimizer']['state'] or not all(
            int(s['step']) == step for s in saved['optimizer']['state'].values()):
            raise ValueError('optimizer steps differ')
        records[arm] = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        if [r['step'] for r in records[arm]] != list(range(1,step+1)):
            raise ValueError('non-contiguous journal')
    for left,right in zip(records[ARMS[0]], records[ARMS[1]]):
        for key in PAIRED_KEYS:
            if left[key] != right[key]:
                raise ValueError('unpaired training: ' + key)
    return dict(complete=True, steps=step, exact_pairing=True,
                journals={a:digest(stage/a/'steps.jsonl') for a in ARMS})


def evaluate(run, stage, output, step, smoke=False):
    options = ['--stage',stage,'--output',output,'--step',step]
    if smoke: options.append('--smoke')
    execute(run, f'{"smoke" if smoke else "formal"}_review_{step:06d}',
            'tools.texture_clean_evaluate', options)


def verify_completed(root):
    from tools.texture_clean_evaluate import verify_stage
    done = read(root/'complete.json')
    verify_artifacts(root, done['artifacts'])
    config = read(root/'formal/protocol.json'); verify_bindings(config)
    if verify_pair({a:root/'formal'/a for a in ARMS}) != read(root/'formal/paired_audit.json'):
        raise ValueError('paired audit changed')
    initialize(root)
    for step in (0, *MILESTONES): verify_stage(root/'evaluation', step)
    verify_stage(root/'smoke/evaluation', 6)
    return done


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if args.verify_only or (args.output/'complete.json').exists():
        print(json.dumps(verify_completed(args.output)), flush=True); return
    run = CleanRun(SimpleNamespace(output=args.output, command='clean_queue', max_hours=12))
    run.thread.start(); run.log_resources()
    try:
        with exclusive_native_evaluation(run):
            initial = initialize(args.output)
            smoke = args.output/'smoke'; protocol(smoke,initial,True)
            if not (smoke/'complete.json').exists():
                for arm in ARMS:
                    direct, resumed = smoke/'direct'/arm, smoke/'resumed'/arm
                    fit(run,smoke,arm,direct)
                    if not (resumed/'resume.pt').exists(): fit(run,smoke,arm,resumed,3)
                    fit(run,smoke,arm,resumed)
                    exact(torch.load(direct/'resume.pt',map_location='cpu',weights_only=True),
                          torch.load(resumed/'resume.pt',map_location='cpu',weights_only=True))
                verify_pair({a:smoke/'direct'/a for a in ARMS})
                evaluate(run,smoke,smoke/'evaluation',6,True)
                names = ['evaluation/stages/step_000006/complete.json'] + [
                    f'{mode}/{arm}/{name}' for mode in ('direct','resumed') for arm in ARMS
                    for name in ('complete.json','resume.pt','adapter.pt','steps.jsonl')]
                atomic_json(smoke/'complete.json',dict(complete=True, exact_restart_both_arms=True,
                    zero_initial_output_and_reload_checked=True, steps=6,
                    artifacts={n:digest(smoke/n) for n in names}))
            verify_artifacts(smoke,read(smoke/'complete.json')['artifacts'])
            if args.smoke_only: return
            formal = args.output/'formal'; protocol(formal,initial)
            evaluate(run,formal,args.output/'evaluation',0)
            for step in MILESTONES:
                # Resume from the same optimizer/RNG with a fixed 2000-step LR horizon.
                # Phase arm first so the primary repair candidate starts immediately.
                audit = formal/f'paired_{step:06d}.json'
                if not audit.exists():
                    for arm in reversed(ARMS):
                        fit(run,formal,arm,formal/arm,step)
                        if not (args.output/'formal_started.json').exists():
                            saved = torch.load(formal/arm/'resume.pt',map_location='cpu',weights_only=True)
                            if saved['step'] < 1: raise ValueError('no formal optimizer update')
                            immutable(args.output/'formal_started.json',dict(utc=now(), arm=arm,
                                observed_updates=saved['step'], planned_steps_per_arm=2000,
                                checkpoint_sha256=digest(formal/arm/'resume.pt')))
                    immutable(audit,partial_audit(formal,step))
                evaluate(run,formal,args.output/'evaluation',step)
            atomic_json(formal/'paired_audit.json',verify_pair({a:formal/a for a in ARMS}))
            atomic_json(args.output/'training_complete.json',dict(complete=True,
                steps_per_arm=2000, fresh_initialization=True, production_model_changed=False))
            from tools.texture_ablation_detail import resource_figure
            import matplotlib
            matplotlib.use('Agg')
            resource = resource_figure(args.output,args.output)
            atomic_json(args.output/'resources.json',resource)
            names = ['initial/complete.json','formal/protocol.json','formal/paired_audit.json',
                'smoke/complete.json','training_complete.json','resources.json','resources.png']
            names += [f'evaluation/stages/step_{s:06d}/complete.json' for s in (0,*MILESTONES)]
            atomic_json(args.output/'complete.json',dict(complete=True,production_model_changed=False,
                artifacts={n:digest(args.output/n) for n in names}))
            verify_completed(args.output)
            run.update(phase='complete_review_required_no_model_promotion')
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(utc=now(),error=repr(error),progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    main(parser.parse_args())
