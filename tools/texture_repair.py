"""Resumable paired G-only repair queue; never promotes the experiment model."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, digest
from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_json
from tools.fusion_finish import memory_snapshot
from routervc.generation.texture_data import DEFAULT, protocol


class RepairRun(Run):
    def log_resources(self):
        super().log_resources()
        with (self.root/'memory.jsonl').open('a') as log:
            log.write(json.dumps(dict(unix_seconds=time.time(), **memory_snapshot()))+'\n')


def execute(run, name, module, options):
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=1',
               '-m',module,*map(str,options)]
    run.update(phase=name)
    with (run.root/(name+'.log')).open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                run.check(); time.sleep(.5)
            if child.returncode: raise RuntimeError(f'{name} failed ({child.returncode}); inspect its log')
        finally:
            if child.poll() is None:
                # Send directly to worker + torchrun; trainer atomically saves at a step boundary.
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def fit(run, root, arm, dest, stop=0):
    execute(run, str(dest.relative_to(run.root)).replace('/','_')+f'_to{stop}',
            'routervc.generation.texture_training',
            ['--protocol',root/'protocol.json','--output',dest,'--arm',arm,'--stop-after',stop])


def verify_pair(root):
    rows = {}
    for arm in ('control','repair'):
        dest = root/arm
        done = read(dest/'complete.json')
        if digest(dest/'adapter.pt') != done['adapter_sha256'] or not done['complete']:
            raise ValueError('completed checkpoint changed')
        rows[arm] = [json.loads(s) for s in (dest/'steps.jsonl').read_text().splitlines()]
    if len(rows['control']) != len(rows['repair']): raise ValueError('unequal arm lengths')
    for a,b in zip(rows['control'],rows['repair']):
        for key in ('step','sample','dataset','crop','E_cap','condition','noise','learning_rate'):
            if a[key] != b[key]: raise ValueError('paired inputs differ: '+key)
    result = dict(complete=True, steps=len(rows['control']), identical_conditions_noise_schedule=True,
                  production_model_changed=False)
    atomic_json(root/'paired_audit.json',result)
    return result


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run = RepairRun(SimpleNamespace(output=args.output,command='repair_queue',max_hours=12))
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            smoke = args.output/'smoke_v2'; protocol(smoke,6,True)
            for arm in ('control','repair'):
                direct, resumed = smoke/'direct'/arm, smoke/'resumed'/arm
                fit(run,smoke,arm,direct)
                if not (resumed/'resume.pt').exists(): fit(run,smoke,arm,resumed,3)
                fit(run,smoke,arm,resumed)
                a = torch.load(direct/'resume.pt',weights_only=True,map_location='cpu')
                b = torch.load(resumed/'resume.pt',weights_only=True,map_location='cpu')
                exact(a,b)  # Raises on the first differing optimizer, model or RNG value.
            verify_pair(smoke/'direct'); verify_pair(smoke/'resumed')
            atomic_json(smoke/'complete.json',dict(complete=True,exact_resume_both_arms=True,steps=6))
            if args.smoke_only: return
            formal = args.output/'formal'; protocol(formal,args.steps,False)
            for arm in ('control','repair'): fit(run,formal,arm,formal/arm)
            verify_pair(formal)
            run.update(phase='paired_training_complete_evaluation_pending')
            atomic_json(args.output/'training_complete.json',dict(complete=True,steps_per_arm=args.steps,
                paired_audit=digest(formal/'paired_audit.json'),no_automatic_promotion=True))
            execute(run,'paired_evaluation','tools.texture_repair_evaluate',
                    ['--root',args.output,'--output',args.output/'evaluation'])
            run.update(phase='complete_no_model_promotion')
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    if args.steps < 1: p.error('positive training steps required')
    main(args)
