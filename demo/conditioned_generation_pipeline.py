"""Exclusive, resumable single-GPU process queue; run inside tmux."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.scalable_codec import atomic_json, file_hash

DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_conditioned_generation_20260928')


def execute(run, name, script, argv, *, distributed=False):
    run.update(phase=name)
    command = [sys.executable]
    if distributed:
        command += ['-m','torch.distributed.run','--standalone','--nproc-per-node=1']
    command += [str(REPO/'demo'/script), *map(str,argv)]
    path = run.root/f'{name}.log'
    with path.open('a') as log:
        log.write('\n'+json.dumps(dict(command=command, attempt_start=time.time()))+'\n'); log.flush()
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                run.check()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; see {path}')
        except BaseException:
            if child.poll() is None:
                # Signal the worker directly first, allowing an atomic training
                # checkpoint before torchrun tears down its process group.
                for p in Path('/proc').iterdir():
                    if p.name.isdigit():
                        try:
                            if os.getpgid(int(p.name)) == child.pid and int(p.name) != child.pid:
                                os.kill(int(p.name), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError):
                            pass
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        # Hold the same native-evaluation mutex for the complete queue,
        # INCLUDING training, so another cooperating evaluator cannot overlap.
        with exclusive_native_evaluation(run):
            pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'],text=True).strip()
            if pids:
                raise RuntimeError(f'GPU busy before starting queue: {pids}')
            if args.command == 'train':
                smoke = DEFAULT.with_name(DEFAULT.name+'_smoke')
                if (not read(smoke/'resume_check.json')['adapter_exact'] or
                        not read(smoke/'evaluation/summary.json')['old_adapter_receiver_exact']):
                    raise RuntimeError('gradient/resume/fresh-reload smoke required first')
            for stage in ('encode','receive','latents'):
                extra = ['--smoke'] if args.command == 'smoke' else []
                execute(run, stage, 'conditioned_generation_cache.py',
                    [stage,'--output',args.output,*extra], distributed=stage == 'latents')
            def train(name, mode, steps, stop=-1):
                execute(run,name,'conditioned_generation_train.py',
                    ['--cache',args.output/'cache.json','--output',args.output/name.split('_attempt')[0],
                     '--mode',mode,'--steps',steps,'--stop-after',stop],distributed=True)
            if args.command == 'smoke':
                train('image_resume_attempt1','image',6,3)
                train('image_resume_attempt2','image',6)
                train('image_direct','image',6)
                import torch
                a = torch.load(args.output/'image_resume/resume.pt',weights_only=True,map_location='cpu')
                b = torch.load(args.output/'image_direct/resume.pt',weights_only=True,map_location='cpu')
                for key, value in a['adapter']['state_dict'].items():
                    torch.testing.assert_close(value,b['adapter']['state_dict'][key],rtol=0,atol=0)
                for key, value in a['optimizer']['state'].items():
                    for name, tensor in value.items():
                        torch.testing.assert_close(tensor,b['optimizer']['state'][key][name],rtol=0,atol=0)
                train('latent_smoke','latent',3)
                atomic_json(args.output/'resume_check.json',dict(adapter_exact=True,optimizer_exact=True,
                    first_stop=3,final_step=6))
            else:
                train('latent','latent',args.steps)
                train('image','image',args.steps)
            run.update(phase='training_complete',completed=args.steps if args.command != 'smoke' else 6)
            atomic_json(args.output/'pipeline.complete.json',dict(command=args.command,steps=args.steps,
                code=file_hash(Path(__file__)),elapsed_seconds=time.monotonic()-run.started))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['smoke','train'])
    p.add_argument('--output',type=Path)
    p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--steps',type=int,default=1000)
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command == 'smoke' else DEFAULT
    main(args)
