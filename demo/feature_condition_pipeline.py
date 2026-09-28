"""Resumable feature cache, correctness smoke and paired training in tmux."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.scalable_codec import atomic_json, file_hash

DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_feature_condition_20260928')


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).strip()
            if pids:
                raise RuntimeError(f'GPU busy: {pids}')
            if args.command == 'train':
                smoke = DEFAULT.with_name(DEFAULT.name+'_smoke')
                if not read(smoke/'resume_check.json')['exact'] or not read(smoke/'decode_check.json')['complete']:
                    raise RuntimeError('correctness smoke required before formal training')
            if args.command != 'decode-smoke':
                execute(run,'cache','feature_condition_cache.py',
                    ['--output',args.output, *(['--smoke'] if args.command == 'smoke' else [])])
            if args.command == 'cache':
                return
            def train(name, mode, steps, stop=-1):
                execute(run,name,'feature_condition_train.py',
                    ['--cache',args.output/'cache.json','--output',args.output/name.split('_attempt')[0],
                     '--mode',mode,'--steps',steps,'--stop-after',stop],distributed=True)
            if args.command == 'smoke':
                train('feature_resume_attempt1','feature',6,3)
                train('feature_resume_attempt2','feature',6)
                train('feature_direct','feature',6)
                import torch
                a = torch.load(args.output/'feature_resume/resume.pt',weights_only=True,map_location='cpu')
                b = torch.load(args.output/'feature_direct/resume.pt',weights_only=True,map_location='cpu')
                for section in ('state_dict','feature_state'):
                    for key,value in a['adapter'][section].items():
                        torch.testing.assert_close(value,b['adapter'][section][key],rtol=0,atol=0)
                for key,value in a['optimizer']['state'].items():
                    for name,tensor in value.items():
                        torch.testing.assert_close(tensor,b['optimizer']['state'][key][name],rtol=0,atol=0)
                train('rgb_smoke','rgb',3)
                atomic_json(args.output/'resume_check.json',dict(exact=True,
                    lora_exact=True,feature_exact=True,optimizer_exact=True,steps=6))
            if args.command in ('smoke','decode-smoke'):
                from demo.feature_condition_smoke import decode_smoke
                decode_smoke(args.output,run)
            elif args.command == 'train':
                train('feature','feature',args.steps)
                train('rgb','rgb',args.steps)
            run.update(phase='complete')
            atomic_json(args.output/f'{args.command}.complete.json',dict(command=args.command,
                code=file_hash(Path(__file__)),elapsed_seconds=time.monotonic()-run.started))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['smoke','decode-smoke','cache','train'])
    p.add_argument('--output',type=Path)
    p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--steps',type=int,default=1000)
    args = p.parse_args()
    if args.output is None:
        args.output = DEFAULT.with_name(DEFAULT.name+'_smoke') if args.command in ('smoke','decode-smoke') else DEFAULT
    main(args)
