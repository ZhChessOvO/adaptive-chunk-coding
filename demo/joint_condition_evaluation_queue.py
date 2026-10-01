"""Exclusive, resumable evaluation/report queue; never trains any model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.joint_condition_train import DEFAULT
from demo.scalable_codec import atomic_json


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            if args.command != 'report':
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                                     '--format=csv,noheader'], text=True).strip()
                from demo.joint_condition_evaluate import evaluate
                evaluate(run.root, run)
            if args.command != 'evaluate':
                from demo.joint_condition_report import report
                report(run.root)
            run.update(phase='complete')
            target = run.root/f'{args.command}.complete.json'
            if not target.exists():
                atomic_json(target, dict(complete=True, elapsed_seconds=time.monotonic()-run.started))
    except BaseException as e:
        atomic_json(run.root/'evaluation.last_failure.json', dict(error=repr(e), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run','evaluate','report'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--max-hours', type=float, default=6.)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        parser.error('Run inside tmux')
    main(args)
