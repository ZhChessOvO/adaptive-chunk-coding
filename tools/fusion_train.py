"""Wait for fixed-policy fusion data, then train only the tiny display fusion F."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_fullview_probe import save
from tools.latent_boundary_report import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'p1_data')
    p.add_argument('--output', type=Path, default=ROOT/'p1_fit')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--stop-after', type=int, default=0)
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run = Run(SimpleNamespace(output=args.output, command='fit', max_hours=48)); run.thread.start()
    try:
        while not (args.data/'complete.json').exists():
            if (args.data/'last_failure.json').exists(): raise RuntimeError('data preparation needs attention')
            run.check(); run.update(phase='waiting_for_frozen_policy_data'); time.sleep(5)
        with exclusive_native_evaluation(run):
            # Load the recipe only when work starts, not hours earlier in the
            # waiting launcher. Checkpoint hashes must describe executed code.
            from routervc.fusion.training import fit
            configure_torch()
            if not (args.output/'resume_verified.json').exists():
                import gc
                import torch
                from demo.routervc_mixed_queue import exact
                from demo.routervc_fullview_probe import digest
                resumed, direct = args.output/'resume_check/resumed', args.output/'resume_check/direct'
                if not (resumed/'resume.pt').exists():
                    fit(resumed, args.data, run, epochs=args.epochs, stop_after=2)
                gc.collect(); torch.cuda.empty_cache()
                fit(resumed, args.data, run, epochs=args.epochs, stop_after=4)
                gc.collect(); torch.cuda.empty_cache()
                fit(direct, args.data, run, epochs=args.epochs, stop_after=4)
                a = torch.load(resumed/'resume.pt', map_location='cpu', weights_only=True)
                b = torch.load(direct/'resume.pt', map_location='cpu', weights_only=True)
                exact(a, b)
                save(args.output/'resume_verified.json', dict(complete=True, steps=4,
                    model_optimizer_rng_exact=True, resumed=digest(resumed/'resume.pt'), direct=digest(direct/'resume.pt')))
                del a, b
                gc.collect(); torch.cuda.empty_cache()
            fit(args.output, args.data, run, epochs=args.epochs, stop_after=args.stop_after)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally: run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__': main()
