"""Observe posterior sampling magnitude; do not change receiver inference."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import feature_interface_analysis as analysis
from demo.scalable_codec import atomic_json,file_hash
from demo.patch_prefix_probe import load_frames,verify_artifacts
from demo.chunk_enhancement_experiment import read,Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.feature_interface_train import DEFAULT


def worker(args):
    from demo import stage_c_a800_teacher as teacher
    Original=teacher.PersistentSeedVR2; records=[]
    class Observed(Original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            encode=self.runner.vae.encode
            @torch.no_grad()
            def observed_encode(*args,**kwargs):
                result=encode(*args,**kwargs); p=result.posterior
                scale=float(self.runner.config.vae.scaling_factor)
                sampled=result.latent; center=p.mode()
                if sampled.ndim!=center.ndim: center=center.squeeze(2)
                records.append(dict(use_sample=self.runner.config.vae.get('use_sample',True),
                    posterior_dtype=str(center.dtype),
                    sampled_minus_mode_rms=float((sampled.float()-center.float()).square().mean().sqrt())*scale,
                    posterior_std_rms=float(p.std.float().square().mean().sqrt())*scale,
                    mode_rms=float(center.float().square().mean().sqrt())*scale))
                return result
            self.runner.vae.encode=observed_encode
    with patch.object(teacher,'PersistentSeedVR2',Observed): analysis.worker(args)
    atomic_json(args.output/'posterior.json',dict(windows=records,code=file_hash(Path(__file__))))


def main(args):
    run=Run(args);run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            for sid in ('mechanism-00-reds','mechanism-01-uvg'):
                src=args.output/'evaluation'/sid/'actual_full';dest=args.output/'posterior_probe'/sid
                config=dict(code=file_hash(Path(__file__)),analysis_code=file_hash(Path(analysis.__file__)),
                            reference=file_hash(src/'result.json'))
                if (dest/'complete.json').exists():
                    saved=read(dest/'complete.json');assert saved['config']==config
                    verify_artifacts(dest,saved['artifacts']);continue
                dest.mkdir(parents=True,exist_ok=True)
                execute(run,'posterior_'+sid,Path(__file__).name,['--worker','--stream',src/'stream.acsg',
                    '--output',dest,'--adapter',args.output/'actual/adapter.pt'],distributed=True)
                np.testing.assert_array_equal(load_frames(dest/'reconstruction.npz'),load_frames(src/'reconstruction.npz'))
                atomic_json(dest/'complete.json',dict(config=config,pixels_exact=True,
                    artifacts={p.name:file_hash(p) for p in (dest/'posterior.json',dest/'numerics.json',
                                                            dest/'decode.json',dest/'reconstruction.npz')}))
            atomic_json(args.output/'posterior_probe/complete.json',dict(complete=True,pixels_exact=True))
            run.update(phase='complete')
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    import os
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--stream',type=Path);p.add_argument('--adapter',type=Path)
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--max-hours',type=float,default=.5)
    args=p.parse_args();args.disable_generation=False;args.command='posterior_probe'
    if args.worker: worker(args)
    else:
        if not os.environ.get('TMUX'): p.error('run inside tmux')
        main(args)
