"""Synthetic GPU architecture/resume audit only; not sender research training."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(
        '/root/autodl-fs/DCVC/runs/routervc_receiver_20261005/sender_architecture_smoke'))
    args=parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('GPU architecture audit must run in tmux')
    run=Run(SimpleNamespace(output=args.output,command='sender_architecture_smoke',max_hours=1.))
    run.thread.start()
    try:
        files=('routervc_sender_router.py','test_routervc_sender_router.py','routervc_sender_smoke.py')
        protocol=dict(scope='synthetic GPU optimization and exact resume, NOT measured sender training',
            code={n:digest(Path(__file__).parent/n) for n in files},device='cuda',training_data=False)
        immutable(args.output/'protocol.json',protocol)
        done=args.output/'complete.json'
        if done.exists():
            saved=read(done)
            if saved['protocol']!=digest(args.output/'protocol.json'):
                raise ValueError('sender architecture audit protocol changed')
            verify_artifacts(args.output,saved['artifacts'])
            return
        with exclusive_native_evaluation(run):
            from demo.test_routervc_sender_router import resume_equivalence
            result=resume_equivalence('cuda',args.output/'fixture')
        paths={str(p.relative_to(args.output)):digest(p)
               for p in (args.output/'fixture').rglob('*') if p.is_file()}
        immutable(done,dict(complete=True,protocol=digest(args.output/'protocol.json'),
            result=result,artifacts=paths,formal_sender_training=False))
        run.update(phase='synthetic_GPU_resume_complete',formal_sender_training=False)
        print(result,flush=True)
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)
        run.lock.close()


if __name__=='__main__':
    main()
