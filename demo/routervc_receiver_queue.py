"""Resumable single-A800 G-only receiver smoke/training queue."""
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_receiver_train import ROOT, CACHE, make_protocol
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.routervc_mixed_queue import exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke','train','verify'))
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--max-hours',type=float,default=24.)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('run receiver queue inside tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('positive finite max-hours required')
    run = Run(SimpleNamespace(output=args.output/'queue', command='receiver_router', max_hours=args.max_hours))
    run.thread.start()
    os.environ['ROUTERVC_RECEIVER_PARENT'] = str(os.getpid())
    try:
        smoke = args.command == 'smoke'
        root = args.output/('smoke' if smoke else 'formal')
        protocol = make_protocol(smoke)
        immutable(root/'protocol.json',protocol)
        if args.command == 'verify':
            execute(run,'verify_only','routervc_receiver_train.py',
                ['--root',root,'--cache',args.cache/'formal','--output',root/'router','--verify-only'])
            return
        with exclusive_native_evaluation(run):
            if smoke:
                def worker(name, stop=0):
                    execute(run,name,'routervc_receiver_train.py',
                        ['--root',root,'--cache',args.cache/'smoke','--output',root/name,
                         '--stop-after',stop],distributed=True)
                if not (root/'resume_checked.json').exists():
                    worker('resumed',1)
                    preserved = {str(p.relative_to(root)):digest(p)
                                 for p in (root/'samples').rglob('*') if p.is_file()}
                    worker('resumed')
                    worker('direct')
                    for name, expected in preserved.items():
                        if digest(root/name) != expected:
                            raise ValueError('resuming rewrote measured labels')
                    import torch
                    left = torch.load(root/'resumed/resume.pt',weights_only=True,map_location='cpu')
                    right = torch.load(root/'direct/resume.pt',weights_only=True,map_location='cpu')
                    exact(left,right)
                    from demo.routervc_receiver_train import initialize
                    initial,_,_ = initialize()
                    for arm, weights in left['models'].items():
                        if all(torch.equal(v,initial[k]) for k,v in weights.items()):
                            raise ValueError('receiver did not update')
                    save(root/'resume_checked.json',dict(complete=True, models_exact=True,
                        optimizers_exact=True, best_selection_exact=True, labels_preserved=True,
                        arms=list(left['models']), G_only_head=True, sender_training=False))
                from demo.routervc_receiver_smoke import fresh_checks
                fresh_checks(run,root,protocol,root/'resumed')
                immutable(root/'complete.json',dict(complete=True,protocol=digest(root/'protocol.json'),
                    artifacts={n:digest(root/n) for n in ('resume_checked.json','fresh_audit.json',
                                                        'resumed/complete.json','direct/complete.json')}))
                run.update(phase='receiver_smoke_complete')
            else:
                smoke_root = args.output/'smoke'
                tested, checked = read(smoke_root/'protocol.json'),read(smoke_root/'complete.json')
                verify_artifacts(smoke_root,checked['artifacts'])
                if not checked['complete'] or checked['protocol'] != digest(smoke_root/'protocol.json'):
                    raise ValueError('receiver smoke incomplete')
                for key in ('code','initial_sha256','scale_sha256','teacher_profile','enhancement','adapter'):
                    if tested[key] != protocol[key]:
                        raise ValueError('formal configuration differs from smoke: '+key)
                execute(run,'formal_training','routervc_receiver_train.py',
                    ['--root',root,'--cache',args.cache/'formal','--output',root/'router'],distributed=True)
                immutable(args.output/'complete.json',dict(complete=True,protocol=digest(root/'protocol.json'),
                    router=digest(root/'router/complete.json'),labels=digest(root/'labels.complete.json'),
                    evaluation_pending=True,sender_training_complete=False))
                run.update(phase='receiver_training_complete_evaluation_pending')
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)
        run.lock.close()


if __name__ == '__main__':
    main()
