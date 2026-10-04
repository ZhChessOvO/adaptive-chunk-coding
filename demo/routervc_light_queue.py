"""Resumable q2 label smoke, teacher and paired Router-only training queue."""
import argparse
import os
from pathlib import Path
from demo.routervc_light_teacher import OUTPUT, CODE, REPO, read, digest, save, immutable
from demo.chunk_enhancement_experiment import Run
from demo.conditioned_generation_pipeline import execute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('smoke','run','verify'))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--cache',type=Path,default=Path('/root/autodl-tmp/DCVC/cache/routervc_light_20261004'))
    p.add_argument('--max-hours',type=float,default=36.)
    args = p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('queue requires tmux')
    root = args.output; args.output = root/'queue'
    run = Run(args); run.thread.start()
    try:
        immutable(run.root/'request.json',dict(code={n:digest(REPO/'demo'/n) for n in CODE},
                  epochs=120,cache=str(args.cache),freeze_UF_E_G=True,semantic_supervision=False))
        smoke, teacher = root/'teacher_smoke',root/'teacher'
        if args.command == 'verify':
            if not read(run.root/'complete.json')['complete']: raise ValueError('queue incomplete')
            execute(run,'verify_teacher','routervc_light_teacher.py',['verify','--output',teacher])
            from demo.routervc_fullview_probe import verify_artifacts
            verify_artifacts(root,read(run.root/'complete.json')['artifacts'])
            return
        execute(run,'01_teacher_smoke','routervc_light_teacher.py',['smoke','--output',smoke,'--max-hours',4])
        execute(run,'02_training_smoke','routervc_visual_train.py',
                ['--labels',smoke/'labels.json','--cache',args.cache/'smoke',
                 '--output',root/'router_smoke','--smoke','--max-hours',2])
        if args.command == 'smoke': return
        execute(run,'03_q2_teacher','routervc_light_teacher.py',
                ['run','--output',teacher,'--smoke-root',smoke,'--max-hours',24])
        execute(run,'04_q2_router','routervc_visual_train.py',
                ['--labels',teacher/'labels.json','--cache',args.cache/'formal',
                 '--output',root/'router','--epochs',120,'--max-hours',6])
        immutable(run.root/'complete.json',dict(complete=True,qstep=2.,trained='Router only',
            artifacts={str(path.relative_to(root)):digest(path) for path in
                (teacher/'complete.json',root/'router/complete.json',root/'router/global_local/model.pt',root/'router/local/model.pt')}))
        run.update(phase='complete',completed=4,total=4)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__': main()
