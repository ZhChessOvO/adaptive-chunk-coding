"""Teacher -> real-data visual training smoke -> mixed-view Router baselines.

Supervises separate lock-owning jobs, so this parent does NOT take the GPU lock.
No component fine-tuning, semantic-protection claim or automatic model promotion.
"""
import argparse
import math
import os
from pathlib import Path

from demo.chunk_enhancement_experiment import Run
from demo.conditioned_generation_pipeline import execute
from demo.routervc_visual_train import immutable
from demo.scalable_codec import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', type=Path,
                        default=Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003'))
    parser.add_argument('--scratch', type=Path,
                        default=Path('/root/autodl-tmp/DCVC/cache/routervc_visual_20261003'))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--max-hours', type=float, default=48.)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('epochs and max-hours must be positive')
    if not os.environ.get('TMUX'):
        raise RuntimeError('revision queue requires tmux')
    args.output, args.command = args.revision/'queue', 'revision_queue'
    run = Run(args); run.thread.start()
    revision = args.revision
    data = revision/'mixedview_data'
    smoke, teacher = revision/'mixedview_teacher_smoke', revision/'mixedview_teacher'
    repo = Path(__file__).resolve().parents[1]
    try:
        immutable(run.root/'request.json', dict(epochs=args.epochs,
            data_sha256=file_hash(data/'complete.json'), scratch=str(args.scratch.resolve()),
            code={name: file_hash(repo/'demo'/name) for name in
                  ('routervc_revision_queue.py', 'routervc_visual_router.py', 'routervc_visual_train.py',
                   'routervc_mixedview_teacher.py', 'routervc_mixedview_teacher_receive.py')},
            semantic_supervision=False, purpose='paired visual-context perceptual baselines'))
        if not (smoke/'complete.json').exists():
            raise ValueError('launch this queue after the separately supervised teacher smoke completes')
        execute(run, '01_verify_teacher_smoke', 'routervc_mixedview_teacher.py',
                ['verify', '--data', data, '--output', smoke, '--max-hours', 2])
        execute(run, '02_visual_training_smoke', 'routervc_visual_train.py',
                ['--labels', smoke/'labels.json', '--cache', args.scratch/'smoke',
                 '--output', revision/'visual_router_smoke', '--smoke', '--max-hours', 3])
        execute(run, '03_mixed_teacher', 'routervc_mixedview_teacher.py',
                ['verify' if (teacher/'complete.json').exists() else 'run',
                 '--data', data, '--output', teacher, '--smoke-root', smoke, '--max-hours', 24])
        execute(run, '04_visual_training', 'routervc_visual_train.py',
                ['--labels', teacher/'labels.json', '--cache', args.scratch/'formal',
                 '--output', revision/'visual_router', '--epochs', args.epochs, '--max-hours', 18])
        immutable(run.root/'complete.json', dict(complete=True, request=file_hash(run.root/'request.json'),
                  artifacts={str(path.relative_to(revision)): file_hash(path) for path in
                             (teacher/'complete.json', revision/'visual_router/complete.json')},
                  semantic_supervision=False, deployment_pending=True))
        run.update(phase='complete', completed=4, total=4)
    except BaseException as error:
        atomic_json(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
