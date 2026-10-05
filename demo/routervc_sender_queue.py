"""Single-A800 sender teacher, paired training and restart/fresh smoke queue.

The receiver checkpoint is an explicit selection, never an automatically chosen
in-progress model. Preparing final-quality labels is not formal optimization.
"""
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute


def matching_smoke(tested, formal):
    """Only the row set/epochs and completion attestation may differ."""
    for key in ('code', 'seed', 'learning_rate', 'weight_decay', 'ranking_weight', 'arms'):
        if tested[key] != formal[key]:
            raise ValueError('sender smoke differs from formal recipe: '+key)
    left, right = dict(tested['teacher']), dict(formal['teacher'])
    for value in (left, right):
        value['receiver'] = {k:v for k,v in value['receiver'].items()
                             if k not in ('complete_path', 'complete_sha256')}
    if left != right:
        raise ValueError('formal sender must use the same fixed R_g/G as its real smoke')
    if formal['teacher']['receiver']['smoke_weights'] is not False:
        raise ValueError('formal sender cannot use a receiver smoke checkpoint')


def get_protocol(args, root, smoke):
    from demo.routervc_sender_train import make_protocol
    receiver, sha, completed = args.receiver, args.receiver_sha256, args.receiver_complete
    if (root/'protocol.json').exists() and receiver is None and sha is None and completed is None:
        previous = read(root/'protocol.json')['teacher']['receiver']
        receiver, sha = previous['path'], previous['sha256']
        completed = previous.get('complete_path')
    if receiver is None or sha is None:
        raise ValueError('first launch requires explicit --receiver and --receiver-sha256; no automatic promotion')
    protocol = make_protocol(smoke=smoke, receiver_checkpoint=receiver,
        expected_sha256=sha, receiver_complete=completed)
    immutable(root/'protocol.json', protocol)
    return protocol


def main():
    from demo.routervc_sender_train import ROOT, CACHE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'train', 'verify'))
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--receiver', type=Path)
    p.add_argument('--receiver-sha256')
    p.add_argument('--receiver-complete', type=Path)
    p.add_argument('--max-hours', type=float, default=48.)
    args = p.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('sender queue requires tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('positive finite max-hours required')
    run = Run(SimpleNamespace(output=args.output/'queue', command='sender_router', max_hours=args.max_hours))
    run.thread.start()
    os.environ['ROUTERVC_SENDER_PARENT'] = str(os.getpid())
    os.environ['ROUTERVC_RECEIVER_PARENT'] = str(os.getpid())
    try:
        smoke = args.command == 'smoke'
        stage = 'smoke' if smoke else 'formal'
        root = args.output/stage
        protocol = get_protocol(args, root, smoke)
        if args.command == 'verify':
            execute(run, 'verify_only', 'routervc_sender_train.py',
                ['--root', root, '--cache', args.cache/stage, '--output', root/'router', '--verify-only'])
            return
        with exclusive_native_evaluation(run):
            if smoke:
                def worker(name, stop=0):
                    execute(run, name, 'routervc_sender_train.py',
                        ['--root', root, '--cache', args.cache/stage, '--output', root/name,
                         '--stop-after', stop], distributed=True)
                if not (root/'resume_checked.json').exists():
                    worker('resumed', 1)
                    preserved = {str(p.relative_to(root)):digest(p)
                                 for p in (root/'samples').rglob('*') if p.is_file()}
                    worker('resumed')
                    worker('direct')
                    for name, sha in preserved.items():
                        if digest(root/name) != sha:
                            raise ValueError('resuming sender training changed measured labels')
                    import torch
                    from demo.routervc_mixed_queue import exact
                    left = torch.load(root/'resumed/resume.pt', weights_only=True, map_location='cpu')
                    right = torch.load(root/'direct/resume.pt', weights_only=True, map_location='cpu')
                    exact(left, right)
                    from demo import routervc_sender_router as sender
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(protocol['seed'])
                        initial = sender.SenderUtilityRouter(sender.Config(**protocol['teacher']['sender_config'])).state_dict()
                    expected_steps = len(protocol['rows'])*protocol['epochs']
                    for arm, weights in left['models'].items():
                        if all(torch.equal(v, initial[k]) for k,v in weights.items()):
                            raise ValueError('sender smoke model did not update')
                        steps = {int(v['step']) for v in left['optimizers'][arm]['state'].values()}
                        if steps != {expected_steps}:
                            raise ValueError('sender smoke optimizer did not complete every update')
                    save(root/'resume_checked.json', dict(complete=True, model_exact=True,
                        optimizer_exact=True, selection_exact=True, labels_preserved=True,
                        weights_changed=True, updates_per_arm=expected_steps,
                        paired_source_ablation=True, label_scope='measured final LPIPS after fixed R_g/G'))
                from demo.routervc_sender_checks import fresh_checks
                fresh_checks(run, root, protocol, root/'resumed')
                immutable(root/'complete.json', dict(complete=True, protocol=digest(root/'protocol.json'),
                    artifacts={n:digest(root/n) for n in ('resume_checked.json', 'fresh_audit.json',
                        'resumed/complete.json', 'direct/complete.json', 'labels.complete.json')}))
                run.update(phase='sender_real_smoke_complete', formal_sender_training=False)
            else:
                smoke_root = args.output/'smoke'
                checked = read(smoke_root/'complete.json')
                if not checked.get('complete') or checked['protocol'] != digest(smoke_root/'protocol.json'):
                    raise ValueError('sender real smoke incomplete')
                verify_artifacts(smoke_root, checked['artifacts'])
                matching_smoke(read(smoke_root/'protocol.json'), protocol)
                execute(run, 'formal_teacher_and_training', 'routervc_sender_train.py',
                    ['--root', root, '--cache', args.cache/stage, '--output', root/'router'], distributed=True)
                immutable(args.output/'complete.json', dict(complete=True,
                    protocol=digest(root/'protocol.json'), router=digest(root/'router/complete.json'),
                    labels=digest(root/'labels.complete.json'), evaluation_pending=True,
                    on_policy_adaptation_complete=False))
                run.update(phase='sender_training_complete_evaluation_pending')
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)
        run.lock.close()


if __name__ == '__main__':
    main()
