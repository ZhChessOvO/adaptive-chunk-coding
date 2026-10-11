"""P2b: exact-restart real smoke, 60-window final teacher, warm-start paired R_s.

Formal completion stops here. Evaluation and receiver/sender promotion are
separate decisions; this queue never launches another adaptation round.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import torch

from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.routervc_mixed_queue import exact
from tools.fusion_finish import memory_snapshot
from routervc.cooperation import sender_data as data, sender_training as fit


class SenderRun(Run):
    def check(self):
        super().check()
        if time.monotonic()-getattr(self, '_last_size_check', 0) >= 30:
            self._last_size_check = time.monotonic()
            total = 0
            for folder, _, names in os.walk(self.root.parent):
                for name in names:
                    path = Path(folder)/name
                    try:
                        if not path.is_symlink(): total += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Atomic checkpoint replacement can race this observation.
            self.progress['pilot_output_bytes'] = total
            if total >= 8*1024**3:
                raise InterruptedError('8 GiB pilot output limit reached; preserve outputs and inspect')

    def log_resources(self):
        super().log_resources()
        with (self.root/'memory.jsonl').open('a') as log:
            log.write(json.dumps(dict(unix_seconds=time.time(), **memory_snapshot()))+'\n')


def execute(run, name, args, *, distributed=False, module='tools.cooperative_sender_worker'):
    command = [sys.executable]
    if distributed: command += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1']
    command += ['-m', module, *map(str, args)]
    path = run.root/(name+'.log')
    run.update(phase=name, log=str(path))
    with path.open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None: run.check(); time.sleep(.5)
            if child.returncode: raise RuntimeError(f'{name} failed ({child.returncode}); see {path}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()


def fresh(run, wire, dest, protocol, *, disabled=False):
    from routervc.cooperation import stream
    inner, config = stream.parse(Path(wire).read_bytes())
    expected = dict(stream=digest(wire), receiver=protocol['teacher']['receiver'], disabled=disabled,
                    profile=protocol['receiver_profile'])
    immutable(dest/'request.json', expected)
    if not (dest/'complete.json').exists():
        options = ['decode', '--stream', wire, '--output', dest]
        if disabled: options += ['--disable-generation']
        else: options += ['--receiver', protocol['teacher']['receiver']['path']]
        execute(run, 'fresh_'+dest.parent.name+'_'+dest.name, options,
                distributed=True, module='tools.cooperative_sender_decode')
    result = read(dest/'complete.json'); verify_artifacts(dest, result['artifacts'])
    if (result['stream'] != digest(wire) or result['actual_bytes'] != Path(wire).stat().st_size
            or result['source_frames_read'] or result['sender_router_loaded']
            or result.get('sender_checkpoint_access_guard') is not True
            or result['generation_disabled'] != disabled or result['mask_bytes'] != 0
            or config['receiver_sha256'] != protocol['teacher']['receiver']['sha256']):
        raise ValueError('fresh decoder contract differs')
    return result


def smoke_checks(run, root, cache, protocol):
    receipts = {}; points = 0
    for row in protocol['rows']:
        sid = row['sample_id']; folder = root/'samples'/sid
        plans = read(folder/'plans.json')['plans']; chosen = plans[1]['candidates'][0]
        for name, relative, disabled in [('empty', 's0/parent', False), ('partial', 's1/parent', False),
                ('child', f's1/add{chosen}', False), ('repeat', f's1/add{chosen}', False),
                ('off', f's1/add{chosen}', True)]:
            teacher = read(folder/relative/'result.json'); dest = root/'fresh'/sid/name
            report = fresh(run, folder/relative/'stream.rvlcoop', dest, protocol, disabled=disabled)
            if (report['base_hash'] != teacher['base_hash'] or report['enhanced_hash'] != teacher['received_hash']
                    or report['output_hash'] != teacher['received_hash' if disabled else 'output_hash']):
                raise ValueError('actual fresh P2/F differs from teacher')
            receipts[str(dest.relative_to(root)/'complete.json')] = digest(dest/'complete.json'); points += 1
        for arm in ('source', 'zero_source'):
            planned = root/'planned'/sid/arm; checkpoint = root/'resumed'/arm/'best.pt'
            if not (planned/'complete.json').exists():
                execute(run, f'plan_{sid}_{arm}', ['plan', '--root', root, '--cache', cache,
                    '--output', planned, '--sample-id', sid, '--checkpoint', checkpoint])
            done = read(planned/'complete.json'); verify_artifacts(planned, done['artifacts'])
            receipts[str(planned.relative_to(root)/'complete.json')] = digest(planned/'complete.json')
            for point in (done['points'][0], done['points'][2]):
                dest = planned/('fresh_'+point['file'].split('.')[0])
                report = fresh(run, planned/point['file'], dest, protocol)
                if report['actual_bytes'] != point['actual_bytes'] or report['base_hash'] != teacher['base_hash']:
                    raise ValueError('planned actual bytes/B differ')
                receipts[str(dest.relative_to(root)/'complete.json')] = digest(dest/'complete.json'); points += 1
    save(root/'fresh_audit.json', dict(complete=True, fresh_points=points, artifacts=receipts,
        teacher_fresh_pixels_exact=True, repeat_exact=True, G_off_no_Rg_or_G_assets=True,
        sourcefree_receiver=True, masks_sent=False, both_sender_arms=True, literal_prefix_pairs=12))


def verify(root, cache, check=lambda:None):
    root, cache = Path(root), Path(cache)
    done = read(root/'complete.json'); verify_artifacts(root, done['artifacts'])
    protocol = read(root/'protocol.json')
    for name, expected in protocol['code'].items():
        if digest(data.REPO/name) != expected: raise ValueError('pinned sender source changed: '+name)
    samples = data.Samples(root, cache, protocol, check)
    try:
        labels = data.verify_labels(root, protocol, samples, check)
        for name in ('resumed', 'direct') if protocol['smoke'] else ('router',):
            fit.verify_training(root/name, protocol, labels_binding=labels, scale_record=read(root/'train_scale.json'))
        if protocol['smoke']:
            audit = read(root/'fresh_audit.json'); verify_artifacts(root, audit['artifacts'])
    finally:
        # Verification is CPU-only, including on a host with an available GPU.
        samples.memory.clear()
    print('COOPERATIVE_SENDER_VERIFIED', root, flush=True)
    return done


def stage(run, args, smoke):
    root = args.output/('smoke' if smoke else 'formal'); cache = args.output/'input_cache'/root.name
    protocol = data.make_protocol(smoke); immutable(root/'protocol.json', protocol)
    if (root/'complete.json').exists(): verify(root, cache, run.check); return
    if not smoke:
        smoke_root = args.output/'smoke'; verify(smoke_root, args.output/'input_cache/smoke', run.check)
        tested = read(smoke_root/'protocol.json')
        for key in ('code', 'teacher', 'initialization', 'receiver_profile', 'packet_profile'):
            if tested[key] != protocol[key]: raise ValueError('formal differs from real smoke')
    common = ['--root', root, '--cache', cache, '--max-hours', args.max_hours]
    if smoke and not (root/'label_resume_checked.json').exists():
        stop = root/'label_stop'
        if not (stop/'intentional_stop.json').exists():
            execute(run, 'smoke_label_stop', ['labels', *common, '--output', stop, '--stop-renderings', 1], distributed=True)
        prior = {str(p.relative_to(root)):digest(p) for p in (root/'samples').glob('*/s*/*/result.json')}
        execute(run, 'smoke_labels', ['labels', *common, '--output', root/'label_worker'], distributed=True)
        if not prior or any(digest(root/n) != h for n,h in prior.items()): raise ValueError('teacher restart rewrote states')
        save(root/'label_resume_checked.json', dict(complete=True, preserved_states=prior))
    elif not (root/'labels.complete.json').exists():
        execute(run, 'formal_labels', ['labels', *common, '--output', root/'label_worker'], distributed=True)
    samples = data.Samples(root, cache, protocol, run.check)
    labels = data.verify_labels(root, protocol, samples, run.check); samples.memory.clear()
    if smoke:
        if not (root/'resume_checked.json').exists():
            if not (root/'resumed/resume.pt').exists():
                execute(run, 'smoke_fit_stop', ['fit', *common, '--output', root/'resumed', '--stop-after', 1])
            execute(run, 'smoke_fit_resume', ['fit', *common, '--output', root/'resumed'])
            execute(run, 'smoke_fit_direct', ['fit', *common, '--output', root/'direct'])
            a = torch.load(root/'resumed/resume.pt', map_location='cpu', weights_only=True)
            b = torch.load(root/'direct/resume.pt', map_location='cpu', weights_only=True)
            exact(a, b)
            save(root/'resume_checked.json', dict(complete=True, models_optimizers_rng_best_exact=True))
        smoke_checks(run, root, cache, protocol)
        names = ['resumed/complete.json', 'direct/complete.json', 'resume_checked.json',
                 'label_resume_checked.json', 'fresh_audit.json']
    else:
        execute(run, 'formal_fit', ['fit', *common, '--output', root/'router'])
        names = ['router/complete.json']
    names += ['protocol.json', 'labels.complete.json', 'train_scale.json']
    save(root/'complete.json', dict(complete=True, smoke=smoke, warm_start=True,
        evaluation_pending=not smoke, automatic_promotion=False,
        artifacts={n:digest(root/n) for n in names}))
    verify(root, cache, run.check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'train', 'all', 'verify'))
    parser.add_argument('--output', type=Path, default=data.ROOT)
    parser.add_argument('--max-hours', type=float, default=48.)
    args = parser.parse_args()
    if args.command == 'verify':
        for name in ('smoke', 'formal'): verify(args.output/name, args.output/'input_cache'/name)
        return
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required for long tasks')
    run = SenderRun(SimpleNamespace(output=args.output/'queue', command='cooperative_sender', max_hours=args.max_hours))
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            if args.command in ('smoke', 'all'): stage(run, args, True)
            if args.command in ('train', 'all'): stage(run, args, False)
        run.update(phase='complete_no_automatic_evaluation_or_promotion')
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__': main()
