"""Single-A800, byte/inode-guarded queue: real smoke, labels, core adaptation.

Sender labels may only bind a completed receiver after the resulting quality
checks. No automatic UF/G finetuning, downloads or old experiment mutation.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.scalable_codec import atomic_bytes
from demo.routervc_mixed_queue import exact
from routervc.latent import routing, router_data as data


def execute(run, name, arguments, *, distributed=False):
    command = [sys.executable]
    if distributed:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=1']
    command += ['-m', 'tools.latent_router_worker', *map(str, arguments)]
    path = run.root/f'{name}.log'
    with path.open('a') as log:
        run.update(phase=name, log=str(path))
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                run.check()
                time.sleep(.5)
            if child.returncode:
                raise RuntimeError(f'{name} failed ({child.returncode}); see {path}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def fresh_checks(run, root, protocol, checkpoint):
    """Two datasets, literal E0/E8/E16 prefixes, independent repeated receivers.

    Fresh single-region G is also compared to the cached teacher hash when the
    learned G1 policy chooses a cell, not merely against another fresh receiver.
    """
    from demo.routervc_receiver_router import load_model
    model, _ = load_model(checkpoint)
    hashes = []
    for index, row in enumerate(protocol['rows']):
        folder = root/'samples'/row['sample_id']
        bank = (folder/'bank.rvlp').read_bytes()
        out = root/'fresh'/row['sample_id']; out.mkdir(parents=True, exist_ok=True)
        streams = {}
        for count in (0, 8, 16):
            raw = routing.subset(bank, data.selection(row['sample_id'], count))
            streams[count] = routing.wrap(raw, digest(checkpoint), protocol['G_assets_hash'], max_g=1)
            atomic_bytes(out/f'e{count}.rvlrg', streams[count])
        if not (streams[16].startswith(streams[8]) and streams[8].startswith(streams[0])):
            raise ValueError('receiver envelope broke literal prefixes')
        for name, count, disabled in (('e0', 0, False), ('e8', 8, False), ('repeat8', 8, False), ('off16',16,True)):
            dest = out/name
            stream = out/f'e{count}.rvlrg'
            args = ['decode', '--stream', stream, '--output', dest, '--receiver',
                    '/nonexistent/no-G-no-router.pt' if disabled else checkpoint]
            if disabled:
                args += ['--disable-generation']
            if not (dest/'complete.json').exists():
                execute(run, f'fresh_{index}_{name}', args, distributed=True)
            result = read(dest/'complete.json'); verify_artifacts(dest, result['artifacts'])
            if result['stream_sha256'] != digest(stream) or result['actual_bytes'] != len(streams[count]):
                raise ValueError('fresh stream identity/bytes differ')
            enc = read(folder/'encoded.json')
            if result['base_hash'] != enc['base_hash'] or result['detail']['base_reference_hashes'] != enc['base_reference_hashes']:
                raise ValueError('fresh B reference mismatch')
            if disabled:
                if result['output_hash'] != enc['full_hash']:
                    raise ValueError('fresh full-E endpoint differs')
            else:
                state = read(folder/f'e{count}.json')
                if result['enhanced_hash'] != state['binding']['received_hash']:
                    raise ValueError('fresh entropy mixed-Y differs from teacher')
                if len(result['generated']) == 1:
                    chosen = result['generated'][0]
                    if result['output_hash'] != state['cells'][str(chosen)]['output_hash']:
                        raise ValueError('fresh G differs from persistent measured teacher')
                elif result['output_hash'] != result['enhanced_hash']:
                    raise ValueError('empty G is not identity')
            hashes.append(digest(dest/'complete.json'))
        if read(out/'e8/complete.json')['output_hash'] != read(out/'repeat8/complete.json')['output_hash']:
            raise ValueError('fresh repeat differs')
    result = dict(complete=True, independent_fresh_receivers=len(hashes), receipts=hashes,
        literal_prefixes=True, G_off_no_router_or_G_assets=True,
        new_mixed_Y_exact=True, G_matches_measured_teacher=True, source_access_guard=True,
        additional_mask_bytes=0, router_header_bytes=routing.HEADER_BYTES)
    save(root/'fresh_audit.json', result)
    return result


def verify(root):
    completed = read(root/'complete.json')
    if not completed['complete']:
        raise ValueError('queue incomplete')
    verify_artifacts(root, completed['artifacts'])
    return completed


def stage(run, args, smoke):
    root = args.output/('smoke' if smoke else 'formal')
    cache = args.cache/('smoke' if smoke else 'formal')
    protocol = data.make_protocol(smoke)
    immutable(root/'protocol.json', protocol)
    if (root/'complete.json').exists():
        return verify(root)
    if not smoke:
        tested = read(args.output/'smoke/protocol.json')
        verify(args.output/'smoke')
        for key in ('code', 'packet_profile', 'receiver_profile', 'G_assets', 'initial_sha256'):
            if tested[key] != protocol[key]:
                raise ValueError('formal differs from tested smoke: '+key)
    common = ['--root', root, '--cache', cache, '--max-hours', args.max_hours]
    if not (root/'labels.complete.json').exists():
        execute(run, 'smoke_labels' if smoke else 'formal_labels',
                ['labels', *common, '--output', root/'label_worker'], distributed=True)
    if smoke:
        if not (root/'resume_checked.json').exists():
            if not (root/'resumed/resume.pt').exists():
                execute(run, 'smoke_stop', ['fit', *common, '--output', root/'resumed', '--stop-after', 1])
            execute(run, 'smoke_resume', ['fit', *common, '--output', root/'resumed'])
            execute(run, 'smoke_direct', ['fit', *common, '--output', root/'direct'])
            left = torch.load(root/'resumed/resume.pt', weights_only=True, map_location='cpu')
            right = torch.load(root/'direct/resume.pt', weights_only=True, map_location='cpu')
            exact(left, right)
            initial = torch.load(protocol['initial_path'], weights_only=True, map_location='cpu')['state_dict']
            if all(torch.equal(initial[k], v) for k, v in left['model'].items()):
                raise ValueError('optimizer did not update receiver')
            save(root/'resume_checked.json', dict(complete=True, weights_optimizer_rng_best_exact=True,
                updates=left['state']['updates'], trained_core_only=True))
        fresh_checks(run, root, protocol, root/'resumed/core/best.pt')
        names = ('protocol.json', 'labels.complete.json', 'resume_checked.json', 'fresh_audit.json',
                 'resumed/complete.json', 'direct/complete.json')
    else:
        execute(run, 'formal_fit', ['fit', *common, '--output', root/'router'])
        names = ('protocol.json', 'labels.complete.json', 'router/complete.json')
    result = dict(complete=True, smoke=smoke, sender_started=False,
        evaluation_pending=not smoke, artifacts={n:digest(root/n) for n in names})
    save(root/'complete.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'train', 'all', 'verify'))
    p.add_argument('--output', type=Path, default=data.ROOT)
    p.add_argument('--cache', type=Path, default=data.CACHE)
    p.add_argument('--max-hours', type=float, default=48.)
    args = p.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    run = Run(SimpleNamespace(output=args.output/'queue', command='latent_routers', max_hours=args.max_hours))
    run.thread.start()
    try:
        if args.command == 'verify':
            for name in ('smoke', 'formal'):
                verify(args.output/name)
        else:
            with exclusive_native_evaluation(run):
                if args.command in ('smoke', 'all'):
                    stage(run, args, True)
                if args.command in ('train', 'all'):
                    stage(run, args, False)
                run.update(phase='receiver_stage_complete_sender_not_started')
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__':
    main()
