"""Monitor byte AND inode headroom around an unchanged resumable tmux queue.

Writes its journal on the fast disk, so an exhausted output mount cannot hide
the reason for stopping. SIGTERM goes to the supervised queue, whose existing
cleanup stops distributed workers at checkpoint boundaries. No research source,
protocol, model or optimizer is modified. This is not a universal process killer:
the supplied command must already handle SIGTERM and stop its own children.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

MOUNTS = ('/root', '/root/autodl-tmp', '/root/autodl-fs')


def disk_snapshot(paths=MOUNTS):
    result = {}
    for path in paths:
        usage, fs = shutil.disk_usage(path), os.statvfs(path)
        result[str(path)] = dict(total_bytes=usage.total, used_bytes=usage.used,
            free_bytes=usage.free, used_fraction=usage.used / usage.total,
            total_inodes=fs.f_files, free_inodes=fs.f_favail,
            used_inodes=fs.f_files - fs.f_ffree)
    return result


def reasons(disks, min_inodes=5000, max_used_fraction=.8):
    failures = []
    for path, disk in disks.items():
        if disk['used_fraction'] >= max_used_fraction:
            failures.append(f'{path}: byte usage {disk["used_fraction"]:.1%}')
        if disk['total_inodes'] > 0 and disk['free_inodes'] < min_inodes:
            failures.append(f'{path}: only {disk["free_inodes"]} free file entries')
    return failures


def gpu_snapshot():
    try:
        return subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return dict(unavailable=repr(error))


def supervise(command, log, *, interval=30., min_inodes=5000, max_used_fraction=.8,
              stop=None, snapshot=disk_snapshot, gpu=gpu_snapshot):
    stop = stop or threading.Event()
    started = time.monotonic()
    child = None
    stopping = False
    log.parent.mkdir(parents=True, exist_ok=True)
    # Open before creating the child; failure to record resources is fail-closed.
    with log.open('a', buffering=1) as journal:
        def record(phase, **fields):
            event = dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                         elapsed_seconds=time.monotonic()-started, **fields)
            journal.write(json.dumps(event) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps(dict(storage_guard=event)), flush=True)

        try:
            disks = snapshot()
            violations = reasons(disks, min_inodes, max_used_fraction)
            record('preflight', disks=disks, violations=violations, command=command)
            if violations or stop.is_set():
                return 75
            child = subprocess.Popen(command, start_new_session=True)
            while child.poll() is None:
                try:
                    disks = snapshot()
                    violations = reasons(disks, min_inodes, max_used_fraction)
                    record('running' if not stopping else 'awaiting_checkpoint_exit',
                           pid=child.pid, disks=disks, violations=violations, gpu=gpu())
                except Exception as error:
                    violations = ['resource journal/check failed: ' + repr(error)]
                    print(violations[0], flush=True)
                if (violations or stop.is_set()) and not stopping:
                    stopping = True
                    print('STORAGE_GUARD_REQUEST_CHECKPOINT_STOP ' + str(violations), flush=True)
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass
                # Wait for this process, not a sleep which delays completed work.
                try:
                    child.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
            record('stopped_for_headroom' if stopping else 'child_exited',
                   returncode=child.returncode, checkpoint_resume_required=stopping)
            return 75 if stopping else child.returncode
        finally:
            if child is not None and child.poll() is None:
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
                # The queue owns shutdown of its separately-sessioned torchrun.
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30.)
    parser.add_argument('--min-inodes', type=int, default=5000)
    parser.add_argument('--max-used-fraction', type=float, default=.8)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not math.isfinite(args.interval) or not 1 <= args.interval <= 60:
        parser.error('command and interval in [1,60] seconds required')
    if args.min_inodes < 1 or not 0 < args.max_used_fraction < 1:
        parser.error('positive inode reserve and byte fraction in (0,1) required')
    if not os.environ.get('TMUX'):
        parser.error('run long queues inside tmux')
    if not args.log.resolve().is_relative_to(Path('/root/autodl-tmp').resolve()):
        parser.error('guard journal must be on the fast disk')
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    raise SystemExit(supervise(command, args.log, interval=args.interval, min_inodes=args.min_inodes,
                              max_used_fraction=args.max_used_fraction, stop=stop))


if __name__ == '__main__':
    main()
