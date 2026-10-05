"""Losslessly compact explicitly selected OLD run frames, never datasets.

Only im<number>.png files are selected. Metrics, streams, checkpoints, figures
and directories remain at their original paths. Each archive embeds a SHA256
manifest; the same command can resume packing/copying/retiring. A verified fast
disk copy exists before any unlink, and a verified file-store copy exists before
bulk retirement. At a zero-inode quota, at most eight already backed-up files
are retired first to permit creating the archive on the file store.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import time

RUNS = Path('/root/autodl-fs/DCVC/runs')
STAGE = Path('/root/autodl-tmp/DCVC/tmp/frame_archives_20261005')
ALLOWED = (
    'a800_joint_evaluation_20260918', 'a800_independent_test_20260918',
    'a800_seedvr2_lora_budget_curve_20260921', 'a800_v6_evaluation_20260920',
)
MANIFEST = '_frame_archive_manifest.json'
ARCHIVE = 'archived_frames_20261005.tar'
NOTICE = 'archived_frames_20261005.json'


def sha(path_or_file):
    if hasattr(path_or_file, 'read'):
        return hashlib.file_digest(path_or_file, 'sha256').hexdigest()
    with Path(path_or_file).open('rb') as handle:
        return sha(handle)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def frame_path(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not re.fullmatch(r'im\d+\.png', rel.name):
        raise ValueError('unsafe frame member: ' + name)
    target = root / rel
    if target.resolve() != root.resolve() / rel:
        raise ValueError('symlink or escaped frame path: ' + name)
    return target


def inventory(root):
    entries = []
    for path in sorted(root.rglob('im*.png')):
        if not re.fullmatch(r'im\d+\.png', path.name):
            continue
        path = frame_path(root, str(path.relative_to(root)))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('only unlinked regular frames may be compacted: ' + str(path))
        entries.append(dict(path=str(path.relative_to(root)), bytes=info.st_size, sha256=sha(path)))
    if not entries:
        raise ValueError('no old numbered frames found')
    return dict(format='old_frames_lossless_tar_v1', root=str(root), files=entries,
                files_count=len(entries), bytes=sum(x['bytes'] for x in entries))


def verify_tar(path, plan):
    expected = {item['path']: item for item in plan['files']}
    if len(expected) != len(plan['files']):
        raise ValueError('duplicate manifest member')
    seen = set()
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            if member.name in seen or not member.isfile():
                raise ValueError('duplicate or non-file tar member')
            seen.add(member.name)
            stream = archive.extractfile(member)
            if member.name == MANIFEST:
                if json.load(stream) != plan:
                    raise ValueError('embedded manifest differs')
            else:
                item = expected.get(member.name)
                if item is None or member.size != item['bytes'] or sha(stream) != item['sha256']:
                    raise ValueError('tar member checksum mismatch: ' + member.name)
    if seen != set(expected) | {MANIFEST}:
        raise ValueError('missing tar members')


def retire_one(root, item):
    path = frame_path(root, item['path'])
    if path.exists():
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('source changed; will not unlink: ' + str(path))
        path.unlink()
        return 1
    return 0


def compact(root, stage):
    start = time.monotonic()
    stage.mkdir(parents=True, exist_ok=True)
    manifest = stage / 'manifest.json'
    if manifest.exists():
        plan = json.loads(manifest.read_text())
        if plan['root'] != str(root):
            raise ValueError('archive source binding changed')
    else:
        plan = inventory(root)
        write_json(manifest, plan)
    print(json.dumps(dict(stage='manifest', root=str(root), files=plan['files_count'], bytes=plan['bytes'])), flush=True)
    local = stage / ARCHIVE
    target = root / ARCHIVE
    if not local.exists() and not target.exists():
        # PNGs are already compressed: use a seekable, lossless uncompressed TAR.
        required = plan['bytes'] + len(plan['files']) * 2048 + 64 * 1024**2
        if shutil.disk_usage(stage).free < required + 5 * 1024**3:
            raise RuntimeError('insufficient fast-disk staging headroom')
        partial = local.with_suffix('.tar.partial')
        with tarfile.open(partial, 'w', format=tarfile.PAX_FORMAT) as archive:
            for i, item in enumerate(plan['files']):
                path = frame_path(root, item['path'])
                if sha(path) != item['sha256']:
                    raise ValueError('source changed before archiving')
                archive.add(path, arcname=item['path'], recursive=False)
                if (i + 1) % 1000 == 0:
                    print(f'PACK {root.name} {i+1}/{len(plan["files"])}', flush=True)
            payload = json.dumps(plan).encode()
            member = tarfile.TarInfo(MANIFEST)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        with partial.open('rb') as handle:
            os.fsync(handle.fileno())
        verify_tar(partial, plan)
        os.replace(partial, local)
    source_archive = local if local.exists() else target
    verify_tar(source_archive, plan)
    archive_hash = sha(source_archive)
    receipt = dict(root=str(root), archive=str(root / ARCHIVE), archive_sha256=archive_hash,
                   archive_bytes=source_archive.stat().st_size, files=plan['files_count'], original_bytes=plan['bytes'],
                   bootstrap_retired=[], complete=False,
                   restore=f'tar --skip-old-files -xf {root / ARCHIVE} -C {root}',
                   note='Lossless frame archive. Free sufficient inodes before restoring. Metrics, streams, models and figures remain unpacked.')
    receipt_path = stage / 'receipt.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old['archive_sha256'] != archive_hash:
            raise ValueError('staged archive changed')
        receipt['bootstrap_retired'] = old['bootstrap_retired']
    write_json(receipt_path, receipt)
    if not target.exists():
        fs = os.statvfs(root)
        if fs.f_files and fs.f_favail < 8:
            bootstrap_count = 0
            for item in plan['files']:
                if os.statvfs(root).f_favail >= 8 or bootstrap_count >= 8:
                    break
                if retire_one(root, item):
                    bootstrap_count += 1
                    receipt['bootstrap_retired'].append(item['path'])
                    write_json(receipt_path, receipt)
            # Some network filesystems publish quota counters asynchronously.
            for _ in range(20):
                if os.statvfs(root).f_favail >= 8:
                    break
                time.sleep(.5)
            if os.statvfs(root).f_favail < 8:
                raise RuntimeError('could not reserve archive metadata entries')
        if shutil.disk_usage(root).free < local.stat().st_size + 1024**3:
            raise RuntimeError('insufficient file-store archive byte headroom')
        temporary = target.with_suffix('.tar.partial')
        with local.open('rb') as source, temporary.open('wb') as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024**2)
            destination.flush()
            os.fsync(destination.fileno())
        if sha(temporary) != archive_hash:
            raise ValueError('file-store archive copy mismatch')
        os.replace(temporary, target)
    if sha(target) != archive_hash:
        raise ValueError('existing persistent archive differs')
    verify_tar(target, plan)
    write_json(root / NOTICE, receipt)
    removed = 0
    for i, item in enumerate(plan['files']):
        removed += retire_one(root, item)
        if (i + 1) % 1000 == 0:
            print(f'RETIRE {root.name} {i+1}/{len(plan["files"])}', flush=True)
    receipt.update(complete=True, removed_this_attempt=removed, seconds=time.monotonic()-start,
                   final_free_inodes=os.statvfs(root).f_favail)
    write_json(receipt_path, receipt)
    write_json(root / NOTICE, receipt)
    if local.exists():
        # The persistent archive has been fully read back and hash-verified.
        local.unlink()
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', choices=ALLOWED, nargs='+')
    parser.add_argument('--execute', action='store_true', help='Without this flag, inventory only.')
    args = parser.parse_args()
    if args.execute and not os.environ.get('TMUX'):
        raise RuntimeError('archive/retirement must run inside tmux')
    for name in args.run:
        root = RUNS / name
        if root.is_symlink() or root.resolve() != RUNS.resolve() / name or not root.is_dir():
            raise ValueError('invalid allowlisted run root')
        if args.execute:
            stage = STAGE / name
            stage.mkdir(parents=True, exist_ok=True)
            with (stage / 'archive.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                compact(root, stage)
        else:
            plan = inventory(root)
            print(json.dumps({k:v for k,v in plan.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
