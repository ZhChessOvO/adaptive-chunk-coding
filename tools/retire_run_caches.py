"""Retire explicitly audited, obsolete training pixel caches (October 15).

Plan is read-only apart from its own manifest. Execution requires both an exact
plan digest and verified formal G startup. Never remove a directory recursively.
Datasets, source pixels, streams, labels, models and compact evaluation evidence
survive. --expanded also retires old full decoded videos and sender caches.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

RUNS = Path('/root/autodl-fs/DCVC/runs')
REAL_RUNS = RUNS.resolve(strict=True)  # AutoDL's mount itself is a symlink.
ACTIVE = RUNS / 'routervc_texture_context_20261015'
AUDIT = RUNS / '_maintenance_20261015'
# Exact old TRAINING teacher caches, not their final evaluation directories.
GROUPS = (
    ('routervc_revision_20261003/mixedview_teacher/encoded', ('candidates.npz',),
     'routervc_revision_20261003/mixedview_teacher/complete.json'),
    ('routervc_revision_20261003/mixedview_teacher/received', ('received_E.npz', 'outputs.npz'),
     'routervc_revision_20261003/mixedview_teacher/complete.json'),
    ('routervc_light_router_20261004/teacher/received', ('received_E.npz', 'outputs.npz'),
     'routervc_light_router_20261004/teacher/complete.json'),
    ('a800_four_state_20261002/received', ('received_E.npz', 'outputs.npz'),
     'a800_four_state_20261002/complete.json'),
    ('routervc_mixed_router_20261004/formal/samples', ('generated.npz',),
     'routervc_mixed_router_20261004/formal/router/complete.json'),
    ('routervc_receiver_20261005/formal/samples', ('generated.npz',),
     'routervc_receiver_20261005/formal/router/complete.json'),
)
PROTECTED = (
    'routervc_texture_context_20261015', 'routervc_texture_clean_20261014',
    'routervc_texture_repair_20261014', 'routervc_texture_loss_ablation_20261014',
    'routervc_fusion_20261010', 'routervc_precision_20261012', 'routervc_precision9_20261012',
    'routervc_revision_20261003/mixedview_data',
    'routervc_revision_20261003/visual_evaluation_recovered', 'a800_online_eg_20261002',
)
PROTECTED_FILES = set()
REASON = 'obsolete pre-latent training teachers; labels already materialized; not consumed by active G training or its queued review'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    tmp.replace(path)


def identity(path):
    info = path.lstat()
    unexpected_link = (path.is_relative_to(RUNS) and
                       path.resolve() != REAL_RUNS/path.relative_to(RUNS))
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or unexpected_link:
        raise ValueError('not a real, non-symlink file: ' + str(path))
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns,
                inode=info.st_ino, device=info.st_dev)


def selected(path):
    """Whitelist filename AND exact subtree; source/weights/receipts never match."""
    if path.is_relative_to(REAL_RUNS):
        path = RUNS/path.relative_to(REAL_RUNS)
    canonical = REAL_RUNS/path.relative_to(RUNS) if path.is_relative_to(RUNS) else path
    if str(path) in PROTECTED_FILES or str(canonical) in PROTECTED_FILES:
        return False
    if any(path.is_relative_to(RUNS / p) for p in PROTECTED):
        return False
    return any(path.is_relative_to(RUNS / rel) and
               (path.name in names or ('*.npz' in names and path.suffix=='.npz'))
               for rel, names, _ in GROUPS)


def protected_identity(path):
    # Preserved assets may intentionally be symlinks to the fast data disk.
    resolved = path.resolve(strict=True)
    return dict(resolved=str(resolved), **identity(resolved))


def strings(data):
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        for value in data.values(): yield from strings(value)
    elif isinstance(data, list):
        for value in data: yield from strings(value)


def dependency_inventory():
    # Covers source PNGs, actual received Y, initial/model assets and evaluation
    # folders. Entire active/fusion/diagnostic trees are additionally protected.
    manifests = [ACTIVE/'probe/protocol.json',
                 RUNS/'routervc_texture_clean_20261014/formal/protocol.json',
                 RUNS/'routervc_texture_clean_20261014/evaluation/protocol.json']
    if (ACTIVE/'training/formal/protocol.json').exists():
        manifests.append(ACTIVE/'training/formal/protocol.json')
    dependencies = {}
    for manifest in manifests:
        for value in strings(read(manifest)):
            if value.startswith('/'):
                path = Path(value)
                if path.is_file():
                    if selected(path): raise ValueError('active dependency selected: ' + value)
                    dependencies[value] = protected_identity(path)
                elif path.is_dir() and any((RUNS/r).is_relative_to(path) for r,_,_ in GROUPS):
                    raise ValueError('ambiguous active directory overlaps targets: ' + value)
    return dependencies


def open_files():
    result = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    value = os.readlink(fd)
                    if value.startswith((str(RUNS)+'/',str(REAL_RUNS)+'/')):
                        result.add(str(Path(value).resolve()))
                except (OSError, FileNotFoundError): pass
        except (OSError, FileNotFoundError): pass
    return result


def disk():
    return {p:dict(zip(('total','used','free'),shutil.disk_usage(p)))
            for p in ('/root','/root/autodl-tmp','/root/autodl-fs')}


def expanded_scope():
    """User explicitly broadened cleanup to cheaply regenerated old evaluations.

    Retain tiny streams, all metrics/figures/models, original source arrays and
    current generation experiments. Remove old reconstructed videos and sender
    feature caches too; these are no longer archival pixel bundles.
    """
    global GROUPS, PROTECTED, PROTECTED_FILES, AUDIT, REASON
    dependencies=dependency_inventory()
    PROTECTED_FILES=set(dependencies) | {v['resolved'] for v in dependencies.values()}
    PROTECTED=tuple(p for p in PROTECTED if p not in (
        'routervc_precision_20261012','routervc_precision9_20261012',
        'routervc_revision_20261003/visual_evaluation_recovered'))
    roots=(
        'routervc_revision_20261003','routervc_light_router_20261004',
        'routervc_receiver_20261005','routervc_mixed_router_20261004',
        'a800_four_state_20261002','routervc_sender_20261005',
        'routervc_latent_20261007','routervc_latent_20261008',
        'routervc_latent_routers_20261008','routervc_latent_sender_20261009',
        'routervc_cooperation_review_20261011','routervc_cooperative_sender_review_20261011',
        'routervc_precision_20261012','routervc_precision9_20261012',
        'routervc_efficiency_20261004','routervc_cooperative_sender_20261011',
        'routervc_cooperation_20261010')
    names=('pixels.npz','reconstruction.npz','generated.npz','received_E.npz',
           'outputs.npz','candidates.npz','expected.npz','native_own.npz')
    GROUPS=tuple((r,names,None) for r in roots)+(
        ('routervc_latent_sender_20261009/input_cache',('*.npz',),None),
        ('routervc_cooperative_sender_20261011/input_cache',('*.npz',),None))
    AUDIT=RUNS/'_maintenance_20261015/expanded'
    REASON='migration preparation authorized by user: obsolete teacher pixels, full reconstructed evaluation videos and regenerateable sender feature caches; retain compact scientific evidence and active dependencies'


def plan():
    path = AUDIT/'plan.json'
    if path.exists():
        print(json.dumps(dict(existing_plan=str(path), sha256=digest(path)))); return
    targets, retained, receipts, totals = [], {}, {}, Counter()
    links, alias_targets = {}, set()
    seen=set()
    dependencies = dependency_inventory()
    for rel, names, receipt in GROUPS:
        folder = RUNS/rel
        if folder.resolve() != REAL_RUNS/rel: raise ValueError('symlinked target subtree')
        if receipt:
            if not read(RUNS/receipt).get('complete'): raise ValueError('unfinished old stage')
            receipts[receipt] = digest(RUNS/receipt)
        for parent, dirs, files in os.walk(folder, followlinks=False):
            for n in dirs[:]:
                link=Path(parent)/n
                if link.is_symlink():
                    links[str(link)]=os.readlink(link);dirs.remove(n)
            for name in sorted(files):
                p = Path(parent)/name
                if str(p) in seen:continue
                seen.add(str(p))
                if p.is_symlink():
                    links[str(p)]=os.readlink(p)
                    if p.is_file():
                        retained[str(p)]=protected_identity(p)
                        alias_targets.add(str(p.resolve()))
                    continue
                if selected(p):
                    ident = identity(p)
                    targets.append(dict(path=str(p), group=rel, **ident))
                    totals[rel] += ident['size']
                else:
                    # Preserve scientific metadata, streams and source arrays.
                    retained[str(p)] = protected_identity(p)
    # Preserve any real file backing an existing retained alias (in particular
    # source pixels). Never follow directory aliases to enlarge deletion scope.
    filtered=[]
    for item in targets:
        p=Path(item['path'])
        if str(REAL_RUNS/p.relative_to(RUNS)) in alias_targets:
            retained[str(p)]=protected_identity(p);totals[item['group']]-=item['size']
        else:filtered.append(item)
    targets=filtered
    opened = open_files()
    if any(str(Path(t['path']).resolve()) in opened for t in targets): raise ValueError('target is currently open')
    data = dict(utc=now(), authorization='user: clean obsolete run intermediates after formal training starts',
        targets=targets, bytes=sum(t['size'] for t in targets), group_bytes=dict(totals),
        retained=retained, retained_links=links, receipts=receipts, dependencies=dependencies, disks=disk(),
        consequences='No backup. Old raw-pixel whole-tree verification and new full-output plots need cache regeneration; labels, model weights, streams, metrics and published figures remain.',
        reason=REASON)
    write(path, data)
    print(json.dumps(dict(plan=str(path),sha256=digest(path),files=len(targets),
                         gib=data['bytes']/2**30,groups=dict(totals)),ensure_ascii=False),flush=True)


def verify_retained(data):
    for path, target in data.get('retained_links',{}).items():
        if not Path(path).is_symlink() or os.readlink(path)!=target:
            raise ValueError('retained alias changed: '+path)
    for path, expected in {**data['retained'], **data['dependencies']}.items():
        # Target inode/device identity also authenticates preserved symlink
        # referents; avoid resolving every immutable parent repeatedly on FUSE.
        info=Path(path).stat()
        actual=dict(size=info.st_size,mtime_ns=info.st_mtime_ns,inode=info.st_ino,device=info.st_dev)
        if not stat.S_ISREG(info.st_mode) or any(actual[k]!=expected[k] for k in actual):
            raise ValueError('protected file changed: '+path)
    for path, expected in data['receipts'].items():
        if digest(RUNS/path) != expected: raise ValueError('historical receipt changed')
    dependencies=dependency_inventory()
    protected_real={v['resolved'] for v in dependencies.values()}
    if any(str(REAL_RUNS/Path(t['path']).relative_to(RUNS)) in protected_real for t in data['targets']):
        raise ValueError('target is a resolved active dependency')


def execute(expected_digest):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    path = AUDIT/'plan.json'
    if digest(path) != expected_digest: raise ValueError('reviewed plan hash required')
    started = read(ACTIVE/'training/formal_started.json')
    if started['steps'] < 25: raise ValueError('formal optimizer startup not verified')
    data = read(path); verify_retained(data)
    with (AUDIT/'delete.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (AUDIT/'complete.json').exists():
            print(json.dumps(read(AUDIT/'complete.json')));return
        start=AUDIT/'execution_start.json'
        if not start.exists():write(start,dict(utc=now(),disks=disk(),formal_started=started))
        before=read(start)['disks']
        opened = open_files()
        if any(str(Path(t['path']).resolve()) in opened for t in data['targets']): raise ValueError('target now open')
        journal = AUDIT/'deleted.jsonl'
        done = {json.loads(s)['path'] for s in journal.read_text().splitlines()} if journal.exists() else set()
        with journal.open('a') as log:
            for i, item in enumerate(data['targets']):
                p = Path(item['path'])
                if not selected(p): raise ValueError('target outside current whitelist')
                if item['path'] in done:
                    if p.exists(): raise ValueError('retired cache unexpectedly recreated')
                    continue
                expected = {k:item[k] for k in ('size','mtime_ns','inode','device')}
                if identity(p) != expected: raise ValueError('target changed since plan: '+str(p))
                p.unlink()
                log.write(json.dumps(dict(item,utc=now()))+'\n');log.flush();os.fsync(log.fileno())
                if (i+1)%250==0: print(json.dumps(dict(deleted=i+1,total=len(data['targets']))),flush=True)
        verify_retained(data)
        if any(Path(t['path']).exists() for t in data['targets']): raise ValueError('some targets remain')
        after=disk()
        summary=dict(complete=True,utc=now(),plan_sha256=expected_digest,files=len(data['targets']),
            apparent_bytes=data['bytes'],group_bytes=data['group_bytes'],before=before,after=after,
            observed_fs_used_reduction=before['/root/autodl-fs']['used']-after['/root/autodl-fs']['used'],
            preserved_files_checked=len(data['retained']),active_dependencies_checked=len(data['dependencies']),
            warning=data['consequences'])
        write(AUDIT/'complete.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','execute'))
    parser.add_argument('--plan-sha256')
    parser.add_argument('--expanded',action='store_true')
    args=parser.parse_args()
    if args.expanded:expanded_scope()
    if args.command=='plan':plan()
    else:execute(args.plan_sha256)
