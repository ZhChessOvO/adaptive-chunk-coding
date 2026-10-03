"""REDS whole views + existing UVG crops, without waiting for UVG originals.

This is a separate input profile: it never changes the full-view-only contract.
Old sequence-level Router groups are retained. Old E/G exposure is recorded,
and a cropped UVG input is never described as full-frame or independent data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from demo import routervc_fullview_data as full


FORMAT = 'routervc_reds_full_uvg_crop_v1'
PREVIOUS_COMPONENT = 'a800_online_eg_20261002/joint/training_manifest.json'
PREVIOUS_LABELS = 'a800_four_state_20261002/labels.json'
PREVIOUS_ROUTER = 'a800_four_state_router_20261002/context/config.json'


def previous_layout(root):
    paths = dict(component=Path(root)/'runs'/PREVIOUS_COMPONENT,
                 labels=Path(root)/'runs'/PREVIOUS_LABELS,
                 router=Path(root)/'runs'/PREVIOUS_ROUTER)
    component, labels, router = (full.read(paths[k]) for k in ('component', 'labels', 'router'))
    if not labels['complete'] or router['data']['labels'] != full.digest(paths['labels']):
        raise ValueError('old Router group configuration is not bound to the given teacher ledger')
    entries = component['entries']
    by_id = {r['sample_id']: r for r in entries}
    if len(by_id) != len(entries) or set(by_id) != {r['sample_id'] for r in labels['samples']}:
        raise ValueError('old component/Router sample IDs differ')
    groups = {}
    allocated = []
    for role, key in (('train', 'train_indices'), ('validation', 'validation_indices')):
        for index in router[key]:
            if type(index) is not int or not 0 <= index < len(labels['samples']):
                raise ValueError('invalid old Router split index')
            entry = by_id[labels['samples'][index]['sample_id']]
            group = (entry['dataset'], entry['sample']['sequence'])
            if group in groups and groups[group] != role:
                raise ValueError('old Router split leaks the same sequence across train/validation')
            groups[group] = role
            allocated.append(index)
    if sorted(allocated) != list(range(len(labels['samples']))):
        raise ValueError('old Router groups do not form a complete disjoint split')
    return entries, groups, {k: dict(path=str(v.resolve()), sha256=full.digest(v)) for k, v in paths.items()}


def uvg_crop_record(row, provenance, *, evaluation=False):
    if evaluation:
        sequence = row['official_sequence']
        selected = row['evaluation_sample']
        files, start, count = selected['png_files'], selected['frame_start'], selected['frame_count']
        crop = dict(selected['crop'])
        sample_id = f'uvg-{sequence.lower()}-f{start:03d}-historical-evaluation-crop'
        expected_hash = None
    else:
        sequence = row['sequence']
        files, start, count = row['source_files'], row['frame_start'], row['frame_count']
        crop = dict(row['original_crop'])
        sample_id = row['sample_id']
        expected_hash = row['selected_source_sha256']
    if sequence not in full.UVG_NAMES or count != 17 or len(files) != count or start < 0:
        raise ValueError('UVG cache provenance does not identify a supported 17-frame crop')
    original = row['original_format']
    if (original['width'], original['height']) != (1920, 1080):
        raise ValueError('unexpected UVG original format')
    if (crop['width'], crop['height']) != (512, 512) or min(crop['x'], crop['y']) < 0:
        raise ValueError('expected recorded native-resolution 512px UVG crop')
    if crop['x']+512 > 1920 or crop['y']+512 > 1080:
        raise ValueError('UVG crop outside recorded original frame')
    paths = [Path(p) for p in files]
    if [p.name for p in paths] != [f'{i:08d}.png' for i in range(count)]:
        raise ValueError('UVG cache needs consecutive zero-based frame names')
    if len({p.parent.resolve() for p in paths}) != 1:
        raise ValueError('UVG cache files must share one recorded sequence directory')
    if any(not p.is_file() for p in paths):
        raise ValueError('missing UVG crop frame')
    if any(full.image_size(p) != [512, 512] for p in (paths[0], paths[-1])):
        raise ValueError('UVG stored-view dimensions differ from crop provenance')
    return dict(sample_id=sample_id, dataset='UVG', sequence=sequence,
        split='historical_evaluation' if evaluation else 'adaptation',
        original_frame_start=start, frame_count=count, original_frame_size=[1920, 1080],
        original_crop=crop, view_kind='existing_spatial_crop', whole_frame=False,
        original_full_view_available=False, global_context_scope='within available 512px crop only',
        transform=full.geometry((512, 512), mode='native'),
        source=dict(kind='png_sequence', files=[str(p.resolve()) for p in paths],
                    file_sizes=[p.stat().st_size for p in paths]),
        provenance=provenance, expected_concatenated_png_sha256=expected_hash,
        component_training_sequence=sequence in full.UVG_TRAIN,
        history='historical UVG crop; five sequence families used for E/G training; all seven first windows evaluated before',
        independent_system_test=False)


def reds_record(row, groups, *, evaluation=False):
    row = dict(row)
    row.update(view_kind='resized_full_frame', original_frame_size=[1280, 720],
        original_full_view_available=True, global_context_scope='complete original field of view',
        independent_system_test=False)
    if evaluation:
        row['router_split'] = 'evaluation'
        row['component_training_sequence'] = False
    else:
        row['router_split'] = groups.get(('REDS', row['sequence']), 'train')
        row['role'] = 'train' if row['router_split'] == 'train' else 'development'
        row['component_training_sequence'] = ('REDS', row['sequence']) in groups
    return row


def set_probabilities(samples, uvg_probability):
    if not math.isfinite(uvg_probability) or not 0 < uvg_probability < 1:
        raise ValueError('UVG training probability must lie strictly between zero and one')
    counts = Counter(s['dataset'] for s in samples if s['router_split'] == 'train')
    if not counts['REDS'] or not counts['UVG']:
        raise ValueError('mixed training requires available REDS and UVG training samples')
    for row in samples:
        mass = uvg_probability if row['dataset'] == 'UVG' else 1-uvg_probability
        row['training_sample_probability'] = mass/counts[row['dataset']] if row['router_split'] == 'train' else 0.


def build_manifest(root=full.DEFAULT_ROOT, *, profile='pilot120', include_evaluation=True,
                   uvg_probability=.25):
    if profile not in ('pilot120', 'all_available'):
        raise ValueError('unknown mixed-view sampling profile')
    root = Path(root)
    old, groups, dependencies = previous_layout(root)
    # No full-UVG inventory is required to make progress: these are explicitly
    # existing cropped inputs, not substitutes in the full-view-only ledger.
    ledger = root/'data/UVG_adaptation/uvg_adaptation_samples.jsonl'
    rows = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        sidecar = full.read(root/'data/UVG_adaptation/samples'/entry['sample_id']/'source_manifest.json')
        for key in ('sample_id', 'sequence', 'source_files', 'frame_start', 'frame_count',
                    'original_crop', 'selected_source_sha256', 'original_format'):
            if entry[key] != sidecar[key]:
                raise ValueError(f'UVG sidecar differs from dataset ledger: {entry["sample_id"]}/{key}')
        rows.append(sidecar)
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate UVG adaptation sample IDs')
    if {r['sequence'] for r in rows} != full.UVG_TRAIN:
        raise ValueError('existing UVG training-side sequence families changed')
    dependencies['uvg_adaptation'] = dict(path=str(ledger.resolve()), sha256=full.digest(ledger))
    pool = full.build_manifest(root, splits=('train', 'val') if include_evaluation else ('train',))
    reds = {r['sample_id']: r for r in pool['samples'] if r['dataset'] == 'REDS'}
    samples = []
    if profile == 'pilot120':
        for entry in old:
            if entry['dataset'] == 'REDS':
                source = entry['sample']
                sid = f"reds-train-{source['sequence']}-f{source['frame_start']:03d}-n17-fullview"
                if sid not in reds:
                    raise ValueError('old REDS source window unavailable as a full-frame input')
                row = reds_record(reds[sid], groups)
                row['historical_crop_sample_id'] = entry['sample_id']
                row['historical_teacher_sample_id'] = entry['sample_id']
                samples.append(row)
            elif entry['dataset'] == 'UVG':
                if entry['sample_id'] not in by_id:
                    raise ValueError('old UVG crop missing from existing adaptation ledger')
                row = by_id[entry['sample_id']]
                path = root/'data/UVG_adaptation/samples'/row['sample_id']/'source_manifest.json'
                sample = uvg_crop_record(row, dict(path=str(path.resolve()), sha256=full.digest(path)))
                sample['historical_teacher_sample_id'] = entry['sample_id']
                samples.append(sample)
            else:
                raise ValueError('unexpected old component dataset')
    else:
        samples.extend(reds_record(row, groups) for row in reds.values() if row['split'] == 'train')
        for row in rows:
            path = root/'data/UVG_adaptation/samples'/row['sample_id']/'source_manifest.json'
            samples.append(uvg_crop_record(row, dict(path=str(path.resolve()), sha256=full.digest(path))))
    for row in samples:
        if row['dataset'] == 'UVG':
            row['router_split'] = groups[('UVG', row['sequence'])]
            row['role'] = 'train' if row['router_split'] == 'train' else 'development'
    if include_evaluation:
        samples.extend(reds_record(row, groups, evaluation=True) for row in reds.values() if row['split'] == 'val')
        for name in full.UVG_NAMES:
            path = root/'assets/evaluation/UVG'/name/'source_manifest.json'
            row = uvg_crop_record(full.read(path), dict(path=str(path.resolve()), sha256=full.digest(path)), evaluation=True)
            row.update(router_split='evaluation', role='evaluation',
                evaluation_subgroup='component_training_sequence' if name in full.UVG_TRAIN else 'sequence_not_used_for_component_training')
            samples.append(row)
    ids = [s['sample_id'] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate mixed-view sample IDs')
    set_probabilities(samples, uvg_probability)
    return dict(format=FORMAT, profile=profile, samples=samples, dependencies=dependencies,
        code={p.name: full.digest(p) for p in (Path(__file__), Path(full.__file__))},
        counts=dict(Counter(f"{s['dataset']}/{s['router_split']}/{s['view_kind']}" for s in samples)),
        sampling=dict(uvg_probability=uvg_probability, reds_probability=1-uvg_probability,
                      scope='train only, equal sample probability within each dataset; development/evaluation probability zero'),
        split=dict(preserved_old_router_groups=[dict(dataset=k[0], sequence=k[1], router_split=v)
                    for k, v in sorted(groups.items())],
            note='Router development split only; existing E/G models used both sides; no claim of unseen whole-system evidence'),
        full_uvg_upload_required=False,
        reporting='Report REDS resized full-frame and UVG spatial crops separately; no undifferentiated full-video claim.',
        labels='Source manifest only. Changed REDS field of view requires new measured labels; old cropped utilities are not transferred.',
        scope='No training, downloads, original-source deletion or full-frame UVG claim')


def verify_crop(sample):
    if sample['view_kind'] != 'existing_spatial_crop' or sample['whole_frame']:
        raise ValueError('expected explicit UVG crop')
    hashes = full.source_hash(sample)
    combined = hashlib.sha256()
    for path in sample['source']['files']:
        if full.image_size(path) != [512, 512]:
            raise ValueError('UVG stored crop shape changed')
        combined.update(Path(path).read_bytes())
    expected = sample['expected_concatenated_png_sha256']
    if expected is not None and combined.hexdigest() != expected:
        raise ValueError('UVG crop no longer matches historical source content hash')
    return hashes


def load_rgb(sample):
    """Load only the recorded available view; no invented UVG off-crop pixels."""
    if sample['view_kind'] == 'existing_spatial_crop':
        verify_crop(sample)
    elif sample['view_kind'] != 'resized_full_frame' or not sample['whole_frame']:
        raise ValueError('unsupported view kind')
    return np.stack([np.asarray(full.transform_frame(image, sample['transform']))
                     for image in full.source_images(sample)])


def prepare_manifest(manifest, output, *, sample_ids=(), router_splits=('train', 'validation'),
                     limit=0, max_hours=4.):
    """Prepare REDS views; verify and reuse UVG source PNGs without duplicating them."""
    manifest, output = Path(manifest), Path(output)
    value = full.read(manifest)
    if value['format'] != FORMAT or limit < 0 or not math.isfinite(max_hours) or max_hours <= 0:
        raise ValueError('invalid mixed-view preparation configuration')
    selected = [s for s in value['samples'] if s['router_split'] in router_splits
                and (not sample_ids or s['sample_id'] in sample_ids)]
    if sample_ids and set(sample_ids)-{s['sample_id'] for s in selected}:
        raise ValueError('unknown sample or excluded Router split')
    selected = selected[:limit or None]
    if not selected:
        raise ValueError('no mixed-view samples selected')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', s['sample_id']) for s in selected):
        raise ValueError('invalid sample ID')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(manifest=str(manifest.resolve()), manifest_sha256=full.digest(manifest),
                       sample_ids=[s['sample_id'] for s in selected], code=value['code'])
        if binding['code'] != {p.name: full.digest(p) for p in (Path(__file__), Path(full.__file__))}:
            raise ValueError('mixed-view implementation changed after manifest was frozen')
        full.immutable(output/'request.json', binding)
        began, last = time.monotonic(), 0.

        def check(force=False):
            nonlocal last
            now = time.monotonic()
            if now-began > max_hours*3600:
                raise TimeoutError('mixed-view preparation deadline reached; resume same command')
            if force or now-last >= 30:
                disks = {}
                for path in ('/root', '/root/autodl-tmp', '/root/autodl-fs'):
                    if Path(path).exists():
                        use = shutil.disk_usage(path)
                        disks[path] = dict(total=use.total, used=use.used, free=use.free)
                        if path != '/root' and use.used/use.total >= .8:
                            raise RuntimeError('data disk at least80% used; preparation paused')
                try:
                    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                        '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
                except (OSError, subprocess.SubprocessError):
                    gpu = 'unavailable; preparation uses no GPU'
                resource = dict(elapsed_seconds=now-began, disks=disks, gpu_observation=gpu, GPU_used=False)
                full.atomic_json(output/'resources.json', resource)
                print(json.dumps(dict(heartbeat=resource)), flush=True)
                last = now

        results = []
        for sample in selected:
            check(force=True)
            folder = output/'samples'/sample['sample_id']
            if sample['view_kind'] == 'resized_full_frame':
                made = full.prepare_sample(sample, folder/'fullview', check)
                record = dict(sample_id=sample['sample_id'], frames_dir=made['frames_dir'],
                    view_kind=sample['view_kind'], router_split=sample['router_split'],
                    source_hashes=made['binding']['source_hashes'], transform=sample['transform'],
                    fullview_complete_sha256=full.digest(folder/'fullview/complete.json'))
            else:
                hashes = verify_crop(sample)
                record = dict(sample_id=sample['sample_id'], frames_dir=str(Path(sample['source']['files'][0]).parent),
                    view_kind=sample['view_kind'], router_split=sample['router_split'], source_hashes=hashes,
                    transform=sample['transform'], original_crop=sample['original_crop'], source_pngs_reused_without_copy=True)
            record.update(complete=True, sample=sample)
            full.immutable(folder/'view.json', record)
            results.append(dict(sample_id=sample['sample_id'], frames_dir=record['frames_dir'],
                view_json=str((folder/'view.json').resolve()), view_sha256=full.digest(folder/'view.json')))
            print(json.dumps(dict(prepared=len(results), total=len(selected), sample=sample['sample_id'])), flush=True)
        done = dict(complete=True, format=FORMAT, binding=binding, samples=results, full_uvg_upload_required=False)
        full.immutable(output/'complete.json', done)
        return done


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    build = commands.add_parser('manifest')
    build.add_argument('--root', type=Path, default=full.DEFAULT_ROOT)
    build.add_argument('--profile', choices=('pilot120', 'all_available'), default='pilot120')
    build.add_argument('--training-only', action='store_true')
    build.add_argument('--uvg-probability', type=float, default=.25)
    build.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--sample-ids', nargs='*', default=[])
    prepare.add_argument('--router-splits', nargs='+', choices=('train', 'validation', 'evaluation'),
                         default=['train', 'validation'])
    prepare.add_argument('--limit', type=int, default=0)
    prepare.add_argument('--max-hours', type=float, default=4.)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'manifest':
        record = build_manifest(args.root, profile=args.profile, include_evaluation=not args.training_only,
                                uvg_probability=args.uvg_probability)
        full.immutable(args.output, record)
        print(json.dumps(dict(counts=record['counts'], full_uvg_upload_required=False)), flush=True)
    else:
        if not os.environ.get('TMUX'):
            raise RuntimeError('long mixed-view preparation requires tmux')
        prepare_manifest(args.manifest, args.output, sample_ids=args.sample_ids,
                         router_splits=args.router_splits, limit=args.limit, max_hours=args.max_hours)


if __name__ == '__main__':
    main()
