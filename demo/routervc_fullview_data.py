"""CPU-only, explicit whole-frame inputs for RouterVC development/evaluation.

No downloads, archive extraction, source deletion, hidden cropping, training or
GPU work. The manifest distinguishes historical 512px crops from full originals.
Preparation preserves the full field of view, records resize/padding/valid area,
and resumes at verified sample boundaries. Invoke long preparation in tmux.
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


FORMAT = 'routervc_fullview_data_v1'
DEFAULT_ROOT = Path('/root/autodl-fs/DCVC')
UVG_NAMES = ('Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySetGo', 'ShakeNDry', 'YachtRide')
UVG_ARCHIVE_BYTES = (925430047, 680772328, 906770507, 770631599, 832143797, 460046003, 724220168)
UVG_TRAIN = frozenset(('Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ShakeNDry'))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'changed configuration: {path}; use a new output path')
    else:
        atomic_json(path, value)


def geometry(original, mode='fit', width=1024, height=576, alignment=64):
    """Return exact-aspect integer fit plus explicit centered edge padding.

    Native mode never resizes. Fit uses a multiple of the reduced aspect ratio,
    avoiding even a small aspect distortion. All evaluations must compare the
    same transform and exclude padding from metrics and the bpp denominator.
    """
    ow, oh = original
    if any(type(v) is not int or v <= 0 for v in (ow, oh, width, height, alignment)):
        raise ValueError('geometry dimensions and alignment must be positive integers')
    if mode == 'native':
        rw, rh = ow, oh
        cw, ch = math.ceil(ow/alignment)*alignment, math.ceil(oh/alignment)*alignment
    elif mode == 'fit':
        if width % alignment or height % alignment:
            raise ValueError('explicit fit canvas must be a multiple of alignment')
        common = math.gcd(ow, oh)
        aw, ah = ow//common, oh//common
        scale = min(width//aw, height//ah)
        if scale < 1:
            raise ValueError('canvas too small to preserve exact source aspect ratio')
        rw, rh = aw*scale, ah*scale
        cw, ch = width, height
    else:
        raise ValueError('mode must be fit or native')
    left, top = (cw-rw)//2, (ch-rh)//2
    return dict(original_size=[ow, oh], resized_size=[rw, rh], coded_size=[cw, ch],
        valid_rect=[left, top, rw, rh], padding=[left, top, cw-rw-left, ch-rh-top],
        mode=mode, crop=None, preserves_full_field_of_view=True,
        aspect_ratio_exact=rw*oh == rh*ow, alignment=alignment,
        resize_filter='Pillow LANCZOS' if [ow, oh] != [rw, rh] else 'none',
        padding_mode='edge', bpp_denominator='frame_count * valid_width * valid_height',
        metrics_scope='valid_rect only; exclude padding; use same transform for every method',
        resolution_claim='native pixels' if mode == 'native' else 'resized whole frame, not native resolution')


def transform_frame(image, spec):
    if list(image.size) != spec['original_size']:
        raise ValueError('source dimensions changed or vary across frames')
    image = image.convert('RGB')
    if list(image.size) != spec['resized_size']:
        image = image.resize(tuple(spec['resized_size']), Image.Resampling.LANCZOS)
    values = np.asarray(image)
    left, top, right, bottom = spec['padding']
    if any(spec['padding']):
        values = np.pad(values, ((top, bottom), (left, right), (0, 0)), mode='edge')
    return Image.fromarray(values)


def png_files(folder):
    return sorted(Path(folder).glob('*.png')) if Path(folder).is_dir() else []


def image_size(path):
    with Image.open(path) as image:
        return list(image.size)


def uvg_inventory(root=DEFAULT_ROOT, full_root=None):
    root = Path(root)
    full_root = Path(full_root) if full_root else root/'data/UVG_full'
    rows = []
    for name, archive_bytes in zip(UVG_NAMES, UVG_ARCHIVE_BYTES):
        archive = f'{name}_1920x1080_120fps_420_8bit_YUV_RAW.7z'
        count = 300 if name == 'ShakeNDry' else 600
        raw_name = 'ReadySteadyGo' if name == 'ReadySetGo' else name
        # Some releases include dimensions in the raw filename. Only inspect
        # explicitly named, sequence-specific locations; never scan checkpoints.
        candidates = []
        for folder in (full_root/name, root/'downloads/uvg'/f'raw-{name}',
                       root/'downloads/uvg-adaptation'/f'raw-{name}'):
            if folder.is_dir():
                candidates.extend(sorted(folder.glob('*.yuv')))
        if full_root.is_dir():
            candidates.extend(sorted(full_root.glob(raw_name+'*.yuv')))
        raw = sorted({str(p.resolve()) for p in candidates
                      if p.is_file() and p.stat().st_size == 1920*1080*3//2*count})
        frames = png_files(full_root/name)
        complete_png = (len(frames) == count and image_size(frames[0]) == [1920, 1080]
                        and image_size(frames[-1]) == [1920, 1080])
        archives = [p for p in (root.parent/archive, root/'downloads/uvg'/archive,
                               root/'downloads/uvg-adaptation'/archive, full_root/archive)
                    if p.is_file()]
        rows.append(dict(sequence=name, frame_count=count, original_size=[1920, 1080], fps=120,
            pixel_format='yuv420p', role='train' if name in UVG_TRAIN else 'evaluation',
            history='all seven first cropped windows were evaluated historically; five sequences used in component training',
            official_archive=archive, official_url='https://ultravideo.fi/video/'+archive,
            historical_archive_bytes=archive_bytes,
            archive_paths=[str(p.resolve()) for p in archives],
            valid_size_archive_paths=[str(p.resolve()) for p in archives if p.stat().st_size == archive_bytes],
            full_png_dir=str((full_root/name).resolve()) if complete_png else None,
            full_png_count=len(frames), raw_yuv_paths=raw,
            full_original_available=bool(complete_png or raw),
            status='available' if complete_png or raw else 'archive_needs_full_extraction' if archives else 'missing_full_original',
            cropped_cache=str((root/'assets/evaluation/UVG'/name).resolve()),
            cropped_cache_is_full_original=False))
    return rows


def roles(dataset, split, sequence):
    if dataset == 'UVG':
        return ('train' if sequence in UVG_TRAIN else 'evaluation',
                'five adaptation sequences reused for training; other two not used for component training; historical evaluation exists')
    if split == 'train':
        return 'train', 'REDS official training split; historical component-training pool'
    if int(sequence) < 6:
        return 'development', 'REDS val000..005 repeatedly used for method development'
    return 'evaluation', 'REDS validation previously used in historical evaluations; not newly untouched evidence'


def sample_record(dataset, split, sequence, paths, start, count, transform, *, raw=None, frame_count=None):
    role, history = roles(dataset, split, sequence)
    if raw is None:
        if len(paths) < start+count:
            raise ValueError(f'not enough original frames: {dataset}/{sequence} start{start}')
        selected = paths[start:start+count]
        source = dict(kind='png_sequence', files=[str(p.resolve()) for p in selected],
                      file_sizes=[p.stat().st_size for p in selected],
                      hash_scope='all selected files hashed when prepared, no pixel content read by inventory')
        if any(image_size(p) != transform['original_size'] for p in (selected[0], selected[-1])):
            raise ValueError('source window dimensions inconsistent')
    else:
        if frame_count < start+count:
            raise ValueError('not enough raw frames')
        source = dict(kind='raw_yuv420p', path=str(Path(raw).resolve()),
            bytes=Path(raw).stat().st_size, fps=120, total_frames=frame_count,
            hash_scope='selected raw frame bytes hashed when prepared; no guessed conversion')
    return dict(sample_id=f'{dataset.lower()}-{split}-{sequence}-f{start:03d}-n{count}-fullview',
                dataset=dataset, split=split, sequence=sequence, role=role, history=history,
                original_frame_start=start, frame_count=count, source=source, transform=transform,
                original_crop=None, whole_frame=True, source_content_hash_pending_preparation=True)


def build_manifest(root=DEFAULT_ROOT, *, starts=(0, 40), count=17, mode='fit', width=1024,
                   height=576, alignment=64, splits=('train', 'val'), full_uvg_root=None,
                   limit_sequences=0):
    root = Path(root)
    if (not starts or len(set(starts)) != len(starts)
            or any(type(v) is not int or v < 0 for v in starts)
            or type(count) is not int or count < 17 or (count-1) % 8):
        raise ValueError('unique nonnegative starts and a count=1+8n >=17 are required')
    if not splits or any(s not in ('train', 'val') for s in splits) or len(set(splits)) != len(splits):
        raise ValueError('select unique REDS train/val splits')
    if limit_sequences < 0:
        raise ValueError('sequence limit must be nonnegative')
    entries, inventory, missing = [], [], []
    for split in splits:
        folder = root/'data/REDS'/f'{split}_sharp'
        sequences = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()) if folder.is_dir() else []
        expected = 240 if split == 'train' else 30
        inventory.append(dict(dataset='REDS', split=split, path=str(folder.resolve()),
                              sequences=len(sequences), expected_sequences=expected))
        expected_names = {f'{i:03d}' for i in range(expected)}
        actual_names = {p.name for p in sequences}
        for name in sorted(expected_names-actual_names):
            missing.append(dict(dataset='REDS', split=split, sequence=name, reason='missing original sequence'))
        if actual_names-expected_names:
            raise ValueError('unexpected REDS sequence directory')
        for sequence in sequences[:limit_sequences or None]:
            frames = png_files(sequence)
            if not frames or image_size(frames[0]) != [1280, 720]:
                raise ValueError(f'not original full REDS 1280x720: {sequence}')
            if len(frames) != 100 or [p.name for p in frames] != [f'{i:08d}.png' for i in range(100)]:
                raise ValueError(f'incomplete/reordered REDS timeline: {sequence}')
            spec = geometry((1280, 720), mode, width, height, alignment)
            for start in starts:
                entries.append(sample_record('REDS', split, sequence.name, frames, start, count, spec))
    uvg = uvg_inventory(root, full_uvg_root)
    for row in uvg:
        if not row['full_original_available']:
            missing.append(dict(dataset='UVG', sequence=row['sequence'], reason=row['status'],
                                official_url=row['official_url'], official_archive=row['official_archive']))
            continue
        frames = png_files(row['full_png_dir']) if row['full_png_dir'] else []
        if frames and [p.name for p in frames] != [f'{i:08d}.png' for i in range(row['frame_count'])]:
            raise ValueError('full UVG PNGs must use complete zero-based eight-digit frame names')
        spec = geometry((1920, 1080), mode, width, height, alignment)
        for start in starts:
            entries.append(sample_record('UVG', 'adaptation' if row['role'] == 'train' else 'heldout',
                row['sequence'], frames, start, count, spec,
                raw=None if frames else row['raw_yuv_paths'][0], frame_count=row['frame_count']))
    return dict(format=FORMAT, source_root=str(root.resolve()), parameters=dict(starts=list(starts),
        count=count, mode=mode, width=width, height=height, alignment=alignment,
        reds_splits=list(splits), limit_sequences=limit_sequences), samples=entries,
        counts=dict(Counter(f"{r['dataset']}/{r['role']}" for r in entries)),
        reds_inventory=inventory, uvg_inventory=uvg, missing_sources=missing,
        coverage='partial dataset availability' if missing else 'requested originals available',
        metric_contract='Never use coded padded pixel count for valid-area bpp; exclude padding from quality metrics.',
        scope='source preparation only, no training, metrics, codec, generator or independent-test claim')


def source_hash(sample):
    source = sample['source']
    if source['kind'] == 'png_sequence':
        if [Path(p).stat().st_size for p in source['files']] != source['file_sizes']:
            raise ValueError('source file size changed after manifest creation')
        return dict(files={p: digest(p) for p in source['files']})
    if source['kind'] != 'raw_yuv420p':
        raise ValueError('unsupported original source format')
    path = Path(source['path'])
    if path.stat().st_size != source['bytes']:
        raise ValueError('raw source size changed')
    w, h = sample['transform']['original_size']
    frame_bytes = w*h*3//2
    size = frame_bytes*sample['frame_count']
    value = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(frame_bytes*sample['original_frame_start'])
        remaining = size
        while remaining:
            block = stream.read(min(1024*1024, remaining))
            if not block:
                raise ValueError('truncated raw source window')
            value.update(block)
            remaining -= len(block)
    return dict(raw_window_sha256=value.hexdigest(), raw_offset=frame_bytes*sample['original_frame_start'],
                raw_length=size)


def source_images(sample):
    source = sample['source']
    if source['kind'] == 'png_sequence':
        for path in source['files']:
            with Image.open(path) as image:
                yield image.convert('RGB')
        return
    w, h = sample['transform']['original_size']
    offset = sample['original_frame_start']*w*h*3//2
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'yuv420p',
        '-video_size', f'{w}x{h}', '-framerate', '120', '-skip_initial_bytes', str(offset),
        '-i', source['path'], '-frames:v', str(sample['frame_count']), '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    expected = sample['frame_count']*w*h*3
    if len(result.stdout) != expected:
        raise ValueError('raw conversion produced an unexpected number of RGB pixels')
    values = np.frombuffer(result.stdout, dtype=np.uint8).reshape(sample['frame_count'], h, w, 3)
    for frame in values:
        yield Image.fromarray(frame)


def prepare_sample(sample, output, check=lambda: None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sample['sample_id']):
        raise ValueError('invalid sample ID')
    spec = sample['transform']
    expected = geometry(tuple(spec['original_size']), spec['mode'], *spec['coded_size'], spec['alignment'])
    if spec != expected or sample['original_crop'] is not None or not sample['whole_frame']:
        raise ValueError('manifest must describe an explicit uncropped full-view transform')
    if type(sample['frame_count']) is not int or sample['frame_count'] < 17 or (sample['frame_count']-1) % 8:
        raise ValueError('sample frame count must be 1+8n and at least17')
    if sample['source']['kind'] == 'png_sequence' and len(sample['source']['files']) != sample['frame_count']:
        raise ValueError('sample PNG timeline length does not match manifest')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(sample=sample, source_hashes=source_hash(sample), code_sha256=digest(__file__),
                   pillow_version=Image.__version__, numpy_version=np.__version__)
    if sample['source']['kind'] == 'raw_yuv420p':
        binding['ffmpeg_version'] = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    done = output/'complete.json'
    if done.exists():
        record = read(done)
        if record['binding'] != binding:
            raise ValueError('prepared source or configuration changed; choose a new output directory')
        for name, expected in record['artifacts'].items():
            if digest(output/name) != expected:
                raise ValueError(f'prepared artifact changed: {output/name}')
        return record
    immutable(output/'request.json', binding)
    frames = output/'frames'
    frames.mkdir(exist_ok=True)
    artifacts = {'request.json': digest(output/'request.json')}
    begun = time.monotonic()
    count = 0
    for index, image in enumerate(source_images(sample)):
        check()
        prepared = transform_frame(image, sample['transform'])
        path = frames/f'{index:08d}.png'
        temporary = path.with_suffix('.png.tmp')
        prepared.save(temporary, format='PNG')
        if path.exists():
            if digest(path) != digest(temporary):
                temporary.unlink()
                raise ValueError('incomplete sample contains changed pixels; choose a new output')
            temporary.unlink()
        else:
            os.replace(temporary, path)
        artifacts[str(path.relative_to(output))] = digest(path)
        count += 1
    if count != sample['frame_count']:
        raise ValueError('prepared frame count mismatch')
    _, _, valid_w, valid_h = sample['transform']['valid_rect']
    record = dict(complete=True, format=FORMAT, sample_id=sample['sample_id'], binding=binding,
        frames_dir=str(frames.resolve()), frame_count=count, transform=sample['transform'],
        valid_video_pixels=count*valid_w*valid_h, artifacts=artifacts,
        preparation_seconds=time.monotonic()-begun,
        ordinary_file_bytes=sum((output/name).stat().st_size for name in artifacts),
        note='Padding-aware metric evaluation is required outside this preparation utility.')
    atomic_json(done, record)
    return record


def prepare_manifest(manifest, output, *, sample_ids=(), selected_roles=(), limit=0, max_hours=2.):
    manifest, output = Path(manifest), Path(output)
    data = read(manifest)
    if data['format'] != FORMAT:
        raise ValueError('unsupported full-view manifest')
    if not math.isfinite(max_hours) or max_hours <= 0 or limit < 0:
        raise ValueError('positive runtime and nonnegative sample limit required')
    selected = [s for s in data['samples'] if (not sample_ids or s['sample_id'] in sample_ids)
                and (not selected_roles or s['role'] in selected_roles)]
    if sample_ids and set(sample_ids)-{s['sample_id'] for s in selected}:
        raise ValueError('unknown sample ID or excluded sample role')
    selected = selected[:limit or None]
    if not selected:
        raise ValueError('no available full-original samples selected')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = dict(manifest=str(manifest.resolve()), manifest_sha256=digest(manifest),
                       sample_ids=[s['sample_id'] for s in selected], code_sha256=digest(__file__))
        immutable(output/'request.json', request)
        begun, heartbeat = time.monotonic(), 0.

        def check(force=False):
            nonlocal heartbeat
            now = time.monotonic()
            if now-begun > max_hours*3600:
                raise TimeoutError('preparation deadline reached; same command resumes complete samples')
            if force or now-heartbeat >= 30:
                disks = {}
                for mount in ('/root', '/root/autodl-tmp', '/root/autodl-fs'):
                    if not Path(mount).exists():
                        continue
                    use = shutil.disk_usage(mount)
                    disks[mount] = dict(total=use.total, used=use.used, free=use.free)
                    if mount != '/root' and use.used/use.total >= .8:
                        raise RuntimeError(f'data disk at least 80% full: {mount}')
                record = dict(elapsed_seconds=now-begun, disks=disks, GPU_used=False,
                              pid=os.getpid(), phase='whole_frame_CPU_preparation')
                try:
                    record['gpu_observation'] = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                         '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL,
                        timeout=5).strip()
                except (OSError, subprocess.SubprocessError):
                    record['gpu_observation'] = 'unavailable; this preparation itself uses no GPU'
                atomic_json(output/'resources.json', record)
                print(json.dumps(dict(heartbeat=record)), flush=True)
                heartbeat = now

        results = []
        for sample in selected:
            check(force=True)
            result = prepare_sample(sample, output/'samples'/sample['sample_id'], check)
            results.append(dict(sample_id=sample['sample_id'], frames_dir=result['frames_dir'],
                complete_json=str((output/'samples'/sample['sample_id']/'complete.json').resolve())))
            print(json.dumps(dict(prepared=len(results), total=len(selected), sample=sample['sample_id'])), flush=True)
        record = dict(complete=True, format=FORMAT, request=request, samples=results,
            scope='only explicitly selected available sources; missing UVG is not silently replaced',
            unavailable_sources=data['missing_sources'])
        immutable(output/'complete.json', record)
        return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    inventory = commands.add_parser('inventory')
    inventory.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    inventory.add_argument('--full-uvg-root', type=Path)
    inventory.add_argument('--output', type=Path, required=True)
    inventory.add_argument('--splits', nargs='+', choices=('train', 'val'), default=['train', 'val'])
    inventory.add_argument('--starts', nargs='+', type=int, default=[0, 40])
    inventory.add_argument('--count', type=int, default=17)
    inventory.add_argument('--mode', choices=('fit', 'native'), default='fit')
    inventory.add_argument('--width', type=int, default=1024)
    inventory.add_argument('--height', type=int, default=576)
    inventory.add_argument('--alignment', type=int, default=64)
    inventory.add_argument('--limit-sequences', type=int, default=0)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--sample-ids', nargs='*', default=[])
    prepare.add_argument('--roles', nargs='*', choices=('train', 'development', 'evaluation'), default=[])
    prepare.add_argument('--limit', type=int, default=0)
    prepare.add_argument('--max-hours', type=float, default=2.)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'inventory':
        data = build_manifest(args.root, starts=args.starts, count=args.count, mode=args.mode,
            width=args.width, height=args.height, alignment=args.alignment, splits=args.splits,
            full_uvg_root=args.full_uvg_root, limit_sequences=args.limit_sequences)
        immutable(args.output, data)
        print(json.dumps(dict(manifest=str(args.output), samples=len(data['samples']),
                              counts=data['counts'], missing_sources=len(data['missing_sources']))), flush=True)
    else:
        if not os.environ.get('TMUX'):
            raise RuntimeError('long source preparation must run in tmux')
        prepare_manifest(args.manifest, args.output, sample_ids=args.sample_ids,
            selected_roles=args.roles, limit=args.limit, max_hours=args.max_hours)


if __name__ == '__main__':
    main()
