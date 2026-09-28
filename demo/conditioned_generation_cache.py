"""Actual ACSE prefix inputs for generation adaptation (no UF/E training)."""
import argparse
from collections import Counter
import gc
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_codec import (configure_torch, encode_enhancement,
    decode_enhancement, load_model)
from demo.chunk_enhancement_experiment import codec, read
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import parse, frame_hash
from demo.scalable_generation_decode import ASSETS, PATCH
from demo.stage_c_seedvr2_lora_finetune import (initialize_single_gpu,
    configure_frozen_vae, deterministic_vae_latent, atomic_torch_save)
from demo.scalable_experiment import check_space, resources

FEATURES = Path('/root/autodl-fs/DCVC/runs/a800_chunk_enhancement_20260926/features.json')
MODES = ('none', 'partial', 'full')


def verify(root, record):
    for name, digest in record['artifacts'].items():
        if file_hash(root / name) != digest:
            raise RuntimeError(f'cache artifact changed: {root / name}')


def entries(limit):
    rows = read(FEATURES)['samples']
    if limit:
        rows = [next(r for r in rows if r['sample']['dataset'] == d)
                for d in ('REDS', 'UVG')]
    return rows


def encode(args):
    configure_torch()
    root = args.output / 'streams'
    root.mkdir(parents=True, exist_ok=True)
    model = load_model(PATCH).requires_grad_(False)
    results = []
    protocol = dict(feature_manifest=file_hash(FEATURES), enhancement=file_hash(PATCH),
                    code=file_hash(Path(__file__)), smoke=args.smoke,
                    roi_size=256, qsteps=[.5, 1., 2.], prefixes=[4, 6, 8])
    if (root/'protocol.json').exists() and read(root/'protocol.json') != protocol:
        raise RuntimeError('cache protocol changed')
    atomic_json(root/'protocol.json', protocol)
    for index, row in enumerate(entries(args.smoke)):
        check_space()
        dest = root / row['sample_id']
        dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            result = read(dest/'complete.json')
            verify(dest, result)
        else:
            for path, key in [('pair_path', 'pair_hash'), ('base_path', 'base_file_hash'),
                              ('feature_path', 'feature_hash')]:
                if file_hash(Path(row[path])) != row[key]:
                    raise RuntimeError(f'input changed: {row[path]}')
            with np.load(row['pair_path']) as pair:
                source, base = pair['source'].copy(), pair['base'].copy()
            if source.shape != (17, 512, 512, 3):
                raise ValueError('expected mixed training 17x512x512 cache')
            chunks = torch.load(row['feature_path'], weights_only=True, map_location='cpu')['chunks']
            rois = [[x, y, 256, 256] for y in (0, 256) for x in (0, 256)]
            rois = rois[index % 4:] + rois[:index % 4]
            q = [.5, 1., 2.][index % 3]
            prefix, wires, full, details = encode_enhancement(model, PATCH,
                Path(row['base_path']).read_bytes(), source, base, chunks, rois, q, compact=True)
            # Order by region then time. A literal partial prefix consequently
            # mixes enhanced/unenhanced areas AND incomplete temporal chunks.
            wires = [wires[t*4+r] for r in range(4) for t in range(3)]
            count = [4, 6, 8][index % 3]
            partial = base.copy()
            for packet in parse(prefix+b''.join(wires[:count])).packets:
                m = packet.meta
                x, y, w, h = m['roi']; t, n = m['start'], m['count']
                partial[t:t+n, y:y+h, x:x+w] = full[t:t+n, y:y+h, x:x+w]
            streams = {'none': prefix, 'partial': prefix+b''.join(wires[:count]),
                       'full': prefix+b''.join(wires)}
            expected = {'none': base, 'partial': partial, 'full': full}
            for mode, wire in streams.items():
                atomic_bytes(dest/f'{mode}.acse', wire)
            assert streams['full'].startswith(streams['partial'])
            assert streams['partial'].startswith(streams['none'])
            result = dict(sample=row['sample'], sample_id=row['sample_id'],
                dataset=row['sample']['dataset'], pair_path=row['pair_path'], pair_hash=row['pair_hash'],
                q=q, rois=rois, partial_packets=count, full_packets=len(wires),
                expected={m:frame_hash(v) for m,v in expected.items()},
                bytes={m:len(v) for m,v in streams.items()},
                artifacts={f'{m}.acse':file_hash(dest/f'{m}.acse') for m in MODES})
            atomic_json(dest/'complete.json', result)
        results.append(result)
        print(f'ENCODE {index+1}/{len(entries(args.smoke))} {row["sample_id"]}', flush=True)
    atomic_json(root/'index.json', dict(entries=results, protocol=protocol))


def receive(args):
    """Source-free worker: read only streams, models and output-hash assertions."""
    configure_torch()
    model, base_codec = load_model(PATCH), codec()
    rows = read(args.output/'streams/index.json')['entries']
    for index, row in enumerate(rows):
        check_space()
        streams = args.output/'streams'/row['sample_id']
        dest = args.output/'received'/row['sample_id']
        dest.mkdir(parents=True, exist_ok=True)
        if (dest/'complete.json').exists():
            verify(dest, read(dest/'complete.json'))
        else:
            verify(streams, row)
            images, reports = {}, {}
            for mode in MODES:
                wire = (streams/f'{mode}.acse').read_bytes()
                images[mode], reports[mode] = decode_enhancement(model, PATCH, base_codec, wire)
                if frame_hash(images[mode]) != row['expected'][mode]:
                    raise RuntimeError(f'fresh prefix reconstruction mismatch: {mode}')
            atomic_npz(dest/'conditions.npz', **images)
            atomic_json(dest/'complete.json', dict(source_frames_read=False, reports=reports,
                artifacts={'conditions.npz':file_hash(dest/'conditions.npz')}))
        print(f'RECEIVE {index+1}/{len(rows)} {row["sample_id"]}', flush=True)


def latents(args):
    args.upstream_root = REPO/'third_party/SeedVR2'
    args.vae_checkpoint = ASSETS['vae']
    device = initialize_single_gpu(0)
    vae, config = configure_frozen_vae(args, device)
    root = args.output/'latents'
    root.mkdir(exist_ok=True)
    rows, results = read(args.output/'streams/index.json')['entries'], []
    for index, row in enumerate(rows):
        check_space()
        path = root/f'{row["sample_id"]}.pt'
        meta = path.with_suffix('.json')
        if meta.exists():
            record = read(meta)
            if file_hash(path) != record['sha256']:
                raise RuntimeError('latent cache changed')
        else:
            dest = args.output/'received'/row['sample_id']
            verify(dest, read(dest/'complete.json'))
            if file_hash(Path(row['pair_path'])) != row['pair_hash']:
                raise RuntimeError('source target cache changed')
            with np.load(row['pair_path']) as pair:
                source = pair['source'].copy()
            payload = {'clean':deterministic_vae_latent(list(source), vae, config, device)}
            with np.load(dest/'conditions.npz') as conditions:
                for mode in MODES:
                    payload[mode] = deterministic_vae_latent(list(conditions[mode]), vae, config, device)
            atomic_torch_save(path, payload)
            record = dict(sample_id=row['sample_id'], dataset=row['dataset'], path=str(path),
                sha256=file_hash(path), pair_path=row['pair_path'], pair_hash=row['pair_hash'],
                vae_sha256=file_hash(ASSETS['vae']), posterior='mode', shape=list(payload['clean'].shape))
            atomic_json(meta, record)
            del payload, source
        results.append(record)
        print(f'LATENT {index+1}/{len(rows)} {row["sample_id"]}', flush=True)
    summary = dict(entries=results, complete=True,
        datasets=dict(Counter(r['dataset'] for r in results)), modes=list(MODES),
        stream_index_hash=file_hash(args.output/'streams/index.json'))
    destination = args.output/'cache.json'
    if destination.exists():
        if read(destination) != summary:
            raise RuntimeError('completed cache manifest changed')
    else:
        atomic_json(destination, summary)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['encode', 'receive', 'latents'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    globals()[args.command](args)
