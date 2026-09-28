"""Source-free extraction of the delta already synthesized by an E decoder.

Observe the original operation without modifying its output or old receiver.
FP16 is used only for the new conditioning side channel, never for E's RGB.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_codec import configure_torch, decode_enhancement, load_model
from demo.chunk_enhancement_experiment import codec, read
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import parse, frame_hash
from demo.scalable_generation_decode import PATCH
from demo.scalable_experiment import check_space
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save

PREVIOUS = Path('/root/autodl-fs/DCVC/runs/a800_conditioned_generation_20260928')


def receive_features(model, model_path, base_codec, data):
    parsed = parse(data)
    pending = iter(parsed.packets)
    records, captured = [], []
    original = model.decompress
    def capture(_module, _inputs, output):
        captured.append(output.detach().to(device='cpu', dtype=torch.float16).clone())
    hook = model.feature_synthesis.register_forward_hook(capture)
    def decompress(*args, **kwargs):
        packet = next(pending)
        captured.clear()
        result = original(*args, **kwargs)
        if len(captured) != int(packet.meta['start'] != 0):
            raise RuntimeError('unexpected feature synthesis branch')
        records.append(dict(packet.meta, delta=captured[0] if captured else None))
        return result
    model.decompress = decompress
    try:
        result = decode_enhancement(model, model_path, base_codec, data, return_base=True)
    finally:
        model.decompress = original
        hook.remove()
    if len(records) != len(parsed.packets):
        raise RuntimeError('packet extraction mismatch')
    return (*result, records)


def selected(records, ids):
    lookup = {p['packet_id']:p for p in records}
    if len(lookup) != len(records) or not set(ids).issubset(lookup):
        raise ValueError('invalid cached packet selection')
    return [lookup[i] for i in ids]


def main(args):
    configure_torch()
    latent = read(PREVIOUS/'cache.json')
    rows = read(PREVIOUS/'streams/index.json')['entries']
    if args.smoke:
        rows = [next(r for r in rows if r['dataset'] == d) for d in ('REDS','UVG')]
    by_id = {r['sample_id']:r for r in latent['entries']}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(code=file_hash(Path(__file__)), previous=file_hash(PREVIOUS/'cache.json'),
                    enhancement=file_hash(PATCH), smoke=args.smoke, feature_dtype='float16')
    if (args.output/'feature_protocol.json').exists() and read(args.output/'feature_protocol.json') != protocol:
        raise RuntimeError('feature-cache protocol changed')
    atomic_json(args.output/'feature_protocol.json', protocol)
    model, base_codec = load_model(PATCH), codec()
    results = []
    for i, row in enumerate(rows):
        check_space()
        begin = time.monotonic()
        sid = row['sample_id']
        dest = args.output/'features'/f'{sid}.pt'
        meta = dest.with_suffix('.json')
        root = PREVIOUS/'streams'/sid
        for name, digest in row['artifacts'].items():
            if file_hash(root/name) != digest:
                raise RuntimeError('input stream changed')
        if meta.exists():
            record = read(meta)
            if file_hash(dest) != record['feature_hash']:
                raise RuntimeError('feature cache changed')
        else:
            image, report, base, packets = receive_features(model, PATCH, base_codec, (root/'full.acse').read_bytes())
            if frame_hash(image) != row['expected']['full']:
                raise RuntimeError('observing features changed enhanced RGB')
            ids = {m:[p.meta['packet_id'] for p in parse((root/f'{m}.acse').read_bytes()).packets]
                   for m in ('none','partial','full')}
            prefix_exact = None
            if args.smoke:
                for m in ('none','partial'):
                    pixels, _, _, received = receive_features(model, PATCH, base_codec, (root/f'{m}.acse').read_bytes())
                    if frame_hash(pixels) != row['expected'][m]:
                        raise RuntimeError('prefix RGB changed')
                    expected = selected(packets, ids[m])
                    for a,b in zip(received,expected,strict=True):
                        if a['delta'] is None:
                            assert b['delta'] is None
                        else:
                            torch.testing.assert_close(a['delta'], b['delta'], rtol=0,atol=0)
                prefix_exact = True
            atomic_torch_save(dest, dict(packets=packets, prefix_ids=ids))
            record = dict(sample_id=sid, feature_path=str(dest), feature_hash=file_hash(dest),
                source_frames_read=False, enhanced_rgb_exact=True, fresh_prefix_features_exact=prefix_exact,
                feature_bytes=dest.stat().st_size, seconds=time.monotonic()-begin)
            atomic_json(meta,record)
        results.append(dict(by_id[sid], **record))
        print(f'FEATURES {i+1}/{len(rows)} {sid} {time.monotonic()-begin:.1f}s',flush=True)
    atomic_json(args.output/'cache.json',dict(entries=results, complete=True,
        modes=['none','partial','full'], protocol=protocol))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
