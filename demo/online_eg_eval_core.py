"""Pinned real-packet preparation and read-only checks for online E/G evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import compact_enhancement_format as compact
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_codec import (configure_torch, decode_features, load_model,
    pack_region, region_features, unpack_region)
from demo.chunk_enhancement_experiment import read, codec, MECHANISM
from demo.internal_condition_pipeline import OLD, MODES
from demo.feature_condition_report import CLIPS, METRICS
from demo.online_eg_train import DEFAULT, INITIAL, PATCH
from demo.online_eg_pipeline import exact
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import parse, frame_hash
from demo.scalable_experiment import load_source

HISTORY = INITIAL.parents[1] / 'evaluation'
ARMS = ('fixed', 'joint')
CASES = ('none', 'partial', 'full', 'full_q05', 'full_q2')
QSTEPS = {'full': 1., 'full_q05': .5, 'full_q2': 2.}
CONTROLS = ('generate', 'protect', 'context', 'feather', 'seed',
            'processing_scale', 'blend', 'window', 'stride')


def split_prefix(wire, count):
    value = parse(wire)
    if not 0 <= count <= len(value.packets):
        raise ValueError('invalid packet prefix')
    return wire[:value.base_end if not count else value.packets[count-1].end_offset]


def prefix_pixels(base, full, wire):
    result = base.copy()
    for p in parse(wire).packets:
        m = p.meta
        t, n = m['start'], m['count']
        x, y, w, h = m['roi']
        result[t:t+n, y:y+h, x:x+w] = full[t:t+n, y:y+h, x:x+w]
    return result


def noise_pair(a, b, *, same_condition=False):
    """Different E SHOULD change conditions, but never diffusion noise or geometry."""
    x, y = a['generation_runtime'], b['generation_runtime']
    assert len(x['condition_windows']) == len(y['condition_windows']) > 0
    assert [(v['region'], v['start'], v['crop']) for v in x['windows']] == [
        (v['region'], v['start'], v['crop']) for v in y['windows']]
    for u, v in zip(x['condition_windows'], y['condition_windows'], strict=True):
        for key in ('before_vae', 'after_vae', 'before_diffusion', 'diffusion_noise'):
            assert u[key] == v[key], f'noise/RNG pairing changed: {key}'
        assert u['diffusion_noise']['dtype'] == 'torch.bfloat16'
        if same_condition:
            assert u['conditions'] == v['conditions']


def training_check(root):
    """No modification of completed training manifests, audit or timings."""
    assert read(root/'train.complete.json')['complete']
    recorded = read(root/'training_audit.json')
    assert recorded['complete'] and recorded['steps'] == 3000 and recorded['paired']
    for name, digest in read(root/'queue_protocol.json')['code'].items():
        assert file_hash(REPO/'demo'/name) == digest
    logs, summaries = [], {}
    initial = torch.load(PATCH, weights_only=True, map_location='cpu')['model']
    for arm in ARMS:
        folder = root/arm
        c, cfg = read(folder/'complete.json'), read(folder/'config.json')
        assert c['steps'] == cfg['steps'] == 3000 and c['mode'] == arm
        assert c['enhancement_changed'] == (arm == 'joint')
        for key in ('adapter', 'enhancement'):
            assert file_hash(folder/f'{key}.pt') == c[key+'_sha256']
        saved = torch.load(folder/'resume.pt', weights_only=True, map_location='cpu')
        assert saved['step'] == 3000 and saved['config'] == cfg
        for key in ('adapter', 'enhancement'):
            endpoint = torch.load(folder/f'{key}.pt', weights_only=True, map_location='cpu')
            exact(saved[key], endpoint)
        if arm == 'fixed':
            exact(initial, saved['enhancement']['model'])
        rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 3001))
        for r in rows:
            assert np.isfinite(r['loss']) and r['lora_gradient_norm'] > 0
            if arm == 'joint' and r['condition'] != 'none':
                assert r['e_gradient_norm'] > 0 and r['g_to_e_rgb_gradient_norm'] > 0
            else:
                assert r['e_gradient_norm'] == 0
        logs.append(rows)
        summaries[arm] = c
    for a, b in zip(*logs, strict=True):
        for k in ('step', 'sample', 'dataset', 'condition', 'crop', 'packet_ids',
                  'diffusion_noise_identity', 'learning_rate'):
            assert a[k] == b[k]
    return dict(complete=True, export_resume_exact=True, fixed_E_weights_exact=True,
                paired=True, steps=3000, summaries=summaries)


def prepare(root, sid, arm):
    """Encode new weights using the OLD packet order, including appended wall packets.

    Do not replace this by the encoder's default time/ROI ordering: that would
    alter the partial-prefix comparison on REDS000.
    """
    configure_torch()
    folder = root/'evaluation'/sid/'prepared'/arm
    folder.mkdir(parents=True, exist_ok=True)
    weights = root/arm/'enhancement.pt'
    row = next(r for r in read(OLD/'summary.json')['results'] if r['sample']['sample_id'] == sid)
    oldfolder = OLD/sid/'prepared'
    oldrecord = read(oldfolder/'complete.json')
    assert oldrecord == row['prepared']
    verify_artifacts(oldfolder, oldrecord['artifacts'])
    identities = dict(model=file_hash(weights), source=row['source_hash'],
        reference=file_hash(oldfolder/'complete.json'), code=file_hash(Path(__file__)))
    done = folder/'complete.json'
    if done.exists():
        record = read(done)
        assert record['identities'] == identities
        verify_artifacts(folder, record['artifacts'])
        return record
    source = load_source(row['sample'])
    assert frame_hash(source) == identities['source']
    oldwire = (oldfolder/'q1.acse').read_bytes()
    parsed = parse(oldwire)
    base_codec = codec()
    base, chunks = decode_features(base_codec, oldwire[:parsed.base_end])
    assert frame_hash(base) == frame_hash(load_frames(oldfolder/'base_expected.npz'))
    model = load_model(weights)
    by_start = {c['start']: c for c in chunks}
    meta = dict(parsed.meta, enhancement_model_sha256=file_hash(weights))
    prefix = compact.base_container(parsed.base, meta)
    records, full_wires, full_pixels = {}, {}, {}
    for case, q in QSTEPS.items():
        begin = time.monotonic()
        wires, details = [], []
        expected = base.copy()
        with torch.no_grad():
            for oldpacket in parsed.packets:
                pm = dict(oldpacket.meta, qstep=q)
                t, n, roi = pm['start'], pm['count'], pm['roi']
                x, y, w, h = roi
                assert t+n <= 17, 'keep the historical first-17-frame E scope'
                a = pack_region(source, t, n, roi, 'cuda', model.spatial_alignment)
                b = pack_region(base, t, n, roi, 'cuda', model.spatial_alignment)
                f = region_features(by_start[t], roi, 'cuda', model.feature_halo, model.spatial_alignment)
                payload, recon, stats = model.compress(a, b, f, q, n)
                packet = compact.packet_bytes(pm, payload, codec=model.FORMAT)
                wires.append(packet)
                expected[t:t+n, y:y+h, x:x+w] = unpack_region(recon, n, roi)
                details.append(dict(pm, packet_bytes=len(packet), payload_bytes=len(payload), **stats))
        torch.cuda.synchronize()
        wire = prefix+b''.join(wires)
        if arm == 'fixed':
            original = (oldfolder/f'q{q:g}.acse').read_bytes()
            assert [p.wire for p in parse(wire).packets] == [p.wire for p in parse(original).packets]
            np.testing.assert_array_equal(expected, load_frames(oldfolder/f'q{q:g}_expected.npz'))
        full_wires[case], full_pixels[case] = wire, expected
        records[case] = dict(encoding_seconds=time.monotonic()-begin, packet_details=details, qstep=q)
    wire = full_wires['full']
    for case, count in [('none', 0), ('partial', (len(parsed.packets)+1)//2)]:
        full_wires[case] = split_prefix(wire, count)
        full_pixels[case] = prefix_pixels(base, full_pixels['full'], full_wires[case])
        records[case] = dict(encoding_seconds=0., prefix_of='full', qstep=1.)
    for case in CASES:
        data, pixels = full_wires[case], full_pixels[case]
        atomic_bytes(folder/f'{case}.acse', data)
        atomic_npz(folder/f'{case}_expected.npz', reconstruction=pixels)
        records[case].update(bytes=len(data), output_hash=frame_hash(pixels),
                            packet_count=len(parse(data).packets))
    assert wire.startswith(full_wires['partial']) and full_wires['partial'].startswith(full_wires['none'])
    artifacts = {p.name:file_hash(p) for p in folder.iterdir() if p.suffix in ('.acse', '.npz')}
    value = dict(complete=True, identities=identities, records=records, artifacts=artifacts,
                 base_hash=frame_hash(base), literal_prefix=True, fixed_packets_exact=arm == 'fixed')
    atomic_json(done, value)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--sample', choices=CLIPS, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    args = p.parse_args()
    prepare(args.root, args.sample, args.arm)
