"""Source-free four-state receiver with region-level atomic replay.

E packets are independently decoded from bytes; shared B/F extraction is reused.
G keeps the pinned receiver math, resetting its hooks after each invocation.
Model residency is an execution optimization, checked against fresh processes.
"""
import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import (REPO, ENHANCEMENT, ADAPTER, ROIS, control, crop,
    isolation, subset, verify, immutable_json)
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_codec import (configure_torch, decode_features, load_model,
    pack_region, region_features, unpack_region)
from demo.chunk_enhancement_experiment import codec, read
from demo.internal_condition_decode import restore
from demo.scalable_generation_decode import ASSETS
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import parse, frame_hash
from demo.scalable_experiment import check_space


class PersistentRGB:
    def __init__(self):
        from demo import stage_c_a800_teacher as teacher
        self.teacher = teacher
        args = SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
            dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'],
            positive_embedding=ASSETS['positive'], negative_embedding=ASSETS['negative'],
            lora_checkpoint=ADAPTER, lora_strength=1., sample_steps=1, cfg_scale=1.,
            dit_dtype='bfloat16')
        self.model = teacher.PersistentSeedVR2(args)

    def __call__(self, pixels, settings):
        runner = self.model.runner
        methods = {k:getattr(runner,k) for k in ('vae_encode','get_condition','inference')}
        try:
            with patch.object(self.teacher, 'PersistentSeedVR2', return_value=self.model):
                generated, runtime = restore(pixels, settings, ADAPTER, [])
        finally:
            # Do not stack observation/injection wrappers across samples.
            for name, method in methods.items():
                setattr(runner, name, method)
        runtime['model_load_seconds'] = 0.
        runtime['persistent_model_load_seconds_once'] = self.model.model_load_seconds
        return fmt.combine(pixels, generated, fmt.weights(pixels.shape, settings)), runtime


@contextmanager
def codec_precision():
    # SeedVR2 init_torch enables TF32 globally. E entropy/synthesis must retain
    # its own FP32 policy, then restore G's original receiver precision.
    backends = ((torch.backends.cuda.matmul, 'allow_tf32'),
                (torch.backends.cudnn, 'allow_tf32'),
                (torch.backends.cudnn, 'benchmark'),
                (torch.backends.cudnn, 'deterministic'))
    saved = [(obj, key, getattr(obj, key)) for obj, key in backends]
    configure_torch()
    try:
        yield
    finally:
        for obj, key, value in saved:
            setattr(obj, key, value)


def receive_E(bank):
    with codec_precision():
        return _receive_E(bank)


@torch.no_grad()
def _receive_E(bank):
    start = time.monotonic()
    model, base_codec = load_model(ENHANCEMENT), codec()
    p = parse(bank)
    assert p.meta['enhancement_model_sha256'] == file_hash(ENHANCEMENT)
    base, chunks = decode_features(base_codec, bank[:p.base_end])
    torch.cuda.synchronize()
    base_seconds = time.monotonic()-start
    torch.cuda.set_stream(torch.cuda.default_stream())
    by_start = {c['start']:c for c in chunks}
    enhanced, timings, packet_ids = base.copy(), [], []
    for i in range(16):
        selected = parse(subset(bank, [i])).packets
        torch.cuda.synchronize(); begin = time.monotonic()
        for packet in selected:
            m = packet.meta
            assert m['qstep'] == 1. and m['codec'] == model.FORMAT
            t, n, roi = m['start'], m['count'], m['roi']
            x, y, w, h = roi
            bottom = pack_region(base, t, n, roi, 'cuda', model.spatial_alignment)
            features = region_features(by_start[t], roi, 'cuda', model.feature_halo, model.spatial_alignment)
            pixels = model.decompress(packet.payload, bottom, features, 1., n)
            enhanced[t:t+n, y:y+h, x:x+w] = unpack_region(pixels, n, roi)
            packet_ids.append(m['packet_id'])
        torch.cuda.synchronize()
        timings.append(time.monotonic()-begin)
    assert len(packet_ids) == len(set(packet_ids)) == len(p.packets) == 48
    del model, base_codec, chunks, by_start
    gc.collect(); torch.cuda.empty_cache()
    return base, enhanced, dict(source_frames_read=False, base_hash=frame_hash(base),
        enhanced_hash=frame_hash(enhanced), base_and_E_model_load_and_base_seconds=base_seconds,
        per_region_e_seconds=timings, decoded_packets=packet_ids, base_reference_unchanged=True)


def receive(root, stop_after=0, verify_only=False):
    configure_torch()
    protocol = read(root/'protocol.json')
    index = read(root/'encoded/index.json')['entries']
    generator = None
    new_regions = 0
    for row in index:
        sid = row['sample_id']
        encoded, dest = root/'encoded'/sid, root/'received'/sid
        dest.mkdir(parents=True, exist_ok=True)
        record = read(encoded/'complete.json')
        assert file_hash(encoded/'complete.json') == row['encoded_manifest']
        verify(encoded, record['artifacts'])
        bank = (encoded/'packets.acse').read_bytes()
        e_done = dest/'E.json'
        if e_done.exists():
            e_report = read(e_done)
            verify(dest, e_report['artifacts'])
        else:
            if verify_only:
                raise RuntimeError('read-only replay cannot decode missing E')
            base, enhanced, e_report = receive_E(bank)
            assert e_report['base_hash'] == record['expected_base_hash']
            assert e_report['enhanced_hash'] == record['expected_E_hash']
            atomic_npz(dest/'received_E.npz', base=base, enhanced=enhanced)
            e_report['artifacts'] = {'received_E.npz':file_hash(dest/'received_E.npz')}
            atomic_json(e_done, e_report)
        with np.load(dest/'received_E.npz', allow_pickle=False) as cache:
            base, enhanced = cache['base'].copy(), cache['enhanced'].copy()
        all_results = []
        for i in range(16):
            check_space()
            folder = dest/f'cell_{i:02d}'
            folder.mkdir(exist_ok=True)
            settings = control(protocol['profile'], sid, i)
            result = folder/'result.json'
            if result.exists():
                value = read(result)
                verify(folder, value['artifacts'])
                assert value['control'] == settings and value['encoded_hash'] == row['encoded_manifest']
            else:
                if verify_only:
                    raise RuntimeError('read-only replay cannot generate missing region')
                if generator is None:
                    generator = PersistentRGB()
                states, reports = {}, {}
                for state in ('G', 'EG'):
                    condition = base if state == 'G' else isolation(base, enhanced, i)
                    inner = subset(bank, [] if state == 'G' else [i])
                    wire = fmt.wrap(inner, settings)
                    atomic_bytes(folder/f'{state}.acsg', wire)
                    start = time.monotonic()
                    output, runtime = generator(condition, settings)
                    mask = fmt.weights(base.shape, settings)
                    np.testing.assert_array_equal(output[mask == 0], condition[mask == 0])
                    states[state] = crop(output, i)
                    reports[state] = dict(source_frames_read=False, runtime=runtime,
                        generation_input_hash=frame_hash(condition), output_hash=frame_hash(output),
                        outside_generate_exact=True, seconds=time.monotonic()-start,
                        total_bytes=len(wire), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                a, b = reports['G']['runtime']['condition_windows'], reports['EG']['runtime']['condition_windows']
                assert len(a) == len(b) == 1
                for key in ('before_vae','after_vae','before_diffusion','diffusion_noise'):
                    assert a[0][key] == b[0][key], f'G/EG randomness changed: {key}'
                atomic_npz(folder/'outputs.npz', **states)
                value = dict(region=i, roi=ROIS[i], control=settings,
                    encoded_hash=row['encoded_manifest'], reports=reports,
                    artifacts={n:file_hash(folder/n) for n in ('G.acsg','EG.acsg','outputs.npz')})
                atomic_json(result, value)
                new_regions += 1
            all_results.append(file_hash(result))
            print(f'RECEIVE {sid} {i+1}/16 new={new_regions}', flush=True)
            if stop_after and new_regions >= stop_after:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                return
        immutable_json(dest/'complete.json', dict(complete=True, region_results=all_results,
            e_report=file_hash(e_done), source_frames_read=False))
    immutable_json(root/'received/complete.json', dict(complete=True,
        samples={r['sample_id']:file_hash(root/'received'/r['sample_id']/'complete.json') for r in index}))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    receive(a.root, a.stop_after, a.verify_only)
