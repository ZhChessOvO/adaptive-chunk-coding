"""Frozen-chain G replacement, with an explicitly versioned, mask-free profile.

The inner B/E bytes and 121-byte envelope size are unchanged. New model IDs are
honest: G-only checkpoints are exported with the legacy *disabled* interface,
without changing any LoRA tensor. Old decoders reject this experimental profile.
"""
import copy
import gc
import hashlib
from pathlib import Path
import struct
import time
from types import SimpleNamespace
from unittest.mock import patch
import zlib

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_npz, atomic_json
from demo.scalable_format import frame_hash
from routervc.cooperation import receiver, stream
from routervc.cooperation.render import render
from routervc.latent import routing, router_data, generation

MAGIC = b'RVLGINT1'
BRANCH_KEYS = ('feature_format', 'feature_enabled', 'feature_state',
               'internal_condition_format', 'branch_kind', 'branch_mode', 'branch_state')


def identity():
    return hashlib.sha256((stream.identity()+digest(Path(__file__))).encode()).hexdigest()


def wrap(inner, model, assets, max_g=8, seed=routing.SEED):
    stream.wrap(inner, model, assets, max_g, seed)  # Reuse all payload validation.
    header = stream.FIELDS.pack(MAGIC, bytes.fromhex(identity()), bytes.fromhex(model),
                                bytes.fromhex(assets), seed, max_g, 0.)
    return header+struct.pack('<I', zlib.crc32(header))+inner


def parse(data):
    if len(data) < stream.HEADER_BYTES:
        raise ValueError('incomplete G integration envelope')
    magic, code, model, assets, seed, max_g, reserved = stream.FIELDS.unpack_from(data)
    if (magic != MAGIC or code.hex() != identity() or reserved != 0 or
            struct.unpack_from('<I', data, stream.FIELDS.size)[0] != zlib.crc32(data[:stream.FIELDS.size])):
        raise ValueError('wrong/corrupt G integration profile')
    inner = data[stream.HEADER_BYTES:]
    if wrap(inner, model.hex(), assets.hex(), max_g, seed) != data:
        raise ValueError('noncanonical G integration envelope')
    return inner, dict(receiver_sha256=model.hex(), assets_sha256=assets.hex(), seed=seed, max_g=max_g)


def compatible_bundle(candidate, legacy):
    from demo.internal_condition_model import validate_bundle
    if validate_bundle(legacy) != ('internal', 'off') or legacy['feature_enabled']:
        raise ValueError('only a disabled legacy interface may be copied')
    for key in ('rank', 'alpha', 'target_modules', 'last_n_blocks', 'trainable_parameter_count'):
        if candidate[key] != legacy[key]:
            raise ValueError('G architecture changed: '+key)
    if set(candidate['state_dict']) != set(legacy['state_dict']):
        raise ValueError('LoRA parameter names changed')
    for key, value in candidate['state_dict'].items():
        if value.shape != legacy['state_dict'][key].shape or not torch.isfinite(value).all():
            raise ValueError('invalid candidate LoRA tensor: '+key)
    result = copy.deepcopy(candidate)
    for key in BRANCH_KEYS:
        result[key] = copy.deepcopy(legacy[key])
    return result


class ExplicitCapture:
    """Same native crop/condition/noise/capture math, explicit shared G bundle."""
    def __init__(self, adapter):
        from demo import stage_c_a800_teacher as teacher
        from demo.scalable_generation_decode import ASSETS
        self.teacher, self.adapter = teacher, adapter
        self.model = teacher.PersistentSeedVR2(SimpleNamespace(
            upstream_root=Path(__file__).resolve().parents[2]/'third_party/SeedVR2',
            dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'],
            positive_embedding=ASSETS['positive'], negative_embedding=ASSETS['negative'],
            lora_checkpoint=adapter, lora_strength=1., sample_steps=1, cfg_scale=1., dit_dtype='bfloat16'))

    def __call__(self, pixels, settings):
        from demo.internal_condition_decode import restore
        from demo import scalable_cooperation_format as fmt
        if len(settings['generate']) != 1 or settings['window'] != 17:
            raise ValueError('one native 17-frame region required')
        runner = self.model.runner
        methods = {k:getattr(runner,k) for k in ('vae_encode','get_condition','inference')}
        captured, original = [], self.model.restore
        def observe(*args, **kwargs):
            frames, runtime = original(*args, **kwargs)
            captured.append(np.stack(frames))
            return frames, runtime
        try:
            with patch.object(self.teacher, 'PersistentSeedVR2', return_value=self.model), \
                    patch.object(self.model, 'restore', observe):
                generated, runtime = restore(pixels, settings, self.adapter, [])
        finally:
            for name, method in methods.items(): setattr(runner, name, method)
        if len(captured) != 1 or len(runtime['windows']) != 1:
            raise ValueError('unexpected temporal tiling')
        raw = captured[0]
        x0,y0,cw,ch = runtime['windows'][0]['crop']
        t,n,x,y,w,h = settings['generate'][0]
        if raw.shape != (n,ch,cw,3): raise ValueError('raw shape differs')
        restored = pixels.copy()
        restored[t:t+n,y:y+h,x:x+w] = raw[:,y-y0:y-y0+h,x-x0:x-x0+w]
        alpha = fmt.weights(pixels.shape, settings)
        np.testing.assert_array_equal(fmt.combine(pixels, restored, alpha), fmt.combine(pixels, generated, alpha))
        runtime['model_load_seconds'] = 0.
        runtime['persistent_model_load_seconds_once'] = self.model.model_load_seconds
        return raw, dict(runtime=runtime, crop=[x0,y0,cw,ch], core=[x,y,w,h], feather_replay_exact=True)


@torch.no_grad()
def decode(data, checkpoint, adapter, output, disabled=False, stop_regions=0, check=lambda:None):
    start = time.monotonic(); inner, config = parse(data)
    binding = dict(wire=hashlib.sha256(data).hexdigest(), profile=identity(), disabled=disabled,
                   adapter=None if disabled else digest(adapter))
    immutable(output/'binding.json', binding)
    peaks, reserved = [], []
    reset = torch.cuda.reset_peak_memory_stats
    def observed(device=None):
        peaks.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device)); reset(device)
    with patch.object(torch.cuda, 'reset_peak_memory_stats', observed):
        torch.cuda.reset_peak_memory_stats()
        base, received, detail = router_data.decode(inner)
        patches, reports, indices, selected = [], [], [], None
        new_regions = 0
        if not disabled:
            model, _ = receiver.load(checkpoint, config['receiver_sha256'])
            selected = receiver.select(model, base, received,
                routing.coverage(routing.packets.parse(inner)), config['max_g'])
            indices = selected['indices']; del model
            hashes = generation.assets(adapter)
            if generation.asset_hash(hashes) != config['assets_sha256']:
                raise ValueError('actual G assets differ from transmitted shared model identity')
            generator = None
            for i in indices:
                check(); folder = output/f'g{i:02d}'
                settings = routing.control(received.shape, hashes, i, config['seed'])
                immutable(folder/'request.json', dict(binding=binding, settings=settings,
                    received_hash=frame_hash(received)))
                if (folder/'complete.json').exists():
                    report = read(folder/'complete.json'); verify_artifacts(folder, report['artifacts'])
                    with np.load(folder/'raw.npz', allow_pickle=False) as z: raw = z['pixels'].copy()
                else:
                    if generator is None: generator = ExplicitCapture(adapter)
                    raw, report = generator(received, settings)
                    atomic_npz(folder/'raw.npz', pixels=raw)
                    report['artifacts'] = {'raw.npz':digest(folder/'raw.npz')}
                    atomic_json(folder/'complete.json', report); new_regions += 1
                reports.append(report)
                patches.append(dict(region=i, pixels=raw, crop=report['crop'], core=report['core']))
                if stop_regions and new_regions >= stop_regions:
                    atomic_json(output/'intentional_stop.json', dict(new_regions=new_regions, binding=binding))
                    return None
            del generator
            gc.collect(); torch.cuda.empty_cache()
        result = render(received, patches, indices) if indices else received.copy()
        torch.cuda.synchronize()
    peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    report = dict(complete=True, selected=selected, generated=indices, detail=detail,
        generation_runtime=reports, base_hash=frame_hash(base), enhanced_hash=frame_hash(received),
        output_hash=frame_hash(result), actual_bytes=len(data), header_bytes=stream.HEADER_BYTES, mask_bytes=0,
        source_frames_read=False, sender_router_loaded=False, generation_disabled=disabled,
        bpp=len(data)*8/int(np.prod(base.shape[:3])), seconds=time.monotonic()-start,
        peak_cuda_allocated_bytes=max(peaks), peak_cuda_reserved_bytes=max(reserved),
        newly_generated_regions=new_regions, resumed_regions=len(indices)-new_regions)
    return base, received, result, report
