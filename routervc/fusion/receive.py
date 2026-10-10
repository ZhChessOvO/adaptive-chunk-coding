"""Receiver-local fusion: shared model and received observations, never source."""
import gc
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from demo.scalable_format import frame_hash
from demo.routervc_fullview_probe import digest
from demo.routervc_receiver_router import load_model
from demo.four_state_receive import codec_precision
from routervc.latent import routing, router_data, generation
from routervc.fusion import stream
from routervc.fusion.capture import CaptureRGB
from routervc.fusion.blend import fuse
from routervc.fusion.inputs import maps, tensors
from routervc.fusion.model import PrecisionFusion, FORMAT


def load_fusion(path, expected):
    if digest(path) != expected: raise ValueError('shared fusion checkpoint differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != FORMAT: raise ValueError('unknown fusion model')
    repo = Path(__file__).resolve().parents[2]
    for name, value in state['code'].items():
        if digest(repo/name) != value: raise ValueError('fusion training source changed: '+name)
    model = PrecisionFusion().cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model'])
    return model


@torch.no_grad()
def learned_pixels(model, base, received, current, multiband, regions, generated):
    values = dict(base=base, received=received, current=current, multiband=multiband)
    values.update(maps(base.shape, regions, generated))
    result = []
    with codec_precision():
        for frame in range(len(base)):
            inputs = tensors(values, [frame], device='cuda')
            out = model(inputs)[0].permute(1, 2, 0).mul(255).round().clip(0, 255).byte().cpu().numpy()
            unchanged = (values['e_band'][frame] == 0) & (values['g_band'][frame] == 0)
            np.testing.assert_array_equal(out[unchanged], current[frame][unchanged])
            result.append(out)
    return np.stack(result)


def decode(data, receiver, checkpoint=None, *, disable_generation=False, check=lambda: None):
    start = time.monotonic()
    original, fusion = stream.parse(data)
    inner, config, _ = routing.parse(original)
    peaks, reserved = [], []
    reset = torch.cuda.reset_peak_memory_stats
    def observed_reset(device=None):
        peaks.append(torch.cuda.max_memory_allocated(device)); reserved.append(torch.cuda.max_memory_reserved(device))
        reset(device)
    with patch.object(torch.cuda, 'reset_peak_memory_stats', observed_reset):
        torch.cuda.reset_peak_memory_stats()
        base, received, detail = router_data.decode(inner)
        generated, reports, patches = [], [], []
        selected = None; current = received.copy()
        if not disable_generation:
            model, _ = load_model(receiver, expected_sha256=config['receiver_sha256'])
            selected = routing.route(base, received, inner, config, model)
            generated = selected['indices']; del model
            if generated:
                hashes = generation.assets()
                if generation.asset_hash(hashes) != config['assets_sha256']: raise ValueError('G assets differ')
                generator = CaptureRGB()
                from demo.scalable_cooperation_format import weights
                for region in generated:
                    check(); settings = routing.control(received.shape, hashes, region, config['seed'])
                    values, raw, report = generator(received, settings)
                    alpha = weights(received.shape, settings)
                    current[alpha > 0] = values[alpha > 0]
                    patches.append(dict(region=region, pixels=raw, crop=report['crop'], core=report['core']))
                    reports.append(report)
                del generator
                gc.collect(); torch.cuda.empty_cache()
        fusion_start = time.monotonic(); load_seconds = 0.
        if disable_generation or fusion['mode'] == 'current':
            output = current
        else:
            controls = fuse(received, current, patches, generated)
            if fusion['mode'] != 'learned': output = controls[fusion['mode']]
            else:
                begin = time.monotonic(); model = load_fusion(checkpoint, fusion['model_sha256'])
                load_seconds = time.monotonic()-begin
                torch.use_deterministic_algorithms(True)
                output = learned_pixels(model, base, received, current, controls['multiband'],
                                        detail['received_regions'], generated)
                del model
        torch.cuda.synchronize()
        fusion_seconds = time.monotonic()-fusion_start
    peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    return base, received, output, dict(complete=True, detail=detail, selected=selected,
        generated=generated, generation_runtime=reports, fusion=fusion,
        original_current_hash=frame_hash(current), output_hash=frame_hash(output),
        base_hash=frame_hash(base), enhanced_hash=frame_hash(received),
        actual_bytes=len(data), old_router_bytes=len(original), additional_fusion_header_bytes=stream.HEADER_BYTES,
        additional_mask_bytes=0, bpp=len(data)*8/np.prod(base.shape[:3]),
        source_frames_read=False, sender_router_loaded=False, generation_disabled=disable_generation,
        fusion_model_loaded=fusion['mode']=='learned' and not disable_generation,
        fusion_total_seconds=fusion_seconds, fusion_model_load_seconds=load_seconds,
        seconds_decode_model_load_included=time.monotonic()-start,
        peak_cuda_allocated_bytes=max(peaks), peak_cuda_reserved_bytes=max(reserved))
