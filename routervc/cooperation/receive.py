"""Source-free fresh decode of cooperative R_g + pinned multiband fusion."""
import gc
import time
from unittest.mock import patch
import numpy as np
import torch
from demo.scalable_format import frame_hash
from routervc.latent import routing, router_data, generation
from routervc.fusion.capture import CaptureRGB
from routervc.cooperation import receiver, stream
from routervc.cooperation.render import render


@torch.no_grad()
def decode(data, checkpoint, disable_generation=False, check=lambda: None):
    start = time.monotonic(); inner, config = stream.parse(data)
    peaks, reserved = [], []
    reset = torch.cuda.reset_peak_memory_stats
    def observed(device=None):
        peaks.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device)); reset(device)
    with patch.object(torch.cuda, 'reset_peak_memory_stats', observed):
        torch.cuda.reset_peak_memory_stats()
        base, received, detail = router_data.decode(inner)
        patches, reports, indices, selected = [], [], [], None
        if not disable_generation:
            model, _ = receiver.load(checkpoint, config['receiver_sha256'])
            selected = receiver.select(model, base, received,
                routing.coverage(routing.packets.parse(inner)), config['max_g'])
            indices = selected['indices']; del model
            if indices:
                hashes = generation.assets()
                if generation.asset_hash(hashes) != config['assets_sha256']: raise ValueError('G differs')
                generator = CaptureRGB()
                for i in indices:
                    check()
                    _, raw, report = generator(received, routing.control(received.shape, hashes, i, config['seed']))
                    patches.append(dict(region=i, pixels=raw, crop=report['crop'], core=report['core']))
                    reports.append(report)
                del generator
                gc.collect(); torch.cuda.empty_cache()
        output = render(received, patches, indices) if indices else received.copy()
        torch.cuda.synchronize()
    peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    return base, received, output, dict(complete=True, selected=selected, generated=indices,
        detail=detail, generation_runtime=reports, base_hash=frame_hash(base),
        enhanced_hash=frame_hash(received), output_hash=frame_hash(output),
        actual_bytes=len(data), header_bytes=stream.HEADER_BYTES, mask_bytes=0,
        source_frames_read=False, sender_router_loaded=False, generation_disabled=disable_generation,
        bpp=len(data)*8/int(np.prod(base.shape[:3])), seconds=time.monotonic()-start,
        peak_cuda_allocated_bytes=max(peaks), peak_cuda_reserved_bytes=max(reserved))
