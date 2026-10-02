"""Online, quantized E -> RGB -> frozen VAE inputs for joint adaptation.

Historical models and decoders are not edited. E packets keep their original
256px footprint; only the G processing window is cropped. A byte-rounding STE
has the receiver's forward values, rather than sending unquantized RGB to G.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from demo.chunk_enhancement_codec import pack_region, region_features
from demo.chunk_enhancement_experiment import read
from demo.conditioned_generation_cache import FEATURES
from demo.feature_condition_cache import PREVIOUS
from demo.roi_condition_train import crop_for_packets
from demo.scalable_codec import file_hash
from demo.scalable_format import parse

FORMAT = 'online_eg_rgb_first_quantized_roi_v1'
STREAMS = PREVIOUS/'streams'


def round_pixels(value):
    """Exact integer-valued 0..255 forward, identity derivative inside bounds."""
    scaled = value.clamp(0, 1)*255
    return scaled + (scaled.round()-scaled).detach()


def normalize_pixels(pixels):
    # Receiver divides uint8 RGB by 255 on CPU, then normalizes on GPU. GPU
    # division rounds differently at a few byte levels (visible after BF16).
    # This 256-value table reproduces that forward exactly, with the usual
    # quantization STE derivative. It carries no information beyond received RGB.
    lookup = torch.arange(256, dtype=torch.float32).div_(255).to(pixels.device)
    exact = lookup[pixels.detach().round().long().clamp(0, 255)]
    normalized = exact + (pixels-pixels.detach())/255
    return normalized.clamp(0, 1)*2-1


def intersection(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    return (x, y, r-x, d-y) if r > x and d > y else None


def training_entries():
    entries = []
    for row in read(FEATURES)['samples']:
        sid = row['sample_id']
        record = read(STREAMS/sid/'complete.json')
        assert record['pair_hash'] == row['pair_hash']
        packets = {}
        for mode in ('none', 'partial', 'full'):
            path = STREAMS/sid/f'{mode}.acse'
            assert file_hash(path) == record['artifacts'][path.name]
            packets[mode] = [p.meta for p in parse(path.read_bytes()).packets]
        entries.append(dict(row, dataset=row['sample']['dataset'], packets=packets))
    assert len(entries) == 120
    return entries


def choose_crop(rng, packets):
    if not packets:
        return (rng.randrange(17)*16, rng.randrange(17)*16, 256, 256)
    # The crop rule is independent of the learned contents and both arms use it.
    return crop_for_packets(rng, [dict(p, delta=True if p['start'] else None) for p in packets])


def tensor_rgb(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).to(device).float()/255


def online_rgb(model, source, base, chunks, packets, crop, *, differentiable=True):
    """Recompute complete intersecting packets, then extract the G ROI.

    The rate/fidelity term covers entire coded packets, not just their visible
    overlap with G's ROI. No source information bypasses E quantization.
    """
    device = next(model.parameters()).device
    x, y, w, h = crop
    result = tensor_rgb(base[:, y:y+h, x:x+w], device)*255
    lookup = {c['start']: c for c in chunks}
    losses, bits, pixels, used = [], [], [], []
    for p in packets:
        overlap = intersection(crop, p['roi'])
        if overlap is None:
            continue
        roi, start, count, q = p['roi'], p['start'], p['count'], p['qstep']
        bottom = pack_region(base, start, count, roi, device, model.spatial_alignment)
        target = pack_region(source, start, count, roi, device, model.spatial_alignment)
        feature = region_features(lookup[start], roi, device, model.feature_halo, model.spatial_alignment)
        def packet_forward(a, b, f, quality=q, valid=count):
            prediction = model(a, b, f, quality, valid)
            # Saturated symbols cannot be coded by the real serializer.
            if prediction['saturated'].item():
                raise RuntimeError('E latent saturation; checkpoint retained, no formal coding claim')
            return prediction['reconstruction'], prediction['bits']
        if differentiable:
            reconstructed, rate = checkpoint(packet_forward, target, bottom, feature, use_reentrant=False)
        else:
            with torch.no_grad():
                reconstructed, rate = packet_forward(target, bottom, feature)
        ph, pw = bottom.shape[-2:]
        pred = reconstructed.reshape(8, 3, ph, pw)[:count, :, :roi[3], :roi[2]]
        truth = target.reshape(8, 3, ph, pw)[:count, :, :roi[3], :roi[2]]
        mse = (pred-truth).square().mean()
        temporal = ((pred[1:]-pred[:-1])-(truth[1:]-truth[:-1])).square().mean() if count > 1 else mse*0
        n = count*roi[2]*roi[3]
        # Retain the A model's q-dependent fidelity/rate trade-off (lambda_scale=4).
        losses.append((128*4/q*mse + 8*temporal)*n)
        bits.append(rate); pixels.append(n); used.append(p['packet_id'])
        ox, oy, ow, oh = overlap
        local = round_pixels(pred)[:, :, oy-roi[1]:oy-roi[1]+oh, ox-roi[0]:ox-roi[0]+ow]
        base_local = tensor_rgb(base[start:start+count, oy:oy+oh, ox:ox+ow], device)*255
        correction = F.pad(local-base_local, (ox-x, x+w-ox-ow, oy-y, y+h-oy-oh))
        result = result + F.pad(correction, (0, 0, 0, 0, 0, 0, start, len(base)-start-count))
    zero = result.new_zeros(())
    return result, dict(bpp=sum(bits, zero)/max(1, sum(pixels)),
                        fidelity=sum(losses, zero)/max(1, sum(pixels)),
                        packets=used, coded_pixels=sum(pixels))


def encode_pixels(runner, pixels):
    """Receiver-identical cropped RGB mean/BF16 VAE with differentiable input.

    Preserve upstream 4-frame causal slicing. Upstream history caches detach:
    gradients flow within each slice; cross-slice history has truncated BPTT.
    Both arms share this forward path, and a smoke checks exact receiver values.
    All slicing state is reset on every checkpoint replay and cleaned afterward.
    """
    sample = normalize_pixels(pixels)
    sample = sample.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(torch.bfloat16)
    runner.vae.set_causal_slicing(**runner.config.vae.slicing)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.autocast('cuda', dtype=torch.bfloat16):
            if hasattr(runner.vae, 'preprocess'):
                sample = runner.vae.preprocess(sample)
            latent = runner.vae.encode(sample).posterior.mode()
            latent = latent.permute(0, 2, 3, 4, 1)
            latent = (latent-float(runner.config.vae.get('shifting_factor', 0.)))*float(runner.config.vae.scaling_factor)
        return latent[0]
    finally:
        runner.vae.set_causal_slicing(split_size=None, memory_device='same')
        for module in runner.vae.modules():
            if hasattr(module, 'memory'):
                module.memory = None
