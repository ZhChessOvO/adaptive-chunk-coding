"""Native RGB-first ROI conditions, independent of immutable historical caches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from demo.chunk_enhancement_experiment import read
from demo.condition_path_decode import tensor_identity
from demo.feature_condition_cache import PREVIOUS
from demo.feature_condition_train import image_terms
from demo.scalable_codec import atomic_json, file_hash
from demo.stage_c_seedvr2_bridge import resize_and_normalize, pad_temporal
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save

FORMAT = 'rgb_first_native_roi_mean_bf16_halo64_v1'
HALO = 64


def context_crop(shape, region, halo=HALO):
    t, n, x, y, w, h = region
    frames, height, width = shape[:3]
    if min(t, x, y, halo) < 0 or min(n, w, h) <= 0:
        raise ValueError('invalid region')
    if t+n > frames or x+w > width or y+h > height:
        raise ValueError('region outside received video')
    x0, y0 = max(0,x-halo), max(0,y-halo)
    x1, y1 = min(width,x+w+halo), min(height,y+h+halo)
    if x0 % 8 or y0 % 8 or (x1-x0) % 16 or (y1-y0) % 16:
        raise ValueError('ROI must preserve the receiver lattice without resizing')
    return x0, y0, x1-x0, y1-y0


def crop_rgb(frames, crop):
    x, y, w, h = crop
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('expected uint8 THWC receiver pixels')
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > frames.shape[2] or y+h > frames.shape[1]:
        raise ValueError('invalid RGB crop')
    if x % 8 or y % 8 or w % 16 or h % 16:
        raise ValueError('unaligned RGB crop')
    return frames[:,y:y+h,x:x+w].copy()


@torch.no_grad()
def encode_roi(runner, frames, crop, device):
    """Same preprocessing, encoder slicing, posterior and dtype as receiver.

    Decoder slicing stays disabled during differentiable image supervision.
    VAE posterior sampling inside encode must not perturb the training RNG.
    """
    values = crop_rgb(frames, crop)
    tensor = torch.from_numpy(values).permute(0,3,1,2).float().div_(255.)
    sample, length = pad_temporal(resize_and_normalize(tensor,crop[3],crop[2],device))
    if length != 17:
        raise ValueError('training/probe encode operates on actual 17-frame windows')
    runner.config.vae.use_sample = False
    runner.vae.set_causal_slicing(**runner.config.vae.slicing)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), \
             torch.autocast('cuda',dtype=torch.bfloat16):
            latent = runner.vae_encode([sample])[0]
        assert latent.dtype == torch.bfloat16 and torch.isfinite(latent).all()
        return latent.detach().cpu().contiguous()
    finally:
        runner.vae.set_causal_slicing(split_size=None,memory_device='same')


def core_image_terms(pred, target, metric, offset=0):
    # Reuse all loss definitions/weights; the old function omits another 16px.
    # A 256px processing crop therefore supervises its 128px core, halo=64.
    border = HALO-16
    if pred.shape[-2:] != (256,256) or target.shape != pred.shape:
        raise ValueError('expected paired 256px processing crops')
    return image_terms(pred[...,border:-border,border:-border],
                       target[...,border:-border,border:-border],metric,offset)


def received_inputs(entries):
    """Validate actual reconstructed RGB once, before any new training."""
    result = {}
    for entry in entries:
        sid = entry['sample_id']
        folder = PREVIOUS/'received'/sid
        record = read(folder/'complete.json')
        path = folder/'conditions.npz'
        expected = record['artifacts']['conditions.npz']
        assert not record['source_frames_read'] and file_hash(path) == expected
        result[sid] = dict(path=str(path),sha256=expected)
    return result


def cached_pair(root, runner, entry, received, crop, mode, vae_hash, device):
    identity = dict(format=FORMAT, sample=entry['sample_id'], crop=list(crop),mode=mode,
        source=entry['pair_hash'], received=received['sha256'], vae=vae_hash,
        code=file_hash(Path(__file__)))
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path = root/f'{key}.pt'
    meta = path.with_suffix('.json')
    if meta.exists():
        record = read(meta)
        assert record['identity'] == identity and file_hash(path) == record['sha256']
        value = torch.load(path,weights_only=True,map_location='cpu')
    else:
        with np.load(entry['pair_path']) as pair:
            clean = encode_roi(runner,pair['source'],crop,device)
        with np.load(received['path']) as pair:
            raw = encode_roi(runner,pair[mode],crop,device)
        value = dict(clean=clean,raw=raw)
        atomic_torch_save(path,value)
        record = dict(identity=identity,sha256=file_hash(path),
            clean=tensor_identity(clean),raw=tensor_identity(raw))
        atomic_json(meta,record)
    for k in ('clean','raw'):
        assert list(value[k].shape) == [5,32,32,16] and value[k].dtype == torch.bfloat16
        assert tensor_identity(value[k]) == record[k]
    return value, key
