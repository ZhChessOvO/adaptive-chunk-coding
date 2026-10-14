"""Paired perceptual augmentation and source-referenced lattice-bias loss.

These are training losses, not validated perceptual quality metrics. In
particular, legitimate periodic source texture is not itself penalized.
"""
import random
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def resize_matrix(size, output=128):
    # Build the exact separable PyTorch antialiased bilinear map on CPU.
    # Applying the fixed map with matmul avoids nondeterministic CUDA AA
    # resize backward while retaining the same linear operation/gradients.
    basis = torch.eye(size, dtype=torch.float32).reshape(size, 1, 1, size)
    return F.interpolate(basis, (1,output), mode='bilinear', align_corners=False,
                         antialias=True)[:,0,0,:].T.contiguous()


def transform_spec(seed):
    rng = random.Random(seed)
    return dict(size=rng.choice((112, 120, 128, 136, 144)),
                dx=rng.randint(-8, 8), dy=rng.randint(-8, 8))


def perceptual_view(image, spec):
    """Shift/scale the loss view only, never a cached VAE latent."""
    if image.shape[-2:] != (256, 256):
        raise ValueError('expected a 256px processing crop')
    size = spec['size']
    if size not in (112, 120, 128, 136, 144) or max(abs(spec['dx']), abs(spec['dy'])) > 8:
        raise ValueError('invalid paired transform')
    x, y = (256-size)//2+spec['dx'], (256-size)//2+spec['dy']
    view = image[..., y:y+size, x:x+size]
    if size != 128:
        matrix = resize_matrix(size).to(device=view.device, dtype=view.dtype)
        view = torch.matmul(torch.matmul(matrix,view),matrix.T)
    return view


def paired_lpips(pred, target, metric, step, seed):
    spec = transform_spec(seed)
    indices = list(range(step % 4, pred.shape[0], 4))
    left = perceptual_view(pred[indices], spec).clamp(-1, 1)
    right = perceptual_view(target[indices], spec)
    return metric(left, right).mean(), spec


def phase_bias_loss(pred, target, periods=(8, 16)):
    """RMS of stationary, high-pass *error* folded over 8/16px phases.

    Averaging cells/time rejects much nonstationary detail; subtracting the
    source rejects correctly reproduced periodic content. Finite scene bias
    remains possible, so this is kept separate from visual/LPIPS evaluation.
    Values use the training RGB range [-1,1]; no clipping hides overshoots.
    """
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError('expected paired TCHW tensors')
    error = (pred-target).float()
    coordinate = torch.arange(-9, 10, device=error.device, dtype=error.dtype)
    kernel = torch.exp(-coordinate.square()/18)
    kernel = kernel/kernel.sum()
    c = error.shape[1]
    blur = F.conv2d(F.pad(error, (9, 9, 0, 0), mode='reflect'),
                    kernel.reshape(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    blur = F.conv2d(F.pad(blur, (0, 0, 9, 9), mode='reflect'),
                    kernel.reshape(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    high = error-blur
    terms = []
    for p in periods:
        h, w = (high.shape[-2]//p)*p, (high.shape[-1]//p)*p
        if min(h, w) < p*2:
            raise ValueError('too few complete phase cells')
        folded = high[..., :h, :w].reshape(high.shape[0], c, h//p, p, w//p, p).mean((0, 2, 4))
        folded = folded-folded.mean((-2, -1), keepdim=True)
        terms.append(torch.sqrt(folded.square().mean()+1e-12)-1e-6)
    return torch.stack(terms).mean()
