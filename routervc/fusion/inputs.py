"""Prepare fusion tensors solely from received RGB, packet addresses and local G."""
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from demo.routervc_policy import grid_rois
from routervc.fusion.blend import geometry


def maps(shape, received_regions, generated):
    t, h, w, _ = shape
    if t != 17 or h % 64 or w % 64:
        # Current pilot shapes have matching latent/RGB grid boundaries. General
        # dimensions need the latent floor-division grid projected separately.
        raise ValueError('fusion pilot requires 17 frames and 64-aligned geometry')
    precision = np.zeros((t, h, w), np.float32)
    for chunk, region in received_regions:
        x, y, rw, rh = grid_rois(h, w)[region]
        precision[1+8*chunk:9+8*chunk, y:y+rh, x:x+rw] = 1
    e_band = np.zeros_like(precision)
    for frame, pe in enumerate(precision):
        edge = np.zeros((h, w), bool)
        dx, dy = pe[:, 1:] != pe[:, :-1], pe[1:] != pe[:-1]
        edge[:, 1:] |= dx; edge[:, :-1] |= dx
        edge[1:] |= dy; edge[:-1] |= dy
        if edge.any(): e_band[frame] = np.clip(1-distance_transform_edt(~edge)/16, 0, 1)
    support, _, distance = geometry(shape, generated)
    band = np.zeros((h, w), np.float32)
    for region in generated:
        x, y, rw, rh = grid_rois(h, w)[region]
        yy, xx = np.mgrid[:rh, :rw]
        d = np.minimum.reduce([xx, rw-1-xx, yy, rh-1-yy])
        band[y:y+rh, x:x+rw] = np.clip(1-d/24., 0, 1)
    band *= np.clip(distance/4., 0, 1)
    return dict(precision=precision, e_band=e_band,
                g_band=np.broadcast_to(band, (t, h, w)),
                g_support=np.broadcast_to(support.astype(np.float32), (t, h, w)))


def tensors(arrays, centers, crop=None, device='cpu'):
    """RGB arrays are uint8 THWC, maps are float THW. No source in this API."""
    t, h, w, _ = arrays['received'].shape
    x, y, cw, ch = crop if crop is not None else (0, 0, w, h)
    neighbors = np.array([[max(0, i-1), i, min(t-1, i+1)] for i in centers])
    result = {}
    for key in ('base', 'received', 'current', 'multiband'):
        a = arrays[key][neighbors, y:y+ch, x:x+cw].transpose(0, 1, 4, 2, 3)
        a = np.ascontiguousarray(a.reshape(len(centers), 9, ch, cw))
        result[key] = torch.from_numpy(a).to(device=device, dtype=torch.float32)/255.
    pe = arrays['precision'][neighbors, y:y+ch, x:x+cw]
    result['precision'] = torch.from_numpy(np.ascontiguousarray(pe)).to(device=device)
    for key in ('e_band', 'g_band', 'g_support'):
        a = arrays[key][centers, y:y+ch, x:x+cw, None].transpose(0, 3, 1, 2)
        result[key] = torch.from_numpy(np.ascontiguousarray(a)).to(device=device)
    return result
