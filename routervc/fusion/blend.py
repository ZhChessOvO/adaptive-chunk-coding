"""No-training overlap/multiband controls; no source, extra G call, or sent mask."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from demo.routervc_policy import grid_rois


def geometry(shape, regions, radius=24):
    _, h, w, _ = shape
    support, band = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for i in regions:
        x, y, rw, rh = grid_rois(h, w)[i]
        yy, xx = np.mgrid[:rh, :rw]
        d = np.minimum.reduce([xx, rw-1-xx, yy, rh-1-yy])
        support[y:y+rh, x:x+rw] = True
        band[y:y+rh, x:x+rw] = d < radius
    distance = np.maximum(distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]-1, 0)
    return support, band, distance.astype(np.float32)


def candidates(received, patches):
    """Overlap-weighted raw residual; only selected G cores may consume it."""
    t, h, w, _ = received.shape
    mass = np.zeros((h, w), np.float32)
    total = np.zeros(received.shape, np.float32)
    for patch in patches:
        x0, y0, cw, ch = patch['crop']; x, y, rw, rh = patch['core']
        yy, xx = np.mgrid[y0:y0+ch, x0:x0+cw]
        signed = np.minimum.reduce([xx-x, x+rw-1-xx, yy-y, y+rh-1-yy])
        alpha = np.clip((signed+16.)/32., 0, 1).astype(np.float32)
        residual = patch['pixels'].astype(np.float32)-received[:, y0:y0+ch, x0:x0+cw]
        total[:, y0:y0+ch, x0:x0+cw] += residual*alpha[None, ..., None]
        mass[y0:y0+ch, x0:x0+cw] += alpha
    return total/np.maximum(mass[None, ..., None], 1e-8)


def fuse(received, current, patches, regions):
    support, band, distance = geometry(received.shape, regions)
    if not regions:
        return dict(overlap=current.copy(), multiband=current.copy())
    delta = candidates(received, patches)
    low = gaussian_filter(delta, sigma=(0, 3, 3, 0), mode='nearest')
    high = delta-low
    output = {}
    for name, residual in (
        ('overlap', np.clip(distance/16, 0, 1)[None, ..., None]*delta),
        ('multiband', np.clip(distance/24, 0, 1)[None, ..., None]*low +
         np.clip(distance/8, 0, 1)[None, ..., None]*high)):
        values = np.rint(received.astype(np.float32)+residual).clip(0, 255).astype(np.uint8)
        # Deliberately preserve all existing non-boundary interior pixels.
        result = current.copy(); result[:, band] = values[:, band]
        np.testing.assert_array_equal(result[:, ~support], received[:, ~support])
        np.testing.assert_array_equal(result[:, ~band], current[:, ~band])
        output[name] = result
    return output
