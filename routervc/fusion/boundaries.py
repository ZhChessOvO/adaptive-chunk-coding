"""Address-derived grid boundaries, with source-referenced (not smoothing) errors."""
import numpy as np

from demo.routervc_policy import grid_rois

CATEGORIES = ('E_nonE', 'G_nonG', 'G_G')


def edges(shape, received_regions, generated):
    """Return each physical edge once per category with its eligible frames.

    E addresses describe latent precision, not exact RGB support after synthesis.
    The I bootstrap has no E and is excluded from E/non-E measurements.
    Categories may overlap; they are not an image segmentation.
    """
    t, h, w, channels = shape
    if channels != 3 or t != 17:
        raise ValueError('expected the current 17-frame RGB profile')
    e = np.zeros((t, 16), bool)
    for chunk, region in received_regions:
        if chunk not in (0, 1) or not 0 <= region < 16:
            raise ValueError('invalid E address')
        e[1 + chunk * 8:1 + (chunk + 1) * 8, region] = True
    g = np.zeros(16, bool)
    if len(set(generated)) != len(generated) or any(not 0 <= i < 16 for i in generated):
        raise ValueError('invalid G addresses')
    g[generated] = True
    rois = grid_rois(h, w)
    result = []
    for a, (x, y, rw, rh) in enumerate(rois):
        neighbors = []
        if a % 4 < 3:
            neighbors.append((a + 1, 'x', x + rw, y, y + rh))
        if a // 4 < 3:
            neighbors.append((a + 4, 'y', y + rh, x, x + rw))
        for b, axis, pos, lo, hi in neighbors:
            selections = (e[:, a] != e[:, b],
                          np.full(t, g[a] != g[b]),
                          np.full(t, g[a] and g[b]))
            for category, selected in zip(CATEGORIES, selections):
                frames = np.flatnonzero(selected).tolist()
                if frames:
                    result.append(dict(category=category, a=a, b=b, axis=axis,
                                       pos=pos, lo=lo, hi=hi, frames=frames))
    return result


def strip(pixels, edge, radius=24):
    """Return [T, along_edge, distance_to_edge, RGB], negative side first."""
    p, lo, hi = edge['pos'], edge['lo'], edge['hi']
    if edge['axis'] == 'x':
        out = pixels[:, lo:hi, p-radius:p+radius]
    else:
        out = pixels[:, p-radius:p+radius, lo:hi].transpose(0, 2, 1, 3)
    if out.shape[2] != 2 * radius:
        raise ValueError('boundary strip exceeds frame')
    return out.astype(np.float32) / 255.


def measure(source, output, edge, radius=24, band=16):
    """Unit-range RGB errors. A real image edge is not penalized if preserved."""
    s, o = strip(source, edge, radius), strip(output, edge, radius)
    f = edge['frames']
    err = o - s
    grad_error = np.diff(err[f], axis=2)
    temporal = [i for i in f if i > 0 and i-1 in f]
    band_error = err[:, :, radius-band:radius+band]
    detail = np.abs(np.diff(o[f], axis=2)).mean(axis=(0, 1, 3))
    return dict(edge_pixels=len(f) * (edge['hi']-edge['lo']),
                crossing_gradient_mae=float(np.abs(grad_error[:, :, radius-1]).mean()),
                band_mae=float(np.abs(band_error[f]).mean()),
                band_temporal_mae=(float(np.abs(band_error[temporal] -
                    band_error[np.array(temporal)-1]).mean()) if temporal else None),
                normal_detail_profile=detail.tolist())


def aggregate(records):
    """Within one view: edge-pixel weighted; across views: handled separately."""
    if not records:
        return None
    weights = np.array([r['edge_pixels'] for r in records], np.float64)
    out = dict(edge_pixels=int(weights.sum()), edges=len(records))
    for key in ('crossing_gradient_mae', 'band_mae', 'band_temporal_mae', 'normal_detail_profile'):
        valid = [i for i, r in enumerate(records) if r[key] is not None]
        out[key] = np.average([records[i][key] for i in valid], axis=0,
                             weights=weights[valid]).tolist() if valid else None
    return out
