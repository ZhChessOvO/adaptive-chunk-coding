"""Source-free shared RouterVC policy and optional Generate-boundary reduction.

Public interface (all inference is on CPU):

* ``grid_rois(height, width)`` -> 16 raster-ordered ``[x, y, w, h]`` cells.
  Dimensions must be multiples of eight and at least 64. The 4x4 topology is
  fixed by the trained backbone, but rectangular and unequal edge cells work.
* ``features(base, received, rois, coverage)`` -> float32 ``[1, 16, 37]``.
  Both videos are uint8 ``[T, H, W, 3]``, T >= 2. They are the decoded base B
  and actual received candidate Y, NOT source X or isolated teacher montages.
  Coverage is the received E fraction per cell in [0, 1]. Current training used
  only zero/one coverage and 17-frame 512px windows; other geometry and partial
  temporal coverage are supported mechanically, not claimed to be calibrated.
* ``load_model(path)`` -> the existing 6,982-parameter utility backbone.
  Caller must authenticate the model hash against its stream/profile. Loading
  uses the historical ``four_state_router_evaluate.model_from_bundle`` API;
  neither its table evaluation nor any source/teacher dataset is accessed.
* ``predict(model, features)`` -> float32 ``[1, 16, 6]`` in TARGETS order:
  direct LPIPS, conditional G LPIPS, direct PSNR, conditional G PSNR,
  direct temporal, conditional G temporal gains. Larger means better.
* ``select_generate(gains, max_g, boundary_lambda)`` -> JSON-safe dict.
  ``gains`` is the 16 CONDITIONAL G(Y) LPIPS gains (prediction channel one),
  never G(B) gains added to an already enhanced candidate. Exhaustively maximize
  sum of selected gains minus lambda times internal G/non-G adjacency edges.
  Equal scores prefer fewer G cells, fewer boundary edges, then smaller bitmask.

The returned mask does not change E packets or merge tiles into execution ROIs.
Boundary reduction and output feathering remain separate. ``max_g`` limits ROI
calls, not measured milliseconds. Lambda zero is the unregularized baseline;
0.004 is a candidate inherited scale, not a demonstrated optimal value.
"""
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from demo.four_state_router import TARGETS
from demo.four_state_router_evaluate import model_from_bundle


REGIONS = 16
FEATURES = 37
GRID_SIDE = 4
BOUNDARY_CANDIDATE = 0.004
CPU_THREADS = 4


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} outside supported range')
    return value


def grid_rois(height, width):
    """Cover the complete frame with aligned cells, without resize or cropping."""
    height = _integer(height, 'height', 64)
    width = _integer(width, 'width', 64)
    if height % 8 or width % 8:
        raise ValueError('height and width must be multiples of eight')
    ys = [8 * (i * (height // 8) // GRID_SIDE) for i in range(GRID_SIDE + 1)]
    xs = [8 * (i * (width // 8) // GRID_SIDE) for i in range(GRID_SIDE + 1)]
    return [[xs[x], ys[y], xs[x + 1] - xs[x], ys[y + 1] - ys[y]]
            for y in range(GRID_SIDE) for x in range(GRID_SIDE)]


def _rois(rois, height, width):
    """The trained context pooling assumes this exact raster order/topology."""
    arr = np.asarray(rois)
    if arr.shape != (REGIONS, 4) or arr.dtype.kind not in 'iu':
        raise ValueError('rois must contain 16 integer [x, y, w, h] entries')
    expected = grid_rois(height, width)
    if not np.array_equal(arr, expected):
        raise ValueError('rois must equal grid_rois(height, width) in raster order')
    return expected


def _patch_features(pixels, roi):
    # Preserve the historical patch_features reduction order and float32 casts
    # exactly; replacing these with approximations changes a shared policy.
    x, y, w, h = roi
    values = pixels[:, y:y+h, x:x+w].astype(np.float32) / 255
    luma = values @ np.array([.2126, .7152, .0722], np.float32)
    dx, dy = np.abs(np.diff(luma, axis=2)), np.abs(np.diff(luma, axis=1))
    dt = np.abs(np.diff(luma, axis=0))
    return np.array([
        *values.mean(axis=(0, 1, 2)), *values.std(axis=(0, 1, 2)),
        luma.mean(), luma.std(), dx.mean(), dx.std(), dy.mean(), dy.std(),
        dt.mean(), dt.std(), luma.mean(axis=(1, 2)).std(),
        (x+w/2)/pixels.shape[2], (y+h/2)/pixels.shape[1],
        float(x == 0 or y == 0 or x+w == pixels.shape[2] or y+h == pixels.shape[1]),
    ], np.float32)


def features(base, received, rois, coverage):
    """Extract B/Y-only features; never reads files, labels or source pixels."""
    base, received = np.asarray(base), np.asarray(received)
    if base.ndim != 4 or base.shape[-1] != 3 or base.shape[0] < 2:
        raise ValueError('base must have shape [T>=2, H, W, 3]')
    if received.shape != base.shape:
        raise ValueError('base and received videos must have identical shapes')
    if base.dtype != np.uint8 or received.dtype != np.uint8:
        raise ValueError('decoded videos must be uint8 RGB, not normalized floats')
    checked = _rois(rois, base.shape[1], base.shape[2])
    coverage = np.asarray(coverage)
    if (coverage.shape != (REGIONS,) or coverage.dtype.kind not in 'biuf'
            or not np.isfinite(coverage).all()
            or np.any(coverage < 0) or np.any(coverage > 1)):
        raise ValueError('coverage must contain 16 finite fractions in [0, 1]')
    bf = np.stack([_patch_features(base, roi) for roi in checked])
    yf = np.stack([_patch_features(received, roi) for roi in checked])
    result = np.concatenate((bf, yf, coverage.astype(np.float32)[:, None]), axis=1)
    return torch.from_numpy(result[None])


def load_model(path):
    """Load the existing CPU bundle; stream/profile hash verification is external."""
    torch.set_num_threads(CPU_THREADS)
    model = model_from_bundle(Path(path))
    if sum(p.numel() for p in model.parameters()) != 6982:
        raise ValueError('unsupported Router backbone size')
    if model.mean.shape != (FEATURES,) or model.target_mean.shape != (len(TARGETS),):
        raise ValueError('unsupported Router feature/target dimensions')
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite Router weights or normalization')
    if torch.any(model.scale <= 0) or torch.any(model.target_scale <= 0):
        raise ValueError('Router normalization scales must be positive')
    return model


@torch.no_grad()
def predict(model, input_features):
    """Run one body on the actual mixed candidate, with no feature experts."""
    if torch.get_num_threads() != CPU_THREADS:
        torch.set_num_threads(CPU_THREADS)
    if (not torch.is_tensor(input_features)
            or input_features.shape != (1, REGIONS, FEATURES)
            or input_features.dtype != torch.float32
            or input_features.device.type != 'cpu'
            or not torch.isfinite(input_features).all()):
        raise ValueError('features must be a finite CPU float32 [1, 16, 37] tensor')
    coverage = input_features[..., -1]
    if torch.any(coverage < 0) or torch.any(coverage > 1):
        raise ValueError('feature coverage must be in [0, 1]')
    if any(v.device.type != 'cpu' for v in (*model.parameters(), *model.buffers())):
        raise ValueError('shared policy inference is CPU-only')
    model.eval()
    result = model(input_features)
    if result.shape != (1, REGIONS, len(TARGETS)) or not torch.isfinite(result).all():
        raise ValueError('invalid Router predictions')
    return result


def _grid_edges(rows, columns):
    return tuple((y*columns+x, ny*columns+nx)
                 for y in range(rows) for x in range(columns)
                 for ny, nx in ((y, x+1), (y+1, x))
                 if ny < rows and nx < columns)


@lru_cache(maxsize=8)
def _enumeration(n, edges):
    """Cache bounded topology only, not private pixels or predicted gains."""
    bits = np.arange(1 << n, dtype=np.uint32)
    masks = ((bits[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    counts = masks.sum(axis=1)
    boundaries = np.zeros(len(bits), dtype=np.uint8)
    for a, b in edges:
        boundaries += masks[:, a] != masks[:, b]
    for value in (bits, masks, counts, boundaries):
        value.flags.writeable = False
    return bits, masks, counts, boundaries


def _components(indices, edges):
    remaining = set(indices)
    graph = {i: set() for i in indices}
    for a, b in edges:
        if a in remaining and b in remaining:
            graph[a].add(b)
            graph[b].add(a)
    components = []
    while remaining:
        todo = [min(remaining)]
        component = []
        while todo:
            node = todo.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            component.append(node)
            todo.extend(sorted(graph[node] & remaining, reverse=True))
        components.append(sorted(component))
    return components


def _select(gains, max_g, boundary_lambda, *, rows, columns):
    """Small-grid implementation also exposed internally for exhaustive tests."""
    n = rows * columns
    if not 1 <= n <= REGIONS:
        raise ValueError('enumeration supports 1..16 cells')
    gains = np.asarray(gains)
    if gains.shape != (n,) or gains.dtype.kind not in 'iuf' or not np.isfinite(gains).all():
        raise ValueError(f'gains must contain {n} finite numbers')
    gains = gains.astype(np.float64)
    max_g = _integer(max_g, 'max_g', 0, n)
    if (isinstance(boundary_lambda, (bool, np.bool_))
            or not isinstance(boundary_lambda, Real)
            or not np.isfinite(boundary_lambda) or boundary_lambda < 0):
        raise ValueError('boundary_lambda must be finite and nonnegative')
    boundary_lambda = float(boundary_lambda)
    edges = _grid_edges(rows, columns)
    bits, masks, counts, boundaries = _enumeration(n, edges)
    gains_sum = (masks * gains[None]).sum(axis=1)
    objective = gains_sum - boundary_lambda * boundaries
    admissible = np.flatnonzero(counts <= max_g)
    best_score = objective[admissible].max()
    candidates = admissible[objective[admissible] == best_score]
    candidates = candidates[counts[candidates] == counts[candidates].min()]
    candidates = candidates[boundaries[candidates] == boundaries[candidates].min()]
    index = int(candidates[0])  # Enumeration is ascending integer bitmask.
    indices = np.flatnonzero(masks[index]).tolist()
    groups = _components(indices, edges)
    return dict(indices=indices, boundary_edges=int(boundaries[index]),
                components=len(groups), component_indices=groups,
                predicted_gain=float(gains_sum[index]), objective=float(objective[index]),
                bitmask=int(bits[index]), g_calls=len(indices), max_g=max_g,
                boundary_lambda=boundary_lambda)


def select_generate(gains, max_g, boundary_lambda=0.0):
    """Select 0..max_g cells exactly on the fixed 4x4 Generate/non-G graph."""
    return _select(gains, max_g, boundary_lambda, rows=GRID_SIDE, columns=GRID_SIDE)
