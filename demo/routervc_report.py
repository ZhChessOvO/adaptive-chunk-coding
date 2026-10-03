"""CPU-only report of saved, freshly decoded RouterVC streams.

No inference, source-based routing, metric recalculation, or isolated-tile
montage is performed here. Quality and audited bytes come from summary.json;
the pixel denominator comes from each actual reconstruction NPZ header.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np


METRICS = ('lpips_alex', 'psnr_db', 'temporal_delta_mae')
STATES = ('B', 'E', 'G', 'EG')
METHODS = ('context_raw', 'context_smooth', 'local_raw', 'local_smooth',
           'base', 'e_only', 'g_only', 'full_g', 'uf_qp32')
COLORS = {'context_raw': '#2768a3', 'context_smooth': '#12509c',
          'local_raw': '#cf8a30', 'local_smooth': '#b36314', 'base': '#666666',
          'e_only': '#358752', 'g_only': '#9266b0', 'full_g': '#c44c71',
          'uf_qp32': '#111111'}
STATE_COLORS = ('#cccccc', '#45a778', '#9467bd', '#e4a23b')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def npz_shape(path, key='reconstruction'):
    """Read the NPY header only; do not inflate every saved video for BPP."""
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f'unsupported NPY header version: {version}')
        shape, _, dtype = readers[version](stream)
    if len(shape) != 4 or shape[-1] != 3 or any(int(n) <= 0 for n in shape):
        raise ValueError(f'expected positive THW3 RGB geometry, got {shape}')
    if dtype != np.dtype('uint8'):
        raise ValueError(f'expected saved uint8 RGB, got {dtype}')
    return tuple(int(n) for n in shape)


def bits_per_pixel(byte_count, shape):
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError('actual stream byte count must be a positive integer')
    if len(shape) != 4 or shape[-1] != 3 or any(int(n) <= 0 for n in shape):
        raise ValueError('BPP requires THW3 geometry')
    return byte_count * 8 / math.prod(shape[:3])


def route_states(route):
    values = route.get('states')
    if values is None:
        coverage = route.get('coverage', [0.] * 16)
        if len(coverage) != 16:
            raise ValueError('RouterVC report expects a 4 x 4 grid')
        indices = route.get('indices', [])
        if any(type(i) is not int or not 0 <= i < 16 for i in indices):
            raise ValueError('invalid Generate cell index')
        generated = set(indices)
        values = [('EG' if coverage[i] > 0 else 'G') if i in generated
                  else ('E' if coverage[i] > 0 else 'B') for i in range(16)]
    if len(values) != 16:
        raise ValueError('RouterVC report expects 16 states')
    output = []
    for value in values:
        if isinstance(value, str) and value in STATES:
            output.append(STATES.index(value))
        elif type(value) is int and 0 <= value < 4:
            output.append(value)
        else:
            raise ValueError(f'invalid routing state: {value}')
    return output


def grid_statistics(states):
    """Internal G/non-G edges and 4-connected G components, not ROI calls."""
    if len(states) != 16 or any(type(s) is not int or not 0 <= s < 4 for s in states):
        raise ValueError('expected 16 integer four-state labels')
    mask = np.asarray(states).reshape(4, 4) >= 2
    edges = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1]) +
                np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    pending = set(zip(*np.where(mask)))
    components = 0
    while pending:
        components += 1
        stack = [pending.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
    return dict(boundary_edges=edges, components=components,
                counts={name: states.count(i) for i, name in enumerate(STATES)})


def finite_number(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative')
    return float(value)


def normalize_record(record, shape):
    """Pure, validated projection used by both report and unit tests."""
    quality = {name: finite_number(record['quality'][name], name) for name in METRICS}
    decode = record['decode']
    route = decode.get('route', {})
    states = route_states(route)
    geometry = grid_statistics(states)
    for name in ('boundary_edges', 'components'):
        recorded = route.get(name)
        # A list of explicit components is also a permitted receiver detail.
        if isinstance(recorded, list):
            recorded = len(recorded)
        if recorded is not None and int(recorded) != geometry[name]:
            raise ValueError(f'{name} does not match the decoded routing states')
    ratio = record.get('ratio')
    if ratio is not None:
        ratio = finite_number(ratio, 'ratio', True)
    max_g = record.get('max_g')
    if max_g is not None and (type(max_g) is not int or not 0 <= max_g <= 16):
        raise ValueError('invalid max_g')
    return dict(sample_id=str(record['sample_id']), dataset=str(record['dataset']),
        method=str(record['method']), ratio=ratio, max_g=max_g,
        bytes=record['bytes'], shape=list(shape), pixels=math.prod(shape[:3]),
        bpp=bits_per_pixel(record['bytes'], shape), quality=quality,
        decode_seconds=finite_number(decode['seconds'], 'decode seconds', True),
        peak_cuda_allocated_bytes=finite_number(decode['peak_cuda_allocated_bytes'], 'peak memory', True),
        states=states, **geometry)


def summarize(records):
    """Each operating point has its own sample support; no budget averaging."""
    grouped = defaultdict(list)
    seen = set()
    for record in records:
        identity = tuple(record[k] for k in ('dataset', 'sample_id', 'method', 'ratio', 'max_g'))
        if identity in seen:
            raise ValueError(f'duplicate operating point: {identity}')
        seen.add(identity)
        for domain in (record['dataset'], 'All'):
            grouped[(domain, record['method'], record['ratio'], record['max_g'])].append(record)
    result = []
    for (domain, method, ratio, max_g), rows in sorted(grouped.items(), key=lambda x: str(x[0])):
        result.append(dict(dataset=domain, method=method, ratio=ratio, max_g=max_g,
            windows=len(rows), sample_ids=sorted(r['sample_id'] for r in rows),
            quality={name: float(np.mean([r['quality'][name] for r in rows])) for name in METRICS},
            bytes_mean=float(np.mean([r['bytes'] for r in rows])),
            bpp_mean=float(np.mean([r['bpp'] for r in rows])),
            bpp_pooled=sum(r['bytes'] for r in rows)*8/sum(r['pixels'] for r in rows),
            decode_seconds_mean=float(np.mean([r['decode_seconds'] for r in rows])),
            peak_cuda_allocated_bytes_max=max(r['peak_cuda_allocated_bytes'] for r in rows),
            boundary_edges_mean=float(np.mean([r['boundary_edges'] for r in rows])),
            components_mean=float(np.mean([r['components'] for r in rows])),
            states_mean={state: float(np.mean([r['counts'][state] for r in rows])) for state in STATES}))
    return result


def fixed_selection(records):
    """First supplied sample per domain, never best-looking/lowest-loss sample."""
    selected = {}
    for record in records:
        if record['dataset'] in ('REDS', 'UVG'):
            selected.setdefault(record['dataset'], record['sample_id'])
    return selected


def choose_operating_point(records, method):
    candidates = [r for r in records if r['method'] == method]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (abs((r.get('ratio') or 0.)-.5),
        abs((r.get('max_g') or 0)-4), str(r.get('folder', ''))))


def slug(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', value)


def save_figure(figure, path):
    temporary = path.with_suffix('.tmp.png')
    figure.savefig(temporary, dpi=160, facecolor='white')
    os.replace(temporary, path)


def draw_charts(destination, aggregates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    artifacts = []
    domains = [d for d in ('REDS', 'UVG', 'All') if any(r['dataset'] == d for r in aggregates)]
    for domain in domains:
        rows = [r for r in aggregates if r['dataset'] == domain]
        budgets = sorted({r['max_g'] for r in rows if r['method'].startswith(('context_', 'local_'))
                          and r['max_g'] is not None}) or [None]
        for max_g in budgets:
            current = [r for r in rows if not r['method'].startswith(('context_', 'local_', 'g_only'))
                       or r['max_g'] == max_g]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            for method in METHODS:
                points = sorted([r for r in current if r['method'] == method], key=lambda r:r['bpp_mean'])
                if not points:
                    continue
                # No line between means over different video subsets.
                matched = len({tuple(r['sample_ids']) for r in points}) == 1
                style = '--' if method.endswith('_raw') else '-'
                if len(points) == 1 or not matched:
                    style = 'None'
                for axis, metric in zip(axes, ('lpips_alex', 'psnr_db')):
                    axis.plot([r['bpp_mean'] for r in points], [r['quality'][metric] for r in points],
                              marker='o', linestyle=style, color=COLORS[method], label=method)
                    for point in points:
                        label = f'n={point["windows"]}'
                        if point['ratio'] is not None:
                            label = f'{point["ratio"]:g}, ' + label
                        axis.annotate(label, (point['bpp_mean'], point['quality'][metric]),
                                      xytext=(3, 4), textcoords='offset points', fontsize=6, alpha=.8)
            axes[0].set_ylabel('Whole-frame LPIPS (lower is better)')
            axes[1].set_ylabel('Whole-frame PSNR / dB (higher is better)')
            for axis in axes:
                axis.set_xlabel('Actual bits / pixel / frame (equal-window mean)')
                axis.grid(alpha=.2)
            axes[1].legend(fontsize=7, loc='best')
            fig.suptitle(f'{domain} | fresh decoded streams | Router G-cell cap={max_g}\n'
                         'Baseline points use their own compute; connected points are observations, not interpolation.', fontsize=10)
            fig.tight_layout()
            filename = f'rd_{slug(domain)}_g{max_g}.png'
            save_figure(fig, destination/filename)
            plt.close(fig)
            artifacts.append(filename)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for method in METHODS:
            points = [r for r in rows if r['method'] == method]
            if not points:
                continue
            axes[0].scatter([r['decode_seconds_mean'] for r in points],
                            [r['quality']['lpips_alex'] for r in points],
                            color=COLORS[method], label=method, s=35, alpha=.8)
            axes[1].scatter([r['boundary_edges_mean'] for r in points],
                            [r['decode_seconds_mean'] for r in points],
                            color=COLORS[method], label=method, s=35, alpha=.8)
        axes[0].set_xlabel('Recorded complete fresh-decode time / s')
        axes[0].set_ylabel('Whole-frame LPIPS (lower is better)')
        axes[1].set_xlabel('Internal Generate / non-Generate grid edges')
        axes[1].set_ylabel('Recorded complete fresh-decode time / s')
        for axis in axes:
            axis.grid(alpha=.2)
        axes[1].legend(fontsize=7)
        fig.suptitle(f'{domain} | actual receiver cost; G-cell cap is not a strict time budget', fontsize=10)
        fig.tight_layout()
        filename = f'cost_{slug(domain)}.png'
        save_figure(fig, destination/filename)
        plt.close(fig)
        artifacts.append(filename)
    return artifacts


def path_from_root(root, value):
    path = Path(value)
    return path if path.is_absolute() else root/path


def reconstruction_path(root, record):
    return path_from_root(root, record['folder'])/'fresh/reconstruction.npz'


def draw_fixed(destination, root, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    artifacts, selected, source_hashes = [], [], {}
    for domain, sid in fixed_selection(records).items():
        same = [r for r in records if r['dataset'] == domain and r['sample_id'] == sid]
        point = choose_operating_point(same, 'context_smooth')
        if point is None:
            point = next((choose_operating_point(same, m) for m in METHODS
                          if choose_operating_point(same, m) is not None), None)
        source_path = path_from_root(root, point['source_path'])
        source_hashes[str(source_path)] = file_hash(source_path)
        with np.load(source_path, allow_pickle=False) as cache:
            source = cache['source']
        if len(source) < 9:
            raise ValueError('fixed frame 9 is unavailable; do not silently change frame')
        with np.load(reconstruction_path(root, point), allow_pickle=False) as cache:
            arrays = {name: cache[name] for name in ('base', 'enhanced', 'reconstruction')}
        if any(value.shape != source.shape for value in arrays.values()):
            raise ValueError('source and decoded outputs have different geometry')
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
        panels = [('Source (evaluation only)', source), ('Shared UF base', arrays['base']),
                  ('Received E (same stream)', arrays['enhanced']),
                  (f'Final: {point["method"]}', arrays['reconstruction'])]
        for axis, (title, value) in zip(axes, panels):
            axis.imshow(value[8]);axis.set_title(title, fontsize=10);axis.axis('off')
        fig.suptitle(f'{domain} | {sid} | fixed frame 9 | actual full-frame receiver output\n'
                     f'ratio={point.get("ratio")}, G cap={point.get("max_g")}, '
                     f'{point["bytes"]} bytes, LPIPS={point["quality"]["lpips_alex"]:.4f}', fontsize=11)
        fig.tight_layout()
        filename = f'fixed_{slug(domain)}_{slug(sid)}.png'
        save_figure(fig, destination/filename);plt.close(fig);artifacts.append(filename)

        methods = [m for m in METHODS if choose_operating_point(same, m) is not None]
        fig, axes = plt.subplots(math.ceil((len(methods)+1)/4), 4,
                                 figsize=(16, 4.1*math.ceil((len(methods)+1)/4)), squeeze=False)
        for axis in axes.flat:
            axis.axis('off')
        axes.flat[0].imshow(source[8]);axes.flat[0].set_title('Source (evaluation only)', fontsize=10)
        for axis, method in zip(list(axes.flat)[1:], methods):
            current = choose_operating_point(same, method)
            with np.load(reconstruction_path(root, current), allow_pickle=False) as cache:
                pixels = cache['reconstruction']
            if pixels.shape != source.shape:
                raise ValueError('fixed comparison geometry mismatch')
            axis.imshow(pixels[8])
            axis.set_title(f'{method} | {current["bytes"]} B\n'
                           f'LPIPS {current["quality"]["lpips_alex"]:.4f} | '
                           f'r={current.get("ratio")}, G<={current.get("max_g")}', fontsize=9)
        fig.suptitle(f'{domain} | {sid} | frame 9; fixed mid-budget choices, not equal-rate claims', fontsize=11)
        fig.tight_layout()
        filename = f'overview_{slug(domain)}_{slug(sid)}.png'
        save_figure(fig, destination/filename);plt.close(fig);artifacts.append(filename)

        route_methods = [m for m in METHODS[:4] if choose_operating_point(same, m) is not None]
        if route_methods:
            fig, axes = plt.subplots(1, len(route_methods), figsize=(4.1*len(route_methods), 4.5), squeeze=False)
            for axis, method in zip(axes.flat, route_methods):
                current = choose_operating_point(same, method)
                values = route_states(current['decode'].get('route', {}))
                stats = grid_statistics(values)
                axis.imshow(np.asarray(values).reshape(4, 4), cmap=ListedColormap(STATE_COLORS), vmin=0, vmax=3)
                for i, value in enumerate(values):
                    axis.text(i%4, i//4, STATES[value], ha='center', va='center', fontsize=13)
                axis.set_xticks(np.arange(-.5, 4, 1), minor=True)
                axis.set_yticks(np.arange(-.5, 4, 1), minor=True)
                axis.grid(which='minor', color='white', linewidth=2)
                axis.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
                axis.set_title(f'{method}\nG edges={stats["boundary_edges"]}, components={stats["components"]}', fontsize=10)
            fig.suptitle(f'{domain} | {sid}\nB: neither; E: packet only; G: generation only; EG: packet then generation', fontsize=11)
            fig.tight_layout()
            filename = f'route_{slug(domain)}_{slug(sid)}.png'
            save_figure(fig, destination/filename);plt.close(fig);artifacts.append(filename)
        selected.append(dict(dataset=domain, sample_id=sid, frame_1based=9,
                             method=point['method'], ratio=point.get('ratio'), max_g=point.get('max_g')))
    return artifacts, selected, source_hashes


def report(root):
    root = Path(root)
    summary_path = root/'summary.json'
    source = read_json(summary_path)
    records = source['records']
    if not records:
        raise ValueError('no completed fresh-stream records; report cannot invent missing results')
    destination = root/'report'
    destination.mkdir(exist_ok=True)
    done = destination/'summary.json'
    dependencies = dict(summary_sha256=file_hash(summary_path), code_sha256=file_hash(__file__))
    if done.exists():
        old = read_json(done)
        if old['dependencies'] != dependencies:
            raise ValueError('report inputs changed; preserve old report and use a new report directory')
        for path, expected in old['input_hashes'].items():
            if file_hash(path) != expected:
                raise ValueError(f'changed input artifact: {path}')
        for name, expected in old['artifacts'].items():
            if file_hash(destination/name) != expected:
                raise ValueError(f'changed report artifact: {name}')
        return old
    normalized, hashes = [], {}
    for record in records:
        path = reconstruction_path(root, record)
        shape = npz_shape(path)
        for name in ('base', 'enhanced'):
            if npz_shape(path, name) != shape:
                raise ValueError(f'{name} has wrong saved geometry')
        if str(path) not in hashes:
            hashes[str(path)] = file_hash(path)
        normalized.append(normalize_record(record, shape))
    aggregates = summarize(normalized)
    artifacts = draw_charts(destination, aggregates)
    fixed, selected, source_hashes = draw_fixed(destination, root, records)
    artifacts.extend(fixed);hashes.update(source_hashes)
    result = dict(complete=True, source_complete=source.get('complete', False), dependencies=dependencies,
        records=normalized, aggregates=aggregates, fixed_visuals=selected,
        methods_present=sorted({r['method'] for r in records}),
        methods_missing=[m for m in METHODS if m not in {r['method'] for r in records}],
        input_hashes=hashes, artifacts={name:file_hash(destination/name) for name in artifacts},
        scope='Saved whole-frame fresh decodes; no isolated-region table results; development evidence.',
        byte_source='Audited actual stream bytes from input summary; BPP denominator read from saved RGB geometry.',
        averaging='Quality and RD bpp are equal-window means; pooled bpp is also supplied. All is not dataset-balanced.',
        timing='Recorded receiver seconds, not estimated ROI area or strict time budgets; encoding is not included.',
        comparison='No BD-rate or universal dominance claim; missing methods and unequal sample support stay visible.',
        no_inference=True, no_metric_recalculation=True)
    atomic_json(done, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    result = report(parser.parse_args().output)
    print(json.dumps(dict(complete=result['complete'], source_complete=result['source_complete'],
                          records=len(result['records']), figures=len(result['artifacts']),
                          methods_missing=result['methods_missing'])))
