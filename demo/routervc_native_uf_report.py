"""CPU-only saved-result comparison against standard native DCVC-UF bytes.

UF's experimental JSON sidecar is a separate diagnostic curve, not standard
codec syntax. RouterVC still pays for its entire transmitted stream. This file
does not change historical reports, metrics, streams, or pinned implementations.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.routervc_analysis import (COLORS, METRICS, ROUTERS, UF_METHODS,
    auxiliary_interpolation, common_support, indexed, load_records, point_key,
    read, record_key, sample_key)
from demo.routervc_report import atomic_json, bits_per_pixel, file_hash, npz_shape, save_figure

DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_20261003/supplement')
CODE = ('routervc_native_uf_report.py', 'run_routervc_native_uf_report.sh',
        'routervc_analysis.py', 'routervc_report.py', 'chunk_enhancement_experiment.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def immutable(path, value):
    if path.exists():
        require(read(path) == value, f'changed binding: {path}; use a new output directory')
    else:
        atomic_json(path, value)


def code_hashes():
    return {name: file_hash(REPO/'demo'/name) for name in CODE}


def checked(path, expected=None, hashes=None):
    path = Path(path).resolve()
    actual = file_hash(path)
    if expected is not None:
        require(actual == expected, f'changed saved evidence: {path}')
    if hashes is not None:
        hashes[str(path)] = actual
    return actual


def verify_native_point(raw, norm, root, hashes):
    """Use each native file's actual bytes and each decoded video's geometry."""
    require(raw['method'] in UF_METHODS and raw['kind'] == 'uf', 'expected a measured UF baseline')
    qp = int(raw['method'].removeprefix('uf_qp'))
    require(raw['qp'] == qp, 'UF method/QP mismatch')
    folder = Path(raw['folder'])
    if not folder.is_absolute():
        folder = root/folder
    stream, sidecar = folder/'stream.bin', folder/'transmitted_meta.json'
    d = raw['decode']
    native = stream.stat().st_size
    metadata = sidecar.stat().st_size
    require(native > 0 and native == raw['native_bytes'] == norm['native_bytes'] == d['native_bytes'],
            'native UF file size differs from recorded native bytes')
    require(metadata == d['metadata_bytes'] and native+metadata == raw['bytes'] == norm['bytes'] == d['total_bytes'],
            'UF native/sidecar byte accounting mismatch')
    require(raw['non_native_bytes'] == metadata, 'unexplained non-native UF bytes')
    stream_hash = checked(stream, d['stream_sha256'], hashes)
    sidecar_hash = checked(sidecar, d['metadata_sha256'], hashes)
    require(raw['artifacts']['stream.bin'] == stream_hash and
            raw['artifacts']['transmitted_meta.json'] == sidecar_hash, 'UF artifact hash mismatch')
    meta = read(sidecar)
    require(meta['format'] == 'RouterVC_native_UF_sidecar_v1' and
            meta['qp'] == qp and meta['stream_sha256'] == stream_hash and
            meta['base_rgb_sha256'] == d['base_hash'], 'UF sidecar disagrees with measured decode')
    shape = npz_shape(folder/'fresh/reconstruction.npz')
    require(list(shape) == norm['shape'] and
            list(shape) == [meta['frame_count'], meta['height'], meta['width'], 3],
            'UF decoded geometry differs from report/sidecar')
    encoded = read(folder/'encode.json')
    require(encoded['complete'] and encoded['qp'] == qp and encoded['native_bytes'] == native and
            encoded['metadata_bytes'] == metadata and encoded['total_bytes'] == native+metadata,
            'UF encoder byte/QP evidence differs')
    require(encoded['artifacts']['stream.bin'] == stream_hash and
            encoded['artifacts']['transmitted_meta.json'] == sidecar_hash, 'UF encoder hash evidence differs')
    provenance = dict(dataset=raw['dataset'], sample_id=raw['sample_id'], method=raw['method'], qp=qp,
        stream=str(stream.resolve()), stream_sha256=stream_hash, native_bytes=native,
        sidecar=str(sidecar.resolve()), sidecar_sha256=sidecar_hash, sidecar_bytes=metadata,
        charged_sidecar_total_bytes=native+metadata, shape=list(shape),
        native_bpp=bits_per_pixel(native, shape),
        charged_sidecar_bpp=bits_per_pixel(native+metadata, shape))
    require(provenance['charged_sidecar_bpp'] == norm['bpp'], 'saved total UF BPP differs from actual geometry')
    return dict(norm, bytes=native, bpp=provenance['native_bpp']), provenance


def curve_points(records, support, cap):
    """No pooling mismatched sample sets or converting an averaged byte ratio."""
    selected = set(map(tuple, support))
    groups = defaultdict(list)
    for row in records:
        if sample_key(row) not in selected:
            continue
        if row['method'] not in UF_METHODS and not (row['method'] in ROUTERS and row['max_g'] == cap):
            continue
        groups[point_key(row)].append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        require(len(rows) == len(selected) and {sample_key(r) for r in rows} == selected,
                'different videos at plotted operating points')
        result.append(dict(method=key[0], ratio=key[1], max_g=key[2], windows=len(rows),
            samples=[list(v) for v in sorted(selected)],
            bpp=float(np.mean([r['bpp'] for r in rows])),
            bytes_mean=float(np.mean([r['bytes'] for r in rows])),
            quality={m:float(np.mean([r['quality'][m] for r in rows])) for m in METRICS}))
    return result


def build_curves(original, native, expected_support):
    indexed(original); indexed(native)
    require(indexed(original).keys() == indexed(native).keys(), 'native report changed operating point set')
    curves, supports = [], {}
    for cap in sorted({r['max_g'] for r in original if r['method'] in ROUTERS}):
        support = common_support(original, cap)
        require(bool(support), f'no matched UF/Router support for G cap {cap}')
        supports[str(cap)] = [list(v) for v in support]
        require(supports[str(cap)] == expected_support[str(cap)], 'support differs from completed analysis')
        for domain in ('REDS', 'UVG', 'All'):
            chosen = [v for v in support if domain == 'All' or v[0] == domain]
            if not chosen:
                continue
            points = curve_points(native, chosen, cap)
            sidecar = [r for r in curve_points(original, chosen, cap) if r['method'] in UF_METHODS]
            curves.append(dict(dataset=domain, max_g=cap, samples=[list(v) for v in chosen], points=points,
                uf_charged_sidecar_diagnostic=sidecar,
                missing_router_methods=[m for m in ROUTERS if not any(p['method'] == m for p in points)]))
    require(supports == expected_support, 'G cap set differs from completed analysis')
    return curves, supports


def draw_curves(destination, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs = []
    for group in curves:
        uf = sorted([p for p in group['points'] if p['method'] in UF_METHODS], key=lambda p:p['bpp'])
        diagnostic = sorted(group['uf_charged_sidecar_diagnostic'], key=lambda p:p['bpp'])
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1))
        for axis, metric in zip(axes, ('lpips_alex', 'psnr_db')):
            axis.plot([p['bpp'] for p in uf], [p['quality'][metric] for p in uf], '-o', color='black',
                      lw=2.3, label='DCVC-UF native stream (standard baseline)')
            axis.plot([p['bpp'] for p in diagnostic], [p['quality'][metric] for p in diagnostic], '--o',
                      color='black', lw=1.4, markerfacecolor='white', alpha=.7,
                      label='UF + experimental JSON sidecar (diagnostic)')
            for point in uf:
                axis.annotate(point['method'].replace('uf_', ''), (point['bpp'], point['quality'][metric]),
                              xytext=(4, 5), textcoords='offset points', fontsize=7)
            for method in ROUTERS:
                points = sorted([p for p in group['points'] if p['method'] == method], key=lambda p:p['bpp'])
                if points:
                    axis.plot([p['bpp'] for p in points], [p['quality'][metric] for p in points],
                              marker='o', linestyle='--' if method.endswith('raw') else '-',
                              color=COLORS[method], label=f'RouterVC {method}')
            axis.set_xlabel('Bits / pixel / frame (complete native UF or complete RouterVC stream)')
            axis.grid(alpha=.2)
        axes[0].set_ylabel('Whole-frame LPIPS (lower is better)')
        axes[1].set_ylabel('Whole-frame PSNR / dB (higher is better)')
        axes[1].legend(fontsize=7, loc='best')
        fig.suptitle(f'{group["dataset"]} | {len(group["samples"])} identical videos at every point | Router G cap {group["max_g"]}\n'
                     'Standard UF includes its native SPS/NAL syntax; RouterVC pays for every header and E packet.', fontsize=10)
        fig.text(.5, .015, 'Four development clips at most; saved fresh-decode quality reused. No new inference or BD-rate.',
                 ha='center', fontsize=8, color='#555555')
        fig.tight_layout(rect=(0, .035, 1, 1))
        name = f'rd_native_UF_{group["dataset"]}_g{group["max_g"]}.png'
        save_figure(fig, destination/name); plt.close(fig); outputs.append(name)
    return outputs


def completed_analysis(root, workflow, hashes, run=None):
    workflow_path = workflow/'complete.json'
    workflow_done = read(workflow_path)
    analysis_path = root/'analysis/summary.json'
    analysis = read(analysis_path)
    require(workflow_done.get('complete') is True and analysis.get('complete') is True,
            'completed workflow and analysis are required')
    checked(workflow_path, hashes=hashes)
    require(str(analysis_path.resolve()) in workflow_done['artifacts'], 'workflow did not bind this analysis')
    checked(analysis_path, workflow_done['artifacts'][str(analysis_path.resolve())], hashes)
    require(analysis['dependencies']['source_summary'] == file_hash(root/'summary.json'),
            'completed analysis refers to a different supplement')
    for name, digest in {**analysis['input_hashes'],
                         **{str(root/'analysis'/n):h for n,h in analysis['artifacts'].items()}}.items():
        if run:
            run.check()
        checked(name, digest, hashes)
    return analysis


def analyze(root, workflow, destination, config, run=None):
    root, workflow, destination = map(Path, (root, workflow, destination))
    require(code_hashes() == config['code'], 'native report code changed while queued')
    dependencies = dict(config=config, source_summary=file_hash(root/'summary.json'),
        source_analysis=file_hash(root/'analysis/summary.json'), source_workflow=file_hash(workflow/'complete.json'))
    saved = destination/'summary.json'
    if saved.exists():
        result = read(saved)
        require(result.get('complete') is True and result['dependencies'] == dependencies,
                'completed native report dependencies changed')
        for path, digest in result['input_hashes'].items():
            if run:
                run.check()
            checked(path, digest)
        for name, digest in result['artifacts'].items():
            checked(destination/name, digest)
    else:
        hashes = {}
        prior = completed_analysis(root, workflow, hashes, run)
        records, evidence = load_records(root, run)
        hashes.update(evidence)
        raw = indexed(read(root/'summary.json')['records'])
        native, provenance = [], []
        for record in records:
            if run:
                run.check()
            if record['method'] in UF_METHODS:
                normalized, source = verify_native_point(raw[record_key(record)], record, root, hashes)
                native.append(normalized); provenance.append(source)
            else:
                native.append(record)
        curves, supports = build_curves(records, native, prior['matched_support_by_G_cap'])
        support_union = sorted({tuple(v) for support in supports.values() for v in support})
        auxiliary = {}
        for cap, support in supports.items():
            chosen = [r for r in native if r['method'] in UF_METHODS or r.get('max_g') == int(cap)]
            auxiliary[cap] = auxiliary_interpolation(chosen, support)
            auxiliary[cap]['scope'] = ('Auxiliary per-video linear interpolation ONLY inside adjacent measured '
                'native-UF rate intervals; RouterVC uses complete-stream rates. No extrapolation or BD-rate.')
        if run:
            run.update(phase='drawing_native_UF_comparison', cpu_only=True)
        artifacts = draw_curves(destination, curves)
        result = dict(complete=True, dependencies=dependencies, input_hashes=hashes,
            matched_support_by_G_cap=supports, matched_sample_count=len(support_union), curve_sets=curves,
            excluded_samples=[list(v) for v in sorted({sample_key(r) for r in records}-set(support_union))],
            native_uf_file_evidence=provenance,
            auxiliary_native_uf_interpolation_by_G_cap=auxiliary,
            artifacts={name:file_hash(destination/name) for name in artifacts},
            byte_scopes=dict(
                standard_uf='Actual stream.bin stat bytes including native SPS/NAL syntax, lengths, QPs and coded payloads; excludes the experimental JSON validation sidecar.',
                diagnostic_uf='Same native stream and SAME decoded quality plus actual transmitted_meta.json bytes; historical experimental framing diagnostic, NOT the standard UF baseline.',
                routervc='All actual RouterVC stream bytes, including native base, RTVC/ACSE headers, strategy/model control and E packets; nothing subtracted.',
                averaging='Compute each video BPP from actual byte count and its saved T*H*W geometry, then equally average videos. Never use a mean payload/total ratio.',
                decoder_evidence='This CPU report reuses saved fresh decoding performed with the JSON sidecar for validation and frame-count configuration. It does not claim a new sidecar-free decoder run.',
                shared_setup='Pre-shared codec/model and sequence conventions are not counted as per-video model payload. Native UF parsing carries dimensions/QP/packet lengths; arbitrary experimental hash checks are not standard codec syntax.',
                effective_frame_count='As in the upstream evaluation convention, effective frame count is supplied by sequence configuration, not charged as the experimental JSON sidecar. These saved clips contain complete 17-frame windows (one I frame plus two eight-frame chunks). An arbitrary partly filled final chunk needs an agreed valid-frame-count convention; this report does not implement that interface.'),
            recommendation='Use standard native-UF rates for subsequent scientific comparisons; retain charged-sidecar results only as an explicitly labeled diagnostic.',
            interpretation='Matched development-video evidence, not independent testing or universal RD dominance.',
            no_inference=True, no_metric_recalculation=True, no_bd_rate=True,
            preserves_historical_results=True, no_GPU_mutex=True)
        require(code_hashes() == config['code'], 'native report code changed while running')
        require(dependencies['source_summary'] == file_hash(root/'summary.json') and
                dependencies['source_analysis'] == file_hash(root/'analysis/summary.json') and
                dependencies['source_workflow'] == file_hash(workflow/'complete.json'), 'source summaries changed during report')
        atomic_json(saved, result)
    completion = dict(complete=True, summary_sha256=file_hash(saved), artifacts=result['artifacts'],
                      no_inference=True, no_metric_recalculation=True)
    immutable(destination/'complete.json', completion)
    return result


def main(args):
    if not os.environ.get('TMUX'):
        raise RuntimeError('run the native UF report inside tmux (CPU only)')
    require(math.isfinite(args.max_hours) and args.max_hours > 0, 'max-hours must be finite and positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    root = args.root.resolve()
    workflow = (args.workflow or root.parent/'workflow').resolve()
    destination = (args.output or root/'native_uf_report').resolve()
    require(destination not in (root, root/'analysis', root/'report', workflow, root.parent),
            'native comparison must use a separate output directory')
    config = dict(version=1, root=str(root), workflow=str(workflow), output=str(destination),
                  code=code_hashes(), cpu_only=True, no_GPU_mutex=True)
    from demo.chunk_enhancement_experiment import Run
    run = Run(SimpleNamespace(output=destination, command='native_uf_report', max_hours=args.max_hours))
    run.thread.start()
    try:
        immutable(destination/'request.json', config)
        run.update(phase='waiting_for_completed_workflow', cpu_only=True)
        while not (workflow/'complete.json').exists():
            require(args.wait_workflow, 'workflow is not complete; use --wait-workflow to queue')
            run.check()
            require(code_hashes() == config['code'], 'native report code changed while queued')
            time.sleep(10)
        run.check()
        result = analyze(root, workflow, destination, config, run)
        run.update(phase='complete', cpu_only=True, matched_samples=result['matched_sample_count'],
                   figures=len(result['artifacts']))
        print(json.dumps(dict(complete=True, output=str(destination), figures=len(result['artifacts']))), flush=True)
    except BaseException as error:
        atomic_json(destination/'last_failure.json', dict(error=repr(error), phase=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--workflow', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wait-workflow', action='store_true')
    parser.add_argument('--max-hours', type=float, default=24.)
    main(parser.parse_args())
