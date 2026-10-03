"""CPU-only digest of saved full-view probe points; never decode or score video.

The context/local curves belong to the existing statistical Router, not the new
visual Router. A single development window cannot establish dataset performance.
Run the CLI in tmux; report generation is atomic and completed reentry is read-only.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


DEFAULT_ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003/fullview_probe_recovered')
SCHEMA = 'routervc-fullview-probe-report-v1'
POINTS = ('context_e0.25_g8', 'context_e0.5_g8', 'local_e0.25_g8',
          'local_e0.5_g8', 'context_e0.5_g_off', 'uf_qp8', 'uf_qp16',
          'uf_qp24', 'uf_qp32', 'wholeframe_g_one_roi')
GROUPS = (
    ('uf', 'Native DCVC-UF', '#242424', 'o', '-'),
    ('context', 'Existing Router: context', '#236bb2', 'o', '-'),
    ('local', 'Existing Router: local', '#e08023', '^', '--'),
    ('off', 'Same-byte context, G off', '#8e4ba7', 'X', 'None'),
    ('full_g', 'UF QP8 + full-frame G (1 ROI)', '#298a56', '*', 'None'),
)
LIMITATIONS = [
    'One REDS val000 development window, 17 frames; not a REDS/UVG dataset conclusion.',
    'Full field of view resized from 1280x720 to 1024x576, not native resolution.',
    'Context/local are the existing statistical Routers, not the new visual Router.',
    'Nine successful old points are reused unchanged; only the final full-frame G point was repaired and completed.',
    'Native UF rate uses stream.bin only; its audit JSON is excluded. RouterVC/G framing and controls remain charged.',
    'No metrics, decoding, training or GPU work is performed by this report.',
    'Recorded receiver timings include loading/validation and differ in execution/cache conditions; not a controlled latency benchmark.',
    'G runtime sums saved per-window model execution; it excludes model loading and is not end-to-end time.',
    'LPIPS/PSNR do not establish semantic fidelity; no best-method selection or model promotion is made.',
]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def export_display(source, destination, max_bytes=5 * 1024**2 - 1):
    """Format-only preview export, never an input to metrics or decoding."""
    from PIL import Image

    with Image.open(source) as original:
        original_size = list(original.size)
        rgb = original.convert('RGB')
        for quality in (95, 90, 85, 80, 70, 60):
            buffer = io.BytesIO()
            rgb.save(buffer, format='JPEG', quality=quality, optimize=True, subsampling=0)
            payload = buffer.getvalue()
            if len(payload) <= max_bytes:
                break
        else:
            raise ValueError('display export exceeds upload limit; original assets unchanged')
    temporary = Path(destination).with_name(Path(destination).name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return dict(path=str(destination), sha256=digest(destination), bytes=len(payload),
                source_path=str(source), source_sha256=digest(source),
                source_size=original_size, display_size=original_size,
                resized=False, jpeg_quality=quality, jpeg_chroma_subsampling=0,
                metrics_input=False, original_assets_unchanged=True,
                caption='Display-only JPEG of the existing fixed frame (index 8). '
                'No resizing or generative editing. Lossy display encoding is not used '
                'for any metric; original PNG and reconstruction NPZ remain unchanged.')


def inside(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'unsafe artifact name: {relative}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'artifact escapes probe: {relative}')
    return path


def collect(root):
    """Verify saved metadata, streams, visual/reconstruction hashes; do not load RGB."""
    root = Path(root).resolve()
    hashes = {}

    def check(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'changed saved artifact: {path}')
        hashes[str(Path(path).resolve())] = actual
        return actual

    for filename in ('summary.json', 'protocol.json', 'import.json', 'complete.json'):
        check(root / filename)
    summary = read(root / 'summary.json')
    protocol = read(root / 'protocol.json')
    receipt = read(root / 'import.json')
    complete = read(root / 'complete.json')
    protocol_hash = hashes[str(root / 'protocol.json')]
    if (not summary.get('complete') or not complete.get('complete')
            or summary.get('points') != 10 or len(summary.get('records', [])) != 10
            or summary.get('protocol_sha256') != protocol_hash
            or complete.get('protocol_sha256') != protocol_hash
            or complete.get('summary_sha256') != hashes[str(root / 'summary.json')]
            or receipt.get('current_protocol') != protocol_hash
            or not summary.get('native_uf_sidecar_excluded_from_rate')
            or not summary.get('no_model_promotion')
            or not protocol.get('whole_view_provenance_verified')
            or summary.get('shape') != [17, 576, 1024, 3]):
        raise ValueError('incomplete or incompatible bounded full-view probe')
    sample = protocol['source_preparation']['record']['binding']['sample']
    if (sample['sample_id'] != 'reds-val-000-f000-n17-fullview'
            or sample['transform']['original_size'] != [1280, 720]
            or sample['transform']['coded_size'] != [1024, 576]
            or sample['transform']['padding'] != [0, 0, 0, 0]
            or not sample['whole_frame']):
        raise ValueError('unexpected source provenance; report scope must be revised explicitly')
    check(root / 'fixed_frame_comparison.png', summary['fixed_frame_sha256'])
    previous = Path(receipt['previous']).resolve()
    check(previous / 'protocol.json', receipt['previous_protocol'])
    check(previous / 'runner_before_baseline_fix.py', receipt['previous_runner'])
    rows, visuals, records = [], [], {}
    for record in summary['records']:
        point = record['point']
        name, kind = point['name'], point['kind']
        if name not in POINTS or name in records:
            raise ValueError('unexpected or duplicate point')
        records[name] = record
        folder = inside(root, name)
        check(folder / 'result.json', summary['artifact_hashes'][name + '/result.json'])
        if (read(folder / 'result.json') != record or not record.get('complete')
                or record['protocol_sha256'] != protocol_hash):
            raise ValueError('summary differs from saved point')
        for filename, expected in record['artifacts'].items():
            check(inside(folder, filename), expected)
        decoded = read(folder / 'fresh/decode.json')
        if record['decode'] != decoded or decoded.get('source_frames_read') is not False:
            raise ValueError('receiver evidence differs or is not source-free')
        reuse = record.get('reused_from')
        if name != 'wholeframe_g_one_roi':
            if not reuse or not reuse.get('fresh_decode_and_metric_times_preserved'):
                raise ValueError('missing old-point recovery evidence')
            old_path = previous / name / 'result.json'
            if Path(reuse['path']).resolve() != old_path:
                raise ValueError('recovery path differs from import receipt')
            check(old_path, reuse['sha256'])
            old = read(old_path)
            for key in ('point', 'bytes', 'bpp', 'quality', 'decode', 'byte_ledger',
                        'decode_seconds', 'elapsed_seconds_this_completion_attempt'):
                if old[key] != record[key]:
                    raise ValueError(f'reused {name} changed {key}')
            for filename, expected in old['artifacts'].items():
                if filename != 'job.json' and record['artifacts'].get(filename) != expected:
                    raise ValueError('reused artifact hash differs')
        elif reuse is not None:
            raise ValueError('final full-G point must be the recovered new completion')
        stream_name = 'stream.bin' if kind == 'uf' else ('stream.acsg' if kind == 'full_g' else 'stream.rtvc')
        stream = folder / stream_name
        actual_bytes = stream.stat().st_size
        ledger = record['byte_ledger']
        if actual_bytes != record['bytes'] or ledger['bytes'] != actual_bytes:
            raise ValueError('rate differs from real stream file size')
        if kind == 'uf':
            sidecar = (folder / 'transmitted_meta.json').stat().st_size
            if (decoded['native_bytes'] != actual_bytes or ledger['native_bytes'] != actual_bytes
                    or ledger['audit_sidecar_bytes'] != sidecar
                    or decoded['metadata_bytes'] != sidecar
                    or decoded['total_bytes'] != actual_bytes + sidecar):
                raise ValueError('UF native/audit-sidecar accounting differs')
        elif decoded['total_bytes'] != actual_bytes:
            raise ValueError('Router/G complete byte accounting differs')
        expected_bpp = actual_bytes * 8 / (17 * 576 * 1024)
        if not math.isclose(record['bpp'], expected_bpp, rel_tol=1e-12):
            raise ValueError('bpp denominator differs from valid full-frame pixels')
        runtime = decoded.get('generation_runtime') or {}
        windows = runtime.get('windows', [])
        g_seconds = sum(w['runtime']['seconds_model_load_excluded'] for w in windows)
        quality = record['quality']
        row = dict(point=name, group=point.get('arm', kind) if kind == 'router' else kind,
                   e_budget_ratio=point.get('ratio'), qp=point.get('qp'),
                   stream_bytes=actual_bytes, bpp=expected_bpp,
                   lpips_alex=quality['lpips_alex'], psnr_db=quality['psnr_db'],
                   temporal_delta_mae=quality['temporal_delta_mae'],
                   recorded_receiver_seconds=decoded['seconds'],
                   recorded_codec_seconds=decoded.get('codec_seconds'),
                   recorded_policy_seconds=decoded.get('policy_seconds'),
                   recorded_g_load_seconds=runtime.get('model_load_seconds'),
                   recorded_g_window_seconds=g_seconds,
                   actual_g_window_calls=len(windows),
                   actual_g_roi_calls=len({w['region'] for w in windows}),
                   recorded_peak_cuda_gib=decoded.get('peak_cuda_allocated_bytes', 0) / 2**30,
                   reused_old_point=bool(reuse), stream_sha256=record['artifacts'][stream_name])
        for key in ('bpp', 'lpips_alex', 'psnr_db', 'recorded_receiver_seconds', 'recorded_g_window_seconds'):
            if not math.isfinite(row[key]):
                raise ValueError('nonfinite saved metric/time')
        rows.append(row)
        visuals.append(dict(point=name, frame_index=protocol['fixed_visual_frame'],
                            path=str(folder / 'fixed_frame.png'),
                            sha256=record['artifacts']['fixed_frame.png']))
    paired = records['context_e0.5_g8'], records['context_e0.5_g_off']
    if paired[0]['artifacts']['stream.rtvc'] != paired[1]['artifacts']['stream.rtvc']:
        raise ValueError('same-byte G-off control is not the same actual stream')
    for arm in ('context', 'local'):
        lo = (root / f'{arm}_e0.25_g8/stream.rtvc').read_bytes()
        hi = (root / f'{arm}_e0.5_g8/stream.rtvc').read_bytes()
        if not hi.startswith(lo):
            raise ValueError('E budget points are not literal packet prefixes')
    full_g = records['wholeframe_g_one_roi']['decode']
    if (full_g.get('actual_G_roi_calls') != 1 or full_g.get('actual_G_window_calls') != 1
            or full_g.get('geometry') != 'one_full_frame_ROI'):
        raise ValueError('full-frame G is not a measured one-ROI baseline')
    visuals.insert(0, dict(point='all_points_and_source', frame_index=protocol['fixed_visual_frame'],
                          path=str(root / 'fixed_frame_comparison.png'),
                          sha256=summary['fixed_frame_sha256']))
    return dict(schema=SCHEMA, input_root=str(root), input_hashes=hashes,
                source=dict(sample_id=sample['sample_id'], shape=summary['shape'],
                            transform=sample['transform'], history=sample['history']),
                recovery=dict(receipt=receipt, reused_points=9, newly_completed_points=1,
                              metrics_and_original_times_preserved=True),
                rows=rows, visuals=visuals, limitations=LIMITATIONS,
                model_promotion=False, semantic_fidelity_measured=False,
                metrics_recomputed=False, gpu_used=False)


def plot_rd(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    handles = []
    for group, label, color, marker, style in GROUPS:
        points = sorted((r for r in rows if r['group'] == group), key=lambda r: r['bpp'])
        for index, (axis, metric) in enumerate(zip(axes, ('lpips_alex', 'psnr_db'))):
            handle, = axis.plot([r['bpp'] for r in points], [r[metric] for r in points],
                                label=label, color=color, marker=marker, linestyle=style,
                                markersize=10 if group == 'full_g' else 7, linewidth=1.8)
            if index == 0:
                handles.append(handle)
            if group == 'uf':
                for row in points:
                    axis.annotate(f"QP{row['qp']}", (row['bpp'], row[metric]),
                                  xytext=(5, 6), textcoords='offset points', fontsize=8)
    for axis, label in zip(axes, ('LPIPS (AlexNet), lower is better', 'PSNR (dB), higher is better')):
        axis.set_xlabel('Actual transmitted bits / pixel')
        axis.set_ylabel(label)
        axis.grid(alpha=.22)
        axis.margins(x=.13, y=.16)
    figure.suptitle('REDS val000: 17 frames, full field of view resized to 1024 x 576\n'
                    'Single-window calibration; not a dataset result', fontsize=13)
    # One row, in explicit order: no column-major reordering across two legends.
    figure.legend(handles=handles, labels=[v[1] for v in GROUPS], loc='lower center',
                  bbox_to_anchor=(.5, .065), ncol=5, frameon=False, fontsize=8,
                  handlelength=1.8, columnspacing=1.1)
    figure.text(.5, .025, 'Native UF excludes audit JSON. Other points include every stream header/control. '
                'Lines join measured points only; no method selected.', ha='center', fontsize=8)
    figure.subplots_adjust(left=.075, right=.985, bottom=.235, top=.80, wspace=.24)
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    figure.savefig(temporary, format='png', dpi=170, facecolor='white')
    plt.close(figure)
    os.replace(temporary, path)


def generate(root=DEFAULT_ROOT, output=None, *, verify_only=False):
    root = Path(root).resolve()
    output = Path(output or root / 'report').resolve()
    if output != root / 'report':
        raise ValueError('report output must be the new probe/report subdirectory')
    report = collect(root)
    binding = dict(schema=SCHEMA, report_code_sha256=digest(__file__),
                   input_hashes=report['input_hashes'])
    binding_path, complete_path = output / 'binding.json', output / 'complete.json'
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError('report inputs/code changed; keep old report and choose a new explicit implementation')
    if complete_path.exists():
        complete = read(complete_path)
        if not complete.get('complete') or complete.get('binding') != binding:
            raise ValueError('completed report binding differs')
        for filename, expected in complete['artifacts'].items():
            if digest(inside(output, filename)) != expected:
                raise ValueError('completed report artifact changed')
        return complete
    if verify_only:
        raise ValueError('report is not complete')
    output.mkdir(parents=True, exist_ok=True)
    if not binding_path.exists():
        save(binding_path, binding)
    plot_rd(report['rows'], output / 'rd.png')
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=list(report['rows'][0]))
    writer.writeheader()
    writer.writerows(report['rows'])
    atomic_text(output / 'timings.csv', table.getvalue())
    for visual in report['visuals']:
        visual['relative_link'] = quote(os.path.relpath(visual['path'], output), safe='/')
    preview = export_display(root / 'fixed_frame_comparison.png', output / 'fixed_frame_display.jpg')
    report['display_export'] = preview
    save(output / 'visuals.json', dict(fixed_frame_index=8, images=report['visuals'],
                                     display_export=preview))
    save(output / 'report.json', report)
    items = ''.join(f'<li><a href="{v["relative_link"]}">{html.escape(v["point"])}</a>'
                    f' (frame index {v["frame_index"]})</li>' for v in report['visuals'])
    caveats = ''.join(f'<li>{html.escape(v)}</li>' for v in LIMITATIONS)
    atomic_text(output / 'index.html', '<!doctype html><meta charset="utf-8">'
                '<title>RouterVC full-view probe</title><style>body{max-width:1250px;'
                'margin:30px auto;font:16px sans-serif}img{width:100%}li{margin:9px 0}</style>'
                '<h1>REDS val000 full-view development probe</h1><img src="rd.png" alt="Saved RD points">'
                '<p><a href="timings.csv">Recorded timings and metrics CSV</a> | '
                '<a href="report.json">Scope, recovery and input hashes</a></p>'
                '<h2>How to read these points</h2><ul>' + caveats + '</ul>'
                '<h2>Existing fixed-frame images</h2>'
                '<img src="fixed_frame_display.jpg" alt="Display-only existing fixed frame montage">'
                '<p>' + html.escape(preview['caption']) + '</p><ul>' + items + '</ul>')
    names = ('binding.json', 'rd.png', 'timings.csv', 'visuals.json', 'report.json',
             'index.html', 'fixed_frame_display.jpg')
    complete = dict(complete=True, schema=SCHEMA, binding=binding,
                    artifacts={name: digest(output / name) for name in names},
                    points=10, reused_points=9, newly_completed_points=1,
                    metrics_recomputed=False, gpu_used=False, model_promotion=False)
    save(complete_path, complete)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        parser.error('run this report in tmux; no GPU/model work is needed')
    done = generate(args.root, args.output, verify_only=args.verify_only)
    print(json.dumps({k: done[k] for k in ('complete', 'points', 'reused_points',
                     'newly_completed_points', 'metrics_recomputed', 'gpu_used')}))


if __name__ == '__main__':
    main()
