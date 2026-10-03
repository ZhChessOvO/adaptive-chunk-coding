"""CPU companion: native UF, Router G4/G8, and one full-frame G ROI.

Only authenticates and displays completed fresh reconstructions. UF captions
use its real native stream bytes; all other methods retain their complete wire
bytes. Historical records, metrics, images, and preview modules stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo import routervc_preview as preview
from demo.scalable_codec import atomic_json, file_hash

DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_20261003/supplement')
FPS, REPEATS, FRAMES = 12., 1, 17
CODE = ('routervc_fullframe_preview.py', 'routervc_preview.py', 'scalable_codec.py',
        'scalable_format.py', 'chunk_enhancement_experiment.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_hashes():
    return {name: file_hash(REPO/'demo'/name) for name in CODE}


def select_samples(records):
    selected = []
    for domain in ('REDS', 'UVG'):
        rows = [r for r in records if r['dataset'] == domain]
        require(bool(rows), f'missing completed {domain} sample')
        sid = rows[0]['sample_id']
        same = [r for r in rows if r['sample_id'] == sid]
        panels = [('Native DCVC-UF: QP32', preview.choose(same, 'uf_qp32')),
                  ('RouterVC: E budget 50%, G <= 4', preview.choose(same, 'context_smooth', .5, 4)),
                  ('RouterVC: E budget 50%, G <= 8', preview.choose(same, 'context_smooth', .5, 8)),
                  ('Full-frame G: one ROI', preview.choose(same, 'full_frame_g'))]
        require(panels[-1][1]['actual_G_roi_calls'] == 1,
                'full_frame_g must be the measured one-ROI baseline, not 16 grid ROIs')
        selected.append(dict(dataset=domain, sample_id=sid, panels=panels))
    return selected


def validate_records(root, sample):
    """First authenticate original records; change only later display captions."""
    source, hashes = preview.validate_selection(root, sample)
    for _, record in sample['panels']:
        folder = preview.resolve(root, record['folder'])
        decoded = folder/'fresh/decode.json'
        require('fresh/decode.json' in record['artifacts'], 'fresh decode report must be authenticated')
        require(preview.read(decoded) == record['decode'], 'summary differs from saved fresh decode')
        d = record['decode']
        require(d['source_frames_read'] is False, 'receiver must not read source frames')
        name = ('stream.bin' if record['method'] == 'uf_qp32' else
                'stream.acsg' if record['method'] == 'full_frame_g' else 'stream.rtvc')
        require(name in record['artifacts'], 'display byte count must use an authenticated real stream')
        stream = folder/name
        require(file_hash(stream) == d['stream_sha256'], 'fresh decode stream hash differs')
        nbytes = stream.stat().st_size
        if record['method'] == 'uf_qp32':
            sidecar = folder/'transmitted_meta.json'
            require('transmitted_meta.json' in record['artifacts'], 'UF diagnostic sidecar must be authenticated')
            require(nbytes == record['native_bytes'] == d['native_bytes'], 'native UF size differs from saved record')
            require(sidecar.stat().st_size == d['metadata_bytes'], 'UF diagnostic sidecar size differs')
            require(nbytes+sidecar.stat().st_size == record['bytes'] == d['total_bytes'],
                    'original UF native/sidecar accounting differs')
        else:
            require(nbytes == record['bytes'] == d['total_bytes'], 'full transmitted wire size differs')
        if record['method'] == 'full_frame_g':
            require(d['actual_G_roi_calls'] == 1, 'fresh decode did not use one full-frame ROI')
    return source, hashes


def load_display_panels(root, sample, source):
    panels = preview.load_panels(root, sample, source)
    require(panels[0]['pixels'].shape[0] == FRAMES, 'fixed comparison requires exactly 17 original frames')
    for panel, (_, original) in zip(panels[1:], sample['panels']):
        # No record is mutated; original total/native accounting is retained.
        wire_bytes = panel.pop('stream_bytes')
        native = original['method'] == 'uf_qp32'
        displayed = original['native_bytes'] if native else wire_bytes
        scope = 'native UF' if native else 'all-wire'
        panel.update(displayed_bytes=displayed, original_record_total_bytes=wire_bytes,
                     display_byte_scope=scope,
                     subtitle=f'{displayed:,} {scope} B | LPIPS {panel["saved_lpips"]:.4f}')
    return panels


def create(root, output, run=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    done, summary = preview.read(root/'complete.json'), preview.read(root/'summary.json')
    require(done.get('complete') is True and summary.get('complete') is True,
            'requires completed supplementary evaluation')
    require(done['summary'] == file_hash(root/'summary.json'), 'completed summary identity differs')
    samples = select_samples(summary['records'])
    inputs, sources = {}, []
    for sample in samples:
        source, hashes = validate_records(root, sample)
        sources.append(source)
        inputs.update(hashes)
    codec = preview.ffmpeg_profile()
    binding = dict(summary_sha256=file_hash(root/'summary.json'),
        complete_sha256=file_hash(root/'complete.json'), code=code_hashes(), input_hashes=inputs,
        sample_ids=[s['sample_id'] for s in samples], frames=FRAMES, fps=FPS, repeats=REPEATS,
        profile='native_UF_vs_router_G4_G8_vs_one_full_frame_G', encoder=codec)
    manifest = output/'manifest.json'
    if manifest.exists():
        previous = preview.read(manifest)
        require(previous['binding'] == binding, 'preview inputs/code changed; use a new output')
        for name, digest in previous['artifacts'].items():
            require(file_hash(output/name) == digest, f'changed preview artifact: {name}')
        return previous
    began, videos = time.monotonic(), []
    for sample, source in zip(samples, sources):
        if run is not None:
            run.check()
            run.update(phase='CPU_display_encoding', sample=sample['sample_id'], completed=len(videos), total=2)
        panels = load_display_panels(root, sample, source)
        slug = re.sub(r'[^A-Za-z0-9_-]', '_', sample['sample_id'])
        path = output/f'{sample["dataset"]}_{slug}_fullframe_compare{codec["extension"]}'
        video = preview.encode_preview(path, panels, f'{sample["dataset"]} | {sample["sample_id"]}',
                                       FPS, REPEATS, codec, run)
        video.update(dataset=sample['dataset'], sample_id=sample['sample_id'],
                     panels=[{k: v for k, v in p.items() if k != 'pixels'} for p in panels])
        videos.append(video)
    require(code_hashes() == binding['code'], 'preview source changed while rendering')
    result = dict(complete=True, binding=binding, videos=videos,
        artifacts={name: file_hash(output/name) for video in videos for name in (video['path'], video['poster'])},
        elapsed_seconds=time.monotonic()-began, no_model_inference=True, no_metric_recalculation=True,
        selection='First REDS and first UVG sample in completed summary; fixed 17 frames and frame-9 poster.',
        accounting='UF caption uses native stream.bin bytes; RouterVC and full-frame G include their entire wires. Original records unchanged.',
        comparison='Different bitrate and computation budgets; not an equal-rate or equal-latency claim.',
        source_usage='Ground truth is used only for display, never supplied to the receiver.',
        playback='12 fps, one pass; no interpolation/resize. Display MP4 bytes and pixels never enter quality or RD metrics.')
    atomic_json(manifest, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    p.add_argument('--wait-complete', action='store_true')
    p.add_argument('--max-hours', type=float, default=24.)
    args = p.parse_args()
    require(bool(os.environ.get('TMUX')), 'run this CPU preview task inside tmux')
    require(math.isfinite(args.max_hours) and args.max_hours > 0, 'invalid max-hours')
    args.output = args.output or args.root/'preview_fullframe'
    args.command = 'fullframe_preview'
    from demo.chunk_enhancement_experiment import Run
    run = Run(args)
    run.thread.start()
    try:
        pinned = code_hashes()
        request = dict(root=str(args.root.resolve()), code=pinned, frames=FRAMES, fps=FPS, repeats=REPEATS)
        request_path = args.output/'request.json'
        if request_path.exists():
            require(preview.read(request_path) == request, 'changed queued preview source/settings')
        else:
            atomic_json(request_path, request)
        while not (args.root/'complete.json').is_file():
            require(args.wait_complete, 'supplement not complete; pass --wait-complete to wait')
            run.check()
            run.update(phase='waiting_for_supplement', cpu_only=True)
            time.sleep(10)
        require(code_hashes() == pinned, 'source changed while waiting')
        result = create(args.root, args.output, run)
        run.update(phase='complete', completed=2, total=2)
        print(json.dumps(dict(complete=True, videos=[v['path'] for v in result['videos']])), flush=True)
    except BaseException as error:
        atomic_json(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
