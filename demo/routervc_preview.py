"""CPU-only motion previews from already audited RouterVC reconstructions.

Videos are display transcodes, NOT experimental compressed streams or metric
inputs. Fixed first REDS/UVG samples, original frame order, no frame synthesis,
no model inference and no metric recalculation. Preview playback FPS is not a
claim about source acquisition FPS or decoder throughput.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash

DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_20261003/supplement')
LABEL_HEIGHT = 56
PAGE_TOP = 42
PAGE_BOTTOM = 44


def read(path):
    return json.loads(Path(path).read_text())


def choose(records, method, ratio=None, max_g=None):
    selected = [r for r in records if r['method'] == method
                and (ratio is None or r.get('ratio') == ratio)
                and (max_g is None or r.get('max_g') == max_g)]
    if len(selected) != 1:
        raise ValueError(f'expected one completed {method} point, found {len(selected)}')
    return selected[0]


def select_samples(records, profile):
    """First appearance per domain, never metric/content-based selection."""
    selected = []
    for domain in ('REDS', 'UVG'):
        rows = [r for r in records if r['dataset'] == domain]
        if not rows:
            raise ValueError(f'completed {domain} preview sample is missing')
        sid = rows[0]['sample_id']
        same = [r for r in rows if r['sample_id'] == sid]
        router = choose(same, 'context_smooth', .5, 4)
        if profile == 'formal':
            panels = [('Base: UF QP8', choose(same, 'base')),
                      ('Native UF QP32', choose(same, 'uf_qp32')),
                      ('RouterVC: E 50%, G <= 4', router),
                      ('All G: 16 separate ROIs', choose(same, 'full_g'))]
        elif profile == 'smoke':
            panels = [('Base: UF QP8', choose(same, 'base')),
                      ('E-only: separate allocation', choose(same, 'e_only', .5, 0)),
                      ('G-only: G <= 4', choose(same, 'g_only', 0., 4)),
                      ('RouterVC: E 50%, G <= 4', router)]
        else:
            raise ValueError('profile must be formal or smoke')
        selected.append(dict(dataset=domain, sample_id=sid, panels=panels))
    return selected


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root/path


def validate_selection(root, sample):
    """Authenticate original source cache and all selected saved artifacts."""
    input_hashes = {}
    source_info = None
    for _, record in sample['panels']:
        folder = resolve(root, record['folder'])
        if not record.get('artifacts'):
            raise ValueError('preview requires audited artifact manifests')
        for name, digest in record['artifacts'].items():
            path = folder/name
            if file_hash(path) != digest:
                raise RuntimeError(f'changed preview input artifact: {path}')
            input_hashes[str(path)] = digest
        reconstruction = folder/'fresh/reconstruction.npz'
        if 'fresh/reconstruction.npz' not in record['artifacts']:
            raise ValueError('reconstruction is not authenticated by result manifest')
        source = resolve(root, record['source_path'])
        expected = record['source_hash']
        if file_hash(source) != expected:
            raise RuntimeError(f'changed original evaluation cache: {source}')
        if source_info is not None and source_info != (source, expected):
            raise ValueError('comparison methods do not share the identical source input')
        source_info = source, expected
        input_hashes[str(source)] = expected
    return source_info[0], input_hashes


def load_panels(root, sample, source_path):
    with np.load(source_path, allow_pickle=False) as cache:
        source = cache['source'].copy()
    if source.dtype != np.uint8 or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError('expected saved uint8 RGB source')
    panels = [dict(title='Source: evaluation reference', subtitle='Not sent to the receiver',
                   pixels=source, method='source', pixel_sha256=frame_hash(source))]
    for title, record in sample['panels']:
        folder = resolve(root, record['folder'])
        with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as cache:
            pixels = cache['reconstruction'].copy()
        if pixels.shape != source.shape or pixels.dtype != np.uint8:
            raise ValueError('side-by-side panels do not have identical uint8 frame geometry')
        digest = frame_hash(pixels)
        if digest != record['decode']['output_hash']:
            raise RuntimeError('saved pixels differ from fresh receiver output hash')
        lpips = float(record['quality']['lpips_alex'])
        if not math.isfinite(lpips):
            raise ValueError('invalid recorded LPIPS')
        panels.append(dict(title=title, subtitle=f'{record["bytes"]:,} actual stream B | LPIPS {lpips:.4f}',
            pixels=pixels, method=record['method'], pixel_sha256=digest,
            stream_bytes=record['bytes'], saved_lpips=lpips))
    return panels


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_frame(panels, frame, sample_label, fps, repeats):
    """Native RGB tiles, fixed 2x3 layout; no spatial resize or interpolation."""
    count, height, width, _ = panels[0]['pixels'].shape
    if len(panels) != 5 or not 0 <= frame < count:
        raise ValueError('preview expects five panels and a valid frame index')
    out_w = width*3
    out_h = PAGE_TOP+2*(height+LABEL_HEIGHT)+PAGE_BOTTOM
    canvas = Image.new('RGB', (out_w, out_h), '#17202b')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'{sample_label} | Frame {frame+1}/{count} | Fixed development sample',
              fill='white', font=font(17))
    for i, panel in enumerate(panels):
        x, y = (i % 3)*width, PAGE_TOP+(i//3)*(height+LABEL_HEIGHT)
        draw.text((x+8, y+5), panel['title'], fill='white', font=font(16))
        draw.text((x+8, y+30), panel['subtitle'], fill='#b8c5d4', font=font(12))
        canvas.paste(Image.fromarray(panel['pixels'][frame]), (x, y+LABEL_HEIGHT))
    x, y = width*2+12, PAGE_TOP+height+LABEL_HEIGHT+25
    notes = ['Motion preview only', '', f'Playback: {fps:g} fps, repeated {repeats} times',
             'Original frame order; no generated in-betweens',
             'Not original capture speed / decoder throughput', '',
             'Native-size panels; no spatial resizing',
             'MP4 display transcode may alter pixels',
             'Saved NPZ is the metric / pixel reference', '',
             'Different byte budgets: not an equal-rate claim']
    for line in notes:
        draw.text((x, y), line, fill='#cbd5e1', font=font(13))
        y += 23
    draw.text((10, out_h-PAGE_BOTTOM+12),
              'Display file only: MP4 bytes are NEVER counted as RouterVC bitrate or used for quality metrics.',
              fill='#b8c5d4', font=font(14))
    return np.asarray(canvas)


def ffmpeg_profile():
    available = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], text=True,
                                        stderr=subprocess.STDOUT)
    version = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    names = {line.split()[1] for line in available.splitlines() if len(line.split()) > 1}
    if 'libx264' in names:
        return dict(codec='libx264', extension='.mp4', container='mp4', browser_playable=True,
            args=['-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
                  '-movflags', '+faststart'], ffmpeg_version=version,
            display_loss='H264/YUV420 display conversion is lossy; never used for metrics')
    if 'ffv1' in names:
        return dict(codec='ffv1', extension='.mkv', container='matroska', browser_playable=False,
            args=['-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0'], ffmpeg_version=version,
            display_loss='lossless RGB fallback; MKV may need download rather than Notion inline playback')
    raise RuntimeError('No existing software libx264/ffv1 encoder; do not silently use GPU or download')


def probe(path):
    result = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,codec_name:format=duration',
        '-of', 'json', str(path)], text=True)
    return json.loads(result)


def encode_preview(path, panels, label, fps, repeats, profile, run=None):
    first = render_frame(panels, 0, label, fps, repeats)
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise ValueError('display encoder requires even layout dimensions; no hidden resize allowed')
    count = len(panels[0]['pixels'])
    temporary = path.with_name(path.stem+'.partial'+path.suffix)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(fps),
               '-i', 'pipe:0', '-an', '-threads', '4', *profile['args'], '-f', profile['container'], str(temporary)]
    log = path.with_suffix(path.suffix+'.ffmpeg.log')
    with log.open('w') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for _ in range(repeats):
                for index in range(count):
                    if run is not None:
                        run.check()
                    pixels = first if index == 0 else render_frame(panels, index, label, fps, repeats)
                    process.stdin.write(pixels.tobytes())
            process.stdin.close()
            deadline = time.monotonic()+120
            while process.poll() is None:
                if run is not None:
                    run.check()
                if time.monotonic() > deadline:
                    raise RuntimeError('preview encoder did not finish within its bounded flush time')
                time.sleep(.2)
            if process.returncode:
                raise RuntimeError(f'preview software encoder failed; see {log}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
    result = probe(temporary)
    stream = result['streams'][0]
    if (int(stream['nb_read_frames']) != count*repeats or stream['width'] != width or stream['height'] != height):
        raise RuntimeError('encoded display video frame count/geometry mismatch')
    os.replace(temporary, path)
    poster = path.with_suffix('.png')
    buffer = io.BytesIO()
    Image.fromarray(render_frame(panels, min(8, count-1), label, fps, repeats)).save(buffer, format='PNG')
    atomic_bytes(poster, buffer.getvalue())
    return dict(path=path.name, poster=poster.name, display_bytes=path.stat().st_size,
                frame_count=count*repeats, unique_original_frames=count, probe=result,
                command=command, browser_playable=profile['browser_playable'])


def create(root, output=None, *, profile='formal', fps=12., repeats=3, run=None):
    root = Path(root)
    output = root/'preview' if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(fps) or fps <= 0 or type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('invalid preview playback FPS/repetitions')
    summary = read(root/'summary.json')
    if not summary.get('complete'):
        raise ValueError('preview requires a completed saved result summary')
    samples = select_samples(summary['records'], profile)
    inputs = {}
    source_paths = []
    for sample in samples:
        source, hashes = validate_selection(root, sample)
        source_paths.append(source)
        inputs.update(hashes)
    codec = ffmpeg_profile()
    binding = dict(summary_sha256=file_hash(root/'summary.json'), code_sha256=file_hash(Path(__file__)),
                   profile=profile, fps=fps, repeats=repeats, encoder=codec, input_hashes=inputs,
                   sample_ids=[s['sample_id'] for s in samples])
    manifest = output/'manifest.json'
    if manifest.exists():
        previous = read(manifest)
        if previous['binding'] != binding:
            raise RuntimeError('preview inputs/settings changed; preserve this preview and choose a new output')
        for name, digest in previous['artifacts'].items():
            if file_hash(output/name) != digest:
                raise RuntimeError(f'preview artifact changed: {name}')
        return previous
    videos = []
    began = time.monotonic()
    for sample, source_path in zip(samples, source_paths):
        if run is not None:
            run.update(phase='CPU_display_encoding', sample=sample['sample_id'], completed=len(videos))
            run.check()
        pixels = load_panels(root, sample, source_path)
        slug = re.sub(r'[^A-Za-z0-9_-]', '_', sample['sample_id'])
        filename = f'{sample["dataset"]}_{slug}{codec["extension"]}'
        video = encode_preview(output/filename, pixels, f'{sample["dataset"]} | {sample["sample_id"]}',
                               fps, repeats, codec, run)
        video.update(dataset=sample['dataset'], sample_id=sample['sample_id'],
                     panels=[{k: v for k, v in p.items() if k != 'pixels'} for p in pixels])
        videos.append(video)
    artifacts = {name: file_hash(output/name) for v in videos for name in (v['path'], v['poster'])}
    result = dict(complete=True, binding=binding, videos=videos, artifacts=artifacts,
        elapsed_seconds=time.monotonic()-began, no_model_inference=True, no_metric_recalculation=True,
        source_usage='evaluation display only; receiver outputs are authenticated saved fresh decodes',
        selection='first REDS and first UVG occurrence in completed summary, not content or quality selection',
        comparison='fixed mid-budget points, not equal-rate matched curves',
        playback='original frame order repeated; presentation FPS is not capture FPS or decoder speed',
        accounting='display MP4/MKV file bytes never enter codec RD or LPIPS metrics')
    atomic_json(manifest, result)
    return result


def main(args):
    if not os.environ.get('TMUX'):
        raise RuntimeError('run CPU preview rendering in tmux')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from demo.chunk_enhancement_experiment import Run
    if args.output is None:
        args.output = args.root/'preview'
    args.command = 'preview'
    run = Run(args)
    run.thread.start()
    try:
        result = create(args.root, args.output, profile=args.profile, fps=args.fps, repeats=args.repeats, run=run)
        run.update(phase='complete', completed=len(result['videos']), total=2)
        print(json.dumps(dict(complete=True, videos=[v['path'] for v in result['videos']])), flush=True)
    except BaseException as error:
        atomic_json(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    p.add_argument('--profile', choices=('formal', 'smoke'), default='formal')
    p.add_argument('--fps', type=float, default=12.)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-hours', type=float, default=2.)
    main(p.parse_args())
