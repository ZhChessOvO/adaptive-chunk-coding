"""Fixed lossless stills and explicitly slow preview movies, from saved controls."""
import argparse
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo.chunk_enhancement_experiment import Run
from demo.routervc_fullview_probe import read, save, digest, verify_artifacts
from tools.latent_boundary_report import ROOT, EVALUATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'p1_visuals')
    args = parser.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run = Run(SimpleNamespace(output=args.output, command='visuals', max_hours=2)); run.thread.start()
    try:
        rows = read(EVALUATION/'summary.json')['scope']['rows']
        rows = [rows[0], rows[6]]
        records = []
        for row in rows:
            run.check(); sid = row['sample_id']; folder = ROOT/'p1_controls/samples'/sid
            verify_artifacts(folder, read(folder/'complete.json')['artifacts'])
            verify_artifacts(folder/'controls', read(folder/'controls/complete.json')['artifacts'])
            with np.load(row['source_path']) as z: source = z['source']
            with np.load(folder/'received.npz') as z: received = z['enhanced']
            with np.load(folder/'current.npz') as z: current = z['pixels']
            with np.load(folder/'controls/controls.npz') as z: multi = z['multiband']
            _, h, w, _ = source.shape; display_w = 512; display_h = round(h*display_w/w)//2*2
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
            frames = []
            for t in range(17):
                image = Image.new('RGB', (display_w*4, display_h+56), 'white'); draw = ImageDraw.Draw(image)
                for i, (title, pixels) in enumerate([('Source', source), ('Received Y', received),
                                                    ('Current feather', current), ('Multiband control', multi)]):
                    tile = Image.fromarray(pixels[t]).resize((display_w, display_h), Image.Resampling.LANCZOS)
                    image.paste(tile, (i*display_w, 56)); draw.text((i*display_w+8, 6), title, font=font, fill='black')
                draw.text((8, 31), f'{sid} | frame {t+1}/17 | preview slowed to 6 fps', font=font, fill='black')
                frames.append(np.asarray(image))
                if t == 8: image.save(run.root/f'{sid}.png')
            movie = run.root/f'{sid}.mp4'
            command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{display_w*4}x{display_h+56}', '-r', '6', '-i', '-', '-an',
                       '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
            subprocess.run(command, input=np.stack(frames).tobytes(), check=True)
            records.append(dict(sample_id=sid, frames=17, preview_fps=6,
                                source_fps_not_represented=True, preview_not_used_for_metrics=True,
                                artifacts={f'{sid}.{ext}':digest(run.root/f'{sid}.{ext}') for ext in ('png', 'mp4')}))
        save(run.root/'complete.json', dict(complete=True, samples=records))
        run.update(phase='complete'); run.log_resources()
    finally: run.stop.set(); run.thread.join(); run.lock.close()


if __name__ == '__main__': main()
