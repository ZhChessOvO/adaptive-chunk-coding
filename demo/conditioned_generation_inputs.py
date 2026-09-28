"""Fixed training-input illustrations, explicitly not evaluation results."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.conditioned_generation_pipeline import DEFAULT
from demo.chunk_enhancement_experiment import read
from demo.scalable_format import parse


def render(root):
    rows=read(root/'streams/index.json')['entries']
    dest=root/'input_figures';dest.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for dataset in ('REDS','UVG'):
        row=next(r for r in rows if r['dataset']==dataset)
        folder=root/'received'/row['sample_id']
        if not (folder/'complete.json').exists():
            continue
        with np.load(folder/'conditions.npz') as data:
            arrays={k:data[k].copy() for k in ('none','partial','full')}
        with np.load(row['pair_path']) as data:
            arrays['GT']=data['source'].copy()
        canvas=Image.new('RGB',(4*256,2*292+50),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{dataset}: {row["sample_id"]}; training inputs, NOT evaluation results',font=font,fill='black')
        draw.text((8,25),'Green outlines: received E packets at this frame; no Generate is run.',font=font,fill='black')
        for ir,frame in enumerate((0,8)):
            for ic,key in enumerate(('none','partial','full','GT')):
                x,y=ic*256,50+ir*292
                canvas.paste(Image.fromarray(arrays[key][frame]).resize((256,256)),(x,y+36))
                draw.text((x+4,y+7),f'{key} / frame {frame}',font=font,fill='black')
                if key in ('partial','full'):
                    for packet in parse((root/'streams'/row['sample_id']/f'{key}.acse').read_bytes()).packets:
                        m=packet.meta
                        if m['start'] <= frame < m['start']+m['count']:
                            rx,ry,w,h=m['roi']
                            draw.rectangle((x+rx//2,y+36+ry//2,x+(rx+w)//2-1,y+36+(ry+h)//2-1),outline='#00cc60',width=2)
        canvas.save(dest/f'{dataset.lower()}_prefix_inputs.png')
        print(dest/f'{dataset.lower()}_prefix_inputs.png')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT)
    render(p.parse_args().output)
