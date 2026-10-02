"""Encode updated E packets; receiver runs later in a separate source-free process."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_codec import configure_torch, load_model, encode_enhancement
from demo.online_eg_data import training_entries, online_rgb
from demo.online_eg_decode import identities
from demo.internal_condition_pipeline import OLD
from demo.chunk_enhancement_experiment import read
from demo.scalable_codec import atomic_bytes,atomic_json,atomic_npz,file_hash
from demo.scalable_format import parse,frame_hash
from demo import scalable_cooperation_format as fmt


def main(args):
    configure_torch()
    entries = training_entries()
    model = load_model(args.enhancement)
    reports = []
    for entry in [next(e for e in entries if e['dataset']==d) for d in ('REDS','UVG')]:
        with np.load(entry['pair_path']) as p:
            source,base = p['source'].copy(),p['base'].copy()
        chunks = torch.load(entry['feature_path'],weights_only=True,map_location='cpu')['chunks']
        rois = [[0,0,256,256]]
        prefix,wires,enhanced,details = encode_enhancement(model,args.enhancement,
            Path(entry['base_path']).read_bytes(),source,base,chunks,rois,1.,compact=True)
        control,_,_,_ = fmt.parse((OLD/read(OLD/'summary.json')['results'][0]['sample']['sample_id']/'cooperate_l05.acsg').read_bytes())
        control.update(identities(args.adapter))
        control.update(generate=[[0,17,64,64,128,128]],protect=[],context=64,strength=1.,processing_scale=1,blend=1.)
        dest = args.output/entry['dataset']; dest.mkdir(parents=True,exist_ok=True)
        for mode,count in [('none',0),('partial',2),('full',3)]:
            inner = prefix+b''.join(wires[:count])
            actual = fmt.wrap(inner,control)
            path = dest/f'{mode}.acsg'; atomic_bytes(path,actual)
            expected = base.copy()
            for packet in parse(inner).packets:
                m = packet.meta; t,n=m['start'],m['count'];x,y,w,h=m['roi']
                expected[t:t+n,y:y+h,x:x+w]=enhanced[t:t+n,y:y+h,x:x+w]
            with torch.no_grad():
                pixels,terms = online_rgb(model,source,base,chunks,[p.meta for p in parse(inner).packets],
                    (0,0,256,256),differentiable=False)
            online = pixels.round().byte().permute(0,2,3,1).cpu().numpy()
            np.testing.assert_array_equal(online,expected[:,:256,:256])
            atomic_npz(dest/f'{mode}_expected.npz',reconstruction=expected)
            reports.append(dict(dataset=entry['dataset'],mode=mode,stream=str(path),bytes=len(actual),
                stream_sha256=file_hash(path),enhanced_hash=frame_hash(expected),
                base_hash=frame_hash(base),online_matches_real_coding=True))
    atomic_json(args.output/'encode.json',dict(complete=True,results=reports,
        enhancement=file_hash(args.enhancement),adapter=file_hash(args.adapter)))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--enhancement',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    main(p.parse_args())
