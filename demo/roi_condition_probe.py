"""Compare RGB-first condition tensors with the untouched fresh receiver."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_experiment import read
from demo.internal_condition_pipeline import OLD
from demo.condition_path_decode import tensor_identity
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.roi_condition_data import context_crop, encode_roi
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_generation_decode import ASSETS
from demo.stage_c_seedvr2_bridge import configure_runner
from demo import scalable_cooperation_format as fmt


def main(args):
    configure_torch()
    runner,device = configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'],vae_checkpoint=ASSETS['vae'],
        positive_embedding=ASSETS['positive'],negative_embedding=ASSETS['negative'],
        lora_checkpoint=None,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
    rows = read(OLD/'summary.json')['results']
    results = {}
    for index in (0,2):
        row = rows[index]; sid=row['sample']['sample_id']
        folder = OLD/sid
        verify_artifacts(folder/'enhance_q1',row['points']['enhance_q1']['artifacts'])
        pixels = load_frames(folder/'enhance_q1/reconstruction.npz')
        wire = folder/'cooperate_l05.acsg'
        assert file_hash(wire) == row['points']['cooperate_l05']['stream_sha256']
        control,*_ = fmt.parse(wire.read_bytes())
        full = encode_roi(runner,pixels[:17],(0,0,pixels.shape[2],pixels.shape[1]),device)
        observations = []
        for region in control['generate']:
            crop = context_crop(pixels.shape,region,control['context'])
            for start in fmt.windows(region[1]):
                t = region[0]+start
                latent = encode_roi(runner,pixels[t:t+17],crop,device)
                obs = dict(start=t,crop=list(crop),identity=tensor_identity(latent))
                if t == 0:
                    x,y,w,h = [v//8 for v in crop]
                    old = full[:,y:y+h,x:x+w]
                    obs['full_then_crop_rms'] = float((old.float()-latent.float()).square().mean().sqrt())
                observations.append(obs)
        results[sid] = observations
    atomic_json(args.output,dict(complete=True,source_frames_read=False,results=results,
        code={f:file_hash(REPO/'demo'/f) for f in ('roi_condition_data.py','roi_condition_probe.py')}))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
