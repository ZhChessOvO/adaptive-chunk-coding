"""Fresh ACSG2 receiver for paired input/internal mean-BF16 conditions."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.scalable_cooperation_decode import profile as previous_profile
from demo.scalable_generation_decode import ASSETS, PATCH
from demo.scalable_codec import BaseCodec, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.chunk_enhancement_codec import configure_torch, load_model
from demo.feature_condition_cache import receive_features
from demo.internal_condition_model import ConditionBranch, injection, validate_bundle, FORMAT
from demo.condition_path_decode import tensor_identity, rng_identity
from demo.feature_condition_decode import identities as old_identities, decode as old_decode
from demo import feature_condition_decode as parent
from unittest.mock import patch


def identities(adapter):
    kind, mode = validate_bundle(torch.load(adapter,weights_only=True,map_location='cpu'))
    result = old_identities(adapter)
    spec = dict(parent=result['profile'], receiver=file_hash(Path(__file__)), format=FORMAT,
        kind=kind,mode=mode,model=file_hash(REPO/'demo/internal_condition_model.py'),
        packet_view=file_hash(REPO/'demo/feature_interface_model.py'),
        upstream={p:file_hash(REPO/'third_party/SeedVR2'/p) for p in (
            'models/dit_v2/nadit.py','models/dit_v2/patch/patch_v1.py',
            'projects/video_diffusion_sr/infer.py')})
    result['profile'] = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return result


def restore(enhanced, control, adapter, packets):
    from demo.stage_c_a800_teacher import PersistentSeedVR2
    args = SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'],
        positive_embedding=ASSETS['positive'], negative_embedding=ASSETS['negative'],
        lora_checkpoint=adapter if control['strength'] else None,
        lora_strength=control['strength'], sample_steps=1, cfg_scale=1.,dit_dtype='bfloat16')
    if control['processing_scale'] != 1 or control['strength'] != 1:
        raise ValueError('feature profile requires native scale and bundle strength one')
    payload = torch.load(adapter, weights_only=True, map_location='cpu')
    kind,mode = validate_bundle(payload)
    feature = ConditionBranch(kind,mode).to('cuda').eval().requires_grad_(False)
    feature.load_state_dict(payload['branch_state'])
    model = PersistentSeedVR2(args)
    runner = model.runner
    runner.config.vae.use_sample = False
    original_encode, original_condition = runner.vae_encode, runner.get_condition
    original_inference = runner.inference
    current, observations = {}, []
    @torch.no_grad()
    def encode(samples):
        before = rng_identity()
        result = original_encode(samples)
        assert all(z.dtype == torch.bfloat16 for z in result)
        observations.append(dict(before_vae=before,after_vae=rng_identity(),
                                 conditions=[tensor_identity(z) for z in result]))
        return result
    @torch.no_grad()
    def get_condition(latent, latent_blur, task):
        if latent.dtype != torch.bfloat16 or latent_blur.dtype != torch.bfloat16:
            raise ValueError('noise/condition dtype mismatch')
        effective,side,coverage = feature(latent_blur,packets,
            start=current['start'],crop=current['crop'])
        if kind == 'internal' and not torch.equal(effective,latent_blur):
            raise RuntimeError('internal branch changed the RGB condition')
        current.update(side=side,mask=coverage,side_rms=float(side.float().square().mean().sqrt()),
                       coverage=float(coverage.mean()))
        observations[-1].update(diffusion_noise=tensor_identity(latent),
            effective_condition=tensor_identity(effective),before_diffusion=rng_identity(),
            rgb_condition_unchanged=torch.equal(effective,latent_blur),internal_statistics=[])
        return original_condition(latent=latent,latent_blur=effective,task=task)
    @torch.no_grad()
    def inference(*args,**kwargs):
        with injection(runner.dit,feature,current['side'],current['mask'],
                       observations[-1]['internal_statistics']):
            return original_inference(*args,**kwargs)
    runner.vae_encode,runner.get_condition,runner.inference = encode,get_condition,inference
    restored, records = enhanced.copy(), []
    for i, region in enumerate(control['generate']):
        t,n,x,y,w,h = region
        halo,scale = control['context'],control['processing_scale']
        x0,y0 = max(0,x-halo),max(0,y-halo)
        x1,y1 = min(enhanced.shape[2],x+w+halo),min(enhanced.shape[1],y+h+halo)
        ch,cw = y1-y0,x1-x0
        if any(v % 8 for v in (x0,y0)) or ch % 16 or cw % 16:
            raise ValueError("feature profile does not silently resize or shift the lattice")
        sums = np.zeros((n,h,w,3),np.float32)
        mass = np.zeros((n,1,1,1),np.float32)
        temporal = ((9-np.abs(np.arange(17)-8))/9).astype(np.float32)[:,None,None,None]
        for j,start in enumerate(fmt.windows(n)):
            current = dict(start=t+start, crop=(x0,y0,cw,ch))
            frames,runtime = model.restore(list(enhanced[t+start:t+start+17,y0:y1,x0:x1]),
                seed=control['seed']+65536*i+j,
                processing_height=(ch*scale+15)//16*16,processing_width=(cw*scale+15)//16*16,
                output_height=ch,output_width=cw)
            result = np.stack(frames)[:,y-y0:y-y0+h,x-x0:x-x0+w]
            sums[start:start+17] += result*temporal
            mass[start:start+17] += temporal
            records.append(dict(region=i,start=t+start,crop=[x0,y0,cw,ch],runtime=runtime,
                                feature_side_rms=current.get("side_rms",0.),
                                feature_coverage=current.get("coverage",0.)))
        restored[fmt.region_slice(region)] = np.rint(sums/mass).clip(0,255).astype(np.uint8)
    return restored,dict(model_load_seconds=model.model_load_seconds,windows=records,
        condition_windows=observations,branch_kind=kind,branch_mode=mode,
        condition_policy='mean BF16, same posterior draws; independent BF16 noise',
        seconds_model_load_excluded=sum(v['runtime']['seconds_model_load_excluded'] for v in records))


def decode(args):
    with patch.object(parent,'identities',identities), patch.object(parent,'restore',restore):
        old_decode(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--disable-generation',action='store_true')
    decode(p.parse_args())

