"""ACSG2 receiver with an explicitly hashed adapted-generator profile.

The adapter path locates pre-shared weights, not hidden coding parameters.
All generation settings and weight digests are transmitted in the container.
Old receivers and their pinned profiles remain byte-for-byte unchanged.
"""
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
from demo.chunk_enhancement_codec import configure_torch, load_model, decode_enhancement


def identities(adapter):
    spec = dict(parent=previous_profile(), receiver=file_hash(Path(__file__)),
                version='ACSG2-enhanced-RGB-adapted-LoRA-v1', input='received_enhanced_rgb')
    assets = dict(ASSETS, lora=Path(adapter))
    return dict({k:file_hash(v) for k,v in assets.items()},
                profile=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())


def restore(enhanced, control, adapter):
    from demo.stage_c_a800_teacher import PersistentSeedVR2
    args = SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'],
        positive_embedding=ASSETS['positive'], negative_embedding=ASSETS['negative'],
        lora_checkpoint=adapter if control['strength'] else None,
        lora_strength=control['strength'], sample_steps=1, cfg_scale=1.,dit_dtype='bfloat16')
    model = PersistentSeedVR2(args)
    restored, records = enhanced.copy(), []
    for i, region in enumerate(control['generate']):
        t,n,x,y,w,h = region
        halo,scale = control['context'],control['processing_scale']
        x0,y0 = max(0,x-halo),max(0,y-halo)
        x1,y1 = min(enhanced.shape[2],x+w+halo),min(enhanced.shape[1],y+h+halo)
        ch,cw = y1-y0,x1-x0
        sums = np.zeros((n,h,w,3),np.float32)
        mass = np.zeros((n,1,1,1),np.float32)
        temporal = ((9-np.abs(np.arange(17)-8))/9).astype(np.float32)[:,None,None,None]
        for j,start in enumerate(fmt.windows(n)):
            frames,runtime = model.restore(list(enhanced[t+start:t+start+17,y0:y1,x0:x1]),
                seed=control['seed']+65536*i+j,
                processing_height=(ch*scale+15)//16*16,processing_width=(cw*scale+15)//16*16,
                output_height=ch,output_width=cw)
            result = np.stack(frames)[:,y-y0:y-y0+h,x-x0:x-x0+w]
            sums[start:start+17] += result*temporal
            mass[start:start+17] += temporal
            records.append(dict(region=i,start=t+start,crop=[x0,y0,cw,ch],runtime=runtime))
        restored[fmt.region_slice(region)] = np.rint(sums/mass).clip(0,255).astype(np.uint8)
    return restored,dict(model_load_seconds=model.model_load_seconds,windows=records,
        seconds_model_load_excluded=sum(v['runtime']['seconds_model_load_excluded'] for v in records))


def decode(args):
    configure_torch()
    started = time.monotonic()
    data = args.stream.read_bytes()
    control,inner_bytes,inner,ncontrol = fmt.parse(data)
    shape = (inner.meta['frame_count'],inner.meta['height'],inner.meta['width'],3)
    alpha = fmt.weights(shape,control)
    execute = not args.disable_generation and bool(np.any(alpha))
    hashes = None
    if execute:
        hashes = identities(args.adapter)
        if any(control[k] != v for k,v in hashes.items()):
            raise ValueError('adapted generator asset/profile mismatch')
    model = load_model(PATCH)
    codec = BaseCodec(REPO/'checkpoints/cvpr2026_image.pth.tar',
                      REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    enhanced,report,base = decode_enhancement(model,PATCH,codec,inner_bytes,return_base=True)
    native_peak = torch.cuda.max_memory_allocated()
    del model,codec
    gc.collect(); torch.cuda.empty_cache()
    if execute:
        generated,runtime = restore(enhanced,control,args.adapter)
        output = fmt.combine(enhanced,generated,alpha)
    else:
        runtime,output = None,enhanced.copy()
    np.testing.assert_array_equal(output[alpha == 0],enhanced[alpha == 0])
    report['enhancement_stage_non_enhanced_exact'] = report.pop('non_enhanced_exact')
    report.update(total_bytes=len(data),generation_control_bytes=ncontrol,
        generation_disabled=args.disable_generation,generation_executed=execute,
        generation_assets_validated=execute,assets=hashes,source_frames_read=False,
        generation_input='enhanced',generation_input_hash=frame_hash(enhanced),
        outside_generate_exact=True,base_reference_unchanged=True,
        output_hash=frame_hash(output),generation_runtime=runtime,
        stream_sha256=file_hash(args.stream),seconds=time.monotonic()-started,pid=os.getpid(),
        peak_cuda_allocated_bytes=max(native_peak,torch.cuda.max_memory_allocated()))
    if sum(report[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                              'incomplete_tail_bytes','generation_control_bytes')) != len(data):
        raise RuntimeError('byte accounting mismatch')
    atomic_npz(args.output/'reconstruction.npz',reconstruction=output)
    atomic_json(args.output/'decode.json',report)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--disable-generation',action='store_true')
    decode(p.parse_args())
