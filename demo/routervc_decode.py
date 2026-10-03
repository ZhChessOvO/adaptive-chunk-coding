"""Fresh RouterVC receiver: only stream + shared weights, never source frames."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import routervc_format as fmt
from demo.routervc_policy import grid_rois, features, predict, select_generate, load_model as load_router
from demo.chunk_enhancement_codec import configure_torch, load_model, decode_enhancement
from demo.scalable_codec import BaseCodec, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.online_eg_decode import identities
from demo.internal_condition_decode import restore
from demo import scalable_cooperation_format as cooperation


def coverage_from_packets(inner, rois):
    """Only received packet coordinates and time determine E coverage."""
    count=inner.meta['frame_count'];coverage=np.zeros((count,len(rois)),bool)
    for packet in inner.packets:
        p=packet.meta
        if p['roi'] not in rois: raise ValueError('packet is not in the RouterVC grid')
        i=rois.index(p['roi']);start,n=p['start'],p['count']
        if start<0 or n<1 or start+n>count or coverage[start:start+n,i].any():
            raise ValueError('invalid or duplicate E coverage')
        coverage[start:start+n,i]=True
    return coverage.mean(0).astype(np.float32)


def route(base, enhanced, inner, config, router):
    torch.set_num_threads(4)
    if file_hash(router)!=config['router'] or fmt.policy_identity()!=config['policy']:
        raise ValueError('shared Router model/policy mismatch')
    rois=grid_rois(base.shape[1],base.shape[2])
    coverage=coverage_from_packets(inner,rois)
    model=load_router(router)
    predictions=predict(model,features(base,enhanced,rois,coverage))
    if torch.is_tensor(predictions): predictions=predictions.detach().cpu().numpy()
    selection=select_generate(predictions[0,:,1],config['max_g'],config['boundary_lambda'])
    selection.update(coverage=coverage.tolist(),predictions=predictions[0].tolist(),
        rois=rois,states=[('EG' if coverage[i]>0 else 'G') if i in selection['indices']
                        else ('E' if coverage[i]>0 else 'B') for i in range(16)])
    return selection


def decode(args):
    configure_torch();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    data=args.stream.read_bytes()
    config,inner_bytes,inner,ncontrol=fmt.parse(data,allow_incomplete_tail=args.allow_incomplete_tail)
    rois=grid_rois(inner.meta['height'],inner.meta['width'])
    run_policy=not args.disable_generation and config['max_g']>0 and config['blend']>0
    if run_policy:
        if inner.meta['frame_count']<17:
            raise ValueError('RouterVC v1 G requires at least 17 frames')
        height,width=inner.meta['height'],inner.meta['width']
        for x,y,w,h in rois:
            cw=min(width,x+w+64)-max(0,x-64);ch=min(height,y+h+64)-max(0,y-64)
            if min(w,h)<=32 or cw%16 or ch%16:
                raise ValueError('RouterVC v1 G needs cores >32 and clipped context size divisible by16; use e.g.512x512 or384x256')
    # No E/G assets needed for the ordinary no-packet, G-off playback path.
    codec=BaseCodec(REPO/'checkpoints/cvpr2026_image.pth.tar',REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    if inner.packets:
        model=load_model(args.enhancement)
        enhanced,report,base=decode_enhancement(model,args.enhancement,codec,inner_bytes,
             allow_incomplete_tail=args.allow_incomplete_tail,return_base=True)
        del model
    else:
        base=codec.decode(inner.base,inner.meta['frame_count'])
        if frame_hash(base)!=inner.meta['base_rgb_sha256']: raise ValueError('UF base mismatch')
        if any(inner.meta[k]!=v for k,v in codec.models.items()): raise ValueError('UF weights mismatch')
        enhanced=base.copy()
        report=dict(base_bytes=len(inner.base),container_header_bytes=inner.base_end-len(inner.base),
            packet_bytes=0,incomplete_tail_bytes=inner.incomplete_tail_bytes,applied_packets=[],
            base_hash=frame_hash(base),non_enhanced_exact=True)
    del codec;gc.collect();torch.cuda.empty_cache()
    codec_seconds=time.monotonic()-start
    policy_start=time.monotonic()
    selection=route(base,enhanced,inner,config,args.router) if run_policy else dict(
        select_generate(np.zeros(16),0,config['boundary_lambda']),rois=rois,
        coverage=coverage_from_packets(inner,rois).tolist(),policy_skipped=True)
    if not run_policy:selection['states']=['E' if c>0 else 'B' for c in selection['coverage']]
    policy_seconds=time.monotonic()-policy_start
    control=fmt.generation_control(config,selection['indices'],rois,len(base))
    cooperation.validate(control,inner)
    alpha=cooperation.weights(base.shape,control)
    if selection['indices']:
        hashes=identities(args.adapter)
        if any(config[k]!=v for k,v in hashes.items()): raise ValueError('G weights/profile mismatch')
        generated,runtime=restore(enhanced,control,args.adapter,[])
        output=cooperation.combine(enhanced,generated,alpha)
    else: output,runtime=enhanced.copy(),None
    np.testing.assert_array_equal(output[alpha==0],enhanced[alpha==0])
    report.update(total_bytes=len(data),generation_control_bytes=ncontrol,
        stream_sha256=file_hash(args.stream),source_frames_read=False,pid=os.getpid(),
        base_reference_unchanged=True,outside_generate_exact=True,
        generation_executed=bool(selection['indices']),generation_assets_validated=bool(selection['indices']),
        shared_router_used=run_policy,explicit_G_map_bytes=0,route=selection,
        config=config,generation_runtime=runtime,generation_input_hash=frame_hash(enhanced),
        output_hash=frame_hash(output),seconds=time.monotonic()-start,
        codec_seconds=codec_seconds,policy_seconds=policy_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    assert sum(report[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
        'incomplete_tail_bytes','generation_control_bytes'))==len(data)
    atomic_npz(args.output/'reconstruction.npz',reconstruction=output,base=base,enhanced=enhanced)
    atomic_json(args.output/'decode.json',report)
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--router',type=Path,required=True)
    p.add_argument('--enhancement',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--disable-generation',action='store_true')
    p.add_argument('--allow-incomplete-tail',action='store_true')
    decode(p.parse_args())
