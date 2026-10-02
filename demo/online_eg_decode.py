"""Source-free ACSG2 receiver for a paired, separately hashed E and G model."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_codec import configure_torch, load_model, decode_enhancement
from demo.internal_condition_decode import identities as old_identities, restore
from demo.internal_condition_model import validate_bundle
from demo.scalable_codec import BaseCodec, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash


def identities(adapter):
    assert validate_bundle(torch.load(adapter,weights_only=True,map_location='cpu')) == ('internal','off')
    hashes = old_identities(adapter)
    hashes['profile'] = hashlib.sha256(json.dumps(dict(parent=hashes['profile'],
        receiver=file_hash(Path(__file__)),version='online_eg_rgb_v1',
        enhancement='hash is mandatory in inner ACSE header',input='enhanced RGB only'),sort_keys=True).encode()).hexdigest()
    return hashes


def decode(args):
    configure_torch(); start = time.monotonic()
    data = args.stream.read_bytes()
    control,inner_bytes,inner,ncontrol = fmt.parse(data)
    alpha = fmt.weights((inner.meta['frame_count'],inner.meta['height'],inner.meta['width'],3),control)
    execute = not args.disable_generation and bool(np.any(alpha))
    hashes = identities(args.adapter) if execute else None
    if execute and any(control[k] != v for k,v in hashes.items()):
        raise ValueError('G model/profile differs from transmitted control')
    model = load_model(args.enhancement)
    codec = BaseCodec(REPO/'checkpoints/cvpr2026_image.pth.tar',REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    enhanced,report,base = decode_enhancement(model,args.enhancement,codec,inner_bytes,return_base=True)
    peak = torch.cuda.max_memory_allocated()
    del model,codec
    gc.collect(); torch.cuda.empty_cache()
    if execute:
        generated,runtime = restore(enhanced,control,args.adapter,[])
        output = fmt.combine(enhanced,generated,alpha)
    else:
        output,runtime = enhanced.copy(),None
    np.testing.assert_array_equal(output[alpha==0],enhanced[alpha==0])
    report.update(total_bytes=len(data),generation_control_bytes=ncontrol,source_frames_read=False,
        enhancement_sha256=file_hash(args.enhancement),generation_executed=execute,
        generation_assets_validated=execute,assets=hashes,generation_input='enhanced_rgb_only',
        generation_input_hash=frame_hash(enhanced),base_reference_unchanged=True,outside_generate_exact=True,
        output_hash=frame_hash(output),generation_runtime=runtime,stream_sha256=file_hash(args.stream),
        seconds=time.monotonic()-start,peak_cuda_allocated_bytes=max(peak,torch.cuda.max_memory_allocated()))
    assert sum(report[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                                  'incomplete_tail_bytes','generation_control_bytes')) == len(data)
    atomic_npz(args.output/'reconstruction.npz',reconstruction=output)
    atomic_json(args.output/'decode.json',report)
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--enhancement',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--disable-generation',action='store_true')
    decode(p.parse_args())
