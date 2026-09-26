"""Source-free receiver: only transmitted bytes and pre-shared model registry.

Run under single-process torchrun. Neither source manifests nor encoder caches
are imported here. Generation input is always the freshly decoded base.
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
sys.path.insert(0, str(REPO))
from demo import scalable_generation_format as fmt
from demo.chunk_enhancement_codec import configure_torch, decode_enhancement, load_model
from demo.scalable_codec import BaseCodec, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash

RUNS = Path("/root/autodl-fs/DCVC/runs")
PATCH = RUNS / "a800_patch_efficiency_20260926/train_l4/final.pt"
MODEL_ROOT = Path("/root/autodl-tmp/DCVC/models/seedvr2")
ASSETS = dict(dit=MODEL_ROOT/"seedvr2_ema_3b_bf16.safetensors", vae=MODEL_ROOT/"ema_vae.pth",
              positive=MODEL_ROOT/"pos_emb.pt", negative=MODEL_ROOT/"neg_emb.pt",
              lora=MODEL_ROOT/"seedvr2_codec_lora_v1_1000step.pt")


def profile():
    files = [Path(__file__), REPO/"demo/scalable_generation_format.py",
             REPO/"demo/stage_c_seedvr2_bridge.py", REPO/"demo/stage_c_a800_teacher.py",
             REPO/"demo/stage_c_seedvr2_lora_utils.py"]
    files += sorted((REPO/"third_party/SeedVR2/configs_3b").rglob("*.yaml"))
    for directory in ("models", "common", "projects"):
        files += sorted((REPO/"third_party/SeedVR2"/directory).rglob("*.py"))
    spec = dict(version="ACSG1", sample_steps=1, cfg_scale=1., dtype="bfloat16",
                input="base_only", resize="ceil16_then_restore_native", seed_rule="seed+65536*region+window_index",
                merge="triangular_temporal_then_inner_linear_feather_round_u8",
                files={str(p.relative_to(REPO)):file_hash(p) for p in files})
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()


def identities():
    return dict({k:file_hash(p) for k,p in ASSETS.items()}, profile=profile())


def restore(base, control, check=lambda:None):
    from demo.stage_c_a800_teacher import PersistentSeedVR2
    args = SimpleNamespace(upstream_root=REPO/"third_party/SeedVR2",
        dit_checkpoint=ASSETS["dit"], vae_checkpoint=ASSETS["vae"],
        positive_embedding=ASSETS["positive"], negative_embedding=ASSETS["negative"],
        lora_checkpoint=ASSETS["lora"], lora_strength=control["strength"],
        sample_steps=1, cfg_scale=1., dit_dtype="bfloat16")
    model = PersistentSeedVR2(args)
    # CPU accumulation, not a full-frame latent computation masquerading as ROI.
    restored = base.copy()
    records = []
    for i,region in enumerate(control["generate"]):
        t,n,x,y,w,h = region
        halo = control["context"]
        x0,y0 = max(0,x-halo),max(0,y-halo)
        x1,y1 = min(base.shape[2],x+w+halo),min(base.shape[1],y+h+halo)
        ch,cw = y1-y0,x1-x0
        sums = np.zeros((n,h,w,3),np.float32)
        mass = np.zeros((n,1,1,1),np.float32)
        temporal = ((9-np.abs(np.arange(17)-8))/9).astype(np.float32)[:,None,None,None]
        for j,start in enumerate(fmt.windows(n)):
            check()
            frames, runtime = model.restore(list(base[t+start:t+start+17,y0:y1,x0:x1]),
                seed=control["seed"]+65536*i+j,
                processing_height=(ch+15)//16*16,processing_width=(cw+15)//16*16,
                output_height=ch,output_width=cw)
            result = np.stack(frames)[:,y-y0:y-y0+h,x-x0:x-x0+w]
            full = x0 == y0 == 0 and cw == base.shape[2] and ch == base.shape[1]
            runtime.update(actual_compute_scope="full-frame" if full else "roi-crop",
                           full_frame_actual_compute=full,roi_compute_measured_here=not full)
            sums[start:start+17] += result*temporal
            mass[start:start+17] += temporal
            records.append(dict(region=i,start=t+start,crop=[x0,y0,cw,ch],runtime=runtime))
        restored[fmt.region_slice(region)] = np.rint(sums/mass).clip(0,255).astype(np.uint8)
    return restored, dict(model_load_seconds=model.model_load_seconds, windows=records,
        seconds_model_load_excluded=sum(v["runtime"]["seconds_model_load_excluded"] for v in records))


def decode(args):
    configure_torch()
    started = time.monotonic()
    data = args.stream.read_bytes()
    control, inner_bytes, inner, control_bytes = fmt.parse(data)
    # Even capability fallback validates the declared profile/asset identities.
    hashes = identities()
    if any(control[k] != v for k,v in hashes.items()):
        raise ValueError("generator model/profile differs from stream")
    model = load_model(PATCH)
    codec = BaseCodec(REPO/"checkpoints/cvpr2026_image.pth.tar",
                      REPO/"checkpoints/cvpr2026_video_hts.pth.tar")
    native_started = time.monotonic()
    enhanced, report, base = decode_enhancement(model,PATCH,codec,inner_bytes,return_base=True)
    native_seconds = time.monotonic()-native_started
    del model,codec
    gc.collect()
    torch.cuda.empty_cache()
    alpha = fmt.weights(base.shape,control,inner)
    runtime = None
    if args.disable_generation:
        output = enhanced.copy()
    else:
        generated,runtime = restore(base,control)
        output = np.rint(enhanced.astype(np.float32)+alpha[...,None]*(
            generated.astype(np.float32)-base.astype(np.float32))).clip(0,255).astype(np.uint8)
    np.testing.assert_array_equal(output[alpha == 0],enhanced[alpha == 0])
    # The native enhancement-only invariant does not describe the final G pixels.
    report["enhancement_stage_non_enhanced_exact"] = report.pop("non_enhanced_exact")
    report.update(total_bytes=len(data),generation_control_bytes=control_bytes,
        generation_disabled=args.disable_generation, outside_generate_exact=True,
        base_reference_unchanged=True, generation_input_hash=frame_hash(base),
        output_hash=frame_hash(output), enhanced_hash=frame_hash(enhanced),
        native_decode_seconds=native_seconds, generation_runtime=runtime,
        seconds=time.monotonic()-started, pid=os.getpid(), stream_sha256=file_hash(args.stream),
        assets=hashes, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    if sum(report[k] for k in ("base_bytes","container_header_bytes","packet_bytes",
                               "incomplete_tail_bytes","generation_control_bytes")) != len(data):
        raise RuntimeError("byte accounting mismatch")
    atomic_npz(args.output/"reconstruction.npz",reconstruction=output)
    atomic_json(args.output/"decode.json",report)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stream",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--disable-generation",action="store_true")
    decode(p.parse_args())
