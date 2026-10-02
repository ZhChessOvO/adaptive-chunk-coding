"""Paired fixed-E versus online E+G LoRA training on one A800."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.chunk_enhancement_codec import configure_torch, load_model
from demo.chunk_enhancement_experiment import read, checkpoint as e_payload
from demo.condition_path_decode import tensor_identity
from demo.feature_condition_train import differentiable_decode, restore_step_log
from demo.internal_condition_model import ConditionBranch, make_bundle
from demo.online_eg_data import (FORMAT, FEATURES, training_entries, choose_crop, online_rgb,
                                 encode_pixels, tensor_rgb, STREAMS)
from demo.roi_condition_data import encode_roi, core_image_terms
from demo.roi_condition_train import DEFAULT as ROI, assert_frozen
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import check_space, resources, now
from demo.scalable_generation_decode import ASSETS, PATCH
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.stage_c_seedvr2_lora_finetune import select_training_entry, append_jsonl
from demo.stage_c_seedvr2_lora_utils import (load_lora_adapter, trainable_lora_parameters,
    load_lora_state_dict, lora_state_dict, atomic_torch_save)

DEFAULT = Path('/root/autodl-fs/DCVC/runs/a800_online_eg_20261002')
INITIAL = ROI/'rgb/adapter.pt'
CODE = ['online_eg_train.py', 'online_eg_data.py', 'roi_condition_data.py',
        'feature_condition_train.py', 'chunk_enhancement_model.py', 'feature_head_enhancement.py',
        'chunk_enhancement_codec.py', 'stage_c_seedvr2_lora_utils.py', 'stage_c_seedvr2_bridge.py']


def lr(step, steps):
    warm = min(100, steps)
    if step <= warm:
        return 1e-5*step/warm
    return 2e-6 + .5*8e-6*(1+math.cos(math.pi*(step-warm)/max(1, steps-warm)))


def norm(parameters):
    values = [p.grad.float().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.


def main(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    configure_torch(); torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1
    entries = training_entries()
    config = dict(version=1, format=FORMAT, mode=args.mode, steps=args.steps, seed=261002,
        initial_lora=file_hash(INITIAL), initial_adapter=file_hash(INITIAL),
        initial_e=file_hash(PATCH), feature_manifest=file_hash(FEATURES),
        assets={k:file_hash(v) for k,v in ASSETS.items()}, code={n:file_hash(REPO/'demo'/n) for n in CODE},
        initial='continue completed 03.14 RGB LoRA + A pad16 E; not earlier initialization',
        crop=256, core=128, halo=64, qsteps=[.5, 1., 2.], conditions=['partial','full','none'],
        uvg_probability=.25, source_roles='90 REDS train + 30 UVG adaptation; no val',
        loss=dict(velocity=.25,clean_latent_l1=.025,latent_temporal=.05,lpips=.5,
                  low_frequency=.1,edge=.05,rgb_temporal=.1),
        e_objective_weight=.1, e_fidelity='512/q * MSE + 8 * temporal MSE; whole intersecting packets',
        rate='noisy likelihood estimate in training; real bytes mandatory at evaluation',
        lr=dict(e_peak=1e-5,lora_peak=1e-5,end=2e-6,warmup=min(100,args.steps)),
        frozen=['UF','UF reconstruction head','base DiT','VAE'],
        decoder_slicing=False, encoder_slicing='receiver split4; upstream detached histories = truncated cross-slice BPTT',
        pixel_quantization='round-to-u8 forward STE; no source bypass',
        no_e='one third G rehearsal, E optimizer skipped',
        extra_feature_interface=False, checkpoint_every=25)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'config.json').exists():
        assert read(args.output/'config.json') == config, 'config/source changed; use another run'
    atomic_json(args.output/'config.json', config)
    atomic_json(args.output/'training_manifest.json', dict(entries=entries, roles=config['source_roles']))
    # Preload bounded 120 samples into host RAM (~6 GiB), avoiding repeated slow FS I/O.
    samples = {}
    for index, entry in enumerate(entries):
        for path, key in [('pair_path','pair_hash'), ('feature_path','feature_hash'), ('base_path','base_file_hash')]:
            assert file_hash(Path(entry[path])) == entry[key], f'changed training asset: {entry[path]}'
        with np.load(entry['pair_path']) as p:
            source, base = p['source'].copy(), p['base'].copy()
        chunks = torch.load(entry['feature_path'], weights_only=True, map_location='cpu')['chunks']
        samples[entry['sample_id']] = source, base, chunks
        if index % 20 == 0:
            print(json.dumps(dict(loading=index+1,total=len(entries),utc=now())), flush=True)
    torch.manual_seed(config['seed'])
    runner, device = configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'], vae_checkpoint=ASSETS['vae'], positive_embedding=ASSETS['positive'],
        negative_embedding=ASSETS['negative'], lora_checkpoint=None,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
    runner.dit.to(device=device, dtype=torch.bfloat16)
    load_lora_adapter(runner.dit, INITIAL, trainable=True, strength=1.)
    runner.dit.train(); runner.vae.requires_grad_(False).eval()
    runner.vae.set_causal_slicing(split_size=None,memory_device='same')
    model = load_model(PATCH)
    if args.mode == 'fixed':
        model.requires_grad_(False)
    # Training likelihood noise is seeded per step, paired across both arms.
    model.train()
    initial_e = {k:v.detach().cpu().clone() for k,v in model.export_state().items()}
    initial_g = torch.load(INITIAL,weights_only=True,map_location='cpu')
    branch = ConditionBranch('internal','off').requires_grad_(False)
    branch.load_state_dict(initial_g['branch_state'])
    gparams = trainable_lora_parameters(runner.dit)
    eparams = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.AdamW(gparams, lr=1e-5,weight_decay=1e-4)
    optimizer_e = torch.optim.Adam(eparams, lr=1e-5) if eparams else None
    import lpips
    metric = lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    completed = 0
    resume_path = args.output/'resume.pt'
    if resume_path.exists():
        saved = torch.load(resume_path,weights_only=True,map_location='cpu')
        assert saved['config'] == config
        load_lora_state_dict(runner.dit,saved['adapter']['state_dict'])
        model.load_export_state(saved['enhancement']['model'])
        optimizer_g.load_state_dict(saved['optimizer_g'])
        if optimizer_e: optimizer_e.load_state_dict(saved['optimizer_e'])
        completed = saved['step']
    rows = restore_step_log(args.output/'steps.jsonl', completed)
    from models.dit_v2 import na
    positive = torch.load(ASSETS['positive'],weights_only=True,map_location=device).to(torch.bfloat16)
    text, text_shape = na.flatten([positive])
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    clean_cache = {}
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()

    def payloads(step):
        assert_frozen(runner, initial_g)
        assert not any(p.requires_grad or p.grad is not None for p in model.head.parameters())
        adapter = make_bundle(dict(initial_g,state_dict=lora_state_dict(runner.dit)),branch,config,step)
        adapter['metadata'].update(frozen_lora=False, online_eg_version=1, lora_parameters=2822656)
        enhancement = e_payload(model,step,config=config)
        enhancement['model'] = {k:v.detach().cpu() for k,v in enhancement['model'].items()}
        return adapter, enhancement

    def save(step, milestone=False):
        adapter, enhancement = payloads(step)
        atomic_torch_save(resume_path,dict(step=step,config=config,adapter=adapter,enhancement=enhancement,
            optimizer_g=optimizer_g.state_dict(),optimizer_e=optimizer_e.state_dict() if optimizer_e else None))
        if milestone:
            atomic_torch_save(args.output/'checkpoints'/f'adapter_{step:06d}.pt',adapter)
            atomic_torch_save(args.output/'checkpoints'/f'enhancement_{step:06d}.pt',enhancement)

    for step in range(completed+1, args.steps+1):
        try: check_space()
        except BaseException:
            save(step-1); raise
        if stop[0]:
            save(step-1); raise InterruptedError('atomic checkpoint saved')
        begin = time.monotonic()
        entry, rng = select_training_entry(entries, step, config['seed'], .25)
        mode = ('partial','full','none')[(step-1)%3]
        packets = entry['packets'][mode]
        crop = choose_crop(rng, packets)
        x,y,w,h = crop
        source,base,chunks = samples[entry['sample_id']]
        target = tensor_rgb(source[:,y:y+h,x:x+w],device)*2-1
        key = (entry['sample_id'],tuple(crop))
        if key not in clean_cache:
            clean_cache[key] = encode_roi(runner,source,crop,device)
        clean = clean_cache[key].to(device)
        torch.manual_seed(config['seed']+step*1000033)
        for opt in (optimizer_g,optimizer_e):
            if opt:
                for group in opt.param_groups: group['lr'] = lr(step,args.steps)
                opt.zero_grad(set_to_none=True)
        pixels, e_terms = online_rgb(model,source,base,chunks,packets,crop,differentiable=args.mode=='joint')
        train_e = args.mode == 'joint' and bool(e_terms['packets'])
        if train_e: pixels.retain_grad()
        # Full encoder recomputation resets slicing state, including on resume.
        cond = checkpoint(lambda p:encode_pixels(runner,p),pixels,use_reentrant=False)
        assert cond.dtype == torch.bfloat16 and cond.shape == clean.shape
        first_alignment = None
        if step <= 2:
            values = pixels.detach().round().byte().permute(0,2,3,1).cpu().numpy()
            native = encode_roi(runner,values,(0,0,w,h),device).to(device)
            torch.testing.assert_close(cond.detach(),native,rtol=0,atol=0)
            first_alignment = True
        generator = torch.Generator(device=device).manual_seed(config['seed']+step*97409)
        noise = torch.randn(clean.shape,device=device,dtype=torch.bfloat16,generator=generator)
        flat_noise, shape = na.flatten([noise])
        flat_condition,_ = na.flatten([torch.cat([cond,torch.ones_like(cond[...,:1])],-1)])
        def dit_forward(condition):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return runner.dit(vid=torch.cat([flat_noise,condition],-1),txt=text,
                    vid_shape=shape,txt_shape=text_shape,timestep=torch.full((1,),1000.,device=device),
                    disable_cache=False).vid_sample
        velocity = checkpoint(dit_forward,flat_condition,use_reentrant=False)
        predicted = (flat_noise.float()-velocity.float()).reshape(clean.shape)
        terms = dict(velocity=F.mse_loss(velocity.float(),(noise-clean).reshape_as(velocity).float()),
            clean_latent_l1=F.l1_loss(predicted,clean.float()),
            latent_temporal=F.mse_loss(predicted[1:]-predicted[:-1],clean[1:].float()-clean[:-1].float()))
        image = checkpoint(lambda z:differentiable_decode(runner,z),predicted,use_reentrant=False)
        terms.update(core_image_terms(image,target,metric,step))
        g_loss = sum(config['loss'][k]*v for k,v in terms.items())
        path_checks = None
        if step == 1 and train_e:
            # Test the perceptual term ALONE, not a nonzero direct-E loss gradient.
            probes = [pixels,model.analysis[-1].weight,model.feature_synthesis[-1].weight]
            grads = torch.autograd.grad(terms['lpips'],probes,retain_graph=True)
            path_checks = [float(g.float().norm()) for g in grads]
            assert all(np.isfinite(v) and v > 0 for v in path_checks)
            pixels.grad = None
        loss = g_loss + config['e_objective_weight']*(e_terms['bpp']+e_terms['fidelity'])
        if not torch.isfinite(loss): raise RuntimeError('nonfinite joint loss')
        loss.backward()
        gnorm,enorm = norm(gparams),norm(eparams)
        pixel_gradient = float(pixels.grad.float().norm()) if train_e else 0.
        if not np.isfinite(gnorm) or gnorm <= 0: raise RuntimeError('invalid LoRA gradient')
        if train_e and (not np.isfinite(enorm) or enorm <= 0 or pixel_gradient <= 0):
            raise RuntimeError('no finite G -> E gradient')
        torch.nn.utils.clip_grad_norm_(gparams,1.,error_if_nonfinite=True)
        if train_e: torch.nn.utils.clip_grad_norm_(eparams,1.,error_if_nonfinite=True)
        assert_frozen(runner,initial_g)
        assert not any(p.grad is not None for p in model.head.parameters())
        optimizer_g.step()
        if train_e: optimizer_e.step()
        torch.cuda.synchronize()
        record = dict(step=step,sample=entry['sample_id'],dataset=entry['dataset'],condition=mode,
            crop=list(crop),packet_ids=e_terms['packets'],loss=float(loss.detach()),g_loss=float(g_loss.detach()),
            terms={k:float(v.detach()) for k,v in terms.items()}, e_bpp_estimate=float(e_terms['bpp'].detach()),
            e_fidelity_loss=float(e_terms['fidelity'].detach()),lora_gradient_norm=gnorm,e_gradient_norm=enorm,
            g_to_e_rgb_gradient_norm=pixel_gradient,first_lpips_path_gradients=path_checks,
            receiver_condition_exact=first_alignment,learning_rate=lr(step,args.steps),
            condition_identity=tensor_identity(cond),diffusion_noise_identity=tensor_identity(noise),
            seconds=time.monotonic()-begin,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),utc=now())
        append_jsonl(args.output/'steps.jsonl',record); rows.append(record)
        print(json.dumps(record),flush=True)
        if step%25 == 0 or step in (args.steps,args.stop_after):
            save(step,milestone=step%1000==0 or step==args.steps)
            atomic_json(args.output/'progress.json',dict(step=step,total=args.steps,mode=args.mode,
                attempt_seconds=time.monotonic()-started,resources=resources()))
        del loss,g_loss,terms,e_terms,pixels,cond,flat_condition,velocity,predicted,image,target,clean,noise
        if step == args.stop_after and step < args.steps:
            torch.distributed.destroy_process_group(); return
    adapter,enhancement = payloads(args.steps)
    atomic_torch_save(args.output/'adapter.pt',adapter)
    atomic_torch_save(args.output/'enhancement.pt',enhancement)
    changed = any(not torch.equal(v.detach().cpu(),initial_e[k]) for k,v in model.export_state().items())
    assert changed == (args.mode == 'joint')
    atomic_json(args.output/'complete.json',dict(steps=args.steps,mode=args.mode,enhancement_changed=changed,
        adapter_sha256=file_hash(args.output/'adapter.pt'),enhancement_sha256=file_hash(args.output/'enhancement.pt'),
        trainable_e_parameters=sum(p.numel() for p in eparams),trainable_g_parameters=2822656,
        dataset_steps=dict(Counter(r['dataset'] for r in rows)),condition_steps=dict(Counter(r['condition'] for r in rows)),
        total_step_seconds=sum(r['seconds'] for r in rows),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),resources=resources()))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['fixed','joint'],required=True)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--stop-after',type=int,default=-1)
    args = parser.parse_args()
    if args.steps < 1: parser.error('positive steps required')
    main(args)
