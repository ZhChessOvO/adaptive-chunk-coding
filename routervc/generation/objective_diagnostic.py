"""Read-only LoRA gradient attribution at a fixed, completed G checkpoint.

Weighted gradients share the actual trainable-parameter space. Their angles
describe local first-order conflicts, NOT causal image-quality improvements or
the preconditioned AdamW trajectory. No optimizer is constructed here.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from demo.chunk_enhancement_codec import configure_torch
from demo.condition_path_decode import tensor_identity
from demo.conditioned_generation_train import differentiable_decode
from demo.roi_condition_data import encode_roi
from demo.roi_condition_train import assert_frozen
from demo.routervc_fullview_probe import read, digest, verify_artifacts
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import check_space, now
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.stage_c_seedvr2_lora_utils import load_lora_adapter, trainable_lora_parameters, lora_state_dict
from routervc.generation.context_geometry import master_noise, noise_crop
from routervc.generation.context_training import image_objectives
from routervc.generation.texture_data import REPO, preload


def gradient_statistics(gradients):
    names = list(gradients)
    matrix = torch.stack([gradients[n].detach().flatten().double().cpu() for n in names])
    if not torch.isfinite(matrix).all():
        raise ValueError('nonfinite gradient')
    gram = matrix @ matrix.T
    norms = gram.diag().clamp_min(0).sqrt()
    if (norms <= 0).any():
        raise ValueError('zero objective gradient')
    total = matrix.sum(0)
    total_squared = total.square().sum()
    cosine = gram / (norms[:, None] * norms[None, :])
    return dict(names=names, norms={n:float(norms[i]) for i,n in enumerate(names)},
        cosine=cosine.tolist(), total_norm=float(total_squared.sqrt()),
        total_projection={n:float((matrix[i] @ total) / total_squared) if total_squared > 0 else None
                          for i,n in enumerate(names)},
        note='weighted LoRA parameter gradients; local SGD geometry, not AdamW or a causal quality claim')


def verify(root):
    config = read(root/'protocol.json')
    for n,s in config['code'].items():
        if digest(REPO/n) != s: raise ValueError('diagnostic source changed: '+n)
    for spec in (*config['assets'].values(), config['initial']):
        if digest(spec['path']) != spec['sha256']: raise ValueError('model changed')
    done = read(root/'complete.json')
    verify_artifacts(root,done['artifacts'])
    return done


def probe(runner, device, metric, text, text_shape, sample, case, config, parameters):
    from models.dit_v2 import na
    source, received = sample
    crop = case['crop']; x,y,w,h = crop; step = case['schedule_step']
    target = torch.from_numpy(source[:,y:y+h,x:x+w].copy()).permute(0,3,1,2).to(device).float()/127.5-1
    clean = encode_roi(runner,source,crop,device).to(device)
    cond = encode_roi(runner,received,crop,device).to(device)
    torch.manual_seed(config['seed']+step*1000033)
    noise = noise_crop(master_noise(config['seed']+step*97409),crop,case['large']).to(device)
    flat_noise, shape = na.flatten([noise])
    flat_cond,_ = na.flatten([torch.cat([cond,torch.ones_like(cond[...,:1])],-1)])
    def forward(condition):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return runner.dit(vid=torch.cat([flat_noise,condition],-1),txt=text,
                vid_shape=shape,txt_shape=text_shape,timestep=torch.full((1,),1000.,device=device),
                disable_cache=False).vid_sample
    velocity = checkpoint(forward,flat_cond,use_reentrant=False)
    predicted = (flat_noise.float()-velocity.float()).reshape(clean.shape)
    terms = dict(velocity=F.mse_loss(velocity.float(),(noise-clean).reshape_as(velocity).float()),
        clean_latent_l1=F.l1_loss(predicted,clean.float()),
        latent_temporal=F.mse_loss(predicted[1:]-predicted[:-1],clean[1:].float()-clean[:-1].float()))
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        image = differentiable_decode(runner,predicted)
    image_losses,phase = image_objectives(image,target,metric,step)
    terms.update(image_losses); terms['phase'] = phase
    weights = dict(config['loss'],phase=config['phase_weight'])
    gradients = {}; magnitudes = {}
    for name,value in terms.items():
        grad = torch.autograd.grad(value*weights[name],parameters,retain_graph=True)
        gradients[name] = torch.cat([g.detach().float().cpu().flatten() for g in grad])
        magnitudes[name] = dict(raw=float(value.detach()),weight=weights[name],weighted=float(value.detach())*weights[name])
        print(json.dumps(dict(case=case['id'],objective=name,norm=float(gradients[name].norm()))),flush=True)
        del grad
    # The actual summed backward can differ slightly from separate BF16 VJPs.
    # Measure that numeric discrepancy, rather than assuming exact linearity.
    total_loss = sum(weights[n]*v for n,v in terms.items())
    grad = torch.autograd.grad(total_loss,parameters)
    actual = torch.cat([g.detach().float().cpu().flatten() for g in grad])
    separate = sum(gradients.values())
    error = float((actual-separate).norm()/actual.norm().clamp_min(1e-30))
    stats = gradient_statistics(gradients)
    result = dict(case=case,values=magnitudes,gradient=stats,combined_backward_relative_error=error,
        observations=dict(condition=tensor_identity(cond),noise=tensor_identity(noise),
                          predicted=tensor_identity(predicted),image=tensor_identity(image)),
        gradient_hashes={n:tensor_identity(g) for n,g in gradients.items()},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    root=args.root
    if (root/'complete.json').exists():
        print(json.dumps(verify(root)),flush=True); return
    config=read(root/'protocol.json')
    for n,s in config['code'].items():
        if digest(REPO/n)!=s:raise ValueError('bound source changed: '+n)
    for spec in (*config['assets'].values(),config['initial']):
        if digest(spec['path'])!=spec['sha256']:raise ValueError('model binding changed')
    configure_torch(); torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count()!=1:raise RuntimeError('one visible A800 required')
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def progress(**kw):
        atomic_json(root/'progress.json',dict(utc=now(),**kw))
        print(json.dumps(kw),flush=True)
        if stop[0]:raise InterruptedError('resume at completed case')
    samples=preload(config['rows'],progress)
    assets={k:Path(v['path']) for k,v in config['assets'].items()}
    torch.manual_seed(config['seed'])
    runner,device=configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=assets['dit'],vae_checkpoint=assets['vae'],positive_embedding=assets['positive'],
        negative_embedding=assets['negative'],lora_checkpoint=None,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
    runner.dit.to(device=device,dtype=torch.bfloat16)
    load_lora_adapter(runner.dit,config['initial']['path'],trainable=True,strength=1.)
    runner.dit.train(); runner.vae.requires_grad_(False).eval()
    runner.vae.set_causal_slicing(split_size=None,memory_device='same')
    initial=torch.load(config['initial']['path'],weights_only=True,map_location='cpu')
    parameters=trainable_lora_parameters(runner.dit); assert_frozen(runner,initial)
    import lpips
    metric=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    from models.dit_v2 import na
    positive=torch.load(assets['positive'],map_location=device,weights_only=True).to(torch.bfloat16)
    text,text_shape=na.flatten([positive]);records=[]
    try:
        for case in config['cases']:
            check_space();progress(phase='gradient_diagnostic',completed=len(records),total=len(config['cases']),case=case['id'])
            path=root/(case['id']+'.json');binding=dict(protocol=digest(root/'protocol.json'),case=case['id'])
            if path.exists():
                record=read(path)
                if record['binding']!=binding:raise ValueError('case binding changed')
            else:
                torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                record=probe(runner,device,metric,text,text_shape,samples[case['sample_id']],case,config,parameters)
                if case['repeat']:
                    again=probe(runner,device,metric,text,text_shape,samples[case['sample_id']],case,config,parameters)
                    for key in ('values','gradient','gradient_hashes','observations','combined_backward_relative_error'):
                        if again[key]!=record[key]:raise ValueError('repeated gradient differs: '+key)
                record.update(binding=binding,exact_repeat=case['repeat'],seconds=time.monotonic()-began)
                assert_frozen(runner,initial)
                for k,v in lora_state_dict(runner.dit).items():
                    if not torch.equal(v,initial['state_dict'][k]):raise ValueError('read-only probe changed weights')
                atomic_json(path,record)
                gc.collect();torch.cuda.empty_cache()
            records.append(record)
            if args.stop_after and len(records)>=args.stop_after:return
        atomic_json(root/'summary.json',dict(cases=records,training=False,unchanged_weights=True,
            scope='train-only LoRA gradient geometry; no test selection; no causal quality proof'))
        names=['protocol.json','summary.json']+[c['id']+'.json' for c in config['cases']]
        atomic_json(root/'complete.json',dict(complete=True,training=False,points=len(records),
            artifacts={n:digest(root/n) for n in names},utc=now()))
        progress(phase='complete',completed=len(records),total=len(records))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stop-after',type=int,default=0);main(p.parse_args())
