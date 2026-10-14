"""Paired native-scale G continuation: local control versus mixed wider views.

Only the training field-of-view recipe differs. Wider inputs also supervise a
wider image interior; this is deliberately NOT a pure input-context ablation.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from demo.chunk_enhancement_codec import configure_torch
from demo.condition_path_decode import tensor_identity
from demo.conditioned_generation_train import differentiable_decode, restore_step_log
from demo.feature_condition_train import image_terms
from demo.roi_condition_data import encode_roi
from demo.roi_condition_train import assert_frozen
from demo.routervc_fullview_probe import read, immutable, digest
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import check_space, now
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.stage_c_seedvr2_lora_finetune import append_jsonl
from demo.stage_c_seedvr2_lora_utils import (load_lora_adapter, trainable_lora_parameters,
    load_lora_state_dict, lora_state_dict, atomic_torch_save)
from routervc.generation.context_geometry import window, master_noise, noise_crop
from routervc.generation.texture_data import REPO, choose, preload
from routervc.generation.texture_losses import phase_bias_loss
from routervc.generation.texture_ablation import learning_rate

ARMS = ('local', 'mixed')


def choose_view(rows, step, seed, arm):
    if arm not in ARMS: raise ValueError('unknown arm')
    row,small = choose(rows,step,seed)
    core = [small[0]+64,small[1]+64,128,128]
    large = window(row['shape'],core,512)
    # Four small and twelve wide updates per sixteen, preserving 3:1 REDS/UVG.
    wide = arm=='mixed' and (step-1)%16 not in (0,5,10,15)
    return row,small,large,large if wide else small


def image_objectives(pred,target,metric,step):
    if pred.shape != target.shape or pred.shape[-1] not in (256,512) or pred.shape[-2]!=pred.shape[-1]:
        raise ValueError('native paired 256 or 512 square required')
    # The reused function removes another 16 px: final border remains 64.
    terms=image_terms(pred[...,48:-48,48:-48],target[...,48:-48,48:-48],metric,step)
    phase=phase_bias_loss(pred[...,64:-64,64:-64],target[...,64:-64,64:-64])
    return terms,phase


def train(args):
    configure_torch();torch.use_deterministic_algorithms(True)
    if not os.environ.get('TMUX') or torch.cuda.device_count()!=1:raise RuntimeError('tmux / one A800 required')
    config=read(args.protocol)
    for name,value in config['code'].items():
        if digest(REPO/name)!=value:raise ValueError('bound source changed: '+name)
    for spec in (*config['assets'].values(),config['initial']):
        if digest(spec['path'])!=spec['sha256']:raise ValueError('model binding changed')
    args.output.mkdir(parents=True,exist_ok=True)
    binding=dict(protocol=digest(args.protocol),arm=args.arm)
    immutable(args.output/'binding.json',binding)
    if (args.output/'complete.json').exists():
        done=read(args.output/'complete.json')
        if done['binding']!=binding or digest(args.output/'adapter.pt')!=done['adapter_sha256']:
            raise ValueError('complete checkpoint differs')
        print('ALREADY_COMPLETE',args.arm,flush=True);return
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def progress(**kw):
        atomic_json(args.output/'progress.json',dict(arm=args.arm,utc=now(),**kw))
        print(json.dumps(dict(arm=args.arm,**kw)),flush=True)
        if stop[0]:raise InterruptedError('preparation boundary')
    samples=preload(config['rows'],progress)
    torch.manual_seed(config['seed']);assets={k:Path(v['path']) for k,v in config['assets'].items()}
    runner,device=configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=assets['dit'],vae_checkpoint=assets['vae'],positive_embedding=assets['positive'],
        negative_embedding=assets['negative'],lora_checkpoint=None,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
    runner.dit.to(device=device,dtype=torch.bfloat16)
    load_lora_adapter(runner.dit,config['initial']['path'],trainable=True,strength=1.)
    runner.dit.train();runner.vae.requires_grad_(False).eval()
    runner.vae.set_causal_slicing(split_size=None,memory_device='same')
    initial=torch.load(config['initial']['path'],map_location='cpu',weights_only=True)
    parameters=trainable_lora_parameters(runner.dit);assert_frozen(runner,initial)
    assert all(p.dtype==torch.float32 for p in parameters)
    optimizer=torch.optim.AdamW(parameters,lr=5e-6,weight_decay=1e-4)
    import lpips
    metric=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    from models.dit_v2 import na
    positive=torch.load(assets['positive'],map_location=device,weights_only=True).to(torch.bfloat16)
    text,text_shape=na.flatten([positive]);random.seed(config['seed']);completed=0
    if (args.output/'resume.pt').exists():
        saved=torch.load(args.output/'resume.pt',map_location='cpu',weights_only=True)
        if saved['binding']!=binding:raise ValueError('resume binding differs')
        load_lora_state_dict(runner.dit,saved['adapter']['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda']);random.setstate(saved['rng_python'])
        completed=saved['step']
    records=restore_step_log(args.output/'steps.jsonl',completed);torch.cuda.reset_peak_memory_stats()
    def bundle(step):
        payload=copy.deepcopy(initial);payload['state_dict']=lora_state_dict(runner.dit)
        payload.setdefault('metadata',{}).update(context_training=binding,step=step,frozen_lora=False)
        return payload
    def save(step,milestone=False):
        assert_frozen(runner,initial);adapter=bundle(step)
        atomic_torch_save(args.output/'resume.pt',dict(binding=binding,step=step,adapter=adapter,
            optimizer=optimizer.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            rng_python=random.getstate()))
        if milestone:atomic_torch_save(args.output/'checkpoints'/f'adapter_{step:06d}.pt',adapter)
    try:
        for step in range(completed+1,config['steps']+1):
            check_space()
            if stop[0]:raise InterruptedError('save at completed update')
            began=time.monotonic();row,small,large,crop=choose_view(config['rows'],step,config['seed'],args.arm)
            source,received=samples[row['sample_id']];x,y,w,h=crop
            target=torch.from_numpy(source[:,y:y+h,x:x+w].copy()).permute(0,3,1,2).to(device).float()/127.5-1
            clean=encode_roi(runner,source,crop,device).to(device);cond=encode_roi(runner,received,crop,device).to(device)
            torch.manual_seed(config['seed']+step*1000033)
            master=master_noise(config['seed']+step*97409)
            noise=noise_crop(master,crop,large).to(device)
            common_noise=noise_crop(master,small,large)
            flat_noise,shape=na.flatten([noise]);flat_cond,_=na.flatten([torch.cat([cond,torch.ones_like(cond[...,:1])],-1)])
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step,config)
            def forward(condition):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    return runner.dit(vid=torch.cat([flat_noise,condition],-1),txt=text,vid_shape=shape,txt_shape=text_shape,
                        timestep=torch.full((1,),1000.,device=device),disable_cache=False).vid_sample
            velocity=checkpoint(forward,flat_cond,use_reentrant=False)
            predicted=(flat_noise.float()-velocity.float()).reshape(clean.shape)
            terms=dict(velocity=F.mse_loss(velocity.float(),(noise-clean).reshape_as(velocity).float()),
                clean_latent_l1=F.l1_loss(predicted,clean.float()),
                latent_temporal=F.mse_loss(predicted[1:]-predicted[:-1],clean[1:].float()-clean[:-1].float()))
            # Exact activation storage offload, no downscaling, sliced decoder, or detached loss.
            # Both arms use the same memory strategy; tensors return to CUDA for backward.
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                image=differentiable_decode(runner,predicted)
            image_losses,phase=image_objectives(image,target,metric,step);terms.update(image_losses)
            loss=sum(config['loss'][k]*v for k,v in terms.items())+config['phase_weight']*phase
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            path_checks=None
            if step in (1,2):
                path_checks={k:float(torch.autograd.grad(v,parameters[-1],retain_graph=True)[0].float().norm())
                             for k,v in (('lpips',terms['lpips']),('phase',phase))}
                if not all(np.isfinite(v) and v>0 for v in path_checks.values()):raise RuntimeError('lost image gradient')
            loss.backward();grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            if grad_norm<=0:raise RuntimeError('zero LoRA gradient')
            assert_frozen(runner,initial);optimizer.step();torch.cuda.synchronize()
            record=dict(step=step,arm=args.arm,sample=row['sample_id'],dataset=row['dataset'],E_cap=row['E_cap'],
                crop=crop,paired_small=small,noise=tensor_identity(noise),paired_noise=tensor_identity(common_noise),
                condition=tensor_identity(cond),loss=float(loss.detach()),phase_error=float(phase.detach()),
                terms={k:float(v.detach()) for k,v in terms.items()},gradient_norm=grad_norm,first_gradients=path_checks,
                learning_rate=learning_rate(step,config),seconds=time.monotonic()-began,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),utc=now())
            append_jsonl(args.output/'steps.jsonl',record);records.append(record);completed=step
            print(json.dumps(record),flush=True)
            if step%25==0 or step in (args.stop_after,config['steps']):
                save(step,milestone=step%100==0 or step==config['steps']);progress(phase='training',step=step,total=config['steps'])
            del loss,terms,image_losses,phase,image,predicted,velocity,flat_cond,flat_noise,noise,cond,clean,target,common_noise,master
            if step==args.stop_after and step<config['steps']:return
        atomic_torch_save(args.output/'adapter.pt',bundle(completed))
        atomic_json(args.output/'complete.json',dict(complete=True,binding=binding,steps=completed,
            adapter_sha256=digest(args.output/'adapter.pt'),dataset_steps=dict(Counter(r['dataset'] for r in records)),
            size_steps=dict(Counter(str(r['crop'][2]) for r in records)),seconds=sum(r['seconds'] for r in records),
            peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records),production_changed=False))
        progress(phase='complete',step=completed,total=config['steps'])
    except BaseException:
        save(completed);raise
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--stop-after',type=int,default=0);train(p.parse_args())
