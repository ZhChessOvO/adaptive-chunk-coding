"""One source-free utility backbone, with optional within-backbone context.

Inputs describe B and the actually decoded candidate Y, never the source X.
The same body predicts direct E gain and conditional G(Y) gain. The latter is
NOT the old independent G(B) expert. This pilot does not deploy a free G policy:
the current receiver still consumes charged explicit G controls.
"""
import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import DEFAULT, ROIS, STATES, immutable_json, verify
from demo.chunk_enhancement_experiment import read
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_codec import atomic_json, file_hash

FORMAT='four_state_conditional_utility_v1'
TARGETS=('direct_lpips_gain','conditional_G_lpips_gain','direct_psnr_gain',
         'conditional_G_psnr_gain','direct_temporal_gain','conditional_G_temporal_gain')


def patch_features(pixels, index):
    """Tile-internal measurements: never accidentally read neighboring E."""
    x,y,w,h=ROIS[index]
    values=pixels[:,y:y+h,x:x+w].astype(np.float32)/255
    luma=values@np.array([.2126,.7152,.0722],np.float32)
    dx,dy=np.abs(np.diff(luma,axis=2)),np.abs(np.diff(luma,axis=1))
    dt=np.abs(np.diff(luma,axis=0))
    return np.array([*values.mean(axis=(0,1,2)),*values.std(axis=(0,1,2)),
        luma.mean(),luma.std(),dx.mean(),dx.std(),dy.mean(),dy.std(),
        dt.mean(),dt.std(),luma.mean(axis=(1,2)).std(),
        (x+w/2)/pixels.shape[2],(y+h/2)/pixels.shape[1],
        float(x==0 or y==0 or x+w==pixels.shape[2] or y+h==pixels.shape[1])],np.float32)


def gain_targets(scores, with_E):
    before=scores['E' if with_E else 'B']
    after=scores['EG' if with_E else 'G']
    base=scores['B']
    values=[]
    for metric,direction in (('lpips_alex',-1),('psnr_db',1),('temporal_delta_mae',-1)):
        values += [direction*(before[metric]-base[metric]),direction*(after[metric]-before[metric])]
    return np.asarray(values,np.float32)


def make_views(base, enhanced, qualities):
    """One no-E view supervises all cells; each isolated-E view only its own cell.

    Neighbor G changes under E are not silently assigned the no-E labels.
    """
    bf=np.stack([patch_features(base,i) for i in range(16)])
    ef=np.stack([patch_features(enhanced,i) for i in range(16)])
    # B features, Y features, E presence; no original/source features.
    baseline=np.concatenate((bf,bf,np.zeros((16,1),np.float32)),axis=1)
    views=np.repeat(baseline[None],17,axis=0)
    targets=np.zeros((17,16,6),np.float32)
    mask=np.zeros((17,16),np.float32)
    for i in range(16):
        targets[0,i]=gain_targets(qualities[i],False);mask[0,i]=1
        views[i+1,i,18:36]=ef[i];views[i+1,i,36]=1
        targets[i+1,i]=gain_targets(qualities[i],True);mask[i+1,i]=1
    return views,targets,mask


class UtilityBackbone(nn.Module):
    def __init__(self, context, mean, scale, target_mean, target_scale, width=64):
        super().__init__()
        self.context=context
        for name,value in [('mean',mean),('scale',scale),('target_mean',target_mean),('target_scale',target_scale)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.embed=nn.Linear(len(mean),width)
        self.hidden=nn.Linear(width,width)
        self.output=nn.Linear(width,6)

    def forward(self, features):
        h=F.gelu(self.embed((features-self.mean)/self.scale))
        if self.context:
            local=h.transpose(1,2).reshape(-1,h.shape[-1],4,4)
            local=F.avg_pool2d(local,3,stride=1,padding=1,count_include_pad=False)
            # Context inside the ONE body, after local embeddings; no experts.
            h=(h+local.flatten(2).transpose(1,2))/2
        h=F.gelu(self.hidden(h))
        raw=self.output(h)*self.target_scale+self.target_mean
        direct_mask=torch.stack((features[...,36],torch.ones_like(features[...,36]))*3,dim=-1)
        return raw*direct_mask


def prepare(root, output):
    manifest=read(root/'labels.json')
    assert manifest['complete']
    dependencies=dict(labels=file_hash(root/'labels.json'),code=file_hash(Path(__file__)))
    record=output/'data.json'
    if record.exists():
        old=read(record);assert old['dependencies']==dependencies
        verify(output,old['artifacts'])
        return torch.load(output/'data.pt',weights_only=True,map_location='cpu')
    records=[]
    for entry in manifest['samples']:
        path=Path(entry['path']);assert file_hash(path)==entry['sha256']
        r=read(path);folder=root/'received'/r['sample_id']
        er=read(folder/'E.json');verify(folder,er['artifacts'])
        with np.load(folder/'received_E.npz',allow_pickle=False) as cache:
            views,targets,mask=make_views(cache['base'],cache['enhanced'],[v['quality'] for v in r['regions']])
        records.append(dict(sample_id=r['sample_id'],dataset=r['dataset'],sequence=r['sequence'],
            features=torch.from_numpy(views),targets=torch.from_numpy(targets),mask=torch.from_numpy(mask),
            utility=torch.tensor([[v['lpips_gain'][s] for s in STATES] for v in r['regions']]),
            e_bytes=[v['costs']['e_packet_bytes'] for v in r['regions']],
            g_shared_bytes=r['regions'][0]['costs']['g_shared_bytes'],
            g_region_bytes=r['regions'][0]['costs']['g_region_bytes']))
        print(f'ROUTER_FEATURES {len(records)}/{len(manifest["samples"])}',flush=True)
    data=dict(records=records,targets=TARGETS,features=37,dependencies=dependencies)
    atomic_torch(output/'data.pt',data)
    atomic_json(record,dict(dependencies=dependencies,artifacts={'data.pt':file_hash(output/'data.pt')}))
    return data


def split(records,smoke=False):
    if smoke: return list(range(len(records))),list(range(len(records)))
    held=set()
    for domain in ('REDS','UVG'):
        sequences=sorted({r['sequence'] for r in records if r['dataset']==domain})
        # Simple grouped 80/20 development split, not an independent codec test.
        rng=np.random.default_rng(20261002)
        rng.shuffle(sequences)
        held.update((domain,s) for s in sequences[::5])
    valid=[i for i,r in enumerate(records) if (r['dataset'],r['sequence']) in held]
    train=[i for i in range(len(records)) if i not in valid]
    assert train and valid and not(set(train)&set(valid))
    return train,valid


def batches(records,indices):
    return [torch.cat([records[i][name] for i in indices]) for name in ('features','targets','mask')]


def train(data, output, *, context, epochs=240, stop_after=0, smoke=False):
    torch.set_num_threads(4);torch.manual_seed(20261002)
    records=data['records'];ti,vi=split(records,smoke)
    x,y,mask=batches(records,ti);vx,vy,vm=batches(records,vi)
    # Statistics come only from Router-training groups, never holdout targets.
    mean=x.reshape(-1,37).mean(0);scale=x.reshape(-1,37).std(0).clamp_min(.01)
    valid_y=y[mask.bool()];ym=valid_y.mean(0);ys=valid_y.std(0).clamp_min(.01)
    model=UtilityBackbone(context,mean,scale,ym,ys)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    cfg=dict(format=FORMAT,context=context,epochs=epochs,smoke=smoke,
             data=data['dependencies'],code=file_hash(Path(__file__)),train_indices=ti,validation_indices=vi,
             target_weights=[1.,1.,.1,.1,.1,.1],seed=20261002,batch_size=64)
    output.mkdir(parents=True,exist_ok=True);immutable_json(output/'config.json',cfg)
    last=output/'resume.pt';start=0;history=[]
    if last.exists():
        saved=torch.load(last,weights_only=True,map_location='cpu');assert saved['config']==cfg
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch'];history=saved['history']
    weights=torch.tensor(cfg['target_weights'])
    began=time.monotonic()
    for epoch in range(start,epochs):
        generator=torch.Generator().manual_seed(20261002+epoch)
        order=torch.randperm(len(x),generator=generator)
        losses=[];model.train()
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            predictions=model(x[batch])
            loss=(F.smooth_l1_loss(predictions/ys,y[batch]/ys,reduction='none')*
                  mask[batch,...,None]*weights).sum()/(mask[batch].sum()*weights.sum())
            if not torch.isfinite(loss):raise RuntimeError('nonfinite Router loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            error=(model(vx)-vy).abs()
            mae=(error*vm[...,None]).sum((0,1))/vm.sum()
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_mae=mae.tolist()))
        if (epoch+1)%5==0 or epoch+1==epochs or epoch+1==stop_after:
            atomic_torch(last,dict(config=cfg,epoch=epoch+1,model=model.state_dict(),
                optimizer=optimizer.state_dict(),history=history))
            atomic_json(output/'progress.json',history[-1])
        if (epoch+1)%20==0 or epoch+1==epochs:
            print(f'ROUTER context={context} epoch={epoch+1}/{epochs} loss={history[-1]["loss"]:.5f}',flush=True)
        if epoch+1==stop_after:return
    if not (output/'complete.json').exists():
        atomic_torch(output/'model.pt',dict(format=FORMAT,config=cfg,model=model.state_dict()))
        atomic_json(output/'complete.json',dict(complete=True,epochs=epochs,context=context,
            parameters=sum(v.numel() for v in model.parameters()),seconds_this_attempt=time.monotonic()-began,
            training_samples=len(ti),validation_samples=len(vi),validation_mae=history[-1]['validation_mae'],
            model_sha256=file_hash(output/'model.pt'),receiver_inputs_only=True,
            profile_deployment=False,independent_codec_evaluation=False))


def exact(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for u,v in zip(a,b):exact(u,v)
    else:assert a==b


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    data=prepare(args.root,args.output)
    if args.smoke:
        train(data,args.output/'resume',context=True,epochs=4,stop_after=2,smoke=True)
        train(data,args.output/'resume',context=True,epochs=4,smoke=True)
        train(data,args.output/'direct',context=True,epochs=4,smoke=True)
        a=torch.load(args.output/'resume/resume.pt',weights_only=True,map_location='cpu')
        b=torch.load(args.output/'direct/resume.pt',weights_only=True,map_location='cpu')
        exact(a,b)
        immutable_json(args.output/'smoke.json',dict(complete=True,exact_resume=True,epochs=4))
    else:
        for name,context in (('context',True),('local',False)):
            train(data,args.output/name,context=context,epochs=args.epochs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int,default=240)
    main(p.parse_args())
