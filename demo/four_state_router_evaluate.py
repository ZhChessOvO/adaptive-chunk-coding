"""Grouped holdout utility-regret probe, NOT a decoded RD experiment."""
import argparse
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.four_state_router import UtilityBackbone, split
from demo.four_state_core import STATES, immutable_json
from demo.scalable_codec import file_hash


@lru_cache(maxsize=4)
def masks(n):
    if not 1<=n<=16:raise ValueError('bounded 1..16 region diagnostic')
    return ((np.arange(1<<n)[:,None]>>np.arange(n)[None])&1).astype(np.int64)


def frontier(utility,e_bytes,shared,per_g):
    utility=np.asarray(utility,dtype=float);e_bytes=np.asarray(e_bytes,dtype=np.int64)
    if utility.shape!=(len(e_bytes),4) or not np.isfinite(utility).all() or np.any(e_bytes<=0):
        raise ValueError('invalid four-state utilities/costs')
    m=masks(len(e_bytes))
    direct=utility[:,1]-utility[:,0]
    marginal=np.where(m,utility[:,3]-utility[:,1],utility[:,2]-utility[:,0])
    order=np.argsort(-marginal,axis=1,kind='stable')
    cumulative=np.c_[np.zeros(len(m)),np.cumsum(np.take_along_axis(marginal,order,axis=1),axis=1)]
    gains=m@direct+utility[:,0].sum()
    rate=m@e_bytes
    return m,order,cumulative,gains,rate,shared,per_g


def solve(table,budget,max_g):
    m,order,cumulative,gains,rate,shared,per_g=table
    if budget<0 or not 0<=max_g<=m.shape[1]:raise ValueError('invalid budget')
    best=None
    for k in range(max_g+1):
        used=rate+(shared+k*per_g if k else 0)
        scores=np.where(used<=budget,gains+cumulative[:,k],-np.inf)
        index=int(np.argmax(scores));score=float(scores[index])
        if not np.isfinite(score):continue
        if best is None or score>best['predicted_utility']+1e-12:
            states=np.where(m[index],1,0)
            states[order[index,:k]]+=2
            best=dict(states=states.tolist(),extra_bytes=int(used[index]),g_calls=k,predicted_utility=score)
    assert best is not None
    return best


def model_from_bundle(path):
    value=torch.load(path,weights_only=True,map_location='cpu')
    s=value['model']
    model=UtilityBackbone(value['config']['context'],s['mean'],s['scale'],s['target_mean'],s['target_scale'])
    model.load_state_dict(s);model.eval()
    return model


@torch.no_grad()
def utility_prediction(model,features):
    p=model(features)
    i=torch.arange(16)
    direct=p[i+1,i,0]
    return torch.stack((torch.zeros_like(direct),direct,p[0,:,1],direct+p[i+1,i,1]),dim=1).numpy()


def evaluate(root):
    torch.set_num_threads(4)
    data=torch.load(root/'data.pt',weights_only=True,map_location='cpu')
    _,valid=split(data['records'])
    inputs={arm:file_hash(root/arm/'model.pt') for arm in ('context','local')}
    path=root/'table_evaluation.json'
    if path.exists():
        from demo.chunk_enhancement_experiment import read
        old=read(path);assert old['models']==inputs and old['data']==file_hash(root/'data.pt')
        return old
    models={a:model_from_bundle(root/a/'model.pt') for a in inputs}
    records=[]
    for index in valid:
        row=data['records'][index]
        actual=row['utility'].numpy()
        tables={'table_oracle':frontier(actual,row['e_bytes'],row['g_shared_bytes'],row['g_region_bytes'])}
        tables.update({a:frontier(utility_prediction(m,row['features']),row['e_bytes'],
            row['g_shared_bytes'],row['g_region_bytes']) for a,m in models.items()})
        for ratio in (.25,.5,.75):
            for calls in (4,8):
                budget=int(sum(row['e_bytes'])*ratio)
                plans={a:solve(t,budget,calls) for a,t in tables.items()}
                for plan in plans.values():
                    plan['teacher_utility']=float(actual[np.arange(16),plan['states']].sum())
                    plan['counts']={s:plan['states'].count(i) for i,s in enumerate(STATES)}
                oracle=plans['table_oracle']['teacher_utility']
                for arm in models:
                    regret=(oracle-plans[arm]['teacher_utility'])/16
                    assert regret>=-1e-7
                    plans[arm]['mean_regional_lpips_regret']=regret
                records.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],
                    ratio=ratio,max_g_calls=calls,budget=budget,plans=plans))
        print(f'ROUTER_TABLE {len(records)//6}/{len(valid)}',flush=True)
    summary={arm:float(np.mean([r['plans'][arm]['mean_regional_lpips_regret'] for r in records])) for arm in models}
    result=dict(complete=True,models=inputs,data=file_hash(root/'data.pt'),records=records,
        mean_regional_lpips_regret=summary,regions=16,operating_points=6,holdout_windows=len(valid),
        scope='Router sequence-group holdout; E/G have seen these component-training data',
        limitation='exact table solver, not true composition oracle; no measured whole-video RD claim',
        compute_budget='maximum ROI calls, NOT a strict measured-time budget',
        no_model_promotion=True,no_real_route_decode=True)
    immutable_json(path,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(evaluate(p.parse_args().output)['mean_regional_lpips_regret'])
