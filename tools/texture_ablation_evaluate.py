"""120 new ROI decodes plus 135 verified historical points, four loss arms."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.routervc_fullview_probe import read, immutable, digest, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from demo.scalable_experiment import quality
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.generation.texture_data import CURRENT, REPO
from tools.texture_ablation import NEW_ARMS, PREVIOUS
from tools.texture_repair import RepairRun
from tools.texture_repair_evaluate import aggregate
from tools.texture_diagnostic import core, run_one, phase_diagnostic
from tools.texture_repair_review import temporal_phase
from tools.texture_ablation_report import draw, trajectory


def verify_evaluation(root):
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    protocol=read(root/'protocol.json')
    for model in protocol['models'].values():
        if digest(model['path'])!=model['sha256']:raise ValueError('candidate checkpoint changed')
    for item in read(root/'summary.json')['cases']:
        for row in item['results']:
            if digest(row['output_path'])!=row['artifacts']['output.npz']:
                raise ValueError('evaluation pixels changed')
    if done['new_points']!=120 or done['reused_points']!=135 or done['logical_points']!=255:
        raise ValueError('unexpected point counts')
    return done


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    if (args.output/'complete.json').exists():
        print(json.dumps(verify_evaluation(args.output)),flush=True);return
    run=RepairRun(SimpleNamespace(output=args.output,command='evaluate',max_hours=3));run.thread.start()
    model=None;metric=None;all_results=[];temporal=[]
    try:
        training=read(args.root/'formal/protocol.json')
        for name,value in training['code'].items():
            if digest(REPO/name)!=value:raise ValueError('bound source changed: '+name)
        old=PREVIOUS/'evaluation'
        if digest(old/'complete.json')!=training['reference_evaluation']:
            raise ValueError('historical evaluation receipt changed')
        verify_artifacts(old,read(old/'complete.json')['artifacts'])
        previous=read(old/'summary.json');old_protocol=read(old/'protocol.json')
        models={f'{a}_{step:06d}':args.root/'formal'/a/'checkpoints'/f'adapter_{step:06d}.pt'
                for a in NEW_ARMS for step in (100,200,300,400)}
        assets={k:v['path'] for k,v in training['assets'].items()}
        protocol=dict(version=1,cases=old_protocol['cases'],
            models={**old_protocol['models'],**{n:dict(path=str(p),sha256=digest(p)) for n,p in models.items()}},
            training_protocol=digest(args.root/'formal/protocol.json'),
            reference_complete=training['reference_evaluation'],code=digest(Path(__file__)),
            scope='fixed raw G ROIs only; old pixels authenticated and reused; no model promotion')
        immutable(args.output/'protocol.json',protocol)
        for item in protocol['models'].values():
            if digest(item['path'])!=item['sha256']:raise ValueError('model changed')
        states={n:torch.load(p,weights_only=True,map_location='cpu')['state_dict'] for n,p in models.items()}
        for old_item in previous['cases']:
            case=old_item['case'];folder=args.output/case['sample_id'];folder.mkdir(exist_ok=True)
            old_folder=old/case['sample_id']
            with np.load(old_folder/'inputs.npz') as z:
                source=z['source'].copy();received=z['received'].copy()
            rows=[]
            for old_row in old_item['results']:
                dest=old_folder/old_row['name'];verify_artifacts(dest,old_row['artifacts'])
                if read(dest/'result.json')!=old_row:raise ValueError('old row/summary mismatch')
                rows.append(dict(old_row,output_path=str(dest/'output.npz'),provenance='verified_historical_reuse'))
            expected={k:rows[0]['observation'][k] for k in ('condition','noise')}
            for name,path in models.items():
                run.check();dest=folder/name;dest.mkdir(exist_ok=True)
                if (dest/'result.json').exists():
                    row=read(dest/'result.json');verify_artifacts(dest,row['artifacts'])
                else:
                    if model is None:
                        model=PersistentSeedVR2(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
                            dit_checkpoint=Path(assets['dit']),vae_checkpoint=Path(assets['vae']),
                            positive_embedding=Path(assets['positive']),negative_embedding=Path(assets['negative']),
                            lora_checkpoint=CURRENT,lora_strength=1.,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
                        metric=LPIPSAlex(True)
                    began=time.monotonic()
                    pixels,latent,observation=run_one(model,received,case,(name,name,1.,'all'),states)
                    q=quality(core(source,case),core(pixels,case),metric)
                    phase,folded=phase_diagnostic(core(pixels,case))
                    error,_=phase_diagnostic(core(pixels,case),core(source,case))
                    phase8,_=phase_diagnostic(core(pixels,case),core(source,case),period=8)
                    atomic_npz(dest/'output.npz',pixels=pixels,phase16=folded)
                    row=dict(name=name,quality=q,phase16=phase,error_phase16=error,error_phase8=phase8,
                        observation=observation,seconds=time.monotonic()-began,
                        output_path=str(dest/'output.npz'),provenance='new_decode',
                        artifacts={'output.npz':digest(dest/'output.npz')})
                    atomic_json(dest/'result.json',row)
                rows.append(row)
                run.update(phase='paired_checkpoint_evaluation',sample=case['sample_id'],arm=name)
                print(json.dumps(dict(sample=case['sample_id'],arm=name,quality=row['quality'])),flush=True)
            for row in rows:
                if {k:row['observation'][k] for k in expected}!=expected:
                    raise ValueError('input condition/noise differs')
                with np.load(row['output_path']) as z:pixels=core(z['pixels'],case)
                phase=temporal_phase(pixels,core(source,case))
                if not np.isclose(phase['stationary_rms'],row['error_phase16']['phase_rms_u8'],rtol=1e-5):
                    raise ValueError('stationary phase audit differs')
                temporal.append(dict(sample=case['sample_id'],group=case['evaluation_group'],
                    dataset=case['dataset'],name=row['name'],**phase))
            draw(folder,case,source,{r['name']:r for r in rows})
            item=dict(case=case,results=rows,paired_conditions_noise=True,
                      reference_inputs=str(old_folder/'inputs.npz'))
            atomic_json(folder/'summary.json',item);all_results.append(item)
        summary=dict(cases=all_results,aggregate=aggregate(all_results),scope=protocol['scope'],
                     production_model_changed=False)
        atomic_json(args.output/'summary.json',summary);trajectory(summary,args.output)
        atomic_json(args.output/'temporal_audit.json',dict(complete=True,rows=temporal))
        names=['protocol.json','summary.json','trajectory.png','temporal_audit.json']
        for item in all_results:
            sid=item['case']['sample_id'];names.extend([sid+'/summary.json',sid+'/comparison.png'])
            if item['case']['evaluation_group']=='known_artifact':
                names.extend([sid+'/comparison.mp4',sid+'/temporal_contact.png'])
        atomic_json(args.output/'complete.json',dict(complete=True,new_points=120,reused_points=135,
            logical_points=255,seconds=time.monotonic()-run.started,
            artifacts={n:digest(args.output/n) for n in names}))
        verify_evaluation(args.output);run.update(phase='complete');run.log_resources()
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
