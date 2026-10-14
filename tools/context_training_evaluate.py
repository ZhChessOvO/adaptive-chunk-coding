"""Post-fit native-FOV review with authenticated frozen references and images."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from demo.chunk_enhancement_codec import configure_torch
from demo.routervc_fullview_probe import read,digest,immutable,verify_artifacts
from demo.scalable_codec import atomic_json,atomic_npz
from demo.scalable_experiment import quality,now
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_seedvr2_lora_utils import load_lora_state_dict
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.generation.context_geometry import rgb_crop,local_core,master_noise,noise_crop
from routervc.generation.texture_data import REPO
from tools.texture_context_probe import DEFAULT,full_inputs,frozen_restore
from tools.texture_context_report import plotting
from tools.texture_diagnostic import phase_diagnostic
from tools.texture_repair_review import temporal_phase
from tools.texture_clean import CleanRun


def verify(root):
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    config=read(root/'protocol.json')
    for spec in config['models'].values():
        if digest(spec['path'])!=spec['sha256']:raise ValueError('model changed')
    for item in read(root/'summary.json')['cases']:
        for row in item['results']:
            if row.get('output_path') and digest(row['output_path'])!=row['output_sha256']:
                raise ValueError('output changed')
    return done


def figure(folder,item,source,received):
    plt=plotting();rows={(r['name'],r['size']):r for r in item['results']};geo=item['geometry']
    fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
    for i,size in enumerate((256,512)):
        for j,name in enumerate(('source','initial','local','mixed')):
            ax=axes[i,j]
            if name=='source':pixels=rgb_crop(source if i==0 else received,geo['core']);title='Source' if i==0 else 'Received Y'
            else:
                row=rows[name,size]
                with np.load(row['output_path']) as z:pixels=local_core(z['pixels'],row['crop'],geo['core'])
                title=f'{name} / {size} context\nLPIPS {row["quality"]["lpips_alex"]:.4f} | phase {row["phase_error"]["phase_rms_u8"]:.3f}'
            ax.imshow(pixels[8],interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(item['case']['sample_id']+'\nSame 128 core, source/received orientation is in the frozen-context page')
    fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
    for ax,name in zip(axes,('source','initial','local','mixed')):
        if name=='source':pixels=rgb_crop(source,geo['large'])
        else:
            with np.load(rows[name,512]['output_path']) as z:pixels=z['pixels'].copy()
        ax.imshow(pixels[8],interpolation='nearest');ax.axis('off');ax.set_title(name+' / whole 512 view')
    fig.suptitle(item['case']['sample_id']+' / raw G before fusion; not whole-system RD')
    fig.savefig(folder/'wide_comparison.png',dpi=130);plt.close(fig)


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    output=args.root/('smoke/evaluation' if args.smoke else 'evaluation')
    if (output/'complete.json').exists():print(json.dumps(verify(output)),flush=True);return
    configure_torch();torch.use_deterministic_algorithms(True)
    run=CleanRun(SimpleNamespace(output=output,command='context_fit_review',max_hours=3));run.thread.start();run.log_resources()
    try:
        stage=args.root/('smoke' if args.smoke else 'formal');train=read(stage/'protocol.json')
        for name,value in train['code'].items():
            if digest(REPO/name)!=value:raise ValueError('bound training/eval source changed: '+name)
        paths={a:stage/('direct' if args.smoke else '')/a/'adapter.pt' for a in ('local','mixed')}
        probe=DEFAULT/'probe';old=read(probe/'summary.json');old_cfg=read(probe/'protocol.json')
        cases=old['cases']
        if args.smoke:cases=[cases[0],cases[2]]
        config=dict(training=digest(stage/'protocol.json'),probe_summary=digest(probe/'summary.json'),
            smoke=args.smoke,models={n:dict(path=str(p),sha256=digest(p)) for n,p in paths.items()},
            cases=[x['case']['sample_id'] for x in cases])
        immutable(output/'protocol.json',config)
        states={n:torch.load(p,map_location='cpu',weights_only=True)['state_dict'] for n,p in paths.items()}
        assets={k:Path(v['path']) for k,v in train['assets'].items()}
        model=PersistentSeedVR2(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
            dit_checkpoint=assets['dit'],vae_checkpoint=assets['vae'],positive_embedding=assets['positive'],negative_embedding=assets['negative'],
            lora_checkpoint=paths['local'],lora_strength=1.,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
        metric=LPIPSAlex(True);all_items=[];n=0
        for item in cases:
            case=item['case'];geo=item['geometry'];source,received=full_inputs(case)
            folder=output/case['sample_id'];folder.mkdir(exist_ok=True);rows=[]
            reference={r['size']:r for r in item['results'] if r['name']=='phase2000' and r['seed_index']==0}
            for size,r in reference.items():
                path=probe/r['relative_dir']/'output.npz'
                if digest(path)!=r['artifacts']['output.npz']:raise ValueError('frozen reference changed')
                rows.append(dict(name='initial',size=size,crop=r['binding']['crop'],quality=r['quality'],phase_error=r['phase_error'],
                    output_path=str(path),output_sha256=digest(path),provenance='authenticated_frozen_reference'))
            seed=reference[256]['binding']['seed'];master=master_noise(seed)
            for name,state in states.items():
                load_lora_state_dict(model.runner.dit,state)
                for size,label in ((256,'small'),(512,'large')):
                    run.check();crop=geo[label];dest=folder/f'{name}_{size}';dest.mkdir(exist_ok=True)
                    binding=dict(protocol=digest(output/'protocol.json'),sample=case['sample_id'],name=name,size=size,seed=seed)
                    if (dest/'result.json').exists():
                        row=read(dest/'result.json')
                        if row['binding']!=binding or digest(row['output_path'])!=row['output_sha256']:raise ValueError('point changed')
                    else:
                        noise=noise_crop(master,crop,geo['large'])
                        pixels,obs=frozen_restore(model,rgb_crop(received,crop),noise,seed)
                        if any(obs[k]!=reference[size]['observation'][k] for k in ('condition','noise')):
                            raise ValueError('post-fit input pairing differs')
                        repeat=False
                        if args.smoke:
                            # Actually swap away and reload the trained adapter before repeating.
                            load_lora_state_dict(model.runner.dit,states['mixed' if name=='local' else 'local'])
                            load_lora_state_dict(model.runner.dit,state)
                            again,again_obs=frozen_restore(model,rgb_crop(received,crop),noise,seed)
                            np.testing.assert_array_equal(pixels,again)
                            if again_obs['condition']!=obs['condition']:raise ValueError('reload condition mismatch')
                            repeat=True
                        target=rgb_crop(source,geo['core']);value=local_core(pixels,crop,geo['core'])
                        phase,_=phase_diagnostic(value,target);path=dest/'output.npz';atomic_npz(path,pixels=pixels)
                        row=dict(name=name,size=size,crop=crop,binding=binding,quality=quality(target,value,metric),
                            phase_error=phase,temporal_phase=temporal_phase(value,target),observation=obs,
                            output_path=str(path),output_sha256=digest(path),reload_repeat_exact=repeat,provenance='new_decode')
                        atomic_json(dest/'result.json',row)
                    rows.append(row);n+=1;run.update(phase='postfit_evaluation',completed=n,total=len(cases)*4)
            result=dict(case=case,geometry=geo,results=rows);figure(folder,result,source,received);all_items.append(result)
        groups=defaultdict(list)
        for item in all_items:
            for row in item['results']:groups[(item['case']['evaluation_group'],item['case']['dataset'],row['name'],row['size'])].append(row)
        aggregate=[dict(group=g,dataset=d,name=n,size=s,count=len(values),
            lpips=float(np.mean([r['quality']['lpips_alex'] for r in values])),
            psnr=float(np.mean([r['quality']['psnr_db'] for r in values])),
            phase=float(np.mean([r['phase_error']['phase_rms_u8'] for r in values])))
            for (g,d,n,s),values in sorted(groups.items())]
        atomic_json(output/'summary.json',dict(cases=all_items,aggregate=aggregate,scope='same 128 core; no RD claim'))
        names=['protocol.json','summary.json']+[str(p.relative_to(output)) for p in output.rglob('*.png')]
        atomic_json(output/'complete.json',dict(complete=True,new_points=n,reused_initial_points=len(cases)*2,
            smoke=args.smoke,artifacts={p:digest(output/p) for p in names},utc=now()))
        verify(output);run.update(phase='complete')
    except BaseException as error:
        atomic_json(output/'last_failure.json',dict(error=repr(error),utc=now()));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join();run.lock.close()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT/'training')
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
