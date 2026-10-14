"""Fixed raw-G evaluation of repair milestones, without changing codec profiles.

Three previously viewed artifact locations and twelve internal validation
windows are reported separately. One deterministic existing G ROI per window
is evaluated, not whole-video RD or an independent benchmark test set.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.routervc_fullview_probe import read, immutable, digest, verify_artifacts
from demo.scalable_codec import atomic_json, atomic_npz
from demo.scalable_experiment import quality
from demo.stage_c_a800_teacher import PersistentSeedVR2
from demo.stage_c_three_path_roi_probe import LPIPSAlex
from routervc.latent.router_data import source as load_source
from routervc.generation.texture_data import PREVIOUS, CURRENT, REPO
from tools.texture_diagnostic import cases as artifact_cases, core, run_one, phase_diagnostic


def cases():
    result = [dict(c, evaluation_group='known_artifact') for c in artifact_cases()]
    previous = read(PREVIOUS/'protocol.json')
    index = {r['sample_id']:r for r in read(PREVIOUS/'complete.json')['samples']}
    for row in previous['rows']:
        if row['router_split'] != 'validation': continue
        folder = Path(index[row['sample_id']]['capture'])
        receipt = read(folder/'complete.json')
        verify_artifacts(folder, receipt['artifacts'])
        selection = read(folder/'selection.json')['indices']
        # Nearest existing G call to center, no source/quality-driven selection.
        region = min(selection, key=lambda i:((i//4-1.5)**2+(i%4-1.5)**2, i))
        report = read(folder/f'g{region:02d}/result.json')
        verify_artifacts(folder/f'g{region:02d}',report['artifacts'])
        result.append(dict(row, region=region, crop=report['crop'], core=report['core'],
            folder=str(folder), seed=report['runtime']['windows'][0]['runtime']['seed'],
            received_sha256=digest(folder/'received.npz'),
            previous_raw_sha256=digest(folder/f'g{region:02d}/raw.npz'),
            previous_observation=report['runtime']['condition_windows'][0],
            evaluation_group='internal_validation'))
    if len(result) != 15: raise ValueError('expected 3 known + 12 internal validation ROIs')
    return result


def draw(folder, case, source, received, results, final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['source','received','initial',f'control_{final:06d}',f'repair_{final:06d}']
    labels = ['Source','Received Y','Current LoRA',f'Control / {final}',f'Repair / {final}']
    values = {'source':core(source,case),'received':core(received,case)}
    for name in names[2:]:
        with np.load(folder/name/'output.npz') as z: values[name]=core(z['pixels'],case)
    fig, axes = plt.subplots(3,5,figsize=(15,8),layout='constrained')
    maps = {}
    for name in names:
        _,maps[name]=phase_diagnostic(values[name],values['source'])
    bound=max(1.,max(float(np.max(np.abs(m))) for m in maps.values()))
    for i,(name,label) in enumerate(zip(names,labels)):
        pixels=values[name]; h,w=pixels.shape[1:3]
        axes[0,i].imshow(pixels[8],interpolation='nearest')
        if name in results:
            q=results[name]['quality']; label+=f"\nLPIPS {q['lpips_alex']:.4f} | PSNR {q['psnr_db']:.2f}"
        axes[0,i].set_title(label,fontsize=10)
        ch,cw=min(h,96),min(w,96)
        axes[1,i].imshow(pixels[8,(h-ch)//2:(h+ch)//2,(w-cw)//2:(w+cw)//2],interpolation='nearest')
        axes[1,i].set_title('Same central detail crop',fontsize=9)
        axes[2,i].imshow(maps[name].mean(-1),cmap='coolwarm',vmin=-bound,vmax=bound,interpolation='nearest')
        axes[2,i].set_title(f"Error phase16 RMS {np.sqrt(np.mean(maps[name]**2)):.3f}",fontsize=9)
        for ax in axes[:,i]: ax.axis('off')
    fig.suptitle(case['sample_id']+f" | region {case['region']} | BEFORE fusion | frame 9",fontsize=11)
    fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    # Lossless decoded pixels are retained in npz; MP4 is a convenient, explicitly lossy preview.
    from PIL import Image,ImageDraw
    height,width=values['source'].shape[1:3]
    canvas_w=5*width*2; canvas_h=height*2+32
    temporary=folder/'comparison.partial.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{canvas_w}x{canvas_h}','-r','8','-i','pipe:0','-an','-c:v','libx264',
             '-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for t in range(17):
            canvas=Image.new('RGB',(canvas_w,canvas_h),'white'); painter=ImageDraw.Draw(canvas)
            for i,(name,label) in enumerate(zip(names,labels)):
                frame=Image.fromarray(values[name][t]).resize((width*2,height*2),Image.Resampling.NEAREST)
                canvas.paste(frame,(i*width*2,32));painter.text((i*width*2+5,8),label,fill='black')
            child.stdin.write(np.asarray(canvas).tobytes())
        child.stdin.close()
        if child.wait(): raise RuntimeError('video preview encoding failed')
        os.replace(temporary,folder/'comparison.mp4')
    finally:
        if child.poll() is None: child.terminate();child.wait()


def aggregate(all_results):
    buckets=defaultdict(list)
    for item in all_results:
        c=item['case']
        for r in item['results']:
            buckets[(c['evaluation_group'],c['dataset'],r['name'])].append(r)
    result=[]
    for (group,dataset,name),rows in sorted(buckets.items()):
        result.append(dict(group=group,dataset=dataset,name=name,n=len(rows),
            lpips=float(np.mean([r['quality']['lpips_alex'] for r in rows])),
            psnr=float(np.mean([r['quality']['psnr_db'] for r in rows])),
            temporal_mae=float(np.mean([r['quality']['temporal_delta_mae'] for r in rows])),
            phase16_error=float(np.mean([r['error_phase16']['phase_rms_u8'] for r in rows])),
            phase16_output=float(np.mean([r['phase16']['phase_rms_u8'] for r in rows]))))
    return result


def draw_trajectory(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=summary['aggregate']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row_index,dataset in enumerate(('REDS','UVG')):
        selected={r['name']:r for r in rows if r['group']=='internal_validation' and r['dataset']==dataset}
        for column,(key,title) in enumerate((('phase16_error','Stationary error / lower'),
                                             ('lpips','LPIPS / lower'),('psnr','PSNR / higher'))):
            ax=axes[row_index,column]
            for arm,color in (('control','#64748b'),('repair','#c45a26')):
                points=sorted((int(n.split('_')[-1]),r[key]) for n,r in selected.items() if n.startswith(arm+'_'))
                x=[0]+[p[0] for p in points];y=[selected['initial'][key]]+[p[1] for p in points]
                ax.plot(x,y,'o-',label=arm,color=color)
            ax.set_title(f'{dataset} ({selected["initial"]["n"]} fixed ROIs) | {title}')
            ax.set_xlabel('Additional G-only training steps');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Internal validation / raw G before fusion / NOT whole-video RD')
    fig.savefig(output/'trajectory.png',dpi=160);plt.close(fig)


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    from tools.texture_repair import RepairRun
    run=RepairRun(SimpleNamespace(output=args.output,command='evaluate',max_hours=3));run.thread.start()
    # Parent repair queue already owns the exclusive GPU mutex.
    model=None; all_results=[]
    try:
        training=read(args.root/'formal/protocol.json'); total=training['steps']
        chosen=cases()
        models=[('initial',CURRENT)]
        for arm in ('control','repair'):
            for step in range(100,total+1,100):
                models.append((f'{arm}_{step:06d}',args.root/'formal'/arm/'checkpoints'/f'adapter_{step:06d}.pt'))
            if total % 100: models.append((f'{arm}_{total:06d}',args.root/'formal'/arm/'adapter.pt'))
        assets={k:v['path'] for k,v in training['assets'].items()}
        protocol=dict(version=1,cases=chosen,models={n:dict(path=str(p),sha256=digest(p)) for n,p in models},
            code=digest(Path(__file__)),helper_code=digest(REPO/'tools/texture_diagnostic.py'),
            training_protocol=digest(args.root/'formal/protocol.json'),
            scope='one fixed raw G ROI/window, not full-system RD; no automatic promotion')
        immutable(args.output/'protocol.json',protocol)
        states={n:torch.load(p,weights_only=True,map_location='cpu')['state_dict'] for n,p in models}
        for case in chosen:
            folder=args.output/case['sample_id'];folder.mkdir(exist_ok=True)
            x,y,w,h=case['crop']; old=Path(case['folder'])
            if digest(old/'received.npz') != case['received_sha256']: raise ValueError('received input changed')
            with np.load(old/'received.npz') as z: received=z['enhanced'][:,y:y+h,x:x+w].copy()
            if 'source_path' in case:
                if digest(case['source_path']) != case['source_sha256']: raise ValueError('source changed')
                with np.load(case['source_path']) as z: source=z['source'][:,y:y+h,x:x+w].copy()
            else: source=load_source(case)[:,y:y+h,x:x+w].copy()
            atomic_npz(folder/'inputs.npz',source=source,received=received)
            rows=[];expected=None
            for name,path in models:
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
                    if name=='initial':
                        with np.load(old/f"g{case['region']:02d}/raw.npz") as z: np.testing.assert_array_equal(pixels,z['pixels'])
                        previous=case['previous_observation']
                        assert observation['condition']==previous['conditions'][0]
                        assert observation['noise']==previous['diffusion_noise']
                    q=quality(core(source,case),core(pixels,case),metric)
                    phase,folded=phase_diagnostic(core(pixels,case))
                    error,_=phase_diagnostic(core(pixels,case),core(source,case))
                    phase8,_=phase_diagnostic(core(pixels,case),core(source,case),period=8)
                    atomic_npz(dest/'output.npz',pixels=pixels,phase16=folded)
                    row=dict(name=name,quality=q,phase16=phase,error_phase16=error,error_phase8=phase8,
                        observation=observation,initial_replay_exact=True if name=='initial' else None,
                        seconds=time.monotonic()-began,artifacts={'output.npz':digest(dest/'output.npz')})
                    atomic_json(dest/'result.json',row)
                pair={k:row['observation'][k] for k in ('condition','noise')}
                if expected is None: expected=pair
                if pair != expected: raise ValueError('evaluation conditions/noise not paired')
                rows.append(row)
                run.update(phase='paired_checkpoint_evaluation',sample=case['sample_id'],arm=name)
                print(json.dumps(dict(sample=case['sample_id'],arm=name,quality=row['quality'],phase=row['error_phase16'])),flush=True)
            draw(folder,case,source,received,{r['name']:r for r in rows},total)
            summary=dict(case=case,results=rows,paired_conditions_noise=True)
            atomic_json(folder/'summary.json',summary);all_results.append(summary)
        summary=dict(cases=all_results,aggregate=aggregate(all_results),
                     scope=protocol['scope'],production_model_changed=False)
        atomic_json(args.output/'summary.json',summary)
        draw_trajectory(summary,args.output)
        names=['protocol.json','summary.json','trajectory.png']+[f"{c['sample_id']}/{n}" for c in chosen
            for n in ('comparison.png','comparison.mp4','summary.json','inputs.npz')]
        atomic_json(args.output/'complete.json',dict(complete=True,points=len(chosen)*len(models),
            artifacts={n:digest(args.output/n) for n in names},seconds=time.monotonic()-run.started))
        run.update(phase='complete');run.log_resources()
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        run.stop.set();run.thread.join();run.lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
