"""Post-run audit and observational numerics replay; never changes old profiles."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.feature_interface_train import DEFAULT
from demo.feature_interface_evaluate import audit_training
from demo.feature_interface_report import report, paired_clips, CLIPS, MODES
from demo.feature_interface_model import FeatureInterface, condition_statistics
from demo import feature_interface_decode as receiver
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import load_source, resources
from demo.conditioned_generation_evaluate import OLD


def rounding_comparison(raw,side,coverage,training_side):
    """Compare cast locations, not an estimate of image-quality causality."""
    inference = (raw+side).to(torch.bfloat16)
    training = raw.to(torch.bfloat16)+training_side
    stats = condition_statistics(raw.to(torch.bfloat16),inference,side.to(torch.bfloat16),coverage)
    stats.update(raw_dtype=str(raw.dtype),train_style_difference_rms=float(
        (inference.float()-training.float()).square().mean().sqrt()),
        train_style_different_fraction=float((inference!=training).float().mean()))
    return stats


def worker(args):
    from demo import stage_c_a800_teacher as teacher
    forward = FeatureInterface.forward; records=[]; projections=[]
    @torch.no_grad()
    def observe(net,raw,packets,**kw):
        side,coverage = forward(net,raw,packets,**kw)
        training_side,_ = forward(net,raw.to(torch.bfloat16),packets,**kw)
        records.append(rounding_comparison(raw,side,coverage,training_side))
        return side,coverage  # Exactly the original result; observation only.
    Original = teacher.PersistentSeedVR2
    class Observed(Original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            def hook(module,inputs,output):
                projections.append(dict(input_dtype=str(inputs[0].dtype),
                    weight_dtype=str(module.weight.dtype),output_dtype=str(output.dtype),
                    cuda_autocast=torch.is_autocast_enabled('cuda')))
            self.runner.dit.vid_in.proj.register_forward_hook(hook)
    with patch.object(FeatureInterface,'forward',observe),patch.object(teacher,'PersistentSeedVR2',Observed):
        receiver.decode(args)
    atomic_json(args.output/'numerics.json',dict(windows=records,projections=projections,
        observational=True,code=file_hash(Path(__file__))))


def summarize(root,probes):
    evaluation = root/'evaluation'; summary=read(evaluation/'summary.json')
    clips=paired_clips(summary['results']); ablations={(r['sample_id'],r['candidate']):r for r in summary['ablations']}
    details=[]
    for c in clips:
        sid=c['sample_id']; full=c['prefixes']['full']
        on=load_frames(evaluation/sid/'actual_full/reconstruction.npz')
        control,*_=fmt.parse((evaluation/sid/'actual_full/stream.acsg').read_bytes())
        mask=fmt.weights(on.shape,control)>0
        changes={}
        for name in ('branch_off','without_content','shuffled','zero'):
            off=load_frames(evaluation/sid/f'{name}_full/reconstruction.npz')
            delta=np.abs(on.astype(np.int16)-off.astype(np.int16))[mask]
            other=full['zero'] if name=='zero' else ablations[(sid,name)]
            changes[name]=dict(generated_channel_mae_255=float(delta.mean()),
                changed_fraction=float((delta!=0).mean()),
                local_lpips_gain=other['roi_quality']['lpips_alex']-full['actual']['roi_quality']['lpips_alex'])
        details.append(dict(sample_id=sid,name=c['name'],bytes=full['actual']['bytes'],changes=changes,
            incremental_E={k:c['prefixes']['none'][k]['roi_quality']['lpips_alex']-
                full[k]['roi_quality']['lpips_alex'] for k in ('rgb','feature','zero','actual')}))
    training={}
    for name in ('actual','zero'):
        rows=[json.loads(line) for line in (root/name/'steps.jsonl').read_text().splitlines()]
        training[name]=dict(first_utc=rows[0]['utc'],last_utc=rows[-1]['utc'],
            last100={k:mean(r['condition_statistics'][k] for r in rows[-100:]) for k in
                     ('side_rms','effective_rms','covered_changed_fraction','rounded_away_fraction')})
    heartbeats=[json.loads(line) for line in (root/'heartbeat.jsonl').read_text().splitlines()]
    # Keep only original formal-run measurements, not later reporting snapshots.
    original=[r for r in heartbeats if r['mode']=='run']
    data=dict(complete=True,training=training,clips=details,numerics=probes,
        queue_seconds=read(root/'run.complete.json')['elapsed_seconds'],
        formal_start=original[0]['utc'],formal_end=original[-1]['utc'],
        evaluation_gpu_sampled_peak_mib=max(int(r['gpu'].split(',')[2]) for r in original
            if str(r.get('phase','')).startswith('mechanism-')),
        current_resources=resources())
    atomic_json(evaluation/'analysis.json',data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    x=np.arange(4)
    for i,key in enumerate(('rgb','feature','zero','actual')):
        axes[0].bar(x+(i-1.5)*.2,[r['incremental_E'][key] for r in details],width=.2,label=key)
    axes[0].set(title='Extra E benefit: LPIPS(no E) - LPIPS(full E)',xticks=x,xticklabels=list(CLIPS.values()))
    axes[0].legend();axes[0].grid(axis='y',alpha=.3)
    for i,key in enumerate(('branch_off','without_content','shuffled')):
        axes[1].bar(x+(i-1)*.25,[r['changes'][key]['generated_channel_mae_255'] for r in details],width=.25,label=key)
    axes[1].set(title='Actual output vs controls: generated-region MAE',ylabel='RGB channel levels / 255',
                xticks=x,xticklabels=list(CLIPS.values()))
    axes[1].legend();axes[1].grid(axis='y',alpha=.3)
    fig.savefig(evaluation/'mechanism_summary.png',dpi=150);plt.close(fig)
    refs={r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
    for sid,region_index,name in [('mechanism-00-reds',-1,'wall_sequence'),('mechanism-02-reds',0,'long33_sequence')]:
        row=refs[sid];t,n,x,y,w,h=row['metric_regions'][region_index]
        videos={'GT':load_source(row['sample']),
            'Frozen RGB':load_frames(PREVIOUS/'evaluation'/sid/'rgb_full/reconstruction.npz'),
            'RGB + coverage':load_frames(evaluation/sid/'zero_full/reconstruction.npz'),
            'Received feature':load_frames(evaluation/sid/'actual_full/reconstruction.npz')}
        frames=[]
        for f in range(len(videos['GT'])):
            canvas=Image.new('RGB',(len(videos)*w*2,h*2+30),'white');draw=ImageDraw.Draw(canvas)
            for i,(label,video) in enumerate(videos.items()):
                crop=Image.fromarray(video[f,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                canvas.paste(crop,(i*w*2,30));draw.text((i*w*2+3,6),label,fill='black',font=font)
            frames.append(canvas)
        frames[0].save(evaluation/f'{name}.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
    print(json.dumps(data,indent=2))


def main(args):
    run=Run(args);run.thread.start();started=time.monotonic()
    try:
        with exclusive_native_evaluation(run):
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            protocol=read(args.output/'queue_protocol.json')
            for name,sha in protocol['code'].items(): assert file_hash(REPO/'demo'/name)==sha
            audit_training(args.output); report(args.output)
            probes={}
            for sid in CLIPS:
                run.check();source=args.output/'evaluation'/sid/'actual_full';dest=args.output/'analysis'/sid
                result=dest/'complete.json'
                config=dict(code=file_hash(Path(__file__)),reference=file_hash(source/'result.json'),
                            adapter=file_hash(args.output/'actual/adapter.pt'))
                if result.exists():
                    item=read(result);assert item['config']==config;verify_artifacts(dest,item['artifacts'])
                else:
                    dest.mkdir(parents=True,exist_ok=True)
                    execute(run,sid,'feature_interface_analysis.py',['--worker','--stream',source/'stream.acsg',
                        '--output',dest,'--adapter',args.output/'actual/adapter.pt'],distributed=True)
                    np.testing.assert_array_equal(load_frames(dest/'reconstruction.npz'),load_frames(source/'reconstruction.npz'))
                    item=dict(config=config,pixels_exact=True,numerics=read(dest/'numerics.json'),
                        artifacts={p.name:file_hash(p) for p in (dest/'reconstruction.npz',dest/'decode.json',dest/'numerics.json')})
                    atomic_json(result,item)
                probes[sid]=item['numerics'];run.update(completed=len(probes),total=4)
            summarize(args.output,probes)
            atomic_json(args.output/'analysis.complete.json',dict(complete=True,seconds=time.monotonic()-started,
                observational_replays=4,pixels_exact=True,code=file_hash(Path(__file__)),resources=resources()))
            run.update(phase='complete')
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--stream',type=Path);p.add_argument('--adapter',type=Path)
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--max-hours',type=float,default=1.)
    args=p.parse_args();args.disable_generation=False;args.command='analysis'
    if args.worker: worker(args)
    else:
        if not os.environ.get('TMUX'): p.error('run inside tmux')
        main(args)
