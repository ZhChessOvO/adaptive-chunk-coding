"""No-recompute resume audit and existing-pixel analysis of coadaptation."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import joint_condition_evaluate as evaluation
from demo import joint_condition_pipeline as training
from demo import internal_condition_pipeline as receiver
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.joint_condition_train import DEFAULT
from demo.feature_condition_report import CLIPS, METRICS
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import resources, load_source


def forbidden(*args, **kwargs):
    raise AssertionError('Completed evaluation must not decode, recompute, or rewrite artifacts')


def resume_without_recompute(root, run):
    paths = [root/'training_audit.json', *[root/'evaluation'/f for f in
        ('summary.json','digest.json','audit.json','figure_artifacts.json','protocol.json')]]
    before = {str(p):file_hash(p) for p in paths}
    def unchanged_audit(path, value):
        assert path == root/'training_audit.json' and value == read(path)
    with patch.object(receiver,'execute',forbidden), \
         patch.object(receiver,'atomic_torch_save',forbidden), \
         patch.object(evaluation,'quality',forbidden), \
         patch.object(evaluation,'region_metrics',forbidden), \
         patch.object(evaluation,'LPIPSAlex',forbidden), \
         patch.object(evaluation,'atomic_json',forbidden), \
         patch.object(training,'atomic_json',unchanged_audit):
        evaluation.evaluate(root,run)
    assert before == {str(p):file_hash(p) for p in paths}
    evaluation.grouped(read(root/'evaluation/summary.json')['results'])
    verify_artifacts(root/'evaluation',read(root/'evaluation/figure_artifacts.json'))
    return dict(points=58,no_decode=True,no_metric_recompute=True,formal_files_unchanged=True,hashes=before)


def summarize(root):
    dest = root/'evaluation'
    lookup = evaluation.grouped(read(dest/'summary.json')['results'])
    digest = read(dest/'digest.json')
    rows = []
    for sid,name in CLIPS.items():
        point = dest/sid/'internal_full'
        actual = load_frames(point/'reconstruction.npz')
        control,*_ = fmt.parse((point/'stream.acsg').read_bytes())
        mask = fmt.weights(actual.shape,control) > 0
        changes = {}
        for arm in ('rgb','zero','off','without','shuffled'):
            other = load_frames(dest/sid/f'{arm}_full/reconstruction.npz')
            delta = np.abs(actual.astype(np.int16)-other.astype(np.int16))[mask]
            changes[arm] = dict(channel_mae_255=float(delta.mean()),changed_fraction=float((delta != 0).mean()))
        rows.append(dict(sample_id=sid,name=name,changes=changes,
            bytes=lookup[sid,'full','internal']['bytes'],
            full_local={k:lookup[sid,'full',k]['roi_quality'] for k in
                ('rgb','internal','zero','off','without','shuffled')},
            prefix_bytes={m:lookup[sid,m,'rgb']['bytes'] for m in evaluation.MODES},
            incremental_E={k:lookup[sid,'none',k]['roi_quality']['lpips_alex']-
                lookup[sid,'full',k]['roi_quality']['lpips_alex'] for k in evaluation.ARMS}))
    domains = {domain:{arm:{scope:{m:mean(lookup[sid,'full',arm][scope][m]
        for sid in CLIPS if domain == 'all' or lookup[sid,'full',arm]['dataset'] == domain)
        for m in METRICS} for scope in ('quality','roi_quality')}
        for arm in ('rgb','internal','zero','off','without','shuffled')}
        for domain in ('all','REDS','UVG')}
    actual,off,rgb = [domains['all'][k]['roi_quality'] for k in ('internal','off','rgb')]
    beats = [json.loads(line) for line in (root/'heartbeat.jsonl').read_text().splitlines()]
    formal = [b for b in beats if b['mode'] == 'evaluate']
    return dict(clips=rows,domains=domains,
        off_vs_rgb=dict(lpips_reduction=rgb['lpips_alex']-off['lpips_alex'],
            relative_percent=100*(1-off['lpips_alex']/rgb['lpips_alex']),
            psnr_change_db=off['psnr_db']-rgb['psnr_db'],
            temporal_change=off['temporal_delta_mae']-rgb['temporal_delta_mae']),
        off_vs_internal=dict(lpips_reduction=actual['lpips_alex']-off['lpips_alex'],
            relative_percent=100*(1-off['lpips_alex']/actual['lpips_alex']),
            psnr_change_db=off['psnr_db']-actual['psnr_db'],
            temporal_change=off['temporal_delta_mae']-actual['temporal_delta_mae']),
        formal_seconds=digest['evaluation_seconds'], first_heartbeat=formal[0]['utc'],
        last_heartbeat=formal[-1]['utc'],current_resources=resources(),
        ordinary_file_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()),
        role='Existing four development clips; branch-off is a post-hoc research candidate, not promoted')


def figures(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = root/'evaluation/supplement'
    dest.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    labels = {'rgb':'RGB equal steps','internal':'Real feature ON','off':'Same LoRA OFF',
              'without':'Same LoRA zero input','shuffled':'Same LoRA shuffled'}
    for ax,metric in zip(axes,METRICS):
        for arm in labels:
            values = [r['full_local'][arm][metric] for r in result['clips']]
            ax.plot(range(4),values,'o-',label=labels[arm])
        ax.set(xticks=range(4),xticklabels=list(CLIPS.values()),ylabel=metric,
               title='Same full-prefix bytes; lower better' if metric != 'psnr_db' else 'Same bytes; higher better')
        ax.tick_params(axis='x',labelrotation=25)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.savefig(dest/'off_tradeoff.png',dpi=160)
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    refs = read(evaluation.OLD/'summary.json')['results']
    for row in refs:
        sid = row['sample']['sample_id']
        videos = {'GT':load_source(row['sample']), **{labels[k]:load_frames(
            root/'evaluation'/sid/f'{k}_full/reconstruction.npz') for k in ('rgb','internal','off')}}
        regions = row['metric_regions']
        width = max(r[4] for r in regions)*2
        canvas = Image.new('RGB',(width*len(videos),sum(r[5]*2+36 for r in regions)),'white')
        draw = ImageDraw.Draw(canvas)
        top = 0
        for t,n,x,y,w,h in regions:
            for i,(label,video) in enumerate(videos.items()):
                crop = Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                canvas.paste(crop,(i*width,top+36))
                draw.text((i*width+3,top+8),label,font=font,fill='black')
            top += h*2+36
        canvas.save(dest/f'{sid}_off.png')
    atomic_json(dest/'artifacts.json',{p.name:file_hash(p) for p in dest.glob('*.png')})


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            resume = resume_without_recompute(run.root,run)
            result = dict(complete=True,resume=resume,**summarize(run.root),analysis_code=file_hash(Path(__file__)))
            figures(run.root,result)
            result['seconds'] = time.monotonic()-run.started
            atomic_json(run.root/'evaluation/analysis.json',result)
            atomic_json(run.root/'analysis.complete.json',dict(complete=True,seconds=result['seconds'],resume=resume))
            run.update(phase='complete',completed=58)
            print(json.dumps(result,indent=2),flush=True)
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--max-hours',type=float,default=1.)
    args = parser.parse_args()
    args.command = 'analysis'
    if not os.environ.get('TMUX'):
        parser.error('Run inside tmux')
    main(args)
