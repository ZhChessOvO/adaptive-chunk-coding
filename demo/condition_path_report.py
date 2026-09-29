"""Audit and visualize the single-factor posterior experiment, CPU only."""
import json
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo.chunk_enhancement_experiment import read
from demo.condition_path_experiment import (ARMS, CONDITIONS, INTERFACE, PREVIOUS, OLD,
    destination, assert_pair, validate_point, point)
from demo.feature_condition_report import CLIPS, METRICS
from demo.conditioned_generation_evaluate import MODES
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import load_source, resources

LABELS = dict(rgb='Fixed RGB LoRA', zero='RGB + coverage', actual='Received features')


def report(root):
    summary = read(root/'summary.json')
    assert summary['complete'] and len(summary['results']) == 30 and len(summary['checks']) == 2
    refs = {r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
    lookup = {(r['sample_id'],r['arm'],r['condition'],r['prefix']):r for r in summary['results']}
    expected = {(sid,a,c,p) for sid in CLIPS for a in ARMS for c in CONDITIONS
                for p in (('full','none') if sid == next(iter(CLIPS)) else ('full',))}
    assert set(lookup) == expected and len(lookup) == 30
    hashes = {(a,c):summary['protocol']['profiles'][f'{a}_{c}'] for a in ARMS for c in CONDITIONS}
    for r in summary['results']+summary['checks']:
        dest = destination(root,r['sample_id'],r['arm'],r['condition'],r['prefix'],r['check'])
        assert read(dest/'result.json') == r
        validate_point(dest,r,refs[r['sample_id']])
        with patch('demo.condition_path_experiment.execute',side_effect=AssertionError('unexpected recomputation')):
            resumed = point(root,refs[r['sample_id']],r['arm'],r['condition'],r['prefix'],
                            {},hashes,None,None,None,r['check'])
            assert resumed == r
    noise_windows = 0
    for sid,a,c,p in expected:
        if c != 'sample': continue
        assert_pair(lookup[(sid,a,'sample',p)],lookup[(sid,a,'mean',p)])
        noise_windows += len(lookup[(sid,a,'sample',p)]['fresh_decode']['generation_runtime']['condition_windows'])
    # Noise/raw condition must also match across the three feature-interface arms.
    for sid in CLIPS:
        for c in CONDITIONS:
            reference = lookup[(sid,'rgb',c,'full')]['fresh_decode']['generation_runtime']['condition_windows']
            for a in ARMS[1:]:
                other = lookup[(sid,a,c,'full')]['fresh_decode']['generation_runtime']['condition_windows']
                for x,y in zip(reference,other,strict=True):
                    assert x['conditions'] == y['conditions'] and x['diffusion_noise'] == y['diffusion_noise']
    first = next(iter(CLIPS))
    for c in CONDITIONS:
        rgb = load_frames(destination(root,first,'rgb',c,'none')/'reconstruction.npz')
        for a in ARMS[1:]:
            np.testing.assert_array_equal(rgb,load_frames(destination(root,first,a,c,'none')/'reconstruction.npz'))
    np.testing.assert_array_equal(load_frames(destination(root,first,'actual','mean','full')/'reconstruction.npz'),
        load_frames(destination(root,first,'actual','mean','full','repeat')/'reconstruction.npz'))
    full = [r for r in summary['results'] if r['prefix'] == 'full']
    aggregates = {field:{domain:{a:{c:{m:mean(r[field][m] for r in full if
        r['arm']==a and r['condition']==c and (domain=='all' or r['dataset']==domain)) for m in METRICS}
        for c in CONDITIONS} for a in ARMS} for domain in ('all','REDS','UVG')}
        for field in ('roi_quality','quality')}
    changes = []
    for sid in CLIPS:
        output_change = {}
        for a in ARMS:
            sample = load_frames(destination(root,sid,a,'sample','full')/'reconstruction.npz')
            center = load_frames(destination(root,sid,a,'mean','full')/'reconstruction.npz')
            control,_,_,_ = fmt.parse((destination(root,sid,a,'mean','full')/'stream.acsg').read_bytes())
            values = (center.astype(np.float32)-sample)[fmt.weights(sample.shape,control)>0]
            output_change[a] = dict(generated_channel_mae_255=float(np.abs(values).mean()),
                                    generated_channel_changed_fraction=float((values!=0).mean()))
        changes.append(dict(sample_id=sid, sample_name=CLIPS[sid],
            mean_sample_output_change=output_change,
            mean_gain={a:lookup[(sid,a,'sample','full')]['roi_quality']['lpips_alex']-
                         lookup[(sid,a,'mean','full')]['roi_quality']['lpips_alex'] for a in ARMS},
            feature_gain_vs={c:{a:lookup[(sid,a,c,'full')]['roi_quality']['lpips_alex']-
                                 lookup[(sid,'actual',c,'full')]['roi_quality']['lpips_alex']
                               for a in ('rgb','zero')} for c in CONDITIONS}))
    heartbeats = [json.loads(l) for l in (root/'heartbeat.jsonl').read_text().splitlines()]
    digest = dict(aggregates=aggregates,changes=changes,elapsed_seconds=summary['elapsed_seconds'],
        report_code_sha256=file_hash(Path(__file__)),
        timing={a:{c:mean(r['fresh_decode']['seconds'] for r in full if r['arm']==a and r['condition']==c)
                   for c in CONDITIONS} for a in ARMS},
        peak_cuda_allocated_bytes=max(r['fresh_decode']['peak_cuda_allocated_bytes'] for r in full),
        sampled_gpu_peak_mib=max(int(r['gpu'].split(',')[2]) for r in heartbeats),resources=resources(),
        local_scope='Fixed first-17-frame regions, area weighted per clip then equal clip mean',
        whole_scope='All 17/17/33/17 frames, equal clip mean',
        role='Four reused development clips, one fixed noise seed schedule; no generalization claim')
    atomic_json(root/'audit.json',dict(complete=True,fresh_decodes=32,legacy_pixel_replays=15,
        resumed_points_without_execution=32,
        paired_noise_windows=noise_windows,paired_rng_and_noise_exact=True,
        raw_condition_and_noise_match_across_arms=True,real_bytes_exact=True,
        inner_payload_unchanged=True,source_free=True,outside_generate_exact=True,
        repeat_exact=True,no_E_branch_exact=True,G_off_without_generator_exact=True))
    atomic_json(root/'digest.json',digest)
    figures(root,lookup,refs,changes)
    print(json.dumps(dict(full_local=aggregates['roi_quality'],changes=changes),indent=2),flush=True)


def figures(root,lookup,refs,changes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(rgb='#0072B2',zero='#E69F00',actual='#009E73')
    fig,axes = plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    x = np.arange(4)
    for i,a in enumerate(ARMS):
        values = [r['mean_gain'][a] for r in changes]
        axes[0].bar(x+(i-1)*.23,values,.23,label=LABELS[a],color=colors[a])
    axes[0].set(title='VAE mean vs sample (all weights fixed)',ylabel='Local LPIPS improvement (sample - mean)')
    for i,c in enumerate(CONDITIONS):
        values = [r['feature_gain_vs'][c]['rgb'] for r in changes]
        axes[1].bar(x+(i-.5)*.32,values,.32,label=c,color=['#777777','#009E73'][i])
    axes[1].set(title='Feature interface vs RGB under the SAME condition',ylabel='Local LPIPS improvement (RGB - feature)')
    for ax in axes:
        ax.axhline(0,color='black',lw=.8); ax.set_xticks(x,list(CLIPS.values()),rotation=12)
        ax.grid(axis='y',alpha=.25); ax.legend(fontsize=8)
    fig.savefig(root/'paired_effects.png',dpi=150); plt.close(fig)
    old = read(INTERFACE/'evaluation/summary.json')['results']
    previous = read(PREVIOUS/'evaluation/summary.json')['results']
    for metric,label in [('lpips_alex','Local LPIPS (lower better)'),('psnr_db','Local PSNR (dB)')]:
        fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,sid in zip(axes.flat,CLIPS):
            for a in ARMS:
                baseline = [next(r for r in (previous if a=='rgb' else old)
                    if r['sample_id']==sid and r['candidate']==a and r['mode']==p) for p in MODES]
                ax.plot([r['bytes'] for r in baseline],[r['roi_quality'][metric] for r in baseline],
                        'o-',color=colors[a],label=LABELS[a]+' / sample')
                r = lookup[(sid,a,'mean','full')]
                ax.plot(r['bytes'],r['roi_quality'][metric],'*',ms=12,color=colors[a],
                        markeredgecolor='black',markeredgewidth=.4,label=LABELS[a]+' / mean, full E')
            ax.set(title=CLIPS[sid],xlabel='Actual stream bytes',ylabel=label)
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.suptitle('Lines: existing sample-condition prefixes; stars: NEW mean-condition full-E only',fontsize=10)
        fig.savefig(root/f'prefix_context_{metric}.png',dpi=150); plt.close(fig)
    fixed_visuals(root,lookup,refs)
    artifacts = {p.name:file_hash(p) for p in root.iterdir() if p.suffix in ('.png','.gif')}
    atomic_json(root/'figure_artifacts.json',artifacts)


def fixed_visuals(root,lookup,refs):
    """Also usable for already completed clips while the serial queue runs."""
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for sid,row in refs.items():
        videos = {'GT':load_source(row['sample']), 'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz')}
        for a in ARMS:
            for c in CONDITIONS:
                videos[f'{a} / {c}'] = load_frames(destination(root,sid,a,c,'full')/'reconstruction.npz')
        regions = row['metric_regions']; width = max(r[4] for r in regions)*2
        canvas = Image.new('RGB',(len(videos)*width,sum(r[5]*2+30 for r in regions)),'white')
        draw = ImageDraw.Draw(canvas); top = 0
        for _,_,x,y,w,h in regions:
            for i,(name,video) in enumerate(videos.items()):
                crop = Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                canvas.paste(crop,(i*width,top+30)); draw.text((i*width+3,top+6),name,font=font,fill='black')
            top += h*2+30
        canvas.save(root/f'{sid}_fixed.png')
        if sid in ('mechanism-00-reds','mechanism-01-uvg'):
            # Same already selected metric region; not selected by new scores.
            _,_,x,y,w,h = regions[-1] if sid=='mechanism-00-reds' else regions[0]
            names = ('GT','E only','actual / sample','actual / mean'); frames = []
            for t in range(17):
                frame = Image.new('RGB',(4*w*2,h*2+30),'white'); draw = ImageDraw.Draw(frame)
                for i,name in enumerate(names):
                    crop = Image.fromarray(videos[name][t,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                    frame.paste(crop,(i*w*2,30)); draw.text((i*w*2+3,6),name,font=font,fill='black')
                frames.append(frame)
            frames[0].save(root/f'{sid}_sequence.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
