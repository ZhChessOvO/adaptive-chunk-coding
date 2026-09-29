"""CPU audit, fair paired summaries and fixed visuals (no model inference)."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.conditioned_generation_evaluate import OLD, MODES
from demo.feature_condition_cache import PREVIOUS
from demo.feature_condition_pipeline import DEFAULT
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, resources

METRICS = ('lpips_alex','psnr_db','temporal_delta_mae')
CLIPS = {'mechanism-00-reds':'REDS000','mechanism-01-uvg':'UVG Beauty',
         'mechanism-02-reds':'REDS001','mechanism-03-uvg':'UVG Jockey'}
LABELS = {'direct':'No G: Base / partial E / full E','previous':'Previous image model',
          'rgb':'RGB continuation','feature':'RGB + feature',
          'feature_off':'Feature-trained LoRA, branch off'}


def aggregate(points, field):
    groups = {}
    for mode in MODES:
        groups[mode] = {}
        for domain in ('all','REDS','UVG'):
            rows = [r for r in points if r['mode'] == mode and
                    (domain == 'all' or r['dataset'] == domain)]
            groups[mode][domain] = {}
            for key in ('direct','previous','rgb','feature'):
                selected = [r for r in rows if r['candidate'] ==
                            ('feature' if key in ('direct','previous') else key)]
                expected = 4 if domain == 'all' else 2
                if len(selected) != expected or len({r['sample_id'] for r in selected}) != expected:
                    raise ValueError('missing or duplicate paired clip/prefix/candidate')
                if key in ('direct','previous'):
                    selected = [r[key] for r in selected]
                groups[mode][domain][key] = {m:mean(r[field][m] for r in selected) for m in METRICS}
    return groups


def report(root):
    evaluation = root/'evaluation'
    summary = read(evaluation/'summary.json')
    assert summary['complete'] and len(summary['results']) == 24
    references = read(OLD/'summary.json')['results']
    rows = {r['sample']['sample_id']:r for r in references}
    details = []
    for path in sorted(evaluation.glob('*/*/result.json')):
        result = read(path); dest = path.parent
        verify_artifacts(dest,result['artifacts'])
        control,inner,parsed,overhead = fmt.parse((dest/'stream.acsg').read_bytes())
        old_name,direct_name = MODES[result['mode']]
        sid = result['sample_id']
        old_control,old_inner,_,old_overhead = fmt.parse((OLD/sid/f'{old_name}.acsg').read_bytes())
        assert inner == old_inner and overhead == old_overhead
        for key in ('generate','protect','context','feather','seed','processing_scale','blend','window','stride'):
            assert control[key] == old_control[key]
        d = read(dest/'decode.json')
        assert d == result['fresh_decode']
        assert not d['source_frames_read'] and d['feature_bytes_added'] == 0
        assert d['total_bytes'] == result['bytes'] == (dest/'stream.acsg').stat().st_size
        assert sum(d[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                                 'incomplete_tail_bytes','generation_control_bytes')) == d['total_bytes']
        assert d['feature_packet_ids'] == [p.meta['packet_id'] for p in parsed.packets if p.meta['start'] > 0]
        actual = load_frames(dest/'reconstruction.npz')
        enhanced = load_frames(OLD/sid/direct_name/'reconstruction.npz')
        base = load_frames(OLD/sid/'base/reconstruction.npz')
        assert d['output_hash'] == frame_hash(actual)
        assert d['generation_input_hash'] == frame_hash(enhanced)
        assert d['base_hash'] == frame_hash(base)
        alpha = fmt.weights(actual.shape,control)
        np.testing.assert_array_equal(actual[alpha == 0],enhanced[alpha == 0])
        if result['candidate'] == 'off':
            np.testing.assert_array_equal(actual,enhanced)
            assert not d['generation_assets_validated']
        else:
            assert d['generation_executed'] and d['assets']['lora'] == control['lora']
            if result['mode'] == 'none' or result['candidate'] in ('rgb','feature_off'):
                assert all(w['feature_side_rms'] == 0 for w in d['generation_runtime']['windows'])
        details.append(dict(path=str(dest),bytes=d['total_bytes'],source_free=True,
            unchanged_payload=True,unchanged_non_generate=True,hashes=True))
    assert len(details) == 31
    first = references[0]['sample']['sample_id']
    for a,b in [('feature_full','repeat_full'),('feature_none','feature_off_none')]:
        np.testing.assert_array_equal(load_frames(evaluation/first/a/'reconstruction.npz'),
                                      load_frames(evaluation/first/b/'reconstruction.npz'))
    points = summary['results']
    local,whole = [aggregate(points,f) for f in ('roi_quality','quality')]
    by_point = {(r['sample_id'],r['mode'],r['candidate']):r for r in points}
    off = {r['sample_id']:r for r in summary['ablations']}
    clips,conditional,ablation = [],[],[]
    for sid,name in CLIPS.items():
        prefixes = {}
        for mode in MODES:
            feature = by_point[(sid,mode,'feature')]
            prefixes[mode] = dict(feature=feature,rgb=by_point[(sid,mode,'rgb')],
                                  previous=feature['previous'],direct=feature['direct'])
        clips.append(dict(sample_id=sid,name=name,prefixes=prefixes))
        gains = {}
        for key in ('previous','rgb','feature'):
            a,b = prefixes['none'][key],prefixes['full'][key]
            gains[key] = dict(lpips_reduction=a['roi_quality']['lpips_alex']-b['roi_quality']['lpips_alex'],
                psnr_increase=b['roi_quality']['psnr_db']-a['roi_quality']['psnr_db'],extra_bytes=b['bytes']-a['bytes'])
        conditional.append(dict(sample_id=sid,name=name,gains=gains))
        full = prefixes['full']
        on_dir,off_dir = [evaluation/sid/f'{k}_full' for k in ('feature','feature_off')]
        on_pixels,off_pixels = [load_frames(p/'reconstruction.npz') for p in (on_dir,off_dir)]
        control,_,_,_ = fmt.parse((on_dir/'stream.acsg').read_bytes())
        generated = fmt.weights(on_pixels.shape,control) > 0
        difference = np.abs(on_pixels.astype(np.int16)-off_pixels.astype(np.int16))[generated]
        windows = full['feature']['fresh_decode']['generation_runtime']['windows']
        ablation.append(dict(sample_id=sid,name=name,off=off[sid]['roi_quality'],
            on=full['feature']['roi_quality'],rgb=full['rgb']['roi_quality'],
            lpips_gain_from_branch=off[sid]['roi_quality']['lpips_alex']-full['feature']['roi_quality']['lpips_alex'],
            psnr_gain_from_branch=full['feature']['roi_quality']['psnr_db']-off[sid]['roi_quality']['psnr_db'],
            generated_pixel_channel_mae_255=float(difference.mean()),
            generated_pixel_channel_changed_fraction=float((difference != 0).mean()),
            mean_window_feature_side_rms=mean(w['feature_side_rms'] for w in windows)))
    timing = {}
    for key in ('rgb','feature'):
        selected = [r for r in points if r['candidate'] == key and r['mode'] == 'full']
        timing[key] = dict(receiver_seconds=mean(r['fresh_decode']['seconds'] for r in selected),
            generation_seconds=mean(r['fresh_decode']['generation_runtime']['seconds_model_load_excluded'] for r in selected),
            process_with_metrics_seconds=mean(r['process_wall_seconds'] for r in selected),
            peak_cuda_allocated_bytes=max(r['fresh_decode']['peak_cuda_allocated_bytes'] for r in selected))
    comparisons = {}
    for comparator in ('rgb','previous'):
        differences = [r['roi_quality']['lpips_alex']-
            (by_point[(r['sample_id'],r['mode'],'rgb')] if comparator == 'rgb' else r['previous'])['roi_quality']['lpips_alex']
            for r in points if r['candidate'] == 'feature']
        comparisons[comparator] = dict(lpips_improved_points=sum(v < 0 for v in differences),
            total=12,mean_local_lpips_change=mean(differences),
            full_prefix_relative_lpips_reduction=1-local['full']['all']['feature']['lpips_alex']/
                local['full']['all'][comparator]['lpips_alex'])
    heartbeats = [json.loads(l) for l in (root/'heartbeat.jsonl').read_text().splitlines()]
    sampled = {mode:max(int(r['gpu'].split(',')[2]) for r in heartbeats if r['mode'] == mode)
               for mode in ('train','feature_evaluation')}
    digest = dict(local=local,whole_frame=whole,clips=clips,conditional=conditional,ablation=ablation,
        comparisons=comparisons,timing=timing,training=summary['training'],
        sampled_gpu_peak_mib=sampled,evaluation_seconds=summary['elapsed_seconds'],
        resources=resources(),local_scope='Fixed ROI, first 17 frames, area weighted within clip; equal clip mean',
        whole_frame_scope='All frames (17,17,33,17), equal clip mean',
        role='Four already-used development clips, not independent evaluation')
    atomic_json(evaluation/'audit.json',dict(complete=True,fresh_decodes=31,details=details,
        repeat_exact=True,no_E_branch_off_exact=True,generation_off_without_weights_exact=True))
    atomic_json(evaluation/'digest.json',digest)
    figures(evaluation,clips,conditional,ablation,rows)
    print(json.dumps(dict(full_local=local['full'],full_whole=whole['full'],
        comparisons=comparisons,ablation=ablation,timing=timing),ensure_ascii=False,indent=2))


def figures(root,clips,conditional,ablation,references):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,ylabel in [('lpips_alex','Local LPIPS (lower better)'),('psnr_db','Local PSNR (dB)')]:
        fig,axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,clip in zip(axes.flat,clips):
            for key in ('direct','previous','rgb','feature'):
                rows = [clip['prefixes'][m][key] for m in MODES]
                ax.plot([r['bytes'] for r in rows],[r['roi_quality'][metric] for r in rows],'o-',label=LABELS[key])
            ax.set(title=clip['name'],xlabel='Actual total stream bytes',ylabel=ylabel)
            ax.grid(alpha=.3);ax.legend(fontsize=8)
        fig.savefig(root/f'prefix_{metric}.png',dpi=150);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    x = np.arange(4)
    for i,key in enumerate(('previous','rgb','feature')):
        for ax,metric in zip(axes,('lpips_reduction','psnr_increase')):
            ax.bar(x+(i-1)*.25,[r['gains'][key][metric] for r in conditional],width=.25,label=LABELS[key])
    for ax,title in zip(axes,('Extra E: local LPIPS reduction','Extra E: local PSNR increase')):
        ax.set(title=title,xticks=x,xticklabels=[r['name'] for r in conditional]);ax.grid(axis='y',alpha=.3);ax.legend(fontsize=8)
    fig.suptitle('No E to full E: same generator; E costs extra bytes')
    fig.savefig(root/'incremental_E.png',dpi=150);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,metric,title in zip(axes[:2],('lpips_alex','psnr_db'),('Local LPIPS: lower better','Local PSNR: higher better')):
        for i,key in enumerate(('rgb','off','on')):
            ax.bar(x+(i-1)*.25,[r[key][metric] for r in ablation],width=.25,
                label={'rgb':'RGB continuation','off':'Feature-trained LoRA, branch off','on':'Same LoRA, branch on'}[key])
        ax.set(title=title,xticks=x,xticklabels=[r['name'] for r in ablation]);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.3)
    gains = [r['lpips_gain_from_branch'] for r in ablation]
    bars = axes[2].bar(x,gains,color=['tab:green' if g > 0 else 'tab:red' for g in gains])
    axes[2].bar_label(bars,fmt='%+.5f',fontsize=8)
    axes[2].axhline(0,color='black',linewidth=.8)
    axes[2].set(title='Branch ON benefit: positive is better',ylabel='Local LPIPS(off) - Local LPIPS(on)',
                xticks=x,xticklabels=[r['name'] for r in ablation])
    axes[2].margins(y=.25);axes[2].grid(axis='y',alpha=.3)
    fig.savefig(root/'branch_ablation.png',dpi=150);plt.close(fig)
    fixed_visuals(root,clips,references)


def fixed_visuals(root,clips,references):
    """Can also inspect finished clips while the serial GPU queue continues."""
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for clip in clips:
        sid = clip['sample_id']; row = references[sid]
        for name in ('rgb','feature'):
            point_dir = root/sid/f'{name}_full'
            verify_artifacts(point_dir,read(point_dir/'result.json')['artifacts'])
        videos = {'GT':load_source(row['sample']), 'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
            'Previous image':load_frames(PREVIOUS/'evaluation'/sid/'image_full/reconstruction.npz'),
            'RGB continuation':load_frames(root/sid/'rgb_full/reconstruction.npz'),
            'RGB + feature':load_frames(root/sid/'feature_full/reconstruction.npz')}
        regions = row['metric_regions']; scale = 2
        width = max(r[4] for r in regions)*scale
        canvas = Image.new('RGB',(len(videos)*width,sum(r[5]*scale+36 for r in regions)),'white')
        draw = ImageDraw.Draw(canvas); top = 0
        for t,n,x0,y0,w,h in regions:
            for i,(name,video) in enumerate(videos.items()):
                image = Image.fromarray(video[8,y0:y0+h,x0:x0+w]).resize((w*scale,h*scale),Image.Resampling.NEAREST)
                canvas.paste(image,(i*width,top+36));draw.text((i*width+3,top+8),name,font=font,fill='black')
            top += h*scale+36
        canvas.save(root/f'{sid}_fixed.png')
        if sid in ('mechanism-00-reds','mechanism-02-reds'):
            region = regions[-1] if sid == 'mechanism-00-reds' else regions[0]
            # A compact 2x preview stays below this Notion workspace's 5 MiB
            # attachment limit; metrics and saved reconstructions are untouched.
            t,n,x0,y0,w,h = region; scale=2
            frames=[]
            for f in range(len(videos['GT'])):
                canvas=Image.new('RGB',(len(videos)*w*scale,h*scale+32),'white');draw=ImageDraw.Draw(canvas)
                for i,(name,video) in enumerate(videos.items()):
                    image=Image.fromarray(video[f,y0:y0+h,x0:x0+w]).resize((w*scale,h*scale),Image.Resampling.NEAREST)
                    canvas.paste(image,(i*w*scale,32));draw.text((i*w*scale+3,7),name,font=font,fill='black')
                frames.append(canvas)
            filename='wall_sequence.gif' if sid == 'mechanism-00-reds' else 'long33_sequence.gif'
            frames[0].save(root/filename,save_all=True,append_images=frames[1:],duration=125,loop=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--preview',action='store_true',help='Only draw already-complete full-prefix pairs')
    args = p.parse_args()
    if args.preview:
        root = args.output/'evaluation'
        rows = {r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
        clips = [dict(sample_id=sid,name=CLIPS[sid]) for sid in CLIPS if all(
            (root/sid/f'{k}_full/result.json').exists() for k in ('rgb','feature'))]
        fixed_visuals(root,clips,rows)
    else:
        report(args.output)
