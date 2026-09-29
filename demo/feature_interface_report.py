"""CPU-only artifact audit, paired summaries and fixed unselected visualizations."""
import json
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.conditioned_generation_evaluate import OLD, MODES
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.feature_condition_report import CLIPS, METRICS
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, resources

KEYS = ('direct','rgb','feature','zero','actual')
LABELS = dict(direct='No generation',rgb='Frozen RGB LoRA',feature='Previous joint feature model',
              zero='Trained RGB + coverage control',actual='Trained received-feature interface',
              without_content='Same interface, feature zeroed',shuffled='Same interface, spatial shuffle',
              branch_off='Same interface OFF')


def paired_clips(points):
    lookup = {(r['sample_id'],r['mode'],r['candidate']):r for r in points}
    expected = {(sid,mode,k) for sid in CLIPS for mode in MODES for k in ('actual','zero')}
    if len(points) != len(expected) or set(lookup) != expected:
        raise ValueError('missing or duplicate paired clip/prefix/candidate')
    clips = []
    for sid,name in CLIPS.items():
        prefixes = {}
        for mode in MODES:
            a,z = [lookup[(sid,mode,k)] for k in ('actual','zero')]
            assert a['bytes'] == z['bytes']
            prefixes[mode] = dict(actual=a,zero=z,direct=a['direct'],**a['previous'])
            assert all(prefixes[mode][k]['bytes'] == a['bytes'] for k in ('rgb','feature'))
        clips.append(dict(sample_id=sid,name=name,prefixes=prefixes,dataset=a['dataset']))
    return clips


def aggregate(clips,field):
    return {mode:{domain:{key:{metric:mean(c['prefixes'][mode][key][field][metric]
        for c in clips if domain == 'all' or c['dataset'] == domain)
        for metric in METRICS} for key in KEYS} for domain in ('all','REDS','UVG')}
        for mode in MODES}


def report(root):
    evaluation = root/'evaluation'; summary = read(evaluation/'summary.json')
    assert summary['complete'] and len(summary['ablations']) == 12 and len(summary['checks']) == 2
    clips = paired_clips(summary['results'])
    refs = {r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
    audit = []
    all_results = summary['results']+summary['ablations']+summary['checks']
    assert len({(r['sample_id'],r['candidate'],r['mode']) for r in all_results}) == 38
    for r in all_results:
        sid,mode = r['sample_id'],r['mode']; dest = evaluation/sid/f"{r['candidate']}_{mode}"
        assert read(dest/'result.json') == r
        verify_artifacts(dest,r['artifacts'])
        control,inner,parsed,overhead = fmt.parse((dest/'stream.acsg').read_bytes())
        oldname,directname = MODES[mode]
        previous,oldinner,_,oldoverhead = fmt.parse((OLD/sid/f'{oldname}.acsg').read_bytes())
        assert inner == oldinner and overhead == oldoverhead
        for k in ('generate','protect','context','feather','seed','processing_scale','blend','window','stride'):
            assert control[k] == previous[k]
        d = read(dest/'decode.json'); assert d == r['fresh_decode']
        assert d['total_bytes'] == r['bytes'] == (dest/'stream.acsg').stat().st_size
        assert sum(d[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
            'incomplete_tail_bytes','generation_control_bytes')) == d['total_bytes']
        assert d['feature_packet_ids'] == [p.meta['packet_id'] for p in parsed.packets if p.meta['start'] > 0]
        assert not d['source_frames_read'] and d['feature_bytes_added'] == 0
        actual = load_frames(dest/'reconstruction.npz')
        enhanced = load_frames(OLD/sid/directname/'reconstruction.npz')
        assert d['base_hash'] == frame_hash(load_frames(OLD/sid/'base/reconstruction.npz'))
        assert d['output_hash'] == frame_hash(actual) and d['generation_input_hash'] == frame_hash(enhanced)
        alpha = fmt.weights(actual.shape,control)
        np.testing.assert_array_equal(actual[alpha == 0],enhanced[alpha == 0])
        if r['candidate'] == 'G_off':
            np.testing.assert_array_equal(actual,enhanced); assert not d['generation_assets_validated']
        else:
            assert d['generation_executed'] and d['assets']['lora'] == control['lora']
            assert all(s['outside_coverage_exact'] for s in d['generation_runtime']['condition_statistics'])
            if mode == 'none' or r['candidate'] == 'branch_off':
                np.testing.assert_array_equal(actual,load_frames(PREVIOUS/'evaluation'/sid/f'rgb_{mode}/reconstruction.npz'))
        audit.append(dict(path=str(dest),bytes=r['bytes'],hashes=True,source_free=True,
                          unchanged_payload=True,unchanged_non_generate=True))
    first = next(iter(CLIPS))
    np.testing.assert_array_equal(load_frames(evaluation/first/'actual_full/reconstruction.npz'),
                                  load_frames(evaluation/first/'repeat_full/reconstruction.npz'))
    ablations = {(r['sample_id'],r['candidate']):r for r in summary['ablations']}
    differences = []
    for clip in clips:
        sid = clip['sample_id']; full = clip['prefixes']['full']; actual = full['actual']
        alternatives = {k:full[k] for k in ('rgb','feature','zero')}
        alternatives.update({k:ablations[(sid,k)] for k in ('without_content','shuffled','branch_off')})
        differences.append(dict(sample_id=sid,name=clip['name'],
            local_lpips=actual['roi_quality']['lpips_alex'],
            lpips_gain_vs={k:r['roi_quality']['lpips_alex']-actual['roi_quality']['lpips_alex']
                          for k,r in alternatives.items()},
            statistics=actual['fresh_decode']['generation_runtime']['condition_statistics']))
    local,whole = aggregate(clips,'roi_quality'),aggregate(clips,'quality')
    timing = {key:dict(receiver_seconds=mean(c['prefixes']['full'][key]['fresh_decode']['seconds'] for c in clips),
        peak_cuda_allocated_bytes=max(c['prefixes']['full'][key]['fresh_decode']['peak_cuda_allocated_bytes'] for c in clips))
        for key in ('actual','zero')}
    heartbeats = [json.loads(l) for l in (root/'heartbeat.jsonl').read_text().splitlines()]
    digest = dict(local=local,whole_frame=whole,full_prefix_differences=differences,timing=timing,
        training=summary['training'],resources=resources(),
        sampled_gpu_peak_mib=max(int(r['gpu'].split(',')[2]) for r in heartbeats),
        local_scope='Fixed regions, first 17 frames, area weighted within clip then equal clip mean',
        whole_scope='17/17/33/17 frames; equal clip mean',
        role='Four existing development clips; 12 prefixes are not 12 independent clips')
    atomic_json(evaluation/'audit.json',dict(complete=True,fresh_decodes=38,details=audit,
        repeat_exact=True,no_E_exact=True,branch_off_rgb_exact=True,G_off_without_weights_exact=True))
    atomic_json(evaluation/'digest.json',digest)
    figures(evaluation,clips,differences,refs)
    print(json.dumps(dict(full_local=local['full'],differences=differences,timing=timing),indent=2))


def figures(root,clips,differences,refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,title in [('lpips_alex','Local LPIPS (lower better)'),('psnr_db','Local PSNR (dB)')]:
        fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,clip in zip(axes.flat,clips):
            for key in KEYS:
                points = [clip['prefixes'][m][key] for m in MODES]
                ax.plot([p['bytes'] for p in points],[p['roi_quality'][metric] for p in points],
                        'o-',label=LABELS[key])
            ax.set(title=clip['name'],xlabel='Actual stream bytes',ylabel=title)
            ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.savefig(root/f'prefix_{metric}.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key in zip(axes,('zero','without_content','shuffled')):
        values = [d['lpips_gain_vs'][key] for d in differences]
        bars = ax.bar(np.arange(4),values,color=['tab:green' if v > 0 else 'tab:red' for v in values])
        ax.bar_label(bars,fmt='%+.5f',fontsize=8); ax.axhline(0,color='black',linewidth=.8)
        ax.set(title='Actual vs '+LABELS[key],xticks=np.arange(4),
               xticklabels=[c['name'] for c in clips],ylabel='LPIPS gain (positive better)')
        ax.tick_params(axis='x',labelrotation=15); ax.margins(y=.25); ax.grid(axis='y',alpha=.3)
    fig.savefig(root/'content_ablation.png',dpi=150); plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for clip in clips:
        sid = clip['sample_id']; row = refs[sid]
        videos = {'GT':load_source(row['sample']),
            'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
            'Frozen RGB LoRA':load_frames(PREVIOUS/'evaluation'/sid/'rgb_full/reconstruction.npz'),
            'RGB + coverage':load_frames(root/sid/'zero_full/reconstruction.npz'),
            'Received features':load_frames(root/sid/'actual_full/reconstruction.npz')}
        regions = row['metric_regions']; width = max(r[4] for r in regions)*2
        canvas = Image.new('RGB',(len(videos)*width,sum(r[5]*2+36 for r in regions)),'white')
        draw = ImageDraw.Draw(canvas); top = 0
        for t,n,x,y,w,h in regions:
            for i,(name,video) in enumerate(videos.items()):
                crop = Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                canvas.paste(crop,(i*width,top+36)); draw.text((i*width+3,top+8),name,font=font,fill='black')
            top += h*2+36
        canvas.save(root/f'{sid}_fixed.png')
