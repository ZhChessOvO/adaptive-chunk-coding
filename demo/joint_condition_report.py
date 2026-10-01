"""Read-only audits, paired numerical summaries and fixed visual comparisons."""
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.joint_condition_evaluate import ARMS, HISTORY, grouped, exact_reference
from demo.joint_condition_pipeline import audit as audit_training
from demo.internal_condition_train import INITIAL, REPO
from demo.internal_condition_pipeline import OLD, MODES, assert_noise
from demo.feature_condition_report import CLIPS, METRICS
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import resources, load_source

LABELS = dict(initial='Initial RGB (reused)', rgb='RGB +3000 steps',
    internal='Real features + LoRA', zero='Zero features + LoRA', direct='Base / E only')


def historical(expected_hash):
    """Reuse only the verified initial RGB endpoint on the identical mean/BF16 path."""
    dest = HISTORY/'evaluation'
    assert file_hash(dest/'summary.json') == expected_hash
    summary = read(dest/'summary.json')
    initial = torch.load(INITIAL, weights_only=True, map_location='cpu')
    off = torch.load(dest/'models/off.pt', weights_only=True, map_location='cpu')
    assert off['branch_mode'] == 'off'
    for k,v in initial['state_dict'].items():
        torch.testing.assert_close(v, off['state_dict'][k], rtol=0, atol=0)
    lookup = {}
    for row in summary['results']:
        if row['candidate'] != 'off':
            continue
        point = dest/row['sample_id']/f"off_{row['mode']}"
        assert read(point/'result.json') == row
        verify_artifacts(point, row['artifacts'])
        d = row['fresh_decode']
        assert d == read(point/'decode.json') and d['assets']['lora'] == file_hash(dest/'models/off.pt')
        assert d['total_bytes'] == row['bytes'] == (point/'stream.acsg').stat().st_size
        assert d['output_hash'] == frame_hash(load_frames(point/'reconstruction.npz'))
        key = (row['sample_id'],row['mode'])
        assert key not in lookup
        lookup[key] = row
    assert set(lookup) == {(sid,m) for sid in CLIPS for m in MODES}
    return lookup


def report(root):
    training = audit_training(root)
    dest = root/'evaluation'
    summary = read(dest/'summary.json')
    assert summary['complete']
    for protocol in (read(root/'queue_protocol.json'),summary['protocol']):
        for f,h in protocol['code'].items():
            assert file_hash(REPO/'demo'/f) == h
    assert summary['training'] == training
    assert summary['protocol']['training_audit'] == file_hash(root/'training_audit.json')
    assert summary['protocol']['references'] == file_hash(OLD/'summary.json')
    lookup = grouped(summary['results'])
    history = historical(summary['protocol']['historical_summary'])
    refs = {r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
    details = []
    for r in summary['results']:
        sid,mode,name = r['sample_id'],r['mode'],r['candidate']
        point = dest/sid/f'{name}_{mode}'
        assert read(point/'result.json') == r
        verify_artifacts(point,r['artifacts'])
        wire = (point/'stream.acsg').read_bytes()
        control,inner,_,overhead = fmt.parse(wire)
        oldcontrol,oldinner,_,oldoverhead = fmt.parse((OLD/sid/f'{MODES[mode][0]}.acsg').read_bytes())
        assert inner == oldinner and overhead == oldoverhead
        for k in ('generate','protect','context','feather','seed','processing_scale','blend','window','stride'):
            assert control[k] == oldcontrol[k]
        d = r['fresh_decode']
        assert d == read(point/'decode.json') and d['total_bytes'] == r['bytes'] == len(wire)
        assert sum(d[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
            'incomplete_tail_bytes','generation_control_bytes')) == len(wire)
        assert not d['source_frames_read'] and d['feature_bytes_added'] == 0
        pixels = load_frames(point/'reconstruction.npz')
        direct = load_frames(OLD/sid/MODES[mode][1]/'reconstruction.npz')
        assert d['output_hash'] == frame_hash(pixels) and d['generation_input_hash'] == frame_hash(direct)
        assert d['base_hash'] == frame_hash(load_frames(OLD/sid/'base/reconstruction.npz'))
        alpha = fmt.weights(pixels.shape,control)
        np.testing.assert_array_equal(pixels[alpha == 0],direct[alpha == 0])
        assert r['bytes'] == history[sid,mode]['bytes']
        assert r['direct'] == history[sid,mode]['direct'] == refs[sid]['points'][MODES[mode][1]]
        if name == 'G_off':
            np.testing.assert_array_equal(pixels,direct)
            assert not d['generation_assets_validated']
        else:
            assert_noise(lookup[sid,mode,'rgb']['fresh_decode'],d)
            assert_noise(history[sid,mode]['fresh_decode'],d)
            profile = summary['protocol']['profiles']['internal' if name == 'repeat' else name]
            assert d['assets'] == profile
            for k,v in profile.items():
                assert control[k] == v
            assert d['assets']['lora'] == control['lora'] and d['generation_executed']
            ref = exact_reference(name,mode)
            if ref:
                np.testing.assert_array_equal(pixels,load_frames(dest/sid/f'{ref[0]}_{ref[1]}/reconstruction.npz'))
        details.append(dict(path=str(point),bytes=len(wire),payload_exact=True,non_generate_exact=True))
    def get(sid,mode,arm):
        if arm == 'initial':
            return history[sid,mode]
        if arm == 'direct':
            return lookup[sid,mode,'rgb']['direct']
        return lookup[sid,mode,arm]
    def aggregate(field):
        return {mode:{domain:{k:{metric:mean(get(sid,mode,k)[field][metric]
            for sid in CLIPS if domain == 'all' or lookup[sid,mode,'rgb']['dataset'] == domain)
            for metric in METRICS} for k in ('initial',*ARMS,'direct')}
            for domain in ('all','REDS','UVG')} for mode in MODES}
    differences = []
    for sid,name in CLIPS.items():
        actual = get(sid,'full','internal')['roi_quality']['lpips_alex']
        differences.append(dict(sample_id=sid,name=name,internal_local_lpips=actual,
            lpips_gain_vs={k:get(sid,'full',k)['roi_quality']['lpips_alex']-actual
                for k in ('initial','rgb','zero','off','without','shuffled')}))
    beats = [json.loads(l) for l in (root/'heartbeat.jsonl').read_text().splitlines()]
    evaluation_beats = [r for r in beats if r['mode'] in ('evaluate','run')]
    digest = dict(report_code=file_hash(Path(__file__)),local=aggregate('roi_quality'), whole_frame=aggregate('quality'),
        differences=differences, training=training, resources=resources(),
        ablations={k:{m:mean(get(s,'full',k)['roi_quality'][m] for s in CLIPS) for m in METRICS}
            for k in ('internal','zero','off','without','shuffled')},
        sampled_gpu_peak_mib=max(int(r['gpu'].split(',')[2]) for r in evaluation_beats),
        evaluation_seconds=summary['elapsed_seconds'],
        receiver={k:dict(seconds=mean(get(s,'full',k)['fresh_decode']['seconds'] for s in CLIPS),
            peak_cuda_allocated_bytes=max(get(s,'full',k)['fresh_decode']['peak_cuda_allocated_bytes'] for s in CLIPS)) for k in ARMS},
        ordinary_file_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()),
        local_scope='Fixed first-17-frame regions; area-weighted within clip, equal clips',
        whole_scope='All 17/17/33/17 frames; equal clips; LPIPS not region-additive',
        role='Four reused development clips; no independent generalization claim')
    atomic_json(dest/'audit.json',dict(complete=True,points=58,reused_initial_points=12,details=details,
        paired_noise=True,no_E_own_LoRA_exact=True,repeat_exact=True,G_off_without_weights_exact=True))
    atomic_json(dest/'digest.json',digest)
    figures(dest,lookup,history,refs,differences)
    print(json.dumps(dict(full_local=digest['local']['full'],ablations=digest['ablations'],
        differences=differences),indent=2),flush=True)


def figures(root,lookup,history,refs,differences):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field in ('roi_quality','quality'):
        for metric in ('lpips_alex','psnr_db'):
            fig,axes = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
            for ax,(sid,name) in zip(axes.flat,CLIPS.items()):
                for key in ('direct','initial',*ARMS):
                    points = [lookup[sid,m,key] if key in ARMS else
                        history[sid,m] if key == 'initial' else lookup[sid,m,'rgb']['direct'] for m in MODES]
                    ax.plot([p['bytes'] for p in points],[p[field][metric] for p in points],
                        'o--' if key == 'initial' else 'o-',label=LABELS[key])
                ax.set(title=name,xlabel='Actual stream bytes',ylabel=f'{field}: {metric}')
                ax.grid(alpha=.25)
                ax.legend(fontsize=6)
            fig.savefig(root/f'prefix_{field}_{metric}.png',dpi=150)
            plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('rgb','zero','without','shuffled')):
        values = [d['lpips_gain_vs'][key] for d in differences]
        bars = ax.bar(np.arange(4),values,color=['tab:green' if v>0 else 'tab:red' for v in values])
        ax.bar_label(bars,fmt='%+.5f',fontsize=8)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(title=f'Real features vs {key}',xticks=np.arange(4),xticklabels=list(CLIPS.values()),
               ylabel='Local LPIPS gain (positive better)')
        ax.tick_params(axis='x',labelrotation=15)
        ax.margins(y=.25)
    fig.savefig(root/'content_ablation.png',dpi=150)
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for index,(sid,name) in enumerate(CLIPS.items()):
        row = refs[sid]
        videos = {'GT':load_source(row['sample']),
            'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
            'initial':load_frames(HISTORY/'evaluation'/sid/'off_full/reconstruction.npz'),
            **{k:load_frames(root/sid/f'{k}_full/reconstruction.npz') for k in ARMS}}
        regions = row['metric_regions']
        width = max(r[4] for r in regions)*2
        canvas = Image.new('RGB',(len(videos)*width,sum(r[5]*2+36 for r in regions)),'white')
        draw = ImageDraw.Draw(canvas)
        top = 0
        for t,n,x,y,w,h in regions:
            for i,(key,video) in enumerate(videos.items()):
                tile = Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                canvas.paste(tile,(i*width,top+36))
                draw.text((i*width+3,top+8),key,font=font,fill='black')
            top += h*2+36
        canvas.save(root/f'{sid}_fixed.png')
        if index in (1,2):
            t,n,x,y,w,h = regions[0]
            frames = []
            for frame in range(17):
                canvas = Image.new('RGB',(w*4,h*2+32),'white')
                draw = ImageDraw.Draw(canvas)
                for i,key in enumerate(('rgb','internal')):
                    tile = Image.fromarray(videos[key][frame,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                    canvas.paste(tile,(i*w*2,32))
                    draw.text((i*w*2+3,8),f'{key}, frame {frame}',font=font,fill='black')
                frames.append(canvas)
            frames[0].save(root/f'{sid}_motion.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
    atomic_json(root/'figure_artifacts.json',{p.name:file_hash(p) for p in sorted(root.iterdir())
        if p.suffix in ('.png','.gif')})
