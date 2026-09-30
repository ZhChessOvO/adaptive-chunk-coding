"""CPU-only audits and fixed figures for the independent internal prompt study."""
import json
from statistics import mean

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.internal_condition_evaluate import ARMS,audit_training
from demo.internal_condition_pipeline import OLD,MODES,assert_noise
from demo.feature_condition_pipeline import DEFAULT as PREVIOUS
from demo.feature_condition_report import CLIPS,METRICS
from demo.patch_prefix_probe import load_frames,verify_artifacts
from demo.scalable_codec import atomic_json,file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import resources,load_source

LABELS=dict(off='RGB only (new mean/BF16 path)',input='Input addition',
            internal='Internal received features',zero='Internal RGB/coverage control',
            direct='No generation',legacy='Prior jointly trained feature model (old path)')


def grouped(points):
    keys=[(r['sample_id'],r['mode'],r['candidate']) for r in points]
    expected={(sid,m,k) for sid in CLIPS for m in MODES for k in ARMS}
    expected|={(sid,'full',k) for sid in CLIPS for k in ('without','shuffled')}
    first=next(iter(CLIPS));expected|={(first,'full',k) for k in ('repeat','G_off')}
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('incomplete or duplicated 58-point evaluation')
    lookup=dict(zip(keys,points))
    for sid in CLIPS:
        for mode in MODES:
            assert len({lookup[sid,mode,k]['bytes'] for k in ARMS})==1
    return lookup


def report(root):
    training=audit_training(root)
    dest=root/'evaluation';summary=read(dest/'summary.json')
    assert summary['complete'];lookup=grouped(summary['results'])
    refs={r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
    audit=[]
    for r in summary['results']:
        sid,mode,name=r['sample_id'],r['mode'],r['candidate'];point=dest/sid/f'{name}_{mode}'
        assert read(point/'result.json')==r
        verify_artifacts(point,r['artifacts'])
        wire=(point/'stream.acsg').read_bytes();control,inner,parsed,overhead=fmt.parse(wire)
        oldcontrol,oldinner,_,oldoverhead=fmt.parse((OLD/sid/f'{MODES[mode][0]}.acsg').read_bytes())
        assert inner==oldinner and overhead==oldoverhead
        for k in ('generate','protect','context','feather','seed','processing_scale','blend','window','stride'):
            assert control[k]==oldcontrol[k]
        d=read(point/'decode.json');assert d==r['fresh_decode']
        assert d['total_bytes']==r['bytes']==len(wire)
        assert sum(d[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
               'incomplete_tail_bytes','generation_control_bytes'))==len(wire)
        assert not d['source_frames_read'] and d['feature_bytes_added']==0
        pixels=load_frames(point/'reconstruction.npz')
        enhanced=load_frames(OLD/sid/MODES[mode][1]/'reconstruction.npz')
        assert d['output_hash']==frame_hash(pixels) and d['generation_input_hash']==frame_hash(enhanced)
        assert d['base_hash']==frame_hash(load_frames(OLD/sid/'base/reconstruction.npz'))
        alpha=fmt.weights(pixels.shape,control)
        np.testing.assert_array_equal(pixels[alpha==0],enhanced[alpha==0])
        if name=='G_off':
            np.testing.assert_array_equal(pixels,enhanced);assert not d['generation_assets_validated']
        else:
            assert_noise(lookup[sid,mode,'off']['fresh_decode'],d)
            assert d['assets']['lora']==control['lora'] and d['generation_executed']
            if mode=='none' or name=='repeat':
                other=dest/sid/('off_none' if mode=='none' else 'internal_full')/'reconstruction.npz'
                np.testing.assert_array_equal(pixels,load_frames(other))
        audit.append(dict(path=str(point),bytes=len(wire),payload_exact=True,non_generate_exact=True))
    def aggregate(field):
        return {mode:{domain:{k:{metric:mean(lookup[sid,mode,k][field][metric]
            for sid in CLIPS if domain=='all' or lookup[sid,mode,k]['dataset']==domain)
            for metric in METRICS} for k in ARMS} for domain in ('all','REDS','UVG')} for mode in MODES}
    differences=[]
    for sid,name in CLIPS.items():
        actual=lookup[sid,'full','internal']['roi_quality']['lpips_alex']
        differences.append(dict(sample_id=sid,name=name,internal_local_lpips=actual,
            lpips_gain_vs={k:lookup[sid,'full',k]['roi_quality']['lpips_alex']-actual
                          for k in ('off','input','zero','without','shuffled')}))
    heartbeat=[json.loads(l) for l in (root/'heartbeat.jsonl').read_text().splitlines()]
    digest=dict(local=aggregate('roi_quality'),whole_frame=aggregate('quality'),
        differences=differences,training=training,resources=resources(),
        sampled_gpu_peak_mib=max(int(r['gpu'].split(',')[2]) for r in heartbeat),
        receiver={k:dict(seconds=mean(lookup[s,'full',k]['fresh_decode']['seconds'] for s in CLIPS),
            peak_cuda_allocated_bytes=max(lookup[s,'full',k]['fresh_decode']['peak_cuda_allocated_bytes'] for s in CLIPS)) for k in ARMS},
        local_scope='Fixed first-17-frame regions, area-weighted within clip then equal clips',
        whole_scope='All 17/17/33/17 frames, equal clips; LPIPS not region-additive',
        role='Four reused development clips; 12 prefix points are not independent clips')
    atomic_json(dest/'audit.json',dict(complete=True,points=58,details=audit,
        paired_noise=True,no_E_exact=True,repeat_exact=True,G_off_without_weights_exact=True))
    atomic_json(dest/'digest.json',digest)
    figures(dest,lookup,refs,differences)
    print(json.dumps(dict(full_local=digest['local']['full'],differences=differences),indent=2),flush=True)


def figures(root,lookup,refs,differences):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    legacy={(r['sample_id'],r['mode']):r for r in read(PREVIOUS/'evaluation/summary.json')['results']
            if r['candidate']=='feature'}
    for field in ('roi_quality','quality'):
        for metric in ('lpips_alex','psnr_db'):
            fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
            for ax,(sid,name) in zip(axes.flat,CLIPS.items()):
                for key in (*ARMS,'direct','legacy'):
                    points=[lookup[sid,m,key] if key in ARMS else
                            lookup[sid,m,'off']['direct'] if key=='direct' else legacy[sid,m] for m in MODES]
                    ax.plot([p['bytes'] for p in points],[p[field][metric] for p in points],
                            'o--' if key=='legacy' else 'o-',label=LABELS[key],alpha=.65 if key=='legacy' else 1.)
                ax.set(title=name,xlabel='Actual stream bytes',ylabel=f'{field}: {metric}')
                ax.grid(alpha=.25);ax.legend(fontsize=6)
            fig.savefig(root/f'prefix_{field}_{metric}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key in zip(axes,('zero','without','shuffled')):
        values=[d['lpips_gain_vs'][key] for d in differences]
        bars=ax.bar(np.arange(4),values,color=['tab:green' if v>0 else 'tab:red' for v in values])
        ax.bar_label(bars,fmt='%+.5f',fontsize=8);ax.axhline(0,color='black',linewidth=.8)
        ax.set(title=f'Internal actual vs {key}',xticks=np.arange(4),
            xticklabels=list(CLIPS.values()),ylabel='Local LPIPS gain (positive better)')
        ax.tick_params(axis='x',labelrotation=15);ax.margins(y=.25)
    fig.savefig(root/'content_ablation.png',dpi=150);plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for index,(sid,name) in enumerate(CLIPS.items()):
        row=refs[sid]
        videos={'GT':load_source(row['sample']),
            'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
            **{k:load_frames(root/sid/f'{k}_full/reconstruction.npz') for k in ARMS}}
        regions=row['metric_regions'];width=max(r[4] for r in regions)*2
        canvas=Image.new('RGB',(len(videos)*width,sum(r[5]*2+36 for r in regions)),'white')
        draw=ImageDraw.Draw(canvas);top=0
        for t,n,x,y,w,h in regions:
            for i,(key,video) in enumerate(videos.items()):
                tile=Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                canvas.paste(tile,(i*width,top+36));draw.text((i*width+3,top+8),key,font=font,fill='black')
            top+=h*2+36
        canvas.save(root/f'{sid}_fixed.png')
        if index in (1,2):
            t,n,x,y,w,h=regions[0];frames=[]
            for frame in range(17):
                canvas=Image.new('RGB',(w*4,h*2+32),'white');draw=ImageDraw.Draw(canvas)
                for i,key in enumerate(('input','internal')):
                    tile=Image.fromarray(videos[key][frame,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                    canvas.paste(tile,(i*w*2,32));draw.text((i*w*2+3,8),f'{key}, frame {frame}',font=font,fill='black')
                frames.append(canvas)
            frames[0].save(root/f'{sid}_motion.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
    atomic_json(root/'figure_artifacts.json',{p.name:file_hash(p) for p in sorted(root.iterdir())
                                           if p.suffix in ('.png','.gif')})
