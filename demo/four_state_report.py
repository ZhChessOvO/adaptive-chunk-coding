"""Read-only teacher diagnostics; never label an isolated montage as a stream."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import DEFAULT, STATES, ROIS, rows, crop, verify
from demo.chunk_enhancement_experiment import read
from demo.scalable_codec import atomic_json, file_hash
from demo.patch_prefix_probe import load_frames


def summarize(records):
    domains = defaultdict(list)
    for record in records:
        domains[record['dataset']].extend(record['regions'])
        domains['All'].extend(record['regions'])
    summary={}
    for name, values in domains.items():
        summary[name]=dict(regions=len(values),
            quality={s:{m:float(np.mean([r['quality'][s][m] for r in values]))
                for m in ('lpips_alex','psnr_db','temporal_delta_mae')} for s in STATES},
            e_packet_bytes_mean=float(np.mean([r['costs']['e_packet_bytes'] for r in values])),
            e_help_without_G=sum(r['lpips_gain']['E']>0 for r in values),
            e_help_with_G=sum(r['quality']['EG']['lpips_alex']<r['quality']['G']['lpips_alex'] for r in values),
            eg_best_lpips=sum(min(STATES,key=lambda s:r['quality'][s]['lpips_alex'])=='EG' for r in values),
            g_seconds={s:float(np.median([r['g_seconds'][s] for r in values])) for s in ('G','EG')})
    return summary


def charts(dest, records, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[n for n in ('REDS','UVG','All') if n in summary]
    fig, axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,metric,title in zip(axes,('lpips_alex','psnr_db','temporal_delta_mae'),
                            ('Local LPIPS (lower better)','Local PSNR (dB)','Temporal MAE (lower better)')):
        for j,name in enumerate(names):
            ax.bar(np.arange(4)+(j-1)*.25,[summary[name]['quality'][s][metric] for s in STATES],
                   .25,label=name)
        ax.set_xticks(range(4),STATES);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend()
    fig.suptitle('Isolated regional counterfactuals on reused training windows; not whole-video RD',fontsize=10)
    fig.tight_layout();fig.savefig(dest/'state_quality.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,5))
    for domain in ('REDS','UVG'):
        rs=[r for v in records if v['dataset']==domain for r in v['regions']]
        if not rs:continue
        ax.scatter([r['lpips_gain']['E'] for r in rs],
                   [r['quality']['G']['lpips_alex']-r['quality']['EG']['lpips_alex'] for r in rs],
                   label=domain,s=18,alpha=.6)
    lo,hi=ax.get_xlim();bot,top=ax.get_ylim();lo=min(lo,bot);hi=max(hi,top)
    ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
    ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
    ax.set_xlabel('LPIPS gain of E without G: B - E')
    ax.set_ylabel('LPIPS gain of E with G: G - EG')
    ax.set_title('Does the same packet help direct display and generation equally?')
    ax.legend();fig.tight_layout();fig.savefig(dest/'conditional_gain.png',dpi=150);plt.close(fig)


def panels(dest, sid, source, variants, annotations, filename, scale=2):
    frame=8
    width,height=source.shape[2]*scale,source.shape[1]*scale
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    canvas=Image.new('RGB',(width*(len(variants)+1),height+86),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((8,4),sid+' | fixed frame 9 | research diagnostic',font=font,fill='black')
    for i,(name,values) in enumerate([('Source',source),*variants.items()]):
        canvas.paste(Image.fromarray(values[frame]).resize((width,height),Image.Resampling.NEAREST),(i*width,82))
        draw.text((i*width+5,26),name,font=font,fill='black')
        draw.multiline_text((i*width+5,47),annotations.get(name,''),font=small,fill='black')
    canvas.save(dest/filename)


def report(root):
    manifest=read(root/'labels.json')
    assert manifest['complete']
    dest=root/'report';dest.mkdir(exist_ok=True)
    done=dest/'summary.json'
    if done.exists():
        saved=read(done)
        assert saved['labels_sha256']==file_hash(root/'labels.json')
        verify(dest,saved['artifacts'])
        return saved
    records=[]
    for row in manifest['samples']:
        path=Path(row['path']);assert file_hash(path)==row['sha256']
        records.append(read(path))
    summary=summarize(records)
    charts(dest,records,summary)
    by_id={r['sample_id']:r for r in records}
    for row in rows(True):
        sid=row['sample_id']
        assert file_hash(Path(row['pair_path']))==row['pair_hash']
        with np.load(row['pair_path'],allow_pickle=False) as cache:source=cache['source'].copy()
        with np.load(root/'received'/sid/'received_E.npz',allow_pickle=False) as cache:
            base,enhanced=cache['base'].copy(),cache['enhanced'].copy()
        i=5
        with np.load(root/'received'/sid/f'cell_{i:02d}'/'outputs.npz',allow_pickle=False) as cache:
            variants=dict(B=crop(base,i),E=crop(enhanced,i),G=cache['G'].copy(),EG=cache['EG'].copy())
        r=by_id[sid]['regions'][i]
        annotations={s:f'LPIPS {r["quality"][s]["lpips_alex"]:.4f}\n{r["costs"]["individual_stream_bytes"][s]} B whole stream' for s in STATES}
        panels(dest,sid+' | cell 5',crop(source,i),variants,annotations,f'fixed_{sid}.png')
        if (root/'composition/summary.json').exists():
            folder=root/'composition'/sid/'stripes'
            r=read(folder/'result.json')
            panels(dest,sid+' | same mixed choices, same noise',source,
                dict(Base=base,Isolated=load_frames(folder/'isolated_prediction.npz'),
                     Actual=load_frames(folder/'fresh/reconstruction.npz')),
                dict(Base='reference',Isolated='prediction ignoring neighbor E\nnot a decoded mixed stream',
                     Actual=f'fresh mixed stream | {r["bytes"]} B'),f'composition_{sid}.png',scale=1)
    saved=dict(complete=True,labels_sha256=file_hash(root/'labels.json'),summary=summary,
        artifacts={p.name:file_hash(p) for p in dest.glob('*.png')},
        scope='regional training-data counterfactuals, no independent evaluation or whole-video RD claim')
    atomic_json(done,saved)
    return saved


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT)
    print(report(p.parse_args().root)['summary'])
