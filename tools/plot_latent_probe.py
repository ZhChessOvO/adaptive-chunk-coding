"""Saved-evidence plots for the first native-latent B/E prototype, CPU only."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from demo.scalable_codec import atomic_json,file_hash
from tools.latent_stream_probe import DEFAULT,WIDTHS,verify
from tools.latent_probe import read


def collect(root):
    done=read(root/'complete.json')
    if not done['complete'] or done['artifacts']['summary.json']!=file_hash(root/'summary.json'):
        raise ValueError('stream experiment incomplete or summary changed')
    samples=read(root/'summary.json')['results'];bindings={}
    rows=[];native=[]
    for sample in samples:
        sid=sample['sample'];folder=root/sid;verify(folder)
        bindings[str(folder/'complete.json')]=file_hash(folder/'complete.json')
        group='REDS' if sid.startswith('reds') else 'UVG crop'
        n=sample['native'];native.append(dict(sample=sid,dataset=group,**n))
        for row in sample['rows']:
            detail=verify(folder/f'w{row["width"]}_{row["level"]}')
            if detail['actual_bytes']!=(folder/f'w{row["width"]}_{row["level"]}.rvl').stat().st_size:
                raise ValueError('stream size changed')
            rows.append(dict(row,dataset=group,native_bytes=n['bytes'],native_bpp=n['bpp'],
                total_over_native=row['actual_bytes']/n['bytes'],
                base_over_native=row['base_bytes']/n['bytes'],
                memory_gib=row['peak_cuda_allocated_bytes']/2**30))
    return rows,native,bindings


def aggregate(rows,native):
    output={}
    for group in ('REDS','UVG crop'):
        reference=[r for r in native if r['dataset']==group]
        if not reference: continue
        result={'samples':len(reference),'native':{
            'bytes':float(np.mean([r['bytes'] for r in reference])),
            'bpp':float(np.mean([r['bpp'] for r in reference])),
            **{k:float(np.mean([r['quality_all9'][k] for r in reference])) for k in ('psnr_db','lpips_alex')}}}
        for width in WIDTHS:
            result[str(width)]={}
            for level in ('B','BE'):
                points=[r for r in rows if r['dataset']==group and r['width']==width and r['level']==level]
                result[str(width)][level]={k:float(np.mean([r[k] for r in points])) for k in
                    ('actual_bytes','bpp','I_bytes','z_bytes','coarse_y_bytes','refinement_payload_bytes',
                     'base_header_bytes','enhancement_header_bytes','total_over_native','base_over_native','seconds','memory_gib')}
                result[str(width)][level].update({k:float(np.mean([r['quality_all9'][k] for r in points])) for k in ('psnr_db','lpips_alex')})
                result[str(width)][level]['P8']={k:float(np.mean([r['quality_P8'][k] for r in points])) for k in ('psnr_db','lpips_alex')}
        output[group]=result
    return output


def draw(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={3:'#2675b6',9:'#d07824',17:'#8e4bb2'}
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,(group,values) in enumerate(summary.items()):
        for col,(key,label) in enumerate((('psnr_db','PSNR (dB, higher better)'),('lpips_alex','LPIPS (lower better)'))):
            ax=axs[row,col]
            for width in WIDTHS:
                points=values[str(width)]
                ax.plot([points[x]['bpp'] for x in ('B','BE')],[points[x][key] for x in ('B','BE')],
                    'o--',color=colors[width],label=f'width {width}: B -> B+E')
            ref=values['native'];ax.scatter([ref['bpp']],[ref[key]],c='black',marker='*',s=140,label='Native UF same context',zorder=9)
            ax.set_title(group);ax.set_xlabel('Actual file bpp (all 9 frames, headers included)');ax.set_ylabel(label)
            ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Frozen UF latent split, G off | I QP32 + one P8 QP48\nEach dashed pair is a different encoding config, NOT a dense RD curve',fontsize=12)
    fig.savefig(out/'rd_pairs.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    keys=['I_bytes','z_bytes','coarse_y_bytes','refinement_payload_bytes','base_header_bytes','enhancement_header_bytes']
    labels=['Native I bootstrap','Shared z','Coarse y','Fine y (group headers included)','Base header/checksum','E header/checksum']
    for ax,(group,v) in zip(axs,summary.items()):
        bottom=np.zeros(3)
        for key,label in zip(keys,labels):
            vals=np.array([v[str(w)]['BE'][key]/1024 for w in WIDTHS])
            ax.bar(range(3),vals,bottom=bottom,label=label);bottom+=vals
        ax.axhline(v['native']['bytes']/1024,color='black',linestyle='--',label='Native total')
        ax.set_xticks(range(3),[f'width {w}' for w in WIDTHS]);ax.set_ylabel('KiB / 9-frame stream');ax.set_title(group)
        ax.legend(fontsize=7)
    fig.savefig(out/'bytes.png',dpi=150);plt.close(fig)


def pictures(root,out,samples):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    names=[]
    for sid in samples:
        folder=root/sid
        # Source is read only by this offline plotter, never by fresh decoders.
        protocol=read(root/'protocol.json')
        source_path=Path(protocol['inputs'][sid]['source_path'])
        if file_hash(source_path)!=protocol['inputs'][sid]['source_sha256']: raise ValueError('visual source changed')
        with np.load(source_path,allow_pickle=False) as f: source=f['source'][5].copy()
        panels=[('Source',source)]
        for width in WIDTHS:
            with np.load(folder/f'w{width}_B/pixels.npz',allow_pickle=False) as f:
                panels.append((f'B (width {width})',f['reconstruction'][5].copy()))
        with np.load(folder/'w3_BE/pixels.npz',allow_pickle=False) as f:
            panels.append(('B+E = native UF',f['reconstruction'][5].copy()))
        h,w=source.shape[:2];pw=384;ph=round(pw*h/w)
        canvas=Image.new('RGB',(5*pw,2*ph+92),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{sid} | saved fresh pixels, global frame 5; G off',fill='black',font=font)
        for i,(label,arr) in enumerate(panels):
            im=Image.fromarray(arr)
            draw.text((i*pw+5,30),label,fill='black',font=font)
            canvas.paste(im.resize((pw,ph)),(i*pw,54))
            cw,ch=w//3,h//3;x,y=(w-cw)//2,(h-ch)//2
            canvas.paste(im.crop((x,y,x+cw,y+ch)).resize((pw,ph)),(i*pw,ph+78))
        name=sid+'.png';canvas.save(out/name);names.append(name)
    return names


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path);p.add_argument('--wait',action='store_true');args=p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required for filesystem-heavy report')
    if args.wait:
        started=time.monotonic()
        while not (args.root/'complete.json').exists():
            if time.monotonic()-started>4*3600: raise TimeoutError('stream evidence did not finish')
            time.sleep(15)
    rows,native,bindings=collect(args.root);summary=aggregate(rows,native)
    out=args.output or args.root.parent/'report';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        receipt=read(out/'complete.json')
        for path,sha in receipt['inputs'].items():
            if file_hash(Path(path))!=sha: raise ValueError('report input changed')
        for name,sha in receipt['artifacts'].items():
            if file_hash(out/name)!=sha: raise ValueError('report artifact changed')
        print('Verified completed latent report without redrawing',flush=True);return
    draw(summary,out);images=pictures(args.root,out,[r['sample'] for r in native])
    atomic_json(out/'summary.json',dict(summary=summary,rows=rows,native=native))
    names=['rd_pairs.png','bytes.png','summary.json']+images
    atomic_json(out/'complete.json',dict(complete=True,inputs=bindings,images=images,
                artifacts={name:file_hash(out/name) for name in names}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
