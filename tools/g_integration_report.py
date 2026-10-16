"""Read-only full-picture G integration audit, figures and slow previews."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo.routervc_fullview_probe import read,digest,immutable,verify_artifacts
from demo.scalable_codec import atomic_json
from demo.scalable_format import frame_hash
from demo.routervc_policy import grid_rois
from tools.cooperative_review import source_rgb
from tools.g_integration import DEFAULT,ARMS,verify,validate
from tools.storage_guard import disk_snapshot
from tools.texture_context_report import plotting

NAMES=('received',*ARMS)
LABELS=dict(received='Received Y / G off',legacy='Legacy production LoRA',
            initial='Starting mixed400',control='Control continuation +400')


def group(cases):
    buckets=defaultdict(list)
    for point in cases: buckets[point['cohort'],point['dataset']].append(point)
    result=[]
    for (cohort,dataset),rows in sorted(buckets.items()):
        out=dict(cohort=cohort,dataset=dataset,count=len(rows),
                 mean_bpp=float(np.mean([r['bpp'] for r in rows])),models={})
        for arm in NAMES:
            out['models'][arm]={k:float(np.mean([r['metrics'][arm]['quality'][k] for r in rows]))
                for k in ('lpips_alex','psnr_db','temporal_delta_mae')}
            out['models'][arm]['phase_per_frame']=float(np.mean([
                r['metrics'][arm]['G_core_phase']['per_frame_rms_mean'] for r in rows]))
        out['control_better_than_legacy']=sum(r['metrics']['control']['quality']['lpips_alex']<
            r['metrics']['legacy']['quality']['lpips_alex'] for r in rows)
        out['control_better_than_initial']=sum(r['metrics']['control']['quality']['lpips_alex']<
            r['metrics']['initial']['quality']['lpips_alex'] for r in rows)
        result.append(out)
    return result


def audit(root,cfg):
    verify(root); evidence=[]
    for j,job in enumerate(cfg['jobs']):
        folder=root/'samples'/job['row']['sample_id']
        for arm in ARMS:
            tags=('main','repeat') if j<2 else ('main',)
            for tag in tags:
                dest=folder/arm/tag; receipt=read(dest/'complete.json')
                verify_artifacts(dest,receipt['artifacts'])
                validate(receipt,job['receipt'],legacy=arm=='legacy')
                with np.load(dest/'pixels.npz') as z:
                    if frame_hash(z['reconstruction'])!=receipt['output_hash']: raise ValueError('pixel hash failed')
                if tag=='repeat' and receipt['output_hash']!=read(folder/arm/'main/complete.json')['output_hash']:
                    raise ValueError('fresh repeat failed')
            evidence.append(dict(sample=job['row']['sample_id'],arm=arm,legacy_exact=arm=='legacy'))
        if j<2:
            dest=folder/'legacy/off';receipt=read(dest/'complete.json');verify_artifacts(dest,receipt['artifacts'])
            validate(receipt,job['receipt'],disabled=True)
    return dict(complete=True,main_points=len(evidence),legacy_historical_exact=6,actual_reload_repeats=6,
                partial_region_recoveries=2,G_off_exact=2,source_free=True,model_promoted=False)


def charts(output,cases,groups):
    plt=plotting();fig,axes=plt.subplots(len(groups),3,figsize=(13,3.2*len(groups)),layout='constrained')
    for i,row in enumerate(groups):
        title=f'{row["dataset"]} / {row["cohort"]} / n={row["count"]}'
        for col,(key,label) in enumerate((('lpips_alex','Whole-frame LPIPS (lower)'),
                ('psnr_db','Whole-frame PSNR dB (higher)'),('phase_per_frame','G-core phase error (auxiliary)'))):
            ax=axes[i,col];v=[row['models'][n][key] for n in NAMES]
            ax.plot(range(4),v,'o-',color='#21808d')
            for x,y in enumerate(v):ax.annotate(f'{y:.4f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(4),['G off','Legacy','Mixed400','Control+400'],fontsize=8)
            ax.set_title(title+'\n'+label,fontsize=10);ax.grid(alpha=.2);ax.margins(y=.3)
    fig.suptitle('Same B/E bytes and G regions / actual native crops + 64 halo / fixed E50 / NOT an RD curve')
    fig.savefig(output/'summary.png',dpi=135);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    labels=[p['sample_id'].replace('-f000-n17-fullview','').replace('-historical-evaluation-crop','') for p in cases]
    for ax,key,title in zip(axes,('lpips_alex','psnr_db'),('Whole-frame LPIPS change (negative is better)',
                                                       'Whole-frame PSNR change / dB (positive is better)')):
        y=np.arange(len(cases))
        for offset,base,color in ((-.18,'legacy','#258190'),(.18,'initial','#dc9455')):
            vals=[p['metrics']['control']['quality'][key]-p['metrics'][base]['quality'][key] for p in cases]
            ax.barh(y+offset,vals,height=.34,label='Control minus '+base,color=color)
        ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
        ax.set_title(title);ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.savefig(output/'paired_deltas.png',dpi=140);plt.close(fig)


def selection(source,point):
    frame=Image.fromarray(source[8]).convert('RGBA'); overlay=Image.new('RGBA',frame.size,(0,0,0,0))
    draw=ImageDraw.Draw(overlay); rois=grid_rois(*source.shape[1:3])
    e={i for _,i in point['E_regions']}
    for i,(x,y,w,h) in enumerate(rois):
        draw.rectangle((x,y,x+w-1,y+h-1),fill=(232,150,40,80) if i in point['G_regions'] else None,
                       outline=(40,210,255,255) if i in e else (255,255,255,160),width=3 if i in e else 1)
        draw.text((x+5,y+5),str(i),fill=(255,255,255,255))
    return np.array(Image.alpha_composite(frame,overlay).convert('RGB'))


def canvas(panels,t,*,width):
    h,w=panels[0][1].shape[1:3]; ph=round(h*width/w)
    # Even dimensions for yuv420p video. No crop/warp; display scaling only.
    ph+=ph%2;head=54
    image=Image.new('RGB',(3*width,2*(ph+head)),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for i,(label,pixels) in enumerate(panels):
        x,y=(i%3)*width,(i//3)*(ph+head)
        frame=Image.fromarray(pixels[t]).resize((width,ph),Image.Resampling.LANCZOS)
        image.paste(frame,(x,y+head));draw.multiline_text((x+8,y+5),label,font=font,fill='black',spacing=3)
    return image


def figures(output,root,job,point):
    source=source_rgb(job['row']);sid=point['sample_id'];arrays={}
    for arm in ARMS:
        with np.load(root/'samples'/sid/arm/'main/pixels.npz') as z:
            arrays[arm]=z['reconstruction'].copy()
            if arm=='legacy':arrays['received']=z['enhanced'].copy()
    panels=[('Source / frame 9\nMetrics use all 17 native frames',source)]
    for arm in NAMES:
        q=point['metrics'][arm]['quality']
        panels.append((f'{LABELS[arm]}\nLPIPS {q["lpips_alex"]:.5f} | PSNR {q["psnr_db"]:.3f}',arrays[arm]))
    mask=selection(source,point)
    panels.append(('Same E/G selection in every model\nOrange: G / Cyan: received E in either chunk',np.repeat(mask[None],17,axis=0)))
    canvas(panels,8,width=min(768,source.shape[2])).save(output/(sid+'.png'))
    # Independent whole-view images preserve native pixels for closer inspection.
    for arm in ('legacy','control'):
        Image.fromarray(arrays[arm][8]).save(output/(sid+'_'+arm+'_native.png'))
    path=output/(sid+'.mp4');tmp=path.with_suffix('.partial.mp4')
    # Same whole picture, 4fps; captions and selection map are static overlays.
    panels[0]=('Source / 17 frames / 4fps slow preview',source)
    first=canvas(panels,0,width=512)
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{first.width}x{first.height}','-r','4','-i','pipe:0','-an','-c:v','libx264',
        '-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as child:
        try:
            for t in range(17):child.stdin.write(canvas(panels,t,width=512).tobytes())
        finally:child.stdin.close()
        if child.wait()!=0:raise RuntimeError('preview failed')
    tmp.replace(path)


def resources(root,cases):
    def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
    hb=lines(root/'heartbeat.jsonl');mem=lines(root/'memory.jsonl')
    groups=defaultdict(list)
    for p in cases:
        for arm in ARMS:groups[p['dataset'],arm].append(p['resources'][arm])
    inference=[dict(dataset=d,arm=a,count=len(rows),mean_seconds=float(np.mean([r['seconds'] for r in rows])),
        peak_allocated_gib=max(r['peak_cuda_allocated_bytes'] for r in rows)/2**30,
        peak_reserved_gib=max(r['peak_cuda_reserved_bytes'] for r in rows)/2**30) for (d,a),rows in groups.items()]
    return dict(queue_minutes=hb[-1]['elapsed_seconds']/60,inference=inference,
        time_scope='cold decoder includes model hashing/loading, B/E, Rg/G/fusion and per-region resume writes; excludes final NPZ and scoring',
        sampled_device_peak_gib=max(float(r['gpu'].splitlines()[0].split(',')[2])/1024 for r in hb),
        cgroup_peak_charged_gib=max(m['charged'] for m in mem)/2**30,
        cgroup_peak_rss_gib=max(m['rss'] for m in mem)/2**30,failcnt_start=mem[0]['failcnt'],failcnt_end=mem[-1]['failcnt'],
        current_disks=disk_snapshot(),run_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))


def main(args):
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    root=args.root;output=root/'report';output.mkdir(exist_ok=True)
    while not (root/'complete.json').exists():
        if not args.wait:raise RuntimeError('evaluation not complete')
        if (root/'last_failure.json').exists():raise RuntimeError('evaluation needs attention')
        time.sleep(30)
    # The completion receipt precedes final verification/heartbeat. Wait for the
    # writer's lock to close before binding logs; never snapshot a moving file.
    lock=(root/'integration.lock').open('a')
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if not args.wait:raise RuntimeError('evaluation is still finalizing')
            time.sleep(30)
    if read(root/'integration.progress.json')['phase']!='complete_review_required':
        raise RuntimeError('final verification did not finish')
    names=['complete.json','protocol.json','summary.json','heartbeat.jsonl','memory.jsonl']
    immutable(output/'protocol.json',dict(inputs={str(root/n):digest(root/n) for n in names},
        code=digest(Path(__file__)),scope='existing outputs only; no inference/fitting/promotion'))
    if (output/'complete.json').exists():
        done=read(output/'complete.json');verify_artifacts(output,done['artifacts']);print('READ_ONLY_REENTRY_OK');return
    cfg=read(root/'protocol.json');result=audit(root,cfg)
    cases=read(root/'summary.json')['cases'];groups=group(cases)
    atomic_json(output/'audit.json',result);atomic_json(output/'groups.json',groups)
    charts(output,cases,groups)
    for job,point in zip(cfg['jobs'],cases,strict=True):figures(output,root,job,point)
    atomic_json(output/'resources.json',resources(root,cases))
    files=[p for p in output.iterdir() if p.is_file()]
    atomic_json(output/'complete.json',dict(complete=True,artifacts={p.name:digest(p) for p in files}))
    print(json.dumps(dict(complete=True,audit=result,groups=groups)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--wait',action='store_true');main(p.parse_args())
