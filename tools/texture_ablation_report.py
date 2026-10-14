"""Fixed-view four-arm reports from real saved pixels; no image synthesis."""
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.texture_diagnostic import core, phase_diagnostic

ARMS = ('control','augmentation_only','phase_only','repair')
LABELS = {'control':'Original loss', 'augmentation_only':'LPIPS view only',
          'phase_only':'Phase loss only', 'repair':'Both changes'}
COLORS = {'control':'#64748b','augmentation_only':'#008b8b',
          'phase_only':'#9a50aa','repair':'#c45a26'}


def panels(case, source, results):
    values=[('Source',core(source,case),None)]
    for name in ('initial',*(a+'_000400' for a in ARMS)):
        row=results[name]
        with np.load(row['output_path']) as z: pixels=core(z['pixels'],case)
        label='Current LoRA' if name=='initial' else LABELS[name.rsplit('_',1)[0]]
        values.append((label,pixels,row))
    return values


def draw(folder,case,source,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=panels(case,source,results)
    maps=[phase_diagnostic(p,values[0][1])[1] for _,p,_ in values]
    bound=max(1.,max(float(np.max(np.abs(m))) for m in maps))
    fig,axes=plt.subplots(3,6,figsize=(17,8),layout='constrained')
    for i,((label,pixels,row),phase) in enumerate(zip(values,maps)):
        h,w=pixels.shape[1:3]
        axes[0,i].imshow(pixels[8],interpolation='nearest')
        if row:
            q=row['quality'];label+=f"\nLPIPS {q['lpips_alex']:.4f} | PSNR {q['psnr_db']:.2f}"
        axes[0,i].set_title(label,fontsize=9)
        ch,cw=min(96,h),min(96,w)
        axes[1,i].imshow(pixels[8,(h-ch)//2:(h+ch)//2,(w-cw)//2:(w+cw)//2],interpolation='nearest')
        axes[1,i].set_title('Same central 96px detail',fontsize=9)
        axes[2,i].imshow(phase.mean(-1),cmap='coolwarm',vmin=-bound,vmax=bound,interpolation='nearest')
        axes[2,i].set_title(f'Error phase16 RMS {np.sqrt(np.mean(phase**2)):.3f}',fontsize=9)
        for ax in axes[:,i]:ax.axis('off')
    fig.suptitle(case['sample_id']+f" | region {case['region']} | raw G before fusion | 400 steps",fontsize=11)
    fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    if case['evaluation_group']!='known_artifact':return
    fig,axes=plt.subplots(6,5,figsize=(13,14),layout='constrained')
    for i,(label,pixels,_) in enumerate(values):
        for j,t in enumerate((0,4,8,12,16)):
            axes[i,j].imshow(pixels[t],interpolation='nearest');axes[i,j].axis('off')
            axes[i,j].set_title(f'{label} / frame {t+1}',fontsize=9)
    fig.suptitle(case['sample_id']+' | same location through time',fontsize=11)
    fig.savefig(folder/'temporal_contact.png',dpi=140);plt.close(fig)
    video(folder,values)


def video(folder,values):
    from PIL import Image,ImageDraw
    h,w=values[0][1].shape[1:3]
    width,height=len(values)*w*2,h*2+32
    temporary=folder/'comparison.partial.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','12',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for t in range(17):
            canvas=Image.new('RGB',(width,height),'white');painter=ImageDraw.Draw(canvas)
            for i,(label,pixels,_) in enumerate(values):
                canvas.paste(Image.fromarray(pixels[t]).resize((w*2,h*2),Image.Resampling.NEAREST),(i*w*2,32))
                painter.text((i*w*2+5,8),label,fill='black')
            child.stdin.write(np.asarray(canvas).tobytes())
        child.stdin.close()
        if child.wait():raise RuntimeError('preview encode failed')
        os.replace(temporary,folder/'comparison.mp4')
    finally:
        if child.poll() is None:child.terminate();child.wait()


def trajectory(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,dataset in enumerate(('REDS','UVG')):
        selected={r['name']:r for r in summary['aggregate']
                  if r['group']=='internal_validation' and r['dataset']==dataset}
        for j,(key,title) in enumerate((('phase16_error','Periodic error / lower'),
                                      ('lpips','LPIPS / lower'),('psnr','PSNR / higher'))):
            ax=axes[i,j]
            for arm in ARMS:
                points=sorted((int(n.rsplit('_',1)[1]),r[key]) for n,r in selected.items() if n.startswith(arm+'_'))
                ax.plot([0]+[p[0] for p in points],[selected['initial'][key]]+[p[1] for p in points],
                    'o-',label=LABELS[arm],color=COLORS[arm])
            ax.set_title(f'{dataset} ({selected["initial"]["n"]} fixed ROIs) | {title}',fontsize=10)
            ax.set_xlabel('Additional G-only training steps');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Paired 2 x 2 loss ablation | internal validation | NOT whole-video RD')
    fig.savefig(output/'trajectory.png',dpi=160);plt.close(fig)
