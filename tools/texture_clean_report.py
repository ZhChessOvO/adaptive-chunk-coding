"""Readable fixed crops/slow previews and learning curves, not synthetic images."""
import numpy as np

from tools.texture_diagnostic import core, phase_diagnostic
from tools.texture_ablation_report import video


def panels(case,source,received,rows,step):
    values = [('Source',core(source,case),None),('Received Y',core(received,case),None)]
    names = [('legacy','Old LoRA / reference'),('clean','Fresh LoRA / step 0')]
    if step:
        names += [(f'control_{step:06d}',f'Original loss / {step}'),
                  (f'phase_only_{step:06d}',f'Periodic loss / {step}')]
    for name,label in names:
        row = rows[name]
        with np.load(row['output_path']) as z: pixels=core(z['pixels'],case)
        values.append((label,pixels,row))
    return values


def draw(folder,case,source,received,rows,step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = panels(case,source,received,rows,step)
    columns = len(values)//2
    fig,axes = plt.subplots(2,columns,figsize=(10 if columns==3 else 8,8),layout='constrained')
    for ax,(label,pixels,row) in zip(axes.flat,values):
        h,w = pixels.shape[1:3]; ch,cw=min(96,h),min(96,w)
        ax.imshow(pixels[8,(h-ch)//2:(h+ch)//2,(w-cw)//2:(w+cw)//2],interpolation='nearest')
        ax.axis('off')
        if row:
            label += f"\nWhole-core LPIPS {row['quality']['lpips_alex']:.4f} | phase16 {row['error_phase16']['phase_rms_u8']:.3f}"
        ax.set_title(label,fontsize=10)
    fig.suptitle(case['sample_id']+'\nSame 96px detail / frame 9 / no filtering / raw G before fusion',fontsize=11)
    fig.savefig(folder/'detail.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(2,len(values),figsize=(3*len(values),6),layout='constrained')
    maps = [phase_diagnostic(p,values[0][1])[1] for _,p,_ in values]
    bound = max(1.,max(float(np.max(np.abs(m))) for m in maps))
    for i,((label,pixels,row),phase) in enumerate(zip(values,maps)):
        axes[0,i].imshow(pixels[8],interpolation='nearest'); axes[0,i].set_title(label,fontsize=9)
        axes[1,i].imshow(phase.mean(-1),cmap='coolwarm',vmin=-bound,vmax=bound,interpolation='nearest')
        axes[1,i].set_title('Source-referenced phase16 error',fontsize=9)
        for ax in axes[:,i]: ax.axis('off')
    fig.suptitle(case['sample_id']+f' | paired step {step} | whole core',fontsize=11)
    fig.savefig(folder/'comparison.png',dpi=140); plt.close(fig)
    names = ['detail.png','comparison.png']
    if case['evaluation_group'] == 'known_artifact':
        fig,axes = plt.subplots(len(values),5,figsize=(13,2.3*len(values)),layout='constrained')
        for i,(label,pixels,_) in enumerate(values):
            for j,t in enumerate((0,4,8,12,16)):
                axes[i,j].imshow(pixels[t],interpolation='nearest'); axes[i,j].axis('off')
                axes[i,j].set_title(f'{label} / frame {t+1}',fontsize=9)
        fig.suptitle(case['sample_id']+' | identical region through time',fontsize=11)
        fig.savefig(folder/'temporal_contact.png',dpi=140); plt.close(fig)
        video(folder,values)
        names += ['temporal_contact.png','comparison.mp4']
    return names


def trajectory(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    group = 'internal_validation' if any(r['group']=='internal_validation'
        for r in summary['aggregate']) else 'known_artifact'
    fig,axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,dataset in enumerate(('REDS','UVG')):
        selected = {r['name']:r for r in summary['aggregate'] if r['dataset']==dataset and r['group']==group}
        for j,(key,title) in enumerate((('phase16_error','Periodic error / lower'),
                                      ('lpips','LPIPS / lower'),('psnr','PSNR / higher'))):
            ax = axes[i,j]
            for arm,label,color in [('control','Fresh / original loss','#64748b'),
                                    ('phase_only','Fresh / periodic loss','#008b8b')]:
                points = sorted((int(n.rsplit('_',1)[1]),r[key]) for n,r in selected.items() if n.startswith(arm+'_'))
                ax.plot([0]+[p[0] for p in points],[selected['clean'][key]]+[p[1] for p in points],
                        'o-',label=label,color=color)
            ax.axhline(selected['legacy'][key],color='#b45309',linestyle='--',label='Old LoRA / reference')
            ax.set_title(f'{dataset} / {selected["clean"]["n"]} fixed ROIs | {title}',fontsize=10)
            ax.set_xlabel('Fresh LoRA optimization steps'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(group+' | raw G before fusion | NOT whole-video RD')
    fig.savefig(output/'trajectory.png',dpi=160); plt.close(fig)
