"""Real-pixel field-of-view locator and paired context reports; no image synthesis."""
from collections import defaultdict
import numpy as np
from routervc.generation.context_geometry import rgb_crop, local_core


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def overview(folder,case,source,received,geo):
    from matplotlib.patches import Rectangle
    plt=plotting();fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,pixels,title in zip(axes,(source,received,source),
            ('Available source view','Received Y / generator input','Source / 512 context')):
        if ax is axes[2]:
            ax.imshow(rgb_crop(pixels,geo['large'])[8],interpolation='nearest')
            ox,oy=geo['large'][:2]
        else:ax.imshow(pixels[8],interpolation='nearest');ox=oy=0
        for key,color in (('large','#00bcd4'),('small','#ffbb00'),('core','#ff2255')):
            x,y,w,h=geo[key];ax.add_patch(Rectangle((x-ox,y-oy),w,h,fill=False,edgecolor=color,linewidth=1.7))
        ax.set_title(title);ax.axis('off')
    label='REDS resized full view' if case['dataset']=='REDS' else 'UVG existing 512 crop, NOT original full frame'
    fig.suptitle(case['sample_id']+'\n'+label+' | cyan:512 / yellow:256 / red:scored128 | frame9',fontsize=11)
    fig.savefig(folder/'overview.png',dpi=130);plt.close(fig)


def comparisons(folder,case,source,received,geo,rows,root):
    plt=plotting(); selected={(r['name'],r['size']):r for r in rows if r['seed_index']==0}
    fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
    for ri,size in enumerate((256,512)):
        for ci,name in enumerate(('source','legacy','clean','phase2000')):
            ax=axes[ri,ci]
            if name=='source':pixels=rgb_crop(source if ri==0 else received,geo['core']);title='Source' if ri==0 else 'Received Y'
            else:
                row=selected[name,size]
                with np.load(root/row['relative_dir']/'output.npz') as z:pixels=local_core(z['pixels'],row['binding']['crop'],geo['core'])
                title=f'{name} / {size} context\nLPIPS {row["quality"]["lpips_alex"]:.4f} | phase {row["phase_error"]["phase_rms_u8"]:.3f}'
            ax.imshow(pixels[8],interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(case['sample_id']+'\nIdentical 128px location / native scale / paired spatial noise / raw G',fontsize=11)
    fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)
    # Larger output is shown alongside its source so semantic context remains visible.
    fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
    for ax,name in zip(axes,('source','received','legacy','phase2000')):
        if name in ('source','received'):pixels=rgb_crop(source if name=='source' else received,geo['large'])
        else:
            with np.load(root/selected[name,512]['relative_dir']/'output.npz') as z:pixels=z['pixels'].copy()
        ax.imshow(pixels[8],interpolation='nearest');ax.axis('off');ax.set_title(name+' / 512 view')
    fig.suptitle(case['sample_id']+'\nWhole processing view (not a routed full-video reconstruction)',fontsize=11)
    fig.savefig(folder/'wide_comparison.png',dpi=130);plt.close(fig)


def summary_figure(summary,root):
    buckets=defaultdict(list)
    for item in summary['cases']:
        for r in item['results']:
            if r['seed_index']==0:
                buckets[(item['case']['evaluation_group'],item['case']['dataset'],r['name'],r['size'])].append(r)
    rows=[]
    for (group,dataset,name,size),values in sorted(buckets.items()):
        rows.append(dict(group=group,dataset=dataset,name=name,size=size,n=len(values),
            lpips=float(np.mean([r['quality']['lpips_alex'] for r in values])),
            psnr=float(np.mean([r['quality']['psnr_db'] for r in values])),
            phase=float(np.mean([r['phase_error']['phase_rms_u8'] for r in values])),
            seconds=float(np.mean([r['observation']['runtime']['seconds_model_load_excluded'] for r in values])),
            peak_gib=max(r['observation']['runtime']['peak_cuda_allocated_bytes'] for r in values)/1024**3))
    plt=plotting();fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,dataset in enumerate(('REDS','UVG')):
        for j,key in enumerate(('lpips','psnr','phase')):
            ax=axes[i,j]
            for name in ('legacy','clean','phase2000'):
                values=[r for r in rows if r['group']=='internal_validation' and r['dataset']==dataset and r['name']==name]
                ax.plot([r['size'] for r in values],[r[key] for r in values],'o-',label=name)
            ax.set_title(dataset+' / '+key);ax.set_xticks([256,512]);ax.set_xlabel('Context width / native pixels');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Previously used internal validation / same 128 core / NO fitting or RD claim')
    fig.savefig(root/'summary.png',dpi=150);plt.close(fig)
    return dict(rows=rows)
