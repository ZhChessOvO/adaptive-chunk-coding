"""CPU-only diagnostics of measured sender labels, never a new quality trial.

Compare one packet's direct Y gain with its measured final R_g/G gain. Their
difference includes changed G inputs, selection and ordinal noise assignment;
it is NOT a controlled estimate of generator cooperation alone. Split TRAIN
and validation, and retain negative targets. No training setting is changed.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read, digest, save, verify_artifacts


def marginal(parent, child):
    before = parent['binding']['selected']
    after = child['binding']['selected']
    if len(after) != len(before)+1 or after[:-1] != before or after[-1] in before:
        raise ValueError('expected exactly one appended, previously absent region')
    extra = child['total_bytes']-parent['total_bytes']
    if extra <= 0 or extra != child['E_packet_bytes']-parent['E_packet_bytes']:
        raise ValueError('whole-stream increment must equal appended packet bytes')
    direct = parent['Goff_quality']['lpips_alex']-child['Goff_quality']['lpips_alex']
    final = parent['quality']['lpips_alex']-child['quality']['lpips_alex']
    psnr = child['quality']['psnr_db']-parent['quality']['psnr_db']
    if not all(math.isfinite(v) for v in (direct,final,psnr)):
        raise ValueError('measured metrics must be finite')
    old, new = parent['route']['indices'], child['route']['indices']
    return dict(region=after[-1], prior_E_regions=len(before), incremental_bytes=extra,
        direct_lpips_gain=direct, final_lpips_gain=final, final_psnr_gain_db=psnr,
        gain_difference=final-direct, final_gain_per_KiB=final*1024/extra,
        G_order_changed=old != new, G_set_changed=set(old) != set(new),
        G_before=old, G_after=new)


def collect(stage):
    stage=Path(stage)
    p=read(stage/'protocol.json');done=read(stage/'labels.complete.json')
    expected={r['sample_id'] for r in p['rows']}
    if (not done.get('complete') or done['protocol']!=digest(stage/'protocol.json')
            or set(done['samples'])!=expected or done['measured_marginals']!=6*len(expected)):
        raise ValueError('complete matching sender labels required')
    rows=[];inputs={str(stage/n):digest(stage/n) for n in ('protocol.json','labels.complete.json')}
    for sample in p['rows']:
        sid=sample['sample_id'];folder=stage/'samples'/sid
        if digest(folder/'complete.json')!=done['samples'][sid]:
            raise ValueError('sample completion changed')
        record=read(folder/'complete.json')
        if not record.get('complete') or record['binding']['sample']!=sample:
            raise ValueError('sample identity changed')
        # Includes streams and render records; does not load full RGB or tensors.
        verify_artifacts(folder,record['artifacts'])
        inputs[str(folder/'complete.json')]=digest(folder/'complete.json')
        for state in range(2):
            parent=read(folder/f'state{state}/parent/result.json')
            additions=sorted(n for n in record['artifacts']
                if n.startswith(f'state{state}/add_') and n.endswith('/result.json'))
            if len(additions)!=3:
                raise ValueError('three measured additions per state required')
            for name in additions:
                child=read(folder/name)
                for value in (parent,child):
                    if (not value.get('complete') or value['binding']['config']!=p['teacher']['wire_config']
                            or value['source_frames_used_by_receiver']
                            or value['source_frames_used_by_generator'] or value['explicit_mask_bytes']!=0):
                        raise ValueError('fixed source-free receiver contract changed')
                row=marginal(parent,child)
                rows.append(dict(sample_id=sid,dataset=sample['dataset'],split=sample['router_split'],
                    state='empty' if state==0 else 'partial',**row))
    if len(rows)!=done['measured_marginals']:
        raise ValueError('label coverage differs')
    return p,rows,inputs


def aggregate(rows):
    groups={}
    for dataset,split,state in dict.fromkeys((r['dataset'],r['split'],r['state']) for r in rows):
        selected=[r for r in rows if (r['dataset'],r['split'],r['state'])==(dataset,split,state)]
        n=len(selected)
        groups[f'{dataset}/{split}/{state}']=dict(measured_marginals=n,
            windows=len({r['sample_id'] for r in selected}),
            positive_final=sum(r['final_lpips_gain']>0 for r in selected),
            negative_final=sum(r['final_lpips_gain']<0 for r in selected),
            zero_final=sum(r['final_lpips_gain']==0 for r in selected),
            direct_positive_final_negative=sum(r['direct_lpips_gain']>0 and r['final_lpips_gain']<0 for r in selected),
            final_positive_direct_nonpositive=sum(r['final_lpips_gain']>0 and r['direct_lpips_gain']<=0 for r in selected),
            G_set_changed=sum(r['G_set_changed'] for r in selected),
            G_order_only_changed=sum(r['G_order_changed'] and not r['G_set_changed'] for r in selected),
            **{key:sum(r[key] for r in selected)/n for key in (
                'direct_lpips_gain','final_lpips_gain','gain_difference','incremental_bytes','final_psnr_gain_db')})
    return groups


def figures(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'REDS':'#276fbf','UVG':'#a55083'}
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for row,split in zip(axes,('train','validation')):
        for ax,state in zip(row,('empty','partial')):
            data=[r for r in rows if r['split']==split and r['state']==state]
            span=max([abs(r[key]) for r in data for key in ('direct_lpips_gain','final_lpips_gain')]+[.001])*1.1
            for dataset,marker in (('REDS','o'),('UVG','^')):
                for changed in (False,True):
                    group=[r for r in data if r['dataset']==dataset and r['G_set_changed']==changed]
                    if group:
                        ax.scatter([r['direct_lpips_gain'] for r in group],[r['final_lpips_gain'] for r in group],
                            marker=marker,s=28,alpha=.65,edgecolors=colors[dataset],
                            facecolors=colors[dataset] if changed else 'none',
                            label=f'{dataset} / G set '+('changed' if changed else 'same'))
            ax.plot([-span,span],[-span,span],color='#888',linestyle=':',linewidth=1)
            ax.axhline(0,color='#bbb',linewidth=.7);ax.axvline(0,color='#bbb',linewidth=.7)
            ax.set(xlim=(-span,span),ylim=(-span,span),
                   xlabel='Direct Y LPIPS gain (parent - child)',
                   ylabel='Final Rg/G LPIPS gain (parent - child)',
                   title=f'{split.upper()} / {state} E state / {len(data)} measured additions')
            if data:ax.legend(fontsize=8,frameon=False)
            ax.grid(alpha=.15)
    fig.suptitle('What the sender learns: final gain need not equal direct reconstruction gain')
    fig.text(.5,.018,'Positive = improvement. Sampled candidates, not a benchmark or independent observations.\n'
             'Final-minus-direct also includes changed G inputs, routing and ordinal noise; not isolated G synergy.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.955));fig.savefig(output/'direct_vs_final.png',dpi=160);plt.close(fig)


def render(stage,output):
    stage,output=Path(stage),Path(output)
    done=output/'complete.json'
    if done.exists():
        receipt=read(done);verify_artifacts(output,receipt['artifacts'])
        if receipt['code_sha256']!=digest(Path(__file__)):
            raise ValueError('target report source changed')
        _,_,inputs=collect(stage)
        if inputs!=receipt['inputs']:raise ValueError('target report inputs changed')
        print('SENDER_TARGET_REPORT_VERIFIED_READ_ONLY',flush=True);return
    protocol,rows,inputs=collect(stage)
    output.mkdir(parents=True,exist_ok=True)
    figures(rows,output)
    save(output/'summary.json',dict(complete=True,smoke=protocol['smoke'],rows=rows,groups=aggregate(rows),
        inference=False,training_modified=False,model_selection=False,
        scope='measured sampled marginal labels only; not new RD or an allocation oracle',
        difference_caveat='G input, selection and ordinal noise all may change; not isolated synergy'))
    save(done,dict(complete=True,inputs=inputs,code_sha256=digest(Path(__file__)),
        artifacts={n:digest(output/n) for n in ('summary.json','direct_vs_final.png')}))
    print('SENDER_TARGET_REPORT_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('CPU label report requires tmux')
    if args.output.resolve()==args.stage.resolve() or args.output.resolve()==(args.stage/'router').resolve():
        raise ValueError('use a separate diagnostic output directory')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='sender_target_report',max_hours=72));run.thread.start()
    try:
        while not (args.stage/'labels.complete.json').exists():
            if not args.wait:raise ValueError('sender labels incomplete')
            run.check();run.update(phase='waiting_for_labels',GPU_used=False);run.stop.wait(30)
        run.update(phase='CPU_target_diagnostics',GPU_used=False)
        render(args.stage,args.output)
    finally:
        run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':
    main()
