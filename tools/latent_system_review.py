"""Post-training new-latent dual-Router RD, native UF, G-off and fixed views.

Same 13 previously used views; all new points use actual bytes and fresh workers.
No checkpoint promotion, new optimization, old feature-patch data or downloads.
"""
import argparse
import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.latent import routing,router_data,sender_data

RATIOS=(0.,.25,.5,1.)
QPS=(0,8,16,24,32,40,48)
CODE=('tools/latent_system_review.py','tools/run_latent_system_review.sh',
      'tools/latent_native17.py','tools/latent_receiver_audit.py')


def receive(run,stream,output,disabled=False):
    from tools.latent_router_queue import execute
    if not (output/'complete.json').exists():
        options=['decode','--stream',stream,'--output',output,'--receiver',
                 '/nonexistent/noG.pt' if disabled else sender_data.RECEIVER]
        if disabled:options+=['--disable-generation']
        execute(run,output.parent.name+'_'+output.name,options,distributed=True)
    result=read(output/'complete.json');verify_artifacts(output,result['artifacts'])
    if result['stream_sha256']!=digest(stream) or result['actual_bytes']!=stream.stat().st_size:
        raise ValueError('review fresh stream identity differs')
    return result


def point(folder,row,arm,budget,received,source,metric):
    from demo.scalable_experiment import quality
    if (folder/'result.json').exists():
        saved=read(folder/'result.json')
        if saved['receipt_sha256']!=digest(folder/'receive/complete.json'):
            raise ValueError('completed score receipt changed')
        return saved
    with np.load(folder/'receive/pixels.npz',allow_pickle=False) as f:
        output=f['reconstruction'].copy()
    if frame_hash(output)!=received['output_hash']:raise ValueError('stored pixels differ')
    result=dict(sample_id=row['sample_id'],dataset=row['dataset'],arm=arm,budget=budget,
        bpp=received['bpp'],actual_bytes=received['actual_bytes'],quality=quality(source,output,metric),
        G_calls=len(received.get('generated',[])),peak_GiB=received['peak_cuda_allocated_bytes']/2**30,
        fresh_seconds=received.get('seconds',received.get('total_seconds')),
        receipt_sha256=digest(folder/'receive/complete.json'))
    save(folder/'result.json',result);return result


def summarize(points,ratios,qps,sample_count):
    expected=sample_count*(4*len(ratios)+len(qps))
    if len(points)!=expected or len({(p['sample_id'],p['arm'],p['budget']) for p in points})!=expected:
        raise ValueError('incomplete system comparison')
    groups={}
    for dataset in ('REDS','UVG'):
        groups[dataset]={}
        for arm in ('source','zero_source','fixed_order','source_Goff','UF'):
            groups[dataset][arm]={}
            for budget in qps if arm=='UF' else ratios:
                values=[p for p in points if p['dataset']==dataset and p['arm']==arm and p['budget']==budget]
                if not values:raise ValueError('missing dataset/arm/budget')
                groups[dataset][arm][str(budget)]=dict(samples=len(values),
                    bpp=float(np.mean([p['bpp'] for p in values])),
                    **{k:float(np.mean([p['quality'][k] for p in values]))
                       for k in ('lpips_alex','psnr_db','temporal_delta_mae')},
                    max_GiB=max(p['peak_GiB'] for p in values),
                    mean_fresh_seconds=float(np.mean([p['fresh_seconds'] for p in values])),
                    mean_G_calls=float(np.mean([p['G_calls'] for p in values])))
    return groups


def panel_label(point):
    # comparison_image reserves one title line per panel; a newline gets covered.
    return f'{point["arm"]} | {point["bpp"]:.4f} bpp | L {point["quality"]["lpips_alex"]:.3f}'


def plots(root,points,groups,rows,ratios,qps,*,figure_root=None,smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.plot_latent_diagnostics import comparison_image
    figure_root=Path(figure_root or root);figure_root.mkdir(parents=True,exist_ok=True)
    caption='SMOKE: two-epoch sender; workflow check only\n' if smoke else ''
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    styles={'source':'s-','zero_source':'o--','fixed_order':'^:','source_Goff':'x--','UF':'k.-'}
    for i,dataset in enumerate(('REDS','UVG')):
        for j,metric in enumerate(('lpips_alex','psnr_db')):
            ax=axes[i,j]
            for arm,style in styles.items():
                vals=[groups[dataset][arm][str(b)] for b in (qps if arm=='UF' else ratios)]
                ax.plot([v['bpp'] for v in vals],[v[metric] for v in vals],style,label=arm)
            ax.set_title('REDS resized full views' if dataset=='REDS' else 'UVG existing crops')
            ax.set_xlabel('Actual whole-stream bpp');ax.set_ylabel(metric);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(caption+'New-latent RouterVC: independent sender + adapted receiver, frozen UF/G\n'
                 f'{len(rows)} reused diagnostics, not full benchmarks; UF I32 with own P reference; G <= 8')
    fig.savefig(figure_root/'system_rd.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,key,label in zip(axes,('max_GiB','mean_fresh_seconds'),('Max CUDA allocated GiB','Mean fresh process seconds')):
        labels=[];values=[]
        for dataset in ('REDS','UVG'):
            for arm in ('UF','source_Goff','source'):
                subset=list(groups[dataset][arm].values())
                value=max(p[key] for p in subset) if key=='max_GiB' else float(np.mean([p[key] for p in subset]))
                labels.append(dataset+'\n'+arm);values.append(value)
        ax.bar(range(len(values)),values)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(len(values)),labels,fontsize=8);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.18)
    fig.suptitle(caption+'All-call process peaks, not total device memory; loading and saving included, not realtime FPS')
    fig.savefig(figure_root/'resources.png',dpi=160);plt.close(fig)
    for row in rows:
        sid=row['sample_id'];folder=root/'samples'/sid
        with np.load(row['source_path'],allow_pickle=False) as f:source=f['source'][8].copy()
        panels=[('Source',source)]
        current=next(p for p in points if p['sample_id']==sid and p['arm']=='source' and p['budget']==.5)
        native=min([p for p in points if p['sample_id']==sid and p['arm']=='UF'],key=lambda p:abs(p['bpp']-current['bpp']))
        cases=[('UF',native['budget'],folder/f'UF_q{native["budget"]}'),
               *((arm,.5,folder/f'{arm}_e050') for arm in ('fixed_order','zero_source','source_Goff','source'))]
        for arm,budget,dest in cases:
            measured=next(p for p in points if p['sample_id']==sid and p['arm']==arm and p['budget']==budget)
            with np.load(dest/'receive/pixels.npz',allow_pickle=False) as f:rgb=f['reconstruction'][8].copy()
            panels.append((panel_label(measured),rgb))
        dest=figure_root/'samples'/sid;dest.mkdir(parents=True,exist_ok=True)
        comparison_image(panels,dest/'fixed_frame.png',title=('SMOKE | ' if smoke else '')+sid+
                         ' | half E-byte cap; nearest UF rate, NOT matched exactly')


def review(args,run):
    import torch
    from demo.chunk_enhancement_codec import configure_torch
    from demo.stage_c_three_path_roi_probe import LPIPSAlex,encode_dcvc_stream
    from demo.scalable_codec import BaseCodec
    from routervc.latent.codec import MODEL_I,MODEL_P
    from routervc.latent.sender import write_prefixes
    from tools.latent_sender_queue import verify
    from tools.latent_receiver_review import source_rows
    from tools.latent_native17 import fresh
    training=args.root/('smoke' if args.smoke else 'formal')
    verify(training,args.root/'input_cache'/('smoke' if args.smoke else 'formal'),run)
    protocol=read(training/'protocol.json');teacher=protocol['teacher']
    modelroot=training/('resumed' if args.smoke else 'router')
    models={arm:modelroot/arm/'best.pt' for arm in ('source','zero_source')}
    rows=source_rows()
    if args.smoke:rows=[next(r for r in rows if r['dataset']==d) for d in ('REDS','UVG')]
    ratios=(0.,.5) if args.smoke else RATIOS
    qps=(8,48) if args.smoke else QPS
    binding=dict(format='new_latent_system_review_v1',smoke=args.smoke,rows=rows,
        code={n:digest(sender_data.REPO/n) for n in CODE},
        training_complete=digest(training/'complete.json'),models={a:dict(path=str(p),sha256=digest(p)) for a,p in models.items()},
        receiver_sha256=sender_data.RECEIVER_SHA,ratios=list(ratios),qps=list(qps),I_qp=32,
        receiver_profile=routing.identity(),G_assets=teacher['G_assets_hash'],
        packet_banks='authenticated receiver_review banks for the exact same sources',
        data_role='13 reused diagnostic windows; REDS resized full views / UVG crops',
        budget='fractions of all candidate E bytes, NOT regions or whole-stream bytes',
        time_scope='bank reuse; sender ordering excludes encoding; fresh decode includes model loading/saving')
    immutable(args.output/'protocol.json',binding)
    if (args.output/'complete.json').exists():
        verify_completed(args.output,run.check);return
    configure_torch();metric=LPIPSAlex(True);points=[];prefix_pairs=0;repeat_checks=0
    for row in rows:
        sid=row['sample_id'];folder=args.output/'samples'/sid;folder.mkdir(parents=True,exist_ok=True)
        bankdir=router_data.ROOT/'receiver_review/samples'/sid
        enc=read(bankdir/'encoded.json');verify_artifacts(bankdir,enc['artifacts'])
        if enc['binding']['row']!=row:raise ValueError('evaluation bank belongs to another source')
        bank=(bankdir/'bank.rvlp').read_bytes()
        with np.load(row['source_path'],allow_pickle=False) as f:source=f['source'].copy()
        plans={arm:write_prefixes(folder/(arm+'_planning'),source,bank,model,smoke=args.smoke,
                         check=run.check,progress=run.update) for arm,model in models.items()}
        costs=np.asarray(routing.bundle_bytes(bank),np.int64)
        previous={}
        for ratio in ratios:
            for arm in ('source','zero_source','fixed_order','source_Goff'):
                run.check();dest=folder/f'{arm}_e{int(ratio*100):03d}';dest.mkdir(parents=True,exist_ok=True)
                if arm=='fixed_order':
                    from demo.routervc_sender_router import prefix_under_budget
                    order=router_data.selection(sid,16)
                    selected=prefix_under_budget(order,costs,int(costs.sum()*ratio))['indices']
                    raw=routing.wrap(routing.subset(bank,selected),sender_data.RECEIVER_SHA,teacher['G_assets_hash'],
                                     max_g=8,seed=teacher['seed'])
                else:
                    sourcearm='source' if arm=='source_Goff' else arm
                    raw=(folder/(sourcearm+'_planning')/f'e{int(100*ratio):03d}.rvlrg').read_bytes()
                if arm in previous:
                    if not raw.startswith(previous[arm]):raise ValueError('system comparison lost literal prefix')
                    prefix_pairs+=1
                previous[arm]=raw
                atomic_bytes(dest/'stream.rvlrg',raw)
                received=receive(run,dest/'stream.rvlrg',dest/'receive',arm=='source_Goff')
                if received['base_hash']!=enc['base_hash']:raise ValueError('system E/G changed B')
                if arm=='source_Goff' and received['output_hash']!=received['enhanced_hash']:
                    raise ValueError('G-off is not actual Y')
                points.append(point(dest,row,arm,ratio,received,source,metric))
                # Deterministic repeat on one REDS and one UVG, same selected source E50.
                if arm=='source' and ratio==.5 and row==next(r for r in rows if r['dataset']==row['dataset']):
                    again=receive(run,dest/'stream.rvlrg',dest/'repeat')
                    if again['output_hash']!=received['output_hash']:raise ValueError('fresh system repeat differs')
                    repeat_checks+=1
        for qp in qps:
            run.check();dest=folder/f'UF_q{qp}';dest.mkdir(parents=True,exist_ok=True)
            if not (dest/'encoded.json').exists():
                if qp==48:raw=(bankdir/'native_own.bin').read_bytes()
                else:
                    codec=BaseCodec(MODEL_I,MODEL_P);configure_torch();torch.cuda.set_stream(codec.stream)
                    with torch.inference_mode():
                        raw,_=encode_dcvc_stream(list(source),32,qp,codec.i_net,codec.p_net,codec.device,32)
                    del codec;gc.collect();torch.cuda.empty_cache();torch.cuda.set_stream(torch.cuda.default_stream())
                atomic_bytes(dest/'native.bin',raw)
                save(dest/'encoded.json',dict(artifacts={'native.bin':digest(dest/'native.bin')},I_qp=32,P_qp=qp))
            verify_artifacts(dest,read(dest/'encoded.json')['artifacts'])
            received=fresh(run,dest/'native.bin',dest/'receive')
            points.append(point(dest,row,'UF',qp,received,source,metric))
    groups=summarize(points,ratios,qps,len(rows))
    save(args.output/'summary.json',dict(points=points,groups=groups,scope=binding))
    plots(args.output,points,groups,rows,ratios,qps,smoke=args.smoke)
    names=['summary.json','system_rd.png','resources.png',*(f'samples/{r["sample_id"]}/fixed_frame.png' for r in rows)]
    save(args.output/'complete.json',dict(complete=True,points=len(points),fresh_decodes=len(points)+repeat_checks,
        repeats=repeat_checks,literal_prefix_pairs=prefix_pairs,protocol=digest(args.output/'protocol.json'),
        artifacts={n:digest(args.output/n) for n in names}))
    verify_completed(args.output,run.check)


def verify_completed(root,check=lambda:None):
    done=read(root/'complete.json');verify_artifacts(root,done['artifacts'])
    if done['protocol']!=digest(root/'protocol.json'):raise ValueError('system protocol differs')
    summary=read(root/'summary.json')
    for p in summary['points']:
        check();name=f'UF_q{p["budget"]}' if p['arm']=='UF' else f'{p["arm"]}_e{int(100*p["budget"]):03d}'
        folder=root/'samples'/p['sample_id']/name
        if read(folder/'result.json')!=p:raise ValueError('system point changed')
        receipt=read(folder/'receive/complete.json');verify_artifacts(folder/'receive',receipt['artifacts'])
        stream=folder/('native.bin' if p['arm']=='UF' else 'stream.rvlrg')
        if digest(stream)!=receipt['stream_sha256'] or stream.stat().st_size!=p['actual_bytes']:
            raise ValueError('system real bytes differ')
    return done


def replot(root,destination,run):
    """Render saved scores/pixels to a NEW directory; never change completed results."""
    root,destination=Path(root),Path(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError('replot destination must be outside completed results')
    verify_completed(root,run.check)
    binding=dict(source_complete=digest(root/'complete.json'),code=digest(Path(__file__)),
                 scores_recomputed=False,inference_executed=False)
    immutable(destination/'protocol.json',binding)
    if (destination/'complete.json').exists():
        verify_artifacts(destination,read(destination/'complete.json')['artifacts']);return
    summary=read(root/'summary.json');p=summary['scope']
    plots(root,summary['points'],summary['groups'],p['rows'],p['ratios'],p['qps'],
          figure_root=destination,smoke=p['smoke'])
    names=['system_rd.png','resources.png',*(f'samples/{r["sample_id"]}/fixed_frame.png' for r in p['rows'])]
    save(destination/'complete.json',dict(complete=True,**binding,
        artifacts={n:digest(destination/n) for n in names}))


def main():
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=sender_data.ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--wait',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--replot-to',type=Path,help='CPU-only figures from completed results into a new directory')
    args=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    args.output=args.output or args.root/('evaluation_smoke' if args.smoke else 'evaluation')
    run=Run(SimpleNamespace(output=args.output/'queue',command='latent_system_review',max_hours=72))
    run.thread.start()
    try:
        complete=args.root/('smoke' if args.smoke else 'formal')/'complete.json'
        while not complete.exists():
            if not args.wait:raise RuntimeError('sender not complete')
            run.update(phase='wait_sender_no_GPU_lock');run.check();time.sleep(20)
        if args.replot_to:replot(args.output,args.replot_to,run)
        elif args.verify_only:verify_completed(args.output,run.check)
        else:
            with exclusive_native_evaluation(run):review(args,run)
        run.update(phase='system_review_complete')
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
