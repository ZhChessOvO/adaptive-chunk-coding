"""Isolated QP40/48/56 native-UF rate-coverage supplement; old points immutable.

The legacy helper restricts its benchmark to four QPs, not the UF model. This
entrypoint expands that helper's allowed values only within this process and
restores them afterwards. It does not change a codec, checkpoint or old profile.
"""
from contextlib import contextmanager
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.routervc_visual_report import ROOT,GROUPS,METRICS,comparisons,figures,mean

EXTRA_QPS=(40,48,56)


@contextmanager
def extended_qps():
    from demo import routervc_baselines as baseline
    old=baseline.QPS
    try:
        baseline.QPS=tuple(sorted(set(old)|set(EXTRA_QPS)))
        yield baseline
    finally:
        baseline.QPS=old


def plan(protocol):
    return [(entry,dict(name=f'uf_qp{qp}',kind='uf',qp=qp))
            for entry in protocol['sources'] for qp in EXTRA_QPS]


def evaluate_one(run,entry,point,metric):
    import numpy as np
    from PIL import Image
    from demo.routervc_visual_evaluate import validate_result,validate_receiver
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_experiment import quality
    sample=entry['sample'];sid=sample['sample_id']
    sample_root=run.root/'samples'/sid;sample_root.mkdir(parents=True,exist_ok=True)
    folder=sample_root/point['name'];folder.mkdir(exist_ok=True)
    source=Path(run.original)/'samples'/sid/'source.npz'
    saved_source=read(source.parent/'source.complete.json')
    if digest(source)!=saved_source['artifacts']['source.npz']:
        raise ValueError('formal source cache changed')
    link=sample_root/'source.npz'
    if link.exists():
        if not link.is_symlink() or link.resolve()!=source.resolve():
            raise ValueError('supplement source link differs')
    else:
        link.symlink_to(source.resolve())
    protocol=read(run.root/'protocol.json')
    immutable(folder/'job.json',dict(point=point,protocol_sha256=digest(run.root/'protocol.json'),
                                    source_sha256=digest(source)))
    if (folder/'result.json').exists():
        with extended_qps():
            return validate_result(run.root,sample_root,point,sample,protocol),metric
    if (folder/'encode.json').exists():
        encoded=read(folder/'encode.json');verify_artifacts(folder,encoded['artifacts'])
        if encoded['source_hash']!=digest(source) or encoded['qp']!=point['qp']:
            raise ValueError('supplement encoder binding changed')
    else:
        execute(run,sid+'_'+point['name']+'_encode','routervc_uf_extension.py',
                ['worker','--operation','encode','--output',folder,'--source',source,
                 '--source-hash',digest(source),'--qp',point['qp']])
    if not (folder/'fresh/decode.json').exists():
        execute(run,sid+'_'+point['name']+'_fresh','routervc_uf_extension.py',
                ['worker','--operation','decode','--output',folder])
    with extended_qps():
        decoded,ledger=validate_receiver(sample_root,point,sample,protocol)
    if metric is None:
        import torch
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        torch.set_num_threads(4);metric=LPIPSAlex(True)
    run.update(phase='CPU_UF_metrics',sample=sid,point=point['name'])
    with np.load(source,allow_pickle=False) as z:original=z['source'].copy()
    with np.load(folder/'fresh/reconstruction.npz',allow_pickle=False) as z:output=z['reconstruction'].copy()
    measured=quality(original,output,metric)
    temporary=folder/'fixed_frame.png.tmp';Image.fromarray(output[8]).save(temporary,format='PNG')
    os.replace(temporary,folder/'fixed_frame.png')
    files=['job.json','encode.json','transmitted_meta.json','stream.bin',
           'fresh/decode.json','fresh/reconstruction.npz','fixed_frame.png']
    result=dict(complete=True,sample_id=sid,point=point,protocol_sha256=digest(run.root/'protocol.json'),
        decode=decoded,byte_ledger=ledger,bytes=ledger['bytes'],
        bpp=ledger['bytes']*8/(original.shape[0]*original.shape[1]*original.shape[2]),
        quality=measured,artifacts={n:digest(folder/n) for n in files})
    save(folder/'result.json',result)
    with extended_qps():
        return validate_result(run.root,sample_root,point,sample,protocol),metric


def finish(root,original,records):
    summary=read(original/'summary.json');rows=list(summary['rows'])
    for record in records:
        sid=record['sample_id'];ref=next(r for r in rows if r['sample_id']==sid)
        rows.append(dict(sample_id=sid,group=ref['group'],sequence=ref['sequence'],
            point=record['point']['name'],bytes=record['bytes'],bpp=record['bpp'],
            **record['quality'],receiver_seconds=record['decode']['seconds'],
            result_path=str(root/'samples'/sid/record['point']['name']/'result.json')))
    groups={}
    for group in GROUPS:
        groups[group]={}
        for name in dict.fromkeys(r['point'] for r in rows):
            subset=[r for r in rows if r['group']==group and r['point']==name]
            groups[group][name]=dict(windows=len(subset),**mean(subset,('bpp',*METRICS,'receiver_seconds')))
    value=dict(complete=True,new_UF_points=len(records),unchanged_Router_points=104,rows=rows,
        group_means=groups,comparison= comparisons(rows),
        rate_scope='native stream.bin only; helper audit JSON excluded, all RouterVC bytes charged',
        no_new_router_or_generator_inference=True,no_model_promotion=True)
    save(root/'summary.json',value);figures(value,root)
    files=['summary.json','rd_dataset_means.png','REDS_fullview_rd.png','UVG_crop_rd.png']
    files += [str((root/'samples'/r['sample_id']/r['point']['name']/'result.json').relative_to(root)) for r in records]
    save(root/'complete.json',dict(complete=True,points=len(records),artifacts={n:digest(root/n) for n in files}))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('run','worker'))
    p.add_argument('--root',type=Path,default=ROOT/'visual_evaluation_recovered')
    p.add_argument('--output',type=Path,default=ROOT/'visual_uf_rate_extension')
    p.add_argument('--operation',choices=('encode','decode'))
    p.add_argument('--source',type=Path);p.add_argument('--source-hash')
    p.add_argument('--qp',type=int,choices=EXTRA_QPS)
    args=p.parse_args(argv)
    if args.mode=='worker':
        if os.environ.get('ROUTERVC_UF_EXTENSION_PARENT') is None:
            raise RuntimeError('worker requires mutex-owning parent')
        with extended_qps() as baseline:
            if args.operation=='encode':
                baseline.encode_uf(SimpleNamespace(**vars(args),reuse_base=None))
            elif args.operation=='decode':baseline.decode_uf(args)
            else:raise ValueError('worker operation required')
        return
    if not os.environ.get('TMUX'):raise RuntimeError('UF supplement requires tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('supplement must not write into the formal evaluation')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.routervc_visual_evaluate import code_hashes
    old=read(args.root/'protocol.json');complete=read(args.root/'complete.json')
    verify_artifacts(args.root,complete['artifacts'])
    if old['code']!=code_hashes():raise ValueError('formal dependencies changed')
    run=Run(SimpleNamespace(output=args.output,command='uf_extension',max_hours=4))
    run.original=args.root;run.thread.start()
    previous=os.environ.get('ROUTERVC_UF_EXTENSION_PARENT')
    try:
        immutable(run.root/'protocol.json',dict(old,schema='routervc-UF-support-extension-v1',
            source_evaluation=str(args.root.resolve()),source_complete_sha256=digest(args.root/'complete.json'),
            extra_qps=list(EXTRA_QPS),new_code_sha256=digest(__file__)))
        done=run.root/'complete.json'
        if done.exists():
            verify_artifacts(run.root,read(done)['artifacts'])
            metric=None
            for entry,point in plan(old):
                _,metric=evaluate_one(run,entry,point,metric)
            print('UF_EXTENSION_VERIFIED_NO_RECOMPUTE',flush=True);return
        with exclusive_native_evaluation(run):
            os.environ['ROUTERVC_UF_EXTENSION_PARENT']=str(os.getpid())
            records=[];metric=None
            for i,(entry,point) in enumerate(plan(old)):
                run.check();record,metric=evaluate_one(run,entry,point,metric);records.append(record)
                run.update(completed=i+1,total=39)
        finish(run.root,args.root,records)
        print('UF_EXTENSION_COMPLETE',flush=True)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        if previous is None:os.environ.pop('ROUTERVC_UF_EXTENSION_PARENT',None)
        else:os.environ['ROUTERVC_UF_EXTENSION_PARENT']=previous
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':main()
