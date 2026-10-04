"""Paired zero-E / regional-G8 ablation using completed visual Router weights.

Both arms and all 13 existing windows; no training or hyperparameter search.
E25/E50 remain the original measured files. Zero-E uses the identical G policy
header so it must be a literal prefix of each corresponding original stream.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
from demo.routervc_visual_report import ROOT,GROUPS,METRICS,mean


def zero_plan(protocol):
    return [(e,dict(name=f'{arm}_e0_g8',kind='router',arm=arm,ratio=0.,max_g=8))
            for e in protocol['sources'] for arm in ('global_local','local')]


def link_verified(target,source):
    source=source.resolve()
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve()!=source:
            raise ValueError('reused input link changed')
    else:target.symlink_to(source,target_is_directory=source.is_dir())


def original_matches_prefix(original,point,folder):
    low=(folder/point['name']/'stream.rtvc').read_bytes()
    for ratio in (.25,.5):
        high=(original/f"{point['arm']}_e{ratio:g}_g8"/'stream.rtvc').read_bytes()
        if not high.startswith(low):raise ValueError('zero-E stream is not an original literal prefix')


def finish(root,original,records):
    formal=read(original/'summary.json')
    rows=[]
    for rec in records:
        source=next(r for r in formal['rows'] if r['sample_id']==rec['sample_id'])
        rows.append(dict(sample_id=rec['sample_id'],group=source['group'],sequence=source['sequence'],
            point=rec['point']['name'],bpp=rec['bpp'],bytes=rec['bytes'],**rec['quality'],
            receiver_seconds=rec['decode']['seconds']))
    rows += [r for r in formal['rows'] if r['point'].endswith('_g8') or r['point']=='wholeframe_g_one_roi']
    groups={}
    for group in GROUPS:
        groups[group]={}
        for name in dict.fromkeys(r['point'] for r in rows):
            selected=[r for r in rows if r['group']==group and r['point']==name]
            groups[group][name]=dict(windows=len(selected),**mean(selected,('bpp',*METRICS,'receiver_seconds')))
    value=dict(complete=True,points=26,rows=rows,group_means=groups,no_model_promotion=True,
        original_measured_points_unchanged=True,zero_E_literal_prefix_pairs=52,
        scope='same receiver G8 budget and model, actual G cells recomputed from B versus received Y')
    save(root/'summary.json',value)
    files=['summary.json']+[str((root/'samples'/r['sample_id']/r['point']['name']/'result.json').relative_to(root)) for r in records]
    save(root/'complete.json',dict(complete=True,points=26,artifacts={n:digest(root/n) for n in files}))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'visual_evaluation_recovered')
    p.add_argument('--output',type=Path,default=ROOT/'visual_zero_E')
    args=p.parse_args(argv)
    if not os.environ.get('TMUX'):raise RuntimeError('zero-E probe requires tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('new output must be separate')
    from demo import routervc_visual_evaluate as evaluation
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    old=read(args.root/'protocol.json')
    verify_artifacts(args.root,read(args.root/'complete.json')['artifacts'])
    if old['code']!=evaluation.code_hashes():raise ValueError('formal source pins changed')
    if evaluation.completed_models(ROOT/'visual_router')!=old['models']:raise ValueError('formal models changed')
    run=Run(SimpleNamespace(output=args.output,command='zero_E',max_hours=4));run.thread.start()
    previous=os.environ.get('ROUTERVC_VISUAL_PARENT')
    try:
        protocol=dict(old,schema='routervc-visual-zero-E-probe-v1',new_code_sha256=digest(__file__),
            source_evaluation=str(args.root.resolve()),source_complete_sha256=digest(args.root/'complete.json'),
            purpose='separate regional G without E from E25/50 using same G8 policy',new_points=26)
        immutable(run.root/'protocol.json',protocol)
        complete=run.root/'complete.json'
        verify=complete.exists()
        if verify:verify_artifacts(run.root,read(complete)['artifacts'])
        records=[];metric=None
        # Completed replay uses only validation; GPU mutex is unnecessary there.
        from contextlib import nullcontext
        with nullcontext() if verify else exclusive_native_evaluation(run):
            os.environ['ROUTERVC_VISUAL_PARENT']=str(os.getpid())
            for index,(entry,point) in enumerate(zero_plan(old)):
                run.check();sample=entry['sample'];source=args.root/'samples'/sample['sample_id']
                folder=run.root/'samples'/sample['sample_id'];folder.mkdir(parents=True,exist_ok=True)
                if digest(source/'source.npz')!=read(source/'source.complete.json')['artifacts']['source.npz']:
                    raise ValueError('original input changed')
                link_verified(folder/'source.npz',source/'source.npz')
                link_verified(folder/'prepared',source/'prepared')
                if verify:
                    result=evaluation.validate_result(run.root,folder,point,sample,protocol)
                else:
                    result,metric=evaluation.evaluate_point(run,folder,point,sample,protocol,metric)
                original_matches_prefix(source,point,folder)
                records.append(result);run.update(completed=index+1,total=26,sample=sample['sample_id'])
        if not verify:finish(run.root,args.root,records)
        print('ZERO_E_VERIFIED' if verify else 'ZERO_E_COMPLETE',flush=True)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        if previous is None:os.environ.pop('ROUTERVC_VISUAL_PARENT',None)
        else:os.environ['ROUTERVC_VISUAL_PARENT']=previous
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':main()
