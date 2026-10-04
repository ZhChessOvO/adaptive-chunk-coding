"""Offline paired content diagnostics from completed B/E/G/EG teacher pixels.

Source-positive regions only; missing detections stay unknown. Automatic OCR
remains diagnostic, not character truth. No training, policy changes or masks.
"""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from demo import routervc_content_labels as content
from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts

ROOT=Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003')


def intersects(a,b):
    return min(a[0]+a[2],b[0]+b[2])>max(a[0],b[0]) and min(a[1]+a[3],b[1]+b[3])>max(a[1],b[1])


def positive_regions(coverage,regions):
    selected=set()
    for frame in coverage['frames']:
        for category in ('text_digits','face'):
            for item in frame[category]:
                if item.get('status')!='present':continue
                selected.update(r['region'] for r in regions if intersects(item['box'],r['roi']))
    return sorted(selected)


def diagnostics_summary(records):
    totals={}
    for category in ('text_digits','face'):
        counted=Counter();deltas=[]
        for record in records:
            roi=record['paired_roi']
            for frame in record['frames']:
                for box in frame[category]:
                    if box['status']!='present' or not intersects(box['box'],roi):continue
                    counted['source_positive_observations']+=1
                    for a,b in (('B','G'),('E','EG')):
                        states=box['states'];x,y=states[a],states[b]
                        key='normalized_landmark_error' if category=='face' else 'normalized_edit_error'
                        # Inspect provider key exactly; no absent values become zeros.
                        va,vb=x.get(key),y.get(key)
                        if va is None or vb is None:
                            counted[a+'_'+b+'_unknown']+=1;continue
                        delta=vb-va
                        counted[a+'_'+b+'_paired']+=1
                        counted[a+'_'+b+'_higher_error' if delta>0 else a+'_'+b+'_not_higher']+=1
                        deltas.append(dict(sample_id=record['sample_id'],region=record['paired_region'],
                            frame=frame['frame_index'],pair=a+'_'+b,delta=delta))
        totals[category]=dict(counts=dict(counted),paired_deltas=deltas,
            scope='OCR agreement with unverified pseudo-reference' if category=='text_digits' else 'detector five-point geometry, not identity')
    return dict(complete=True,regions=len(records),windows=len({r['sample_id'] for r in records}),
        train_windows=len({r['sample_id'] for r in records if r['source_role']=='train'}),
        validation_windows=len({r['sample_id'] for r in records if r['source_role']=='validation'}),
        categories=totals,trained=False,character_errors_trainable=False,labels_transmitted=False,
        counting='region/frame observations overlap and are not independent faces or text examples')


def paired_pixels(label,region):
    source_path=Path(label['source_path']);received=Path(label['received_path'])
    if digest(source_path)!=label['source_sha256'] or digest(received)!=label['received_sha256']:
        raise ValueError('teacher source or received E changed')
    with np.load(source_path,allow_pickle=False) as z:source=z['source'].copy()
    with np.load(received,allow_pickle=False) as z:
        base=z['base'].copy();enhanced=z['enhanced' if 'enhanced' in z else 'all_E'].copy()
    cell=received.parent/f'cell_{region:02d}';result=read(cell/'result.json')
    verify_artifacts(cell,result['artifacts'])
    with np.load(cell/'outputs.npz',allow_pickle=False) as z:
        candidates=dict(B=base,E=enhanced,G=z['G'].copy(),EG=z['EG'].copy())
    paired,roi=content.materialize_paired_region(source,candidates,region)
    return source,paired,roi,dict(source=digest(source_path),received=digest(received),
        cell_result=digest(cell/'result.json'),cell_outputs=digest(cell/'outputs.npz'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'content_candidate_review')
    args=p.parse_args(argv)
    if not os.environ.get('TMUX'):raise RuntimeError('content review requires tmux')
    from demo.chunk_enhancement_experiment import Run
    run=Run(SimpleNamespace(output=args.output,command='content_review',max_hours=4));run.thread.start()
    try:
        manifest=ROOT/'mixedview_teacher/labels.json'
        teacher=read(manifest)
        coverage=ROOT/'content_coverage_pilot120'
        verify_artifacts(coverage,read(coverage/'summary.json')['artifacts'])
        immutable(run.root/'request.json',dict(teacher_sha256=digest(manifest),
            coverage_sha256=digest(coverage/'summary.json'),code_sha256=digest(__file__),
            provider_sha256=digest(content.__file__),assets_sha256=digest(content.ASSETS/'assets.json'),
            no_training=True,positive_source_selection_only=True))
        jobs=[]
        for row in teacher['samples']:
            if row['router_split'] not in ('train','validation'):raise ValueError('evaluation data may not enter this review')
            if digest(row['path'])!=row['sha256']:raise ValueError('teacher label changed')
            label=read(row['path'])
            observed=read(coverage/'samples'/row['sample_id']/'coverage.json')
            jobs += [(row,label,region) for region in positive_regions(observed,label['regions'])]
        immutable(run.root/'plan.json',dict(jobs=[dict(sample_id=r['sample_id'],region=i,
            source_role=r['router_split']) for r,_,i in jobs],total=len(jobs)))
        provider=None;records=[]
        for index,(row,label,region) in enumerate(jobs):
            run.check();folder=run.root/'samples'/row['sample_id']/f'cell_{region:02d}'
            marker=folder/'complete.json';folder.mkdir(parents=True,exist_ok=True)
            binding=dict(request=digest(run.root/'request.json'),teacher_sha256=row['sha256'],region=region)
            if marker.exists():
                saved=read(marker)
                if saved['binding']!=binding:raise ValueError('content review binding changed')
                verify_artifacts(folder,saved['artifacts']);record=read(folder/'labels.json')
            else:
                source,candidates,roi,pixels=paired_pixels(label,region)
                if provider is None:provider=content.OpenCVProvider(content.ASSETS)
                record=content.assess_clip(source,candidates,provider,row['sample_id'],row['router_split'])
                record.update(paired_region=region,paired_roi=roi,binding=binding,pixel_binding=pixels,
                    region_targets=content.region_targets(record,[roi]),
                    outside_region='B for all states',training_started=False)
                save(folder/'labels.json',record)
                save(marker,dict(complete=True,binding=binding,artifacts={'labels.json':digest(folder/'labels.json')}))
                del source,candidates
            records.append(record)
            run.update(completed=index+1,total=len(jobs),sample=row['sample_id'],region=region)
        save(run.root/'summary.json',diagnostics_summary(records))
        save(run.root/'complete.json',dict(complete=True,regions=len(records),
            summary_sha256=digest(run.root/'summary.json'),no_training=True))
        print('CONTENT_REVIEW_COMPLETE',flush=True)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__=='__main__':main()
