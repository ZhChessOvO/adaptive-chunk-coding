"""Completion-only paired objective report; no inference or model promotion."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_json
from demo.scalable_experiment import now
from routervc.generation.context_geometry import rgb_crop
from tools.g_objective_trial import DEFAULT, verify
from tools.g_objective_trial_evaluate import indexed, NAMES
from tools.storage_guard import disk_snapshot
from tools.texture_context_probe import full_inputs
from tools.texture_context_report import plotting

COMPARISONS = ('control_minus_initial', 'relaxed_minus_control', 'relaxed_minus_initial')


def comparisons(cases):
    result = []
    for item in cases:
        rows = indexed(item)
        for size, scope in ((256, 'core128'), (512, 'core128'), (512, 'interior384')):
            values = {}
            for name in NAMES:
                row = rows[name, size]
                q = row['quality' if scope == 'core128' else 'wide_384_quality']
                values[name] = dict(lpips=q['lpips_alex'], psnr=q['psnr_db'],
                                    temporal_mae=q['temporal_delta_mae'])
                if scope == 'core128':
                    values[name]['phase'] = row['phase_error']['phase_rms_u8']
                if not all(np.isfinite(x) for x in values[name].values()):
                    raise ValueError('nonfinite score')
            record = dict(sample=item['case']['sample_id'], dataset=item['case']['dataset'],
                          group=item['case']['evaluation_group'], size=size, score_scope=scope, **values)
            for key, left, right in zip(COMPARISONS,
                    ('control', 'relaxed_rgb_temporal', 'relaxed_rgb_temporal'),
                    ('initial', 'control', 'initial')):
                record[key] = {m:values[left][m]-values[right][m] for m in values[left]}
            result.append(record)
    return result


def grouped(pairs):
    buckets = defaultdict(list)
    for row in pairs:
        buckets[row['group'], row['dataset'], row['size'], row['score_scope']].append(row)
    result = []
    for (group, dataset, size, scope), rows in sorted(buckets.items()):
        record = dict(group=group, dataset=dataset, size=size, score_scope=scope, count=len(rows))
        for name in (*NAMES, *COMPARISONS):
            record[name] = {m:float(np.mean([r[name][m] for r in rows])) for m in rows[0][name]}
        record['improvements'] = {name: {m:sum(r[name][m]*(1 if m == 'psnr' else -1)>0 for r in rows)
            for m in rows[0][name]} for name in COMPARISONS}
        result.append(record)
    return result


def charts(output, groups, pairs):
    plt = plotting()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for i, dataset in enumerate(('REDS', 'UVG')):
        row = next(g for g in groups if g['group']=='internal_validation' and
                   g['dataset']==dataset and g['score_scope']=='interior384')
        for j, (key, title) in enumerate((('lpips','LPIPS (lower)'), ('psnr','PSNR / dB (higher)'),
                                         ('temporal_mae','Temporal error (auxiliary)'))):
            ax = axes[i,j]; y = [row[n][key] for n in NAMES]
            ax.plot(range(3), y, 'o-', color='#2f6b95')
            for x, value in enumerate(y): ax.annotate(f'{value:.5f}', (x,value), xytext=(0,7), textcoords='offset points', ha='center', fontsize=9)
            ax.set_xticks(range(3), ['Start', '+400 / time 0.1', '+400 / time 0.025'], fontsize=8)
            ax.set_title(f'{dataset if dataset=="REDS" else "HoneyBee crops"} / {row["count"]} locations\n{title}')
            ax.margins(y=.3); ax.grid(alpha=.25)
    fig.suptitle('512px processing / separate 384px interior scores\nPreviously used internal locations; NOT full-video RD', fontsize=12)
    fig.savefig(output/'wide_scores.png', dpi=145); plt.close(fig)
    selected = [r for r in pairs if r['score_scope']=='interior384']
    fig, ax = plt.subplots(figsize=(11,7), layout='constrained')
    labels = [r['sample'].replace('-f000-n17-fullview','').replace('-historical-evaluation-crop','') for r in selected]
    y = np.arange(len(selected)); a = [r['control_minus_initial']['lpips'] for r in selected]
    b = [r['relaxed_minus_control']['lpips'] for r in selected]
    ax.barh(y-.18,a,height=.34,label='Additional training: control - start',color='#378aa3')
    ax.barh(y+.18,b,height=.34,label='Coefficient change: trial - control',color='#e19142')
    ax.set_yticks(y,labels,fontsize=8); ax.invert_yaxis(); ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('LPIPS change (negative is better) / same 384px interior')
    ax.set_title('Separate extra-training gains from the coefficient effect / all 15 fixed locations')
    ax.legend(fontsize=9); ax.grid(axis='x',alpha=.2)
    fig.savefig(output/'paired_lpips.png',dpi=140); plt.close(fig)


def temporal_contacts(output, item):
    source, _ = full_inputs(item['case']); rows = indexed(item)
    panels = [('Source',rgb_crop(source,item['geometry']['large']))]
    for name, label in zip(NAMES, ('Starting mixed400','Control +400','RGB time /4 +400')):
        with np.load(rows[name,512]['output_path']) as z: panels.append((label,z['pixels'].copy()))
    plt = plotting(); fig, axes = plt.subplots(3,4,figsize=(16,12),layout='constrained')
    for row, t in enumerate((0,8,16)):
        for col,(label,pixels) in enumerate(panels):
            ax=axes[row,col];ax.imshow(pixels[t],interpolation='nearest');ax.axis('off')
            ax.set_title(f'{label} / frame {t+1}',fontsize=10)
    fig.suptitle(item['case']['sample_id']+'\nSaved real frames at whole 512 view; stills do not establish absence of flicker',fontsize=11)
    fig.savefig(output/(item['case']['sample_id']+'_time.png'),dpi=105);plt.close(fig)


def resources(root, summary):
    def lines(path): return [json.loads(s) for s in path.read_text().splitlines()]
    hb, mem = lines(root/'heartbeat.jsonl'), lines(root/'memory.jsonl')
    inference = {}
    for size in (256,512):
        rows = [r['observation']['runtime'] for i in summary['cases'] for r in i['results']
                if r['size']==size and r['name']!='initial']
        inference[str(size)] = dict(mean_seconds=float(np.mean([r['seconds_model_load_excluded'] for r in rows])),
            peak_cuda_allocated_gib=max(r['peak_cuda_allocated_bytes'] for r in rows)/2**30,
            note='17-frame raw-G call; excludes loading, codec, routing and fusion; not real-time FPS')
    return dict(queue_seconds=hb[-1]['elapsed_seconds'],completed_utc=hb[-1]['utc'],
        arms={n:read(root/'formal'/n/'complete.json') for n in NAMES[1:]}, inference=inference,
        sampled_device_peak_gib=max(float(h['gpu'].split(',')[2])/1024 for h in hb),
        cgroup_peak_charged_gib=max(m['charged'] for m in mem)/2**30,
        cgroup_peak_rss_gib=max(m['rss'] for m in mem)/2**30,
        failcnt_start=mem[0]['failcnt'],failcnt_end=mem[-1]['failcnt'],
        memory_note='charged includes cache/shmem/other processes; not RSS; failcnt is not OOM-kill count',
        end_disks=hb[-1]['disks'], review_disks=disk_snapshot())


def main(args):
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    torch.set_num_threads(4)
    root=args.root; output=root/'report_20261016';output.mkdir(exist_ok=True)
    paths=[root/n for n in ('complete.json','evaluation/summary.json','heartbeat.jsonl','memory.jsonl')]+[Path(__file__)]
    inputs={str(p):digest(p) for p in paths}
    immutable(output/'protocol.json',dict(inputs=inputs,scope='read-only review; separate scoring areas; no model change'))
    if (output/'complete.json').exists():
        done=read(output/'complete.json');verify_artifacts(output,done['artifacts'])
        print(json.dumps(dict(complete=True,read_only_reentry=True)),flush=True);return
    verify(root);print('TRIAL_COMPLETION_AUDIT_PASSED',flush=True)
    summary=read(root/'evaluation/summary.json'); pairs=comparisons(summary['cases']); groups=grouped(pairs)
    atomic_json(output/'comparisons.json',dict(pairs=pairs,groups=groups,production_changed=False))
    atomic_json(output/'resources.json',resources(root,summary)); charts(output,groups,pairs)
    for item in summary['cases']:
        if item['case']['evaluation_group']!='known_artifact':continue
        temporal_contacts(output,item)
        src=root/'evaluation'/item['case']['sample_id']/'slow_preview.mp4'
        if src.stat().st_size>=5*2**20:
            dest=output/(item['case']['sample_id']+'_preview.mp4')
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),'-an',
                '-c:v','libx264','-threads','2','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
            if dest.stat().st_size>=5*2**20:raise ValueError('Notion workspace upload limit')
    names=[p for p in output.iterdir() if p.is_file() and p.name!='complete.json']
    atomic_json(output/'complete.json',dict(complete=True,production_changed=False,utc=now(),
        artifacts={p.name:digest(p) for p in names},verified_trial_receipt=digest(root/'complete.json')))
    print('REPORT_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    main(parser.parse_args())
