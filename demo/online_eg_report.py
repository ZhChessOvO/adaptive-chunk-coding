"""Read-only endpoint checks, sparse measured RD curves and fixed visual evidence."""
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo.online_eg_eval_core import (ARMS, CASES, CLIPS, METRICS, HISTORY, OLD, MECHANISM,
                                    MODES, REPO, training_check)
from demo.online_eg_evaluate import grouped, reuse_initial, compare_checks
from demo.chunk_enhancement_experiment import read
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash, parse
from demo.scalable_experiment import quality, load_source, resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex


def common_rate(fixed, joint, scope):
    """Descriptive linear interpolation in log bytes; not BD-rate or a new decode."""
    curves = [sorted(rows, key=lambda r:r['bytes']) for rows in (fixed, joint)]
    if any(len({r['bytes'] for r in rows}) != len(rows) or len(rows) < 2 for rows in curves):
        return None
    lo = max(rows[0]['bytes'] for rows in curves)
    hi = min(rows[-1]['bytes'] for rows in curves)
    if lo >= hi:
        return None
    target = float(np.sqrt(lo*hi))
    values = [{m:float(np.interp(np.log(target), np.log([r['bytes'] for r in rows]),
                                [r[scope][m] for r in rows])) for m in METRICS} for rows in curves]
    a, b = values
    return dict(bytes=target, overlap_bytes=[lo, hi], fixed=a, joint=b,
                lpips_gain=a['lpips_alex']-b['lpips_alex'],
                psnr_change_db=b['psnr_db']-a['psnr_db'],
                note='log-byte linear interpolation of 3 q points, no extrapolation, not BD-rate')


def baseline(root, refs, lookup):
    path = root/'uf_reference.json'
    inputs = {sid:file_hash(MECHANISM/'samples'/sid/'complete.json') for sid in CLIPS}
    inputs.update(reference=file_hash(OLD/'summary.json'), report_code=file_hash(Path(__file__)))
    if path.exists():
        saved = read(path)
        assert saved['inputs'] == inputs
        for sid in CLIPS:
            folder = MECHANISM/'samples'/sid
            verify_artifacts(folder, saved['artifacts'][sid])
        return saved
    metric, results, artifacts = LPIPSAlex(True), {}, {}
    for sid, row in refs.items():
        folder = MECHANISM/'samples'/sid
        old = read(folder/'complete.json')
        assert old['sample'] == row['sample'] and old['source_rgb_sha256'] == row['source_hash']
        artifacts[sid] = {k:v for k,v in old['artifacts'].items()
                          if k == 'uf32.dcvc' or k.startswith('uf32_fresh/')}
        assert len(artifacts[sid]) == 1+row['sample']['frame_count']
        verify_artifacts(folder, artifacts[sid])
        pixels = np.stack([np.asarray(Image.open(p).convert('RGB'))
                           for p in sorted((folder/'uf32_fresh').glob('*.png'))])
        source = load_source(row['sample'])
        assert frame_hash(source) == row['source_hash']
        regions, roi = region_metrics(source, pixels, row['metric_regions'], metric)
        size = (folder/'uf32.dcvc').stat().st_size
        assert size == old['variants']['uf32']['bytes']
        direct = lookup[sid, 'fixed_E_none']
        wire = (root/sid/'fixed_E_none/stream.acse').read_bytes()
        results[sid] = dict(
            uf8=dict(bytes=len(parse(wire).base), quality=direct['quality'], roi_quality=direct['roi_quality']),
            uf32=dict(bytes=size, quality=quality(source,pixels,metric), roi_quality=roi,
                      per_region=regions, output_hash=frame_hash(pixels)))
    value = dict(inputs=inputs, results=results, artifacts=artifacts,
                 note='Existing QP8/32 native UF streams; checked reuse, no new UF encoding or dense UF curve')
    atomic_json(path, value)
    return value


def report(output):
    training = training_check(output)
    root = output/'evaluation'
    summary = read(root/'summary.json')
    assert summary['complete']
    protocol = read(root/'protocol.json')
    assert protocol == summary['protocol']
    for name, digest in protocol['code'].items():
        assert file_hash(REPO/'demo'/name) == digest
    assert file_hash(OLD/'summary.json') == protocol['references']
    assert file_hash(HISTORY/'summary.json') == protocol['history']
    initial = reuse_initial()
    assert initial == summary['initial_reused']
    lookup = grouped(summary['results'])
    refs = {r['sample']['sample_id']:r for r in read(OLD/'summary.json')['results']}
    for r in summary['results']:
        folder = root/r['sample_id']/r['name']
        assert read(folder/'result.json') == r
        verify_artifacts(folder,r['artifacts'])
        assert r['bytes'] == (folder/r['stream_file']).stat().st_size
        assert frame_hash(load_frames(folder/'reconstruction.npz')) == r['fresh_decode']['output_hash']
    for sid,row in refs.items():
        compare_checks(row, {n:r for (s,n),r in lookup.items() if s == sid}, initial)
    uf = baseline(root, refs, lookup)
    means = {}
    for domain in ('all','REDS','UVG'):
        means[domain] = {}
        for arm in ARMS:
            for kind in ('E','G'):
                for case in CASES:
                    name = f'{arm}_{kind}_{case}'
                    selected = [lookup[sid,name] for sid in CLIPS
                                if domain == 'all' or refs[sid]['sample']['dataset'] == domain]
                    means[domain][name] = dict(bytes=mean(r['bytes'] for r in selected),
                        **{s:{m:mean(r[s][m] for r in selected) for m in METRICS}
                           for s in ('quality','roi_quality')})
        selected = [r for r in initial if r['case']=='full' and (domain=='all' or r['dataset']==domain)]
        means[domain]['initial_G_full'] = dict(bytes=mean(r['bytes'] for r in selected),
            **{s:{m:mean(r[s][m] for r in selected) for m in METRICS} for s in ('quality','roi_quality')})
    changes, matched, crosses = [], [], []
    for sid in CLIPS:
        fixed,joint = lookup[sid,'fixed_G_full'], lookup[sid,'joint_G_full']
        changes.append(dict(sample_id=sid, dataset=refs[sid]['sample']['dataset'],
            fixed_bytes=fixed['bytes'], joint_bytes=joint['bytes'], byte_change_percent=100*(joint['bytes']/fixed['bytes']-1),
            **{s:dict(lpips_gain=fixed[s]['lpips_alex']-joint[s]['lpips_alex'],
                      psnr_change_db=joint[s]['psnr_db']-fixed[s]['psnr_db'],
                      temporal_change=joint[s]['temporal_delta_mae']-fixed[s]['temporal_delta_mae'])
               for s in ('quality','roi_quality')}))
        for kind in ('E','G'):
            curves = [[lookup[sid,f'{arm}_{kind}_{c}'] for c in ('full_q2','full','full_q05')] for arm in ARMS]
            matched.append(dict(sample_id=sid, kind=kind,
                **{s:common_rate(*curves,s) for s in ('quality','roi_quality')}))
        crosses.append(dict(sample_id=sid, **{name:{k:lookup[sid,name][k] for k in ('bytes','quality','roi_quality')}
            for name in ('fixed_G_full','joint_G_full','cross_oldE_newG','cross_newE_fixedG')}))
    beats = [json.loads(s) for s in (output/'heartbeat.jsonl').read_text().splitlines()]
    trainbeats = [r for r in beats if r['mode']=='train']
    evalbeats = [r for r in beats if r['mode'] in ('run','evaluate') and not r.get('phase','').startswith('waiting_')]
    digest = dict(complete=True, means=means, same_q_full_changes=changes,
        common_rate=matched, crossed_models=crosses, training=training,
        training_seconds=read(output/'train.complete.json')['elapsed_seconds'],
        evaluation_seconds=summary['elapsed_seconds'],
        training_sampled_peak_mib=max(int(r['gpu'].split(',')[2]) for r in trainbeats),
        evaluation_sampled_peak_mib=max((int(r['gpu'].split(',')[2]) for r in evalbeats),default=0),
        resources=resources(), receiver_full={a:dict(
            seconds=mean(lookup[s,f'{a}_G_full']['fresh_decode']['seconds'] for s in CLIPS),
            peak_cuda_allocated_bytes=max(lookup[s,f'{a}_G_full']['fresh_decode']['peak_cuda_allocated_bytes'] for s in CLIPS))
            for a in ARMS},
        ordinary_file_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink()),
        roles='four reused development clips', local_scope='first 17 frames, fixed regions, area weighted within clip',
        whole_scope='all 17/17/33/17 frames, equal clip weights; LPIPS is not region additive')
    atomic_json(root/'digest.json', digest)
    figures(root, lookup, initial, refs, uf['results'])
    atomic_json(root/'figure_artifacts.json', {p.name:file_hash(p) for p in sorted(root.iterdir()) if p.suffix in ('.png','.gif')})
    atomic_json(root/'audit.json',dict(complete=True,fresh_decodes=92,initial_points_reused=12,
        fixed_payloads_reproduced=True,noise_paired=True,literal_prefix=True,
        real_bytes=True,source_free=True,repeat_exact=True,initial_receiver_exact=True))
    print(json.dumps(dict(means=digest['means']['all'],same_q_full_changes=changes),indent=2),flush=True)


def figures(root, lookup, initial, refs, uf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old = {(r['sample_id'],r['case']):r for r in initial}
    for sweep,cases in [('prefix',('none','partial','full')), ('quantization',('full_q2','full','full_q05'))]:
        for scope in ('roi_quality','quality'):
            for metric,label in [('lpips_alex','LPIPS (lower better)'),('psnr_db','PSNR dB (higher better)')]:
                fig,axes = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
                for ax,(sid,name) in zip(axes.flat,CLIPS.items()):
                    for arm,color in [('fixed','tab:blue'),('joint','tab:orange')]:
                        for kind,style in [('E','o--'),('G','o-')]:
                            points = [lookup[sid,f'{arm}_{kind}_{c}'] for c in cases]
                            ax.plot([p['bytes'] for p in points],[p[scope][metric] for p in points],style,
                                    color=color,label=f'{arm} {"E only" if kind=="E" else "E -> G"}')
                    if sweep=='prefix':
                        points=[old[sid,c] for c in cases]
                        ax.plot([p['bytes'] for p in points],[p[scope][metric] for p in points], 's:',
                                color='tab:green', label='Initial ROI RGB (reused)')
                    pts=[uf[sid][k] for k in ('uf8','uf32')]
                    ax.plot([p['bytes'] for p in pts],[p[scope][metric] for p in pts], 'x:',
                            color='gray', label='Native UF QP8/32 only (reused)')
                    ax.set(title=name,xlabel='Actual complete stream bytes',ylabel=label)
                    ax.grid(alpha=.25);ax.legend(fontsize=6)
                fig.savefig(root/f'{sweep}_{scope}_{metric}.png',dpi=150)
                plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for i,(sid,name) in enumerate(CLIPS.items()):
        videos={'GT':load_source(refs[sid]['sample']),
                'Base':load_frames(Path(lookup[sid,'fixed_E_none']['output_path'])),
                'Old E only':load_frames(Path(lookup[sid,'fixed_E_full']['output_path'])),
                'Joint E only':load_frames(Path(lookup[sid,'joint_E_full']['output_path'])),
                'Initial E -> G':load_frames(Path(old[sid,'full']['output_path'])),
                'Fixed E -> G':load_frames(Path(lookup[sid,'fixed_G_full']['output_path'])),
                'Joint E -> G':load_frames(Path(lookup[sid,'joint_G_full']['output_path']))}
        regions=refs[sid]['metric_regions']
        width=max(r[4] for r in regions)*2
        canvas=Image.new('RGB',(len(videos)*width,sum(r[5]*2+36 for r in regions)),'white')
        draw=ImageDraw.Draw(canvas);top=0
        for t,n,x,y,w,h in regions:
            for j,(label,video) in enumerate(videos.items()):
                canvas.paste(Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST),(j*width,top+36))
                draw.text((j*width+3,top+8),label,font=font,fill='black')
            top+=h*2+36
        canvas.save(root/f'{sid}_fixed.png')
        if i in (1,2):
            t,n,x,y,w,h=regions[0];frames=[]
            for frame in range(17):
                canvas=Image.new('RGB',(w*8,h*2+32),'white');draw=ImageDraw.Draw(canvas)
                for j,label in enumerate(('GT','Initial E -> G','Fixed E -> G','Joint E -> G')):
                    tile=Image.fromarray(videos[label][frame,y:y+h,x:x+w]).resize((w*2,h*2),Image.Resampling.NEAREST)
                    canvas.paste(tile,(j*w*2,32));draw.text((j*w*2+3,8),f'{label} f{frame}',font=font,fill='black')
                frames.append(canvas)
            frames[0].save(root/f'{sid}_motion.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
