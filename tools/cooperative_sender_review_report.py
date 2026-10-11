"""P2b audit and fixed-view RD figures. Scientific discussion belongs in Notion."""
import json
from pathlib import Path

import numpy as np

from demo.routervc_fullview_probe import read, save, digest, verify_artifacts
from routervc.cooperation import sender_data as data
from tools.latent_boundary_report import EVALUATION


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def region_mask(regions, *, packet_coordinates=False):
    """E receipts store (chunk, region), whereas local G stores region alone."""
    indices = []
    for value in regions:
        if packet_coordinates:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError('E positions must be chunk/region pairs')
            chunk, index = value
            if not isinstance(chunk, int) or chunk < 0: raise ValueError('invalid chunk')
        else:
            index = value
        if not isinstance(index, int) or not 0 <= index < 16: raise ValueError('invalid region')
        indices.append(index)
    mask = np.zeros(16)
    mask[sorted(set(indices))] = 1
    return mask.reshape(4, 4)


def training_report(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    training = data.ROOT/'formal/router'
    done = read(training/'complete.json'); verify_artifacts(training, done['artifacts'])
    history = read(training/'history.json'); initial = read(training/'initial_validation.json')
    if [h['epoch'] for h in history] != list(range(1, 121)): raise ValueError('incomplete epoch history')
    if (root/'training_audit.json').exists():
        saved = read(root/'training_audit.json')
        if saved['training_complete'] != digest(training/'complete.json'): raise ValueError('training changed')
        if digest(root/'training_diagnostic.png') != saved['figure_sha256']: raise ValueError('training figure changed')
        return saved
    values = {}
    for arm in ('source', 'zero_source'):
        selected = done['selected'][arm]
        values[arm] = dict(epoch=selected['epoch'], checkpoint_sha256=digest(training/arm/'best.pt'),
            initial=initial[arm], selected=selected['validation'], last=history[-1]['validation'][arm],
            selected_minus_initial_regret=selected['validation']['regret']-initial[arm]['regret'],
            initial_was_checkpoint_candidate=False)
    beats = [json.loads(line) for line in (data.ROOT/'queue/heartbeat.jsonl').read_text().splitlines()]
    memory = [json.loads(line) for line in (data.ROOT/'queue/memory.jsonl').read_text().splitlines()]
    if beats[-1]['phase'] != 'complete_no_automatic_evaluation_or_promotion': raise ValueError('queue not complete')
    sampled = [float(b['gpu'].split(',')[2]) for b in beats if isinstance(b.get('gpu'),str) and b['gpu'].count(',')==4]
    plt = pyplot(); fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, group in zip(axes, ('Dataset-equal mean', 'REDS', 'UVG')):
        for arm, label, color in [('source', 'Source-visible', '#146aab'), ('zero_source', 'Zero-source control', '#da7820')]:
            def value(v): return v['regret'] if group=='Dataset-equal mean' else v['groups'][group]['regret']
            ax.plot([h['epoch'] for h in history], [value(h['validation'][arm])*1000 for h in history],
                    color=color, linewidth=1.2, label=label)
            ax.axhline(value(initial[arm])*1000, color=color, linestyle=':', alpha=.6)
            epoch = values[arm]['epoch']
            ax.scatter([epoch], [value(values[arm]['selected'])*1000], s=60, color=color, edgecolor='k', zorder=5)
        ax.set_title(group); ax.set_xlabel('Epoch'); ax.set_ylabel('Candidate regret x 1000 (lower is better)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Training diagnostic, NOT RD | dots: preselected checkpoints; dotted lines: warm start')
    fig.savefig(root/'training_diagnostic.png', dpi=160); plt.close(fig)
    result = dict(complete=True, training_complete=digest(training/'complete.json'),
        epochs=120, updates_per_arm=done['updates_per_arm'], parameters_per_arm=done['parameters_per_arm'],
        arms=values, queue_seconds=beats[-1]['elapsed_seconds'], completion_utc=beats[-1]['utc'],
        gpu_sampled_peak_MiB=max(sampled), final_disks=beats[-1]['disks'],
        peak_rss_bytes=max(m['rss'] for m in memory), peak_cache_bytes=max(m['cache'] for m in memory),
        cgroup_failcnt_range=[min(m['failcnt'] for m in memory),max(m['failcnt'] for m in memory)],
        pilot_output_bytes=beats[-1]['pilot_output_bytes'],
        validation_scope='9 REDS windows + 3 HoneyBee crop windows, checkpoint selection; not independent test',
        limitations=['Three measured candidates per state, not a full oracle',
            'Source best trained epoch has worse aggregate regret than its warm start',
            'Do not reselect checkpoint after looking at actual RD',
            'Crop/full-view cause cannot be identified from different datasets alone'],
        figure_sha256=digest(root/'training_diagnostic.png'))
    save(root/'training_audit.json', result); return result


def aggregate(points):
    from tools.cooperative_sender_review import ARMS, RATIOS
    groups = []
    for cohort in ('diagnostic', 'grouped_validation'):
        for dataset in ('REDS', 'UVG'):
            for cap in (RATIOS if cohort=='diagnostic' else (None,)):
                subset = [p for p in points if p['cohort']==cohort and p['dataset']==dataset
                          and (cap is None or p['cap']==cap)]
                arms = {}; differences = {}
                expected = None
                for arm in ARMS:
                    rows = [p for p in subset if p['arm']==arm]
                    keys = {(p['sample_id'],p['cap']) for p in rows}
                    if not rows or len(keys)!=len(rows) or (expected is not None and keys!=expected):
                        raise ValueError('missing or duplicated matched sender arm')
                    expected = keys
                    arms[arm] = dict(samples=len(rows),
                        **{k:float(np.mean([p['quality'][k] for p in rows])) for k in
                           ('lpips_alex', 'psnr_db', 'temporal_delta_mae')},
                        **{k:float(np.mean([p[k] for p in rows])) for k in
                           ('bpp', 'actual_bytes', 'fresh_seconds', 'G_calls')},
                        max_GiB=max(p['peak_GiB'] for p in rows),
                        max_reserved_GiB=max(p['reserved_GiB'] for p in rows),
                        no_G={k:float(np.mean([p['received_quality'][k] for p in rows]))
                              for k in ('lpips_alex','psnr_db','temporal_delta_mae')})
                old = {(p['sample_id'],p['cap']):p for p in subset if p['arm']=='old_sender'}
                for arm in ARMS[1:]:
                    rows = [p for p in subset if p['arm']==arm]
                    delta = np.array([p['quality']['lpips_alex']-old[p['sample_id'],p['cap']]['quality']['lpips_alex'] for p in rows])
                    differences[arm] = dict(delta_lpips=float(delta.mean()),
                        delta_bpp=arms[arm]['bpp']-arms['old_sender']['bpp'],
                        improved=int(np.sum(delta < -1e-8)), worsened=int(np.sum(delta > 1e-8)),
                        ties=int(np.sum(np.abs(delta) <= 1e-8)))
                groups.append(dict(cohort=cohort, dataset=dataset, cap=cap, arms=arms, differences=differences))
    return groups


def report(root, protocol, points, check=lambda:None):
    from tools.cooperative_sender_review import ARMS, location, source_rgb, verified_file
    from tools.plot_latent_diagnostics import comparison_image
    if len(points)!=192 or len({(p['cohort'],p['sample_id'],p['cap'],p['arm']) for p in points})!=192:
        raise ValueError('incomplete actual stream evaluation')
    prior = read(EVALUATION/'summary.json')
    verified_file(EVALUATION/'summary.json', protocol['previous_summary'])
    native = [p for p in prior['points'] if p['arm']=='UF']
    for p in native:
        check(); folder = EVALUATION/'samples'/p['sample_id']/f'UF_q{p["budget"]}'
        verified_file(folder/'receive/complete.json', p['receipt_sha256'])
        receipt = read(folder/'receive/complete.json'); verify_artifacts(folder/'receive',receipt['artifacts'])
        if receipt['actual_bytes'] != p['actual_bytes'] or (folder/'native.bin').stat().st_size != p['actual_bytes']:
            raise ValueError('native reference no longer agrees with actual file')
    groups = aggregate(points)
    save(root/'summary.json', dict(complete=True, groups=groups, points=points,
        protocol=protocol, native_groups={d:prior['groups'][d]['UF'] for d in ('REDS','UVG')},
        caveats=['Same E-byte cap, not necessarily same actual bytes; use RD horizontal axis',
            'Diagnostic views and checkpoint-selection windows, not a full independent test set',
            'Exact-stream reused receipts retain their ORIGINAL timing and memory observations',
            'Fresh decoder time includes loading/decode/G/F, excludes saving and scoring',
            'Native UF timings have a different save boundary: no precise speedup claim',
            'No-G lines score saved received Y; two new source prefixes have real model-free fresh checks',
            'No semantic supervision, adaptive width or causal crop/full-frame test in this evaluation']))
    plt = pyplot(); fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    styles = {'old_sender':('Old Rs / P2 Rg','o--'), 'source':('P2b source / P2 Rg','s-'),
              'zero_source':('P2b zero-source / P2 Rg','^:')}
    for i,dataset in enumerate(('REDS','UVG')):
        rows = [g for g in groups if g['cohort']=='diagnostic' and g['dataset']==dataset]
        for j,metric in enumerate(('lpips_alex','psnr_db')):
            ax = axes[i,j]
            for arm,(label,style) in styles.items():
                ax.plot([g['arms'][arm]['bpp'] for g in rows], [g['arms'][arm][metric] for g in rows], style, label=label)
            ax.plot([g['arms']['source']['bpp'] for g in rows],
                    [g['arms']['source']['no_G'][metric] for g in rows], 'x--', color='gray', label='P2b source; no G')
            uf = list(prior['groups'][dataset]['UF'].values())
            ax.plot([v['bpp'] for v in uf], [v[metric] for v in uf], 'k.-', label='DCVC-UF I32 / P sweep')
            ax.set_title('REDS resized full view (6)' if dataset=='REDS' else 'UVG existing crops (7)')
            ax.set_xlabel('Actual complete-stream bpp'); ax.set_ylabel(metric); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Sender cooperation | fixed width3 / P2 Rg / G / multiband | diagnostic scope')
    fig.savefig(root/'sender_rd.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax,cohort in zip(axes,('diagnostic','grouped_validation')):
        for dataset in ('REDS','UVG'):
            subset = [p for p in points if p['dataset']==dataset and p['cohort']==cohort]
            old = {(p['sample_id'],p['cap']):p for p in subset if p['arm']=='old_sender'}
            for arm,marker in (('source','o'),('zero_source','x')):
                ps = [p for p in subset if p['arm']==arm]
                ax.scatter([p['bpp']-old[p['sample_id'],p['cap']]['bpp'] for p in ps],
                    [p['quality']['lpips_alex']-old[p['sample_id'],p['cap']]['quality']['lpips_alex'] for p in ps],
                    marker=marker, label=dataset+' '+arm, alpha=.8)
        ax.axhline(0,color='k',linewidth=.6); ax.axvline(0,color='k',linewidth=.6)
        ax.set_title(cohort); ax.set_xlabel('New minus old bpp (left = fewer bytes)')
        ax.set_ylabel('New minus old LPIPS (down = better)'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(root/'sender_deltas.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax,key,title in zip(axes,('peak_GiB','fresh_seconds'),('Peak allocated GiB','Mean fresh decode seconds')):
        values=[]; labels=[]
        for dataset in ('REDS','UVG'):
            for arm in ARMS:
                ps=[p[key] for p in points if p['cohort']=='diagnostic' and p['dataset']==dataset and p['arm']==arm]
                values.append(max(ps) if key=='peak_GiB' else float(np.mean(ps))); labels.append(dataset+'\n'+arm)
        ax.bar(range(6),values); ax.set_xticks(range(6),labels,fontsize=7); ax.set_ylabel(title)
        ax.set_ylim(0,max(values)*1.16)
        for i,v in enumerate(values): ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Same receiver runtime boundary | exact-stream reuse retains original observations')
    fig.savefig(root/'sender_resources.png',dpi=160); plt.close(fig)
    names=['summary.json','sender_rd.png','sender_deltas.png','sender_resources.png']
    index={(p['cohort'],p['sample_id'],p['cap'],p['arm']):p for p in points}
    for job in protocol['jobs']:
        if job['cohort']=='diagnostic' and job['cap']!=.5: continue
        check(); sid=job['row']['sample_id']; folder=location(root,job)
        chosen={a:index[job['cohort'],sid,job['cap'],a] for a in ARMS}
        source=source_rgb(job['row']); panels=[('Source',source[8])]
        if job['cohort']=='diagnostic':
            uf=min([p for p in native if p['sample_id']==sid],key=lambda p:abs(p['bpp']-chosen['source']['bpp']))
            with np.load(EVALUATION/'samples'/sid/f'UF_q{uf["budget"]}'/'receive/pixels.npz') as z:
                panels.append((f'UF nearest: {uf["bpp"]:.4f} bpp',z['reconstruction'][8].copy()))
        for arm,p in chosen.items():
            with np.load(Path(p['receive'])/'pixels.npz') as z:
                panels.append((f'{arm}: {p["bpp"]:.4f} bpp / LPIPS {p["quality"]["lpips_alex"]:.4f}',z['reconstruction'][8].copy()))
        with np.load(Path(chosen['source']['receive'])/'pixels.npz') as z:
            panels.append(('New source received Y; no G',z['enhanced'][8].copy()))
        comparison_image(panels,folder/'fixed_frame.png',sid+' | frame 9 + fixed center crop | nearest UF is NOT rate-matched')
        fig,axes=plt.subplots(2,3,figsize=(9,5),constrained_layout=True)
        for col,(arm,p) in enumerate(chosen.items()):
            for row,key in enumerate(('E_regions','G_regions')):
                mask=region_mask(p[key],packet_coordinates=key=='E_regions')
                axes[row,col].imshow(mask,vmin=0,vmax=1,cmap='Blues' if row==0 else 'Greens')
                axes[row,col].set_title(arm+' '+key);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        fig.suptitle('E inferred from packets / G chosen locally | no action masks transmitted')
        fig.savefig(folder/'local_choices.png',dpi=140);plt.close(fig)
        names += [str((folder/n).relative_to(root)) for n in ('fixed_frame.png','local_choices.png')]
    return names
