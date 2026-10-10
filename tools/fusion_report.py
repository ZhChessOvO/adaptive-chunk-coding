"""Read-only P1 audit, plots and fixed previews; never train or infer models."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace
import zlib

import numpy as np

from tools.latent_sender_report import read, digest, write

MODES = ('current', 'overlap', 'multiband', 'learned')
LABELS = ('Feather', 'Overlap', 'Multiband', 'Learned F')


def change(old, new):
    if old <= 0:
        raise ValueError('relative LPIPS comparison requires a positive baseline')
    return dict(delta=new-old, reduction_percent=100*(1-new/old))


def statistics(summary):
    result = {}
    for dataset in ('REDS', 'UVG'):
        group = summary['whole_frame'][dataset]
        rows = [r for r in summary['per_view'] if r['dataset'] == dataset]
        for mode in MODES:
            for metric in ('lpips_alex', 'psnr_db', 'temporal_delta_mae'):
                actual = np.mean([r['quality'][mode][metric] for r in rows])
                if not np.isclose(actual, group[mode][metric], rtol=0, atol=1e-12):
                    raise ValueError('group mean differs from per-view records')
        result[dataset] = dict(count=len(rows),
            multiband_vs_feather=change(group['current']['lpips_alex'], group['multiband']['lpips_alex']),
            learned_vs_feather=change(group['current']['lpips_alex'], group['learned']['lpips_alex']),
            learned_vs_multiband=change(group['multiband']['lpips_alex'], group['learned']['lpips_alex']),
            learned_better_than_multiband=sum(r['quality']['learned']['lpips_alex'] <
                r['quality']['multiband']['lpips_alex'] for r in rows),
            multiband_better_than_feather=sum(r['quality']['multiband']['lpips_alex'] <
                r['quality']['current']['lpips_alex'] for r in rows),
            F_seconds_cached_input_mean=float(np.mean([
                r['learned_seconds_cached_multiband_no_model_load'] for r in rows])),
            F_peak_allocated_bytes=max(r['learned_peak_cuda_allocated_bytes'] for r in rows))
    return result


def check_envelope(wire, original, mode, profile, checkpoint):
    if len(wire) != len(original)+77 or wire[77:] != original:
        raise ValueError('inner stream changed or uncharged bytes')
    magic, code, model, index = struct.unpack('<8s32s32sB', wire[:73])
    if (magic != b'RVLFUS01' or code.hex() != profile or index != MODES.index(mode)
            or struct.unpack('<I', wire[73:77])[0] != zlib.crc32(wire[:73])
            or model.hex() != (checkpoint if mode == 'learned' else '0'*64)):
        raise ValueError('fusion identity or CRC differs')


def verify(folder, done=None):
    done = done or read(folder/'complete.json')
    if not done['complete']:
        raise ValueError('incomplete input: '+str(folder))
    for name, expected in done.get('artifacts', {}).items():
        if digest(folder/name) != expected:
            raise ValueError('artifact changed: '+str(folder/name))
    return done


def figures(output, summary, validations, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#7b8494', '#d59c28', '#167dac', '#8a4daa')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, dataset in enumerate(('REDS', 'UVG')):
        for row, values in enumerate((summary['whole_frame'][dataset], summary['boundary'][dataset]['G_G'])):
            numbers = [values[m]['lpips_alex'] if row == 0 else values[m] for m in MODES]
            ax = axes[row, col]
            ax.plot(range(4), numbers, color='#c7cad0', zorder=1)
            for i, (value, color) in enumerate(zip(numbers, colors)):
                ax.scatter(i, value, color=color, s=60, zorder=2)
                ax.annotate(f'{value:.5f}', (i, value), xytext=(0, 9),
                            textcoords='offset points', ha='center', fontsize=10)
            ax.set_xticks(range(4), LABELS); ax.margins(x=.16, y=.3); ax.grid(axis='y', alpha=.2)
            ax.set_title(dataset + (' | whole frame' if row == 0 else ' | G/G boundary context'))
            ax.set_ylabel('LPIPS (lower is better)')
    fig.suptitle('Fixed width3, E cap 50%, identical packets and G candidates\n'
                 '6 REDS full views / 7 UVG crops; boundary crops = 128 px; NOT an RD curve')
    fig.savefig(output/'quality.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    names = [r['sample_id'].replace('reds-val-', 'REDS ').replace('-f000-n17-fullview', '')
             .replace('uvg-', 'UVG ').replace('-f000-historical-evaluation-crop', '') for r in summary['per_view']]
    for j, (reference, mode) in enumerate((('current', 'multiband'), ('multiband', 'learned'))):
        values = [r['quality'][mode]['lpips_alex']-r['quality'][reference]['lpips_alex'] for r in summary['per_view']]
        axes[j].barh(names, values, color=['#167dac' if v < 0 else '#c96645' for v in values])
        axes[j].axvline(0, color='black', linewidth=.8); axes[j].invert_yaxis()
        axes[j].set_xlabel('LPIPS difference (negative = better)'); axes[j].grid(axis='x', alpha=.2)
        axes[j].set_title(('Multiband minus feather', 'Learned F minus multiband')[j])
    fig.suptitle('All 13 reused diagnostic views | axes have different scales')
    fig.savefig(output/'per_view.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = sorted({h['epoch'] for h in history})
    axes[0].plot(epochs, [np.mean([h['lpips'] for h in history if h['epoch']==e]) for e in epochs])
    axes[0].set_ylabel('Mean training patch LPIPS'); axes[0].set_title('48 boundary patches per epoch')
    for mode, color in (('current', colors[0]), ('multiband', colors[2]), ('learned', colors[3])):
        axes[1].plot([v['epoch'] for v in validations],
            [np.mean([v['groups'][d][mode]['lpips_alex'] for d in ('REDS','UVG')]) for v in validations],
            marker='o', color=color, label=dict(zip(MODES,LABELS))[mode])
    selected = min(validations, key=lambda v:v['selection_score'])
    axes[1].scatter(selected['epoch'], selected['selection_score'], s=160, marker='*', color='black', zorder=5)
    axes[1].set_ylabel('Validation LPIPS'); axes[1].set_title('Equal REDS/UVG means; fixed 5 frames'); axes[1].legend()
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.suptitle('Only the 26,328-parameter fusion model trained | star = selected epoch 20\n'
                 'Training patches and whole-view validation are different metrics')
    fig.savefig(output/'training.png', dpi=150); plt.close(fig)


def previews(run, root, rows):
    from PIL import Image, ImageDraw, ImageFont
    from routervc.fusion.boundaries import edges
    records = []
    # Same first REDS/UVG as previous fixed previews, plus the identified failure case.
    selected = [rows[0], rows[6], next(r for r in rows if 'readysetgo' in r['sample_id'])]
    for row in selected:
        run.check(); sid = row['sample_id']; dest = run.root/'previews'/sid; dest.mkdir(parents=True, exist_ok=True)
        if (dest/'complete.json').exists():
            records.append(dict(sample_id=sid, **verify(dest))); continue
        cache = root/'p1_controls/samples'/sid
        if digest(Path(row['source_path'])) != row['source_sha256']:
            raise ValueError('source pixels changed')
        with np.load(row['source_path']) as z: source = z['source']
        with np.load(cache/'current.npz') as z: current = z['pixels']
        with np.load(cache/'controls/controls.npz') as z: multi = z['multiband']
        with np.load(root/'p1_evaluation/samples'/sid/'learned.npz') as z: learned = z['pixels']
        regions = read(cache/'received.json')['detail']['received_regions']
        generated = read(cache/'complete.json')['generated']
        boundaries = edges(source.shape, regions, generated)
        edge = next(e for e in boundaries if e['category']=='G_G' and 8 in e['frames'])
        pos = (edge['lo']+edge['hi'])//2
        x,y = (edge['pos']-48,pos-48) if edge['axis']=='x' else (pos-48,edge['pos']-48)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        frames = []
        for t in range(17):
            run.check(); canvas = Image.new('RGB', (1536, 438), 'white'); draw = ImageDraw.Draw(canvas)
            for j,(name,value) in enumerate(zip(('Source','Feather','Multiband','Learned F'),(source,current,multi,learned))):
                draw.text((j*384+8, 5), name, font=font, fill='black')
                crop = Image.fromarray(value[t,y:y+96,x:x+96]).resize((384,384), Image.Resampling.NEAREST)
                canvas.paste(crop, (j*384,54))
            draw.text((8,29), f'{sid} | first G/G edge | frame {t+1}/17 | 6 fps slow preview', font=font, fill='black')
            if t == 8: canvas.save(dest/'frame9.png')
            frames.append(np.asarray(canvas))
        target = dest/'preview.mp4'; temp = dest/'preview.partial.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x438',
            '-r','6','-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(temp)],input=np.stack(frames).tobytes(),check=True)
        temp.replace(target)
        done = dict(complete=True, crop=[x,y,96,96], edge=edge, preview_fps=6,
            source_fps_not_represented=True, metrics_from_lossless_pixels_not_preview=True,
            artifacts={n:digest(dest/n) for n in ('frame9.png','preview.mp4')})
        write(dest/'complete.json', done); records.append(dict(sample_id=sid, **done))
        run.update(phase='fixed_previews', sample=sid, completed=len(records))
    return records


def report(root, run):
    evaluation = root/'p1_evaluation'
    binding = dict(handoff=digest(root/'p1_handoff/complete.json'),
        code={n:digest(Path(__file__).resolve().parents[1]/n) for n in
            ('tools/fusion_report.py','tools/latent_sender_report.py','routervc/fusion/boundaries.py')})
    if (run.root/'complete.json').exists():
        done = verify(run.root)
        if done['binding'] != binding: raise ValueError('report binding changed; use a new output directory')
        print('FUSION_REPORT_VERIFIED', flush=True); return
    from demo.routervc_fullview_probe import immutable
    immutable(run.root/'protocol.json', binding)
    for stage in ('p1_fit', 'p1_evaluation', 'p1_boundary_learned', 'p1_handoff'):
        verify(root/stage)
    summary = read(root/'p1_handoff/summary.json')
    for item in summary['upstream'].values():
        if digest(Path(item['path'])) != item['sha256']: raise ValueError('upstream binding changed')
    protocol = read(evaluation/'protocol.json'); records = read(evaluation/'summary.json')['results']
    old_root = Path('/root/autodl-fs/DCVC/runs/routervc_latent_sender_20261009/evaluation')
    prefix_pairs = 0
    for i, row in enumerate(records):
        run.check(); sid = row['sample_id']; folder = evaluation/'samples'/sid
        verify(folder, row)
        original = (old_root/'samples'/sid/'source_e050/stream.rvlrg').read_bytes()
        for mode in MODES:
            wire = (folder/f'{mode}.rvlf').read_bytes()
            check_envelope(wire, original, mode, protocol['fusion_profile'], protocol['checkpoint_sha256'])
            if len(wire) != row['actual_bytes'] or row['additional_mask_bytes'] != 0:
                raise ValueError('byte accounting differs')
        header = (folder/'learned.rvlf').read_bytes()[:77]
        prefixes = [header+(old_root/'samples'/sid/f'source_e{cap:03d}/stream.rvlrg').read_bytes()
                    for cap in (0,25,50,100)]
        if any(not b.startswith(a) for a,b in zip(prefixes,prefixes[1:])):
            raise ValueError('literal prefix broken')
        prefix_pairs += 3
        run.update(phase='audit_existing_files', completed=i+1, total=len(records), sample=sid)
    audit = read(evaluation/'fresh_audit.json')
    for r in audit['receipts']:
        folder = evaluation/'samples'/r['sample_id']/'fresh'/r['variant']
        if digest(folder/'complete.json') != r['receipt']: raise ValueError('fresh receipt changed')
        receipt = verify(folder)
        if receipt['source_frames_read'] or receipt['sender_router_loaded'] or receipt['additional_mask_bytes']:
            raise ValueError('source-free contract changed')
    validations = [read(p) for p in sorted((root/'p1_fit').glob('validation_0*.json'))]
    if min(validations,key=lambda v:v['selection_score'])['epoch'] != summary['training']['selected_epoch']:
        raise ValueError('selected checkpoint is not the prescribed validation minimum')
    history = [json.loads(line) for line in (root/'p1_fit/train.jsonl').read_text().splitlines()]
    if [h['step'] for h in history] != list(range(1,2881)):
        raise ValueError('unexpected optimizer history; inspect before reporting')
    figures(run.root, summary, validations, history)
    movies = previews(run, root, protocol['rows'])
    stages = {}
    for folder in ('p1_data','p1_fit','p1_evaluation/queue','p1_handoff'):
        events = [json.loads(line) for line in (root/folder/'heartbeat.jsonl').read_text().splitlines()]
        active = [e for e in events if e.get('phase') and not e['phase'].startswith('waiting_')]
        stages[folder] = dict(first_active_utc=active[0]['utc'], last_active_utc=active[-1]['utc'],
            active_sample_span_seconds=(datetime.fromisoformat(active[-1]['utc'])-
                                       datetime.fromisoformat(active[0]['utc'])).total_seconds(),
            sampled_device_peak_MiB=max(int(e['gpu'].split(',')[2]) for e in active),
            note='30-second device-wide samples; phase start approximate, not framework peak')
    result = dict(complete=True, binding=binding, statistics=statistics(summary), stages=stages,
        audited_paired_views=len(records), audited_stream_variants=len(records)*4,
        literal_prefix_pairs=prefix_pairs, fresh_receipts_verified=len(audit['receipts']),
        training_resume_verified=read(root/'p1_fit/resume_verified.json'),
        sourcefree_checks=audit, previews=movies, model_promoted=False,
        inference_executed=False, quality_metrics_recomputed=False,
        scope='Existing P1 diagnostic results only; no full RD sweep or independent unseen benchmark',
        training=summary['training'], final_validation=read(root/'p1_fit/validation_final_full17.json')['groups'])
    write(run.root/'summary.json', result)
    names=['protocol.json','summary.json','quality.png','per_view.png','training.png']
    for m in movies:
        names += [f'previews/{m["sample_id"]}/{n}' for n in ('complete.json','frame9.png','preview.mp4')]
    write(run.root/'complete.json', dict(complete=True,binding=binding,artifacts={n:digest(run.root/n) for n in names}))
    run.update(phase='report_complete'); run.log_resources()


def main():
    from demo.chunk_enhancement_experiment import Run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-fs/DCVC/runs/routervc_fusion_20261010'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run=Run(SimpleNamespace(output=args.output or args.root/'report_20261010',command='report',max_hours=2))
    run.thread.start()
    try: report(args.root,run)
    except BaseException as error:
        write(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally: run.stop.set();run.thread.join();run.lock.close()


if __name__ == '__main__': main()
