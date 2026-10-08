"""Fixed-E review after core adaptation: old/new R_g on 13 available views.

This is a receiver adaptation diagnostic, NOT a trained new sender result or a
full benchmark. Existing six REDS val full views and seven UVG crops are used;
no new downloads or manufactured full-frame UVG. Every point fresh-decodes.
"""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.latent import routing, router_data as data

OLD = Path('/root/autodl-fs/DCVC/runs/routervc_light_router_20261004/evaluation/protocol.json')
COUNTS = (0, 4, 8, 16)


def source_rows():
    old = read(OLD)
    result = []
    for entry in old['sources']:
        sample = entry['sample']; sid = sample['sample_id']
        original = old['inputs'][sid]
        if digest(original['source_path']) != original['source_sha256']:
            raise ValueError('diagnostic source changed')
        result.append(dict(sample_id=sid, dataset=sample['dataset'], sequence=sample['sequence'],
            view_kind=sample['view_kind'], source_path=original['source_path'],
            source_sha256=original['source_sha256']))
    if len(result) != 13:
        raise ValueError('expected six REDS val views and seven UVG crops')
    return result


def summarize(points):
    keys = {(p['sample_id'], p['arm'], p['count']) for p in points}
    samples = {p['sample_id']:p['dataset'] for p in points}
    if (len(keys) != len(points) or len(points) != 13*3*4
            or sum(d == 'REDS' for d in samples.values()) != 6
            or sum(d == 'UVG' for d in samples.values()) != 7):
        raise ValueError('expected complete unique 13-view paired results')
    for sid in samples:
        for count in COUNTS:
            values = [p for p in points if p['sample_id'] == sid and p['count'] == count]
            if len(values) != 3 or len({p['total_bytes'] for p in values}) != 1:
                raise ValueError('receiver arms must compare at identical actual bytes')
    result = {}
    for dataset in ('REDS', 'UVG'):
        result[dataset] = {}
        for arm in ('old_core', 'adapted_core', 'G_off'):
            result[dataset][arm] = {}
            for count in COUNTS:
                values = [p for p in points if p['dataset'] == dataset and p['arm'] == arm and p['count'] == count]
                if not values:
                    raise ValueError('incomplete receiver comparison')
                result[dataset][arm][str(count)] = dict(samples=len(values),
                    bpp=float(np.mean([p['bpp'] for p in values])),
                    **{k:float(np.mean([p['quality'][k] for p in values]))
                       for k in ('psnr_db','lpips_alex','temporal_delta_mae')},
                    mean_G_calls=float(np.mean([len(p['generated']) for p in values])),
                    peak_GiB=max(p['peak_cuda_allocated_bytes'] for p in values)/2**30,
                    fresh_seconds=float(np.mean([p['seconds'] for p in values])))
    return result


def plots(root, summary, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11,8), constrained_layout=True)
    for row, dataset in enumerate(('REDS','UVG')):
        for col, metric in enumerate(('lpips_alex','psnr_db')):
            ax = axes[row,col]
            for arm, marker in (('old_core','o--'),('adapted_core','s-'),('G_off','x:')):
                values = [summary[dataset][arm][str(c)] for c in COUNTS]
                ax.plot([v['bpp'] for v in values],[v[metric] for v in values],marker,label=arm)
            ax.set_xlabel('Actual whole-stream bpp'); ax.set_ylabel(metric)
            ax.set_title('REDS resized full views' if dataset == 'REDS' else 'UVG existing crops')
            ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Receiver adaptation only: identical E prefixes, same frozen G, up to G8\n'
                 'No new sender; these are previously used 13 diagnostics, not a full benchmark')
    fig.savefig(root/'receiver_rd.png',dpi=150); plt.close(fig)
    history = read(training/'router/history.json')
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    epochs = history['epochs']
    for ax,key in zip(axes,('regret','loss')):
        ax.plot([e['epoch'] for e in epochs],[e['validation'][key] for e in epochs],label='adaptation')
        ax.axhline(history['initial'][key],linestyle='--',color='gray',label='old core initialization')
        ax.set_xlabel('Epoch');ax.set_ylabel('Local G ranking '+key); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Router validation: dataset-equal, six-density-equal; not whole-video RD')
    fig.savefig(root/'receiver_training.png',dpi=150);plt.close(fig)


def run_review(args, run):
    from demo.chunk_enhancement_codec import configure_torch
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from demo.scalable_experiment import quality
    from tools.latent_router_queue import execute, verify
    from tools.verify_latent_routers import verify_stage
    training = args.training/'formal'
    verify(training)
    deep_verification = verify_stage(training)
    protocol = read(training/'protocol.json')
    fit = read(training/'router/complete.json')
    verify_artifacts(training/'router',fit['artifacts'])
    if protocol['smoke'] or fit['updates'] != 96*120:
        raise ValueError('review requires the completed formal core training, not smoke')
    models = dict(old_core=Path(protocol['initial_path']), adapted_core=training/'router/core/best.pt')
    rows = source_rows()
    plan = dict(format='latent_receiver_fixed_E_review_v1', training_complete=digest(training/'complete.json'),
        models={k:dict(path=str(p),sha256=digest(p)) for k,p in models.items()}, rows=rows,
        teacher_and_checkpoint_verification=deep_verification,
        verifier_code=digest(Path(__file__).with_name('verify_latent_routers.py')),
        counts=list(COUNTS), max_g=8, seed=protocol['seed'], receiver_profile=routing.identity(),
        G_assets_hash=protocol['G_assets_hash'], code=digest(Path(__file__)),
        sender_trained=False, source_group='previously used diagnostics; REDS val views and UVG crops',
        byte_matching='same E bytes and same-size receiver header; not independent per-budget allocations')
    immutable(args.output/'protocol.json',plan)
    if (args.output/'complete.json').exists():
        verify_artifacts(args.output,read(args.output/'complete.json')['artifacts'])
        return
    configure_torch(); metric = LPIPSAlex(True)
    points = []
    for row in rows:
        run.check(); sid = row['sample_id']
        folder = args.output/'samples'/sid; folder.mkdir(parents=True,exist_ok=True)
        with np.load(row['source_path'],allow_pickle=False) as f:
            source = f['source'].copy()
        enc = data.prepare_bank(folder,row,source,run.check)
        bank = (folder/'bank.rvlp').read_bytes()
        previous = {}
        for count in COUNTS:
            inner = routing.subset(bank,data.selection(sid,count))
            for arm in ('old_core','adapted_core','G_off'):
                model = models['adapted_core' if arm == 'G_off' else arm]
                stream = routing.wrap(inner,digest(model),protocol['G_assets_hash'],max_g=8,seed=protocol['seed'])
                if arm in previous and not stream.startswith(previous[arm]):
                    raise ValueError('comparison lost literal-prefix property')
                previous[arm] = stream
                dest = folder/f'{arm}_e{count}';dest.mkdir(parents=True,exist_ok=True)
                path = dest/'stream.rvlrg'
                if path.exists() and path.read_bytes() != stream:
                    raise ValueError('point stream changed')
                atomic_bytes(path,stream)
                if not (dest/'receive/complete.json').exists():
                    options = ['decode','--stream',path,'--output',dest/'receive','--receiver',model]
                    if arm == 'G_off': options += ['--disable-generation']
                    execute(run,f'{sid}_{arm}_e{count}',options,distributed=True)
                received = read(dest/'receive/complete.json')
                verify_artifacts(dest/'receive',received['artifacts'])
                if received['stream_sha256'] != digest(path) or received['actual_bytes'] != path.stat().st_size:
                    raise ValueError('actual received byte identity differs')
                if received['base_hash'] != enc['base_hash']:
                    raise ValueError('E/G changed base')
                with np.load(dest/'receive/pixels.npz',allow_pickle=False) as f:
                    output = f['reconstruction'].copy()
                if frame_hash(output) != received['output_hash']:
                    raise ValueError('stored fresh output differs')
                if (dest/'result.json').exists():
                    result = read(dest/'result.json')
                    if result['receive_sha256'] != digest(dest/'receive/complete.json'):
                        raise ValueError('point receiver receipt changed')
                else:
                    measured = quality(source,output,metric)
                    result = dict(sample_id=sid,dataset=row['dataset'],arm=arm,count=count,
                        bpp=received['bpp'],total_bytes=received['actual_bytes'],quality=measured,
                        generated=received['generated'],seconds=received['seconds'],
                        peak_cuda_allocated_bytes=received['peak_cuda_allocated_bytes'],
                        receive_sha256=digest(dest/'receive/complete.json'))
                    save(dest/'result.json',result)
                points.append(result)
        # One fixed frame per view, paired to E8 and actual measured whole video.
        from tools.plot_latent_diagnostics import comparison_image
        panels = [('Source',source[8])]
        for arm in ('G_off','old_core','adapted_core'):
            with np.load(folder/f'{arm}_e8/receive/pixels.npz',allow_pickle=False) as f:
                panels.append((arm,f['reconstruction'][8].copy()))
        comparison_image(panels,folder/'fixed_frame.png',title=sid+' | E8, same bytes')
    summary = summarize(points)
    save(args.output/'summary.json',dict(points=points,groups=summary,formal_receiver_only=True))
    plots(args.output,summary,training)
    names = ['summary.json','receiver_rd.png','receiver_training.png',
             *(f'samples/{r["sample_id"]}/fixed_frame.png' for r in rows)]
    save(args.output/'complete.json',dict(complete=True,points=len(points),protocol=digest(args.output/'protocol.json'),
        independent_fresh_points=len(points),sender_trained=False,
        artifacts={n:digest(args.output/n) for n in names}))


def main():
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=data.ROOT)
    p.add_argument('--output',type=Path,default=data.ROOT/'receiver_review')
    p.add_argument('--wait',action='store_true')
    p.add_argument('--max-hours',type=float,default=48.)
    args=p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    run=Run(SimpleNamespace(output=args.output/'queue',command='receiver_review',max_hours=args.max_hours))
    run.thread.start()
    try:
        while not (args.training/'formal/complete.json').exists():
            if not args.wait: raise RuntimeError('formal receiver stage incomplete')
            run.update(phase='waiting_for_formal_receiver');run.check();time.sleep(20)
        with exclusive_native_evaluation(run): run_review(args,run)
        run.update(phase='receiver_review_complete_sender_choice_pending')
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__': main()
