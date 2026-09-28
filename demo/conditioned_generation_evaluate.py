"""Paired actual-stream evaluation of the two generation-adaptation recipes."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.conditioned_generation_decode import identities
from demo.conditioned_generation_pipeline import execute, DEFAULT
from demo.scalable_generation_decode import ASSETS
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, quality
from demo.scalable_cooperation_experiment import region_metrics
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.stage_c_three_path_roi_probe import LPIPSAlex

OLD = Path('/root/autodl-fs/DCVC/runs/a800_scalable_cooperation_20260928')
MODES = {'none':('generate_l05','base'), 'partial':('prefix_cooperate','prefix_enhance'),
         'full':('cooperate_l05','enhance_q1')}


def point(root, row, mode, candidate, adapter, hashes, source, metric, run, *, disabled=False):
    old_name, direct_name = MODES[mode]
    sid = row['sample']['sample_id']
    old_root = OLD/sid
    data = (old_root/f'{old_name}.acsg').read_bytes()
    if file_hash(old_root/f'{old_name}.acsg') != row['points'][old_name]['stream_sha256']:
        raise RuntimeError('reference stream changed')
    control, inner, _, _ = fmt.parse(data)
    old_control = control.copy()
    control.update(hashes)
    control['strength'] = .5 if candidate == 'receiver_regression' else 1.
    wire = fmt.wrap(inner,control)
    if len(wire) != len(data) or fmt.parse(wire)[1] != inner:
        raise RuntimeError('adaptation changed E image payload or syntax cost')
    dest = root/sid/f'{candidate}_{mode}'
    dest.mkdir(parents=True,exist_ok=True)
    stream = dest/'stream.acsg'
    result_path = dest/'result.json'
    if result_path.exists():
        result = read(result_path)
        verify_artifacts(dest,result['artifacts'])
        if stream.read_bytes() != wire:
            raise RuntimeError('resume stream differs')
        return result
    atomic_bytes(stream,wire)
    args = ['--stream',stream,'--output',dest,'--adapter',
            Path('/nonexistent/generator.pt') if disabled else adapter]
    if disabled:
        args += ['--disable-generation']
    begin = time.monotonic()
    execute(run,f'{sid}_{candidate}_{mode}','conditioned_generation_decode.py',args,distributed=True)
    output = load_frames(dest/'reconstruction.npz')
    report = read(dest/'decode.json')
    verify_artifacts(old_root/direct_name,row['points'][direct_name]['artifacts'])
    enhanced = load_frames(old_root/direct_name/'reconstruction.npz')
    if report['generation_input_hash'] != frame_hash(enhanced):
        raise RuntimeError('generator was not conditioned on the actual enhanced pixels')
    alpha = fmt.weights(source.shape,control)
    np.testing.assert_array_equal(output[alpha == 0],enhanced[alpha == 0])
    if disabled:
        np.testing.assert_array_equal(output,enhanced)
        assert not report['generation_assets_validated']
    if candidate == 'receiver_regression':
        verify_artifacts(old_root/old_name,row['points'][old_name]['artifacts'])
        np.testing.assert_array_equal(output,load_frames(old_root/old_name/'reconstruction.npz'))
    regions, roi = region_metrics(source,output,row['metric_regions'],metric)
    result = dict(sample_id=sid,mode=mode,candidate=candidate,bytes=len(wire),
        quality=quality(source,output,metric),per_region=regions,roi_quality=roi,
        process_wall_seconds=time.monotonic()-begin,fresh_decode=report,
        unchanged_inner_payload=True,same_bytes_as_reference=True,
        baseline=row['points'][old_name],direct=row['points'][direct_name],
        artifacts={p.name:file_hash(p) for p in (stream,dest/'reconstruction.npz',dest/'decode.json')})
    atomic_json(result_path,result)
    return result


def evaluate(args, run):
    root = args.output/'evaluation'
    root.mkdir(exist_ok=True)
    references = read(OLD/'summary.json')['results']
    adapters = {'image':args.output/'image_resume/adapter.pt'} if args.smoke else {
        k:args.output/k/'adapter.pt' for k in ('latent','image')}
    hashes = {k:identities(v) for k,v in adapters.items()}
    protocol = dict(code=file_hash(Path(__file__)),profiles=hashes,
        reference=file_hash(OLD/'summary.json'),smoke=args.smoke,
        reference_roles='same four previously used development clips, no independent-test claim',
        modes=MODES,steps=6 if args.smoke else read(args.output/'image/complete.json')['steps'])
    if (root/'protocol.json').exists() and read(root/'protocol.json') != json.loads(json.dumps(protocol)):
        raise RuntimeError('evaluation protocol changed')
    atomic_json(root/'protocol.json',protocol)
    metric = LPIPSAlex(True)  # CPU only: never concurrent GPU metrics with UF.
    results = []
    for row in references[:1] if args.smoke else references:
        source = load_source(row['sample'])
        if frame_hash(source) != row['source_hash']:
            raise RuntimeError('evaluation source changed')
        sid = row['sample']['sample_id']
        for candidate, adapter in adapters.items():
            for mode in (['full'] if args.smoke else MODES):
                results.append(point(root,row,mode,candidate,adapter,hashes[candidate],source,metric,run))
        if row == references[0]:
            candidate = 'image'
            repeated = point(root,row,'full','image_repeat',adapters[candidate],hashes[candidate],source,metric,run)
            np.testing.assert_array_equal(load_frames(root/sid/'image_full/reconstruction.npz'),
                                          load_frames(root/sid/'image_repeat_full/reconstruction.npz'))
            point(root,row,'full','generation_off',adapters[candidate],hashes[candidate],source,metric,run,disabled=True)
            point(root,row,'full','receiver_regression',ASSETS['lora'],identities(ASSETS['lora']),source,metric,run)
    atomic_json(root/'summary.json',dict(complete=True,smoke=args.smoke,results=results,protocol=protocol,
        repeat_exact=True,generation_off_without_weights_exact=True,old_adapter_receiver_exact=True))
    if not args.smoke:
        figures(root,references,results)


def figures(root, references, results):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for row in references:
        sid = row['sample']['sample_id']
        source = load_source(row['sample'])
        images = {'GT':source, 'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
                  'Old G(E)':load_frames(OLD/sid/'cooperate_l05/reconstruction.npz'),
                  'Latent adapt':load_frames(root/sid/'latent_full/reconstruction.npz'),
                  'Image adapt':load_frames(root/sid/'image_full/reconstruction.npz')}
        regions = row['metric_regions']
        cell = 256
        canvas = Image.new('RGB',(cell*5,(cell+32)*len(regions)),'white')
        draw = ImageDraw.Draw(canvas)
        for ir, region in enumerate(regions):
            t,n,x,y,w,h = region
            for ic,(label,frames) in enumerate(images.items()):
                crop = Image.fromarray(frames[min(8,len(frames)-1),y:y+h,x:x+w]).resize((cell,cell))
                canvas.paste(crop,(ic*cell,ir*(cell+32)+32))
                draw.text((ic*cell+4,ir*(cell+32)+5),label,font=font,fill='black')
        canvas.save(root/f'{sid}_fixed.png')
        if sid == references[0]['sample']['sample_id']:
            t,n,x,y,w,h = regions[-1]
            animation = []
            for frame in range(17):
                strip = Image.new('RGB',(cell*5,cell+32),'white')
                d = ImageDraw.Draw(strip)
                for ic,(label,frames) in enumerate(images.items()):
                    strip.paste(Image.fromarray(frames[frame,y:y+h,x:x+w]).resize((cell,cell)),(ic*cell,32))
                    d.text((ic*cell+4,4),label,font=font,fill='black')
                animation.append(strip)
            animation[0].save(root/'wall_sequence.gif',save_all=True,append_images=animation[1:],duration=125,loop=0)
    for metric,label in [('lpips_alex','Local LPIPS (lower better)'),('psnr_db','Local PSNR (dB)')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,row in zip(axes.flat,references):
            sid=row['sample']['sample_id']
            for candidate in ('Old G(E)','latent','image'):
                points = ([row['points'][MODES[m][0]] for m in MODES] if candidate == 'Old G(E)' else
                          [next(r for r in results if r['sample_id']==sid and r['candidate']==candidate and r['mode']==m) for m in MODES])
                ax.plot([r['bytes'] for r in points],[r['roi_quality'][metric] for r in points],marker='o',label=candidate)
            ax.set_title(sid);ax.set_xlabel('Actual total bytes');ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend()
        fig.savefig(root/f'paired_{metric}.png',dpi=150);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-hours',type=float,default=4.)
    args=p.parse_args();args.command='evaluation'
    run=Run(args);run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            evaluate(args,run)
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)
