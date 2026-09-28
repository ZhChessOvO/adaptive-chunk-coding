"""Audit completed adaptations and draw undistorted fixed regional comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.conditioned_generation_pipeline import DEFAULT
from demo.conditioned_generation_evaluate import OLD, MODES
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo import scalable_cooperation_format as fmt
from demo.patch_prefix_probe import verify_artifacts, load_frames
from demo.chunk_enhancement_experiment import read
from demo.scalable_experiment import load_source, resources


def report(root):
    evaluation = root/'evaluation'
    summary = read(evaluation/'summary.json')
    if not summary['complete'] or summary['smoke'] or len(summary['results']) != 24:
        raise RuntimeError('expected complete two-recipe, four-clip, three-prefix comparison')
    reference = read(OLD/'summary.json')
    rows = {r['sample']['sample_id']:r for r in reference['results']}
    checks, details = {}, []
    for result_file in sorted(evaluation.glob('*/*/result.json')):
        point = read(result_file); dest = result_file.parent
        verify_artifacts(dest,point['artifacts'])
        c,inner,_,overhead = fmt.parse((dest/'stream.acsg').read_bytes())
        old_name, direct_name = MODES[point['mode']]
        old_root = OLD/point['sample_id']
        old_c,old_inner,_,old_overhead = fmt.parse((old_root/f'{old_name}.acsg').read_bytes())
        assert inner == old_inner and overhead == old_overhead
        for k in ('generate','protect','context','feather','seed','processing_scale','blend','window','stride'):
            assert c[k] == old_c[k]
        r = point['fresh_decode']
        assert not r['source_frames_read']
        assert r['total_bytes'] == point['bytes'] == (dest/'stream.acsg').stat().st_size
        assert sum(r[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                                 'incomplete_tail_bytes','generation_control_bytes')) == r['total_bytes']
        output = load_frames(dest/'reconstruction.npz')
        enhanced = load_frames(old_root/direct_name/'reconstruction.npz')
        assert r['output_hash'] == frame_hash(output)
        assert r['generation_input_hash'] == frame_hash(enhanced)
        alpha = fmt.weights(output.shape,c)
        np.testing.assert_array_equal(output[alpha==0],enhanced[alpha==0])
        if point['candidate'] == 'generation_off':
            np.testing.assert_array_equal(output,enhanced)
        if point['candidate'] == 'receiver_regression':
            np.testing.assert_array_equal(output,load_frames(old_root/old_name/'reconstruction.npz'))
        details.append(dict(path=str(dest),bytes=r['total_bytes'],control_bytes=overhead,
                            same_payload=True,same_controls=True))
    assert len(details) == 27
    first = next(iter(rows))
    np.testing.assert_array_equal(load_frames(evaluation/first/'image_full/reconstruction.npz'),
                                  load_frames(evaluation/first/'image_repeat_full/reconstruction.npz'))
    training = {}
    for candidate in ('latent','image'):
        directory = root/candidate
        complete = read(directory/'complete.json')
        assert file_hash(directory/'adapter.pt') == complete['adapter_sha256']
        exported = torch.load(directory/'adapter.pt',weights_only=True,map_location='cpu')
        resume = torch.load(directory/'resume.pt',weights_only=True,map_location='cpu')
        assert resume['step'] == complete['steps']
        for k,v in exported['state_dict'].items():
            torch.testing.assert_close(v,resume['adapter']['state_dict'][k],rtol=0,atol=0)
        log = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert [v['step'] for v in log] == list(range(1,complete['steps']+1))
        assert all(np.isfinite(v['loss']) and np.isfinite(v['gradient_norm']) for v in log)
        training[candidate] = dict(summary=complete,log=log)
    a,b = training['latent']['log'],training['image']['log']
    assert len(a) == len(b)
    for x,y in zip(a,b):
        for k in ('step','sample','dataset','condition','crop'):
            assert x[k] == y[k]
    cache = read(root/'cache.json')
    assert cache['datasets'] == {'REDS':90,'UVG':30}
    for row in cache['entries']:
        assert file_hash(Path(row['path'])) == row['sha256']
    groups = {}
    for mode in MODES:
        groups[mode] = {}
        for dataset in ('all','REDS','UVG'):
            groups[mode][dataset] = {}
            for candidate in ('baseline','direct','latent','image'):
                selected = [r for r in summary['results'] if r['mode']==mode and
                    r['candidate']==('image' if candidate in ('baseline','direct') else candidate) and
                    (dataset=='all' or rows[r['sample_id']]['sample']['dataset']==dataset)]
                points = [r[candidate] if candidate in ('baseline','direct') else r for r in selected]
                groups[mode][dataset][candidate] = {
                    k:float(np.mean([p['roi_quality'][k] for p in points]))
                    for k in ('lpips_alex','psnr_db','temporal_delta_mae')}
    checks.update(fresh_decodes=len(details),same_image_payload=True,same_control_geometry=True,
        bytes_and_hashes=True,adapter_resume_exact=True,paired_training_schedule_exact=True,
        generation_off_without_weights=True,repeat_exact=True,old_receiver_equivalent=True,
        unchanged_non_generate=True)
    # A better G(Y) alone does not establish better USE of E. Compare the
    # no-E -> full-E change for each generator, with everything else fixed.
    conditional = []
    for sid,row in rows.items():
        gains = {}
        for candidate in ('baseline','latent','image'):
            if candidate == 'baseline':
                none,full = [row['points'][MODES[m][0]] for m in ('none','full')]
            else:
                none,full = [next(r for r in summary['results'] if r['sample_id']==sid and
                                 r['mode']==m and r['candidate']==candidate) for m in ('none','full')]
            gains[candidate] = dict(
                lpips_benefit_from_E=none['roi_quality']['lpips_alex']-full['roi_quality']['lpips_alex'],
                psnr_benefit_from_E=full['roi_quality']['psnr_db']-none['roi_quality']['psnr_db'],
                extra_E_bytes=full['bytes']-none['bytes'],
                full_E_psnr_change_from_direct=full['roi_quality']['psnr_db']-
                    row['points']['enhance_q1']['roi_quality']['psnr_db'])
        conditional.append(dict(sample_id=sid,gains=gains))
    atomic_json(evaluation/'audit.json',dict(checks=checks,artifacts=details,resources=resources()))
    atomic_json(evaluation/'analysis.json',dict(groups=groups,conditional_effect=conditional,
        training={k:v['summary'] for k,v in training.items()},
        label='area-weighted local metrics per clip, then equal clip means; 4 development clips'))
    native_figures(evaluation,rows)
    training_figures(evaluation,training)
    temporal_figure(evaluation,rows)
    print(json.dumps(dict(checks=checks,full_prefix=groups['full']),ensure_ascii=False,indent=2))


def native_figures(root, rows):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for sid,row in rows.items():
        source = load_source(row['sample'])
        names = ['GT','E only','Old G(E)','Latent adapt','Image adapt']
        frames = [source,load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
                  load_frames(OLD/sid/'cooperate_l05/reconstruction.npz'),
                  load_frames(root/sid/'latent_full/reconstruction.npz'),
                  load_frames(root/sid/'image_full/reconstruction.npz')]
        regions = row['metric_regions']
        # All original pixels use the same integer scale; no aspect distortion.
        scale = 2
        width = max(r[4] for r in regions)*scale
        height = sum(r[5]*scale+36 for r in regions)
        canvas = Image.new('RGB',(len(names)*width,height),'white')
        draw = ImageDraw.Draw(canvas); top = 0
        for t,n,x,y,w,h in regions:
            for i,(name,video) in enumerate(zip(names,frames)):
                patch = Image.fromarray(video[8,y:y+h,x:x+w]).resize((w*scale,h*scale),Image.Resampling.NEAREST)
                canvas.paste(patch,(i*width,top+36))
                draw.text((i*width+3,top+8),name,font=font,fill='black')
            top += h*scale+36
        canvas.save(root/f'{sid}_native_fixed.png')


def training_figures(root, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for candidate,value in training.items():
        log=value['log'];steps=np.array([v['step'] for v in log])
        # Shared objective term only: total loss weights differ between recipes.
        velocity=np.array([v['terms']['velocity'] for v in log])
        smooth=np.convolve(velocity,np.ones(25)/25,'valid')
        axes[0].plot(steps[24:],smooth,label=candidate)
    log=training['image']['log']
    smooth=np.convolve([v['terms']['lpips'] for v in log],np.ones(25)/25,'valid')
    axes[1].plot(np.arange(25,len(log)+1),smooth,label='image objective')
    for ax,title in zip(axes,['Shared latent error (train)','LPIPS loss (train, not evaluation)']):
        ax.set_title(title);ax.set_xlabel('step');ax.grid(alpha=.3);ax.legend()
    fig.savefig(root/'training_diagnostics.png',dpi=150);plt.close(fig)


def temporal_figure(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for sid,row in rows.items():
        if row['sample']['frame_count'] != 33:
            continue
        source=load_source(row['sample'])
        videos={'E only':load_frames(OLD/sid/'enhance_q1/reconstruction.npz'),
                'Old G(E)':load_frames(OLD/sid/'cooperate_l05/reconstruction.npz'),
                'Latent adapt':load_frames(root/sid/'latent_full/reconstruction.npz'),
                'Image adapt':load_frames(root/sid/'image_full/reconstruction.npz')}
        t,n,x,y,w,h=row['metric_regions'][0]
        movies={'GT':source,**videos};frames=[]
        for f in range(33):
            canvas=Image.new('RGB',(5*w*3,h*3+32),'white');draw=ImageDraw.Draw(canvas)
            for i,(name,video) in enumerate(movies.items()):
                image=Image.fromarray(video[f,y:y+h,x:x+w]).resize((w*3,h*3),Image.Resampling.NEAREST)
                canvas.paste(image,(i*w*3,32))
                draw.text((i*w*3+3,7),name,font=font,fill='black')
            frames.append(canvas)
        frames[0].save(root/'long33_sequence.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
        fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
        for name,video in videos.items():
            error=(video[:,y:y+h,x:x+w].astype(np.float64)-source[:,y:y+h,x:x+w])**2
            ax.plot(np.arange(33),error.mean((1,2,3)),label=name)
        ax.axvline(16.5,color='gray',ls='--',label='E packets only in first 17 frames')
        ax.set(xlabel='Frame index',ylabel='Local RGB MSE (lower better)',title='Fixed 33-frame ROI; not LPIPS')
        ax.grid(alpha=.3);ax.legend();fig.savefig(root/'long33_error.png',dpi=150);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT)
    report(p.parse_args().output)
