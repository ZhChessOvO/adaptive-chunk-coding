"""Bounded P3: actual width3/5/7, frozen Routers, calibration-only selection.

Eight training calibration windows and six internal evaluation windows. All
long work is resumable in tmux; completion never launches another training.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.cooperation import sender_data as prior, sender as fixed_sender
from routervc.precision import entropy, codec, format as packets, stream, budget
from tools.cooperative_sender import SenderRun, execute
from tools.cooperative_review import source_rgb

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_precision_20261012')
REPO = Path(__file__).resolve().parents[1]
RATIOS = (0., .25, .5, 1.)
SENDER_SHA = '870356ac4662ce50d77bcf2ac8e9cc0a66fd1ae34155c38701b57c4689eb70a8'


def prepare(root):
    original = read(prior.ROOT/'formal/protocol.json')
    for name, expected in original['code'].items():
        if digest(REPO/name) != expected: raise ValueError('frozen dependency changed: '+name)
    path = prior.ROOT/'formal/router/source/best.pt'
    if digest(path) != SENDER_SHA: raise ValueError('preselected sender changed')
    fixed_sender.load(path)
    selected = []
    for split,cohort in (('train','calibration'),('validation','internal_evaluation')):
        reds = [r for r in original['rows'] if r['router_split']==split and r['dataset']=='REDS']
        ids = np.linspace(0,len(reds)-1,4 if split=='train' else 3).round().astype(int)
        selected += [dict(reds[i],cohort=cohort) for i in ids]
        uvg = [r for r in original['rows'] if r['router_split']==split and r['dataset']=='UVG']
        if split=='train':
            uvg = [next(r for r in uvg if r['sequence']==s) for s in sorted({r['sequence'] for r in uvg})]
        selected += [dict(r,cohort=cohort) for r in uvg]
    if len(selected) != 14 or sum(r['cohort']=='calibration' for r in selected) != 8:
        raise ValueError('unexpected bounded sample inventory')
    code = set(original['code']) | {str(p.relative_to(REPO)) for p in (REPO/'routervc/precision').iterdir()
        if p.suffix in ('.py','.cpp')}
    code.update('tools/'+n for n in ('precision_pilot.py','precision_worker.py','precision_decode.py',
        'precision_report.py','run_precision.sh','test_precision.py','cooperative_sender.py',
        'cooperative_review.py','plot_latent_diagnostics.py','fusion_finish.py','storage_guard.py'))
    protocol = dict(format='precision_small_width_v1', widths=list(entropy.WIDTHS), ratios=list(RATIOS), rows=selected,
        teacher=original['teacher'], sender=dict(path=str(path),sha256=SENDER_SHA,epoch=8),
        source_protocol=digest(prior.ROOT/'formal/protocol.json'),
        code={n:digest(REPO/n) for n in sorted(code)}, codec_profile=codec.identity(), receiver_profile=stream.identity(),
        calibration_windows=8,evaluation_windows=6,logical_points=168, mask_bytes=0,
        frozen_models=True,zero_shot_width_transfer=True,no_optimization=True,
        independent_test=False,selector='three_neighbour_table_v1, four cap-grid points only',
        selector_inputs='actual bytes and source/B MSE/texture; no evaluation final scores',
        widths_between_two_and_eight=True,automatic_promotion=False)
    immutable(root/'protocol.json',protocol)
    return protocol


def check_protocol(root):
    p = read(root/'protocol.json')
    for name,expected in p['code'].items():
        if digest(REPO/name) != expected: raise ValueError('bound source changed: '+name)
    for model in (p['sender'],p['teacher']['receiver']):
        if digest(model['path']) != model['sha256']: raise ValueError('selected model changed')
    if p['codec_profile'] != codec.identity() or p['receiver_profile'] != stream.identity():
        raise ValueError('profile changed')
    if p['widths'] != [3,5,7]: raise ValueError('width scope changed')
    return p


def bank_folder(root,row,width): return root/'banks'/row['sample_id']/f'w{width}'
def point_folder(root,row,width,cap): return root/row['cohort']/row['sample_id']/f'w{width}'/f'e{round(cap*100):03d}'


def plan(run,root,row,width):
    folder = bank_folder(root,row,width)
    if not (folder/'complete.json').exists():
        execute(run,'prepare_'+row['sample_id']+f'_w{width}',
            ['--root',root,'--sample-id',row['sample_id'],'--width',width],module='tools.precision_worker')
    done = read(folder/'complete.json'); verify_artifacts(folder,done['artifacts'])
    previous = None
    for p in done['points']:
        raw = (folder/p['file']).read_bytes()
        if previous is not None and not raw.startswith(previous): raise ValueError('prefix changed')
        previous = raw
    return done


def fresh(run,path,dest,protocol,disabled=False,legacy=False):
    profile = protocol['receiver_profile']
    if legacy:
        from routervc.cooperation import stream as old
        profile = old.identity()
    immutable(dest/'request.json',dict(stream=digest(path), disabled=disabled,
        receiver=None if disabled else protocol['teacher']['receiver'],profile=profile))
    if not (dest/'complete.json').exists():
        options = ['decode','--stream',path,'--output',dest]
        if disabled: options += ['--disable-generation']
        else: options += ['--receiver',protocol['teacher']['receiver']['path']]
        execute(run,'fresh_'+dest.parent.name+'_'+dest.name,options,distributed=True,
                module='tools.cooperative_sender_decode' if legacy else 'tools.precision_decode')
    receipt = read(dest/'complete.json'); verify_artifacts(dest,receipt['artifacts'])
    if (receipt['stream'] != digest(path) or receipt['actual_bytes'] != path.stat().st_size
            or receipt['source_frames_read'] or receipt['sender_router_loaded']
            or receipt.get('sender_checkpoint_access_guard') is not True
            or receipt['generation_disabled'] != disabled or receipt['mask_bytes'] != 0
            or receipt['header_bytes'] != 121): raise ValueError('fresh receiver contract failed')
    if disabled and (receipt['generated'] or receipt['selected'] is not None
                     or receipt['output_hash'] != receipt['enhanced_hash']): raise ValueError('Goff not Y')
    return receipt


def verify_point(point):
    for path,expected in point['artifacts'].items():
        if digest(path) != expected: raise ValueError('point artifact changed: '+path)
    receipt = read(Path(point['receive'])/'complete.json')
    if receipt['stream'] != point['stream_sha256'] or receipt['actual_bytes'] != point['actual_bytes']:
        raise ValueError('point receipt changed')
    return receipt


def evaluate(run,root,protocol,row,width,cap,metric):
    from demo.scalable_experiment import quality
    from routervc.fusion.blend import geometry
    folder = point_folder(root,row,width,cap); folder.mkdir(parents=True,exist_ok=True)
    if (folder/'point.json').exists():
        point = read(folder/'point.json'); verify_point(point); return point
    bankdir = bank_folder(root,row,width); wire = bankdir/f'e{round(cap*100):03d}.rvlw'
    encoded = read(bankdir/'encoded.json'); dest = folder/'receive'; reused = False
    for prior_cap in RATIOS:
        candidate = point_folder(root,row,width,prior_cap)/'point.json'
        if candidate.exists():
            p = read(candidate)
            if p['stream_sha256']==digest(wire):
                verify_point(p); dest = Path(p['receive']); reused = True; break
    run.update(phase='fresh_decode' if not reused else 'reuse_identical_stream',sample=row['sample_id'],width=width,cap=cap)
    receipt = read(dest/'complete.json') if reused else fresh(run,wire,dest,protocol)
    if (receipt['base_hash'] != encoded['base_hash'] or receipt['detail']['base_reference_hashes'] != encoded['base_reference_hashes']
            or receipt['detail']['bin_width'] != width): raise ValueError('B reference / width differs')
    with np.load(dest/'pixels.npz',allow_pickle=False) as z:
        base,y,output = z['base'],z['enhanced'],z['reconstruction']
    for pixels,key in ((base,'base_hash'),(y,'enhanced_hash'),(output,'output_hash')):
        if frame_hash(pixels) != receipt[key]: raise ValueError('decoded pixels changed')
    support,_,_ = geometry(output.shape,receipt['generated'])
    np.testing.assert_array_equal(output[:,~support],y[:,~support])
    source = source_rgb(row)
    if source.shape != output.shape or abs(receipt['bpp']-wire.stat().st_size*8/np.prod(source.shape[:3]))>1e-12:
        raise ValueError('source geometry / complete-file bpp differs')
    run.update(phase='cpu_picture_scoring',sample=row['sample_id'],width=width,cap=cap)
    base_q_path = bankdir/'base_quality.json'
    if base_q_path.exists(): base_quality = read(base_q_path)
    else:
        base_quality = quality(source,base,metric); save(base_q_path,base_quality)
    point = dict(complete=True,cohort=row['cohort'],sample_id=row['sample_id'],dataset=row['dataset'],
        width=width,cap=cap,view_kind=row['view_kind'],wire=str(wire),stream_sha256=digest(wire),receive=str(dest),
        actual_bytes=receipt['actual_bytes'],bpp=receipt['bpp'],base_bytes=encoded['base_bytes'],
        quality=quality(source,output,metric),received_quality=quality(source,y,metric),base_quality=base_quality,
        sender_features=encoded['sender_features'],G_calls=len(receipt['generated']),G_regions=receipt['generated'],
        E_regions=receipt['detail']['received_regions'],fresh_seconds=receipt['seconds'],
        peak_GiB=receipt['peak_cuda_allocated_bytes']/2**30,reserved_GiB=receipt['peak_cuda_reserved_bytes']/2**30,
        reused_identical_stream=reused, output_hash=receipt['output_hash'],
        artifacts={str(p):digest(p) for p in (wire,dest/'complete.json',dest/'pixels.npz',base_q_path)})
    save(folder/'point.json',point); return point


def smoke(run,root,p,metric):
    if (root/'smoke.json').exists():
        result = read(root/'smoke.json'); verify_artifacts(root,result['artifacts']); return
    names = []; processes = 0
    for dataset in ('REDS','UVG'):
        row = next(r for r in p['rows'] if r['cohort']=='calibration' and r['dataset']==dataset)
        for width in p['widths']:
            plan(run,root,row,width); bankdir = bank_folder(root,row,width); encoded = read(bankdir/'encoded.json')
            point = evaluate(run,root,p,row,width,.5,metric); processes += 1
            names.append(str((point_folder(root,row,width,.5)/'point.json').relative_to(root)))
            for level in ('base','full'):
                dest = root/'startup'/row['sample_id']/f'w{width}'/level
                inner = packets.subset((bankdir/'bank.rvlp').read_bytes(),[] if level=='base' else list(range(16)))
                teacher = p['teacher']; path = dest/'stream.rvlw'; dest.mkdir(parents=True,exist_ok=True)
                atomic_bytes(path,stream.wrap(inner,teacher['receiver']['sha256'],teacher['G_assets_hash'],8,teacher['seed']))
                report = fresh(run,path,dest/'receive',p,disabled=True); processes += 1
                if report['output_hash'] != encoded['base_hash' if level=='base' else 'full_hash']:
                    raise ValueError('fresh B/full endpoint differs')
                if report['detail']['base_reference_hashes'] != encoded['base_reference_hashes']:
                    raise ValueError('fresh E changed references')
                names += [str((dest/n).relative_to(root)) for n in ('stream.rvlw','receive/complete.json','receive/pixels.npz')]
            if width == 3:
                from routervc.latent import routing
                from routervc.cooperation import stream as old
                order = read(bankdir/'complete.json'); indices = next(x['indices'] for x in order['points'] if x['cap']==.5)
                old_wire = old.wrap(routing.subset(Path(row['bank_path']).read_bytes(),indices),
                    p['teacher']['receiver']['sha256'],p['teacher']['G_assets_hash'],8,p['teacher']['seed'])
                dest = root/'startup'/row['sample_id']/'legacy_w3'; dest.mkdir(parents=True,exist_ok=True)
                atomic_bytes(dest/'stream.rvlcoop',old_wire)
                report = fresh(run,dest/'stream.rvlcoop',dest/'receive',p,legacy=True); processes += 1
                if report['output_hash'] != point['output_hash'] or report['actual_bytes'] != point['actual_bytes']:
                    raise ValueError('legacy width3 full-system parity failed')
                names += [str((dest/n).relative_to(root)) for n in ('stream.rvlcoop','receive/complete.json','receive/pixels.npz')]
            if width == 5:
                dest = root/'startup'/row['sample_id']/'repeat_w5'
                repeat = fresh(run,Path(point['wire']),dest,p); processes += 1
                if repeat['output_hash'] != point['output_hash']: raise ValueError('deterministic repeat failed')
                names += [str((dest/n).relative_to(root)) for n in ('complete.json','pixels.npz')]
    save(root/'smoke.json',dict(complete=True,protocol=digest(root/'protocol.json'),fresh_processes=processes,
        width3_legacy_system_exact=True,all_widths_full_endpoint_exact=True,sourcefree_senderfree=True,
        repeat_exact=True,Goff_model_free=True,artifacts={n:digest(root/n) for n in names}))


def decisions(root,protocol,row,table):
    candidates = []; facts = {}
    for w in protocol['widths']:
        folder = bank_folder(root,row,w); facts[w] = read(folder/'encoded.json')
        for p in read(folder/'complete.json')['points']:
            candidates.append(dict(width=w,cap=p['cap'],actual_bytes=p['actual_bytes'],base_bytes=facts[w]['base_bytes'],
                sender_features=facts[w]['sender_features'],key=f'w{w}_e{round(p["cap"]*100):03d}'))
    b,e = facts[3]['base_bytes'],facts[3]['full_e_bytes']
    budgets = [int(.75*b),int(.9*b),b,b+int(.25*e),b+int(.5*e),b+e]
    result = dict(sample_id=row['sample_id'],dataset=row['dataset'],candidates=candidates,
        table_sha256=digest(root/'calibration.json'),decisions=[dict(budget_bytes=limit,
        automatic=budget.choose(table,candidates,limit),fixed={str(w):budget.fixed(candidates,limit,w) for w in protocol['widths']},
        base_fraction=budget.base_fraction(candidates,limit)) for limit in budgets],
        final_scores_used=False,grid_only=True,switch_penalty=0,independent_segments=True)
    immutable(root/'decisions'/f'{row["sample_id"]}.json',result)


def verify(root):
    p = check_protocol(root); done = read(root/'complete.json'); verify_artifacts(root,done['artifacts'])
    verify_artifacts(root,read(root/'smoke.json')['artifacts'])
    for row in p['rows']:
        for w in p['widths']:
            folder = bank_folder(root,row,w); verify_artifacts(folder,read(folder/'complete.json')['artifacts'])
            for cap in RATIOS: verify_point(read(point_folder(root,row,w,cap)/'point.json'))
    print('PRECISION_PILOT_VERIFIED',root,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('all','smoke','verify','report'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args = parser.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if args.command=='verify': verify(args.output); return
    if args.command=='report':
        from tools.precision_report import report
        report(args.output,check_protocol(args.output)); return
    configure_torch()
    run = SenderRun(SimpleNamespace(output=args.output/'queue',command='precision',max_hours=48));run.thread.start()
    try:
        p = prepare(args.output);check_protocol(args.output)
        if (args.output/'complete.json').exists(): verify(args.output); return
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        with exclusive_native_evaluation(run):
            metric = LPIPSAlex(True); smoke(run,args.output,p,metric)
            if args.command=='smoke': return
            immutable(args.output/'formal_started.json',dict(started=True,protocol=digest(args.output/'protocol.json'),
                startup=digest(args.output/'smoke.json'),logical_points=168,optimization=False))
            run.update(phase='formal_precision_evaluation_started',total=168)
            points = []; names = []
            for cohort in ('calibration','internal_evaluation'):
                table = None
                if cohort=='internal_evaluation':
                    table = budget.fit(points);immutable(args.output/'calibration.json',table)
                for row in [r for r in p['rows'] if r['cohort']==cohort]:
                    for w in p['widths']: plan(run,args.output,row,w)
                    if table is not None: decisions(args.output,p,row,table)
                    for w in p['widths']:
                        for cap in RATIOS:
                            points.append(evaluate(run,args.output,p,row,w,cap,metric))
                            names.append(str((point_folder(args.output,row,w,cap)/'point.json').relative_to(args.output)))
                            run.update(phase='formal_point_complete',completed=len(points),total=168)
            from tools.precision_report import report
            names += report(args.output,p,run.check)
            names += ['protocol.json','smoke.json','formal_started.json','calibration.json']
            names += [str(path.relative_to(args.output)) for path in sorted((args.output/'decisions').glob('*.json'))]
            save(args.output/'complete.json',dict(complete=True,logical_points=len(points),
                newly_decoded_points=sum(not x['reused_identical_stream'] for x in points),no_optimization=True,
                automatic_promotion=False,artifacts={n:digest(args.output/n) for n in names}))
            verify(args.output);run.update(phase='complete_wait_for_review',completed=168)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__ == '__main__': main()
