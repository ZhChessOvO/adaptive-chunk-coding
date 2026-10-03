"""CPU-only bounded evaluation contracts. No formal data or GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from demo import routervc_visual_evaluate as module
from demo.test_routervc_visual_policy import fixture_payload


def samples():
    rows = []
    for sequence in module.REDS_SEQUENCES:
        rows.append(dict(sample_id=f'reds-val-{sequence}-f000-n17-fullview', dataset='REDS',
            sequence=sequence, router_split='evaluation', role='development' if sequence in ('000','005') else 'evaluation',
            original_frame_start=0, frame_count=17, whole_frame=True, view_kind='resized_full_frame',
            transform=dict(coded_size=[1024,576], valid_rect=[0,0,1024,576], padding=[0]*4),
            history='historical development exposure', component_training_sequence=False))
    for sequence in module.UVG_SEQUENCES:
        rows.append(dict(sample_id=f'uvg-{sequence.lower()}-f000-historical-evaluation-crop', dataset='UVG',
            sequence=sequence, router_split='evaluation', role='evaluation', original_frame_start=0,
            frame_count=17, whole_frame=False, view_kind='existing_spatial_crop',
            transform=dict(coded_size=[512,512], valid_rect=[0,0,512,512], padding=[0]*4),
            history='historical crop exposure', component_training_sequence=sequence not in ('ReadySetGo','YachtRide')))
    return rows


def formal_models(root):
    binding = dict(smoke=False, train_ids=['train-only'], valid_ids=['valid-only'])
    arms = {}
    for arm in module.ARMS:
        folder = root/arm; folder.mkdir(parents=True)
        payload = fixture_payload(use_global=arm == 'global_local')
        config = dict(payload['training_binding'], data=binding, epochs=120)
        payload['training_binding'] = config
        module.save(folder/'config.json', config)
        torch.save(payload, folder/'model.pt')
        module.save(folder/'complete.json', dict(complete=True, epochs=120, semantic_supervision=False,
                    config_sha256=module.digest(folder/'config.json'),
                    artifacts={'model.pt':module.digest(folder/'model.pt')}))
        arms[arm] = module.digest(folder/'complete.json')
    module.save(root/'complete.json', dict(complete=True, semantic_supervision=False, binding=binding, arms=arms))


class EvaluationTests(unittest.TestCase):
    def test_exact_thirteen_samples_including_development_and_crop_exposure(self):
        rows = samples()
        selected = module.selected_samples(dict(samples=rows+[dict(sample_id='not-approved',router_split='train')]))
        self.assertEqual(len(selected), 13)
        self.assertEqual([r['sequence'] for r in selected[:6]], list(module.REDS_SEQUENCES))
        self.assertTrue(all(r['view_kind'] == 'existing_spatial_crop' for r in selected[6:]))
        self.assertEqual(module.source_shape(selected[0]), (17,576,1024,3))
        self.assertEqual(module.source_shape(selected[-1]), (17,512,512,3))
        for altered in (rows[:-1], rows+[rows[0]], [dict(rows[0],router_split='train'),*rows[1:]],
                        [*rows[:-1],dict(rows[-1],whole_frame=True)]):
            with self.assertRaises(ValueError):
                module.selected_samples(dict(samples=altered))

    def test_point_plan_is_169_plus_two_smoke_controls_not_a_dataset_claim(self):
        points = module.point_plan()
        self.assertEqual(len(points), 13)
        self.assertEqual(len({p['name'] for p in points}), 13)
        routed = [p for p in points if p['kind'] == 'router']
        self.assertEqual(len(routed), 8)
        self.assertEqual({(p['arm'],p['ratio'],p['max_g']) for p in routed},
                         {(a,r,g) for a in module.ARMS for r in (.25,.5) for g in (4,8)})
        self.assertEqual([p['qp'] for p in points if p['kind'] == 'uf'], [8,16,24,32])
        self.assertEqual(len(samples())*len(points),169)

    def test_formal_models_must_finish_both_arms_and_payload_matches_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.assertIsNone(module.completed_models(root))
            formal_models(root)
            value=module.completed_models(root)
            self.assertEqual(set(value['arms']),set(module.ARMS))
            self.assertFalse(value['semantic_supervision'])
            # Rebind all file hashes around a smoke bundle: internal binding still rejects it.
            path=root/'global_local/model.pt'
            payload=torch.load(path,weights_only=True)
            payload['training_binding']['data']['smoke']=True
            torch.save(payload,path)
            complete=module.read(root/'global_local/complete.json')
            complete['artifacts']['model.pt']=module.digest(path)
            module.save(root/'global_local/complete.json',complete)
            done=module.read(root/'complete.json');done['arms']['global_local']=module.digest(root/'global_local/complete.json')
            module.save(root/'complete.json',done)
            with self.assertRaisesRegex(ValueError,'internal training binding'):
                module.completed_models(root)

    def test_present_invalid_completion_is_error_not_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); formal_models(root)
            for update in ({'semantic_supervision':True},{'binding':{'smoke':True}},{'arms':{'local':'0'*64}}):
                original=module.read(root/'complete.json')
                module.save(root/'complete.json',{**original,**update})
                with self.assertRaises(ValueError):module.completed_models(root)
                module.save(root/'complete.json',original)
            path=root/'local/model.pt';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'artifact'):
                module.completed_models(root)

    def test_phase_deadline_resets_between_wait_and_evaluation_not_per_point(self):
        clock=[0.]; base=Mock()
        deadline=module.PhaseDeadline(base_check=base,now=lambda:clock[0])
        deadline.begin('wait',48*3600)
        clock[0]=48*3600-1;deadline.check()
        clock[0]=48*3600
        with self.assertRaisesRegex(InterruptedError,'wait'):deadline.check()
        deadline.begin('evaluation',12*3600)
        clock[0]+=12*3600-1;deadline.check()
        clock[0]+=1
        with self.assertRaisesRegex(InterruptedError,'evaluation'):deadline.check()
        with self.assertRaises(ValueError):deadline.begin('invalid',float('nan'))
        self.assertEqual(base.call_count,4)

    def test_wait_models_has_no_gpu_lock_and_checks_each_poll(self):
        run=SimpleNamespace(check=Mock(),update=Mock(),stop=SimpleNamespace(wait=Mock()))
        inspect=Mock(side_effect=[None,None,{'complete':'ready'}])
        with patch('demo.chunk_enhancement_evaluate.exclusive_native_evaluation',
                   side_effect=AssertionError('waiting must not take GPU lock')):
            self.assertEqual(module.wait_for_models(run,Path('unused'),inspect=inspect),{'complete':'ready'})
        self.assertEqual(run.check.call_count,4)
        self.assertEqual(run.stop.wait.call_count,2)
        self.assertTrue(all(c.kwargs['holds_GPU_mutex'] is False for c in run.update.call_args_list))
        with self.assertRaisesRegex(ValueError,'corrupt'):
            module.wait_for_models(run,Path('unused'),inspect=Mock(side_effect=ValueError('corrupt')))

    def test_worker_commands_source_free_and_off_omits_router_assets(self):
        protocol=dict(enhancement={'path':'/E.pt'},adapter={'path':'/G.pt'},
                      models={'arms':{a:{'path':'/'+a+'.pt'} for a in module.ARMS}})
        points=module.point_plan()+[dict(name='off',kind='off',arm='global_local')]
        for point in points:
            script,argv,distributed=module.receiver_command(Path('/point'),point,protocol)
            self.assertNotIn('--source',argv);self.assertNotIn('--input',argv)
            if point['kind']=='router':
                self.assertEqual(script,'routervc_visual_decode.py');self.assertIn('--worker',argv)
                self.assertTrue(distributed)
            if point['kind']=='off':
                self.assertIn('--disable-generation',argv);self.assertNotIn('--router',argv)
                self.assertIn('/missing/unused-G.pt',argv)
            if point['kind']=='full_g':
                self.assertEqual(script,'routervc_baselines.py');self.assertEqual(argv[0],'decode-g')

    def test_full_frame_baseline_uses_actual_empty_ACSE2_and_dynamic_geometry(self):
        from demo.test_routervc_encode import make_bank
        from demo.test_routervc_visual_wire import config
        from demo import scalable_cooperation_format as cooperation
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);module.save(root/'protocol.json',{})
            folder=root/'sample';(folder/'prepared').mkdir(parents=True)
            (folder/'source.npz').write_bytes(b'opaque-source-binding')
            base=np.zeros((17,64,96,3),np.uint8)
            (folder/'prepared/bank.acse').write_bytes(make_bank(base))
            module.save(folder/'prepared/complete.json',{'binding':{'source':{'source_shape':list(base.shape)}}})
            point={'name':'wholeframe_g_one_roi','kind':'full_g'}
            protocol={'configs':{'global_local_g8':config(8)}}
            with patch('demo.conditioned_generation_pipeline.execute',side_effect=AssertionError('no worker for envelope')):
                out=module.prepare_point(SimpleNamespace(root=root),folder,point,protocol)
            control,raw,parsed,header=cooperation.parse((out/'stream.acsg').read_bytes())
            self.assertEqual(raw[:5],b'ACSE\x02')
            self.assertEqual(len(parsed.packets),0)
            self.assertEqual(control['generate'],[[0,17,0,0,96,64]])

    def test_compact_report_is_grouped_and_completed_reentry_preserves_mtimes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);module.save(root/'protocol.json',{})
            module.save(root/'smoke.complete.json',{'complete':True})
            protocol={'sources':[{'sample':s} for s in samples()]};records=[]
            for sample in samples():
                folder=root/'samples'/sample['sample_id'];folder.mkdir(parents=True)
                np.savez(folder/'source.npz',source=np.zeros((17,8,8,3),np.uint8))
                for point in module.point_plan():
                    out=folder/point['name'];out.mkdir()
                    Image.new('RGB',(8,8)).save(out/'fixed_frame.png')
                    if point['kind']=='router':
                        (out/'stream.rtvc').write_bytes(b'prefix'+(b'long' if point['ratio']==.5 else b''))
                    record=dict(sample_id=sample['sample_id'],point=point,bytes=100,bpp=.02,
                        quality=dict(lpips_alex=.4,psnr_db=24.,temporal_delta_mae=2.,rgb_mse=3.),
                        decode=dict(seconds=10.,policy_seconds=.1,peak_cuda_allocated_bytes=0))
                    module.save(out/'result.json',record);records.append(record)
            summary=module.finish(SimpleNamespace(root=root),protocol,records)
            self.assertEqual(summary['points'],169)
            self.assertEqual(summary['group_means']['REDS_fullview']['uf_qp8']['windows'],6)
            self.assertEqual(summary['group_means']['UVG_crop']['uf_qp8']['windows'],7)
            self.assertFalse(summary['independent_system_test'])
            self.assertEqual(len(summary['fixed_visuals']),13)
            before={str(p):(module.digest(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(module.finish(SimpleNamespace(root=root),protocol,records),summary)
            for path,expected in before.items():
                self.assertEqual((module.digest(path),Path(path).stat().st_mtime_ns),expected)
            (root/'report/points.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'CSV'):
                module.finish(SimpleNamespace(root=root),protocol,records)

    def test_verify_source_missing_cannot_write_or_load_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);module.save(root/'protocol.json',{'fixture':True})
            run=SimpleNamespace(root=root);entry=dict(sample=samples()[0],original_file_hashes={'files':{}})
            with patch('demo.routervc_fullview_data.source_hash',return_value=entry['original_file_hashes']),\
                    patch('demo.routervc_mixedview_data.load_rgb',side_effect=AssertionError('no pixels')),\
                    patch('demo.scalable_codec.atomic_npz',side_effect=AssertionError('no writes')):
                with self.assertRaisesRegex(ValueError,'read-only'):
                    module.prepare_source(run,entry,verify_only=True)
            self.assertFalse((root/'samples').exists())

    def test_completed_point_reentry_never_infers_or_measures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);point=module.point_plan()[0]
            folder=root/'sample';path=folder/point['name']/'result.json'
            module.save(path,{'complete':True,'original_seconds':12.3});mtime=path.stat().st_mtime_ns
            original=module.read(path)
            with patch.object(module,'validate_result',return_value=original),\
                    patch.object(module,'prepare_point',side_effect=AssertionError('no inference')),\
                    patch('demo.scalable_experiment.quality',side_effect=AssertionError('no metrics')):
                actual,metric=module.evaluate_point(SimpleNamespace(root=root),folder,point,samples()[0],{},None)
            self.assertEqual(actual,original);self.assertIsNone(metric)
            self.assertEqual(path.stat().st_mtime_ns,mtime)

    def test_validate_result_binds_sample_and_real_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);module.save(root/'protocol.json',{})
            sample=samples()[0];folder=root/sample['sample_id'];point=module.point_plan()[0]
            decoded={'seconds':1.};ledger={'bytes':123}
            record=dict(complete=True,point=point,sample_id=sample['sample_id'],
                        protocol_sha256=module.digest(root/'protocol.json'),artifacts={},
                        decode=decoded,byte_ledger=ledger,bytes=123,bpp=8*123/(17*576*1024))
            path=folder/point['name']/'result.json';module.save(path,record)
            with patch.object(module,'validate_receiver',return_value=(decoded,ledger)):
                self.assertEqual(module.validate_result(root,folder,point,sample,{}),record)
                for update in ({'sample_id':'wrong'},{'bytes':124},{'bpp':1.}):
                    module.save(path,{**record,**update})
                    with self.assertRaises(ValueError):module.validate_result(root,folder,point,sample,{})

    def test_prefix_check_separates_arms_and_g_caps(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for arm in module.ARMS:
                for cap in (4,8):
                    for ratio in (.25,.5):
                        path=folder/f'{arm}_e{ratio:g}_g{cap}'/'stream.rtvc'
                        path.parent.mkdir();path.write_bytes(f'{arm}{cap}'.encode()+(b'long' if ratio==.5 else b''))
            module.check_prefixes(folder)
            (folder/'local_e0.5_g8/stream.rtvc').write_bytes(b'not a prefix')
            with self.assertRaisesRegex(ValueError,'literal byte prefixes'):module.check_prefixes(folder)

    def test_smoke_records_no_generation_without_claiming_branch_exercised(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);module.save(root/'protocol.json',{})
            run=SimpleNamespace(root=root);sample=samples()[0];folder=root/'samples'/sample['sample_id']
            def fake_evaluate(run,where,point,sample,protocol,metric):
                module.save(where/point['name']/'result.json',{'fixture':point['name']})
                return dict(decode={'generation_executed':False,'route':{'indices':[],'coverage':[0.]*16}}),metric
            with patch.object(module,'prepare_source',return_value=folder),patch.object(module,'evaluate_point',side_effect=fake_evaluate):
                module.smoke(run,{'sources':[{'sample':sample}]},None)
            saved=module.read(root/'smoke.complete.json')
            self.assertEqual(saved['actual_G_calls'],0)
            self.assertFalse(saved['generation_executed']);self.assertIsNone(saved['quality_threshold'])
            self.assertEqual(len(saved['artifacts']),3)

    def test_requires_tmux_and_finite_bounded_hours(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'tmux'):module.main([])
        with patch.dict('os.environ',{'TMUX':'unit-test'}):
            for argv in (['--wait-hours','49'],['--max-hours','13'],['--max-hours','nan'],['--wait-hours','0']):
                with self.assertRaises(ValueError):module.main(argv)

    def test_verify_only_preflight_never_creates_inputs_or_waits_for_models(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',{'TMUX':'unit-test'}):
            root=Path(temp); manifest=root/'manifest.json'
            module.save(manifest,dict(samples=samples()))
            with patch('demo.chunk_enhancement_experiment.Run',side_effect=AssertionError('Run constructed')), \
                 patch.object(module,'immutable',side_effect=AssertionError('immutable called')), \
                 patch.object(module,'wait_for_models',side_effect=AssertionError('wait called')):
                for missing in ('request.json','protocol.json','complete.json'):
                    output=root/missing.removesuffix('.json')
                    for name in ('request.json','protocol.json','complete.json'):
                        if name != missing: module.save(output/name,{})
                    before={str(path.relative_to(root)) for path in root.rglob('*')}
                    with self.assertRaisesRegex(ValueError,f'verify-only requires existing {missing}'):
                        module.main(['--verify-only','--manifest',str(manifest),'--output',str(output)])
                    self.assertEqual(before,{str(path.relative_to(root)) for path in root.rglob('*')})
                output=root/'incomplete-models'
                for name in ('request.json','protocol.json','complete.json'):module.save(output/name,{})
                before={str(path.relative_to(root)) for path in root.rglob('*')}
                with self.assertRaisesRegex(ValueError,'completed formal models; it will not wait'):
                    module.main(['--verify-only','--manifest',str(manifest),'--output',str(output),
                                 '--models',str(root/'missing-models')])
                self.assertEqual(before,{str(path.relative_to(root)) for path in root.rglob('*')})


if __name__=='__main__':
    unittest.main()
