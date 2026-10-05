"""CPU fixed-E receiver evaluation contracts; no model inference on GPU."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from demo import routervc_receiver_evaluate as evaluation
from demo import routervc_receiver_format as newfmt
from demo import routervc_visual_format as oldfmt
from demo.routervc_light_packets import subset_bank
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.scalable_format import frame_hash
from demo.test_routervc_encode import make_bank


def configs():
    old = dict({key:'0'*64 for key in oldfmt.HASHES}, policy=oldfmt.policy_identity(),
        seed=20261003, max_g=8, boundary_lambda=0., strength=1., blend=1.,
        window=17, stride=8, context=64, feather=16)
    new = {('receiver_router' if key == 'router' else key):value for key,value in old.items()}
    new['policy'] = newfmt.policy_identity()
    return old, new


def runtime_report(indices=(0, 1)):
    return dict(route=dict(indices=list(indices), rois=[[i*64,0,64,64] for i in range(16)]),
        generation_runtime=dict(windows=[dict(region=i,start=0,crop=[k*64,0,128,128])
            for i,k in enumerate(indices)], condition_windows=[dict(before_vae=str(i),after_vae=str(i),
            before_diffusion=str(i),diffusion_noise=str(i),conditions=str(i)) for i,_ in enumerate(indices)]))


class ReceiverEvaluationTests(unittest.TestCase):
    def test_fixed_point_count_budgets_and_no_sender_variations(self):
        plan = evaluation.point_plan()
        self.assertEqual(len(plan), 9)
        self.assertEqual(len({p['name'] for p in plan}), 9)
        for arm in ('shared','core','halo'):
            self.assertEqual([p['ratio'] for p in plan if p['arm'] == arm], [0.,.25,.5])
        self.assertEqual(sum(p['reused'] for p in plan), 3)
        self.assertTrue(all(p['max_g'] == 8 for p in plan))

    def test_rewrap_preserves_every_inner_byte_and_real_prefix(self):
        old, new = configs()
        bank = make_bank(np.zeros((17,64,96,3),np.uint8), q=2.)
        previous = None
        for selected in ([],[5],[5,0,15]):
            inner = subset_bank(bank,selected)
            wire, header, inner_hash = evaluation.rewrap_fixed_E(oldfmt.wrap(inner,old),new)
            parsed = newfmt.parse(wire)
            self.assertEqual(parsed[1],inner)
            self.assertEqual(len(wire),len(inner)+header)
            self.assertEqual(header,parsed[3])
            if previous is not None:
                self.assertTrue(wire.startswith(previous))
            previous = wire

    def test_rewrap_rejects_changed_noise_controls_and_q(self):
        old,new = configs()
        bank = make_bank(np.zeros((17,64,96,3),np.uint8),q=2.)
        for key,value in (('seed',20261005),('context',32),('max_g',4)):
            with self.assertRaises(ValueError):
                evaluation.rewrap_fixed_E(oldfmt.wrap(subset_bank(bank,[1]),old),dict(new,**{key:value}))
        wrong = make_bank(np.zeros((17,64,96,3),np.uint8),q=1.)
        with self.assertRaises(ValueError):
            evaluation.rewrap_fixed_E(oldfmt.wrap(wrong,old),new)

    def test_noise_pairing_requires_same_ordinal_and_core_geometry(self):
        old = runtime_report()
        self.assertEqual(evaluation.noise_overlap(old,deepcopy(old))['paired_windows'],2)
        shifted = runtime_report((1,2))
        self.assertEqual(evaluation.noise_overlap(old,shifted)['paired_windows'],0)
        changed = deepcopy(old)
        changed['generation_runtime']['condition_windows'][0]['diffusion_noise'] = 'bad'
        with self.assertRaises(ValueError):
            evaluation.noise_overlap(old,changed)
        self.assertEqual(evaluation.noise_overlap(old,dict(generation_runtime=None))['paired_windows'],0)

    def test_prepare_uses_received_pixels_and_replay_never_repredicts(self):
        old_config,new_config = configs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old_folder=root/'historical';(old_folder/'fresh').mkdir(parents=True)
            base=np.zeros((17,64,96,3),np.uint8);received=np.full_like(base,20)
            inner=subset_bank(make_bank(base,q=2.),[5])
            atomic_bytes(old_folder/'stream.rtvc',oldfmt.wrap(inner,old_config))
            atomic_npz(old_folder/'fresh/reconstruction.npz',base=base,enhanced=received)
            old=dict(decode=dict(base_hash=frame_hash(base),generation_input_hash=frame_hash(received)))
            evaluation.save(old_folder/'result.json',old)
            old_encoded=dict(plan=dict(selected_indices=[5]))
            protocol=dict(configs={'core':new_config},models=dict(arms=dict(core=dict(path='unused-model'))))
            evaluation.save(root/'protocol.json',protocol)
            point=next(p for p in evaluation.point_plan() if p['arm']=='core' and p['ratio']==.25)
            seen=[]
            def route(b,y,*args,**kwargs):
                np.testing.assert_array_equal(b,base);np.testing.assert_array_equal(y,received)
                seen.append(True);return dict(indices=[2])
            with patch.object(evaluation,'reference',return_value=(old_folder,old,old_encoded)), \
                    patch('demo.routervc_receiver_policy.route',side_effect=route):
                first=evaluation.encode_point(root,protocol,'sample',point)
                folder=root/'samples/sample'/point['name']
                before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.iterdir()}
                self.assertEqual(first,evaluation.encode_point(root,protocol,'sample',point))
                self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.iterdir()})
            self.assertEqual(len(seen),1)
            self.assertEqual(first['fixed_E_indices'],[5])
            self.assertEqual(newfmt.parse((folder/'stream.rvrc').read_bytes())[1],inner)

    def test_unfinished_training_is_wait_not_acceptance_of_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(evaluation.completed_models(tmp))

    def test_verify_only_cannot_start_partial_evaluation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            run=SimpleNamespace(root=Path(tmp))
            with self.assertRaises(ValueError), patch('demo.conditioned_generation_pipeline.execute',
                    side_effect=AssertionError('no inference')):
                evaluation.evaluate(run,{},verify_only=True)

    def test_completed_replay_cannot_infer_score_or_rewrite_records(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=dict(points=evaluation.point_plan(),sources=[dict(sample=dict(sample_id=f's{i}',
                dataset='REDS' if i<6 else 'UVG')) for i in range(13)])
            evaluation.save(root/'complete.json',dict(artifacts={}))
            for entry in protocol['sources']:
                sid=entry['sample']['sample_id']
                for point in protocol['points']:
                    folder=root/'samples'/sid/point['name'];folder.mkdir(parents=True)
                    evaluation.save(folder/'result.json',dict(complete=True))
                    if not point['reused']:
                        atomic_bytes(folder/'stream.rvrc',b'header'+b'E'*evaluation.RATIOS.index(point['ratio']))
            before={str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns)
                    for p in root.rglob('*') if p.is_file()}
            run=SimpleNamespace(root=root,check=lambda:None,update=lambda **kw:None)
            forbidden=AssertionError('completed replay must not infer, score or rewrite')
            with patch.object(evaluation,'check_point',return_value={}), \
                    patch.object(evaluation,'recovery_checks',return_value=['repeat','off']) as checks, \
                    patch.object(evaluation,'encode_point',side_effect=forbidden), \
                    patch.object(evaluation,'save',side_effect=forbidden), \
                    patch('demo.conditioned_generation_pipeline.execute',side_effect=forbidden), \
                    patch('demo.scalable_experiment.quality',side_effect=forbidden), \
                    patch('demo.stage_c_three_path_roi_probe.LPIPSAlex',side_effect=forbidden):
                evaluation.evaluate(run,protocol,verify_only=True)
            self.assertEqual(checks.call_count,4)
            self.assertTrue(all(call.kwargs['verify_only'] for call in checks.call_args_list))
            after={str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns)
                   for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)


if __name__=='__main__':
    unittest.main()
