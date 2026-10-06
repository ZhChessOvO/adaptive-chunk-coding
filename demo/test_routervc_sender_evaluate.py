"""CPU orchestration contracts for the asymmetric sender RD diagnostic."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from demo import routervc_sender_evaluate as evaluation
from demo import routervc_receiver_format as fmt
from demo import routervc_visual_format as oldfmt
from demo.routervc_light_packets import subset_bank, bank_info
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from demo.test_routervc_encode import make_bank
from demo.test_routervc_receiver_evaluate import configs


class SenderEvaluation(unittest.TestCase):
    def test_three_senders_share_budgets(self):
        points=evaluation.point_plan()
        self.assertEqual(len(points),9)
        self.assertEqual(len({p['name'] for p in points}),9)
        for arm in evaluation.ARMS:
            self.assertEqual([p['ratio'] for p in points if p['arm']==arm],[0.,.25,.5])
        self.assertTrue(all(p['max_g']==8 for p in points))
        self.assertEqual(13*len(points),117)

    def test_missing_formal_training_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(evaluation.completed_models(Path(tmp)))

    def test_smoke_cannot_be_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'formal').mkdir()
            evaluation.save(root/'complete.json',dict(complete=True))
            evaluation.save(root/'formal/protocol.json',dict(smoke=True))
            with self.assertRaisesRegex(ValueError,'formal sender training'):
                evaluation.completed_models(root)

    def test_verify_partial_never_decodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=SimpleNamespace(root=Path(tmp))
            with patch.object(evaluation,'decode',side_effect=AssertionError('no decode')), \
                    self.assertRaisesRegex(ValueError,'completed evaluation'):
                evaluation.evaluate(run,{},verify_only=True)

    def test_fresh_decoder_never_receives_source_or_sender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=dict(receiver=dict(path='Rg.pt'),enhancement=dict(path='E.pt'),adapter=dict(path='G.pt'))
            with patch('demo.conditioned_generation_pipeline.execute') as execute:
                evaluation.decode(None,protocol,root/'stream.rvrc',root/'off','full_E',off=True)
            args=execute.call_args.args[3]
            self.assertIn('--disable-generation',args)
            self.assertIn(root/'off/absent_Rg.pt',args)
            self.assertIn(root/'off/absent_G.pt',args)
            self.assertNotIn('--source',args)
            self.assertNotIn('--sender',args)

    def test_decode_restart_does_not_repeat_finished_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evaluation.save(root/'decode.json',{})
            with patch('demo.conditioned_generation_pipeline.execute',side_effect=AssertionError('no rerun')):
                evaluation.decode(None,{},root/'stream',root,'finished')

    def test_fixed_old_payload_new_common_noise_config_and_replay(self):
        old_config,config=configs();config['seed']=20261005
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);historical=root/'old';historical.mkdir()
            base=np.zeros((17,64,96,3),np.uint8);all_e=np.full_like(base,20)
            bank=make_bank(base,q=2.);inner=subset_bank(bank,[5])
            atomic_bytes(historical/'stream.rtvc',oldfmt.wrap(inner,old_config))
            evaluation.save(historical/'result.json',dict(complete=True))
            evaluation.save(root/'prior.json',{})
            protocol=dict(config=config,receiver=dict(path='Rg.pt'),prior_protocol=str(root/'prior.json'))
            evaluation.save(root/'protocol.json',protocol)
            candidate=evaluation.encoder.CandidateCache(bank,base,base,all_e,{},0.)
            point=next(p for p in evaluation.point_plan() if p['arm']=='fixed_old_E' and p['ratio']==.5)
            reference=(historical,{},dict(plan=dict(selected_indices=[5])))
            with patch.object(evaluation.receiver_eval,'reference',return_value=reference), \
                    patch('demo.routervc_receiver_policy.route',return_value=dict(indices=[2])) as route:
                result=evaluation.encode_point(root,protocol,dict(sample_id='s'),point,candidate)
                before=route.call_count
                self.assertEqual(result,evaluation.encode_point(root,protocol,dict(sample_id='s'),point,candidate))
                self.assertEqual(route.call_count,before)
            parsed=fmt.parse((root/'samples/s'/point['name']/'stream.rvrc').read_bytes())
            self.assertEqual(parsed[1],inner)
            self.assertEqual(parsed[0]['seed'],20261005)
            self.assertEqual(result['ledger']['total_bytes'],len(parsed[1])+parsed[3])
            self.assertEqual(result['ledger']['packet_bytes'],bank_info(bank)['e_bytes'][5])
            self.assertIsNone(result['ledger']['allocation_seconds'])

    def test_saved_order_reused_and_model_binding_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=dict(config={'seed':1},models=dict(arms=dict(source=dict(path='source.pt',sha256='a'*64)),
                completion='complete.json',completion_sha256='b'*64))
            plan=dict(sender=dict(sha256='a'*64),config={'seed':1},order=[3,4])
            with patch.object(evaluation.encoder,'load_sender',return_value=object()) as load, \
                    patch.object(evaluation.encoder,'conditional_order',return_value=plan) as order, \
                    patch.object(evaluation.encoder,'encode_prefix',return_value=(b'',{})) as validate:
                one=evaluation.load_order(root,protocol,'s','source',None)
                two=evaluation.load_order(root,protocol,'s','source',None)
                self.assertEqual(one,two);self.assertEqual(load.call_count,1);self.assertEqual(order.call_count,1)
                validate.assert_called_once_with(None,plan,0)
                changed=deepcopy(protocol);changed['models']['arms']['source']['sha256']='c'*64
                with self.assertRaisesRegex(ValueError,'order/model'):
                    evaluation.load_order(root,changed,'s','source',None)

    def test_changed_external_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source').write_bytes(b'changed')
            protocol=dict(inputs=dict(s=dict(source_path=str(root/'source'),source_sha256='0'*64)))
            with self.assertRaisesRegex(ValueError,'input changed'):
                evaluation.verify_sample(root,protocol,dict(sample_id='s'))

    def test_completed_replay_cannot_reallocate_decode_score_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=dict(points=evaluation.point_plan(),receiver={'path':'Rg.pt'},expected_points=18,
                sources=[dict(sample=dict(sample_id='reds',dataset='REDS')),
                         dict(sample=dict(sample_id='uvg',dataset='UVG'))])
            evaluation.save(root/'complete.json',dict(artifacts={}))
            for entry in protocol['sources']:
                for point in protocol['points']:
                    folder=root/'samples'/entry['sample']['sample_id']/point['name']
                    (folder/'fresh').mkdir(parents=True)
                    evaluation.save(folder/'result.json',{})
                    evaluation.save(folder/'fresh/decode.json',{'output_hash':'same'})
                    atomic_bytes(folder/'stream.rvrc',b'header'+b'E'*evaluation.RATIOS.index(point['ratio']))
            before={str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns)
                    for p in root.rglob('*') if p.is_file()}
            run=SimpleNamespace(root=root,check=lambda:None,update=lambda **kw:None)
            forbidden=AssertionError('completed replay must be read-only')
            with patch.object(evaluation,'verify_sample') as verify, \
                    patch.object(evaluation,'check_point',return_value={}), \
                    patch.object(evaluation.receiver_eval,'recovery_checks',return_value=[]) as checks, \
                    patch.object(evaluation,'prepare_candidates',side_effect=forbidden), \
                    patch.object(evaluation,'encode_point',side_effect=forbidden), \
                    patch.object(evaluation,'decode',side_effect=forbidden), \
                    patch.object(evaluation,'save',side_effect=forbidden), \
                    patch('demo.scalable_experiment.quality',side_effect=forbidden), \
                    patch('demo.stage_c_three_path_roi_probe.LPIPSAlex',side_effect=forbidden):
                evaluation.evaluate(run,protocol,verify_only=True)
                self.assertEqual(verify.call_count,2)
                self.assertEqual(checks.call_count,4)
                self.assertTrue(all(c.kwargs['verify_only'] for c in checks.call_args_list))
            after={str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns)
                   for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_saved_result_and_sender_plan_are_pinned(self):
        names=evaluation.code_hashes()
        self.assertIn('routervc_sender_encode.py',names)
        self.assertIn('routervc_sender_evaluate.py',names)
        self.assertIn('routervc_receiver_decode.py',names)


if __name__ == '__main__':
    unittest.main()
