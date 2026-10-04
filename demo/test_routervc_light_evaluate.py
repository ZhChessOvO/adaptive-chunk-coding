import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from demo import routervc_light_evaluate as evaluate
from demo import routervc_light_decode as decode
from demo.test_routervc_encode import make_bank
from demo.routervc_light_teacher import save,digest
from demo.scalable_codec import atomic_npz
from demo.scalable_format import frame_hash


class LightEvaluationTests(unittest.TestCase):
    def test_paired_budget_plan(self):
        plan=evaluate.point_plan()
        self.assertEqual(len(plan),12)
        self.assertEqual(len({p['name'] for p in plan}),12)
        for version in ('old','new'):
            for arm in evaluate.ARMS:
                self.assertEqual([p['ratio'] for p in plan if p['version']==version and p['arm']==arm],[0.,.25,.5])
        self.assertTrue(all(p['max_g']==8 for p in plan))

    def test_sender_uses_same_q2_bank_costs_and_no_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sid='sample';folder=root/'samples'/sid;folder.mkdir(parents=True)
            base=np.zeros((17,64,96,3),np.uint8);enhanced=np.full_like(base,20)
            bank=make_bank(base,q=2.);path=root/'bank.acse';path.write_bytes(bank)
            atomic_npz(folder/'candidates.npz',base=base,all_E=enhanced)
            configs={f'{v}_{a}':dict(version=v,arm=a) for v in ('old','new') for a in evaluate.ARMS}
            p=dict(inputs={sid:dict(bank_path=str(path),bank_sha256=digest(path))},points=evaluate.point_plan(),
                   models={v:dict(arms={a:dict(path=f'{v}/{a}') for a in evaluate.ARMS}) for v in ('old','new')},configs=configs)
            save(root/'protocol.json',p)
            save(folder/'prepare.complete.json',dict(protocol=digest(root/'protocol.json'),
                base_hash=frame_hash(base),all_E_hash=frame_hash(enhanced),
                artifacts={'candidates.npz':digest(folder/'candidates.npz')}))
            utility=np.tile([0.,.1,.2,.4],(16,1));pred=np.zeros((17,16,6))
            with patch('demo.routervc_visual_policy.load_model',side_effect=lambda path:path), \
                    patch.object(evaluate,'predict_utility',return_value=(utility,pred)) as predict, \
                    patch('demo.routervc_visual_policy.route',return_value={'indices':[1]}), \
                    patch('demo.routervc_visual_format.wrap',side_effect=lambda inner,config:b'fixedheader'+inner):
                evaluate.encode_sample(root,sid)
                self.assertEqual(predict.call_count,4)
                before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.rglob('*.json')}
                evaluate.encode_sample(root,sid)
                self.assertEqual(predict.call_count,4)
                self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.rglob('*.json')})
            for arm in evaluate.ARMS:
                for ratio in (0.,.25,.5):
                    values=[evaluate.read(folder/f'{v}_{arm}_e{ratio:g}_g8/encode.json') for v in ('old','new')]
                    self.assertEqual(values[0]['plan']['budget_e_packet_bytes'],values[1]['plan']['budget_e_packet_bytes'])

    def test_memory_resets_accumulate_without_changing_pixel_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp))
            def original(args):
                torch.cuda.reset_peak_memory_stats();torch.cuda.reset_peak_memory_stats()
                return dict(peak_cuda_allocated_bytes=3,generation_runtime={'windows':[{'runtime':{'peak_cuda_allocated_bytes':4}}]},output_hash='unchanged')
            with patch('demo.routervc_visual_decode.receive',side_effect=original), \
                    patch('torch.cuda.reset_peak_memory_stats'), \
                    patch('torch.cuda.max_memory_allocated',side_effect=[10,4,3]), \
                    patch('torch.cuda.max_memory_reserved',side_effect=[20,8,6]):
                result=decode.receive(args)
            self.assertEqual(result['peak_cuda_allocated_bytes'],10)
            self.assertEqual(result['generation_all_calls_peak_cuda_bytes'],4)
            self.assertEqual(result['output_hash'],'unchanged')


if __name__=='__main__':unittest.main()
