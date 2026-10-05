"""CPU tests for sparse FINAL sender labels and source-free renderer callbacks."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from demo import routervc_sender_data as data
from demo import routervc_receiver_format as fmt
from demo.routervc_visual_router import grid_rois
from demo.routervc_fullview_probe import save, digest
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_format import frame_hash
from demo.test_routervc_encode import make_bank


def config():
    return dict({key:'0'*64 for key in fmt.HASHES}, policy=fmt.policy_identity(),
        seed=20261005, max_g=8, boundary_lambda=0., strength=1., blend=1.,
        window=17, stride=8, context=64, feather=16)


class SenderDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.base = np.zeros((17,64,96,3),np.uint8)
        self.all_e = np.full_like(self.base,100)
        self.source = np.full_like(self.base,40)
        self.bank = make_bank(self.base,q=2.)
        self.config = config()
        self.costs = data.bank_info(self.bank)['e_bytes']

    def route(self, base, received, inner, settings, receiver, *, expected_policy):
        self.assertIs(base,self.base)
        self.assertFalse(np.shares_memory(received,self.source))
        self.assertEqual(expected_policy,fmt.policy_identity())
        self.assertEqual(settings,self.config)
        # Exercise changed route/ordinal selection after adding E to region 0.
        indices = [1,2] if received[0,0,0,0] else [0,2]
        return dict(indices=indices,rois=grid_rois(64,96), source_frames_used_by_router=False)

    def generate(self, pixels, control):
        self.assertFalse(np.shares_memory(pixels,self.source))
        self.assertEqual(set(control)-set(self.config), {'processing_scale','protect','generate'})
        result=pixels.copy()
        for t,n,x,y,w,h in control['generate']:
            result[t:t+n,y:y+h,x:x+w]=60
        from demo import scalable_cooperation_format as cooperation
        result=cooperation.combine(pixels,result,cooperation.weights(pixels.shape,control))
        return result,dict(windows=[],condition_windows=[],seed=control['seed'])

    def score(self, source, result):
        self.assertIs(source,self.source)
        return dict(lpips_alex=float(np.abs(source.astype(float)-result).mean()/255),
                    psnr_db=20.,temporal_delta_mae=.1)

    def measure(self,dest,selected,**kwargs):
        return data.measure_state(dest,self.bank,self.base,self.all_e,selected,self.source,
            self.config,'explicit-receiver',binding={'fixture':True},generate=self.generate,
            score=self.score,route=self.route,**kwargs)

    def test_balanced_fixed_states_preserve_split_and_old_prefix(self):
        rows=[dict(sample_id=f'{split}-{dataset}-{i}',dataset=dataset,router_split=split,sequence=str(i//3))
              for split in ('train','validation') for dataset in ('REDS','UVG') for i in range(24)]
        assigned=data.assign_plans(rows)
        self.assertEqual(data.assign_plans(rows[::-1]),assigned)
        for dataset in ('REDS','UVG'):
            for split in ('train','validation'):
                found=[r for r in assigned if r['dataset']==dataset and r['router_split']==split]
                for count in data.STATE_COUNTS:
                    for kind in ('random','old_prefix'):
                        self.assertEqual(sum(r['sender_plan']['count']==count and
                                             r['sender_plan']['kind']==kind for r in found),3)
        row=next(r for r in assigned if r['sender_plan']['kind']=='old_prefix')
        old=list(reversed(range(16)))
        plan=data.plan_states(row,old,self.costs)
        self.assertEqual(plan[0]['selected'],[])
        self.assertEqual(plan[1]['selected'],old[:row['sender_plan']['count']])
        self.assertEqual(plan,data.plan_states(row,old,self.costs))

    def test_three_unique_unreceived_candidates_and_known_proposal_roles(self):
        order=list(reversed(range(16)))
        for selected in ([],order[:2],order[:4],order[:8],order[:12]):
            chosen=data.candidates(selected,order,self.costs,'window')
            self.assertEqual(len(set(chosen)),3)
            self.assertTrue(set(chosen).isdisjoint(selected))
            self.assertEqual(chosen[0],next(i for i in order if i not in selected))
            self.assertEqual(chosen,data.candidates(selected,order,self.costs,'window'))
        with self.assertRaises(ValueError):
            data.candidates(order[:14],order,self.costs,'window')

    def test_short_legacy_prefix_is_explicit_random_fallback_not_fake_prefix(self):
        utility=np.zeros((16,4),float)
        utility[:,1]=-np.arange(1,17)/10.
        utility[:,3]=utility[:,1]
        old=data.allocate(utility,np.asarray(self.costs),sum(self.costs),8,mode='prefix')['prefix_order']
        self.assertEqual(old,[])
        proposal=data.complete_proposal_order(utility,self.costs,old)
        self.assertEqual(sorted(proposal),list(range(16)))
        row=dict(sample_id='short',sender_plan=dict(count=12,kind='old_prefix',seed=7))
        plans=data.plan_states(row,old,self.costs,proposal)
        self.assertEqual(plans[1]['kind'],'random_fallback')
        self.assertEqual(plans[1]['requested_kind'],'old_prefix')
        self.assertEqual(plans[1]['original_old_prefix'],[])
        self.assertEqual(plans[1]['requested_count'],12)
        self.assertEqual(plans[1]['actual_count'],12)
        self.assertIsNotNone(plans[1]['fallback_reason'])
        self.assertEqual(len(plans[1]['candidates']),3)
        prefix=[3,1]
        proposal=data.complete_proposal_order(utility,self.costs,prefix)
        self.assertEqual(proposal[:2],prefix)
        row['sender_plan']['count']=2
        plans=data.plan_states(row,prefix,self.costs,proposal)
        self.assertEqual(plans[1]['selected'],prefix)
        self.assertEqual(plans[1]['kind'],'old_prefix')

    def test_every_new_parent_child_reroutes_and_resume_never_regenerates(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)
            parent=self.measure(dest/'parent',[])
            child=self.measure(dest/'child',[0])
            self.assertNotEqual(parent['route']['indices'],child['route']['indices'])
            self.assertNotEqual(parent['render_key'],child['render_key'])
            self.assertEqual(child['total_bytes']-parent['total_bytes'],self.costs[0])
            self.assertEqual(parent['header_bytes'],child['header_bytes'])
            self.assertFalse((dest/'parent/reconstruction.npz').exists())
            self.assertFalse(parent['source_frames_used_by_receiver'])
            self.assertFalse(parent['source_frames_used_by_generator'])
            self.assertEqual(parent['explicit_mask_bytes'],0)
            original=digest(dest/'parent/result.json')
            fail=Mock(side_effect=AssertionError('completed state must not reroute/render/score'))
            replay=data.measure_state(dest/'parent',self.bank,self.base,self.all_e,[],self.source,
                self.config,'explicit-receiver',binding={'fixture':True},generate=fail,score=fail,route=fail)
            self.assertEqual(replay,parent)
            self.assertEqual(digest(dest/'parent/result.json'),original)
            fail.assert_not_called()

    def test_fixed_smoke_pixels_retained_and_asset_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)
            result=self.measure(dest,[],retain_pixels=True)
            with np.load(dest/'reconstruction.npz',allow_pickle=False) as pixels:
                self.assertEqual(frame_hash(pixels['reconstruction']),result['output_hash'])
                self.assertEqual(frame_hash(pixels['enhanced']),result['generation_input_hash'])
            with self.assertRaises(ValueError):
                self.measure(dest,[1],retain_pixels=True)
            with patch.object(data,'verify_artifacts',side_effect=ValueError('tampered')):
                with self.assertRaises(ValueError):
                    self.measure(dest,[],retain_pixels=True)

    def test_eight_full_renders_and_receiver_callbacks_never_receive_gt(self):
        row=dict(sample_id='window',sender_plan=dict(kind='old_prefix',count=4,seed=3))
        plans=data.plan_states(row,list(range(16)),self.costs)
        route=Mock(side_effect=self.route)
        generate=Mock(side_effect=self.generate)
        score=Mock(side_effect=self.score)
        with tempfile.TemporaryDirectory() as directory:
            records=[]
            for state,plan in enumerate(plans):
                measured={}
                choices=[('parent',plan['selected'])]+[(str(i),plan['selected']+[i]) for i in plan['candidates']]
                for name,selected in choices:
                    measured[name]=data.measure_state(Path(directory)/str(state)/name,self.bank,self.base,
                        self.all_e,selected,self.source,self.config,'explicit-receiver',binding={'fixture':True},
                        generate=generate,score=score,route=route)
                records.append(dict(parent=measured['parent'],children={str(i):measured[str(i)] for i in plan['candidates']}))
            targets,_=data.measured_targets(plans,records,self.costs)
            self.assertEqual(targets['weight'].sum(),6)
            self.assertEqual(route.call_count,8)
            self.assertEqual(generate.call_count,8)
            self.assertEqual(score.call_count,16)
            for call in route.call_args_list+generate.call_args_list:
                self.assertFalse(any(value is self.source for value in call.args))

    def test_render_identity_keeps_order_and_noise_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            first=self.measure(Path(directory)/'a',[])
            def reversed_route(*args,**kwargs):
                result=self.route(*args,**kwargs)
                result['indices']=result['indices'][::-1]
                return result
            second=data.measure_state(Path(directory)/'b',self.bank,self.base,self.all_e,[],self.source,
                self.config,'explicit-receiver',binding={'fixture':True},generate=self.generate,
                score=self.score,route=reversed_route)
            self.assertEqual(first['generation_input_hash'],second['generation_input_hash'])
            self.assertNotEqual(first['render_key'],second['render_key'])
            self.assertIn('65536',first['render_identity']['noise_scheme'])
            self.assertEqual(first['render_identity']['wire_config']['receiver_router'],self.config['receiver_router'])

    def test_six_masked_targets_negative_marginal_and_actual_bytes(self):
        plans=[dict(candidates=[0,1,2]),dict(candidates=[3,4,5])]
        records=[]
        for plan in plans:
            parent=dict(total_bytes=1000,quality={'lpips_alex':.2},Goff_quality={'lpips_alex':.3},
                        route={'indices':[0,2]})
            children={str(i):dict(total_bytes=1000+self.costs[i],quality={'lpips_alex':.25 if n==0 else .1},
                Goff_quality={'lpips_alex':.25},route={'indices':[1,2]}) for n,i in enumerate(plan['candidates'])}
            records.append(dict(parent=parent,children=children))
        target,diagnostics=data.measured_targets(plans,records,self.costs)
        self.assertEqual(target['label_scope'],data.LABEL_SCOPE)
        self.assertEqual(target['weight'].sum(),6)
        self.assertLess(target['value'][0,0],0)
        self.assertEqual(target['weight'][0,15],0)
        self.assertEqual(len(diagnostics),6)
        self.assertTrue(all(r['G_selection_changed'] for r in diagnostics))
        wrong=deepcopy(records);wrong[0]['children']['0']['total_bytes']+=310
        with self.assertRaises(ValueError):
            data.measured_targets(plans,wrong,self.costs)

    def test_explicit_receiver_hash_and_formal_completion_required(self):
        fake=dict(arm='core',epoch=1,binding={'protocol':{'smoke':True}})
        with patch('demo.routervc_receiver_router.load_model',return_value=(None,fake)) as load:
            with self.assertRaisesRegex(ValueError,'completed R_g'):
                data.make_settings('/nonexistent/receiver.pt','0'*64,smoke=False)
            load.assert_called_once()
        with self.assertRaises(ValueError):
            data.make_settings('/not/guessed',None,smoke=True)

    def test_formal_receiver_cannot_use_smoke_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'core/best.pt';checkpoint.parent.mkdir()
            atomic_torch(checkpoint,{'fixture':True})
            sha=digest(checkpoint)
            save(root/'complete.json',dict(complete=True,artifacts={'core/best.pt':sha}))
            fake=dict(arm='core',epoch=1,binding={'protocol':{'smoke':True}})
            with patch('demo.routervc_receiver_router.load_model',return_value=(None,fake)):
                with self.assertRaisesRegex(ValueError,'smoke-only'):
                    data.make_settings(checkpoint,sha,smoke=False,receiver_complete=root/'complete.json')

    def test_bounded_cache_no_implicit_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'formal';cache=Path(directory)/'cache';cache.mkdir()
            protocol=dict(teacher=dict(label_scope=data.LABEL_SCOPE,max_g=8,wire_config={'max_g':8}))
            samples=data.Samples(root,cache,protocol,max_cached=2)
            for i in range(3):
                row=dict(sample_id=str(i),dataset='REDS')
                path=cache/f'{i}.pt';atomic_torch(path,{'fixture':i})
                dest=root/'samples'/str(i);dest.mkdir(parents=True)
                save(dest/'complete.json',dict(binding=dict(protocol=samples.protocol_hash,sample=row,
                     teacher=protocol['teacher']),cache_sha256=digest(path),artifacts={},reused_artifacts={}))
                self.assertEqual(samples.get(row,generate=False)['fixture'],i)
            self.assertEqual(list(samples.memory),['1','2'])
            with self.assertRaisesRegex(ValueError,'cannot generate'):
                samples.get(dict(sample_id='missing'),generate=False)
            with self.assertRaises(ValueError):
                data.Samples(root,cache,protocol,max_cached=3)


if __name__=='__main__':
    unittest.main()
