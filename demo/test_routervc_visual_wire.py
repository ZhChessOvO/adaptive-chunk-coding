"""CPU-only visual profile/entrypoint contracts; native/G execution is stubbed.

These tests are not a claim that a new trained Router passed GPU fresh decode.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from demo import routervc_format as old_format
from demo import routervc_visual_format as fmt
from demo import routervc_visual_encode as sender
from demo import routervc_visual_decode as receiver
from demo.routervc_encode import bank_info, subset_bank
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash, parse as parse_inner
from demo.test_routervc_encode import make_bank


def config(max_g=0):
    return dict({key:'0'*64 for key in fmt.HASHES}, seed=1, max_g=max_g,
        boundary_lambda=.004, strength=1., blend=1., window=17, stride=8, context=64,
        feather=16, policy=fmt.policy_identity())


class WireTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((17,64,96,3), np.uint8)
        self.enhanced = np.full_like(self.base, 64)
        self.bank = make_bank(self.base)

    def test_same_310_byte_header_and_literal_prefix_no_masks(self):
        c = config()
        a = fmt.wrap(subset_bank(self.bank,[5]), c)
        b = fmt.wrap(subset_bank(self.bank,[5,1]), c)
        self.assertTrue(b.startswith(a))
        got, inner, parsed, header = fmt.parse(a)
        self.assertEqual(got, c); self.assertEqual(header,310)
        self.assertEqual(len(a), header+parsed.base_end+sum(len(p.wire) for p in parsed.packets))
        self.assertEqual(old_format.parse(a)[1],inner)
        self.assertFalse({'generate','protect','mask'} & set(got))

    def test_new_profile_rejects_old_or_unknown_but_old_stream_is_unchanged(self):
        c = config(); c['policy'] = old_format.policy_identity()
        wire = old_format.wrap(self.bank,c)
        self.assertEqual(old_format.parse(wire)[0],c)
        with self.assertRaisesRegex(ValueError,'profile mismatch'):
            fmt.parse(wire)
        with self.assertRaisesRegex(ValueError,'profile mismatch'):
            fmt.wrap(self.bank,c)
        c['policy'] = 'f'*64
        with self.assertRaises(ValueError):
            fmt.parse(old_format.wrap(self.bank,c))

    def test_legacy_generated_policy_rejects_visual_profile(self):
        from demo import routervc_decode as legacy
        with patch.object(legacy,'file_hash',return_value='0'*64), patch.object(legacy,'load_router') as load:
            with self.assertRaisesRegex(ValueError,'model/policy mismatch'):
                legacy.route(self.base,self.base,parse_inner(self.bank),config(4),Path('unused'))
            load.assert_not_called()

    def test_corrupt_and_partial_tail_keep_accounting(self):
        wire = fmt.wrap(subset_bank(self.bank,[0]),config())
        with self.assertRaises(ValueError):
            fmt.parse(wire[:-1])
        _,_,inner,header = fmt.parse(wire[:-1],allow_incomplete_tail=True)
        self.assertEqual(len(inner.packets),2)
        self.assertEqual(header+inner.base_end+sum(len(p.wire) for p in inner.packets)+inner.incomplete_tail_bytes,len(wire)-1)
        with self.assertRaises(ValueError):
            fmt.parse(wire[:25]+bytes([wire[25]^1])+wire[26:])

    def test_profile_binds_new_preprocessing_and_reused_helpers(self):
        hashes = fmt.code_identity()
        self.assertIn('routervc_visual_router.py',hashes)
        self.assertIn('routervc_encode.py',hashes)
        baseline = fmt.policy_identity()
        changed = dict(hashes); changed['routervc_visual_router.py']='0'*64
        with patch.object(fmt,'code_identity',return_value=changed):
            self.assertNotEqual(fmt.policy_identity(),baseline)

    def test_selection_budget_and_nested_prefixes(self):
        utility = np.zeros((16,4)); utility[:,1] = np.arange(1,17)/100
        utility[:,2] = .03; utility[:,3] = utility[:,1]+.02
        prediction = np.zeros((17,16,6))
        with patch.object(sender.policy,'predict_utility',return_value=(utility,prediction)):
            a = sender.select(self.bank,self.base,self.enhanced,object(),400,4)
            b = sender.select(self.bank,self.base,self.enhanced,object(),2000,4)
            c = sender.select(self.bank,self.base,self.enhanced,object(),0,0,mode='independent')
        self.assertTrue(subset_bank(self.bank,b['selected_indices']).startswith(subset_bank(self.bank,a['selected_indices'])))
        for budget,plan in ((400,a),(2000,b),(0,c)):
            used = len(subset_bank(self.bank,plan['selected_indices']))-parse_inner(self.bank).base_end
            self.assertEqual(plan['e_packet_bytes'],used); self.assertLessEqual(used,budget)
            self.assertFalse(plan['semantic_heads_used']); self.assertFalse(plan['source_frames_used_by_router'])
        self.assertEqual(c['selected_indices'],[])

    def test_sender_source_entrypoint_and_no_recompute_resume(self):
        from demo.test_routervc_visual_policy import fixture_payload
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'encode'; output.mkdir()
            source=root/'source.npz'; atomic_npz(source,source=self.base)
            router=root/'model.pt'; torch.save(fixture_payload(),router)
            enhancement=root/'E.pt'; enhancement.write_bytes(b'E-model-fixture')
            args=SimpleNamespace(input=source,input_key='source',start=0,count=17,output=output,
                router=router,enhancement=enhancement,adapter=root/'G.pt',prepared_dir=None,
                e_budget=800,e_ratio=None,max_g=0,boundary_lambda=.004,seed=1,mode='prefix')
            c=config(); c['router']=file_hash(router)
            def prepare(path, destination, **kwargs):
                destination.mkdir()
                atomic_bytes(destination/'bank.acse',self.bank)
                atomic_npz(destination/'candidates.npz',base=self.base,all_E=self.enhanced)
                record=dict(all_E_rgb_sha256=frame_hash(self.enhanced))
                atomic_json(destination/'complete.json',record)
                return record
            run=SimpleNamespace(check=lambda:None,update=lambda **kw:None)
            with patch.object(fmt,'make_config',return_value=c), patch.object(sender,'prepare',side_effect=prepare) as prepared:
                result=sender.encode(args,run)
                before=(output/'encode.json').stat().st_mtime_ns
                self.assertEqual(sender.encode(args,run),result)
                self.assertEqual((output/'encode.json').stat().st_mtime_ns,before)
                prepared.assert_called_once()
            self.assertEqual(sum(result['byte_breakdown'].values()),(output/'stream.rtvc').stat().st_size)
            self.assertFalse(result['generation_executed_at_sender'])
            self.assertIsNone(result['expected_shared_route'])
            self.assertEqual(result['byte_breakdown']['explicit_E_mask_bytes'],0)
            with patch.object(fmt,'make_config',return_value=c):
                args.e_budget=900
                with self.assertRaises(RuntimeError):
                    sender.encode(args,run)

    def test_invalid_router_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'encode'; output.mkdir()
            source=root/'source.npz'; atomic_npz(source,source=self.base)
            router=root/'model.pt'; router.write_bytes(b'not-a-model')
            enhancement=root/'E.pt'; enhancement.write_bytes(b'E')
            args=SimpleNamespace(input=source,input_key='source',start=0,count=17,output=output,
                router=router,enhancement=enhancement,adapter=root/'G',prepared_dir=None,
                e_budget=0,e_ratio=None,max_g=0,boundary_lambda=0.,seed=1,mode='prefix')
            c=config(); c['router']=file_hash(router)
            with patch.object(fmt,'make_config',return_value=c), patch.object(sender,'prepare') as prepare:
                with self.assertRaisesRegex(ValueError,'checkpoint'):
                    sender.encode(args,SimpleNamespace(update=lambda **kw:None))
                prepare.assert_not_called()

    def receive_fixture(self, root, *, indices=(), truncated=False, generated=False):
        base = np.zeros((17,256,384,3),np.uint8) if generated else self.base
        bank = make_bank(base); inner = subset_bank(bank,list(indices)); c = config(1 if generated else 0)
        wire = fmt.wrap(inner,c)
        stream = root/'input.rtvc'; atomic_bytes(stream,wire[:-1] if truncated else wire)
        output = root/'receiver'; output.mkdir()
        args = SimpleNamespace(stream=stream,output=output,router=None,enhancement=root/'missing-E',
            adapter=root/'missing-G',disable_generation=not generated,allow_incomplete_tail=truncated)
        models = {key:'0'*64 for key in ('model_i_sha256','model_p_sha256')}
        codec = SimpleNamespace(models=models,decode=Mock(return_value=base.copy()))
        def decode_e(model, path, codec_arg, raw, **kwargs):
            parsed = parse_inner(raw,allow_incomplete_tail=truncated); enhanced=base.copy()
            for packet in parsed.packets:
                pm=packet.meta; x,y,w,h=pm['roi']; start,count=pm['start'],pm['count']
                enhanced[start:start+count,y:y+h,x:x+w]=64
            report = dict(base_bytes=len(parsed.base),container_header_bytes=parsed.base_end-len(parsed.base),
                packet_bytes=sum(len(p.wire) for p in parsed.packets),incomplete_tail_bytes=parsed.incomplete_tail_bytes,
                base_hash=frame_hash(base),non_enhanced_exact=True,applied_packets=[])
            return enhanced,report,base.copy()
        stack=ExitStack()
        for name in ('reset_peak_memory_stats','empty_cache'):
            stack.enter_context(patch.object(receiver.torch.cuda,name))
        stack.enter_context(patch.object(receiver.torch.cuda,'max_memory_allocated',return_value=0))
        stack.enter_context(patch.object(receiver,'configure_torch'))
        stack.enter_context(patch.object(receiver,'BaseCodec',return_value=codec))
        load=stack.enter_context(patch.object(receiver,'load_model',return_value=object()))
        stack.enter_context(patch.object(receiver,'decode_enhancement',side_effect=decode_e))
        return args,base,c,stack,load

    def test_G_off_needs_no_router_or_generator_and_no_E_without_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,base,c,stack,load=self.receive_fixture(Path(tmp))
            with stack, patch.object(receiver.policy,'route',side_effect=AssertionError('must not route')), \
                    patch.object(receiver,'identities',side_effect=AssertionError('must not read G assets')), \
                    patch.object(receiver,'restore',side_effect=AssertionError('must not generate')):
                report=receiver.receive(args)
            load.assert_not_called()
            self.assertFalse(report['shared_router_used']); self.assertFalse(report['generation_executed'])
            self.assertEqual(report['route']['states'],['B']*16)
            self.assertEqual(report['total_bytes'],args.stream.stat().st_size)
            receiver.validate_decoded(args.output,args.stream,c)
            with np.load(args.output/'reconstruction.npz') as saved:
                np.testing.assert_array_equal(saved['reconstruction'],base)

    def test_G_off_E_packets_and_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,base,c,stack,load=self.receive_fixture(Path(tmp),indices=(5,),truncated=True)
            with stack, patch.object(receiver.policy,'route',side_effect=AssertionError('must not route')):
                report=receiver.receive(args)
            load.assert_called_once()
            self.assertEqual(report['route']['states'][5],'E')
            self.assertAlmostEqual(report['route']['coverage'][5],9/17,places=6)
            self.assertGreater(report['incomplete_tail_bytes'],0)
            receiver.validate_decoded(args.output,args.stream,c)

    def test_generated_path_uses_received_mixed_Y_and_only_changes_selected_ROI(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args,base,c,stack,load=self.receive_fixture(root,indices=(5,),generated=True)
            args.router=root/'router'; args.router.write_bytes(b'router-placeholder')
            c['router']=file_hash(args.router)
            atomic_bytes(args.stream,fmt.wrap(subset_bank(make_bank(base),[5]),c))
            def route(b,y,inner,config,router,**kwargs):
                self.assertEqual(kwargs['expected_policy'],fmt.policy_identity())
                self.assertTrue(np.any(y != b))
                return dict(indices=[5],rois=receiver.validate_generation_geometry(base.shape,1))
            with stack, patch.object(receiver.policy,'route',side_effect=route), \
                    patch.object(receiver,'identities',return_value={k:c[k] for k in receiver.cooperation.HASHES}), \
                    patch.object(receiver,'restore',side_effect=lambda pixels,*_: (pixels+32,{'stub':True})):
                report=receiver.receive(args)
            self.assertTrue(report['outside_generate_exact']); self.assertTrue(report['generation_executed'])
            self.assertEqual(report['explicit_G_map_bytes'],0)
            receiver.validate_decoded(args.output,args.stream,c)

    def test_supervisor_G_off_resume_without_assets_or_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args,base,c,stack,load=self.receive_fixture(root)
            # Move the test's in-process reconstruction under the supervisor's fresh path.
            args.output=root/'supervised'; args.output.mkdir()
            fresh=args.output/'fresh'; fresh.mkdir()
            child_args=SimpleNamespace(**vars(args)); child_args.output=fresh
            with stack:
                receiver.receive(child_args)
            run=SimpleNamespace(check=lambda:None)
            with patch.object(receiver,'execute',side_effect=AssertionError('must reuse fresh files')):
                result=receiver.decode(args,run)
                before=(args.output/'decode.complete.json').stat().st_mtime_ns
                self.assertEqual(receiver.decode(args,run),result)
                self.assertEqual((args.output/'decode.complete.json').stat().st_mtime_ns,before)
            self.assertEqual(result['binding']['used_assets'],dict(router=None,adapter=None,enhancement=None))
            atomic_bytes(fresh/'reconstruction.npz',b'corrupt')
            with self.assertRaises(RuntimeError):
                receiver.decode(args,run)

    def test_encode_requires_explicit_router_and_worker_guard(self):
        with self.assertRaises(SystemExit):
            sender.main(['--input','unused','--output','unused','--e-budget','0'])
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'supervising tmux'):
                sender.main(['--worker','--input','unused','--output','unused','--router','unused','--e-budget','0'])
            with self.assertRaisesRegex(RuntimeError,'supervising tmux'):
                receiver.main(['--worker','--stream','unused','--output','unused','--disable-generation'])
            with self.assertRaisesRegex(RuntimeError,'tmux'):
                fmt.supervised(SimpleNamespace(max_hours=1),lambda *_:None)


if __name__ == '__main__':
    unittest.main()
