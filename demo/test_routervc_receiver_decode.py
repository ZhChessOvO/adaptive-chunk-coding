"""CPU contracts for asymmetric Rg wire/receiver; native codec/G are stubbed.

These do not replace real entropy decode and fresh GPU smoke validation.
"""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from demo import routervc_format as legacy
from demo import routervc_receiver_format as fmt
from demo import routervc_receiver_policy as policy
from demo import routervc_receiver_decode as receiver
from demo import routervc_receiver_router as rg
from demo.routervc_light_packets import subset_bank
from demo.scalable_codec import atomic_bytes, file_hash
from demo.scalable_format import frame_hash, parse as parse_inner
from demo.test_routervc_encode import make_bank


def config(max_g=0):
    return dict({key: '0'*64 for key in fmt.HASHES}, policy=fmt.policy_identity(),
        seed=1, max_g=max_g, boundary_lambda=0., strength=1., blend=1.,
        window=17, stride=8, context=64, feather=16)


def model_fixture(path, arm='core'):
    model = rg.ReceiverGUtilityRouter(rg.Config(channels=8, hidden=8, local_size=16, global_size=16))
    torch.save(rg.export_payload(model, arm, {'smoke': True}, 1,
                               {'loss': 0., 'regret': 0.}), path)
    return rg.load_model(path)[0]


class ReceiverContracts(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((17, 64, 96, 3), np.uint8)
        self.bank = make_bank(self.base, q=2.)

    def test_distinct_magic_profile_and_actual_header_accounting(self):
        c = config()
        inner = subset_bank(self.bank, [5])
        wire = fmt.wrap(inner, c)
        got, raw, parsed, header = fmt.parse(wire)
        self.assertEqual(got, c)
        self.assertEqual(raw, inner)
        self.assertEqual(wire[:4], fmt.MAGIC)
        self.assertNotEqual(fmt.MAGIC, b'RTVC')
        self.assertEqual(header, fmt.HEADER.size+fmt.CONTROL.size+32*len(fmt.HASHES))
        self.assertEqual(header, len(wire)-len(inner))
        self.assertEqual(len(wire), header+parsed.base_end+sum(len(p.wire) for p in parsed.packets))
        self.assertFalse({'router', 'sender_router', 'generate', 'mask', 'protect'} & set(got))
        with self.assertRaises(ValueError):
            legacy.parse(wire)
        old = {('router' if k == 'receiver_router' else k): v for k, v in c.items()}
        with self.assertRaises(ValueError):
            fmt.parse(legacy.wrap(inner, old))

    def test_literal_complete_and_partial_packet_prefixes(self):
        c = config()
        short = fmt.wrap(subset_bank(self.bank, [5]), c)
        long = fmt.wrap(subset_bank(self.bank, [5, 1]), c)
        self.assertTrue(long.startswith(short))
        _, _, parsed, header = fmt.parse(short)
        for end, packets in ((parsed.base_end, 0), (parsed.packets[0].end_offset, 1),
                             (parsed.packets[1].end_offset, 2), (parsed.packets[2].end_offset, 3)):
            self.assertEqual(len(fmt.parse(short[:header+end])[2].packets), packets)
        with self.assertRaises(ValueError):
            fmt.parse(short[:-1])
        _, _, partial, h = fmt.parse(short[:-1], allow_incomplete_tail=True)
        self.assertEqual(len(partial.packets), 2)
        self.assertEqual(h+partial.base_end+sum(len(p.wire) for p in partial.packets)
                         +partial.incomplete_tail_bytes, len(short)-1)

    def test_corrupt_unsupported_and_mask_config_fail_closed(self):
        c = config()
        wire = fmt.wrap(self.bank, c)
        for bad in (wire[:7], wire[:30], b'BAD!'+wire[4:], wire[:4]+b'\x02'+wire[5:],
                    wire[:25]+bytes([wire[25]^1])+wire[26:]):
            with self.assertRaises(ValueError):
                fmt.parse(bad)
        for key, value in [('policy', 'f'*64), ('receiver_router', 'bad'), ('max_g', True),
                           ('boundary_lambda', float('nan')), ('window', 9), ('mask', []),
                           ('sender_router', '0'*64)]:
            with self.assertRaises(ValueError):
                fmt.wrap(self.bank, {**c, key: value})

    def test_profile_pins_receiver_not_sender(self):
        hashes = fmt.code_identity()
        self.assertIn('routervc_receiver_router.py', hashes)
        self.assertNotIn('routervc_sender_router.py', hashes)
        previous = fmt.policy_identity()
        with patch.object(fmt, 'code_identity', return_value={**hashes, 'routervc_receiver_router.py': '0'*64}):
            self.assertNotEqual(fmt.policy_identity(), previous)

    def test_rg_route_uses_only_conditional_g_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'Rg.pt'
            model = model_fixture(path)
            c = config(4)
            c['receiver_router'] = file_hash(path)
            gains = torch.zeros(1, 16, 3)
            gains[0, :, 0] = torch.arange(16)/100
            gains[0, :, 1:] = -1000  # Diagnostics must not change G selection.
            mixed = self.base.copy()
            rois = rg.visual.grid_rois(64, 96)
            x, y, w, h = rois[5]
            mixed[:, y:y+h, x:x+w] = 80
            inner = parse_inner(subset_bank(self.bank, [5]))
            with patch.object(rg, 'predict', return_value=gains) as predict:
                result = policy.route(self.base, mixed, inner, c, model,
                                      expected_policy=fmt.policy_identity())
            self.assertEqual(set(result['indices']), {12, 13, 14, 15})
            self.assertEqual(result['states'][5], 'E')
            self.assertEqual(result['states'][15], 'G')
            np.testing.assert_array_equal(predict.call_args.args[1], self.base)
            np.testing.assert_array_equal(predict.call_args.args[2], mixed)
            self.assertEqual(predict.call_args.args[3][5], 1.)
            self.assertFalse(result['sender_router_loaded'])
            self.assertFalse(result['unreceived_candidates_used'])

    def test_core_and_halo_models_roundtrip_source_free_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            inner = parse_inner(subset_bank(self.bank, []))
            for arm, halo in (('core', 0), ('halo', 64)):
                path = Path(tmp)/f'{arm}.pt'
                model_fixture(path, arm)
                c = config(4)
                c['receiver_router'] = file_hash(path)
                first = policy.route(self.base, self.base, inner, c, path,
                                     expected_policy=fmt.policy_identity())
                again = policy.route(self.base, self.base, inner, c, path,
                                     expected_policy=fmt.policy_identity())
                self.assertEqual(first, again)
                self.assertEqual(first['input_halo'], halo)
                self.assertEqual(np.shape(first['predictions']), (16, 3))

    def test_policy_authentication_geometry_and_invalid_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'Rg.pt'
            model = model_fixture(path)
            c = config(4)
            c['receiver_router'] = file_hash(path)
            inner = parse_inner(subset_bank(self.bank, []))
            for change in ({'receiver_router': '0'*64}, {'policy': '0'*64},
                           {'max_g': True}, {'boundary_lambda': -1}):
                with self.assertRaises(ValueError):
                    policy.route(self.base, self.base, inner, {**c, **change}, path,
                                 expected_policy=fmt.policy_identity())
            with self.assertRaisesRegex(ValueError, 'geometry'):
                policy.route(self.base[:2], self.base[:2], inner, c, model,
                             expected_policy=fmt.policy_identity())
            with patch.object(rg, 'predict', return_value=torch.full((1, 16, 3), float('nan'))):
                with self.assertRaisesRegex(ValueError, 'predictions'):
                    policy.route(self.base, self.base, inner, c, model,
                                 expected_policy=fmt.policy_identity())

    def fixture(self, root, *, indices=(), truncated=False, generated=False):
        base = np.zeros((17, 256, 384, 3), np.uint8) if generated else self.base
        bank = make_bank(base, q=2.)
        c = config(1 if generated else 0)
        wire = fmt.wrap(subset_bank(bank, list(indices)), c)
        stream = root/'stream.rvrc'
        atomic_bytes(stream, wire[:-1] if truncated else wire)
        output = root/'receiver'
        output.mkdir()
        args = SimpleNamespace(stream=stream, output=output, router=None,
            enhancement=root/'missing-E', adapter=root/'missing-G',
            disable_generation=not generated, allow_incomplete_tail=truncated)
        models = {key: '0'*64 for key in ('model_i_sha256', 'model_p_sha256')}
        codec = SimpleNamespace(models=models, decode=Mock(return_value=base.copy()))
        def decode_e(model, path, codec_arg, raw, **kwargs):
            parsed = parse_inner(raw, allow_incomplete_tail=truncated)
            enhanced = base.copy()
            for packet in parsed.packets:
                p = packet.meta
                x, y, w, h = p['roi']
                enhanced[p['start']:p['start']+p['count'], y:y+h, x:x+w] = 64
            report = dict(base_bytes=len(parsed.base), container_header_bytes=parsed.base_end-len(parsed.base),
                packet_bytes=sum(len(p.wire) for p in parsed.packets), incomplete_tail_bytes=parsed.incomplete_tail_bytes,
                base_hash=frame_hash(base), non_enhanced_exact=True, applied_packets=[])
            return enhanced, report, base.copy()
        stack = ExitStack()
        for name in ('reset_peak_memory_stats', 'empty_cache'):
            stack.enter_context(patch.object(receiver.torch.cuda, name))
        for name in ('max_memory_allocated', 'max_memory_reserved'):
            stack.enter_context(patch.object(receiver.torch.cuda, name, return_value=0))
        stack.enter_context(patch.object(receiver, 'configure_torch'))
        stack.enter_context(patch.object(receiver, 'BaseCodec', return_value=codec))
        load = stack.enter_context(patch.object(receiver, 'load_model', return_value=object()))
        stack.enter_context(patch.object(receiver, 'decode_enhancement', side_effect=decode_e))
        return args, base, c, stack, load

    def test_G_off_no_E_reads_no_router_generator_or_enhancement(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, base, c, stack, load = self.fixture(Path(tmp))
            with stack, patch.object(receiver.policy, 'route', side_effect=AssertionError('no route')), \
                    patch.object(receiver, 'identities', side_effect=AssertionError('no G assets')), \
                    patch.object(receiver, 'restore', side_effect=AssertionError('no G')):
                report = receiver.receive(args)
            load.assert_not_called()
            self.assertFalse(report['receiver_router_used'])
            self.assertFalse(report['sender_router_loaded'])
            self.assertFalse(report['generation_executed'])
            self.assertEqual(report['route']['states'], ['B']*16)
            receiver.validate_decoded(args.output, args.stream, c)
            with np.load(args.output/'reconstruction.npz') as saved:
                np.testing.assert_array_equal(saved['reconstruction'], base)

    def test_G_off_partial_E_preserves_coverage_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, base, c, stack, load = self.fixture(Path(tmp), indices=(5,), truncated=True)
            with stack, patch.object(receiver.policy, 'route', side_effect=AssertionError('no route')):
                report = receiver.receive(args)
            load.assert_called_once()
            self.assertEqual(report['route']['states'][5], 'E')
            self.assertAlmostEqual(report['route']['coverage'][5], 9/17, places=6)
            self.assertGreater(report['incomplete_tail_bytes'], 0)
            self.assertEqual(report['total_bytes'], args.stream.stat().st_size)
            receiver.validate_decoded(args.output, args.stream, c)

    def test_generation_receives_actual_Y_and_keeps_outside_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, base, c, stack, load = self.fixture(root, indices=(5,), generated=True)
            args.router = root/'Rg.pt'
            args.router.write_bytes(b'fixture-model')
            c['receiver_router'] = file_hash(args.router)
            atomic_bytes(args.stream, fmt.wrap(subset_bank(make_bank(base, q=2.), [5]), c))
            def route(b, y, inner, cfg, rg_path, **kwargs):
                self.assertTrue(np.any(b != y))
                self.assertEqual(kwargs['expected_policy'], fmt.policy_identity())
                return dict(indices=[5], rois=receiver.validate_generation_geometry(base.shape, 1))
            with stack, patch.object(policy, 'route', side_effect=route), \
                    patch.object(receiver, 'identities', return_value={k: c[k] for k in receiver.cooperation.HASHES}), \
                    patch.object(receiver, 'restore', side_effect=lambda y, *a: (y+32, {'windows': []})):
                report = receiver.receive(args)
            self.assertTrue(report['outside_generate_exact'])
            self.assertTrue(report['receiver_router_used'])
            self.assertFalse(report['sender_router_loaded'])
            self.assertTrue(report['generation_executed'])
            self.assertEqual(report['explicit_G_map_bytes'], 0)
            receiver.validate_decoded(args.output, args.stream, c)

    def test_all_reset_memory_segments_are_accounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            def fake_receive(args):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_peak_memory_stats()
                return dict(peak_cuda_allocated_bytes=30, generation_runtime={
                    'windows': [{'runtime': {'peak_cuda_allocated_bytes': 100}}]})
            with patch.object(receiver, '_receive', side_effect=fake_receive), \
                    patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                    patch.object(torch.cuda, 'max_memory_allocated', side_effect=[200, 100, 30]), \
                    patch.object(torch.cuda, 'max_memory_reserved', side_effect=[300, 150, 60]):
                report = receiver.receive(args)
            self.assertEqual(report['peak_cuda_allocated_bytes'], 200)
            self.assertEqual(report['peak_cuda_reserved_bytes'], 300)
            self.assertEqual(report['generation_all_calls_peak_cuda_bytes'], 100)

    def test_supervised_G_off_resume_is_readonly_and_checks_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, base, c, stack, load = self.fixture(root)
            args.output = root/'parent'
            args.output.mkdir()
            child = SimpleNamespace(**vars(args))
            child.output = args.output/'fresh'
            child.output.mkdir()
            with stack:
                receiver.receive(child)
            run = SimpleNamespace(check=lambda: None)
            with patch.object(receiver, 'execute', side_effect=AssertionError('no rerun')):
                result = receiver.decode(args, run)
                before = (args.output/'decode.complete.json').stat().st_mtime_ns
                self.assertEqual(receiver.decode(args, run), result)
                self.assertEqual((args.output/'decode.complete.json').stat().st_mtime_ns, before)
            self.assertEqual(result['binding']['used_assets'],
                             dict(receiver_router=None, adapter=None, enhancement=None))
            atomic_bytes(child.output/'reconstruction.npz', b'corrupt')
            with self.assertRaises(RuntimeError):
                receiver.decode(args, run)

    def test_worker_requires_tmux_supervisor_and_no_sender_flag(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'supervising tmux'):
                receiver.main(['--worker', '--stream', 'unused', '--output', 'unused', '--disable-generation'])
            with self.assertRaisesRegex(RuntimeError, 'tmux'):
                fmt.supervised(SimpleNamespace(max_hours=1), lambda *_: None)
        with self.assertRaises(SystemExit):
            receiver.main(['--stream', 'unused', '--output', 'unused', '--sender-router', 'forbidden'])


if __name__ == '__main__':
    unittest.main()
