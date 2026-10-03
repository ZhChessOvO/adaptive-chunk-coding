"""CPU orchestration tests; fake payloads never execute CUDA or generation."""
from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from demo import routervc as cli
from demo import routervc_format as fmt
from demo.routervc_encode import subset_bank
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash, parse
from demo.test_routervc_encode import make_bank


class FakeRun:
    gpu_wait_seconds = .5

    def __init__(self, root):
        self.root = root
        self.progress = {}

    def update(self, **kw):
        self.progress.update(kw)

    def check(self):
        pass


def config(max_g=4, lam=.004):
    return dict({key: '0'*64 for key in fmt.HASHES}, seed=20261003, max_g=max_g,
        boundary_lambda=lam, strength=1., blend=1., window=17, stride=8, context=64, feather=16)


class CliTests(unittest.TestCase):
    def test_budget_parser_requires_exactly_one_choice(self):
        value = cli.parser().parse_args(['encode', '--input', 'source.npz', '--output', 'run', '--e-ratio', '.5'])
        self.assertEqual(value.mode, 'prefix')
        self.assertEqual(value.max_g, 4)
        cli._budgets(value)
        value.e_budget = 10
        with self.assertRaises(ValueError):
            cli._budgets(value)
        value.e_budget = None
        value.e_ratio = float('nan')
        with self.assertRaises(ValueError):
            cli._budgets(value)

    def test_G_geometry_is_explicit_and_early(self):
        self.assertEqual(len(cli.validate_generation_geometry((17, 512, 512, 3), 4)), 16)
        self.assertEqual(len(cli.validate_generation_geometry((33, 256, 384, 3), 8)), 16)
        for shape in ((17, 264, 520, 3), (17, 128, 128, 3)):
            with self.assertRaisesRegex(ValueError, 'No implicit resize'):
                cli.validate_generation_geometry(shape, 4)
            self.assertEqual(len(cli.validate_generation_geometry(shape, 0)), 16)

    def test_tmux_required_before_any_run(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(cli, 'Run') as run:
            with self.assertRaisesRegex(RuntimeError, 'tmux'):
                cli.main(['encode', '--input', 'source.npz', '--output', 'run', '--e-budget', '100'])
            run.assert_not_called()

    def test_encode_real_framing_resume_and_no_G(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'encoded'
            output.mkdir()
            prepared = output/'prepared'
            prepared.mkdir()
            base = np.zeros((17, 256, 384, 3), np.uint8)
            enhanced = np.full_like(base, 42)
            bank = make_bank(base)
            atomic_npz(root/'input.npz', source=base)
            atomic_bytes(prepared/'bank.acse', bank)
            atomic_npz(prepared/'candidates.npz', base=base, all_E=enhanced)
            prep = dict(all_E_rgb_sha256=frame_hash(enhanced), base_seconds=1.,
                        candidate_encode_seconds=2., candidate_decode_seconds=3.)
            atomic_json(prepared/'complete.json', prep)
            for name in ('E.pt', 'router.pt', 'adapter.pt'):
                atomic_bytes(root/name, name.encode())
            args = cli.parser().parse_args(['encode', '--input', str(root/'input.npz'),
                '--output', str(output), '--e-ratio', '.5', '--enhancement', str(root/'E.pt'),
                '--router', str(root/'router.pt'), '--adapter', str(root/'adapter.pt')])
            inner = subset_bank(bank, [0])
            mixed = base.copy()
            mixed[:, :64, :96] = enhanced[:, :64, :96]
            plan = dict(selected_indices=[0], e_packet_bytes=len(inner)-parse(bank).base_end,
                        mixed_rgb_sha256=frame_hash(mixed), prediction_seconds=.1, allocation_seconds=.2)
            with ExitStack() as stack:
                stack.enter_context(patch.object(cli.fmt, 'make_config', return_value=config()))
                prepare_call = stack.enter_context(patch.object(cli, 'prepare', return_value=prep))
                select_call = stack.enter_context(patch.object(cli, 'select', return_value=plan))
                route_call = stack.enter_context(patch.object(cli, 'route', return_value={'indices': [2]}))
                stack.enter_context(patch.object(cli, 'execute', side_effect=AssertionError('encoder must not generate')))
                first = cli.encode(args, FakeRun(output))
                self.assertEqual(first['actual_on_disk_bytes'], (output/'stream.rtvc').stat().st_size)
                self.assertEqual(sum(first['byte_breakdown'].values()), first['actual_on_disk_bytes'])
                self.assertEqual(first['expected_mixed_hash'], frame_hash(mixed))
                self.assertFalse(first['generation_executed_at_sender'])
                self.assertEqual(first['explicit_G_mask_bytes'], 0)
                before = (output/'encode.json').read_bytes()
                again = cli.encode(args, FakeRun(output))
                self.assertEqual(first, again)
                self.assertEqual(before, (output/'encode.json').read_bytes())
                self.assertEqual(prepare_call.call_count, 1)
                self.assertEqual(select_call.call_count, 1)
                self.assertEqual(route_call.call_count, 1)
                args.e_ratio = .75
                with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                    cli.encode(args, FakeRun(output))

    def test_encode_bad_geometry_never_prepares_or_loads_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_npz(root/'input.npz', source=np.zeros((17, 264, 520, 3), np.uint8))
            args = cli.parser().parse_args(['encode', '--input', str(root/'input.npz'),
                '--output', str(root), '--e-budget', '0'])
            with patch.object(cli, 'prepare') as prepare_call, patch.object(cli.fmt, 'make_config') as config_call:
                with self.assertRaisesRegex(ValueError, 'No implicit resize'):
                    cli.encode(args, FakeRun(root))
                prepare_call.assert_not_called()
                config_call.assert_not_called()

    def test_fresh_decode_resume_and_missing_unused_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'decoded'
            output.mkdir()
            base = np.zeros((17, 64, 96, 3), np.uint8)
            inner = subset_bank(make_bank(base), [])
            cfg = config(max_g=0)
            stream = root/'stream.rtvc'
            atomic_bytes(stream, fmt.wrap(inner, cfg))
            args = cli.parser().parse_args(['decode', '--stream', str(stream), '--output', str(output),
                '--enhancement', str(root/'missingE.pt'), '--router', str(root/'missingRouter.pt'),
                '--adapter', str(root/'missingG.pt')])
            def fake_execute(run, name, script, argv, distributed):
                self.assertTrue(distributed)
                self.assertEqual(script, 'routervc_decode.py')
                self.assertNotIn('--input', argv)
                destination = Path(argv[argv.index('--output')+1])
                atomic_npz(destination/'reconstruction.npz', reconstruction=base, base=base, enhanced=base)
                atomic_json(destination/'decode.json', dict(stream_sha256=file_hash(stream),
                    total_bytes=stream.stat().st_size, source_frames_read=False, base_reference_unchanged=True,
                    outside_generate_exact=True, config=cfg, output_hash=frame_hash(base),
                    base_hash=frame_hash(base), generation_input_hash=frame_hash(base)))
            with patch.object(cli, 'execute', side_effect=fake_execute) as execute:
                first = cli.decode(args, FakeRun(output))
                before = (output/'decode.complete.json').read_bytes()
                again = cli.decode(args, FakeRun(output))
                self.assertEqual(first, again)
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(first['binding']['used_assets'], dict(enhancement=None, router=None, adapter=None))
                self.assertEqual(before, (output/'decode.complete.json').read_bytes())
                atomic_bytes(output/'fresh/reconstruction.npz', b'corrupt')
                with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
                    cli.decode(args, FakeRun(output))


if __name__ == '__main__':
    unittest.main()
