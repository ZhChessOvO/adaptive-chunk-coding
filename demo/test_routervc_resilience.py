"""CPU-only framing and recovery tests; no models or CUDA execution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from demo import routervc_resilience as probe
from demo import routervc_format as fmt
from demo.routervc_encode import subset_bank
from demo.routervc_decode import coverage_from_packets
from demo.routervc_policy import grid_rois, select_generate
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.test_routervc_encode import make_bank


def example():
    pixels = np.zeros((17, 256, 384, 3), np.uint8)
    bank = make_bank(pixels)
    config = dict({name: '0'*64 for name in fmt.HASHES}, seed=42, max_g=4, boundary_lambda=.004,
                  strength=1., blend=1., window=17, stride=8, context=64, feather=16)
    return pixels, fmt.wrap(subset_bank(bank, [5, 0]), config)


class FakeRun:
    def __init__(self, root):
        self.root = root
        self.progress = {}

    def check(self):
        pass

    def update(self, **kw):
        self.progress.update(kw)


def prepared(root):
    base, wire = example()
    streams, damaged, details = probe.make_cases(wire)
    for name, value in dict(streams, crc_rejected_cpu=damaged).items():
        atomic_bytes(root/f'{name}.rtvc', value)
    protocol = dict(details=details, cases={name: dict(
        allow_incomplete_tail=name == 'truncated_next_packet',
        missing_all_optional_assets=name == 'base_no_models') for name in streams})
    return base, protocol


def fake_receiver(base, calls):
    def execute(run, name, script, argv, *, distributed):
        calls.append(name)
        assert script == 'routervc_decode.py' and distributed and '--input' not in argv
        stream = Path(argv[argv.index('--stream')+1])
        output = Path(argv[argv.index('--output')+1])
        config, _, inner, header = fmt.parse(stream.read_bytes(),
            allow_incomplete_tail='--allow-incomplete-tail' in argv)
        rois = grid_rois(base.shape[1], base.shape[2])
        coverage = coverage_from_packets(inner, rois).tolist()
        enhanced = base.copy()
        for packet in inner.packets:
            p = packet.meta
            x, y, w, h = p['roi']
            enhanced[p['start']:p['start']+p['count'], y:y+h, x:x+w] = 20
        if name == 'base_no_models':
            for option in ('--enhancement', '--adapter', '--router'):
                assert str(argv[argv.index(option)+1]).startswith('/missing/')
        route = dict(select_generate(np.zeros(16), 0, config['boundary_lambda']),
                     coverage=coverage, rois=rois, states=['E' if c else 'B' for c in coverage])
        report = dict(stream_sha256=file_hash(stream), total_bytes=stream.stat().st_size,
            config=config, base_bytes=len(inner.base), container_header_bytes=inner.base_end-len(inner.base),
            packet_bytes=sum(len(p.wire) for p in inner.packets), incomplete_tail_bytes=inner.incomplete_tail_bytes,
            generation_control_bytes=header, source_frames_read=False, base_reference_unchanged=True,
            outside_generate_exact=True, base_hash=frame_hash(base), output_hash=frame_hash(enhanced),
            generation_input_hash=frame_hash(enhanced), route=route, generation_executed=False,
            generation_assets_validated=False, shared_router_used=bool(config['max_g']), seconds=2.5)
        atomic_npz(output/'reconstruction.npz', base=base, enhanced=enhanced, reconstruction=enhanced)
        atomic_json(output/'decode.json', report)
    return execute


class ResilienceTests(unittest.TestCase):
    def test_complete_packet_and_incomplete_tail_are_distinct_bytes_same_content(self):
        _, wire = example()
        cases, _, details = probe.make_cases(wire)
        self.assertTrue(cases['truncated_next_packet'].startswith(cases['first_packet']))
        self.assertEqual(len(cases['truncated_next_packet'])-len(cases['first_packet']), details['ignored_tail_bytes'])
        self.assertEqual(details['partial_temporal_coverage'][5], float(np.float32(1/17)))
        self.assertEqual(np.count_nonzero(details['partial_temporal_coverage']), 1)
        _, _, first, _ = fmt.parse(cases['first_packet'])
        _, _, tail, _ = fmt.parse(cases['truncated_next_packet'], allow_incomplete_tail=True)
        self.assertEqual(first.packets, tail.packets)
        base_config, _, base, _ = fmt.parse(cases['base_no_models'])
        self.assertEqual(base_config['max_g'], 0)
        self.assertFalse(base.packets)
        self.assertEqual(base.base, first.base)

    def test_strict_truncation_and_any_complete_CRC_corruption_reject_on_CPU(self):
        _, wire = example()
        cases, damaged, details = probe.make_cases(wire)
        with patch.object(probe, 'execute', side_effect=AssertionError('CPU rejection must not execute')):
            self.assertIn('truncated', probe.reject(cases['truncated_next_packet']))
            self.assertIn('checksum', probe.reject(damaged))
            self.assertIn('checksum', probe.reject(damaged, allow_incomplete_tail=True))
        self.assertEqual(set(details['rejection']), {'strict_truncated', 'crc_strict', 'crc_permissive'})

    def test_resume_and_readonly_verify_do_not_repeat_fresh_receivers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, protocol = prepared(root)
            calls = []
            with patch.object(probe, 'execute', side_effect=fake_receiver(base, calls)):
                first = probe.run_checks(FakeRun(root), protocol)
                before = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in root.glob('*/*.json')}
                again = probe.run_checks(FakeRun(root), protocol, verify_only=True)
                resumed = probe.run_checks(FakeRun(root), protocol)
            self.assertEqual(calls, ['first_packet', 'truncated_next_packet', 'base_no_models'])
            self.assertEqual(first, again)
            self.assertEqual(first, resumed)
            for path, old in before.items():
                self.assertEqual((file_hash(Path(path)), Path(path).stat().st_mtime_ns), old)

    def test_interrupted_queue_preserves_completed_first_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, protocol = prepared(root)
            calls = []
            receiver = fake_receiver(base, calls)
            def interrupted(run, name, script, argv, *, distributed):
                if name == 'truncated_next_packet':
                    raise InterruptedError('test interruption')
                return receiver(run, name, script, argv, distributed=distributed)
            with patch.object(probe, 'execute', side_effect=interrupted), self.assertRaises(InterruptedError):
                probe.run_checks(FakeRun(root), protocol)
            before = (root/'first_packet/decode.json').read_bytes()
            with patch.object(probe, 'execute', side_effect=receiver):
                probe.run_checks(FakeRun(root), protocol)
            self.assertEqual(calls, ['first_packet', 'truncated_next_packet', 'base_no_models'])
            self.assertEqual(before, (root/'first_packet/decode.json').read_bytes())

    def test_verify_never_starts_missing_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, protocol = prepared(root)
            with patch.object(probe, 'execute') as execute, self.assertRaisesRegex(RuntimeError, 'missing completed'):
                probe.run_checks(FakeRun(root), protocol, verify_only=True)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
