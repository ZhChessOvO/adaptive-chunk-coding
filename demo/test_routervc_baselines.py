"""Independent CPU review checks for real-byte and fixed-route baselines."""
from contextlib import ExitStack
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from demo import routervc_baselines as baseline
from demo import routervc_format as rvc
from demo import scalable_cooperation_format as cooperation
from demo.routervc_policy import grid_rois
from demo.routervc_encode import subset_bank
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import canonical_json, frame_hash
from demo.test_routervc_encode import make_bank


class BaselineReviewTests(unittest.TestCase):
    def test_native_fresh_receiver_never_reads_source_and_charges_sidecar(self):
        pixels = np.zeros((17, 64, 64, 3), np.uint8)
        weights = dict(model_i_sha256='1'*64, model_p_sha256='2'*64)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for qp in (8, 32):
                target = folder/f'qp{qp}'
                target.mkdir()
                raw = b'opaque-native-compressed-bits'
                metadata = baseline.uf_metadata(qp, pixels.shape, raw, pixels, weights)
                atomic_bytes(target/'stream.bin', raw)
                atomic_bytes(target/'transmitted_meta.json', canonical_json(metadata)+b'\n')
                codec = Mock(models=weights)
                codec.decode.return_value = pixels
                with ExitStack() as stack:
                    stack.enter_context(patch('demo.scalable_codec.BaseCodec', return_value=codec))
                    stack.enter_context(patch('demo.chunk_enhancement_codec.configure_torch'))
                    stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
                    stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=0))
                    stack.enter_context(patch.object(baseline.np, 'load', side_effect=AssertionError('source/npz read forbidden')))
                    baseline.decode_uf(SimpleNamespace(output=target))
                codec.decode.assert_called_once_with(raw, 17)
                decoded = baseline.read(target/'fresh/decode.json')
                self.assertFalse(decoded['source_frames_read'])
                self.assertEqual(decoded['total_bytes'], (target/'stream.bin').stat().st_size+
                                 (target/'transmitted_meta.json').stat().st_size)
                self.assertEqual(decoded['metadata_bytes'], (target/'transmitted_meta.json').stat().st_size)
                self.assertEqual(decoded['output_hash'], frame_hash(pixels))
                self.assertEqual(baseline.read(target/'transmitted_meta.json')['qp'], qp)
                self.assertNotEqual(raw[:4], b'ACSE')
                job = dict(kind='uf', method=f'uf_qp{qp}', qp=qp, folder=str(target), sample_id='synthetic',
                           source_path='/missing/source.npz', source_hash='f'*64,
                           base_qp8_hash=frame_hash(pixels))
                names = ('stream.bin', 'transmitted_meta.json', 'fresh/decode.json',
                         'fresh/reconstruction.npz')
                record = dict(job=job, decode=decoded, bytes=decoded['total_bytes'],
                              artifacts={name: file_hash(target/name) for name in names})
                atomic_json(target/'result.json', record)
                before = {name: (file_hash(target/name), (target/name).stat().st_mtime_ns)
                          for name in (*names, 'result.json')}
                run = SimpleNamespace(check=lambda: None, update=lambda **kw: None)
                with patch('demo.conditioned_generation_pipeline.execute', side_effect=AssertionError('no reinference')), \
                     patch('demo.scalable_experiment.quality', side_effect=AssertionError('no metric recomputation')), \
                     patch('demo.stage_c_three_path_roi_probe.LPIPSAlex', side_effect=AssertionError('no metric load')):
                    replayed = baseline.evaluate(run, [job], verify_only=True)
                self.assertEqual(replayed, [record])
                after = {name: (file_hash(target/name), (target/name).stat().st_mtime_ns)
                         for name in (*names, 'result.json')}
                self.assertEqual(before, after)

    def test_all_G_cells_and_actual_ROI_calls_are_distinct(self):
        base = np.zeros((17, 256, 384, 3), np.uint8)
        inner = subset_bank(make_bank(base), [])
        rois = grid_rois(256, 384)
        cfg = dict({name: '0'*64 for name in rvc.HASHES}, seed=42, max_g=16,
                   boundary_lambda=0., strength=1., blend=1., window=17, stride=8, context=64, feather=16)
        with tempfile.TemporaryDirectory() as temp:
            for full_frame in (True, False):
                folder = Path(temp)/str(full_frame)
                folder.mkdir()
                control = rvc.generation_control(cfg, list(range(16)), rois, 17)
                if full_frame:
                    control['generate'] = [[0, 17, 0, 0, 384, 256]]
                atomic_bytes(folder/'stream.acsg', cooperation.wrap(inner, control))
                codec = Mock(models=dict(model_i_sha256='0'*64, model_p_sha256='0'*64))
                codec.decode.return_value = base
                calls = 1 if full_frame else 16
                with ExitStack() as stack:
                    stack.enter_context(patch('demo.scalable_codec.BaseCodec', return_value=codec))
                    stack.enter_context(patch('demo.chunk_enhancement_codec.configure_torch'))
                    stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
                    stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=0))
                    stack.enter_context(patch('torch.cuda.empty_cache'))
                    stack.enter_context(patch('demo.online_eg_decode.identities',
                                              return_value={k: '0'*64 for k in cooperation.HASHES}))
                    restore = stack.enter_context(patch('demo.internal_condition_decode.restore',
                                              return_value=(base+5, {'windows': list(range(calls))})))
                    stack.enter_context(patch.object(baseline.np, 'load', side_effect=AssertionError('source read forbidden')))
                    baseline.decode_g(SimpleNamespace(output=folder, adapter=Path('/missing/G.pt')))
                restore.assert_called_once()
                decoded = baseline.read(folder/'fresh/decode.json')
                self.assertEqual(decoded['route']['grid_G_cells'], 16)
                self.assertEqual(decoded['actual_G_roi_calls'], calls)
                self.assertEqual(decoded['actual_G_window_calls'], calls)
                self.assertEqual(decoded['total_bytes'], (folder/'stream.acsg').stat().st_size)
                self.assertFalse(decoded['source_frames_read'])
                self.assertEqual(decoded['packet_bytes'], 0)

    def test_fixed_G_controls_survive_removing_E_not_reallocation(self):
        base = np.zeros((17, 256, 384, 3), np.uint8)
        bank = make_bank(base)
        selected = subset_bank(bank, [2, 7])
        cfg = dict({name: '0'*64 for name in rvc.HASHES}, seed=42, max_g=4,
                   boundary_lambda=.004, strength=1., blend=1., window=17, stride=8, context=64, feather=16)
        wire = rvc.wrap(selected, cfg)
        config, inner, parsed, _ = rvc.parse(wire)
        fixed = [1, 2, 3]
        control = rvc.generation_control(config, fixed, grid_rois(256, 384), 17)
        without_E = cooperation.wrap(inner[:parsed.base_end], control)
        got_control, _, got_inner, control_bytes = cooperation.parse(without_E)
        self.assertEqual(got_control, control)
        self.assertEqual(got_inner.packets, ())
        self.assertEqual(got_inner.base, parsed.base)
        self.assertEqual(got_control['seed'], cfg['seed'])
        self.assertEqual(got_control['generate'], [[0, 17, *grid_rois(256, 384)[i]] for i in fixed])
        self.assertEqual(len(without_E), got_inner.base_end+control_bytes)
        self.assertGreater(control_bytes, 0)

    def test_G_off_uses_original_wire_and_ignores_missing_G_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            atomic_bytes(folder/'stream.rtvc', b'exact-original-routervc-wire')
            before = file_hash(folder/'stream.rtvc')
            def fake_receiver(args):
                self.assertTrue(args.disable_generation)
                self.assertEqual(args.stream, folder/'stream.rtvc')
                self.assertEqual(args.adapter, Path('/missing/generator.pt'))
                self.assertEqual(args.router, Path('/missing/router.pt'))
                atomic_json(args.output/'decode.json', dict(base_bytes=12, packet_bytes=34,
                            total_bytes=78, source_frames_read=False, generation_executed=False))
            with patch('demo.routervc_decode.decode', side_effect=fake_receiver) as decoder:
                baseline.decode_no_g(SimpleNamespace(output=folder))
            decoder.assert_called_once()
            self.assertEqual(before, file_hash(folder/'stream.rtvc'))
            report = baseline.read(folder/'fresh/decode.json')
            self.assertEqual(report['packet_bytes'], 34)
            self.assertEqual(report['actual_G_roi_calls'], 0)

    def test_paired_noise_allows_condition_change_but_rejects_noise_or_geometry_drift(self):
        reference = dict(generation_runtime=dict(
            windows=[dict(start=0, crop=[0, 0, 192, 192], runtime=dict(seed=42))],
            condition_windows=[dict(diffusion_noise={'sha256': 'a'*64},
                                    effective_condition={'sha256': 'e'*64})]))
        current = copy.deepcopy(reference)
        current['generation_runtime']['condition_windows'][0]['effective_condition']['sha256'] = 'b'*64
        self.assertEqual(baseline.paired_noise(reference, current),
                         dict(paired_windows=1, diffusion_noise_equal=True))
        changed = copy.deepcopy(current)
        changed['generation_runtime']['condition_windows'][0]['diffusion_noise']['sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'geometry/noise'):
            baseline.paired_noise(reference, changed)
        changed = copy.deepcopy(current)
        changed['generation_runtime']['windows'][0]['crop'][2] = 256
        with self.assertRaisesRegex(ValueError, 'geometry/noise'):
            baseline.paired_noise(reference, changed)
        with self.assertRaisesRegex(ValueError, 'unpaired skipped'):
            baseline.paired_noise(reference, dict(generation_runtime=None))
        self.assertEqual(baseline.paired_noise(dict(generation_runtime=None), dict(generation_runtime=None)),
                         dict(paired_windows=0, diffusion_noise_equal=True))


if __name__ == '__main__':
    unittest.main()
