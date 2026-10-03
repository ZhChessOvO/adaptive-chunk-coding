"""CPU-only accounting, support, interpolation and no-recompute replay checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from demo.routervc_analysis import UF_METHODS, auxiliary_interpolation
from demo.routervc_native_uf_report import (analyze, build_curves, curve_points,
    verify_native_point)
from demo.routervc_report import atomic_json, file_hash


def record(sample='r1', dataset='REDS', method='context_smooth', bpp=.1, byte_count=100, cap=4):
    return dict(sample_id=sample, dataset=dataset, method=method, ratio=.5,
                max_g=0 if method in UF_METHODS else cap, bytes=byte_count, bpp=bpp,
                quality=dict(lpips_alex=.2, psnr_db=30., temporal_delta_mae=2.))


def saved_point(root):
    folder = root/'point'; (folder/'fresh').mkdir(parents=True)
    (folder/'stream.bin').write_bytes(b'actual native test stream')
    stream_hash = file_hash(folder/'stream.bin')
    shape = (17, 16, 32, 3)
    pixels = np.zeros(shape, np.uint8)
    np.savez_compressed(folder/'fresh/reconstruction.npz', base=pixels, enhanced=pixels, reconstruction=pixels)
    meta = dict(format='RouterVC_native_UF_sidecar_v1', qp=8, frame_count=17, height=16, width=32,
                stream_sha256=stream_hash, base_rgb_sha256='a'*64)
    atomic_json(folder/'transmitted_meta.json', meta)
    native = (folder/'stream.bin').stat().st_size
    sidecar = (folder/'transmitted_meta.json').stat().st_size
    artifacts = {name:file_hash(folder/name) for name in ('stream.bin', 'transmitted_meta.json')}
    atomic_json(folder/'encode.json', dict(complete=True, qp=8, native_bytes=native,
        metadata_bytes=sidecar, total_bytes=native+sidecar, artifacts=artifacts))
    d = dict(native_bytes=native, metadata_bytes=sidecar, total_bytes=native+sidecar,
        stream_sha256=stream_hash, metadata_sha256=artifacts['transmitted_meta.json'], base_hash='a'*64)
    raw = dict(sample_id='r1', dataset='REDS', method='uf_qp8', kind='uf', qp=8, folder=str(folder),
        native_bytes=native, non_native_bytes=sidecar, bytes=native+sidecar, artifacts=artifacts, decode=d)
    norm = dict(record(method='uf_qp8', byte_count=native+sidecar), native_bytes=native,
                shape=list(shape), bpp=8*(native+sidecar)/(17*16*32))
    return raw, norm


class NativeUFTests(unittest.TestCase):
    def test_native_uses_actual_stat_hash_and_individual_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw, norm = saved_point(root)
            original = copy.deepcopy(norm)
            converted, evidence = verify_native_point(raw, norm, root, {})
            self.assertEqual(converted['bytes'], len(b'actual native test stream'))
            self.assertEqual(converted['bpp'], 8*len(b'actual native test stream')/(17*16*32))
            self.assertEqual(converted['quality'], norm['quality'])
            self.assertEqual(norm, original)
            self.assertEqual(evidence['native_bytes']+evidence['sidecar_bytes'], norm['bytes'])
            stream = Path(raw['folder'])/'stream.bin'
            stream.write_bytes(b'x'*stream.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'changed saved evidence'):
                verify_native_point(raw, norm, root, {})

    def test_recorded_native_byte_or_geometry_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw, norm = saved_point(root)
            with self.assertRaisesRegex(ValueError, 'native UF file size'):
                verify_native_point(dict(raw, native_bytes=999), norm, root, {})
            with self.assertRaisesRegex(ValueError, 'decoded geometry'):
                verify_native_point(raw, dict(norm, shape=[17, 32, 16, 3]), root, {})

    def test_matched_smoke_excludes_unmatched_UVG_from_all_curves(self):
        original = [record(method=m, bpp=.2) for m in UF_METHODS]
        original += [record(), record('u1', 'UVG')]
        native = [dict(r, bpp=.1) if r['method'] in UF_METHODS else r for r in original]
        groups, support = build_curves(original, native, {'4':[['REDS', 'r1']]})
        self.assertEqual(support, {'4':[['REDS', 'r1']]})
        self.assertEqual([g['dataset'] for g in groups], ['REDS', 'All'])
        for group in groups:
            self.assertTrue(all(p['windows'] == 1 and p['samples'] == [['REDS', 'r1']] for p in group['points']))
            self.assertEqual([p['bpp'] for p in group['uf_charged_sidecar_diagnostic']], [.2]*4)
            router = next(p for p in group['points'] if p['method'] == 'context_smooth')
            self.assertEqual(router['bpp'], .1)
        with self.assertRaisesRegex(ValueError, 'support differs'):
            build_curves(original, native, {'4':[['REDS', 'r1'], ['UVG', 'u1']]})

    def test_bpp_is_mean_of_per_video_bpp_not_mean_byte_ratio(self):
        rows = [record('r1', method='uf_qp8', bpp=8*100/1000, byte_count=100),
                record('r2', method='uf_qp8', bpp=8*200/10000, byte_count=200)]
        point = curve_points(rows, [('REDS', 'r1'), ('REDS', 'r2')], 4)[0]
        self.assertAlmostEqual(point['bpp'], .48)
        self.assertNotAlmostEqual(point['bpp'], 8*300/11000)

    def test_native_interpolation_does_not_use_shifted_sidecar_or_extrapolate(self):
        rows = [record(method='uf_qp8', bpp=.01), record(method='uf_qp16', bpp=.03), record(bpp=.02)]
        out = auxiliary_interpolation(rows, [('REDS', 'r1')])['records'][0]
        self.assertEqual(out['uf']['status'], 'interpolated_inside_adjacent_interval')
        rows[-1]['bpp'] = .04
        out = auxiliary_interpolation(rows, [('REDS', 'r1')])['records'][0]
        self.assertEqual(out['uf']['status'], 'outside_measured_range')
        self.assertNotIn('router_minus_uf_estimate', out)

    def test_completed_resume_keeps_summary_figures_and_never_reloads_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'supplement'; workflow = Path(directory)/'workflow'
            out = root/'native_uf_report'
            (root/'analysis').mkdir(parents=True); workflow.mkdir(); out.mkdir()
            for path in (root/'summary.json', root/'analysis/summary.json', workflow/'complete.json'):
                atomic_json(path, dict(complete=True))
            image = out/'test.png'; image.write_bytes(b'unit-test-placeholder-not-real-result')
            config = dict(code={'test':'hash'})
            deps = dict(config=config, source_summary=file_hash(root/'summary.json'),
                source_analysis=file_hash(root/'analysis/summary.json'), source_workflow=file_hash(workflow/'complete.json'))
            summary = dict(complete=True, dependencies=deps, input_hashes={str(root/'summary.json'):deps['source_summary']},
                          artifacts={'test.png':file_hash(image)})
            atomic_json(out/'summary.json', summary)
            before = {p:(file_hash(p), p.stat().st_mtime_ns) for p in (image, out/'summary.json')}
            with patch('demo.routervc_native_uf_report.code_hashes', return_value=config['code']), \
                 patch('demo.routervc_native_uf_report.load_records', side_effect=AssertionError('no reload')), \
                 patch('demo.routervc_native_uf_report.draw_curves', side_effect=AssertionError('no redraw')):
                self.assertEqual(analyze(root, workflow, out, config), summary)
                self.assertEqual(analyze(root, workflow, out, config), summary)
            self.assertEqual(before, {p:(file_hash(p), p.stat().st_mtime_ns) for p in before})
            self.assertEqual(json.loads((out/'complete.json').read_text())['summary_sha256'], file_hash(out/'summary.json'))


if __name__ == '__main__':
    unittest.main()
