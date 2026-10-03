"""Tiny saved-file fixtures; no dataset, decoder, metric model or GPU."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from demo import routervc_fullview_probe_report as report


class SavedProbeFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'current'
        self.old = Path(temporary.name) / 'previous'
        self.root.mkdir()
        self.old.mkdir()
        report.save(self.old / 'protocol.json', {'old': True})
        (self.old / 'runner_before_baseline_fix.py').write_text('# old runner\n')
        self.protocol = dict(
            whole_view_provenance_verified=True, fixed_visual_frame=8,
            source_preparation={'record': {'binding': {'sample': {
                'sample_id': 'reds-val-000-f000-n17-fullview', 'whole_frame': True,
                'history': 'reused development sample', 'transform': {
                    'original_size': [1280, 720], 'coded_size': [1024, 576],
                    'padding': [0, 0, 0, 0]}}}}})
        report.save(self.root / 'protocol.json', self.protocol)
        self.protocol_hash = report.digest(self.root / 'protocol.json')
        report.save(self.root / 'import.json', dict(
            previous=str(self.old), previous_protocol=report.digest(self.old / 'protocol.json'),
            current_protocol=self.protocol_hash,
            previous_runner=report.digest(self.old / 'runner_before_baseline_fix.py'),
            purpose='recover completed points after ACSE1/ACSE2 full-G envelope fix'))
        self.records = []
        for index, name in enumerate(report.POINTS):
            folder = self.root / name
            (folder / 'fresh').mkdir(parents=True)
            kind = ('uf' if name.startswith('uf_') else 'full_g' if name == 'wholeframe_g_one_roi'
                    else 'off' if name.endswith('_off') else 'router')
            point = dict(name=name, kind=kind)
            if kind == 'uf':
                point['qp'] = int(name.removeprefix('uf_qp'))
                payload = b'u' * (point['qp'] + 10)
            elif kind == 'full_g':
                payload = b'f' * 50
            else:
                point.update(arm='context' if name.startswith('context') else 'local',
                             ratio=.25 if '0.25' in name else .5)
                payload = point['arm'].encode() * (2 if point['ratio'] == .25 else 3)
            stream = 'stream.bin' if kind == 'uf' else 'stream.acsg' if kind == 'full_g' else 'stream.rtvc'
            (folder / stream).write_bytes(payload)
            (folder / 'fixed_frame.png').write_bytes(b'fixture image bytes')
            (folder / 'fresh/reconstruction.npz').write_bytes(b'fixture reconstruction bytes')
            decoded = dict(source_frames_read=False, total_bytes=len(payload), seconds=12.5,
                           peak_cuda_allocated_bytes=128)
            ledger = dict(bytes=len(payload), native_bytes=18, audit_sidecar_bytes=0)
            if kind == 'uf':
                (folder / 'transmitted_meta.json').write_bytes(b'{}\n')
                decoded.update(native_bytes=len(payload), metadata_bytes=3, total_bytes=len(payload)+3)
                ledger.update(native_bytes=len(payload), audit_sidecar_bytes=3)
            elif kind in ('full_g', 'router'):
                count = 1 if kind == 'full_g' else 8
                decoded['generation_runtime'] = dict(model_load_seconds=2., windows=[
                    dict(region=i, runtime={'seconds_model_load_excluded': .5}) for i in range(count)])
                if kind == 'full_g':
                    decoded.update(actual_G_roi_calls=1, actual_G_window_calls=1,
                                   geometry='one_full_frame_ROI')
                else:
                    decoded.update(codec_seconds=2.1, policy_seconds=.4)
            report.save(folder / 'fresh/decode.json', decoded)
            names = [stream, 'fixed_frame.png', 'fresh/reconstruction.npz', 'fresh/decode.json']
            if kind == 'uf':
                names.append('transmitted_meta.json')
            record = dict(complete=True, point=point, protocol_sha256=self.protocol_hash,
                          bytes=len(payload), bpp=len(payload)*8/(17*576*1024),
                          quality=dict(lpips_alex=.7-index*.03, psnr_db=21+index*.2,
                                       temporal_delta_mae=3.), decode=decoded,
                          decode_seconds=12.5, elapsed_seconds_this_completion_attempt=20.,
                          byte_ledger=ledger,
                          artifacts={n: report.digest(folder/n) for n in names})
            if kind != 'full_g':
                old_path = self.old / name / 'result.json'
                report.save(old_path, dict(record, protocol_sha256='old'))
                record['reused_from'] = dict(path=str(old_path), sha256=report.digest(old_path),
                                           fresh_decode_and_metric_times_preserved=True)
            self.records.append(record)
        from PIL import Image
        Image.new('RGB', (64, 48), (32, 64, 128)).save(self.root / 'fixed_frame_comparison.png')
        self.refresh()

    def refresh(self):
        for record in self.records:
            report.save(self.root / record['point']['name'] / 'result.json', record)
        summary = dict(complete=True, points=10, records=self.records,
                       protocol_sha256=self.protocol_hash, no_model_promotion=True,
                       native_uf_sidecar_excluded_from_rate=True, shape=[17, 576, 1024, 3],
                       artifact_hashes={r['point']['name']+'/result.json': report.digest(
                           self.root / r['point']['name'] / 'result.json') for r in self.records},
                       fixed_frame_sha256=report.digest(self.root / 'fixed_frame_comparison.png'))
        report.save(self.root / 'summary.json', summary)
        report.save(self.root / 'complete.json', dict(complete=True, protocol_sha256=self.protocol_hash,
                    summary_sha256=report.digest(self.root / 'summary.json')))

    def test_native_rate_and_reuse_are_verified_not_recomputed(self):
        result = report.collect(self.root)
        self.assertEqual(len(result['rows']), 10)
        self.assertEqual(result['recovery']['reused_points'], 9)
        self.assertEqual(result['recovery']['newly_completed_points'], 1)
        self.assertFalse(result['metrics_recomputed'])
        self.assertFalse(result['gpu_used'])
        self.assertFalse(result['semantic_fidelity_measured'])
        uf = next(r for r in result['rows'] if r['point'] == 'uf_qp8')
        self.assertEqual(uf['stream_bytes'], 18)  # Not the helper's 21 bytes.
        full = result['rows'][-1]
        self.assertEqual(full['actual_g_roi_calls'], 1)
        self.assertEqual(full['recorded_g_window_seconds'], .5)
        self.assertIsNone(full['recorded_policy_seconds'])  # Unknown is not zero.

    def test_completed_reentry_preserves_output_and_original_mtimes(self):
        before = {str(p): (report.digest(p), p.stat().st_mtime_ns)
                  for folder in (self.root, self.old) for p in folder.rglob('*') if p.is_file()}
        done = report.generate(self.root)
        output = self.root / 'report'
        saved = {str(p): (report.digest(p), p.stat().st_mtime_ns) for p in output.iterdir()}
        with patch.object(report, 'plot_rd', side_effect=AssertionError('must not redraw')):
            self.assertEqual(report.generate(self.root), done)
            self.assertEqual(report.generate(self.root, verify_only=True), done)
        for path, value in {**before, **saved}.items():
            self.assertEqual((report.digest(path), Path(path).stat().st_mtime_ns), value)
        with (output / 'timings.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 10)
        self.assertEqual((output / 'rd.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        images = report.read(output / 'visuals.json')['images']
        self.assertEqual(len(images), 11)
        self.assertTrue(all(v['relative_link'].startswith('../') for v in images))
        preview = report.read(output / 'visuals.json')['display_export']
        self.assertLess(preview['bytes'], 5 * 1024**2)
        self.assertFalse(preview['metrics_input'])
        self.assertFalse(preview['resized'])
        self.assertEqual(preview['source_size'], preview['display_size'])

    def test_recovered_metric_change_rejected(self):
        self.records[0]['quality']['lpips_alex'] += .1
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'changed quality'):
            report.collect(self.root)

    def test_changed_bytes_rejected_even_if_new_artifact_hash_is_rebound(self):
        record = self.records[-1]
        stream = self.root / record['point']['name'] / 'stream.acsg'
        stream.write_bytes(stream.read_bytes() + b'x')
        record['artifacts']['stream.acsg'] = report.digest(stream)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'real stream file size'):
            report.collect(self.root)

    def test_completed_report_tamper_rejected(self):
        with patch.object(report, 'plot_rd', side_effect=lambda rows, path: path.write_bytes(b'plot fixture')):
            report.generate(self.root)
        (self.root / 'report/timings.csv').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'report artifact changed'):
            report.generate(self.root)

    def test_changed_input_binding_rejected(self):
        with patch.object(report, 'plot_rd', side_effect=lambda rows, path: path.write_bytes(b'plot fixture')):
            report.generate(self.root)
        receipt = report.read(self.root / 'import.json')
        receipt['purpose'] += ' (changed)'
        report.save(self.root / 'import.json', receipt)
        with self.assertRaisesRegex(ValueError, 'inputs/code changed'):
            report.generate(self.root)

    def test_partial_run_resumes_but_verify_cannot_generate(self):
        with self.assertRaisesRegex(ValueError, 'not complete'):
            report.generate(self.root, verify_only=True)
        with patch.object(report, 'plot_rd', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                report.generate(self.root)
        binding_mtime = (self.root / 'report/binding.json').stat().st_mtime_ns
        with patch.object(report, 'plot_rd', side_effect=lambda rows, path: path.write_bytes(b'plot fixture')):
            self.assertTrue(report.generate(self.root)['complete'])
        self.assertEqual(binding_mtime, (self.root / 'report/binding.json').stat().st_mtime_ns)

    def test_cannot_write_over_existing_probe_root(self):
        with self.assertRaisesRegex(ValueError, 'new probe/report'):
            report.generate(self.root, self.root)

    def test_module_import_is_torch_free(self):
        subprocess.run([sys.executable, '-c', 'import sys; '
                        'import demo.routervc_fullview_probe_report; assert "torch" not in sys.modules'],
                       check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
