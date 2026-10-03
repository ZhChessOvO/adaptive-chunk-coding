"""Pure-CPU checks for the independent full-frame comparison preview."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from demo import routervc_fullframe_preview as companion
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash


class FullframePreviewTests(unittest.TestCase):
    def records(self, root):
        rows = []
        source = np.zeros((17, 64, 96, 3), np.uint8)
        for domain in ('REDS', 'UVG'):
            sp = root/f'{domain}_source.npz'
            atomic_npz(sp, source=source)
            for method, cap in [('uf_qp32', 0), ('context_smooth', 4),
                                ('context_smooth', 8), ('full_frame_g', 16)]:
                folder = root/domain/f'{method}_{cap}'
                native = method == 'uf_qp32'
                name = 'stream.bin' if native else 'stream.acsg' if method == 'full_frame_g' else 'stream.rtvc'
                atomic_bytes(folder/name, b'wire'*20)
                total = 80
                artifacts = [name, 'fresh/decode.json', 'fresh/reconstruction.npz']
                d = dict(output_hash=frame_hash(source), source_frames_read=False,
                         stream_sha256=file_hash(folder/name), total_bytes=total)
                if native:
                    atomic_bytes(folder/'transmitted_meta.json', b'sidecar')
                    total += 7
                    d.update(native_bytes=80, metadata_bytes=7, total_bytes=total)
                    artifacts.append('transmitted_meta.json')
                if method == 'full_frame_g':
                    d['actual_G_roi_calls'] = 1
                atomic_npz(folder/'fresh/reconstruction.npz', reconstruction=source)
                atomic_json(folder/'fresh/decode.json', d)
                rows.append(dict(dataset=domain, sample_id=domain+'_first', method=method,
                    ratio=.5 if method == 'context_smooth' else 0., max_g=cap,
                    source_path=str(sp), source_hash=file_hash(sp), folder=str(folder),
                    bytes=total, native_bytes=80, quality={'lpips_alex': .1}, decode=d,
                    actual_G_roi_calls=1 if method == 'full_frame_g' else cap,
                    artifacts={n: file_hash(folder/n) for n in artifacts}))
        atomic_json(root/'summary.json', dict(complete=True, records=rows))
        atomic_json(root/'complete.json', dict(complete=True, summary=file_hash(root/'summary.json')))
        return rows

    def test_exact_baselines_and_native_caption_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.records(root)
            before = copy.deepcopy(rows)
            samples = companion.select_samples(rows)
            sample = samples[0]
            source, _ = companion.validate_records(root, sample)
            panels = companion.load_display_panels(root, sample, source)
            self.assertEqual([p['method'] for p in panels],
                             ['source', 'uf_qp32', 'context_smooth', 'context_smooth', 'full_frame_g'])
            self.assertEqual(panels[1]['displayed_bytes'], 80)
            self.assertEqual(panels[1]['original_record_total_bytes'], 87)
            self.assertIn('80 native UF B', panels[1]['subtitle'])
            self.assertIn('all-wire', panels[2]['subtitle'])
            self.assertEqual(before, rows)
            rows[3]['actual_G_roi_calls'] = 16
            with self.assertRaisesRegex(ValueError, 'one-ROI'):
                companion.select_samples(rows)

    def test_hash_checked_resume_and_native_stream_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.records(root)
            out = root/'preview_fullframe'
            def fake_encode(path, panels, label, fps, repeats, profile, run):
                self.assertEqual((fps, repeats), (12., 1))
                atomic_bytes(path, b'video')
                atomic_bytes(path.with_suffix('.png'), b'poster')
                return dict(path=path.name, poster=path.with_suffix('.png').name)
            with patch.object(companion.preview, 'ffmpeg_profile', return_value={'extension': '.mp4'}), \
                 patch.object(companion.preview, 'encode_preview', side_effect=fake_encode) as render:
                first = companion.create(root, out)
                stamp = (out/'manifest.json').stat().st_mtime_ns
                again = companion.create(root, out)
                self.assertEqual(first, again)
                self.assertEqual(stamp, (out/'manifest.json').stat().st_mtime_ns)
                self.assertEqual(render.call_count, 2)
                self.assertTrue(first['no_model_inference'])
                atomic_bytes(Path(rows[0]['folder'])/'stream.bin', b'changed')
                with self.assertRaisesRegex(RuntimeError, 'artifact'):
                    companion.create(root, out)

    def test_completed_marker_binds_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.records(root)
            atomic_json(root/'complete.json', dict(complete=True, summary='0'*64))
            with self.assertRaisesRegex(ValueError, 'summary identity'):
                companion.create(root, root/'preview')


if __name__ == '__main__':
    unittest.main()
