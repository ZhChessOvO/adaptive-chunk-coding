"""CPU display-only selection, native-pixel layout and resume checks."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from demo import routervc_preview as preview
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash


class PreviewTests(unittest.TestCase):
    def records(self, root):
        rows = []
        for domain in ('REDS', 'UVG'):
            source = np.full((2, 64, 96, 3), 120, np.uint8)
            sp = root/f'{domain}_source.npz'
            atomic_npz(sp, source=source)
            for name, ratio, max_g in [('base', 0., 0), ('e_only', .5, 0),
                                       ('g_only', 0., 4), ('context_smooth', .5, 4)]:
                folder = root/domain/name
                atomic_npz(folder/'fresh/reconstruction.npz', reconstruction=source)
                rows.append(dict(dataset=domain, sample_id=domain+'_first', method=name,
                    ratio=ratio, max_g=max_g, source_path=str(sp), source_hash=file_hash(sp),
                    folder=str(folder), bytes=123, quality={'lpips_alex': .1},
                    decode={'output_hash': frame_hash(source)},
                    artifacts={'fresh/reconstruction.npz': file_hash(folder/'fresh/reconstruction.npz')}))
        return rows

    def test_explicit_profile_does_not_silently_substitute_missing_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.records(Path(tmp))
            selected = preview.select_samples(rows, 'smoke')
            self.assertEqual([v['sample_id'] for v in selected], ['REDS_first', 'UVG_first'])
            self.assertEqual(len(selected[0]['panels']), 4)
            with self.assertRaisesRegex(ValueError, 'uf_qp32'):
                preview.select_samples(rows, 'formal')

    def test_render_keeps_native_panel_pixels(self):
        pixels = np.full((2, 256, 384, 3), [10, 100, 200], np.uint8)
        panels = [dict(title='Panel', subtitle='Existing pixels', pixels=pixels) for _ in range(5)]
        result = preview.render_frame(panels, 1, 'test', 12, 3)
        for i in range(5):
            x = i % 3*384
            y = preview.PAGE_TOP+i//3*(256+preview.LABEL_HEIGHT)+preview.LABEL_HEIGHT
            np.testing.assert_array_equal(result[y:y+256, x:x+384], pixels[1])

    def test_verified_resume_never_renders_again_or_changes_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.records(root)
            atomic_json(root/'summary.json', dict(complete=True, records=rows))
            original = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in root.rglob('*.npz')}
            def fake_encode(path, panels, label, fps, repeats, profile, run):
                atomic_bytes(path, b'display-video')
                atomic_bytes(path.with_suffix('.png'), b'display-poster')
                return dict(path=path.name, poster=path.with_suffix('.png').name)
            with patch.object(preview, 'ffmpeg_profile', return_value={'extension': '.mp4'}), \
                 patch.object(preview, 'encode_preview', side_effect=fake_encode) as encode:
                first = preview.create(root, profile='smoke')
                before = (root/'preview/manifest.json').read_bytes()
                again = preview.create(root, profile='smoke')
                self.assertEqual(first, again)
                self.assertEqual(before, (root/'preview/manifest.json').read_bytes())
                self.assertEqual(encode.call_count, 2)
                self.assertTrue(first['no_model_inference'])
                self.assertTrue(first['no_metric_recalculation'])
                self.assertEqual(original, {str(p): (file_hash(p), p.stat().st_mtime_ns)
                                          for p in root.rglob('*.npz')})
                atomic_bytes(Path(rows[0]['source_path']), b'changed source')
                with self.assertRaisesRegex(RuntimeError, 'evaluation cache'):
                    preview.create(root, profile='smoke')


if __name__ == '__main__':
    unittest.main()
