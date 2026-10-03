"""CPU-only whole-frame preparation, provenance and no-crop regressions."""
from pathlib import Path
import hashlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from demo import routervc_fullview_data as data


class FullViewTests(unittest.TestCase):
    def test_full_16_9_fit_is_exact_and_unpadded(self):
        for size in ((1280, 720), (1920, 1080)):
            spec = data.geometry(size)
            self.assertEqual(spec['coded_size'], [1024, 576])
            self.assertEqual(spec['valid_rect'], [0, 0, 1024, 576])
            self.assertEqual(spec['padding'], [0, 0, 0, 0])
            self.assertIsNone(spec['crop'])
            self.assertTrue(spec['aspect_ratio_exact'])

    def test_native_padding_does_not_resize_or_crop(self):
        reds = data.geometry((1280, 720), mode='native')
        self.assertEqual(reds['resized_size'], [1280, 720])
        self.assertEqual(reds['coded_size'], [1280, 768])
        self.assertEqual(reds['valid_rect'], [0, 24, 1280, 720])
        uvg = data.geometry((1920, 1080), mode='native')
        self.assertEqual(uvg['coded_size'], [1920, 1088])
        self.assertEqual(uvg['valid_rect'], [0, 4, 1920, 1080])

    def test_non_16_9_fit_preserves_borders_and_marks_pad(self):
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[0] = [255, 0, 0]
        pixels[-1] = [0, 255, 0]
        pixels[:, 0] = [0, 0, 255]
        spec = data.geometry((64, 48), width=64, height=64)
        out = np.asarray(data.transform_frame(Image.fromarray(pixels), spec))
        np.testing.assert_array_equal(out[8:56], pixels)
        self.assertEqual(spec['valid_rect'], [0, 8, 64, 48])
        np.testing.assert_array_equal(out[0], pixels[0])
        np.testing.assert_array_equal(out[-1], pixels[-1])

    def test_geometry_validation(self):
        for kw in (dict(width=1000), dict(alignment=0), dict(mode='crop'), dict(width=-1)):
            with self.assertRaises(ValueError):
                data.geometry((1280, 720), **kw)
        with self.assertRaises(ValueError):
            data.geometry((101, 103), width=64, height=64)

    def test_missing_uvg_never_promotes_old_crops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root/'assets/evaluation/UVG/HoneyBee'
            old.mkdir(parents=True)
            Image.new('RGB', (512, 512)).save(old/'00000000.png')
            rows = data.uvg_inventory(root)
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(not r['full_original_available'] for r in rows))
            self.assertTrue(all(r['status'] == 'missing_full_original' for r in rows))
            self.assertEqual(rows[4]['official_archive'], 'ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z')
            self.assertEqual([r['sequence'] for r in rows if r['role'] == 'evaluation'], ['ReadySetGo', 'YachtRide'])

    def test_archive_exists_does_not_mean_full_original_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'DCVC'
            root.mkdir()
            name = data.uvg_inventory(root)[0]['official_archive']
            (root.parent/name).write_bytes(b'not an extracted archive')
            row = data.uvg_inventory(root)[0]
            self.assertEqual(row['status'], 'archive_needs_full_extraction')
            self.assertFalse(row['full_original_available'])
            self.assertEqual(row['valid_size_archive_paths'], [])

    def test_deterministic_manifest_with_roles_and_original_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'data/REDS/val_sharp/000'
            folder.mkdir(parents=True)
            # Small highly-compressible full-size PNGs are sufficient for CPU tests.
            for i in range(100):
                Image.new('RGB', (1280, 720), (i, 0, 0)).save(folder/f'{i:08d}.png')
            first = data.build_manifest(root, splits=('val',))
            self.assertEqual(first, data.build_manifest(root, splits=('val',)))
            self.assertEqual(len(first['samples']), 2)
            self.assertEqual([s['original_frame_start'] for s in first['samples']], [0, 40])
            self.assertEqual(first['samples'][1]['source']['files'][0], str(folder/'00000040.png'))
            self.assertTrue(all(s['role'] == 'development' for s in first['samples']))
            self.assertEqual(first['counts'], {'REDS/development': 2})
            self.assertEqual(sum(s['dataset'] == 'UVG' for s in first['missing_sources']), 7)
            self.assertEqual(data.roles('REDS', 'train', '000')[0], 'train')
            self.assertEqual(data.roles('REDS', 'val', '029')[0], 'evaluation')

    def test_rejects_cropped_reds_as_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'data/REDS/val_sharp/000'
            folder.mkdir(parents=True)
            Image.new('RGB', (512, 512)).save(folder/'00000000.png')
            with self.assertRaisesRegex(ValueError, 'not original full REDS'):
                data.build_manifest(root, splits=('val',))

    def make_sample(self, root):
        source = root/'originals'
        source.mkdir()
        for i in range(17):
            Image.new('RGB', (64, 48), (i, 20, 200)).save(source/f'{i:08d}.png')
        return data.sample_record('REDS', 'val', '000', data.png_files(source), 0, 17,
                                  data.geometry((64, 48), width=64, height=64))

    def test_prepare_exact_valid_pixels_and_resume_no_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = self.make_sample(root)
            first = data.prepare_sample(sample, root/'prepared')
            self.assertEqual(first['valid_video_pixels'], 17*64*48)
            self.assertEqual(first['frame_count'], 17)
            self.assertEqual(len(first['artifacts']), 18)
            with Image.open(root/'prepared/frames/00000000.png') as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.getpixel((0, 0)), (0, 20, 200))
            with patch.object(data, 'transform_frame', side_effect=AssertionError('no recompute')):
                self.assertEqual(first, data.prepare_sample(sample, root/'prepared'))

    def test_interrupted_sample_resumes_and_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = self.make_sample(root)
            calls = 0
            def stop():
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise InterruptedError('test interruption')
            with self.assertRaises(InterruptedError):
                data.prepare_sample(sample, root/'prepared', stop)
            self.assertEqual(len(data.png_files(root/'prepared/frames')), 3)
            self.assertFalse((root/'prepared/complete.json').exists())
            result = data.prepare_sample(sample, root/'prepared')
            self.assertTrue(result['complete'])
            Image.new('RGB', (64, 48), (99, 99, 99)).save(sample['source']['files'][0])
            with self.assertRaises(ValueError):
                data.prepare_sample(sample, root/'prepared')

    def test_changed_output_detected_and_never_repaired_silently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = self.make_sample(root)
            data.prepare_sample(sample, root/'prepared')
            (root/'prepared/frames/00000000.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                data.prepare_sample(sample, root/'prepared')

    def test_timeline_validation_and_prepare_tmux(self):
        with tempfile.TemporaryDirectory() as temporary:
            for kw in (dict(starts=(0, 0)), dict(starts=(-1,)), dict(count=16), dict(limit_sequences=-1)):
                with self.assertRaises(ValueError):
                    data.build_manifest(Path(temporary), **kw)
            with patch.dict('os.environ', {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'tmux'):
                    data.main(['prepare', '--manifest', 'unused.json', '--output', temporary])

    def test_raw_selected_window_hash_and_conversion(self):
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg is optional until restored raw UVG is prepared')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root/'tiny.yuv'
            frames = [bytes([16+i])*(64*48)+bytes([128])*(64*48//2) for i in range(18)]
            raw.write_bytes(b''.join(frames))
            sample = data.sample_record('UVG', 'heldout', 'YachtRide', [], 1, 17,
                data.geometry((64, 48), width=64, height=64), raw=raw, frame_count=18)
            hashes = data.source_hash(sample)
            self.assertEqual(hashes['raw_window_sha256'], hashlib.sha256(b''.join(frames[1:])).hexdigest())
            images = list(data.source_images(sample))
            self.assertEqual(len(images), 17)
            self.assertEqual(images[0].size, (64, 48))
            self.assertLess(images[0].getpixel((0, 0))[0], images[-1].getpixel((0, 0))[0])

    def test_invalid_sample_manifest_cannot_hide_crop_or_escape_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = self.make_sample(root)
            sample['sample_id'] = '../outside'
            with self.assertRaisesRegex(ValueError, 'sample ID'):
                data.prepare_sample(sample, root/'prepared')
            sample['sample_id'] = 'safe'
            sample['original_crop'] = [0, 0, 32, 32]
            with self.assertRaisesRegex(ValueError, 'uncropped'):
                data.prepare_sample(sample, root/'prepared')

    def test_prepare_manifest_records_only_selected_samples_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = self.make_sample(root)
            manifest = root/'manifest.json'
            data.atomic_json(manifest, dict(format=data.FORMAT, samples=[sample],
                                           missing_sources=[{'dataset': 'UVG', 'reason': 'missing'}]))
            # Do not inspect the actual host disks or GPU in a unit test.
            fake_usage = shutil._ntuple_diskusage(100, 10, 90)
            with patch.object(data.shutil, 'disk_usage', return_value=fake_usage), \
                 patch.object(data.subprocess, 'check_output', return_value='no active GPU work'):
                first = data.prepare_manifest(manifest, root/'out', selected_roles=['development'])
                with patch.object(data, 'transform_frame', side_effect=AssertionError('recompute')):
                    second = data.prepare_manifest(manifest, root/'out', selected_roles=['development'])
            self.assertEqual(first, second)
            self.assertEqual(len(first['samples']), 1)
            self.assertEqual(first['unavailable_sources'][0]['dataset'], 'UVG')
            with self.assertRaisesRegex(ValueError, 'unknown sample'):
                data.prepare_manifest(manifest, root/'other', sample_ids=['missing'])


if __name__ == '__main__':
    unittest.main()
