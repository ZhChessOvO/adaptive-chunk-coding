"""CPU-only mixed-view provenance, split preservation and reuse checks."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from demo import routervc_fullview_data as full
from demo import routervc_mixedview_data as mixed


class MixedViewTests(unittest.TestCase):
    def make_uvg(self, root, name='HoneyBee'):
        folder = root/name
        folder.mkdir()
        files = []
        value = hashlib.sha256()
        for i in range(17):
            file = folder/f'{i:08d}.png'
            Image.new('RGB', (512, 512), (i, 30, 60)).save(file)
            files.append(str(file))
            value.update(file.read_bytes())
        return dict(sample_id='uvg-honeybee-f032-center', sequence=name, source_files=files,
            frame_start=32, frame_count=17, original_crop=dict(name='center', x=704, y=284, width=512, height=512),
            original_format=dict(width=1920, height=1080, frame_count=600),
            selected_source_sha256=value.hexdigest())

    def test_uvg_retains_view_kind_frame_offset_crop_and_original_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self.make_uvg(root)
            sample = mixed.uvg_crop_record(row, {'source': 'fixture'})
            self.assertEqual(sample['original_frame_size'], [1920, 1080])
            self.assertEqual(sample['transform']['original_size'], [512, 512])
            self.assertEqual(sample['original_frame_start'], 32)
            self.assertEqual(sample['original_crop']['x'], 704)
            self.assertFalse(sample['whole_frame'])
            self.assertEqual(sample['view_kind'], 'existing_spatial_crop')
            self.assertFalse(sample['original_full_view_available'])
            self.assertEqual(mixed.load_rgb(sample).shape, (17, 512, 512, 3))

    def test_uvg_not_used_component_training_is_still_historical_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self.make_uvg(Path(temporary), 'YachtRide')
            eval_row = dict(official_sequence='YachtRide', original_format=row['original_format'],
                evaluation_sample=dict(frame_start=0, frame_count=17, crop=row['original_crop'], png_files=row['source_files']))
            sample = mixed.uvg_crop_record(eval_row, {}, evaluation=True)
            self.assertFalse(sample['component_training_sequence'])
            self.assertFalse(sample['independent_system_test'])
            self.assertEqual(sample['original_frame_start'], 0)

    def test_bad_crop_geometry_and_changed_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self.make_uvg(Path(temporary))
            sample = mixed.uvg_crop_record(row, {})
            Image.new('RGB', (512, 512), (255, 0, 0)).save(row['source_files'][0])
            with self.assertRaises(ValueError):
                mixed.verify_crop(sample)
            row['original_crop']['x'] = 1900
            with self.assertRaisesRegex(ValueError, 'outside'):
                mixed.uvg_crop_record(row, {})

    def test_fullview_reds_keeps_old_router_group(self):
        source = dict(dataset='REDS', split='train', sequence='010', whole_frame=True,
                      transform=full.geometry((1280, 720)), role='train')
        sample = mixed.reds_record(source, {('REDS', '010'): 'validation'})
        self.assertEqual(sample['router_split'], 'validation')
        self.assertEqual(sample['role'], 'development')
        self.assertEqual(sample['view_kind'], 'resized_full_frame')
        self.assertTrue(sample['component_training_sequence'])
        self.assertFalse(sample['independent_system_test'])
        sample = mixed.reds_record(dict(source, split='val', role='evaluation'), {}, evaluation=True)
        self.assertEqual(sample['router_split'], 'evaluation')
        self.assertFalse(sample['component_training_sequence'])

    def test_dataset_balanced_probabilities_ignore_validation_and_evaluation(self):
        samples = [dict(dataset='REDS', router_split='train') for _ in range(3)]
        samples += [dict(dataset='UVG', router_split='train') for _ in range(2)]
        samples += [dict(dataset='UVG', router_split=s) for s in ('validation', 'evaluation')]
        mixed.set_probabilities(samples, .25)
        self.assertAlmostEqual(sum(s['training_sample_probability'] for s in samples), 1.)
        self.assertAlmostEqual(sum(s['training_sample_probability'] for s in samples if s['dataset'] == 'UVG'), .25)
        self.assertTrue(all(s['training_sample_probability'] == 0 for s in samples[-2:]))
        for probability in (0., 1., float('nan')):
            with self.assertRaises(ValueError):
                mixed.set_probabilities(samples, probability)

    def test_previous_group_binding_and_sequence_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root/'runs'/mixed.PREVIOUS_COMPONENT
            labels = root/'runs'/mixed.PREVIOUS_LABELS
            config = root/'runs'/mixed.PREVIOUS_ROUTER
            entries = [dict(sample_id='a', dataset='REDS', sample=dict(sequence='000')),
                       dict(sample_id='b', dataset='UVG', sample=dict(sequence='HoneyBee'))]
            full.atomic_json(component, dict(entries=entries))
            full.atomic_json(labels, dict(complete=True, samples=[dict(sample_id='a'), dict(sample_id='b')]))
            full.atomic_json(config, dict(data=dict(labels=full.digest(labels)), train_indices=[0], validation_indices=[1]))
            actual, groups, dependencies = mixed.previous_layout(root)
            self.assertEqual(actual, entries)
            self.assertEqual(groups[('UVG', 'HoneyBee')], 'validation')
            self.assertEqual(dependencies['labels']['sha256'], full.digest(labels))
            entries[1].update(dataset='REDS', sample=dict(sequence='000'))
            full.atomic_json(component, dict(entries=entries))
            with self.assertRaisesRegex(ValueError, 'leaks'):
                mixed.previous_layout(root)

    def test_preparation_reuses_uvg_png_without_claiming_fullview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = mixed.uvg_crop_record(self.make_uvg(root), {})
            sample.update(router_split='validation', role='development')
            manifest = root/'manifest.json'
            full.atomic_json(manifest, dict(format=mixed.FORMAT, samples=[sample],
                code={p.name: full.digest(p) for p in (Path(mixed.__file__), Path(full.__file__))}))
            fake_usage = shutil._ntuple_diskusage(100, 10, 90)
            with patch.object(mixed.shutil, 'disk_usage', return_value=fake_usage), \
                 patch.object(mixed.subprocess, 'check_output', return_value='CPU only'):
                made = mixed.prepare_manifest(manifest, root/'prepared')
                resumed = mixed.prepare_manifest(manifest, root/'prepared')
            self.assertEqual(made, resumed)
            self.assertFalse(made['full_uvg_upload_required'])
            result = full.read(made['samples'][0]['view_json'])
            self.assertEqual(result['frames_dir'], str(root/'HoneyBee'))
            self.assertTrue(result['source_pngs_reused_without_copy'])
            self.assertFalse(result['sample']['whole_frame'])
            self.assertEqual(list((root/'prepared').rglob('*.png')), [])

    def test_prepare_tmux_and_profile_validation(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'tmux'):
                mixed.main(['prepare', '--manifest', 'none', '--output', 'none'])
        with self.assertRaisesRegex(ValueError, 'profile'):
            mixed.build_manifest(profile='missing')


if __name__ == '__main__':
    unittest.main()
