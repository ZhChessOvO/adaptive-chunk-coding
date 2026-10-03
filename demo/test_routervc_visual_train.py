"""Tiny CPU training/restart tests; formal teacher data and GPUs are untouched."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from demo import routervc_visual_router as visual
from demo import routervc_visual_train as train
from demo.test_routervc_visual_router import pixels, qualities, content_teacher


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def tiny_config(use_global=True):
    return visual.VisualRouterConfig(use_global=use_global, channels=4, hidden=8,
                                    local_size=16, global_size=16)


def tiny_data(config=None, *, semantic=False):
    config = config or tiny_config()
    base, enhanced = pixels(64, 96)
    cases = [visual.build_training_case(base, enhanced, qualities(), 0, with_e,
             config=config, content_teacher=content_teacher() if semantic else None)
             for with_e in (False, True)]
    return visual.batch_examples(cases)


def teacher_fixture(folder):
    """Two synthetic measured samples, grouped train/validation, real hashes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base, enhanced = pixels(64, 96)
    rows = []
    for i, split in enumerate(('train', 'validation')):
        sid = f'sample-{i}'
        received = folder / f'{sid}.npz'
        np.savez(received, base=base, enhanced=enhanced)
        label = folder / f'{sid}.json'
        value = dict(sample_id=sid, dataset='REDS', sequence=f'{i:03d}', router_split=split,
                     source_role='train', shape=list(base.shape), view_kind='synthetic_fullview',
                     content_annotation_status='unknown', regions=[dict(region=k, roi=r, quality=q)
                     for k, (r, q) in enumerate(zip(visual.grid_rois(64, 96), qualities()))])
        dump(label, value)
        rows.append(dict(sample_id=sid, dataset='REDS', sequence=f'{i:03d}', router_split=split,
                         source_role='train', view_kind='synthetic_fullview',
                         path=str(label), sha256=train.file_hash(label),
                         reconstruction_path=str(received), reconstruction_sha256=train.file_hash(received)))
    manifest = folder / 'labels.json'
    dump(manifest, dict(complete=True, samples=rows, states=list(visual.STATES), region_count=32))
    return manifest, rows


def alter_label_and_manifest(manifest, index, *, label_updates=None, row_updates=None):
    value = train.read(manifest)
    row = value['samples'][index]
    label = train.read(row['path'])
    label.update(label_updates or {})
    dump(row['path'], label)
    row['sha256'] = train.file_hash(Path(row['path']))
    row.update(row_updates or {})
    dump(manifest, value)


def snapshots(folder):
    return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in Path(folder).rglob('*') if p.is_file()}


class VisualTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree_exact(self, actual, expected):
        if torch.is_tensor(actual):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_tree_exact(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_tree_exact(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_three_epoch_resume_is_tensor_and_optimizer_exact(self):
        data = tiny_data()
        binding = dict(labels='a'*64, train_ids=['train'], valid_ids=['valid'], smoke=True)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            kwargs = dict(config=tiny_config(), data_binding=binding, epochs=3,
                          batch_size=2, device='cpu', seed=33)
            stopped = train.train_arm(data, data, folder/'resumed', stop_after=1, **kwargs)
            self.assertIsNone(stopped)
            self.assertFalse((folder/'resumed/complete.json').exists())
            self.assertEqual(torch.load(folder/'resumed/resume.pt', weights_only=True)['epoch'], 1)
            train.train_arm(data, data, folder/'resumed', **kwargs)
            train.train_arm(data, data, folder/'direct', **kwargs)
            for name in ('resume.pt', 'model.pt'):
                a = torch.load(folder/'resumed'/name, weights_only=True, map_location='cpu')
                b = torch.load(folder/'direct'/name, weights_only=True, map_location='cpu')
                self.assert_tree_exact(a, b)
            self.assertEqual(train.read(folder/'resumed/complete.json')['epochs'], 3)

    def test_mid_epoch_interrupt_replays_last_complete_epoch(self):
        data = tiny_data()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            kwargs = dict(config=tiny_config(False), data_binding={'labels': 'b'*64},
                          epochs=2, batch_size=1, device='cpu', seed=35)
            # epoch1: check + batch1 + batch2. Interrupt at epoch2 batch2.
            calls = 0
            def check():
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise InterruptedError('synthetic preemption')
            with self.assertRaisesRegex(InterruptedError, 'preemption'):
                train.train_arm(data, data, folder/'resumed', check=check, **kwargs)
            checkpoint = torch.load(folder/'resumed/resume.pt', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)
            train.train_arm(data, data, folder/'resumed', **kwargs)
            train.train_arm(data, data, folder/'direct', **kwargs)
            self.assert_tree_exact(torch.load(folder/'resumed/resume.pt', weights_only=True),
                                   torch.load(folder/'direct/resume.pt', weights_only=True))

    def test_complete_reentry_preserves_all_mtimes_and_semantic_false(self):
        data = tiny_data()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'train'
            kwargs = dict(config=tiny_config(), data_binding={'labels': 'c'*64},
                          epochs=2, batch_size=2, device='cpu')
            result = train.train_arm(data, data, output, **kwargs)
            before = snapshots(output)
            def forbidden():
                raise AssertionError('completed training must not run optimizer/check loop')
            repeated = train.train_arm(data, data, output, check=forbidden, **kwargs)
            self.assertEqual(result, repeated)
            self.assertEqual(before, snapshots(output))
            payload = torch.load(output/'model.pt', weights_only=True)
            self.assertFalse(payload['semantic_supervision'])
            self.assertFalse(payload['content_heads_usable'])
            self.assertFalse(result['semantic_supervision'])
            self.assertFalse(result['deployed_receiver'])

    def test_changed_teacher_binding_rejected_on_resume_and_completion(self):
        data = tiny_data()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'train'
            kwargs = dict(config=tiny_config(), epochs=2, batch_size=2, device='cpu')
            train.train_arm(data, data, output, data_binding={'labels': 'd'*64}, stop_after=1, **kwargs)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train.train_arm(data, data, output, data_binding={'labels': 'e'*64}, **kwargs)
            train.train_arm(data, data, output, data_binding={'labels': 'd'*64}, **kwargs)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train.train_arm(data, data, output, data_binding={'labels': 'e'*64}, **kwargs)

    def test_scales_and_weights_ignore_validation_values(self):
        data = tiny_data()
        valid = deepcopy(data)
        valid['targets']['gains']['value'] += 1000.
        scale = train.train_scales(data)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            kwargs = dict(config=tiny_config(), data_binding={'labels': 'f'*64},
                          epochs=2, batch_size=2, device='cpu', seed=37)
            train.train_arm(data, data, folder/'ordinary', **kwargs)
            train.train_arm(data, valid, folder/'different_validation', **kwargs)
            a = torch.load(folder/'ordinary/resume.pt', weights_only=True)
            b = torch.load(folder/'different_validation/resume.pt', weights_only=True)
            self.assert_tree_exact(a['model'], b['model'])
            self.assert_tree_exact(a['optimizer'], b['optimizer'])
            torch.testing.assert_close(a['scale'], scale, rtol=0, atol=0)
            torch.testing.assert_close(b['scale'], scale, rtol=0, atol=0)
            self.assertNotEqual(a['history'][-1]['validation'], b['history'][-1]['validation'])

    def test_unknown_semantic_targets_remain_untrained(self):
        data = tiny_data()
        for name in ('content_gain', 'generation_harm', 'importance'):
            data['targets'][name]['value'].fill_(float('nan'))
        with tempfile.TemporaryDirectory() as folder:
            result = train.train_arm(data, data, Path(folder), config=tiny_config(),
                data_binding={'labels': '0'*64}, epochs=2, batch_size=2, device='cpu')
            self.assertFalse(result['semantic_supervision'])
            self.assertTrue(np.isfinite(result['validation']['gain_mae']).all())

    def test_unknown_auxiliary_gains_do_not_pollute_validation(self):
        data = tiny_data()
        data['targets']['gains']['value'][..., 2:] = float('nan')
        data['targets']['gains']['weight'][..., 2:] = 0.
        model = visual.VisualUtilityRouter(tiny_config())
        result = train.validation(model, data, train.train_scales(data), 2, 'cpu')
        self.assertTrue(math.isfinite(result['loss']))
        self.assertTrue(all(value is None or math.isfinite(value) for value in result['gain_mae']))

    def test_semantically_supervised_input_cannot_be_mislabeled_baseline(self):
        ordinary, annotated = tiny_data(), tiny_data(semantic=True)
        with tempfile.TemporaryDirectory() as folder:
            for index, (a, b) in enumerate(((annotated, ordinary), (ordinary, annotated))):
                with self.assertRaisesRegex(ValueError, 'semantic'):
                    train.train_arm(a, b, Path(folder)/str(index), config=tiny_config(),
                        data_binding={'labels': '1'*64}, epochs=1, batch_size=2, device='cpu')

    def test_teacher_hash_changed_and_evaluation_split_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, rows = teacher_fixture(Path(folder))
            self.assertEqual(len(train.teacher_entries(manifest)), 2)
            payload = train.read(manifest)
            payload['samples'][0]['router_split'] = 'evaluation'
            dump(manifest, payload)
            with self.assertRaisesRegex(ValueError, 'evaluation'):
                train.teacher_entries(manifest)
            dump(manifest, dict(complete=True, samples=rows))
            Path(rows[0]['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                train.teacher_entries(manifest)

    def test_train_validation_sequence_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _ = teacher_fixture(Path(folder))
            alter_label_and_manifest(manifest, 1, label_updates={'sequence': '000'},
                                     row_updates={'sequence': '000'})
            with self.assertRaisesRegex(ValueError, 'sequence|group'):
                train.teacher_entries(manifest)

    def test_teacher_internal_identity_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _ = teacher_fixture(Path(folder))
            alter_label_and_manifest(manifest, 0, label_updates={'router_split': 'validation'})
            with self.assertRaises(ValueError):
                train.teacher_entries(manifest)

    def test_evaluation_source_cannot_be_relabeled_training(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, _ = teacher_fixture(Path(folder))
            alter_label_and_manifest(manifest, 0, label_updates={'source_role': 'evaluation'},
                                     row_updates={'source_role': 'evaluation'})
            with self.assertRaises(ValueError):
                train.teacher_entries(manifest)

    def test_cache_resume_hashes_and_global_ablation_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest, _ = teacher_fixture(folder/'teacher')
            cache = folder/'cache'
            calls = 0
            def check():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise InterruptedError('synthetic cache interruption')
            with self.assertRaisesRegex(InterruptedError, 'cache interruption'):
                train.prepare_cache(manifest, cache, tiny_config(), check)
            first = (cache/'sample-0.pt').stat().st_mtime_ns
            records = train.prepare_cache(manifest, cache, tiny_config())
            self.assertEqual(len(records), 2)
            self.assertEqual(first, (cache/'sample-0.pt').stat().st_mtime_ns)
            before = snapshots(cache)
            again = train.prepare_cache(manifest, cache, tiny_config(False))
            self.assertEqual(records, again)
            self.assertEqual(before, snapshots(cache))
            batch = train.load_cases(records)
            self.assertEqual(batch['inputs']['coverage'].shape[0], 64)
            self.assertFalse(visual.supervision_metadata(batch['targets'])['semantic_supervision'])
            saved = Path(records[0]['cache_path'])
            saved.write_bytes(saved.read_bytes() + b'bad')
            with self.assertRaisesRegex(ValueError, 'changed'):
                train.load_cases(records)


if __name__ == '__main__':
    unittest.main()
