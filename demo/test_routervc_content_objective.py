"""CPU-only checks for the prospective offline content teacher contract."""
from copy import deepcopy
import unittest

import numpy as np

from demo.routervc_content_objective import (
    CATEGORIES, SCHEMA, SCOPES, STATES, build_teacher_targets,
)


def fixture(regions=2):
    metadata = dict(schema=SCHEMA, scope='offline_train_evaluation',
                    source_role='train', sample_id='synthetic-contract-only',
                    source_sha256='a' * 64, annotation_provenance='synthetic unit test',
                    importance_definition='externally annotated weight in [0,1]',
                    scale_provenance='fixed unit test units, not fitted to candidates',
                    state_order=list(STATES), category_order=list(CATEGORIES),
                    region_ids=[str(i) for i in range(regions)],
                    metrics={c: dict(name='synthetic-' + c, definition='test fixture error',
                                     scope=SCOPES[c], scale=1., direction='lower_is_better')
                             for c in CATEGORIES})
    p = np.tile([.4, .3, .2, .1], (regions, 1))
    w = np.ones((regions, 3))
    e = np.tile(np.asarray([.4, .1, .5, .2])[None, :, None], (regions, 1, 3))
    status = np.full((regions, 3), 'present')
    return p, w, e, status, metadata


class ContentObjectiveTest(unittest.TestCase):
    def test_face_landmarks_are_explicit_not_identity(self):
        p, w, e, status, metadata = fixture(1)
        metadata['metrics']['face']['scope'] = 'face_landmarks'
        metadata['metrics']['face']['definition'] = 'normalized five-point geometry error, not identity'
        result = build_teacher_targets(p, w, e, status, metadata)
        self.assertEqual(result['metadata']['metrics']['face']['scope'], 'face_landmarks')
        self.assertFalse(result['semantic_guarantee'])
        metadata['metrics']['face']['scope'] = 'unspecified_semantics'
        with self.assertRaises(ValueError):
            build_teacher_targets(p, w, e, status, metadata)

    def test_zero_importance_exactly_recovers_lpips(self):
        p, w, e, status, metadata = fixture()
        w[:] = 0
        e[:] = np.nan  # assessed zero importance needs no metric inference
        result = build_teacher_targets(p, w, e, status, metadata)
        np.testing.assert_array_equal(result['objective_cost'], p)
        np.testing.assert_array_equal(result['generation_harm'], 0)
        np.testing.assert_array_equal(result['targets']['objective_gain']['value'],
                                      result['targets']['lpips_gain']['value'])
        self.assertTrue(result['objective_known'].all())

    def test_eg_harm_compares_to_e_not_base(self):
        result = build_teacher_targets(*fixture(1))
        # EG .2 is better than B .4 but worse than its real parent E .1.
        np.testing.assert_allclose(result['generation_harm'], [[.1, .1]])
        np.testing.assert_allclose(result['objective_cost'], [[.8, .4, .8, .4]])

    def test_harm_is_positive_increment_not_absolute_error(self):
        p, w, e, status, metadata = fixture(1)
        e[0, :, :] = np.asarray([.9, .7, .6, .4])[:, None]
        result = build_teacher_targets(p, w, e, status, metadata)
        np.testing.assert_array_equal(result['generation_harm'], [[0., 0.]])
        self.assertGreater(result['content_cost'][0, 2], 0)

    def test_harm_does_not_cancel_across_categories(self):
        p, w, e, status, metadata = fixture(1)
        e[0, 0] = [.2, .8, .4]
        e[0, 2] = [.8, .2, .4]
        result = build_teacher_targets(p, w, e, status, metadata)
        self.assertAlmostEqual(result['generation_harm'][0, 0], .2)

    def test_no_annotations_gives_no_content_or_harm_training(self):
        p, w, e, _, metadata = fixture()
        w[:] = np.nan
        e[:] = np.nan
        status = np.full(w.shape, 'unknown')
        result = build_teacher_targets(p, w, e, status, metadata)
        for name in ('content_gain', 'category_content_gain', 'generation_harm',
                     'category_generation_harm', 'objective_gain', 'importance'):
            np.testing.assert_array_equal(result['targets'][name]['weight'], 0)
            self.assertTrue(np.isfinite(result['targets'][name]['value']).all())
        self.assertTrue(np.isnan(result['objective_cost']).all())
        np.testing.assert_array_equal(result['targets']['lpips_gain']['weight'], 1)

    def test_lpips_ablation_works_without_annotations(self):
        p, w, e, _, metadata = fixture()
        w[:] = np.nan
        e[:] = np.nan
        result = build_teacher_targets(p, w, e, np.full(w.shape, 'unknown'), metadata,
                                       content_weight=0, harm_weight=0)
        np.testing.assert_array_equal(result['objective_cost'], p)
        np.testing.assert_array_equal(result['targets']['objective_gain']['weight'], 1)
        np.testing.assert_array_equal(result['targets']['generation_harm']['weight'], 0)

    def test_partial_annotation_usable_by_category_not_aggregate(self):
        p, w, e, status, metadata = fixture(1)
        status = status.astype('U7')
        status[0, 1] = 'unknown'
        w[0, 1] = np.nan
        e[0, :, 1] = np.nan
        result = build_teacher_targets(p, w, e, status, metadata)
        np.testing.assert_array_equal(result['targets']['generation_harm']['weight'], 0)
        np.testing.assert_array_equal(result['targets']['category_generation_harm']['weight'],
                                      [[[1., 0., 1.], [1., 0., 1.]]])

    def test_missing_eg_error_only_masks_eg_related_labels(self):
        p, w, e, status, metadata = fixture(1)
        e[0, 3, 0] = np.nan
        result = build_teacher_targets(p, w, e, status, metadata)
        np.testing.assert_array_equal(result['targets']['objective_gain']['weight'], [[1, 1, 0]])
        np.testing.assert_array_equal(result['targets']['generation_harm']['weight'], [[1, 0]])

    def test_absent_requires_actual_explicit_assessment(self):
        p, w, e, _, metadata = fixture(1)
        status = np.full(w.shape, 'absent')
        with self.assertRaises(ValueError):
            build_teacher_targets(p, w, e, status, metadata)
        w[:] = 0
        e[:] = np.nan
        result = build_teacher_targets(p, w, e, status, metadata)
        np.testing.assert_array_equal(result['objective_cost'], p)
        np.testing.assert_array_equal(result['targets']['importance']['weight'], 1)

    def test_unknown_cannot_be_filled_as_safe(self):
        p, w, e, _, metadata = fixture(1)
        status = np.full(w.shape, 'unknown')
        for bad_w, bad_e in ((np.zeros_like(w), np.full_like(e, np.nan)),
                             (np.full_like(w, np.nan), np.zeros_like(e))):
            with self.assertRaises(ValueError):
                build_teacher_targets(p, bad_w, bad_e, status, metadata)

    def test_fixed_normalization_preserves_importance_magnitude(self):
        p, w, e, status, metadata = fixture(1)
        baseline = build_teacher_targets(p, w, e, status, metadata)
        weighted = build_teacher_targets(p, .25 * w, e, status, metadata)
        np.testing.assert_allclose(weighted['content_cost'], .25 * baseline['content_cost'])
        np.testing.assert_allclose(weighted['generation_harm'], .25 * baseline['generation_harm'])
        for spec in metadata['metrics'].values():
            spec['scale'] = 2
        scaled = build_teacher_targets(p, w, e, status, metadata)
        np.testing.assert_allclose(scaled['content_cost'], .5 * baseline['content_cost'])

    def test_no_g_states_are_not_harm_negative_examples(self):
        result = build_teacher_targets(*fixture(1))
        self.assertEqual(result['targets']['generation_harm']['value'].shape, (1, 2))
        self.assertEqual(result['generation_parents'], ('G_over_B', 'EG_over_E'))
        self.assertEqual(result['states'], STATES)

    def test_optional_importance_and_no_receiver_packet_output(self):
        result = build_teacher_targets(*fixture(1), include_importance_target=False)
        self.assertNotIn('importance', result['targets'])
        self.assertTrue(result['offline_supervision_only'])
        self.assertFalse(result['receiver_inputs'])
        self.assertFalse(result['semantic_guarantee'])
        for name in ('protect', 'protection', 'packet', 'route', 'mask'):
            self.assertNotIn(name, result)

    def test_metadata_is_required_and_copied(self):
        args = list(fixture(1))
        metadata = args[-1]
        for key in ('annotation_provenance', 'scale_provenance', 'source_sha256',
                    'region_ids', 'metrics'):
            bad = deepcopy(metadata)
            del bad[key]
            with self.assertRaises(ValueError, msg=key):
                build_teacher_targets(*args[:-1], bad)
        result = build_teacher_targets(*args)
        metadata['metrics']['face']['scale'] = 99
        self.assertEqual(result['metadata']['metrics']['face']['scale'], 1)

    def test_reject_generic_metrics_mislabeled_as_content_scope(self):
        args = list(fixture(1))
        args[-1]['metrics']['face']['scope'] = 'psnr'
        with self.assertRaises(ValueError):
            build_teacher_targets(*args)

    def test_reject_bad_shapes_values_and_contract(self):
        for replacement in (np.zeros((1, 3)), [[.4, .3, np.inf, .1]], [[-.1, .3, .2, .1]]):
            args = list(fixture(1))
            args[0] = replacement
            with self.assertRaises(ValueError):
                build_teacher_targets(*args)
        for key, value in (('scope', 'receiver'), ('source_role', 'deployment'),
                           ('state_order', ['B', 'G', 'E', 'EG'])):
            args = list(fixture(1))
            args[-1][key] = value
            with self.assertRaises(ValueError):
                build_teacher_targets(*args)
        for key in ('content_weight', 'harm_weight'):
            for value in (-1, np.nan, np.inf, True):
                with self.assertRaises(ValueError):
                    build_teacher_targets(*fixture(1), **{key: value})


if __name__ == '__main__':
    unittest.main()
