"""Small deterministic CPU tests; no data, pretrained weights or GPU required."""
from dataclasses import replace
import copy
import tempfile
from pathlib import Path
import json
import unittest

import numpy as np
import torch

from demo import routervc_visual_router as module


def pixels(height=128, width=192):
    rng = np.random.default_rng(3)
    base = rng.integers(0, 180, size=(3, height, width, 3), dtype=np.uint8)
    return base, (base.astype(np.int16) + 20).astype(np.uint8)


def qualities():
    return [{s: dict(lpips_alex=p, psnr_db=q, temporal_delta_mae=t) for s, p, q, t in
             [('B', .5, 20., 5.), ('E', .4, 22., 4.), ('G', .3, 19., 6.), ('EG', .2, 21., 4.5)]}
            for _ in range(16)]


def content_teacher():
    from demo.routervc_content_objective import build_teacher_targets, CATEGORIES, SCHEMA, SCOPES
    importance = np.full((16, 3), np.nan)
    error = np.full((16, 4, 3), np.nan)
    status = np.full((16, 3), 'unknown', dtype='<U7')
    importance[0, 0] = 1.
    status[0, 0] = 'present'
    error[0, :, 0] = [.4, .2, .8, .3]
    metadata = dict(schema=SCHEMA, scope='offline_train_evaluation', source_role='train',
        sample_id='synthetic', source_sha256='0'*64, annotation_provenance='unit test',
        importance_definition='synthetic positive', scale_provenance='unit interval',
        state_order=list(module.STATES), category_order=list(CATEGORIES),
        region_ids=[str(i) for i in range(16)], metrics={k: dict(name='synthetic',
        definition='test fixture not a real metric', scope=SCOPES[k], scale=1., direction='lower_is_better') for k in CATEGORIES})
    p = np.array([[q[s]['lpips_alex'] for s in module.STATES] for q in qualities()])
    return build_teacher_targets(p, importance, error, status, metadata)


class VisualRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(5)
        self.config = module.VisualRouterConfig(channels=8, hidden=16, local_size=16, global_size=24)

    def test_dynamic_rectangular_receiver_inputs(self):
        base, enhanced = pixels()
        values = module.build_receiver_inputs(base, enhanced, np.ones(16), config=self.config)
        self.assertEqual(values['local_pairs'].shape, (1, 16, 3, 6, 16, 16))
        self.assertEqual(values['global_pairs'].shape, (1, 3, 6, 24, 24))
        self.assertEqual(values['geometry'].shape, (1, 16, 6))
        self.assertEqual(module.grid_rois(128, 192)[0], [0, 0, 48, 32])
        self.assertEqual(set(values), set(module.INPUT_KEYS))

    def test_shared_encoder_and_equal_capacity_ablation(self):
        base, enhanced = pixels()
        inputs = module.build_receiver_inputs(base, enhanced, np.ones(16), config=self.config)
        full = module.VisualUtilityRouter(self.config)
        local = module.VisualUtilityRouter(replace(self.config, use_global=False))
        local.load_state_dict(full.state_dict())
        self.assertEqual(full.metadata()['parameters'], local.metadata()['parameters'])
        changed = {**inputs, 'global_pairs': inputs['global_pairs'] * -1}
        with torch.no_grad():
            a, b = local(inputs), local(changed)
            for k in a:
                torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
            self.assertFalse(torch.equal(full(inputs)['gains'], full(changed)['gains']))
        self.assertEqual(full.metadata()['experts'], 0)

    def test_direct_E_zero_without_packets(self):
        base, _ = pixels()
        values = module.build_receiver_inputs(base, base, np.zeros(16), config=self.config)
        outputs = module.VisualUtilityRouter(self.config)(values)
        self.assertTrue(torch.equal(outputs['gains'][..., ::2], torch.zeros(1, 16, 3)))
        self.assertEqual(outputs['content_gain'][..., 0, :].abs().sum().item(), 0.)

    def test_conditional_targets_not_independent_G_sum(self):
        q = qualities()
        q[0]['EG']['lpips_alex'] = .35
        no_e = module.build_view_targets(q, 0)
        with_e = module.build_view_targets(q, 1)
        self.assertAlmostEqual(no_e['gains']['value'][0, 0, 1].item(), .2)
        self.assertAlmostEqual(with_e['gains']['value'][0, 0, 1].item(), .05)
        self.assertAlmostEqual(with_e['gains']['value'][0, 0, 0].item(), .1)
        self.assertEqual(with_e['gains']['weight'][0, 1:].sum().item(), 0.)

    def test_baseline_has_no_semantic_supervision(self):
        target = module.build_view_targets(qualities(), 0)
        meta = module.supervision_metadata(target)
        self.assertFalse(meta['semantic_supervision'])
        self.assertFalse(meta['content_fidelity_supervision'])
        self.assertTrue(meta['untrained_semantic_outputs_must_not_route'])

    def test_pure_lpips_labels_supported(self):
        q = [{s: {'lpips_alex': v['lpips_alex']} for s, v in row.items()} for row in qualities()]
        t = module.build_view_targets(q, 0)
        self.assertEqual(t['gains']['weight'][..., 2:].sum().item(), 0.)
        self.assertEqual(t['gains']['weight'][..., :2].sum().item(), 32.)

    def test_real_known_content_only_masks_and_parent_harm(self):
        teacher = content_teacher()
        base = module.build_view_targets(qualities(), 0, content_teacher=teacher)
        enhanced = module.build_view_targets(qualities(), 1, content_teacher=teacher)
        self.assertEqual(base['importance']['weight'].sum().item(), 1.)
        self.assertEqual(base['generation_harm']['weight'].sum().item(), 1.)
        self.assertAlmostEqual(base['generation_harm']['value'][0, 0, 0].item(), .4)
        self.assertAlmostEqual(enhanced['generation_harm']['value'][0, 0, 0].item(), .1)
        self.assertAlmostEqual(enhanced['content_gain']['value'][0, 0, 0, 0].item(), .2)
        self.assertTrue(module.supervision_metadata(base)['content_fidelity_supervision'])

    def test_unknown_NaN_cannot_poison_loss_or_become_safe_label(self):
        base, enhanced = pixels()
        view = module.build_training_view(base, enhanced, qualities(), 0, config=self.config)
        for name in ('content_gain', 'generation_harm', 'importance'):
            view['targets'][name]['value'].fill_(float('nan'))
        model = module.VisualUtilityRouter(self.config)
        total, terms = module.masked_training_loss(model(view['inputs']), view['targets'])
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(terms['generation_harm'].item(), 0.)
        total.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.output.weight.grad[6:].abs().sum().item(), 0.)
        view['targets']['importance']['weight'][0, 0, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'unknown target'):
            module.masked_training_loss(model(view['inputs']), view['targets'])

    def test_shared_encoder_receives_training_gradients(self):
        base, enhanced = pixels()
        view = module.build_training_view(base, enhanced, qualities(), 1, config=self.config,
                                          content_teacher=content_teacher())
        model = module.VisualUtilityRouter(self.config)
        loss, _ = module.masked_training_loss(model(view['inputs']), view['targets'])
        loss.backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.output.weight.grad[15:].abs().sum().item(), 0.)

    def test_lazy_view_and_batch(self):
        base, enhanced = pixels()
        before = base.copy()
        a = module.build_training_view(base, enhanced, qualities(), 0, config=self.config)
        b = module.build_training_view(base, enhanced, qualities(), 1, config=self.config)
        combined = module.batch_examples([a, b])
        self.assertEqual(combined['inputs']['coverage'].shape, (2, 16, 1))
        self.assertEqual(combined['inputs']['coverage'].sum().item(), 1.)
        self.assertTrue(np.array_equal(base, before))
        model = module.VisualUtilityRouter(self.config)
        self.assertEqual(model(combined['inputs'])['gains'].shape, (2, 16, 6))

    def test_single_target_case_equals_full_view_prediction(self):
        base, enhanced = pixels()
        model = module.VisualUtilityRouter(self.config).eval()
        for with_e in (False, True):
            case = module.build_training_case(base, enhanced, qualities(), 5, with_e, config=self.config)
            full = module.build_training_view(base, enhanced, qualities(), 6 if with_e else 0, config=self.config)
            self.assertEqual(case['inputs']['local_pairs'].shape[1], 1)
            torch.testing.assert_close(case['inputs']['global_pairs'], full['inputs']['global_pairs'], rtol=0, atol=0)
            with torch.no_grad():
                a, b = model(case['inputs']), model(full['inputs'])
            for key in a:
                torch.testing.assert_close(a[key], b[key][:, 5:6], rtol=1e-5, atol=1e-7)
            self.assertEqual(case['receiver_metadata']['frame_indices'], [0, 1, 2])
            self.assertEqual(case['receiver_metadata']['region_indices'], [5])

    def test_one_optimization_step_on_single_target_cases(self):
        base, enhanced = pixels()
        cases = [module.build_training_case(base, enhanced, qualities(), 0, flag, config=self.config)
                 for flag in (False, True)]
        batch = module.batch_examples(cases)
        model = module.VisualUtilityRouter(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.encoder[0].weight.detach().clone()
        loss, terms = module.masked_training_loss(model(batch['inputs']), batch['targets'])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.encoder[0].weight))
        self.assertFalse(module.supervision_metadata(batch['targets'])['semantic_supervision'])

    def test_loader_accepts_both_old_and_new_received_names(self):
        base, enhanced = pixels()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rois = module.grid_rois(128, 192)
            path = folder / 'labels.json'
            path.write_text(json.dumps(dict(regions=[dict(roi=r, quality=q) for r, q in zip(rois, qualities())])))
            for key in ('enhanced', 'all_E'):
                np.savez(folder / 'received.npz', base=base, **{key: enhanced})
                values = module.load_measured_sample(path, folder / 'received.npz')
                self.assertTrue(np.array_equal(values['enhanced'], enhanced))
                self.assertEqual(values['rois'], rois)

    def test_invalid_inputs_and_extra_source_field_rejected(self):
        base, enhanced = pixels()
        with self.assertRaises(ValueError):
            module.build_receiver_inputs(base.astype(float), enhanced, np.zeros(16), config=self.config)
        with self.assertRaises(ValueError):
            module.build_receiver_inputs(base, enhanced, np.full(16, np.nan), config=self.config)
        values = module.build_receiver_inputs(base, enhanced, np.zeros(16), config=self.config)
        with self.assertRaisesRegex(ValueError, 'fields'):
            module.VisualUtilityRouter(self.config)({**values, 'source': torch.zeros(1)})

    def test_state_dict_roundtrip(self):
        base, enhanced = pixels()
        model = module.VisualUtilityRouter(self.config)
        second = module.VisualUtilityRouter(self.config)
        second.load_state_dict(copy.deepcopy(model.state_dict()))
        values = module.build_receiver_inputs(base, enhanced, np.ones(16), config=self.config)
        with torch.no_grad():
            a, b = model(values), second(values)
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
