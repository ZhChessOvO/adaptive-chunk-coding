"""CPU contracts for the independent receiver-only G utility model."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import routervc_receiver_router as rg
from demo import routervc_visual_router as visual
from demo import routervc_mixed_router as mixed


class ReceiverRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.config = rg.Config(channels=8, hidden=8, local_size=16, global_size=16)
        rng = np.random.default_rng(14)
        self.base = rng.integers(0, 256, (3, 64, 96, 3), dtype=np.uint8)
        self.received = self.base.copy()
        self.received[:, :32, :48] = 100
        self.coverage = np.zeros(16, dtype=np.float32)
        self.coverage[[0, 1, 4, 5]] = 1.

    def model(self):
        return rg.ReceiverGUtilityRouter(self.config)

    def inputs(self, halo=0):
        return rg.build_inputs(self.base, self.received, self.coverage, halo=halo, config=self.config)

    def payload(self, arm='core'):
        return rg.export_payload(self.model(), arm, {'smoke': True}, 1, {'loss': .1, 'regret': .01})

    def save(self, directory, payload):
        path = Path(directory)/'receiver.pt'
        torch.save(payload, path)
        return path

    def test_model_has_only_three_g_outputs_and_no_sender(self):
        model = self.model()
        output = model(self.inputs())
        self.assertEqual(output.shape, (1, 16, 3))
        self.assertEqual(model.output.out_features, 3)
        self.assertFalse(model.metadata()['sender_required'])
        self.assertFalse(model.metadata()['semantic_heads'])
        self.assertFalse(model.metadata()['direct_E_outputs'])
        self.assertEqual(sum(p.numel() for p in rg.ReceiverGUtilityRouter().parameters()), 24363)

    def test_migration_is_exact_g_projection(self):
        old = visual.VisualUtilityRouter(self.config).eval()
        new = rg.initialize_from_mixed(self.model(), old.state_dict()).eval()
        for halo in (0, 64):
            inputs = self.inputs(halo)
            with torch.no_grad():
                expected = old(inputs)['gains'][..., [1, 3, 5]]
                actual = new(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            rg.initialize_from_mixed(new, new.state_dict())

    def test_input_profiles_match_historical_preprocessing(self):
        for halo in (0, 64):
            actual = self.inputs(halo)
            expected = mixed.build_inputs(self.base, self.received, self.coverage,
                                           halo=halo, config=self.config)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.inputs()['local_pairs'], self.inputs(64)['local_pairs']))
        torch.testing.assert_close(self.inputs()['geometry'], self.inputs(64)['geometry'], rtol=0, atol=0)

    def test_inputs_reject_source_fields_bad_pixels_and_coverage(self):
        with self.assertRaises(TypeError):
            rg.build_inputs(self.base, self.received, self.coverage, source=self.base)
        with self.assertRaises(ValueError):
            self.model()({**self.inputs(), 'source': torch.zeros(1)})
        with self.assertRaises(ValueError):
            rg.build_inputs(self.base.astype(np.float32), self.received, self.coverage)
        for coverage in (np.zeros(15), np.full(16, np.nan), np.full(16, 1.1)):
            with self.assertRaises(ValueError):
                rg.build_inputs(self.base, self.received, coverage)
        with self.assertRaises(ValueError):
            rg.build_inputs(self.base, self.received, self.coverage, halo=True)

    def test_g_not_gated_by_e_coverage(self):
        model = self.model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output.bias.copy_(torch.tensor([.3, -.4, .1]))
        inputs = rg.build_inputs(self.base, self.base, np.zeros(16), config=self.config)
        expected = torch.tensor([.3, -.4, .1]).expand(1, 16, 3)
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)

    def test_targets_keep_unknown_and_negative_values(self):
        old = {'gains': {'value': torch.arange(96.).reshape(1, 16, 6),
                         'weight': torch.ones(1, 16, 6)}}
        old['gains']['value'][0, 2, 1] = -.2
        old['gains']['value'][0, 3, 3] = float('nan')
        old['gains']['weight'][0, 3, 3] = 0
        result = rg.g_only_targets(old)
        self.assertEqual(result['value'].shape, (1, 16, 3))
        self.assertLess(result['value'][0, 2, 0], 0)
        self.assertTrue(torch.isnan(result['value'][0, 3, 1]))
        self.assertEqual(result['weight'][0, 3, 1], 0)
        result['value'][0, 0, 0] = 111
        self.assertEqual(old['gains']['value'][0, 0, 1], 1)

    def test_perfect_prediction_huber_zero_scales_loss_only(self):
        pred = torch.arange(48., dtype=torch.float32).reshape(1, 16, 3)/100
        total, terms = rg.training_loss(pred, {'value': pred.clone(), 'weight': torch.ones_like(pred)},
                                       gain_scale=[.02, 1., .1], ranking_weight=0)
        self.assertEqual(total, 0.)
        self.assertEqual(terms['huber'], 0.)

    def test_ranking_rewards_right_order_and_keeps_negative_pairs(self):
        truth = torch.tensor([[[.3, 0., 0.], [.1, 0., 0.], [-.1, 0., 0.], [-.4, 0., 0.]]])
        target = {'value': truth, 'weight': torch.ones_like(truth)}
        _, right = rg.training_loss(truth, target)
        wrong = truth.flip(1).clone().requires_grad_()
        total, reversed_ = rg.training_loss(wrong, target)
        self.assertLess(right['ranking'], reversed_['ranking'])
        self.assertGreater(reversed_['ranking'], 0)
        total.backward()
        self.assertTrue(torch.isfinite(wrong.grad).all())
        self.assertNotEqual(wrong.grad[0, -1, 0].item(), 0.)

    def test_pairwise_loss_does_not_cross_views(self):
        # Each view is locally tied, but between-view labels/predictions differ.
        target_value = torch.zeros(2, 16, 3)
        target_value[1, :, 0] = 5
        pred = target_value.flip(0).clone()
        target = {'value': target_value, 'weight': torch.ones_like(pred)}
        _, terms = rg.training_loss(pred, target)
        self.assertEqual(terms['ranking'], 0.)

    def test_unknown_targets_do_not_contaminate_loss_or_gradient(self):
        pred = torch.zeros(2, 16, 3, requires_grad=True)
        values = torch.full_like(pred, float('nan'))
        weights = torch.zeros_like(pred)
        values[0, 0, 0], weights[0, 0, 0] = -.3, 1
        total, terms = rg.training_loss(pred, {'value': values, 'weight': weights})
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(terms['ranking'], 0.)
        total.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(torch.count_nonzero(pred.grad), 1)
        self.assertGreater(pred.grad[0, 0, 0], 0)

    def test_all_unknown_zero_loss_with_backward(self):
        pred = torch.zeros(1, 16, 3, requires_grad=True)
        total, _ = rg.training_loss(pred, {'value': torch.full_like(pred, float('nan')),
                                          'weight': torch.zeros_like(pred)})
        self.assertEqual(total, 0.)
        total.backward()
        self.assertEqual(torch.count_nonzero(pred.grad), 0)

    def test_bad_targets_and_scales_are_rejected(self):
        pred = torch.zeros(1, 16, 3)
        target = {'value': pred.clone(), 'weight': torch.ones_like(pred)}
        for scale in ([0, 1, 1], [1, 1], [float('nan'), 1, 1]):
            with self.assertRaises(ValueError):
                rg.training_loss(pred, target, gain_scale=scale)
        bad = deepcopy(target); bad['value'][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            rg.training_loss(pred, bad)
        bad = deepcopy(target); bad['weight'][0, 0, 0] = -1
        with self.assertRaises(ValueError):
            rg.training_loss(pred, bad)
        with self.assertRaises(ValueError):
            rg.training_loss(pred, target, ranking_weight=-1)

    def test_negative_gains_can_select_nothing(self):
        negative = -np.ones(16)
        self.assertEqual(rg.regret(negative, negative), 0.)
        self.assertEqual(mixed.best_g_indices(negative, 8), [])
        self.assertGreater(rg.regret(np.ones(16), negative), 0.)

    def test_export_reload_hash_rng_and_source_free(self):
        for arm in rg.ARMS:
            with tempfile.TemporaryDirectory() as directory:
                payload = self.payload(arm)
                path = self.save(directory, payload)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                before = torch.random.get_rng_state()
                loaded, actual = rg.load_model(path, expected_sha256=digest)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                self.assertEqual(actual['input_halo'], rg.INPUT_HALOS[arm])
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                # No files are opened after loading the one model: B/Y arrays
                # suffice without original frames, teachers or sender checkpoint.
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('unexpected asset read')):
                    result = rg.predict(loaded, self.base, self.received, self.coverage)
                with torch.no_grad():
                    expected = loaded(self.inputs(rg.INPUT_HALOS[arm]))
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    rg.load_model(path, expected_sha256='0'*64)

    def test_reload_rejects_wrong_format_metadata_source_and_nonfinite(self):
        changes = (
            lambda p: p.update(format=visual.FORMAT),
            lambda p: p.update(input_halo=64),
            lambda p: p.update(source='forbidden'),
            lambda p: p.update(semantic_supervision=True),
            lambda p: p['code'].update({'routervc_receiver_router.py': '0'*64}),
            lambda p: p['metadata'].update(sender_required=True),
            lambda p: p['state_dict']['output.bias'].fill_(float('nan')),
            lambda p: p['state_dict'].update({'output.bias': torch.zeros(18)}),
            lambda p: p['state_dict'].update({'output.bias': torch.zeros(3, dtype=torch.float64)}),
        )
        with tempfile.TemporaryDirectory() as directory:
            initial = self.payload()
            for mutate in changes:
                payload = deepcopy(initial); mutate(payload)
                with self.assertRaises(ValueError):
                    rg.load_model(self.save(directory, payload))

    def test_public_predict_rejects_unbound_model(self):
        with self.assertRaises(ValueError):
            rg.predict(self.model(), self.base, self.received, self.coverage)


if __name__ == '__main__':
    unittest.main()
