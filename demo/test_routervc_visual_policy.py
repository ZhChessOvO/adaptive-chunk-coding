"""Small CPU contracts for actual-pixel visual routing, no GPU or video data."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import routervc_visual_policy as policy
from demo import routervc_visual_router as visual
from demo.routervc_encode import compose_candidates, subset_bank
from demo.scalable_format import parse
from demo.test_routervc_encode import make_bank


def fixture_payload(*, use_global=True, constant=False):
    """Synthetic checkpoint format, not a claim that random weights are trained."""
    config = visual.VisualRouterConfig(use_global=use_global, channels=4, hidden=8,
                                       local_size=16, global_size=16)
    torch.manual_seed(16)
    model = visual.VisualUtilityRouter(config)
    if constant:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output.bias[:6] = torch.tensor([.1, .2, 1., 2., 3., 4.])
    training = dict(format=policy.TRAIN_FORMAT, model=asdict(config),
                    semantic_supervision=False, code={
                        'routervc_visual_router.py': hashlib.sha256(Path(visual.__file__).read_bytes()).hexdigest(),
                        'routervc_visual_train.py': '1'*64},
                    gain_scale_is_loss_weight_only=[100., 200., 300., 400., 500., 600.])
    return dict(format=visual.FORMAT, architecture=asdict(config), state_dict=model.state_dict(),
                training_binding=training, semantic_supervision=False, content_heads_usable=False,
                metadata=model.metadata())


class VisualPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.path = self.folder / 'model.pt'
        self.payload = fixture_payload()
        self.save()
        self.base = np.zeros((17, 64, 96, 3), np.uint8)
        self.enhanced = np.zeros_like(self.base)
        self.rois = visual.grid_rois(64, 96)
        for i, (x, y, w, h) in enumerate(self.rois):
            self.enhanced[:, y:y+h, x:x+w] = 10*(i+1)
        self.bank = make_bank(self.base)

    def save(self):
        torch.save(self.payload, self.path)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def config(self):
        return dict(router=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    policy=policy.policy_identity(), max_g=8, boundary_lambda=0.)

    def test_load_cpu_evaluation_and_random_state_preservation(self):
        rng = torch.random.get_rng_state().clone()
        model = policy.load_model(self.path, expected_sha256=self.config()['router'])
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertFalse(model.training)
        self.assertTrue(all(p.device.type == 'cpu' and not p.requires_grad for p in model.parameters()))
        actual = policy.predict(model, self.base, self.base, np.zeros(16))
        self.assertEqual(actual.shape, (1, 16, 6))
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual[..., [0, 2, 4]], torch.zeros(1, 16, 3), rtol=0, atol=0)

    def test_checkpoint_fail_closed(self):
        def bad_state_shape(p):
            p['state_dict']['output.bias'] = torch.zeros(17)
        def bad_state_dtype(p):
            p['state_dict']['output.bias'] = p['state_dict']['output.bias'].double()
        def bad_state_finite(p):
            p['state_dict']['output.bias'][15] = float('nan')
        def bad_training_code(p):
            p['training_binding']['code']['routervc_visual_router.py'] = '0'*64
        changes = [
            lambda p: p.update(format='old-statistical-router'),
            lambda p: p.update(semantic_supervision=True),
            lambda p: p.update(semantic_supervision=0),
            lambda p: p.update(content_heads_usable=True),
            lambda p: p.pop('content_heads_usable'),
            lambda p: p.update(unexpected_protection_mask=[]),
            lambda p: p['architecture'].update(use_global='yes'),
            lambda p: p['architecture'].update(channels=5),
            lambda p: p['training_binding'].update(semantic_supervision=True),
            lambda p: p['training_binding'].update(model={}),
            lambda p: p['metadata'].update(gain_names=['wrong']),
            lambda p: p['state_dict'].update(unexpected_tensor=torch.zeros(1)),
            bad_state_shape, bad_state_dtype, bad_state_finite, bad_training_code,
        ]
        original = deepcopy(self.payload)
        for index, change in enumerate(changes):
            self.payload = deepcopy(original)
            change(self.payload)
            self.save()
            with self.subTest(index=index), self.assertRaises(ValueError):
                policy.load_model(self.path)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            policy.load_model(self.path, expected_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'expected Router hash'):
            policy.load_model(self.path, expected_sha256='bad')

    def test_no_gain_scale_inverse_and_no_semantic_outputs_used(self):
        self.payload = fixture_payload(constant=True)
        self.save()
        model = policy.load_model(self.path)
        fractions = np.array([0., .5, 1.] + [0.]*13, np.float32)
        result = policy.predict(model, self.base, self.enhanced, fractions)[0]
        expected = torch.tensor([.1, .2, 1., 2., 3., 4.]).repeat(16, 1)
        expected[:, [0, 2, 4]] *= torch.from_numpy(fractions[:, None])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        original = model.forward
        def poisoned_semantics(inputs):
            outputs = original(inputs)
            for key in ('content_gain', 'generation_harm', 'importance_logits'):
                outputs[key].fill_(float('nan'))
            return outputs
        with patch.object(model, 'forward', side_effect=poisoned_semantics):
            again = policy.predict(model, self.base, self.enhanced, fractions)[0]
        torch.testing.assert_close(result, again, rtol=0, atol=0)

    def test_unbound_model_invalid_pixels_and_nan_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unbound model'):
            policy.predict(visual.VisualUtilityRouter(), self.base, self.base, np.zeros(16))
        model = policy.load_model(self.path)
        for base, received, coverage in (
                (self.base.astype(float), self.base, np.zeros(16)),
                (self.base, self.base[:3], np.zeros(16)),
                (self.base, self.base, np.full(16, np.nan)),
                (self.base, self.base, np.full(16, 1.1)),
                (self.base, self.base, np.zeros(15))):
            with self.assertRaises(ValueError):
                policy.predict(model, base, received, coverage)
        with patch.object(model, 'forward', return_value={'gains': torch.full((1, 16, 6), float('nan'))}):
            with self.assertRaisesRegex(ValueError, 'predictions'):
                policy.predict(model, self.base, self.base, np.zeros(16))

    def test_sender_rebuilds_seventeen_real_global_views_and_conditional_utility(self):
        model = policy.load_model(self.path)
        seen, inputs_seen = [], []
        original = visual.build_receiver_inputs
        def observe(base, received, coverage, rois=None, **kwargs):
            seen.append((received.copy(), np.asarray(coverage).copy()))
            values = original(base, received, coverage, rois, **kwargs)
            inputs_seen.append({k: v.clone() for k, v in values.items()})
            return values
        with patch.object(visual, 'build_receiver_inputs', side_effect=observe):
            utility, predictions = policy.predict_utility(self.bank, self.base, self.enhanced, model)
        self.assertEqual(len(seen), 17)
        self.assertEqual(predictions.shape, (17, 16, 6))
        np.testing.assert_array_equal(seen[0][0], self.base)
        for i in range(16):
            expected = compose_candidates(self.base, self.enhanced, [i], self.rois)
            np.testing.assert_array_equal(seen[i+1][0], expected)
            np.testing.assert_array_equal(seen[i+1][1], np.eye(16, dtype=np.float32)[i])
            self.assertFalse(torch.equal(inputs_seen[0]['global_pairs'], inputs_seen[i+1]['global_pairs']))
            truth = original(self.base, expected, seen[i+1][1], self.rois, config=model.config)
            for key in truth:
                torch.testing.assert_close(inputs_seen[i+1][key], truth[key], rtol=0, atol=0)
            self.assertAlmostEqual(float(utility[i, 1]), float(predictions[i+1, i, 0]))
            self.assertAlmostEqual(float(utility[i, 2]), float(predictions[0, i, 1]))
            self.assertAlmostEqual(float(utility[i, 3]), float(predictions[i+1, i, 0]+predictions[i+1, i, 1]))
        np.testing.assert_array_equal(utility[:, 0], 0)
        np.testing.assert_array_equal(self.base, 0)  # Sender input is never mutated.

    def test_sender_receiver_same_y_exact_with_actual_mixed_conditions(self):
        model = policy.load_model(self.path)
        _, predictions = policy.predict_utility(self.bank, self.base, self.enhanced, model)
        for i in (0, 5, 15):
            inner = parse(subset_bank(self.bank, [i]))
            mixed = compose_candidates(self.base, self.enhanced, [i], self.rois)
            result = policy.route(self.base, mixed, inner, self.config(), self.path)
            np.testing.assert_array_equal(result['predictions'], predictions[i+1])
            self.assertFalse(result['semantic_heads_used'])
            self.assertEqual(result['explicit_G_map_bytes'], 0)
            self.assertEqual(result['mask_removal_savings_bytes'], 0)
        chosen = [0, 5, 10]
        mixed = compose_candidates(self.base, self.enhanced, chosen, self.rois)
        inner = parse(subset_bank(self.bank, chosen))
        coverage = policy.coverage_from_packets(inner, self.rois)
        expected = policy.predict(model, self.base, mixed, coverage)[0].numpy()
        result = policy.route(self.base, mixed, inner, self.config(), self.path)
        np.testing.assert_array_equal(result['predictions'], expected)
        repeated = policy.route(self.base, mixed, inner, self.config(), self.path)
        self.assertEqual(result, repeated)
        json.dumps(result, allow_nan=False)

    def test_packet_coverage_zero_partial_full_and_partial_tail(self):
        wire = subset_bank(self.bank, [5])
        complete = parse(wire)
        for end, fraction in ((complete.base_end, 0),
                              (complete.packets[0].end_offset, 1/17),
                              (complete.packets[1].end_offset, 9/17),
                              (complete.packets[-1].end_offset, 1)):
            inner = parse(wire[:end])
            actual = policy.coverage_from_packets(inner, self.rois)
            expected = np.zeros(16, np.float32); expected[5] = fraction
            np.testing.assert_array_equal(actual, expected)
            received = self.base.copy()
            for packet in inner.packets:
                x, y, w, h = packet.meta['roi']; start = packet.meta['start']; n = packet.meta['count']
                received[start:start+n, y:y+h, x:x+w] = self.enhanced[start:start+n, y:y+h, x:x+w]
            routed = policy.route(self.base, received, inner, self.config(), self.path)
            np.testing.assert_array_equal(routed['coverage'], expected)
        partial = parse(wire[:-1], allow_incomplete_tail=True)
        self.assertEqual(policy.coverage_from_packets(partial, self.rois)[5], np.float32(9/17))

    def test_route_authentication_geometry_and_duplicate_packets_fail_closed(self):
        inner = parse(subset_bank(self.bank, []))
        for update in ({'router': '0'*64}, {'policy': '0'*64}, {'max_g': True},
                       {'boundary_lambda': -1}, {'boundary_lambda': float('nan')}):
            with self.assertRaises(ValueError):
                policy.route(self.base, self.base, inner, {**self.config(), **update}, self.path)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            policy.route(self.base[:2], self.base[:2], inner, self.config(), self.path)
        one = parse(subset_bank(self.bank, [0])).packets[0]
        duplicate = SimpleNamespace(meta=inner.meta, packets=[one, one])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            policy.coverage_from_packets(duplicate, self.rois)
        # New wire profile owns an aggregate digest and supplies it explicitly.
        config = {**self.config(), 'policy': 'f'*64}
        result = policy.route(self.base, self.base, inner, config, self.path, expected_policy='f'*64)
        self.assertEqual(result['sampled_frame_indices'], [0, 8, 16])

    def test_fixed_ties_boundaries_and_budget_prefixes(self):
        self.assertEqual(policy.select_generate(np.zeros(16), 16)['indices'], [])
        self.assertEqual(policy.select_generate(np.ones(16), 1)['indices'], [0])
        gains = np.full(16, -10.); gains[[5, 15]] = 1.
        self.assertEqual(policy.select_generate(gains, 1)['indices'], [15])  # Fewer boundary edges.
        utility = np.zeros((16, 4)); utility[:, [1, 3]] = 1.
        costs = [10]*16
        for mode in ('prefix', 'independent'):
            small = policy.allocate(utility, costs, 20, 0, mode=mode)
            large = policy.allocate(utility, costs, 40, 0, mode=mode)
            self.assertEqual(small['selected_indices'], [0, 1])
            self.assertEqual(large['selected_indices'], [0, 1, 2, 3])
            self.assertEqual(small['e_packet_bytes'], 20)
            self.assertEqual(small, policy.allocate(utility, costs, 20, 0, mode=mode))
            self.assertFalse(small['generation_mask_transmitted'])
        self.assertEqual(policy.allocate(np.zeros((16, 4)), costs, 160, 16)['selected_indices'], [])
        bad = [(utility, [10.5]*16, 20, 0), (utility, [0]*16, 20, 0),
               (utility, costs, True, 0), (utility, costs, 20, True),
               (np.full((16, 4), np.nan), costs, 20, 0)]
        for args in bad:
            with self.assertRaises(ValueError):
                policy.allocate(*args)

    def test_identity_covers_preprocessing_and_changes_fail_closed(self):
        hashes = policy.code_hashes()
        self.assertIn('routervc_visual_router.py', hashes)
        self.assertIn('routervc_visual_policy.py', hashes)
        self.assertIn('routervc_encode.py', hashes)
        before = policy.policy_identity()
        with patch.object(policy, 'code_hashes', return_value={**hashes, 'routervc_visual_router.py': '0'*64}):
            self.assertNotEqual(before, policy.policy_identity())

    def test_local_ablation_does_not_depend_on_faraway_pixels(self):
        self.payload = fixture_payload(use_global=False)
        self.save()
        model = policy.load_model(self.path)
        first = policy.predict(model, self.base, self.base, np.zeros(16))
        received = compose_candidates(self.base, self.enhanced, [15], self.rois)
        second = policy.predict(model, self.base, received, np.zeros(16))
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
