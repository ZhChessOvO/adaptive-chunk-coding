"""CPU-only synthetic sender allocation contracts, NOT trained model results.

Opaque payloads and synthetic source-free receive certificates exercise wiring;
real entropy decode and measured R_s/R_g/G quality remain separate GPU checks.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import routervc_sender_encode as encode
from demo import routervc_sender_router as sender
from demo import routervc_receiver_format as fmt
from demo.routervc_encode import compose_candidates
from demo.routervc_light_packets import bank_info
from demo.scalable_codec import atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.test_routervc_encode import make_bank
from demo.test_routervc_receiver_decode import config as receiver_config


def sha(wire):
    return hashlib.sha256(wire).hexdigest()


class SenderEncodeContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = np.zeros((17, 64, 96, 3), np.uint8)
        self.source = np.full_like(self.base, 120)
        self.all_e = np.full_like(self.base, 80)
        self.bank = make_bank(self.base, q=2.)
        atomic_npz(self.root/'received_E.npz', base=self.base, enhanced=self.all_e)
        self.receive = dict(complete=True, source_frames_read=False, sender_candidate_pixels_read=False,
            base_reference_unchanged=True, bank_sha256=sha(self.bank), base_hash=frame_hash(self.base),
            enhanced_hash=frame_hash(self.all_e), artifacts={'received_E.npz': file_hash(self.root/'received_E.npz')})
        atomic_json(self.root/'E.json', self.receive)
        self.args = dict(receive_record=self.root/'E.json', expected_receive_sha256=file_hash(self.root/'E.json'),
            expected_bank_sha256=sha(self.bank), expected_source_rgb_sha256=frame_hash(self.source))
        self.candidates = self.authenticate()
        self.config = receiver_config(8)
        self.model_path = self.root/'source'/'best.pt'
        self.model_path.parent.mkdir()
        self.model_cfg = sender.Config(global_size=32, local_size=16, channels=4, detail_channels=4, hidden=8)
        self.model = sender.SenderUtilityRouter(self.model_cfg)
        self.payload = self.make_payload()
        torch.save(self.payload, self.model_path)

    def authenticate(self, **changes):
        return encode.authenticate_candidates(self.bank, self.source, self.base, self.all_e,
                                              **{**self.args, **changes})

    def make_payload(self, smoke=True, zero_source=False):
        cfg = replace(self.model_cfg, zero_source=zero_source)
        model = sender.SenderUtilityRouter(cfg)
        teacher = dict(format='routervc_sender_final_teacher_v1', profile=fmt.PROFILE,
            label_scope=sender.LABEL_SCOPE, wire_config=self.config,
            receiver=dict(sha256=self.config['receiver_router'], smoke_weights=smoke),
            enhancement=dict(sha256='0'*64), max_g=8, boundary_lambda=0.)
        protocol = dict(format='routervc_sender_training_v1', smoke=smoke, teacher=teacher)
        return sender.export_payload(model, dict(protocol=protocol, architecture=asdict(cfg),
            arm='zero_source' if zero_source else 'source', labels={'synthetic_test_only': True}, scale={}),
            step=0 if smoke else 4, selection_metric={'synthetic_test_only': True})

    def context(self, **changes):
        return encode.load_sender(self.model_path, self.config,
            **{**dict(expected_sha256=file_hash(self.model_path), allow_smoke=True), **changes})

    def completion(self):
        path = self.root/'complete.json'
        binding = torch.load(self.model_path, weights_only=True)['binding']
        atomic_json(self.root/'config.json', dict(protocol=binding['protocol'],
            architectures={binding['arm']: binding['architecture']}, labels=binding['labels'], scale=binding['scale']))
        atomic_json(path, dict(complete=True, role=encode.ROLE,
            config=file_hash(self.root/'config.json'), artifacts={'source/best.pt': file_hash(self.model_path)}))
        return dict(completion_record=path, expected_completion_sha256=file_hash(path))

    def conditional(self, model, source, base, received, all_e, coverage, costs, max_g):
        self.assertIs(source, self.source)
        self.assertEqual(max_g, 8)
        ids = np.flatnonzero(coverage).tolist()
        np.testing.assert_array_equal(received, compose_candidates(base, all_e, ids, bank_info(self.bank)['rois']))
        result = torch.full((1, 16), -1.)
        # Static first-state sorting would choose 0,1,2. Conditioning flips 1/2.
        if not ids:
            result[0, 0], result[0, 1], result[0, 2] = 3*int(costs[0]), 2*int(costs[1]), int(costs[2])
        elif ids == [0]:
            result[0, 2], result[0, 1] = 2*int(costs[2]), int(costs[1])
        elif ids == [0, 2]:
            result[0, 1] = int(costs[1])
        return result

    def plan(self, predictor=None):
        with patch.object(sender, 'predict', side_effect=predictor or self.conditional):
            return encode.conditional_order(self.candidates, self.context())

    def test_candidate_authentication_and_full_e_proof(self):
        self.assertTrue(self.candidates.binding['candidate_cache_authenticated'])
        self.assertFalse(self.candidates.binding['candidate_encode_measured_here'])
        self.assertIsNone(self.candidates.binding['candidate_encode_seconds'])
        for change in ({'expected_receive_sha256': '0'*64}, {'expected_bank_sha256': '0'*64},
                       {'expected_source_rgb_sha256': '0'*64}):
            with self.assertRaises(ValueError):
                self.authenticate(**change)
        for key, bad in (('enhanced_hash', '0'*64), ('base_hash', '0'*64), ('bank_sha256', '0'*64),
                         ('complete', False), ('source_frames_read', True), ('sender_candidate_pixels_read', True),
                         ('base_reference_unchanged', False), ('artifacts', {})):
            atomic_json(self.root/'bad.json', {**self.receive, key: bad})
            with self.assertRaises(ValueError):
                self.authenticate(receive_record=self.root/'bad.json', expected_receive_sha256=file_hash(self.root/'bad.json'))

    def test_q1_incomplete_bank_geometry_and_changed_cache_rejected(self):
        for bad in (make_bank(self.base, q=1.), make_bank(self.base, q=2., skip=(9, 1))):
            with self.assertRaises(ValueError):
                encode.authenticate_candidates(bad, self.source, self.base, self.all_e,
                    **{**self.args, 'expected_bank_sha256': sha(bad)})
        with self.assertRaises(ValueError):
            encode.authenticate_candidates(self.bank, self.source[:-1], self.base, self.all_e, **self.args)
        atomic_npz(self.root/'received_E.npz', base=self.base, enhanced=self.base)
        with self.assertRaises(ValueError):
            self.authenticate()

    def test_full_candidate_pixels_are_not_source_substitutes(self):
        with self.assertRaises(ValueError):
            encode.authenticate_candidates(self.bank, self.source, self.base, self.source, **self.args)
        changed = self.all_e.copy()
        changed[0, 0, 0, 0] += 1
        with self.assertRaises(ValueError):
            encode.conditional_order(replace(self.candidates, all_e=changed), self.context())

    def test_smoke_requires_explicit_permission_and_completion_authenticates(self):
        with self.assertRaises(ValueError):
            self.context(allow_smoke=False)
        completion = self.completion()
        context = self.context(**completion)
        self.assertFalse(context.binding['formal_weights'])
        with self.assertRaises(ValueError):
            self.context(**{**completion, 'expected_completion_sha256': '0'*64})
        atomic_json(completion['completion_record'], dict(complete=True, role=encode.ROLE, artifacts={}))
        with self.assertRaises(ValueError):
            self.context(**{**completion, 'expected_completion_sha256': file_hash(completion['completion_record'])})

    def test_formal_provenance_requires_completion_step_and_exact_policy(self):
        # Synthetic metadata tests contract only; never a formal quality claim.
        payload = self.make_payload(smoke=False)
        torch.save(payload, self.model_path)
        with self.assertRaises(ValueError):
            self.context(allow_smoke=False)
        context = self.context(allow_smoke=False, **self.completion())
        self.assertTrue(context.binding['formal_weights'])
        payload['step'] = 0
        torch.save(payload, self.model_path)
        with self.assertRaises(ValueError):
            self.context(allow_smoke=False, **self.completion())

    def test_rg_g_budget_seed_hash_profile_all_bound(self):
        for key, value in (('max_g', 4), ('seed', 99), ('receiver_router', '1'*64), ('blend', .5)):
            with self.assertRaises(ValueError):
                encode.load_sender(self.model_path, {**self.config, key: value},
                    expected_sha256=file_hash(self.model_path), allow_smoke=True)
        payload = deepcopy(self.payload)
        payload['binding']['protocol']['teacher']['label_scope'] = 'isolated_direct_E_gain'
        torch.save(payload, self.model_path)
        with self.assertRaises(ValueError):
            self.context()

    def test_conditional_order_recomputes_actual_y_no_g_or_old_prepare(self):
        with (patch('demo.routervc_encode.prepare', side_effect=AssertionError('must not encode old q1')),
              patch('demo.routervc_receiver_policy.route', side_effect=AssertionError('must not run R_g/G'))):
            plan = self.plan()
        self.assertEqual(plan['order'], [0, 2, 1])
        self.assertEqual([row['selected_before'] for row in plan['trace']], [[], [0], [0, 2], [0, 2, 1]])
        self.assertEqual(plan['trace'][0]['received_hash'], frame_hash(self.base))
        self.assertNotEqual(plan['trace'][0]['received_hash'], plan['trace'][1]['received_hash'])
        self.assertEqual(plan['stop_reason'], 'no_positive_predicted_remaining_gain')
        self.assertEqual(plan['generator_calls'], 0)

    def test_multiple_budgets_are_same_literal_order_no_prediction_replay(self):
        costs = bank_info(self.bank)['e_bytes']
        budgets = [0, costs[0], costs[0]+costs[2], sum(costs)]
        with patch.object(sender, 'predict', side_effect=self.conditional) as prediction:
            plan, results = encode.plan_and_encode(self.candidates, self.context(), budgets)
        self.assertEqual(prediction.call_count, 4)
        self.assertEqual(plan['order'], [0, 2, 1])
        self.assertEqual([r['indices'] for _, r in results], [[], [0], [0, 2], [0, 2, 1]])
        for (short, _), (long, _) in zip(results, results[1:]):
            self.assertTrue(long.startswith(short))
        with patch.object(sender, 'predict', side_effect=AssertionError('frozen order cannot rerun')):
            self.assertEqual(encode.encode_prefix(self.candidates, plan, budgets[2]), results[2])

    def test_costly_middle_bundle_stops_prefix_does_not_skip(self):
        def prediction(model, x, b, y, e, coverage, costs, max_g):
            value = torch.full((1, 16), -1.)
            for i, strength in ((0, 3), (15, 2), (1, 1)):
                if not coverage[i]:
                    value[0, i] = strength*int(costs[i])
            return value
        plan = self.plan(prediction)
        self.assertEqual(plan['order'], [0, 15, 1])
        costs = plan['packet_bytes']
        _, ledger = encode.encode_prefix(self.candidates, plan, costs[0]+costs[1])
        self.assertEqual(ledger['indices'], [0])
        self.assertEqual(ledger['unused_e_budget'], costs[1])

    def test_bytes_packet_payloads_unchanged_and_no_masks_or_sender_on_wire(self):
        plan = self.plan()
        wire, ledger = encode.encode_prefix(self.candidates, plan, sum(plan['packet_bytes']))
        config, raw, parsed, header = fmt.parse(wire)
        info = bank_info(self.bank)
        expected = [p.wire for i in plan['order'] for p in info['groups'][i]]
        self.assertEqual([p.wire for p in parsed.packets], expected)
        self.assertEqual(parsed.base, info['parsed'].base)
        self.assertEqual(ledger['packet_count'], 9)
        self.assertEqual(ledger['header_bytes'], header)
        self.assertEqual(ledger['packet_bytes'], sum(map(len, expected)))
        self.assertEqual(len(wire), header+parsed.base_end+ledger['packet_bytes'])
        self.assertEqual(ledger['total_bytes'], len(wire))
        self.assertEqual(config, self.config)
        self.assertFalse({'source', 'sender_router', 'mask', 'order', 'prediction'} & set(config))
        self.assertNotIn('routervc_sender_router.py', fmt.CODE)
        for key in ('explicit_E_mask_bytes', 'explicit_G_map_bytes', 'protection_mask_bytes', 'sender_router_bytes'):
            self.assertEqual(ledger[key], 0)
        self.assertFalse(ledger['payloads_regenerated'])
        self.assertFalse(ledger['candidate_encode_measured_here'])

    def test_no_positive_gain_emits_only_base_header(self):
        for value in (0., -1.):
            plan = self.plan(lambda *args: torch.full((1, 16), value))
            wire, ledger = encode.encode_prefix(self.candidates, plan, 100000)
            self.assertEqual(plan['order'], [])
            self.assertEqual(len(plan['trace']), 1)
            self.assertEqual(ledger['packet_bytes'], 0)
            self.assertEqual(ledger['packet_count'], 0)
            self.assertEqual(len(fmt.parse(wire)[2].packets), 0)

    def test_positive_ties_are_deterministic_no_duplicates_max_sixteen(self):
        def equal_density(model, x, b, y, e, coverage, costs, max_g):
            return torch.tensor(costs, dtype=torch.float32).reshape(1, 16)
        plan = self.plan(equal_density)
        self.assertEqual(plan['order'], list(range(16)))
        self.assertEqual(len(plan['trace']), 16)
        self.assertEqual(plan['stop_reason'], 'all_regions_selected')

    def test_invalid_prediction_budget_or_tampered_plan_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(lambda *args: torch.full((1, 16), float('nan')))
        plan = self.plan()
        for budget in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                encode.encode_prefix(self.candidates, plan, budget)
        for key, value in (('order', [2, 0, 1]), ('config', {**self.config, 'max_g': 0}),
                           ('packet_bytes', [1]*16)):
            with self.assertRaises(ValueError):
                encode.encode_prefix(self.candidates, {**plan, key: value}, 100000)

    def test_saved_plan_reusable_after_new_cache_authentication(self):
        plan = self.plan()
        atomic_json(self.root/'plan.json', plan)
        loaded = json.loads((self.root/'plan.json').read_text())
        fresh = self.authenticate()
        self.assertEqual(fresh.binding, self.candidates.binding)
        self.assertEqual(encode.encode_prefix(fresh, loaded, 10000),
                         encode.encode_prefix(self.candidates, plan, 10000))

    def test_source_and_zero_source_arms_both_authenticated_independently(self):
        for zero_source in (False, True):
            payload = self.make_payload(zero_source=zero_source)
            payload['state_dict']['final_gain.2.weight'].zero_()
            payload['state_dict']['final_gain.2.bias'].fill_(-1.)
            torch.save(payload, self.model_path)
            context = self.context()
            self.assertEqual(context.model.config.zero_source, zero_source)
            self.assertEqual(context.binding['arm'], 'zero_source' if zero_source else 'source')
            # A real CPU R_s forward, with synthetic weights. Negative output
            # makes its one-step contract quick; this is not trained behavior.
            plan = encode.conditional_order(self.candidates, context)
            self.assertEqual(plan['order'], [])

    def test_sender_context_binding_or_enhancement_change_rejected(self):
        context = self.context()
        context.model.final_gain[-1].bias.data.add_(1.)
        with self.assertRaises(ValueError):
            encode.conditional_order(self.candidates, context)
        context = self.context()
        context.config['max_g'] = 4
        with self.assertRaises(ValueError):
            encode.conditional_order(self.candidates, context)
        payload = deepcopy(self.payload)
        payload['binding']['protocol']['teacher']['enhancement']['sha256'] = '1'*64
        torch.save(payload, self.model_path)
        with self.assertRaises(ValueError):
            encode.conditional_order(self.candidates, self.context())


if __name__ == '__main__':
    unittest.main()
