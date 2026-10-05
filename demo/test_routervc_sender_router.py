"""CPU-only independent source-aware sender contracts; no teacher/GPU training."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from demo import routervc_sender_router as sender


def fixture():
    config = sender.Config(global_size=32, local_size=16, channels=4,
                           detail_channels=4, hidden=8)
    rng = np.random.default_rng(1025)
    x = rng.integers(0, 256, (17, 64, 96, 3), dtype=np.uint8)
    b = (x//2).astype(np.uint8)
    y = b.copy()
    candidate = np.clip(b.astype(np.int16)+30, 0, 255).astype(np.uint8)
    coverage = np.zeros(16, np.float32)
    costs = np.arange(1, 17, dtype=np.int64)*100
    return config, [x, b, y, candidate, coverage, costs, 8]


def targets():
    values = torch.linspace(-.2, .2, 16).reshape(1, 16)
    return dict(value=values, weight=torch.ones_like(values), label_scope=sender.LABEL_SCOPE)


def tree_equal(test, a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a:
            tree_equal(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            tree_equal(test, x, y)
    else:
        test.assertEqual(a, b)


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def _assert_exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _assert_exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_exact(x, y)
    else:
        assert a == b


def resume_equivalence(device, output_directory):
    """Root-owned synthetic smoke: default142897param Rs, six vs3+save+3.

Never called on CUDA by this CPU test suite. Root may call this inside its
tmux/shared-GPU-lock queue with CUBLAS_WORKSPACE_CONFIG already exported before
CUDA initialization. It writes a reloadable checkpoint, never a formal model.
"""
    device = torch.device(device)
    if device.type == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('export deterministic CUBLAS_WORKSPACE_CONFIG before the CUDA smoke')
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    cpu_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(20261005)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        config = sender.Config()
        _, args = fixture()
        inputs = {k: v.to(device) for k, v in sender.build_inputs(*args, config=config).items()}
        truth = targets()
        initial_model = sender.SenderUtilityRouter(config)
        initial = _cpu_tree(initial_model.state_dict())
        def fresh():
            model = sender.SenderUtilityRouter(config).to(device)
            model.load_state_dict(initial)
            return model, torch.optim.AdamW(model.parameters(), lr=1e-4)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = sender.training_loss(model(inputs), truth, inputs['packet_bytes'], gain_scale=.1)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        continuous, co = fresh()
        for _ in range(6):
            update(continuous, co)
        expected_model, expected_optimizer = _cpu_tree(continuous.state_dict()), _cpu_tree(co.state_dict())
        del continuous, co
        interrupted, io = fresh()
        for _ in range(3):
            update(interrupted, io)
        checkpoint = dict(sender=sender.export_payload(interrupted, {'synthetic_smoke_only': True}, 3, {}),
                          optimizer=_cpu_tree(io.state_dict()), cursor=3,
                          cpu_rng=torch.random.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        path = folder/'synthetic_resume.pt'
        torch.save(checkpoint, path)
        del interrupted, io
        state = torch.load(path, weights_only=True, map_location='cpu')
        resumed, ro = fresh()
        resumed.load_state_dict(state['sender']['state_dict'])
        ro.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['cpu_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        for _ in range(state['cursor'], 6):
            update(resumed, ro)
        actual = _cpu_tree(resumed.state_dict())
        _assert_exact(expected_model, actual)
        _assert_exact(expected_optimizer, _cpu_tree(ro.state_dict()))
        assert any(not torch.equal(v, initial[k]) for k, v in actual.items())
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return dict(complete=True, device=str(device), parameters=initial_model.metadata()['parameters'],
                    continuous_steps=6, interrupted_steps=3, resumed_steps=3,
                    model_exact=True, optimizer_exact=True, weights_changed=True,
                    deterministic_algorithms=True, source=sender.code_identity(),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                    checkpoint=str(path), synthetic_only=True, formal_sender_training=False)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = flags
        torch.set_num_threads(cpu_threads)


class SenderContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(105)
        self.config, self.args = fixture()
        self.inputs = sender.build_inputs(*self.args, config=self.config)
        self.model = sender.SenderUtilityRouter(self.config)

    def test_full_window_full_view_and_no_input_mutation(self):
        self.assertEqual(self.inputs['global_video'].shape, (1, 18, 17, 32, 32))
        self.assertEqual(self.inputs['local_video'].shape, (1, 16, 21, 3, 16, 16))
        self.assertEqual(self.inputs['packet_bytes'].dtype, torch.int64)
        np.testing.assert_array_equal(self.args[1], self.args[2])
        original = [v.copy() if isinstance(v, np.ndarray) else v for v in self.args]
        self.assertEqual(self.model(self.inputs).shape, (1, 16))
        for a, b in zip(self.args, original):
            np.testing.assert_array_equal(a, b)
        meta = self.model.metadata()
        self.assertEqual(meta['global_frame_indices'], list(range(17)))
        self.assertEqual(meta['detail_frame_indices'], [0, 8, 16])
        self.assertFalse(meta['diffusion_executed'])
        self.assertFalse(meta['mask_transmitted'])
        self.assertFalse(meta['receiver_weights_shared'])

    def test_non_detail_frame_source_information_reaches_prediction(self):
        modified = [v.copy() if isinstance(v, np.ndarray) else v for v in self.args]
        modified[0][7] = 255-modified[0][7]  # Frame7 is absent from sampled local detail.
        second = sender.build_inputs(*modified, config=self.config)
        torch.testing.assert_close(self.inputs['local_video'], second['local_video'], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.inputs['global_video'], second['global_video']))
        self.assertFalse(torch.equal(self.model(self.inputs), self.model(second)))

    def test_temporal_order_changes_3D_features_and_predictions(self):
        reversed_ = [v[::-1].copy() if i < 4 else v for i, v in enumerate(self.args)]
        second = sender.build_inputs(*reversed_, config=self.config)
        self.assertFalse(torch.equal(self.model(self.inputs), self.model(second)))
        # The same set of pixels is present; order, not extra image information, changed.
        torch.testing.assert_close(self.inputs['global_video'].flip(2), second['global_video'], rtol=0, atol=0)

    def test_actual_cost_coverage_candidate_and_G_budget_are_used(self):
        baseline = self.model(self.inputs)
        variants = []
        for index, value in ((4, np.eye(16, dtype=np.float32)[5]),
                             (5, np.arange(16, 0, -1, dtype=np.int64)*800), (6, 0)):
            args = list(self.args)
            args[index] = value
            variants.append(sender.build_inputs(*args, config=self.config))
        args = list(self.args)
        args[3] = np.full_like(args[3], 200)
        variants.append(sender.build_inputs(*args, config=self.config))
        for value in variants:
            self.assertFalse(torch.equal(baseline, self.model(value)))

    def test_no_source_ablation_has_same_weights_and_no_residual_leak(self):
        config = replace(self.config, zero_source=True)
        model = sender.SenderUtilityRouter(config)
        model.load_state_dict(self.model.state_dict())
        first = sender.build_inputs(*self.args, config=config)
        changed = list(self.args)
        changed[0] = 255-self.args[0]
        second = sender.build_inputs(*changed, config=config)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        torch.testing.assert_close(model(first), model(second), rtol=0, atol=0)
        self.assertEqual(model.metadata()['parameters'], self.model.metadata()['parameters'])
        self.assertFalse(model.metadata()['source_used'])
        first['global_video'][0, 12, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'leaked'):
            model(first)

    def test_letterbox_padding_not_mistaken_for_region_coordinates(self):
        # 64x96 ->32x21, padding top5 bottom6. Extent is actual picture support.
        torch.testing.assert_close(self.inputs['global_extent'],
            torch.tensor([[0., 5/32, 1., 21/32]]), rtol=0, atol=0)
        tall = [v.transpose(0, 2, 1, 3).copy() if i < 4 else v for i, v in enumerate(self.args)]
        inp = sender.build_inputs(*tall, config=self.config)
        torch.testing.assert_close(inp['global_extent'], torch.tensor([[5/32, 0., 21/32, 1.]]), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(self.model(inp)).all())

    def test_fixed_sampling_matches_original_grid_forward_and_CPU_gradient(self):
        # Read-only CPU reference to the removed op; no production grid_sample.
        for h, w in ((12, 12), (6, 12), (1, 3)):
            original = torch.randn(3, 2, 3, h, w, dtype=torch.float64, requires_grad=True)
            fixed = original.detach().clone().requires_grad_(True)
            extent = torch.tensor([[0., 0., 1., 1.], [0., 5/32, 1., 21/32],
                                   [5/32, 0., 21/32, 1.]], dtype=torch.float64)
            centers = (torch.arange(8, dtype=torch.float64)+.5)/8
            gx = extent[:, 0, None]+extent[:, 2, None]*centers
            gy = extent[:, 1, None]+extent[:, 3, None]*centers
            grid = torch.stack((gx[:, None, :].expand(-1, 8, -1), gy[:, :, None].expand(-1, -1, 8)), -1)*2-1
            grid = grid[:, None].expand(-1, 3, -1, -1, -1).reshape(-1, 8, 8, 2)
            reference = F.grid_sample(original.permute(0, 2, 1, 3, 4).reshape(-1, 2, h, w),
                grid, mode='bilinear', padding_mode='border', align_corners=False)
            reference = reference.reshape(3, 3, 2, 8, 8).permute(0, 2, 1, 3, 4)
            actual = sender._aligned_picture(fixed, extent)
            torch.testing.assert_close(actual, reference, rtol=0, atol=1e-12)
            upstream = torch.randn_like(actual)
            (actual*upstream).sum().backward()
            (reference*upstream).sum().backward()
            torch.testing.assert_close(fixed.grad, original.grad, rtol=0, atol=1e-12)

    def test_fixed_temporal_bins_match_original_adaptive_average(self):
        for frames, bins in ((5, 3), (3, 3), (9, 3), (5, 7)):
            values = torch.randn(2, 3, frames, 4, 6, dtype=torch.float64)
            expected = F.adaptive_avg_pool3d(values, (bins, 4, 6))
            actual = sender._temporal_average(values, bins)
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

    def test_sender_training_avoids_documented_nondeterministic_sampling_ops(self):
        with patch.object(F, 'grid_sample', side_effect=AssertionError('removed CUDA-backward risk')), \
                patch.object(F, 'adaptive_avg_pool3d', side_effect=AssertionError('removed CUDA-backward risk')), \
                patch.object(F, 'interpolate', side_effect=AssertionError('no learned CUDA resize')):
            prediction = self.model(self.inputs)
            prediction.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_sender_and_receiver_are_distinct_architectures_and_parameters(self):
        from demo import routervc_receiver_router as receiver
        rg = receiver.ReceiverGUtilityRouter()
        self.assertTrue(any(isinstance(m, torch.nn.Conv3d) for m in self.model.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.Conv3d) for m in rg.modules()))
        self.assertFalse({p.data_ptr() for p in self.model.parameters()} & {p.data_ptr() for p in rg.parameters()})
        self.assertNotEqual(self.model.metadata()['format'], rg.metadata()['format'])
        self.assertEqual(set(sender.code_identity()), {'routervc_sender_router.py'})
        self.assertEqual(self.model.final_gain[-1].out_features, 1)
        self.assertEqual(rg.output.out_features, 3)

    def test_invalid_input_contracts_fail_closed(self):
        for index, value in ((0, self.args[0][:3]), (0, self.args[0].astype(float)),
                             (4, np.full(16, np.nan)), (4, np.zeros(15)),
                             (5, np.zeros(16, np.int64)), (5, np.ones(16)), (6, True), (6, 17)):
            args = list(self.args)
            args[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                sender.build_inputs(*args, config=self.config)
        with self.assertRaises(ValueError):
            sender.Config(frame_count=16)
        with self.assertRaises(ValueError):
            self.model({**self.inputs, 'teacher_gain': torch.zeros(16)})
        bad = dict(self.inputs)
        bad['global_video'] = bad['global_video'].clone()
        bad['global_video'][0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.model(bad)

    def test_signed_final_gain_and_explicit_final_label_contract(self):
        with torch.no_grad():
            self.model.final_gain[-1].weight.zero_()
            self.model.final_gain[-1].bias.fill_(-.25)
        output = self.model(self.inputs)
        torch.testing.assert_close(output, torch.full_like(output, -.25), rtol=0, atol=0)
        loss, detail = sender.training_loss(output, targets(), self.inputs['packet_bytes'], gain_scale=.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(detail['huber'].detach()), 0.)
        wrong = {**targets(), 'label_scope': 'direct_E_only_lpips'}
        with self.assertRaisesRegex(ValueError, 'FINAL'):
            sender.training_loss(output, wrong, self.inputs['packet_bytes'])

    def test_unknown_labels_are_not_fabricated_and_receive_no_gradient(self):
        prediction = torch.zeros(1, 16, requires_grad=True)
        t = targets()
        t['weight'][0, 2:] = 0
        t['value'][0, 2:] = float('nan')
        loss, _ = sender.training_loss(prediction, t, self.inputs['packet_bytes'])
        loss.backward()
        torch.testing.assert_close(prediction.grad[0, 2:], torch.zeros(14), rtol=0, atol=0)
        t['weight'].zero_()
        loss, detail = sender.training_loss(prediction, t, self.inputs['packet_bytes'])
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(detail['ranking'].detach()), 0.)

    def test_cost_weighted_ranking_uses_gain_per_byte_not_raw_gain(self):
        costs = torch.ones(1, 16, dtype=torch.int64)*100
        costs[0, 0] = 1000
        truth = targets()
        truth['weight'].zero_()
        truth['weight'][0, :2] = 1
        truth['value'][0, :2] = torch.tensor([.2, .1])  # region1 wins per byte.
        good = torch.zeros(1, 16)
        good[0, :2] = torch.tensor([.2, .1])
        wrong = torch.zeros(1, 16)
        wrong[0, :2] = torch.tensor([2., .1])
        _, gd = sender.training_loss(good, truth, costs, gain_scale=.1)
        _, bd = sender.training_loss(wrong, truth, costs, gain_scale=.1)
        self.assertLess(float(gd['ranking']), float(bd['ranking']))
        # Confidence weight0 means no pair contributes, not an implicit safe label.
        truth['weight'][0, 1] = 0
        _, detail = sender.training_loss(wrong, truth, costs)
        self.assertEqual(float(detail['ranking']), 0.)

    def test_deterministic_positive_ranking_and_true_prefix_budget(self):
        gains = np.zeros(16)
        gains[:5] = [.2, .1, .1, -.1, .4]
        costs = np.ones(16, np.int64)*100
        costs[0] = 200
        coverage = np.zeros(16)
        coverage[4] = 1
        order = sender.rank_candidates(gains, costs, coverage)
        self.assertEqual(order, [0, 1, 2])
        self.assertEqual(sender.prefix_under_budget(order, costs, 150)['indices'], [])
        self.assertEqual(sender.prefix_under_budget(order, costs, 250)['indices'], [0])
        self.assertEqual(sender.prefix_under_budget(order, costs, 350)['indices'], [0, 1])
        self.assertEqual(sender.prefix_under_budget(order, costs, 500)['packet_bytes'], 400)
        self.assertEqual(sender.rank_candidates(-np.ones(16), costs, coverage), [])

    def test_checkpoint_roundtrip_rng_no_receiver_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sender.pt'
            payload = sender.export_payload(self.model, {'stage': 'CPU fixture, not trained'}, 0, {})
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rng = torch.random.get_rng_state().clone()
            loaded, got = sender.load_model(path, expected_sha256=digest)
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            self.assertFalse(loaded.training)
            tree_equal(self, loaded.state_dict(), self.model.state_dict())
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('no Rg or source file IO')):
                result = sender.predict(loaded, *self.args)
            torch.testing.assert_close(result, self.model(self.inputs), rtol=0, atol=0)
            self.assertEqual(got['step'], 0)

    def test_checkpoint_rejects_receiver_tampering_and_nonfinite_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sender.pt'
            payload = sender.export_payload(self.model, {}, 0, {})
            mutations = [lambda p: p.update(format='routervc_receiver_g_only_v1'),
                lambda p: p.update(sender_mask=[]), lambda p: p['architecture'].update(zero_source='yes'),
                lambda p: p.update(code={}), lambda p: p['metadata'].update(receiver_weights_shared=True),
                lambda p: p['state_dict'].update({'final_gain.2.bias': torch.tensor([float('nan')])}),
                lambda p: p['state_dict'].update({'final_gain.2.bias': torch.zeros(1, dtype=torch.float64)})]
            for fn in mutations:
                bad = deepcopy(payload)
                fn(bad)
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    sender.load_model(path)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                sender.load_model(path, expected_sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'authenticated'):
                sender.predict(self.model, *self.args)

    def test_independent_sender_optimizer_resume_exact(self):
        """Six CPU steps vs three + serialized model/optimizer + three, no Rg."""
        initial = deepcopy(self.model.state_dict())
        def fresh():
            model = sender.SenderUtilityRouter(self.config)
            model.load_state_dict(initial)
            return model, torch.optim.AdamW(model.parameters(), lr=1e-3)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = sender.training_loss(model(self.inputs), targets(), self.inputs['packet_bytes'], gain_scale=.1)
            loss.backward()
            optimizer.step()
        continuous, co = fresh()
        for _ in range(6):
            update(continuous, co)
        interrupted, io = fresh()
        for _ in range(3):
            update(interrupted, io)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resume.pt'
            torch.save(dict(sender=sender.export_payload(interrupted, {'labels': sender.LABEL_SCOPE}, 3, {}),
                            optimizer=io.state_dict(), cursor=3, random_state=torch.random.get_rng_state()), path)
            resume = torch.load(path, weights_only=True, map_location='cpu')
            resumed, ro = fresh()
            resumed.load_state_dict(resume['sender']['state_dict'])
            ro.load_state_dict(resume['optimizer'])
            torch.random.set_rng_state(resume['random_state'])
            for _ in range(resume['cursor'], 6):
                update(resumed, ro)
        tree_equal(self, continuous.state_dict(), resumed.state_dict())
        tree_equal(self, co.state_dict(), ro.state_dict())
        self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in continuous.state_dict().items()))

    def test_batch_contract_and_finite_gradient_all_branches(self):
        batch = {k: torch.cat((v, v), 0) for k, v in self.inputs.items()}
        prediction = self.model(batch)
        self.assertEqual(prediction.shape, (2, 16))
        target = targets()
        target = dict(value=target['value'].repeat(2, 1), weight=target['weight'].repeat(2, 1),
                      label_scope=sender.LABEL_SCOPE)
        loss, _ = sender.training_loss(prediction, target, batch['packet_bytes'], gain_scale=.1)
        loss.backward()
        for branch in (self.model.video_encoder, self.model.detail_encoder,
                       self.model.grid_context, self.model.final_gain):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch.parameters()), 0)


if __name__ == '__main__':
    unittest.main()
