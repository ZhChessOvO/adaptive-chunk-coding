"""CPU-only paired R_g training and exact-restart contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import routervc_receiver_router as receiver
from demo import routervc_receiver_train as train


def tiny_config():
    return receiver.Config(channels=4, hidden=8, local_size=16, global_size=16)


def fixture(config=None):
    config = config or tiny_config()
    rng = torch.Generator().manual_seed(31)
    inputs = dict(global_pairs=torch.rand(6, config.temporal_frames, 6,
                                         config.global_size, config.global_size, generator=rng),
        local_pairs=torch.rand(6, 16, config.temporal_frames, 6,
                               config.local_size, config.local_size, generator=rng),
        geometry=torch.rand(6, 16, 6, generator=rng), coverage=torch.zeros(6, 16, 1))
    for index, count in enumerate((0, 2, 4, 8, 12, 16)):
        inputs['coverage'][index, :count] = 1.
    target = torch.randn(6, 16, 3, generator=rng) * .2
    return dict(inputs=inputs, halo_pairs=inputs['local_pairs'].flip(-1),
        targets=dict(value=target, weight=torch.ones_like(target)),
        view_names=list(train.VIEW_NAMES), dataset='REDS', sample_id='fixture')


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            exact(left, right)
    else:
        assert a == b


class ZeroModel:
    def eval(self):
        return self

    def __call__(self, inputs):
        return inputs['coverage'].new_zeros((6, 16, 3))


class ReceiverTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The public receiver loader also uses four CPU threads.
        torch.set_num_threads(4)

    def test_context_arms_share_all_targets_and_nonlocal_inputs(self):
        data = fixture()
        core = train.arm_batch(data, 'core', 'cpu')
        halo = train.arm_batch(data, 'halo', 'cpu')
        exact(core['targets'], halo['targets'])
        for key in ('global_pairs', 'geometry', 'coverage'):
            exact(core['inputs'][key], halo['inputs'][key])
        self.assertFalse(torch.equal(core['inputs']['local_pairs'], halo['inputs']['local_pairs']))
        self.assertEqual(tuple(core['targets']['value'].shape), (6, 16, 3))

    def test_invalid_views_arms_and_target_contract_rejected(self):
        data = fixture()
        with self.assertRaises(ValueError):
            train.arm_batch(data, 'sender', 'cpu')
        data['view_names'].reverse()
        with self.assertRaises(ValueError):
            train.arm_batch(data, 'core', 'cpu')
        data = fixture()
        data['targets']['value'] = torch.zeros(6, 16, 6)
        with self.assertRaises(ValueError):
            train.arm_batch(data, 'core', 'cpu')

    def test_regret_preserves_negative_g_and_optional_off(self):
        truth = np.array([1., 2.] + [-1.] * 14)
        self.assertEqual(train.local_regret(truth, truth), 0.)
        self.assertEqual(train.local_regret(-np.ones(16), -np.ones(16)), 0.)
        self.assertGreater(train.local_regret(np.ones(16), -np.ones(16)), 0.)
        with self.assertRaises(ValueError):
            train.local_regret(np.full(16, np.nan), truth)

    def test_validation_equal_weights_datasets_then_six_states(self):
        rows = [dict(dataset='REDS') for _ in range(3)] + [dict(dataset='UVG')]
        data = fixture()

        def get(row):
            value = deepcopy(data)
            for index in range(6):
                value['targets']['value'][index, :, 0] = (index+1) * (1 if row['dataset'] == 'REDS' else 3)
            return value

        result = train.validate(ZeroModel(), 'core', rows, get, torch.ones(3), 'cpu')
        self.assertAlmostEqual(result['regret'], 2.625)
        self.assertAlmostEqual(result['by_state']['e0']['regret'], .75)
        self.assertAlmostEqual(result['by_state']['e16']['regret'], 4.5)
        self.assertEqual(result['groups']['REDS']['samples'], 3)
        self.assertEqual(result['groups']['UVG']['samples'], 1)

    def test_validation_refuses_unknown_lpips_in_ranking(self):
        data = fixture()
        data['targets']['weight'][0, 0, 0] = 0
        with self.assertRaises(ValueError):
            train.validate(ZeroModel(), 'core', [dict(dataset='REDS')],
                           lambda _: data, torch.ones(3), 'cpu')

    def test_exact_resume_models_optimizer_initial_and_best(self):
        config = tiny_config()
        data = fixture(config)
        protocol = dict(rows=[dict(sample_id='a', dataset='REDS', router_split='train'),
                              dict(sample_id='b', dataset='UVG', router_split='train')],
                        smoke=True, epochs=2, seed=20261005, learning_rate=1e-4,
                        weight_decay=1e-4, ranking_weight=.1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            initial = receiver.ReceiverGUtilityRouter(config).state_dict()
        kwargs = dict(initial_state=initial, scale=torch.tensor([.1, 1., 1.]),
                      config=config, device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []

            def get(row):
                calls.append(row['sample_id'])
                return data

            train.run_training(root / 'resumed', protocol, get, stop_after=1, **kwargs)
            self.assertEqual(len(calls), 1)  # No expensive initial validation before optimization.
            self.assertFalse((root / 'resumed/complete.json').exists())
            saved = torch.load(root / 'resumed/resume.pt', weights_only=True)
            self.assertEqual(saved['state']['cursor'], 1)
            self.assertIsNone(saved['state']['initial_validation'])
            for arm in receiver.ARMS:
                self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in saved['models'][arm].items()))
                self.assertEqual({int(s['step']) for s in saved['optimizers'][arm]['state'].values()}, {1})
            train.run_training(root / 'resumed', protocol, get, **kwargs)
            train.run_training(root / 'direct', protocol, get, **kwargs)
            a = torch.load(root / 'resumed/resume.pt', weights_only=True)
            b = torch.load(root / 'direct/resume.pt', weights_only=True)
            exact(a, b)
            self.assertEqual(a['state']['epoch'], 2)
            self.assertEqual(a['state']['cursor'], 0)
            self.assertEqual(a['state']['updates'], 4)
            self.assertEqual(len(a['state']['history']), 2)
            self.assertEqual(set(a['state']['initial_validation']), set(receiver.ARMS))
            for arm in receiver.ARMS:
                model, payload = receiver.load_model(root / 'resumed' / arm / 'best.pt')
                self.assertEqual(payload['epoch'], a['state']['best'][arm]['epoch'])
                exact(model.state_dict(), a['state']['best'][arm]['weights'])
                model, payload = receiver.load_model(root / 'resumed' / arm / 'last.pt')
                self.assertEqual(payload['epoch'], 2)
                exact(model.state_dict(), a['models'][arm])
            before = {str(p.relative_to(root/'resumed')): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in (root/'resumed').rglob('*') if p.is_file()}
            result = train.run_training(root/'resumed', protocol,
                lambda _: self.fail('completed replay requested teacher data'), **kwargs)
            after = {str(p.relative_to(root/'resumed')): (p.read_bytes(), p.stat().st_mtime_ns)
                     for p in (root/'resumed').rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse(result['sender_training_complete'])
            self.assertTrue(result['whole_video_RD_pending'])

    def test_changed_binding_and_scale_rejected_on_resume(self):
        config, data = tiny_config(), fixture()
        protocol = dict(rows=[dict(sample_id='a', dataset='REDS', router_split='train')],
                        smoke=True, epochs=2, seed=9, learning_rate=1e-4)
        initial = receiver.ReceiverGUtilityRouter(config).state_dict()
        kwargs = dict(initial_state=initial, scale=torch.ones(3), config=config, device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train.run_training(root, protocol, lambda _: data, stop_after=1, **kwargs)
            changed = dict(protocol, seed=10)
            with self.assertRaises(ValueError):
                train.run_training(root, changed, lambda _: data, **kwargs)
            kwargs['scale'] = torch.ones(3) * 2
            with self.assertRaises(AssertionError):
                train.run_training(root, protocol, lambda _: data, **kwargs)

    def test_final_checkpoint_repairs_stale_or_missing_json_exports(self):
        config, data = tiny_config(), fixture()
        protocol = dict(rows=[dict(sample_id='a', dataset='REDS', router_split='train')],
                        smoke=True, epochs=2, seed=9, learning_rate=1e-4)
        initial = receiver.ReceiverGUtilityRouter(config).state_dict()
        kwargs = dict(initial_state=initial, scale=torch.ones(3), config=config, device='cpu')
        original_save = train.save

        def interrupt_final_history(path, value):
            if Path(path).name == 'history.json' and len(value) == 2:
                raise InterruptedError('simulate crash after final epoch checkpoint')
            return original_save(path, value)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(train, 'save', side_effect=interrupt_final_history):
                with self.assertRaises(InterruptedError):
                    train.run_training(root, protocol, lambda _: data, **kwargs)
            state = torch.load(root/'resume.pt', weights_only=True)['state']
            self.assertEqual(state['epoch'], 2)
            self.assertEqual(state['cursor'], 0)
            self.assertEqual(len(state['history']), 2)
            self.assertEqual(len(train.read(root/'history.json')), 1)
            (root/'initial_validation.json').unlink()
            self.assertFalse((root/'complete.json').exists())
            result = train.run_training(root, protocol,
                lambda _: self.fail('final checkpoint resume must not request teachers'), **kwargs)
            self.assertTrue(result['complete'])
            self.assertEqual(train.read(root/'history.json'), state['history'])
            self.assertEqual(train.read(root/'initial_validation.json'), state['initial_validation'])
            for name, expected in result['artifacts'].items():
                self.assertEqual(train.digest(root/name), expected)


if __name__ == '__main__':
    unittest.main()
