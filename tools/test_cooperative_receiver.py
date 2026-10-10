import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from demo import routervc_receiver_router as old
from demo.test_routervc_receiver_train import tiny_config, fixture, exact
from routervc.cooperation import receiver, training, data, stream
from routervc.cooperation.render import render


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_state_inputs_and_encoder_warmstart(self):
        initial = old.ReceiverGUtilityRouter(tiny_config())
        with patch.object(old, 'load_model', return_value=(initial, {})):
            model = receiver.initialize('fixture', 'a'*64)
        batch = fixture(tiny_config())['inputs']
        selected = torch.zeros(batch['coverage'].shape)
        inputs = dict(batch, selected=selected)
        torch.testing.assert_close(model(inputs), initial(batch)/16, rtol=1e-5, atol=1e-6)
        model.body[0].weight.data[:, -21:] = .1
        a = model(inputs); selected[:, 3, 0] = 1
        self.assertFalse(torch.equal(a, model(inputs)))
        with self.assertRaises(ValueError): model(dict(inputs, source=torch.zeros(1)))

    def test_mask_unknown_and_training_scales(self):
        p = torch.zeros(1, 16, 3, requires_grad=True)
        t = torch.zeros_like(p); w = torch.zeros_like(p)
        t[:, 0] = 2; w[:, 0] = 1; t[:, 1] = float('nan')
        loss, _ = old.training_loss(p, dict(value=t, weight=w)); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(p.grad[:, 1:].abs().sum(), 0)
        row = dict(router_split='train')
        scale = training.scales([row], lambda _:dict(targets=dict(value=t.detach(), weight=w)))
        torch.testing.assert_close(scale, torch.ones(3)*2)
        with self.assertRaises(ValueError): training.scales([dict(router_split='validation')], lambda _:None)

    def test_candidate_exploration_and_local_empty(self):
        selected = dict(indices=list(range(8)), predictions=[[i, 0, 0] for i in range(16)])
        pool, states, extra = data.candidates(dict(sample_id='example'), selected)
        self.assertEqual(len(extra), 2); self.assertTrue(all(i not in selected['indices'] for i in extra))
        self.assertEqual(states[0], []); self.assertTrue(all(set(s) <= set(pool) for s in states))
        pixels = np.zeros((17, 128, 128, 3), np.uint8)
        np.testing.assert_array_equal(render(pixels, [], []), pixels)
        self.assertEqual(stream.HEADER_BYTES, 121)

    def test_stream_prefix_corruption_and_actual_overhead(self):
        from tools.test_latent_routing import bank
        from routervc.latent import routing
        lo, hi = [stream.wrap(routing.subset(bank(), s), 'ab'*32, 'cd'*32)
                  for s in ([], [5, 6])]
        self.assertTrue(hi.startswith(lo))
        inner, config = stream.parse(hi)
        self.assertEqual(len(hi)-len(inner), 121)
        self.assertEqual(config['max_g'], 8)
        for i in (0, 15, 120):
            broken = bytearray(hi); broken[i] ^= 1
            with self.assertRaises(ValueError): stream.parse(bytes(broken))

    def test_local_sequential_policy_reuses_visual_embedding(self):
        model = receiver.Receiver(tiny_config()).eval()
        x = np.zeros((17, 128, 128, 3), np.uint8)
        def conditional(embedding, state):
            result = torch.full((1,16,3), -1.)
            if state.sum() == 0: result[0,2,0] = 3
            elif state[0,2] == 1 and state.sum() == 1: result[0,9,0] = 2
            return result
        with patch.object(model, 'encode', wraps=model.encode) as encoded, \
                patch.object(model, 'from_embedding', side_effect=conditional):
            result = receiver.select(model, x, x, np.zeros(16, np.float32), 8)
        self.assertEqual(result['indices'], [2,9]); self.assertEqual(encoded.call_count, 1)
        self.assertEqual(result['mask_bytes'], 0)

    def test_exact_training_restart(self):
        torch.manual_seed(3); initial = receiver.Receiver(tiny_config())
        b = fixture(tiny_config()); inputs = b['inputs']
        inputs = dict(inputs, selected=torch.zeros_like(inputs['coverage']))
        targets = b['targets']
        rows = [dict(sample_id=d, dataset=d, router_split='train') for d in ('REDS', 'UVG')]
        protocol = dict(rows=rows, smoke=True, epochs=2, initial_path='fixture', initial_sha256='a'*64,
                        seed=1008, learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1)
        def init(*_):
            with torch.random.fork_rng(devices=[]): result = receiver.Receiver(tiny_config())
            result.load_state_dict(initial.state_dict()); return result
        with tempfile.TemporaryDirectory() as tmp, patch.object(receiver, 'initialize', side_effect=init):
            a, b = Path(tmp)/'resume', Path(tmp)/'direct'
            get = lambda _:dict(inputs=inputs, targets=targets)
            scale = torch.ones(3)
            training.fit(a, protocol, get, scale, device='cpu', stop_after=2)
            training.fit(a, protocol, get, scale, device='cpu')
            training.fit(b, protocol, get, scale, device='cpu')
            l = torch.load(a/'resume.pt', weights_only=True); r = torch.load(b/'resume.pt', weights_only=True)
            exact(l, r)
            self.assertEqual(l['state']['updates'], 4)
            self.assertTrue(any(not torch.equal(v, initial.state_dict()[k]) for k,v in l['model'].items()))


if __name__ == '__main__': unittest.main()
