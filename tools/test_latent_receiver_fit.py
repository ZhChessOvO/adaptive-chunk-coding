from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from demo import routervc_receiver_router as receiver
from demo.test_routervc_receiver_train import tiny_config, fixture, exact
from routervc.latent.receiver_fit import fit, loss_scales
from routervc.latent.router_data import selection, COUNTS


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_train_only_scales(self):
        rows = [dict(router_split='train'), dict(router_split='validation')]
        def get(row):
            self.assertEqual(row['router_split'], 'train')
            return dict(targets=dict(value=torch.ones(6,16,3)*2, weight=torch.ones(6,16,3)))
        torch.testing.assert_close(loss_scales(rows,get),torch.ones(3)*2)

    def test_nested_independent_sample_orders(self):
        states = [selection('one', n) for n in COUNTS]
        for a,b in zip(states,states[1:]): self.assertEqual(a,b[:len(a)])
        self.assertNotEqual(states[-1],selection('two',16))

    def test_exact_resume_and_real_update(self):
        config = tiny_config()
        torch.manual_seed(3)
        initial = receiver.ReceiverGUtilityRouter(config)
        before = {k:v.clone() for k,v in initial.state_dict().items()}
        batch = fixture(config)
        rows = [dict(router_split='train', dataset=d, sample_id=d) for d in ('REDS','UVG')]
        protocol = dict(rows=rows, smoke=True, epochs=2, initial_path='fixture', initial_sha256='a'*64,
                        seed=1008, learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(receiver, 'load_model', return_value=(initial,{})):
            a, b = Path(tmp)/'resume', Path(tmp)/'direct'
            fit(a, protocol, lambda row:batch, device='cpu', stop_after=1)
            fit(a, protocol, lambda row:batch, device='cpu')
            fit(b, protocol, lambda row:batch, device='cpu')
            left = torch.load(a/'resume.pt', weights_only=True)
            right = torch.load(b/'resume.pt', weights_only=True)
            exact(left,right)
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in left['model'].items()))
            self.assertEqual(left['state']['updates'],4)


if __name__ == '__main__': unittest.main()
