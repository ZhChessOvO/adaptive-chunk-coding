"""CPU contracts for actual trajectories, fresh final labels and warm restarts."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import routervc_sender_router as network
from demo.routervc_fullview_probe import save, digest
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_format import frame_hash
from demo.test_routervc_sender_train import fixture, tiny_config, exact
from tools.test_latent_routing import bank
from routervc.latent import routing
from routervc.cooperation import sender_data as data, sender_training as fit, stream
from tools.cooperative_sender_decode import deny_sender_checkpoint


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_cap_assignment_and_actual_prefix(self):
        rows = [dict(sample_id=f'{d}-{s}-{i}', dataset=d, router_split=s)
                for d in ('REDS', 'UVG') for s in ('train', 'validation') for i in range(5)]
        assigned = data.assign_caps(list(reversed(rows)))
        self.assertEqual(assigned, data.assign_caps(rows))
        for d in ('REDS', 'UVG'):
            for s in ('train', 'validation'):
                self.assertEqual([r['sender_parent_cap'] for r in assigned if r['dataset']==d and r['router_split']==s],
                                 [.25, .5, .25, .5, .25])
        costs = [100]*16; order = dict(order=list(range(16)), bundle_bytes=costs)
        self.assertEqual(data.trajectory_parent(order, .25, costs), [0,1,2,3])
        self.assertEqual(data.trajectory_parent(order, .5, costs), list(range(8)))
        with self.assertRaises(ValueError): data.trajectory_parent(order, 1., costs)
        with self.assertRaises(ValueError): data.trajectory_parent(order, .25, [101]*16)

    def test_candidates_use_actual_received_state_and_do_not_drop_negative_proposals(self):
        source = np.zeros((17, 64, 64, 3), np.uint8); source[:, 16:32, 16:32] = 80
        received = np.zeros_like(source); gains = [-i-1. for i in range(16)]
        order = dict(steps=[dict(selected=[], received_hash=frame_hash(received), gains=gains, ranked=[])])
        p = data.proposals([], order, source, received, [100]*16)
        self.assertEqual(p['candidates'][:2], [0,5]); self.assertTrue(p['sender_would_stop'])
        self.assertEqual(len(set(p['candidates'])), 3)
        self.assertEqual(p, data.proposals([], order, source, received, [100]*16))
        with self.assertRaises(ValueError): data.proposals([0], order, source, received, [100]*16)
        with self.assertRaises(ValueError): data.proposals([], order, source, received+1, [100]*16)

    def test_new_final_teacher_recomputes_G_for_child_and_reuses_only_exact_state(self):
        received_calls = []; base = np.zeros((17, 256, 256, 3), np.uint8)
        def decode(inner):
            n = int(routing.coverage(routing.packets.parse(inner)).sum())
            return base.copy(), base+n, dict(base_reference_hashes=['base-reference'])
        def generate(y, control):
            received_calls.append(frame_hash(y))
            return y.copy(), y.copy(), dict(core=[0,0,32,32], crop=[0,0,32,32])
        def score(source, output, metric):
            return dict(lpips_alex=float(output.mean()), psnr_db=30., temporal_delta_mae=0.)
        teacher = dict(receiver=dict(path='fixture.pt', sha256='ab'*32),
                       G_assets={}, G_assets_hash='cd'*32, max_g=8, seed=routing.SEED)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol = dict(teacher=teacher)
            save(root/'protocol.json', protocol)
            samples = data.Samples(root, root/'cache', protocol)
            samples.model = object(); samples.generator = generate; samples.metric = object()
            encoded = dict(base_hash=frame_hash(base), base_reference_hashes=['base-reference'])
            with patch.object(data.router_data, 'decode', side_effect=decode), \
                    patch.object(data.receiver, 'select', return_value=dict(indices=[0], mask_bytes=0)), \
                    patch('routervc.cooperation.render.render', side_effect=lambda y,p,i:y+len(p)), \
                    patch('demo.scalable_experiment.quality', side_effect=score):
                parent = samples._measure(root/'parent', bank(), [], base, encoded)
                child = samples._measure(root/'child', bank(), [1], base, encoded)
                self.assertEqual(len(received_calls), 2)
                self.assertNotEqual(received_calls[0], received_calls[1])
                self.assertLess(parent['quality']['lpips_alex']-child['quality']['lpips_alex'], 0)
                self.assertEqual(child['total_bytes']-parent['total_bytes'], routing.bundle_bytes(bank())[1])
                replay = samples._measure(root/'child', bank(), [1], base, encoded)
                self.assertEqual(child, replay); self.assertEqual(len(received_calls), 2)
                with self.assertRaises(ValueError): samples._measure(root/'child', bank(), [2], base, encoded)
                samples.stop_after_renderings = 3
                with self.assertRaises(InterruptedError): samples._measure(root/'stop', bank(), [3], base, encoded)
                self.assertTrue((root/'stop/result.json').exists())

    def test_cooperative_bytes_are_literal_prefixes_without_sender_identity(self):
        low, high = [stream.wrap(routing.subset(bank(), s), 'ab'*32, 'cd'*32) for s in ([], [2,7])]
        self.assertTrue(high.startswith(low))
        inner, config = stream.parse(high)
        self.assertEqual(len(high)-len(inner), 121)
        self.assertEqual(len(high)-len(low), sum(routing.bundle_bytes(bank())[i] for i in (2,7)))
        self.assertNotIn('sender', str(config)); self.assertNotIn('mask', str(config))

    def test_fresh_receiver_rejects_both_historical_and_new_sender_checkpoints(self):
        for name in ('routervc_sender_20261005', 'routervc_latent_sender_20261009',
                     'routervc_cooperative_sender_20261011'):
            with self.assertRaises(RuntimeError):
                deny_sender_checkpoint('open', (f'/root/autodl-fs/DCVC/runs/{name}/source/best.pt', 'r', 0))
        deny_sender_checkpoint('open', ('/root/autodl-fs/DCVC/runs/routervc_cooperation_20261010/router/best.pt', 'r', 0))

    def test_trained_warm_start_exact_restart_and_new_label_contract(self):
        batch = fixture(); config = tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); torch.manual_seed(41)
            initial = network.SenderUtilityRouter(config)
            optimizer = torch.optim.AdamW(initial.parameters(), lr=1e-4)
            loss, _ = network.training_loss(initial(batch['inputs']), batch['targets'], batch['packet_bytes'])
            loss.backward(); optimizer.step()
            path = root/'trained_source.pt'
            atomic_torch(path, network.export_payload(initial, dict(arm='source'), 1, dict(epoch=1)))
            protocol = dict(format=fit.FORMAT, smoke=True,
                rows=[dict(sample_id='reds', dataset='REDS', router_split='train'),
                      dict(sample_id='uvg', dataset='UVG', router_split='train')],
                teacher=dict(sender_config=asdict(config)), arms=list(fit.ARMS), epochs=2, seed=19,
                learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1,
                initialization=dict(path=str(path), sha256=digest(path),
                    kind='same_trained_source_sender_for_both_arms', optimizer_reset=True))
            labels = dict(format=fit.LABEL_FORMAT, complete=True, protocol_sha256=fit.semantic_hash(protocol),
                samples={r['sample_id']:'aa'*32 for r in protocol['rows']}, windows=2,
                measured_final_renderings=16, measured_marginals=12, label_scope=network.LABEL_SCOPE,
                current_sender_policy_used=True, no_UF_E_G_Rg_F_updates=True)
            get = lambda _:batch
            scale = fit.fit_scale(protocol, get)
            fit.run_training(root/'resumed', protocol, get, scale_record=scale, labels_binding=labels,
                             device='cpu', stop_after=1)
            fit.run_training(root/'resumed', protocol, get, scale_record=scale, labels_binding=labels, device='cpu')
            fit.run_training(root/'direct', protocol, get, scale_record=scale, labels_binding=labels, device='cpu')
            resumed = torch.load(root/'resumed/resume.pt', map_location='cpu', weights_only=True)
            direct = torch.load(root/'direct/resume.pt', map_location='cpu', weights_only=True)
            exact(resumed, direct)
            self.assertEqual(resumed['state']['updates'], 4)
            for arm in fit.ARMS:
                self.assertTrue(any(not torch.equal(v, initial.state_dict()[k]) for k,v in resumed['models'][arm].items()))
            changed = dict(labels, current_sender_policy_used=False)
            with self.assertRaises(ValueError):
                fit.run_training(root/'bad', protocol, get, scale_record=scale, labels_binding=changed, device='cpu')
            bad_scale = deepcopy(scale); bad_scale['value'] *= 2
            with self.assertRaises(ValueError):
                fit.run_training(root/'bad_scale', protocol, get, scale_record=bad_scale, labels_binding=labels, device='cpu')


if __name__ == '__main__': unittest.main()
