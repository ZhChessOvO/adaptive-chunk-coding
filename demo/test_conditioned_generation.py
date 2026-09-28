import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from demo.conditioned_generation_train import image_terms, differentiable_decode, restore_step_log


class MSEMetric(torch.nn.Module):
    def forward(self, a, b):
        return (a-b).square().mean((1,2,3))


class ConditionedGenerationTests(unittest.TestCase):
    def test_digest_keeps_domains_prefixes_and_metric_scopes_separate(self):
        from demo.conditioned_generation_digest import aggregate, CLIPS, MODES, METRICS
        rows = []
        for i, sid in enumerate(CLIPS):
            for j, mode in enumerate(MODES):
                def point(value):
                    return {field: {k: value + offset for k in METRICS}
                            for field, offset in (('roi_quality', 0), ('quality', 100))}
                for candidate, offset in (('latent', 10), ('image', 20)):
                    rows.append(dict(sample_id=sid, mode=mode, candidate=candidate,
                                     **point(i+j*10+offset),
                                     baseline=point(i+j*10), direct=point(i+j*10+30)))
        local = aggregate(rows, 'roi_quality')
        whole = aggregate(rows, 'quality')
        self.assertEqual(local['full']['all']['image']['lpips_alex'], 41.5)
        self.assertEqual(local['none']['REDS']['baseline']['lpips_alex'], 1.)
        self.assertEqual(local['partial']['UVG']['direct']['lpips_alex'], 42.)
        self.assertEqual(whole['full']['all']['image']['lpips_alex'], 141.5)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1], 'quality')
        with self.assertRaises(ValueError):
            aggregate(rows+[rows[-1]], 'quality')

    def test_power_loss_log_tail_and_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'steps.jsonl'
            path.write_text('{"step":1}\n{"step":2}\n{"step":3')
            self.assertEqual(restore_step_log(path,1),[{'step':1}])
            self.assertEqual(path.read_text(),'{"step": 1}\n')
            path.write_text('{"step":1}\n{"step":2')
            with self.assertRaises(RuntimeError):
                restore_step_log(path,2)
            path.write_text('{"step":1}\ncorrupt\n')
            with self.assertRaises(json.JSONDecodeError):
                restore_step_log(path,1)

    def test_optional_adapter_not_needed_for_fallback(self):
        from demo import conditioned_generation_decode as decoder
        from demo.test_scalable_cooperation import CooperationTests
        from demo.scalable_format import parse, frame_hash
        from demo import scalable_cooperation_format as fmt
        inner, control = CooperationTests().fixture()
        parsed = parse(inner)
        base = np.full((33,128,128,3),20,np.uint8)
        enhanced = base.copy(); enhanced[1:9,64:,64:] = 40
        report = dict(source_frames_read=False,applied_packets=[1],
            base_bytes=len(parsed.base),container_header_bytes=parsed.base_end-len(parsed.base),
            packet_bytes=sum(len(p.wire) for p in parsed.packets),incomplete_tail_bytes=0,
            total_bytes=len(inner),base_hash=frame_hash(base),output_hash=frame_hash(enhanced),
            non_enhanced_exact=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for disabled,blend in ((True,1.),(False,0.)):
                stream = root/f'{disabled}-{blend}.acsg'
                stream.write_bytes(fmt.wrap(inner,dict(control,blend=blend)))
                args = SimpleNamespace(stream=stream,output=root/stream.stem,
                    disable_generation=disabled,adapter=Path('/missing/adapter.pt'))
                with patch.multiple(decoder,configure_torch=lambda:None,
                    load_model=lambda _:object(),BaseCodec=lambda *_:object(),
                    decode_enhancement=lambda *a,**kw:(enhanced,dict(report),base)), \
                    patch.object(decoder,'identities',side_effect=FileNotFoundError('missing')) as assets, \
                    patch.object(decoder,'restore',side_effect=AssertionError('G executed')) as restore, \
                    patch.object(decoder.torch.cuda,'max_memory_allocated',return_value=0), \
                    patch.object(decoder.torch.cuda,'empty_cache'), \
                    patch.object(decoder.torch.distributed,'is_initialized',return_value=False):
                    decoder.decode(args)
                assets.assert_not_called(); restore.assert_not_called()
                self.assertFalse(json.loads((args.output/'decode.json').read_text())['generation_executed'])
                with np.load(args.output/'reconstruction.npz') as value:
                    np.testing.assert_array_equal(value['reconstruction'],enhanced)

    def test_image_loss_identity_and_gradients(self):
        target = torch.zeros(17,3,64,64)
        pred = torch.rand_like(target).requires_grad_()
        terms = image_terms(pred,target,MSEMetric())
        sum(terms.values()).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.norm()),0)
        self.assertTrue(all(float(v) == 0 for v in image_terms(target,target,MSEMetric()).values()))

    def test_latent_crop_border_is_excluded(self):
        target = torch.zeros(17,3,64,64)
        pred = target.clone()
        pred[...,:16,:] = 10
        self.assertTrue(all(float(v) == 0 for v in image_terms(pred,target,MSEMetric()).values()))

    def test_static_bias_not_temporal_error(self):
        target = torch.zeros(17,3,64,64)
        result = image_terms(target+.2,target,MSEMetric())
        self.assertEqual(float(result['rgb_temporal']),0)
        self.assertGreater(float(result['low_frequency']),0)

    def test_differentiable_layout_and_scaling(self):
        class VAE:
            def decode(self, z):
                return SimpleNamespace(sample=z)
        runner = SimpleNamespace(vae=VAE(),config=SimpleNamespace(vae={'shifting_factor':0}))
        class Config(dict):
            scaling_factor = 2.
        runner.config.vae = Config()
        latent = torch.ones(5,8,8,3,requires_grad=True)
        output = differentiable_decode(runner,latent)
        self.assertEqual(tuple(output.shape),(5,3,8,8))
        self.assertTrue(torch.all(output == .5))
        output.sum().backward()
        self.assertTrue(torch.all(latent.grad == .5))


if __name__ == '__main__':
    unittest.main()
