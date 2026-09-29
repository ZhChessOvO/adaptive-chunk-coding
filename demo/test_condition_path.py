import unittest
from copy import deepcopy
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import torch

from demo.condition_path_decode import (FORMAT, observe_runner, posterior_condition,
    tensor_identity, validate_condition)
from demo.condition_path_experiment import assert_pair


class FakeRunner:
    """Mimics upstream encode always sampling before choosing mode or sample."""
    def __init__(self):
        self.config = SimpleNamespace(vae=SimpleNamespace(use_sample=True))
        self.vae = SimpleNamespace(encode=self.encode)

    def encode(self, center):
        output = namedtuple('Output', ['latent','posterior'])
        sampled = center + .1 * torch.randn_like(center)
        return output(sampled, SimpleNamespace(mode=lambda:center))

    def vae_encode(self, samples):
        result = self.vae.encode(samples[0])
        return [result.latent if self.config.vae.use_sample else result.posterior.mode()]

    def get_condition(self, latent, latent_blur, task):
        return latent_blur


class ConditionTests(unittest.TestCase):
    def run_path(self, mode):
        runner, records = FakeRunner(), []
        observe_runner(runner, mode, records)
        torch.manual_seed(261001)
        center = torch.ones(5, 2, 2, 16)
        latent = runner.vae_encode([center])[0]
        noise = torch.randn_like(latent)
        out = runner.get_condition(noise, latent, 'sr')
        return out, noise, records[0]

    def test_noise_and_rng_pair_exactly(self):
        sample, noise, a = self.run_path('sample')
        mean, other, b = self.run_path('mean')
        self.assertFalse(torch.equal(sample, mean))
        self.assertTrue(torch.equal(mean, torch.ones_like(mean)))
        torch.testing.assert_close(noise, other, rtol=0, atol=0)
        for k in ('before_vae', 'after_vae', 'before_diffusion', 'diffusion_noise'):
            self.assertEqual(a[k], b[k])

    def test_sample_path_is_unchanged(self):
        sample, noise, _ = self.run_path('sample')
        torch.manual_seed(261001)
        expected = torch.ones_like(sample) + .1 * torch.randn_like(sample)
        expected_noise = torch.randn_like(sample)
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)

    def test_hash_supports_bf16_and_checks_dtype_and_shape(self):
        value = torch.ones(2, 3, dtype=torch.bfloat16)
        self.assertEqual(tensor_identity(value), tensor_identity(value.clone()))
        self.assertNotEqual(tensor_identity(value), tensor_identity(value.float()))
        self.assertNotEqual(tensor_identity(value), tensor_identity(value.reshape(3, 2)))

    def test_bf16_mean_preserves_fp32_sample_layout_and_rng(self):
        center = torch.randn(1,16,5,2,3).bfloat16()
        latent = torch.randn_like(center.float())
        output = namedtuple('Output', ['latent','posterior'])(latent,SimpleNamespace(mode=lambda:center))
        before = torch.get_rng_state()
        mean = posterior_condition(output,'mean')
        self.assertEqual(mean.latent.dtype,latent.dtype)
        self.assertEqual(mean.latent.stride(),latent.stride())
        torch.testing.assert_close(mean.latent,center.float(),rtol=0,atol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertIs(posterior_condition(output,'sample'),output)

    def test_no_implicit_mode_default(self):
        with patch('demo.condition_path_decode.validate_bundle'):
            for bundle in ({}, {'vae_condition':'mean'},
                           {'vae_condition':'invalid', 'condition_path_format':FORMAT}):
                with self.assertRaises(ValueError): validate_condition(bundle)
            self.assertEqual(validate_condition(dict(vae_condition='mean', condition_path_format=FORMAT)), 'mean')

    def test_repeated_run_exact(self):
        a, _, first = self.run_path('mean')
        b, _, second = self.run_path('mean')
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(first, second)

    def test_pair_rejects_noise_and_rng_mismatch(self):
        _, _, a = self.run_path('sample')
        _, _, b = self.run_path('mean')
        def result(record):
            return dict(bytes=100, fresh_decode=dict(generation_input_hash='same',
                generation_runtime=dict(condition_windows=[record])))
        assert_pair(result(a), result(b))
        for key in ('before_vae','after_vae','before_diffusion','diffusion_noise'):
            broken = deepcopy(b); broken[key] = 'changed'
            with self.assertRaises(AssertionError): assert_pair(result(a), result(broken))
        broken = deepcopy(b); broken['conditions'][0]['dtype'] = 'torch.bfloat16'
        with self.assertRaises(AssertionError): assert_pair(result(a), result(broken))


if __name__ == '__main__':
    unittest.main()
