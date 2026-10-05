"""CPU contracts for real sender smoke and recipe-bound queue startup."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from demo.routervc_sender_checks import compare_teacher
from demo.routervc_sender_queue import matching_smoke
from demo.scalable_format import frame_hash


class FreshContracts(unittest.TestCase):
    def fixture(self, generated=False):
        base = np.zeros((17, 64, 64, 3), np.uint8)
        enhanced = base+3
        pixels = dict(base=base, enhanced=enhanced, reconstruction=enhanced.copy())
        expected = dict(total_bytes=1000, header_bytes=310, E_packet_bytes=400,
            generation_input_hash=frame_hash(enhanced), output_hash=frame_hash(enhanced),
            route=dict(indices=[2] if generated else []), generation_executed=generated,
            base_hash=frame_hash(base), generation_runtime={'sentinel': 'runtime'})
        report = dict(source_frames_read=False, sender_router_loaded=False,
            unreceived_candidates_read=False, outside_generate_exact=True, total_bytes=1000,
            generation_control_bytes=310, packet_bytes=400, generation_input_hash=frame_hash(enhanced),
            output_hash=frame_hash(enhanced), route=copy.deepcopy(expected['route']),
            generation_executed=generated, explicit_E_mask_bytes=0, explicit_G_map_bytes=0,
            protection_mask_bytes=0, generation_runtime={'sentinel': 'runtime'})
        return expected, report, pixels

    def test_no_G_exact(self):
        expected, report, pixels = self.fixture()
        compare_teacher(expected, report, pixels, pixels)

    def test_bytes_masks_and_source_must_match(self):
        for key, bad in (('total_bytes', 999), ('generation_control_bytes', 309),
                ('packet_bytes', 399), ('source_frames_read', True), ('explicit_G_map_bytes', 1),
                ('outside_generate_exact', False), ('generation_input_hash', 'wrong')):
            with self.subTest(key=key):
                expected, report, pixels = self.fixture()
                report[key] = bad
                with self.assertRaises(ValueError):
                    compare_teacher(expected, report, pixels, pixels)

    def test_pixels_not_just_scores(self):
        expected, report, pixels = self.fixture()
        fresh = copy.deepcopy(pixels)
        fresh['reconstruction'][0, 0, 0, 0] += 1
        with self.assertRaises(AssertionError):
            compare_teacher(expected, report, pixels, fresh)

    def test_G_requires_noise_and_conditions(self):
        expected, report, pixels = self.fixture(True)
        with patch('demo.online_eg_eval_core.noise_pair') as checked:
            compare_teacher(expected, report, pixels, pixels)
            checked.assert_called_once_with(report, expected, same_condition=True)


class QueueContracts(unittest.TestCase):
    def fixture(self):
        return dict(code={'model.py':'h'}, seed=1, learning_rate=.0001, weight_decay=.0001,
            ranking_weight=.1, arms=['source','zero_source'], teacher=dict(
                receiver=dict(path='/fixed.pt', sha256='a'*64, smoke_weights=False),
                wire_config={'max_g':8}, sender_config={'frame_count':17}))

    def test_complete_attestation_only_difference(self):
        tested = self.fixture(); formal = copy.deepcopy(tested)
        formal['teacher']['receiver'].update(complete_path='/complete.json', complete_sha256='b'*64)
        matching_smoke(tested, formal)

    def test_changed_receiver_and_recipe_rejected(self):
        for key in ('receiver', 'learning_rate', 'code', 'wire_config'):
            tested = self.fixture(); formal = copy.deepcopy(tested)
            if key == 'receiver': formal['teacher']['receiver']['sha256'] = 'f'*64
            elif key == 'wire_config': formal['teacher'][key]['max_g'] = 4
            elif key == 'code': formal[key] = {'model.py':'new'}
            else: formal[key] = .001
            with self.subTest(key=key), self.assertRaises(ValueError):
                matching_smoke(tested, formal)

    def test_smoke_receiver_forbidden_in_formal(self):
        tested = self.fixture(); tested['teacher']['receiver']['smoke_weights'] = True
        with self.assertRaises(ValueError):
            matching_smoke(tested, copy.deepcopy(tested))


if __name__ == '__main__':
    unittest.main()
