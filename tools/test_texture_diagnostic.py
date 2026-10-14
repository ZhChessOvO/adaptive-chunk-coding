import unittest
import numpy as np
from tools.texture_diagnostic import phase_diagnostic, core


class DiagnosticTests(unittest.TestCase):
    def test_constant_has_no_grid(self):
        a = np.full((2, 96, 96, 3), 99, np.uint8)
        result, _ = phase_diagnostic(a)
        self.assertEqual(result['phase_rms_u8'], 0.)

    def test_phase_grid_and_reference(self):
        a = np.full((3, 96, 96, 3), 99, np.float32)
        a[:, ::8, ::8] += 40
        result, _ = phase_diagnostic(a)
        self.assertGreater(result['phase_rms_u8'], 4.)
        result, _ = phase_diagnostic(a, a)
        self.assertEqual(result['phase_rms_u8'], 0.)

    def test_relative_core(self):
        a = np.arange(2*16*16*3).reshape(2, 16, 16, 3)
        c = dict(crop=[32, 16, 16, 16], core=[36, 24, 8, 4])
        np.testing.assert_array_equal(core(a, c), a[:, 8:12, 4:12])


if __name__ == '__main__': unittest.main()
