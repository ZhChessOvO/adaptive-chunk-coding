"""CPU-only contracts for descriptive completed-result interpolation."""
import unittest

from tools.latent_sender_report import matched_value


class MatchedRateTest(unittest.TestCase):
    def setUp(self):
        self.points = [{'bpp': 1., 'score': 4.}, {'bpp': 4., 'score': 2.}]

    def test_exact_endpoints(self):
        self.assertEqual(matched_value(self.points, 1., 'score'), 4.)
        self.assertEqual(matched_value(self.points, 4., 'score'), 2.)

    def test_log_rate_midpoint(self):
        self.assertEqual(matched_value(self.points, 2., 'score'), 3.)

    def test_no_extrapolation(self):
        self.assertIsNone(matched_value(self.points, .5, 'score'))
        self.assertIsNone(matched_value(self.points, 5., 'score'))

    def test_order_independent(self):
        self.assertEqual(matched_value(self.points[::-1], 2., 'score'), 3.)

    def test_bad_curves_rejected(self):
        for points in ([], self.points + self.points[:1], [{'bpp': 0, 'score': 2}]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                matched_value(points, 2., 'score')


if __name__ == '__main__':
    unittest.main()
