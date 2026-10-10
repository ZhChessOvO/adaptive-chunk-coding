import unittest
import numpy as np
from routervc.fusion.blend import fuse, geometry
from demo.routervc_policy import grid_rois


class BlendTest(unittest.TestCase):
    def test_scope_and_shared_edge(self):
        y = np.full((17, 256, 256, 3), 100, np.uint8)
        patches = []
        for region in (0, 1):
            x, yy, w, h = grid_rois(256, 256)[region]
            patches.append(dict(core=[x, yy, w, h], crop=[0, 0, 192, 128],
                                pixels=np.full((17, 128, 192, 3), 150, np.uint8)))
        out = fuse(y, y, patches, [0, 1])
        support, band, _ = geometry(y.shape, [0, 1])
        for value in out.values():
            np.testing.assert_array_equal(value[:, ~support], y[:, ~support])
            np.testing.assert_array_equal(value[:, ~band], y[:, ~band])
            self.assertEqual(value[0, 32, 63, 0], 150)
            self.assertEqual(value[0, 32, 64, 0], 150)

    def test_empty(self):
        y = np.zeros((17, 256, 256, 3), np.uint8)
        for out in fuse(y, y, [], []).values(): np.testing.assert_array_equal(y, out)


if __name__ == '__main__': unittest.main()
