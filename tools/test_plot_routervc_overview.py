import unittest

from tools.plot_routervc_overview import center_box, means, memory_peak, nearest_rate


class OverviewTest(unittest.TestCase):
    def test_nearest_rate_does_not_select_by_quality(self):
        rows=[dict(point='q1',bpp=.01,lpips_alex=.01),dict(point='q2',bpp=.019,lpips_alex=.9)]
        self.assertEqual(nearest_rate(rows,.02)['point'],'q2')

    def test_nearest_rate_uses_log_distance(self):
        rows=[dict(point='low',bpp=.5),dict(point='high',bpp=1.8)]
        self.assertEqual(nearest_rate(rows,1)['point'],'high')

    def test_bad_rates(self):
        for rows,bpp in (([],1),([dict(point='a',bpp=0)],1),([dict(point='a',bpp=1)],0)):
            with self.assertRaises(ValueError):nearest_rate(rows,bpp)

    def test_peak_includes_earlier_roi(self):
        d=dict(peak_cuda_allocated_bytes=2**30,generation_runtime=dict(windows=[
            dict(runtime=dict(peak_cuda_allocated_bytes=4*2**30)),
            dict(runtime=dict(peak_cuda_allocated_bytes=2*2**30))]))
        self.assertEqual(memory_peak(d),4)
        self.assertEqual(memory_peak(dict(peak_cuda_allocated_bytes=2**30)),1)

    def test_center_not_quality_based(self):
        self.assertEqual(center_box(1024,576),(341,192,682,384))
        self.assertEqual(center_box(512,512),(171,171,341,341))

    def test_group_mean_is_window_equal(self):
        self.assertEqual(means([{'bpp':1},{'bpp':3}],('bpp',)),{'bpp':2})
        with self.assertRaises(ValueError):means([],('bpp',))


if __name__=='__main__':unittest.main()
