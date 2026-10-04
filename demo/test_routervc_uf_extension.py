import unittest
import numpy as np
from demo import routervc_uf_extension as e
from demo import routervc_baselines as b


class ExtensionTests(unittest.TestCase):
    def test_allowed_quality_values_are_scoped_and_restore_on_error(self):
        old=b.QPS
        with self.assertRaises(RuntimeError):
            with e.extended_qps():
                self.assertTrue(set(e.EXTRA_QPS).issubset(b.QPS))
                raise RuntimeError('interrupted')
        self.assertEqual(b.QPS,old)

    def test_new_native_metadata_is_authenticated_without_altering_old_validator(self):
        pixels=np.zeros((17,64,64,3),dtype=np.uint8)
        weights=dict(model_i_sha256='0'*64,model_p_sha256='1'*64)
        with e.extended_qps():
            meta=b.uf_metadata(48,pixels.shape,b'bits',pixels,weights)
            b.validate_uf_metadata(meta,b'bits')
            with self.assertRaises(ValueError):b.validate_uf_metadata(meta,b'changed')
        with self.assertRaises(ValueError):b.validate_uf_metadata(meta,b'bits')

    def test_only_three_extra_points_per_existing_sample(self):
        entries=[dict(sample=dict(sample_id=str(i))) for i in range(13)]
        planned=e.plan(dict(sources=entries))
        self.assertEqual(len(planned),39)
        self.assertTrue(all(point['qp'] in (40,48,56) for _,point in planned))
        self.assertEqual([entry for entry,_ in planned[::3]],entries)


if __name__=='__main__':unittest.main()
