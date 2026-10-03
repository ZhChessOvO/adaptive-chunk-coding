"""CPU protocol tests; synthetic arrays/metadata, no assets or inference."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from demo import routervc_mixedview_teacher as teacher


class MixedTeacherTest(unittest.TestCase):
    def test_dynamic_grid_preserves_whole_view(self):
        for height, width in ((576, 1024), (512, 512)):
            mask = np.zeros((height, width), np.uint8)
            rois = teacher.grid(height, width)
            self.assertEqual(len(rois), 16)
            for x, y, w, h in rois:
                mask[y:y+h, x:x+w] += 1
            np.testing.assert_array_equal(mask, 1)
        self.assertEqual(teacher.grid(576, 1024)[0], [0, 0, 256, 144])
        self.assertEqual(teacher.grid(576, 1024)[5], [256, 144, 256, 144])

    def test_isolated_e_never_changes_neighbors(self):
        base = np.zeros((2, 8, 8, 3), np.uint8)
        enhanced = np.full_like(base, 9)
        roi = [2, 2, 2, 2]
        output = teacher.isolate(base, enhanced, roi)
        self.assertEqual(int(output.sum()), 2*2*2*3*9)
        np.testing.assert_array_equal(teacher.crop(output, roi), 9)
        np.testing.assert_array_equal(base, 0)

    def test_pair_control_seed_and_geometry(self):
        rois = teacher.grid(576, 1024)
        a = teacher.control({'profile':'synthetic'}, 'sample', 0, rois)
        b = teacher.control({'profile':'synthetic'}, 'sample', 0, rois)
        self.assertEqual(a, b)
        self.assertEqual(a['generate'], [[0, 17, 0, 0, 256, 144]])
        self.assertEqual(a['protect'], [])
        self.assertNotEqual(a['seed'], teacher.control({}, 'sample', 5, rois)['seed'])

    def test_uvg_controls_exactly_match_historical_geometry(self):
        # Pure-byte seed comparison, without importing the old GPU framework.
        import hashlib
        sid = 'uvg-beauty-f032-center'
        for index in (0, 5, 15):
            settings = teacher.control({}, sid, index, teacher.grid(512, 512))
            self.assertEqual(settings['seed'], int(hashlib.sha256(
                f'four-state-v1/{sid}/{index}'.encode()).hexdigest()[:12], 16))
            x, y = index % 4 * 128, index // 4 * 128
            self.assertEqual(settings['generate'], [[0, 17, x, y, 128, 128]])

    def test_shared_cost_is_not_repeated_per_region(self):
        # Match the documented shared-head aggregation for a synthetic B/E/G/EG layout.
        base, e, shared, roi_control = 100, [10, 20, 30, 40], 50, 12
        states = ['B', 'E', 'G', 'EG']
        total = teacher.aggregate_diagnostic_bytes(base, e, states, shared, roi_control)
        self.assertEqual(total, 234)
        self.assertNotEqual(total, sum(base+n+shared+roi_control for n in e))
        self.assertEqual(teacher.aggregate_diagnostic_bytes(base, e, ['B']*4, shared, roi_control), base)
        with self.assertRaises(ValueError):
            teacher.aggregate_diagnostic_bytes(base, e, ['bad']*4, shared, roi_control)

    def test_unknown_content_and_four_states_explicit(self):
        self.assertEqual(teacher.STATES, ('B', 'E', 'G', 'EG'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source.npz').write_bytes(b'synthetic source')
            (root/'received.npz').write_bytes(b'synthetic decoded')
            sample = dict(sample_id='sample', dataset='REDS', sequence='000', router_split='train',
                          view_kind='resized_full_frame', historical_teacher_sample_id='old')
            record = teacher._base_label({'sample':sample}, root/'source.npz', root/'received.npz',
                                         (17,576,1024,3))
            self.assertEqual(record['content_annotation_status'], 'unknown')
            self.assertFalse(record['semantic_labels_available'])
            self.assertFalse(record['receiver_uses_source'])
            self.assertFalse(record['independent_system_test'])
            self.assertEqual(record['sample'], sample)

    def test_completed_point_keeps_time_hash_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'result.json'
            teacher.immutable(path, dict(seconds=12.4))
            original = teacher.digest(path), path.stat().st_mtime_ns
            teacher.immutable(path, dict(seconds=12.4))
            self.assertEqual(original, (teacher.digest(path), path.stat().st_mtime_ns))
            with self.assertRaises(ValueError):
                teacher.immutable(path, dict(seconds=.01))

    def test_tampered_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); teacher.save(root/'p.json', {'value':1})
            artifacts = {'p.json':teacher.digest(root/'p.json')}
            teacher.verify(root, artifacts)
            teacher.save(root/'p.json', {'value':2})
            with self.assertRaises(ValueError):
                teacher.verify(root, artifacts)

    def test_imports_are_gpu_framework_free(self):
        script = ('import sys; import demo.routervc_mixedview_teacher; '
                  'import demo.routervc_mixedview_teacher_receive; assert "torch" not in sys.modules')
        subprocess.run([sys.executable, '-c', script], cwd=teacher.REPO, check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
