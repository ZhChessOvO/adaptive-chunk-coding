import tempfile
import unittest
from pathlib import Path

from tools.fusion_finish import pending_stage


class FinishQueueTests(unittest.TestCase):
    def test_waits_in_order_and_requires_all_completions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ('p1_data', 'p1_fit', 'p1_evaluation'):
                self.assertEqual(pending_stage(root), stage)
                (root/stage).mkdir(parents=True)
                (root/stage/'complete.json').touch()
            self.assertIsNone(pending_stage(root))

    def test_upstream_failure_stops_but_completed_resume_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'p1_data').mkdir()
            (root/'p1_data/last_failure.json').touch()
            with self.assertRaises(RuntimeError):
                pending_stage(root)
            (root/'p1_data/complete.json').touch()
            self.assertEqual(pending_stage(root), 'p1_fit')
            (root/'p1_fit').mkdir()
            (root/'p1_fit/complete.json').touch()
            (root/'p1_evaluation/queue').mkdir(parents=True)
            (root/'p1_evaluation/queue/last_failure.json').touch()
            with self.assertRaises(RuntimeError):
                pending_stage(root)


if __name__ == '__main__':
    unittest.main()
