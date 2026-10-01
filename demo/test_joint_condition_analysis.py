"""The audit must never replace completed pixels, metrics or formal timings."""
from pathlib import Path
import unittest
from unittest.mock import patch

from demo import joint_condition_analysis as analysis


class ResumeTests(unittest.TestCase):
    def exercise(self, action=None, altered=False):
        root = Path('/example/completed')
        def read(path):
            return {'results':[1]} if path.name == 'summary.json' else {}
        calls = [0]
        def digest(path):
            calls[0] += 1
            return 'changed' if altered and calls[0] > 6 else 'original'
        def evaluate(output, run):
            if action:
                action()
            analysis.training.atomic_json(root/'training_audit.json',{})
        with patch.object(analysis,'read',side_effect=read), \
             patch.object(analysis,'file_hash',side_effect=digest), \
             patch.object(analysis,'verify_artifacts'), \
             patch.object(analysis.evaluation,'grouped') as grouped, \
             patch.object(analysis.evaluation,'evaluate',side_effect=evaluate):
            result = analysis.resume_without_recompute(root,object())
            grouped.assert_called_once_with([1])
            return result

    def test_valid_completed_resume(self):
        self.assertTrue(self.exercise()['formal_files_unchanged'])

    def test_new_decode_forbidden(self):
        with self.assertRaisesRegex(AssertionError,'must not decode'):
            self.exercise(lambda:analysis.receiver.execute())

    def test_metric_recompute_forbidden(self):
        with self.assertRaisesRegex(AssertionError,'must not decode'):
            self.exercise(lambda:analysis.evaluation.quality())

    def test_summary_write_forbidden(self):
        with self.assertRaisesRegex(AssertionError,'must not decode'):
            self.exercise(lambda:analysis.evaluation.atomic_json())

    def test_model_recreation_forbidden(self):
        with self.assertRaisesRegex(AssertionError,'must not decode'):
            self.exercise(lambda:analysis.receiver.atomic_torch_save())

    def test_hash_change_detected(self):
        with self.assertRaises(AssertionError):
            self.exercise(altered=True)


if __name__ == '__main__':
    unittest.main()
