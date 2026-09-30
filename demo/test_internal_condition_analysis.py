"""The completed-result resume check must not silently launch new work."""
from pathlib import Path
import unittest
from unittest.mock import patch

from demo import internal_condition_analysis as analysis


class ResumeTests(unittest.TestCase):
    def exercise(self, action=None, altered=False):
        root = Path('/example/completed')
        summary = dict(results=[1], training={}, protocol={}, paired_noise=True,
            no_E_exact=True, repeat_exact=True, G_off_without_weights_exact=True)

        def read(path):
            return {} if path.name == 'training_audit.json' else summary

        def evaluate(output, run):
            if action:
                action()
            analysis.evaluation.atomic_json(root / 'training_audit.json', {})
            record = dict(summary, results=[2]) if altered else dict(summary)
            analysis.evaluation.atomic_json(root / 'evaluation/summary.json', record)

        with patch.object(analysis, 'read', side_effect=read), \
             patch.object(analysis, 'file_hash', return_value='unchanged'), \
             patch.object(analysis, 'grouped') as grouped, \
             patch.object(analysis.evaluation, 'evaluate', side_effect=evaluate):
            result = analysis.resume_without_recompute(root, object())
            grouped.assert_called_once_with([1])
            return result

    def test_completed_results_preserved(self):
        self.assertTrue(self.exercise()['formal_summary_unchanged'])

    def test_child_decode_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'must not decode'):
            self.exercise(lambda: analysis.pipeline.execute())

    def test_metric_recompute_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'must not decode'):
            self.exercise(lambda: analysis.evaluation.quality())

    def test_altered_results_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'results'):
            self.exercise(altered=True)


if __name__ == '__main__':
    unittest.main()
