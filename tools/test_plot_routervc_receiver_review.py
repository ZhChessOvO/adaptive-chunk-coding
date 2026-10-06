"""Input provenance tests for the supplemental, inference-free plots."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.plot_routervc_receiver_review import digest, inputs


class CompletedInputs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for subdir, names in (
            ('report', ('summary.json',)),
            ('formal/router', ('history.json', 'initial_validation.json')),
        ):
            folder = self.root / subdir
            folder.mkdir(parents=True)
            artifacts = {}
            for name in names:
                path = folder / name
                path.write_text(json.dumps({'name': name}))
                artifacts[name] = digest(path)
            (folder / 'complete.json').write_text(json.dumps({
                'complete': True, 'artifacts': artifacts}))

    def test_three_completed_inputs(self):
        values, hashes = inputs(self.root)
        self.assertEqual(len(hashes), 3)
        self.assertEqual([v['name'] for v in values],
                         ['summary.json', 'history.json', 'initial_validation.json'])

    def test_changed_summary_rejected(self):
        (self.root / 'report/summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'completed input changed'):
            inputs(self.root)

    def test_incomplete_training_rejected(self):
        path = self.root / 'formal/router/complete.json'
        receipt = json.loads(path.read_text())
        receipt['complete'] = False
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'completed input changed'):
            inputs(self.root)

    def test_missing_artifact_rejected(self):
        (self.root / 'formal/router/history.json').unlink()
        with self.assertRaises(FileNotFoundError):
            inputs(self.root)


if __name__ == '__main__':
    unittest.main()
