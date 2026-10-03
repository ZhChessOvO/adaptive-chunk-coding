"""Queue ordering only; fake child jobs never touch a GPU or real data."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from demo import routervc_revision_queue as queue
from demo.scalable_codec import atomic_json


class RevisionQueueTests(unittest.TestCase):
    def execute_queue(self, root, formal_complete=False):
        revision = root/'revision'
        atomic_json(revision/'mixedview_data/complete.json', dict(complete=True))
        atomic_json(revision/'mixedview_teacher_smoke/complete.json', dict(complete=True))
        if formal_complete:
            atomic_json(revision/'mixedview_teacher/complete.json', dict(complete=True))
        calls = []

        def make_run(args):
            args.output.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(root=args.output, thread=Mock(), stop=Mock(),
                                   progress={}, log_resources=Mock(), update=Mock())

        def execute(run, name, script, argv):
            calls.append((name, script, list(map(str, argv))))
            out = Path(argv[argv.index('--output')+1])
            if script == 'routervc_visual_train.py' or argv[0] == 'run':
                atomic_json(out/'complete.json', dict(complete=True, test_fixture=True))

        argv = ['queue', '--revision', str(revision), '--scratch', str(root/'scratch'), '--epochs', '3']
        with patch.dict(os.environ, {'TMUX': 'synthetic-test'}), patch('sys.argv', argv), \
                patch.object(queue, 'Run', side_effect=make_run), patch.object(queue, 'execute', side_effect=execute):
            queue.main()
        self.assertTrue((revision/'queue/complete.json').exists())
        return calls

    def test_smoke_precedes_formal_teacher_and_training(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = self.execute_queue(Path(folder))
        self.assertEqual([c[0] for c in calls], ['01_verify_teacher_smoke', '02_visual_training_smoke',
                                               '03_mixed_teacher', '04_visual_training'])
        self.assertEqual(calls[0][2][0], 'verify')
        self.assertIn('--smoke', calls[1][2])
        self.assertEqual(calls[2][2][0], 'run')
        self.assertNotIn('--smoke', calls[3][2])
        self.assertEqual(calls[3][2][-3:], ['3', '--max-hours', '18'])

    def test_completed_teacher_uses_verify_not_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = self.execute_queue(Path(folder), formal_complete=True)
        self.assertEqual(calls[2][2][0], 'verify')

    def test_requires_tmux(self):
        with patch.dict(os.environ, {}, clear=True), patch('sys.argv', ['queue']):
            with self.assertRaisesRegex(RuntimeError, 'tmux'):
                queue.main()


if __name__ == '__main__':
    unittest.main()
