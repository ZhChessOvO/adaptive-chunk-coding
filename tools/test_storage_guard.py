import json
from pathlib import Path
import sys
import tempfile
import unittest
from tools import storage_guard as guard


def snapshot(free=50000, used=.65, total=200000):
    return {'file-store': dict(used_fraction=used, total_inodes=total, free_inodes=free)}


class GuardTests(unittest.TestCase):
    def test_free_bytes_do_not_override_inode_exhaustion(self):
        self.assertIn('file entries', guard.reasons(snapshot(0))[0])

    def test_byte_limit_preserved(self):
        self.assertIn('byte usage', guard.reasons(snapshot(used=.8))[0])

    def test_unknown_inode_total_ignored(self):
        self.assertEqual(guard.reasons(snapshot(free=0, total=0)), [])

    def test_boundary(self):
        self.assertEqual(guard.reasons(snapshot(free=5000)), [])
        self.assertTrue(guard.reasons(snapshot(free=4999)))

    def test_full_mount_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'should_not_exist'
            result = guard.supervise([sys.executable, '-c',
                f'from pathlib import Path; Path({str(marker)!r}).touch()'],
                Path(tmp) / 'guard.jsonl', snapshot=lambda: snapshot(0), gpu=lambda: '')
            self.assertEqual(result, 75)
            self.assertFalse(marker.exists())

    def test_child_exit_status_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(guard.supervise([sys.executable, '-c', 'raise SystemExit(7)'],
                Path(tmp) / 'guard.jsonl', interval=.05, snapshot=snapshot, gpu=lambda: ''), 7)

    def test_exhausting_mount_signals_resumable_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'checkpoint'
            ready = Path(tmp) / 'ready'
            code = ('import signal,time; from pathlib import Path; '
                    f'signal.signal(signal.SIGTERM,lambda *_:(Path({str(marker)!r}).touch(),exit(0))); '
                    f'Path({str(ready)!r}).touch(); '
                    'time.sleep(10)')
            count = 0
            def changing():
                nonlocal count
                count += 1
                return snapshot(4999 if count >= 8 and ready.exists() else 50000)
            log = Path(tmp) / 'guard.jsonl'
            result = guard.supervise([sys.executable, '-c', code], log, interval=.03,
                                     snapshot=changing, gpu=lambda: '')
            self.assertEqual(result, 75)
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads(log.read_text().splitlines()[-1])['phase'], 'stopped_for_headroom')

    def test_actual_mount_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            disks = guard.disk_snapshot([tmp])
            self.assertIn('free_inodes', disks[tmp])


if __name__ == '__main__':
    unittest.main()
