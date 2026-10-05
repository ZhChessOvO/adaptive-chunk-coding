import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools import archive_old_frames as a


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'old_run'
        self.root.mkdir()
        self.stage = Path(self.temp.name) / 'stage'
        (self.root / 'frames').mkdir()
        (self.root / 'frames/im00001.png').write_bytes(b'frame one')
        (self.root / 'frames/im00002.png').write_bytes(b'frame two')
        (self.root / 'visualization.png').write_bytes(b'figure')
        (self.root / 'result.json').write_text('{"quality":1}')

    def test_roundtrip_and_resume(self):
        first = a.compact(self.root, self.stage)
        self.assertTrue(first['complete'])
        self.assertEqual(first['files'], 2)
        self.assertFalse((self.root / 'frames/im00001.png').exists())
        self.assertEqual((self.root / 'visualization.png').read_bytes(), b'figure')
        self.assertEqual((self.root / 'result.json').read_text(), '{"quality":1}')
        with tarfile.open(self.root / a.ARCHIVE) as archive:
            self.assertEqual(archive.extractfile('frames/im00001.png').read(), b'frame one')
        second = a.compact(self.root, self.stage)
        self.assertEqual(first['archive_sha256'], second['archive_sha256'])
        self.assertEqual(second['removed_this_attempt'], 0)

    def test_changed_original_not_retired(self):
        plan = a.inventory(self.root)
        (self.root / 'frames/im00001.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            a.retire_one(self.root, plan['files'][0])
        self.assertTrue((self.root / 'frames/im00001.png').exists())

    def test_unsafe_members_and_symlinks_rejected(self):
        for name in ('../im1.png', '/im1.png', 'model.pt', 'a/../im1.png'):
            with self.assertRaises(ValueError):
                a.frame_path(self.root, name)
        (self.root / 'im9.png').symlink_to(self.root / 'frames/im00001.png')
        with self.assertRaises(ValueError):
            a.inventory(self.root)

    def test_mount_alias_allowed_but_nested_symlink_not(self):
        alias = Path(self.temp.name) / 'mount_alias'
        alias.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(len(a.inventory(alias)['files']), 2)
        (self.root / 'nested').symlink_to(self.root / 'frames', target_is_directory=True)
        with self.assertRaises(ValueError):
            a.frame_path(alias, 'nested/im00001.png')

    def test_corrupt_archive_prevents_any_retirement(self):
        self.stage.mkdir()
        a.write_json(self.stage / 'manifest.json', a.inventory(self.root))
        (self.stage / a.ARCHIVE).write_bytes(b'invalid')
        with self.assertRaises(tarfile.ReadError):
            a.compact(self.root, self.stage)
        self.assertTrue((self.root / 'frames/im00001.png').exists())

    def test_missing_member_rejected(self):
        plan = a.inventory(self.root)
        path = self.root / 'incomplete.tar'
        with tarfile.open(path, 'w') as archive:
            content = json.dumps(plan).encode()
            member = tarfile.TarInfo(a.MANIFEST)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        with self.assertRaises(ValueError):
            a.verify_tar(path, plan)

    def test_mid_retirement_restart_preserves_all_content(self):
        original = a.retire_one
        def interrupted(root, item):
            if item['path'].endswith('im00002.png'):
                raise InterruptedError('simulated interruption')
            return original(root, item)
        with patch.object(a, 'retire_one', interrupted):
            with self.assertRaises(InterruptedError):
                a.compact(self.root, self.stage)
        self.assertTrue((self.root / a.ARCHIVE).exists())
        result = a.compact(self.root, self.stage)
        self.assertTrue(result['complete'])
        self.assertEqual(result['removed_this_attempt'], 1)

    def test_zero_inode_bootstrap_keeps_verified_copy(self):
        for i in range(3, 12):
            (self.root / f'frames/im{i:05d}.png').write_bytes(bytes([i]))
        real_statvfs = a.os.statvfs
        def quota(path):
            if Path(path) == self.root:
                remaining = len(list((self.root / 'frames').glob('im*.png')))
                return SimpleNamespace(f_files=200000, f_favail=11-remaining)
            return real_statvfs(path)
        # shutil.disk_usage uses statvfs too; it retains its real byte counters.
        real_usage = a.shutil.disk_usage(self.root)
        with patch.object(a.os, 'statvfs', quota), patch.object(a.shutil, 'disk_usage', return_value=real_usage):
            result = a.compact(self.root, self.stage)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['bootstrap_retired']), 8)
        with tarfile.open(self.root / a.ARCHIVE) as archive:
            self.assertEqual(len(archive.getmembers()), 12)


if __name__ == '__main__':
    unittest.main()
