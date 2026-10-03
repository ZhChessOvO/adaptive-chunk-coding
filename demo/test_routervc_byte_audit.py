"""Tiny synthetic, CPU-only tests; never read or rewrite formal experiments."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

from demo import compact_enhancement_format as compact
from demo import routervc_byte_audit as audit


def wire(packets=2):
    base = b'native-UF-test-payload'
    meta = dict(compact.CONSTANTS, width=64, height=64, frame_count=17,
                enhancement_codec=compact.CODECS[-1], **{k: '0' * 64 for k in compact.HASHES})
    inner = compact.base_container(base, meta)
    for i in range(packets):
        pm = dict(packet_id=i + 1, start=1 + i * 8, count=8, roi=[0, 0, 16, 16],
                  qstep=1., codec=meta['enhancement_codec'])
        inner += compact.packet_bytes(pm, b'payload' * (i + 1), codec=pm['codec'])
    body = audit.RTVC_CONTROL.pack(1, 8, .004, 1., 1., 17, 8, 64, 16) + bytes(32 * 8)
    return audit.RTVC_HEADER.pack(b'RTVC', 1, len(body), zlib.crc32(body)) + body + inner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.encoded_json(value))


def fixture(root, points=2):
    jobs = []
    for i in range(points):
        folder = root / f'sample-{i}' / 'context'
        folder.mkdir(parents=True)
        data = wire(i)
        parsed = audit.inspect_stream(data)
        counts = parsed['bytes']
        job = dict(folder=str(folder), bytes=len(data), stream_sha256=audit.digest(data),
                   sample_id=f'sample-{i}', dataset='REDS' if i == 0 else 'UVG',
                   method='context', ratio=i / points, max_g=8)
        decoded = dict(base_bytes=counts['native_uf_bytes'], container_header_bytes=counts['acse_header_bytes'],
                       packet_bytes=counts['e_packet_header_bytes'] + counts['e_payload_bytes'],
                       incomplete_tail_bytes=0, generation_control_bytes=counts['rtvc_shared_header_bytes'],
                       total_bytes=len(data), stream_sha256=job['stream_sha256'], explicit_G_map_bytes=0,
                       config=parsed['config'])
        (folder / 'stream.rtvc').write_bytes(data)
        write_json(folder / 'job.json', job)
        write_json(folder / 'fresh/decode.json', decoded)
        write_json(folder / 'result.json', dict(decode=decoded, artifacts={
            name: audit.file_hash(folder / name) for name in ('stream.rtvc', 'fresh/decode.json')}))
        jobs.append(job)
    write_json(root / 'jobs.json', dict(jobs=jobs))
    write_json(root / 'protocol.json', dict(role='synthetic test'))
    write_json(root / 'summary.json', dict(complete=True))
    write_json(root / 'complete.json', dict(complete=True, points=points,
        summary=audit.file_hash(root / 'summary.json'), protocol=audit.file_hash(root / 'protocol.json')))
    return jobs


class ByteAuditTests(unittest.TestCase):
    def test_exact_decomposition_and_no_imaginary_mask_savings(self):
        record = audit.inspect_stream(wire())
        self.assertEqual(record['bytes']['rtvc_shared_header_bytes'], 310)
        self.assertEqual(record['bytes']['acse_header_bytes'], 190)
        self.assertEqual(record['bytes']['e_packet_header_bytes'], 2 * 41)
        self.assertEqual(record['bytes']['e_payload_bytes'], 21)
        self.assertEqual(sum(record['bytes'].values()), len(wire()))
        self.assertEqual(sum(record['separate_mask_bytes'].values()), 0)
        self.assertTrue(record['schema_evidence']['e_coordinates_are_charged'])

    def test_base_only(self):
        r = audit.inspect_stream(wire(0))
        self.assertEqual(r['packet_count'], 0)
        self.assertEqual(r['bytes']['e_payload_bytes'], 0)
        self.assertEqual(r['total_bytes'], 500 + len(b'native-UF-test-payload'))

    def test_partial_tail_is_charged_not_payload(self):
        truncated = wire()[:-2]
        with self.assertRaisesRegex(ValueError, 'truncated'):
            audit.inspect_stream(truncated)
        r = audit.inspect_stream(truncated, allow_incomplete_tail=True)
        self.assertEqual(r['packet_count'], 1)
        self.assertEqual(r['bytes']['incomplete_tail_bytes'], 41 + 14 - 2)
        self.assertEqual(sum(r['bytes'].values()), len(truncated))

    def test_corruption_is_not_a_tolerated_tail(self):
        data = wire()
        corrupt = data[:-1] + bytes([data[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            audit.inspect_stream(corrupt, allow_incomplete_tail=True)
        for bad in (data[:8], b'BAD!' + data[4:], data[:20] + bytes([data[20] ^ 1]) + data[21:]):
            with self.assertRaises(ValueError):
                audit.inspect_stream(bad)

    def test_unknown_variable_outer_fields_rejected(self):
        data = wire()
        body = data[13:310] + b'fake-mask'
        expanded = audit.RTVC_HEADER.pack(b'RTVC', 1, len(body), zlib.crc32(body)) + body + data[310:]
        with self.assertRaisesRegex(ValueError, 'schema'):
            audit.inspect_stream(expanded)

    def test_resume_and_verify_preserve_every_historical_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / 'history', Path(temp) / 'audit'
            fixture(root)
            original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            partial = audit.audit(root, output, expected_count=2, stop_after=1)
            self.assertFalse(partial['complete'])
            saved = (output / 'points/0000.json').stat().st_mtime_ns
            done = audit.audit(root, output, expected_count=2)
            self.assertTrue(done['complete'])
            self.assertEqual(saved, (output / 'points/0000.json').stat().st_mtime_ns)
            complete = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob('*')
                        if p.is_file() and p.name != '.lock'}
            audit.audit(root, output, expected_count=2, verify_only=True)
            audit.audit(root, output, expected_count=2)
            self.assertTrue(all(p.read_bytes() == b for p, b in original.items()))
            self.assertTrue(all((p.read_bytes(), p.stat().st_mtime_ns) == v for p, v in complete.items()))
            self.assertEqual(done['mask_removal_savings_bytes'], 0)

    def test_changed_source_and_corrupt_cached_audit_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / 'history', Path(temp) / 'audit'
            fixture(root)
            audit.audit(root, output, expected_count=2)
            point = output / 'points/0000.json'
            old = point.read_bytes()
            point.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'audit output changed'):
                audit.audit(root, output, expected_count=2)
            point.write_bytes(old)
            source = root / 'sample-0/context/stream.rtvc'
            source.write_bytes(source.read_bytes() + b'junk')
            with self.assertRaises(ValueError):
                audit.audit(root, output, expected_count=2)

    def test_no_writes_inside_history_or_verify_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'history'
            fixture(root)
            with self.assertRaisesRegex(ValueError, 'separate'):
                audit.audit(root, root / 'report', expected_count=2)
            self.assertFalse((root / 'report').exists())
            with self.assertRaisesRegex(ValueError, 'missing audit'):
                audit.audit(root, Path(temp) / 'missing', expected_count=2, verify_only=True)

    def test_out_of_scope_job_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'history'
            jobs = fixture(root)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                audit.inspect_point(root, dict(jobs[0], folder=temp))

    def test_cli_requires_tmux(self):
        with mock.patch.dict('os.environ', {'TMUX': ''}):
            with self.assertRaisesRegex(ValueError, 'tmux'):
                audit.main([])

    def test_audit_imports_no_torch_or_cuda(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import demo.routervc_byte_audit; assert "torch" not in sys.modules'],
                       cwd=audit.REPO, check=True, timeout=20)

    def test_invalid_time_and_count_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / 'history', Path(temp) / 'audit'
            fixture(root)
            for seconds in (0, -1, float('nan'), True):
                with self.assertRaisesRegex(ValueError, 'time limit'):
                    audit.audit(root, output, expected_count=2, max_seconds=seconds)
            with self.assertRaisesRegex(ValueError, 'point count'):
                audit.audit(root, output, expected_count=3)


if __name__ == '__main__':
    unittest.main()
