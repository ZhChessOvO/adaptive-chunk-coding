"""CPU completion queue contracts, without running any experiment or child."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from demo import routervc_finalize as workflow
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash


class FakeRun:
    def __init__(self, root):
        self.root = root
        self.progress = {}
        self.checks = 0

    def check(self):
        self.checks += 1

    def update(self, **kw):
        self.progress.update(kw)


class FinalizeTests(unittest.TestCase):
    def test_wait_requires_all_true_markers_and_never_sleeps_over_15_seconds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markers = {name: root/f'{name}.json' for name in ('main', 'supplement', 'resilience')}
            atomic_json(markers['main'], {'complete': True})
            atomic_json(markers['supplement'], {'complete': False})
            run = FakeRun(root)
            def fake_sleep(seconds):
                self.assertEqual(seconds, 10.)
                self.assertEqual(run.progress['missing'], ['supplement', 'resilience'])
                atomic_json(markers['supplement'], {'complete': True})
                atomic_json(markers['resilience'], {'complete': True})
            workflow.wait_for_completions(run, markers, sleep=fake_sleep)
            self.assertEqual(run.checks, 2)
            with self.assertRaises(ValueError):
                workflow.wait_for_completions(run, markers, interval=16)

    def test_public_commands_are_CPU_analysis_not_model_or_baseline_GPU_verify(self):
        commands = workflow.child_commands(Path('/main'), Path('/main/supplement'))
        self.assertEqual([name for name, _, _ in commands], ['audit', 'analysis', 'preview'])
        text = ' '.join(str(x) for _, command, _ in commands for x in command)
        self.assertNotIn('routervc_baselines.py', text)
        self.assertNotIn('routervc_decode.py', text)
        self.assertIn('run_routervc_analysis.sh', text)
        self.assertIn('--repeats 1', text)
        self.assertIn('--profile formal', text)

    def test_complete_reentry_preserves_elapsed_and_all_output_mtimes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root/'workflow'
            destination.mkdir()
            run = FakeRun(destination)
            marker = root/'original.complete.json'
            atomic_json(marker, dict(complete=True, elapsed_seconds=123.))
            dependencies = {str(marker): file_hash(marker)}
            commands = [(name, ['fake_cpu', name], root/f'{name}.json') for name in ('audit', 'analysis', 'preview')]
            def child(_run, name, command):
                atomic_json(root/f'{name}.json', dict(complete=True, elapsed_seconds=5.))
            def evidence(*args):
                return {str(path): file_hash(path) for _, _, path in commands}
            with patch.object(workflow, 'wait_for_completions', return_value=0.), \
                 patch.object(workflow, 'completion_evidence', return_value=dependencies), \
                 patch.object(workflow, 'child_commands', return_value=commands), \
                 patch.object(workflow, 'run_child', side_effect=child) as launcher, \
                 patch.object(workflow, 'output_evidence', side_effect=evidence):
                first = workflow.finalize(run, root, root/'supplement', root/'resilience')
                before = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in root.rglob('*.json')}
                second = workflow.finalize(run, root, root/'supplement', root/'resilience')
                after = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in root.rglob('*.json')}
                self.assertEqual(first, second)
                self.assertEqual(before, after)
                self.assertEqual(launcher.call_count, 3)
                self.assertTrue(first['no_GPU_mutex'])
                self.assertTrue(first['original_timings_preserved'])
                atomic_json(root/'analysis.json', dict(complete=True, elapsed_seconds=99.))
                with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
                    workflow.finalize(run, root, root/'supplement', root/'resilience')

    def test_code_changes_after_wait_are_rejected_before_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(workflow, 'wait_for_completions', return_value=0.), \
                 patch.object(workflow, 'verify_code', side_effect=RuntimeError('source pin changed')), \
                 patch.object(workflow, 'run_child') as launcher:
                with self.assertRaisesRegex(RuntimeError, 'source pin changed'):
                    workflow.finalize(FakeRun(root), root/'main', root/'supplement', root/'resilience')
                launcher.assert_not_called()

    def test_resilience_requires_three_cases_and_real_CPU_rejection_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            supplement, resilience = root/'supplement', root/'resilience'
            atomic_json(root/'protocol.json', {})
            atomic_json(root/'summary.json', {'complete': True})
            main_summary = file_hash(root/'summary.json')
            atomic_json(root/'complete.json', dict(complete=True, summary=main_summary,
                                                  protocol=file_hash(root/'protocol.json')))
            atomic_json(supplement/'protocol.json', {})
            atomic_json(supplement/'summary.json', dict(main_summary=main_summary,
                        baseline_protocol=file_hash(supplement/'protocol.json')))
            atomic_json(supplement/'complete.json', dict(complete=True, readonly_completed_point_replay=True,
                        summary=file_hash(supplement/'summary.json')))
            atomic_json(resilience/'protocol.json', dict(details=dict(rejection=dict(
                strict_truncated='truncated packet', crc_strict='CRC mismatch', crc_permissive='CRC mismatch'))))
            records = {}
            for name in ('first_packet', 'truncated_next_packet', 'base_no_models'):
                atomic_bytes(resilience/f'{name}.rtvc', name.encode())
                decoded = dict(stream_sha256=file_hash(resilience/f'{name}.rtvc'))
                atomic_json(resilience/name/'decode.json', decoded)
                atomic_bytes(resilience/name/'reconstruction.npz', b'mocked RGB artifact')
                records[name] = dict(complete=True, decoded=decoded,
                    artifacts={filename: file_hash(resilience/name/filename)
                               for filename in ('decode.json', 'reconstruction.npz')})
                atomic_json(resilience/name/'complete.json', records[name])
            atomic_bytes(resilience/'crc_rejected_cpu.rtvc', b'bad CRC')
            atomic_json(resilience/'cpu_rejections.json', dict(complete=True, gpu_started=False,
                strict_truncated='truncated packet', crc_strict='CRC mismatch', crc_permissive='CRC mismatch',
                artifacts={name: file_hash(resilience/name) for name in
                           ('truncated_next_packet.rtvc', 'crc_rejected_cpu.rtvc')}))
            evidence = dict(complete=True, fresh_decodes=3, truncated_pixels_and_route_exact=True,
                missing_E_G_router_fallback_exact=True, strict_truncation_and_CRC_rejected_before_GPU=True,
                source_frames_read=False, protocol=file_hash(resilience/'protocol.json'), records=records)
            atomic_json(resilience/'complete.json', evidence)
            hashes = workflow.completion_evidence(root, supplement, resilience)
            self.assertIn(str((resilience/'cpu_rejections.json').resolve()), hashes)
            atomic_json(resilience/'complete.json', dict(evidence, records={}))
            with self.assertRaisesRegex(RuntimeError, 'exactly the three'):
                workflow.completion_evidence(root, supplement, resilience)
            atomic_json(resilience/'complete.json', evidence)
            atomic_json(resilience/'cpu_rejections.json', dict(complete=True, gpu_started=False))
            with self.assertRaisesRegex(RuntimeError, 'rejection evidence'):
                workflow.completion_evidence(root, supplement, resilience)


if __name__ == '__main__':
    unittest.main()
