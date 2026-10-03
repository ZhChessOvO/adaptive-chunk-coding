"""Synthetic saved-artifact audits; no CUDA, model forward or metric calls."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from demo import routervc_audit as audit
from demo import routervc_format as fmt
from demo import compact_enhancement_format as compact
from demo.routervc_policy import grid_rois, select_generate
from demo.routervc_report import normalize_record, summarize
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash, parse
from demo.routervc_encode import subset_bank
from demo.test_routervc_encode import make_bank


def fixture(folder):
    repo, teacher, router = folder/'repo', folder/'teacher', folder/'router'
    root = folder/'run_smoke'
    for path in (root, repo/'demo', teacher, router/'context', router/'local'):
        path.mkdir(parents=True)
    for name in set(audit.DEPENDENCIES) | {'routervc_format.py', 'routervc_decode.py', 'routervc_policy.py',
                                        'routervc_pipeline.py', 'run_routervc.sh'}:
        atomic_bytes(repo/'demo'/name, name.encode())
    models = {}
    for name in ('E', 'G', 'I', 'P'):
        models[name] = folder/f'{name}.pt'
        atomic_bytes(models[name], name.encode())
    for arm in ('context', 'local'):
        atomic_bytes(router/arm/'model.pt', arm.encode())
    atomic_json(teacher/'complete.json', {'complete': True})
    paths = audit.Paths(repo=repo, teacher=teacher, router=router,
                        enhancement=models['E'], adapter=models['G'], model_i=models['I'], model_p=models['P'])
    identities = {k: '0'*64 for k in fmt.cooperation.HASHES}
    identities['lora'] = file_hash(models['G'])
    policy_hash = '1'*64
    protocol = dict(smoke=True, samples=['sample'], code={p.name: file_hash(p) for p in (repo/'demo').iterdir()},
        teacher=file_hash(teacher/'complete.json'), enhancement=file_hash(models['E']),
        adapter=file_hash(models['G']), router={a: file_hash(router/a/'model.pt') for a in ('context', 'local')})
    atomic_json(root/'protocol.json', protocol)
    base = np.zeros((17, 256, 384, 3), np.uint8)
    enhanced, output = base.copy(), base.copy()
    rois = grid_rois(256, 384)
    x, y, w, h = rois[0]
    enhanced[:, y:y+h, x:x+w] = 20
    output[:] = enhanced
    x, y, w, h = rois[1]
    output[:, y+1:y+h-1, x+1:x+w-1] = 30
    original = parse(make_bank(base))
    meta = dict(original.meta, enhancement_model_sha256=file_hash(models['E']),
                model_i_sha256=file_hash(models['I']), model_p_sha256=file_hash(models['P']))
    bank = compact.base_container(original.base, meta) + b''.join(p.wire for p in original.packets)
    inner = subset_bank(bank, [0])
    config = dict(identities, router=file_hash(router/'context/model.pt'), policy=policy_hash,
        seed=20261003, max_g=4, boundary_lambda=.004, strength=1., blend=1.,
        window=17, stride=8, context=64, feather=16)
    wire = fmt.wrap(inner, config)
    parsed = parse(inner)
    coverage = [1.] + [0.]*15
    gains = np.full(16, -1.)
    gains[1] = 1
    route = dict(select_generate(gains, 4, .004), rois=rois, coverage=coverage,
                 states=['E', 'G'] + ['B']*14)
    destination = root/'sample/context_smooth_r0.5_g4'
    (destination/'fresh').mkdir(parents=True)
    atomic_bytes(destination/'stream.rtvc', wire)
    source_path = folder/'source.npz'
    atomic_npz(source_path, source=base)
    packet_bytes = sum(len(p.wire) for p in parsed.packets)
    job = dict(sample_id='sample', dataset='REDS', method='context_smooth', arm='context',
        ratio=.5, max_g=4, budget=packet_bytes, bytes=len(wire),
        stream_sha256=file_hash(destination/'stream.rtvc'), selected_E=[0],
        source_path=str(source_path), source_hash=file_hash(source_path), folder=str(destination),
        base_hash=frame_hash(base), enhanced_hash=frame_hash(enhanced), expected_route=route,
        plan=dict(budget_e_packet_bytes=packet_bytes, e_packet_bytes=packet_bytes, selected_indices=[0]))
    counts = dict(base_bytes=len(parsed.base), container_header_bytes=parsed.base_end-len(parsed.base),
        packet_bytes=packet_bytes, incomplete_tail_bytes=0, generation_control_bytes=len(wire)-len(inner))
    decode = dict(counts, total_bytes=len(wire), stream_sha256=job['stream_sha256'], config=config,
        source_frames_read=False, base_reference_unchanged=True, outside_generate_exact=True,
        base_hash=job['base_hash'], generation_input_hash=job['enhanced_hash'],
        output_hash=frame_hash(output), explicit_G_map_bytes=0, seconds=2.,
        peak_cuda_allocated_bytes=100, route=route, generation_executed=True,
        generation_assets_validated=True, shared_router_used=True)
    atomic_npz(destination/'fresh/reconstruction.npz', base=base, enhanced=enhanced, reconstruction=output)
    atomic_json(destination/'fresh/decode.json', decode)
    record = dict(job, decode=decode, quality=dict(lpips_alex=.2, psnr_db=27., temporal_delta_mae=4.),
        artifacts={name: file_hash(destination/name) for name in
                   ('stream.rtvc', 'fresh/decode.json', 'fresh/reconstruction.npz')})
    atomic_json(destination/'job.json', job)
    atomic_json(destination/'result.json', record)
    atomic_json(destination/'sender_timing.json', {'selection_and_simulation_seconds': .3})
    atomic_json(root/'jobs.json', dict(jobs=[job], real_prefixes=True))
    atomic_json(root/'summary.json', dict(complete=True, records=[record]))
    for name in ('repeat', 'G_off'):
        target = root/'checks'/name
        target.mkdir(parents=True)
        pixels, detail = output, decode
        if name == 'G_off':
            offroute = dict(select_generate(np.zeros(16), 0, .004), rois=rois, coverage=coverage,
                            states=['E']+['B']*15, policy_skipped=True)
            detail = dict(decode, route=offroute, output_hash=frame_hash(enhanced),
                          generation_executed=False, generation_assets_validated=False, shared_router_used=False)
            pixels = enhanced
        atomic_json(target/'decode.json', detail)
        atomic_npz(target/'reconstruction.npz', base=base, enhanced=enhanced, reconstruction=pixels)
    atomic_json(root/'smoke_audit.json', dict(complete=True, fresh_repeat_exact=True,
        missing_G_router_fallback=True, protocol=file_hash(root/'protocol.json')))
    (root/'report').mkdir()
    atomic_bytes(root/'report/fixed.png', b'synthetic plot; no inference')
    normalized = [normalize_record(record, base.shape)]
    atomic_json(root/'report/summary.json', dict(complete=True, source_complete=True,
        dependencies=dict(summary_sha256=file_hash(root/'summary.json'), code_sha256=file_hash(repo/'demo/routervc_report.py')),
        input_hashes={str(source_path): file_hash(source_path),
                     str(destination/'fresh/reconstruction.npz'): file_hash(destination/'fresh/reconstruction.npz')},
        artifacts={'fixed.png': file_hash(root/'report/fixed.png')}, records=normalized,
        aggregates=summarize(normalized), no_inference=True, no_metric_recalculation=True))
    atomic_json(root/'complete.json', dict(complete=True, points=1, elapsed_seconds=10.,
        protocol=file_hash(root/'protocol.json'), summary=file_hash(root/'summary.json')))
    return root, paths, identities, policy_hash


class AuditTests(unittest.TestCase):
    def patches(self, identities, policy_hash):
        stack = ExitStack()
        stack.enter_context(patch.object(audit, 'current_generation_identity', return_value=identities))
        stack.enter_context(patch.object(audit.fmt, 'policy_identity', return_value=policy_hash))
        for target in ('demo.routervc_decode.decode', 'demo.routervc_decode.route',
                       'demo.routervc_decode.restore', 'demo.routervc_pipeline.execute',
                       'demo.routervc_pipeline.quality', 'demo.routervc_policy.predict'):
            stack.enter_context(patch(target, side_effect=AssertionError('audit must not recompute')))
        return stack

    def test_complete_audit_reentry_preserves_every_original_and_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root, paths, identities, policy_hash = fixture(Path(temp))
            before = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            with self.patches(identities, policy_hash):
                result = audit.audit(root, paths=paths)
                stamp, content = (root/'audit.json').stat().st_mtime_ns, (root/'audit.json').read_bytes()
                again = audit.audit(root, paths=paths)
            self.assertEqual(result, again)
            self.assertEqual(result['points'], 1)
            self.assertEqual(result['unique_video_files_checked'], 3)
            self.assertEqual(result['original_elapsed_seconds'], 10.)
            self.assertEqual(result['points_detail'][0]['original_decode_seconds'], 2.)
            self.assertEqual(stamp, (root/'audit.json').stat().st_mtime_ns)
            self.assertEqual(content, (root/'audit.json').read_bytes())
            for path, old in before.items():
                self.assertEqual((file_hash(Path(path)), Path(path).stat().st_mtime_ns), old)

    def test_each_large_npz_is_file_hashed_only_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root, paths, identities, policy_hash = fixture(Path(temp))
            original = audit._sha256
            counts = {}
            def counted(path):
                name = str(path)
                counts[name] = counts.get(name, 0) + 1
                return original(path)
            with self.patches(identities, policy_hash), patch.object(audit, '_sha256', side_effect=counted):
                audit.audit(root, paths=paths)
            videos = {name: n for name, n in counts.items() if name.endswith('.npz')}
            self.assertEqual(len(videos), 4)
            self.assertEqual(set(videos.values()), {1})

    def test_tampered_smoke_repeat_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as temp:
            root, paths, identities, policy_hash = fixture(Path(temp))
            path = root/'checks/repeat/reconstruction.npz'
            with np.load(path) as p:
                values = {k: p[k].copy() for k in p.files}
            values['reconstruction'][0, 0, 0, 0] ^= 1
            atomic_npz(path, **values)
            with self.patches(identities, policy_hash), self.assertRaisesRegex(RuntimeError, 'outside G changed'):
                audit.audit(root, paths=paths)
            self.assertFalse((root/'audit.json').exists())

    def test_current_code_model_and_source_are_checked(self):
        for target in ('code', 'model', 'source'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root, paths, identities, policy_hash = fixture(Path(temp))
                changed = {'code': paths.repo/'demo/routervc_pipeline.py',
                           'model': paths.enhancement, 'source': Path(temp)/'source.npz'}[target]
                atomic_bytes(changed, b'changed')
                with self.patches(identities, policy_hash), self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    audit.audit(root, paths=paths)

    def test_job_binding_and_report_values_checked(self):
        for target in ('job', 'report'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root, paths, identities, policy_hash = fixture(Path(temp))
                path = next(root.glob('sample/*/job.json')) if target == 'job' else root/'report/summary.json'
                value = json.loads(path.read_text())
                if target == 'job':
                    value['budget'] += 1
                else:
                    value['records'][0]['quality']['lpips_alex'] = .01
                atomic_json(path, value)
                with self.patches(identities, policy_hash), self.assertRaises(RuntimeError):
                    audit.audit(root, paths=paths)

    def test_formal_requires_same_smoke_models_not_just_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root, paths, identities, policy_hash = fixture(Path(temp))
            smoke = Path(temp)/'other_smoke'
            smoke.mkdir()
            original_protocol = json.loads((root/'protocol.json').read_text())
            atomic_json(smoke/'protocol.json', original_protocol)
            atomic_json(smoke/'complete.json', {'complete': True})
            atomic_json(smoke/'smoke_audit.json', dict(complete=True, protocol=file_hash(smoke/'protocol.json')))
            formal = dict(original_protocol, smoke=False)
            atomic_json(root/'protocol.json', formal)
            complete = json.loads((root/'complete.json').read_text())
            complete['protocol'] = file_hash(root/'protocol.json')
            atomic_json(root/'complete.json', complete)
            changed = dict(original_protocol, adapter='f'*64)
            atomic_json(smoke/'protocol.json', changed)
            atomic_json(smoke/'smoke_audit.json', dict(complete=True, protocol=file_hash(smoke/'protocol.json')))
            with self.patches(identities, policy_hash), self.assertRaisesRegex(RuntimeError, 'formal/smoke adapter mismatch'):
                audit.audit(root, paths=paths, smoke_root=smoke)

    def test_incomplete_run_refused_without_any_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root/'complete.json', {'complete': False})
            with patch.object(audit, 'current_generation_identity') as identity:
                with self.assertRaisesRegex(RuntimeError, 'not complete'):
                    audit.audit(root)
                identity.assert_not_called()


if __name__ == '__main__':
    unittest.main()
