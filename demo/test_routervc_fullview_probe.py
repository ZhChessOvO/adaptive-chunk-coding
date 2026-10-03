"""Protocol-only tests: no torch import, models, CUDA or video datasets."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from demo import routervc_fullview_probe as probe


class FullviewProtocolTest(unittest.TestCase):
    def test_ten_points_no_automatic_model_selection(self):
        points = probe.point_plan()
        self.assertEqual(len(points), 10)
        self.assertEqual(len({p['name'] for p in points}), 10)
        routes = [p for p in points if p['kind'] == 'router']
        self.assertEqual({(p['arm'], p['ratio']) for p in routes},
                         {('context', .25), ('context', .5), ('local', .25), ('local', .5)})
        self.assertEqual(len([p for p in points if p['kind'] == 'full_g']), 1)

    def test_geometry_explicit_no_crop_resize(self):
        probe.validate_shape((17, 576, 1024, 3))
        for shape in ((17, 512, 512, 3), (33, 576, 1024, 3), (17, 1080, 1920, 3)):
            with self.assertRaises(ValueError):
                probe.validate_shape(shape)

    def test_native_uf_primary_rate_excludes_audit_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'stream.bin').write_bytes(b'abcd')
            (root/'transmitted_meta.json').write_bytes(b'{}\n')
            report = dict(native_bytes=4, metadata_bytes=3, total_bytes=7)
            result = probe.byte_ledger('uf', root, report)
            self.assertEqual(result['bytes'], 4)
            self.assertEqual(result['audit_sidecar_bytes'], 3)
            with self.assertRaises(ValueError):
                probe.byte_ledger('uf', root, dict(report, total_bytes=4))

    def test_all_router_and_generator_headers_remain_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind, file in (('router', 'stream.rtvc'), ('off', 'stream.rtvc'),
                               ('full_g', 'stream.acsg')):
                (root/file).write_bytes(b'abcdef')
                result = probe.byte_ledger(kind, root, dict(total_bytes=6, base_bytes=2))
                self.assertEqual(result['bytes'], 6)

    def test_receiver_commands_never_carry_source_and_never_own_lock(self):
        root = Path('/synthetic/output')
        for point in probe.point_plan():
            script, argv, distributed = probe.receiver_command(root, point)
            self.assertNotEqual(script, 'routervc.py')
            self.assertNotIn('--input', argv)
            self.assertNotIn('--source', argv)
            self.assertNotIn('source.npz', [str(v) for v in argv])
            if point['kind'] == 'off':
                self.assertIn('--disable-generation', argv)
                self.assertIn(Path('/missing/fullview-router.pt'), argv)
                self.assertIn(Path('/missing/fullview-generator.pt'), argv)
                self.assertFalse(distributed)

    def test_encoders_share_one_prepared_bank(self):
        root = Path('/synthetic/output')
        for point in probe.point_plan()[:4]:
            args = probe.encode_arguments(root, point)
            self.assertEqual(args[args.index('--prepared-dir')+1], root/'prepared')
            self.assertEqual(args[args.index('--max-g')+1], 8)
            self.assertEqual(args[args.index('--boundary-lambda')+1], 0)
            self.assertEqual(args[args.index('--mode')+1], 'prefix')

    def test_immutable_resume_preserves_original_time_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            original = dict(seconds=12.34, complete=True)
            probe.immutable(path, original)
            before = (probe.digest(path), path.stat().st_mtime_ns)
            probe.immutable(path, original)
            self.assertEqual(before, (probe.digest(path), path.stat().st_mtime_ns))
            self.assertEqual(probe.read(path)['seconds'], 12.34)
            with self.assertRaises(ValueError):
                probe.immutable(path, dict(original, seconds=.01))

    def test_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe.save(root/'a.json', dict(complete=True))
            expected = {'a.json': probe.digest(root/'a.json')}
            probe.verify_artifacts(root, expected)
            probe.save(root/'a.json', dict(complete=False))
            with self.assertRaises(ValueError):
                probe.verify_artifacts(root, expected)

    def test_module_import_is_torch_free(self):
        script = 'import sys; import demo.routervc_fullview_probe; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-c', script], cwd=probe.REPO, check=True, timeout=10)


class RecoveryTest(unittest.TestCase):
    """Small files exercise real hash/protocol checks; video decoding is stubbed."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.previous = Path(self.temporary.name)/'previous'
        self.current = Path(self.temporary.name)/'current'
        self.previous.mkdir(); self.current.mkdir()
        self.run = SimpleNamespace(root=self.current)
        snapshot = self.previous/'runner_before_baseline_fix.py'
        snapshot.write_text('# previous runner\n')
        weights = {k: {'path': '/weights/'+k, 'sha256': k+'-hash'}
                   for k in ('enhancement', 'adapter', 'context', 'local', 'uf_i', 'uf_p')}
        self.old = dict(source={'path': '/verified/input'}, source_shape=list(probe.SHAPE),
                        source_rgb_sha256='rgb-hash', source_preparation=None,
                        weights=weights, configs={'context': {'seed': 4}, 'local': {'seed': 4}},
                        points=probe.point_plan(), code={
                            'demo/routervc_fullview_probe.py': probe.digest(snapshot),
                            'demo/routervc_encode.py': 'encoder-code-hash',
                            'demo/unchanged_dependency.py': 'dependency-hash'})
        self.protocol = json.loads(json.dumps(self.old))
        self.protocol['code']['demo/routervc_fullview_probe.py'] = 'new-runner-hash'
        probe.save(self.previous/'protocol.json', self.old)
        probe.save(self.current/'protocol.json', self.protocol)
        (self.previous/'source.npz').write_bytes(b'source-pixels')
        probe.save(self.previous/'source.complete.json', {
            'protocol_sha256': probe.digest(self.previous/'protocol.json'),
            'artifacts': {'source.npz': probe.digest(self.previous/'source.npz')}})
        prepared = self.previous/'prepared'; prepared.mkdir()
        for name in ('base.acse', 'bank.acse', 'candidates.npz'):
            (prepared/name).write_bytes(name.encode())
        source_binding = dict(kind='npz',key='source',start=0,requested_count=None,
                              source_rgb_sha256='rgb-hash', source_shape=list(probe.SHAPE),
                              file_sha256=probe.digest(self.previous/'source.npz'),
                              path=str(self.previous/'source.npz'))
        binding = dict(version=1,source_code='encoder-code-hash',padded_frames=0,
                       source=source_binding, enhancement='enhancement-hash',
                       model_i='uf_i-hash', model_p='uf_p-hash', base_qp=8, qstep=1.)
        probe.save(prepared/'base.complete.json', dict(binding=binding,base_rgb_sha256='base-hash',seconds=10.,
                   artifacts={'base.acse': probe.digest(prepared/'base.acse')}))
        probe.save(prepared/'complete.json', dict(binding=binding, base_rgb_sha256='base-hash',base_seconds=10.,
                   padding=dict(temporal_frames=0,spatial_pixels=0,input_shape=list(probe.SHAPE)),
                   artifacts={name: probe.digest(prepared/name)
                              for name in ('base.acse', 'bank.acse', 'candidates.npz')}))
        self.decoded = dict(seconds=12.5, output_hash='decoded-pixels')
        self.ledger = dict(bytes=73)
        self.originals = {}
        for point in probe.point_plan()[:-1]:
            folder = self.previous/point['name']; folder.mkdir()
            probe.save(folder/'job.json', dict(point=point,
                       protocol_sha256=probe.digest(self.previous/'protocol.json')))
            (folder/'stream.fixture').write_bytes(point['name'].encode())
            (folder/'fresh').mkdir()
            probe.save(folder/'fresh/decode.json', self.decoded)
            result = dict(complete=True, point=point,
                          protocol_sha256=probe.digest(self.previous/'protocol.json'),
                          decode=self.decoded, byte_ledger=self.ledger,
                          quality={'lpips_alex': .123}, decode_seconds=12.5,
                          elapsed_seconds_this_completion_attempt=34.75,
                          artifacts={name: probe.digest(folder/name) for name in
                                     ('job.json', 'stream.fixture', 'fresh/decode.json')})
            probe.save(folder/'result.json', result)
            self.originals[point['name']] = result
        self.receiver = patch.object(probe, 'validate_receiver',
                                     return_value=(self.decoded, self.ledger))
        self.receiver.start(); self.addCleanup(self.receiver.stop)
        def validate_prepared(root, protocol):
            prepared = probe.read(root/'prepared/complete.json')
            base = probe.read(root/'prepared/base.complete.json')
            source = probe.read(root/'source.complete.json')
            self.assertEqual(source['protocol_sha256'],probe.digest(root/'protocol.json'))
            probe.verify_artifacts(root,source['artifacts'])
            probe.validate_prepared_binding(prepared,base,protocol,probe.digest(root/'source.npz'))
            probe.verify_artifacts(root/'prepared',prepared['artifacts'])
            probe.verify_artifacts(root/'prepared',base['artifacts'])
            return prepared
        self.prepared = patch.object(probe,'validate_prepared',side_effect=validate_prepared)
        self.prepared.start(); self.addCleanup(self.prepared.stop)

    def test_import_nine_preserves_metrics_times_and_original_files(self):
        original_files = {str(p.relative_to(self.previous)): (probe.digest(p), p.stat().st_mtime_ns)
                          for p in self.previous.rglob('*') if p.is_file()}
        probe.import_completed(self.previous, self.run, self.protocol)
        self.assertFalse((self.current/'wholeframe_g_one_roi').exists())
        protocol_hash = probe.digest(self.current/'protocol.json')
        for point in probe.point_plan()[:-1]:
            folder = self.current/point['name']
            result = probe.validate_result(self.current, point, protocol_hash)
            self.assertEqual(result['quality'], self.originals[point['name']]['quality'])
            self.assertEqual(result['decode_seconds'], 12.5)
            self.assertEqual(result['elapsed_seconds_this_completion_attempt'], 34.75)
            self.assertEqual(probe.read(folder/'job.json'),
                             dict(point=point, protocol_sha256=protocol_hash))
            self.assertEqual(result['reused_from']['sha256'],
                             probe.digest(self.previous/point['name']/'result.json'))
            self.assertTrue(result['reused_from']['fresh_decode_and_metric_times_preserved'])
        imported_files = {str(p.relative_to(self.current)): (probe.digest(p), p.stat().st_mtime_ns)
                          for p in self.current.rglob('*') if p.is_file()}
        probe.import_completed(self.previous, self.run, self.protocol)
        for relative, expected in imported_files.items():
            path = self.current/relative
            self.assertEqual((probe.digest(path), path.stat().st_mtime_ns), expected)
        for relative, expected in original_files.items():
            path = self.previous/relative
            self.assertEqual((probe.digest(path), path.stat().st_mtime_ns), expected)

    def test_source_geometry_weights_configs_points_mismatches_rejected(self):
        for key in ('source', 'source_shape', 'source_rgb_sha256', 'source_preparation',
                    'weights', 'configs', 'points'):
            with self.subTest(key=key):
                altered = dict(self.protocol, **{key: 'mismatch'})
                with self.assertRaisesRegex(ValueError, key):
                    probe.import_completed(self.previous, self.run, altered)

    def test_changed_dependency_and_old_runner_snapshot_rejected(self):
        altered = json.loads(json.dumps(self.protocol))
        altered['code']['demo/unchanged_dependency.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'dependency'):
            probe.import_completed(self.previous, self.run, altered)
        (self.previous/'runner_before_baseline_fix.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            probe.import_completed(self.previous, self.run, self.protocol)

    def test_same_output_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'new output'):
            probe.import_completed(self.previous, SimpleNamespace(root=self.previous), self.protocol)

    def test_extra_dependency_rejected(self):
        altered = json.loads(json.dumps(self.protocol))
        altered['code']['demo/new_dependency.py'] = 'new'
        with self.assertRaisesRegex(ValueError, 'dependency'):
            probe.import_completed(self.previous,self.run,altered)

    def test_prepared_source_models_and_base_binding_rejected(self):
        prepared = probe.read(self.previous/'prepared/complete.json')
        base = probe.read(self.previous/'prepared/base.complete.json')
        source_hash = probe.digest(self.previous/'source.npz')
        probe.validate_prepared_binding(prepared,base,self.protocol,source_hash)
        alterations = [('source','file_sha256','wrong'),('source','source_rgb_sha256','wrong'),
                       ('source','source_shape',[17,512,512,3]),('source','start',1),
                       ('binding','enhancement','wrong'),('binding','model_i','wrong'),
                       ('binding','model_p','wrong'),('binding','source_code','wrong'),
                       ('binding','qstep',2.),('binding','padded_frames',1)]
        for scope,key,value in alterations:
            with self.subTest(scope=scope,key=key):
                altered = json.loads(json.dumps(prepared))
                target = altered['binding']['source'] if scope=='source' else altered['binding']
                target[key] = value
                with self.assertRaises(ValueError):
                    probe.validate_prepared_binding(altered,base,self.protocol,source_hash)
        altered = json.loads(json.dumps(base)); altered['binding']['qstep'] = 2.
        with self.assertRaisesRegex(ValueError,'base.complete binding'):
            probe.validate_prepared_binding(prepared,altered,self.protocol,source_hash)

    def test_prepared_artifact_corruption_rejected(self):
        (self.previous/'prepared/bank.acse').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            probe.import_completed(self.previous, self.run, self.protocol)

    def test_completed_point_artifact_corruption_rejected(self):
        (self.previous/probe.point_plan()[0]['name']/'stream.fixture').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            probe.import_completed(self.previous, self.run, self.protocol)

    def test_missing_completed_artifact_rejected(self):
        (self.previous/probe.point_plan()[0]['name']/'stream.fixture').unlink()
        with self.assertRaises(FileNotFoundError):
            probe.import_completed(self.previous, self.run, self.protocol)

    def test_destination_conflict_is_not_overwritten(self):
        prepared = self.current/'prepared'; prepared.mkdir()
        (prepared/'bank.acse').write_bytes(b'not-the-same-bank')
        with self.assertRaisesRegex(ValueError, 'destination differs'):
            probe.import_completed(self.previous, self.run, self.protocol)
        self.assertEqual((prepared/'bank.acse').read_bytes(), b'not-the-same-bank')


class FullGenerationEnvelopeTest(unittest.TestCase):
    def test_empty_acse2_bank_subset_not_legacy_base_and_idempotent(self):
        # Stub only heavyweight worker imports. Assert the envelope receives the
        # empty-E compact bank, never the legacy ACSE1 prepare/base.acse bytes.
        import demo
        modules = {name: ModuleType(name) for name in (
            'demo.conditioned_generation_pipeline', 'demo.scalable_codec',
            'demo.routervc_format', 'demo.scalable_cooperation_format', 'demo.routervc_encode')}
        modules['demo.conditioned_generation_pipeline'].execute = lambda *a, **k: self.fail('no worker expected')
        modules['demo.scalable_codec'].atomic_bytes = lambda path, data: Path(path).write_bytes(data)
        modules['demo.routervc_format'].generation_control = lambda config, e, g, n: {'generate': [], 'count': n}
        calls = []
        def subset(data, indices):
            calls.append((data, indices))
            self.assertEqual(data, b'ACSE2-candidate-bank')
            self.assertEqual(indices, [])
            return b'ACSE2-empty-E'
        def wrap(inner, control):
            self.assertEqual(inner, b'ACSE2-empty-E')
            self.assertEqual(control['generate'], [[0, 17, 0, 0, 1024, 576]])
            return b'ACSG-header:'+inner
        modules['demo.routervc_encode'].subset_bank = subset
        modules['demo.scalable_cooperation_format'].wrap = wrap
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
                patch.object(demo, 'routervc_format', modules['demo.routervc_format'], create=True), \
                patch.object(demo, 'scalable_cooperation_format', modules['demo.scalable_cooperation_format'], create=True):
            root = Path(directory); (root/'prepared').mkdir()
            (root/'prepared/base.acse').write_bytes(b'ACSE1-legacy-base')
            (root/'prepared/bank.acse').write_bytes(b'ACSE2-candidate-bank')
            protocol = {'configs': {'context': {'seed': 3}}}
            probe.save(root/'protocol.json', protocol)
            point = probe.point_plan()[-1]; run = SimpleNamespace(root=root)
            probe.prepare_point(run, point, protocol)
            stream = root/point['name']/'stream.acsg'
            before = (probe.digest(stream), stream.stat().st_mtime_ns)
            probe.prepare_point(run, point, protocol)
            self.assertEqual((probe.digest(stream), stream.stat().st_mtime_ns), before)
            self.assertEqual(len(calls), 2)
            self.assertEqual(probe.byte_ledger('full_g', stream.parent,
                             {'total_bytes': stream.stat().st_size})['bytes'], stream.stat().st_size)
            stream.write_bytes(b'wrong-prior-envelope')
            with self.assertRaisesRegex(ValueError, 'wire changed'):
                probe.prepare_point(run, point, protocol)


class SourcePreparationTest(unittest.TestCase):
    def test_record_bound_to_exact_png_directory_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); frames = root/'frames'; frames.mkdir()
            for index in range(17):
                (frames/f'{index:04}.png').write_bytes(str(index).encode())
            transform = dict(crop=None,valid_rect=[0,0,1024,576],coded_size=[1024,576])
            record = dict(complete=True,frames_dir=str(frames),frame_count=17,transform=transform,
                          binding=dict(sample=dict(whole_frame=True,transform=transform)),
                          artifacts={str(p.relative_to(root)):probe.digest(p) for p in frames.glob('*.png')})
            path = root/'complete.json'; probe.save(path,record)
            _,verified = probe.validate_source_preparation(frames,path)
            self.assertTrue(verified)
            other = root/'different_frames'; other.mkdir()
            with self.assertRaisesRegex(ValueError,'frames_dir'):
                probe.validate_source_preparation(other,path)
            npz = root/'source.npz'; npz.write_bytes(b'not-png')
            with self.assertRaisesRegex(ValueError,'PNG input'):
                probe.validate_source_preparation(npz,path)
            (frames/'0016.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact set'):
                probe.validate_source_preparation(frames,path)

    def test_json_is_independent_and_payload_link_fallback_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.json'; probe.save(source,{'old':True})
            target = root/'target.json'; probe.copy_file(source,target)
            self.assertNotEqual(source.stat().st_ino,target.stat().st_ino)
            payload = root/'source.npz'; payload.write_bytes(b'fixed')
            with patch.object(probe.os,'link',side_effect=OSError('cross-device')):
                probe.copy_file(payload,root/'target.npz')
            self.assertEqual((root/'target.npz').read_bytes(),b'fixed')
            self.assertNotEqual(payload.stat().st_ino,(root/'target.npz').stat().st_ino)


if __name__ == '__main__':
    unittest.main()
