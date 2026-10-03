"""Independent CPU-only audit of completed RouterVC pipeline artifacts.

Run under tmux with CUDA_VISIBLE_DEVICES='' after complete.json exists. This
does not call a decoder, Router, generator, or metric. It validates saved pixels
and records, and writes only audit.json, atomically. Re-entry checks the evidence
again and preserves the original audit file and all original timings unchanged.

File SHA256s and saved-video pixel hashes are cached within one invocation, so
repeated references to a large reconstruction NPZ do not rehash/inflate it.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo import routervc_format as fmt
from demo.routervc_policy import grid_rois
from demo.routervc_decode import coverage_from_packets
from demo.routervc_report import grid_statistics, route_states, normalize_record, summarize
from demo.scalable_format import frame_hash


RUNS = Path('/root/autodl-fs/DCVC/runs')
# Runtime inputs of this pipeline/report/audit, not every RouterVC experiment.
# Unrelated baselines or future public-CLI changes must not invalidate this run.
DEPENDENCIES = (
    'routervc_audit.py', 'routervc_report.py', 'four_state_core.py',
    'four_state_router.py', 'four_state_router_evaluate.py',
    'chunk_enhancement_codec.py', 'chunk_enhancement_experiment.py',
    'chunk_enhancement_evaluate.py', 'conditioned_generation_pipeline.py',
    'scalable_codec.py', 'scalable_format.py', 'compact_enhancement_format.py',
    'scalable_cooperation_format.py', 'scalable_generation_format.py',
    'scalable_experiment.py', 'stage_c_three_path_roi_probe.py',
    'online_eg_decode.py', 'internal_condition_decode.py',
)


@dataclass(frozen=True)
class Paths:
    repo: Path = REPO
    teacher: Path = RUNS/'a800_four_state_20261002'
    router: Path = RUNS/'a800_four_state_router_20261002'
    enhancement: Path = RUNS/'a800_online_eg_20261002/joint/enhancement.pt'
    adapter: Path = RUNS/'a800_online_eg_20261002/joint/adapter.pt'
    model_i: Path = REPO/'checkpoints/cvpr2026_image.pth.tar'
    model_p: Path = REPO/'checkpoints/cvpr2026_video_hts.pth.tar'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def current_generation_identity(adapter):
    # This reads/checksums pre-shared weights and their source-profile only;
    # identities loads the small adapter on CPU, never instantiates G or CUDA.
    from demo.online_eg_decode import identities
    return identities(adapter)


class Reader:
    def __init__(self):
        self.hashes = {}
        self.jsons = {}
        self.videos = {}

    def digest(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = _sha256(path)
        value = self.hashes[key]
        if expected is not None:
            require(value == expected, f'artifact hash mismatch: {path}')
        return value

    def json(self, path):
        path = Path(path).resolve()
        self.digest(path)
        key = str(path)
        if key not in self.jsons:
            with path.open() as stream:
                self.jsons[key] = json.load(stream)
        return self.jsons[key]

    def video(self, path, *, generated_mask=None, enhanced_mask=None):
        path = Path(path).resolve()
        self.digest(path)
        key = str(path)
        if key not in self.videos:
            values, arrays = {}, {}
            shape = None
            with np.load(path, allow_pickle=False) as cache:
                for field in ('base', 'enhanced', 'reconstruction'):
                    pixels = cache[field]
                    require(pixels.dtype == np.uint8 and pixels.ndim == 4 and pixels.shape[-1] == 3,
                            f'invalid saved RGB video: {path}:{field}')
                    if shape is None:
                        shape = pixels.shape
                    require(pixels.shape == shape, f'inconsistent RGB geometry: {path}')
                    values[field] = frame_hash(pixels)
                    arrays[field] = pixels
            if generated_mask is not None:
                require(np.array_equal(arrays['reconstruction'][~generated_mask],
                                       arrays['enhanced'][~generated_mask]),
                        f'pixels outside G changed: {path}')
            if enhanced_mask is not None:
                require(np.array_equal(arrays['enhanced'][~enhanced_mask],
                                       arrays['base'][~enhanced_mask]),
                        f'pixels outside received E changed: {path}')
            self.videos[key] = dict(shape=list(shape), hashes=values)
        return self.videos[key]

    def artifacts(self, folder, values):
        folder = Path(folder).resolve()
        require(isinstance(values, dict) and values, f'empty artifact manifest: {folder}')
        for name, expected in values.items():
            path = (folder/name).resolve()
            require(path.is_relative_to(folder), f'artifact escapes its folder: {name}')
            self.digest(path, expected)


def _finite_quality(quality):
    for name in ('lpips_alex', 'psnr_db', 'temporal_delta_mae'):
        value = quality[name]
        require(type(value) in (int, float) and math.isfinite(value), f'nonfinite {name}')


def _decoder(reader, directory, wire, job, *, expected_route=None, generation_off=False):
    """Validate an already executed fresh receiver; never construct model input."""
    config, _, inner, header = fmt.parse(wire)
    data = reader.json(directory/'decode.json')
    require(data['stream_sha256'] == job['stream_sha256'], 'decoder stream hash mismatch')
    require(data['total_bytes'] == len(wire) == job['bytes'], 'decoder actual byte mismatch')
    require(data['config'] == config, 'decoder configuration differs from wire')
    counts = dict(base_bytes=len(inner.base), container_header_bytes=inner.base_end-len(inner.base),
                  packet_bytes=sum(len(p.wire) for p in inner.packets),
                  incomplete_tail_bytes=inner.incomplete_tail_bytes, generation_control_bytes=header)
    require(sum(counts.values()) == len(wire), 'stream byte decomposition mismatch')
    require(all(data[k] == v for k, v in counts.items()), 'decoder byte decomposition mismatch')
    require(data['source_frames_read'] is False and data['base_reference_unchanged'] is True
            and data['outside_generate_exact'] is True, 'receiver invariants failed')
    require(data['base_hash'] == job['base_hash'] and data['generation_input_hash'] == job['enhanced_hash'],
            'decoder B/Y identity differs from sender')
    require(data['explicit_G_map_bytes'] == 0, 'unexpected explicit G mask charge')
    require(type(data['seconds']) in (int, float) and math.isfinite(data['seconds'])
            and data['seconds'] >= 0, 'invalid preserved decoder timing')
    expected_shape = [inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3]
    route = data['route']
    rois = grid_rois(inner.meta['height'], inner.meta['width'])
    require(route['rois'] == rois, 'receiver grid mismatch')
    require(route['coverage'] == coverage_from_packets(inner, rois).tolist(), 'E coverage differs from packets')
    indices = route['indices']
    require(indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < 16 for i in indices),
            'invalid or duplicate generated cells')
    require(len(indices) <= config['max_g'], 'G call budget exceeded')
    expected_states = [('EG' if route['coverage'][i] > 0 else 'G') if i in indices
                       else ('E' if route['coverage'][i] > 0 else 'B') for i in range(16)]
    require(route['states'] == expected_states, 'four-state labels do not match E/G decisions')
    statistics = grid_statistics(route_states(route))
    for key in ('boundary_edges', 'components'):
        require(route[key] == statistics[key], f'incorrect saved {key}')
    require(data['generation_executed'] is bool(indices), 'generation execution flag mismatch')
    require(data['generation_assets_validated'] is bool(indices), 'generation asset flag mismatch')
    g_mask = np.zeros(expected_shape[:3], bool)
    for index in indices:
        x, y, w, h = rois[index]
        # Inner feather has alpha zero only at the tile perimeter; the rest is
        # allowed to change. No region merging or protection exists in RTVC v1.
        pad = int(config['feather'] > 0)
        if config['blend'] > 0:
            g_mask[:, y+pad:y+h-pad, x+pad:x+w-pad] = True
    e_mask = np.zeros(expected_shape[:3], bool)
    for packet in inner.packets:
        p = packet.meta
        x, y, w, h = p['roi']
        e_mask[p['start']:p['start']+p['count'], y:y+h, x:x+w] = True
    video = reader.video(directory/'reconstruction.npz', generated_mask=g_mask, enhanced_mask=e_mask)
    require(video['shape'] == expected_shape, 'wire and saved RGB dimensions differ')
    for field, key in (('base', 'base_hash'), ('enhanced', 'generation_input_hash'),
                       ('reconstruction', 'output_hash')):
        require(video['hashes'][field] == data[key], f'saved {field} pixel hash mismatch')
    if expected_route is not None:
        require(route == expected_route, 'sender/receiver routes differ')
    if generation_off or config['max_g'] == 0 or config['blend'] == 0:
        require(not indices and data['shared_router_used'] is False, 'G-off executed policy/generation')
        require(data['output_hash'] == job['enhanced_hash'], 'G-off pixels differ from received E')
    else:
        require(data['shared_router_used'] is True, 'shared policy unexpectedly skipped')
    return data, config, inner, counts


def _smoke_checks(reader, root):
    saved_audit = reader.json(root/'smoke_audit.json')
    require(saved_audit['complete'] and saved_audit['fresh_repeat_exact']
            and saved_audit['missing_G_router_fallback'], 'incomplete smoke checks')
    reader.digest(root/'protocol.json', saved_audit['protocol'])
    complete = reader.json(root/'complete.json')
    require(complete['complete'], 'smoke completion is false')
    reader.digest(root/'protocol.json', complete['protocol'])
    reader.digest(root/'summary.json', complete['summary'])
    jobs = reader.json(root/'jobs.json')['jobs']
    job = next(j for j in jobs if j['method'] == 'context_smooth' and j['ratio'] == .5)
    folder = Path(job['folder']).resolve()
    require(folder.is_relative_to(root.resolve()), 'smoke job escapes smoke root')
    require(reader.json(folder/'job.json') == job, 'smoke JOB binding mismatch')
    record = reader.json(folder/'result.json')
    require(all(record.get(k) == v for k, v in job.items()), 'smoke result/JOB binding mismatch')
    require(record in reader.json(root/'summary.json')['records'], 'smoke result missing from completed summary')
    reader.artifacts(folder, record['artifacts'])
    reader.digest(folder/'stream.rtvc', job['stream_sha256'])
    wire = (folder/'stream.rtvc').read_bytes()
    original, _, _, _ = _decoder(reader, folder/'fresh', wire, job, expected_route=job['expected_route'])
    require(original == record['decode'], 'smoke original receiver differs from result')
    repeat, _, _, _ = _decoder(reader, root/'checks/repeat', wire, job, expected_route=job['expected_route'])
    fallback, _, _, _ = _decoder(reader, root/'checks/G_off', wire, job, generation_off=True)
    require(repeat['output_hash'] == original['output_hash'], 'fresh repeat pixels differ')
    require(fallback['output_hash'] == job['enhanced_hash'], 'missing-assets fallback pixels differ')
    return dict(fresh_repeat_exact=True, G_off_exact=True,
                original_smoke_audit=reader.digest(root/'smoke_audit.json'))


def audit(root, *, paths=Paths(), smoke_root=None, progress=None):
    root = Path(root).resolve()
    if progress:
        progress('checking completion, source pins and shared model identities')
    reader = Reader()
    complete = reader.json(root/'complete.json')
    require(complete['complete'] is True, 'pipeline is not complete')
    reader.digest(root/'protocol.json', complete['protocol'])
    reader.digest(root/'summary.json', complete['summary'])
    protocol = reader.json(root/'protocol.json')
    summary = reader.json(root/'summary.json')
    manifest = reader.json(root/'jobs.json')
    require(summary['complete'] is True and manifest['real_prefixes'] is True, 'incomplete summary/jobs')
    jobs, records = manifest['jobs'], summary['records']
    require(len(jobs) == len(records) == complete['points'] and jobs, 'point count mismatch')
    require(len(set(j['folder'] for j in jobs)) == len(jobs), 'duplicate job output folder')
    require(set(j['sample_id'] for j in jobs) == set(protocol['samples']), 'protocol sample support mismatch')
    for name, expected in protocol['code'].items():
        require(Path(name).name == name, 'unsafe protocol source name')
        reader.digest(paths.repo/'demo'/name, expected)
    reader.digest(paths.teacher/'complete.json', protocol['teacher'])
    reader.digest(paths.enhancement, protocol['enhancement'])
    reader.digest(paths.adapter, protocol['adapter'])
    for arm, expected in protocol['router'].items():
        require(arm in ('context', 'local'), 'unknown Router arm')
        reader.digest(paths.router/arm/'model.pt', expected)
    uf_hashes = dict(model_i_sha256=reader.digest(paths.model_i), model_p_sha256=reader.digest(paths.model_p))
    generation_identity = current_generation_identity(paths.adapter)
    require(generation_identity['lora'] == protocol['adapter'], 'current LoRA identity differs from protocol')
    policy_identity = fmt.policy_identity()
    # Record actual runtime sources. The pinned G identity above recursively
    # checks its historical source profile; unrelated baseline/CLI experiments
    # are intentionally excluded from this evidence boundary.
    for name in DEPENDENCIES:
        reader.digest(paths.repo/'demo'/name)
    reader.digest(Path(__file__))
    if not protocol['smoke']:
        smoke_root = Path(smoke_root) if smoke_root else root.with_name(root.name+'_smoke')
        previous = reader.json(smoke_root/'protocol.json')
        previous_complete = reader.json(smoke_root/'complete.json')
        previous_audit = reader.json(smoke_root/'smoke_audit.json')
        require(previous_complete['complete'] and previous_audit['complete'], 'smoke is not complete')
        reader.digest(smoke_root/'protocol.json', previous_audit['protocol'])
        for key in ('code', 'router', 'enhancement', 'adapter'):
            require(previous[key] == protocol[key], f'formal/smoke {key} mismatch')
    prefixes = {}
    point_summaries, normalized_records = [], []
    for index, (job, record) in enumerate(zip(jobs, records)):
        folder = Path(job['folder']).resolve()
        require(folder.is_relative_to(root) and folder != root, 'job folder escapes audit root')
        require(reader.json(folder/'job.json') == job, 'saved JOB differs from jobs manifest')
        require(reader.json(folder/'result.json') == record, 'result differs from completed summary')
        require(all(record.get(k) == v for k, v in job.items()), 'result/JOB binding mismatch')
        reader.artifacts(folder, record['artifacts'])
        reader.digest(folder/'stream.rtvc', job['stream_sha256'])
        wire = (folder/'stream.rtvc').read_bytes()
        data, config, inner, counts = _decoder(reader, folder/'fresh', wire, job,
                                               expected_route=job['expected_route'])
        require(data == record['decode'], 'result decoder record differs from fresh decode.json')
        require(config['router'] == protocol['router'][job['arm']], 'stream Router differs from protocol')
        require(config['policy'] == policy_identity, 'current shared-policy source differs from stream')
        require(all(config[k] == v for k, v in generation_identity.items()), 'current G model/profile differs from stream')
        require(inner.meta['enhancement_model_sha256'] == protocol['enhancement'], 'stream E weights differ from protocol')
        require(all(inner.meta[k] == v for k, v in uf_hashes.items()), 'stream UF weights differ from local models')
        require(config['max_g'] == job['max_g'], 'JOB/control G budget mismatch')
        require(counts['packet_bytes'] <= job['budget'] == job['plan']['budget_e_packet_bytes'], 'E byte budget exceeded/mismatch')
        require(counts['packet_bytes'] == job['plan']['e_packet_bytes'], 'selected E packet cost mismatch')
        require(job['plan']['selected_indices'] == job['selected_E'], 'selected E plan mismatch')
        rois = grid_rois(inner.meta['height'], inner.meta['width'])
        received_e = {rois.index(p.meta['roi']) for p in inner.packets}
        require(received_e == set(job['selected_E']), 'actual E packets differ from selected regions')
        _finite_quality(record['quality'])
        normalized_records.append(normalize_record(record, reader.videos[str((folder/'fresh/reconstruction.npz').resolve())]['shape']))
        reader.digest(Path(job['source_path']), job['source_hash'])
        timing = folder/'sender_timing.json'
        require(timing.exists(), 'missing original sender timing')
        reader.json(timing)
        key = (job['sample_id'], job['method'], job['max_g'])
        prefixes.setdefault(key, []).append((job['ratio'], wire))
        point_summaries.append(dict(sample_id=job['sample_id'], method=job['method'],
            ratio=job['ratio'], max_g=job['max_g'], bytes=len(wire), byte_breakdown=counts,
            output_hash=data['output_hash'], original_decode_seconds=data['seconds']))
        if progress:
            progress(f'checked point {index+1}/{len(jobs)}')
    prefix_pairs = 0
    for values in prefixes.values():
        ordered = sorted(values, key=lambda v: v[0])
        for (_, small), (_, large) in zip(ordered, ordered[1:]):
            require(large.startswith(small), 'completed operating points are not literal byte prefixes')
            prefix_pairs += 1
    smoke_checks = _smoke_checks(reader, root if protocol['smoke'] else smoke_root)
    report = reader.json(root/'report/summary.json')
    require(report['complete'] and report['source_complete'], 'incomplete report')
    reader.digest(root/'summary.json', report['dependencies']['summary_sha256'])
    reader.digest(paths.repo/'demo/routervc_report.py', report['dependencies']['code_sha256'])
    for path, expected in report['input_hashes'].items():
        reader.digest(Path(path), expected)
    reader.artifacts(root/'report', report['artifacts'])
    require(report['records'] == normalized_records, 'report rows differ from audited results')
    require(report['aggregates'] == summarize(normalized_records), 'report aggregates differ from audited results')
    require(report['no_inference'] and report['no_metric_recalculation'], 'report changed evaluation scope')
    result = dict(complete=True, format='routervc_completed_artifact_audit_v1', root=str(root),
        mode='smoke' if protocol['smoke'] else 'formal', points=len(jobs),
        prefix_pairs=prefix_pairs, smoke_checks=smoke_checks, figures=len(report['artifacts']),
        original_elapsed_seconds=complete['elapsed_seconds'],
        current_generation_identity=generation_identity, current_policy_identity=policy_identity,
        no_decoder=True, no_router_inference=True, no_generator=True, no_metric_recalculation=True,
        original_records_unchanged=True, sources_checked=True, models_checked=True,
        pixel_hashes_checked=True, points_detail=point_summaries,
        file_sha256=dict(sorted(reader.hashes.items())),
        unique_video_files_checked=len(reader.videos))
    target = root/'audit.json'
    if target.exists():
        with target.open() as stream:
            previous = json.load(stream)
        require(previous == result, 'audit evidence changed; preserve the existing audit and investigate')
        return previous
    with tempfile.NamedTemporaryFile(mode='w', dir=root, prefix='audit.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get('TMUX')), 'run full artifact audits in tmux')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'set CUDA_VISIBLE_DEVICES to the empty string')
    # Heartbeats go to the tmux console only, not the original experiment files.
    from demo.scalable_experiment import resources
    stopped = threading.Event()
    began = time.monotonic()
    state = {'phase': 'starting'}
    def heartbeat():
        while not stopped.wait(30):
            print(json.dumps(dict(audit=state['phase'], elapsed_seconds=time.monotonic()-began,
                                  resources=resources())), flush=True)
    def progress(message):
        state['phase'] = message
        print('AUDIT '+message, flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        result = audit(args.output, smoke_root=args.smoke_root, progress=progress)
    finally:
        stopped.set()
        thread.join(timeout=3)
    print(json.dumps({k: result[k] for k in ('complete', 'mode', 'points', 'figures', 'prefix_pairs')}))


if __name__ == '__main__':
    main()
