"""RouterVC sender: immutable UF base, real candidate E packets, learned allocation.

The source is used only by the codec. Router inputs are B, decoded candidate Y
and received-E coverage. Generation is NOT encoded as a region mask here: the
receiver will recompute it from the actual mixed Y and its shared policy.

``independent`` optimizes the isolated four-state utility table separately at
each budget. ``prefix`` uses a budget-independent greedy order and transmits a
literal byte prefix of whole region bundles. These are different allocations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash, parse, sha256

DEFAULT_ENHANCEMENT = Path('/root/autodl-fs/DCVC/runs/a800_online_eg_20261002/joint/enhancement.pt')
DEFAULT_ROUTER = Path('/root/autodl-fs/DCVC/runs/a800_four_state_router_20261002/context/model.pt')


def _read(path):
    return json.loads(Path(path).read_text())


def validate_frames(frames):
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8
            or frames.ndim != 4 or frames.shape[-1] != 3):
        raise ValueError('expected uint8 RGB [T,H,W,3]')
    t, h, w, _ = frames.shape
    if t < 17 or (t-1) % 8 or t > 100000:
        raise ValueError('RouterVC currently requires T >= 17 and T = 1 + 8*n; no silent tail padding')
    if min(h, w) < 64 or max(h, w) > 8192 or h % 8 or w % 8:
        raise ValueError('RouterVC requires H/W divisible by 8, between 64 and 8192')
    return t, h, w


def load_input(path, *, key='source', start=0, count=None):
    """Explicit PNG directory or RGB npz input; no implicit resize/crop/tail pad."""
    path = Path(path)
    if type(start) is not int or start < 0 or (count is not None and (type(count) is not int or count < 1)):
        raise ValueError('invalid source frame selection')
    if path.is_dir():
        from PIL import Image
        files = sorted(path.glob('*.png'))
        selected = files[start:start+count if count is not None else None]
        if not selected or (count is not None and len(selected) != count):
            raise ValueError('source directory has fewer requested PNG frames')
        frames = []
        for item in selected:
            with Image.open(item) as image:
                frames.append(np.asarray(image.convert('RGB')).copy())
        if any(f.shape != frames[0].shape for f in frames):
            raise ValueError('source PNG dimensions differ')
        pixels = np.stack(frames)
        provenance = dict(kind='png_directory', path=str(path.resolve()),
                          files={str(p.resolve()): file_hash(p) for p in selected})
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as value:
            if key not in value:
                raise ValueError(f'npz does not contain explicit key {key!r}')
            pixels = value[key][start:start+count if count is not None else None].copy()
        if count is not None and len(pixels) != count:
            raise ValueError('npz has fewer requested frames')
        provenance = dict(kind='npz', path=str(path.resolve()), key=key, file_sha256=file_hash(path))
    else:
        raise ValueError('source must be a PNG directory or .npz file')
    validate_frames(pixels)
    return pixels, dict(provenance, start=start, requested_count=count,
                        source_shape=list(pixels.shape), source_rgb_sha256=frame_hash(pixels))


def bank_info(bank):
    """Validate a full bank of 16 independent, single-layer region bundles."""
    from demo.routervc_policy import grid_rois
    parsed = parse(bank)
    t, h, w = (parsed.meta[k] for k in ('frame_count', 'height', 'width'))
    if t < 17 or (t-1) % 8:
        raise ValueError('unsupported bank length; expected 1 + 8*n frames, at least 17')
    rois = [list(r) for r in grid_rois(h, w)]
    expected = [(0, 1)] + [(start, 8) for start in range(1, t, 8)]
    groups = [[] for _ in rois]
    for packet in parsed.packets:
        pm = packet.meta
        if pm['roi'] not in rois or pm['qstep'] != 1.:
            raise ValueError('bank contains non-grid ROI or non-q1 packet')
        groups[rois.index(pm['roi'])].append(packet)
    for group in groups:
        actual = [(p.meta['start'], p.meta['count']) for p in group]
        if actual != expected:
            raise ValueError('each candidate must contain the complete ordered I + P8 region bundle')
    return dict(parsed=parsed, rois=rois, groups=groups,
                e_bytes=[sum(len(p.wire) for p in g) for g in groups], chunks=expected)


def subset_bank(bank, indices):
    """Append exact original entropy packets, in explicit whole-region order."""
    info = bank_info(bank)
    indices = list(indices)
    if (len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 16 for i in indices)):
        raise ValueError('invalid or duplicate E region index')
    prefix = bank[:info['parsed'].base_end]
    return prefix + b''.join(p.wire for i in indices for p in info['groups'][i])


def compose_candidates(base, all_e, indices, rois):
    """E packets are independent display corrections, not a new UF reference."""
    result = base.copy()
    for i in indices:
        x, y, w, h = rois[i]
        result[:, y:y+h, x:x+w] = all_e[:, y:y+h, x:x+w]
    return result


def _validate_candidates(bank, base, all_e):
    validate_frames(base)
    validate_frames(all_e)
    info = bank_info(bank)
    m = info['parsed'].meta
    if base.shape != all_e.shape or base.shape != (m['frame_count'], m['height'], m['width'], 3):
        raise ValueError('candidate reconstruction dimensions differ from bank')
    if frame_hash(base) != m['base_rgb_sha256']:
        raise ValueError('candidate base does not match the actual bank reconstruction')
    return info


def _verify_record(folder, record, binding):
    if record['binding'] != binding:
        raise RuntimeError(f'changed sender inputs/configuration at {folder}; choose a new output')
    for name, digest in record['artifacts'].items():
        if file_hash(folder/name) != digest:
            raise RuntimeError(f'changed sender artifact: {folder/name}')


def prepare(input_path, output, *, enhancement=DEFAULT_ENHANCEMENT,
            model_i=REPO/'checkpoints/cvpr2026_image.pth.tar',
            model_p=REPO/'checkpoints/cvpr2026_video_hts.pth.tar',
            input_key='source', start=0, count=None):
    """Source -> QP8 UF -> q1 candidate bank -> entropy-decoded B/all_E cache.

    All files are atomic. Base and completed candidate phases are independently
    resumable. A long running caller must provide tmux, heartbeat and GPU mutex.
    This function never launches G, and does not read teacher quality labels.
    """
    from demo.chunk_enhancement_codec import (configure_torch, decode_features,
        encode_enhancement, decode_enhancement, load_model)
    from demo.scalable_codec import BaseCodec
    from demo.scalable_format import base_container
    from demo.scalable_experiment import check_space
    from demo.routervc_policy import grid_rois

    began = time.monotonic()
    output, enhancement, model_i, model_p = map(Path, (output, enhancement, model_i, model_p))
    output.mkdir(parents=True, exist_ok=True)
    source, provenance = load_input(input_path, key=input_key, start=start, count=count)
    # Reject unsupported Router topology before loading any GPU model.
    rois = grid_rois(source.shape[1], source.shape[2])
    binding = dict(version=1, source=provenance, source_code=file_hash(Path(__file__)),
                   enhancement=file_hash(enhancement), model_i=file_hash(model_i), model_p=file_hash(model_p),
                   base_qp=8, qstep=1., padded_frames=0)
    done = output/'complete.json'
    if done.exists():
        record = _read(done)
        _verify_record(output, record, binding)
        return record
    check_space()
    configure_torch()
    codec = BaseCodec(model_i, model_p)
    configure_torch()
    base_done = output/'base.complete.json'
    if base_done.exists():
        base_record = _read(base_done)
        _verify_record(output, base_record, binding)
        container = (output/'base.acse').read_bytes()
    else:
        phase = time.monotonic()
        raw, encode_details = codec.encode(source, qp=8)
        base = codec.decode(raw, len(source))
        container = base_container(raw, codec.metadata(source, base))
        atomic_bytes(output/'base.acse', container)
        base_record = dict(binding=binding, seconds=time.monotonic()-phase,
            native_encode=encode_details, base_rgb_sha256=frame_hash(base),
            artifacts={'base.acse': file_hash(output/'base.acse')})
        atomic_json(base_done, base_record)
    phase = time.monotonic()
    base, chunks = decode_features(codec, container)
    actual_chunks = [(c['start'], c['count']) for c in chunks]
    if actual_chunks != [(0, 1)] + [(s, 8) for s in range(1, len(source), 8)]:
        raise RuntimeError('UF feature chunks differ from the supported continuous I/P8 schedule')
    model = load_model(enhancement).requires_grad_(False)
    prefix, packets, encoded_e, details = encode_enhancement(model, enhancement,
        container, source, base, chunks, rois, 1., compact=True)
    bank = prefix + b''.join(packets)
    candidate_encode_seconds = time.monotonic()-phase
    phase = time.monotonic()
    all_e, decode_report, decoded_base = decode_enhancement(model, enhancement, codec, bank, return_base=True)
    if not np.array_equal(base, decoded_base) or not np.array_equal(encoded_e, all_e):
        raise RuntimeError('entropy-decoded E candidates differ from sender reconstruction')
    info = _validate_candidates(bank, base, all_e)
    atomic_bytes(output/'bank.acse', bank)
    atomic_npz(output/'candidates.npz', base=base, all_E=all_e)
    record = dict(complete=True, binding=binding, rois=info['rois'],
        actual_chunks=[list(v) for v in actual_chunks],
        padding=dict(temporal_frames=0, spatial_pixels=0, input_shape=list(source.shape)),
        base_container_bytes=info['parsed'].base_end, base_native_bytes=len(info['parsed'].base),
        e_packet_bytes=info['e_bytes'], candidate_bank_bytes=len(bank),
        base_rgb_sha256=frame_hash(base), all_E_rgb_sha256=frame_hash(all_e),
        packet_details=details, base_seconds=base_record['seconds'],
        candidate_encode_seconds=candidate_encode_seconds,
        candidate_decode_seconds=time.monotonic()-phase,
        wall_seconds_this_attempt=time.monotonic()-began, decoded_candidate_report=decode_report,
        artifacts={name: file_hash(output/name) for name in ('base.acse', 'bank.acse', 'candidates.npz')})
    atomic_json(done, record)
    return record


@torch.no_grad()
def predict_utility(bank, base, all_e, router):
    """Seventeen source-free views: B, then isolated decoded E at each cell."""
    from demo.routervc_policy import features, load_model
    info = _validate_candidates(bank, base, all_e)
    torch.set_num_threads(4)
    model = load_model(Path(router)) if isinstance(router, (str, Path)) else router
    model.eval()
    baseline = features(base, base, info['rois'], [False]*16)
    enhanced = features(base, all_e, info['rois'], [True]*16)
    views = baseline.repeat(17, 1, 1)
    for i in range(16):
        views[i+1, i] = enhanced[0, i]
    p = model(views)
    if p.shape != (17, 16, 6) or not torch.isfinite(p).all():
        raise RuntimeError('invalid Router predictions')
    index = torch.arange(16)
    direct = p[index+1, index, 0]
    utility = torch.stack((torch.zeros_like(direct), direct, p[0, :, 1], direct+p[index+1, index, 1]), dim=1)
    return utility.cpu().numpy(), p.cpu().numpy()


def _table_score(utility, chosen, max_g):
    selected = np.zeros(len(utility), dtype=bool)
    selected[chosen] = True
    direct = np.where(selected, utility[:, 1], utility[:, 0])
    conditional = np.where(selected, utility[:, 3]-utility[:, 1], utility[:, 2]-utility[:, 0])
    order = np.argsort(-conditional, kind='stable')
    generated = [int(i) for i in order[:max_g] if conditional[i] > 0]
    states = selected.astype(np.int64)
    states[generated] += 2
    return float(direct.sum()+conditional[generated].sum()), states.tolist()


def prefix_order(utility, e_bytes, max_g):
    """Budget-independent greedy marginal utility/byte rank (not exact RD)."""
    utility = np.asarray(utility, dtype=np.float64)
    e_bytes = np.asarray(e_bytes, dtype=np.int64)
    if (utility.shape != (len(e_bytes), 4) or not 1 <= len(e_bytes) <= 16
            or not np.isfinite(utility).all() or np.any(e_bytes <= 0)
            or type(max_g) is not int or not 0 <= max_g <= len(e_bytes)):
        raise ValueError('invalid progressive utility/costs/G budget')
    chosen, records = [], []
    score, _ = _table_score(utility, chosen, max_g)
    remaining = set(range(len(e_bytes)))
    while remaining:
        options = []
        for i in sorted(remaining):
            value, _ = _table_score(utility, chosen+[i], max_g)
            gain = value-score
            options.append((gain/e_bytes[i], -i, i, value, gain))
        _, _, i, new_score, gain = max(options)
        if gain <= 0:
            break
        chosen.append(i)
        remaining.remove(i)
        records.append(dict(index=i, marginal_utility=gain, packet_bytes=int(e_bytes[i]),
                            gain_per_byte=gain/e_bytes[i], cumulative_utility=new_score))
        score = new_score
    return chosen, records


def select(bank, base, all_e, router, extra_e_budget, max_g_calls, *, mode='independent'):
    """Choose E using LPIPS-primary predictions; no labels, source or G mask.

    The returned G states are an isolated-table *prediction only*. The actual
    receiver must decide G again on its mixed reconstruction. E budget excludes
    the shared RVC header; the enclosing sender counts that header separately.
    """
    from demo.four_state_router_evaluate import frontier, solve
    if (type(extra_e_budget) is not int or extra_e_budget < 0
            or type(max_g_calls) is not int or not 0 <= max_g_calls <= 16):
        raise ValueError('invalid E-byte/G-call budget')
    if mode not in ('independent', 'prefix'):
        raise ValueError('allocation must be independent or prefix')
    began = time.monotonic()
    info = _validate_candidates(bank, base, all_e)
    utility, prediction = predict_utility(bank, base, all_e, router)
    prediction_seconds = time.monotonic()-began
    phase = time.monotonic()
    rank, rank_details = [], []
    if mode == 'independent':
        plan = solve(frontier(utility, info['e_bytes'], 0, 0), extra_e_budget, max_g_calls)
        indices = [i for i, state in enumerate(plan['states']) if state in (1, 3)]
        score, states = plan['predicted_utility'], plan['states']
    else:
        rank, rank_details = prefix_order(utility, info['e_bytes'], max_g_calls)
        indices, cost = [], 0
        for i in rank:
            if cost + info['e_bytes'][i] > extra_e_budget:
                break  # skipping a large bundle would destroy byte-prefix nesting
            indices.append(i)
            cost += info['e_bytes'][i]
        score, states = _table_score(utility, indices, max_g_calls)
    inner = subset_bank(bank, indices)
    used = len(inner)-info['parsed'].base_end
    if used != sum(info['e_bytes'][i] for i in indices) or used > extra_e_budget:
        raise RuntimeError('serialized packet budget mismatch')
    mixed = compose_candidates(base, all_e, indices, info['rois'])
    return dict(mode=mode, selected_indices=indices, e_packet_bytes=used,
        budget_e_packet_bytes=extra_e_budget, unused_e_budget_bytes=extra_e_budget-used,
        max_g_calls=max_g_calls, provisional_states=states, predicted_utility=score,
        utility=utility.tolist(), prediction=prediction.tolist(),
        prefix_order=rank, prefix_ranking=rank_details,
        prediction_seconds=prediction_seconds, allocation_seconds=time.monotonic()-phase,
        mixed_rgb_sha256=frame_hash(mixed), inner_sha256=sha256(inner),
        inner_bytes=len(inner), base_container_bytes=info['parsed'].base_end,
        router_sha256=file_hash(Path(router)) if isinstance(router, (str, Path)) else None,
        source_frames_used_by_router=False, generation_mask_transmitted=False,
        g_policy='receiver redecides on actual mixed Y; provisional states are not transmitted',
        allocation_scope='isolated-table exact independent allocation' if mode == 'independent'
                         else 'greedy marginal gain/byte order; literal nested complete-region prefix',
        budget_scope='E packet bytes only; shared RouterVC header and base charged separately')


def select_files(bank_path, candidates_path, output, *, router=DEFAULT_ROUTER,
                 extra_e_budget, max_g_calls, mode='independent'):
    """Reuse a prepared or historical bank and matching decoded candidate npz."""
    bank_path, candidates_path, output, router = map(Path, (bank_path, candidates_path, output, router))
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(bank=file_hash(bank_path), candidates=file_hash(candidates_path), router=file_hash(router),
                   extra_e_budget=extra_e_budget, max_g_calls=max_g_calls, mode=mode,
                   source_code=file_hash(Path(__file__)))
    done = output/'selection.json'
    if done.exists():
        previous = _read(done)
        _verify_record(output, previous, binding)
        return previous
    with np.load(candidates_path, allow_pickle=False) as cache:
        base = cache['base'].copy()
        # The formal teacher calls this field "enhanced"; accept it explicitly.
        key = 'all_E' if 'all_E' in cache else 'enhanced'
        all_e = cache[key].copy()
    bank = bank_path.read_bytes()
    result = select(bank, base, all_e, router, extra_e_budget, max_g_calls, mode=mode)
    inner = subset_bank(bank, result['selected_indices'])
    atomic_bytes(output/'selected.acse', inner)
    result.update(binding=binding, artifacts={'selected.acse': file_hash(output/'selected.acse')})
    atomic_json(done, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='GPU sender; run under supervised tmux GPU mutex')
    prep.add_argument('--input', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--input-key', default='source')
    prep.add_argument('--start', type=int, default=0)
    prep.add_argument('--count', type=int)
    prep.add_argument('--enhancement', type=Path, default=DEFAULT_ENHANCEMENT)
    prep.add_argument('--model-i', type=Path, default=REPO/'checkpoints/cvpr2026_image.pth.tar')
    prep.add_argument('--model-p', type=Path, default=REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    allocation = commands.add_parser('select', help='CPU learned E allocation, no source read')
    allocation.add_argument('--bank', type=Path, required=True)
    allocation.add_argument('--candidates', type=Path, required=True)
    allocation.add_argument('--output', type=Path, required=True)
    allocation.add_argument('--router', type=Path, default=DEFAULT_ROUTER)
    allocation.add_argument('--e-budget', type=int, required=True)
    allocation.add_argument('--max-g', type=int, default=4)
    allocation.add_argument('--mode', choices=('independent', 'prefix'), default='independent')
    args = p.parse_args()
    if args.command == 'prepare':
        result = prepare(args.input, args.output, enhancement=args.enhancement,
            model_i=args.model_i, model_p=args.model_p, input_key=args.input_key,
            start=args.start, count=args.count)
    else:
        result = select_files(args.bank, args.candidates, args.output, router=args.router,
            extra_e_budget=args.e_budget, max_g_calls=args.max_g, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
