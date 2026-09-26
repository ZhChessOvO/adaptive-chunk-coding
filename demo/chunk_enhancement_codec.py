"""Real-base feature extraction and a source-free regional enhancement receiver."""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from demo.scalable_codec import BaseCodec, atomic_bytes, file_hash
from demo.scalable_format import base_container, frame_hash, packet_bytes, parse
from demo.chunk_enhancement_model import ChunkEnhancement


def configure_torch():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def atomic_torch(path, data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    atomic_bytes(Path(path), buffer.getvalue())


@torch.inference_mode()
def decode_features(codec: BaseCodec, container: bytes, *, mutate_copies=False):
    from demo.stage_c_three_path_roi_probe import decode_dcvc_stream

    parsed = parse(container)
    for k, value in codec.models.items():
        if parsed.meta[k] != value:
            raise ValueError("base checkpoint does not match stream")
    count = parsed.meta["frame_count"]
    chunks = []

    def observe(start, sps, outputs):
        valid = min(len(outputs), count - start)
        if start == 0:
            # Explicit I-frame policy: replicated RGB, no pretend P8 feature.
            h8, w8 = (sps["height"] + 63) // 64 * 8, (sps["width"] + 63) // 64 * 8
            features = torch.zeros(1, 1024, h8, w8, dtype=torch.float16)
        else:
            feature, context = codec.p_net.proxy.get_decoded_features()
            features = torch.cat((feature, context), 1).contiguous().cpu()
            if mutate_copies:
                feature.zero_()
                context.zero_()
                again_f, again_c = codec.p_net.proxy.get_decoded_features()
                torch.testing.assert_close(features, torch.cat((again_f, again_c), 1).cpu(),
                                           rtol=0, atol=0)
        chunks.append({"start": start, "count": valid, "features": features})

    torch.cuda.set_stream(codec.stream)
    frames = decode_dcvc_stream(parsed.base, count, codec.i_net, codec.p_net,
                                codec.device, chunk_observer=observe)
    torch.cuda.synchronize()
    base = np.stack(frames)
    if frame_hash(base) != parsed.meta["base_rgb_sha256"]:
        raise RuntimeError("feature observation changed original base reconstruction")
    return base, chunks


def pack_region(frames, start, count, roi, device, alignment=64):
    if alignment not in (16, 64) or not 1 <= count <= 8:
        raise ValueError("invalid spatial alignment or chunk length")
    x, y, w, h = roi
    data = np.ascontiguousarray(frames[start:start+count, y:y+h, x:x+w])
    if data.shape != (count, h, w, 3):
        raise ValueError("region outside video")
    t = torch.from_numpy(data).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
    if count < 8:
        t = torch.cat((t, t[-1:].expand(8-count, -1, -1, -1)), 0)
    t = t.reshape(1, 24, h, w)
    return F.pad(t, (0, -w % alignment, 0, -h % alignment), mode="replicate")


def region_features(chunk, roi, device, halo=0, alignment=64):
    x, y, w, h = roi
    if x % 8 or y % 8:
        raise ValueError("feature-conditioned region origin must align to 8 pixels")
    if halo:
        if halo < 0:
            raise ValueError("negative feature halo")
        # Extract real neighboring base features, including the padded ROI.
        # At full-frame boundaries the external feature halo is zero-filled.
        grid = chunk["features"]
        left, top = x//8-halo, y//8-halo
        right = x//8+(w+alignment-1)//alignment*(alignment//8)+halo
        bottom = y//8+(h+alignment-1)//alignment*(alignment//8)+halo
        gh, gw = grid.shape[-2:]
        f = grid[..., max(top, 0):min(bottom, gh), max(left, 0):min(right, gw)]
        return F.pad(f.to(device, torch.float32),
                     (max(0, -left), max(0, right-gw), max(0, -top), max(0, bottom-gh)))
    h8, w8 = (h + 7) // 8, (w + 7) // 8
    f = chunk["features"][:, :, y//8:y//8+h8, x//8:x//8+w8].to(device, torch.float32)
    if f.shape[-2:] != (h8, w8):
        raise ValueError("region outside decoded feature grid")
    return F.pad(f, (0, (-w % alignment + w)//8 - w8, 0, (-h % alignment + h)//8 - h8), mode="replicate")


def unpack_region(tensor, count, roi):
    _, _, w, h = roi
    data = tensor.reshape(8, 3, *tensor.shape[-2:])[:count, :, :h, :w]
    return data.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def load_model(path, device="cuda:0"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from demo.feature_head_enhancement import FeatureHeadEnhancement, Pad16FeatureHeadEnhancement
    architectures = {ChunkEnhancement.FORMAT: ChunkEnhancement,
                     FeatureHeadEnhancement.FORMAT: FeatureHeadEnhancement,
                     Pad16FeatureHeadEnhancement.FORMAT: Pad16FeatureHeadEnhancement}
    if checkpoint.get("format") not in architectures:
        raise ValueError("not a chunk enhancement checkpoint")
    model = architectures[checkpoint["format"]](**checkpoint["model_config"]).to(device)
    if hasattr(model, "load_export_state"):
        model.load_export_state(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def encode_enhancement(model, model_path, original_container, source, base, chunks, rois, qstep,
                       *, compact=False):
    parsed = parse(original_container)
    if parsed.packets:
        raise ValueError("expected base-only input")
    meta = dict(parsed.meta, enhancement_codec=model.FORMAT,
                enhancement_model_sha256=file_hash(model_path), feature_format="uf_hts_F_ctx_v1")
    if compact:
        from demo.compact_enhancement_format import base_container as compact_base, packet_bytes as compact_packet
        prefix = compact_base(parsed.base, meta)
    else:
        prefix = base_container(parsed.base, meta)
    alignment = model.spatial_alignment
    wires, details = [], []
    result = base.copy()
    device = next(model.parameters()).device
    for chunk in chunks:
        start, count = chunk["start"], chunk["count"]
        for roi in rois:
            x, y, w, h = roi
            for previous in details:
                ox, oy, ow, oh = previous["roi"]
                if previous["start"] == start and max(x, ox) < min(x+w, ox+ow) and max(y, oy) < min(y+h, oy+oh):
                    raise ValueError("overlapping single-layer enhancement regions")
            bottom = pack_region(base, start, count, roi, device, alignment)
            target = pack_region(source, start, count, roi, device, alignment)
            feature = region_features(chunk, roi, device, getattr(model, "feature_halo", 0), alignment)
            payload, recon, stats = model.compress(target, bottom, feature, qstep, count)
            pm = {"packet_id": len(wires)+1, "codec": model.FORMAT,
                  "start": start, "count": count, "roi": list(roi), "qstep": qstep}
            wire = compact_packet(pm, payload, codec=model.FORMAT) if compact else packet_bytes(pm, payload)
            wires.append(wire)
            result[start:start+count, y:y+h, x:x+w] = unpack_region(recon, count, roi)
            details.append(dict(pm, **stats, packet_bytes=len(wire),
                                framing_bytes=len(wire)-len(payload)))
    return prefix, wires, result, details


@torch.no_grad()
def decode_enhancement(model, model_path, codec, data, *, allow_incomplete_tail=False,
                       return_base=False):
    parsed = parse(data, allow_incomplete_tail=allow_incomplete_tail)
    if (parsed.meta.get("enhancement_codec") != model.FORMAT
            or parsed.meta.get("enhancement_model_sha256") != file_hash(model_path)
            or parsed.meta.get("feature_format") != "uf_hts_F_ctx_v1"):
        raise ValueError("enhancement codec/weights/features do not match stream")
    base, chunks = decode_features(codec, data[:parsed.base_end])
    by_start = {c["start"]: c for c in chunks}
    output = base.copy()
    occupied = np.zeros(base.shape[:3], dtype=bool)
    device = next(model.parameters()).device
    applied = []
    for packet in parsed.packets:
        m = packet.meta
        if m.get("codec") != model.FORMAT:
            raise ValueError("unsupported enhancement payload")
        start, count, roi, q = m.get("start"), m.get("count"), m.get("roi"), m.get("qstep")
        if (type(start) is not int or start not in by_start or type(count) is not int
                or count != by_start[start]["count"] or not isinstance(roi, list) or len(roi) != 4
                or any(type(v) is not int for v in roi)
                or type(q) not in (int, float) or not np.isfinite(q) or not 0.125 <= q <= 8):
            raise ValueError("invalid packet geometry or quality")
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) < 1 or x+w > base.shape[2] or y+h > base.shape[1]:
            raise ValueError("invalid region bounds")
        region = np.s_[start:start+count, y:y+h, x:x+w]
        if occupied[region].any():
            raise ValueError("duplicate/overlapping single-layer region")
        bottom = pack_region(base, start, count, roi, device, model.spatial_alignment)
        feature = region_features(by_start[start], roi, device, getattr(model, "feature_halo", 0), model.spatial_alignment)
        result = model.decompress(packet.payload, bottom, feature, q, count)
        output[region] = unpack_region(result, count, roi)
        occupied[region] = True
        applied.append(m["packet_id"])
    if not np.array_equal(output[~occupied], base[~occupied]):
        raise RuntimeError("unselected pixel changed")
    report = {"source_frames_read": False, "applied_packets": applied,
              "base_bytes": len(parsed.base), "container_header_bytes": parsed.base_end-len(parsed.base),
              "packet_bytes": sum(len(p.wire) for p in parsed.packets),
              "incomplete_tail_bytes": parsed.incomplete_tail_bytes, "total_bytes": len(data),
              "base_hash": frame_hash(base), "output_hash": frame_hash(output),
              "non_enhanced_exact": True}
    if sum(report[k] for k in ("base_bytes", "container_header_bytes", "packet_bytes", "incomplete_tail_bytes")) != len(data):
        raise RuntimeError("byte accounting mismatch")
    return (output, report, base) if return_base else (output, report)
