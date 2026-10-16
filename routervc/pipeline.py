"""Source-aware sending and source-free receiving of the final RouterVC chain."""

import gc
import time
import numpy as np
import torch
from routervc.assets import Assets
from routervc.codec import precision, format as packets
from routervc.routers import sender, receiver, routing
from routervc.fusion.render import render
from routervc.utils.arrays import frame_hash
from routervc.utils.precision import codec_precision
from routervc import stream


@torch.no_grad()
def ordering(source, bank, model, max_g=8, progress=lambda **kw: None):
    base, full, detail = precision.decode(bank)
    costs = np.asarray(packets.bundle_bytes(bank), np.int64)
    order, trace = [], []
    for step in range(16):
        inner = packets.subset(bank, order)
        b, received, current = precision.decode(inner)
        if (
            not np.array_equal(b, base)
            or current["base_reference_hashes"] != detail["base_reference_hashes"]
        ):
            raise ValueError("enhancement changed the base reference chain")
        coverage = routing.coverage(packets.parse(inner))
        prediction = sender.predict(
            model, source, base, received, full, coverage, costs, max_g
        )[0].numpy()
        ranked = sender.rank_candidates(prediction, costs, coverage)
        trace.append(
            dict(selected=order.copy(), predictions=prediction.tolist(), ranked=ranked)
        )
        if not ranked:
            break
        order.append(ranked[0])
        progress(phase="sender_order", selected=order.copy())
    return dict(order=order, trace=trace, bundle_bytes=costs.tolist())


@torch.no_grad()
def encode(
    source,
    width=3,
    e_ratio=0.5,
    max_g=8,
    variant="formal",
    seed=20261008,
    all_enhancements=False,
    progress=lambda **kw: None,
):
    if len(source) != 17 or not 0 <= e_ratio <= 1:
        raise ValueError("17-frame source and E ratio in [0,1] required")
    began = time.monotonic()
    assets = Assets()
    with codec_precision():
        codec = precision.PrecisionCodec()
        bank, base, full, detail = codec.encode(source, width)
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = sender.load_model(
        assets.path("sender"), expected_sha256=assets.sha("sender")
    )
    if all_enhancements:
        record = dict(
            order=list(range(16)), bundle_bytes=packets.bundle_bytes(bank), trace=[]
        )
    elif e_ratio == 0:
        record = dict(order=[], bundle_bytes=packets.bundle_bytes(bank), trace=[])
    else:
        record = ordering(source, bank, model, max_g, progress)
    costs = np.asarray(record["bundle_bytes"], np.int64)
    selected = sender.prefix_under_budget(
        record["order"], costs, int(costs.sum() * e_ratio)
    )
    inner = packets.subset(bank, selected["indices"])
    data = stream.wrap(
        inner, assets.sha("receiver"), assets.generator_identity(variant), max_g, seed
    )
    return data, dict(
        width=width,
        E_ratio=e_ratio,
        **selected,
        order=record,
        actual_bytes=len(data),
        header_bytes=stream.HEADER_BYTES,
        mask_bytes=0,
        base_bytes=len(packets.subset(bank, [])),
        bpp=len(data) * 8 / int(np.prod(source.shape[:3])),
        seconds=time.monotonic() - began,
        generation_executed_at_sender=False,
        source_hash=frame_hash(source),
        codec=detail
    )


@torch.no_grad()
def decode(
    data, variant="formal", disable_generation=False, progress=lambda **kw: None
):
    began = time.monotonic()
    assets = Assets()
    inner, config = stream.parse(data)
    torch.cuda.reset_peak_memory_stats()
    base, received, detail = precision.decode(inner)
    peak = torch.cuda.max_memory_allocated()
    patches, runtimes, selection = [], [], None
    if not disable_generation:
        if assets.generator_identity(variant) != config["assets_sha256"]:
            raise ValueError("wrong G variant/shared assets")
        model, _ = receiver.load(assets.path("receiver"), config["receiver_sha256"])
        selection = receiver.select(
            model,
            base,
            received,
            routing.coverage(packets.parse(inner)),
            config["max_g"],
        )
        if selection["indices"]:
            from routervc.generation.engine import Generator

            generator = Generator(assets, variant)
            for region in selection["indices"]:
                patch, runtime = generator(received, region, config["seed"])
                patches.append(patch)
                runtimes.append(runtime)
                peak = max(peak, torch.cuda.max_memory_allocated())
                progress(
                    phase="generation",
                    completed=len(patches),
                    total=len(selection["indices"]),
                )
            del generator
            gc.collect()
            torch.cuda.empty_cache()
    result = (
        render(received, patches, selection["indices"]) if patches else received.copy()
    )
    torch.cuda.synchronize()
    report = dict(
        actual_bytes=len(data),
        bpp=len(data) * 8 / int(np.prod(base.shape[:3])),
        header_bytes=stream.HEADER_BYTES,
        mask_bytes=0,
        source_frames_read=False,
        sender_router_loaded=False,
        generation_disabled=disable_generation,
        selection=selection,
        base_hash=frame_hash(base),
        enhanced_hash=frame_hash(received),
        output_hash=frame_hash(result),
        detail=detail,
        generation_runtime=runtimes,
        seconds=time.monotonic() - began,
        peak_cuda_allocated_bytes=peak,
        G_variant=variant,
    )
    return base, received, result, report
