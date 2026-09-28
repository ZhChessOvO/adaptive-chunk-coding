"""Bounded paired development probe: same region, E payload and G configuration.

Only this evaluator reads source pixels. Every measurement uses a fresh receiver;
completed points and prepared streams are checksummed before resuming.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import compact_enhancement_format as compact
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_codec import configure_torch, decode_features, encode_enhancement, load_model
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.chunk_enhancement_experiment import Run, codec, fresh_decode, read, MECHANISM
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_experiment import load_source, quality, now, resources
from demo.scalable_format import parse, frame_hash
from demo.scalable_cooperation_decode import identities, PATCH
from demo.scalable_generation_experiment import generated_decode as legacy_decode
from demo.stage_c_three_path_roi_probe import LPIPSAlex

RUNS = Path("/root/autodl-fs/DCVC/runs")
EVAL = RUNS/"a800_patch_efficiency_20260926/evaluation_l4"
OLD = RUNS/"a800_scalable_generate_20260926"
DEFAULT = RUNS/"a800_scalable_cooperation_20260928"
WALL = [256,128,128,128]


def generated_decode(stream,directory,run,disabled=False):
    import os
    import signal
    import subprocess
    directory.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=1",
               str(REPO/"demo/scalable_cooperation_decode.py"),"--stream",str(stream),"--output",str(directory)]
    if disabled:
        command.append("--disable-generation")
    with (directory/"process.log").open("w") as log:
        child = subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                run.check()
                if time.monotonic()-started > 1800:
                    raise TimeoutError("one fresh receiver exceeded 30 minutes")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"fresh receiver failed: {directory/'process.log'}")
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise
    return load_frames(directory/"reconstruction.npz"),read(directory/"decode.json")


def prepare(root,reference,source,run):
    """Reuse every original packet; add a fixed wall ROI only to REDS000."""
    sid = reference["sample"]["sample_id"]
    dest = root/"prepared"
    dest.mkdir(exist_ok=True)
    record = dest/"complete.json"
    if record.exists():
        result = read(record)
        verify_artifacts(dest,result["artifacts"])
        if result["source_hash"] != frame_hash(source):
            raise RuntimeError("source changed on resume")
        return result
    verify_artifacts(EVAL/sid,reference["artifacts"])
    wall = sid == "mechanism-00-reds"
    rois = reference["rois"]+([WALL] if wall else [])
    initial = (EVAL/sid/"q1.acse").read_bytes()
    parsed = parse(initial)
    base = load_frames(MECHANISM/"samples"/sid/"base.npz","base")
    if frame_hash(base) != parsed.meta["base_rgb_sha256"]:
        raise RuntimeError("base pixels changed")
    if wall:
        run.update(phase="prepare_fixed_wall_packets",sample=sid)
        model,base_codec = load_model(PATCH),codec()
        actual,chunks = decode_features(base_codec,initial[:parsed.base_end])
        np.testing.assert_array_equal(actual,base)
        chunks = [v for v in chunks if v["start"] < 17]
    records = {}
    for q in (1.,2.,.5):
        name = f"q{q:g}"
        wire = (EVAL/sid/f"{name}.acse").read_bytes()
        expected = load_frames(EVAL/sid/name/"reconstruction.npz")
        original = parse(wire)
        if wire[:original.base_end] != initial[:parsed.base_end]:
            raise RuntimeError("q sweep changed shared base")
        if wall:
            prefix,wires,wall_pixels,_ = encode_enhancement(model,PATCH,wire[:original.base_end],
                source,base,chunks,[WALL],q,compact=True)
            if prefix != wire[:original.base_end]:
                raise RuntimeError("new ROI changed base container")
            next_id = max(p.meta["packet_id"] for p in original.packets)
            added = parse(prefix+b"".join(wires))
            for index,packet in enumerate(added.packets,1):
                wire += compact.packet_bytes(dict(packet.meta,packet_id=next_id+index),
                    packet.payload,codec=packet.meta["codec"])
            x,y,w,h = WALL
            expected[:17,y:y+h,x:x+w] = wall_pixels[:17,y:y+h,x:x+w]
        atomic_bytes(dest/f"{name}.acse",wire)
        atomic_npz(dest/f"{name}_expected.npz",reconstruction=expected)
        records[name] = dict(bytes=len(wire),output_hash=frame_hash(expected),
                             packets=len(parse(wire).packets),old_packet_bytes_unchanged=True)
    if wall:
        del model,base_codec,chunks,actual
        torch.cuda.set_stream(torch.cuda.default_stream())
        gc.collect()
        torch.cuda.empty_cache()
    atomic_npz(dest/"base_expected.npz",reconstruction=base)
    result = dict(rois=rois,source_hash=frame_hash(source),records=records,
        wall_roi_added=WALL if wall else None,
        artifacts={p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.name != "complete.json"})
    atomic_json(record,result)
    return result


def specifications(inner_bytes,control,*,diagnostics=False,smoke=False):
    wire = inner_bytes["q1"]
    parsed = parse(wire)
    base = wire[:parsed.base_end]
    midpoint = math.ceil(len(parsed.packets)/2)
    partial = wire[:parsed.packets[midpoint-1].end_offset]
    specs = []
    def add(name,data,generated=False,disabled=False,expected="base"):
        specs.append(dict(name=name,data=data,generated=generated,disabled=disabled,expected=expected))
    add("base",base)
    add("enhance_q1",wire,expected="q1")
    for key,strength in (("l0",0.),("l025",.25),("l05",.5)):
        if smoke and key != "l05":
            continue
        c = dict(control,strength=strength)
        add("generate_"+key,fmt.wrap(base,c),True)
        add("cooperate_"+key,fmt.wrap(wire,c),True,expected="q1")
    if not smoke:
        add("prefix_enhance",partial,expected=None)
        add("prefix_cooperate",fmt.wrap(partial,control),True,expected="prefix")
    add("cooperate_repeat",fmt.wrap(wire,control),True,expected="q1")
    add("generation_off",fmt.wrap(wire,control),True,True,expected="q1")
    if smoke or diagnostics:
        add("base_generation_off",fmt.wrap(base,control),True,True)
        add("zero_blend",fmt.wrap(wire,dict(control,blend=0.)),True,expected="q1")
    if diagnostics and not smoke:
        for suffix,changes in (("context128",dict(context=128)),("scale2",dict(processing_scale=2))):
            c = dict(control,**changes)
            add("generate_"+suffix,fmt.wrap(base,c),True)
            add("cooperate_"+suffix,fmt.wrap(wire,c),True,expected="q1")
        for q in ("q2","q0.5"):
            add("enhance_"+q,inner_bytes[q],expected=q)
            add("cooperate_"+q,fmt.wrap(inner_bytes[q],control),True,expected=q)
    return specs


def region_metrics(source,output,regions,metric):
    result = [quality(source[fmt.region_slice(r)],output[fmt.region_slice(r)],metric) for r in regions]
    sizes = [r[1]*r[4]*r[5] for r in regions]
    pooled = {k:float(np.average([v[k] for v in result],weights=sizes)) for k in result[0]}
    return result,pooled


def evaluate_point(root,spec,source,base,expected,regions,metric,run):
    name = spec["name"]
    stream = root/(name+(".acsg" if spec["generated"] else ".acse"))
    data = spec["data"]
    if stream.exists() and stream.read_bytes() != data:
        raise RuntimeError("stream changed while resuming")
    atomic_bytes(stream,data)
    dest = root/name
    dest.mkdir(exist_ok=True)
    record = dest/"point.json"
    if record.exists():
        point = read(record)
        verify_artifacts(dest,point["artifacts"])
        if point["stream_sha256"] != file_hash(stream):
            raise RuntimeError("resume stream differs")
        actual,report = load_frames(dest/"reconstruction.npz"),read(dest/"decode.json")
    else:
        started = time.monotonic()
        if spec["generated"]:
            actual,report = generated_decode(stream,dest,run,spec["disabled"])
        else:
            actual,report = fresh_decode(stream,PATCH,dest)
        process_seconds = time.monotonic()-started
        local,pooled = region_metrics(source,actual,regions,metric)
        point = dict(bytes=len(data),stream_sha256=file_hash(stream),
            quality=quality(source,actual,metric),per_region=local,roi_quality=pooled,
            process_wall_seconds=process_seconds,fresh_decode=report,
            artifacts={f:file_hash(dest/f) for f in ("reconstruction.npz","decode.json")})
    if (report["source_frames_read"] or report["base_hash"] != frame_hash(base)
            or report["total_bytes"] != len(data) or report["output_hash"] != frame_hash(actual)
            or report != point["fresh_decode"]):
        raise RuntimeError("fresh decode integrity failed")
    if spec["generated"]:
        c,_,_,ncontrol = fmt.parse(data)
        alpha = fmt.weights(base.shape,c)
        if (report["generation_input_hash"] != frame_hash(expected)
                or report["generation_control_bytes"] != ncontrol
                or not report["base_reference_unchanged"]):
            raise RuntimeError("G did not receive the expected E reconstruction")
        np.testing.assert_array_equal(actual[alpha == 0],expected[alpha == 0])
        if spec["disabled"] or not np.any(alpha):
            np.testing.assert_array_equal(actual,expected)
            if report["generation_executed"] or report["generation_runtime"] is not None:
                raise RuntimeError("disabled generation executed")
    elif expected is not None:
        np.testing.assert_array_equal(actual,expected)
    atomic_json(record,point)
    print(json.dumps(dict(variant=name,bytes=len(data),roi=point["roi_quality"])),flush=True)
    return actual,point


def experiment(args,run):
    configure_torch()
    previous = read(EVAL/"summary.json")
    hashes = identities()
    protocol = dict(version=2,smoke=args.smoke,samples=[r["sample"] for r in previous["results"]],
        upstream_summary_sha256=file_hash(EVAL/"summary.json"),patch_sha256=file_hash(PATCH),assets=hashes,
        data_role="four previously used REDS/UVG development clips, not independent evaluation",
        regions="same old two E rectangles; add [256,128,128,128] brick wall to REDS000",
        E="A existing q1 packets (additional wall packets encoded once); first 17 frames",
        G="same spatial rectangles as E; all frames; shared settings and seeds in each pair",
        generation=dict(seed=260928,strength=.5,window=17,stride=8,context=64,feather=16,
                        processing_scale=1,blend=1.),
        lora_strengths=[0.,.25,.5],diagnostic_samples=["mechanism-00-reds","mechanism-03-uvg"],
        diagnostics="one variable at a time: context128, scale2 (native output), q2/q0.5",
        masks="explicit charged diagnostic controls, not a trained receiver-derived policy",
        lpips="Alex CPU; native independently cropped regions first 17 frames; nonadditive globally",
        code={str(p.relative_to(REPO)):file_hash(p) for p in
              (Path(__file__),REPO/"demo/scalable_generation_experiment.py",
               REPO/"demo/chunk_enhancement_codec.py",REPO/"demo/compact_enhancement_format.py",
               REPO/"demo/stage_c_three_path_roi_probe.py",REPO/"demo/run_scalable_cooperation.sh")})
    pp = args.output/"protocol.json"
    if pp.exists() and read(pp) != protocol:
        raise RuntimeError("protocol/code changed; use separate outputs")
    atomic_json(pp,protocol)
    metric,rows = LPIPSAlex(True),[]
    selected = previous["results"][:1] if args.smoke else previous["results"][:args.limit or None]
    for index,reference in enumerate(selected):
        run.check()
        sample = reference["sample"]
        sid = sample["sample_id"]
        root = args.output/sid
        root.mkdir(exist_ok=True)
        source = load_source(sample)
        prepared = prepare(root,reference,source,run)
        n = len(source)
        regions = [[0,17,*r] for r in prepared["rois"]]
        control = dict(protocol["generation"],**hashes,
                       generate=[[0,n,*r] for r in prepared["rois"]],protect=[])
        wires = {q:(root/"prepared"/f"{q}.acse").read_bytes() for q in ("q1","q2","q0.5")}
        base = load_frames(root/"prepared/base_expected.npz")
        expected = {q:load_frames(root/"prepared"/f"{q}_expected.npz") for q in wires}
        expected["base"] = base
        specs = specifications(wires,control,diagnostics=sid in protocol["diagnostic_samples"],smoke=args.smoke)
        points,arrays = {},{}
        for number,spec in enumerate(specs):
            run.check()
            run.update(phase="fresh_decode",sample=sid,variant=spec["name"],
                       completed=index,total=len(selected),point=number,total_points=len(specs))
            actual,point = evaluate_point(root,spec,source,base,expected.get(spec["expected"]),
                                         regions,metric,run)
            points[spec["name"]] = point
            if spec["name"] == "prefix_enhance":
                expected["prefix"] = actual
            if spec["name"] in ("cooperate_l05","cooperate_repeat","enhance_q1","generation_off"):
                arrays[spec["name"]] = actual
        np.testing.assert_array_equal(arrays["cooperate_l05"],arrays["cooperate_repeat"])
        np.testing.assert_array_equal(arrays["enhance_q1"],arrays["generation_off"])
        if not (root/"cooperate_l05.acsg").read_bytes().startswith((root/"generate_l05.acsg").read_bytes()):
            raise RuntimeError("G-only is not literal base prefix of E->G")
        if not args.smoke and not (root/"cooperate_l05.acsg").read_bytes().startswith((root/"prefix_cooperate.acsg").read_bytes()):
            raise RuntimeError("middle cooperative stream not a prefix")
        row = dict(sample=sample,rois=prepared["rois"],metric_regions=regions,points=points,
            source_hash=frame_hash(source),prepared=prepared,
            checks=dict(same_base=True,same_E_payload=True,repeat_pixel_exact=True,
                        generation_off_exact=True,unselected_exact=True,literal_prefix=True))
        atomic_json(root/"result.json",row)
        rows.append(row)
        atomic_json(args.output/"summary.json",dict(complete=False,results=rows,protocol=protocol,
            utc=now(),seconds=time.monotonic()-run.started,resources=resources()))
        run.update(phase="sample_complete",completed=index+1,total=len(selected))
    # One legacy fresh decode remains byte/pixel-compatible with its original profile.
    legacy_sid = previous["results"][0]["sample"]["sample_id"]
    dest = args.output/"legacy_v1_regression"
    if not (dest/"complete.json").exists():
        output,report = legacy_decode(OLD/legacy_sid/"combined.acsg",dest,run)
        np.testing.assert_array_equal(output,load_frames(OLD/legacy_sid/"combined/reconstruction.npz"))
        atomic_json(dest/"complete.json",dict(exact=True,report=report,
            artifacts={f:file_hash(dest/f) for f in ("reconstruction.npz","decode.json")}))
    else:
        verify_artifacts(dest,read(dest/"complete.json")["artifacts"])
    complete = args.smoke or len(rows) == 4
    summary = dict(complete=complete,smoke=args.smoke,results=rows,protocol=protocol,
        utc=now(),seconds=time.monotonic()-run.started,resources=resources(),
        legacy_v1_regression=read(dest/"complete.json"))
    atomic_json(args.output/"summary.json",summary)
    if complete:
        atomic_bytes(args.output/"experiment.complete",b"complete\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=DEFAULT)
    p.add_argument("--max-hours",type=float,default=6)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    args.command = "cooperation-probe"
    run = Run(args)
    run.thread.start()
    try:
        run.log_resources()
        with exclusive_native_evaluation(run):
            experiment(args,run)
        run.update(phase="complete")
    except BaseException as error:
        atomic_json(args.output/"last_error.json",dict(utc=now(),error=repr(error)))
        raise
    finally:
        run.stop.set()
        run.thread.join(timeout=2)
        run.log_resources()
        run.lock.close()
