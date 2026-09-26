"""Fixed four-clip development ablation, actual control bits and fresh receivers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_generation_format as fmt
from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.chunk_enhancement_experiment import Run, fresh_decode, read, MECHANISM
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_experiment import load_source, quality, now, resources
from demo.scalable_format import parse, frame_hash
from demo.scalable_generation_decode import identities, PATCH, RUNS
from demo.stage_c_three_path_roi_probe import LPIPSAlex

EVAL = RUNS/"a800_patch_efficiency_20260926/evaluation_l4"
DEFAULT = RUNS/"a800_scalable_generate_20260926"


def local_quality(source,output,regions,metric):
    # Independently cropped LPIPS, no resize. Not an additive global decomposition.
    rows = [quality(source[fmt.region_slice(r)],output[fmt.region_slice(r)],metric) for r in regions]
    sizes = [r[1]*r[4]*r[5] for r in regions]
    return {k:float(np.average([v[k] for v in rows],weights=sizes)) for k in rows[0]}


def generated_decode(stream,directory,run,disabled=False):
    directory.mkdir(parents=True,exist_ok=True)
    args = [sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=1",
            str(REPO/"demo/scalable_generation_decode.py"),"--stream",str(stream),"--output",str(directory)]
    if disabled:
        args.append("--disable-generation")
    with (directory/"process.log").open("w") as log:
        child = subprocess.Popen(args,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                run.check()
                if time.monotonic()-started > 1800:
                    raise TimeoutError("one decoder exceeded 30 minutes")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"fresh decoder failed ({child.returncode}): {directory/'process.log'}")
        except BaseException:
            import os,signal
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise
    return load_frames(directory/"reconstruction.npz"),read(directory/"decode.json")


def visual(root,source,outputs,points,rois,gregion):
    h,w = source.shape[1:3]
    names = ["source","base","enhance","generate","combined","full_generate"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
    canvas = Image.new("RGB",(3*w,2*(h+68)),"white")
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(names):
        x,y = i%3*w,i//3*(h+68)
        pixels = source[8] if name == "source" else outputs[name][8]
        canvas.paste(Image.fromarray(pixels),(x,y+68))
        draw.text((x+4,y+3),name,font=font,fill="black")
        if name != "source":
            p = points[name]
            draw.text((x+4,y+24),f"{p['bytes']} B | LPIPS {p['quality']['lpips_alex']:.4f}",font=font,fill="black")
            draw.text((x+4,y+45),f"E-ROI LPIPS {p['enhance_roi']['lpips_alex']:.4f}",font=font,fill="black")
        else:
            draw.text((x+4,y+24),"E: orange boxes; G: cyan strip",font=font,fill="black")
            for xx,yy,ww,hh in rois:
                draw.rectangle((x+xx,y+68+yy,x+xx+ww-1,y+68+yy+hh-1),outline="orange",width=2)
            _,_,xx,yy,ww,hh = gregion
            draw.rectangle((x+xx,y+68+yy,x+xx+ww-1,y+68+yy+hh-1),outline="cyan",width=2)
    canvas.save(root/"fixed_frame.png")
    # Exact, enlarged crops for discussing structures; nearest-neighbor, no AI edits.
    regions = [("Enhance",rois[0]),("Generate",gregion[2:])]
    crop_canvas = Image.new("RGB",(6*256,2*300),"white")
    draw = ImageDraw.Draw(crop_canvas)
    for j,(label,(xx,yy,ww,hh)) in enumerate(regions):
        hh = min(hh,128)
        for i,name in enumerate(names):
            pixels = source[8] if name == "source" else outputs[name][8]
            image = Image.fromarray(pixels[yy:yy+hh,xx:xx+ww])
            image.thumbnail((256,256),Image.Resampling.NEAREST)
            scale = min(256/ww,256/hh)
            image = image.resize((round(ww*scale),round(hh*scale)),Image.Resampling.NEAREST)
            crop_canvas.paste(image,(i*256,j*300+40))
            draw.text((i*256+3,j*300+2),f"{label}: {name}",font=font,fill="black")
    crop_canvas.save(root/"fixed_crops.png")


def plot(root,rows):
    fig,axes = plt.subplots(len(rows),2,figsize=(12,3.4*len(rows)),squeeze=False)
    for row,axs in zip(rows,axes):
        points = row["points"]
        for ax,field,title in zip(axs,("quality","enhance_roi"),("Whole-frame LPIPS","Enhanced-region cropped LPIPS")):
            es = [points["base"]]+[row["local_patch_diagnostic"][q] for q in ("q2","q1","q0.5")]
            ax.plot([v["bytes"] for v in es],[v[field]["lpips_alex"] for v in es],"o-",label="Enhance (q re-encodings)")
            for name,mark in (("generate","s"),("combined","*"),("full_generate","^")):
                p = points[name]
                ax.scatter(p["bytes"],p[field]["lpips_alex"],marker=mark,s=65,label=name)
            uf = [row["native_uf"][k] for k in ("8","32")]
            ax.plot([p["bytes"] for p in uf],[p[field]["lpips_alex"] for p in uf],"x--",label="Native UF (2 references)")
            ax.set_title(row["sample"]["sample_id"]+" | "+title)
            ax.set_xlabel("Total bytes (including control)")
            ax.set_ylabel("LPIPS (lower is better)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root/"rd_diagnostic.png",dpi=140)
    plt.close(fig)


def experiment(args,run):
    configure_torch()
    previous = read(EVAL/"summary.json")
    hashes = identities()
    protocol = dict(version=1, samples=[r["sample"] for r in previous["results"]],
        upstream_summary_sha256=file_hash(EVAL/"summary.json"), patch_sha256=file_hash(PATCH), assets=hashes,
        data_role="four previously used development clips; fixed geometric actions, no learned router",
        E="existing two rectangles, first 17 frames, A/q1",G="rightmost quarter, all frames",
        generation=dict(seed=260926,strength=.5,window=17,stride=8,context=64,feather=16),
        lpips="Alex; whole-frame and separately cropped regions; no resizing; not additive",
        code={str(p.relative_to(REPO)):file_hash(p) for p in
              (Path(__file__),REPO/"demo/chunk_enhancement_codec.py")})
    pp = args.output/"protocol.json"
    if pp.exists() and read(pp) != protocol:
        raise RuntimeError("protocol/code changed; do not mix results")
    atomic_json(pp,protocol)
    metric = LPIPSAlex(True)
    rows = []
    for i,reference in enumerate(previous["results"][:args.limit or None]):
        run.check()
        sample = reference["sample"]
        sid = sample["sample_id"]
        root = args.output/sid
        root.mkdir(exist_ok=True)
        verify_artifacts(EVAL/sid,reference["artifacts"])
        source = load_source(sample)
        n,h,w = source.shape[:3]
        wire = (EVAL/sid/"q1.acse").read_bytes()
        parsed = parse(wire)
        base = load_frames(MECHANISM/"samples"/sid/"base.npz","base")
        if frame_hash(base) != parsed.meta["base_rgb_sha256"]:
            raise RuntimeError("base cache hash differs")
        e_regions = [[0,17,*r] for r in reference["rois"]]
        g_region = [0,n,3*w//4,0,w//4,h]
        control = dict(protocol["generation"],**hashes,generate=[g_region],
                       protect=[[0,n,*r] for r in reference["rois"]])
        full_control = dict(control,generate=[[0,n,0,0,w,h]],protect=[])
        streams = dict(base=wire[:parsed.base_end],enhance=wire,
            generate=fmt.wrap(wire[:parsed.base_end],control),combined=fmt.wrap(wire,control),
            full_generate=fmt.wrap(wire[:parsed.base_end],full_control))
        streams["combined_repeat"] = streams["combined"]
        streams["generation_disabled"] = streams["combined"]
        outputs,points = {},{}
        for name,data in streams.items():
            run.check()
            run.update(sample=sid,variant=name,completed=i,total=4,phase="fresh_decode")
            stream = root/(name+(".acse" if name in ("base","enhance") else ".acsg"))
            if stream.exists() and stream.read_bytes() != data:
                raise RuntimeError("materialized stream changed")
            atomic_bytes(stream,data)
            dest = root/name
            record = dest/"point.json"
            if record.exists():
                point = read(record)
                verify_artifacts(dest,point["artifacts"])
                if point["stream_sha256"] != file_hash(stream):
                    raise RuntimeError("resumed stream differs")
                actual = load_frames(dest/"reconstruction.npz")
                report = read(dest/"decode.json")
            else:
                if name in ("base","enhance"):
                    actual,report = fresh_decode(stream,PATCH,dest)
                else:
                    actual,report = generated_decode(stream,dest,run,disabled=name == "generation_disabled")
                point = dict(bytes=stream.stat().st_size,stream_sha256=file_hash(stream),
                    quality=quality(source,actual,metric),
                    enhance_roi=local_quality(source,actual,e_regions,metric),
                    generate_roi=local_quality(source,actual,[g_region],metric),
                    fresh_decode=report,artifacts={f:file_hash(dest/f) for f in ("reconstruction.npz","decode.json")})
            if (report["source_frames_read"] or report["base_hash"] != frame_hash(base)
                    or report["total_bytes"] != len(data) or report["output_hash"] != frame_hash(actual)):
                raise RuntimeError("fresh byte/hash/source validation failed")
            outputs[name],points[name] = actual,point
            atomic_json(record,point)
            print(json.dumps(dict(sample=sid,variant=name,bytes=len(data),quality=point["quality"])),flush=True)
        np.testing.assert_array_equal(outputs["base"],base)
        np.testing.assert_array_equal(outputs["enhance"],load_frames(EVAL/sid/"q1/reconstruction.npz"))
        np.testing.assert_array_equal(outputs["combined"],outputs["combined_repeat"])
        np.testing.assert_array_equal(outputs["enhance"],outputs["generation_disabled"])
        alpha = fmt.weights(base.shape,control,parsed)
        np.testing.assert_array_equal(outputs["combined"][alpha == 0],outputs["enhance"][alpha == 0])
        np.testing.assert_array_equal(outputs["combined"][alpha > 0],outputs["generate"][alpha > 0])
        # Actual combined stream base prefix equals the Generate-only stream byte for byte.
        if not streams["combined"].startswith(streams["generate"]):
            raise RuntimeError("generation changed when enhancement appended")
        diag = {}
        for q in ("q2","q1","q0.5"):
            output = load_frames(EVAL/sid/q/"reconstruction.npz")
            diag[q] = dict(bytes=(EVAL/sid/f"{q}.acse").stat().st_size,
                quality=quality(source,output,metric),enhance_roi=local_quality(source,output,e_regions,metric))
        native32 = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in
                          sorted((MECHANISM/"samples"/sid/"uf32_fresh").glob("*.png"))])
        native = {"8":dict(bytes=len(parsed.base),quality=points["base"]["quality"],
                          enhance_roi=points["base"]["enhance_roi"]),
                  "32":dict(bytes=(MECHANISM/"samples"/sid/"uf32.dcvc").stat().st_size,
                            quality=quality(source,native32,metric),
                            enhance_roi=local_quality(source,native32,e_regions,metric))}
        visual(root,source,outputs,points,reference["rois"],g_region)
        row = dict(sample=sample,points=points,local_patch_diagnostic=diag,native_uf=native,
            source_frame_hash=frame_hash(source),rois=reference["rois"],generate_region=g_region,
            validations=dict(same_base=True,enhance_unchanged=True,generation_shared=True,
                repeat_fresh_pixel_exact=True,no_generation_fallback_exact=True,
                combined_is_generate_plus_literal_enhancement_suffix=True))
        atomic_json(root/"result.json",row)
        rows.append(row)
        run.update(completed=i+1,total=4,phase="sample_complete")
        atomic_json(args.output/"summary.json",dict(complete=len(rows)==4,results=rows,utc=now(),
            seconds=time.monotonic()-run.started,resources=resources(),protocol=protocol))
        plot(args.output,rows)
    if len(rows) == 4:
        atomic_bytes(args.output/"experiment.complete",b"complete\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=DEFAULT)
    p.add_argument("--max-hours",type=float,default=6)
    p.add_argument("--limit",type=int,default=0)
    args = p.parse_args()
    args.command = "generate-probe"
    run = Run(args)
    run.thread.start()
    try:
        run.log_resources()
        with exclusive_native_evaluation(run):
            experiment(args,run)
        run.update(phase="complete")
    finally:
        run.stop.set()
        run.thread.join()
        run.log_resources()
        run.lock.close()
