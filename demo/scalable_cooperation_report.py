"""Audit actual bytes/pixels and produce honest paired diagnostic figures."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_cooperation_decode import identities, PATCH
from demo.scalable_cooperation_experiment import DEFAULT, EVAL, OLD
from demo.scalable_experiment import load_source, now, resources
from demo.scalable_format import parse, frame_hash

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def expected_key(name):
    if name == "base" or name.startswith("generate_") or name == "base_generation_off":
        return "base"
    if name.startswith("prefix_"):
        return "prefix"
    for q in ("q2","q0.5"):
        if name.endswith("_"+q):
            return q
    return "q1"


def audit(root,summary):
    if not summary["complete"] or not (root/"experiment.complete").exists():
        raise RuntimeError("experiment incomplete")
    protocol = summary["protocol"]
    if identities() != protocol["assets"] or file_hash(PATCH) != protocol["patch_sha256"]:
        raise RuntimeError("models/profile changed")
    if file_hash(EVAL/"summary.json") != protocol["upstream_summary_sha256"]:
        raise RuntimeError("upstream summary changed")
    for path,digest in protocol["code"].items():
        if file_hash(REPO/path) != digest:
            raise RuntimeError(f"measured code changed: {path}")
    rows,count,peak = [],0,0
    for row in summary["results"]:
        sid = row["sample"]["sample_id"]
        directory = root/sid
        verify_artifacts(directory/"prepared",row["prepared"]["artifacts"])
        source = load_source(row["sample"])
        if frame_hash(source) != row["source_hash"]:
            raise RuntimeError("source changed")
        expected = {q:load_frames(directory/"prepared"/f"{q}_expected.npz") for q in ("base","q1","q2","q0.5")}
        if "prefix_enhance" in row["points"]:
            expected["prefix"] = load_frames(directory/"prefix_enhance/reconstruction.npz")
        for name,point in row["points"].items():
            dest = directory/name
            verify_artifacts(dest,point["artifacts"])
            generated = (directory/(name+".acsg")).exists()
            stream = directory/(name+(".acsg" if generated else ".acse"))
            data = stream.read_bytes()
            report = read(dest/"decode.json")
            actual = load_frames(dest/"reconstruction.npz")
            before = expected[expected_key(name)]
            if (len(data) != point["bytes"] or file_hash(stream) != point["stream_sha256"]
                    or report != point["fresh_decode"] or report["source_frames_read"]
                    or report["total_bytes"] != len(data) or report["output_hash"] != frame_hash(actual)
                    or report["base_hash"] != frame_hash(expected["base"])):
                raise RuntimeError("integrity mismatch")
            if generated:
                control,inner_bytes,parsed,ncontrol = fmt.parse(data)
                mask = fmt.weights(actual.shape,control)
                if (report["generation_input_hash"] != frame_hash(before)
                        or report["generation_control_bytes"] != ncontrol
                        or not report["base_reference_unchanged"]):
                    raise RuntimeError("wrong generator condition/control")
                np.testing.assert_array_equal(actual[mask == 0],before[mask == 0])
                fields = ("base_bytes","container_header_bytes","packet_bytes",
                          "incomplete_tail_bytes","generation_control_bytes")
                if sum(report[k] for k in fields) != len(data):
                    raise RuntimeError("byte parts do not sum")
                if not report["generation_executed"]:
                    np.testing.assert_array_equal(actual,before)
                if name.startswith("cooperate_") and expected_key(name) in ("q1","q2","q0.5"):
                    if inner_bytes != (directory/"prepared"/f"{expected_key(name)}.acse").read_bytes():
                        raise RuntimeError("cooperation silently recoded E")
                for window in (report["generation_runtime"] or {}).get("windows",[]):
                    peak = max(peak,window["runtime"]["peak_cuda_allocated_bytes"])
            else:
                parsed = parse(data)
                np.testing.assert_array_equal(actual,before)
            if parsed.meta["base_rgb_sha256"] != report["base_hash"]:
                raise RuntimeError("base differs")
            peak = max(peak,report.get("peak_cuda_allocated_bytes",0))
            count += 1
        def frames(name):
            return load_frames(directory/name/"reconstruction.npz")
        np.testing.assert_array_equal(frames("cooperate_l05"),frames("cooperate_repeat"))
        np.testing.assert_array_equal(frames("enhance_q1"),frames("generation_off"))
        full = (directory/"cooperate_l05.acsg").read_bytes()
        if not full.startswith((directory/"generate_l05.acsg").read_bytes()):
            raise RuntimeError("not literal base prefix")
        if "prefix_cooperate" in row["points"] and not full.startswith((directory/"prefix_cooperate.acsg").read_bytes()):
            raise RuntimeError("not literal middle prefix")
        comparisons = {}
        for suffix in ("l0","l025","l05","context128","scale2"):
            g,e = row["points"].get("generate_"+suffix),row["points"].get("cooperate_"+suffix)
            if g and e:
                comparisons[suffix] = {k:e["roi_quality"][k]-g["roi_quality"][k] for k in e["roi_quality"]}
        rows.append(dict(sample=sid,paired_EG_minus_G=comparisons,
                         repeat_exact=True,fallback_exact=True,condition_verified=True))
    legacy = root/"legacy_v1_regression"
    verify_artifacts(legacy,summary["legacy_v1_regression"]["artifacts"])
    old_sid = summary["results"][0]["sample"]["sample_id"]
    np.testing.assert_array_equal(load_frames(legacy/"reconstruction.npz"),
                                  load_frames(OLD/old_sid/"combined/reconstruction.npz"))
    beats = [json.loads(line) for line in (root/"heartbeat.jsonl").read_text().splitlines()]
    # A resumed evaluator reports the latest attempt's duration in summary.json.
    # Preserve observed time from prior attempts rather than presenting a fast
    # cache verification as the cost of the original experiment.
    attempt_ends = []
    previous = 0.
    for beat in beats:
        elapsed = beat["elapsed_seconds"]
        if elapsed < previous:
            attempt_ends.append(previous)
        previous = elapsed
    attempt_ends.append(max(previous,summary["seconds"]))
    return dict(passed=True,utc=now(),fresh_decodes=count+1,rows=rows,
                legacy_v1_exact=True,peak_cuda_allocated_bytes=peak,
                peak_sampled_gpu_mib=max(int(v["gpu"].split(",")[2]) for v in beats),
                elapsed_seconds=sum(attempt_ends),last_attempt_seconds=summary["seconds"],
                attempts=len(attempt_ends),
                elapsed_note="Sum of observed attempt durations; an abruptly interrupted attempt may miss its final heartbeat interval.",
                resources=resources())


def visual(root,row):
    source = load_source(row["sample"])
    sid = row["sample"]["sample_id"]
    directory = root/sid
    names = ["source","base","enhance_q1","generate_l05","cooperate_l05"]
    if "cooperate_l0" in row["points"]:
        names.append("cooperate_l0")
    arrays = {name:load_frames(directory/name/"reconstruction.npz") for name in names if name != "source"}
    arrays["source"] = source
    h,w = source.shape[1:3]
    font = ImageFont.truetype(FONT,15)
    canvas = Image.new("RGB",(3*w,2*(h+65)),"white")
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(names):
        x,y = i%3*w,i//3*(h+65)
        canvas.paste(Image.fromarray(arrays[name][8]),(x,y+65))
        draw.text((x+4,y+3),name,font=font,fill="black")
        if name != "source":
            point = row["points"][name]
            draw.text((x+4,y+24),f"{point['bytes']} B | ROI LPIPS {point['roi_quality']['lpips_alex']:.4f}",font=font,fill="black")
            draw.text((x+4,y+44),f"ROI PSNR {point['roi_quality']['psnr_db']:.2f}",font=font,fill="black")
        else:
            draw.text((x+4,y+24),"Same E/G boxes; frame 8",font=font,fill="black")
            for j,(xx,yy,ww,hh) in enumerate(row["rois"]):
                draw.rectangle((xx,y+65+yy,xx+ww-1,y+65+yy+hh-1),outline="orange",width=2)
                draw.text((xx+2,y+65+yy+2),str(j),font=font,fill="cyan")
    canvas.save(directory/"fixed_frame.png")
    crop_names = ["source","base","enhance_q1","generate_l05","cooperate_l05"]
    if "generate_l0" in row["points"]:
        crop_names += ["generate_l0","cooperate_l0"]
        arrays["generate_l0"] = load_frames(directory/"generate_l0/reconstruction.npz")
    crops = Image.new("RGB",(256*len(crop_names),290*len(row["rois"])),"white")
    draw = ImageDraw.Draw(crops)
    for j,(x,y,w,h) in enumerate(row["rois"]):
        for i,name in enumerate(crop_names):
            crop = Image.fromarray(arrays[name][8,y:y+h,x:x+w]).resize((256,round(256*h/w)),Image.Resampling.NEAREST)
            crops.paste(crop,(i*256,j*290+30))
            draw.text((i*256+3,j*290+3),f"ROI {j}: {name}",font=font,fill="black")
    crops.save(directory/"fixed_crops.png")
    # Source pixels, never AI-edited. Short GIFs supplement stills, not metrics.
    roi = row["rois"][-1] if row["prepared"]["wall_roi_added"] else row["rois"][0]
    x,y,w,h = roi
    movie = []
    for t in range(len(source)):
        frame = Image.new("RGB",(224*5,round(224*h/w)+30),"white")
        d = ImageDraw.Draw(frame)
        for i,name in enumerate(names[:5]):
            crop = Image.fromarray(arrays[name][t,y:y+h,x:x+w]).resize((224,round(224*h/w)),Image.Resampling.NEAREST)
            frame.paste(crop,(i*224,30))
            d.text((i*224+3,3),f"{name} | t={t}",font=font,fill="black")
        movie.append(frame)
    movie[0].save(directory/"temporal_comparison.gif",save_all=True,append_images=movie[1:],
                  duration=125,loop=0,optimize=False)
    if len(source) > 17:
        fig,ax = plt.subplots(figsize=(9,3))
        for name in names[1:5]:
            difference = (arrays[name][:,y:y+h,x:x+w].astype(np.float32)
                          -source[:,y:y+h,x:x+w].astype(np.float32))
            ax.plot(np.arange(len(source)),np.mean(difference**2,axis=(1,2,3)),label=name)
        ax.axvline(16.5,linestyle="--",color="gray",label="E packets cover frames 0-16 only")
        ax.set(xlabel="Frame index (continuous base reference)",ylabel="Fixed ROI RGB MSE",
               title="33-frame diagnostic: G uses 17/8 overlapping windows")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(directory/"temporal_roi_error.png",dpi=150)
        plt.close(fig)


def plots(root,rows):
    fig,axes = plt.subplots(len(rows),2,figsize=(12,3.5*len(rows)),squeeze=False)
    for row,axs in zip(rows,axes):
        points = row["points"]
        for ax,field in zip(axs,("lpips_alex","psnr_db")):
            e_keys = ["base"]+[k for k in ("enhance_q2","enhance_q1","enhance_q0.5") if k in points]
            ep = [points[k] for k in e_keys]
            ax.plot([p["bytes"] for p in ep],[p["roi_quality"][field] for p in ep],"o-",label="E (independent q encodings)")
            if "cooperate_q2" in points:
                cp = [points[k] for k in ("generate_l05","cooperate_q2","cooperate_l05","cooperate_q0.5")]
                ax.plot([p["bytes"] for p in cp],[p["roi_quality"][field] for p in cp],"*--",label="E -> G, LoRA .5 (q sweep)")
            for suffix,marker in (("l0","s"),("l025","^"),("l05","*")):
                for mode in ("generate","cooperate"):
                    name = mode+"_"+suffix
                    if name in points:
                        p = points[name]
                        ax.scatter(p["bytes"],p["roi_quality"][field],marker=marker,s=60,label=name)
            ax.set_title(row["sample"]["sample_id"]+" | "+field)
            ax.set_xlabel("Actual total bytes (base, E, controls)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=6)
    fig.suptitle("Paired same-region development diagnostic; no BD-rate claim")
    fig.tight_layout()
    fig.savefig(root/"paired_rd.png",dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for row in rows:
        p = row["points"]
        keys = [k for k in ("generate_l05","prefix_cooperate","cooperate_l05") if k in p]
        for ax,field in zip(axes,("lpips_alex","psnr_db")):
            ax.plot([p[k]["bytes"] for k in keys],[p[k]["roi_quality"][field] for k in keys],"o-",label=row["sample"]["sample_id"])
            ax.set_xlabel("Same stream: base -> middle -> all E packets")
            ax.set_ylabel("ROI "+field)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/"cooperative_prefix.png",dpi=150)
    plt.close(fig)


def diagnostics(root, row):
    """Fixed frame/ROI configuration grid; never select by the best output."""
    if "cooperate_scale2" not in row["points"]:
        return
    directory = root/row["sample"]["sample_id"]
    source = load_source(row["sample"])
    index = len(row["rois"])-1 if row["prepared"]["wall_roi_added"] else 0
    x,y,w,h = row["rois"][index]
    keys = ["source","base","enhance_q1", "generate_l0","generate_l025","generate_l05",
            "cooperate_l0","cooperate_l025","cooperate_l05",
            "cooperate_context128","cooperate_scale2","cooperate_q0.5"]
    tilew,tileh = 320,round(320*h/w)+70
    canvas = Image.new("RGB",(3*tilew,4*tileh),"white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT,15)
    for i,name in enumerate(keys):
        px,py = i%3*tilew,i//3*tileh
        frames = source if name == "source" else load_frames(directory/name/"reconstruction.npz")
        crop = Image.fromarray(frames[8,y:y+h,x:x+w]).resize((tilew,tileh-70),Image.Resampling.NEAREST)
        canvas.paste(crop,(px,py+70))
        draw.text((px+3,py+3),name,font=font,fill="black")
        if name != "source":
            point = row["points"][name]
            m = point["per_region"][index]
            draw.text((px+3,py+25),f"ROI {index} LPIPS {m['lpips_alex']:.3f} | PSNR {m['psnr_db']:.2f}",font=font,fill="black")
            draw.text((px+3,py+47),f"{point['bytes']} B (whole stream)",font=font,fill="black")
        else:
            draw.text((px+3,py+25),f"Fixed ROI {index}, frame 8, nearest-neighbor",font=font,fill="black")
    canvas.save(directory/"configuration_diagnostic.png")


def analysis(rows):
    """Convenience tables derived only from audited native reconstructions."""
    samples = []
    for row in rows:
        points = {}
        sample = row["sample"]
        # Load shape rather than assuming a common resolution/frame count.
        n,h,w,_ = load_source(sample).shape
        for name,p in row["points"].items():
            r = p["fresh_decode"]
            runtime = r.get("generation_runtime") or {}
            points[name] = dict(bytes=p["bytes"],bpp=8*p["bytes"]/(n*h*w),
                roi=p["roi_quality"],per_region=p["per_region"],whole=p["quality"],
                receiver_wall_seconds=p["process_wall_seconds"],
                generation_seconds=runtime.get("seconds_model_load_excluded",0),
                generation_load_seconds=runtime.get("model_load_seconds",0),
                generation_calls=len(runtime.get("windows",[])),
                control_bytes=r.get("generation_control_bytes",0))
        samples.append(dict(sample=sample["sample_id"],rois=row["rois"],points=points))
    common = set.intersection(*(set(s["points"]) for s in samples))
    mean = {name:{field:float(np.mean([s["points"][name]["roi"][field] for s in samples]))
                  for field in samples[0]["points"][name]["roi"]} for name in sorted(common)}
    return dict(samples=samples,equal_clip_mean_roi=mean,
                note="Development diagnostic. Native ROI metric over first 17 frames; equal-clip means are not pooled video RD or BD-rate. Timings separate G loading from execution; each receiver is fresh.")


def main(root):
    summary = read(root/"summary.json")
    checked = audit(root,summary)
    for row in summary["results"]:
        visual(root,row)
        diagnostics(root,row)
    plots(root,summary["results"])
    atomic_json(root/"analysis.json",analysis(summary["results"]))
    checked["stored_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    checked["artifacts"] = {str(p.relative_to(root)):file_hash(p) for p in root.rglob("*")
        if p.is_file() and p.suffix in (".acse",".acsg",".npz",".png",".gif")}
    atomic_json(root/"audit.json",checked)
    print(json.dumps({k:v for k,v in checked.items() if k != "artifacts"},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=DEFAULT)
    main(p.parse_args().output)
