"""Read-only experiment audit plus a compact machine-readable resource report."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo import scalable_generation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_experiment import resources, now
from demo.scalable_format import frame_hash, parse
from demo.scalable_generation_experiment import DEFAULT, EVAL, MECHANISM


def audit(root):
    summary = read(root/"summary.json")
    if not summary["complete"] or not (root/"experiment.complete").exists():
        raise RuntimeError("experiment incomplete")
    protocol = summary["protocol"]
    for relative,digest in protocol["code"].items():
        if file_hash(REPO/relative) != digest:
            raise RuntimeError("evaluation implementation changed")
    from demo.scalable_generation_decode import identities, PATCH
    if identities() != protocol["assets"] or file_hash(PATCH) != protocol["patch_sha256"]:
        raise RuntimeError("model/profile mismatch")
    evaluated = read(EVAL/"summary.json")
    if file_hash(EVAL/"summary.json") != protocol["upstream_summary_sha256"]:
        raise RuntimeError("upstream summary changed")
    rows, peak_allocated, count = [],0,0
    for record in summary["results"]:
        sid = record["sample"]["sample_id"]
        directory = root/sid
        points = record["points"]
        arrays = {}
        for name,point in points.items():
            dest = directory/name
            verify_artifacts(dest,point["artifacts"])
            path = directory/(name+(".acse" if name in ("base","enhance") else ".acsg"))
            report = read(dest/"decode.json")
            data = path.read_bytes()
            if (len(data) != point["bytes"] or file_hash(path) != point["stream_sha256"]
                    or report != point["fresh_decode"] or report["source_frames_read"]
                    or report["total_bytes"] != len(data)):
                raise RuntimeError("decode/byte integrity failure")
            output = load_frames(dest/"reconstruction.npz")
            if frame_hash(output) != report["output_hash"]:
                raise RuntimeError("output hash mismatch")
            arrays[name] = output
            peak_allocated = max(peak_allocated,report.get("peak_cuda_allocated_bytes",0))
            if name not in ("base","enhance"):
                c,inner_bytes,inner,ncontrol = fmt.parse(data)
                if (ncontrol != report["generation_control_bytes"] or
                        not report["outside_generate_exact"] or
                        report["generation_input_hash"] != inner.meta["base_rgb_sha256"]):
                    raise RuntimeError("generation report mismatch")
                for window in (report["generation_runtime"] or {}).get("windows",[]):
                    peak_allocated = max(peak_allocated,window["runtime"]["peak_cuda_allocated_bytes"])
            else:
                inner = parse(data)
            if report["base_hash"] != inner.meta["base_rgb_sha256"]:
                raise RuntimeError("base changed")
            count += 1
        np.testing.assert_array_equal(arrays["combined"],arrays["combined_repeat"])
        np.testing.assert_array_equal(arrays["enhance"],arrays["generation_disabled"])
        c,_,parsed,_ = fmt.parse((directory/"combined.acsg").read_bytes())
        alpha = fmt.weights(arrays["base"].shape,c,parsed)
        np.testing.assert_array_equal(arrays["combined"][alpha == 0],arrays["enhance"][alpha == 0])
        np.testing.assert_array_equal(arrays["combined"][alpha > 0],arrays["generate"][alpha > 0])
        if not (directory/"combined.acsg").read_bytes().startswith((directory/"generate.acsg").read_bytes()):
            raise RuntimeError("not a literal append")
        original = next(v for v in evaluated["results"] if v["sample"]["sample_id"] == sid)
        verify_artifacts(EVAL/sid,original["artifacts"])
        old = MECHANISM/"samples"/sid
        previous = read(old/"complete.json")
        if record["source_frame_hash"] != previous["source_rgb_sha256"]:
            raise RuntimeError("source pixels differ from reused UF references")
        if "artifacts" in previous:
            verify_artifacts(old,previous["artifacts"])
        region = points["generate"]["fresh_decode"]["generation_runtime"]
        full = points["full_generate"]["fresh_decode"]["generation_runtime"]
        rows.append(dict(sample=sid, lpips={n:p["quality"]["lpips_alex"] for n,p in points.items()},
            roi_lpips={n:p["enhance_roi"]["lpips_alex"] for n,p in points.items()},
            bytes={n:p["bytes"] for n,p in points.items()},
            generation_seconds=region["seconds_model_load_excluded"],
            full_generation_seconds=full["seconds_model_load_excluded"],
            generation_time_reduction=1-region["seconds_model_load_excluded"]/full["seconds_model_load_excluded"],
            generation_model_load_seconds=region["model_load_seconds"],
            receiver_seconds={n:points[n]["fresh_decode"]["seconds"] for n in ("generate","combined","full_generate")},
            protected_pixels_exact=True, repeat_fresh_exact=True))
    beats = [json.loads(line) for line in (root/"heartbeat.jsonl").read_text().splitlines()]
    peak_sampled = max(int(v["gpu"].split(",")[2]) for v in beats)
    files = {str(p.relative_to(root)):file_hash(p) for p in root.rglob("*") if p.is_file()
             and p.suffix in (".acse",".acsg",".npz",".png")}
    result = dict(passed=True,utc=now(),fresh_decodes=count,rows=rows,
        checked_artifact_hashes=files, peak_cuda_allocated_bytes=peak_allocated,
        peak_sampled_gpu_mib=peak_sampled,elapsed_seconds=summary["seconds"],
        stored_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),resources=resources())
    atomic_json(root/"audit.json",result)
    print(json.dumps({k:v for k,v in result.items() if k != "checked_artifact_hashes"},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=DEFAULT)
    audit(p.parse_args().output)
