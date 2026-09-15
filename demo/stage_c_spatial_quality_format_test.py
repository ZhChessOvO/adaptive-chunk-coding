#!/usr/bin/env python3
"""E22: round-trip and corruption tests for spatial-quality stream syntax.

The test uses dummy entropy payloads on purpose.  It proves only that the new
versioned I/P syntax, action map, and exact byte accounting are unambiguous.
The actual no-training spatial codec and fresh decoder are exercised separately
by stage_c_spatial_quality_codec.py; this syntax test makes no quality claim.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.stream_helper import (
    NalType,
    SPSHelper,
    pack_spatial_actions,
    read_header,
    read_ip_remaining,
    read_spatial_ip_remaining,
    read_sps_remaining,
    spatial_action_count,
    validate_spatial_actions,
    write_ip,
    write_spatial_ip,
    write_sps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/e22_spatial_quality_format_test"))
    return parser.parse_args()


def expect_error(name: str, function, contains: str) -> dict:
    try:
        function()
    except (ValueError, struct.error) as error:
        if contains not in str(error):
            raise AssertionError(
                f"{name}: expected error containing {contains!r}, got {error!r}")
        return {"name": name, "passed": True, "error": str(error)}
    raise AssertionError(f"{name}: malformed syntax was accepted")


def parse_stream(data: bytes, frame_count: int) -> tuple[list[dict], dict]:
    source = io.BytesIO(data)
    sps_helper = SPSHelper()
    records = []
    decoded_frames = 0
    sps_bytes = 0
    syntax_bytes = 0
    action_bytes = 0
    entropy_bytes = 0
    while decoded_frames < frame_count:
        start = source.tell()
        header = read_header(source)
        if header["nal_type"] == NalType.NAL_SPS:
            sps = read_sps_remaining(source, header["sps_id"])
            sps_helper.add_sps_by_id(sps)
            sps_bytes += source.tell() - start
            continue
        if header["nal_type"] not in (NalType.NAL_I_SQ, NalType.NAL_P_SQ):
            raise ValueError("unexpected coding-unit type in E22 stream")
        sps = sps_helper.get_sps_by_id(header["sps_id"])
        if sps is None:
            raise ValueError("spatial coding unit references an unknown SPS")
        payload = read_spatial_ip_remaining(source)
        validate_spatial_actions(
            sps["width"], sps["height"], payload["cell_size"],
            payload["actions"])
        total = source.tell() - start
        entropy = len(payload["bit_stream"])
        action = payload["action_map_bytes"]
        syntax = total - entropy - action
        records.append({
            "nal_type": payload_type_name(header["nal_type"]),
            "sps_id": header["sps_id"],
            "quality_profile": {
                "Generate": payload["qp_generate"],
                "Base": payload["qp_base"],
                "Enhance": payload["qp_enhance"],
            },
            "cell_size": payload["cell_size"],
            "action_count": len(payload["actions"]),
            "actions": payload["actions"],
            "action_map_bytes": action,
            "syntax_bytes_excluding_action_and_entropy": syntax,
            "entropy_payload_bytes": entropy,
            "coding_unit_total_bytes": total,
            "ec_part": payload["ec_part"],
            "reset_feature_memory": payload["reset_feature_memory"],
        })
        syntax_bytes += syntax
        action_bytes += action
        entropy_bytes += entropy
        decoded_frames += 1 if header["nal_type"] == NalType.NAL_I_SQ else min(
            8, frame_count - decoded_frames)
    if source.tell() != len(data):
        raise ValueError("trailing bytes after spatial-quality stream")
    accounting = {
        "total_bytes": len(data),
        "sps_bytes": sps_bytes,
        "coding_unit_syntax_bytes": syntax_bytes,
        "action_map_bytes": action_bytes,
        "entropy_payload_bytes": entropy_bytes,
    }
    if sum(value for key, value in accounting.items() if key != "total_bytes") != len(data):
        raise RuntimeError("E22 byte accounting does not equal file size")
    return records, accounting


def payload_type_name(value: NalType) -> str:
    return "I-spatial" if value == NalType.NAL_I_SQ else "P-spatial"


def build_valid_stream() -> tuple[bytes, list[list[int]]]:
    width = height = 512
    cell_size = 64
    action_count = spatial_action_count(width, height, cell_size)
    maps = [
        [(index + phase) % 3 for index in range(action_count)]
        for phase in range(3)
    ]
    entropy_payloads = [
        bytes(range(17)),
        bytes((index * 7) % 256 for index in range(193)),
        bytes((index * 11) % 256 for index in range(131)),
    ]
    output = io.BytesIO()
    sps = {"sps_id": 0, "height": height, "width": width}
    write_sps(output, sps)
    for index, (actions, payload) in enumerate(zip(maps, entropy_payloads)):
        write_spatial_ip(
            output,
            is_i_frame=index == 0,
            sps_id=0,
            quality_profile=(8, 16, 32),
            cell_size=cell_size,
            actions=actions,
            ec_part=2,
            reset_feature_memory=int(index == 2),
            bit_stream=payload,
        )
    return output.getvalue(), maps


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid, expected_maps = build_valid_stream()
    stream_path = args.output_dir / "syntax_roundtrip.dqvc"
    stream_path.write_bytes(valid)
    records, accounting = parse_stream(stream_path.read_bytes(), frame_count=17)
    if [record["actions"] for record in records] != expected_maps:
        raise RuntimeError("round-trip changed an action map")

    corruption_tests = []
    legacy = io.BytesIO()
    write_ip(legacy, True, 0, 16, 2, 0, b"legacy-payload")
    legacy.seek(0)
    legacy_header = read_header(legacy)
    legacy_payload = read_ip_remaining(legacy)
    if (legacy_header["nal_type"] != NalType.NAL_I
            or legacy_payload != (16, 2, 0, b"legacy-payload")):
        raise RuntimeError("new header parser changed legacy scalar-QP syntax")
    corruption_tests.append(expect_error(
        "reserved action id",
        lambda: pack_spatial_actions([0, 1, 2, 3]),
        "ids 0, 1, or 2",
    ))
    corruption_tests.append(expect_error(
        "geometry mismatch",
        lambda: validate_spatial_actions(512, 512, 64, [0] * 63),
        "expected 64",
    ))
    corruption_tests.append(expect_error(
        "invalid quality ordering",
        lambda: write_spatial_ip(
            io.BytesIO(), True, 0, (16, 8, 32), 64, [0] * 64,
            1, 0, b"payload"),
        "Generate QP < Base QP < Enhance QP",
    ))
    # In the first coding unit, byte 12 is the final packed action-map byte.
    # Its top two bits are padding only when action_count is not divisible by
    # four, so exercise canonical padding directly with a 1-entry map.
    one_action = io.BytesIO()
    write_spatial_ip(
        one_action, True, 0, (8, 16, 32), 64, [0], 1, 0, b"x")
    malformed = bytearray(one_action.getvalue())
    # header(1), QPs(3), log2(1), flags(1), count(1), then action byte.
    malformed[7] = 0xC0
    corruption_tests.append(expect_error(
        "nonzero padding bits",
        lambda: read_spatial_ip_remaining(io.BytesIO(malformed[1:])),
        "padding bits",
    ))
    corruption_tests.append(expect_error(
        "truncated stream",
        lambda: parse_stream(valid[:-1], frame_count=17),
        "unpack requires",
    ))

    result = {
        "experiment": "E22 spatial-quality syntax round-trip",
        "status": "syntax-complete-see-e24-spatial-codec",
        "stream": str(stream_path),
        "format": {
            "coding_unit_map_scope": "one map per I frame or eight-frame P chunk",
            "action_ids": {"0": "Base", "1": "Generate", "2": "Enhance"},
            "quality_profile": {"Generate": 8, "Base": 16, "Enhance": 32},
            "cell_size_pixels": 64,
            "action_map_encoding": "canonical two-bit values; id 3 reserved",
            "maps_are_available_before_entropy_decode": True,
            "budget_known_before_encoding": True,
        },
        "records": records,
        "byte_accounting": accounting,
        "tests": {
            "roundtrip_maps_equal": True,
            "roundtrip_file_size_equal": accounting["total_bytes"] == stream_path.stat().st_size,
            "legacy_scalar_qp_syntax_unchanged": True,
            "corruption_tests": corruption_tests,
        },
        "scientific_boundary": {
            "entropy_payloads_are_dummy": True,
            "stock_scalar_qp_payload_used_as_spatial_result": False,
            "spatial_codec_implemented_elsewhere": True,
            "this_test_exercises_spatial_codec": False,
            "result_is_a_quality_claim": False,
            "next_required_work": (
                "controller training after approval, then optional spatial-codec fine-tuning"),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "stream": str(stream_path),
        "byte_accounting": accounting,
        "corruption_tests_passed": len(corruption_tests),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
