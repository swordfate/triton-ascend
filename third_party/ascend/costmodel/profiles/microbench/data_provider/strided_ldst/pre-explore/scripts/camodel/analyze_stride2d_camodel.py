#!/usr/bin/env python3
"""Parse a rank-2 CAModel run and expose the white-box 2-D facts.

Reuses analyze_strided_camodel.py for the OPPROF dump parsing (MTE2 queues, AIV
vector window, SIMT LSU/DC TagRam, BIU send/recv) and post-joins the rank-2
metadata from the probe manifest, so the 2-D structural quantities can be
compared against the 1-D ones.

Reported per case:
  * L_2d, mean_warp_lines, bucket_worst_32k16, page_cross, row_gap_bytes
    (from stride_feature_reference.tile_features -- the same definitions the
     semi-white-box fit uses);
  * MTE2 command count and issue->retire span;
  * BIU send count / bytes / size histogram / unique 128B lines;
  * derived ratios: bytes per BIU command, commands per element,
    elements per 128B line, and the load window in cycles.

Usage:
    python3 scripts/analyze_stride2d_camodel.py <run_dir> -o parsed.json --output-csv parsed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_strided_camodel as az  # noqa: E402
import stride_feature_reference as sfr  # noqa: E402


def derive(record, case):
    R, C, OS, IS = int(case["R"]), int(case["C"]), int(case["OS"]), int(case["IS"])
    f = sfr.tile_features(R, C, OS, IS, 4)
    biu = record.get("biu") or {}
    mte = (record.get("mte_queues") or {}).get("mte2") or {}
    mte2 = mte.get("MOV_SRC_TO_DST_ALIGNv2") or next(iter(mte.values()), {}) if mte else {}
    elements = R * C
    send_count = biu.get("send_count")
    send_bytes = biu.get("send_bytes")
    out = {
        "form": case.get("form"),
        "R": R, "C": C, "OS": OS, "IS": IS,
        "elements": elements,
        "contiguous": bool(case.get("contiguous")),
        "row_gap_bytes": f["row_gap_bytes"],
        "inner_stride_bytes": f["inner_stride_bytes"],
        "span_bytes": f["span_bytes"],
        "L_2d": f["L"],
        "elements_per_line": (elements / f["L"]) if f["L"] else None,
        "mean_warp_lines": f["mean_warp_lines"],
        "bucket_worst_32k16": f["bucket_worst_32k16"],
        "page_cross": f["page_cross"],
        "mte2_push_count": mte2.get("count") if isinstance(mte2, dict) else None,
        "mte2_push": mte2.get("push") if isinstance(mte2, dict) else None,
        "mte2_retire": mte2.get("retire") if isinstance(mte2, dict) else None,
        "load_engine": record.get("load_engine"),
        "load_instr_count": record.get("load_instr_count"),
        "load_active_cycles": record.get("load_active_cycles"),
        "instr_active_cycles": record.get("instr_active_cycles"),
        "biu_send_count": send_count,
        "biu_send_bytes": send_bytes,
        "biu_send_sizes": biu.get("send_sizes"),
        "biu_unique_128B_lines": biu.get("send_unique_128B_lines"),
        "biu_first_send": biu.get("first_send"),
        "biu_last_send": biu.get("last_send"),
        "biu_fill_span": record.get("biu_fill_span"),
    }
    if send_count:
        out["bytes_per_biu_cmd"] = (send_bytes / send_count) if send_bytes else None
    out["cmds_per_element"] = (send_count / elements) if send_count else None
    out["lines_per_element"] = (f["L"] / elements) if elements else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--kernel-name", default="stride2d_load_kernel")
    ap.add_argument("--profile", type=Path, default=None)
    ap.add_argument("-o", "--output-json", type=Path, default=Path("stride2d_camodel_parsed.json"))
    ap.add_argument("--output-csv", type=Path, default=Path("stride2d_camodel_parsed.csv"))
    args = ap.parse_args()

    run_dir = args.run_dir
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    cases = manifest["cases"]
    profiles = [args.profile] if args.profile else sorted(run_dir.glob("OPPROF_*"))
    if not profiles:
        raise SystemExit("no OPPROF_* directory found")
    profile = profiles[0]
    kernel_dir = profile / args.kernel_name
    if not kernel_dir.is_dir():
        kernel_dir = profile
    blocks = az.parse_block_windows(run_dir / "run.log")
    durations = az.parse_reported_durations(run_dir / "run.log")

    rows = []
    for index, case in enumerate(cases):
        record = az.summarize_case(case, run_dir, kernel_dir, index, blocks, durations)
        rows.append(derive(record, case))

    args.output_json.write_text(json.dumps(
        {"run_dir": str(run_dir), "cases": rows}, indent=2, sort_keys=True),
        encoding="utf-8")
    fields = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_json} and {args.output_csv} ({len(rows)} cases)")
    for r in rows:
        print(f"  {r['form']:14s} {r['R']}x{r['C']} OS={r['OS']:4d} IS={r['IS']} "
              f"L={r['L_2d']:4.0f} el/line={r['elements_per_line'] or 0:6.2f} "
              f"mte2={r['mte2_push_count']} biu_cmds={r['biu_send_count']} "
              f"biu_bytes={r['biu_send_bytes']} "
              f"B/cmd={r['bytes_per_biu_cmd'] and round(r['bytes_per_biu_cmd'],2)} "
              f"load_active={r['load_active_cycles']}")


if __name__ == "__main__":
    main()
