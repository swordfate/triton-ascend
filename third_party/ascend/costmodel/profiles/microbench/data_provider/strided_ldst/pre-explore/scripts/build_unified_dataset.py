#!/usr/bin/env python3
"""One featurisation for every strided row, rank-1 and rank-2 alike.

The v1 datasets used two incompatible feature namespaces (31 f_* columns for
rank-1, 23 different ones for rank-2, only 5 shared).  A single formula cannot
span both unless both are described by the SAME quantities.  A strided access
-- 1-D or 2-D -- is just an element-address list, and
stride_feature_reference.address_list_features() is already the byte-domain
description of an address list.  So every row is re-featurised through it and
the path-specific extras are added on top.

Output columns (all byte-domain or pure counts, never element-domain stride):
    elements, L, line_elems_max/min, line_density,
    mean/min/max_warp_lines, bucket_pairs_2k8, bucket_worst_2k8,
    bucket_pairs_32k16, bucket_worst_32k16, span_bytes, page_cross, cross,
    K, K_stg, dup, line_request_size, line_request_size_cpp4,
    min_L_32, max_L_minus_32, dupL_minL_128, cross_minL_32, lowW_4_L,
    n_cmd_block, tail_elems, iters_per_thread,
    inner_stride_bytes, row_gap_bytes, rank
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stride_feature_reference as sfr  # noqa: E402

TEMPLATE_THREADS = 1024
WARP = 32


def unified_features(element_bytes, num_warps, *, block=None, stride=None,
                     rows=None, cols=None, os_=None, is_=None, path="simd",
                     op="load"):
    if block is not None:
        addrs = [i * int(stride) for i in range(int(block))]
        rank = 1
        elements = int(block)
        inner_stride = int(stride)
        row_gap = 0
    else:
        addrs = sfr.tile_addresses(int(rows), int(cols), int(os_), int(is_))
        rank = 2
        elements = int(rows) * int(cols)
        inner_stride = int(is_)
        inner_span = (int(cols) - 1) * int(is_)
        row_gap = int(os_) - inner_span - 1

    threads = (TEMPLATE_THREADS if path == "template"
               else WARP * max(1, int(num_warps)))
    f = dict(sfr.address_list_features(addrs, element_bytes, threads=threads))

    stride_bytes = inner_stride * element_bytes
    L = f["L"]
    f["rank"] = float(rank)
    f["K"] = float(sfr.ldg_instructions(elements, num_warps))
    f["K_stg"] = float(sfr.stg_instructions(elements, num_warps))
    f["dup"] = float(sfr.dup_factor(elements, num_warps))
    f["line_request_size"] = float(
        sfr.line_request_size(elements, stride_bytes, element_bytes))
    # C++ ships a hardcoded 4 bytes/element in stridedLineRequestSize().
    f["line_request_size_cpp4"] = float(
        sfr.line_request_size(elements, stride_bytes, 4))
    f["min_L_32"] = min(L, 32.0)
    f["max_L_minus_32"] = max(0.0, L - 32.0)
    f["dupL_minL_128"] = f["dup"] * L * min(L, 128.0)
    f["cross_minL_32"] = f["cross"] * min(L, 32.0)
    f["lowW_4_L"] = max(0.0, 4.0 - float(num_warps)) * L
    f["n_cmd_block"] = float(elements)
    f["tail_elems"] = max(0.0, elements - 64.0)
    f["iters_per_thread"] = float(
        max(1, math.ceil(elements / TEMPLATE_THREADS)))
    f["inner_stride_bytes"] = float(stride_bytes)
    f["row_gap_bytes"] = float(row_gap * element_bytes)

    # ---- rank-2 "comb" structure -------------------------------------------
    # A rank-2 strided access is R combs of C teeth.  The union of touched
    # lines is already in L, but the SHAPE of the comb matters: rows may share
    # a line (row_gap < 128 B) or not, and the number of row segments changes
    # how many times the memory system restarts its burst.
    nr = int(rows) if rows is not None else 1
    nc = int(cols) if cols is not None else (int(block) if block is not None else 1)
    os_el = int(os_) if os_ is not None else nc
    is_el = int(is_) if is_ is not None else (int(stride) if stride is not None else 1)
    row_span_b = float((nc - 1) * is_el * element_bytes + element_bytes)
    row_pitch_b = float(os_el * element_bytes)
    row_gap_b = max(0.0, row_pitch_b - row_span_b)
    row_lines = float(math.ceil(row_span_b / 128.0))
    f["rows"] = float(nr)
    f["cols"] = float(nc)
    f["row_span_bytes"] = row_span_b
    f["row_pitch_bytes"] = row_pitch_b
    f["row_lines"] = row_lines
    f["gap_lines"] = float(math.floor(row_gap_b / 128.0))
    f["comb_lines"] = float(nr) * row_lines              # lines if rows never shared
    f["comb_overlap"] = float(nr) * row_lines - L        # lines saved by sharing
    f["rows_share_line"] = 1.0 if (nr > 1 and row_gap_b < 128.0) else 0.0
    f["row_segments"] = float(nr) if nr > 1 else 0.0
    f["gap_per_line"] = row_gap_b / 128.0
    f["row_span_div_gap"] = row_span_b / max(1.0, row_gap_b) if nr > 1 else 0.0
    f["tail_row_elems"] = float(nc) if nr > 1 else 0.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-template", action="store_true",
                    help="drop the simt-template path entirely: verified that "
                         "compile_mode=simd_simt / simt_only never emit "
                         "triton_stride_load/store, so the template is out of scope")
    ap.add_argument("--v1-1d", type=Path,
                    default=HERE.parent / "results/dtype_calibration_v1/dataset.csv")
    ap.add_argument("--v1-2d", type=Path,
                    default=HERE.parent / "results/dtype_calibration_v1/dataset_2d.csv")
    ap.add_argument("--v4-2d", type=Path, default=None,
                    help="campaign-B 2D grid (R-scan / W-scan / wide / dense strides)")
    ap.add_argument("--v3-2d", type=Path, default=None,
                    help="expanded 2D shape grid (dataset_2d.csv from the v10 run)")
    ap.add_argument("--v2-1d", type=Path,
                    default=HERE.parent / "results/dtype_calibration_v2_rotate512/dataset.csv")
    args = ap.parse_args()

    rows = []

    # ---- historical rank-1 rotE-loop datasets (all f32) -------------------
    # model_v4_numwarps : pure-SIMT load, W in 1..64, 23 strides
    p = HERE.parent / "results/model_v4_numwarps/dataset.csv"
    if p.exists():
        with p.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r["stride"]) < 2:
                    continue
                rows.append({"op": "load", "path": "simt_only", "dtype": "float32",
                             "element_bytes": 4, "num_warps": int(r["num_warps"]),
                             "target_ns": float(r["target_ns"]),
                             "target_kind": r["target_kind"], "_src": "hist_v4",
                             "_kw": {"block": int(r["block"]),
                                     "stride": int(r["stride"])}})
    # model_store_v1 : SIMD (100) and pure-SIMT (644) stores, W in 1..32
    p = HERE.parent / "results/model_store_v1/dataset.csv"
    if p.exists():
        with p.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r["stride"]) < 2:
                    continue
                path = "simd" if r["mode"] == "simd" else "simt_only"
                rows.append({"op": "store", "path": path, "dtype": "float32",
                             "element_bytes": 4, "num_warps": int(r["num_warps"]),
                             "target_ns": float(r["target_ns"]),
                             "target_kind": r["target_kind"], "_src": "hist_store",
                             "_kw": {"block": int(r["block"]),
                                     "stride": int(r["stride"])}})
    # model_template_stride_v2 : template load/store, W = 32, valid rows only
    p = HERE.parent / "results/model_template_stride_v2/dataset.csv"
    if p.exists():
        with p.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("valid")) not in ("1", "True"):
                    continue
                if int(r["stride"]) < 2:
                    continue
                op = "load" if "load" in r["path"] else "store"
                rows.append({"op": op, "path": "template", "dtype": "float32",
                             "element_bytes": 4, "num_warps": int(r["num_warps"]),
                             "target_ns": float(r["target_ns"]),
                             "target_kind": r["target_kind"],
                             "_src": "hist_template",
                             "_kw": {"block": int(r["block"]),
                                     "stride": int(r["stride"])}})
    # model_v2 SIMD load is SINGLE-SHOT (different domain) -> loaded separately
    p = HERE.parent / "results/model_v2/dataset.csv"
    if p.exists():
        with p.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["mode"] != "simd" or int(r["stride"]) < 2:
                    continue
                rows.append({"op": "load", "path": "simd", "dtype": "float32",
                             "element_bytes": 4, "num_warps": 1,
                             "target_ns": float(r["target_ns"]),
                             "target_kind": r["target_kind"],
                             "_src": "hist_v2_single_shot",
                             "_kw": {"block": int(r["block"]),
                                     "stride": int(r["stride"])}})

    if args.v1_1d.exists():
        with args.v1_1d.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, "_src": "v1_1d",
                             "_kw": {"block": int(r["block"]),
                                     "stride": int(r["stride"])}})
    if args.v1_2d.exists():
        with args.v1_2d.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, "_src": "v1_2d",
                             "_kw": {"rows": int(r["R"]), "cols": int(r["C"]),
                                     "os_": int(r["OS"]), "is_": int(r["IS"])}})
    if args.v2_1d.exists():
        with args.v2_1d.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, "_src": "v2_1d",
                             "_kw": {"block": int(r["block"]),
                                     "stride": int(r["stride"])}})

    if args.v3_2d is not None and args.v3_2d.exists():
        with args.v3_2d.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, "_src": "v3_2d",
                             "_kw": {"rows": int(r["R"]), "cols": int(r["C"]),
                                     "os_": int(r["OS"]), "is_": int(r["IS"])}})

    if args.v4_2d is not None and args.v4_2d.exists():
        with args.v4_2d.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({**r, "_src": "v4_2d",
                             "_kw": {"rows": int(r["R"]), "cols": int(r["C"]),
                                     "os_": int(r["OS"]), "is_": int(r["IS"])}})

    # v2 (512 MB window) supersedes v1 for the same cell.
    PRIORITY = {"hist_v2_single_shot": 0, "hist_v4": 0, "hist_store": 0,
                "hist_template": 0, "v1_1d": 1, "v1_2d": 2, "v2_1d": 3,
                "v3_2d": 4, "v4_2d": 5}
    dedup, order = {}, []
    for r in rows:
        key = (r["op"], r["path"], r["dtype"], int(r["num_warps"]),
               tuple(sorted(r["_kw"].items())))
        if key in dedup and PRIORITY[r["_src"]] <= PRIORITY[dedup[key]["_src"]]:
            continue
        if key not in dedup:
            order.append(key)
        dedup[key] = r
    rows = [dedup[k] for k in order]

    if args.exclude_template:
        rows = [r for r in rows if r["path"] != "template"]
        print(f"  (template excluded -> {len(rows)} rows)")

    feature_names = None
    out_rows = []
    for r in rows:
        f = unified_features(int(r["element_bytes"]), int(r["num_warps"]),
                             path=r["path"], op=r["op"], **r["_kw"])
        if feature_names is None:
            feature_names = list(f.keys())
        rec = {
            "op": r["op"], "path": r["path"], "dtype": r["dtype"],
            "element_bytes": int(r["element_bytes"]),
            "num_warps": int(r["num_warps"]),
            "target_ns": float(r["target_ns"]),
            "target_kind": r["target_kind"], "source": r["_src"],
            "domain": ("single_shot" if r["_src"] == "hist_v2_single_shot"
                       else "rotate"),
            "rank": int(f["rank"]),
            "elements": int(f["elements"]),
            "inner_stride_bytes": int(f["inner_stride_bytes"]),
            # carry the probe's own quality verdict through the re-featurisation
            "valid": int(str(r.get("valid", 1)) in ("1", "True")),
            "quality": r.get("quality", "ok"),
            "target_spread": r.get("target_spread", ""),
        }
        rec.update({f"f_{k}": v for k, v in f.items()})
        out_rows.append(rec)

    cols = ["op", "path", "dtype", "element_bytes", "num_warps", "rank",
            "elements", "inner_stride_bytes", "target_ns", "target_kind",
            "source", "domain", "valid", "quality", "target_spread"] + [f"f_{k}" for k in feature_names]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out}: {len(out_rows)} rows, "
          f"{len(feature_names)} features")
    for src in sorted({r["source"] for r in out_rows}):
        sub = [r for r in out_rows if r["source"] == src]
        print(f"   {src}: {len(sub)} rows, ranks={sorted({r['rank'] for r in sub})}, "
              f"dtypes={sorted({r['dtype'] for r in sub})}")


if __name__ == "__main__":
    main()
