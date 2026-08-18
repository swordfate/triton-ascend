#!/usr/bin/env python3
"""Fit rate functions from the A5 microbenchmark data.

Reads the files collected in ascend_results/ and writes
ascend_results/fitted_rate_parameters.json with the parameters that Step 6
will later write into the SIMD/SIMT profile (or consume from C++).

Usage:
    python bench/simt_autoscope/fit_rates.py \
        --results-dir ascend_results \
        --out ascend_results/fitted_rate_parameters.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SYS_CNT_MHZ = 988.9


def ms_to_cycles(ms: float) -> float:
    return ms * 1e-3 * SYS_CNT_MHZ * 1e6


def parse_predicate_log(path: Path):
    """Return {mode: {active_lanes: {warps: (cycles_per_iter, effective_warps_per_cycle)}}}."""
    data = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("mode") or line.startswith("INC") or line.startswith("TK") or line.startswith("---") or line.startswith("In file") or line.startswith("SIMT"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                mode = int(parts[0]); active = int(parts[1]); warps = int(parts[2])
                cpi = float(parts[3]); rate = float(parts[4])
            except ValueError:
                continue
            data.setdefault(mode, {}).setdefault(active, {})[warps] = (cpi, rate)
    return data


def parse_gm_pattern_log(path: Path):
    """Return {(mode,pattern,stride,warps): (cpi, bpc, wpc)}."""
    data = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("mode") or line.startswith("INC") or line.startswith("TK") or line.startswith("---") or line.startswith("In file") or line.startswith("SIMT"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                mode = int(parts[0]); pattern = int(parts[1]); stride = int(parts[2]); warps = int(parts[3])
                cpi = float(parts[4]); bpc = float(parts[5]); wpc = float(parts[6])
            except ValueError:
                continue
            data[(mode, pattern, stride, warps)] = (cpi, bpc, wpc)
    return data


def power_fit(xs, ys):
    logx = [math.log(x) for x in xs]
    logy = [math.log(y) for y in ys]
    n = len(xs)
    mx = sum(logx) / n; my = sum(logy) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(logx, logy))
    vx = sum((a - mx) ** 2 for a in logx)
    b = cov / vx if vx > 1e-12 else 0.0
    a = my - b * mx
    return math.exp(a), b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="ascend_results")
    ap.add_argument("--out", default="ascend_results/fitted_rate_parameters.json")
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    out_path = Path(args.out)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)

    fitted = {"_comment": "Fitted from A5 microbenchmarks; SYS_CNT_MHZ=988.9"}

    # ---------------- SIMD memory ----------------
    simd_mem = {}
    with (res_dir / "simd_memory_microbench.jsonl.txt").open() as f:
        simd_rows = [json.loads(l) for l in f if l.strip()]
    for r in simd_rows:
        lab = r["label"]
        cyc = ms_to_cycles(r["latency_ms"])
        full_read = float(lab.get("full_tensor_read_bytes", r["read_bytes"]))
        read_bpc = full_read / cyc
        key = (lab.get("pattern"), lab.get("stride"), lab.get("n"))
        if lab["pattern"] == "strided" and lab["n"] == 4194304:
            simd_mem[("strided", lab["stride"])] = read_bpc
        elif lab["pattern"] == "contiguous" and lab["n"] == 4194304:
            simd_mem[("contiguous", 1)] = read_bpc
        elif lab["pattern"] == "gather" and lab["n"] == 4194304:
            simd_mem[("gather", 0)] = read_bpc
        elif lab["pattern"] == "masked" and lab["n"] == 4194304:
            active = float(lab.get("active_ratio", 1.0))
            simd_mem[("masked_full_tensor", active)] = full_read / cyc
            simd_mem[("masked_requested", active)] = (full_read * active) / cyc

    contig_read = simd_mem[("contiguous", 1)]
    stride_xs = [s for (p, s) in simd_mem if p == "strided"]
    stride_ys = [simd_mem[("strided", s)] for s in stride_xs]
    stride_a, stride_b = power_fit(stride_xs, stride_ys)
    gather_read = simd_mem[("gather", 0)]

    fitted["simd_memory"] = {
        "mte2_mte3_bytes_per_cycle_contiguous": round(contig_read, 2),
        "strided_power_fit": {"A": round(stride_a, 3), "exponent": round(stride_b, 3)},
        "strided_read_B_per_cycle": {str(s): round(simd_mem[("strided", s)], 3) for s in stride_xs},
        "gather_read_B_per_cycle": round(gather_read, 3),
        "masked_50_read_B_per_cycle_full_tensor": round(simd_mem[("masked_full_tensor", 0.5)], 3),
        "masked_50_read_B_per_cycle_requested": round(simd_mem[("masked_requested", 0.5)], 3),
        "note": "B/cycle is one-direction read bytes; copy kernel has symmetric read+write. full_tensor matches current C++ loadBytes semantics; requested is active_ratio*n*4.",
    }

    # ---------------- SIMT GM memory pattern ----------------
    gm = parse_gm_pattern_log(res_dir / "simt_gm_memory_pattern_host.log")
    gm_fit = {}
    for mode, name in [(0, "load"), (1, "store")]:
        contig = gm[(mode, 0, 1, 32)][2]
        gather = gm[(mode, 2, 1, 32)][2]
        xs = [s for s in [2, 4, 8, 16] if (mode, 1, s, 32) in gm]
        ys = [gm[(mode, 1, s, 32)][2] for s in xs]
        a, b = power_fit(xs, ys)
        gm_fit[name] = {
            "contiguous_warp_instr_per_cycle_32warps": round(contig, 4),
            "strided_power_fit": {"A": round(a, 4), "exponent": round(b, 4)},
            "strided_warp_instr_per_cycle": {str(s): round(ys[i], 4) for i, s in enumerate(xs)},
            "gather_warp_instr_per_cycle": round(gather, 4),
        }
    fitted["simt_gm_memory"] = gm_fit

    # ---------------- SIMT predicate ----------------
    pred = parse_predicate_log(res_dir / "simt_predicate_host.log")
    pred_fit = {}
    for mode, name in [(0, "no_mask_add"), (1, "bounds_mask_add"), (2, "predicated_select"), (3, "masked_gm_load")]:
        pred_fit[name] = {
            "effective_warps_per_cycle_32warps_active32": round(pred[mode][32][32][1], 4),
            "effective_warps_per_cycle_32warps_active1": round(pred[mode][1][32][1], 4),
            "by_active_lanes": {
                str(active): {
                    str(w): round(pred[mode][active][w][1], 4)
                    for w in sorted(pred[mode][active])
                }
                for active in sorted(pred[mode])
            },
        }
    fitted["simt_predicate"] = pred_fit

    # ---------------- SIMD components ----------------
    comp_rows = []
    with (res_dir / "simd_components_microbench.jsonl.txt").open() as f:
        comp_rows = [json.loads(l) for l in f if l.strip()]
    elem_fit = {}
    for r in comp_rows:
        if r["kind"] != "elementwise":
            continue
        cyc = ms_to_cycles(r["latency_ms"])
        vector_ops = r["elements"] / 64.0
        vec_per_cycle = vector_ops / cyc
        key = (r["dtype"], r["op"], r["n"])
        if r["n"] == 4194304:
            elem_fit.setdefault(r["dtype"], {})[r["op"]] = round(vec_per_cycle, 4)
    fitted["simd_elementwise_vector_instr_per_cycle_n4M"] = elem_fit

    dot_rows = [r for r in comp_rows if r["kind"] == "dot"]
    dot_cycles = [ms_to_cycles(r["latency_ms"]) for r in dot_rows]
    min_cycles = min(dot_cycles)
    # Profile rate=4096, setup=128 still matches the largest measured dot.
    fitted["simd_dot"] = {
        "measured_cycles": {f"{r['M']}x{r['N']}x{r['K']}": round(ms_to_cycles(r["latency_ms"])) for r in dot_rows},
        "keep_flops_per_system_cycle": 4096.0,
        "keep_startup_system_cycles": 128.0,
        "small_kernel_min_cycles": round(min_cycles),
        "note": "All three measured shapes are latency-floor dominated (~16.3k SYS_CNT cycles). Keep 4096/128 and add max(..., min_cycles).",
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fitted, f, indent=2, ensure_ascii=False)
    print(json.dumps(fitted, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
