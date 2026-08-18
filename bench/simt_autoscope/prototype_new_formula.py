#!/usr/bin/env python3
"""Prototype the component-targeted cost formula before editing C++.

Reads the 5-case benchmark report JSON and the fitted rate parameters, then
recomputes raw all_simd / all_simt_only with:

- irregular_addressing penalty applied to SIMD memory cycles;
- tiny_dot penalty applied to SIMD dot cycles;
- mask/reduction/loop/control/rank1 lowering penalties applied to SIMD compute;
- pattern-dependent SIMD/SIMT memory rates (contiguous vs gather);
- SIMT predicate instructions estimated as mask_tensor_ops * ceil(max_numel/32)
  with the newly fitted bounds-mask rate.

Usage:
    python bench/simt_autoscope/prototype_new_formula.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SYS_CNT_MHZ = 988.9
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    bench_path = REPO_ROOT / "ascend_results" / "simt_autoscope_bench.jsonl"
    fitted_path = REPO_ROOT / "ascend_results" / "fitted_rate_parameters.json"
    fitted = json.loads(fitted_path.read_text())
    rows = [json.loads(l) for l in bench_path.read_text().strip().splitlines() if l.strip()]

    reports = {}
    measured = {}
    for r in rows:
        measured.setdefault(r["case"], {})[r["route"]] = r["latency_ms"]
        if r.get("report_json"):
            reports[r["case"]] = r["report_json"]

    p = {
        "simd_setup": 21.212121, "simt_setup": 141.0, "issue_scale": 8.0,
        "simd_dot_rate": 4096.0, "simd_dot_setup": 128.0,
        "simt_dot_rate": 141.0, "simt_dot_setup": 64.0,
        "mask_per_rank": 0.022, "mask_cap": 0.35,
        "red_per": 0.02, "red_cap": 0.15,
        "loop_per": 0.008, "loop_cap": 0.15,
        "control": 0.03, "rank1": 0.75,
        "irr_per": 0.8, "irr_cap": 0.5,
        "tiny_dot": 0.06, "tiny_dot_flops_max": 16384,
        "pred_rate": fitted["simt_predicate"]["bounds_mask_add"]["effective_warps_per_cycle_32warps_active32"],
        "simd_mem_contig": fitted["simd_memory"]["mte2_mte3_bytes_per_cycle_contiguous"],
        "simd_mem_gather": fitted["simd_memory"]["gather_read_B_per_cycle"],
        "simt_load_contig": fitted["simt_gm_memory"]["load"]["contiguous_warp_instr_per_cycle_32warps"],
        "simt_store_contig": fitted["simt_gm_memory"]["store"]["contiguous_warp_instr_per_cycle_32warps"],
        "simt_load_gather": fitted["simt_gm_memory"]["load"]["gather_warp_instr_per_cycle"],
        "simt_store_gather": fitted["simt_gm_memory"]["store"]["gather_warp_instr_per_cycle"],
    }

    print(f"{'case':28s} {'meas':>8s} {'old_raw':>8s} {'new_raw':>8s}")
    for case, rep in sorted(reports.items()):
        f = rep["features"]
        gather = f["loaded_index_dependent_memory_ops"] > 0

        simd_rate = p["simd_mem_gather"] if gather else p["simd_mem_contig"]
        simd_mem = max(f["load_bytes"] / simd_rate, f["store_bytes"] / simd_rate)
        irr_density = rep["structure"]["irregular_density"]
        irr_pen = min(p["irr_cap"], irr_density * p["irr_per"])
        simd_mem *= (1.0 + irr_pen)

        flops = f["dot_flops"]
        simd_dot = p["simd_dot_setup"] + flops / p["simd_dot_rate"]
        if 0 < flops <= p["tiny_dot_flops_max"]:
            underfill = max(0.0, 1.0 - flops / p["tiny_dot_flops_max"])
            simd_dot *= (1.0 + p["tiny_dot"] * underfill)

        mask_pen = min(p["mask_cap"], f["mask_rank_sum"] * p["mask_per_rank"])
        weighted_red = f["weighted_ops"].get("reduce", 0)
        red_pen = min(p["red_cap"], weighted_red * p["red_per"])
        loop_pen = min(p["loop_cap"], f["static_loop_trip_count_sum"] * p["loop_per"])
        ctrl_pen = p["control"] if f["has_control_flow"] else 0.0
        rank1_pen = p["rank1"] if f["rank1_indirect_vector_reduce"] else 0.0
        simd_compute = rep["compute_only"]["simd"] * (
            1.0 + mask_pen + red_pen + loop_pen + ctrl_pen + rank1_pen
        )

        simd_payload = max(simd_compute + simd_dot, simd_mem)
        all_simd = p["simd_setup"] + p["issue_scale"] * simd_payload

        lr = p["simt_load_gather"] if gather else p["simt_load_contig"]
        sr = p["simt_store_gather"] if gather else p["simt_store_contig"]
        simt_mem = f["load_warp_instructions"] / lr + f["store_warp_instructions"] / sr
        simt_shuffle = rep["simt_execution"]["shuffle_system_cycles"]
        pred_instr = f["mask_tensor_ops"] * math.ceil(f["max_tensor_numel"] / 32)
        simt_pred = pred_instr / p["pred_rate"]
        simt_dot = p["simt_dot_setup"] + flops / p["simt_dot_rate"]
        simt_payload = max(rep["compute_only"]["simt"] + simt_shuffle + simt_dot, simt_mem) + simt_pred
        all_simt = p["simt_setup"] + p["issue_scale"] * simt_payload

        old_raw = (
            rep["event_route_calibration"]["raw_candidate_costs"]["all_simd"]
            / rep["event_route_calibration"]["raw_candidate_costs"]["all_simt_only"]
        )
        new_raw = all_simd / all_simt
        meas = measured.get(case, {}).get("simd", float("nan")) / measured.get(case, {}).get("simt_only", float("nan"))
        print(f"{case:28s} {meas:8.3f} {old_raw:8.3f} {new_raw:8.3f}")


if __name__ == "__main__":
    main()
