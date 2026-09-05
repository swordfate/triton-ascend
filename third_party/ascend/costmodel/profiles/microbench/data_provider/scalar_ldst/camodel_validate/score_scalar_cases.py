#!/usr/bin/env python3
"""Run scalar cases through the StageCostModel and print SIMD/SIMT route scores.

This is used to compare our cost-model ranking with CAModel SIMD-vs-SIMT
measurements. It runs on real NPU only for compilation and route dump.
"""
import json
import os
import tempfile
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

HERE = Path(__file__).resolve().parent


def _vector_core_count():
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(props["num_vectorcore"])


@triton.jit
def scalar_camodel_cases(
    a_ptr, idx_ptr, out_ptr,
    N_LOAD: tl.constexpr, DEP_LEN: tl.constexpr, N_STORE: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.load(a_ptr + pid)
    for i in tl.static_range(1, N_LOAD):
        s += tl.load(a_ptr + pid + i * 16)
    cur = pid
    for i in tl.static_range(DEP_LEN):
        cur = tl.load(idx_ptr + cur * 4 + i)
        s += cur
    for i in tl.static_range(N_STORE):
        tl.store(out_ptr + pid * 8 + i, s)


CASES = {
    "direct_load_1": (1, 0, 1),
    "direct_load_4": (4, 0, 1),
    "direct_store_4": (0, 0, 4),
    "dep_load_4": (1, 4, 1),
}


def score_case(name, grid, report_path):
    n_load, dep_len, n_store = CASES[name]
    a_cpu = torch.arange(grid + 8 * 16 + 64, dtype=torch.int32)
    idx_cpu = torch.arange(grid * 4 + 64, dtype=torch.int32) % (grid * 4)
    out_cpu = torch.zeros(grid * 8, dtype=torch.int32)
    a = a_cpu.npu()
    idx = idx_cpu.npu()
    out = out_cpu.npu()
    options = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": _vector_core_count(),
    }
    scalar_camodel_cases[(grid,)](
        a, idx, out,
        N_LOAD=n_load, DEP_LEN=dep_len, N_STORE=n_store,
        **options,
    )
    torch.npu.synchronize()
    with open(report_path) as f:
        report = json.load(f)
    sm = report.get("stage_model", {})
    routes = sm.get("routes", {})
    return {
        "effective": report.get("effective_decision_kind"),
        "all_simd": routes.get("all_simd", {}).get("total_system_cycles"),
        "all_simt_only": routes.get("all_simt_only", {}).get("total_system_cycles"),
        "mixed": routes.get("mixed_simd_simt", {}).get("total_system_cycles"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2)
    args = ap.parse_args()
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    grid = args.grid
    print("case,grid,effective,all_simd,all_simt_only,mixed")
    for name in CASES:
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "route.json"
            s = score_case(name, grid, rp)
            print(f"{name},{grid},{s['effective']},{s['all_simd']},{s['all_simt_only']},{s['mixed']}")
            import sys
            sys.stdout.flush()


if __name__ == "__main__":
    main()
