#!/usr/bin/env python3
"""Measure and compare load-only / store-only / load+store SIMD vs SIMT ratios.

This is the reproducible script behind README section 4.5.

It runs the same scalar_ldst_kernel used by demo_scalar_scenarios.py with:
  - N_LD=ops,  N_ST=0   : load-only
  - N_LD=0,    N_ST=ops : store-only
  - N_LD=ops,  N_ST=ops : load+store

For each variant it:
  1. measures real SIMD, SIMT F1, SIMT F4 median time;
  2. runs cost model route report (simd_simt + auto) and prints all_simd / all_simt_only F1 and F4 ratios.

Run:
    source ~/env_ascend.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 check_scalar_ldst_split_ratios.py --grid 2048 --ops 32
"""
import argparse
import json
import pathlib
import statistics

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def scalar_ldst_kernel(a, b, out, N_LD: tl.constexpr, N_ST: tl.constexpr):
    pid = tl.program_id(0)
    s = 0
    for i in tl.static_range(N_LD):
        s += tl.load(a + pid * N_LD + i)
    for i in tl.static_range(N_ST):
        tl.store(b + pid * N_ST + i, s + i)
    tl.store(out + pid, s)


def vector_core_count():
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(props["num_vectorcore"])


def route_by_factor(routes, factor):
    if factor == routes.get("all_simt_only", {}).get("route_superblock_factor"):
        return routes.get("all_simt_only")
    for plan in routes.get("all_simt_only_by_factor", []):
        if plan.get("route_superblock_factor") == factor:
            return plan
    return None


def launch_opts(mode, sf, grid, report_path=None):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": grid,
        }
    if mode == "simt_only":
        return {
            "num_warps": 1,
            "compile_mode": "simt_only",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": True,
            "superblock_factor": sf,
            "logical_program_count_hint": grid,
        }
    # simd_simt auto route
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": vector_core_count(),
    }
    if report_path is not None:
        opts["auto_simt_scope_dump"] = str(report_path)
    return opts


def make_launch(grid, nld, nst, mode, sf, report_path=None):
    a = torch.arange(grid * max(nld, 1), dtype=torch.int32, device="npu") % 7
    b = torch.zeros(grid * max(nst, 1), dtype=torch.int32, device="npu")
    out = torch.zeros(grid, dtype=torch.int32, device="npu")
    opts = launch_opts(mode, sf, grid, report_path)

    def launch():
        scalar_ldst_kernel[(grid,)](
            a, b, out, N_LD=nld, N_ST=nst, **opts
        )

    return launch


def measure(launch, reps=15):
    for _ in range(5):
        launch()
    torch.npu.synchronize()
    times = []
    for _ in range(reps):
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        launch()
        en.record()
        torch.npu.synchronize()
        times.append(st.elapsed_time(en))
    return statistics.median(times)


def costmodel_ratios(grid, nld, nst):
    report_path = pathlib.Path(f"/tmp/ldst_split_{nld}_{nst}.json")
    if report_path.exists():
        report_path.unlink()
    launch = make_launch(grid, nld, nst, "simd_simt", 1, report_path)
    launch()
    torch.npu.synchronize()
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text().strip().splitlines()[-1])
    routes = report["stage_model"]["routes"]
    simd = routes["all_simd"]["total_system_cycles"]
    f1 = route_by_factor(routes, 1)
    f4 = route_by_factor(routes, 4)
    ratios = {"simd": simd}
    ratios["f1"] = f1.get("total_system_cycles") if f1 else None
    ratios["f4"] = f4.get("total_system_cycles") if f4 else None
    ratios["model_f1"] = simd / ratios["f1"] if simd and ratios["f1"] else None
    ratios["model_f4"] = simd / ratios["f4"] if simd and ratios["f4"] else None
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2048)
    ap.add_argument("--ops", type=int, default=32)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--model-only", action="store_true")
    args = ap.parse_args()

    grid = args.grid
    ops = args.ops
    print(f"# grid={grid} ops={ops} num_warps=1")
    if args.model_only:
        print("variant,model_simd_over_simt_f1,model_simd_over_simt_f4")
    else:
        print("variant,simd_ms,simt_f1_ms,simt_f4_ms,model_simd_over_simt_f1,model_simd_over_simt_f4")
    for name, nld, nst in [
        ("load-only", ops, 0),
        ("store-only", 0, ops),
        ("load+store", ops, ops),
    ]:
        model = costmodel_ratios(grid, nld, nst)
        if args.model_only:
            if model is None:
                print(f"{name},NA,NA", flush=True)
            else:
                m1 = "NA" if model["model_f1"] is None else f"{model['model_f1']:.3f}"
                m4 = "NA" if model["model_f4"] is None else f"{model['model_f4']:.3f}"
                print(f"{name},{m1},{m4}", flush=True)
            continue
        simd = measure(make_launch(grid, nld, nst, "simd", 1), args.reps)
        simt1 = measure(make_launch(grid, nld, nst, "simt_only", 1), args.reps)
        simt4 = measure(make_launch(grid, nld, nst, "simt_only", 4), args.reps)
        if model is None:
            print(
                f"{name},{simd:.6f},{simt1:.6f},{simt4:.6f},NA,NA",
                flush=True,
            )
        else:
            m1 = "NA" if model["model_f1"] is None else f"{model['model_f1']:.3f}"
            m4 = "NA" if model["model_f4"] is None else f"{model['model_f4']:.3f}"
            print(
                f"{name},{simd:.6f},{simt1:.6f},{simt4:.6f},{m1},{m4}",
                flush=True,
            )


if __name__ == "__main__":
    main()
