#!/usr/bin/env python3
"""Run cost-model route reports for pure scalar scenarios and print SIMD/SIMT ratios.

This is the current lightweight validation entry after scalar-ldst calibration.

It launches the same three kernels as demo_scalar_scenarios.py in
`simd_simt + auto` mode, reads the route JSON, and prints:
    case, all_simd, all_simt_F1, all_simt_F4, selected

The route JSON now includes all legal all_simt_only factors
(`all_simt_only_by_factor`), so both F1 and F4 cost-model ratios are printed.

Run on the Ascend server:
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 check_pure_scalar_costmodel_ratios.py
"""
import json
import pathlib

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


def run_case(name, launch, report_path, grid):
    if report_path.exists():
        report_path.unlink()
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": vector_core_count(),
    }
    launch(opts)
    torch.npu.synchronize()
    if not report_path.exists():
        print(f"{name},NO_REPORT")
        return None
    report = json.loads(report_path.read_text().strip().splitlines()[-1])
    routes = report["stage_model"]["routes"]
    simt1 = route_by_factor(routes, 1)
    simt4 = route_by_factor(routes, 4)
    row = {
        "case": name,
        "effective": report.get("effective_decision_kind"),
        "selected_sb": report.get("selected_superblock_factor"),
        "all_simd": routes.get("all_simd", {}).get("total_system_cycles"),
        "all_simt_only": routes.get("all_simt_only", {}).get("total_system_cycles"),
        "all_simt_factor": routes.get("all_simt_only", {}).get("route_superblock_factor"),
        "all_simt_f1": simt1.get("total_system_cycles") if simt1 else None,
        "all_simt_f4": simt4.get("total_system_cycles") if simt4 else None,
    }
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2048)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    grid = args.grid
    n = args.n
    report_dir = pathlib.Path("/tmp/scalar_cm_reports")
    report_dir.mkdir(exist_ok=True)

    a = torch.arange(grid * n, dtype=torch.int32, device="npu") % 7
    b = torch.zeros(grid * n, dtype=torch.int32, device="npu")
    out = torch.zeros(grid, dtype=torch.int32, device="npu")

    rows = []
    row = run_case(
        "ldst",
        lambda opts: scalar_ldst_kernel[(grid,)](
            a, b, out, N_LD=n, N_ST=n, **opts
        ),
        report_dir / "ldst.json",
        grid,
    )
    if row:
        rows.append(row)

    print("case,effective,selected_sb,all_simd,all_simt_f1,all_simt_f4,ratio_simd_over_simt_f1,ratio_simd_over_simt_f4")
    for r in rows:
        simd = r["all_simd"]
        f1 = r["all_simt_f1"]
        f4 = r["all_simt_f4"]
        ratio1 = simd / f1 if simd and f1 else float("nan")
        ratio4 = simd / f4 if simd and f4 else float("nan")
        f1_s = f"nan" if f1 is None else f"{f1:.1f}"
        f4_s = f"nan" if f4 is None else f"{f4:.1f}"
        print(
            f"{r['case']},{r['effective']},{r['selected_sb']},"
            f"{simd:.1f},{f1_s},{f4_s},{ratio1:.3f},{ratio4:.3f}"
        )


if __name__ == "__main__":
    main()
