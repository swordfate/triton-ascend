#!/usr/bin/env python3
"""Run cost-model route reports for pure scalar scenarios and print SIMD/SIMT ratios.

This is the current lightweight validation entry after scalar-ldst calibration.

It launches the same three kernels as demo_scalar_scenarios.py in
`simd_simt + auto` mode, reads the route JSON, and prints:
    case, all_simd, all_simt_F1_if_available, all_simt_F4, selected

The route JSON normally only retains the best all_simt_only route, so if F1 is
needed please run with COSTMODEL_LOG_LEVEL=2 and inspect the trace, or extend
the report consumer to emit all legal factors.

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


@triton.jit
def scalar_compute_kernel(a, out, N: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.load(a + pid)
    for i in tl.static_range(N):
        s = s * 3 + i
        s = s ^ 0x2c6b
    tl.store(out + pid, s)


@triton.jit
def scalar_control_kernel(a, out, N: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.load(a + pid)
    acc = 0
    for i in tl.static_range(N):
        if (s & 1) == 0:
            acc += i
        else:
            acc -= i
        s = (s >> 1) ^ 0x2c6b
    tl.store(out + pid, acc)


def vector_core_count():
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(props["num_vectorcore"])


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
    row = {
        "case": name,
        "effective": report.get("effective_decision_kind"),
        "selected_sb": report.get("selected_superblock_factor"),
        "all_simd": routes.get("all_simd", {}).get("total_system_cycles"),
        "all_simt_only": routes.get("all_simt_only", {}).get("total_system_cycles"),
        "all_simt_factor": routes.get("all_simt_only", {}).get("route_superblock_factor"),
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

    a2 = torch.randint(0, 100, (grid,), dtype=torch.int32, device="npu")
    out2 = torch.zeros(grid, dtype=torch.int32, device="npu")
    row = run_case(
        "compute",
        lambda opts: scalar_compute_kernel[(grid,)](a2, out2, N=n, **opts),
        report_dir / "compute.json",
        grid,
    )
    if row:
        rows.append(row)

    a3 = torch.randint(0, 1 << 30, (grid,), dtype=torch.int32, device="npu")
    out3 = torch.zeros(grid, dtype=torch.int32, device="npu")
    row = run_case(
        "control",
        lambda opts: scalar_control_kernel[(grid,)](a3, out3, N=n, **opts),
        report_dir / "control.json",
        grid,
    )
    if row:
        rows.append(row)

    print("case,effective,selected_sb,all_simd,all_simt_only,ratio_simd_over_simt")
    for r in rows:
        simd = r["all_simd"]
        simt = r["all_simt_only"]
        ratio = simd / simt if simd and simt else float("nan")
        print(
            f"{r['case']},{r['effective']},{r['selected_sb']},"
            f"{simd:.1f},{simt:.1f},{ratio:.3f}"
        )


if __name__ == "__main__":
    main()
