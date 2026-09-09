#!/usr/bin/env python3
"""Cost-model ratio for uniform dependent scalar load/store kernels."""
import argparse
import json
import pathlib

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def dep_load_kernel(next_ptr, start_ptr, out, N: tl.constexpr):
    pid = tl.program_id(0)
    cur = tl.load(start_ptr + pid)
    s = 0
    for i in tl.static_range(N):
        cur = tl.load(next_ptr + cur)
        s += cur
    tl.store(out + pid, s)


@triton.jit
def dep_store_kernel(next_ptr, data_ptr, start_ptr, out, N: tl.constexpr):
    pid = tl.program_id(0)
    cur = tl.load(start_ptr + pid)
    s = 0
    for i in tl.static_range(N):
        cur = tl.load(next_ptr + cur)
        tl.store(data_ptr + cur, s + i)
        s += cur
    tl.store(out + pid, s)


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
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": int(props["num_vectorcore"]),
    }
    launch(opts)
    torch.npu.synchronize()
    if not report_path.exists():
        print(f"{name},NO_REPORT")
        return None
    report = json.loads(report_path.read_text().strip().splitlines()[-1])
    routes = report["stage_model"]["routes"]
    simd = routes["all_simd"]["total_system_cycles"]
    f1 = route_by_factor(routes, 1)
    f4 = route_by_factor(routes, 4)
    f1v = f1["total_system_cycles"] if f1 else None
    f4v = f4["total_system_cycles"] if f4 else None
    return name, simd / f1v if simd and f1v else None, simd / f4v if simd and f4v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=4)
    ap.add_argument("--grid", type=int, default=2048)
    args = ap.parse_args()
    grid = args.grid
    ops = args.ops
    total = max(1, grid * ops)
    next_ptr = torch.randperm(total, dtype=torch.int32).npu()
    start_ptr = torch.randint(0, total, (grid,), dtype=torch.int32).npu()
    data = torch.zeros(total, dtype=torch.int32, device="npu")
    out = torch.zeros(grid, dtype=torch.int32, device="npu")
    report_dir = pathlib.Path("/tmp/scalar_dep_cm_reports")
    report_dir.mkdir(exist_ok=True)

    rows = []
    for name, launch in [
        ("depl", lambda opts: dep_load_kernel[(grid,)](next_ptr, start_ptr, out, N=ops, **opts)),
        ("deps", lambda opts: dep_store_kernel[(grid,)](next_ptr, data, start_ptr, out, N=ops, **opts)),
    ]:
        row = run_case(name, launch, report_dir / f"{name}.json", grid)
        if row:
            rows.append(row)
    print("variant,model_simd_over_simt_f1,model_simd_over_simt_f4")
    for name, r1, r4 in rows:
        print(f"{name},{r1:.3f},{r4:.3f}")


if __name__ == "__main__":
    main()
