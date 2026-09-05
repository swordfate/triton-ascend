#!/usr/bin/env python3
"""Compare cost-model SIMD/SIMT scores with real runtime for scalar-only kernels.

Unlike compare_route_runtime.py, these cases contain no vector tile / dot /
recurrence. They only execute scalar loads/stores.

Driver mode:
    python3 compare_scalar_only_runtime.py --grid 512

Worker mode (used internally):
    TRITON_ASCEND_COMPILE_MODE=simd \
    python3 compare_scalar_only_runtime.py --case direct_load_4 --mode simd --grid 512
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
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
def scalar_only_kernel(
    a_ptr,
    idx_ptr,
    out_ptr,
    N_LOAD: tl.constexpr,
    DEP_LEN: tl.constexpr,
    N_STORE: tl.constexpr,
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
    "direct_load_8": (8, 0, 1),
    "direct_store_1": (1, 0, 1),
    "direct_store_4": (1, 0, 4),
    "direct_store_8": (1, 0, 8),
    "dep_load_4": (1, 4, 1),
    "dep_load_8": (1, 8, 1),
}


def launch_opts(report_path, grid, mode):
    opts = {
        "num_warps": 1,
        "compile_mode": mode,
        "auto_simt_scope_mode": "off" if mode != "simd_simt" else "auto",
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": _vector_core_count(),
    }
    if mode == "simd_simt":
        opts["enable_auto_blockify"] = True
        opts["auto_simt_scope_dump"] = str(report_path)
    return opts


def make_launch(case, grid, mode, report_path):
    n_load, dep_len, n_store = CASES[case]
    a_cpu = torch.arange(grid + 8 * 16 + 64, dtype=torch.int32)
    idx_cpu = torch.arange(grid * 4 + 64, dtype=torch.int32) % (grid * 4)
    out_cpu = torch.zeros(grid * 8, dtype=torch.int32)
    a = a_cpu.npu()
    idx = idx_cpu.npu()
    out = out_cpu.npu()

    def launch():
        scalar_only_kernel[(grid,)](
            a, idx, out,
            N_LOAD=n_load,
            DEP_LEN=dep_len,
            N_STORE=n_store,
            **launch_opts(report_path, grid, mode),
        )
    return launch


def measure_worker(case, mode, grid):
    os.environ["TRITON_ASCEND_COMPILE_MODE"] = mode
    os.environ["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "off" if mode != "simd_simt" else "auto"
    report = Path(tempfile.gettempdir()) / f"scalar_only_{case}_route.json"
    if report.exists():
        report.unlink()
    launch = make_launch(case, grid, mode, report)
    for _ in range(5):
        launch()
    torch.npu.synchronize()
    times = []
    for _ in range(30):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e))
    print(f"RESULT {case} {mode} {statistics.median(times):.6f}")


def get_scores(case, grid):
    os.environ["TRITON_ASCEND_COMPILE_MODE"] = "simd_simt"
    os.environ["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "auto"
    report = Path(tempfile.gettempdir()) / f"scalar_only_{case}_route.json"
    if report.exists():
        report.unlink()
    launch = make_launch(case, grid, "simd_simt", report)
    launch()
    torch.npu.synchronize()
    if not report.exists():
        raise SystemExit(f"no route report for {case}")
    with open(report) as f:
        data = json.load(f)
    routes = data["stage_model"]["routes"]
    return (routes["all_simd"]["total_system_cycles"],
            routes["all_simt_only"]["total_system_cycles"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--mode", choices=["simd", "simt_only", "simd_simt"])
    ap.add_argument("--grid", type=int, default=512)
    args = ap.parse_args()

    if args.case and args.mode:
        measure_worker(args.case, args.mode, args.grid)
        return

    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    grid = args.grid
    print(f"# grid={grid}", file=sys.stderr)
    print("case,score_simd,score_simt,measured_simd_ms,measured_simt_ms,"
          "score_winner,measured_winner,consistent")
    for case in sorted(CASES):
        score_simd, score_simt = get_scores(case, grid)
        measured = {}
        for mode in ["simd", "simt_only"]:
            env = os.environ.copy()
            env["TRITON_ASCEND_COMPILE_MODE"] = mode
            env["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "off"
            proc = subprocess.run(
                [sys.executable, __file__, "--case", case, "--mode", mode,
                 "--grid", str(grid)],
                env=env, text=True, capture_output=True)
            if proc.returncode != 0:
                print(f"worker failed {case} {mode}\n{proc.stdout}\n{proc.stderr}",
                      file=sys.stderr)
                raise SystemExit(proc.returncode)
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT"):
                    _, _, m, val = line.split()
                    measured["simt" if m == "simt_only" else m] = float(val)
            if "simd" not in measured and "simt" not in measured:
                print(f"no result {case} {mode}\n{proc.stdout}\n{proc.stderr}",
                      file=sys.stderr)
                raise SystemExit(1)
        score_winner = "simd" if score_simd < score_simt else "simt"
        measured_winner = "simd" if measured["simd"] < measured["simt"] else "simt"
        consistent = score_winner == measured_winner
        print(f"{case},{score_simd:.3f},{score_simt:.3f},"
              f"{measured['simd']:.6f},{measured['simt']:.6f},"
              f"{score_winner},{measured_winner},{consistent}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
