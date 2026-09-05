#!/usr/bin/env python3
"""Tiny Triton scalar load/store cases for CAModel validation.

Each case is intentionally tiny so CAModel can finish:
  direct_load_N   : N independent scalar loads
  direct_store_N  : N scalar stores
  dep_load_N      : N dependent scalar loads (address depends on previous load)

Run on real NPU first to confirm route/compilation:
    python3 scalar_camodel_cases.py direct_load_4 --grid 2

Run under CAModel:
    msprof op simulator --kernel-name=scalar_camodel_cases \
        python3 scalar_camodel_cases.py direct_load_4 --grid 2
"""
import argparse
import os

import torch
import torch_npu
import triton
import triton.language as tl

if not os.environ.get("TRITON_ASCEND_COMPILE_MODE"):
    os.environ["TRITON_ASCEND_COMPILE_MODE"] = "simd"
if not os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE"):
    os.environ["TRITON_ASCEND_AUTO_SIMT_SCOPE"] = "off"


@triton.jit
def scalar_camodel_cases(
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
    # case -> (N_LOAD, DEP_LEN, N_STORE)
    "direct_load_1": (1, 0, 1),
    "direct_load_2": (2, 0, 1),
    "direct_load_4": (4, 0, 1),
    "direct_store_1": (0, 0, 1),
    "direct_store_2": (0, 0, 2),
    "direct_store_4": (0, 0, 4),
    "dep_load_1": (1, 1, 1),
    "dep_load_2": (1, 2, 1),
    "dep_load_4": (1, 4, 1),
}


def run_case(name: str, grid: int):
    if name not in CASES:
        raise SystemExit(f"unknown case {name}; available: {sorted(CASES)}")
    n_load, dep_len, n_store = CASES[name]

    a_cpu = torch.arange(grid + 8 * 16 + 64, dtype=torch.int32)
    idx_cpu = torch.arange(grid * 4 + 64, dtype=torch.int32) % (grid * 4)
    out_cpu = torch.zeros(grid * 8, dtype=torch.int32)

    a = a_cpu.npu()
    idx = idx_cpu.npu()
    out = out_cpu.npu()

    scalar_camodel_cases[(grid,)](
        a,
        idx,
        out,
        N_LOAD=n_load,
        DEP_LEN=dep_len,
        N_STORE=n_store,
        num_warps=1,
    )
    torch.npu.synchronize()
    print(f"SCALAR_CAMODEL_CASE_DONE {name} grid={grid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--grid", type=int, default=2)
    args = ap.parse_args()
    run_case(args.case, args.grid)


if __name__ == "__main__":
    main()
