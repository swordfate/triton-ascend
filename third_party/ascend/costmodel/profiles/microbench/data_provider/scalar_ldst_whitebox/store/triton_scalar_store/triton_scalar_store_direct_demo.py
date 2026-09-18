#!/usr/bin/env python3
"""Minimal direct scalar-store probes to check whether scalar store really uses MTE3.

Modes:
  const   : tl.store(out_ptr + pid, 1.0)                  # no cast / no arithmetic
  const0  : tl.store(out_ptr, 1.0) with grid=1            # absolute simplest
  cast    : tl.store(out_ptr + pid, pid.to(tl.float32))   # conversion only
  add     : v = pid.to(tl.float32) + 1.0; tl.store(..., v)
  arg     : tl.store(out_ptr + pid, v), v is a runtime scalar argument

Run under msopprof simulator, one mode per process, e.g.:
  MODE=const bash run_triton_scalar_store_direct_camodel.sh
"""
import argparse

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver as triton_driver
from triton.runtime.jit import JITFunction


@triton.jit
def _triton_scalar_store_const(out_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, 1.0)


@triton.jit
def _triton_scalar_store_const0(out_ptr):
    tl.store(out_ptr, 1.0)


@triton.jit
def _triton_scalar_store_cast(out_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, pid.to(tl.float32))


@triton.jit
def _triton_scalar_store_add(out_ptr):
    pid = tl.program_id(0)
    v = pid.to(tl.float32) + 1.0
    tl.store(out_ptr + pid, v)


@triton.jit(do_not_specialize=["value"])
def _triton_scalar_store_arg(out_ptr, value):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, value)


KERNELS = {
    "const": _triton_scalar_store_const,
    "const0": _triton_scalar_store_const0,
    "cast": _triton_scalar_store_cast,
    "add": _triton_scalar_store_add,
    "arg": _triton_scalar_store_arg,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=sorted(KERNELS), default="const")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    try:
        props = triton_driver.active.utils.get_device_properties(torch.npu.current_device())
        vec_cores = int(props.get("num_vectorcore", 0))
    except Exception:
        vec_cores = 0

    original_run = JITFunction.run

    def run_with_hint(self, *f_args, **f_kwargs):
        if vec_cores and "physical_vector_core_count_hint" not in f_kwargs:
            f_kwargs["physical_vector_core_count_hint"] = vec_cores
        return original_run(self, *f_args, **f_kwargs)

    JITFunction.run = run_with_hint

    grid = 1 if args.mode == "const0" else args.grid
    out = torch.zeros(max(grid, 1), dtype=torch.float32).to("npu")
    opts = {
        "num_warps": 1,
        "compile_mode": "simd",
        "auto_simt_scope_mode": "off",
        "enable_auto_blockify": False,
        "superblock_factor": 1,
        "logical_program_count_hint": grid,
    }
    kernel = KERNELS[args.mode]
    if args.mode == "arg":
        kernel[(grid,)](out, 2.5, **opts)
    else:
        kernel[(grid,)](out, **opts)

    torch_npu.npu.synchronize()
    print(f"LAUNCHED triton_scalar_store_{args.mode} grid={grid}", flush=True)


if __name__ == "__main__":
    main()
