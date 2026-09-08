#!/usr/bin/env python3
"""Triton arithmetic calibration draft for f32.add / sin / cos.

Method: use the SAME Triton kernel for both `simd` and `simt_only`.
The only thing changed between modes is `compile_mode`; the backend decides
how to lower the arithmetic (including any Taylor expansion automatically).

Run on Ascend server:
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=1
    python3 bench_arith_triton.py --modes simd simt_only --ops add sin cos
"""
from __future__ import annotations

import argparse
import statistics

import torch
import torch_npu
import triton
import triton.language as tl


# Number of independent vector chains per program.  The CCE tput probe shows
# that ILP>=4 is enough to saturate SIMD vadd; keeping the same kernel for SIMT
# measures the actual SIMT lowering of the same Triton source.
ILP = 4

DEFAULT_GRID = 56       # one AIV per program, adjust to device core count
DEFAULT_BLOCK = 1024    # elements per vector program


@triton.jit
def _apply_op(x, MODE: tl.constexpr, k):
    if MODE == 0:       # add
        return x + k
    elif MODE == 1:     # sin
        return tl.sin(x)
    else:               # cos
        return tl.cos(x)


@triton.jit
def arith_kernel(x, out, ITERS, MODE: tl.constexpr,
                 BLOCK: tl.constexpr, ILP: tl.constexpr):
    pid = tl.program_id(0)
    k = 1.0000001
    base = pid * (BLOCK * ILP)
    offs0 = base + tl.arange(0, BLOCK)
    offs1 = offs0 + BLOCK
    offs2 = offs1 + BLOCK
    offs3 = offs2 + BLOCK

    x0 = tl.load(x + offs0)
    x1 = tl.load(x + offs1)
    x2 = tl.load(x + offs2)
    x3 = tl.load(x + offs3)

    for _ in tl.range(0, ITERS):
        x0 = _apply_op(x0, MODE, k)
        x1 = _apply_op(x1, MODE, k)
        x2 = _apply_op(x2, MODE, k)
        x3 = _apply_op(x3, MODE, k)

    tl.store(out + offs0, x0)
    tl.store(out + offs1, x1)
    tl.store(out + offs2, x2)
    tl.store(out + offs3, x3)


def _opts(mode: str, grid: int, superblock_factor: int = 1):
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
            "superblock_factor": superblock_factor,
            "logical_program_count_hint": grid,
        }
    raise ValueError(mode)


def _median_launch_ms(launch, reps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _tensor_for_kernel(grid: int, block: int):
    n = grid * block * ILP
    x = torch.rand(n, dtype=torch.float32, device="npu")
    out = torch.zeros(n, dtype=torch.float32, device="npu")
    return x, out


def measure(mode, op, ops, grid, block, reps, sf):
    x, out = _tensor_for_kernel(grid, block)
    opts = _opts(mode, grid, sf)

    def launch():
        arith_kernel[(grid,)](
            x, out, ITERS=ops, MODE=op, BLOCK=block, ILP=ILP, **opts
        )

    return _median_launch_ms(launch, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--ops", nargs="+", default=["add", "sin", "cos"],
                    choices=["add", "sin", "cos"])
    ap.add_argument("--ops-list", type=int, nargs="+", default=[0, 16, 64, 128, 256, 512])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    ap.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    ap.add_argument("--superblock-factor", type=int, default=1)
    args = ap.parse_args()

    op_code = {"add": 0, "sin": 1, "cos": 2}

    print("mode,op,ops,median_ms")
    for mode in args.modes:
        for op_name in args.ops:
            op = op_code[op_name]
            for ops in args.ops_list:
                try:
                    ms = measure(
                        mode, op, ops, args.grid, args.block,
                        args.reps, args.superblock_factor,
                    )
                    print(f"{mode},{op_name},{ops},{ms:.6f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"{mode},{op_name},{ops},ERROR:{type(exc).__name__}:{exc}",
                          flush=True)


if __name__ == "__main__":
    main()
