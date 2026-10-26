#!/usr/bin/env python3
"""Minimal kernel for CAModel runs of the row-tail / column-tail geometries.

Usage (do not call directly - use run_align_camodel.sh, which sets up the
simulator environment and msprof):

    python3 align_camodel.py <rows> <cols> [dtype]

It launches exactly one program of a packed rank-two tile with an extent mask,
which is the shape whose lowering changes when the trailing extent stops being a
power of two.  Nothing is timed here: the point is to let msprof op simulator
capture the microarchitectural trace (burst parameters, per-instruction cycles).
"""

import sys

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def align_kernel(source, output, R: tl.constexpr, C: tl.constexpr,
                 RPAD: tl.constexpr, CPAD: tl.constexpr,
                 ROW_STRIDE: tl.constexpr):
    rows = tl.arange(0, RPAD)[:, None]
    cols = tl.arange(0, CPAD)[None, :]
    offsets = rows * ROW_STRIDE + cols
    mask = (rows < R) & (cols < C)
    value = tl.load(source + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, value, mask=mask)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16,
          "int8": torch.int8, "int64": torch.int64}


def main():
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    dtype = sys.argv[3] if len(sys.argv) > 3 else "fp32"
    rpad = 1 << max(0, (rows - 1).bit_length())
    cpad = 1 << max(0, (cols - 1).bit_length())
    elements = rpad * cpad + 64
    torch_dtype = DTYPES[dtype]
    source = torch.ones(elements, dtype=torch_dtype, device="npu")
    output = torch.zeros(elements, dtype=torch_dtype, device="npu")
    align_kernel[(1, )](source, output, rows, cols, rpad, cpad, cols,
                       compile_mode="simd", num_warps=4)
    torch.npu.synchronize()
    print("ALIGN_RUN_OK rows=%d cols=%d rpad=%d cpad=%d dtype=%s"
          % (rows, cols, rpad, cpad, dtype))


if __name__ == "__main__":
    main()
