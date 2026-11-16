#!/usr/bin/env python3
"""CAModel simulator input: minimal straight-copy triton kernel.
Usage: python3 wave_sim_tiny.py <grid>   (keep grid small; sim is slow)
Data: grid x 1024 f32 = grid x 4KB (tiny on purpose)."""
import sys

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit(do_not_specialize=["n"])
def gc_copy(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)


def main():
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    block = 1024
    n = grid * block
    src = torch.randn((n,), dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    cores = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    gc_copy[(grid,)](src, dst, src.numel(), BLOCK=block,
                     num_warps=4, compile_mode="simd",
                     logical_program_count_hint=grid,
                     physical_vector_core_count_hint=cores,
                     auto_simt_scope_mode="off")
    torch.npu.synchronize()
    ok = torch.allclose(dst, src)
    print("LAUNCH_OK grid=%d bytes=%d correct=%s" % (grid, n * 4, ok))


main()
