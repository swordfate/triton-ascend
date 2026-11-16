#!/usr/bin/env python3
"""diag_acopyg64.py - compile the guard A_copy_f32_4k_g64 geometry once with
traces on stderr."""
import os
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '2')
import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit(do_not_specialize=['n'])
def gc_scopy(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)


def main():
    torch.npu.set_device(0)
    n = 64 * 1024
    src = torch.randn((n,), dtype=torch.float32, device='npu')
    dst = torch.empty_like(src)
    cores = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    gc_scopy[(64,)](src, dst, n, BLOCK=1024,
                    num_warps=4, compile_on_910_95=True,
                    compile_mode='simd_simt', superblock_factor=1,
                    logical_program_count_hint=64,
                    physical_vector_core_count_hint=cores,
                    auto_simt_scope_mode='auto',
                    auto_simt_scope_dump='/tmp/diag_ac64_route.json')
    torch.npu.synchronize()
    print('COMPILE_OK')


main()
