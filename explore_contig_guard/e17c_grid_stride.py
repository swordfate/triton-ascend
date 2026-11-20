#!/usr/bin/env python3
"""E17c: grid-stride vs contiguous addressing for the mixed body.

muls_add executes as 56 programs x 37 grid-stride trips (stride
56*4KB between trips).  Question: does the E17 max+k*min formula
(110.7 ns/trip for L=8KB,S=4KB contiguous) hold under grid-stride
addressing, or does the per-program scatter cost extra?
  gs  : block_id = pid + step*GRID (muls_add pattern), two 4KB loads
        (x,y buffers) + one 4KB store, K=37
  cont: same body, contiguous (base + step*BLOCK) addressing, K=37
"""
import statistics
import sys
from pathlib import Path

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)

@triton.jit
def e17c_gs(x, y, dst, tick_ptr, n_blocks, K: tl.constexpr,
            BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    for step in range(K):
        block_id = pid + step * nprog
        off = block_id * BLOCK + lane
        m = block_id < n_blocks
        v0 = tl.load(x + off, mask=m, other=0.0)
        v1 = tl.load(y + off, mask=m, other=0.0)
        tl.store(dst + off, v0 * 2.5 + v1, mask=m)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)

@triton.jit
def e17c_cont(x, y, dst, tick_ptr, n_blocks, K: tl.constexpr,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * K * BLOCK
    t0 = _sys_cnt()
    for step in range(K):
        off = base + step * BLOCK + lane
        v0 = tl.load(x + off)
        v1 = tl.load(y + off)
        tl.store(dst + off, v0 * 2.5 + v1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

GRID = 56
K = 37
BLOCK = 1024


def measure(mod, name, x, y, dst, tick, n_blocks):
    kn = getattr(mod, name)

    def fn():
        kn[(GRID,)](x, y, dst, tick, n_blocks, K=K, BLOCK=BLOCK,
                    num_warps=4, compile_mode="simd",
                    auto_simt_scope_mode="off",
                    logical_program_count_hint=GRID,
                    physical_vector_core_count_hint=56)

    for _ in range(5):
        fn()
    torch.npu.synchronize()
    all_ns = []
    for _ in range(30):
        fn()
        torch.npu.synchronize()
        ticks = tick.cpu().reshape(GRID, 2)
        ns = (ticks[:, 1] - ticks[:, 0]) * 1.00008
        all_ns.extend(ns.tolist())
    whole = statistics.median(all_ns)
    return whole / K, whole


def main():
    gen_path = Path("/root/work/explore_contig/_gen_e17c.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e17c", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    n_blocks = GRID * K
    x = torch.randn((n_blocks * BLOCK,), dtype=torch.float32, device="npu")
    y = torch.randn((n_blocks * BLOCK,), dtype=torch.float32, device="npu")
    dst = torch.zeros((n_blocks * BLOCK,), dtype=torch.float32, device="npu")
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")

    gs_per, gs_whole = measure(mod, "e17c_gs", x, y, dst, tick, n_blocks)
    print("grid-stride 37 trips: per-trip=%7.1f ns  whole=%9.1f ns" %
          (gs_per, gs_whole))
    # contiguous reference: same total bytes, contiguous per-program
    xc = torch.randn((GRID * K * BLOCK,), dtype=torch.float32, device="npu")
    yc = torch.randn((GRID * K * BLOCK,), dtype=torch.float32, device="npu")
    dstc = torch.zeros((GRID * K * BLOCK,), dtype=torch.float32,
                       device="npu")
    cont_per, cont_whole = measure(mod, "e17c_cont", xc, yc, dstc, tick,
                                   GRID * K)
    print("contiguous   37 trips: per-trip=%7.1f ns  whole=%9.1f ns" %
          (cont_per, cont_whole))
    print("\nmodels (per trip, L=8KB load + S=4KB store):")
    print("  Tier 2 (E17 max+k*min): %.1f" %
          (max(96.4, 79.6) + 0.18 * min(96.4, 79.6)))
    print("  Tier 3 (per-op max)   : %.1f" %
          max(2 * (105.31 + 4096 / 167.56), 90.95 + 4096 / 213.13))
    print("  byte-sum/83.26 +12.79 : %.1f" % (12.79 + 12288 / 83.26))
    print("\nmuls_add real body measured per-trip ~179 ns (6.625us/37)")


main()
