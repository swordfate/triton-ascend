#!/usr/bin/env python3
"""E17b: K-axis sweep for the mixed envelope's iteration factor.
Shapes: mixed(4,4), mixed(8,4), ld_only(4KB), st_only(4KB), all grid=56.
K in {1,2,4,8,16,32,64}; SYS_CNT loop window; per-iteration ns = median/K.
Goal: measure g(K) = perIter(K)/perIter(32) per shape for the Tier-2
replacement (steady max+κ*min gives m at K=32; g(K) rescales by K).
"""
import json
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
def e17b_ld(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(iters):
        acc += tl.load(src + base + step * BLOCK + lane)
    t1 = _sys_cnt()
    tl.store(dst + pid, tl.sum(acc, axis=0))
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)

@triton.jit
def e17b_st(dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
    t0 = _sys_cnt()
    for step in range(iters):
        tl.store(dst + base + step * BLOCK + lane, v)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)

@triton.jit
def e17b_mix(src, dst, tick_ptr, iters: tl.constexpr, LB: tl.constexpr,
             SB: tl.constexpr):
    pid = tl.program_id(0)
    lane_l = tl.arange(0, LB)
    lane_s = tl.arange(0, SB)
    base_l = pid * iters * LB
    base_s = pid * iters * SB
    t0 = _sys_cnt()
    for step in range(iters):
        v = tl.load(src + base_l + step * LB + lane_l)
        tl.store(dst + base_s + step * SB + lane_s, v + 1.0)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

GRID = 56
KS = [1, 2, 4, 8, 16, 32, 64]


def measure(mod, kn, tensors, K):
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")

    def fn():
        kn[(GRID,)](*tensors, tick, iters=K, num_warps=4,
                    compile_mode="simd", auto_simt_scope_mode="off",
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
    gen_path = Path("/root/work/explore_contig/_gen_e17b.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e17b", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    out = {}
    shapes = [
        ("mix_4_4", "e17b_mix", 1024, 1024),
        ("mix_8_4", "e17b_mix", 2048, 1024),
        ("ld_4K", "e17b_ld", 1024, None),
        ("st_4K", "e17b_st", 1024, None),
    ]
    for tag, kn_name, LB, SB in shapes:
        maxK = max(KS)
        pool = torch.randn((GRID * maxK * LB,), dtype=torch.float32,
                           device="npu")
        dst = torch.zeros((GRID * maxK * max(SB, LB),), dtype=torch.float32,
                          device="npu")
        out[tag] = {}
        per = {}
        for K in KS:
            tensors = ((pool, dst) if kn_name == "e17b_mix"
                       else ((pool, dst) if kn_name == "e17b_ld" else (dst,)))
            per[K], whole = measure(mod, kn_name, tensors, K)
            out[tag][str(K)] = {"per_iter": per[K], "whole": whole}
            print("%-8s K=%2d  per-iter=%8.1f ns  whole=%9.1f ns" %
                  (tag, K, per[K], whole))
            sys.stdout.flush()
        base = per[32]
        out[tag]["g"] = {str(K): round(per[K] / base, 3) for K in KS}
        print("%-8s g(K): %s" % (tag, out[tag]["g"]))
        sys.stdout.flush()
    Path("/root/work/explore_contig/e17b_results.json").write_text(
        json.dumps(out, indent=1))
    print("saved e17b_results.json")


main()
