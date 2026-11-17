#!/usr/bin/env python3
"""E17: two-direction sweep for the user-proposed Tier-2 replacement.

Model to validate:  perIter(L,S) = max(TL(L), TS(S)) [+ kappa*min]
where TL/TS are concave per-direction time curves (rate ramps with
transaction size).  Sweep:
  ld_only : L in {4,8,16,32} KB      (acc += v, K=16/32 slope)
  st_only : S in {4,8,16,32} KB
  mixed   : (L,S) in {(4,4),(8,4),(16,4),(8,8),(16,8),(8,16),(16,16)} KB
            non-aliasing (src != dst), store depends on load (v+1)
  inplace : (L,S)=(8,8) with dst == src slice (loop-carried dependency)
Timing: SYS_CNT loop window (guide 6.1), grid=56, slope over K=16/32.
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
def e17_ld(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
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
def e17_st(dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
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
def e17_mix(src, dst, tick_ptr, iters: tl.constexpr,
            LB: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    lane_l = tl.arange(0, LB)
    lane_s = tl.arange(0, SB)
    base_l = pid * iters * LB
    base_s = pid * iters * SB
    t0 = _sys_cnt()
    for step in range(iters):
        v = tl.load(src + base_l + step * LB + lane_l)
        sv = tl.reshape(v + 1.0, (SB,)) if LB != SB else v + 1.0
        tl.store(dst + base_s + step * SB + lane_s, sv)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)

@triton.jit
def e17_inplace(buf, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    t0 = _sys_cnt()
    for step in range(iters):
        v = tl.load(buf + base + step * BLOCK + lane)
        tl.store(buf + base + step * BLOCK + lane, v + 1.0)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

GRID = 56


def measure(mod, name, tensors, K, blocks):
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")
    kn = getattr(mod, name)

    def fn():
        kn[(GRID,)](*tensors, tick, iters=K, num_warps=4,
                    **blocks, compile_mode="simd",
                    auto_simt_scope_mode="off",
                    logical_program_count_hint=GRID,
                    physical_vector_core_count_hint=56)

    for _ in range(5):
        fn()
    torch.npu.synchronize()
    all_ns = []
    for _ in range(20):
        fn()
        torch.npu.synchronize()
        ticks = tick.cpu().reshape(GRID, 2)
        ns = (ticks[:, 1] - ticks[:, 0]) * 1.00008
        all_ns.extend(ns.tolist())
    med = statistics.median(all_ns)
    return med / K


def main():
    gen_path = Path("/root/work/explore_contig/_gen_e17.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e17", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    out = {}
    # ld-only sweep
    for kb in (4, 8, 16, 32):
        blk = kb * 256
        pool = torch.randn((GRID * 40 * blk,), dtype=torch.float32,
                           device="npu")
        dst = torch.zeros((GRID,), dtype=torch.float32, device="npu")
        t16 = measure(mod, "e17_ld", (pool, dst), 16, {"BLOCK": blk})
        t32 = measure(mod, "e17_ld", (pool, dst), 32, {"BLOCK": blk})
        slope = (t32 - t16) / 16.0
        out[f"ld_{kb}K"] = {"t16": t16, "t32": t32, "slope": slope,
                            "rate": kb * 1024 / slope}
        print("ld_only  %2dKB: t16=%7.1f t32=%7.1f slope=%7.1f ns  rate=%6.1f B/ns" %
              (kb, t16, t32, slope, kb * 1024 / slope))
        sys.stdout.flush()
    # st-only sweep
    for kb in (4, 8, 16, 32):
        blk = kb * 256
        dst = torch.zeros((GRID * 40 * blk,), dtype=torch.float32,
                          device="npu")
        t16 = measure(mod, "e17_st", (dst,), 16, {"BLOCK": blk})
        t32 = measure(mod, "e17_st", (dst,), 32, {"BLOCK": blk})
        slope = (t32 - t16) / 16.0
        out[f"st_{kb}K"] = {"t16": t16, "t32": t32, "slope": slope,
                            "rate": kb * 1024 / slope}
        print("st_only  %2dKB: t16=%7.1f t32=%7.1f slope=%7.1f ns  rate=%6.1f B/ns" %
              (kb, t16, t32, slope, kb * 1024 / slope))
        sys.stdout.flush()
    # mixed matrix (non-aliasing), LKB/SKB
    for lkb, skb in ((4, 4), (8, 4), (16, 4), (8, 8), (16, 8), (8, 16),
                     (16, 16)):
        lb, sb = lkb * 256, skb * 256
        pool = torch.randn((GRID * 40 * lb,), dtype=torch.float32,
                           device="npu")
        dst = torch.zeros((GRID * 40 * sb,), dtype=torch.float32,
                          device="npu")
        t16 = measure(mod, "e17_mix", (pool, dst), 16,
                      {"LB": lb, "SB": sb})
        t32 = measure(mod, "e17_mix", (pool, dst), 32,
                      {"LB": lb, "SB": sb})
        slope = (t32 - t16) / 16.0
        out[f"mix_{lkb}_{skb}"] = {"t16": t16, "t32": t32, "slope": slope}
        print("mix %2dKB/%2dKB: t16=%7.1f t32=%7.1f slope=%7.1f ns" %
              (lkb, skb, t16, t32, slope))
        sys.stdout.flush()
    # in-place (loop-carried dependency), 8KB/8KB
    buf = torch.randn((GRID * 40 * 2048,), dtype=torch.float32, device="npu")
    t16 = measure(mod, "e17_inplace", (buf,), 16, {"BLOCK": 2048})
    t32 = measure(mod, "e17_inplace", (buf,), 32, {"BLOCK": 2048})
    slope = (t32 - t16) / 16.0
    out["inplace_8_8"] = {"t16": t16, "t32": t32, "slope": slope}
    print("inplace 8KB/8KB (loop-carried dep): slope=%7.1f ns" % slope)
    Path("/root/work/explore_contig/e17_results.json").write_text(
        json.dumps(out, indent=1))
    print("saved e17_results.json")


main()
