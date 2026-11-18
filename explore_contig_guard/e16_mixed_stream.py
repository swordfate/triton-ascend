#!/usr/bin/env python3
"""E16: is the mixed load+store loop body (a) a shared-card-bandwidth
envelope (Tier 2: (L+S)/83.26 + startup), (b) two independent channels
running in parallel (max of separate stream times), or (c) per-op
startup-dominated (Tier 3)?

Kernels (grid=56, BLOCK=1024 f32 = 4KB per access, K=16/32):
  ld_only : loop{acc+=load} + tiny sink store   -> load stream steady rate
  st_only : loop{store 1.0}                     -> store stream steady rate
  mixed   : loop{v=load; store(v+1)}            -> 1L+1S per iteration
  mixed21 : loop{v0=load0; v1=load1; store}     -> 2L+1S per iteration
Timing: SYS_CNT ticks inside the kernel around the loop (guide 6.1),
ns = ticks*1.00008; median across 56 programs x 30 launches.
Models (per iteration, 4KB/access):
  Tier 2: 12.79 + total_bytes/83.26
  Tier 3: max over accesses of (startup + bytes/dir_rate)   [simd]
  Parallel-channel: max(measured ld_only, measured st_only) per access
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
def e16_ld_only(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
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
def e16_st_only(dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
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
def e16_mixed(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    t0 = _sys_cnt()
    for step in range(iters):
        v = tl.load(src + base + step * BLOCK + lane)
        tl.store(dst + base + step * BLOCK + lane, v + 1.0)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)

@triton.jit
def e16_mixed21(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    t0 = _sys_cnt()
    for step in range(iters):
        v0 = tl.load(src + base + step * BLOCK + lane)
        v1 = tl.load(src + base + iters * BLOCK + step * BLOCK + lane)
        tl.store(dst + base + step * BLOCK + lane, v0 + v1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

GRID = 56
LAUNCHES = 30


def run_variant(mod, name, K, BLOCK=1024):
    total = GRID * (K + 8) * BLOCK
    src = torch.randn((total,), dtype=torch.float32, device="npu")
    dst = torch.zeros((total,), dtype=torch.float32, device="npu")
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")
    kn = getattr(mod, name)
    args = {
        "e16_ld_only": (src, dst, tick),
        "e16_st_only": (dst, tick),
        "e16_mixed": (src, dst, tick),
        "e16_mixed21": (src, dst, tick),
    }[name]

    def fn():
        kn[(GRID,)](*args, iters=K, BLOCK=BLOCK, num_warps=4,
                    compile_mode="simd", auto_simt_scope_mode="off",
                    logical_program_count_hint=GRID,
                    physical_vector_core_count_hint=56)

    for _ in range(5):
        fn()
    torch.npu.synchronize()
    all_ns = []
    for _ in range(LAUNCHES):
        fn()
        torch.npu.synchronize()
        ticks = tick.cpu().reshape(GRID, 2)
        ns = (ticks[:, 1] - ticks[:, 0]) * 1.00008
        all_ns.extend(ns.tolist())
    med = statistics.median(all_ns)
    return med / K, med          # per-iteration ns, whole-loop ns


def main():
    gen_path = Path("/root/work/explore_contig/_gen_e16.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e16", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    rows = {}
    for name in ("e16_ld_only", "e16_st_only", "e16_mixed", "e16_mixed21"):
        rows[name] = {}
        for K in (16, 32):
            per, whole = run_variant(mod, name, K)
            rows[name][K] = (per, whole)
            print("%-12s K=%2d  loop=%8.1f ns  per-iter=%7.1f ns" %
                  (name, K, whole, per))
            sys.stdout.flush()
    # slope-based per-iteration (removes fixed loop overhead)
    print("\n== slope (T32-T16)/16 ==")
    for name in rows:
        slope = (rows[name][32][1] - rows[name][16][1]) / 16.0
        rows[name]["slope"] = slope
        print("%-12s %7.1f ns/iter" % (name, slope))
    # models
    L = S = 4096
    t2_11 = 12.79 + (L + S) / 83.26
    t2_21 = 12.79 + (2 * L + S) / 83.26
    t3_11 = max(105.31 + L / 167.56, 90.95 + S / 213.13)
    t3_21 = max(2 * (105.31 + L / 167.56), 90.95 + S / 213.13)
    ld, st = rows["e16_ld_only"]["slope"], rows["e16_st_only"]["slope"]
    print("\n== models vs measured (per iteration, ns) ==")
    print("mixed 1L+1S: measured %.1f | Tier2 %.1f | Tier3 %.1f | parallel-chan %.1f" %
          (rows["e16_mixed"]["slope"], t2_11, t3_11, max(ld, st)))
    print("mixed 2L+1S: measured %.1f | Tier2 %.1f | Tier3 %.1f | parallel-chan %.1f" %
          (rows["e16_mixed21"]["slope"], t2_21, t3_21, max(2 * ld, st)))
    print("ld_only: measured %.1f | Tier3 %.1f | bytes/83.26 %.1f" %
          (ld, 105.31 + L / 167.56, L / 83.26))
    print("st_only: measured %.1f | Tier3 %.1f | bytes/83.26 %.1f" %
          (st, 90.95 + S / 213.13, S / 83.26))
    out = {n: {str(k): v for k, v in r.items()} for n, r in rows.items()}
    Path("/root/work/explore_contig/e16_results.json").write_text(
        json.dumps(out, indent=1))


main()
