#!/usr/bin/env python3
"""E16b: why is the achieved load rate (~49-83 B/ns) far below the tier-3
constant 167.56 B/cycle?  Variants (ld-only, grid=56, SYS_CNT loop window):
  base      : acc += load, BLOCK=1024 K=32 (serial acc dependency, 4KB/iter)
  indep4    : 4 independent accumulator chains, 4 loads/iter (16KB/iter)
  bigblock  : acc += load, BLOCK=4096 K=8 (16KB transactions)
  indep4_bb : indep4 with BLOCK=4096 K=4 (64KB/iter)
  *_cold    : same with buffer rotation across launches (DRAM-cold)
  single    : K=1, one 64KB load (tier-3 formula's own regime)
Reports per-program achieved bytes/ns.
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
def e16b_base(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
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
def e16b_indep4(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK * 4
    acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(iters):
        off = base + step * BLOCK
        acc0 += tl.load(src + off + lane)
        acc1 += tl.load(src + off + BLOCK + lane)
        acc2 += tl.load(src + off + 2 * BLOCK + lane)
        acc3 += tl.load(src + off + 3 * BLOCK + lane)
    t1 = _sys_cnt()
    s = tl.sum(acc0, axis=0) + tl.sum(acc1, axis=0) \\
        + tl.sum(acc2, axis=0) + tl.sum(acc3, axis=0)
    tl.store(dst + pid, s)
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)

@triton.jit
def e16b_single(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * BLOCK
    t0 = _sys_cnt()
    v = tl.load(src + base + lane)
    t1 = _sys_cnt()
    tl.store(dst + pid, tl.sum(v, axis=0))
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

GRID = 56
LAUNCHES = 30


def measure(mod, name, pool, K, BLOCK, streams, rotate):
    per_launch = GRID * K * BLOCK * streams          # elements
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")
    dst = torch.zeros((GRID,), dtype=torch.float32, device="npu")
    kn = getattr(mod, name)

    def fn(i):
        off = (i % rotate) * per_launch if rotate else 0
        src = pool[off:off + per_launch]
        kn[(GRID,)](src, dst, tick, iters=K, BLOCK=BLOCK, num_warps=4,
                    compile_mode="simd", auto_simt_scope_mode="off",
                    logical_program_count_hint=GRID,
                    physical_vector_core_count_hint=56)

    for i in range(5):
        fn(i)
    torch.npu.synchronize()
    all_ns = []
    for i in range(LAUNCHES):
        fn(i)
        torch.npu.synchronize()
        ticks = tick.cpu().reshape(GRID, 2)
        ns = (ticks[:, 1] - ticks[:, 0]) * 1.00008
        all_ns.extend(ns.tolist())
    med = statistics.median(all_ns)
    bytes_per_prog = K * BLOCK * 4 * streams
    return med, med / K, bytes_per_prog / (med / K)


def main():
    gen_path = Path("/root/work/explore_contig/_gen_e16b.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e16b", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    cases = [
        ("e16b_base", 32, 1024, 1, 0),
        ("e16b_indep4", 32, 1024, 4, 0),
        ("e16b_base", 8, 4096, 1, 0),
        ("e16b_indep4", 8, 4096, 4, 0),
        ("e16b_base", 32, 1024, 1, 16),
        ("e16b_indep4", 32, 1024, 4, 16),
        ("e16b_single", 1, 16384, 1, 16),
    ]
    print("%-16s %3s %5s %5s %10s %9s %11s" %
          ("variant", "K", "BLOCK", "strm", "loop_ns", "ns/iter", "B/ns/prog"))
    out = {}
    for name, K, BLOCK, streams, rotate in cases:
        per_launch = GRID * K * BLOCK * streams
        rot = rotate or 1
        pool = torch.randn((per_launch * rot,), dtype=torch.float32,
                           device="npu")
        med, per_iter, bns = measure(mod, name, pool, K, BLOCK, streams,
                                     rotate)
        tag = (name.replace("e16b_", "") +
               ("_cold" if rotate else "") +
               (f"_s{streams}" if streams > 1 else ""))
        print("%-16s %3d %5d %5d %10.1f %9.1f %11.1f" %
              (tag, K, BLOCK, streams, med, per_iter, bns))
        out[tag] = {"K": K, "BLOCK": BLOCK, "streams": streams,
                    "loop_ns": med, "ns_per_iter": per_iter,
                    "B_ns_per_prog": bns}
        sys.stdout.flush()
    print("\ntier-3 constant: 167.56 B/cyc | tier-2 aggregate: 83.26 B/cyc")
    Path("/root/work/explore_contig/e16b_results.json").write_text(
        json.dumps(out, indent=1))


main()
