#!/usr/bin/env python3
"""E25: is exact-shape grouping needed, or can the lookup key (bucket)
carry the grouping?  Discriminator: a pair of loads that land in the
SAME byte bucket but have DIFFERENT element counts (1000 x f32 = 4000B
and 1008 x f32 = 4032B, both bucket 4096).

  today (exact-shape grouping): two groups, n=1 each -> per-access
          analytical 129.19 + 129.38 = 258.6 ns
  proposed (bucket grouping):   one group, n=2 -> first+marginal
          160.1 + 36.6 = 196.7 ns (same as an identical 2x4KB pair)

Measure (SYS_CNT loads-only windows + profiler whole-kernel sanity):
  same2_1024   2 identical 4KB loads        (anchor; group price 196.7)
  same2_1000   2 identical 1000-elem masked loads (group hits too)
  odd2_1000_1008  1000-elem + 1008-elem masked pair   (the discriminator)
If odd2 measures like same2 (not +60ns heavier), exact-shape grouping
over-prices mixed-bucket pairs and the merge is justified.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK = 1024
NS_TO_TICK = 1.00008
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""
HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n" + COMMON


def gen_module(path):
    parts = [HEADER]
    parts.append("""
@triton.jit
def e25_same2(src1, src2, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    a0 = tl.zeros((BLOCK,), dtype=tl.float32)
    a1 = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    a0 += tl.load(src1 + lane)
    a1 += tl.load(src2 + lane)
    t1 = _sys_cnt()
    tl.store(out + lane, a0 * 0 + a1 * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    # masked odd-size loads: tensor 1000 elems, arange 1024, mask lane < N
    parts.append("""
@triton.jit
def e25_same2_1000(src1, src2, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    a0 = tl.zeros((BLOCK,), dtype=tl.float32)
    a1 = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    a0 += tl.load(src1 + lane, mask=lane < 1000, other=0.0)
    a1 += tl.load(src2 + lane, mask=lane < 1000, other=0.0)
    t1 = _sys_cnt()
    tl.store(out + lane, a0 * 0 + a1 * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit
def e25_odd2(src1, src2, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    a0 = tl.zeros((BLOCK,), dtype=tl.float32)
    a1 = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    a0 += tl.load(src1 + lane, mask=lane < 1000, other=0.0)
    a1 += tl.load(src2 + lane, mask=lane < 1008, other=0.0)
    t1 = _sys_cnt()
    tl.store(out + lane, a0 * 0 + a1 * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e25", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def med_ns(fn, tick, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    xs = []
    for _ in range(reps):
        fn()
        torch.npu.synchronize()
        xs.append((int(tick[1].item()) - int(tick[0].item())) * NS_TO_TICK)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e25.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    gen_path = str(Path(args.out).parent / "_gen_e25.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    src1 = torch.randn(BLOCK, device="npu")
    src2 = torch.randn(BLOCK, device="npu")
    out = torch.zeros(BLOCK, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    variants = {
        "same2_1024": lambda: mod.e25_same2[(1,)](
            src1, src2, out, tick, BLOCK, **opts),
        "same2_1000": lambda: mod.e25_same2_1000[(1,)](
            src1, src2, out, tick, BLOCK, **opts),
        "odd2_1000_1008": lambda: mod.e25_odd2[(1,)](
            src1, src2, out, tick, BLOCK, **opts),
    }
    rows = []
    for pass_i in range(args.passes):
        row = {"pass": pass_i}
        for name, fn in variants.items():
            row[name] = round(med_ns(fn, tick, reps=args.reps), 1)
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("E25 DONE", flush=True)


if __name__ == "__main__":
    main()
