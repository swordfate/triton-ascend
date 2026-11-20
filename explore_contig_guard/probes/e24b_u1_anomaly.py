#!/usr/bin/env python3
"""E24b: locate the U=1 anomaly (burst 331ns vs loop 66ns at ONE pair).

Both windows hold the same instruction sequence (t0 -> 1 load -> 1 store
-> t1), so the 5x gap is unexplained.  Isolate by variants (SYS_CNT):
  loop1        loop_ind, iters=1                    (e24 baseline)
  burst1       burst_u1                             (e24 baseline)
  loop1_ld     window = load only (acc sink after t1), no store
  burst1_ld    window = load only
  loop1_st     window = store only (const value)
  burst1_st    window = store only
  burst1_pre   priming load BEFORE t0, then window = load (2nd load)
  loop1_pre    same for the loop body
If burst1_ld still ~5x loop1_ld, the gap is in load placement/scheduling;
if only the store variants differ, it is store-drain related; if the
preload closes the gap, it is a first-load-after-idle effect.
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
@triton.jit(do_not_specialize=['iters'])
def v_loop(src, dst, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    for step in range(iters):
        x = tl.load(src + step * BLOCK + lane)
        tl.store(dst + step * BLOCK + lane, x + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit
def v_burst(src, dst, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    x0 = tl.load(src + 0 * BLOCK + lane)
    tl.store(dst + 0 * BLOCK + lane, x0 + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    # load-only windows
    parts.append("""
@triton.jit(do_not_specialize=['iters'])
def v_loop_ld(src, dst, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(iters):
        acc += tl.load(src + step * BLOCK + lane)
    t1 = _sys_cnt()
    tl.store(out + lane, acc * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit
def v_burst_ld(src, dst, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    acc += tl.load(src + 0 * BLOCK + lane)
    t1 = _sys_cnt()
    tl.store(out + lane, acc * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    # store-only windows
    parts.append("""
@triton.jit(do_not_specialize=['iters'])
def v_loop_st(src, dst, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
    t0 = _sys_cnt()
    for step in range(iters):
        tl.store(dst + step * BLOCK + lane, v)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit
def v_burst_st(src, dst, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
    t0 = _sys_cnt()
    tl.store(dst + 0 * BLOCK + lane, v)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    # primed (second-load) windows
    parts.append("""
@triton.jit
def v_burst_pre(src, dst, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    pre = tl.load(src + 15 * BLOCK + lane)
    t0 = _sys_cnt()
    x0 = tl.load(src + 0 * BLOCK + lane)
    t1 = _sys_cnt()
    tl.store(dst + 0 * BLOCK + lane, x0 + tl.sum(pre) * 0 + 1)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit(do_not_specialize=['iters'])
def v_loop_pre(src, dst, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    pre = tl.load(src + 15 * BLOCK + lane)
    t0 = _sys_cnt()
    for step in range(iters):
        x = tl.load(src + step * BLOCK + lane)
        tl.store(dst + step * BLOCK + lane, x + tl.sum(pre) * 0 + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e24b", path)
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
        xs.append((int(tick[0 + 1].item()) - int(tick[0].item()))
                  * NS_TO_TICK)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e24b.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    gen_path = str(Path(args.out).parent / "_gen_e24b.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    src = torch.randn(BLOCK * 32, device="npu")
    dst = torch.zeros(BLOCK * 32, device="npu")
    out = torch.zeros(BLOCK, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")

    variants = {
        "loop1": lambda: mod.v_loop[(1,)](src, dst, out, tick, 1, BLOCK,
                                          **OPTS),
        "burst1": lambda: mod.v_burst[(1,)](src, dst, out, tick, BLOCK,
                                            **OPTS),
        "loop1_ld": lambda: mod.v_loop_ld[(1,)](src, dst, out, tick, 1,
                                                BLOCK, **OPTS),
        "burst1_ld": lambda: mod.v_burst_ld[(1,)](src, dst, out, tick,
                                                  BLOCK, **OPTS),
        "loop1_st": lambda: mod.v_loop_st[(1,)](src, dst, out, tick, 1,
                                                BLOCK, **OPTS),
        "burst1_st": lambda: mod.v_burst_st[(1,)](src, dst, out, tick,
                                                  BLOCK, **OPTS),
        "loop1_pre": lambda: mod.v_loop_pre[(1,)](src, dst, out, tick, 1,
                                                  BLOCK, **OPTS),
        "burst1_pre": lambda: mod.v_burst_pre[(1,)](src, dst, out, tick,
                                                    BLOCK, **OPTS),
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
    print("E24B DONE", flush=True)


OPTS = {"compile_mode": "simd", "num_warps": 4}

if __name__ == "__main__":
    main()
