#!/usr/bin/env python3
"""E25c: clean confirmation for the grouping-removal question.
SYS_CNT windows around the load bodies (acc sink stored AFTER t1):
  w16     one 16KB load (shape (4,1024) f32)
  w32     one 32KB load (shape (8,1024) f32)
  w16x2   two 16KB loads together  -> current model: group 396.2 ns
  w1632   16KB + 32KB together     -> current: 503.9 (2x analytical)
                                     bucket-merge: 396.2
Delta (w1632 - w16x2): current predicts +107.7, bucket-merge predicts 0.
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

    def one(name, rows):
        return """
@triton.jit
def e25c_%s(src1, src2, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    row = tl.arange(0, %d)
    off = row[:, None] * BLOCK + lane[None, :]
    s = 0.0
    t0 = _sys_cnt()
    v = tl.load(src1 + off)
    s += tl.sum(v)
    t1 = _sys_cnt()
    tl.store(out + lane, s * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (name, rows)

    def two(name, rows):
        return """
@triton.jit
def e25c_%s(src1, src2, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    row = tl.arange(0, %d)
    off = row[:, None] * BLOCK + lane[None, :]
    s0 = 0.0
    s1 = 0.0
    t0 = _sys_cnt()
    s0 += tl.sum(tl.load(src1 + off))
    s1 += tl.sum(tl.load(src2 + off))
    t1 = _sys_cnt()
    tl.store(out + lane, (s0 + s1) * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (name, rows)

    parts.append(one("w16", 4))
    parts.append(one("w32", 8))
    parts.append(two("w16x2", 4))
    # w1632: (4,1024) from src1 + (8,1024) from src2
    parts.append("""
@triton.jit
def e25c_w1632(src1, src2, out, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    r4 = tl.arange(0, 4)
    r8 = tl.arange(0, 8)
    off4 = r4[:, None] * BLOCK + lane[None, :]
    off8 = r8[:, None] * BLOCK + lane[None, :]
    s0 = 0.0
    s1 = 0.0
    t0 = _sys_cnt()
    s0 += tl.sum(tl.load(src1 + off4))
    s1 += tl.sum(tl.load(src2 + off8))
    t1 = _sys_cnt()
    tl.store(out + lane, (s0 + s1) * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e25c", path)
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
    ap.add_argument("--out", default="/tmp/e25c.jsonl")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    gen_path = str(Path(args.out).parent / "_gen_e25c.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    src1 = torch.randn(8192, device="npu")
    src2 = torch.randn(8192, device="npu")
    out = torch.zeros(BLOCK, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    variants = {
        "w16": lambda: mod.e25c_w16[(1,)](src1, src2, out, tick, BLOCK,
                                          **opts),
        "w32": lambda: mod.e25c_w32[(1,)](src1, src2, out, tick, BLOCK,
                                          **opts),
        "w16x2": lambda: mod.e25c_w16x2[(1,)](src1, src2, out, tick, BLOCK,
                                              **opts),
        "w1632": lambda: mod.e25c_w1632[(1,)](src1, src2, out, tick, BLOCK,
                                              **opts),
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
    print("E25C DONE", flush=True)


if __name__ == "__main__":
    main()
