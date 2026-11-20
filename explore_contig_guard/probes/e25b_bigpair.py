#!/usr/bin/env python3
"""E25b: the po2-space discriminator for exact-shape vs bucket grouping.

16KB load (shape (4,1024) f32) and 32KB load (shape (8,1024) f32) both
land in bucket 16384 (the 32KB clamps), but their element counts differ
-> today they form TWO n=1 groups (per-access analytical 203.1 + 300.8
= 503.9 ns); under bucket grouping they merge into ONE n=2 group at the
16384 row (264.1 + 132.1 = 396.2 ns).  Whole-kernel profiler timing
decides.

Variants (straight-line, one 2D load per tensor + sink stores, grid=1,
simd):
  same2_4k    2 x (1024,) f32 = 4KB each   (group anchor 196.7)
  same2_16k   2 x (4,1024)  f32 = 16KB     (group anchor 396.2)
  odd_16_32   (4,1024) + (8,1024)          (discriminator)
  same2_32k   2 x (8,1024)  f32 = 32KB     (clamped row, 396.2)
"""
import argparse
import csv
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK = 1024
HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n"


def gen_module(path):
    parts = [HEADER]

    def two_load_kernel(name, rows):
        return """
@triton.jit
def e25b_%s(src1, src2, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    row = tl.arange(0, %d)
    off = row[:, None] * BLOCK + lane[None, :]
    a0 = tl.load(src1 + off)
    a1 = tl.load(src2 + off)
    tl.store(out + lane, tl.sum(a0) * 0 + tl.sum(a1) * 0 + 1.0)
""" % (name, rows)

    parts.append(two_load_kernel("same2_16k", 4))
    parts.append(two_load_kernel("odd_16_32", 4))      # placeholder; real
    parts.append(two_load_kernel("same2_32k", 8))      # one replaced below
    # same2_4k: two 1D 4KB loads
    parts.append("""
@triton.jit
def e25b_same2_4k(src1, src2, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    a0 = tl.load(src1 + lane)
    a1 = tl.load(src2 + lane)
    tl.store(out + lane, a0 * 0 + a1 * 0 + 1.0)
""")
    # odd_16_32: (4,1024) 16KB from src1 + (8,1024) 32KB from src2
    parts.append("""
@triton.jit
def e25b_odd_16_32(src1, src2, out, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    r4 = tl.arange(0, 4)
    r8 = tl.arange(0, 8)
    off4 = r4[:, None] * BLOCK + lane[None, :]
    off8 = r8[:, None] * BLOCK + lane[None, :]
    a0 = tl.load(src1 + off4)
    a1 = tl.load(src2 + off8)
    tl.store(out + lane, tl.sum(a0) * 0 + tl.sum(a1) * 0 + 1.0)
""")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e25b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=12):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e25b_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=pairs),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        fn()
        torch.npu.synchronize()
        for _ in range(4):
            fn()
        torch.npu.synchronize()
        for _ in range(pairs):
            fn()
            p.step()
    torch.npu.synchronize()
    durs = []
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for r in csv.DictReader(h):
                dur = r.get("Duration(us)")
                name = r.get("Name") or ""
                if dur and name_filter in name:
                    durs.append(float(dur))
    shutil.rmtree(trace, ignore_errors=True)
    return round(statistics.median(durs), 3) if durs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e25b.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()

    gen_path = str(Path(args.out).parent / "_gen_e25b.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    src1 = torch.randn(8192, device="npu")
    src2 = torch.randn(8192, device="npu")
    out = torch.zeros(BLOCK, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    variants = {
        "same2_4k": lambda: mod.e25b_same2_4k[(1,)](
            src1, src2, out, BLOCK, **opts),
        "same2_16k": lambda: mod.e25b_same2_16k[(1,)](
            src1, src2, out, BLOCK, **opts),
        "odd_16_32": lambda: mod.e25b_odd_16_32[(1,)](
            src1, src2, out, BLOCK, **opts),
        "same2_32k": lambda: mod.e25b_same2_32k[(1,)](
            src1, src2, out, BLOCK, **opts),
    }
    rows = []
    for pass_i in range(args.passes):
        row = {"pass": pass_i}
        for name, fn in variants.items():
            row[name] = prof_med_us(fn, "e25b_%s" % name,
                                    pairs=args.pairs)
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)
        time.sleep(0.3)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("E25B DONE", flush=True)


if __name__ == "__main__":
    main()
