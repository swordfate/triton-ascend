#!/usr/bin/env python3
"""E24c: discriminate why burst1's window includes first-load latency
(~300ns) while loop1's does not (66ns).
  burst_const  straight-line 1 pair, compile-time constant address
               (the e24b v_burst baseline, repeated)
  burst_pid    straight-line 1 pair, address = src + pid*0*BLOCK*0 + ...
               with pid = tl.program_id(0) (runtime value)
  burst_off    straight-line 1 pair, address offset via runtime arg
               (do_not_specialize)
  loop1        the scf.for baseline
Also dumps the ISA (ttadapter/source) around the SYS_CNT reads for
burst_const vs loop1 so the load/counter scheduling order is visible.
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
def c_burst_const(src, dst, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    x0 = tl.load(src + 0 * BLOCK + lane)
    tl.store(dst + 0 * BLOCK + lane, x0 + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit
def c_burst_pid(src, dst, tick_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    t0 = _sys_cnt()
    x0 = tl.load(src + pid * 0 * BLOCK + lane)
    tl.store(dst + pid * 0 * BLOCK + lane, x0 + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit(do_not_specialize=['off'])
def c_burst_off(src, dst, tick_ptr, off, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    x0 = tl.load(src + off * BLOCK + lane)
    tl.store(dst + off * BLOCK + lane, x0 + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    parts.append("""
@triton.jit(do_not_specialize=['iters'])
def c_loop(src, dst, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    for step in range(iters):
        x = tl.load(src + step * BLOCK + lane)
        tl.store(dst + step * BLOCK + lane, x + 1)
    t1 = _sys_cnt()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e24c", path)
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


def dump_isa(mod, tag, handle):
    h = handle
    for key in ("ttir", "ttadapter", "source"):
        text = h.asm.get(key)
        if not isinstance(text, str) or len(text) < 30:
            continue
        lines = text.splitlines()
        hits = [i for i, l in enumerate(lines)
                if ("load" in l or "store" in l or "SYS_CNT" in l
                    or "inline" in l or "scf.for" in l or "for" in l)]
        print("ISA %s %s: %d hit lines of %d" % (tag, key, len(hits),
                                                 len(lines)), flush=True)
        shown = set()
        for i in hits[:24]:
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                if j not in shown:
                    print("   %4d %s" % (j, lines[j][:110]), flush=True)
                    shown.add(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e24c.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--isa", action="store_true")
    args = ap.parse_args()

    gen_path = str(Path(args.out).parent / "_gen_e24c.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    src = torch.randn(BLOCK * 32, device="npu")
    dst = torch.zeros(BLOCK * 32, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    if args.isa:
        dump_isa(mod, "burst_const", mod.c_burst_const[(1,)](
            src, dst, tick, BLOCK, **opts))
        dump_isa(mod, "loop1", mod.c_loop[(1,)](
            src, dst, tick, 1, BLOCK, **opts))

    variants = {
        "burst_const": lambda: mod.c_burst_const[(1,)](
            src, dst, tick, BLOCK, **opts),
        "burst_pid": lambda: mod.c_burst_pid[(1,)](
            src, dst, tick, BLOCK, **opts),
        "burst_off": lambda: mod.c_burst_off[(1,)](
            src, dst, tick, 0, BLOCK, **opts),
        "loop1": lambda: mod.c_loop[(1,)](
            src, dst, tick, 1, BLOCK, **opts),
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
    print("E24C DONE", flush=True)


if __name__ == "__main__":
    main()
