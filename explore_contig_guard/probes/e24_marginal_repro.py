#!/usr/bin/env python3
"""E24: reproduce the E6 marginal conclusion with the device-truth caliber.

Claim under test (E6, quoted in CONTIGUOUS_REVIEW_20261118 2.1):
  straight-line (unrolled interleaved pairs) per-instance marginal ~37 ns
  at U<=8 (MTE2 queue depth 16, several accesses in flight);
  runtime-loop per-round marginal ~102 ns (one access in flight per
  round, issue-throttled).

Method (SYS_CNT in-kernel timestamps, the E17f caliber - immune to
launch/profiler overhead that inflated the E23 event readings):
  t0 = _sys_cnt() ... body ... t1 = _sys_cnt(); duration = (t1-t0)*1.00008 ns
  - loop_ind(K): K rounds of (4KB load -> dependent store), K in {1,2,4,8,16,32}
  - burst(U):    U interleaved pairs fully unrolled, U in {1,2,4,8,16}
  Both at grid=(1,), BLOCK=1024 fp32, fixed warm buffers, simd, warps=4.
  Marginals = adjacent-U/K differences of medians + least-squares slope
  over U in {1,2,4,8} / K in {4,8,16,32}; 30 launches x 3 passes.

Usage: python3 e24_marginal_repro.py --out /tmp/e24.jsonl
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK = 1024
KS = (1, 2, 4, 8, 16, 32)
US = (1, 2, 4, 8, 16, 32)
NS_TO_TICK = 1.00008  # SYS_CNT at 1 GHz
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
    # loop_ind: runtime scf.for, one load->store pair per round (E6 body)
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e24_loop_ind(src, dst, out, tick_ptr, iters, "
        "BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    t0 = _sys_cnt()\n"
        "    for step in range(iters):\n"
        "        x = tl.load(src + step * BLOCK + lane)\n"
        "        tl.store(dst + step * BLOCK + lane, x + 1)\n"
        "    t1 = _sys_cnt()\n"
        "    tl.store(tick_ptr + 0, t0)\n"
        "    tl.store(tick_ptr + 1, t1)\n\n")
    for u in US:
        # burst_u: fully unrolled interleaved pairs (E6 burst body)
        pairs = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(u))
        parts.append(
            "@triton.jit\ndef e24_burst_u%d(src, dst, out, tick_ptr, "
            "BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n"
            "    t0 = _sys_cnt()\n"
            "%s\n"
            "    t1 = _sys_cnt()\n"
            "    tl.store(tick_ptr + 0, t0)\n"
            "    tl.store(tick_ptr + 1, t1)\n\n" % (u, pairs))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e24", path)
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
        t0 = int(tick[0].item())
        t1 = int(tick[1].item())
        xs.append((t1 - t0) * NS_TO_TICK)
    return statistics.median(xs)


def slope(points):
    """Least-squares slope over [(x, y_ns)]."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e24.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--modes", default=["simd"], nargs="+")
    args = ap.parse_args()

    outp = Path(args.out)
    gen_path = str(outp.parent / "_gen_e24.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    nmax = max(max(KS), max(US))
    src = torch.randn(BLOCK * nmax, device="npu")
    dst = torch.zeros(BLOCK * nmax, device="npu")
    out = torch.zeros(1, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")

    records = []
    for mode in args.modes:
        opts = {"compile_mode": mode, "num_warps": 4}
        if mode == "simt_only":
            opts["superblock_factor"] = 1
        for pass_i in range(args.passes):
            row = {"pass": pass_i, "mode": mode}
            for k in KS:
                fn = lambda k=k: mod.e24_loop_ind[(1,)](
                    src, dst, out, tick, k, BLOCK, **opts)
                row["loop_K%d" % k] = round(med_ns(fn, tick,
                                                   reps=args.reps), 1)
            for u in US:
                fn = lambda u=u: getattr(mod, "e24_burst_u%d" % u)[(1,)](
                    src, dst, out, tick, BLOCK, **opts)
                row["burst_U%d" % u] = round(med_ns(fn, tick,
                                                    reps=args.reps), 1)
            records.append(row)
            print("PASS", pass_i, json.dumps(row), flush=True)

    # analysis on pass medians (per K/U across passes)
    summary = {"loop_med": {}, "burst_med": {}}
    for k in KS:
        vals = [r["loop_K%d" % k] for r in records]
        summary["loop_med"][k] = round(statistics.median(vals), 1)
    for u in US:
        vals = [r["burst_U%d" % u] for r in records]
        summary["burst_med"][u] = round(statistics.median(vals), 1)
    loop_pts = [(k, summary["loop_med"][k]) for k in (4, 8, 16, 32)]
    burst_pts = [(u, summary["burst_med"][u]) for u in (1, 2, 4, 8)]
    summary["loop_slope_ns_per_round"] = round(slope(loop_pts), 1)
    summary["burst_slope_ns_per_instance"] = round(slope(burst_pts), 1)
    summary["loop_adjacent"] = {
        str(k): round(summary["loop_med"][k] - summary["loop_med"][k // 2], 1)
        for k in (2, 4, 8, 16, 32)}
    summary["burst_adjacent"] = {
        str(u): round(summary["burst_med"][u] - summary["burst_med"][u // 2], 1)
        for u in (2, 4, 8, 16)}
    print("SUMMARY", json.dumps(summary), flush=True)

    with open(outp, "w", encoding="utf-8") as f:
        for rec in records + [{"exp": "e24", "kind": "summary",
                               "engine": "simd", **summary}]:
            f.write(json.dumps(rec) + "\n")
    print("E24 DONE ->", outp, flush=True)


if __name__ == "__main__":
    main()
