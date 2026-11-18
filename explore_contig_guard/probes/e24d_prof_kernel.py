#!/usr/bin/env python3
"""E24d: whole-kernel profiler timing (NO in-kernel SYS_CNT) of the two
E6 shapes at matched work - does the whole kernel differ between a
K-round loop and a U-pair unrolled body?

  e24d_loop   runtime scf.for, 1 load->store pair per round, K rounds
  e24d_burst  straight-line U interleaved pairs (textgen)
K/U in {1,2,4,8,16,32}; grid=(1,), BLOCK=1024 fp32, simd, warps=4.
Timing = torch_npu.profiler kernel Duration medians (12 pairs/session,
one session per point, 3 passes) - the E7/E23 caliber, launch overhead
included in each reading (same floor for both shapes).
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
NS = (1, 2, 4, 8, 16, 32)
HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n"


def gen_module(path):
    parts = [HEADER]
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e24d_loop(src, dst, iters, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for step in range(iters):\n"
        "        x = tl.load(src + step * BLOCK + lane)\n"
        "        tl.store(dst + step * BLOCK + lane, x + 1)\n\n")
    for n in NS:
        pairs = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(n))
        parts.append(
            "@triton.jit\ndef e24d_burst_%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (n, pairs))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e24d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=12):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e24d_"))
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
    ap.add_argument("--out", default="/tmp/e24d.jsonl")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()

    gen_path = str(Path(args.out).parent / "_gen_e24d.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    nmax = max(NS)
    src = torch.randn(BLOCK * nmax, device="npu")
    dst = torch.zeros(BLOCK * nmax, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}

    rows = []
    for pass_i in range(args.passes):
        row = {"pass": pass_i}
        for n in NS:
            row["loop_K%d" % n] = prof_med_us(
                lambda n=n: mod.e24d_loop[(1,)](src, dst, n, BLOCK, **opts),
                "e24d_loop", pairs=args.pairs)
            row["burst_U%d" % n] = prof_med_us(
                lambda n=n: getattr(mod, "e24d_burst_%d" % n)[(1,)](
                    src, dst, BLOCK, **opts),
                "e24d_burst_%d" % n, pairs=args.pairs)
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)
        time.sleep(0.5)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("E24D DONE", flush=True)


if __name__ == "__main__":
    main()
