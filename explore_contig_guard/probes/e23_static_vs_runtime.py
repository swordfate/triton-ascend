#!/usr/bin/env python3
"""E23: A (runtime-trip-count for) vs B (same-size constexpr for) performance,
and what it means for the contiguous calibration taxonomy.  Re-runs the
E7/E11 question on the final wheel.

Shapes (identical work: N x 4KiB fp32 load+store pairs, streaming, grid=1):
  A_rt   for step in range(iters), iters at runtime (do_not_specialize)
  B_ce   for step in range(CT), CT is a tl.constexpr arg   (E11: stays a loop)
  B_sr   for step in tl.static_range(N)                    (E11: fully unrolled)
  C_sl   textgen straight-line N pairs                     (burst reference)

Per (mode, N): device-side kernel medians (torch_npu.profiler, 12 pairs,
4 shapes interleaved in one session) x 3 passes; plus the E11 ISA scf.for
scan for A/B_ce/B_sr.

Usage: python3 e23_static_vs_runtime.py --out /tmp/e23.jsonl [--modes simd simt_only]
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
NS = (8, 32)
SHAPES = ("A_rt", "B_ce", "B_sr", "C_sl")
HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n"


def gen_module(path):
    parts = [HEADER]
    # A_rt: one kernel, runtime trip count (shared by all N)
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e23_rt(src, dst, iters, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for step in range(iters):\n"
        "        x = tl.load(src + step * BLOCK + lane)\n"
        "        tl.store(dst + step * BLOCK + lane, x + 1)\n\n")
    for n in NS:
        # B_ce: constexpr trip count via a constexpr arg (plain range)
        parts.append(
            "@triton.jit\ndef e23_ce_%d(src, dst, CT: tl.constexpr, "
            "BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n"
            "    for step in range(CT):\n"
            "        x = tl.load(src + step * BLOCK + lane)\n"
            "        tl.store(dst + step * BLOCK + lane, x + 1)\n\n" % n)
        # B_sr: static_range (frontend unrolls)
        parts.append(
            "@triton.jit\ndef e23_sr_%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n"
            "    for step in tl.static_range(%d):\n"
            "        x = tl.load(src + step * BLOCK + lane)\n"
            "        tl.store(dst + step * BLOCK + lane, x + 1)\n\n"
            % (n, n))
        # C_sl: straight-line textgen
        pairs = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(n))
        parts.append(
            "@triton.jit\ndef e23_sl_%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (n, pairs))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e23", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan(text):
    return {"scf.for": text.count("scf.for"),
            "cf.br": text.count("cf.br"),
            "for(": text.count("for(") + text.count("for (")}


def prof_medians(launch_fns, name_filters, pairs=12, extra_warmup=6):
    """Device-side kernel durations (us) under one profiler session.
    Returns {filter: median_us} (E7's prof_session protocol)."""
    import torch_npu.profiler as prof
    total = pairs * len(launch_fns)
    trace = Path(tempfile.mkdtemp(prefix="e23_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=total),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for fn in launch_fns:
            fn()
        torch.npu.synchronize()
        for _ in range(extra_warmup):
            for fn in launch_fns:
                fn()
        torch.npu.synchronize()
        for _ in range(pairs):
            for fn in launch_fns:
                fn()
                p.step()
    torch.npu.synchronize()
    rows = {f: [] for f in name_filters}
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for r in csv.DictReader(h):
                dur = r.get("Duration(us)")
                name = r.get("Name") or ""
                if not dur:
                    continue
                for f in name_filters:
                    if f in name:
                        rows[f].append(float(dur))
    shutil.rmtree(trace, ignore_errors=True)
    return {f: (round(statistics.median(v), 3) if v else None)
            for f, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e23.jsonl")
    ap.add_argument("--modes", default=["simd", "simt_only"], nargs="+")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()

    outp = Path(args.out)
    gen_path = str(outp.parent / "_gen_e23.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    n_elems = BLOCK * 1024
    src = torch.randn(n_elems, device="npu")
    dst = torch.zeros(n_elems, device="npu")

    results = []
    for mode in args.modes:
        opts = {"compile_mode": mode, "num_warps": 4}
        if mode == "simt_only":
            opts["superblock_factor"] = 1
        # ---- ISA scan (E11 style) ----
        handles = {
            "A_rt": mod.e23_rt[(1,)](src, dst, 8, BLOCK, **opts),
            "B_ce8": mod.e23_ce_8[(1,)](src, dst, CT=8, BLOCK=BLOCK, **opts),
            "B_sr8": mod.e23_sr_8[(1,)](src, dst, BLOCK, **opts),
        }
        for kname, h in handles.items():
            scans = {}
            for key in sorted(h.asm.keys()):
                text = h.asm.get(key)
                if isinstance(text, str) and len(text) > 30:
                    scans[key] = scan(text)
            rec = {"exp": "e23", "kind": "isa", "kernel": kname,
                   "mode": mode, "scans": scans}
            results.append(rec)
            print("ISA", kname, mode, json.dumps(scans), flush=True)

        # ---- timing (device medians, profiler protocol) ----
        for n in NS:
            fns = [
                lambda n=n: mod.e23_rt[(1,)](src, dst, n, BLOCK, **opts),
                lambda n=n: getattr(mod, "e23_ce_%d" % n)[(1,)](
                    src, dst, CT=n, BLOCK=BLOCK, **opts),
                lambda n=n: getattr(mod, "e23_sr_%d" % n)[(1,)](
                    src, dst, BLOCK, **opts),
                lambda n=n: getattr(mod, "e23_sl_%d" % n)[(1,)](
                    src, dst, BLOCK, **opts),
            ]
            filters = ["e23_rt", "e23_ce_%d" % n, "e23_sr_%d" % n,
                       "e23_sl_%d" % n]
            for pass_i in range(args.passes):
                meds = prof_medians(fns, filters, pairs=args.pairs)
                row = {"exp": "e23", "kind": "time", "mode": mode, "N": n,
                       "pass": pass_i}
                for shape, filt in zip(SHAPES, filters):
                    row[shape] = meds[filt]
                results.append(row)
                print("TIME", mode, "N%d" % n, "pass%d" % pass_i,
                      {k: row[k] for k in SHAPES}, flush=True)
                time.sleep(0.5)

    with open(outp, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print("E23 DONE ->", outp, flush=True)


if __name__ == "__main__":
    main()
