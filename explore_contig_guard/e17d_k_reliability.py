#!/usr/bin/env python3
"""E17d: K-axis measurement reliability for the mixed body (4,4).
Factors: method (syscnt window / profiler whole-kernel), card (2/3),
dst rotation (fixed vs rotate), K in {1,2,4,8,16,32,64}.
Per point: median of 30 launches x 56 programs.
"""
import argparse
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
def e17d_mix(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base_l = pid * iters * BLOCK
    base_s = pid * iters * BLOCK
    t0 = _sys_cnt()
    for step in range(iters):
        v = tl.load(src + base_l + step * BLOCK + lane)
        tl.store(dst + base_s + step * BLOCK + lane, v + 1.0)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

GRID = 56
KS = [1, 2, 4, 8, 16, 32, 64]


def run_case(mod, method, pool_s, pool_d, K, dstmode, LAUNCHES=30):
    BLOCK = 1024
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")
    span = GRID * K * BLOCK

    def fn(ridx):
        off = (ridx % 8) * span
        s = pool_s[off:off + span]
        d = pool_d[off:off + span]
        mod.e17d_mix[(GRID,)](s, d, tick, iters=K, BLOCK=BLOCK,
                              num_warps=4, compile_mode="simd",
                              auto_simt_scope_mode="off",
                              logical_program_count_hint=GRID,
                              physical_vector_core_count_hint=56)

    for _ in range(5):
        fn(0)
    torch.npu.synchronize()
    all_ns = []
    for i in range(LAUNCHES):
        fn(0 if dstmode == "fixed" else i)
        torch.npu.synchronize()
        if method == "syscnt":
            ticks = tick.cpu().reshape(GRID, 2)
            all_ns.extend(((ticks[:, 1] - ticks[:, 0]) * 1.00008).tolist())
        else:
            pass
    if method == "syscnt":
        return statistics.median(all_ns) / K, statistics.median(all_ns)
    # profiler: whole-kernel Duration
    import torch_npu.profiler as prof
    import shutil
    import tempfile
    import csv
    trace = Path(tempfile.mkdtemp(prefix="e17d_"))
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(
                          str(trace))):
        torch.npu.synchronize()
        for i in range(LAUNCHES):
            fn(0 if dstmode == "fixed" else i)
    torch.npu.synchronize()
    durs = []
    for tpath in sorted(trace.rglob("kernel_details.csv")):
        with tpath.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "e17d" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]) * 1000)
    shutil.rmtree(trace, ignore_errors=True)
    whole = statistics.median(durs)
    return whole / K, whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--methods", default="syscnt,profiler")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    gen_path = Path("/root/work/explore_contig/_gen_e17d.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e17d", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    maxK = max(KS)
    pool_s = torch.randn((8 * GRID * maxK * 1024,), dtype=torch.float32,
                         device="npu")
    pool_d = torch.zeros((8 * GRID * maxK * 1024,), dtype=torch.float32,
                         device="npu")

    out = {}
    for method in args.methods.split(","):
        for dstmode in ("fixed", "rotate"):
            row = {}
            for K in KS:
                per, whole = run_case(mod, method, pool_s, pool_d, K,
                                      dstmode)
                row[str(K)] = {"per_iter": round(per, 1),
                               "whole": round(whole, 1)}
            out[f"{method}/{dstmode}"] = row
            print("card%d %-9s %-6s per-iter: %s" %
                  (args.card, method, dstmode,
                   {k: v["per_iter"] for k, v in row.items()}))
            sys.stdout.flush()
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
