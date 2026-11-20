#!/usr/bin/env python3
"""E28c: profiler whole-kernel cross-check of the flat SYS_CNT mixed
readings (simt, fp32, grid=1).  Plain bodies, no SYS_CNT — kernel
Duration from torch_npu.profiler decides whether the byte-invariant
plateau (~1269 ns/iter for 8/12/12/16KB) is real or a counter artifact."""
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

HEADER = "import triton\nimport triton.language as tl\n"


def gen_module(path):
    parts = [HEADER]
    for nl, ns, e in ((1, 1, 1024), (2, 1, 1024), (1, 2, 1024),
                      (2, 2, 1024), (1, 1, 4096), (2, 2, 4096)):
        loads = "\n".join(
            "        a%d += tl.load(src%d + p)" % (j, j + 1) for j in range(nl))
        stores = "\n".join(
            "        tl.store(dst%d + p, a%d + %d)" % (j % 2 + 1, j % nl, j + 1)
            for j in range(ns))
        accs = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=tl.float32)" % (j, e)
            for j in range(nl))
        parts.append("""
@triton.jit(do_not_specialize=['iters'])
def e28c_L%d_S%d_e%d(src1, src2, dst1, dst2, out, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %d)
%s
    t0 = 0
    for step in range(iters):
        p = step * %d + lane
%s
%s
    tl.store(out + lane, (%s) * 0 + 1.0)
""" % (nl, ns, e, e, accs, e, loads, stores,
       " + ".join("a%d" % j for j in range(nl)) or "0"))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e28c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=12):
    import torch_npu.profiler as prof
    import tempfile
    import shutil
    import csv
    trace = Path(tempfile.mkdtemp(prefix="e28c_"))
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
    out_root = Path("/tmp/e28c.jsonl")
    gen_path = "/tmp/_gen_e28c.py"
    gen_module(gen_path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g28c", gen_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    src1 = torch.randn(16 * 1024, device="npu")
    src2 = torch.randn(16 * 1024, device="npu")
    dst1 = torch.zeros(16 * 1024, device="npu")
    dst2 = torch.zeros(16 * 1024, device="npu")
    out = torch.zeros(1024, device="npu")
    opts = {"compile_mode": "simt_only", "num_warps": 4,
            "superblock_factor": 1}
    ITERS = 64

    pts = [(1, 1, 1024), (2, 1, 1024), (1, 2, 1024), (2, 2, 1024),
           (1, 1, 4096), (2, 2, 4096)]
    rows = []
    for pass_i in range(3):
        row = {}
        for nl, ns, e in pts:
            key = "L%d_S%d_e%d" % (nl, ns, e)
            kname = "e28c_L%d_S%d_e%d" % (nl, ns, e)
            fn = lambda k=kname, a=(nl, ns, e): getattr(
                m, "e28c_L%d_S%d_e%d" % (a[0], a[1], a[2]))[(1,)](
                src1, src2, dst1, dst2, out, ITERS, a[2], **opts)
            v = prof_med_us(fn, kname, pairs=12)
            row[key] = v
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)
    out_root.write_text("\n".join(json.dumps(r) for r in rows))
    print("E28C DONE")


main()
