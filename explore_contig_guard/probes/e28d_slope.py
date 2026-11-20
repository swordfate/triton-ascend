#!/usr/bin/env python3
"""E28d: per-iter marginal by whole-kernel differencing (K-slope).

simt, fp32, grid=1.  For each body: profiler kernel Duration at K=8 and
K=64; per-iter = (D64 - D8) / 56.  Any additive inflation (launch,
profiler) cancels in the difference; kernel completion includes the
store drain, so the slope is the TRUE per-iteration cost.

Points: 4 mixed variants (the byte-invariant-in-SYS_CNT quartet) +
loads-only / store-only anchors + a 16KB-pair + 64KB-total point.
"""
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/work/triton-ascend/explore_contig_guard/probes")
from e28c_prof_check import gen_module, load_module, prof_med_us  # noqa


def main():
    gen_path = "/tmp/_gen_e28d.py"
    # generate the full point set incl. loads-only / store-only anchors
    pts_all = [(1, 1, 1024), (2, 1, 1024), (1, 2, 1024), (2, 2, 1024),
               (1, 1, 4096), (2, 2, 4096),
               (1, 0, 1024), (2, 0, 1024), (2, 0, 4096),
               (0, 1, 1024), (0, 2, 1024)]
    lines = ["import triton\nimport triton.language as tl\n\n"]
    for nl, ns, e in pts_all:
        loads = "\n".join(
            "        a%d += tl.load(src%d + p)" % (j, j + 1)
            for j in range(nl))
        if nl:
            stores = "\n".join(
                "        tl.store(dst%d + p, a%d + %d)"
                % (j % 2 + 1, j % nl, j + 1) for j in range(ns))
        else:
            stores = "\n".join(
                "        tl.store(dst%d + p, tl.full((%d,), %d, "
                "dtype=tl.float32))" % (j % 2 + 1, e, j + 1)
                for j in range(ns))
        accs = "\n".join(
            "    a%d = tl.zeros((%d,), dtype=tl.float32)" % (j, e)
            for j in range(nl))
        sink = " + ".join("a%d" % j for j in range(nl)) or "0.0"
        lines.append("""
@triton.jit(do_not_specialize=['iters'])
def e28d_L%d_S%d_e%d(src1, src2, dst1, dst2, out, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %d)
%s
    for step in range(iters):
        p = step * %d + lane
%s
%s
    tl.store(out + lane, (%s) * 0 + 1.0)
""" % (nl, ns, e, e, accs, e, loads, stores, sink))
    Path(gen_path).write_text("\n".join(lines), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g28d", gen_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    src1 = torch.randn(16 * 1024, device="npu")
    src2 = torch.randn(16 * 1024, device="npu")
    dst1 = torch.zeros(16 * 1024, device="npu")
    dst2 = torch.zeros(16 * 1024, device="npu")
    out = torch.zeros(1024, device="npu")
    opts = {"compile_mode": "simt_only", "num_warps": 4,
            "superblock_factor": 1}

    pts = [("M1_1_1024", 1, 1, 1024), ("M2_1_1024", 2, 1, 1024),
           ("M1_2_1024", 1, 2, 1024), ("M2_2_1024", 2, 2, 1024),
           ("L1_1024", 1, 0, 1024), ("L2_1024", 2, 0, 1024),
           ("L2_4096", 2, 0, 4096),
           ("S1_1024", 0, 1, 1024), ("S2_1024", 0, 2, 1024)]
    rows = []
    for pass_i in range(3):
        row = {}
        for tag, nl, ns, e in pts:
            kname = "e28d_L%d_S%d_e%d" % (nl, ns, e)
            d8 = prof_med_us(
                lambda k=8, n=nl, s=ns, ee=e: getattr(
                    m, "e28d_L%d_S%d_e%d" % (n, s, ee))[(1,)](
                    src1, src2, dst1, dst2, out, k, ee, **opts),
                kname, pairs=12)
            d64 = prof_med_us(
                lambda n=nl, s=ns, ee=e: getattr(
                    m, "e28d_L%d_S%d_e%d" % (n, s, ee))[(1,)](
                    src1, src2, dst1, dst2, out, 64, ee, **opts),
                kname, pairs=12)
            row["%s_K8" % tag] = d8
            row["%s_K64" % tag] = d64
            row["%s_slope" % tag] = round((d64 - d8) / 56.0, 2) \
                if (d8 is not None and d64 is not None) else None
        rows.append(row)
        print("PASS", pass_i, json.dumps(row), flush=True)

    print("\n== per-iter slope (ns/iter, true marginal) ==")
    for tag, nl, ns, e in pts:
        key = "%s_slope" % tag
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            print("%-12s lB=%5d sB=%5d  slope=%8.2f ns/iter" %
                  (tag, nl * e * 4, ns * e * 4, statistics.median(vals)))
    Path("/tmp/e28d.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    print("E28D DONE")


main()
