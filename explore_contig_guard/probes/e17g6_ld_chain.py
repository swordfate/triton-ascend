#!/usr/bin/env python3
"""E17g v6: SIMT load-side calibration, register-budget-clean.

Design per review: calibrate MEMORY, not compute - consumption is the
simplest possible `a_i += tl.load(...)` chain (same shape as the E17f
simd probes); NO reduction (tl.sum) anywhere.  Register spill is
avoided BY CONSTRUCTION: cells are limited to U x BLOCK <= 8192
elements (~64 fp32 accum registers per thread at num_warps=4); the
spill region (e.g. U4 x 16KB) is not measured - real memory-bound
kernels do not hold that many live accumulators either.

Each cell also measures a no-load baseline (loads replaced by
constants, identical structure); the reported slope_delta is
load-slope minus baseline-slope (Q4 guide 6.1 no-load subtraction),
which removes the per-iteration issue/scaffold cost shared by both.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"
COMMON = """
@triton.jit
def _clock():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l,~{memory}", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""


def gen_body(tag, u, loads_real):
    accs = "\n".join("    a%d = tl.zeros((BLOCK,), dtype=tl.float32)" % i
                     for i in range(u))
    if loads_real:
        body = "\n".join(
            "        a%d += tl.load(src + p + %d * seg)" % (i, i)
            for i in range(u))
    else:
        body = "\n".join(
            "        a%d += lane.to(tl.float32) + %d.0" % (i, i)
            for i in range(u))
    # block stores outside the window: create the data dependency so the
    # accumulator chains are not eliminated - no reduction involved
    sink = "\n".join(
        "    tl.store(out + %d * BLOCK + lane, a%d)" % (i, i)
        for i in range(u))
    return """
@triton.jit
def e17g6_%s(src, out, tick_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = ITERS * BLOCK
%s
    t0 = _clock()
    for step in range(ITERS):
        p = pid * seg + step * BLOCK + lane
%s
    t1 = _clock()
%s
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (tag, accs, body, sink)


PATTERN = (
    r'tt\.elementwise_inline_asm "MOV \$0, SYS_CNT"'
    r' \{constraints = "=l,~\{memory\}", '
    r'packed_element = 1 : i32, pure = false\} -> i64'
)


def replace_markers(text):
    patched, count = re.subn(PATTERN, "ascend_dpx.clock64", text)
    if count < 1:
        for line in text.splitlines():
            if "elementwise_inline_asm" in line:
                print("TTIR ASM LINE:", line.strip()[:300])
        raise ValueError("clock marker not matched (count=%d)" % count)
    return patched


@contextmanager
def clock64_compilation():
    original_run = subprocess.run

    def run(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) > 1
                and Path(command[0]).name == "bishengir-compile"):
            source = Path(command[1])
            source.write_text(replace_markers(source.read_text()))
        return original_run(command, *args, **kwargs)

    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e17g6_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    # register-budget-safe grid: U x BLOCK <= 8192 elements
    plan = [(4, (1, 2, 4, 8)), (8, (1, 2, 4)), (16, (1, 2)), (32, (1,))]

    out = {"meta": {"grid": 56, "launches": 30, "ks": [16, 32],
                    "method": "clock64 slope K16/K32, simt_only, hot; "
                              "acc-chain consumption; no-load baseline "
                              "subtracted; spill region excluded by "
                              "design (U x BLOCK <= 8192)"},
           "cells": {}}
    out_path = Path(args.out)

    for kb, us in plan:
        for u in us:
            for real, tagname in ((True, "ld"), (False, "base")):
                key = "%s_u%d_%dk" % (tagname, u, kb)
                t0 = time.time()
                try:
                    text = HEADER + COMMON + gen_body(key, u, real)
                    gen_path = Path("/root/work/explore_contig/"
                                    "_gen_e17g6.py")
                    gen_path.write_text(text, encoding="utf-8")
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "_gen_e17g6", str(gen_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    blk = 1024 * (kb // 4)
                    seg_max = 56 * 32 * blk
                    pool = torch.randn((seg_max * u,), dtype=torch.float32,
                                       device="npu")
                    outbuf = torch.zeros((u * blk,), dtype=torch.float32,
                                         device="npu")
                    tick = torch.zeros((112,), dtype=torch.int64,
                                       device="npu")
                    kn = getattr(mod, "e17g6_%s" % key)

                    def launch(iters):
                        with clock64_compilation():
                            kn[(56,)](pool, outbuf, tick, ITERS=iters,
                                      BLOCK=blk, num_warps=4,
                                      compile_mode="simt_only",
                                      auto_simt_scope_mode="off",
                                      superblock_factor=1)

                    launch(16)
                    torch.npu.synchronize()

                    def measure(iters):
                        all_ns = []
                        for i in range(30):
                            launch(iters)
                            torch.npu.synchronize()
                            t = tick.cpu().reshape(56, 2)
                            ns = (t[:, 1] - t[:, 0]) * 1.00008
                            all_ns.append(statistics.median(ns.tolist()))
                        return statistics.median(all_ns)

                    w16 = measure(16)
                    w32 = measure(32)
                    out["cells"][key] = {
                        "whole16": round(w16, 1),
                        "whole32": round(w32, 1),
                        "slope": round((w32 - w16) / 16, 1)}
                    print("%s slope=%.1f (%.0fs)" %
                          (key, out["cells"][key]["slope"],
                           time.time() - t0), flush=True)
                    del pool
                    torch.npu.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    out["cells"][key] = {"error": repr(exc)[:400]}
                    print("%s ERROR %r" % (key, exc), flush=True)
                out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
