#!/usr/bin/env python3
"""E17k: store-only probe reconciliation (4KB, grid=56).

Three store-only variants plus the mix reference:
  e17k_st_plain : store-only, no sink (E17f/E17i shape)
  e17k_st_sink  : store-only + in-window completion store (load-free)
  e17k_mix      : 1 load + 1 store per iteration (completion forced by
                  the store's dependency on the load chain)
All constexpr K=16/32 slope, hot.  Resolves the 106.8 (E17f) vs 199
(E17i) store-only contradiction and validates the mix anchor (107.4).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""

KERN = """
@triton.jit
def e17k_%s(dst, src, tick_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = ITERS * BLOCK
    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(ITERS):
        p = pid * seg + step * BLOCK + lane
%s
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

VARIANTS = {
    "st_plain": "        tl.store(dst + p, v)",
    "st_sink": ("        tl.store(dst + p, v)\n"
                "        acc += v * 1e-30"),
    "mix": ("        acc += tl.load(src + p)\n"
            "        tl.store(dst + p, acc + 1.0)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e17k_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    parts = [HEADER, COMMON]
    for tag, body in VARIANTS.items():
        parts.append(KERN % (tag, body))
    gen_path = Path("/root/work/explore_contig/_gen_e17k.py")
    gen_path.write_text("".join(parts), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e17k", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("module built", flush=True)

    blk = 1024
    seg_max = 56 * 32 * blk
    pool = torch.randn((seg_max,), dtype=torch.float32, device="npu")
    dstp = torch.zeros((seg_max,), dtype=torch.float32, device="npu")
    outbuf = torch.zeros((56,), dtype=torch.float32, device="npu")

    out = {"meta": {"ks": [16, 32], "launches": 30, "body": "4KB"},
           "cells": {}}
    out_path = Path(args.out)

    for tag in VARIANTS:
        kn = getattr(mod, "e17k_%s" % tag)
        tick = torch.zeros((112,), dtype=torch.int64, device="npu")

        def launch(iters):
            kn[(56,)](dstp, pool, tick, ITERS=iters, BLOCK=blk,
                      num_warps=4, compile_mode="simd",
                      auto_simt_scope_mode="off")

        launch(16)
        torch.npu.synchronize()
        rows = {}
        for iters in (16, 32):
            all_ns = []
            for i in range(30):
                launch(iters)
                torch.npu.synchronize()
                t = tick.cpu().reshape(56, 2)
                ns = (t[:, 1] - t[:, 0]) * 1.00008
                all_ns.append(statistics.median(ns.tolist()))
            rows[iters] = round(statistics.median(all_ns), 1)
        slope = (rows[32] - rows[16]) / 16
        out["cells"][tag] = {"whole16": rows[16], "whole32": rows[32],
                             "slope": round(slope, 1)}
        print("%-10s slope=%.1f  whole16=%s whole32=%s" %
              (tag, slope, rows[16], rows[32]), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
