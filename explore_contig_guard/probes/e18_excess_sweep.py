#!/usr/bin/env python3
"""E18 v2: simd tiny-store latency-chain excess (unified-set phase 4).

Fixes vs v1: (1) masked tiny stores use block pointers (tiny + off +
lane, mask=lane==0) - the scalar-pointer form failed to compile;
(2) a completion store sits INSIDE the window before t1 (4B masked,
depends on the accumulator chain) so small-K windows include the
pipeline drain instead of deferring it past t1.

Body: one 16KB load chain + S tiny dependent stores (4B each), K swept
{2,4,8,16,32,64}, runtime iters, SYS_CNT whole-window, grid=56, hot.

excess(K, S) = whole(K, S) - whole(K, S=0): the tiny-store
serialization penalty isolated from the load stream.
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


def gen_body(s):
    sts = "\n".join(
        "        tl.store(tiny + p_t + %d + lane, acc, mask=lane == 0)"
        % j for j in range(s))
    st_block = ("\n" + sts) if s else ""
    return """
@triton.jit
def e18_s%d(src, tiny, done, tick_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = 32 * BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(iters):
        p = pid * seg + step * BLOCK + lane
        p_t = pid * iters * %d + step * %d
        acc += tl.load(src + p)
%s
    tl.store(done + pid * BLOCK + lane, acc, mask=lane == 0)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (s, max(s, 1), max(s, 1), st_block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e18_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    parts = [HEADER, COMMON]
    for s in (0, 1, 2, 4):
        parts.append(gen_body(s))
    gen_path = Path("/root/work/explore_contig/_gen_e18.py")
    gen_path.write_text("".join(parts), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e18", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("module built", flush=True)

    blk = 4096  # 16KB fp32 per iteration
    seg_max = 56 * 64 * blk
    pool = torch.randn((seg_max,), dtype=torch.float32, device="npu")
    done = torch.zeros((56 * blk,), dtype=torch.float32, device="npu")
    tiny = torch.zeros((56 * 64 * 4 + blk,), dtype=torch.float32,
                       device="npu")

    KS = [2, 4, 8, 16, 32, 64]
    out = {"meta": {"grid": 56, "launches": 30,
                    "body": "16KB load chain + S x 4B dependent masked "
                            "stores, hot, SYS_CNT whole-window incl. "
                            "in-window completion store"},
           "cells": {}}
    out_path = Path(args.out)

    for s in (0, 1, 2, 4):
        kn = getattr(mod, "e18_s%d" % s)
        tick = torch.zeros((112,), dtype=torch.int64, device="npu")

        def launch(iters):
            kn[(56,)](pool, tiny, done, tick, iters=iters, BLOCK=blk,
                      num_warps=4, compile_mode="simd",
                      auto_simt_scope_mode="off")

        launch(4)
        torch.npu.synchronize()
        row = {}
        for iters in KS:
            all_ns = []
            for i in range(30):
                launch(iters)
                torch.npu.synchronize()
                t = tick.cpu().reshape(56, 2)
                ns = (t[:, 1] - t[:, 0]) * 1.00008
                all_ns.append(statistics.median(ns.tolist()))
            row[str(iters)] = round(statistics.median(all_ns), 1)
        out["cells"]["s%d" % s] = row
        print("s%d: %s" % (s, row), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
