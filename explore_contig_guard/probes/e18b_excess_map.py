#!/usr/bin/env python3
"""E18b: tiny-store excess geometry map (unified-set phase 4).

E18 v2 found two regimes at 16KB load: S<=2 saturates (one-time ~200/
450ns), S=4 is sustained (~194ns/iter, masked-store instruction slots).
This sweep maps excess(K, S, loadBytes) over S in {0..4} and loadBytes
in {2KB, 4KB, 16KB} to fit E_sat(loadBytes, S) and the sustained-term
threshold.  Body: one load chain + S tiny dependent stores (4B masked,
block pointers), SYS_CNT whole-window with in-window completion store.
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
def e18b_s%d(src, tiny, done, tick_ptr, iters, BLOCK: tl.constexpr):
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
                                     "e18b_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    parts = [HEADER, COMMON]
    for s in (0, 1, 2, 3, 4):
        parts.append(gen_body(s))
    gen_path = Path("/root/work/explore_contig/_gen_e18b.py")
    gen_path.write_text("".join(parts), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e18b",
                                                  str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("module built", flush=True)

    BYTES = {"2k": 512, "4k": 1024, "16k": 4096}
    KS = [2, 4, 8, 16, 32, 64]
    out = {"meta": {"grid": 56, "launches": 30,
                    "body": "loadBytes chain + S x 4B dependent masked "
                            "stores, hot, SYS_CNT whole-window incl. "
                            "completion store"},
           "cells": {}}
    out_path = Path(args.out)

    for bname, blk in BYTES.items():
        seg_max = 56 * 64 * blk
        pool = torch.randn((seg_max,), dtype=torch.float32, device="npu")
        done = torch.zeros((56 * blk,), dtype=torch.float32, device="npu")
        tiny = torch.zeros((56 * 64 * 4 + blk,), dtype=torch.float32,
                           device="npu")
        for s in (0, 1, 2, 3, 4):
            kn = getattr(mod, "e18b_s%d" % s)
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
            key = "%s_s%d" % (bname, s)
            out["cells"][key] = row
            print("%s: %s (%.0fs)" % (key, row, time.time()), flush=True)
            out_path.write_text(json.dumps(out, indent=1))
        del pool
        torch.npu.empty_cache()
    print("saved", args.out)


main()
