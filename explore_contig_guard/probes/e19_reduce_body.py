#!/usr/bin/env python3
"""E19: reduction-epilogue body calibration (row-retirement cluster 2).

Replicates the reduce_sum_epilogue body exactly: per iteration =
2KB load (RBLOCK=512 fp32) + 32B bias load + fp32 reduction
(tl.sum) + 32B dependent store.  whole(K) swept K in {1,2,4,8,16,32};
compare against the envelope price (max(TL,TS) at clamped 4KB buckets
= 106.8 ns/iter + compute) and against the row tier's price (the
measured rows currently carry this family).
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

KERNEL = """
@triton.jit
def e19(src, bias, out, tick_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    for step in range(iters):
        base = pid * iters * BLOCK + step * BLOCK
        value = tl.load(src + base + r)
        b = tl.load(bias + pid)
        s = tl.sum(value, axis=0)
        tl.store(out + pid * iters + step, s + b)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e19_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    gen_path = Path("/root/work/explore_contig/_gen_e19.py")
    gen_path.write_text(HEADER + COMMON + KERNEL, encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e19", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    blk = 512  # 2KB fp32
    seg_max = 56 * 32 * blk
    pool = torch.randn((seg_max,), dtype=torch.float32, device="npu")
    bias = torch.randn((56,), dtype=torch.float32, device="npu")
    outb = torch.zeros((56 * 64,), dtype=torch.float32, device="npu")

    KS = [1, 2, 4, 8, 16, 32]
    out = {"meta": {"grid": 56, "launches": 30,
                    "body": "2KB load + 32B bias load + tl.sum + 32B "
                            "dependent store, hot, SYS_CNT whole"},
           "whole": {}}
    out_path = Path(args.out)
    tick = torch.zeros((112,), dtype=torch.int64, device="npu")

    def launch(iters):
        mod.e19[(56,)](pool, bias, outb, tick, iters=iters, BLOCK=blk,
                       num_warps=4, compile_mode="simd",
                       auto_simt_scope_mode="off")

    launch(2)
    torch.npu.synchronize()
    for iters in KS:
        all_ns = []
        for i in range(30):
            launch(iters)
            torch.npu.synchronize()
            t = tick.cpu().reshape(56, 2)
            ns = (t[:, 1] - t[:, 0]) * 1.00008
            all_ns.append(statistics.median(ns.tolist()))
        out["whole"][str(iters)] = round(statistics.median(all_ns), 1)
        print("K=%-3d whole=%8.1f ns  per-iter=%7.1f" %
              (iters, out["whole"][str(iters)],
               out["whole"][str(iters)] / iters), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
