#!/usr/bin/env python3
"""E17j: grid-axis factor for the simd envelope (unified-set cluster 1).

The E17f TL/TS constants were calibrated at grid=56 (full wave).  The
2x4096x8 audit cluster (hint=2) shows the envelope overpricing by ~12%
at tiny grids - fewer programs share DRAM, so the per-program price
drops.  This probe measures the mix (1L+1S 4KB) and load-only (4KB)
steady-state slope at grid in {1,2,4,8,14,28,56}, K=16/32 constexpr,
SYS_CNT window (simd), to fit gridFactor(P) = slope(P)/slope(56).
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


def gen_kernel(tag, kind):
    if kind == "mix":
        body = """        acc += tl.load(src + p)
        tl.store(dst + p, acc + 1.0)"""
        args = "src, dst"
        call = "pool, dstp"
    else:
        body = "        acc += tl.load(src + p)"
        args = "src"
        call = "pool"
    return """
@triton.jit
def e17j_%s(%s, out, tick_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = ITERS * BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(ITERS):
        p = pid * seg + step * BLOCK + lane
%s
    t1 = _sys_cnt()
    tl.store(out + pid, tl.max(acc, axis=0) * 1e-30)
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (tag, args, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e17j_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    parts = [HEADER, COMMON]
    for kind in ("mix", "ld"):
        parts.append(gen_kernel(kind, kind))
    gen_path = Path("/root/work/explore_contig/_gen_e17j.py")
    gen_path.write_text("".join(parts), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e17j", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("module built", flush=True)

    blk = 1024  # 4KB fp32
    seg_max = 56 * 32 * blk
    pool = torch.randn((seg_max,), dtype=torch.float32, device="npu")
    dstp = torch.zeros((seg_max,), dtype=torch.float32, device="npu")
    outbuf = torch.zeros((56,), dtype=torch.float32, device="npu")

    GRIDS = [1, 2, 4, 8, 14, 28, 56]
    out = {"meta": {"ks": [16, 32], "launches": 30,
                    "body": "4KB mix (1L+1S) and 4KB load-only, "
                            "SYS_CNT slope, hot"},
           "mix": {}, "ld": {}}
    out_path = Path(args.out)

    for kind in ("mix", "ld"):
        kn = getattr(mod, "e17j_%s" % kind)
        for g in GRIDS:
            tick = torch.zeros((112,), dtype=torch.int64, device="npu")

            def launch(iters):
                if kind == "mix":
                    kn[(g,)](pool, dstp, outbuf, tick, ITERS=iters,
                             BLOCK=blk, num_warps=4, compile_mode="simd",
                             auto_simt_scope_mode="off")
                else:
                    kn[(g,)](pool, outbuf, tick, ITERS=iters, BLOCK=blk,
                             num_warps=4, compile_mode="simd",
                             auto_simt_scope_mode="off")

            launch(16)
            torch.npu.synchronize()

            def measure(iters):
                all_ns = []
                for i in range(30):
                    launch(iters)
                    torch.npu.synchronize()
                    t = tick.cpu()[:g * 2].reshape(g, 2)
                    ns = (t[:, 1] - t[:, 0]) * 1.00008
                    all_ns.append(statistics.median(ns.tolist()))
                return statistics.median(all_ns)

            w16 = measure(16)
            w32 = measure(32)
            slope = (w32 - w16) / 16
            out[kind][str(g)] = round(slope, 1)
            print("%s g=%-3d slope=%.1f (%.0fs)" %
                  (kind, g, slope, time.time()), flush=True)
            out_path.write_text(json.dumps(out, indent=1))
    base_m = out["mix"]["56"]
    base_l = out["ld"]["56"]
    out["mix_factor"] = {g: round(v / base_m, 3)
                         for g, v in out["mix"].items()}
    out["ld_factor"] = {g: round(v / base_l, 3)
                        for g, v in out["ld"].items()}
    out_path.write_text(json.dumps(out, indent=1))
    print("mix factors:", out["mix_factor"])
    print("ld factors :", out["ld_factor"])
    print("saved", args.out)


main()
