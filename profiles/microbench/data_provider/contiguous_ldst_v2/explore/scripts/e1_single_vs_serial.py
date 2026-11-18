"""E1: single-shot vs serial-dependent for-loop (Q1 + Q2), with an explicit
L2-state axis (user's L2 hypothesis).

Questions
  Q1  Is a single-shot access just the K==1 special case of an acc-chain
      loop, or does the loop context itself change the per-access price?
  Q2  As K grows, does the per-round marginal amortize (loop startup /
      first-touch latency / final drain)?
  Q4a Is the difference mainly L2 state?  Three window regimes per K:
        same  - every iteration re-reads the SAME block (winmask=0).
        warm  - K distinct blocks per launch, the same region every launch
                (footprint = K*4KB <= L2 after the first launch).
        cold  - K distinct blocks, and the base offset rotates over a
                >256 MB region set, so every launch reads never-cached
                lines (footprint flows: launches never revisit a region).

Method: every point is one ABAB profiler session
  A = probe kernel, B = control kernel (store-only / tiny store),
mirroring probe_forms_single_shot.py so numbers are comparable with the
existing calibration.  Absolute medians of both sides are recorded too,
so we can answer *total* kernel time questions, not only differences.

Usage (server):
  bash run_exp.sh e1_single_vs_serial.py <card> e1_fp32_c<card>.jsonl \
      --dtypes fp32 --modes simd simt_only
"""
import argparse
import torch
import triton
import triton.language as tl

from common import (add_common_args, gate_or_die, health_check, pair_diff_ns,
                    prof_session, stats_us, std_opts, JsonlOut)

DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
      "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}
BLOCK = 1024          # fp32: 4 KB per block
COLD_TOTAL_MB = 320   # rotation span, safely above the 128 MB L2


@triton.jit(do_not_specialize=["iters", "winmask", "boff"])
def e1_load_loop(src, dst, iters, winmask, boff,
                 BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + boff + (step & winmask) * BLOCK + lane)
    tl.store(dst + lane, acc)


@triton.jit
def e1_load_once(src, dst, boff, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    x = tl.load(src + boff + lane)
    tl.store(dst + lane, x)


@triton.jit
def e1_ctl_store(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


@triton.jit(do_not_specialize=["iters", "winmask", "boff"])
def e1_store_loop(dst, iters, winmask, boff, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    val = tl.full((BLOCK,), 1, dst.dtype.element_ty)
    for step in range(iters):
        tl.store(dst + boff + (step & winmask) * BLOCK + lane, val)


@triton.jit
def e1_store_once(dst, boff, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + boff + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


@triton.jit
def e1_ctl_tiny(dst):
    tl.store(dst, 0)


KS = [1, 2, 4, 8, 16, 32, 64, 128]


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--dtypes", default="fp32")
    ap.add_argument("--modes", default="simd simt_only", nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    big_n = COLD_TOTAL_MB * 1024 * 1024 // 4          # fp32 elements
    src = (torch.randn(big_n, device="npu") * 7)
    dstbig = torch.zeros(big_n, device="npu")
    dst = torch.zeros(BLOCK * 2, device="npu")

    for dtype in args.dtypes.split(","):
        for mode in args.modes:
            opts = std_opts(mode, args.warps)
            by = BLOCK * EB[dtype]
            region_elems = max(1, COLD_TOTAL_MB * 1024 * 1024 // EB[dtype])
            sbuf = src.to(DT[dtype]) if dtype != "fp32" else src
            dbuf = dstbig.to(DT[dtype]) if dtype != "fp32" else dstbig
            n_regions_big = region_elems // (BLOCK * 512)   # >=512 blocks/region

            def run_point(tag, fns, names, extra):
                rows = prof_session(fns, names, args.pairs)
                rec = {"exp": "e1", "mode": mode, "dtype": dtype, "by": by,
                       "w": args.warps, "tag": tag, **extra}
                for nm, fl in zip(names, ["probe", "ctl"]):
                    rec[fl] = stats_us(rows[nm])
                d = pair_diff_ns(rows[names[0]], rows[names[1]])
                if d:
                    rec["diff"] = d
                out.write(**rec)
                print(tag, rec.get("probe"), rec.get("diff"), flush=True)

            # ---- load side: single-shot vs acc-chain loop --------------
            # single shot (K==1 reference, no loop at all), warm vs cold
            for tag, nreg in (("ss_warm", 1), ("ss_cold", n_regions_big)):
                def fss(i=[0]):
                    if nreg <= 1:
                        off = 0
                    else:
                        off = (i[0] % nreg) * BLOCK
                        i[0] += 1
                    e1_load_once[(1,)](sbuf, dst, off, BLOCK, **opts)
                run_point("load_" + tag, [fss, lambda: e1_ctl_store[(1,)](
                    dst, BLOCK, **opts)], ["e1_load_once", "e1_ctl_store"],
                    {"kind": "load", "K": 1, "window": tag})

            for K in ks:
                # same-block: winmask=0 -> every iteration re-reads block 0
                def fsame():
                    e1_load_loop[(1,)](sbuf, dst, K, 0, 0, BLOCK, **opts)
                run_point("load_same_K%d" % K,
                          [fsame, lambda: e1_ctl_store[(1,)](dst, BLOCK, **opts)],
                          ["e1_load_loop", "e1_ctl_store"],
                          {"kind": "load", "K": K, "window": "same"})

                # warm distinct: K distinct blocks, same region each launch
                def fwarm():
                    e1_load_loop[(1,)](sbuf, dst, K, K - 1, 0, BLOCK, **opts)
                run_point("load_warm_K%d" % K,
                          [fwarm, lambda: e1_ctl_store[(1,)](dst, BLOCK, **opts)],
                          ["e1_load_loop", "e1_ctl_store"],
                          {"kind": "load", "K": K, "window": "warm"})

                # cold: rotate base over a >256 MB span
                ir = [0]

                def fcold():
                    off = (ir[0] % n_regions_big) * BLOCK * 512
                    ir[0] += 1
                    e1_load_loop[(1,)](sbuf, dst, K, K - 1, off, BLOCK, **opts)
                run_point("load_cold_K%d" % K,
                          [fcold, lambda: e1_ctl_store[(1,)](dst, BLOCK, **opts)],
                          ["e1_load_loop", "e1_ctl_store"],
                          {"kind": "load", "K": K, "window": "cold"})

            # ---- store side --------------------------------------------
            for tag, nreg in (("ss_warm", 1), ("ss_cold", n_regions_big)):
                def fss(i=[0]):
                    off = 0 if nreg <= 1 else (i[0] % nreg) * BLOCK
                    if nreg > 1:
                        i[0] += 1
                    e1_store_once[(1,)](dbuf, off, BLOCK, **opts)
                run_point("store_" + tag,
                          [fss, lambda: e1_ctl_tiny[(1,)](dst, **opts)],
                          ["e1_store_once", "e1_ctl_tiny"],
                          {"kind": "store", "K": 1, "window": tag})

            for K in ks:
                def fsame():
                    e1_store_loop[(1,)](dbuf, K, 0, 0, BLOCK, **opts)
                run_point("store_same_K%d" % K,
                          [fsame, lambda: e1_ctl_tiny[(1,)](dst, **opts)],
                          ["e1_store_loop", "e1_ctl_tiny"],
                          {"kind": "store", "K": K, "window": "same"})

                def fwarm():
                    e1_store_loop[(1,)](dbuf, K, K - 1, 0, BLOCK, **opts)
                run_point("store_warm_K%d" % K,
                          [fwarm, lambda: e1_ctl_tiny[(1,)](dst, **opts)],
                          ["e1_store_loop", "e1_ctl_tiny"],
                          {"kind": "store", "K": K, "window": "warm"})

                ir = [0]

                def fcold():
                    off = (ir[0] % n_regions_big) * BLOCK * 512
                    ir[0] += 1
                    e1_store_loop[(1,)](dbuf, K, K - 1, off, BLOCK, **opts)
                run_point("store_cold_K%d" % K,
                          [fcold, lambda: e1_ctl_tiny[(1,)](dst, **opts)],
                          ["e1_store_loop", "e1_ctl_tiny"],
                          {"kind": "store", "K": K, "window": "cold"})

    print("E1 DONE", flush=True)


if __name__ == "__main__":
    main()
