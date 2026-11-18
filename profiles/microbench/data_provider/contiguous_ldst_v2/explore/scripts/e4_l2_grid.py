"""E4: L2 residency / grid-size effects (user Q4).

Part A - grid sweep, single-shot: P in {1,2,4,8,16,32,56,112} programs, each
  loading one distinct 4 KiB block and storing it (memcpy-style).  Warm (same
  regions every launch) vs cold (base rotates over a >256 MB span, so no
  launch revisits a region).  Question: does per-program cost / total time
  depend on P beyond linear work scaling, and does L2 state change it?

Part B - cross-program reuse at P=56: all programs read the SAME 4 KiB block
  (perfect reuse) vs disjoint blocks (no reuse), identical total bytes.

Part C - footprint sweep through L2 (streaming regime): grid=56 loop kernel
  (acc-chain-free? no - independent), footprint per launch in
  {8,32,64,96,128,160,224,288} MB spanning the 128 MB L2; two-point K-marginal
  (K in {2048, 4096}) gives the per-round streaming cost.  The per-round cost
  as a function of footprint shows where the L2 cliff is, per engine and
  direction.  UNROLL=1, no cross-iteration dependency (store version writes
  through the same window).

Usage:
  bash run_exp.sh e4_l2_grid.py <card> e4_c<card>.jsonl --modes simd simt_only
"""
import argparse
import torch
import triton
import triton.language as tl

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024  # 4 KiB fp32
ROT_SPAN_MB = 320


@triton.jit(do_not_specialize=["boff"])
def e4_copy1(src, dst, boff, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = boff + pid * BLOCK
    x = tl.load(src + base + lane)
    tl.store(dst + base + lane, x)


@triton.jit(do_not_specialize=["iters", "winmask"])
def e4_stream(src, dst, iters, winmask, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    span = (winmask + 1) * BLOCK
    base = pid * span
    lane = tl.arange(0, BLOCK)
    if 1:  # load variant is selected by passing zero-sized store? keep separate
        acc = tl.zeros((BLOCK,), src.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(src + base + (step & winmask) * BLOCK + lane)
        tl.store(dst + pid * BLOCK + lane, acc)


@triton.jit(do_not_specialize=["iters", "winmask"])
def e4_stream_store(src, dst, iters, winmask, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    span = (winmask + 1) * BLOCK
    base = pid * span
    lane = tl.arange(0, BLOCK)
    val = tl.full((BLOCK,), 1, dst.dtype.element_ty)
    for step in range(iters):
        tl.store(dst + base + (step & winmask) * BLOCK + lane, val)


@triton.jit
def e4_ctl_tiny(dst):
    tl.store(dst, 0)


@triton.jit
def e4_ctl_4k(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


GRIDS = [1, 2, 4, 8, 16, 32, 56, 112]
FOOTPRINTS_MB = [8, 32, 64, 96, 128, 160, 224, 288]


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default="simd simt_only", nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--ks", default="2048,4096")
    ap.add_argument("--parts", default="ABC", help="subset of A,B,C to run")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    parts = set(args.parts)

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    n_total = int((ROT_SPAN_MB + 64) * 1024 * 1024 // 4)
    src = torch.randn(n_total, device="npu")
    dstb = torch.zeros(n_total, device="npu")
    ctl_dst = torch.zeros(BLOCK * 2, device="npu")

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        # ---------- Part A: grid sweep, single-shot copy ----------------
        for warm in ((True, False) if 'A' in parts else ()):
            nregions = 1 if warm else (ROT_SPAN_MB * 1024 * 1024 // 4) // (BLOCK * 512)
            ir = [0]

            def fprobe(P, nregions=nregions, ir=ir):
                off = 0 if warm else (ir[0] % nregions) * BLOCK * 512
                ir[0] += 1
                e4_copy1[(P,)](src, dstb, off, BLOCK, **opts)

            for P in GRIDS:
                rows = prof_session(
                    [lambda P=P: fprobe(P),
                     lambda: e4_ctl_4k[(1,)](ctl_dst, BLOCK, **opts)],
                    ["e4_copy1", "e4_ctl_4k"], args.pairs)
                rec = {"exp": "e4", "part": "A", "mode": mode, "P": P,
                       "state": "warm" if warm else "cold",
                       "probe": stats_us(rows["e4_copy1"]),
                       "ctl": stats_us(rows["e4_ctl_4k"]),
                       "diff": pair_diff_ns(rows["e4_copy1"], rows["e4_ctl_4k"])}
                out.write(**rec)
                print("A", mode, P, "warm" if warm else "cold", rec["probe"],
                      rec.get("diff"), flush=True)

        # ---------- Part B: same vs disjoint at P=56 --------------------
        @triton.jit(do_not_specialize=["boff"])
        def e4_same(src, dst, boff, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            lane = tl.arange(0, BLOCK)
            x = tl.load(src + boff + lane)
            tl.store(dst + boff + lane, x)

        for same in ((True, False) if 'B' in parts else ()):
            def f(same=same):
                if same:
                    e4_same[(56,)](src, dstb, 0, BLOCK, **opts)
                else:
                    e4_copy1[(56,)](src, dstb, 0, BLOCK, **opts)
            rows = prof_session([f, lambda: e4_ctl_4k[(1,)](ctl_dst, BLOCK, **opts)],
                                ["e4_", "e4_ctl_4k"], args.pairs)
            keys = [k for k in rows if k != "e4_ctl_4k"]
            probe_key = keys[0]
            rec = {"exp": "e4", "part": "B", "mode": mode, "P": 56,
                   "reuse": "same_block" if same else "disjoint",
                   "probe": stats_us(rows[probe_key]),
                   "ctl": stats_us(rows["e4_ctl_4k"]),
                   "diff": pair_diff_ns(rows[probe_key], rows["e4_ctl_4k"])}
            out.write(**rec)
            print("B", mode, rec["reuse"], rec["probe"], flush=True)

        # ---------- Part C: footprint sweep through L2 ------------------
        for fmb in (FOOTPRINTS_MB if 'C' in parts else []):
            winblocks = max(2, fmb * 1024 * 1024 // (4 * BLOCK * 56))
            winmask = 1
            while winmask < winblocks:
                winmask *= 2
            winmask -= 1
            for direction in ("load", "store"):
                costs = {}
                for K in ks:
                    kname = "e4_stream_store" if direction == "store" else "e4_stream"
                    def f(K=K):
                        if direction == "load":
                            e4_stream[(56,)](src, dstb, K, winmask, BLOCK, **opts)
                        else:
                            e4_stream_store[(56,)](src, dstb, K, winmask, BLOCK, **opts)
                    rows = prof_session(
                        [f, lambda: e4_ctl_tiny[(1,)](ctl_dst, **opts)],
                        [kname, "e4_ctl_tiny"], args.pairs)
                    costs[K] = stats_us(rows[kname])
                    ctl_stats = stats_us(rows["e4_ctl_tiny"])
                if len(ks) == 2 and all(costs.values()):
                    slope_ns = (costs[ks[1]]["min_us"] - costs[ks[0]]["min_us"]) * 1000.0 / (ks[1] - ks[0])
                else:
                    slope_ns = None
                rec = {"exp": "e4", "part": "C", "mode": mode,
                       "direction": direction, "footprint_mb": fmb,
                       "winmask": winmask, "per_round_us": costs,
                       "marginal_ns_per_round": round(slope_ns, 1) if slope_ns else None,
                       "ctl": ctl_stats}
                out.write(**rec)
                print("C", mode, direction, fmb, "MB",
                      "marginal", rec["marginal_ns_per_round"], "ns/round",
                      flush=True)
    print("E4 DONE", flush=True)


if __name__ == "__main__":
    main()
