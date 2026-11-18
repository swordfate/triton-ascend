"""E6: is a dynamic for-loop essentially different from straight-line code with
the same accesses?  (Reviewer question: why model runtime-for separately?)

Real-kernel context for each cell:
  burst  - straight-line / static-unrolled multi access: fused elementwise
           (vector_add family: 2-3 loads different tensors + 1 store),
           view/insert_slice copies, small unrolled n-ary ops.
  loop   - dynamic trip count: grid-stride elementwise over runtime n
           (memory_stream / muls_add), reduction accumulation loops
           (sum/mean/var over runtime K, matvec k-loop, fbgemm row loop).

This probe measures the SAME access pattern as a burst and as a loop and
compares per-instance marginals (real streaming addresses base+step*BLOCK,
no window re-read; footprint K*4KB stays < L2 -> warm regime):

  loop_ind     for step in range(iters):   # runtime iters
                   x = load(src + off + step*BLOCK + lane)
                   store(dst + off + step*BLOCK + lane, x + 1)
  burst_ind    U interleaved load/store pairs, U in {1,2,4,8}  (= E3c ind)

Verdict rule: if loop marginal (large K) == burst marginal, the loop is just
"the same accesses, K of them" and the taxonomy collapses; if they differ,
the loop context (backedge + limited lookahead + in-order issue) is a real
axis and (a)/(b) must stay separate.

Usage:
  bash run_exp.sh e6_loop_vs_burst.py 2 e6_c2.jsonl --modes simd simt_only
"""
import argparse
import importlib.util
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024
KS = (1, 2, 4, 8, 16, 32, 64, 128)
US = (1, 2, 4, 8)

HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n"


def gen_module(path):
    parts = [HEADER]
    for u in US:
        inter = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(u))
        ctl = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, "
            "tl.full((BLOCK,), %d, dst.dtype.element_ty))" % (j, j + 1)
            for j in range(u))
        parts.append(
            "@triton.jit\ndef e6_burst_u%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (u, inter))
        parts.append(
            "@triton.jit\ndef e6_burst_ctl_u%d(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (u, ctl))
    parts.append(
        "@triton.jit(do_not_specialize=['iters', 'boff'])\n"
        "def e6_loop_ind(src, dst, iters, boff, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for step in range(iters):\n"
        "        x = tl.load(src + boff + step * BLOCK + lane)\n"
        "        tl.store(dst + boff + step * BLOCK + lane, x + 1)\n")
    parts.append(
        "@triton.jit\n"
        "def e6_ctl_1(dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    tl.store(dst + lane, tl.full((BLOCK,), 1, dst.dtype.element_ty))\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_e6", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(tag, fns, names, pairs, sessions, out, extra):
    best = None
    sessions_out = []
    for _ in range(sessions):
        rows = prof_session(fns, names, pairs)
        d = pair_diff_ns(rows[names[0]], rows[names[1]])
        sessions_out.append({"probe": stats_us(rows[names[0]]),
                             "ctl": stats_us(rows[names[1]]), "diff": d})
        if d and d["cv"] <= 0.4:
            if best is None or d["diff_med_ns"] < best:
                best = d["diff_med_ns"]
    out.write(tag=tag, diff_med_ns_min=best, sessions=sessions_out, **extra)
    print(tag, "min_diff=%s" % best, flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default="simd simt_only", nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=3)
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    gen_path = str(Path(args.out).parent / "_gen_e6.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    n = BLOCK * 1024                     # 4 MB fp32 working set per direction
    src = torch.randn(n, device="npu")
    dst = torch.zeros(n, device="npu")

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        for K in KS:
            measure("loop_ind_%s_K%d" % (mode, K),
                    [lambda K=K: mod.e6_loop_ind[(1,)](src, dst, K, 0, BLOCK,
                                                       **opts),
                     lambda: mod.e6_ctl_1[(1,)](dst, BLOCK, **opts)],
                    ["e6_loop_ind", "e6_ctl_1"],
                    args.pairs, args.sessions, out,
                    {"exp": "e6", "kind": "loop_ind", "mode": mode, "K": K})
        for u in US:
            probe = getattr(mod, "e6_burst_u%d" % u)
            ctl = getattr(mod, "e6_burst_ctl_u%d" % u)
            measure("burst_%s_u%d" % (mode, u),
                    [lambda p=probe: p[(1,)](src, dst, BLOCK, **opts),
                     lambda c=ctl: c[(1,)](dst, BLOCK, **opts)],
                    ["e6_burst_u%d" % u, "e6_burst_ctl_u%d" % u],
                    args.pairs, args.sessions, out,
                    {"exp": "e6", "kind": "burst", "mode": mode, "U": u})
    print("E6 DONE", flush=True)


if __name__ == "__main__":
    main()
