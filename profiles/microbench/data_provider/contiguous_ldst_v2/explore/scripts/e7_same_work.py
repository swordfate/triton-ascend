"""E7: same total work, different code shape - the definitive (a)/(b) proof.

Reviewer question: "does a runtime-trip-count loop really fail to reach the
concurrency of unrolled/straight-line code?  Prove it with a concrete example,
so we can justify modeling runtime loops separately."

Design: fix the TOTAL number of 4 KiB load+store pairs at N (same bytes moved,
same addresses streaming, warm L2), then vary only the code shape:

  A_loop1    runtime scf.for, trip count N, body = 1 pair
             -> N rounds, 1 pair in flight per round (approx.)
  C_burst    straight-line N pairs  (TTIR: N load + N store ops)
  D_static   `for step in tl.static_range(N)` -> frontend unrolls -> same IR
             shape as C (sanity: static loop must equal burst)
  E_loopu4   runtime scf.for, trip count N/4, body = static_range(4) pairs
             -> 4 pairs in flight per round; if the mechanism is per-round
             issue throttle, E lands between A and C, near C.

Controls do the SAME stores on constants:
  A/E control: runtime loop of full stores (same trip count)
  C/D control: straight-line N full stores

diff(A) vs diff(C) at equal N is the whole proof: same work, different time.
N in {8, 32} (compile time keeps U=32 straight-line manageable).

Usage:
  bash run_exp.sh e7_same_work.py 3 e7_c3.jsonl --modes simd simt_only
"""
import argparse
import importlib.util
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024
NS = (8, 32)

HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n"


def gen_module(path):
    parts = [HEADER]
    for n in NS:
        pairs = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(n))
        stores = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, "
            "tl.full((BLOCK,), %d, dst.dtype.element_ty))" % (j, j + 1)
            for j in range(n))
        static_pairs = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(n))
        parts.append(
            "@triton.jit\ndef e7_burst_%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (n, pairs))
        parts.append(
            "@triton.jit\ndef e7_burst_ctl_%d(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (n, stores))
        parts.append(
            "@triton.jit\ndef e7_static_%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n"
            "    for step in tl.static_range(%d):\n"
            "        x = tl.load(src + step * BLOCK + lane)\n"
            "        tl.store(dst + step * BLOCK + lane, x + 1)\n\n" % (n, n))
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e7_loop1(src, dst, iters, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for step in range(iters):\n"
        "        x = tl.load(src + step * BLOCK + lane)\n"
        "        tl.store(dst + step * BLOCK + lane, x + 1)\n\n")
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e7_loop1_ctl(dst, iters, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for step in range(iters):\n"
        "        tl.store(dst + step * BLOCK + lane, "
        "tl.full((BLOCK,), 1, dst.dtype.element_ty))\n\n")
    parts.append(
        "@triton.jit(do_not_specialize=['iters'])\n"
        "def e7_loopu4(src, dst, iters, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for step in range(iters):\n"
        "        for j in tl.static_range(4):\n"
        "            x = tl.load(src + (step * 4 + j) * BLOCK + lane)\n"
        "            tl.store(dst + (step * 4 + j) * BLOCK + lane, x + 1)\n\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_e7", path)
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
    gen_path = str(Path(args.out).parent / "_gen_e7.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    n_elems = BLOCK * 1024
    src = torch.randn(n_elems, device="npu")
    dst = torch.zeros(n_elems, device="npu")

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        for n in NS:
            # A: runtime loop, 1 pair per round
            measure("A_loop1_%s_N%d" % (mode, n),
                    [lambda n=n: mod.e7_loop1[(1,)](src, dst, n, BLOCK, **opts),
                     lambda n=n: mod.e7_loop1_ctl[(1,)](dst, n, BLOCK, **opts)],
                    ["e7_loop1", "e7_loop1_ctl"],
                    args.pairs, args.sessions, out,
                    {"exp": "e7", "shape": "loop1", "mode": mode, "N": n})
            # C: straight-line burst
            measure("C_burst_%s_N%d" % (mode, n),
                    [lambda n=n: getattr(mod, "e7_burst_%d" % n)[(1,)](
                        src, dst, BLOCK, **opts),
                     lambda n=n: getattr(mod, "e7_burst_ctl_%d" % n)[(1,)](
                         dst, BLOCK, **opts)],
                    ["e7_burst_%d" % n, "e7_burst_ctl_%d" % n],
                    args.pairs, args.sessions, out,
                    {"exp": "e7", "shape": "burst", "mode": mode, "N": n})
            # D: static_range loop (frontend unrolls -> burst shape)
            measure("D_static_%s_N%d" % (mode, n),
                    [lambda n=n: getattr(mod, "e7_static_%d" % n)[(1,)](
                        src, dst, BLOCK, **opts),
                     lambda n=n: getattr(mod, "e7_burst_ctl_%d" % n)[(1,)](
                         dst, BLOCK, **opts)],
                    ["e7_static_%d" % n, "e7_burst_ctl_%d" % n],
                    args.pairs, args.sessions, out,
                    {"exp": "e7", "shape": "static", "mode": mode, "N": n})
        # E: runtime loop with 4 static pairs per round (N=32 -> 8 rounds)
        if 32 in NS:
            measure("E_loopu4_%s_N32" % mode,
                    [lambda: mod.e7_loopu4[(1,)](src, dst, 8, BLOCK, **opts),
                     lambda: mod.e7_loop1_ctl[(1,)](dst, 8, BLOCK, **opts)],
                    ["e7_loopu4", "e7_loop1_ctl"],
                    args.pairs, args.sessions, out,
                    {"exp": "e7", "shape": "loopu4", "mode": mode, "N": 32})
    print("E7 DONE", flush=True)


if __name__ == "__main__":
    main()
