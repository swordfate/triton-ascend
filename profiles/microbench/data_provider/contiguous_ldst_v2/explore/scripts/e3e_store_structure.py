"""E3e: WHY is simt independent-load linear while simt chain is sub-linear?

E3c established (4 KiB fp32, grid=1, W=4):
  simt ind   (U loads -> U live vars -> U per-load stores): ~277 ns/load LINEAR
  simt chain (U loads -> multiply chain -> ONE store):      sub-linear ~90-166
  simd both: sub-linear ~30-50.

Two candidate mechanisms:
  H1  per-load STORE interference: on SIMT the LSU is a single path; a store
      that depends on load j cannot overlap with issuing load j+1, so the
      sequence load,store,load,store pays latency per pair.  (SIMD is immune:
      MTE2/MTE3 are separate hardware queues.)
  H2  live-value pressure: ind keeps U loaded blocks live simultaneously
      (register / DCache-refill tracking pressure grows with U); chain keeps
      ONE live block.

Discriminating variants (all warm, 4 KiB fp32, grid=1, 3 sessions, min):
  v1_1store   U loads -> U live vars -> ONE final store of their sum.
      H1 predicts sub-linear (no intermediate stores);
      H2 predicts still linear (U blocks still live).
  v2_endstore U loads -> U live vars -> ALL U stores AFTER the loads.
      If order is preserved, loads finish issuing before any store:
      H1 predicts sub-linear; H2 predicts linear.
  v3_interleave (reference, = E3c ind): a store after every load
      -> known linear on simt.

Controls perform the SAME arithmetic/stores on tl.full constants, so the
diff prices exactly the U loads (the adds cancel too).

Usage:
  bash run_exp.sh e3e_store_structure.py 2 e3e_c2.jsonl --modes simd simt_only
"""
import argparse
import importlib.util
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024
US = (1, 2, 4, 8)

HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n"


def gen_module(path):
    parts = [HEADER]
    for u in US:
        # ---- v1: one final store of the sum (adds in probe AND control) ----
        loads = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)" % (j, j)
            for j in range(u))
        chain = "x0"
        for j in range(1, u):
            chain = "(%s + x%d)" % (chain, j)
        parts.append(
            "@triton.jit\ndef e3e_1store_u%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n"
            "    s = %s + 1\n"
            "    tl.store(dst + lane, s)\n\n" % (u, loads, chain))
        consts = "\n".join(
            "    c%d = tl.full((BLOCK,), %d, dst.dtype.element_ty)"
            % (j, j + 1) for j in range(u))
        cchain = "c0"
        for j in range(1, u):
            cchain = "(%s + c%d)" % (cchain, j)
        parts.append(
            "@triton.jit\ndef e3e_1store_ctl_u%d(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n"
            "    s = %s + 1\n"
            "    tl.store(dst + lane, s)\n\n" % (u, consts, cchain))

        # ---- v2: all stores after all loads --------------------------------
        loads2 = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)" % (j, j)
            for j in range(u))
        stores2 = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j)
            for j in range(u))
        parts.append(
            "@triton.jit\ndef e3e_endstore_u%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n%s\n\n" % (u, loads2, stores2))
        consts2 = "\n".join(
            "    c%d = tl.full((BLOCK,), %d, dst.dtype.element_ty)"
            % (j, j + 1) for j in range(u))
        stores2c = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, c%d + 1)" % (j, j)
            for j in range(u))
        parts.append(
            "@triton.jit\ndef e3e_endstore_ctl_u%d(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n%s\n\n"
            % (u, consts2, stores2c))

        # ---- v3: store after every load (E3c reference) --------------------
        inter = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)\n"
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j, j, j)
            for j in range(u))
        inter_ctl = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, "
            "tl.full((BLOCK,), %d, dst.dtype.element_ty))" % (j, j + 1)
            for j in range(u))
        parts.append(
            "@triton.jit\ndef e3e_inter_u%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (u, inter))
        parts.append(
            "@triton.jit\ndef e3e_inter_ctl_u%d(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (u, inter_ctl))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_e3e", path)
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
    gen_path = str(Path(args.out).parent / "_gen_e3e.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    src = torch.randn(BLOCK * 64, device="npu")
    dst = torch.zeros(BLOCK * 64, device="npu")

    variants = [
        ("v1_1store", "e3e_1store_u%d", "e3e_1store_ctl_u%d"),
        ("v2_endstore", "e3e_endstore_u%d", "e3e_endstore_ctl_u%d"),
        ("v3_interleave", "e3e_inter_u%d", "e3e_inter_ctl_u%d"),
    ]
    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        for vtag, probe_t, ctl_t in variants:
            for u in US:
                probe = getattr(mod, probe_t % u)
                ctl = getattr(mod, ctl_t % u)
                measure("%s_%s_u%d" % (vtag, mode, u),
                        [lambda p=probe: p[(1,)](src, dst, BLOCK, **opts),
                         lambda c=ctl: c[(1,)](dst, BLOCK, **opts)],
                        [probe_t % u, ctl_t % u],
                        args.pairs, args.sessions, out,
                        {"exp": "e3e", "mode": mode, "variant": vtag, "U": u})
    print("E3E DONE", flush=True)


if __name__ == "__main__":
    main()
