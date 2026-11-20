"""cal_e_loopgrid.py - Formula B coverage campaign: loop-body prices across
grid (loop-context P-axis) plus the missing mixed body indpt2
(two independent loads + one store per round - the muls_add/add shape the
100-case elementwise family kept falling back on).

One prof_pairs_session per (mode, dtype, bytes, body, grid) covers all K
points; diff vs the 1-store control; T(K) = P + K*m fitted downstream.

Usage:
  bash run_exp.sh cal_e_loopgrid.py <card> calE_simd.jsonl \
      --modes simd --bodies chain indpt indpt2 store --grids 1 56 224
"""
import argparse
import importlib.util
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_pairs_session, std_opts)

DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
      "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}

HEADER = "import triton\nimport triton.language as tl\n\n"


def gen_module(path, body, dtype_key, ks):
    parts = [HEADER]
    for k in ks:
        if body == "chain":
            body_code = (
                "    acc = tl.zeros((BLOCK,), src.dtype.element_ty)\n"
                "    for step in range(iters):\n"
                "        acc += tl.load(src + boff + step * BLOCK + lane)\n"
                "    tl.store(dst + lane, acc)\n")
            sig = "(src, dst, iters, boff, BLOCK: tl.constexpr)"
        elif body == "indpt":
            body_code = (
                "    for step in range(iters):\n"
                "        x = tl.load(src + boff + step * BLOCK + lane)\n"
                "        tl.store(dst + boff + step * BLOCK + lane, x + 1)\n")
            sig = "(src, dst, iters, boff, BLOCK: tl.constexpr)"
        elif body == "indpt2":
            body_code = (
                "    for step in range(iters):\n"
                "        x = tl.load(src + boff + step * BLOCK + lane)\n"
                "        y = tl.load(src + boff + (iters + step) * BLOCK "
                "+ lane)\n"
                "        tl.store(dst + boff + step * BLOCK + lane, "
                "(x + y) * 0.5)\n")
            sig = "(src, dst, iters, boff, BLOCK: tl.constexpr)"
        else:
            body_code = (
                "    for step in range(iters):\n"
                "        tl.store(dst + boff + step * BLOCK + lane, "
                "tl.full((BLOCK,), 1, dst.dtype.element_ty))\n")
            sig = "(dst, iters, boff, BLOCK: tl.constexpr)"
        parts.append(
            "@triton.jit(do_not_specialize=['iters', 'boff'])\n"
            "def ce_%s_k%d%s:\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (body, k, sig,
                                                       body_code))
    parts.append(
        "@triton.jit\n"
        "def ce_ctl_1store(dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    tl.store(dst + lane, tl.full((BLOCK,), 1, "
        "dst.dtype.element_ty))\n")
    parts.append(
        "@triton.jit\n"
        "def ce_ref8(src, dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for j in tl.static_range(8):\n"
        "        x = tl.load(src + j * BLOCK + lane)\n"
        "        tl.store(dst + j * BLOCK + lane, x + 1)\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_cal_e", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default=["simd"], nargs="+")
    ap.add_argument("--dtypes", default=["fp32"], nargs="+")
    ap.add_argument("--bytes-list", type=int, nargs="+",
                    default=[1024, 4096, 16384])
    ap.add_argument("--bodies", nargs="+",
                    default=["chain", "indpt", "indpt2", "store"])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 8, 32, 128])
    ap.add_argument("--grids", type=int, nargs="+", default=[1, 56, 224])
    ap.add_argument("--warps-simd", type=int, nargs="+", default=[4])
    ap.add_argument("--warps-simt", type=int, nargs="+", default=[4])
    ap.add_argument("--sessions", type=int, default=2)
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)

    for mode in args.modes:
        warp_list = args.warps_simd if mode == "simd" else args.warps_simt
        for warps in warp_list:
            for dtype in args.dtypes:
                for nbytes in args.bytes_list:
                    block = max(1, nbytes // EB[dtype])
                    for body in args.bodies:
                        # indpt2 streams 2*K blocks from src
                        stride_blocks = (2 * max(args.ks) if body == "indpt2"
                                         else max(args.ks))
                        gen_path = str(Path(args.out).parent /
                                       ("_gen_cal_e_%s_%s_%d.py" %
                                        (dtype, body, nbytes)))
                        gen_module(gen_path, body, dtype, args.ks)
                        mod = load_module(gen_path)
                        n = (320 + 64) * 1024 * 1024 // EB[dtype]
                        src = torch.randn(n, device="npu")
                        dst = torch.zeros(n, device="npu")
                        for grid in args.grids:
                            opts = std_opts(mode, warps)
                            spec = []
                            for k in args.ks:
                                probe = getattr(mod, "ce_%s_k%d" % (body, k))
                                if body == "store":
                                    spec.append((
                                        lambda pr=probe, k=k: pr[(grid,)](
                                            dst, k, 0, block, **opts),
                                        "ce_%s_k%d" % (body, k),
                                        lambda: getattr(
                                            mod, "ce_ctl_1store")[(
                                                grid,)](dst, block, **opts),
                                        "ce_ctl_1store"))
                                else:
                                    spec.append((
                                        lambda pr=probe, k=k: pr[(grid,)](
                                            src, dst, k, 0, block, **opts),
                                        "ce_%s_k%d" % (body, k),
                                        lambda: getattr(
                                            mod, "ce_ctl_1store")[(
                                                grid,)](dst, block, **opts),
                                        "ce_ctl_1store"))
                            if mode == "simt_only":
                                spec.append((
                                    lambda: getattr(mod, "ce_ref8")[(
                                        grid,)](src, dst, block, **opts),
                                    "ce_ref8",
                                    lambda: getattr(mod, "ce_ref8")[(
                                        grid,)](src, dst, block, **opts),
                                    "ce_ref8_ctl"))
                            for si in range(args.sessions):
                                durs = prof_pairs_session(spec, rounds=6)
                                for k in args.ks:
                                    d = pair_diff_ns(
                                        durs.get("ce_%s_k%d" % (body, k), []),
                                        durs.get("ce_ctl_1store", []))
                                    if not d:
                                        continue
                                    out.write(kind="loopB", mode=mode,
                                              dtype=dtype, bytes=nbytes,
                                              body=body, warps=warps,
                                              grid=grid, k=k, sessions=si,
                                              ns=d["median"],
                                              q1=d.get("q1"), q3=d.get("q3"),
                                              cv=d.get("cv"))
                            print("done %s %s b%d %s g%d" %
                                  (mode, dtype, nbytes, body, grid),
                                  flush=True)


if __name__ == "__main__":
    main()
