"""cal_d_grid.py - Formula A coverage campaign: groupA prices across
grid (P-axis in the straight-line context), simt warp count, and the
small-byte buckets the 100-case regressions exposed (32..512B).

One prof_pairs_session per (mode, dtype, bytes, warps, grid) cell covers
all U points; diff = per-launch group cost at that contention level.
Per-program cost = diff / ceil(grid / 56).

Usage:
  bash run_exp.sh cal_d_grid.py <card> calD_simd.jsonl \
      --modes simd --dtypes fp16 fp32 --bytes-list 32 64 128 256 512 1024 4096 16384 \
      --grids 1 56 224 --sessions 2
"""
import argparse
import importlib.util
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_pairs_session, std_opts)

DT = {"int8": torch.int8, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
      "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "bf16": 2, "fp32": 4, "int64": 8}

HEADER = "import triton\nimport triton.language as tl\n\n"


def gen_module(path, us, dtype_key):
    eb = EB[dtype_key]
    parts = [HEADER]
    for u in us:
        loads = "\n".join(
            "    x%d = tl.load(src + boff + %d * BLOCK + lane)" % (j, j)
            for j in range(u))
        chain = "x0"
        for j in range(1, u):
            chain = "(%s + x%d)" % (chain, j)
        tail = ("    tl.store(dst + lane, %s + 1)" % chain) if u > 1 \
            else "    tl.store(dst + lane, x0 + 1)"
        consts = "\n".join(
            "    c%d = tl.full((BLOCK,), %d, dst.dtype.element_ty)"
            % (j, j + 1) for j in range(u))
        cchain = "c0"
        for j in range(1, u):
            cchain = "(%s + c%d)" % (cchain, j)
        ctl_tail = "    tl.store(dst + lane, %s + 1)" % cchain
        parts.append(
            "@triton.jit(do_not_specialize=['boff'])\n"
            "def cd_ld_u%d(src, dst, boff, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n%s\n\n" % (u, loads, tail))
        parts.append(
            "@triton.jit(do_not_specialize=['boff'])\n"
            "def cd_ld_ctl_u%d(dst, boff, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n%s\n\n"
            % (u, consts, ctl_tail))
    # simt state reference (same as cal_a): 8 x BLOCK load+store pairs
    parts.append(
        "@triton.jit\n"
        "def cd_ref8(src, dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for j in tl.static_range(8):\n"
        "        x = tl.load(src + j * BLOCK + lane)\n"
        "        tl.store(dst + j * BLOCK + lane, x + 1)\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_cal_d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default=["simd"], nargs="+")
    ap.add_argument("--dtypes", default=["fp32"], nargs="+")
    ap.add_argument("--bytes-list", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024, 4096, 16384])
    ap.add_argument("--us", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--grids", type=int, nargs="+", default=[1, 56, 224])
    ap.add_argument("--warps-simd", type=int, nargs="+", default=[4])
    ap.add_argument("--warps-simt", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--sessions", type=int, default=2)
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    ub_budget = 96 * 1024
    cores = 56

    for mode in args.modes:
        warp_list = args.warps_simd if mode == "simd" else args.warps_simt
        for warps in warp_list:
            for dtype in args.dtypes:
                for nbytes in args.bytes_list:
                    us = [u for u in args.us
                          if u * nbytes <= ub_budget]
                    if not us:
                        print("skip b%d: no U fits UB" % nbytes, flush=True)
                        continue
                    block = max(1, nbytes // EB[dtype])
                    gen_path = str(Path(args.out).parent /
                                   ("_gen_cal_d_%s_%d.py" % (dtype, nbytes)))
                    gen_module(gen_path, us, dtype)
                    mod = load_module(gen_path)
                    n = (320 + 64) * 1024 * 1024 // EB[dtype]
                    src = torch.randn(n, device="npu")
                    dst = torch.zeros(n, device="npu")
                    for grid in args.grids:
                        opts = std_opts(mode, warps)
                        spec = []
                        for u in us:
                            spec.append((
                                lambda u=u: getattr(
                                    mod, "cd_ld_u%d" % u)[(grid,)](
                                    src, dst, 0, block, **opts),
                                "cd_ld_u%d" % u,
                                lambda u=u: getattr(
                                    mod, "cd_ld_ctl_u%d" % u)[(grid,)](
                                    dst, 0, block, **opts),
                                "cd_ld_ctl_u%d" % u))
                        if mode == "simt_only":
                            spec.append((
                                lambda: getattr(mod, "cd_ref8")[(
                                    grid,)](src, dst, block, **opts),
                                "cd_ref8",
                                lambda: getattr(mod, "cd_ref8")[(
                                    grid,)](src, dst, block, **opts),
                                "cd_ref8_ctl"))
                        waves = max(1, (grid + cores - 1) // cores)
                        for si in range(args.sessions):
                            durs = prof_pairs_session(spec, rounds=6)
                            for u in us:
                                d = pair_diff_ns(durs.get("cd_ld_u%d" % u, []),
                                                 durs.get("cd_ld_ctl_u%d" % u,
                                                          []))
                                if not d:
                                    continue
                                out.write(kind="groupA", mode=mode,
                                          dtype=dtype, bytes=nbytes,
                                          warps=warps, grid=grid, waves=waves,
                                          u=u, sessions=si,
                                          ns=d["diff_med_ns"],
                                          q1=d.get("diff_q1_ns"),
                                          q3=d.get("diff_q3_ns"),
                                          cv=d.get("cv"),
                                          ns_per_prog=d["diff_med_ns"] / waves)
                        print("done %s %s b%d w%d g%d" %
                              (mode, dtype, nbytes, warps, grid), flush=True)


if __name__ == "__main__":
    main()
