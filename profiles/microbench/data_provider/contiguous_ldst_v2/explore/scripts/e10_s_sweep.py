"""E10: S-sweep - how does the number of DEPENDENT stores modify the cost of
U=8 independent loads?  (Reviewer Q1: real kernels have S between 1 and U.)

Construction (U=8 loads fixed, S = number of early per-load stores):
  probe_S  x_j = load(src + j*BLOCK + lane)          for j in 0..7
           store(dst + j*BLOCK + lane, x_j + 1)      for j in 0..S-1
           t = x_S; t = t + x_{S+1}; ... ; t += x_7  (remaining loads)
           store(dst + S*BLOCK + lane, t + 1)        (single final store)
  ctl_S    identical stores/sum structure on tl.full constants
  -> diff = exactly the 8 loads (all stores and adds cancel).

S=8 reduces to the E3c ind shape (linear); S=0 to a pure load group + 1 sum
store (sub-linear per E3e v1).  The sweep gives the interpolation law
T(S) between the two endpoints at fixed U=8.

Usage:
  bash run_exp.sh e10_s_sweep.py 2 e10_c2.jsonl --modes simt_only simd
"""
import argparse
import importlib.util
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024
U = 8
SS = (0, 1, 2, 4, 8)

HEADER = "import triton\nimport triton.language as tl\n\nBLOCK = 1024\n\n"


def gen_module(path):
    parts = [HEADER]
    for s in SS:
        loads = "\n".join(
            "    x%d = tl.load(src + %d * BLOCK + lane)" % (j, j)
            for j in range(U))
        early = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, x%d + 1)" % (j, j)
            for j in range(s))
        if s < U:
            final_loads = "x%d" % s
            for j in range(s + 1, U):
                final_loads = "(%s + x%d)" % (final_loads, j)
            final = "    tl.store(dst + %d * BLOCK + lane, %s + 1)" % (s, final_loads)
        else:
            final = ""
        parts.append(
            "@triton.jit\ndef e10_probe_s%d(src, dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n%s\n%s\n\n"
            % (s, loads, early, final))
        consts = "\n".join(
            "    c%d = tl.full((BLOCK,), %d, dst.dtype.element_ty)"
            % (j, j + 1) for j in range(U))
        early_c = "\n".join(
            "    tl.store(dst + %d * BLOCK + lane, c%d + 1)" % (j, j)
            for j in range(s))
        if s < U:
            final_c = "c%d" % s
            for j in range(s + 1, U):
                final_c = "(%s + c%d)" % (final_c, j)
            final_c = "    tl.store(dst + %d * BLOCK + lane, %s + 1)" % (s, final_c)
        else:
            final_c = ""
        parts.append(
            "@triton.jit\ndef e10_ctl_s%d(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n%s\n%s\n\n"
            % (s, consts, early_c, final_c))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_e10", path)
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
    ap.add_argument("--modes", default=["simt_only", "simd"], nargs="+")
    ap.add_argument("--dtypes", default=["fp32"], nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=3)
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    gen_path = str(Path(args.out).parent / "_gen_e10.py")
    gen_module(gen_path)
    mod = load_module(gen_path)

    DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
          "int64": torch.int64}
    for dtype in args.dtypes:
        if dtype == "fp32":
            src = torch.randn(BLOCK * 64, device="npu")
        elif dtype == "fp16":
            src = torch.randn(BLOCK * 64, device="npu").half()
        elif dtype == "int8":
            src = torch.randint(-120, 120, (BLOCK * 64,), dtype=torch.int8,
                                device="npu")
        else:
            src = torch.randint(-1000, 1000, (BLOCK * 64,), dtype=torch.int64,
                                device="npu")
        dst = torch.zeros(BLOCK * 64, device="npu").to(DT[dtype])

        for mode in args.modes:
            opts = std_opts(mode, args.warps)
            for s in SS:
                probe = getattr(mod, "e10_probe_s%d" % s)
                ctl = getattr(mod, "e10_ctl_s%d" % s)
                measure("sweep_%s_%s_U%d_S%d" % (mode, dtype, U, s),
                        [lambda p=probe, s_=src, d_=dst: p[(1,)](s_, d_,
                                                                 BLOCK, **opts),
                         lambda c=ctl, d_=dst: c[(1,)](d_, BLOCK, **opts)],
                        ["e10_probe_s%d" % s, "e10_ctl_s%d" % s],
                        args.pairs, args.sessions, out,
                        {"exp": "e10", "mode": mode, "dtype": dtype,
                         "U": U, "S": s})
    print("E10 DONE", flush=True)


if __name__ == "__main__":
    main()
