"""cal_a_group.py v2 - Formula A calibration, ONE profiler session per cell
covering all N points (named-per-point kernels; ~10x faster than v1).

  direction=load : U independent loads -> ONE sum store (S=1); ctl does the
                   same sum/store on constants.  diff = exactly U loads.
  direction=store: U streaming stores; ctl = one store.  U-sweep slope gives
                   first/marginal/c_transfer.

Per (mode, dtype, bytes, residency): one prof_pairs_session covering all U,
3 sessions, per-point min of session diffs (cv<=0.4).

Usage:
  bash run_exp.sh cal_a_group.py <card> calA_warm_ld_simd.jsonl \
      --direction load --modes simd --dtypes int8 fp16 fp32 int64 \
      --bytes-list 1024 4096 16384 --us 1 4 8 16 32
"""
import argparse
import importlib.util
import statistics
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_pairs_session, stats_us, std_opts)

DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
      "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}

HEADER = "import triton\nimport triton.language as tl\n\n"


def gen_module(path, direction, us, dtype_key):
    eb = EB[dtype_key]
    parts = [HEADER]
    if direction == "load":
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
                "def ca_ld_u%d(src, dst, boff, BLOCK: tl.constexpr):\n"
                "    lane = tl.arange(0, BLOCK)\n%s\n%s\n\n" % (u, loads, tail))
            parts.append(
                "@triton.jit(do_not_specialize=['boff'])\n"
                "def ca_ld_ctl_u%d(dst, boff, BLOCK: tl.constexpr):\n"
                "    lane = tl.arange(0, BLOCK)\n%s\n%s\n\n"
                % (u, consts, ctl_tail))
    else:
        for u in us:
            stores = "\n".join(
                "    tl.store(dst + boff + %d * BLOCK + lane, "
                "tl.full((BLOCK,), 1, dst.dtype.element_ty))" % j
                for j in range(u))
            parts.append(
                "@triton.jit(do_not_specialize=['boff'])\n"
                "def ca_st_u%d(dst, boff, BLOCK: tl.constexpr):\n"
                "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (u, stores))
        parts.append(
            "@triton.jit\n"
            "def ca_st_ctl1(dst, BLOCK: tl.constexpr):\n"
            "    lane = tl.arange(0, BLOCK)\n"
            "    tl.store(dst + lane, tl.full((BLOCK,), 1, "
            "dst.dtype.element_ty))\n")
    # simt state reference: straight-line 8 x 4KB load+store pairs (golden
    # ~3.95us, E8 cross-card agreement).  Used when mode==simt_only.
    parts.append(
        "@triton.jit\n"
        "def ca_ref8(src, dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for j in tl.static_range(8):\n"
        "        x = tl.load(src + j * BLOCK + lane)\n"
        "        tl.store(dst + j * BLOCK + lane, x + 1)\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_cal_a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--direction", choices=("load", "store"), default="load")
    ap.add_argument("--modes", default=["simd", "simt_only"], nargs="+")
    ap.add_argument("--dtypes", default=["fp32"], nargs="+")
    ap.add_argument("--bytes-list", default=[4096], nargs="+", type=int)
    ap.add_argument("--us", default=[1, 4, 8, 16, 32], nargs="+", type=int)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--cells-file", default="", help="mode,dtype,bytes,residency lines to re-measure")
    ap.add_argument("--residency", default="all",
                    choices=("warm", "cold", "all"))
    args = ap.parse_args()
    residency_list = (("warm", "cold") if args.residency == "all"
                      else (args.residency,))
    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    # UB budget: a U-deep load chain keeps U result blocks live; the backend
    # refuses >~256KB of UB, so cap U accordingly (load direction only —
    # stores do not hold live results).
    ub_budget = 128 * 1024
    # optional targeted re-measure: --cells-file has lines
    # mode,dtype,bytes,residency ; only those cells are run.
    wanted = None
    if args.cells_file:
        wanted = set()
        for line in open(args.cells_file, encoding="utf-8"):
            parts_c = [x.strip() for x in line.split(",") if x.strip()]
            if len(parts_c) == 4:
                wanted.add(tuple(parts_c))
    for dtype in args.dtypes:
        for nbytes in args.bytes_list:
            if wanted is not None:
                skip_dir = [r for r in residency_list
                            if (r, dtype, str(nbytes), args.direction)
                            not in wanted]
                if len(skip_dir) == len(residency_list):
                    continue
            us = [u for u in args.us
                  if args.direction == "store" or u * nbytes <= ub_budget]
            if not us:
                print("skip %s b%d: no U fits UB budget" % (dtype, nbytes),
                      flush=True)
                continue
            block = max(1, nbytes // EB[dtype])
            stride_elems = max(block * max(us),
                               (2 * 1024 * 1024) // EB[dtype])
            gen_path = str(Path(args.out).parent /
                           ("_gen_cal_a_%s_%d.py" % (dtype, nbytes)))
            gen_module(gen_path, args.direction, us, dtype)
            mod = load_module(gen_path)
            n = (320 + 64) * 1024 * 1024 // EB[dtype]
            src = torch.randn(n, device="npu")
            dst = torch.zeros(n, device="npu")
            region_elems = (320 * 1024 * 1024) // EB[dtype]
            for mode in args.modes:
                opts = std_opts(mode, args.warps)
                for residency in residency_list:
                    nreg = (1 if residency == "warm" else
                            max(2, region_elems // stride_elems))
                    ir = [0]

                    def boff():
                        if nreg <= 1:
                            return 0
                        v = (ir[0] % nreg) * stride_elems
                        ir[0] += 1
                        return v

                    if args.direction == "load":
                        spec = []
                        for u in us:
                            spec.append((
                                lambda u=u, b=boff(): getattr(
                                    mod, "ca_ld_u%d" % u)[(1,)](
                                    src, dst, b, block, **opts),
                                "ca_ld_u%d" % u,
                                lambda u=u, b=boff(): getattr(
                                    mod, "ca_ld_ctl_u%d" % u)[(1,)](
                                    dst, b, block, **opts),
                                "ca_ld_ctl_u%d" % u))
                    else:
                        spec = []
                        for u in us:
                            spec.append((
                                lambda u=u, b=boff(): getattr(
                                    mod, "ca_st_u%d" % u)[(1,)](
                                    dst, b, block, **opts),
                                "ca_st_u%d" % u,
                                lambda: getattr(mod, "ca_st_ctl1")[(
                                    1,)](dst, block, **opts),
                                "ca_st_ctl1"))
                    if mode == "simt_only":
                        spec.append((
                            lambda: getattr(mod, "ca_ref8")[(
                                1,)](src, dst, block, **opts),
                            "ca_ref8",
                            lambda: getattr(mod, "ca_ref8")[(
                                1,)](src, dst, block, **opts),
                            "ca_ref8_x"))
                    rows_all = []
                    ref_meds = []
                    try:
                        for sess in range(args.sessions):
                            rows = prof_pairs_session(spec, args.pairs)
                            rows_all.append(rows)
                            if mode == "simt_only" and rows.get("ca_ref8"):
                                ref_meds.append(statistics.median(
                                    rows["ca_ref8"]))
                    except Exception as exc:   # compile explosion etc.
                        for u in us:
                            out.write(tag="calA_%s_%s_%s_b%d_u%d" % (
                                          mode, dtype, residency, nbytes, u),
                                      status="cell-fail",
                                      error=str(exc)[:200], cal="A",
                                      direction=args.direction, mode=mode,
                                      dtype=dtype, bytes=nbytes,
                                      residency=residency, N=u)
                        print("CELL-FAIL %s %s b%d: %s" % (
                            mode, residency, nbytes, str(exc)[:120]),
                            flush=True)
                        continue
                    golden = 3950.0  # us*1000 -> ns; E8 cross-card agreement
                    ref_state = (statistics.median(ref_meds) * 1000.0
                                 / golden) if ref_meds else None
                    for u in us:
                        pname = spec[[t[1] for t in spec].index(
                            ("ca_ld_u%d" % u) if args.direction == "load"
                            else ("ca_st_u%d" % u))][1]
                        diffs = []
                        for rows in rows_all:
                            pname_p = ("ca_ld_u%d" % u) if \
                                args.direction == "load" else "ca_st_u%d" % u
                            cname = ("ca_ld_ctl_u%d" % u) if \
                                args.direction == "load" else "ca_st_ctl1"
                            d = pair_diff_ns(rows[pname_p], rows[cname])
                            if d and d["cv"] <= 0.4:
                                diffs.append(d["diff_med_ns"])
                        best = min(diffs) if diffs else None
                        norm = (round(best / ref_state, 1)
                                if best and ref_state else None)
                        out.write(tag="calA_%s_%s_%s_b%d_u%d" % (
                                      mode, dtype, residency, nbytes, u),
                                  diff_med_ns_min=best, diff_med_ns_norm=norm,
                                  ref_state=round(ref_state, 3)
                                  if ref_state else None,
                                  sessions=[
                                      {"diff": d} for d in diffs],
                                  cal="A", direction=args.direction,
                                  mode=mode, dtype=dtype, bytes=nbytes,
                                  residency=residency, N=u)
                        print("calA_%s_%s_%s_b%d_u%d min=%s norm=%s" % (
                            mode, dtype, residency, nbytes, u, best, norm),
                            flush=True)
    print("CAL-A DONE", flush=True)


if __name__ == "__main__":
    main()
