"""cal_b_loop.py v2 - Formula B calibration, ONE profiler session per cell
covering all K points (constexpr-K named kernels; E11 verified constexpr
`range` keeps scf.for in the final CCE, so semantics match runtime loops).

  body=chain : acc += load(streaming); one final store
  body=indpt : x = load(...); store(... + x + 1) per round
  body=store : store(..., const) per round

Controls: chain/indpt -> one 4KB store; store -> tiny scalar store.
Residency: warm fixed region / cold >=2MiB-stride rotation over >320MB.
One prof_pairs_session per (mode, dtype, bytes, residency); 3 sessions;
per-K min of session diffs (cv<=0.4).

Usage:
  bash run_exp.sh cal_b_loop.py <card> calB_warm_chain_simd.jsonl \
      --body chain --modes simd --dtypes int8 fp16 fp32 int64 \
      --bytes-list 1024 4096 16384 --ks 2 4 8 16 32 64 128
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


def gen_module(path, body, dtype_key, ks):
    eb = EB[dtype_key]
    parts = [HEADER]
    for k in ks:
        # Runtime iters + distinct names per K.  Constexpr K gets fully
        # unrolled by the pure-simt hivmc pipeline (compile explosion at
        # K>=64 for large blocks); E11 semantics (loop stays a loop) are
        # preserved because do_not_specialize keeps bounds opaque.
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
        else:
            body_code = (
                "    for step in range(iters):\n"
                "        tl.store(dst + boff + step * BLOCK + lane, "
                "tl.full((BLOCK,), 1, dst.dtype.element_ty))\n")
            sig = "(dst, iters, boff, BLOCK: tl.constexpr)"
        parts.append(
            "@triton.jit(do_not_specialize=['iters', 'boff'])\n"
            "def cb_%s_k%d%s:\n"
            "    lane = tl.arange(0, BLOCK)\n%s\n\n" % (body, k, sig, body_code))
    parts.append(
        "@triton.jit\n"
        "def cb_ctl_1store(dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    tl.store(dst + lane, tl.full((BLOCK,), 1, dst.dtype.element_ty))\n")
    parts.append(
        "@triton.jit\n"
        "def cb_ctl_tiny(dst):\n"
        "    tl.store(dst, 0)\n")
    # simt state reference (golden ~3.95us, E8 cross-card agreement)
    parts.append(
        "@triton.jit\n"
        "def cb_ref8(src, dst, BLOCK: tl.constexpr):\n"
        "    lane = tl.arange(0, BLOCK)\n"
        "    for j in tl.static_range(8):\n"
        "        x = tl.load(src + j * BLOCK + lane)\n"
        "        tl.store(dst + j * BLOCK + lane, x + 1)\n")
    # NOTE: call_args is informational; callers build lambdas explicitly.
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_cal_b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--body", choices=("chain", "indpt", "store"),
                    default="chain")
    ap.add_argument("--modes", default=["simd", "simt_only"], nargs="+")
    ap.add_argument("--dtypes", default=["fp32"], nargs="+")
    ap.add_argument("--bytes-list", default=[4096], nargs="+", type=int)
    ap.add_argument("--ks", default=[2, 4, 8, 16, 32, 64, 128],
                    nargs="+", type=int)
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
    n = (320 + 64) * 1024 * 1024 // 4
    src = torch.randn(n, device="npu")
    dst = torch.zeros(n, device="npu")
    region_elems = (320 * 1024 * 1024) // 4

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
                bad = [r for r in residency_list
                       if (r, dtype, str(nbytes), args.body) not in wanted]
                if len(bad) == len(residency_list):
                    continue
            block = max(1, nbytes // EB[dtype])
            stride_elems = max(block * max(args.ks),
                               (2 * 1024 * 1024) // EB[dtype])
            gen_path = str(Path(args.out).parent /
                           ("_gen_cal_b_%s_%s_%d.py" % (args.body, dtype,
                                                        nbytes)))
            gen_module(gen_path, args.body, dtype, args.ks)
            mod = load_module(gen_path)
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

                    spec = []
                    for k in args.ks:
                        pname = "cb_%s_k%d" % (args.body, k)
                        cname = ("cb_ctl_tiny" if args.body == "store"
                                 else "cb_ctl_1store")
                        probe_fn = getattr(mod, pname)
                        ctl_fn = getattr(mod, cname)
                        if args.body == "store":
                            spec.append((
                                lambda pf=probe_fn, k=k, b=boff(): pf[(1,)](
                                    dst, k, b, block, **opts),
                                pname,
                                lambda c=ctl_fn: c[(1,)](dst, **opts),
                                cname))
                        else:
                            spec.append((
                                lambda pf=probe_fn, k=k, b=boff(): pf[(1,)](
                                    src, dst, k, b, block, **opts),
                                pname,
                                lambda c=ctl_fn, bl=block: c[(1,)](
                                    dst, bl, **opts),
                                cname))
                    if mode == "simt_only":
                        spec.append((
                            lambda: getattr(mod, "cb_ref8")[(
                                1,)](src, dst, block, **opts),
                            "cb_ref8",
                            lambda: getattr(mod, "cb_ref8")[(
                                1,)](src, dst, block, **opts),
                            "cb_ref8_x"))
                    rows_all = []
                    ref_meds = []
                    try:
                        for _ in range(args.sessions):
                            rows = prof_pairs_session(spec, args.pairs)
                            rows_all.append(rows)
                            if mode == "simt_only" and rows.get("cb_ref8"):
                                ref_meds.append(statistics.median(
                                    rows["cb_ref8"]))
                    except Exception as exc:   # compile explosion etc.
                        for k in args.ks:
                            out.write(tag="calB_%s_%s_%s_%s_b%d_k%d" % (
                                          mode, dtype, residency, args.body,
                                          nbytes, k),
                                      status="cell-fail",
                                      error=str(exc)[:200], cal="B",
                                      body=args.body, mode=mode, dtype=dtype,
                                      bytes=nbytes, residency=residency, K=k)
                        print("CELL-FAIL %s %s b%d: %s" % (
                            mode, residency, nbytes, str(exc)[:120]),
                            flush=True)
                        continue
                    golden = 3950.0
                    ref_state = (statistics.median(ref_meds) * 1000.0
                                 / golden) if ref_meds else None
                    for k in args.ks:
                        pname = "cb_%s_k%d" % (args.body, k)
                        cname = ("cb_ctl_tiny" if args.body == "store"
                                 else "cb_ctl_1store")
                        diffs = []
                        for rows in rows_all:
                            d = pair_diff_ns(rows[pname], rows[cname])
                            if d and d["cv"] <= 0.4:
                                diffs.append(d["diff_med_ns"])
                        best = min(diffs) if diffs else None
                        norm = (round(best / ref_state, 1)
                                if best and ref_state else None)
                        out.write(tag="calB_%s_%s_%s_%s_b%d_k%d" % (
                                      mode, dtype, residency, args.body,
                                      nbytes, k),
                                  diff_med_ns_min=best, diff_med_ns_norm=norm,
                                  ref_state=round(ref_state, 3)
                                  if ref_state else None,
                                  sessions=[{"diff": d} for d in diffs],
                                  cal="B", body=args.body, mode=mode,
                                  dtype=dtype, bytes=nbytes,
                                  residency=residency, K=k)
                        print("calB_%s_%s_%s_%s_b%d_k%d min=%s norm=%s" % (
                            mode, dtype, residency, args.body, nbytes, k,
                            best, norm), flush=True)
    print("CAL-B DONE", flush=True)


if __name__ == "__main__":
    main()
