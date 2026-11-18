"""E3: multiple same-shape tl.loads in one kernel - concurrency, dependency,
and stage splitting (user Q3).

Part A (timing, NPU): U in {1,2,3,4,6,8} same-shape 4 KiB fp32 loads.
  sl_ind   - U independent straight-line loads, sum consumed by one store.
             Q: does T(U) grow ~Ux (fully serialized) or sub-linear (loads
             overlap through the LSU/MTE queues)?
  sl_chain - same loads, but the USES are chained: c = x0 * x1 * ... * x_{U-1}.
             The load *addresses* stay independent; only the consumption is
             sequential.  Comparing sl_chain vs sl_ind separates "load issue
             overlap" from "consumer dependency".
  st_group - U independent straight-line stores (same question, write side).
  Control: a single 4 KiB store kernel (the loads' incremental cost is
  diff-vs-control, calibration style).  grid=1 (single-shot context) and,
  for sl_ind, also grid=56 (multi-program overlap).

Part B (compile-only, stage split): the same sl_ind kernels are compiled with
  TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP + DETAILED_REPORT; route.json records
  how the StagePartitioner grouped the U loads - one stage holding all U
  accesses, or one stage per load.  Results go to results/e3_stage_<u>.json.

Usage:
  bash run_exp.sh e3_multiload.py <card> e3_c<card>.jsonl --modes simd simt_only
"""
import argparse
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    pair_diff_ns, prof_session, stats_us, std_opts)

BLOCK = 1024  # 4 KiB fp32
US = [1, 2, 3, 4, 6, 8]


# ---------------- Part A kernels ------------------------------------------
@triton.jit
def e3_sl(src, dst, U: tl.constexpr, CHAIN: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    x = tl.load(src + lane)
    if CHAIN:
        c = x
        for j in tl.static_range(1, U):
            c = c * tl.load(src + j * BLOCK + lane)
        tl.store(dst + lane, c)
    else:
        acc = x
        for j in tl.static_range(1, U):
            acc += tl.load(src + j * BLOCK + lane)
        tl.store(dst + lane, acc)


@triton.jit
def e3_st(dst, U: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    v = tl.full((BLOCK,), 3, dst.dtype.element_ty)
    for j in tl.static_range(U):
        tl.store(dst + j * BLOCK + lane, v)


@triton.jit
def e3_ctl(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


def measure_point(tag, fns, names, pairs, out, extra):
    rows = prof_session(fns, names, pairs)
    rec = {"exp": "e3", "tag": tag, **extra}
    for nm, fl in zip(names, ["probe", "ctl"]):
        rec[fl] = stats_us(rows[nm])
    d = pair_diff_ns(rows[names[0]], rows[names[1]])
    if d:
        rec["diff"] = d
    out.write(**rec)
    print(tag, rec.get("probe"), rec.get("diff"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default="simd simt_only", nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stage-dump", default="results")
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    out = JsonlOut(args.out)
    src = torch.randn(BLOCK * 8 * 1024, device="npu")   # 32 MB fp32
    dstbig = torch.zeros(BLOCK * 8 * 1024, device="npu")
    dst = torch.zeros(BLOCK * 2, device="npu")

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        # ---- Part B first: stage split (compile-only, dump env) --------
        dump_dir = Path(args.stage_dump)
        dump_dir.mkdir(parents=True, exist_ok=True)
        for U in US:
            env = dict(os.environ)
            env["TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP"] = str(dump_dir / ("e3_stage_u%d_%s.json" % (U, mode)))
            env["TRITON_ASCEND_COSTMODEL_DETAILED_REPORT"] = "1"
            prev = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                e3_sl[(1,)](src, dst, U, 0, BLOCK, **opts)
            finally:
                for k, v in prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            p = dump_dir / ("e3_stage_u%d_%s.json" % (U, mode))
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                stages = []
                sm = data.get("stage_model") or {}
                for st in sm.get("logical_stages", []):
                    impls = st.get("implementations") or []
                    cyc = next((i.get("total_system_cycles") for i in impls
                                if (i.get("implementation") or {}).get("mode") == "simd"
                                and (i.get("implementation") or {}).get("superblock_factor", 1) == 1), None)
                    stages.append({"id": st.get("id"), "model": st.get("model"),
                                   "simd_cycles": cyc})
                (dump_dir / ("e3_stagesum_u%d_%s.json" % (U, mode))).write_text(
                    json.dumps(stages, indent=1), encoding="utf-8")
                print("stages U=%d %s: %s" % (U, mode,
                      [(s["id"], s["model"], s["simd_cycles"]) for s in stages]),
                      flush=True)

        # ---- Part A: timing -------------------------------------------
        for U in US:
            measure_point(
                "sl_ind_g1_u%d" % U,
                [lambda U=U: e3_sl[(1,)](src, dst, U, 0, BLOCK, **opts),
                 lambda: e3_ctl[(1,)](dst, BLOCK, **opts)],
                ["e3_sl", "e3_ctl"], args.pairs, out,
                {"mode": mode, "kind": "load", "grid": 1, "U": U, "chain": False})
            measure_point(
                "sl_chain_g1_u%d" % U,
                [lambda U=U: e3_sl[(1,)](src, dst, U, 1, BLOCK, **opts),
                 lambda: e3_ctl[(1,)](dst, BLOCK, **opts)],
                ["e3_sl", "e3_ctl"], args.pairs, out,
                {"mode": mode, "kind": "load", "grid": 1, "U": U, "chain": True})
            measure_point(
                "st_g1_u%d" % U,
                [lambda U=U: e3_st[(1,)](dstbig, U, BLOCK, **opts),
                 lambda: e3_ctl[(1,)](dst, BLOCK, **opts)],
                ["e3_st", "e3_ctl"], args.pairs, out,
                {"mode": mode, "kind": "store", "grid": 1, "U": U})
        for U in (1, 2, 4, 8):
            measure_point(
                "sl_ind_g56_u%d" % U,
                [lambda U=U: e3_sl[(56,)](src, dst, U, 0, BLOCK, **opts),
                 lambda: e3_ctl[(1,)](dst, BLOCK, **opts)],
                ["e3_sl", "e3_ctl"], args.pairs, out,
                {"mode": mode, "kind": "load", "grid": 56, "U": U})
    print("E3 DONE", flush=True)


if __name__ == "__main__":
    main()
