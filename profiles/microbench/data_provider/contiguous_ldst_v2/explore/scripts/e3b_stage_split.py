"""E3b: how does the StagePartitioner group multiple same-shape loads?

Compile-only.  The SIMD/SIMT cost model only runs in the AUTO route
(compile_mode="simd_simt"), which the timing probes deliberately avoid
(they force simd / simt_only).  This script recompiles the E3 straight-line
U-load kernels under auto mode with route dump enabled, then summarizes the
stage list: how many stages, which cost model each got, and the simd/f1 cycle
count - i.e. whether U loads land in ONE stage (one closed-form group) or U
separate stages (U closed-form evaluations summed serially).

Usage:
  bash run_exp.sh e3b_stage_split.py 3 e3b_c3.jsonl
"""
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import JsonlOut, current_card

BLOCK = 1024
US = [1, 2, 3, 4, 6, 8]


@triton.jit
def e3_sl(src, dst, U: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for j in tl.static_range(U):
        acc += tl.load(src + j * BLOCK + lane)
    tl.store(dst + lane, acc)


@triton.jit
def e3_loop(src, dst, iters, U: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * U * BLOCK
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        row = base + (step & 63) * U * BLOCK
        for j in tl.static_range(U):
            acc += tl.load(src + row + j * BLOCK + lane)
    tl.store(dst + pid * BLOCK + lane, acc)


def summarize(route_path):
    data = json.loads(Path(route_path).read_text(encoding="utf-8"))
    sm = data.get("stage_model") or {}
    rows = []
    for st in sm.get("logical_stages", []):
        best = None
        for impl in st.get("implementations") or []:
            i = impl.get("implementation") or {}
            if i.get("mode") == "simd" and i.get("superblock_factor", 1) == 1:
                best = impl
        rows.append({
            "id": st.get("id"),
            "model": st.get("model"),
            "iterations": st.get("iteration_count"),
            "simd_f1_cycles": best.get("total_system_cycles") if best else None,
            "evidence": best.get("formula_evidence") if best else None,
        })
    decision = {k: data.get(k) for k in
                ("effective_decision_kind", "effective_runtime_factor")}
    return rows, decision


def main():
    out = JsonlOut("/root/work/explore_contig/results/e3b_c3.jsonl")
    dump_dir = Path("/root/work/explore_contig/results")
    src = torch.randn(BLOCK * 8 * 1024, device="npu")
    dst = torch.zeros(BLOCK * 2, device="npu")
    opts = dict(compile_mode="simd_simt", num_warps=4)

    for tag, kernel, extra in (("sl", e3_sl, ()), ("loop", e3_loop, (2048,))):
        for U in US:
            dump = dump_dir / ("e3b_%s_u%d.json" % (tag, U))
            env = {"TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP": str(dump),
                   "TRITON_ASCEND_AUTO_SIMT_SCOPE": "auto",
                   "TRITON_ASCEND_COSTMODEL_DETAILED_REPORT": "1"}
            prev = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                if tag == "sl":
                    e3_sl[(1,)](src, dst, U, BLOCK, **opts)
                else:
                    e3_loop[(1,)](src, dst, 2048, U, BLOCK, **opts)
            finally:
                for k, v in prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            if not dump.exists():
                print("no dump for", tag, U, flush=True)
                continue
            rows, decision = summarize(dump)
            out.write(exp="e3b", tag=tag, U=U, card=current_card(),
                      stages=rows, **decision)
            print("== %s U=%d  route=%s factor=%s" % (
                tag, U, decision.get("effective_decision_kind"),
                decision.get("effective_runtime_factor")), flush=True)
            for r in rows:
                print("   %-38s %-26s iters=%-4s simd_f1=%-8s %s" % (
                    r["id"], r["model"], r["iterations"], r["simd_f1_cycles"],
                    r["evidence"]), flush=True)
    print("E3B DONE", flush=True)


if __name__ == "__main__":
    main()
