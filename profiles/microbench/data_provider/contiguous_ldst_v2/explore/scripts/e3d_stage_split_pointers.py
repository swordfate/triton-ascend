"""E3d: WHY does vector_add split its two loads into two stages while the E3b
kernel keeps U loads in ONE stage?  (Reviewer Q2c - affects pricing.)

Hypothesis from reading StageBoundaryAnalysis::analyze
(Analysis/StagePartitioner.cpp:3081):
  * every top-level op is its own "semantic root";
  * adjacent roots merge into one stage only via mergePlainStage, which
    requires BOTH roots to be OUTSIDE any SIMT anchor group
    (anchorGroup < 0) and same kind/schedule;
  * under AutoBlockify V1 each tt.load becomes its own anchored root; two
    loads on two different tensors belong to DIFFERENT anchor groups, so
    sameCompoundAnchor=false and mergePlainStage's anchorGroup<0 guard also
    fails -> break -> one stage per load.

Variants (compile-only, auto route, stage dump):
  v0  vector_add verbatim (pid offsets + mask + two pointers, grid=97,
      n=98432, harness-style kwargs incl. logical_program_count_hint)
      -> expect the v8-style split (baseline to reproduce);
  v1  E3b-style: SAME pointer, two 4 KiB blocks, no mask, no pid (grid=1)
      -> earlier observed single memory stage;
  v2  TWO different pointers, no mask, no pid (grid=1)  -> merge or split?
  v3  v2 WITHOUT the harness kwargs (no program hints)   -> hint effect?
  v4  v0 masks removed (two pointers, pid offsets, no mask, grid=97)
      -> mask/pid effect under hints?

Usage:
  bash run_exp.sh e3d_stage_split_pointers.py 3 e3d_c3.jsonl
"""
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import JsonlOut, current_card, std_opts

BLOCK = 1024
DUMP = Path("/root/work/explore_contig/results")


@triton.jit
def vec_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def two_ptr_nomask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + lane)
    y = tl.load(y_ptr + lane)
    tl.store(out_ptr + lane, x + y)


@triton.jit
def same_ptr_two_blocks(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    x = tl.load(src + lane)
    y = tl.load(src + BLOCK + lane)
    tl.store(dst + lane, x + y)


def compile_with_dump(name, launch, tag):
    dump = DUMP / ("e3d_%s.json" % tag)
    env = {"TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP": str(dump),
           "TRITON_ASCEND_AUTO_SIMT_SCOPE": "auto",
           "TRITON_ASCEND_COSTMODEL_DETAILED_REPORT": "1"}
    prev = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        launch()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if not dump.exists():
        print("no dump for", tag, flush=True)
        return None
    data = json.loads(dump.read_text(encoding="utf-8"))
    sm = data.get("stage_model") or {}
    rows = []
    for st in sm.get("logical_stages", []):
        best = None
        for impl in st.get("implementations") or []:
            i = impl.get("implementation") or {}
            if i.get("mode") == "simd" and i.get("superblock_factor", 1) == 1:
                best = impl
        rows.append({"id": st.get("id"), "model": st.get("model"),
                     "cycles": best.get("total_system_cycles") if best else None,
                     "evidence": best.get("formula_evidence") if best else None})
    decision = {k: data.get(k) for k in
                ("effective_decision_kind", "effective_runtime_factor")}
    return rows, decision


def main():
    out = JsonlOut("/root/work/explore_contig/results/e3d_c3.jsonl")
    n = 98432
    x = torch.randn(n, device="npu")
    y = torch.randn(n, device="npu")
    o = torch.zeros(n, device="npu")
    src = torch.randn(BLOCK * 8, device="npu")
    dstb = torch.zeros(BLOCK * 8, device="npu")
    hint_kwargs = dict(logical_program_count_hint=97,
                       physical_vector_core_count_hint=56)

    cases = [
        ("v0_vec_add_hints",
         lambda: vec_add[(97,)](x, y, o, n, 1024,
                                compile_mode="simd_simt", num_warps=4,
                                compile_on_910_95=True, **hint_kwargs)),
        ("v1_same_ptr",
         lambda: same_ptr_two_blocks[(1,)](src, dstb, BLOCK,
                                           compile_mode="simd_simt", num_warps=4,
                                           compile_on_910_95=True,
                                           logical_program_count_hint=1)),
        ("v2_two_ptr",
         lambda: two_ptr_nomask[(1,)](x, y, o, BLOCK,
                                      compile_mode="simd_simt", num_warps=4,
                                      compile_on_910_95=True,
                                      logical_program_count_hint=1)),
        ("v3_two_ptr_nohints",
         lambda: two_ptr_nomask[(1,)](x, y, o, BLOCK,
                                      compile_mode="simd_simt", num_warps=4,
                                      compile_on_910_95=True)),
        ("v4_vec_add_nomask",
         lambda: two_ptr_nomask[(97,)](x, y, o, BLOCK,
                                       compile_mode="simd_simt", num_warps=4,
                                       compile_on_910_95=True, **hint_kwargs)),
    ]
    for tag, launch in cases:
        result = compile_with_dump(tag.split("_")[0], launch, tag)
        if not result:
            continue
        rows, decision = result
        out.write(exp="e3d", tag=tag, card=current_card(), stages=rows,
                  **decision)
        print("== %s route=%s f=%s" % (tag,
                                       decision.get("effective_decision_kind"),
                                       decision.get("effective_runtime_factor")),
              flush=True)
        for r in rows:
            print("   %-40s %-26s %10.1f %s" % (r["id"], r["model"],
                                                r["cycles"] or -1,
                                                r["evidence"]), flush=True)
    print("E3D DONE", flush=True)


if __name__ == "__main__":
    main()
