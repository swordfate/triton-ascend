#!/usr/bin/env python3
"""Board profiler differential for K diff-line scalar loads/stores.

The target kernel takes eight independent pointer tensors and executes exactly
K unrolled operations of the form ``tl.load(p_i + pid)`` / ``tl.store(p_i +
pid, val)`` (K=1..8, no for loop).  A structurally matched arithmetic baseline
runs in the same profiler session; we report the min-per-name delta of
Duration / aiv_total_cycles / aiv_time.
"""
import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["off"])
def load_k(p0, p1, p2, p3, p4, p5, p6, p7, out_ptr, off, K: tl.constexpr):
    pid = tl.program_id(0) + off
    acc = 0.0
    if K > 0:
        acc += tl.load(p0 + pid)
    if K > 1:
        acc += tl.load(p1 + pid)
    if K > 2:
        acc += tl.load(p2 + pid)
    if K > 3:
        acc += tl.load(p3 + pid)
    if K > 4:
        acc += tl.load(p4 + pid)
    if K > 5:
        acc += tl.load(p5 + pid)
    if K > 6:
        acc += tl.load(p6 + pid)
    if K > 7:
        acc += tl.load(p7 + pid)
    tl.store(out_ptr, acc)


@triton.jit(do_not_specialize=["off", "val"])
def store_k(p0, p1, p2, p3, p4, p5, p6, p7, out_ptr, off, val, K: tl.constexpr):
    pid = tl.program_id(0) + off
    if K > 0:
        tl.store(p0 + pid, val)
    if K > 1:
        tl.store(p1 + pid, val)
    if K > 2:
        tl.store(p2 + pid, val)
    if K > 3:
        tl.store(p3 + pid, val)
    if K > 4:
        tl.store(p4 + pid, val)
    if K > 5:
        tl.store(p5 + pid, val)
    if K > 6:
        tl.store(p6 + pid, val)
    if K > 7:
        tl.store(p7 + pid, val)


@triton.jit(do_not_specialize=["off"])
def arith_k(p0, p1, p2, p3, p4, p5, p6, p7, out_ptr, off, K: tl.constexpr):
    pid = tl.program_id(0) + off
    acc = 0.0
    if K > 0:
        acc += pid.to(tl.float32)
    if K > 1:
        acc += (pid + 1).to(tl.float32)
    if K > 2:
        acc += (pid + 2).to(tl.float32)
    if K > 3:
        acc += (pid + 3).to(tl.float32)
    if K > 4:
        acc += (pid + 4).to(tl.float32)
    if K > 5:
        acc += (pid + 5).to(tl.float32)
    if K > 6:
        acc += (pid + 6).to(tl.float32)
    if K > 7:
        acc += (pid + 7).to(tl.float32)
    tl.store(out_ptr, acc)


def opts(mode):
    common = dict(num_warps=1, superblock_factor=1, logical_program_count_hint=1,
                  physical_vector_core_count_hint=1, auto_simt_scope_mode="off")
    if mode == "simd":
        return {**common, "compile_mode": "simd", "enable_auto_blockify": False}
    return {**common, "compile_mode": "simt_only", "enable_auto_blockify": True}


def profile_pair(target, baseline, tag, root, reps, warmup, step):
    for _ in range(warmup):
        target(0)
        baseline(0)
    torch.npu.synchronize()
    out_dir = Path(root) / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cfg = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False, data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
        record_shapes=False, profile_memory=False, with_stack=False,
        experimental_config=cfg,
    ):
        for i in range(reps):
            off = (i + 1) * step
            if i % 2 == 0:
                target(off)
                baseline(off)
            else:
                baseline(off)
                target(off)
        torch.npu.synchronize()
    csvs = sorted(out_dir.rglob("kernel_details.csv"))
    if not csvs:
        raise RuntimeError("no kernel_details.csv")
    rows = list(csv.DictReader(csvs[0].open(newline="")))
    return rows


def min_col(rows, name_substr, col):
    vals = []
    for row in rows:
        if name_substr not in (row.get("Name") or ""):
            continue
        v = row.get(col)
        if v in (None, "", "N/A"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    return min(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--kinds", nargs="+", default=["load", "store"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--step", type=int, default=4096)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("profiler_pair_results"))
    args = ap.parse_args()
    torch_npu.npu.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for mode in args.modes:
        for kind in args.kinds:
            for k in args.ks:
                o = opts(mode)
                span = (args.reps + 2) * args.step + max(args.ks) + 4096
                ptrs = [torch.arange(span, dtype=torch.float32).to("npu")
                        for _ in range(8)]
                out = torch.zeros(1, dtype=torch.float32).to("npu")
                val = 1.0
                if kind == "load":
                    load_k.warmup(*ptrs, out, 0, K=k, grid=(1,), **o)
                    target = lambda off: load_k[(1,)](*ptrs, out, off, K=k, **o)
                    target_name = "load_k"
                else:
                    store_k.warmup(*ptrs, out, 0, val, K=k, grid=(1,), **o)
                    target = lambda off: store_k[(1,)](*ptrs, out, off, val, K=k, **o)
                    target_name = "store_k"
                arith_k.warmup(*ptrs, out, 0, K=k, grid=(1,), **o)
                baseline = lambda off: arith_k[(1,)](*ptrs, out, off, K=k, **o)

                per_round = []
                for rnd in range(args.rounds):
                    rows = profile_pair(target, baseline, f"{mode}_{kind}_k{k}_{rnd}",
                                        args.out, args.reps, args.warmup, args.step)
                    cols = {}
                    for col in ["Duration(us)", "aiv_total_cycles", "aiv_time(us)",
                                "aiv_scalar_time(us)", "aiv_vec_time(us)"]:
                        tv = min_col(rows, target_name, col)
                        bv = min_col(rows, "arith_k", col)
                        cols[col] = (tv, bv, None if tv is None or bv is None else tv - bv)
                    per_round.append(cols)
                rec = {"mode": mode, "kind": kind, "K": k, "num_ptrs": 8}
                for col in per_round[0]:
                    ds = [r[col][2] for r in per_round if r[col][2] is not None]
                    rec[col + "_target"] = statistics.median([r[col][0] for r in per_round])
                    rec[col + "_base"] = statistics.median([r[col][1] for r in per_round])
                    rec[col + "_delta"] = statistics.median(ds) if ds else None
                rows_out.append(rec)
                print(f"{mode:9s} {kind:5s} K={k}: "
                      f"Duration_delta={rec['Duration(us)_delta']:.3f} us, "
                      f"aiv_total_delta={rec['aiv_total_cycles_delta']}, "
                      f"aiv_time_delta={rec['aiv_time(us)_delta']:.3f} us", flush=True)
    payload = {"cases": rows_out, "args": vars(args) | {"out": str(args.out)}}
    (args.out / "profiler_pair_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with (args.out / "profiler_pair_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())
