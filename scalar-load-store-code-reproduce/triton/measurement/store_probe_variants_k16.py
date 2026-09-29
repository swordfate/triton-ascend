#!/usr/bin/env python3
"""Store timing variants: matched sink baseline, readback, barrier/readback.

Goal: diagnose whether the K=2/K=4 store rows are timing artifacts caused by
an unmatched sink store/async write completion.  All targets end with the same
sink `tl.store(out_ptr, acc)` as their baseline.

Variants:
  0 store_sink    : K stores + matched arithmetic; baseline = _arith_k
  1 store_read    : K stores + K readback loads;      baseline = _load_k
  2 store_read_bar: same as 1 with tl.debug_barrier() between stores and loads
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


@triton.jit
def _arith_k(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, out_ptr, off, K: tl.constexpr):
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
    if K > 8:
        acc += (pid + 8).to(tl.float32)
    if K > 9:
        acc += (pid + 9).to(tl.float32)
    if K > 10:
        acc += (pid + 10).to(tl.float32)
    if K > 11:
        acc += (pid + 11).to(tl.float32)
    if K > 12:
        acc += (pid + 12).to(tl.float32)
    if K > 13:
        acc += (pid + 13).to(tl.float32)
    if K > 14:
        acc += (pid + 14).to(tl.float32)
    if K > 15:
        acc += (pid + 15).to(tl.float32)
    tl.store(out_ptr, acc)


@triton.jit
def _load_k(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, out_ptr, off, K: tl.constexpr):
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
    if K > 8:
        acc += tl.load(p8 + pid)
    if K > 9:
        acc += tl.load(p9 + pid)
    if K > 10:
        acc += tl.load(p10 + pid)
    if K > 11:
        acc += tl.load(p11 + pid)
    if K > 12:
        acc += tl.load(p12 + pid)
    if K > 13:
        acc += tl.load(p13 + pid)
    if K > 14:
        acc += tl.load(p14 + pid)
    if K > 15:
        acc += tl.load(p15 + pid)
    tl.store(out_ptr, acc)


@triton.jit(do_not_specialize=["off", "val"])
def _store_probe(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, out_ptr, off, val,
                 K: tl.constexpr, VARIANT: tl.constexpr):
    pid = tl.program_id(0) + off
    acc = 0.0
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
    if K > 8:
        tl.store(p8 + pid, val)
    if K > 9:
        tl.store(p9 + pid, val)
    if K > 10:
        tl.store(p10 + pid, val)
    if K > 11:
        tl.store(p11 + pid, val)
    if K > 12:
        tl.store(p12 + pid, val)
    if K > 13:
        tl.store(p13 + pid, val)
    if K > 14:
        tl.store(p14 + pid, val)
    if K > 15:
        tl.store(p15 + pid, val)
    if VARIANT == 0:
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
        if K > 8:
            acc += (pid + 8).to(tl.float32)
        if K > 9:
            acc += (pid + 9).to(tl.float32)
        if K > 10:
            acc += (pid + 10).to(tl.float32)
        if K > 11:
            acc += (pid + 11).to(tl.float32)
        if K > 12:
            acc += (pid + 12).to(tl.float32)
        if K > 13:
            acc += (pid + 13).to(tl.float32)
        if K > 14:
            acc += (pid + 14).to(tl.float32)
        if K > 15:
            acc += (pid + 15).to(tl.float32)
    if VARIANT >= 1:
        if VARIANT == 2:
            tl.debug_barrier()
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
        if K > 8:
            acc += tl.load(p8 + pid)
        if K > 9:
            acc += tl.load(p9 + pid)
        if K > 10:
            acc += tl.load(p10 + pid)
        if K > 11:
            acc += tl.load(p11 + pid)
        if K > 12:
            acc += tl.load(p12 + pid)
        if K > 13:
            acc += tl.load(p13 + pid)
        if K > 14:
            acc += tl.load(p14 + pid)
        if K > 15:
            acc += tl.load(p15 + pid)
    tl.store(out_ptr, acc)


def opts(mode):
    common = dict(num_warps=1, superblock_factor=1,
                  logical_program_count_hint=1,
                  physical_vector_core_count_hint=1,
                  auto_simt_scope_mode="off")
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
        l2_cache=False, data_simplification=False)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=cfg):
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
    return list(csv.DictReader(csvs[0].open(newline="")))


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


COLS = ["Duration(us)", "aiv_total_cycles", "aiv_mte3_time(us)",
        "aiv_mte2_time(us)", "aiv_time(us)", "aiv_vec_time(us)",
        "aiv_scalar_time(us)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--variants", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--step", type=int, default=4096)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("store_variant_results"))
    args = ap.parse_args()
    torch_npu.npu.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    out = []
    for mode in args.modes:
        o = opts(mode)
        for k in args.ks:
            span = (args.reps + 2) * args.step + max(args.ks) + 4096
            ptrs = [torch.arange(span, dtype=torch.float32).to("npu")
                    for _ in range(16)]
            out_buf = torch.zeros(1, dtype=torch.float32).to("npu")
            val = 1.0
            for variant in args.variants:
                if variant == 0:
                    base_name = "_arith_k"
                    base = lambda off: _arith_k[(1,)](
                        *ptrs, out_buf, off, K=k, **o)
                else:
                    base_name = "_load_k"
                    base = lambda off: _load_k[(1,)](
                        *ptrs, out_buf, off, K=k, **o)
                _store_probe.warmup(*ptrs, out_buf, 0, val,
                                    K=k, VARIANT=variant, grid=(1,), **o)
                target = lambda off: _store_probe[(1,)](
                    *ptrs, out_buf, off, val, K=k, VARIANT=variant, **o)
                per_round = []
                for rnd in range(args.rounds):
                    tag = f"{mode}_store_k{k}_v{variant}_{rnd}"
                    rows = profile_pair(target, base, tag, args.out, args.reps,
                                        args.warmup, args.step)
                    rec = {}
                    for col in COLS:
                        tv = min_col(rows, "_store_probe", col)
                        bv = min_col(rows, base_name, col)
                        rec[col] = None if tv is None or bv is None else tv - bv
                    per_round.append(rec)
                rec = {"mode": mode, "kind": "store", "K": k,
                       "variant": variant, "rounds": args.rounds,
                       "reps": args.reps}
                for col in COLS:
                    ds = [r[col] for r in per_round if r[col] is not None]
                    rec[col + "_delta"] = statistics.median(ds) if ds else None
                    rec[col + "_min"] = min(ds) if ds else None
                    rec[col + "_max"] = max(ds) if ds else None
                out.append(rec)
                print(f"{mode:9s} v{variant} K={k}: "
                      f"Duration={rec['Duration(us)_delta']} us, "
                      f"mte3={rec['aiv_mte3_time(us)_delta']} us, "
                      f"total={rec['aiv_total_cycles_delta']} cyc", flush=True)
    payload = {"cases": out, "args": vars(args) | {"out": str(args.out)}}
    (args.out / "store_variant_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with (args.out / "store_variant_results.csv").open("w", newline="") as f:
        fields = list(out[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print("WROTE", args.out)


if __name__ == "__main__":
    sys.exit(main())
