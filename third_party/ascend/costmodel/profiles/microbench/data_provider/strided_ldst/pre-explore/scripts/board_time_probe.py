#!/usr/bin/env python3
"""Measure the strided-load probe on the real Ascend board.

This is the real-card counterpart of ``strided_load_probe.py`` (which is run
under ``msopprof simulator``).  It runs the exact same Triton kernel with the
same mode/stride/BLOCK launch options, profiles it with torch_npu profiler,
and stores the parsed ``kernel_details.csv`` metrics.

Reported metrics per case:
  - Duration(us)         : end-to-end kernel duration seen by the profiler
  - aiv_total_cycles     : AIV total cycles reported by torch_npu
  - aiv_vec_time(us)     : AIV vector-pipe active time
  - aiv_scalar_time(us)  : AIV scalar-pipe active time
  - aiv_mte2_time(us)    : AIV MTE2 active time (SIMD GM->UB path)
  - aiv_mte3_time(us)    : AIV MTE3 active time (scalar store / UB->GM)

Only one logical program (grid=1) is launched by default so it can be compared
with the single-program CAModel dumps.
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
import triton.runtime.driver as triton_driver


@triton.jit
def strided_load_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK) * STRIDE
    v = tl.load(x_ptr + pid * BLOCK * STRIDE + offs)
    tl.store(out_ptr + pid, tl.sum(v, axis=0))


@triton.jit
def empty_kernel(out_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, 1.0)


def launch_options(mode, grid, num_warps, ncore):
    common = {
        "num_warps": num_warps,
        "superblock_factor": 1,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": ncore,
        "auto_simt_scope_mode": "off",
    }
    if mode == "simd":
        return {**common, "compile_mode": "simd", "enable_auto_blockify": False}
    if mode == "simt_only":
        return {**common, "compile_mode": "simt_only", "enable_auto_blockify": True}
    raise ValueError(mode)


def n_vectorcore(device):
    try:
        props = triton_driver.active.utils.get_device_properties(device)
        return int(props.get("num_vectorcore", 0)) or 1
    except Exception:
        return 1


def run_profile(launch, tag, root, reps, warmup):
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()

    out_dir = root / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=cfg,
    ):
        for _ in range(reps):
            launch()
        torch.npu.synchronize()

    csvs = sorted(out_dir.rglob("kernel_details.csv"))
    if not csvs:
        raise RuntimeError(f"no kernel_details.csv under {out_dir}")
    rows = []
    with csvs[0].open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows, csvs[0]


def numeric_column(rows, column):
    values = []
    for row in rows:
        value = row.get(column)
        if value in (None, "", "N/A"):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    if not values:
        return None
    values.sort()
    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "max": values[-1],
    }


def profile_case(launch, tag, root, reps, warmup, kernel_name):
    rows, csv_path = run_profile(launch, tag, root, reps, warmup)
    target = [r for r in rows if kernel_name in (r.get("Name") or "")]
    if not target:
        raise RuntimeError(f"kernel {kernel_name} not found in {csv_path}; names={[r.get('Name') for r in rows][:10]}")
    columns = [
        "Duration(us)",
        "aiv_total_cycles",
        "aiv_time(us)",
        "aiv_vec_time(us)",
        "aiv_scalar_time(us)",
        "aiv_mte2_time(us)",
        "aiv_mte3_time(us)",
        "Block Num",
    ]
    result = {"csv": str(csv_path), "rows": len(target)}
    for column in columns:
        result[column] = numeric_column(target, column)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--blocks", nargs="+", type=int, default=[32])
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--num-warps", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("board_time_results"))
    ap.add_argument("--trace-root", type=Path, default=Path("board_time_traces"))
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    ncore = n_vectorcore(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    args.trace_root.mkdir(parents=True, exist_ok=True)

    results = []
    for block in args.blocks:
        for mode in args.modes:
            opts = launch_options(mode, args.grid, args.num_warps, ncore)
            for stride in args.strides:
                tag = f"{mode}_b{block}_s{stride}"
                total = args.grid * block * max(1, stride) + 4096
                x = (torch.arange(total, dtype=torch.float32) % 7.0).contiguous().to("npu")
                out = torch.zeros(args.grid, dtype=torch.float32).to("npu")
                # Compile before profiling so only launches are inside the trace.
                strided_load_kernel.warmup(
                    x, out, BLOCK=block, STRIDE=stride, grid=(args.grid,), **opts
                )

                def launch():
                    strided_load_kernel[(args.grid,)](
                        x, out, BLOCK=block, STRIDE=stride, **opts
                    )

                metrics = profile_case(launch, tag, args.trace_root, args.reps,
                                       args.warmup, "strided_load_kernel")
                row = {
                    "mode": mode,
                    "stride": stride,
                    "block": block,
                    "grid": args.grid,
                    "num_warps": args.num_warps,
                    "device": args.device,
                    "n_vectorcore": ncore,
                    "tag": tag,
                }
                row.update(metrics)
                results.append(row)
                print(f"DONE {tag}: Duration min/med={metrics['Duration(us)']['min']:.3f}/"
                      f"{metrics['Duration(us)']['median']:.3f} us, "
                      f"aiv_total_cycles min/med={metrics['aiv_total_cycles']['min']:.0f}/"
                      f"{metrics['aiv_total_cycles']['median']:.0f}", flush=True)

    payload = {
        "probe": "board_time_probe",
        "device": args.device,
        "n_vectorcore": ncore,
        "grid": args.grid,
        "num_warps": args.num_warps,
        "reps": args.reps,
        "warmup": args.warmup,
        "cases": results,
    }
    (args.out / "board_time_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {args.out / 'board_time_results.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
