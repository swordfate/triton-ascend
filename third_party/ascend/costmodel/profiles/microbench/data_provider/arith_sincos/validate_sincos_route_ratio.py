#!/usr/bin/env python3
"""Compare cost-model SIMD/SIMT predicted ratio with measured Triton runtime ratio.

Run on Ascend server:
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=1
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_CACHE_DIR=$(pwd)/cache/sincos_route_validation
    python3 validate_sincos_route_ratio.py
"""
import json
import os
import pathlib
import statistics

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def unary_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr, OP: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    if OP == 0:
        y = x + 1.0
    elif OP == 1:
        y = tl.sin(x)
    else:
        y = tl.cos(x)
    tl.store(y_ptr + offs, y, mask=mask)


def _options(mode, grid, report_path=None):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": grid,
        }
    if mode == "simt_only":
        return {
            "num_warps": 1,
            "compile_mode": "simt_only",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": True,
            "superblock_factor": 1,
            "logical_program_count_hint": grid,
        }
    if mode == "auto":
        opts = {
            "num_warps": 1,
            "compile_mode": "simd_simt",
            "auto_simt_scope_mode": "auto",
            "auto_simt_scope_dump": str(report_path),
            "enable_auto_blockify": True,
            "logical_program_count_hint": grid,
        }
        if report_path.exists():
            report_path.unlink()
        return opts
    raise ValueError(mode)


def _median_ms(launch, reps=15):
    for _ in range(5):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _report_path(name: str) -> pathlib.Path:
    env_path = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "").strip()
    if not env_path:
        default_dir = pathlib.Path("/tmp/sincos_route_validation")
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"{name}_route.json"

    p = pathlib.Path(env_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        # Keep the user-specified JSON path but do not overwrite different cases.
        return p.with_name(f"{p.stem}_{name}{p.suffix}")
    # If the env value is a directory, write per-case files inside it.
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}_route.json"


def _run_case(name, opcode, n, block, report_path):
    grid = (n // block,)
    x = torch.rand(n, dtype=torch.float32, device="npu")
    y = torch.zeros(n, dtype=torch.float32, device="npu")
    measured = {}

    # Explicit SIMD and SIMT measurements.
    for mode in ("simd", "simt_only"):
        opts = _options(mode, grid[0])

        def launch(mode=mode, opts=opts):
            y.zero_()
            unary_kernel[grid](x, y, N=n, BLOCK=block, OP=opcode, **opts)

        measured[mode] = _median_ms(launch)

    # Auto cost-model report.
    opts = _options("auto", grid[0], report_path)
    y.zero_()
    unary_kernel[grid](x, y, N=n, BLOCK=block, OP=opcode, **opts)
    torch.npu.synchronize()
    if not report_path.exists():
        raise RuntimeError(f"cost model did not write {report_path}")
    lines = report_path.read_text().strip().splitlines()
    report = json.loads(lines[-1])
    routes = report["stage_model"]["routes"]
    simd_score = routes["all_simd"]["total_system_cycles"]
    simt_score = routes["all_simt_only"]["total_system_cycles"]
    predicted_ratio = simd_score / simt_score
    measured_ratio = measured["simd"] / measured["simt_only"]
    return {
        "case": name,
        "n": n,
        "simd_ms": measured["simd"],
        "simt_ms": measured["simt_only"],
        "measured_simd_over_simt": measured_ratio,
        "all_simd_score": simd_score,
        "all_simt_score": simt_score,
        "predicted_simd_over_simt": predicted_ratio,
        "effective": report.get("effective_decision_kind"),
        "unmodeled": report.get("unmodeled_cost_terms"),
    }


def main():
    rows = []
    for name, opcode in [("add", 0), ("sin", 1), ("cos", 2)]:
        for n in (4096, 16384):
            rows.append(_run_case(name, opcode, n, 1024, _report_path(name)))

    print()
    print("case,n,simd_ms,simt_ms,measured_ratio,all_simd_score,all_simt_score,predicted_ratio,effective,unmodeled")
    for r in rows:
        print(
            f"{r['case']},{r['n']},{r['simd_ms']:.6f},{r['simt_ms']:.6f},"
            f"{r['measured_simd_over_simt']:.4f},{r['all_simd_score']:.2f},"
            f"{r['all_simt_score']:.2f},{r['predicted_simd_over_simt']:.4f},"
            f"{r['effective']},{r['unmodeled']}"
        )


if __name__ == "__main__":
    main()
