#!/usr/bin/env python3
"""比较 transpose 在 cost model 中的 SIMD/SIMT 预测比值与 Triton 实测比值。

在服务器上运行：
    source ~/env_ascend.sh
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=1
    export PYTHONPATH=/home/c00946898/triton-ascend/build/lib.linux-x86_64-cpython-311:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
    export TRITON_BACKENDS_IN_TREE=1
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_CACHE_DIR=$(pwd)/cache/transpose_route_validation
    python3 validate_transpose_route_ratio.py
"""
import json
import pathlib
import statistics

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def trans_kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    y = tl.trans(x)
    tl.store(y_ptr + offs_n[:, None] * M + offs_m[None, :], y)


def _options(mode, grid, report_path=None):
    if mode == "simd":
        return {
            "num_warps": 1,
            "compile_mode": "simd",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": False,
            "superblock_factor": 1,
            "logical_program_count_hint": grid[0] * grid[1],
        }
    if mode == "simt_only":
        return {
            "num_warps": 1,
            "compile_mode": "simt_only",
            "auto_simt_scope_mode": "off",
            "enable_auto_blockify": True,
            "superblock_factor": 1,
            "logical_program_count_hint": grid[0] * grid[1],
        }
    if mode == "auto":
        opts = {
            "num_warps": 1,
            "compile_mode": "simd_simt",
            "auto_simt_scope_mode": "auto",
            "auto_simt_scope_dump": str(report_path),
            "enable_auto_blockify": True,
            "logical_program_count_hint": grid[0] * grid[1],
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


def _run_case(name, m, n, bm, bn, tmpdir):
    grid = (m // bm, n // bn)
    x = torch.rand(m, n, dtype=torch.float32, device="npu")
    y = torch.empty(n, m, dtype=torch.float32, device="npu")
    measured = {}

    for mode in ("simd", "simt_only"):
        opts = _options(mode, grid)

        def launch(mode=mode, opts=opts):
            y.zero_()
            trans_kernel[grid](x, y, M=m, N=n, BM=bm, BN=bn, **opts)

        measured[mode] = _median_ms(launch)

    report_path = tmpdir / f"{name}_route.json"
    opts = _options("auto", grid, report_path)
    y.zero_()
    trans_kernel[grid](x, y, M=m, N=n, BM=bm, BN=bn, **opts)
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
        "m": m,
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
    tmpdir = pathlib.Path("/tmp/transpose_route_validation")
    tmpdir.mkdir(exist_ok=True)
    rows = []
    for m, n in ((64, 64), (128, 128)):
        rows.append(_run_case(f"trans_{m}x{n}", m, n, 32, 32, tmpdir))

    print()
    print("case,m,n,simd_ms,simt_ms,measured_ratio,all_simd_score,all_simt_score,predicted_ratio,effective,unmodeled")
    for r in rows:
        print(
            f"{r['case']},{r['m']},{r['n']},{r['simd_ms']:.6f},{r['simt_ms']:.6f},"
            f"{r['measured_simd_over_simt']:.4f},{r['all_simd_score']:.2f},"
            f"{r['all_simt_score']:.2f},{r['predicted_simd_over_simt']:.4f},"
            f"{r['effective']},{r['unmodeled']}"
        )


if __name__ == "__main__":
    main()
