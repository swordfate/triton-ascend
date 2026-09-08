#!/usr/bin/env python3
"""Run a simple Triton transpose through the auto cost model.

Used to inspect where StagePartitioner puts tt.trans.
Set COSTMODEL_LOG_LEVEL=1 before running to see stage-boundary logs.
"""
import json
import os
import pathlib

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


def main():
    M = N = 64
    BM = BN = 32
    grid = (M // BM, N // BN)
    x = torch.rand(M, N, dtype=torch.float32, device="npu")
    y = torch.empty(N, M, dtype=torch.float32, device="npu")
    dump_env = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "")
    if dump_env:
        report = pathlib.Path(dump_env).expanduser()
    else:
        report = pathlib.Path("/tmp/transpose_partition_route.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        report.unlink()
    scope_env = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE", "auto")
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": scope_env,
        "auto_simt_scope_dump": str(report),
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid[0] * grid[1],
    }
    trans_kernel[grid](x, y, M=M, N=N, BM=BM, BN=BN, **opts)
    torch.npu.synchronize()
    if report.exists():
        data = json.loads(report.read_text().strip().splitlines()[-1])
        print("effective:", data.get("effective_decision_kind"))
        print("unmodeled:", data.get("unmodeled_cost_terms"))
        print("report:", report)


if __name__ == "__main__":
    main()
