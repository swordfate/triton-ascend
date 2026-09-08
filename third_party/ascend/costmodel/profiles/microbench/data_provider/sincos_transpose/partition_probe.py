#!/usr/bin/env python3
"""Run one Triton sin/cos/transpose kernel through the auto cost model.

This is a lightweight single-kernel probe.  Use it to confirm where
StagePartitioner puts f32.sin / f32.cos / f32.trans, and to dump the cost-model
JSON to TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP.

Examples:
    python3 partition_probe.py sin
    python3 partition_probe.py cos
    python3 partition_probe.py trans
"""
import argparse
import json
import os
import pathlib

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
        y = tl.sin(x)
    else:
        y = tl.cos(x)
    tl.store(y_ptr + offs, y, mask=mask)


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


def _report_path() -> pathlib.Path:
    env_path = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "").strip()
    if not env_path:
        return pathlib.Path("/tmp/partition_probe_route.json")
    p = pathlib.Path(env_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix != ".json":
        p.mkdir(parents=True, exist_ok=True)
        p = p / "partition_route.json"
    return p


def _common_opts(report, programs):
    return {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE", "auto"),
        "auto_simt_scope_dump": str(report),
        "enable_auto_blockify": True,
        "logical_program_count_hint": programs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["sin", "cos", "trans"], nargs="?", default="sin")
    args = ap.parse_args()

    report = _report_path()
    if report.exists():
        report.unlink()

    if args.op == "trans":
        M = N = 64
        BM = BN = 32
        grid = (M // BM, N // BN)
        x = torch.rand(M, N, dtype=torch.float32, device="npu")
        y = torch.empty(N, M, dtype=torch.float32, device="npu")
        opts = _common_opts(report, grid[0] * grid[1])
        trans_kernel[grid](x, y, M=M, N=N, BM=BM, BN=BN, **opts)
    else:
        N = 4096
        BLOCK = 1024
        grid = (N // BLOCK,)
        x = torch.rand(N, dtype=torch.float32, device="npu")
        y = torch.empty(N, dtype=torch.float32, device="npu")
        opcode = 0 if args.op == "sin" else 1
        opts = _common_opts(report, grid[0])
        unary_kernel[grid](x, y, N=N, BLOCK=BLOCK, OP=opcode, **opts)

    torch.npu.synchronize()
    if not report.exists():
        raise RuntimeError(f"cost model did not write {report}")

    data = json.loads(report.read_text().strip().splitlines()[-1])
    print("op:", args.op)
    print("effective:", data.get("effective_decision_kind"))
    print("unmodeled:", data.get("unmodeled_cost_terms"))
    print("report:", report)
    for stage in data.get("stage_model", {}).get("logical_stages", []):
        oe = stage.get("workload", {}).get("operation_elements_per_iteration", {})
        if oe:
            print(stage.get("id"), stage.get("model"), oe)


if __name__ == "__main__":
    main()
