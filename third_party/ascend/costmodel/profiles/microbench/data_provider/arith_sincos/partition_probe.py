#!/usr/bin/env python3
"""Run a simple Triton sin/cos kernel through the auto cost model.

Use this to inspect where StagePartitioner puts f32.sin / f32.cos.
It runs only ONE kernel, so it is cheaper than validate_sincos_route_ratio.py.

Examples:
    python3 partition_probe.py sin
    python3 partition_probe.py cos

If TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP is set, the cost-model JSON is written
to that path (for .json) or to that directory (for a directory path).
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


def _report_path() -> pathlib.Path:
    env_path = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP", "").strip()
    if not env_path:
        return pathlib.Path("/tmp/sincos_partition_route.json")
    p = pathlib.Path(env_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix != ".json":
        p.mkdir(parents=True, exist_ok=True)
        p = p / "partition_route.json"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["sin", "cos"], nargs="?", default="sin")
    args = ap.parse_args()

    N = 4096
    BLOCK = 1024
    grid = (N // BLOCK,)
    opcode = 0 if args.op == "sin" else 1

    x = torch.rand(N, dtype=torch.float32, device="npu")
    y = torch.empty(N, dtype=torch.float32, device="npu")

    report = _report_path()
    if report.exists():
        report.unlink()
    scope_env = os.environ.get("TRITON_ASCEND_AUTO_SIMT_SCOPE", "auto")
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": scope_env,
        "auto_simt_scope_dump": str(report),
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid[0],
    }
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
