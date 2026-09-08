import json

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(
    not is_compile_on_910_95(),
    reason="SIMD/SIMT cost model only supports 910_95",
)


@triton.jit
def sin_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.sin(x)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def cos_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.cos(x)
    tl.store(y_ptr + offs, y, mask=mask)


def _run_route(kernel, N, BLOCK, tmp_path, tag):
    grid = (N // BLOCK,)
    x = torch.rand(N, dtype=torch.float32, device="npu")
    y = torch.zeros(N, dtype=torch.float32, device="npu")
    report_path = tmp_path / f"{tag}_route.json"
    opts = {
        "num_warps": 1,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
        "logical_program_count_hint": grid[0],
    }
    kernel[grid](x, y, N=N, BLOCK=BLOCK, **opts)
    torch.npu.synchronize()
    return x, y, report_path


def _assert_report(report_path, expected_op):
    assert report_path.exists(), f"missing route report {report_path}"
    lines = report_path.read_text().strip().splitlines()
    report = json.loads(lines[-1])
    assert report["effective_decision_kind"] in ("all_simd", "all_simt_only")
    assert report.get("unmodeled_cost_terms") == []
    found = False
    for stage in report["stage_model"]["logical_stages"]:
        ops = stage.get("workload", {}).get("operation_elements_per_iteration", {})
        if ops.get(expected_op, 0) > 0:
            found = True
    assert found, f"{expected_op} not present in any stage workload"
    return report


def test_sin_costmodel_route(tmp_path):
    N, BLOCK = 4096, 1024
    x, y, report_path = _run_route(sin_kernel, N, BLOCK, tmp_path, "sin")
    torch.testing.assert_close(y, torch.sin(x), rtol=1e-3, atol=1e-3)
    _assert_report(report_path, "f32.sin")


def test_cos_costmodel_route(tmp_path):
    N, BLOCK = 4096, 1024
    x, y, report_path = _run_route(cos_kernel, N, BLOCK, tmp_path, "cos")
    torch.testing.assert_close(y, torch.cos(x), rtol=1e-3, atol=1e-3)
    _assert_report(report_path, "f32.cos")
