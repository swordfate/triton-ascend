# coding=utf-8
# -------------------------------------------------
# File Name： test__cumsum
# Source: FlagGems src/flag_gems/ops/cumsum.py
#   reduce_then_scan_root_scan_kernel_row (persistent path, N <= 16384)
#   shape (4, 2048) float32, dim=-1 → grid (4, 1, 1), TILE_SIZE=2048, num_warps=4
# -------------------------------------------------
import sys
import pytest
import torch

import triton
import triton.language as tl
import torch_npu
from typing import Optional

sys.path.append("..")
from test_common import convert_tensor_with_device_type, profiling_test, validate_cmp, cal_precision, compare_data_precision


@tl.constexpr
def get_scan_accum_type(inp_dtype: tl.dtype) -> tl.dtype:
    if inp_dtype.is_bf16() or inp_dtype.is_fp16():
        return tl.float32
    if inp_dtype.is_int():  # signed or not(including bool)
        return tl.int64
    else:
        return inp_dtype


@triton.jit
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    """Almost The same kernel as the persistent scan kernel"""
    pid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=0).to(acc_dtype)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


def fn_triton(data, input_data):
    # 调用 kernel
    reduce_then_scan_root_scan_kernel_row[data["grid"]](**input_data)
    return


@pytest.mark.functiontest
def test_get_last_loc_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    reduce_then_scan_root_scan_kernel_row[data["grid"]](**input_data)
    # profiling_test(fn_triton, (data, input_data))

    try:
        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
        print("pass")
    except ValueError as e:
        pytest.fail(f"The testcase failed")


@pytest.mark.perftest
def test_perf(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    profiling_test(fn_triton, (data, input_data))


if __name__ == '__main__':
    test_get_last_loc_kernel('_cumsum_v2.pt')
