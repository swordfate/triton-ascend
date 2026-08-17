"""Built-in Triton kernels used by run_triton_benchmark.py.

This file must be imported after the caller has set the TRITON_ASCEND_*
environment variables, because importing triton before that can cache the
wrong compile-route options.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_silu_mul_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x * tl.sigmoid(x) * y
    tl.store(out_ptr + offs, z, mask=mask)


@triton.jit
def rowwise_reduce_masked_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    R,
    C,
    BLOCK_C: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C
    offs = row * C + cols
    m = tl.load(mask_ptr + offs, mask=c_mask, other=0).to(tl.int1)
    x = tl.load(x_ptr + offs, mask=c_mask, other=0.0)
    x = tl.where(m, x, NEG_INF)
    row_max = tl.max(x, axis=0)
    tl.store(out_ptr + row, row_max)


@triton.jit
def indirect_elementwise_kernel(
    src_ptr, idx_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    val = tl.load(src_ptr + idx, mask=mask, other=0.0)
    out = val * 2.0 + 1.0
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def block_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    acc = tl.dot(a, b)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def single_block_cumsum_kernel(
    in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, y, mask=mask)
