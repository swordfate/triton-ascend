# test/kernels/test_element_mul_kernel.py

import pytest
import torch
import triton

from liger_kernel.utils import infer_device
from test.utils import assert_verbose_allclose, set_seed, supports_bfloat16

# Import your kernel function (adjust path as needed)
from liger_kernel.ops.utils import element_mul_kernel  # ← 替换为你的实际模块路径


device = infer_device()
set_seed(42)


def torch_element_mul_reference(X: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation: X * grad_output (broadcast scalar over last dim)
    Assumes grad_output is a scalar tensor (0-dim or 1-element).
    """
    if grad_output.numel() != 1:
        raise ValueError("grad_output must be a scalar")
    return X * grad_output.item()


def _test_element_mul_once(
    B: int,
    T: int,
    H: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    # Input tensor: [B * T, H]
    X = torch.randn(B * T, H, device=device, dtype=dtype, requires_grad=False)
    grad_output = torch.tensor([2.5], device=device, dtype=torch.float32)  # scalar gradient

    # Clone for reference
    X_ref = X.clone()
    X_triton = X.clone()

    # Reference output
    ref_out = torch_element_mul_reference(X_ref, grad_output)

    # Triton kernel launch
    n_cols = H
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grid = lambda meta: (B * T,)
    element_mul_kernel[grid](
        X_ptr=X_triton,
        X_stride=X_triton.stride(0),
        grad_output_ptr=grad_output,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Compare
    assert_verbose_allclose(ref_out, X_triton, atol=atol, rtol=rtol)


# Helper: copy from your utils or inline
def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"n={n} exceeds max block size {MAX_FUSED_SIZE}")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# Test parameters
_SHAPE_PARAMS = [
    (1, 1, 1),       # minimal
    (2, 1024, 1),
    (3, 3, 1024),
    (7, 31, 63),
    (1, 1, 8192),   # large H
]

_DTYPE_PARAMS = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
    pytest.param(
        torch.bfloat16,
        1e-2,
        1e-2,
        marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
    ),
]


@pytest.mark.parametrize("B,T,H", _SHAPE_PARAMS)
@pytest.mark.parametrize("dtype,atol,rtol", _DTYPE_PARAMS)
def test_element_mul_kernel_correctness(B, T, H, dtype, atol, rtol):
    if "cuda" in device and dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bfloat16 requires Ampere+ GPU")
    _test_element_mul_once(B, T, H, dtype, atol, rtol)


# Edge case: single element
def test_element_mul_single_element():
    X = torch.tensor([3.0], device=device, dtype=torch.float32)
    grad_output = torch.tensor([2.0], device=device, dtype=torch.float32)

    ref = X * 2.0

    element_mul_kernel[(1,)](
        X_ptr=X,
        X_stride=1,
        grad_output_ptr=grad_output,
        n_cols=1,
        BLOCK_SIZE=1,
        num_warps=1,
    )

    assert torch.allclose(ref, X, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
