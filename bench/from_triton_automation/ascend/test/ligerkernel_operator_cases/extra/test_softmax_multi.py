# tests/test_softmax_multiblock_correctness.py

import torch
import pytest

from test.utils import assert_verbose_allclose, set_seed
from liger_kernel.utils import infer_device

# Skip if LigerSoftmax is not available
try:
    from liger_kernel.ops.softmax import LigerSoftmaxFunction
except ImportError as e:
    pytest.skip(f"LigerSoftmaxFunction not available: {e}", allow_module_level=True)

device = infer_device()
set_seed(42)


# ==============================================================================
# 🔑 CRITICAL: Patch the calculate_settings used INSIDE softmax.py
# This ensures n_cols > 1024 triggers multi-block.
# ==============================================================================
import liger_kernel.ops.softmax as softmax_module

def _patched_calculate_settings(n_cols: int):
    """
    Force BLOCK_SIZE = 1024 so that any n_cols > 1024 uses multi-block kernel.
    """
    BLOCK_SIZE = 1024
    num_warps = 4
    if n_cols > 2048:
        num_warps = 8
    if n_cols > 8192:
        num_warps = 16
    return BLOCK_SIZE, num_warps

# Apply patch to the exact function used in softmax.py
softmax_module.calculate_settings = _patched_calculate_settings


def _ref_softmax(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return torch.nn.functional.softmax(x, dim=-1)


# ==============================================================================
# 🧪 Forward correctness tests (multi-block regime)
# ==============================================================================
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2048),      # minimal multi-block
        (2, 4096),
        (1, 8192),
        (4, 12288),
        (1, 16384),
        (2, 32768),     # large vocab
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_softmax_multiblock_forward(shape, dtype):
    if dtype == torch.bfloat16 and device == "cpu":
        pytest.skip("bfloat16 not fully supported on CPU")

    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=False)
    ref_out = _ref_softmax(x.detach().clone())
    tri_out = LigerSoftmaxFunction.apply(x)

    # Set tolerances based on dtype
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2

    assert_verbose_allclose(tri_out, ref_out, atol=atol, rtol=rtol)


# ==============================================================================
# 🧪 Backward (gradient) correctness tests
# ==============================================================================
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2048),
        (2, 4096),
        (1, 8192),
        (4, 12288),
        (1, 16384),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_softmax_multiblock_backward(shape, dtype):
    if dtype == torch.bfloat16 and device == "cpu":
        pytest.skip("bfloat16 not fully supported on CPU")

    # Create identical inputs for both paths
    x_tri = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    x_ref = x_tri.detach().clone().requires_grad_(True)

    grad_output = torch.randn_like(x_tri)

    # Triton path
    out_tri = LigerSoftmaxFunction.apply(x_tri)
    out_tri.backward(grad_output)
    grad_tri = x_tri.grad

    # Reference path
    out_ref = _ref_softmax(x_ref)
    out_ref.backward(grad_output)
    grad_ref = x_ref.grad

    # Set tolerances
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2

    assert_verbose_allclose(grad_tri, grad_ref, atol=atol, rtol=rtol)


# ==============================================================================
# 🧪 Extreme value stability test
# ==============================================================================
def test_softmax_multiblock_extreme_values():
    shape = (2, 4096)
    # Create tensor with large values to test numerical stability
    x_base = torch.full(shape, 1e6, device=device, dtype=torch.float32)
    x_base[0, 0] = 1e7  # one outlier

    x_tri = x_base.clone().detach().requires_grad_(True)
    x_ref = x_base.clone().detach().requires_grad_(True)

    grad_output = torch.randn_like(x_tri)

    # Triton
    out_tri = LigerSoftmaxFunction.apply(x_tri)
    out_tri.backward(grad_output)
    grad_tri = x_tri.grad

    # Reference
    out_ref = _ref_softmax(x_ref)
    out_ref.backward(grad_output)
    grad_ref = x_ref.grad

    # Check both output and gradient
    assert_verbose_allclose(out_tri, out_ref, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(grad_tri, grad_ref, atol=1e-5, rtol=1e-5)


# ==============================================================================
# 🧪 Optional: Verify multi-block was actually used (via internal logic)
# ==============================================================================
def test_softmax_uses_multi_block_flag():
    """
    Since we patched BLOCK_SIZE=1024, n_cols=2048 must use multi-block.
    We verify by checking the internal return flag from _softmax_forward.
    """
    from liger_kernel.ops.softmax import _softmax_forward

    x = torch.randn(1, 2048, device=device, dtype=torch.float16)
    _, _, _, multi_block_launch = _softmax_forward(x)
    assert multi_block_launch, "Expected multi-block path for n_cols=2048"


# ==============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
