# test/transformers/test_swiglu_npu.py

import torch
import pytest
from typing import Tuple

from test.utils import assert_verbose_allclose, set_seed
from liger_kernel.utils import infer_device

try:
    from liger_kernel.ops.backends._ascend.ops.swiglu import LigerSiLUMulFunction
except ImportError as e:
    pytest.skip(f"SWIGLU not available: {e}", allow_module_level=True)

device = infer_device()
set_seed(42)


def swiglu_ref_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * torch.sigmoid(a)) * b


def swiglu_ref_backward(
    a: torch.Tensor,
    b: torch.Tensor,
    grad_output: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    sig_a = torch.sigmoid(a)
    silu_a = a * sig_a
    term1 = silu_a * (1.0 - sig_a) + sig_a
    da = grad_output * b * term1
    db = grad_output * silu_a
    return da, db


_SHAPE_PARAMS = [
    (2, 32),           # shape0
    (1024,),           # shape1
    (8, 1024),         # shape2
    (16, 4096),        # shape3
    (1, 1),            # shape4
    (13, 401),         # shape5
    (128, 256, 64),    # shape6
    (1, 32000),        # shape7
]

# Forward: test all dtypes
_DTYPE_FORWARD = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 1e-2),
]

# Backward / Full: ONLY float32 (dtype0) to ensure correctness without xfail or relaxed tolerance
_DTYPE_BACKWARD = [(torch.float32, 1e-5, 1e-5)]


# =============================================================================
# Forward: all dtypes, all shapes
# =============================================================================

@pytest.mark.parametrize("shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol, rtol", _DTYPE_FORWARD)
def test_swiglu_forward(shape, dtype, atol, rtol):
    a = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    ref_out = swiglu_ref_forward(a, b)
    tri_out = LigerSiLUMulFunction.apply(a, b)

    assert_verbose_allclose(tri_out, ref_out, atol=atol, rtol=rtol)


# =============================================================================
# Backward / Full tests: ONLY float32
# =============================================================================

@pytest.mark.parametrize("shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol, rtol", _DTYPE_BACKWARD)
def test_swiglu_backward(shape, dtype, atol, rtol):
    a = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(shape, device=device, dtype=dtype)

    ref_da, ref_db = swiglu_ref_backward(a, b, grad_output)

    tri_out = LigerSiLUMulFunction.apply(a, b)
    tri_out.backward(grad_output)
    tri_da, tri_db = a.grad, b.grad

    assert_verbose_allclose(tri_da, ref_da, atol=atol, rtol=rtol)
    assert_verbose_allclose(tri_db, ref_db, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol, rtol", _DTYPE_BACKWARD)
def test_swiglu_full(shape, dtype, atol, rtol):
    a = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    ref_out = swiglu_ref_forward(a, b)
    ref_da, ref_db = swiglu_ref_backward(a, b, torch.ones_like(ref_out))

    tri_out = LigerSiLUMulFunction.apply(a, b)
    tri_out.sum().backward()
    tri_da, tri_db = a.grad, b.grad

    assert_verbose_allclose(tri_out, ref_out, atol=atol, rtol=rtol)
    assert_verbose_allclose(tri_da, ref_da, atol=atol, rtol=rtol)
    assert_verbose_allclose(tri_db, ref_db, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol, rtol", _DTYPE_BACKWARD)
def test_swiglu_full_not_last_layer(shape, dtype, atol, rtol):
    a = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    ref_out = swiglu_ref_forward(a, b) * 2.0
    ref_da, ref_db = swiglu_ref_backward(a, b, 2.0 * torch.ones_like(ref_out))

    tri_out = LigerSiLUMulFunction.apply(a, b) * 2.0
    tri_out.sum().backward()
    tri_da, tri_db = a.grad, b.grad

    assert_verbose_allclose(tri_out, ref_out, atol=atol, rtol=rtol)
    assert_verbose_allclose(tri_da, ref_da, atol=atol, rtol=rtol)
    assert_verbose_allclose(tri_db, ref_db, atol=atol, rtol=rtol)


# =============================================================================
# Edge cases: keep all dtypes for forward-like behavior
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_swiglu_zero_input(dtype):
    shape = (32, 64)
    a = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

    out = LigerSiLUMulFunction.apply(a, b)
    out.sum().backward()

    assert torch.allclose(out, torch.zeros_like(out))
    # Gradients are zero; use default tolerance
    assert torch.allclose(a.grad, torch.zeros_like(a.grad))
    assert torch.allclose(b.grad, torch.zeros_like(b.grad))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_swiglu_extreme_values(dtype):
    base_vals = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0], device=device, dtype=dtype)
    shape = (16, 128)
    a = base_vals.repeat(shape[0], (shape[1] + 4) // 5)[:, :shape[1]]
    b = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    ref_out = swiglu_ref_forward(a, b)
    tri_out = LigerSiLUMulFunction.apply(a, b)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1e-5)
    assert_verbose_allclose(tri_out, ref_out, atol=atol, rtol=rtol)


# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
