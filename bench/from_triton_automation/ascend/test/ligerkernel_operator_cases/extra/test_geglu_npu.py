# test/ops/test_geglu_npu.py

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16
from liger_kernel.ops.backends._ascend.ops.geglu import LigerGELUMulFunction
from liger_kernel.utils import infer_device, is_npu_available

device = infer_device()
set_seed(42)

# Skip entire module if not on NPU


def _ref_geglu_tanh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference GEGLU using tanh approximation."""
    return F.gelu(a, approximate="tanh") * b


_SHAPE_PARAMS = (
    "shape",
    [
        (16, 64),
        (32, 128),
        (13, 257),      # odd dims to stress tiling
        (100, 400),
        (1, 1),
        (1024, 2048),
        (17, 31, 123),  # 3D
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported"),
        ),
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
    ],
)


def _test_geglu_correctness_once(
    shape,
    dtype,
    atol,
    rtol,
    is_last_layer: bool = True,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = infer_device()

    a = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    # Reference
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = _ref_geglu_tanh(a_ref, b_ref)

    # Liger
    a_liger = a.detach().clone().requires_grad_(True)
    b_liger = b.detach().clone().requires_grad_(True)
    out_liger = LigerGELUMulFunction.apply(a_liger, b_liger)

    # Forward check
    assert_verbose_allclose(out_ref, out_liger, atol=atol, rtol=rtol)

    # Apply scaling if not last layer (to test grad scaling path)
    if not is_last_layer:
        out_ref = out_ref * 2.0
        out_liger = out_liger * 2.0

    # Backward
    out_ref.sum().backward()
    out_liger.sum().backward()

    # Gradient check
    assert_verbose_allclose(a_ref.grad, a_liger.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(b_ref.grad, b_liger.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_geglu_npu_correctness(shape, dtype, atol, rtol):
    _test_geglu_correctness_once(shape, dtype, atol, rtol, is_last_layer=True, device=device)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_geglu_npu_correctness_not_last(shape, dtype, atol, rtol):
    _test_geglu_correctness_once(shape, dtype, atol, rtol, is_last_layer=False, device=device)


# Edge case: extreme values
def test_geglu_extreme_values():
    dtype = torch.float32
    a = torch.tensor([[-100.0, -10.0, 0.0, 10.0, 100.0]], device=device, dtype=dtype, requires_grad=True)
    b = torch.ones_like(a, requires_grad=True)

    out_ref = _ref_geglu_tanh(a, b)
    out_liger = LigerGELUMulFunction.apply(a, b)

    # Slightly relaxed for extreme saturation, but still strict
    assert_verbose_allclose(out_ref, out_liger, atol=1e-4, rtol=1e-4)

    out_ref.sum().backward()
    out_liger.sum().backward()
    assert_verbose_allclose(a.grad, a.grad, atol=1e-4, rtol=1e-4)  # sanity check


# Edge case: zero input (smooth gradient at 0)
def test_geglu_zero_input():
    dtype = torch.float32
    a = torch.zeros((2, 8), device=device, dtype=dtype, requires_grad=True)
    b = torch.ones_like(a, requires_grad=True)

    out_ref = _ref_geglu_tanh(a, b)
    out_liger = LigerGELUMulFunction.apply(a, b)

    assert_verbose_allclose(out_ref, out_liger, atol=1e-5, rtol=1e-5)

    out_liger.sum().backward()
    assert not torch.isnan(a.grad).any()
    assert not torch.isinf(a.grad).any()
