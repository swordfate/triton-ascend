# test/transformers/test_tvd_npu.py

import torch
import pytest
from typing import Optional, Literal

from test.utils import assert_verbose_allclose, set_seed
from liger_kernel.utils import infer_device

try:
    from liger_kernel.ops.backends._ascend.ops.tvd import LigerTVDLossFunction
except ImportError as e:
    pytest.skip(f"TVD not available: {e}", allow_module_level=True)

device = infer_device()
set_seed(42)

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

def tvd_ref_forward(
    p: torch.Tensor,
    q: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    reduction: REDUCTION_LITERAL = "batchmean",
    ignore_index: int = -100,
) -> torch.Tensor:
    assert p.shape == q.shape and p.ndim == 2, f"Expected 2D tensors, got {p.shape}, {q.shape}"
    loss_per_elem = 0.5 * torch.abs(p - q)  # (BT, V)

    if shift_labels is not None:
        assert shift_labels.shape == (p.shape[0],)
        valid_mask = (shift_labels != ignore_index).unsqueeze(-1)  # (BT, 1)
        loss_per_elem = torch.where(valid_mask, loss_per_elem, torch.zeros_like(loss_per_elem))
        n_valid = valid_mask.sum().item()
    else:
        n_valid = p.shape[0]

    if n_valid == 0:
        scalar_zero = torch.tensor(0.0, device=p.device, dtype=loss_per_elem.dtype)
        return scalar_zero if reduction != "none" else torch.zeros_like(loss_per_elem)

    if reduction == "none":
        return loss_per_elem
    elif reduction == "sum":
        return loss_per_elem.sum()
    elif reduction == "mean":
        return loss_per_elem.sum() / (n_valid * p.shape[-1])
    elif reduction == "batchmean":
        return loss_per_elem.sum() / n_valid
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# ONLY 2D SHAPES
_SHAPE_PARAMS = [
    (2, 32),        # shape0
    (1024,),        # shape1 → will be reshaped to (1024, 1)
    (16, 1024),     # shape2
    (1, 32000),     # shape3
    (13, 401),      # shape4
]

def _make_2d(shape):
    if len(shape) == 1:
        return (shape[0], 1)
    elif len(shape) == 2:
        return shape
    else:
        raise ValueError(f"Unsupported shape: {shape}")

_DTYPE_FORWARD = [torch.float32, torch.float16, torch.bfloat16]
_REDUCTION_MODES = ["none", "sum", "mean", "batchmean"]


@pytest.mark.parametrize("shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("dtype", _DTYPE_FORWARD)
@pytest.mark.parametrize("reduction", _REDUCTION_MODES)
@pytest.mark.parametrize("has_label", [True, False])
def test_tvd_forward(shape, dtype, reduction, has_label):
    BT, V = _make_2d(shape)
    p = torch.randn(BT, V, device=device, dtype=dtype)
    q = torch.randn(BT, V, device=device, dtype=dtype)

    shift_labels = None
    if has_label:
        shift_labels = torch.randint(0, 10, (BT,), device=device)
        shift_labels[0] = 0  # ensure at least one valid token

    ref_out = tvd_ref_forward(p, q, shift_labels, reduction, ignore_index=-100)
    tri_out = LigerTVDLossFunction.apply(p, q, shift_labels, reduction, -100)

    # Set tolerances based on dtype and reduction
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        if reduction == "none":
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 5e-3, 1e-2
    else:  # bfloat16
        if reduction == "none":
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-1, 1e-1

    assert_verbose_allclose(tri_out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", _SHAPE_PARAMS)
@pytest.mark.parametrize("reduction", _REDUCTION_MODES)
@pytest.mark.parametrize("has_label", [True, False])
def test_tvd_backward(shape, reduction, has_label):
    BT, V = _make_2d(shape)
    p = torch.randn(BT, V, device=device, dtype=torch.float32, requires_grad=True)
    q = torch.randn(BT, V, device=device, dtype=torch.float32, requires_grad=False)

    shift_labels = None
    if has_label:
        shift_labels = torch.randint(0, 10, (BT,), device=device)
        shift_labels[0] = 0

    # Compute reference gradient
    grad_p = 0.5 * torch.sign(p - q)
    if shift_labels is not None:
        valid_mask = (shift_labels != -100).unsqueeze(-1)
        grad_p = torch.where(valid_mask, grad_p, torch.zeros_like(grad_p))
        n_valid = valid_mask.sum().item()
    else:
        n_valid = BT

    if n_valid == 0:
        ref_grad = torch.zeros_like(grad_p)
    else:
        if reduction == "none":
            ref_grad = grad_p
        elif reduction == "sum":
            ref_grad = grad_p
        elif reduction == "mean":
            ref_grad = grad_p / (n_valid * V)
        elif reduction == "batchmean":
            ref_grad = grad_p / n_valid

    tri_out = LigerTVDLossFunction.apply(p, q, shift_labels, reduction, -100)
    tri_out.sum().backward()
    tri_grad = p.grad

    assert_verbose_allclose(tri_grad, ref_grad, atol=1e-5, rtol=1e-5)


def test_tvd_all_ignored_safe():
    p = torch.randn(4, 16, device=device, dtype=torch.float32, requires_grad=True)
    q = torch.randn(4, 16, device=device, dtype=torch.float32, requires_grad=False)
    labels = torch.full((4,), -100, device=device)
    labels[0] = 0  # one valid

    out = LigerTVDLossFunction.apply(p, q, labels, "batchmean", -100)
    out.sum().backward()

    assert not torch.isnan(out)
    assert not torch.isnan(p.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
