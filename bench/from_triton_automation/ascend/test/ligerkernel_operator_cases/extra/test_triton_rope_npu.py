# test/transformers/test_triton_rope_npu.py

import torch
import pytest
from typing import Tuple

from test.utils import assert_verbose_allclose, set_seed
from liger_kernel.utils import infer_device

try:
    from liger_kernel.ops.backends._ascend.ops.rope import LigerRopeFunction
except ImportError as e:
    pytest.skip(f"LigerRopeFunction not available: {e}", allow_module_level=True)

device = infer_device()
set_seed(42)


def _ref_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference RoPE implementation.
    Assumes:
        q, k: (B, H, T, D)
        cos, sin: (B_cos, T, D//2)
    """
    B_q = q.shape[0]
    if cos.shape[0] == 1 and B_q > 1:
        cos = cos.expand(B_q, -1, -1)
        sin = sin.expand(B_q, -1, -1)

    half_d = q.shape[-1] // 2
    q1, q2 = q[..., :half_d], q[..., half_d:]
    k1, k2 = k[..., :half_d], k[..., half_d:]

    cos = cos.unsqueeze(1)  # (B, 1, T, D//2)
    sin = sin.unsqueeze(1)

    q_rot1 = q1 * cos - q2 * sin
    q_rot2 = q2 * cos + q1 * sin
    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k2 * cos + k1 * sin

    return torch.cat([q_rot1, q_rot2], dim=-1), torch.cat([k_rot1, k_rot2], dim=-1)


def _create_cos_sin(
    seq_len: int,
    head_dim: int,
    batch_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create cos/sin of shape (batch_size, seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0, "head_dim must be even"
    half_d = head_dim // 2

    inv_freq = 1.0 / (10000 ** (torch.arange(half_d, device=device).float() / half_d))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, half_d)

    cos = freqs.cos().to(dtype).unsqueeze(0).expand(batch_size, -1, -1)
    sin = freqs.sin().to(dtype).unsqueeze(0).expand(batch_size, -1, -1)
    return cos, sin


# Test configurations
_SHAPES = [
    (1, 32, 8, 4, 64),      # (B, T, HQ, HK, D)
    (1, 64, 16, 8, 128),
    (2, 128, 32, 8, 128),
]

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("B,T,HQ,HK,D", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("cos_batch_size", [1])  # usually 1 for efficiency
@pytest.mark.parametrize("is_last", [True, False])
def test_triton_rope_correctness(B, T, HQ, HK, D, dtype, cos_batch_size, is_last):
    # Skip bf16 on CPU if not supported well
    if dtype == torch.bfloat16 and device == "cpu":
        pytest.skip("bfloat16 has limited support on CPU")

    # Create inputs
    q = torch.randn(B, HQ, T, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, HK, T, D, device=device, dtype=dtype, requires_grad=True)

    cos_ref, sin_ref = _create_cos_sin(T, D, cos_batch_size, dtype)
    cos_ker = cos_ref.clone().detach()
    sin_ker = sin_ref.clone().detach()

    # Reference forward
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    out_ref_q, out_ref_k = _ref_apply_rotary_pos_emb(q_ref, k_ref, cos_ref, sin_ref)

    # Kernel forward
    q_ker = q.clone().detach().requires_grad_(True)
    k_ker = k.clone().detach().requires_grad_(True)
    out_ker_q, out_ker_k = LigerRopeFunction.apply(q_ker, k_ker, cos_ker, sin_ker)

    # Set tolerances
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2

    assert_verbose_allclose(out_ker_q, out_ref_q, atol=atol, rtol=rtol)
    assert_verbose_allclose(out_ker_k, out_ref_k, atol=atol, rtol=rtol)

    # Simulate downstream usage if not last layer
    if not is_last:
        out_ref_q = out_ref_q * 2.0
        out_ref_k = out_ref_k * 2.0
        out_ker_q = out_ker_q * 2.0
        out_ker_k = out_ker_k * 2.0

    # Backward pass
    (out_ref_q.sum() + out_ref_k.sum()).backward()
    (out_ker_q.sum() + out_ker_k.sum()).backward()

    assert_verbose_allclose(q_ker.grad, q_ref.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(k_ker.grad, k_ref.grad, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
