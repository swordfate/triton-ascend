# test/transformers/test_qwen2vl_mrope.py

import torch
import pytest
from typing import Optional

from test.utils import assert_verbose_allclose, set_seed
from liger_kernel.utils import infer_device

# Only import the Ascend (NPU) backend implementation
try:
    from liger_kernel.ops.backends._ascend.ops.qwen2vl_mrope import LigerQwen2VLMRopeFunction
except ImportError as e:
    pytest.skip(f"Qwen2VL M-RoPE not available on NPU: {e}", allow_module_level=True)

device = infer_device()
set_seed(42)


def apply_multimodal_rotary_pos_emb_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference implementation (device-agnostic).
    Matches HuggingFace Qwen2-VL M-RoPE logic.
    """
    cos_t, cos_h, cos_w = cos.chunk(3, dim=0)  # each: (1, bsz, sl, hd)
    sin_t, sin_h, sin_w = sin.chunk(3, dim=0)

    cos_t = cos_t.squeeze(0)
    cos_h = cos_h.squeeze(0)
    cos_w = cos_w.squeeze(0)
    sin_t = sin_t.squeeze(0)
    sin_h = sin_h.squeeze(0)
    sin_w = sin_w.squeeze(0)

    hd = q.shape[-1]
    assert hd % 2 == 0, "head_dim must be even"
    half_hd = hd // 2

    t_end = mrope_section[0]
    h_end = t_end + mrope_section[1]
    w_end = half_hd

    # Build left half [0, half_hd)
    left_cos = torch.cat([
        cos_t[..., :t_end],
        cos_h[..., t_end:h_end],
        cos_w[..., h_end:w_end],
    ], dim=-1)
    left_sin = torch.cat([
        sin_t[..., :t_end],
        sin_h[..., t_end:h_end],
        sin_w[..., h_end:w_end],
    ], dim=-1)

    # Right half is duplicate of left (RoPE property)
    cos_full = torch.cat([left_cos, left_cos], dim=-1)  # (bsz, sl, hd)
    sin_full = torch.cat([left_sin, left_sin], dim=-1)

    def rotate_half(x):
        x1 = x[..., :half_hd]
        x2 = x[..., half_hd:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = q * cos_full.unsqueeze(1) + rotate_half(q) * sin_full.unsqueeze(1)
    k_embed = k * cos_full.unsqueeze(1) + rotate_half(k) * sin_full.unsqueeze(1)
    return q_embed, k_embed


# Valid test cases: sum(mrope_section) <= head_dim // 2
_FORWARD_CASES = [
    # (bsz, seq_len, head_dim, n_qh, n_kh, mrope_section)
    (2, 128, 128, 32, 32, [16, 24]),   # 16+24=40 <= 64
    (2, 128, 128, 32, 8,  [16, 24]),
    (1, 32,  64,  32, 32, [8, 16]),    # 8+16=24 <= 32
    (1, 32,  64,  32, 8,  [8, 16]),
    (1, 16,  64,  16, 16, [16, 16]),   # 16+16=32 == 32
    (1, 16,  64,  16, 4,  [16, 16]),
]

_BACKWARD_CASES = [
    (2, 64, 64, 16, 16, [8, 16]),
    (2, 64, 64, 16, 4,  [8, 16]),
    (1, 16, 64, 16, 16, [16, 16]),
    (1, 16, 64, 16, 4,  [16, 16]),
]


@pytest.mark.parametrize("bsz,sl,hd,n_qh,n_kh,mrope_section,dtype", [
    (*case, dtype) for case in _FORWARD_CASES for dtype in [torch.float32, torch.float16]
])
def test_qwen2vl_mrope_forward(bsz, sl, hd, n_qh, n_kh, mrope_section, dtype):
    q = torch.randn(bsz, n_qh, sl, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_kh, sl, hd, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(3, bsz, sl, hd, device=device, dtype=torch.float32)
    sin = torch.randn(3, bsz, sl, hd, device=device, dtype=torch.float32)

    ref_q, ref_k = apply_multimodal_rotary_pos_emb_ref(q.clone(), k.clone(), cos, sin, mrope_section)
    tri_q, tri_k = LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section)

    atol = 1e-5 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-3

    assert_verbose_allclose(tri_q, ref_q, atol=atol, rtol=rtol)
    assert_verbose_allclose(tri_k, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize("bsz,sl,hd,n_qh,n_kh,mrope_section", _BACKWARD_CASES)
def test_qwen2vl_mrope_backward(bsz, sl, hd, n_qh, n_kh, mrope_section):
    dtype = torch.float32
    q = torch.randn(bsz, n_qh, sl, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_kh, sl, hd, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(3, bsz, sl, hd, device=device, dtype=torch.float32)
    sin = torch.randn(3, bsz, sl, hd, device=device, dtype=torch.float32)

    # Reference backward
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    ref_q_out, ref_k_out = apply_multimodal_rotary_pos_emb_ref(q_ref, k_ref, cos, sin, mrope_section)
    (ref_q_out.sum() + ref_k_out.sum()).backward()
    ref_dq, ref_dk = q_ref.grad, k_ref.grad

    # Triton/Ascend backward
    tri_q_out, tri_k_out = LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section)
    (tri_q_out.sum() + tri_k_out.sum()).backward()
    tri_dq, tri_dk = q.grad, k.grad

    assert_verbose_allclose(tri_dq, ref_dq, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(tri_dk, ref_dk, atol=1e-5, rtol=1e-5)


def test_qwen2vl_mrope_edge_case_all_valid():
    """Test with all tokens valid (no ignore index)."""
    bsz, sl, hd, n_qh, n_kh = 1, 8, 64, 8, 8
    mrope_section = [16, 16]
    dtype = torch.float32

    q = torch.randn(bsz, n_qh, sl, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_kh, sl, hd, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(3, bsz, sl, hd, device=device, dtype=torch.float32)
    sin = torch.randn(3, bsz, sl, hd, device=device, dtype=torch.float32)

    ref_q, ref_k = apply_multimodal_rotary_pos_emb_ref(q.clone(), k.clone(), cos, sin, mrope_section)
    tri_q, tri_k = LigerQwen2VLMRopeFunction.apply(q, k, cos, sin, mrope_section)

    assert_verbose_allclose(tri_q, ref_q, atol=1e-5, rtol=1e-5)
    assert_verbose_allclose(tri_k, ref_k, atol=1e-5, rtol=1e-5)

    (tri_q.sum() + tri_k.sum()).backward()
    assert not torch.isnan(q.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
