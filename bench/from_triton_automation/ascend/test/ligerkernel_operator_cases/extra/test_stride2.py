import torch
import torch_npu
import triton
import triton.language as tl


def torch_test(q, k, head_dim_half, bias):
    d_indices = torch.arange(0, head_dim_half)
    k[d_indices * 2 + bias] = q[d_indices * 2 + bias]
    # k[d_indices * 2 + 1 + bias] = -q[d_indices * 2 + 1 + bias]
    return k


@triton.jit
def _llama4_rope_kernel(
    q_ptr,
    k_ptr,
    # freqs_real_ptr,
    # freqs_imag_ptr,
    # q_row_stride,
    # k_row_stride,
    # q_head_stride,
    # k_head_stride,
    # freqs_row_stride,
    # seq_len,
    # batch_size,
    # imag_sign,
    head_dim_half: tl.constexpr,
    bias: tl.constexpr
    # n_q_heads: tl.constexpr,
    # n_k_heads: tl.constexpr,
    # BLOCK_SIZE: tl.constexpr,
):
    """
    H100-optimized RoPE kernel with improved parallelization across heads and dimensions.
    Grid: (batch*seq, head)
    """

    d_indices = tl.program_id(0) + tl.arange(0, head_dim_half)
    mask_d = d_indices < head_dim_half

    q_head_ptr = q_ptr
    q_real = tl.load(q_head_ptr + d_indices * 2 + bias, mask=mask_d, other=0.0)
    # q_imag = tl.load(q_head_ptr + d_indices * 2 + 1 + bias, mask=mask_d, other=0.0)

    new_q_real = q_real
    # new_q_imag = q_imag * (-1)
    k_head_ptr = k_ptr
    tl.store(k_head_ptr + d_indices * 2 + bias, new_q_real, mask=mask_d)
    # tl.store(k_head_ptr + d_indices * 2 + 1 + bias, new_q_imag, mask=mask_d)

head_dim_half = 16
bias = 4
len = bias + head_dim_half * 2
q = torch.ones(len).npu()
k = torch.zeros_like(q).npu()
k_ref = torch.zeros_like(q).npu()

_llama4_rope_kernel[(1,)](q, k, head_dim_half, bias)
k_ref = torch_test(q, k_ref, head_dim_half, bias)

print(k)
print(k_ref)
assert torch.allclose(k, k_ref)

