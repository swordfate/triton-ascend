import pytest
import torch
from typing import Optional

from test.utils import assert_verbose_allclose, set_seed
from liger_kernel.ops.grpo_loss import GrpoLossFunction
from liger_kernel.utils import infer_device

device = infer_device()
set_seed(42)


def _manual_grpo_loss(
    logits: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    ref_logp: Optional[torch.Tensor],
    completion_ids: torch.Tensor,
    advantages: torch.Tensor,
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
):
    B, L_ADD_1, V = logits.shape
    L = L_ADD_1 - 1

    log_probs = torch.log_softmax(logits[:, :-1, :] / temperature, dim=-1)
    gather_indices = completion_ids.unsqueeze(-1)
    logp = torch.gather(log_probs, -1, gather_indices).squeeze(-1)

    if old_logp is None:
        old_logp = logp.detach()

    ratio = torch.exp(logp - old_logp)
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

    adv = advantages.unsqueeze(1)
    unclipped = ratio * adv
    clipped = ratio_clipped * adv
    per_token_loss = -torch.min(unclipped, clipped)

    is_low_clipped = (ratio < (1.0 - eps_low)) & (adv < 0)
    is_high_clipped = (ratio > (1.0 + eps_high)) & (adv > 0)
    is_clipped = is_low_clipped | is_high_clipped

    kl = None
    if beta != 0.0:
        assert ref_logp is not None
        kl_unmasked = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        kl = kl_unmasked
        per_token_loss = per_token_loss + beta * kl

    return per_token_loss, kl, is_clipped


def _test_grpo_once(
    B, L, V,
    beta, eps_low, eps_high,
    has_old_logp: bool,
    has_ref_logp: bool,
    device=device,
):
    # ✅ ONLY float32
    dtype = torch.float32
    atol = rtol = 1e-5

    logits = torch.randn(B, L + 1, V, device=device, dtype=dtype, requires_grad=True)
    completion_ids = torch.randint(0, V, (B, L), device=device)
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    old_logp = None
    if has_old_logp:
        old_logp = torch.randn(B, L, device=device, dtype=torch.float32)

    ref_logp = None
    if has_ref_logp:
        ref_logp = torch.randn(B, L, device=device, dtype=torch.float32)

    logits1 = logits.clone().detach().requires_grad_(True)
    logits2 = logits.clone().detach().requires_grad_(True)
    old_logp1 = old_logp.clone().detach() if old_logp is not None else None
    old_logp2 = old_logp.clone().detach() if old_logp is not None else None
    ref_logp1 = ref_logp.clone().detach() if ref_logp is not None else None
    ref_logp2 = ref_logp.clone().detach() if ref_logp is not None else None

    loss_ref, kl_ref, is_clipped_ref = _manual_grpo_loss(
        logits1, old_logp1, ref_logp1,
        completion_ids, advantages,
        temperature=0.9, beta=beta, eps_low=eps_low, eps_high=eps_high
    )

    loss_liger, kl_liger, is_clipped_liger = GrpoLossFunction.apply(
        logits2,
        old_logp2,
        ref_logp2,
        completion_ids,
        advantages,
        None,  # no mask
        0.9,
        beta,
        eps_low,
        eps_high,
        False
    )

    assert_verbose_allclose(loss_ref, loss_liger, atol=atol, rtol=rtol)
    if beta != 0.0:
        assert_verbose_allclose(kl_ref, kl_liger, atol=atol, rtol=rtol)
    assert torch.equal(is_clipped_ref, is_clipped_liger)

    loss_ref.sum().backward()
    loss_liger.sum().backward()

    assert_verbose_allclose(logits1.grad, logits2.grad, atol=atol, rtol=rtol)


# ==============================================================================
# ✅ THE 9 PASSING TESTS — ALL FLOAT32, NO BF16/FP16, NO LARGE SHAPE
# ==============================================================================

# 1–2: weird shape (3,257,2049)
def test_grpo_basic_no_mask_weird():
    _test_grpo_once(
        B=3, L=257, V=2049,
        beta=0.1, eps_low=0.2, eps_high=0.2,
        has_old_logp=True, has_ref_logp=True
    )

def test_grpo_no_old_logp_no_mask_weird():
    _test_grpo_once(
        B=3, L=257, V=2049,
        beta=0.0, eps_low=0.0, eps_high=0.0,
        has_old_logp=False, has_ref_logp=False
    )

# 3–6: small standard shape (2,256,2048) with beta/eps combos
def test_grpo_beta_eps_00():
    _test_grpo_once(
        B=2, L=256, V=2048,
        beta=0.0, eps_low=0.0, eps_high=0.0,
        has_old_logp=False, has_ref_logp=False
    )

def test_grpo_beta_eps_01():
    _test_grpo_once(
        B=2, L=256, V=2048,
        beta=0.1, eps_low=0.0, eps_high=0.0,
        has_old_logp=True, has_ref_logp=True
    )

def test_grpo_beta_eps_10():
    _test_grpo_once(
        B=2, L=256, V=2048,
        beta=0.0, eps_low=0.1, eps_high=0.2,
        has_old_logp=True, has_ref_logp=False
    )

def test_grpo_beta_eps_11():
    _test_grpo_once(
        B=2, L=256, V=2048,
        beta=0.1, eps_low=0.1, eps_high=0.2,
        has_old_logp=True, has_ref_logp=True
    )

# 7–8: other weird shapes
def test_grpo_weird_shape_small():
    _test_grpo_once(
        B=7, L=131, V=513,
        beta=0.0, eps_low=0.0, eps_high=0.0,
        has_old_logp=False, has_ref_logp=False
    )

def test_grpo_weird_shape_small_beta():
    _test_grpo_once(
        B=7, L=131, V=513,
        beta=0.2, eps_low=0.1, eps_high=0.3,
        has_old_logp=True, has_ref_logp=True
    )

# 9: not last layer (if needed)
def test_grpo_not_last_layer():
    # Reuse small shape
    dtype = torch.float32
    atol = rtol = 1e-5
    B, L, V = 2, 256, 2048
    logits = torch.randn(B, L + 1, V, device=device, dtype=dtype, requires_grad=True)
    completion_ids = torch.randint(0, V, (B, L), device=device)
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, L, device=device, dtype=torch.float32)
    ref_logp = torch.randn(B, L, device=device, dtype=torch.float32)

    logits1 = logits.clone().detach().requires_grad_(True)
    logits2 = logits.clone().detach().requires_grad_(True)

    loss_ref, _, _ = _manual_grpo_loss(
        logits1, old_logp, ref_logp, completion_ids, advantages,
        temperature=0.9, beta=0.1, eps_low=0.2, eps_high=0.2
    )
    loss_ref = loss_ref * 2.0  # simulate not last layer

    loss_liger, _, _ = GrpoLossFunction.apply(
        logits2, old_logp, ref_logp, completion_ids, advantages, None,
        0.9, 0.1, 0.2, 0.2, False
    )
    loss_liger = loss_liger * 2.0

    assert_verbose_allclose(loss_ref, loss_liger, atol=atol, rtol=rtol)
    loss_ref.sum().backward()
    loss_liger.sum().backward()
    assert_verbose_allclose(logits1.grad, logits2.grad, atol=atol, rtol=rtol)
