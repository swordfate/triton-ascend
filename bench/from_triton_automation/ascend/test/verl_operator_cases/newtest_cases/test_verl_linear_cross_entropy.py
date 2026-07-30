# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliate
import os

import pytest
import torch

from benchmark.utils import Benchmark, OpType
from triton_ascend_kernels.loss.verl_linear_cross_entropy import linear_cross_entropy
from triton_ascend_kernels.loss.verl_cross_entropy_kernels import (
    run_triton_d_weight,
    run_triton_d_hidden,
    run_epilogue_tp,
    run_epilogue_update,
)

MAX_TEST_CASES = int(os.environ.get("MAX_TEST_CASES", 5))

LINEAR_CE_CASES = [
    (1, 1937, 3584, 152064),
    (1, 2169, 896, 151936),
    (1, 1530, 2048, 32256),
    (1, 1388, 4096, 102400),
    (1, 8192, 4096, 102400)
][:MAX_TEST_CASES]


def run_torch_linear_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: str = "none",
):
    hidden_2d = hidden.squeeze(0).to(torch.float32)  # [num_tokens, hidden_size]
    weight_t = weight.transpose(0, 1).to(torch.float32)  # [hidden_size, vocab_size]

    logits = torch.matmul(hidden_2d, weight_t)  # [num_tokens, vocab_size]
    logits /= temperature

    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy_a = torch.logsumexp(logits, dim=-1)
    entropy_b = torch.sum(pd * logits, dim=-1)
    entropy = entropy_a - entropy_b  # [num_tokens]

    logprobs = torch.nn.functional.cross_entropy(
        logits, labels.squeeze(0), reduction=reduction
    )
    logprobs = torch.neg(logprobs)
    return logprobs, entropy


def run_torch_d_weight(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: int,
):
    logits = (
        torch.matmul(hidden.to(torch.float32), weight.T.to(torch.float32)) / temperature
    )

    if reduction == 0:
        loss_vec = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    elif reduction == 1:
        loss_vec = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    else:
        loss_vec = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    loss = loss_vec.sum()
    if hidden.grad is not None:
        hidden.grad = None
    if weight.grad is not None:
        weight.grad = None
    loss.backward()
    d_weight_expected = weight.grad.detach().to(torch.float32)
    return logits, d_weight_expected


def run_torch_d_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: int,
):
    logits = (
        torch.matmul(hidden.to(torch.float32), weight.T.to(torch.float32)) / temperature
    )

    if reduction == 0:
        loss_vec = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    elif reduction == 1:
        loss_vec = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    else:
        loss_vec = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    loss = loss_vec.sum()
    if hidden.grad is not None:
        hidden.grad = None
    loss.backward()
    d_hidden_expected = hidden.grad.detach().to(torch.float32)
    return logits, d_hidden_expected


def run_torch_epilogue_tp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduced_max: torch.Tensor,
    original_max: torch.Tensor,
    accu: torch.Tensor,
    entropy_b_in: torch.Tensor,
    vocab_size: int,
    num_tokens: int,
    temperature: float,
    reduction: str = "none",
):
    vocab_per_split = 1024
    temperature = 1.5
    device = hidden.device
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    hidden_fp32 = hidden.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    logits = torch.matmul(hidden_fp32, weight_fp32.T) / temperature

    for s in range(num_splits):
        start = s * vocab_per_split
        end = min(start + vocab_per_split, vocab_size)
        local_logits = logits[:, start:end]
        local_max = local_logits.max(dim=1).values
        local_exp = torch.exp(local_logits - local_max.unsqueeze(-1))
        local_accu = local_exp.sum(dim=1)
        local_entropy_b = torch.sum(
            local_exp * (local_logits - local_max.unsqueeze(-1)), dim=1
        )

        reduced_max[:, s] = local_max.to(reduced_max.dtype)
        original_max[:, s] = local_max.to(original_max.dtype)
        accu[:, s] = local_accu.to(accu.dtype)
        entropy_b_in[:, s] = local_entropy_b.to(entropy_b_in.dtype)

    max_ref = reduced_max.to(torch.float32).max(dim=1).values
    scale = torch.exp(original_max.to(torch.float32) - max_ref.unsqueeze(1))
    accu_ref = torch.sum(scale * accu.to(torch.float32), dim=1)
    entropy_b_ref = torch.sum(scale * entropy_b_in.to(torch.float32), dim=1)

    return max_ref, accu_ref, entropy_b_ref


def run_torch_epilogue_tp_update(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    max_val: torch.Tensor,
    exp_logits: torch.Tensor,
    accu: torch.Tensor,
    num_tokens: int,
):
    device = "npu"
    lse = torch.logsumexp(logits, dim=-1)
    softmax = torch.softmax(logits, dim=-1)

    entropy_b_expected = (
        torch.sum(exp_logits * (logits - max_val.unsqueeze(-1)), dim=-1) / accu
    )
    entropy_expected = lse - torch.sum(
        softmax * (logits - max_val.unsqueeze(-1)), dim=-1
    )
    logprobs_expected_vec = torch.log_softmax(logits, dim=-1)[
        torch.arange(num_tokens, device=device), labels
    ]

    return entropy_b_expected, entropy_expected, logprobs_expected_vec


@pytest.mark.parametrize(
    "batch_size, num_tokens, hidden_size, vocab_size", LINEAR_CE_CASES
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_linear_cross_entropy(
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype,
):
    temperature = 1.5

    hidden = (
        torch.empty((batch_size, num_tokens, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    weight = (
        torch.empty((vocab_size, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    labels = torch.randint(0, vocab_size, (batch_size, num_tokens), device="npu")

    (torch_logprobs, torch_entropy) = run_torch_linear_ce(
        hidden, weight, labels, temperature
    )
    (kernel_logprobs, kernel_entropy) = linear_cross_entropy(
        hidden, weight, labels, temperature
    )
    torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-3, rtol=2e-4)
    torch.testing.assert_close(torch_entropy, kernel_entropy, atol=5e-3, rtol=5e-4)


@pytest.mark.parametrize(
    "batch_size, num_tokens, hidden_size, vocab_size", LINEAR_CE_CASES
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_linear_cross_entropy_bwd(
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype,
):
    temperature = 1.5

    hidden = (
        torch.empty((batch_size, num_tokens, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    weight = (
        torch.empty((vocab_size, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    labels = torch.randint(0, vocab_size, (batch_size, num_tokens), device="npu")

    (torch_logprobs, torch_entropy) = run_torch_linear_ce(
        hidden, weight, labels, temperature
    )
    (kernel_logprobs, kernel_entropy) = linear_cross_entropy(
        hidden, weight, labels, temperature
    )
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    os.environ["ENABLE_UNPUBLISHED_FEATURE"] = "1"
    g_entropy = torch.empty((num_tokens,), dtype=dtype, device="npu").uniform_(
        -0.5, 0.5
    )
    g_logprobs = torch.empty((num_tokens,), dtype=dtype, device="npu").uniform_(-1, 1)

    (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
        (torch_entropy, torch_logprobs),
        (hidden, weight),
        (g_entropy, g_logprobs),
        retain_graph=False,
    )
    (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad(
        (kernel_entropy, kernel_logprobs),
        (hidden, weight),
        (g_entropy, g_logprobs),
        retain_graph=False,
    )

    torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=2e-2, rtol=4e-2)
    torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=2e-2, rtol=4e-2)
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "0"
    os.environ["ENABLE_UNPUBLISHED_FEATURE"] = "0"


@pytest.mark.parametrize(
    "batch_size, num_tokens, hidden_size, vocab_size", LINEAR_CE_CASES
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_efficient_entropy_backward_kernel_d_weight(
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype,
):
    temperature = 1.5
    reduction = 0
    hidden = (
        torch.empty((num_tokens, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    weight = (
        torch.empty((vocab_size, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    labels = torch.randint(0, vocab_size, (num_tokens,), device="npu")

    logits, d_weight_torch = run_torch_d_weight(
        hidden, weight, labels, temperature, reduction
    )

    max_val = logits.max(dim=-1).values.to(torch.float32)
    exp_logits = torch.exp(logits.to(torch.float32) - max_val.unsqueeze(-1))
    accu = exp_logits.sum(dim=-1)
    if reduction == 0:
        d_logprobs = -torch.ones((num_tokens,), dtype=torch.float32, device="npu")
    elif reduction == 1:
        d_logprobs = torch.tensor([-1.0], dtype=torch.float32, device="npu")
    else:
        d_logprobs = torch.tensor([-1.0], dtype=torch.float32, device="npu")
    d_entropy = torch.zeros((num_tokens,), dtype=torch.float32, device="npu")
    entropy_b = torch.zeros((num_tokens,), dtype=torch.float32, device="npu")
    d_weight = torch.zeros((vocab_size, hidden_size), device="npu", dtype=torch.float32)
    d_weight = run_triton_d_weight(
        hidden,
        weight,
        labels,
        max_val,
        accu,
        d_entropy,
        d_logprobs,
        entropy_b,
        d_weight,
    )

    torch.testing.assert_close(d_weight, d_weight_torch, atol=2e-2, rtol=4e-2)


@pytest.mark.parametrize(
    "batch_size, num_tokens, hidden_size, vocab_size", LINEAR_CE_CASES
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_efficient_entropy_backward_kernel_d_hidden(
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype,
):
    temperature = 1.5
    reduction = 0
    hidden = (
        torch.empty((num_tokens, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    weight = (
        torch.empty((vocab_size, hidden_size), dtype=dtype, device="npu")
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    labels = torch.randint(0, vocab_size, (num_tokens,), device="npu")

    logits, d_hidden_torch = run_torch_d_hidden(
        hidden, weight, labels, temperature, reduction
    )

    max_val = logits.max(dim=-1).values.to(torch.float32)
    exp_logits = torch.exp(logits.to(torch.float32) - max_val.unsqueeze(-1))
    accu = exp_logits.sum(dim=-1)
    if reduction == 0:
        d_logprobs = -torch.ones((num_tokens,), dtype=torch.float32, device="npu")
    elif reduction == 1:
        d_logprobs = torch.tensor([-1.0], dtype=torch.float32, device="npu")
    else:
        d_logprobs = torch.tensor([-1.0], dtype=torch.float32, device="npu")
    d_entropy = torch.zeros((num_tokens,), dtype=torch.float32, device="npu")
    entropy_b = torch.zeros((num_tokens,), dtype=torch.float32, device="npu")
    d_hidden = torch.zeros((num_tokens, hidden_size), device="npu", dtype=torch.float32)
    d_hidden = run_triton_d_hidden(
        hidden,
        weight,
        labels,
        max_val,
        accu,
        d_entropy,
        d_logprobs,
        entropy_b,
        d_hidden,
    )
    torch.testing.assert_close(d_hidden, d_hidden_torch, atol=2e-2, rtol=4e-2)


@pytest.mark.parametrize(
    "batch_size, num_tokens, hidden_size, vocab_size", LINEAR_CE_CASES
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_epilogue_tp(
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype,
):
    temperature = 1.5
    vocab_per_split = 1024
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    device = "npu"

    hidden = (
        torch.empty((num_tokens, hidden_size), dtype=dtype, device=device)
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    weight = (
        torch.empty((vocab_size, hidden_size), dtype=dtype, device=device)
        .uniform_(-0.5, 0.5)
        .requires_grad_()
    )
    labels = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int64, device=device
    )

    reduced_max = torch.empty((num_tokens, num_splits), dtype=dtype, device=device)
    original_max = torch.empty_like(reduced_max)
    accu = torch.empty_like(reduced_max)
    entropy_b_in = torch.empty_like(reduced_max)

    (max_ref, accu_ref, entropy_b_ref) = run_torch_epilogue_tp(
        hidden,
        weight,
        labels,
        reduced_max,
        original_max,
        accu,
        entropy_b_in,
        vocab_size,
        num_tokens,
        temperature,
    )
    (max_res, accu_res, entropy_b_res) = run_epilogue_tp(
        num_tokens, vocab_size, reduced_max, original_max, accu, entropy_b_in
    )

    torch.testing.assert_close(max_ref, max_res, atol=5e-3, rtol=5e-4)
    torch.testing.assert_close(accu_ref, accu_res, atol=5e-3, rtol=5e-4)
    torch.testing.assert_close(entropy_b_ref, entropy_b_res, atol=5e-3, rtol=5e-4)


@pytest.mark.parametrize(
    "batch_size, num_tokens, hidden_size, vocab_size", LINEAR_CE_CASES
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_epilogue_tp_update(
    batch_size: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype,
):
    temperature = 1.5
    reduction = 0
    device = "npu"

    hidden = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype, device=device)
    labels = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int64, device=device
    )

    logits = (
        torch.matmul(hidden.to(torch.float32), weight.to(torch.float32).T) / temperature
    )
    max_val = logits.max(dim=-1).values
    exp_logits = torch.exp(logits - max_val.unsqueeze(-1))
    accu = exp_logits.sum(dim=-1)
    entropy_b_in = torch.sum(exp_logits * (logits - max_val.unsqueeze(-1)), dim=-1)
    logprobs_in = logits[torch.arange(num_tokens, device=device), labels]

    entropy_b = entropy_b_in.clone()
    logprobs = logprobs_in.clone()

    (entropy_b, entropy, logprobs, logprobs_scalar) = run_epilogue_update(
        num_tokens, vocab_size, logprobs, max_val, accu, entropy_b, reduction=reduction
    )

    (entropy_b_ref, entropy_ref, logprobs_ref) = run_torch_epilogue_tp_update(
        hidden, weight, labels, logits, max_val, exp_logits, accu, num_tokens
    )

    if reduction == 0:
        torch.testing.assert_close(entropy_b, entropy_b_ref, atol=5e-3, rtol=5e-4)
        torch.testing.assert_close(entropy, entropy_ref, atol=5e-3, rtol=5e-4)
        torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-3, rtol=2e-4)
    elif reduction == 1:
        logprobs_ref_scalar = torch.sum(logprobs_ref)
        torch.testing.assert_close(entropy_b, entropy_b_ref, atol=5e-3, rtol=5e-4)
        torch.testing.assert_close(entropy, entropy_ref, atol=5e-3, rtol=5e-4)
        torch.testing.assert_close(
            logprobs_scalar, logprobs_ref_scalar, atol=1e-3, rtol=2e-4
        )
    else:
        logprobs_ref_scalar = torch.mean(logprobs_ref)
        torch.testing.assert_close(entropy_b, entropy_b_ref, atol=5e-3, rtol=5e-4)
        torch.testing.assert_close(entropy, entropy_ref, atol=5e-3, rtol=5e-4)
        torch.testing.assert_close(
            logprobs_scalar, logprobs_ref_scalar, atol=1e-3, rtol=2e-4
        )
