# benchmarks/tvd_100us_benchmark.py

import torch
import triton
from typing import Literal
from utils import QUANTILES, SingleBenchmarkRunInput, SingleBenchmarkRunOutput
from utils import _test_memory, parse_benchmark_script_args, run_benchmarks
from liger_kernel.utils import infer_device

device = infer_device()  # No restriction — supports npu, cuda, xpu

try:
    from liger_kernel.ops.backends._ascend.ops.tvd import LigerTVDLossFunction
except ImportError as e:
    raise ImportError(
        "Failed to import LigerTVDLossFunction. "
        "Make sure liger-kernel is installed in dev mode: pip install -e ."
    ) from e


REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]


def bench_speed_tvd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = input.x  # BT = batch_size * seq_len
    extra = input.extra_benchmark_config
    V = extra["V"]  # vocab_size
    reduction: REDUCTION_LITERAL = extra["reduction"]
    has_label = extra.get("has_label", False)
    ignore_index = extra.get("ignore_index", -100)
    dtype = extra.get("dtype", torch.float32)

    p = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)

    shift_labels = None
    if has_label:
        shift_labels = torch.randint(0, V, (B,), device=device, dtype=torch.long)
        # Randomly set some to ignore_index
        mask_ignore = torch.rand(B, device=device) < 0.1
        shift_labels[mask_ignore] = ignore_index

    def call_forward():
        return LigerTVDLossFunction.apply(p, q, shift_labels, reduction, ignore_index)

    def call_backward():
        loss = LigerTVDLossFunction.apply(p, q, shift_labels, reduction, ignore_index)
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss.backward()
        else:
            # For "none", sum to scalar for backward
            loss.sum().backward()

    mode = input.kernel_operation_mode
    if mode == "forward":
        fn = call_forward
        if p.grad is not None:
            p.grad.zero_()
        if q.grad is not None:
            q.grad.zero_()
    elif mode == "backward":
        fn = call_backward
        if p.grad is not None:
            p.grad.zero_()
        if q.grad is not None:
            q.grad.zero_()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ms_50, ms_20, ms_80 = triton.testing.do_bench(
        fn,
        quantiles=QUANTILES,
        rep=500,
        warmup=100
    )
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_tvd(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = input.x
    extra = input.extra_benchmark_config
    V = extra["V"]
    reduction: REDUCTION_LITERAL = extra["reduction"]
    has_label = extra.get("has_label", False)
    ignore_index = extra.get("ignore_index", -100)
    dtype = extra.get("dtype", torch.float32)

    p = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)

    shift_labels = None
    if has_label:
        shift_labels = torch.randint(0, V, (B,), device=device, dtype=torch.long)
        mask_ignore = torch.rand(B, device=device) < 0.1
        shift_labels[mask_ignore] = ignore_index

    def call_forward():
        return LigerTVDLossFunction.apply(p, q, shift_labels, reduction, ignore_index)

    def call_backward():
        loss = LigerTVDLossFunction.apply(p, q, shift_labels, reduction, ignore_index)
        if loss.numel() == 1:
            loss.backward()
        else:
            loss.sum().backward()

    mode = input.kernel_operation_mode
    if mode == "forward":
        fn = call_forward
        if p.grad is not None:
            p.grad.zero_()
        if q.grad is not None:
            q.grad.zero_()
    elif mode == "backward":
        fn = call_backward
        if p.grad is not None:
            p.grad.zero_()
        if q.grad is not None:
            q.grad.zero_()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mem_50, mem_20, mem_80 = _test_memory(fn, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_sweep = {
        "x_name": "BT",
        "x_label": "total_rows (B × T)",
        "x_values": [512, 1024, 2048, 4096],
        "overwrite": args.overwrite,
    }

    extra_configs = [
        # Typical LM head: vocab ～32k
        {"V": 32768, "reduction": "batchmean", "has_label": True, "dtype": torch.float32},
        {"V": 32768, "reduction": "mean", "has_label": False, "dtype": torch.float32},
        # Smaller vocab for faster test
        {"V": 8192, "reduction": "none", "has_label": True, "dtype": torch.float32},
        {"V": 8192, "reduction": "sum", "has_label": False, "dtype": torch.float32},
    ]

    print("Running TVD Loss (forward + backward) benchmark targeting ～100–500 µs...")

    for mode in ["forward", "backward"]:
        run_benchmarks(
            bench_test_fn=bench_speed_tvd,
            kernel_operation_modes=[mode],
            metric_name="speed",
            metric_unit="ms",
            kernel_name=f"tvd_{mode}_100us",
            kernel_providers=["liger"],
            extra_benchmark_configs=extra_configs,
            **common_sweep,
        )

        run_benchmarks(
            bench_test_fn=bench_memory_tvd,
            kernel_operation_modes=[mode],
            metric_name="memory",
            metric_unit="MB",
            kernel_name=f"tvd_{mode}_100us",
            kernel_providers=["liger"],
            extra_benchmark_configs=extra_configs,
            **common_sweep,
        )
