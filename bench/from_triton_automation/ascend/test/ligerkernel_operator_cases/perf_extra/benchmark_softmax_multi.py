# benchmarks/benchmark_softmax_multiblock.py

import torch
import triton

from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

from liger_kernel.ops.softmax import LigerSoftmaxFunction
from liger_kernel.ops.utils import calculate_settings  # for patching
from liger_kernel.utils import infer_device

device = infer_device()


# === MONKEY-PATCH: Override calculate_settings to force small BLOCK_SIZE ===
def patched_calculate_settings(n_cols: int):
    """
    Return fixed BLOCK_SIZE=1024 so that n_cols > 1024 triggers multi-block.
    This avoids modifying the kernel source.
    """
    BLOCK_SIZE = 1024
    num_warps = 4
    if n_cols > 2048:
        num_warps = 8
    if n_cols > 8192:
        num_warps = 16
    return BLOCK_SIZE, num_warps


# Apply the patch before importing or using softmax
import liger_kernel.ops.softmax as softmax_module
original_calculate_settings = softmax_module.calculate_settings
softmax_module.calculate_settings = patched_calculate_settings


# Now define wrappers (they will use patched version)
def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)


def liger_softmax(x: torch.Tensor) -> torch.Tensor:
    return LigerSoftmaxFunction.apply(x)


# Restore original after? Not needed since script exits, but good practice in tests
# We'll keep patched for entire run


def bench_speed_softmax_multiblock(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    n_cols = input.x
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]

    if n_cols <= 1024:
        raise ValueError(f"Use n_cols > 1024 to trigger multi-block, got {n_cols}")

    x = torch.randn(B * T, n_cols, device=device, dtype=torch.float16, requires_grad=True)

    def fwd():
        if input.kernel_provider == "liger":
            return liger_softmax(x)
        else:
            return torch_softmax(x)

    if input.kernel_operation_mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(fwd, quantiles=QUANTILES, rep=100, warmup=20)
    elif input.kernel_operation_mode == "backward":
        y = fwd()
        loss = y.sum()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: loss.backward(retain_graph=True),
            quantiles=QUANTILES,
            grad_to_none=[x],
            rep=100,
            warmup=20,
        )
    elif input.kernel_operation_mode == "full":
        def full():
            y = fwd()
            y.sum().backward(retain_graph=True)
        ms_50, ms_20, ms_80 = triton.testing.do_bench(full, quantiles=QUANTILES, rep=100, warmup=20)
    else:
        raise ValueError(f"Unknown mode: {input.kernel_operation_mode}")

    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_softmax_multiblock(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    n_cols = input.x
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]
    if n_cols <= 1024:
        raise ValueError(f"Use n_cols > 1024 to trigger multi-block, got {n_cols}")

    x = torch.randn(B * T, n_cols, device=device, dtype=torch.float16, requires_grad=True)

    def full():
        y = liger_softmax(x) if input.kernel_provider == "liger" else torch_softmax(x)
        y.sum().backward(retain_graph=True)

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_args = {
        "kernel_name": "softmax_multiblock",
        "x_name": "n_cols",
        "x_label": "feature dimension",
        # Must be > 1024 to exceed patched BLOCK_SIZE=1024
        "x_values": [2048, 4096, 8192, 16384, 32768],
        "kernel_providers": ["liger", "torch"],
        "extra_benchmark_configs": [
            {"B": 1, "T": 2048},
            {"B": 4, "T": 1024},
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_memory_softmax_multiblock,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_args,
    )

    run_benchmarks(
        bench_test_fn=bench_speed_softmax_multiblock,
        kernel_operation_modes=["forward", "backward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_args,
    )
