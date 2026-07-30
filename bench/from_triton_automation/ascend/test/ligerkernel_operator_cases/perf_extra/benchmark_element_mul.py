# benchmarks/element_mul_100us_benchmark.py

import torch
import triton
from utils import QUANTILES, SingleBenchmarkRunInput, SingleBenchmarkRunOutput
from utils import _test_memory, parse_benchmark_script_args, run_benchmarks
from liger_kernel.utils import infer_device

device = infer_device()

# Import the kernel and helper
try:
    from liger_kernel.ops.utils import element_mul_kernel, calculate_settings
except ImportError as e:
    raise ImportError("Make sure liger_kernel is installed and element_mul_kernel is accessible.") from e


def bench_speed_element_mul(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = input.x
    extra = input.extra_benchmark_config
    H = extra["H"]
    dtype = extra.get("dtype", torch.float16)

    # Input tensor: (B, H)
    X = torch.randn(B, H, device=device, dtype=dtype)
    grad_output = torch.tensor(2.5, device=device, dtype=dtype)  # scalar multiplier

    n_cols = H
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    def call():
        # Launch kernel directly
        element_mul_kernel[(B,)](
            X,
            X.stride(0),
            grad_output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    ms_50, ms_20, ms_80 = triton.testing.do_bench(
        call,
        quantiles=QUANTILES,
        rep=500,
        warmup=100
    )
    return SingleBenchmarkRunOutput(y_20=ms_20, y_50=ms_50, y_80=ms_80)


def bench_memory_element_mul(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    B = input.x
    extra = input.extra_benchmark_config
    H = extra["H"]
    dtype = extra.get("dtype", torch.float16)

    X = torch.randn(B, H, device=device, dtype=dtype)
    grad_output = torch.tensor(2.5, device=device, dtype=dtype)

    n_cols = H
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    def call():
        element_mul_kernel[(B,)](
            X,
            X.stride(0),
            grad_output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    mem_50, mem_20, mem_80 = _test_memory(call, quantiles=QUANTILES)
    return SingleBenchmarkRunOutput(y_20=mem_20, y_50=mem_50, y_80=mem_80)


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    # Target ～100 µs → need larger total elements
    # Total elements = B * H
    common_sweep = {
        "x_name": "B",
        "x_label": "batch_size",
        "x_values": [1024, 2048, 4096, 8192],  # increase B to hit target
        "overwrite": args.overwrite,
    }

    extra_configs = [
        {"H": 4096, "dtype": torch.float16},
        {"H": 8192, "dtype": torch.bfloat16},
        {"H": 16384, "dtype": torch.float16},  # for large models
    ]

    print("Running element_mul_kernel benchmark targeting ～100–500 µs...")

    # Speed benchmark
    run_benchmarks(
        bench_test_fn=bench_speed_element_mul,
        kernel_operation_modes=["forward"],
        metric_name="speed",
        metric_unit="ms",
        kernel_name="element_mul_100us",
        kernel_providers=["liger"],
        extra_benchmark_configs=extra_configs,
        **common_sweep,
    )

    # Memory benchmark
    run_benchmarks(
        bench_test_fn=bench_memory_element_mul,
        kernel_operation_modes=["forward"],
        metric_name="memory",
        metric_unit="MB",
        kernel_name="element_mul_100us",
        kernel_providers=["liger"],
        extra_benchmark_configs=extra_configs,
        **common_sweep,
    )
