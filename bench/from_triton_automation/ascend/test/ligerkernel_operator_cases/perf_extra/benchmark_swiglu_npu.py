import torch
import torch.nn as nn
import triton

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from utils import QUANTILES
from utils import SingleBenchmarkRunInput
from utils import SingleBenchmarkRunOutput
from utils import _test_memory
from utils import parse_benchmark_script_args
from utils import run_benchmarks

#from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()

if device == "npu":
    from liger_kernel.ops.backends._ascend.ops.swiglu import LigerSiLUMulFunction
    print("### use npu kernel!!")
else:
    from liger_kernel.ops import LigerSiLUMulFunction

class LigerSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


def bench_speed_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    bsz = extra_benchmark_config["B"]
    hidden_size = extra_benchmark_config["hidden_size"]
    dtype = extra_benchmark_config["dtype"]
    intermediate_size = extra_benchmark_config["intermediate_size"]
    hidden_act = extra_benchmark_config["hidden_act"]

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for SwiGLU")

    def fwd():
        return layer(x)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            fwd,
            grad_to_none=[x],
            quantiles=QUANTILES,
            rep=10,
        )
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            grad_to_none=[x],
            quantiles=QUANTILES,
            rep=10,
        )
    else:

        def full():
            y = fwd()
            y.backward(torch.randn_like(y), retain_graph=True)

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x],
            quantiles=QUANTILES,
            rep=10,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    bsz = extra_benchmark_config["B"]
    hidden_size = extra_benchmark_config["hidden_size"]
    dtype = extra_benchmark_config["dtype"]
    intermediate_size = extra_benchmark_config["intermediate_size"]
    hidden_act = extra_benchmark_config["hidden_act"]

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )

    x_shape = (bsz, seq_len, hidden_size)

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for SwiGLU")

    def fwd():
        return layer(x)

    def full():
        y = fwd()
        y.backward(torch.randn_like(y), retain_graph=True)

    if mode == "forward":
        mem_50, mem_20, mem_80 = _test_memory(fwd, quantiles=QUANTILES)
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        mem_50, mem_20, mem_80 = _test_memory(lambda: y.backward(do, retain_graph=True), quantiles=QUANTILES)
    else:
        mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "swiglu",
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, 14)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [
            {
                "B": 4,
                "hidden_size": 4096,
                "dtype": torch.bfloat16,
                "intermediate_size": 11008,
                "hidden_act": "silu",
            }
        ],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_swiglu,
        kernel_operation_modes=["forward","backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
