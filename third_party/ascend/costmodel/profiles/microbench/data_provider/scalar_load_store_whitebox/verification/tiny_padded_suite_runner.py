#!/usr/bin/env python3
"""Tiny padded_* runner for the three padded kernels (gather/scatter/wgrad).

Fixed shape/config:
  sl=4, hs=2, ne=2, top_k=1
  grid=(4,), BLOCK_X=64, superblock_factor=1, num_warps=1
  NUM_COLUMNS=2, logical_program_count_hint=4,
  physical_vector_core_count_hint=num_vectorcore (56)

Inputs are deterministic (top_expert all ones); every program executes both
`bin_idx > 0` conditional loads. All tensors are created on CPU and copied
with `.to("npu")`; no torch NPU kernel is created.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

KERNEL_DIR = Path("/home/c00946898/scalar_dominate_autotune_latest")


def load_module(file_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, KERNEL_DIR / f"{file_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True,
                        choices=["padded_copy_gather", "padded_copy_scatter", "padded_copy_wgrad"])
    parser.add_argument("--mode", required=True,
                        choices=["costmodel", "camodel_simt_only", "camodel_simd"])
    args = parser.parse_args()

    import torch
    import torch_npu
    from triton.runtime import driver as triton_driver
    from triton.runtime.jit import JITFunction

    torch_npu.npu.set_device(0)
    props = triton_driver.active.utils.get_device_properties(torch.npu.current_device())
    vec_cores = int(props.get("num_vectorcore", 56))

    original_run = JITFunction.run

    def run_with_hint(self, *f_args, **f_kwargs):
        f_kwargs.setdefault("physical_vector_core_count_hint", vec_cores)
        return original_run(self, *f_args, **f_kwargs)

    JITFunction.run = run_with_hint

    sl, hs, ne, top_k = 4, 2, 2, 1
    x = torch.ones((sl, hs), dtype=torch.float16)
    top_expert = torch.ones((sl * top_k,), dtype=torch.int32)
    bin_ids, indices = torch.sort(top_expert)
    tokens = torch.bincount(top_expert, minlength=ne).to(torch.int32)
    padded = ((tokens + 127) // 128) * 128
    padded_bins = torch.cumsum(padded, dim=0).to(torch.int32)
    bins = torch.cumsum(tokens, dim=0).to(torch.int32)
    weights = torch.ones((sl * top_k,), dtype=torch.float16)
    grads = torch.ones((sl, hs), dtype=torch.float16)

    x_npu = x.to("npu")
    indices_npu = indices.to("npu")
    bin_ids_npu = bin_ids.to("npu")
    bins_npu = bins.to("npu")
    padded_bins_npu = padded_bins.to("npu")
    weights_npu = weights.to("npu")
    grads_npu = grads.to("npu")

    if args.kernel == "padded_copy_gather":
        mod = load_module("npu_padded_copy_gather", "k_tiny_gather")
        autotuner = mod._padded_copy_gather
        out_rows = int(padded_bins[-1].item())
        out = torch.empty((out_rows, hs), dtype=torch.float16, device="npu")
        call = dict(
            a=x_npu, b=out, indices=indices_npu, bin_ids=bin_ids_npu, weights=None,
            bins=bins_npu, padded_bins=padded_bins_npu,
            NUM_COLUMNS=hs, TOP_K=top_k, A_TO_B=True, SCALE=False,
        )
    elif args.kernel == "padded_copy_scatter":
        mod = load_module("npu_padded_copy_scatter", "k_tiny_scatter")
        autotuner = mod._padded_copy_scatter
        out = torch.empty((sl * top_k, hs), dtype=torch.float16, device="npu")
        call = dict(
            a=out, b=x_npu, indices=indices_npu, bin_ids=bin_ids_npu, weights=weights_npu,
            bins=bins_npu, padded_bins=padded_bins_npu,
            NUM_COLUMNS=hs, TOP_K=top_k, A_TO_B=False, SCALE=True,
        )
    else:
        mod = load_module("npu_padded_copy_scatter_wgrad_camodel", "k_tiny_wgrad")
        autotuner = mod._padded_copy_wgrad
        out = torch.empty((sl * top_k,), dtype=torch.float16, device="npu")
        call = dict(
            x=x_npu, grad=grads_npu, wgrad=out, indices=indices_npu,
            bin_ids=bin_ids_npu, bins=bins_npu, padded_bins=padded_bins_npu,
            NUM_COLUMNS=hs, TOP_K=top_k,
        )

    autotuner.configs = autotuner.configs[:1]  # BLOCK_X=64, superblock_factor=1, num_warps=1
    autotuner[(sl * top_k,)](**call, logical_program_count_hint=sl * top_k)
    torch_npu.npu.synchronize()
    print(f"LAUNCHED {args.kernel} mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
