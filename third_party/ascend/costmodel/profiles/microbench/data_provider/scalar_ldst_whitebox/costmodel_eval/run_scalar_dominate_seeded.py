#!/usr/bin/env python3
"""Run one scalar-dominated Megablocks kernel at a fixed shape/config.

Used both for costmodel report generation (TRITON_ASCEND_AUTO_SIMT_SCOPE=report)
and for CAModel (compile_mode=simd or simt_only + msopprof).

Fixed launch config: BLOCK_X=64, superblock_factor=1, num_warps=1.
Default shape: sl=4, hs=256, ne=4, top_k=2.
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


def force_one_config(autotuner):
    chosen = None
    for c in autotuner.configs:
        if c.num_warps == 1 and c.kwargs.get("superblock_factor") == 1 and c.kwargs.get("BLOCK_X") == 64:
            chosen = c
            break
    if chosen is None:
        for c in autotuner.configs:
            if c.num_warps == 1 and c.kwargs.get("superblock_factor") == 1:
                chosen = c
                break
    if chosen is None:
        raise SystemExit("no BLOCK_X=64/superblock_factor=1/num_warps=1 config")
    autotuner.configs = [chosen]
    print(f"CONFIG BLOCK_X={chosen.kwargs.get('BLOCK_X')} superblock_factor="
          f"{chosen.kwargs.get('superblock_factor')} num_warps={chosen.num_warps}", flush=True)


def padded_data(sl, hs, ne, top_k):
    import torch
    x = torch.randn((sl, hs), dtype=torch.float16)
    top_expert = torch.randint(0, ne, (sl * top_k,), dtype=torch.int32)
    bin_ids, indices = torch.sort(top_expert)
    tokens = torch.bincount(top_expert, minlength=ne).to(torch.int32)
    padded = ((tokens + 127) // 128) * 128
    padded_bins = torch.cumsum(padded, dim=0).to(torch.int32)
    bins = torch.cumsum(tokens, dim=0).to(torch.int32)
    weights = torch.rand((sl * top_k,), dtype=torch.float16)
    grads = torch.randn((sl, hs), dtype=torch.float16)
    return dict(
        sl=sl, hs=hs, ne=ne, top_k=top_k,
        x=x.to("npu"), indices=indices.to("npu"), bin_ids=bin_ids.to("npu"),
        bins=bins.to("npu"), padded_bins=padded_bins.to("npu"),
        weights=weights.to("npu"), grads=grads.to("npu"),
    )


def binned_data(sl, hs, ne, top_k):
    import torch
    num_experts = ne
    expert_capacity = (sl * top_k) // ne
    x = torch.randn((sl, hs), dtype=torch.float16)
    top_expert = torch.randint(0, ne, (sl * top_k,), dtype=torch.int32)
    bin_ids, indices = torch.sort(top_expert)
    tokens = torch.bincount(top_expert, minlength=ne).to(torch.int32)
    bins = torch.cumsum(tokens, dim=0).to(torch.int32)
    weights = torch.rand((sl * top_k,), dtype=torch.float16)
    x3 = torch.randn((num_experts, expert_capacity, hs), dtype=torch.float16)
    grad = torch.randn((sl, hs), dtype=torch.float16)
    return dict(
        sl=sl, hs=hs, ne=ne, top_k=top_k,
        num_experts=num_experts, expert_capacity=expert_capacity,
        x=x.to("npu"), x3=x3.to("npu"), grad=grad.to("npu"),
        indices=indices.to("npu"), bin_ids=bin_ids.to("npu"),
        bins=bins.to("npu"), weights=weights.to("npu"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=[
        "padded_copy_gather", "padded_copy_scatter", "padded_copy_wgrad",
        "binned_copy_gather", "binned_copy_scatter", "binned_copy_wgrad",
    ])
    ap.add_argument("--sl", type=int, default=4)
    ap.add_argument("--hs", type=int, default=256)
    ap.add_argument("--ne", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch_npu
    import triton
    from triton.runtime import driver as triton_driver
    from triton.runtime.jit import JITFunction

    torch.manual_seed(args.seed)
    torch_npu.npu.set_device(0)
    try:
        props = triton_driver.active.utils.get_device_properties(torch.npu.current_device())
        vec_cores = int(props.get("num_vectorcore", 0))
    except Exception:
        vec_cores = 0

    original_run = JITFunction.run

    def run_with_hint(self, *f_args, **f_kwargs):
        if vec_cores and "physical_vector_core_count_hint" not in f_kwargs:
            f_kwargs["physical_vector_core_count_hint"] = vec_cores
        return original_run(self, *f_args, **f_kwargs)

    JITFunction.run = run_with_hint

    k = args.kernel
    if k == "padded_copy_gather":
        mod = load_module("npu_padded_copy_gather", "k_pg")
        force_one_config(mod._padded_copy_gather)
        d = padded_data(args.sl, args.hs, args.ne, args.top_k)
        print("SEED", args.seed, "bin_ids0", int(d["bin_ids"][0]), "bins", d["bins"].tolist(), "padded_bins", d["padded_bins"].tolist(), flush=True)
        mod.padded_gather(d["x"], d["indices"], d["bin_ids"], None,
                          d["bins"], d["padded_bins"], args.top_k)
    elif k == "padded_copy_scatter":
        mod = load_module("npu_padded_copy_scatter", "k_ps")
        force_one_config(mod._padded_copy_scatter)
        d = padded_data(args.sl, args.hs, args.ne, args.top_k)
        print("SEED", args.seed, "bin_ids0", int(d["bin_ids"][0]), "bins", d["bins"].tolist(), "padded_bins", d["padded_bins"].tolist(), flush=True)
        mod.padded_scatter(d["x"], d["indices"], d["weights"], d["bin_ids"],
                           d["bins"], d["padded_bins"], args.top_k)
    elif k == "padded_copy_wgrad":
        mod = load_module("npu_padded_copy_scatter_wgrad_camodel", "k_pw")
        force_one_config(mod._padded_copy_wgrad)
        d = padded_data(args.sl, args.hs, args.ne, args.top_k)
        print("SEED", args.seed, "bin_ids0", int(d["bin_ids"][0]), "bins", d["bins"].tolist(), "padded_bins", d["padded_bins"].tolist(), flush=True)
        mod.padded_scatter_wgrad(d["x"], d["grads"], d["indices"], d["bin_ids"],
                                 d["bins"], d["padded_bins"], args.top_k)
    elif k == "binned_copy_gather":
        mod = load_module("npu_prec_binned_kernel_simd_modified", "k_bg")
        force_one_config(mod._binned_copy_gather)
        d = binned_data(args.sl, args.hs, args.ne, args.top_k)
        print("SEED", args.seed, "bins", d["bins"].tolist(), flush=True)
        mod.binned_gather(d["x"], d["indices"], None, d["bins"],
                          d["expert_capacity"], args.top_k)
    elif k == "binned_copy_scatter":
        mod = load_module("npu_prec_binned_kernel_simd_modified", "k_bs")
        force_one_config(mod._binned_copy_scatter)
        d = binned_data(args.sl, args.hs, args.ne, args.top_k)
        print("SEED", args.seed, "bins", d["bins"].tolist(), flush=True)
        mod.binned_scatter(d["x3"], d["indices"], None, d["bins"], args.top_k)
    elif k == "binned_copy_wgrad":
        mod = load_module("npu_prec_binned_kernel_simd_modified", "k_bw")
        force_one_config(mod._binned_copy_wgrad)
        d = binned_data(args.sl, args.hs, args.ne, args.top_k)
        print("SEED", args.seed, "bins", d["bins"].tolist(), flush=True)
        mod.binned_scatter_wgrad(d["x3"], d["grad"], d["indices"], d["bins"], args.top_k)
    else:
        raise SystemExit(k)

    torch_npu.npu.synchronize()
    print(f"LAUNCHED {k} shape=({args.sl},{args.hs},{args.ne},{args.top_k})", flush=True)


if __name__ == "__main__":
    main()
