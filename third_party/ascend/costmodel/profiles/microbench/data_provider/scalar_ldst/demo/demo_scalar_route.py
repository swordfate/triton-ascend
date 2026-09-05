#!/usr/bin/env python3
"""Demo: compare StageCostModel predicted cycles with on-device measured time.

The script builds a scalar-heavy kernel (padded copy gather) and a mixed
scalar+vector/dot kernel, dumps the route report, prints predicted route and
stage cycles, and measures median wall time.

Core-count note:
- This real Ascend950PR reports num_vectorcore=56.
- If you run the same kernel through CAModel/simulator and the simulator reports
  32 or 64 vector cores, do NOT reuse the 56 hint directly.  The cost model
  uses physical_vector_core_count_hint to compute runtime_wave_count:
      wave_count = ceil(logical_program_count / superblock_factor / physical_cores)
  Therefore use the simulator's core count when comparing simulator outputs,
  or apply an explicit wave-count conversion when comparing real-NPU and
  simulator results.
"""
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


def _vector_core_count():
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(props["num_vectorcore"])


def _launch_options(report_path, logical_programs, num_warps=1):
    return {
        "num_warps": num_warps,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
        "enable_auto_blockify": True,
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": _vector_core_count(),
    }


@triton.jit
def padded_copy_gather(
    a, b, indices, bin_ids, weights, bins, padded_bins,
    NUM_COLUMNS: tl.constexpr, TOP_K: tl.constexpr, BLOCK_X: tl.constexpr,
    SCALE: tl.constexpr,
):
    index_a = tl.load(indices + tl.program_id(0))
    bin_idx = tl.load(bin_ids + tl.program_id(0))
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)
    offset = index_a // TOP_K
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    scale = tl.load(weights + index_a) if SCALE else 1
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(a + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(b + offsets, x.to(b.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


@triton.jit
def gather_dot_min(
    a_ptr, b_ptr, indices_ptr, out_ptr, M, N, stride_am, stride_ak,
    stride_bk, stride_bn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gather_k = tl.load(indices_ptr + offs_k)
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + gather_k[None, :] * stride_ak)
    b = tl.load(b_ptr + gather_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    result = tl.dot(a, b)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, result)


def _report_summary(report_path):
    with open(report_path) as f:
        report = json.load(f)
    sm = report.get("stage_model", {})
    print("effective_decision_kind:", report.get("effective_decision_kind"))
    print("recommended_decision_kind:", report.get("recommended_decision_kind"))
    print("stage_model.applied:", sm.get("applied"))
    if sm.get("routes"):
        for route_name, route in sm["routes"].items():
            if route.get("legal"):
                print(f"  route {route_name}: total_system_cycles={route.get('total_system_cycles')}")
    for st in sm.get("logical_stages", []):
        model = st.get("model")
        impls = st.get("implementations", [])
        print(f"  stage {st.get('id')} model={model} iterations={st.get('iteration_count')}")
        for impl in impls:
            mode = impl["implementation"]["mode"]
            cycles = impl.get("total_system_cycles")
            scalar_mem = impl.get("resource_system_cycles", {}).get("scalar_memory_per_iteration")
            print(f"    {mode}: cycles={cycles} scalar_memory_per_iter={scalar_mem}")
    return report


def _measure_ms(launch, repeat=50, warmup=10):
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()
    starts = []
    ends = []
    for _ in range(repeat):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        torch.npu.synchronize()
        starts.append(s)
        ends.append(e)
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.median(times)


def run_scalar_heavy(tmpdir):
    SL, HS, NE, TOP_K, BLOCK_X = 1024, 1536, 128, 4, 64
    capacity = (SL * TOP_K) // NE
    torch.manual_seed(0)
    x = torch.randn((SL, HS), dtype=torch.float16, device="npu")
    top = torch.randint(0, NE, (SL * TOP_K,), dtype=torch.int32, device="npu")
    bin_ids, indices = torch.sort(top)
    tokens = torch.bincount(top, minlength=NE)
    bins = torch.cumsum(tokens, dim=0).to(torch.int32)
    padded = torch.div(tokens + 127, 128, rounding_mode="trunc") * 128
    padded_bins = torch.cumsum(padded, dim=0).to(torch.int32)
    weights = torch.rand((SL * TOP_K,), dtype=torch.float16, device="npu")
    out = torch.zeros((padded_bins[-1].item(), HS), dtype=torch.float16, device="npu")
    programs = indices.shape[0]
    report = Path(tmpdir) / "scalar_heavy_route.json"
    launch = lambda: padded_copy_gather[(programs,)](
        x, out, indices, bin_ids, weights, bins, padded_bins,
        NUM_COLUMNS=HS, TOP_K=TOP_K, BLOCK_X=BLOCK_X, SCALE=True,
        **_launch_options(report, programs),
    )
    launch()
    torch.npu.synchronize()
    print("\n--- scalar-heavy padded_copy_gather ---")
    _report_summary(report)
    ms = _measure_ms(launch)
    print(f"measured median: {ms:.3f} ms")


def run_mixed(tmpdir):
    logical = _vector_core_count()
    block = 16
    source_k = 256
    a = torch.randn((logical * block, source_k), dtype=torch.float16, device="npu")
    b = torch.randn((source_k, block), dtype=torch.float16, device="npu")
    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255, 1, 2, 3, 4, 6, 7, 8, 9],
                           dtype=torch.int32, device="npu")
    out = torch.empty((logical * block, block), dtype=torch.float32, device="npu")
    report = Path(tmpdir) / "mixed_route.json"
    launch = lambda: gather_dot_min[(logical, 1)](
        a, b, indices, out, a.shape[0], b.shape[1], a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block, BLOCK_N=block, BLOCK_K=block,
        **_launch_options(report, logical, num_warps=4),
    )
    launch()
    torch.npu.synchronize()
    print("\n--- mixed scalar-gather + dot ---")
    _report_summary(report)
    ms = _measure_ms(launch)
    print(f"measured median: {ms:.3f} ms")


def main():
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    print(f"real device num_vectorcore = {_vector_core_count()}")
    with tempfile.TemporaryDirectory() as td:
        run_scalar_heavy(td)
        run_mixed(td)


if __name__ == "__main__":
    main()
