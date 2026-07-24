#!/usr/bin/env python3
"""
Phase 1: 硬件 Benchmark —— 在真 Ascend NPU 上测同样的操作
目标: 拿到真实执行时间，和 costmodel 预估对比

用法: python costmodel_phase1_bench.py
前提: 有 Ascend NPU 硬件 + torch_npu 环境


与 costmodel_phase1_basic.py 的对应关系:
  bench.py 的测试           ↔  basic.py 的测试           数据大小
  ─────────────────────────────────────────────────────────────
  benchmark_pure_add        ↔  test_scaling 各尺寸        256~262144
  benchmark_load_add_store  ↔  test_scaling 各尺寸        256~262144
  benchmark_loop_overhead   ↔  test_scaling (不同大小)     256~262144
  benchmark_various_ops     ↔  test_ops (各算子)          65536

对比时注意: costmodel 预估的是纯计算/搬运时间，不含 launch overhead。
大尺寸 (65536+) 下 launch overhead 可忽略，对比才有意义。
"""

import torch
import torch_npu
import time
import json


def time_kernel(kernel_fn, warmup=5, rep=100):
    """测 kernel 平均执行时间 (微秒)"""
    # warmup
    for _ in range(warmup):
        kernel_fn()

    # 正式测量
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        kernel_fn()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) / rep * 1e6  # 转为微秒
    return elapsed


def benchmark_pure_add():
    """测试: 纯 element-wise add (4096 elements, 无 loop overhead)"""
    size = 4096
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)

    def kernel():
        z = x + y

    elapsed = time_kernel(kernel)
    print(f"  Pure Add ({size} elements): {elapsed:.3f} us")
    print(f"    理论最小: {size / 128 / 1850:.3f} us (128-wide @ 1.85GHz)")
    return elapsed


def benchmark_load_add_store():
    """测试: 先 load, 计算, 再 store (模拟三个操作)"""
    size = 4096
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    def kernel():
        tmp = x + y          # load x, load y, add, result in register
        out.copy_(tmp)        # store to memory

    elapsed = time_kernel(kernel)
    print(f"  Load→Add→Store ({size} elements): {elapsed:.3f} us")
    return elapsed


def benchmark_loop_overhead():
    """测试: 不同大小下的 scaling 行为"""
    results = {}
    for size in [256, 1024, 4096, 16384, 65536, 262144]:
        x = torch.randn(size, device="npu", dtype=torch.float32)
        y = torch.randn(size, device="npu", dtype=torch.float32)

        def kernel():
            _ = x + y

        elapsed = time_kernel(kernel, rep=max(20, int(10000 / size)))
        per_element = elapsed / size * 1e3  # ns per element
        results[f"size_{size}"] = {
            "total_us": round(elapsed, 3),
            "ns_per_element": round(per_element, 3),
        }
        print(f"  Size {size:>6}: {elapsed:8.3f} us  ({per_element:6.3f} ns/element)")

    return results


def benchmark_various_ops():
    """测试: 不同算子类型的耗时差异"""
    size = 65536
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)

    ops = {
        "add": lambda: x + y,
        "sub": lambda: x - y,
        "mul": lambda: x * y,
        "div": lambda: x / y,
        "exp": lambda: torch.exp(x),
        "sqrt": lambda: torch.sqrt(x),
        "tanh": lambda: torch.tanh(x),
        "sigmoid": lambda: torch.sigmoid(x),
    }

    results = {}
    base_time = time_kernel(lambda: x + y)  # 用 add 做 baseline

    for name, fn in ops.items():
        elapsed = time_kernel(fn, rep=max(20, int(5000 * 256 / size)))
        ratio = elapsed / base_time
        results[name] = {
            "us": round(elapsed, 3),
            "ratio_vs_add": round(ratio, 2),
        }
        print(f"  {name:>8}: {elapsed:8.3f} us  (vs add: {ratio:.2f}x)")

    return results


def main():
    print("=" * 60)
    print("Phase 1: 硬件 Benchmark")
    print("=" * 60)

    all_results = {}

    print("\n--- 测试 1: Pure Add ---")
    all_results["pure_add"] = benchmark_pure_add()

    print("\n--- 测试 2: Load→Add→Store ---")
    all_results["load_add_store"] = benchmark_load_add_store()

    print("\n--- 测试 3: Scaling (不同大小下每元素耗时) ---")
    all_results["scaling"] = benchmark_loop_overhead()

    print("\n--- 测试 4: 不同算子耗时对比 ---")
    all_results["op_types"] = benchmark_various_ops()

    print(f"\n{'='*60}")
    print("硬件 Benchmark 完成！")
    print(f"请将以上结果与 costmodel_phase1_basic.py 的输出对比。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
