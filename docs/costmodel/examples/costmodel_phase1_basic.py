#!/usr/bin/env python3
"""
Phase 1: CostModel 基础验证
目标: 确认 costmodel 能跑，理解输入输出格式，与硬件实测数据对比

用法:
  python costmodel_phase1_basic.py          # 单独跑 costmodel
  python costmodel_phase1_basic.py --bench costmodel_phase1_bench.py  # 跑对比

前提:
  1. 已 pip install triton-ascend (或已从源码 build)
  2. 有 torch + torch_npu 环境 (benchmark 才需要)
"""

import json
import re
import sys


def make_ascend_ir_add(N, data_type='f32'):
    """生成: ascend.vector_load → ascend.add → ascend.vector_store"""
    bytes_val = N * (2 if data_type == 'f16' else 4)
    return f"""
module {{
  func.func @main(%arg0: tensor<{N}x{data_type}>, %arg1: tensor<{N}x{data_type}>) -> tensor<{N}x{data_type}> {{
    %0 = ascend.vector_load %arg0 {{bytes = {bytes_val} : i64}} : tensor<{N}x{data_type}> -> tensor<{N}x{data_type}>
    %1 = ascend.add %0, %arg1 : (tensor<{N}x{data_type}>, tensor<{N}x{data_type}>) -> tensor<{N}x{data_type}>
    ascend.vector_store %1 {{bytes = {bytes_val} : i64}} : tensor<{N}x{data_type}>
    return %1 : tensor<{N}x{data_type}>
  }}
}}"""


def make_ascend_ir_op(op_name, N, data_type='f32'):
    """生成各种单算子 (纯计算，无 load/store)"""
    bytes_val = N * (2 if data_type == 'f16' else 4)
    unary_ops = {'exp', 'log', 'sqrt', 'rsqrt', 'tanh', 'sigmoid', 'relu', 'neg', 'abs', 'cast', 'reduce_sum'}
    cmp_ops = {'cmp_eq', 'cmp_ne', 'cmp_lt', 'cmp_le', 'cmp_gt', 'cmp_ge'}
    ternary_ops = {'select'}

    if op_name in unary_ops:
        inner = f"%r = ascend.{op_name} %arg0 : tensor<{N}x{data_type}> -> tensor<{N}x{data_type}"
    elif op_name in cmp_ops:
        inner = f"%r = ascend.{op_name} %arg0, %arg1 : (tensor<{N}x{data_type}>, tensor<{N}x{data_type}>) -> tensor<{N}xi1>"
    elif op_name in ternary_ops:
        inner = f"%r = ascend.{op_name} %arg0, %arg1, %arg1 : tensor<{N}x{data_type}>, tensor<{N}x{data_type}>, tensor<{N}x{data_type}> -> tensor<{N}x{data_type}>"
    else:
        inner = f"%r = ascend.{op_name} %arg0, %arg1 : (tensor<{N}x{data_type}>, tensor<{N}x{data_type}>) -> tensor<{N}x{data_type}"

    return f"""
module {{
  func.func @main(%arg0: tensor<{N}x{data_type}>, %arg1: tensor<{N}x{data_type}>) -> tensor<{N}x{data_type}> {{
    {inner}
    return %r : tensor<{N}x{data_type}>
  }}
}}"""


def run_costmodel(mlir_text, label, extra_args=None):
    """运行 costmodel 并返回 latency (微秒)"""
    from triton._C.libtriton import ascend as ascend_capi

    args = list(extra_args or [])
    if "-ascend-perf-model" not in args:
        args.insert(0, "-ascend-perf-model")
    if "-allow-unregistered-dialect" not in args:
        args.append("-allow-unregistered-dialect")

    print(f"  {label:>50} ... ", end="", flush=True)
    try:
        result = ascend_capi.run_costmodel_inproc(mlir_text, args)
        match = re.search(r"Estimated Time:\s+([0-9.]+)\s*us", result)
        latency = float(match.group(1)) if match else float("inf")
        print(f"{latency:10.4f} us")
        return latency
    except Exception as e:
        print(f"ERROR: {e}")
        return None


# ============================================================
# 测试 1: Scaling —— 匹配 bench.py 的 benchmark_scaling 大小
# ============================================================
def test_scaling():
    print(f"\n{'='*60}")
    print("测试 1: Load→Add→Store 不同大小 (匹配 bench.py scaling)")
    print(f"{'='*60}")

    sizes = [256, 1024, 4096, 16384, 65536, 262144]
    results = {}
    for N in sizes:
        mlir = make_ascend_ir_add(N, 'f32')
        label = f"Load→Add→Store {N} FP32"
        results[N] = run_costmodel(mlir, label)
    return {"scaling_fp32": results}


# ============================================================
# 测试 2: 不同算子 —— 匹配 bench.py 的各种 ops
# ============================================================
def test_ops():
    print(f"\n{'='*60}")
    print("测试 2: 不同算子 65536 FP32 (匹配 bench.py benchmark_various_ops)")
    print(f"{'='*60}")

    N = 65536
    ops = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid',
           'relu', 'neg', 'abs', 'reduce_sum']

    bitwidth = 32  # FP32
    vector_width = 2048 // bitwidth  # 64
    print(f"\n  理论背景 (从 AscendModelOps.cpp:57-67):")
    print(f"    vectorWidth = {2048} / {bitwidth} = {vector_width} elements/op")
    print(f"    {N} elements → ceil({N}/{vector_width}) = {N//vector_width} ops")
    print(f"    add: {N//vector_width} * 1 + startup cycles")
    print(f"    exp: {N//vector_width} * 9 + startup cycles (校准值)")
    print(f"    log: {N//vector_width} * 12 + startup cycles")

    results = {}
    for op_name in ops:
        mlir = make_ascend_ir_op(op_name, N, 'f32')
        results[op_name] = run_costmodel(mlir, f"{op_name:<12} {N} FP32")
    return results


# ============================================================
# 综合对比 (如果提供了 bench.py 路径)
# ============================================================
def print_comparison():
    print(f"\n{'='*70}")
    print("对比说明")
    print(f"{'='*70}")
    print("""
  costmodel 预估的是"纯计算/搬运"时间，硬件实测包含了:
    - 硬件计算/搬运时间 (costmodel 建模)
    - Kernel launch overhead (≈1-5 us, costmodel 不建模)
    - 驱动开销 + PCIe 数据传输 (costmodel 不建模)
    - Pipeline 调度开销 (PipeAnalysis 部分建模)

  合理的预期:
    costmodel 预估 < 硬件总耗时
    costmodel 预估 ≈ 硬件总耗时 - launch_overhead (当数据量足够大时)

  首次验证: 跑大尺寸 (65536+), 看 costmodel 和 (真实-lanch_overhead) 的比例
""")

    print(f"\n{'='*70}")
    print("使用建议")
    print(f"{'='*70}")
    print("""
  1. 先在 910B 服务器上跑: python costmodel_phase1_basic.py
  2. 再跑硬件 benchmark:   python costmodel_phase1_bench.py
  3. 对比两边的输出:
     - 同尺寸下 costmodel 是否远小于真实值? (预期: 是, 没算 overhead)
     - 不同算子的相对比例是否正确? (add≈sub≈mul < div < exp < log < tanh)
     - scaling 趋势是否一致? (costmodel 和真实都是 O(N) 增长)
""")


def main():
    results = {}

    test_scaling_result = test_scaling()
    results.update(test_scaling_result)

    ops_result = test_ops()
    results["ops_fp32_65536"] = ops_result

    # 输出汇总
    print(f"\n{'='*70}")
    print("CostModel 结果汇总")
    print(f"{'='*70}")

    scaling = results.get("scaling_fp32", {})
    print("\nScaling (Load→Add→Store FP32):")
    print(f"  {'Size':>10}  {'CostModel (us)':>15}")
    print(f"  {'-'*10}  {'-'*15}")
    for N in [256, 1024, 4096, 16384, 65536, 262144]:
        v = scaling.get(N)
        if v is not None:
            print(f"  {N:>10}  {v:15.4f}")

    ops = results.get("ops_fp32_65536", {})
    if ops:
        print(f"\n不同算子 (65536 FP32):")
        base = ops.get("add", 1)
        print(f"  {'Op':>12}  {'CostModel (us)':>15}  {'vs add':>8}")
        print(f"  {'-'*12}  {'-'*15}  {'-'*8}")
        for name in ['add', 'sub', 'mul', 'div', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid']:
            v = ops.get(name)
            if v is not None and base:
                print(f"  {name:>12}  {v:15.4f}  {v/base:6.2f}x")

    print_comparison()


if __name__ == "__main__":
    main()
