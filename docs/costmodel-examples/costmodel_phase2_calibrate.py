#!/usr/bin/env python3
"""
Phase 2: 单个算子精度校准
目标: 手写 AscendModel IR，逐个算子对比 costmodel vs 真实硬件

用法: python costmodel_phase2_calibrate.py
前提: Phase 1 已跑通
"""

import json
import torch
import torch_npu
import time
import re
from triton._C.libtriton import ascend as ascend_capi


# ============================================================
# CostModel 预估
# ============================================================

def costmodel_templates():
    """返回各种算子的 AscendModel IR 模板.
    N 是元素数, 会自动填入 bytes = N * sizeof(float) = N * 4
    """
    def _build(op_ir, N, is_compute_only=False):
        """生成完整 MLIR module"""
        bytes_val = N * 4
        if is_compute_only:
            # 纯计算 (无 load/store)
            inner = op_ir
        else:
            # 带 load/store (更接近真实)
            inner = f"""
    %load = ascend.vector_load %arg0 {{bytes = {bytes_val} : i64}} : tensor<{N}xf32> -> tensor<{N}xf32>
    %compute = {op_ir}
    ascend.vector_store %compute {{bytes = {bytes_val} : i64}} : tensor<{N}xf32>
    tf.return %compute : tensor<{N}xf32>
"""
        return f"""
module {{
  func.func @main(%arg0: tensor<{N}xf32>, %arg1: tensor<{N}xf32>) -> tensor<{N}xf32> {{
    {inner}
  }}
}}"""

    templates = {}

    # --- 纯计算算子 (无 load/store) ---
    for name, op_ir in [
        ("add", "%r = ascend.add %arg0, %arg1 : (tensor<{N}xf32>, tensor<{N}xf32>) -> tensor<{N}xf32"),
        ("sub", "%r = ascend.sub %arg0, %arg1 : (tensor<{N}xf32>, tensor<{N}xf32>) -> tensor<{N}xf32"),
        ("mul", "%r = ascend.mul %arg0, %arg1 : (tensor<{N}xf32>, tensor<{N}xf32>) -> tensor<{N}xf32"),
        ("div", "%r = ascend.div %arg0, %arg1 : (tensor<{N}xf32>, tensor<{N}xf32>) -> tensor<{N}xf32"),
        ("exp", "%r = ascend.exp %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("log", "%r = ascend.log %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("sqrt", "%r = ascend.sqrt %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("tanh", "%r = ascend.tanh %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("sigmoid", "%r = ascend.sigmoid %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("relu", "%r = ascend.relu %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("neg", "%r = ascend.neg %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("abs", "%r = ascend.abs %arg0 : tensor<{N}xf32> -> tensor<{N}xf32"),
        ("reduce_sum", "%r = ascend.reduce_sum %arg0 axis 0 : tensor<{N}xf32> -> tensor<1xf32"),
    ]:
        # Create both compute_only and full (load+compute+store) versions
        inner_compute = f"""{op_ir}
    return %r : tensor<{N}xf32>""".replace("{N}", str(N))
        inner_full = f"""%load = ascend.vector_load %arg0 {{bytes = {N*4} : i64}} : tensor<{N}xf32> -> tensor<{N}xf32>
    {op_ir}
    ascend.vector_store %r {{bytes = {N*4} : i64}} : tensor<{N}xf32>
    return %r : tensor<{N}xf32>""".replace("{N}", str(N))

        templates[f"{name}_compute"] = f"""
module {{
  func.func @main(%arg0: tensor<{N}xf32>, %arg1: tensor<{N}xf32>) -> tensor<{N}xf32> {{
    {inner_compute}
  }}
}}"""

        templates[f"{name}_full"] = f"""
module {{
  func.func @main(%arg0: tensor<{N}xf32>, %arg1: tensor<{N}xf32>) -> tensor<{N}xf32> {{
    {inner_full}
  }}
}}"""

    return templates


def run_costmodel_for_op(mlir_text):
    """跑 costmodel，返回微秒数"""
    try:
        output = ascend_capi.run_costmodel_inproc(mlir_text,
            ["-ascend-perf-model", "-allow-unregistered-dialect"])
        match = re.search(r"Estimated Time:\s+([0-9.]+)\s*us", output)
        return float(match.group(1)) if match else float("inf")
    except Exception as e:
        print(f"    CostModel Error: {e}")
        return None


# ============================================================
# 硬件 Benchmark
# ============================================================

def time_kernel(kernel_fn, warmup=10, rep=100):
    torch.npu.synchronize()
    for _ in range(warmup):
        kernel_fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        kernel_fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / rep * 1e6


def real_benchmarks():
    """在真 NPU 上测所有算子"""
    results = {}
    # 用大 tensor 测，降低 overhead 占比
    N = 65536
    x = torch.randn(N, device="npu", dtype=torch.float32)
    y = torch.randn(N, device="npu", dtype=torch.float32)

    ops = {
        "add": lambda: x + y,
        "sub": lambda: x - y,
        "mul": lambda: x * y,
        "div": lambda: x / y,
        "exp": lambda: torch.exp(x),
        "log": lambda: torch.log(x.abs() + 1e-8),
        "sqrt": lambda: torch.sqrt(x.abs()),
        "tanh": lambda: torch.tanh(x),
        "sigmoid": lambda: torch.sigmoid(x),
        "relu": lambda: torch.nn.functional.relu(x),
        "neg": lambda: -x,
        "abs": lambda: torch.abs(x),
        "reduce_sum": lambda: x.sum(),
    }

    # Baseline: 空操作
    def empty():
        pass
    overhead = time_kernel(empty)
    print(f"  Launch overhead: {overhead:.3f} us")

    for name, fn in ops.items():
        total = time_kernel(fn)
        net = total - overhead  # 减掉 launch overhead
        results[name] = {
            "total_us": round(total, 3),
            "net_us": round(net, 3),
            "us_per_element": round(net / N * 1e3, 5),  # ns/elem
        }
        print(f"  {name:>12}: total={total:9.3f} us, net={net:9.3f} us, "
              f"per_elem={net/N*1e3:.4f} ns")

    return results, overhead


# ============================================================
# 对比分析
# ============================================================

def compare():
    print("=" * 70)
    print("Phase 2: 单算子精度校准")
    print("=" * 70)

    # 1. CostModel 预估
    print("\n>>> CostModel 预估 (Ascend 910B)")
    templates = costmodel_templates()
    costmodel_results = {}
    for name in ["add", "sub", "mul", "div", "exp", "log", "sqrt", "tanh", "sigmoid", "relu"]:
        key = f"{name}_full"  # 用 load+compute+store 版本
        if key in templates:
            us = run_costmodel_for_op(templates[key])
            costmodel_results[name] = us
            if us is not None:
                print(f"  {name:>12}: {us:9.3f} us")

    # 2. 硬件 benchmark
    print(f"\n>>> 硬件 Benchmark (Ascend 910B, N=65536)")
    real_results, overhead = real_benchmarks()

    # 3. 对比
    print(f"\n{'='*70}")
    print(f"{'算子':>12} | {'CostModel':>10} | {'真实(去overhead)':>16} | {'比率':>8} | {'评估'}")
    print("-" * 70)

    baseline_ratio = None
    for name in ["add", "sub", "mul", "div", "exp", "log", "sqrt", "tanh", "sigmoid", "relu"]:
        cm_us = costmodel_results.get(name)
        real_us = real_results.get(name, {}).get("net_us")
        if cm_us and real_us:
            ratio = cm_us / real_us
            if baseline_ratio is None and name == "add":
                baseline_ratio = ratio
            # 用 add 做 anchoring: 假设 add 的比例是整个模型的基础偏差
            normalized_ratio = ratio / baseline_ratio if baseline_ratio else ratio
            status = "✓" if 0.5 < normalized_ratio < 2.0 else "✗ 需校准"
            print(f"  {name:>12} | {cm_us:8.3f} us | {real_us:14.3f} us | {ratio:6.2f}x | {status}")

    print(f"\n解读:")
    print(f"  比率 = costmodel预估 / 真实测量")
    print(f"  add 的比率({baseline_ratio:.2f}x)是整体偏差，其他算子的相对偏差才是重点")
    print(f"  比如: mul 的比率应该接近 add (因为都是 1 cycle)")
    print(f"        div 的比率应该 ≈ add的比率 × (div_cycle/add_cycle)")
    print(f"\n下一步:")
    print(f"  1. 如果 add/sub/mul 的相对比率接近 1.0: 基础 modeling 正确")
    print(f"  2. 如果 div/exp/log 的相对比率偏离: 需调 calibration 里的")
    print(f"     'vector_op_cycles_per_vec_instruction' 参数")
    print(f"  3. 修改 ascend_910b.json 的 calibration 部分, 重新跑本脚本")


if __name__ == "__main__":
    compare()
