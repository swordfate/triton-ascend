import pytest
import torch
import time
import psutil
import os

# ===========================
# 双标杆测试
# ===========================

@pytest.mark.skipif(torch.cuda.is_available(), reason="Only run on CPU or Ascend NPU")
def test_bfloat16_gradient_known_issue_double_benchmark():
    """
    双标杆测试：
    1. 功能标杆：验证 bfloat16 梯度计算是否正确
    2. 性能标杆：验证是否在非 GPU 设备上运行，且资源使用合理
    """

    # ✅ 1. 功能标杆：验证计算逻辑正确
    device = torch.device("cpu")  # 显式指定 CPU
    # 若在 Ascend NPU 上运行，可改为：device = torch.device("npu")

    x = torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True, device=device)
    y = x.sum()
    y.backward()

    # 验证梯度是否正确（应为全 1）
    expected_grad = torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad, atol=1e-3), \
        f"Gradient mismatch: expected {expected_grad}, got {x.grad}"

    # ✅ 2. 性能标杆：验证运行环境 + 资源使用情况

    # --- 检查是否真的在非 GPU 上运行 ---
    assert not torch.cuda.is_available(), "GPU is unexpectedly available!"

    # --- 获取当前进程资源使用情况 ---
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()

    # 执行计算（已执行，但为性能计时，再跑一次）
    x = torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True, device=device)
    y = x.sum()
    y.backward()

    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB

    # --- 性能标杆验证 ---
    execution_time = end_time - start_time
    memory_usage = memory_after - memory_before

    # 可设定合理阈值（根据实际硬件调整）
    assert execution_time < 1.0, f"Execution time too long: {execution_time:.3f}s"
    assert memory_usage < 50, f"Memory usage too high: {memory_usage:.2f} MB"

    # ✅ 输出双标杆结果（可选：用于日志或 CI 报告）
    print(f"✅ Double Benchmark Passed:")
    print(f"   - Device: {device}")
    print(f"   - Execution Time: {execution_time:.3f}s")
    print(f"   - Memory Usage: {memory_usage:.2f} MB")
    print(f"   - Gradient Correct: True")
