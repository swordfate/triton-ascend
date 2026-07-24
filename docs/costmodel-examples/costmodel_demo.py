#!/usr/bin/env python3
"""
CostModel 入门 Demo
====================
基于设计文档中的例子，从简单到复杂逐步演示 costmodel 的使用。

用法: python costmodel_demo.py [level]
  level 1: Hello World - 手写 AscendModel IR (最简单，不需要 TTIR)
  level 2: 设计文档中的 triton_unk_fused_add_1 例子
  level 3: 设计文档中的 fa_with_mask (Flash Attention) 例子
  level 4: 演示 arg-bindings 对 trip count 的影响
  level 5: 演示如何从真实 Triton kernel 获取 TTIR

前提: 已在 Linux 服务器上安装 triton-ascend
"""

import sys
import os


# ============================================================================
# Level 1: Hello World —— 手写 AscendModel IR
# ============================================================================

def demo_level1():
    """
    最简单的情况：手写 AscendModel IR，跳过 TTIR→AscendModel 转换步骤，
    直接测底层 cycle 估算管线。

    设计文档对应：3.3 节 Vector 指令建模
    """
    print("=" * 70)
    print("Level 1: Hello World —— 手写 AscendModel IR")
    print("=" * 70)

    from triton._C.libtriton import ascend

    # 最简单的例子：4 个 float 的 vector add，无 load/store，无循环
    mlir = """
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ascend.add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
    print("\n输入:\n" + mlir)
    print("这个 IR 描述的是: 对 4 个 f32 元素做 element-wise add")
    print("预期: 4 elem / 64 elem_per_cycle = 1 instruction * 1 cycle = 1 cycle + startup_latency")
    print()

    result = ascend.run_costmodel_inproc(mlir,
        ["-ascend-perf-model", "-allow-unregistered-dialect"])
    print("CostModel 输出:")
    print("  " + result.strip())

    # 进阶：加上 load 和 store
    mlir2 = """
module {
  func.func @main(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
    %0 = ascend.vector_load %arg0 {bytes = 1024 : i64} : tensor<256xf32> -> tensor<256xf32>
    %1 = ascend.add %0, %arg1 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    ascend.vector_store %1 {bytes = 1024 : i64} : tensor<256xf32>
    return %1 : tensor<256xf32>
  }
}
"""
    print("\n" + "=" * 70)
    print("进阶: 加上 load 和 store (256 elements)")
    print("=" * 70)
    print("这个 IR 描述: load 256 个 f32 → add → store 256 个 f32")
    print("预期包含了 VecMTE2(load) + Vector(add) + MTE3(store) 的 cycle")
    print()

    result2 = ascend.run_costmodel_inproc(mlir2,
        ["-ascend-perf-model", "-allow-unregistered-dialect"])
    print("CostModel 输出:")
    print("  " + result2.strip())


# ============================================================================
# Level 2: 设计文档的 triton_unk_fused_add_1 例子
# ============================================================================

def demo_level2():
    """
    Level 2: 仓库自带 vecadd.mlir — 真实 Triton 编译产出的 TTIR

    基于 test/Triton/vecadd.mlir, 这是 triton-ascend 编译器对
    标准向量加法 kernel 生成的完整 TTIR。

    通过 -ascend-perf-model 走完整 6-Pass 管线:
    ConvertTritonToAscend → InsertDataTransfers → AssignOpIDs →
    EstimateCycles → PipelineAnalysis → PerfReport
    """
    print("\n" + "=" * 70)
    print("Level 2: vecadd.mlir (标准 Triton kernel TTIR)")
    print("=" * 70)

    from triton._C.libtriton import ascend
    import os

    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ttir_path = os.path.join(repo_dir, "test", "Triton", "vecadd.mlir")
    with open(ttir_path) as f:
        ttir = f.read()

    # 截取第一部分 (去掉注释掉的 tritonGPU IR)
    ttir_clean = ttir.split("\n// module")[0]

    print(f"\n读取: {ttir_path}")
    print(f"\nTTIR 包含:")
    print("  - tt.get_program_id x : i32  →  SPMD program id")
    print("  - tt.make_range / tt.splat   →  地址计算 (无硬件开销)")
    print("  - tt.load / arith.addf        →  vector_load + add")
    print("  - scf.for 循环                →  trip count 由 arg-bindings 确定")
    print("  - tt.store                    →  vector_store")
    print()

    # vecadd.mlir 有带动态 bound 的循环 (scf.for %arg6 to %arg4 step %arg5)
    # 需要 arg-bindings 告诉 costmodel 循环跑多少次
    # %arg4 = 上层传入的循环上限, %arg5 = 步长
    result = ascend.run_costmodel_inproc(ttir_clean, [
        "-ascend-perf-model",
        "-ascend-perf-model=arg-bindings=arg4=1024,arg5=256",
        "-allow-unregistered-dialect",
    ])
    print("CostModel 输出:")
    print("  " + result.strip())


# ============================================================================
# Level 3: 设计文档的 fa_with_mask 例子
# ============================================================================

def demo_level3():
    """
    Level 3: Cube (matmul) + Vector (add) — 跨路径搬运演示

    用 AscendModel IR 直接构造 Cube+Vector 混合场景:
    一个 cube_load → matmul → cube_store → vector_load → add → vector_store
    """
    print("\n" + "=" * 70)
    print("Level 3: Cube + Vector 混合 (跨路径搬运)")
    print("=" * 70)

    from triton._C.libtriton import ascend

    # 直接用 AscendModel IR (跳过 TTIR 转换，专注看跨路径行为)
    mlir = """
module {
  func.func @cube_vector_mix(
      %arg0: tensor<64x64xf16>, %arg1: tensor<64x128xf16>,
      %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {

    %lhs = ascend.cube_load %arg0 {bytes = 8192 : i64}
        : tensor<64x64xf16> -> tensor<64x64xf16>
    %rhs = ascend.cube_load %arg1 {bytes = 16384 : i64}
        : tensor<64x128xf16> -> tensor<64x128xf16>

    %dot = ascend.matmul %lhs, %rhs {M = 64 : i64, N = 128 : i64, K = 64 : i64}
        : (tensor<64x64xf16>, tensor<64x128xf16>) -> tensor<64x128xf32>
    ascend.cube_store %dot {bytes = 32768 : i64} : tensor<64x128xf32>

    %vec_in = ascend.vector_load %arg2 {bytes = 32768 : i64}
        : tensor<64x128xf32> -> tensor<64x128xf32>
    %result = ascend.add %dot, %vec_in
        : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    ascend.vector_store %result {bytes = 32768 : i64} : tensor<64x128xf32>

    return %result : tensor<64x128xf32>
  }
}
"""
    print("\nAscendModel IR (手写, 跳过 TTIR 转换):")
    print("  ascend.cube_load    →  CubeMTE2   (HBM→L1, 8192+16384 bytes)")
    print("  ascend.matmul 64x64x128 →  Cube   (64*128*64 fractal ops)")
    print("  ascend.cube_store   →  FixPipe    (L0C→HBM, 32768 bytes)")
    print("  ascend.vector_load  →  VecMTE2   (HBM→UB)")
    print("  ascend.add          →  Vector    (64*128 elements)")
    print("  ascend.vector_store →  MTE3      (UB→HBM)")
    print("  ★ Cube 路径: max(matmul, cube_load, cube_store)")
    print("  ★ Vector 路径: max(add, vector_load + vector_store)")
    print("  ★ 总: max(Cube路径, Vector路径)")
    print()

    result = ascend.run_costmodel_inproc(mlir, [
        "-ascend-perf-model",
        "-allow-unregistered-dialect",
    ])
    print("CostModel 输出:")
    print("  " + result.strip())


# ============================================================================
# Level 4: arg-bindings 演示
# ============================================================================

def demo_level4():
    """
    设计文档第 6 节：arg-bindings。

    用 vecadd.mlir (含动态 loop), 同一个 TTIR 不同的 arg-bindings
    得到不同预估时间 (因为循环 trip count 不同)。
    """
    print("\n" + "=" * 70)
    print("Level 4: arg-bindings 对 trip count 的影响")
    print("=" * 70)

    from triton._C.libtriton import ascend
    import os

    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ttir_path = os.path.join(repo_dir, "test", "Triton", "vecadd.mlir")
    with open(ttir_path) as f:
        ttir = f.read()
    ttir_clean = ttir.split("\n// module")[0]

    print(f"\n读取: {ttir_path}")
    print("循环: scf.for %arg6 = 0 to %arg4 step %arg5")
    print("  %arg4 = 循环上限, %arg5 = 步长")
    print("  循环内: 2x tt.load + arith.addf + tt.store")
    print()

    test_cases = [
        ("arg4=256,arg5=256",  "trip=1 (256/256)"),
        ("arg4=1024,arg5=256", "trip=4 (1024/256)"),
        ("arg4=4096,arg5=256", "trip=16 (4096/256)"),
    ]

    for bindings, desc in test_cases:
        result = ascend.run_costmodel_inproc(ttir_clean, [
            "-ascend-perf-model",
            "-allow-unregistered-dialect",
            f"-ascend-perf-model=arg-bindings={bindings}",
        ])
        print(f"  arg-bindings: {bindings:>30} ({desc})")
        print(f"    → {result.strip()}")


# ============================================================================
# Level 5: 从真实 Triton kernel 获取 TTIR
# ============================================================================

def demo_level5():
    """
    设计文档第 5 节：如何从真实 Triton kernel 获取 TTIR。

    方法一：设环境变量 dump IR
    方法二：程序内调用 triton.compile()
    """
    print("\n" + "=" * 70)
    print("Level 5: 从真实 Triton kernel 获取 TTIR")
    print("=" * 70)

    print("""
方法一：环境变量 dump (最简单)
─────────────────────────────
  export TRITON_DUMP_DIR=/tmp/triton_dump

  然后正常写 Triton kernel 并执行一次，/tmp/triton_dump/ 下
  会生成 kernel.ttir.mlir 文件。

  import torch
  import triton
  import triton.language as tl

  @triton.jit
  def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(axis=0)
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < N
      x = tl.load(x_ptr + offsets, mask=mask)
      y = tl.load(y_ptr + offsets, mask=mask)
      tl.store(out_ptr + offsets, x + y, mask=mask)

  x = torch.randn(1024, device='npu')
  y = torch.randn(1024, device='npu')
  out = torch.empty_like(x)
  grid = lambda m: (triton.cdiv(1024, m['BLOCK_SIZE']),)
  add_kernel[grid](x, y, out, 1024, BLOCK_SIZE=256)

  # 然后去 /tmp/triton_dump/ 找 kernel.ttir.mlir


方法二：程序内获取 TTIR (不写磁盘)
─────────────────────────────
  参考设计文档 5.1 节的 emit_ttir_for_costmodel 思路：

  from triton.compiler import ASTSource
  from triton._C.libtriton import ir as triton_ir

  # 1. 创建 MLIR context
  context = triton_ir.context()
  triton_ir.load_dialects(context)

  # 2. 通过 ASTSource 生成 TTIR module
  src = ASTSource(fn, signature, constexprs, attrs)
  module = src.make_ir(target, options, codegen_fns, module_map, context)

  # 3. 拿到 TTIR 文本
  ttir_text = str(module)

  # 4. 构造 arg-bindings
  # 从 JITFunction.run() 中复用 binder 逻辑获取参数值
  # arg_bindings = f"arg{idx}={value}"


方法三：用 triton-opt 工具 (如果已安装)
─────────────────────────────
  # 先编译得到 TTIR，然后用 triton-opt 跑 costmodel:
  triton-opt kernel.ttir.mlir \\
      --pass-pipeline="builtin.module(convert-triton-to-ascend,...)"
""")

    # 实际演示：如果 triton 可用就现场获取
    try:
        import torch
        import triton
        import triton.language as tl
        from triton._C.libtriton import ascend

        @triton.jit
        def tiny_add(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        # 用最小的参数编译，获取 TTIR
        print("\n尝试现场获取 TTIR...")
        N = 256
        x = torch.randn(N, device='cpu')
        y = torch.randn(N, device='cpu')
        out = torch.empty_like(x)

        # 通过 compile 获取 TTIR
        from triton.compiler import compile as triton_compile
        from triton.runtime.jit import JITFunction

        # 简单方式：让 triton 编译但不执行
        compiled = triton_compile(
            JITFunction(tiny_add.fn).fn if hasattr(tiny_add, 'fn') else tiny_add,
            # 这里需要完整的参数，简化演示
        )
        print("  注意：完整获取需要 Ascend NPU 环境")

    except ImportError as e:
        print(f"\n当前环境缺少依赖 ({e})，无法现场演示，请在有 NPU 的服务器上运行。")


# ============================================================================
# Main
# ============================================================================

def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    demos = {
        1: demo_level1,
        2: demo_level2,
        3: demo_level3,
        4: demo_level4,
        5: demo_level5,
    }

    if level > 0:
        demos[level]()
    else:
        print(__doc__)
        print("\n用法: python costmodel_demo.py [1|2|3|4|5]")
        print("  1: Hello World (手写 AscendModel IR)")
        print("  2: triton_unk_fused_add_1 (TTIR + 完整 6-Pass)")
        print("  3: fa_with_mask (Cube+Vector 混合)")
        print("  4: arg-bindings 对 trip count 的影响")
        print("  5: 如何从 Triton kernel 获取 TTIR")
        print()

        # 默认跑 level 1 (最简单)
        print("默认运行 Level 1...\n")
        demo_level1()


if __name__ == "__main__":
    main()
