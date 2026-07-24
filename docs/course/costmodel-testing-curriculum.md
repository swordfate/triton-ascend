# CostModel 测试驱动学习教案

## 为什么用测试来学？

CostModel 没有"一条命令跑通"的 demo，但测试覆盖了每个模块的正确行为。**每个测试 = 一个独立教学单元**——跑通 → 理解它验证什么 → 看对应源码 → 下一个。

```
costmodel 测试体系
├── C++ GTest (需要 TRITON_BUILD_UT=ON 编译)
│   ├── CostModelHardwareConfig    5 个测试  ← 硬件配置层
│   ├── CostModelPipelineScheduler 7 个测试  ← 调度器层
│   └── CostModelPasses            5 个测试  ← Pass 管线层
│
├── Python unittest (Mock C++ 接口，不需要编译)
│   ├── test_costmodel_runtime.py          11 个测试 ← Python 运行时
│   └── test_compiler_costmodel_contract.py 1 个测试 ← 编译器集成
│
└── Python 集成 demo (需要 triton-ascend 已安装)
    └── costmodel_demo.py          5 个 level ← 端到端验证
```

---

## 编译前提

```bash
# 必须用 TRITON_BUILD_UT=ON 覆盖默认的 OFF
LLVM_SYSPATH=/workspace/llvm-install/ \
TRITON_CODEGEN_BACKENDS=ascend \
TRITON_BUILD_UT=ON \
pip install -e . --no-build-isolation -v --no-deps
```

CTest 可执行文件名 vs GTest 类名/方法名是两层过滤：

```bash
cd build/<cmake_dir>
ctest -R CostModel                              # CTest 正则过滤（到可执行文件级别）
./third_party/.../CostModelPasses \
    --gtest_filter='*EstimateCycles*'           # GTest 内部过滤（到单个 TEST case）
```

---

## 第一课：硬件配置层（HardwareConfig）

**目标**：理解 costmodel 的"数据"——硬件参数怎么表达、怎么加载、怎么用。

### Step 1.1：默认 910B 配置

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelHardwareConfig --gtest_filter='*Default910B*'
```

`HardwareConfigTest.cpp:27-40`——创建默认配置，验证主频、Cube/Vector 算力、HBM 信息非零。理解 `getClockFrequencyGHz()`、`getCubeTFLOPS()`、`getMemorySpace("hbm")` 这些基础 getter。

### Step 1.2：自定义 JSON 解析

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelHardwareConfig --gtest_filter='*ParsesCustomJson*'
```

`HardwareConfigTest.cpp:79-173`——手写一个"微型 NPU" JSON 配置（主频 2GHz、HBM 1GB、Cube 64 TFLOPS FP16、Vector 64 元素宽），验证每个字段都正确读取。理解 JSON schema 和 `estimateVectorCycles()`/`estimateMemoryCyclesWithLatency()` API。

### Step 1.3：真实配置文件加载

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelHardwareConfig --gtest_filter='*LoadFromCommitConfigFile*'
```

验证仓库中的 `ascend_910b.json` 能成功加载和校验。

**源码导读**：`HardwareConfig.h`（类定义）→ `HardwareConfig.cpp`（JSON 解析 + hardcoded fallback + 校准参数）→ `ascend_910b.json`（真实配置）。

---

## 第二课：调度器层（PipelineScheduler）

**目标**：理解 costmodel 怎么模拟硬件并行/串行——**这是整个 costmodel 最核心的算法**。

### Step 2.1：不同硬件单元并行 → 同一单元串行 → 数据依赖

```bash
# 并行：Cube + Vector 同时跑 → total = max(100, 80) = 100
./build/third_party/ascend/unittest/costmodel_ut/CostModelPipelineScheduler --gtest_filter='*DifferentHardwareUnits*'

# 串行：两个 Vector Op → total = 40 + 60 = 100
./build/third_party/ascend/unittest/costmodel_ut/CostModelPipelineScheduler --gtest_filter='*SameHardwareUnitSerializes*'

# 依赖：Op2 依赖 Op1 输出 → Op2.start = Op1.end
./build/third_party/ascend/unittest/costmodel_ut/CostModelPipelineScheduler --gtest_filter='*DependenciesDelayConsumers*'
```

这三个测试是 ASAP 调度算法的三条核心规则：不同硬件单元并行、同一单元串行、数据依赖必须等待。

### Step 2.2：边界情况

```bash
# 循环依赖检测（RejectsCyclic）
# Kernel 级估算（KernelCyclesApplyBarrier）— 从单程序 cycle 推到完整 kernel
# Roofline 瓶颈分析（RooflineAnalyzer）
./build/third_party/ascend/unittest/costmodel_ut/CostModelPipelineScheduler
```

`KernelCyclesApplyBarrier` 验证公式：`(totalCycles + barrierCycles) * (1 + scalarFactor) * ceil(programs/cores)`。`RooflineAnalyzer` 基于调度结果判断哪个硬件单元是瓶颈。

**源码导读**：`PipelineAnalysis.h`（PipelineScheduler 类 + DependencyGraph + RooflineAnalyzer）→ `PipelineAnalysis.cpp`（ASAP 算法实现）。

---

## 第三课：Pass 管线层（Passes）

**目标**：理解 6 个 Pass 怎么串联——**这是理解"编译期 costmodel 如何工作"的关键**。

### Step 3.1：AssignOpIDs —— 最简单的 Pass

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelPasses --gtest_filter='*AssignOpIDs*'
```

手写一段 AscendModel IR（含 `ascend.add` 和 `arith.addi`），跑 AssignOpIDsPass，验证只有 AscendModel dialect 的 Op 被分配 ID。理解 MLIR Pass 的基本模式：`module.walk()` + 写 attribute。

### Step 3.2：EstimateCycles —— 核心 Pass

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelPasses --gtest_filter='*EstimateCycles*'
```

手写 `vector_load → add → vector_store` 的 AscendModel IR，跑 EstimateCyclesPass，验证每个 Op 被标注 `estimated_cycles`、`bytes`、`flops`、`hw_unit`。

### Step 3.3：PipelineAnalysis —— 串联跑

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelPasses --gtest_filter='*PipelineAnalysis*'
```

跑 AssignOpIDs → EstimateCycles → PipelineAnalysis 串联，验证 module 上写出了 `scheduled_cycles_one_iter`、`roofline_cycles`、`simple_sum_cycles` 三个属性。**理解 Pass 之间通过 IR attributes 传递数据**。

### Step 3.4：完整管线 + 错误处理

```bash
./build/third_party/ascend/unittest/costmodel_ut/CostModelPasses --gtest_filter='*PerfReport*'
./build/third_party/ascend/unittest/costmodel_ut/CostModelPasses --gtest_filter='*InvalidArgBindings*'
```

**源码导读**：`PassRegistration.cpp`（管线注册 6 步）→ `EstimateCycles.cpp`（三遍遍历 + loop trip count 解析）→ `ConvertTritonToAscend.cpp`（12 种转换模式）。

---

## 第四课：Python 运行时层

**目标**：理解 Python 侧怎么调用 C++ costmodel。**这些测试 Mock 了 C++ 接口，不需要编译。**

### Step 4.1：Python 运行时单元测试

```bash
python -m pytest third_party/ascend/unittest/costmodel_ut/test_costmodel_runtime.py -v
```

11 个测试覆盖 Python 侧的核心逻辑：

| 测试 | 验证内容 |
|------|---------|
| `test_parse_latency_and_jobs` | 正则解析 C++ 输出 `"Estimated Time: 3.25 us"` → 浮点数 |
| `test_cache_namespace_variants` | 缓存 key 生成逻辑 |
| `test_store_and_load_costmodel_latency` | 缓存读写 + 脏多线程数据处理 |
| `test_make_key_and_extra_args` | arg-bindings → extra_args 转换 |
| `test_run_costmodel_reads_file_and_adds_allow_unregistered_dialect` | 文件路径自动读成 MLIR 文本 |
| `test_run_costmodel_exception_paths` | C++ 异常 → Python None |
| `test_normalize_items_and_eval_item` | 输入 item 规范化（TTIR 文本 + config → 规范化元组） |
| `test_eval_item_miss_and_pending_eval` | 缓存 miss → trigger 真正 C++ 评估 |
| `test_evaluate_pending_empty` | 空 pending 列表不报错 |
| `test_evaluate_pending_parallel_exception_tolerated` | 多线程中单个任务失败不影响其他 |
| `test_costmodel_bench_paths` | **主入口** `costmodel_bench()` 边界情况：空输入、格式错误、normalize 失败回退 |

**Mock 策略**：所有测试 patch 掉 `_evaluate_pending_items`（它内部的 `run_costmodel()` 才调 C++），验证 Python 侧的缓存、解析、规范化、多线程、异常处理逻辑正确。C++ 正确性由第一课～第三课的 GTest 覆盖。

### Step 4.2：编译器集成契约

```bash
python -m pytest third_party/ascend/unittest/costmodel_ut/test_compiler_costmodel_contract.py -v
```

验证 `enable_costmodel_backend=True` 选项能正确传递到编译器 `parse_options()`。

---

## 第五课：端到端集成（需要 triton-ascend 已安装）

上面都是独立测试。以下用 `costmodel_demo.py` 做端到端验证，直接调 `ascend.run_costmodel_inproc()` 走完整 C++ Pass 管线。

### Step 5.1：手写 AscendModel IR 验证 C++ 管线

```bash
python docs/costmodel-examples/costmodel_demo.py 1
```

最简单的 `ascend.add`，跳过 TTIR 转换，直接测底层 cycle 估算管线。这是 C++ PassesTest 的 Python 等价版。

### Step 5.2：真实 TTIR 走完整 6-Pass 管线

```bash
python docs/costmodel-examples/costmodel_demo.py 2
```

读取 `test/Triton/vecadd.mlir`（标准 Triton 向量加法的 TTIR），走 ConvertTritonToAscend → InsertDataTransfers → AssignOpIDs → EstimateCycles → PipelineAnalysis → PerfReport 全链路。

### Step 5.3：arg-bindings 对预估的影响

```bash
python docs/costmodel-examples/costmodel_demo.py 4
```

同一段 TTIR，不同 `arg-bindings` 得到不同预估时间（因为循环 trip count 不同）。

### Step 5.4：TTIR 是如何产出的——从真实 Triton kernel 获取

```bash
python docs/costmodel-examples/costmodel_demo.py 5
```

演示三种获取 TTIR 的方法：

| 方法 | 命令/代码 | 适用场景 |
|------|----------|---------|
| 环境变量 dump | `TRITON_DUMP_DIR=/tmp/dump` 后运行 kernel → 取 `kernel.ttir.mlir` | 最简单，适合一次性调试 |
| 程序内获取 | `ASTSource.make_ir()` → `ast_to_ttir()` → 拿到 TTIR 文本 | costmodel 集成到 autotune |
| triton-opt 命令行 | `triton-opt kernel.ttir.mlir --pass-pipeline=...` | 手动调试单个 Pass |

关键代码路径：
```
Python kernel (@triton.jit)
  → ast_to_ttir()              ← make_ir() 调用, 产出 raw TTIR
  → make_ttir()                ← stages["ttir"], inline + CSE + LICM + loop unroll
  → TTIR 文本                  ← 这就是 costmodel 的输入
  → ascend.run_costmodel_inproc(ttir, args)
  → C++ Pass 管线 (ConvertTritonToAscend → ... → PipelineAnalysis)
  → "Estimated Time: X.XX us"
```

---

## 学习路线

```
Day 1: 编译 + 跑通 C++ 测试
  ├── TRITON_BUILD_UT=ON pip install
  ├── ctest -R CostModel
  ├── 读 HardwareConfigTest → 理解硬件参数层
  └── 读 PipelineSchedulerTest → 理解 ASAP 调度算法

Day 2: 理解 Pass 管线
  ├── 读 PassesTest → 理解 6 个 Pass 逐个和串联
  ├── 读 EstimateCycles.cpp → 理解三遍遍历 + loop trip count 解析
  └── 读 ConvertTritonToAscend.cpp → 理解 TTIR → AscendModel 转换

Day 3: 理解 Python 层 + 端到端
  ├── 跑 Python 测试 → 理解缓存/多线程/输入格式
  ├── 跑 costmodel_demo.py 1~5 → 端到端验证
  └── 读 costmodel_runtime.py → 理解 Python 侧完整调用链
```

---

## 测试与源码对应表

| 测试 | 被测源码 | 教什么 |
|------|---------|--------|
| `CostModelHardwareConfig.*` | `HardwareConfig.h` / `HardwareConfig.cpp` | 硬件参数对象 + JSON 解析 + 校准参数 |
| `CostModelPipelineScheduler.*` | `PipelineAnalysis.h` / `PipelineAnalysis.cpp` | ASAP 调度算法 + Roofline 分析 |
| `CostModelPasses.*` | `EstimateCycles.cpp` + `PipelineAnalysisPass.cpp` + 各 Op `.cpp` | Pass 管线 + cycle 估算 + 依赖调度 |
| `test_costmodel_runtime.py` | `costmodel_runtime.py` | Python 缓存/解析/规范化/多线程 |
| `costmodel_demo.py` | 全链路 C++ Pass | 端到端集成验证 |
