# CAModel 视角：理解 Ascend NPU 硬件行为

> 本文档从 CAModel 官方文档出发，帮你建立对 Ascend NPU 硬件流水线行为的直观理解。
> 这是优化 costmodel 的大前提——你得先知道硬件真实怎么跑，才能写出对的仿真模型。

---

## 一、Ascend NPU Core 的物理架构

CAModel 产出的 trace 图把一个 Core 内部的硬件单元全部展示出来了。

### 1.1 七大流水线（Pipeline）

CAModel trace 图上有 7 个关键字段，代表 NPU Core 内部的 7 个独立硬件单元：

```
                  ┌──────────────────────────────┐
                  │         GM / HBM              │  ← 全局内存 / 片外 HBM
                  └──────┬───────────┬───────────┘
                         │           │
                    MTE2 ↓           ↑ MTE3
                    (GM→UB)          (UB→GM)
                         │           │
              ┌──────────┴───────────┴──────────┐
              │         Unified Buffer (UB)      │  ← 片上共享缓冲区
              └──┬──────────┬──────────┬─────────┘
                 │          │          │
            MTE1 ↓    CUBE  │    VECTOR│
           (UB↔L0)    ↓     │      ↓   │
                 │   ┌──────┐│  ┌──────┐│
                 │   │Cube  ││  │Vector││
                 │   │Core  ││  │Core  ││
                 │   └──┬───┘│  └──────┘│
                 │      │    │           │
                 │   FIXP ↓  │           │
                 │   (L0C→)  │           │
                 │      │    │           │
                 └──────┴────┴───────────┘
```

| 流水线 | 全称 | 数据方向 | 负责什么 |
|--------|------|---------|---------|
| **SCALAR** | 标量运算单元 | — | 循环控制、地址计算、标量比较 |
| **MTE2** | Memory Transfer Engine 2 | GM/L2 → UB/L1/L0 | **从全局内存搬数据进来**（最常见的瓶颈） |
| **MTE1** | Memory Transfer Engine 1 | L1 → UB/L0A/L0B | 细粒度片内搬运（L1 Cache ↔ UB/寄存器） |
| **CUBE** | Cube Core | — | 矩阵乘（MatMul）、卷积 |
| **FIXP** | FixPipe | L0C → OUT/L1 | Cube 结果写回（仅 A2 系列） |
| **VECTOR** | Vector Core | — | 向量运算（Add/Mul/Exp/ReLU…） |
| **MTE3** | Memory Transfer Engine 3 | UB → GM/L2/L1 | **把结果搬出到全局内存** |
| **FLOWCTRL** | 控制流 | — | 分支/跳转指令 |
| **ICmiss** | ICache Miss | — | 指令缓存未命中 |

### 1.2 1C2V 架构

每个物理 Core 由 **1 个 AIC（AI Core Controller）+ 2 个 AIV（AI Vector）** 组成：

```
物理 Core 0:
  ├─ AIC
  │   ├─ SCALAR
  │   ├─ MTE2(AIC)
  │   ├─ MTE1
  │   ├─ CUBE
  │   └─ FIXP
  ├─ AIV0
  │   ├─ MTE2(AIV)
  │   ├─ VECTOR
  │   └─ MTE3
  └─ AIV1
      ├─ MTE2(AIV)
      ├─ VECTOR
      └─ MTE3
```

**这是 costmodel 调度器最核心的建模对象**。

---

## 二、流水线的并行与串行规则

### 2.1 不同流水线可以并行

这是硬件层面的并行——Cube 在算矩阵乘的同时，Vector 可以做向量运算，MTE2 可以同时从 HBM 搬下一批数据进来。

```
cycle:    0        50       100       150       200
SCALAR    ████
MTE2      ██████████████████████████████████████████
CUBE             ██████████████████████████
VECTOR                  ████     ████     ████
MTE3                                 ██████████
```

### 2.2 同一流水线必须串行

同一个流水线上的两个操作必须一个结束另一个才能开始：

```
VECTOR:
  前一个 Add 结束 ──→ 下一个 Mul 才能开始
  ████████           ████████
```

### 2.3 数据依赖会导致等待

即使在不同流水线上，如果 Op B 的输入依赖 Op A 的输出，B 必须等 A 完成：

```
CUBE (MatMul):
  ████████████████
                  ↓ 等待 Cube 结果写入 UB
MTE1 (搬运结果):
                   ████
```

### 2.4 同步事件（SetFlag/WaitFlag）

CAModel trace 上的大片空隙通常就是同步等待——一个流水线发了 `set_flag`，另一个等 `wait_flag`。这是多流水线协同的核心机制。

---

## 三、CAModel Trace 图的读法

### 3.1 一个真实 trace 示例

CAModel 产出的 `core_0.0.json` / `core_0.1.json` / `core_0.2.json` 用 `chrome://tracing` 打开，看到的画面大致是：

```
core_0.0.json (AIC):
  cycle:  0    200   400   600   800   1000  1200  1400
  SCALAR  ██          ██          ██
  MTE2    ████████████      ████████████      ████████
    MTE1      ██  ██          ██  ██          ██  ██
    CUBE      ██████████      ██████████      ████████
    FIXP            ████            ████            ████

core_0.1.json (AIV0):
  cycle:  0    200   400   600   800   1000  1200  1400
  MTE2    ████████████      ████████████      ████████
  VECTOR      ████  ██████      ████  ██████      ████
  MTE3                ██████              ██████
```

### 3.2 读图的三个关键问题

1. **哪里有空洞？** — 大段空白说明流水线在空等，可能是同步开销或调度没做好
2. **哪条线最长？** — 最活跃的流水线就是瓶颈，优化它才有收益
3. **AIC 和 AIV 有时间重叠吗？** — 没有重叠说明 Cube 和 Vector 没并行起来

### 3.3 快捷键

| 键 | 功能 |
|----|------|
| W | 放大 |
| S | 缩小 |
| A | 左移 |
| D | 右移 |

---

## 四、CAModel Trace 打点：精确分析代码段

可以在 kernel 代码里埋打点，精确测量某段代码的耗时：

```cpp
TRACE_START(0x1);                              // 开始打点
DataCopy(zGm, zLocal, this->totalLength);      // 要测量的代码段
TRACE_STOP(0x1);                               // 结束打点
```

trace 图上会显示：

```
USER_DEFINE_1_DELAY  ████    ← 指令发射等待时间
USER_DEFINE_1        ██████  ← 实际执行时间
```

- 支持 10 个用户定义类型（`0x0`~`0x9`，对应 `USER_DEFINE_0`~`USER_DEFINE_9`）
- START/STOP 必须配对
- 不支持跨核打点（不能在 Cube Core 打 START 在 Vector Core 打 STOP）

---

## 五、性能瓶颈的典型诊断方法

CAModel 文档教的标准分析套路：

| trace 图特征 | 诊断 | 优化方向 |
|-------------|------|---------|
| MTE2 时间线最密集 | **带宽瓶颈 (MTE2 bound)** | 增大搬运粒度；数据排布从 BSH 改为 BNSD 使搬运连续 |
| CUBE 和 VECTOR 之间有大段空隙 | **CV 并行度不足** | 让 Cube 提前计算多块存 GML，Vector 分多次取 |
| FIXP 占比极高 | **FixPipe 瓶颈** | 检查 workspace 是否 512B 对齐 |
| VECTOR 时间线最密集 | **向量计算瓶颈** | 优化 softmax/update 等向量算子；利用 AIV 并发 |
| SCALAR 占比高 | **标量化不足** | 循环展开、向量化 |
| ICmiss 频繁 | **指令缓存抖动** | 减少代码分支；函数内联 |

---

## 六、Perf-Sim：CAModel 的软件近似版

PTO-ISA 的 Perf-Sim 做的事情：

- **CAModel**：拿 `kernel.o` 二进制，在真正的 NPU 周期精确仿真器上逐指令跑 → 产出 trace 图
- **Perf-Sim**：拿 C++ 源码，在普通 CPU 进程里跑 `__COSTMODEL` 版本的 kernel → 用公式估算每条指令耗时 → 产出类似的 trace 图

Perf-Sim 的精度不如 CAModel（特别是 Scalar 流水线没有精细化建模、L2 Cache 模型仅预留接口），但它的价值在于：
1. 不需要 NPU 硬件
2. 可以快速跑很多 config 对比趋势
3. 源码完全开源，能看到"公式怎么来的"

### Perf-Sim 的输出格式（跟 CAModel 对比）

| | CAModel | Perf-Sim |
|---|---|---|
| Trace 图 | `core_0.0.json` 等 (chrome://tracing) | `<op_name>.json` (chrome://tracing) |
| 流水线统计 | dump 日志文件 | `pipeline_summary.csv` |
| 终端报告 | — | 各 pipeline 的 busy cycles 汇总 |
| 瓶颈分析 | 靠人工读图 | 靠人工读图 + CSV |

Perf-Sim CSV 的表头（每个 Core 输出 3 行：AIC/AIV0/AIV1）：
```
op_name,core_id,unit,total_cycles,active_start_cycle,active_end_cycle,
active_cycles,busy_cycles,scalar_cycles,mte2_aic_cycles,mte2_aiv_cycles,
mte1_cycles,cube_cycles,fixp_cycles,vec_cycles,mte3_cycles
```

---

## 七、把你学的串起来：从硬件到 costmodel

```
CAModel 教你             PTO-ISA Formula 教你        Triton-Ascend CostModel (你在做)
───────────────────      ─────────────────────       ─────────────────────────────
硬件怎么跑               怎么公式化近似               怎么嵌进编译器 pipeline
                                   
7 条流水线               每条指令:                     每个 AscendModel Op:
SCALAR/MTE2/CUBE/        cycles = slope*rows*cols      有 estimateCycles() 方法
VECTOR/MTE1/MTE3/FIXP    + bias
                         搬运指令:
SCALAR 在 AIC 上          cycles = bytes/bw * freq     HardwareConfig::estimateMemoryCycles()
MTE2/CUBE/FIXP 在 AIC 上
VECTOR 在 AIV 上          1C2V 架构模拟:                PipelineScheduler (ASAP):
MTE2/MTE3 也在 AIV 上     LAUNCH_KERNEL 宏             不同 HW Unit 并行/同 Unit 串行
                         逐 core/subcore 循环           数据依赖延迟消费
                         记录每条指令的 pipe 和耗时      输出 kernel cycles + roofline
                         
瓶颈:                     瓶颈:                         瓶颈:
MTE2 密集→带宽瓶颈        MTE2 busy cycles 高          RooflineAnalyzer::bottleneckUnit
CUBE-VECTOR 间隙大→调度   CUBE/VEC busy cycles 失衡    PipelineScheduler 的三种估算
```

---

## 八、快速参考：关键数字

### Ascend 910B 基础参数

```
主频:       1.85 GHz     → 1 cycle ≈ 0.54 ns
Cube FP16:  320 TFLOPS  (推测)
HBM 带宽:   1555 GB/s   (910B 规格)
UB 大小:    约 1 MB
L2 Cache:   约 192 MB  (推测)
```

### 数据搬运带宽 (A2/A3 架构，来自 PTO-ISA arch_config.hpp)

```
GM → UB:     100.9 GB/s    (外部搬入，最常用的通道)
GM → L1:     135.0 GB/s
UB → GM:     188.46 GB/s   (外部搬出)
UB → UB:     1024 GB/s     (片内搬运，最快)
L1 → L0A:    441  GB/s     (喂给 Cube Core 的 A 矩阵)
L1 → L0B:    220.5 GB/s    (喂给 Cube Core 的 B 矩阵)
L0C → GM:    70.0 GB/s
L0C → L1:    128.0 GB/s
L1 → BT:     32.0 GB/s
L1 → FB:     32.0 GB/s
L1 FILL:     32.0 GB/s
```

### Hill 带宽模型（更精确，拟合自 B3 实测）

```
GM → UB:     peak=247.16 GiB/s, K=30643 bytes
GM → L1:     peak=28.61 GiB/s,   K=1107 bytes
UB → GM:     peak=28.19 GiB/s,   K=1755 bytes
L0C → GM:    peak=41.25 GiB/s,   K=29104 bytes

公式: bw_eff(bytes) = peak * bytes / (K + bytes)
```

### TMATMUL 的 cycle 公式

```
KHeadCycles = 6
MTile = 16, NTile = 16
KTile = 32 / sizeof(dtype)    (dtype=float→8, dtype=half→16)
cycle_per_repeat = fp32 ? 2 : 1
repeats = ceil(M/16) * ceil(K/KTile) * ceil(N/16)
cycles = 6 + cycle_per_repeat * repeats
```

### 一般向量指令的 cycle 公式

```
cycles = slope * rows * cols + bias
```

其中 `slope` 和 `bias` 通过实际硬件测量拟合得到，存于 `formula_params.csv`。不同 op、不同 dtype、不同 cols 有不同参数。
