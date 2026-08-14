# Costmodel 打分公式全集与 rate 语义详解

> 基于 `SimdSimtCostModel.cpp` 完整代码追踪 + ROPE/causal_conv1d/silu_mul/cumsum 四个实测 kernel 的数据核对。
> 分支：`kx_simt_costmodel`（C++ 代码）/ `feature/costmodel-dev`（本文档）

---

## 一、三条路线的完整打分公式

### 1.0 特征层输入（`analyzeSimdSimtFeatures` 产出）

所有 op 计数先按循环加权：

```
opElements[kind]     = Σ_op elements × loopMultiplier(op)
loadBytes/storeBytes = Σ dataElements × loopMultiplier × bitWidth/8
dotFlops             = Σ 2×M×N×K × loopMultiplier
loadWarpInstructions / storeWarpInstructions   ← 提取时统计（mask 处理后的 warp 指令数）
maskRankSum          = Σ 每个 mask tensor 的 rank（按出现次数，不去重）
weightedReductions   = mapValue(weightedOps, "reduce")
weightedScans        = mapValue(weightedOps, "scan")

loopMultiplier(op) = Π 每层 scf.for 的 tripCount；未知循环 fallback = 1
```

### 1.1 公共计算项（SimdSimtCostModel.cpp:2428-2570）

**① 逐 op 计算**（:2428-2455）
```
simdCycles[kind] = ceil(elements / vectorLanes) / simd.throughput[kind] × factor[kind]
simtCycles[kind] = elements / simt.throughput[kind] × factor[kind]
vectorLanes = 2048bit / 元素位宽（f32 → 64 lanes）
```
f32.add 基准：simd.throughput=3.30（微基准实测），simt.throughput=141.0（微基准实测）

**② 内存**（:2481-2526）
```
simdLoadCycles  = loadBytes / 202.25      ← legacy seed（假带宽）
simdStoreCycles = storeBytes / 202.25     ← 同上
simdMemoryCycles = 二者之和
simtLoadCycles  = loadWarpInstructions / 0.176    ← 微基准实测
simtStoreCycles = storeWarpInstructions / 0.129   ← 微基准实测
simtMemoryCycles = 二者之和
```

**③ dot**（:2572-2593）
```
simdDotCycles = 128 + dotFlops/4096    ← setup 是 seed，4096 是 cube 峰值
simtDotCycles = 64 + dotFlops/141      ← setup 是 seed，141 是标量 FMA
```

**④ shuffle**（:2532-2555，SIMT 专属）
```
shuffleLevels = ⌈log2(32)⌉ = 5
simtShuffleInstructions = (weightedReductions + weightedScans) × ⌈maxNumel/32⌉ × 5
simtShuffleCycles = simtShuffleInstructions / 0.817    ← 微基准（32warp/ILP4/shfl-up）
```

**⑤ predicate**（:2559-2570，SIMT 专属）
```
simtPredicateInstructions = maskRankSum × ⌈maxNumel/32⌉
simtPredicateCycles = simtPredicateInstructions / 0.038   ← ★ 单一 FBGEMM workload 的 stall 统计
```

### 1.2 all_simd

```
P_payload_simd = max( simdCompute + simdDot , simdMemory )

all_simd_raw = simdSetup + P_payload_simd × programIssueScale
             = 21.212  + P_payload_simd × 8.0

P_struct = irregular_addressing + mask_materialization + reduction_lowering
         + static_loop_control + control_flow + rank1_indirect_reduction + tiny_dot
         = min(0.5, irregularDensity×0.8) + min(0.35, maskRankSum×0.022)
         + min(0.15, weightedReductions×0.02) + min(0.15, staticLoopTrips×0.008)
         + [0.03 if 有控制流] + [0.75 if rank1 间接归约] + [0.06×underfill if tiny dot]

all_simd = all_simd_raw × (1 + P_struct)
```

### 1.3 all_simt_only

```
P_payload_simt = max( simtCompute + simtShuffle + simtDot , simtMemory ) + simtPredicate

all_simt_only = simtSetup + P_payload_simt × programIssueScale
              = 141 + P_payload_simt × 8.0
              （无结构惩罚——predicate/shuffle 已在 payload 内）
```

### 1.4 mixed_simd_simt

```
# 分区：anchor ops 按 SIMT 价，其余按 SIMD 价（:2470-2479, 2692-2701）
P_regular_compute = max(0, simdCompute − Σ_anchor⌈elements/64⌉/rate×factor)
P_anchor_compute  = Σ_anchor elements/simt.rate×factor
P_regular_memory  = (loadBytes−anchor.loadBytes)/202.25 + (storeBytes−anchor.storeBytes)/202.25
P_anchor_memory   = anchor.loadWarpInst/0.176 + anchor.storeWarpInst/0.129
P_anchor_shuffle  = (anchorReductions+anchorScans) × ⌈anchorMaxNumel/32⌉ × 5 / 0.817
P_anchor_predicate = anchor.maskRankSum × ⌈anchorMaxNumel/32⌉ / 0.038

P_regular = max( P_regular_compute + P_regular_dot , P_regular_memory )
P_anchor  = max( P_anchor_compute + P_anchor_dot + P_anchor_shuffle , P_anchor_memory )
          + P_anchor_predicate

# 剩余结构惩罚（把 anchor 已算的从全 kernel 惩罚里扣除，:2731-2756）
P_residual = 剩余 irregular/mask/reduction/loop/control_flow/rank1/tiny_dot 分量之和

mixed_simd_simt = mixedSetupFallback + ( P_regular×(1+P_residual) + P_anchor ) × 8.0 + boundary
                = 223(32 warps)    + ( ... ) × 8.0                              + 0

# 无 anchor 时的退化形态（:2786-2792）
mixed_simd_simt = max(all_simd, all_simt_only) + 223
```

### 1.5 event 校准与 gate（:2795-2889）

```
uncalibrated = candidateCosts（上面三式的结果）
若有 domain 命中：
    all_simd   ×= m_simd
    all_simt   ×= m_simt
    mixed      ×= m_mixed

decision       = 合法候选中的 argmin
requiredGain   = max(64, bestScore × 0.10)
advantage      = decision==all_simd ? runnerUp−best : all_simd−best
gatePassed     = advantage > requiredGain
              && 无 unsupported 项
              && rankingConfidence ≥ minimumConfidenceForDecision
              && coverage 覆盖
```

### 1.6 常数来源总表

| 常数 | 值 | 来源 | 可信度 |
|------|-----|------|--------|
| simdSetup | 21.212 | legacy seed | 低 |
| simtSetup | 141 | 微基准 empty-VF（含 UB 写+barrier） | 中 |
| mixedSetupFallback | 182-223（按 warps） | 微基准 harness 差值（mode1−mode6） | 低 |
| simd.f32.add 3.30 | 微基准 tput.cce | 高 |
| simt.f32.add 141.0 | 微基准 tput.cce | 高 |
| 其他 op factor | 1.0~12.0 | 相对 f32.add 的估算 | 中低 |
| simd 内存 202.25 B/cycle | **legacy seed（假）** | 极低 |
| simt load/store 0.176/0.129 | 微基准 | 中 |
| dot setup 128/64，4096/141 | seed | 低 |
| shuffle 0.817 | 微基准（单一 code shape） | 中 |
| **predicate 0.038** | **单一 FBGEMM workload stall 统计** | **极低（★头号问题）** |
| programIssueScale 8.0 | 3 个 kernel Event 拟合 | 低 |
| 7 个结构惩罚分量 | 手调/拟合 | 低 |
| event multipliers | 每 domain 手测 | 低（补丁） |
| marginRatio 0.10 + 下限 64 | 策略 | — |

**结构性不对称**：SIMD 的"难算的活"走乘法惩罚（×1+P），SIMT 的"难算的活"走加法项（+shuffle+predicate）。乘法惩罚有 cap 且量级被低估，加法项无 cap 且 rate 被高估——两条路线的"难度建模"在数学形态上不对称，是排序系统性偏向 SIMD 的结构性原因之一。

---

## 二、SIMT 各 rate 的语义：workload-effective warp 指令吞吐

### 2.1 统一语义

单位都是 `warp_instruction / system_cycle`（SYS_CNT @ 988.9MHz）：

```
rate = 源级 warp 指令数 / 整个微基准 kernel 的实测 elapsed cycles
```

- 分子：**源级 warp 指令**（1 条 warp 指令 = 32 lanes 一起执行）
- 分母：**整个微基准 kernel 的墙钟 SYS_CNT cycles**——包括循环控制、地址生成、依赖停顿、sink 开销

所以用 `cycles = instructions / rate` 反推成本时，模型隐含假设：**目标 kernel 的 code shape 与微基准相同**（同样的 ILP、同样的 warp 数、同样的依赖模式），那 rate 里含的停顿比例才一致。

### 2.2 各 rate 的具体来源与 scope

| rate | 值 | 微基准 scope | 备注 |
|------|-----|-------------|------|
| `simt.f32.add` | 141.0 | 32 warps，ILP8，runtime-loop 源级 fadd | "not a proven bare-ALU peak" |
| `simt.shuffle` | 0.817 | 32 warps，ILP4，4 条独立链，**shfl-up 单方向** | "not an intrinsic shuffle-pipeline peak" |
| `simt.gm.load` | 0.176 | 32 warps，顺序旋转 128MiB，每线程 8 个独立 op | 含地址生成和 sink |
| `simt.gm.store` | 0.129 | 同上 | 同上 |
| `simt.predicate` | **0.038** | **FBGEMM 单 workload 统计**（camodel_effective） | "fallback calibration seeds, not isolated instruction peak" |

数值的物理含义对比：

- **141 scalar_op/cycle**：32 warps × ILP8 同时跑，每个 cycle 平均完成 141 个源级 fadd → 每个 warp 指令约 0.23 cycle。合理。
- **0.817 warp_inst/cycle**：shuffle 有 shuffle unit 的串行化约束，1 个 warp 指令约 1.2 cycle。合理。
- **0.038 warp_inst/cycle**：**1 条谓词指令要 26 个 cycles**。物理上不可能是指令发射速度——它反映的是 FBGEMM workload 里谓词指令和访存指令交替时，谓词整体被停顿拖慢后的"有效节奏"。谓词硬件本身大概 1-2 cycles/inst。

### 2.3 "effective" 的三层含义

profile 描述里反复出现的 "effective" 指三件事都被折进了 rate：

1. **含自身开销**：循环增量/比较/分支、地址生成都在分母里
2. **含依赖停顿**：ILP 不足时的 stall 也在分母里
3. **绑定特定 code shape**：32 warps、ILP4/8、特定访存模式——换一个 shape，effective rate 就变

所以这些 rate 不是"发射速率"（issue rate），是"该 workload 在该 code shape 下实际完成的速率"（workload-effective throughput）。这恰恰是模型里 `cycles = instructions / rate` 这个除法成立的**唯一前提**——而真实 kernel 几乎从不满足这个前提。

### 2.4 由此产生的两个具体问题

**问题 1：predicate 0.038 是灾难性的错配**。它来自一个"谓词指令很少、且被访存严重拖慢"的 workload。ROPE 的谓词是 mask select/cmp 和 load 紧密交错的——按 26 cycles/条算，64,421 cycles/迭代，整个 kernel 的实测才 49k cycles。错配了 ~20x。

**问题 2：profile 里有 latency 测量，模型却不用**。微基准 profile 里明明有：

```
simd.f32.add.dependent_latency = 1.818 cycles    ← 依赖链上一条 add 的延迟
simt.shuffle.dependent_latency = 27.28 cycles    ← 依赖链上一条 shfl 的延迟
```

但打分代码（SimdSimtCostModel.cpp）**只消费了 throughput 测量，没有任何 latency 项**。小 kernel（cumsum、silu 的一轮迭代）是 latency-bound 的——依赖链长度决定运行时间，吞吐公式完全抓不到。silu 的 sigmoid 链（exp→div→mul 依赖链）就是这种情况。

### 2.5 一句话总结

```
rate = 该微基准 workload 的有效 warp 指令吞吐（含停顿、含开销、绑定 code shape）
模型用法 = 指令数 ÷ rate → 假设目标 kernel 与微基准同 shape
失效场景 = predicate（错配 20x）、shuffle（32warp→4warp 不适用）、
          latency-bound 小 kernel（模型无 latency 项）
```

这些 rate 是"**某个特定 workload 的实测完成速率**"，不是"**硬件的发射能力**"。模型把它们当硬件能力用，就是当前 SIMT 侧误差的结构性来源。

---

## 三、simtPredicateInstructions 公式的实例走查

### 3.1 公式的意图

SIMT 上每个带 mask 的操作，都要先**逐 lane 算谓词**（cmp/select），再执行。所以谓词指令数应该 ≈ 「mask 的使用规模 × 覆盖整个 tensor 需要的 warp 指令数」。

```
simtPredicateInstructions = maskRankSum × ceil(maxNumel / 32)
```

- `maskRankSum` = mask 使用规模（用 rank 加权）
- `ceil(maxNumel/32)` = 覆盖最大的 tensor 需要几条 warp 指令（32 lanes 一条）

### 3.2 maskRankSum=153 从哪来：一个 mask 被数了 5-8 次

ROPE 里 `q = tl.load(Q+offs_q, mask=(mask_h[:,None]) & (mask_c[None,:]), other=0)` 这一个语句在 TTIR 里的完整链：

```mlir
%a = arith.cmpi slt, %offs_c, %cst : tensor<16xi1>        ← ① 产出 mask_c：result 是 mask → maskRankSum += 1
%b = arith.cmpi slt, %offs_h, %cst : tensor<16xi1>        ← ① 产出 mask_h：                    += 1
%c = tt.expand_dims %b {axis=1} : 16xi1 → 16x1xi1          ← ② 输入 mask(rank1) + 输出 mask(rank2) → += 1 + 2 = 3
%d = tt.expand_dims %a {axis=0} : 16xi1 → 1x16xi1          ← ② 同上                             += 3
%e = tt.broadcast %c : 16x1xi1 → 16x16xi1                 ← ② 输入(rank2) + 输出(rank2)         += 2 + 2 = 4
%f = tt.broadcast %d : 1x16xi1 → 16x16xi1                 ← ② 同上                             += 4
%g = arith.andi %e, %f : 16x16xi1                          ← ③ 两个 mask 输入(2+2) + 一个 mask 结果(2) → += 6
%q = tt.load %ptr, %g, %other : tensor<16x16xf16>         ← ④ load 的 mask 操作数(rank2)       += 2
```

**一个逻辑 mask（`mask_h & mask_c`）贡献了 1+1+3+3+4+4+6+2 = 24**。其中真正"算谓词"的只有两个 cmpi 和一个 andi（4 条指令的量），其余 20 是同一 mask 的 shape 变换和消费点重复计数。

全 kernel 统计：53 个 op 触碰 mask，累计 `maskRankSum = 153`。但 JSON 里同时报告了去重后的真实规模：`unique_mask_values = 37`（只有 37 个不同的 mask 值），`unique_mask_rank_sum = 66`。**153 是 66 的 2.3 倍**——一半以上是重复计数。

### 3.3 ×16 从哪来：maxNumel=512 → 覆盖全 tensor 要 16 条 warp 指令

ROPE 最大的 tensor 是 512 元素（16×32）。512/32 = 16 条 warp 指令。

所以：`153 × 16 = 2448` 条谓词 warp 指令（JSON 实测这个数）。

### 3.4 ÷0.038 → 64,421 cycles

`2448 / 0.038 = 64,421` cycles/迭代。0.038 = 每条谓词 warp 指令 26 cycles。而真实的谓词指令（cmp/select）在 SIMT 上 ~1-2 cycles 一条。

### 3.5 一个例子暴露三个错误

用上面那条 `q = tl.load(mask_h & mask_c)` 链算账：

| 项 | 公式算的 | 实际应该 |
|----|---------|---------|
| 指令数 | 24 × 16 = 384 条 | 真正算谓词的：2 cmp + 1 andi + 1 次 predicated load ≈ 4 × 16 = 64 条 |
| 每条耗时 | 26 cycles（÷0.038） | ~1-2 cycles |
| 合计 | 384 × 26 ≈ 10,000 cycles | 64 × 1.5 ≈ 96 cycles |

**高估 ~100x**。三个错误各占一份：

1. **重复计数**（×2.3）：同一 mask 的 shape 变换和消费点全算一遍 → 应该用 `uniqueMaskRankSum=66`
2. **粒度错配**（×2~4）：rank-1 的小 mask（16 元素）也被按 maxNumel=512 收费（16 条 warp 指令），实际 1 条就够
3. **rate 0.038**（×26）：把 FBGEMM workload 的停顿统计当成了谓词硬件的发射速度

这 100x 再被 ×8.0（program_issue_scale）放大，就是 ROPE SIMT 分数 521,299 里 predicate 独占 515,368 的完整来历。

---

## 附：相关文档

- `costmodel_rope_kernel_diagnosis.md` — ROPE kernel 的 15-block 打分走查与根因
- `computed_index_gather_gap.md` — costmodel 与 template 路径的判定口径差异
- `attention_indirect_gqa_calibration_plan.md` — attention domain 的校准方案
