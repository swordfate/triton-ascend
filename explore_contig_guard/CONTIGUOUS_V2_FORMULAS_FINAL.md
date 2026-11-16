# 连续访存 v2 标定：公式手册（最终版）

基线：`2e9efcacb891eb`（重构前）。本文描述基线之后新引入的、截至当前 HEAD
的**全部**连续访存公式：进入条件、公式本体、参数表、以及各层之间的逻辑关系。
代码位置：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`
（函数 `mapContiguousV2Completion`，约 420–1130 行）与
`SimdSimtCostModel.cpp`（launch 边界隐藏，`completeCandidateCosts`）。

---

## 0. 总入口与总准入（所有公式的公共前置）

v2 钩子在 `mapWorkload` 的 layout 计价分支里、legacy 曲线**之前**调用：

```
mapStridedShortMemoryCompletion(...)      // StridedShort 短行跨步（老机制，先跑）
mapContiguousV2Completion(...)            // v2：认领全部连续访问（本文）
<legacy load-group / GM 曲线 / 未覆盖尾项>  // 只处理 v2 没认领的
```

**总准入（eligible 扫描）**——一条访问进入 v2 必须全部满足：

| # | 条件 | 备注 |
|---|---|---|
| 1 | `count > 0` | |
| 2 | layoutClass ∈ {ContiguousShort, ContiguousWide}，**或**（本轮新增）StridedWide 且 `rowBytes ≤ 64B` 且同 stage 已有连续 eligible **load** | 混流域准入：小行距跨步 load 与连续流共享 DMA 引擎，用组合体一行计价（permute 的 32B/96 元素距 c-load） |
| 3 | `hasStaticLayout` 且 `elementBits/8 > 0` | |
| 4 | 地址不依赖 load 结果、无非仿射张量运算、无张量控制流选择 | gather/间接寻址不进 v2（别的责任域） |
| 5 | masked **store** 且 `activeElementRatio < 1` 不进（保留 RMW 路径） | masked load 进 |
| 6 | 未被本轮其他机制认领 | |

环境门（A/B 用，默认全开）：`ASCEND_V2_DENY_LOOP`、`ASCEND_V2_DENY_SIMT`。
认领即 `claimAll()`：把访问的 **全部实例**（`access.count`，嵌套循环壳会给出
count>1）写入 layout 覆盖账本，保证 legacy 尾项不再重复计价。

**认领单位语义**：所有"行"的值 = 整核 profiler 中位时长的纳秒数
（grid > 56 时除以 wave 数），即一次性包含该 body 的完整访存时间；
公式只把其中可显现的部分按 `directFixedShare`/斜率折算进 per-iteration 价格。

---

## 1. 循环体 stage（hasLoop）——按优先级依次尝试，命中即返回

`K = stage.iterationCount`；`domBytes` = 单实例最大访问字节；
`v2Waves = max(1, ceil(hint/56))`；`v2WaveDecay = v2Waves^(-0.05)`。

### 1.1 Tier 1a：tiny-store 延迟链（tiny_direct 行）

**进入条件**（全部满足）：
- `K ≥ 1` 且 tiny_direct 表非空；
- store-bearing 体：所有 store 块 ≤ 128B 且 `maxStoreChunk` 不被清零；
  load-only 体：仅当 bodyMasked（rms 证据）；
- 行匹配：`nL/nS` 相同、`(m标志 != 0) == bodyMasked`、
  `|log2(domBytes) − log2(rB)| ≤ 1`；
- bestMaxK 门：`K ≤ bestMaxK×1.5 + 1`（不外推 K）。

**公式**：
```
m   = 斜率中位数（所有 K 对的 (T_j−T_i)/(K_j−K_i)）
A   = max(0, T[centreK] − centreK·m)          // centre = K 排序中位
rowT= 行表中距 K 最近的 T
excess = max(0, rowT − (A + K·m))             // 延迟链超出线性趋势的部分
perIter = (A·directFixedShare + K·m + excess·tinyStoreChainShare) / K
          · v2WaveDecay
认领：resources.load += perIter（估计器再 ×K）
```

**参数**：`tiny_direct` 20 行；`direct_fixed_share = 0.05`；
`tiny_store_chain_share = 0.636`。

### 1.2 Tier 1：direct 行整核拟合（主力公式）

**进入条件**：
- `hint ≤ 224`（行只在 ≤4 wave 内测过，不做远外推）；
- 行族匹配：`nL/nS` 相同、`(m标志==0) == !bodyMasked`、
  `|log2(dom) − log2(rB)| ≤ 1` 取最近桶 `bestB`；
- 网格括号（本轮修正）：hint **精确命中**某测量网格 → 用该网格**完整 K 集合**；
  否则取上下邻网格、按 log2-grid 距离做几何插值
  （`t = exp(ln tLo·(1−w) + ln tHi·w)`），K 缺失时取邻网格最近 K。

**公式**：
```
m   = 斜率中位数
mHi = 只用 (centreK, >centreK] 点对的斜率中位数；若 mHi < 1.05·m 则 mHi = m
useM = (K > 16 且 mHi > m) ? mHi : m
chunkScale = clamp(domBytes / bestB, 0.5, 2.0)
perIter = (A·directFixedShare + K·useM·chunkScale) / K · v2WaveDecay
认领：resources.load += perIter（估计器再 ×K）
```
单行族（只有 1 个 K）：`m = T·0.5/K`，`A = T − K·m`（各半拆分）。

**参数**：`direct` simd 295 行 / simt 291+ 行
（键 `n<nL>_s<nS>_b<chunk>_k<K>_g<grid>_m<0|1>`，值 = 整核 T(K) ns）；
`direct_fixed_share = 0.05`；`chunkScale ∈ [0.5, 2]`。

### 1.3 Tier 2：聚合带宽包络（仅 simd）

**进入条件**：simd 且 `independentLoadGroupBytesPerCycle > 0`
（即 simd profile 提供了聚合包络）。

**公式**：
```
perIter = (independentLoadGroupIterationStartupCycles
           + (loadBytes + storeBytes) / independentLoadGroupBytesPerCycle)
          · v2WaveDecay
```

### 1.4 Tier 3：通用逐操作解析式（保底，永不脱靶）

**公式**（每条 eligible 访问）：
```
simd: per = (store? storeOperationStartup : loadOperationStartup)
            + bytes / (store? storeBytesPerCycle : loadBytesPerCycle)
simt: per = divideCeil(elements·instances, 32)
            / (store? storeWarpInstructionsPerCycle
                    : loadWarpInstructionsPerCycle)
load 侧求和进 resources.load，store 侧求和进 resources.store
```

**层间关系**：Tier1a → Tier1 → Tier2 → Tier3 严格降级；任何一层命中都
`claimAll()` 全量认领并 return，保证"连续流永远由 v2 计价，legacy 只见
非连续"。行缺失时逐层退化，最坏落到 Tier3 解析式（仍有 setup+带宽下限）。

---

## 2. 直线 stage（无循环）——Formula A 同形状组

**进入条件**：stage 无循环壳；simd（simt 直线组的行是 warp 数相关的，
`groupA` 里以 `/w<W>` 后缀单独存放）。

**分组**：按 `(isStore, 元素数, 字节宽, masked)` 分组，同组同价。

**公式（拟合组命中）**：
```
桶 bucket = argmin_b |log2(bytes·masked?activeRatio:1) − log2(b)|，
            b ∈ {32,64,128,256,512,1024,4096,16384}
res = footprint ≤ l2CacheBytes(128MB) ? "warm" : "cold"
key = "{load|store}/{dtype}/{bucket}/{res}"（simt 再试 +"/w<W>"）
total = setup + groupA[key] · (store ? 1.0 : v2Waves^(-0.32))
```
load 侧带 wave 衰减 `waves^-0.32`（guard WDA 拟合：copy/add 的整核价格
随 wave 数衰减），store 侧不衰减。

**公式（掩码组 / 未命中组）**：主动退到逐操作解析式（同 Tier3 形状），
load 侧同样乘 `waves^-0.32`；掩码组按**活跃字节**
（`bytes × activeElementRatio`）计。

---

## 3. launch 边界隐藏（calH，SimdSimtCostModel.cpp）

**进入条件**：直线（无循环壳）、连续流覆盖、`memStages ≥ 2`、
合法 plan、`launchWaves ∈ [2,3]`（launchWaves = ceil(hint/56)，用的是
launch 网格数，不是被 AutoBlockify 钳到 ≤56 的 runtime 数）。

**公式**：`候选 elapsed −= 640ns`（线性常数，来源：vector_add 2-wave 锚点
+ calH 扫描；物理含义：边界被连续 DMA 流的首段延迟覆盖）。

---

## 4. 本轮（calM 攻坚）新增/修正的 5 处结构逻辑

| # | 修复 | 机制 | 动机（实测证据） |
|---|---|---|---|
| 1 | claimAll 按 `access.count` 记账 | 嵌套循环壳的每条访问带 count>1（permute 的壳跑 2 遍），旧版只记 1 份 → legacy 尾项把没记账的另一半再算一遍，叠加在整核行之上 | permute +63%→−2% |
| 2 | 混流域准入 | 小行距（≤64B）StridedWide load 与连续 eligible load 共存时并入认领，组合体一行计价 | permute 的 z 载入孤立价 470ns/iter vs 合流真值 216 |
| 3 | SIMT rowwise-reduction group 分支取 `max(groupBody, serial 流, issue)` | 标定组曲线不得遮蔽诚实的 load 流 claim | rowred −25%→+1.6% |
| 4 | 掩码 padded 瓦片在两个未覆盖尾项里按活跃字节计 | 掩码瓦片尾部 lane 不取数；max_dim1 的 (64,32)→(64,8) 只搬 2KB | max_dim1 +12.1%→过线 |
| 5 | tier-1 网格括号：hint 精确命中网格时用该网格完整 K 集合 | 旧逻辑无条件 `--loIt`，注入的 g32 单行使 guard 的 g56 家族 K 集合坍缩成单点、单行拟合 m=T/2/K 虚高 2 倍 | guard B_copy_f16/B_add 回归修复 |

---

## 5. 参数总表（runtime 读取
`costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json`）

| 参数 | 值 | 用在哪 |
|---|---|---|
| `direct` | simd 295 / simt 291+ 行 | Tier1 |
| `tiny_direct` | 20 行 | Tier1a |
| `group_a` | simd 48 / simt 78 行 | Formula A |
| `loop_b` | simd 60 / simt 59 行 | （Formula B 遗留表，直线/循环公开域当前未直接消费，保留作数据） |
| `sdirect` | 93 行 | （store-only 短行，legacy 路径） |
| `direct_fixed_share` | 0.05 | Tier1/Tier1a 的 A 项显现份额 |
| `tiny_store_chain_share` | 0.636 | Tier1a 的 excess 份额 |
| `l2_cache_bytes` | 134217728 | Formula A warm/cold |
| `p_axis` | 12 | Formula A 的 P 轴档位 |
| `boundary_hide` | 6 行 | calH 表（当前 hideShare 常量=0，由 launch-waves 门实现生效） |
| wave 衰减指数 | 循环体 −0.05；直线 load 侧 −0.32 | 两处 |
| waves 分母 | 56 物理核 | 全部 wave 计算 |

**本轮新增行（calM 攻坚，+40 行）**：
- permute：`n2_s0_b2048_k{4,8,12,16}_g{8,32,56}_m0`（loads-only 体）+
  `n3_s0_b2048_k{8,12,16}_g32_m0`（混流组合体）；
- fused_sm：`n1_s1_b2048_k{1,4,16,64}_g8_m0`（全核减归约项的派生行）+
  `k4_g32_m0`；
- rowred（simt）：`n1_s0_b8192_k{4,16,64}_g8_m0`、`k{4,8,16,64}_g56_m0`
  （multibuffer=True 的 loads-only 体；旧行没开 multibuffer，测浅 2 倍）；
- masked_sm：`n1_s1_b65536_k{4,8,16}_g{8,56}_m0`、`k64_g8_m0`、
  `n1_s1_b48000_k{4,8,16,18}_g8_m0`（精确桶，全核减 vecExec+floors）。

**派生行口径**（泛化性关键）：`T行 = 全真实体实测 − 模型自身已标定的
计算项×K`。这样行值只含记忆侧贡献，且不引入自由系数；同族其他几何
（如 guard 的非掩码 copy）走旧 g56 行，互不干扰（网格括号精确命中后
天然分流）。

---

## 6. 泛化性边界（检视要点）

1. **行缺失时的行为**：每层都有定义好的下一层；Tier3 解析式保证
   setup+带宽下限，永远不会返回 0 或负数。
2. **K 外推**：Tier1a 有 bestMaxK 门；Tier1 不设门但受 `hint ≤ 224` 与
   `useM(K>16 用 mHi)` 的拖尾控制；超过 224 直接落 Tier2 聚合包络。
3. **网格外推**：hint 超过最大测量网格时钳到最大网格（hd14 后精确命中
   用该网格全套 K，缺 K 的下网格不再污染 K 集合）。
4. **chunk 桶**：log2 距离 ≤1 才认领（Tier1）或 clamp [0.5,2]（chunkScale），
   防止把 2KB 行套到 64KB 体上。
5. **掩码语义**：模型不感知掩码时（bodyMasked=0 但真实体掩码），
   用"全核−计算项"派生行吸收；掩码 store ratio<1 仍走 RMW 老路。
6. **混流**：只有"同 stage 已有连续 load"才放宽 StridedWide 小行距 load，
   不影响纯跨步 stage（strided-layout 责任域）。
