# 版本 4：PHYSICS —— 物理第一性重建

每个公式都从物理过程出发构造（传输字节数、BIU 命令数、同块冲突惩罚、warp 发射开销等），
允许 max/min 与乘积惩罚项，不使用 log/指数/幂次。拟合口径：加权最小二乘，权重 1/T（即 MAPE）。

## 汇总表

| 公式 | 场景 | 点数 | 物理项数 | **物理 MAPE** | 贪心对照 | 差异 |
|---|---|---:|---:|---:|---:|---:|
| `simd_load_wide_r1` | SIMD load 宽分支（1D） | 32 | 2 | **4.11%** | 3.46% (6 项) | +0.65 |
| `simd_load_wide_r2` | SIMD load 宽分支（2D） | 222 | 8 | **13.10%** | 17.10% (17 项) | -4.00 |
| `simd_load_gather_r1` | SIMD load 聚集分支（1D） | 284 | 7 | **20.18%** | 12.14% (78 项) | +8.04 |
| `simd_load_gather_r2` | SIMD load 聚集分支（2D） | 678 | 8 | **13.18%** | 11.08% (73 项) | +2.10 |
| `simd_store_r1` | SIMD store（1D） | 291 | 6 | **9.69%** | 4.38% (105 项) | +5.31 |
| `simd_store_r2` | SIMD store（2D） | 851 | 5 | **25.26%** | 5.63% (113 项) | +19.63 |
| `simt_load_r1` | 纯 SIMT load（1D） | 855 | 8 | **10.22%** | 7.50% (89 项) | +2.72 |
| `simt_load_r2` | 纯 SIMT load（2D） | 1666 | 9 | **10.79%** | 7.24% (81 项) | +3.55 |
| `simt_store_r1` | 纯 SIMT store（1D） | 962 | 8 | **18.95%** | 5.47% (113 项) | +13.48 |
| `simt_store_r2` | 纯 SIMT store（2D） | 3224 | 8 | **25.24%** | 17.75% (49 项) | +7.49 |
| | | **9065** | | **18.57%** | | |

---

## 跨公式的关键物理发现

### A. MTE 最小 burst = 32 字节是硬门槛

`C * eb < 32`（一行的有效载荷装不满一个最小 burst）时，编译器无法用「连续块搬运 + extract_slice」，
退化为逐元素命令路径，单元素代价高 10–30 倍。
**两个子代理独立发现了同一门槛**（`simd_load_wide_r2` 的 src=32B gate、`simd_load_gather_r2` 的 C·eb=32B 分区）。
这是之前所有 `a + b·R·max(32,row_bytes)` 型公式卡在 26–28% 的原因：它们只描述了宽路径。

### B. 冲突粒度因场景而异（不是统一的 128B）

| 场景 | 最优粒度 | 来源 |
|---|---|---|
| SIMD gather 1D | **256 B** | 扫描 B∈{16..4096} |
| SIMD gather 2D | **128 B** | 同上，2D 梳状地址集 |
| SIMT load 1D | **256 B** | 与 SIMD gather 一致 |
| SIMT load 2D | **1 KB** | conc1k |
| SIMD store | **128 B** | 写回路径 RMW |
| SIMT store 1D | **256 B** | 建议用 L256 而非 L128 重算 |

### C. 抵扣项（负系数）真实存在

同一 line 被多行命中时只取回一次，所以冲突惩罚应被折扣。
`simd_load_gather_r2` 里 `-71.63 · [C·eb<32] · multi@128`（multi = 被 ≥2 行命中的 128B line 数）价值 0.93 个 MAPE 点。

### D. SIMT 内存侧事务数修正

不是 `iters · mwl`，而是 **`W · iters · mwl`**（所有 W 个 warp 的 line 请求都计入，实测 ≈1.2 ns/请求）。
这是高 W 下 T 继续上升的原因。

### E. 双瓶颈确认

`T = max(k · W, 访存代价)`，实测 `k` = 1.54–2.43 ns/warp（与早期 2.18 一致）。
W≥16 且 mwl≤1 时 63 个点全落在 134–162 ns，与元素数无关。

### F. 被证伪的假设（都是原模型里自设的）

| 假设 | 出处 | 实测结论 |
|---|---|---|
| **128B cache line** | `build_v4_numwarps_dataset.py:32` | ❌ 对 MTE2 不适用；冲突粒度随场景为 128B–1KB |
| **4 KB page** | 同文件 :35 | ❌ CPU 概念；4KB 附近现象是**双稳态**，不是干净阈值 |
| **stride=1 属于 wide 分支** | 照搬 C++ 谓词 | ⚠️ 探针主动拒绝测 stride=1；extract_slice 代价实测 ≈ 0 |

### G. 数据陷阱（影响结论可信度）

1. `simd_load_wide_r1` 的 32 点 stride 全是 2，span 与 n·eb 完全共线，**无法分离 n 与 eb**。
2. `simd_store` 两个槽都**没有 ISB==eb 的连续写形状**，禁止外推回连续写。
3. `simd_load_gather_r2` 子代理**复现不出 73 项 = 11.08%**（它用 614 键池饱和到 16.12%，624 键并集到 13.57% in-sample / 17.57% CV）。该对照值需核实。
4. 探针 rotate 窗口是 step=8192 元素；当 `n·stride > 8192` 时相邻迭代写入区域重叠，斜率可能被缓存命中压低。
5. 实测存在**双稳态**：同一配置两次独立运行可差 2.6 倍（如 `float32 s1 block=1024`）。

---

## `simd_load_wide_r1` — SIMD load 宽分支（1D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，元素域 stride <= 2，rank-1。CAModel 白盒：BIU 连续搬运整块再 extract，biu_cmds 从 n 降到 1-9。

- 路径/操作/秩：simd / load / rank-1
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（1 种）：1
- elements（8 种）：8、16、32、64、128、256、512、4
- 内层步长（字节）（4 种）：2、4、8、16
- 总点数：**32**

**派生物理量：**

- span_bytes = (n - 1) * stride_elements * element_bytes + element_bytes  (连续搬运的字节数)

### 2. 公式与各项物理意义

```
T = 65.3389 + 0.00865176 * span_bytes
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `intercept` | 65.3389 | MTE2 固定启动 + 首字节延迟 |
| `span_bytes` | 0.00865176 | 等效搬运时间；1/coef = 116 GB/s 等效带宽 |

### 3. in-sample 拟合误差

- 点数：**32**
- 物理项数：**2**
- **MAPE：4.11%**
- 贪心对照：**6 项 / 3.46%**（物理式差异 +0.65 个点，项数约为 1/3）

**物理洞察与局限：**

32 点的 stride 全是 2，span 与 n*eb 共线，无法分离 n 与 eb。stride=1 被探针拒绝测量。实测 T(stride2)/T(stride1)=0.96-1.13，extract_slice 代价约等于 0。贪心 6 项=3.46%。

---

## `simd_load_wide_r2` — SIMD load 宽分支（2D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，内层元素域 stride <= 2，rank-2。已剔除 R=1 的伪 2D 形状。

- 路径/操作/秩：simd / load / rank-2
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（1 种）：1
- 形状 (R×C)（23 种）：4x4、4x8、4x16、8x8、8x16、4x1、16x1、16x16、1x16、1x32、1x64、2x8、2x16、2x32、8x4、16x4、16x8、8x32、4x32、2x64、32x8、16x32、32x16
- 内层步长（字节）（5 种）：1、2、4、8、16
- 总点数：**246**

**派生物理量：**

- R,C = tile rows/cols (R>=2; R==1 shapes are 1-D and dropped)
- eb = element bytes (1,2,4,8)
- ISB = inner stride bytes (2*eb for C>=2, eb for C==1)
- rb = (C-1)*ISB + eb     = contiguous byte footprint of one row
- src = C*ISB             = source bytes spanned by one row's strided pattern (= rb + eb when IS=2)
- pitch = OS*eb           = row-to-row distance (bytes)
- span = (R-1)*pitch + rb = total tile address footprint (bytes)
- n = R*C = element count
- GATE: src >= 32 B  <=>  a row's strided source covers at least one full 32-B memory sector

### 2. 公式与各项物理意义

```
src = C*ISB ; [x]+ = max(0,x)
A. src >= 32 B  (wide contiguous-block path)   : T = 67.41 + 2.980*R + 2.176*R*[eb-4]+      (n=157, MAPE 12.51%)
B. src <  32 B and C >= 4 (short-row fallback) : T = 145.7 + 5.384*n + 0.4251*span         (n=33,  MAPE 11.06%)
C. src <  32 B and C == 1 (degenerate column)  : T = 97.97 + 3.273*n                        (n=32,  MAPE 18.05%)
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `A.const` | 67.41 | MTE2 fixed start-up / first-byte latency - reproduces the 65.34 ns constant of simd_load_wide_r1 |
| `A.R` | 2.98 | per-row descriptor issue cost on the wide path (ns/row) |
| `A.R*[eb-4]+` | 2.176 | extra per-row cost for elements wider than 4 B (int64: +8.7 ns/row); the row is handled as more than one 4-B transaction |
| `B.const` | 145.7 | fixed cost of the fallback path for a real 2-D tile |
| `B.n` | 5.384 | per-ELEMENT command cost when a row is too short for a block transfer (ns/element) |
| `B.span` | 0.4251 | address-spread cost of the fallback walk (ns per byte of tile span) |
| `C.const` | 97.97 | fixed cost of a degenerate (R,1) column load |
| `C.n` | 3.273 | per-element cost of the degenerate 1-D column load (ns/element) |

### 3. in-sample 拟合误差

- 点数：**222**
- 物理项数：**8**
- **MAPE：13.10%**
- 贪心对照：**17 项 / 17.10%**（物理式差异 -4.00 个点，项数约为 1/2）

**物理洞察与局限：**

GATE (the key new finding). For every C>=2 row IS=2, hence src = C*ISB = rb + eb. The 222 points split EXACTLY at src = 32 B: 157 points with src>=32 have T in [70.7, 547.9]; 65 points with src<32 have T in [90.9, 2528.0]. src<32 means one row's strided source does not even cover a single 32-B memory sector, so the compiler cannot use the contiguous block-copy + extract_slice lowering (the one confirmed for the wide branch) and the load degenerates to a per-element strided-command path that costs 10-30x more per element. On this grid src>=32 is simultaneously equivalent to C*eb>=16 (>=16 B of payload per row) and to rb >= 32-eb; the data cannot separate these three readings, and no measured shape falls between rb=15 and rb=28. This is why forms such as T = a + b*R*max(32,row_bytes) stall at 26-28%: they model only branch A, and branch A is not byte-driven at all. BRANCH A physics: after the block path is available the cost is 67 ns of start-up plus ~3 ns per row descriptor, plus a large per-row penalty for 8-B elements. Bytes moved (R*rb) and element count n are nearly irrelevant inside A - (R=16,C=8,eb=4,rb=60)=110.9 ns and (R=16,C=16,eb=4,rb=124)=107.1 ns; at fixed R=16,eb=8 the whole C range 4..32 moves T only from 303 to 341 ns, while at fixed n=64 the R range 4..16 moves T from 120 to 303 ns - so branch A is descriptor/row driven, not byte driven. BRANCH B physics: per-element command cost (5.38 ns/elem) plus an address-spread term (0.43 ns per byte of span); same shape as simd_load_gather_r1 but with a far larger per-element constant. BRANCH C: C==1 tiles have no inner dimension, so ISB is meaningless - these 32 points are really 1-D strided loads that the slot rule put here because ISB=eb<=2*eb. They are internally inconsistent (for int8/int64 a larger pitch is FASTER, for float16/float32 SLOWER), so only a 2-term form is defensible; the 1-D formula simd_load_wide_r1 (65.34+0.008652*span) gives 50.5% on them and does NOT transfer. VERDICT: the 8-coefficient physical model (13.10%) BEATS the user's greedy (17.10%, 17 items) and comes within 0.7 pt of a 17-term greedy over a 76-feature library (12.39%); with 15 coefficients it matches it (12.27%). A 2-branch variant that folds C==1 into branch B (T = 77.22 + 8.979*n - 0.06528*span) gives 15.68%. REMAINING RESIDUAL (median 11.2%, p90 26.8%, max 69%) has no structure we could tie to a physical quantity: the worst points are R=4 int64 rows (block path, over-predicted because the int64 per-row penalty is not yet active at R=4) and C==1 columns. The slot also carries irreducible measurement noise (target_spread up to 0.97).

---

## `simd_load_gather_r1` — SIMD load 聚集分支（1D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，元素域 stride > 2，rank-1。每元素一条 BIU 命令（CAModel: biu_send_sizes={eb:n}）。

- 路径/操作/秩：simd / load / rank-1
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（1 种）：1
- elements（8 种）：16、128、8、256、4、512、32、64
- 内层步长（字节）（19 种）：3、4、5、6、8、10、12、16、20、24、32、40、48、64、128、256、512、1024、2048
- 总点数：**284**

**派生物理量：**

- w = 最挤的 256 字节块里的元素数
- L256 = 触及的 256 字节块数
- 块粒度 B=256 由扫描 B in {16,32,64,128,256,512,1024,2048,4096} 得出（最优 256，不是 128）

### 2. 公式与各项物理意义

```
T = 96.03 + 4.2413*(w-1) + 0.7178*n + 0.05572*n*(w-1) + 0.1333*L256 + 0.6548*eb + -0.00983*ISB
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `intercept` | 96.03 | 固定启动开销 |
| `w-1` | 4.2413 | 同块冲突导致的额外启动开销 |
| `n` | 0.7178 | 每元素一条 BIU 命令的基线代价 |
| `n*(w-1)` | 0.05572 | 同块元素的排队代价（惩罚项） |
| `L256` | 0.1333 | 触及的 256B 块数 |
| `eb` | 0.6548 | 数据宽度效应 |
| `ISB` | -0.00983 | stride 越大块越分散，排队越少（减轻惩罚） |

### 3. in-sample 拟合误差

- 点数：**284**
- 物理项数：**7**
- **MAPE：20.18%**
- 贪心对照：**78 项 / 12.14%**（物理式差异 +8.04 个点，项数约为 1/11）

**物理洞察与局限：**

核心物理：每元素一条 BIU 命令，其单价被同 256B 块的拥挤度 (w-1) 抬高。贪心 78 项=12.14%，物理 7 项=20.18%。

---

## `simd_load_gather_r2` — SIMD load 聚集分支（2D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.load，内层元素域 stride > 2，rank-2。已剔除 R=1。

- 路径/操作/秩：simd / load / rank-2
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（1 种）：1
- 形状 (R×C)（21 种）：4x4、4x8、4x16、8x8、8x16、16x16、1x16、1x32、1x64、2x8、2x16、2x32、8x4、2x64、4x32、16x32、8x32、32x16、16x4、16x8、32x8
- 内层步长（字节）（10 种）：3、4、6、8、12、16、24、32、64、128
- 总点数：**750**

**派生物理量：**

- addr(i,j) = i*OS + j*ISB  (bytes, base offset 0 / aligned buffer); R rows x C elements
- L@B    = number of distinct B-byte blocks touched by the 2-D address set
- Lrow@B = sum over rows of the number of B-byte blocks that row touches
- share@B= Lrow@B - L@B   (blocks fetched once but touched by several rows)
- multi@B= number of B-byte blocks touched by >= 2 different rows
- pairs@B= sum over blocks of cnt*(cnt-1)  (same-block element collision pairs; = n-L when cnt=2)
- w@B    = max over blocks of the element count (worst crowding)
- burst_short = 1 if C*eb < 32 else 0   (MTE minimum burst is 32 B)
- block-size scan: burst-short line penalty best at B=64 (15.81%) -> 32 (15.87) -> 128 (17.02) -> 256 (18.89); collision-pair term best at B=128 (15.81%) -> 64 (15.84) -> 32 (16.49) -> 256 (16.81)

### 2. 公式与各项物理意义

```
T = 93.7579 + 0.53865*n + 40.637*B36 + 0.084423*P128 + 13.563*S1024 - 71.629*D128 + 1.328*C + 25.055*BR
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `intercept` | 93.7579 | fixed launch + first-byte latency (ns) |
| `n` | 0.538652 | one BIU command per element (CAModel white box: biu_send_count=n) |
| `B36` | 40.6369 | under-filled 64B block fetches in the burst-short regime ([C*eb<32] x L@64B) |
| `P128` | 0.0844234 | same-128B-line collision pairs sum(cnt*(cnt-1)) over the 2-D address set |
| `S1024` | 13.563 | row x block incidences minus distinct blocks at 1KB grain (share = Lrow-L) |
| `D128` | -71.6294 | 128B lines touched by >=2 rows, x[C*eb<32] (fitted coefficient is NEGATIVE: the same-fetch deduction) |
| `C` | 1.32797 | explicit column count: per-row teeth (row-burst overhead / lane utilisation) |
| `BR` | 25.0555 | row bursts issued in the burst-short regime ([C*eb<32] x R) |

### 3. in-sample 拟合误差

- 点数：**678**
- 物理项数：**8**
- **MAPE：13.18%**
- 贪心对照：**73 项 / 11.08%**（物理式差异 +2.10 个点，项数约为 1/9）

**物理洞察与局限：**

2-D gather is the SAME one-BIU-command-per-element machine as gather_r1 (CAModel: biu_send_count = n), but the 256B crowding computed on the 1-D stride pattern is the WRONG block set for a comb tile: corr(w@256B, T) = 0.121 and corr(w@128B, T) = 0.149, and a 1+n model is 45.4%.  Two mechanisms dominate instead. (1) A REGIME SWITCH at C*eb = 32 B - the MTE minimum burst.  When the payload of one row is narrower than a single minimum burst the tile cannot be walked as row bursts, and each block fetch plus each row burst is paid separately: [C*eb<32]*L@64B (+40.6 ns per 64B block) and [C*eb<32]*R (+25.1 ns per row burst).  This one indicator takes 1+n from 45.4% to 21.1%; it is the same threshold the repo already recorded as 'SIMD 2-4x slower when C*eb < 32' (fit_v11_stepwise comment). (2) SAME-BLOCK CROWDING of the 2-D address set, but at 128 B rather than 256 B: pairs@128B = sum over blocks of cnt*(cnt-1) (elements colliding inside one 128B line), +0.0844 ns per colliding pair. The DEDUCTION asked for is real and measurable: when several rows land in the same 128B line the line is fetched once for all of them, so the model credits -71.6 * [C*eb<32] * multi@128B (multi = 128B lines touched by >= 2 rows); that single term is worth 0.93 MAPE points (15.81 -> 14.88).  In the equivalent parameterisation 49.9*[C*eb<32]*Lrow@64B - 77.0*[C*eb<32]*share@64B the credit (77.0) is 1.55x the price of one row-line incidence (49.9), i.e. sharing across rows is worth more than one full duplicate fetch.  By contrast share@1KB has a POSITIVE coefficient: rows that share a 1KB block are rows whose pitch is small - a comb with dense teeth - and that costs extra. Residuals are flat across dtype (11.4-15.1%), C (11.0-15.2%), regime (13.1/13.3%) and campaign (12.3/14.7%).  The data is intrinsically noisy: target_spread up to 1.51 and the two 2-D campaigns (v1_2d/v3_2d) differ by up to 2.6x on repeated configurations. COMPARISON WITH GREEDY: this 8-term physical form (13.18% in-sample, 13.46% 5-fold CV) beats a fully data-driven greedy over a 624-key pool, which saturates at k=47 -> 13.57% in-sample and 17.57% in CV (i.e. the physics generalises, the greedy overfits). greedy union-pool (624 keys, unconstrained WLS, w=1/T) saturates at k=47 -> 13.57% in-sample, but its 5-fold CV (same term list refit per fold) is 17.57%; parent-reported 73-term greedy = 11.08% could not be reproduced on this exact 678-pt set (my reproductions: 614-key v11 pool k=35 -> 16.12%, same pool with R=1 kept (750 pt) k=53 -> 14.09%)

---

## `simd_store_r1` — SIMD store（1D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.store，rank-1，MTE3 写路径。

- 路径/操作/秩：simd / store / rank-1
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（1 种）：1
- elements（8 种）：16、4、8、32、256、64、512、128
- 内层步长（字节）（19 种）：2、3、4、5、6、8、10、12、16、24、32、40、48、64、128、256、512、1024、2048
- 总点数：**291**

**派生物理量：**

- addr(i) = i*ISB                        # rank-1，基址对齐（用 f_L / f_line_elems_max 全量校验通过）
- addr(r,c) = r*row_pitch_bytes + c*ISB  # rank-2
- kB = 落在同一 B 字节块内的元素数
- mxB  = max kB        (B=32,128)
- SB   = sum kB^2

### 2. 公式与各项物理意义

```
T = 10.75 + 0.4601*n + 87.05*mx128 + 2.977*mx32 + 0.4146*S32 + 0.08581*S512
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `1` | 10.7547 | MTE3 固定启动/首字节延迟 (ns) |
| `n` | 0.4601 | BIU 写命令数 = 元素数 (CAModel: biu_send_count=n)，每元素一条命令 |
| `mx128` | 87.0517 | 最挤 128B line 内元素数 = 该 line 上 read-modify-write 串行链长度 (ns/元素) |
| `mx32` | 2.9766 | 最挤 32B sector 内元素数 (次级 RMW 串行链) |
| `S32` | 0.4146 | sum over 32B sectors of k^2 = 32B 写扇区级冲突对总和 (ns/对) |
| `S512` | 0.0858 | sum over 512B blocks of k^2 = 大块级冲突对总和 (ns/对) |

### 3. in-sample 拟合误差

- 点数：**291**
- 物理项数：**6**
- **MAPE：9.69%**
- 贪心对照：**105 项 / 4.38%**（物理式差异 +5.31 个点，项数约为 1/18）

**物理洞察与局限：**

仿真域: 291 点全部 rank-1，ISB 从 2*eb 到 2048B。物理图像: 每元素一条 BIU 写命令；stride>=2 时每个元素的写只覆盖 128B line 的一小部分 -> 部分写 -> 该 line 需要 read-modify-write；同一 line 上的 k 个写必须串行（line 是单一 RMW 资源），因此关键路径 ~ 最挤 line 的 k (系数 87 ns ≈ 一次 RMW 往返)，这与 r1 中 L=1（所有元素挤在一条 line）时 T ≈ 87-95 ns/元素 完全一致。剩余项 S32/S512 描述 RMW 引擎并发度有限时，不同块之间的冲突对也要付费。独立验证: 用户 1D float16 store stride=2 block=4 -> 实测 374.3 ns，公式 380.7 (+1.7%); block=8 -> 实测 768.4，公式 766.7 (-0.2%)。slot-tuned 变体 one+mx128+S32+S512 (4 项) = 9.84%。数据局限: 本 slot 最小 ISB = 2*eb，**没有任何 ISB==eb 的连续写形状**；连续写会走编译器 wide/整块分支（用户实测 ~79 ns，与 stride=2 的 374/768 ns 差 4.7x/9.8x），该分支本公式不覆盖，禁止外推。

---

## `simd_store_r2` — SIMD store（2D）

### 1. 目标场景与实测条件构成

compile_mode=simd，tl.store，rank-2。已剔除 R=1。

- 路径/操作/秩：simd / store / rank-2
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（1 种）：1
- 形状 (R×C)（23 种）：4x4、4x8、4x16、8x8、16x16、8x16、1x16、2x8、32x16、2x16、1x32、1x64、2x32、8x4、16x8、8x32、4x32、2x64、16x32、16x4、32x8、4x1、16x1
- 内层步长（字节）（12 种）：1、2、3、4、6、8、12、16、24、32、64、128
- 总点数：**943**

**派生物理量：**

- addr(r,c) = r*row_pitch_bytes + c*ISB  (基址对齐, base=0)
- 全部 851 点的 block 统计已与数据集自带 f_L / f_line_elems_max 逐点核对一致

### 2. 公式与各项物理意义

```
T = 0 + 0.2845*n + 85.72*mx128 + 29.65*mx32 + 1.193*S32 + 0.02079*S512
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `1` | 0 | 固定项（NNLS 归零：启动延迟已被 mx 项吸收） |
| `n` | 0.2845 | BIU 写命令数 = 元素数 |
| `mx128` | 85.7206 | 最挤 128B line 内元素数 = RMW 串行链长度 |
| `mx32` | 29.6514 | 最挤 32B sector 内元素数（rank-2 里同一 sector 常被多行共享，权重显著上升） |
| `S32` | 1.1929 | 32B 扇区级冲突对总和 |
| `S512` | 0.0208 | 512B 块级冲突对总和 |

### 3. in-sample 拟合误差

- 点数：**851**
- 物理项数：**5**
- **MAPE：25.26%**
- 贪心对照：**113 项 / 5.63%**（物理式差异 +19.63 个点，项数约为 1/23）

**物理洞察与局限：**

拟合域: 原始 943 点，剔除 92 个 R=1（实为 1D）形状后 851 点，4 dtype、R in {2..32}、C in {1..64}。物理同 r1（每元素一条写命令 + 部分写 RMW），但 rank-2 的 **行分段** 让 RMW 串行链更重：mx128 系数与 r1 相同 (~86)，而 mx32 系数从 r1 的 3.0 涨到 29.7 —— 同一 32B 扇区被同一行的多个元素共享，行内元素间距 ISB 越小、扇区内越挤，RMW 串行链越长。已知结构性局限（残差不是噪声）: 存在字节级地址集合几乎相同、但 T 差 7.2 倍的两点 —— eb=1,C=16,ISB=16,pitch=512,R=32 与 eb=1,C=32,ISB=16,pitch=1024,R=16，两者 n=512、L=64、每线 8 元素、B<=256 的所有 S_B 完全相同，T 却分别为 18943 与 2625 ns；另外在同一地址集合下 eb=1 与 eb=2 也能差 ~2x（r1 完全没有这种 eb 依赖）。这说明 rank-2 还存在一个与 **行分段/发射顺序** 相关的代价分量，纯静态占用统计无法表达，因此 25% 是本特征族的下限。残差偏向: 系统性低估 11.5%，R=32 子群误差 43%。

---

## `simt_load_r1` — 纯 SIMT load（1D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.load，rank-1，LSU 路径。

- 路径/操作/秩：simt_only / load / rank-1
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（7 种）：1、2、4、8、16、32、64
- elements（8 种）：4、16、64、256、8、32、512、128
- 内层步长（字节）（18 种）：2、3、4、5、6、8、10、12、16、24、32、48、64、128、256、512、1024、2048
- 总点数：**855**

**派生物理量：**

- mwl = f_mean_warp_lines = 一个 32 线程 warp 波次触及的不同 128B line 数（按行主序展平元素、每 32 个元素一波）
- L = 整个 tile 触及的 128B line 数
- iters = ceil(n/(32*W)) = 每个 warp 在一个 tl.load 里要发的 load 指令数
- eb = 元素字节数；W = num_warps

### 2. 公式与各项物理意义

```
T = max( 2.327*W , 101.6 + 9.515*min(mwl,4) + 0.4993*L + 19.20*iters + 0.7775*mwl*eb , 0.7780*W*mwl , 0.1652*W*mwl*eb )
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `2.327*W` | 2.327 | warp 发射/调度开销：每个 warp 每轮循环一次 load 指令，k≈2.33 ns/warp（与已知 k≈2.18 一致） |
| `101.6` | 101.6 | 固定启动：首次 128B line 取回的 DCache/DRAM 暴露延迟 |
| `9.515*min(mwl,4)` | 9.515 | 每个 warp 波次前若干个 line 请求串行进入 DCache 流水线（饱和在 4 个未完成请求） |
| `0.4993*L` | 0.4993 | 本 tile 触及的 128B line 总数的吞吐代价 |
| `19.20*iters` | 19.2 | 每个 warp 的第 2..iters 条 load 指令的串行暴露延迟（iters=ceil(n/(32W))） |
| `0.7775*mwl*eb` | 0.7775 | 波次载荷字节数（mwl*eb = 一次 warp 波次搬到的有效字节）的搬运代价 |
| `0.7780*W*mwl` | 0.778 | 全网关 line 请求吞吐：W 个 warp 各产生 mwl 条 line 请求（≈1.29 GB/s/请求 等效） |
| `0.1652*W*mwl*eb` | 0.1652 | 按字节计价的同一请求吞吐（eb>=5 时即 int64 占主导） |

### 3. in-sample 拟合误差

- 点数：**855**
- 物理项数：**8**
- **MAPE：10.22%**
- 贪心对照：**89 项 / 7.50%**（物理式差异 +2.72 个点，项数约为 1/11）

**物理洞察与局限：**

双瓶颈：max(warp 发射 k*W, 访存代价, line 请求吞吐 c*W*mwl)。实测 W>=16 且 mwl<=1 时 T 与元素数无关(134-162ns, n=4..256)，印证 k*W 分支。W*mwl 分支解释了 W=64 时 int64 大 mwl 的跳升（实测 T≈1.2*W*mwl）。贪心 89 项=7.50% 为对照。残余偏差：int64 仍偏低约 -5%。

---

## `simt_load_r2` — 纯 SIMT load（2D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.load，rank-2，LSU 路径。已剔除 R=1。

- 路径/操作/秩：simt_only / load / rank-2
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（7 种）：1、2、4、8、16、32、64
- 形状 (R×C)（34 种）：4x8、4x4、16x16、4x16、8x8、8x16、16x1、1x16、1x32、1x64、2x8、2x16、2x32、8x4、2x64、32x16、4x1、4x32、16x32、32x8、8x2、32x1、16x4、16x2、64x8、8x32、64x1、2x128、4x64、32x2、16x8、64x2、8x64、64x4
- 内层步长（字节）（12 种）：1、2、3、4、6、8、12、16、24、32、64、128
- 总点数：**1762**

**派生物理量：**

- mwl = f_mean_warp_lines（对 rank-2：把 R*C 个元素按行主序展平后每 32 个一波，取各波不同 128B line 数的均值）
- L = tile 触及的 128B line 数；iters = ceil(n/(32*W))；n = R*C
- conc1k = max over 1KB 块 of （该块内被 tile 触及的 128B line 数）

### 2. 公式与各项物理意义

```
T = max( 2.425*W , 109.6 + 1.233*min(mwl,8) + 4.539*min(mwl,4) + 0.8335*L + 10.20*iters + 6.111*conc1k , 0.6423*W*mwl , 0.1737*W*mwl*eb )
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `2.425*W` | 2.425 | warp 发射/调度开销 k≈2.43 ns/warp（rank-2 与 rank-1 同量级） |
| `109.6` | 109.6 | 固定启动：首次 line 取回延迟（真 2D 略高于 1D） |
| `1.233*min(mwl,8)` | 1.233 | warp 波次 line 请求的线性部分（小 mwl 段） |
| `4.539*min(mwl,4)` | 4.539 | warp 波次前 4 条 line 请求的串行进入成本（DCache 流水线未满） |
| `0.8335*L` | 0.8335 | 触及 128B line 总数的吞吐代价 |
| `10.20*iters` | 10.2 | 第 2..iters 条 load 指令的串行暴露延迟 |
| `6.111*conc1k` | 6.111 | 同 1KB 块内的 line 冲突深度（=最挤 1KB 块里被触及的 128B line 数）：comb/2D 结构的 bank 串行化 |
| `0.6423*W*mwl` | 0.6423 | line 请求吞吐：W 个 warp 各 mwl 条请求 |
| `0.1737*W*mwl*eb` | 0.1737 | 按字节计价的请求吞吐（int64 主导） |

### 3. in-sample 拟合误差

- 点数：**1666**
- 物理项数：**9**
- **MAPE：10.79%**
- 贪心对照：**81 项 / 7.24%**（物理式差异 +3.55 个点，项数约为 1/9）

**物理洞察与局限：**

剔除 R=1（1x16/1x32/1x64 本就是 1D 形状）；R=1 的 96 点用同公式外推 MAPE 仅 +2 个点。C<32 跳变：63 对受控配对的 T 比值中位数 1.42（p25=1.10,p75=1.97），而同样配对的 mean_warp_lines 比值中位数=2.00 —— 说明『一个 warp 跨多行』的惩罚已经被 mwl 特征本身编码，额外加 C / 32-C 惩罚项只能再改善 <0.1 个点。贪心 81 项=7.24% 为对照。残余偏差：int64 偏低约 -10%。

---

## `simt_store_r1` — 纯 SIMT store（1D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.store，rank-1，LSU 路径。

- 路径/操作/秩：simt_only / store / rank-1
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（7 种）：1、2、4、8、16、32、64
- elements（8 种）：256、64、4、16、128、32、8、512
- 内层步长（字节）（17 种）：2、3、4、6、8、10、12、16、24、32、48、64、128、256、512、1024、2048
- 总点数：**962**

**派生物理量：**

- L = 触及的 128B line 数（元素地址 j*ISB 覆盖的不同 128B line 数）
- P = Sum over warp 波次(32 lane 连续元素, 行主序展平) of 该波次触及的不同 128B line 数
- S = Sum over warp 波次 of 该波次触及的 128B line 段数(rank-1 恒为每波次 1)
- W = num_warps
- [eb<4] = 亚 32-bit 存储指示（int8/float16=1, float32/int64=0）
- wf = 4/min(eb,4) = 32-bit LSU 数据通路的欠填浪费因子 (int8=4, float16=2, float32=int64=1)

### 2. 公式与各项物理意义

```
T = max( 24.64 , 2.184*num_warps , 1.3915*L*[eb<4] + 7.6308*(S/W) + 0.48866*(L/eb) + 0.029833*(n*eb) + 0.013872*[eb<4]*n + 1.9038*wf + 0.039732*L*wf + 0.266406*[eb<4]*P/W )
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `intercept` | 24.64 | 固定发射地板（实测 25ns，36.5% 的点落在这里） |
| `num_warps` | 2.184 | 每 warp 发射开销 k≈2.18 ns/warp（W=16/32/64 处实测 34/68/140ns 拟合） |
| `L*[eb<4]` | 1.3915 | 窄存储时的 line 事务代价 |
| `S/W` | 7.6308 | 每 warp 的行段数 = warp 波次的主事务数 |
| `L/eb` | 0.488664 | 每字节对应的 line 数（窄 dtype 单位 line 承载的字节更少，代价更高） |
| `n*eb` | 0.029833 | 总写入字节数（带宽项） |
| `[eb<4]*n` | 0.013872 | 窄存储的元素级代价 |
| `wf` | 1.90379 | 32-bit 数据通路欠填因子本身的代价 |
| `L*wf` | 0.039732 | 欠填因子加权的 line 事务 |
| `[eb<4]*P/W` | 0.266406 | 窄存储下每 warp 的 line 事务 |

### 3. in-sample 拟合误差

- 点数：**962**
- 物理项数：**8**
- **MAPE：18.95%**
- 贪心对照：**113 项 / 5.47%**（物理式差异 +13.48 个点，项数约为 1/14）

**物理洞察与局限：**

双瓶颈 T=max(发射, 访存)。25ns 地板来自循环发射开销（T 是对 iters 的斜率，固定启动已被扣掉）。W>=16 后 T 由 k*W 主导，W<=8 时由访存主导，与 W>=16 时 T 与元素数无关的既有结论一致。亚 32-bit 存储（int8/float16）存在明显的窄存储惩罚，且只在大 stride（ISB>=32）时暴露：同地址图案下 f16/f32 的 T 比值 ISB<=16 时≈1.0，ISB=32 时≈1.5，ISB>=64 时≈3.0-3.5。贪心 113 项 = 5.47%，本物理式 18.95%（明显劣于贪心，主因是 r1 在 ISB 中段存在未被公式捕获的非单调结构，例如 ISB=64/128 时 T 随 n 近似线性但斜率随 ISB 缓变）。

---

## `simt_store_r2` — 纯 SIMT store（2D）

### 1. 目标场景与实测条件构成

compile_mode=simt_only，tl.store，rank-2，LSU 路径。已剔除 R=1。

- 路径/操作/秩：simt_only / store / rank-2
- dtype（4 种）：float16(2B)、float32(4B)、int64(8B)、int8(1B)
- num_warps（7 种）：1、2、4、8、16、32、64
- 形状 (R×C)（33 种）：4x8、16x16、4x4、4x16、8x8、8x16、2x16、8x4、16x4、16x8、1x16、1x32、1x64、2x8、2x32、8x32、16x32、32x8、32x16、4x32、2x64、8x2、16x1、16x2、32x1、32x2、64x1、64x2、2x128、4x64、64x4、8x64、64x8
- 内层步长（字节）（11 种）：2、3、4、6、8、12、16、24、32、64、128
- 总点数：**3416**

**派生物理量：**

- 已剔除 R=1 的伪 2D 形状：3416 -> 3224 点（192 点剔除）
- R,C = 形状 (行列)；n = R*C；pitch = f_row_pitch_bytes
- warp 波次 = 32 个行主序连续元素（SIMT 把 (R,C) tile 展平到 32 lane）
- P = Sum over warp 波次 of 该波次触及的不同 128B line 数（带重复计数）
- S = Sum over warp 波次 of 该波次跨的行段数；P/S = 每行段平均 line 数
- L = 整个 tile 触及的不同 128B line 数
- [eb<4] = 亚 32-bit 存储指示；wf = 4/min(eb,4)

### 2. 公式与各项物理意义

```
T = max( 24.59 , 1.542*num_warps , 0.123364*L*wf + 0.0284396*min(R,32)^2*C + 0.225891*[eb<4]*n + 0.159854*L*eb + 2.23596*L*[eb<4] + 0.582813*num_warps + 0.964844*(P/S) + 0.00891571*[eb<4]*min(R,32)^2*C )
```

| 项 | 系数 | 物理意义 |
|---|---:|---|
| `intercept` | 24.59 | 固定发射地板 |
| `num_warps` | 1.542 | 每 warp 发射开销（r2 实测拟合 1.54 ns/warp，落在已知 1.5-2.2 区间内） |
| `L*wf` | 0.123364 | 欠填因子加权的 line 事务（窄存储每条 line 的有效字节更少） |
| `min(R,32)^2*C` | 0.02844 | warp 跨行惩罚：一行装不满一个 warp(C<32) 时行数被反复重访；实测 T 峰值出现在 R=32(=warp 宽度)，故取 min(R,32) |
| `[eb<4]*n` | 0.225891 | 窄存储的元素级代价 |
| `L*eb` | 0.159854 | 有效字节 × line 数（带宽项） |
| `L*[eb<4]` | 2.23596 | 窄存储的 line 事务代价 |
| `num_warps` | 0.582813 | warp 发射的线性附加项 |
| `P/S` | 0.964844 | 每个行段平均的 line 数（行越长、跨行越少则该项小） |
| `[eb<4]*min(R,32)^2*C` | 0.008916 | 窄存储下的跨行惩罚放大 |

### 3. in-sample 拟合误差

- 点数：**3224**
- 物理项数：**8**
- **MAPE：25.24%**
- 贪心对照：**49 项 / 17.75%**（物理式差异 +7.49 个点，项数约为 1/6）

**物理洞察与局限：**

双瓶颈 T=max(发射, 访存)。实测 T 对 (dtype,R,C,ISB,W) 几乎是确定性函数（同配置重复测量组内 MAPE 仅 5.5%），但对 W 近似无关（W=1..32 同形状 T 差 <15%），只在 W=64 处出现 k*W 瓶颈。核心现象：固定 n 时 T 随 R 每翻倍跳升约 2.2-2.6 倍，峰值出现在 R=32（=warp 宽度），R>32 后因行间共享 line 而回落（如 n=256: R=4->38, R=8->98, R=16->219, R=32->549, R=64->308）。主要残余：R>=32 的强跳变只被 min(R,32)^2*C 部分捕获，该区段仍系统性低估 50-65%，这是 25.24% 误差的主要来源。贪心 49 项 = 17.75%，本物理式 25.24%。

---
