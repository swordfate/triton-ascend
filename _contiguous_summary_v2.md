# Triton-Ascend 连续访存（stride == 1）代价标定

> 本文按**最终标定结果**描述模型：公式、数据、解释、验证。历史过程与被取代的结论压缩到第 8 节。
> 模型位置：`lib/AscendModel/RouteModel/Models/ContiguousLdStCostModel.cpp`，
> 参数：`profiles/simd_simt/ascend_950pr_simd_simt_v1.json` 的 `simd/simt.memory.contiguous_ldst`。

## 1. 这是什么

对 `stride == 1` 的 load / store，给出**单次访存的执行代价**（system cycles），供 RouteModel 计算每个 stage 的访存分量。

覆盖范围：

- 元素宽度：int8 / fp16 / fp32 / int64；
- 单次访存字节数：1 B ~ 128 KiB；
- 线程组织：SIMD（向量）与 SIMT（每线程 `te` 个元素、`W` 个 warp）。

两条路径分开定价：

| 路径 | 定价方式 |
|---|---|
| SIMD | 一条闭式公式（按字节数）|
| SIMT | 按 `te`、字节数、元素宽度分段的闭式公式 + 大 `te` 的实测表 |

不在范围内：多条访存并发的相互影响（NACC>1 的现象记在第 7.4/7.5 节，但**未进公式**）。

## 2. 两种输入形态

分类依据：**TTIR 里有没有 `mask`**。

### 2.1 形态 A：无掩码

[test_ubtuner.py:53-58](third_party/ascend/unittest/ubtuner_ut/test_ubtuner.py#L53-L58)：

```python
@triton.jit
def kernel_allowed(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)          # 块宽 = N，无 mask
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + offs, a * b)
```

CostModel 能拿到：块宽 `B`、dtype、`W`、方向。走 §3.2。

### 2.2 形态 B：带掩码

[01-vector-add.py:51-61](third_party/ascend/unittest/autotune_ut/01-vector-add.py#L51-L61)：

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements                     # n_elements 是运行期参数
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(output_ptr + offsets, x + y, mask=mask)
```

TTIR 里是 `tt.splat %n_elements` → `cmpi slt`，**有效元素数不可知**。
CostModel 能拿到：块宽、dtype、`W`、方向、**「有掩码」这一位**。走 §3.3。

### 2.3 两形态对照

| | 形态 A | 形态 B |
|---|---|---|
| TTIR | 无 `mask` | 有 `mask`，有效数不可知 |
| 公式 | §3.2 | §3.3 |
| 数据 | `raw/raw_pure_unified.jsonl`（755 点）<br>+ `raw/raw_simd_npow2_A.jsonl`（268 点，SIMD 非 2 幂）| `raw/raw_formb_v2.jsonl`（722 点）<br>+ `raw/raw_simd_npow2_B.jsonl`（136 点，SIMD）|

> **掩码按块宽计价**。实测 634 对同块宽样本：有效元素数减半只省 13%（比值中位 0.87、p90 1.01），
> 所以块宽是一阶正确变量，再细分有效数没有收益。

---

## 3. 标定数据

### 3.1 测量方法

- **k-marginal**：同一 kernel 跑 `iters ∈ {2048, 8192}` 两次，`cost = (t(8192) − t(2048)) / 6144`；
- `grid = 1`，每程序在 `k_low` 个块的窗口内轮转；
- 单次 `spread > 5%` 丢弃；
- **整批采集**：同一轮标定的所有点必须在一次连续作业里跑完（见 §2.3 的跨时段漂移）；
- **每次采集前必须跑 `scripts/probe_speed.py`**：本例实测 65.6~66.3 ns/iter 为健康，>150 不可用。
  降级卡能让数据整体慢 3~20 倍，而单次测量内部完全自洽，事后无法识别。

### 3.2 总体测试范围

#### 各维度的取值（所有公式共用）

| 项 | 范围 |
|---|---|
| **dtype** | int8 / fp16 / fp32 / int64（`eb` = 1 / 2 / 4 / 8 B）|
| **方向** | load / store |
| **形态 A · SIMD 块宽** | 34 档：2 的幂 `2…16384` **＋ 非 2 幂** `31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1000, 1023, 1025, 1536, 2047, 2049, 3072, 4095, 4097, 6144, 8191, 8193, 12288, 16383` |
| **形态 A · SIMT 元素数** | `1, 2, 4, 8, …, 16384`（只能 2 的幂）|
| **形态 B · SIMD 块宽** | 同 A 的 34 档（含非 2 幂）|
| **形态 B · SIMT 块宽** | `2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192`（13 档，只能 2 的幂）|
| **`num_warps`** | 1 / 2 / 4 / 8 / 16 / 32 / 64（SIMT）；SIMD 固定 1 |
| **二维几何** `(R,C)` | `32×64, 32×128, 64×64, 16×256, 31×127, 32×127, 31×128, 63×33` |
| 起点对齐 | 32 B |
| `grid` | 1 |
| k（`iters`）| 2048 / 8192 |
| 测量重复 | 每配置 3 次取最小值（压 §2.3 那个双峰）|

#### 各物理量的实测覆盖

| 物理量 | 定义 | 覆盖范围 |
|---|---|---|
| `bytes` | `n·eb` | **1 B ~ 131 KB** |
| `lines` | `bytes/128` | **0.01 ~ 1024** |
| `te` | `n/(32W)` | **0.001 ~ 512** |
| **`LDG`** | `⌈te·eb/8⌉` | **1 ~ 256** |
| `schedR` | `⌈W/4⌉` | **1 ~ 16** |
| `spill` | `2·te·eb > regB(W)` | **覆盖**（形态 B 里 4.8% 的点溢出）|
| 块宽是否 2 的幂 | — | **SIMD 两侧都覆盖；SIMT 只能 2 的幂** |

#### 各公式的配置数

| 公式 | 数据文件 | 配置数 | MAPE |
|---|---|---:|---:|
| SIMD load | `raw_simd_npow2_A.jsonl` | 133 | **6.97%** |
| SIMD store | `raw_simd_npow2_A.jsonl` | 135 | **5.24%** |
| SIMT A load | `raw_pure_unified.jsonl` | 251 | **7.73%** |
| SIMT A store | `raw_pure_unified.jsonl` | 324 | **8.47%** |
| SIMT B load | `raw_formb_v2.jsonl` | 352 | **9.47%** |
| SIMT B store | `raw_formb_v2.jsonl` | 361 | **9.34%** |
| 二维验证 | `raw_tile2d_verify.jsonl` | 80 组 | 见 §5.1 |

#### 非 2 幂的覆盖

| | 形态 A | 形态 B |
|---|---|---|
| **SIMD** | ✅ 189 / 268 点 | ✅ 96 / 136 点 |
| **SIMT** | ❌ 不可能（§4.3）| ❌ 不可能 |

SIMT 的块宽只能是 2 的幂 → 其 `lines = bytes/128` 也稀疏，这是若干薄段点数少的根本原因（§6）。

#### 其它文件（过程记录 / 已被取代）

| 文件 | 条数 | 状态 |
|---|---:|---|
| `raw_dynmask_anchor.jsonl` | 285 | 用于 §1.3 的 634 对静态/动态对比 |
| `raw_dynmask_footprint16mb.jsonl` | 239 | 已被 `raw_formb_v2` 取代 |
| `raw_npow2_fp.jsonl` + `raw_npow2_w14.jsonl` | 1459 | 旧「按有效元素数 n 分段」的掩码数据，该分支已不采用 |
| `raw_unmasked_npow2_*.jsonl` | 708 | 同上对照组 |
| `raw_simd_npow2.jsonl` | 40 | SIMD 非 2 幂的早期小批，已被密集版取代 |
| `raw_tile2d.jsonl` | 1170 | 早期二维探针，被 `raw_tile2d_verify` 取代 |
| `raw_ablation_program_gap.jsonl` 等 | 83 | 探针设计过程记录 |

### 3.3 数据质量

| 项 | 值 |
|---|---|
| 形态 B 采集时的卡健康值 | 65.6~66.3 ns/iter，**7 次锚点波动 ±0.5%** |
| spread 中位 | 形态 B **0.14%**；SIMD 非 2 幂 **0.026%（A）/ 0.035%（B）** |
| 失败点 | 形态 B 6 点、SIMD 非 2 幂 3 点，全是寄存器/UB 溢出到无法编译的极端组合 |
| **跨时段漂移** | 同一配置在同一时段内**极稳**（B=2048 四次全 104.3；B=4096 四次 149.1~149.2；B=8192 四次 271.2~271.7，波动 ≤0.2%），但**绝对水平会随卡的状态漂移**：B=8192 在两小时内曾从 ~329 变到 ~272（−17%），B=4096 也出现过 149 / 194（+30%）交替。**所以必须整批采集、批内比较，绝不能跨天拼接数据**；§2.1 的健康锚点只能发现严重降级，发现不了这种 20~30% 的漂移 |

---

## 4. 最终公式

### 4.1 记号

| 记号 | 定义 | 含义 |
|---|---|---|
| `B` / `n` | 块宽 / 元素总数 | — |
| `eb` | 元素字节（1/2/4/8）| — |
| `bytes` | `n·eb` | 搬运字节数 |
| `W` | warp 数（一个 warp = 32 lane）| — |
| `te` | `n/(32W)` | 每线程元素数 |
| **`LDG`** | **`⌈te·eb/8⌉`** | **每线程的 load/store 指令条数（一条搬 8 B）** |
| `lines` | `bytes/128` | 128 B DCache line 数 |
| `schedR` | `⌈W/4⌉` | warp 调度轮数（每核 4 个 scheduler）|
| `fill` | `32·eb/256` | 一个 warp 占 256 B 向量宽度的比例 |
| `warpB` | `32·eb` | 一个 warp 的字节宽度 |
| `kof(x)` | `2^min(2,⌊log₂ max(1,x)⌋)` | 分档常数 1/2/4 |
| `regB(W)` | W≤8→508、16→256、32→128、64→64（字节）| 每线程寄存器档位 |
| `spill` | `2·te·eb > regB(W)` | 寄存器溢出（系数 2 = 累加值 + 载入值）|

### 4.2 形态 A 的公式

#### SIMD load / store（**4 个系数**，133 / 135 点，6.97% / 5.24%）

```
T_load  = 66.38 + 0.005412·bytes + 5.80·[bytes%32≠0] + 73.06·(⌈bytes/16384⌉−1)
T_store = 83.24 + 0.004262·bytes + 10.69·[bytes%32≠0] + 40.40·(⌈bytes/16384⌉−1)
```

| 项 | 物理含义 |
|---|---|
| `a` | MTE 搬运的固定启动代价（descriptor 下发 + 首个 burst 响应）|
| `b·bytes` | 整块连续字节的搬运时间 |
| `c·[bytes%32≠0]` | **最后一个 32 B DataBlock 不满**时要单独发一次带掩码的块 |
| `d·(⌈bytes/16384⌉−1)` | **每超过 16 KB 就要多一个 descriptor**（32 KB 传输 ≈ 2 段）|

这两条同时覆盖 2 的幂与非 2 的幂块宽，**不需要分开建**。
分 dtype 表现：load 4.0 / 4.1 / 7.4 / 12.6%（int8/fp16/fp32/int64），store 3.6 / 5.4 / 4.9 / 7.0%。

> 与旧版对比：旧 ①（`65.78 + 0.006112·bytes`）只在 2 的幂上标定，6.37%；新版在对齐子集 6.85%、
> 在含非 2 幂的 133 点上 6.97%，且 store 从 6.12% 改善到 4.89%（对齐子集）。

#### SIMT load（3 条公式，251 点，**7.73%**）

| 段 | n | 公式 |
|---|---:|---|
| `lines ≤ 2` | 153 | `40.26 + 83.27·[bytes≥128] + 1.121·⌈W/4⌉ + 9.890·fill + 0.1282·part·bytes + 0.01557·W²` |
| `2 < lines ≤ 32` | 79 | `109.58 + 5.435·rounds + 0.1223·32·eb + 1.761·⌈W/4⌉ + 33.57·1/W + 0.002717·K1` |
| `lines > 32` | 19 | `-40.52 + 510.8·1/W + 3.361·lines + 56.53·spill·lines − 0.0001990·bytes·max(0,W−16) + 1.423·[te<1]·W` |

#### SIMT store（5 段，按 C++ 选择规则）

选择规则（与 `ContiguousLdStCostModel.cpp` 一致）：`te < 1` 走 tail 段；`te >= 1` 后，
`int8` 单独一段、`W <= 4` 一段、`W > 4` 再按元素宽度分三段。

| 段 | n | MAPE | 公式（截距 + Σ coef·feat）|
|---|---:|---:|---|
| `te>=1,int8` | 34 | 25.77% | `7.512 + 0.1027·LN`（`waves_by` 在实现中恒为 0）|
| `te>=1,wide,W<=4` | 71 | 11.83% | `25.35 + 0.9759·rounds + 0.0258·byh1k` |
| `te>=1,wide,W>4,eb2` | 22 | 16.73% | `20.75 + 0.0050·by + 0.0082·K1 − 0.0008·LN` |
| `te>=1,wide,W>4,eb4` | 20 | 11.89% | `11.62 + 0.7454·lines + 0.0040·K1 + 0.0019·LN` |
| `te>=1,wide,W>4,eb8` | 16 | 6.49% | `2.982 + 3.995·lines − 17.11·schedR + 2.230·rounds − 20.78·invW` |

特征定义（C++ `fillFeatures`）：

```
by      = elements × elemBytes        lines   = by / 128
te      = elements / (32·W)           rounds  = ceil(te)
byh1k   = max(0, by − 1024)           LN      = by · ln(1 + te)
K1      = by · kof(lines / W)         kof(x)  = 2^clamp(floor(log2(max(x,1))), 0, 2)
schedR  = ceil(W / 4)                 invW    = 1 / W
```

> `te<1` 的 tail 段本次未改动。

### 4.3 形态 B 的公式

> **系数约定**：形态 B 用 `T = Σ coefs[i]·feats[i]`，其中 `one` 是截距占位；
> 形态 A（`FINAL_SPEC.json`）用 `T = coefs[0] + Σ coefs[i+1]·feats[i]`。两者不同，勿混用。

#### SIMD：与形态 A 完全相同

实测 136 对同几何样本，**比值中位 1.000（p10 0.997、p90 1.002）**，在全部 34 档块宽（含非 2 幂）上都成立。
机制：mask 变成 `memref.subview [%10]` 的 DMA 长度参数，仍是一次 DMA。
所以 SIMD 形态 B 直接用 §3.2 的两条公式，**零新增系数**。

#### SIMT load（4 段，352 点，**9.47%**）

| 段 | n | MAPE | 公式 |
|---|---:|---:|---|
| `lines<=2` | 179 | 7.50% | `5.180·te·eb + 38.50·LDG + 0.02608·W² + 74.33·[bytes≥128] + 12.06·wpl + 7.263·[int8]` |
| `2<l<=32` | 109 | 9.38% | `18.32·te + 30.41·LDG + 0.01433·W² − 52.68·1/W + 78.74·[bytes≥128] + 0.2053·32·eb` |
| `l>32_nospill` | 50 | 17.92% | `1.236·lines + 34.39·te + 10.45·LDG + 10.34·⌈W/4⌉ − 0.01819·W² + 529.8·(⌈bytes/16384⌉−1)` |
| `l>32_spill` | 14 | 5.31% | `106.4·lines + 69.05·te − 12.80·te·eb − 5589·W + 2342·1/W + 988.5·[fp16]` |

#### SIMT store（6 段，361 点，**9.03%**）

| 段 | n | MAPE | 公式 |
|---|---:|---:|---|
| `te<1` | 194 | 3.93% | `27.12 + 4.135·⌈W/4⌉ + 0.01553·W² − 0.02295·32·eb − 5.712·[int8] − 1.463·[fp16]` |
| `narrow_l<=8` | 36 | 15.88% | `0.05427·bytes + 10.58·te − 2.093·te·eb + 19.19·LDG − 2.360·W + 0.06397·W²` |
| `narrow_8<l<=32` | 26 | 18.03% | `−14.40·⌈W/4⌉ − 430.3·fill + 6.411·⌈n/32⌉ + 28.05·lines/⌈W/4⌉` |
| `narrow_l>32` | 21 | 14.76% | `0.3752·bytes + 31.56·te − 288.9·te·eb + 2177·LDG − 0.6745·W² − 755.2·[fp16]` |
| `eb4_W<=4` | 48 | 13.96% | `3.182·lines + 9.565·te − 2.662·te·eb + 11.24·LDG + 23.10·1/W + 3.219·spill·LDG` |
| `eb4_W>4` | 36 | 13.22% | `0.03330·bytes + 22.72·te − 11.45·te·eb + 67.22·LDG − 4.823·W + 0.02562·W²` |
分段依据：

| 判据 | 物理含义 |
|---|---|
| `te < 1` / `te ≥ 1` | 每线程不足一个元素（多数线程空转）/ 满负荷 |
| `narrow`（int8 或 fp16）| 一个 warp 只占 **32 B / 64 B**，都填不满一条 128 B line → 写侧要 RMW |
| `lines` 分档 | 触及几条 128 B line：少 / 中 / 带宽主导 |
| `eb ≥ 4` 按 `W ≤ 4` / `W > 4` | 一条 warp 指令正好覆盖 1~2 条 line；`W` 大时 floor 随寄存器档位抬高 |
| `spill` | `2·te·eb > regB(W)` 触发寄存器溢出 |

### 4.4 汇总

| 形态 | 公式 | 段数 | 系数 | 数据点 | MAPE |
|---|---|---:|---:|---:|---:|
| A | SIMD load | 1 | 4 | 133 | **6.97%** |
| A | SIMD store | 1 | 4 | 135 | **5.24%** |
| A | SIMT load | 3 | 18 | 251 | **7.73%** |
| A | SIMT store | 4 | 21 | 324 | **8.47%** |
| B | SIMD | 0 | 0 | — | 复用 A |
| B | SIMT load | 4 | 24 | 352 | **9.47%** |
| B | SIMT store | 6 | 34 | 361 | **9.03%** |

七条公式 **全部 < 10%**，合计 **1 556 个实测点**。单段不要求各自 <10%（见 §6）。

## 5. 物理解释

### 5.1 有硬件出处的量

| 量 | 定义 | 硬件过程 |
|---|---|---|
| `bytes` | `n·eb` | 要搬运的字节数（MTE 计价单位）|
| `blocks` | `bytes/32` | 32 B DataBlock 数，MTE 最小搬运粒度 |
| `lines` | `bytes/128` | SIMT DCache line 数（line = 128 B）|
| **`LDG`** | `⌈te·eb/8⌉` | 每线程 load/store 指令条数，**一条搬 8 B** |
| `schedR` | `⌈W/4⌉` | 每核 4 个 warp scheduler，`W` 个 warp 分几轮发射 |
| `fill` | `32·eb/256` | 一个 warp 占 256 B 向量宽度（VL=256 B）的比例 |
| `warpB` | `32·eb` | 一个 warp 的字节宽度：fp16 64 B / fp32 128 B / int64 256 B |
| `regB(W)` | 508/256/128/64 B | 每线程寄存器档位 |

### 5.2 四条实测物理规律

**① SIMT 的每条 LDG 搬 8 B**

fp32 load，W=1，无溢出：`cost/LDG` 恒为 **93 / 88.5 / 88.6 / 86.1**（te = 4/8/16/32），
溢出后跳到 141 / 184。相关性从 `te` 的 0.745 提升到 `LDG` 的 **0.880**。

**② 只依赖 `W` 的发射地板**

`B ≤ 32` 时成本与 `B` 无关（int8 load）：W=1..8 → **48**，16 → 55，32 → 79，64 → **167**。
`47 + W²/36` 拟合得很准（16→54、32→75、64→161）。

**③ 跨过 128 B 的台阶**

fp32 load：`B=8`（32 B）→ 47，`B=32`（128 B）→ 138。触及第一条 DCache line 时要付 `83.27·[bytes≥128]`。

**④ 寄存器悬崖**

判据 `2·te·eb > regB(W)`。`B=4096, fp32, W=1`（te=128，2×512 B > 508 B）实测 **1560.2**，
而 `B=2048` 同配置只有 **602** —— 块宽翻倍、成本涨 2.6 倍。

### 5.3 非 2 幂块宽的底层机制

**SIMD 允许非 2 的幂块宽，SIMT 不允许**（实测，同一份探针只改 `compile_mode`）：

| 模式 | BLOCK=127 |
|---|---|
| SIMD load / store | ✅ 80.4 / 103.1（对照 128 → 68.1 / 82.5）|
| SIMT load / store | ❌ `ValueError: arange's range must be a power of 2` |

**IR 里没有 padding**。编译 `BLOCK=127` 与 `128` 两份，TTIR 与 ttadapter **结构完全相同**
（都是 2994 / 3760 字节，归一化后 diff 为空），形状就是 `tensor<127xf32>` / `memref<127xf32>`。
所以代价不是编译器补齐，而是**硬件侧真的搬了 508 B**：

| 项 | 来源 |
|---|---|
| `[bytes % 32 ≠ 0]` | 508 B = 15.875 个 DataBlock，最后一块不满 → 要单独发一次带掩码的块 |
| `⌈bytes/16384⌉ − 1` | 每超过 16 KB 要再发一个 descriptor |

两条都由密集数据（34 档块宽 × 4 dtype × 2 方向）验证：加上这两项后 133/135 点的 MAPE 为 6.97% / 5.24%。

#### 为什么 SIMD 能、SIMT 不能：代码链路

1. **前端按模式构造 builder**（`python/triton/compiler/code_generator.py`，由 patch 引入）：

```python
compile_mode = "simt" if (hasattr(options, "force_simt_only") and options.force_simt_only) else "simd"
self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, getattr(options, "arch", ""),
                                                     compile_mode=compile_mode)
```

2. **builder 自报模式**（`third_party/ascend/ascend_ir.cc:66`）：

```cpp
bool isSimtMode() const { return compile_mode == "simt"; }
```

暴露为 `is_simt_mode()`（`ascend_ir.cc:685`）。

3. **`arange` 的 2 幂校验按模式分支**（`third_party/ascend/patch/triton-ascend-3.6.0.patch`）：

```diff
-        if (range & (range - 1)) != 0:
-            raise ValueError("arange's range must be a power of 2")
+        # Check if compile_mode is simt, then range must be a power of 2
+        if self.builder.is_simt_mode():
+            if (range & (range - 1)) != 0:
+                raise ValueError("arange's range must be a power of 2")
```

同一份 patch 还注释掉了 block shape 的 2 幂校验：

```diff
-            raise ValueError(f"Shape element {i} must be a power of 2")
+        #     raise ValueError(f"Shape element {i} must be a power of 2")
```

这就是 `(31,127)` 这种二维块能直接建出来的原因。

| `compile_mode` | `force_simt_only` | builder 的 mode | 非 2 幂 `arange` |
|---|---|---|---|
| `simd` | False | `"simd"` | ✅ 允许 |
| `simd_simt` / `simd_simt_template`（混合）| False | `"simd"` | ✅ 允许 |
| **`simt_only`** | **True** | **`"simt"`** | ❌ 拒绝 |

> **注意**：仓库工作区的 `python/triton/language/semantic.py` 仍是**无条件**校验——
> 源码树里那份 `python/triton` 是未打 patch 的原始 Triton，门控在 patch 里、打包时才应用。
> 只看仓库源码会误判「SIMD 也不支持」，必须看安装包或实测。

#### 关于「32 B 对齐要不要 padding」

曾有一个说法：SIMD 访存要求 32 B 对齐，所以 `(32,127)` 展平后天然对齐、`(31,127)` 对不齐，可能要走 padding。**实测与代码都不支持这个说法**：

| 几何 | 元素数 | fp32 字节数 | 32 B 对齐 | 实测 cost（load）|
|---|---:|---:|---|---:|
| `(32,127)` | 4064 | 16256 = **508 × 32** | ✅ | 150.2 |
| `(31,127)` | 3937 | 15748 = 492 × 32 + **4** | ❌ | 155.1 |

差别只有 **+4.9 cycle（+3.3%）**，而公式里 `c·[bytes%32≠0]` 一项就是 **+5.80** —— 与实测完全对上。
如果是「另走一条 padding 路径」，代价应该接近掩码路径那种量级（2.3 倍），而不是 3%。

**代码侧也没有针对连续访存的 padding**：

| 位置 | 检查结果 |
|---|---|
| TTIR / ttadapter | `tensor<127xf32>` / `memref<127xf32>`，与 128 版**结构完全相同**（§4.3）|
| `lib/TritonToLinalg/LoadStoreConverter.cpp` | 无 contiguous/32 B 对齐分支，直接 `memref.copy` 原长度 |
| `lib/AutoBlockify/*` | 无 padding / align 逻辑 |
| `lib/TritonToLinalg/TritonOpConverter.cpp:3719-3724` | 有 32 B 对齐标注，但**只用于 `index_select` 的跨步 DMA**（`hfusion.stride_align_value_in_byte = 32`），与连续访存无关 |

**SIMD 与 SIMT 在这件事上没有差别**——两者都走同一套 `TritonToLinalg`，都没有 padding pass。
差别只在前端 `arange` 能否接受非 2 幂（§4.3 的代码链路）。

所以：**连续访存的「不对齐」只体现为最后一个 32 B DataBlock 不满，代价是一次固定的额外发出，不需要建模成 padding 路径。**

**物理原因**：SIMD 走 MTE 块搬运，只需要「从哪、多少字节」，块宽任意；
SIMT 的线程布局是 32 lane × `W` 个 warp，块必须能被 lane 数整齐切分。
`c` 与 `d` 都为正，符合「额外发出一次」的物理解释。

### 5.4 纯拟合项

| 项 | 出现在 | 说明 |
|---|---|---|
| `K1 = bytes·kof(lines/W)` | SIMT A ③⑤ | 把「每 warp 的 line 数」压成 1/2/4 三档，分档数本身是拟合的 |
| `K2 = bytes·2^min(2,⌊log₂(te/16)⌋)` | SIMT A ⑤ | 同上，按 `te/16` 分档 |
| `W²` / `ln(1+W)` | SIMT A ③⑤ | 地板随 `W` 的增长形式，二次/对数都只是拟合 |
| `lines·W`、`LDG·⌈W/4⌉` | SIMT B | 交互项，有量纲意义但系数无独立出处 |

### 5.5 一条必须写明的偏差

SIMD 的掩码**不是无条件免费**——取决于是否真的掩掉了东西：

| KEEP（块宽 4096）| 4096（无掩码）| 4095 | 4064 | 3072 | 2048 |
|---|---:|---:|---:|---:|---:|
| cost | **2682** | 6338 | 6420 | 5990 | 5343 |

只掩 1 个元素就已 2.36 倍——**掩码是开关，不是比例**。
§3.3 说「动态 mask 免费」是因为那次实验里 `M = B`（实际全 true），开关没触发；
真实 kernel 的**最后一块**是部分 true，会触发 → 按全 true 计价会**低估 2.3 倍**，按 `1/N_block` 摊薄。

## 6. 适用条件与已知弱点

1. **布局**：连续块（`stride == 1`、仿射地址、静态 shape），起点 32 B 对齐；
2. **秩**：一维 `(e,)` 或**行跨距 = 列数**的二维 `(R,C)`。二维直接展平、`n = R·C` 代入本模型即可，
   行/列是否 2 的幂都不影响（实测比值 1.000±0.01，见 §5.1）。行跨距 > 列数的二维子块不在本域；
3. **`grid == 1`**，SIMT 侧 `superblock_factor == 1`；
4. **元素数**：形态 A 四种组合都测到 **16384**；形态 B 块宽测到 **8192**；
5. **`W ∈ {1, 2, 4, 8, 16, 32, 64}`**；SIMD 无 `W` 概念；
6. **单卡、同一批次**（跨卡实测可差 16.5 倍，且随他人负载变化）；
7. **无同步、非间接寻址**；
8. 形态划分：TTIR 无 `mask` → §3.2；有 `mask` → §3.3。

**不认领**：行跨距 > 列数的二维子块、`stride > 1`、间接寻址、`grid > 1`、`superblock_factor > 1`、超出上表元素数。

### 6.1 二维 tile 直接展平，套用一维公式

**结论**：二维 tile `(R, C)`（行跨距 = C）与一维 `(R·C,)` 等价，**直接把 `n = R·C` 代入一维公式即可**，
行或列是不是 2 的幂都不影响。

验证方式：每个几何**同时**测二维 `(R,C)` 与一维 `R·C`，同一次运行、同一窗口，直接比比值；
每个配置跑 3 次取最小值（见 §2.3 的双峰问题）。

| 模式 | 几何类别 | 组数 | 2D/1D 比值中位 | min | max |
|---|---|---:|---:|---:|---:|
| SIMD | 都 2 幂 | 32 | **1.0002** | 0.9988 | 1.0131 |
| SIMD | 其一非 2 幂 | 16 | **1.0000** | 0.9989 | 1.0096 |
| SIMD | 都非 2 幂 | 16 | **1.0000** | 0.9934 | 1.0007 |
| SIMT | 都 2 幂 | 12 | **0.9997** | 0.9857 | 1.0008 |

覆盖几何：`32×64`、`32×128`、`64×64`、`16×256`、`31×127`、`32×127`、`31×128`、`63×33`，4 dtype × 2 方向。

**SIMT 遇到非 2 幂的行或列会直接编译失败**（`arange's range must be a power of 2`），与一维一致。

用一维 SIMD 公式预测这些二维实测（`bytes = R·C·eb`）：

| 方向 | 几何类别 | 组数 | 误差中位 | 绝对误差中位 | max |
|---|---|---:|---:|---:|---:|
| load | 都 2 幂 | 16 | −1.8% | 3.8% | 11.0% |
| load | 其一非 2 幂 | 8 | −1.3% | 3.7% | 7.1% |
| load | 都非 2 幂 | 8 | −1.8% | 1.8% | 30.3% |
| store | 都 2 幂 | 16 | +1.1% | 1.4% | 13.1% |
| store | 其一非 2 幂 | 8 | +3.5% | 3.5% | 10.9% |
| store | 都非 2 幂 | 8 | −4.0% | 6.5% | 23.3% |

数据：`raw/raw_tile2d_verify.jsonl`（240 条记录 / 80 个配置）。

##### 探针的核心函数（`scripts/probe_tile2d.py`）

同一次运行里既测二维 `(R,C)` 也测一维 `R·C`，两者共用同一块 buffer、同一个窗口：

```python
@triton.jit(do_not_specialize=["iters"])
def tile2d(src, dst, iters, R: tl.constexpr, C: tl.constexpr,
           WINMASK: tl.constexpr, direction: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    off = r * C + c                     # 行跨距 = C，所以整块连续
    n: tl.constexpr = R * C             # 展平后的元素数
    if direction == 0:
        acc = tl.zeros((R, C), src.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(src + (step & WINMASK) * n + off)
        tl.store(dst + off, acc)
    else:
        v = tl.full((R, C), 1, dst.dtype.element_ty)
        for step in range(iters):
            tl.store(dst + (step & WINMASK) * n + off, v + step)

@triton.jit(do_not_specialize=["iters"])
def flat1d(src, dst, iters, N: tl.constexpr,
           WINMASK: tl.constexpr, direction: tl.constexpr):
    lane = tl.arange(0, N)              # 对照组：同一个 R*C 当成一维块
    ...
```

要点：

| 项 | 说明 |
|---|---|
| `off = r * C + c` | 行跨距取 `C`，所以 `(R,C)` 在内存里是 `R·C` 个连续元素 |
| 两次测量共用 `n = R·C` 的步进 | 二维与一维访问的地址跨度完全一致 |
| `grid = 1` + `iters` 轮转窗口 | 与所有一维探针相同的 k-marginal 口径 |
| 每配置跑 3 次取最小值 | 压掉 §2.3 那个 30% 双峰 |


---

### 6.2 已知弱点

| 项 | 数据量 | MAPE | 说明 |
|---|---:|---:|---|
| SIMD load int64 | 32 | 12.6% | 误差集中在最大块（16384 元素 = 128 KB）|
| SIMT B load `l>32_nospill` | 50 | 17.9% | 段内 `W` 跨 1~64、成本跨 50 倍。只用物理量重搜（4~6 项）最好 17.9%/16.4%（后者 LOO-CV 反而更差），**已到物理极限** |
| SIMT B store `narrow, 8<lines≤32` | 26 | 18.0% | 最弱段；该档只有 26 点（块宽必须 2 的幂，`lines` 取值稀疏）。改用 `schedR/fill/⌈n/32⌉/(lines/⌈W/4⌉)` 四个物理量后从 22.4% 降到 18.0% |
| SIMT A ③ 段内 `W≤4` 的子片 | 8 | 25.0% | 该段总共 19 点，`W≤4` 只占 8 点，点数太少 |

**族级 MAPE 都在 10% 以内**：SIMD 6.97% / 5.24%、SIMT A 7.73% / 8.47%、SIMT B 9.47% / 9.03%。
段不要求各自 <10%。

#### 试过但否定的方向

| 方向 | 结果 |
|---|---|
| 段内用 `max(串行项, 带宽项)` 瓶颈形式 | load 35%、store 38~47%，远差于线性求和 |
| 不分段单条公式 | load 15~18%、store 22~24% |
| 3~5 段替代 6 段 | 11.6~14.3%，超过 10% |
| 用 `LDG·eb`、`te·eb²` 等 dtype 缩放特征救弱段 | 17.92% vs 18.04%，无改善 |

---

## 7. 验证

口径：同一份 costmodel report 中取与编译模式对应的 route（simd 取 all_simd，simt_only 取 all_simt_only）的 total_system_cycles，按 1.65 GHz 换算为 µs；实测取 NPU profiler 的 contig_kernel 持续时间（2048 次迭代，3 次取最小）。

kernel 定义：copy = 每轮 1 load + 1 store；add3 = 每轮 2 load + 1 store + 1 add。全部无 mask、stride == 1。

| 分组 | n | 中位误差 | 最大误差 | <10% |
|---|---:|---:|---:|---:|
| copy / simd | 18 | 8.9% | 45.3% | 9/18 |
| copy / simt_only | 16 | 62.5% | 821.0% | 5/16 |
| add3 / simd | 18 | 34.6% | 45.4% | 0/18 |
| add3 / simt_only | 16 | 38.4% | 391.9% | 0/16 |
| 全部 | 68 | 33.4% | 821.0% | 14/68 |

### 7.1 68 个自建整核：改前 / 改后逐点对比

口径同 6.1：预测取对应 route 的 total_system_cycles / 1.65 GHz，实测取 profiler 的 contig_kernel 持续时间。改前 = 基线二进制，改后 = 本 PR 二进制，同一批 kernel 各跑一次。

| 分组 | n | 改前中位 | 改后中位 | 改前 <10% | 改后 <10% |
|---|---:|---:|---:|---:|---:|
| copy / simd | 17 | 39.3% | 10.8% | 0/17 | 8/17 |
| copy / simt_only | 16 | 62.5% | 62.5% | 2/16 | 2/16 |
| add3 / simd | 18 | 33.5% | 33.5% | 1/18 | 1/18 |
| add3 / simt_only | 16 | 48.6% | 48.6% | 0/16 | 0/16 |
| 全部 | 67 | 40.9% | 37.5% | 3/67 | 11/67 |

| # | kind | mode | dtype | 几何 | W | 每轮访存 | 实测 us | 改前预测 | 改后预测 | 改前误差 | 改后误差 |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | add3 | simd | fp16 | 1x127 | 1 | 2L+1S | 201.4 | 265.2 | 265.2 | +31.7% | +31.7% |
| 2 | add3 | simd | fp16 | 1x1000 | 1 | 2L+1S | 214.2 | 291.1 | 291.1 | +35.9% | +35.9% |
| 3 | add3 | simd | fp16 | 1x1024 | 1 | 2L+1S | 200.7 | 291.8 | 291.8 | +45.3% | +45.3% |
| 4 | add3 | simd | fp16 | 31x127 | 1 | 2L+1S | 469.3 | 378.1 | 378.1 | -19.4% | -19.4% |
| 5 | add3 | simd | fp16 | 32x127 | 1 | 2L+1S | 463.3 | 381.9 | 381.9 | -17.6% | -17.6% |
| 6 | add3 | simd | fp16 | 1x4096 | 1 | 2L+1S | 576.1 | 382.8 | 382.8 | -33.6% | -33.6% |
| 7 | add3 | simd | fp16 | 32x128 | 1 | 2L+1S | 265.9 | 382.8 | 382.8 | +44.0% | +44.0% |
| 8 | add3 | simd | fp16 | 64x64 | 1 | 2L+1S | 385.8 | 382.8 | 382.8 | -0.8% | -0.8% |
| 9 | add3 | simd | fp16 | 1x16384 | 1 | 2L+1S | 1037.4 | 746.9 | 746.9 | -28.0% | -28.0% |
| 10 | add3 | simd | fp32 | 1x127 | 1 | 2L+1S | 211.3 | 116.2 | 116.2 | -45.0% | -45.0% |
| 11 | add3 | simd | fp32 | 1x1000 | 1 | 2L+1S | 219.9 | 136.5 | 136.5 | -37.9% | -37.9% |
| 12 | add3 | simd | fp32 | 1x1024 | 1 | 2L+1S | 208.1 | 138.4 | 138.4 | -33.5% | -33.5% |
| 13 | add3 | simd | fp32 | 31x127 | 1 | 2L+1S | 817.0 | 485.7 | 485.7 | -40.5% | -40.5% |
| 14 | add3 | simd | fp32 | 32x127 | 1 | 2L+1S | 758.9 | 500.9 | 500.9 | -34.0% | -34.0% |
| 15 | add3 | simd | fp32 | 1x4096 | 1 | 2L+1S | 751.4 | 504.7 | 504.7 | -32.8% | -32.8% |
| 16 | add3 | simd | fp32 | 32x128 | 1 | 2L+1S | 755.7 | 504.7 | 504.7 | -33.2% | -33.2% |
| 17 | add3 | simd | fp32 | 64x64 | 1 | 2L+1S | 741.9 | 504.7 | 504.7 | -32.0% | -32.0% |
| 18 | add3 | simd | fp32 | 1x16384 | 1 | 2L+1S | 3150.4 | 1970.2 | 1970.2 | -37.5% | -37.5% |
| 19 | add3 | simt_only | fp16 | 1x1024 | 1 | 2L+1S | 1083.2 | 5336.6 | 5336.6 | +392.7% | +392.7% |
| 20 | add3 | simt_only | fp16 | 1x1024 | 4 | 2L+1S | 557.7 | 1386.7 | 1386.7 | +148.7% | +148.7% |
| 21 | add3 | simt_only | fp16 | 1x4096 | 1 | 2L+1S | 14859.2 | 21345.9 | 21345.9 | +43.7% | +43.7% |
| 22 | add3 | simt_only | fp16 | 1x4096 | 4 | 2L+1S | 3381.4 | 5546.5 | 5546.5 | +64.0% | +64.0% |
| 23 | add3 | simt_only | fp16 | 32x128 | 1 | 2L+1S | 12980.9 | 21345.9 | 21345.9 | +64.4% | +64.4% |
| 24 | add3 | simt_only | fp16 | 32x128 | 4 | 2L+1S | 3338.5 | 5546.5 | 5546.5 | +66.1% | +66.1% |
| 25 | add3 | simt_only | fp16 | 64x64 | 1 | 2L+1S | 15397.0 | 21345.9 | 21345.9 | +38.6% | +38.6% |
| 26 | add3 | simt_only | fp16 | 64x64 | 4 | 2L+1S | 3346.5 | 5546.5 | 5546.5 | +65.7% | +65.7% |
| 27 | add3 | simt_only | fp32 | 1x1024 | 1 | 2L+1S | 3167.0 | 5336.6 | 5336.6 | +68.5% | +68.5% |
| 28 | add3 | simt_only | fp32 | 1x1024 | 4 | 2L+1S | 903.3 | 1386.7 | 1386.7 | +53.5% | +53.5% |
| 29 | add3 | simt_only | fp32 | 1x4096 | 1 | 2L+1S | 27616.9 | 21345.9 | 21345.9 | -22.7% | -22.7% |
| 30 | add3 | simt_only | fp32 | 1x4096 | 4 | 2L+1S | 7840.8 | 5546.5 | 5546.5 | -29.3% | -29.3% |
| 31 | add3 | simt_only | fp32 | 32x128 | 1 | 2L+1S | 28325.3 | 21345.9 | 21345.9 | -24.6% | -24.6% |
| 32 | add3 | simt_only | fp32 | 32x128 | 4 | 2L+1S | 7840.0 | 5546.5 | 5546.5 | -29.3% | -29.3% |
| 33 | add3 | simt_only | fp32 | 64x64 | 1 | 2L+1S | 28115.6 | 21345.9 | 21345.9 | -24.1% | -24.1% |
| 34 | add3 | simt_only | fp32 | 64x64 | 4 | 2L+1S | 7836.6 | 5546.5 | 5546.5 | -29.2% | -29.2% |
| 35 | copy | simd | fp16 | 1x127 | 1 | 1L+1S | 216.3 | 132.6 | 192.9 | -38.7% | -10.8% |
| 36 | copy | simd | fp16 | 1x1000 | 1 | 1L+1S | 239.9 | 145.5 | 212.1 | -39.3% | -11.6% |
| 37 | copy | simd | fp16 | 1x1024 | 1 | 1L+1S | 229.0 | 145.9 | 212.1 | -36.3% | -7.4% |
| 38 | copy | simd | fp16 | 31x127 | 1 | 1L+1S | 287.5 | 189.1 | 275.8 | -34.2% | -4.1% |
| 39 | copy | simd | fp16 | 32x127 | 1 | 1L+1S | 273.0 | 190.9 | 275.8 | -30.1% | +1.0% |
| 40 | copy | simd | fp16 | 1x4096 | 1 | 1L+1S | 274.3 | 191.4 | 275.8 | -30.2% | +0.5% |
| 41 | copy | simd | fp16 | 32x128 | 1 | 1L+1S | 273.3 | 191.4 | 275.8 | -30.0% | +0.9% |
| 42 | copy | simd | fp16 | 64x64 | 1 | 1L+1S | 280.5 | 191.4 | 275.8 | -31.8% | -1.7% |
| 43 | copy | simd | fp16 | 1x16384 | 1 | 1L+1S | 989.0 | 373.5 | 730.0 | -62.2% | -26.2% |
| 44 | copy | simd | fp32 | 1x127 | 1 | 1L+1S | 232.3 | 116.2 | 195.6 | -50.0% | -15.8% |
| 45 | copy | simd | fp32 | 1x1000 | 1 | 1L+1S | 247.7 | 136.5 | 234.5 | -44.9% | -5.3% |
| 46 | copy | simd | fp32 | 31x127 | 1 | 1L+1S | 593.6 | 247.4 | 364.5 | -58.3% | -38.6% |
| 47 | copy | simd | fp32 | 32x127 | 1 | 1L+1S | 626.5 | 251.2 | 364.5 | -59.9% | -41.8% |
| 48 | copy | simd | fp32 | 1x4096 | 1 | 1L+1S | 656.2 | 260.5 | 364.8 | -60.3% | -44.4% |
| 49 | copy | simd | fp32 | 32x128 | 1 | 1L+1S | 656.7 | 330.5 | 364.5 | -49.7% | -44.5% |
| 50 | copy | simd | fp32 | 64x64 | 1 | 1L+1S | 554.8 | 252.1 | 364.5 | -54.6% | -34.3% |
| 51 | copy | simd | fp32 | 1x16384 | 1 | 1L+1S | 1364.7 | 993.2 | 1459.8 | -27.2% | +7.0% |
| 52 | copy | simt_only | fp16 | 1x1024 | 1 | 1L+1S | 496.6 | 2868.6 | 2868.6 | +477.6% | +477.6% |
| 53 | copy | simt_only | fp16 | 1x1024 | 4 | 1L+1S | 424.9 | 754.4 | 754.4 | +77.6% | +77.6% |
| 54 | copy | simt_only | fp16 | 1x4096 | 1 | 1L+1S | 1245.8 | 11473.6 | 11473.6 | +821.0% | +821.0% |
| 55 | copy | simt_only | fp16 | 1x4096 | 4 | 1L+1S | 898.2 | 3017.2 | 3017.2 | +235.9% | +235.9% |
| 56 | copy | simt_only | fp16 | 32x128 | 1 | 1L+1S | 1246.4 | 11473.6 | 11473.6 | +820.5% | +820.5% |
| 57 | copy | simt_only | fp16 | 32x128 | 4 | 1L+1S | 898.4 | 3017.2 | 3017.2 | +235.9% | +235.9% |
| 58 | copy | simt_only | fp16 | 64x64 | 1 | 1L+1S | 1246.2 | 11473.6 | 11473.6 | +820.7% | +820.7% |
| 59 | copy | simt_only | fp16 | 64x64 | 4 | 1L+1S | 962.9 | 3017.2 | 3017.2 | +213.3% | +213.3% |
| 60 | copy | simt_only | fp32 | 1x1024 | 1 | 1L+1S | 2823.2 | 2868.6 | 2868.6 | +1.6% | +1.6% |
| 61 | copy | simt_only | fp32 | 1x1024 | 4 | 1L+1S | 772.0 | 754.4 | 754.4 | -2.3% | -2.3% |
| 62 | copy | simt_only | fp32 | 1x4096 | 1 | 1L+1S | 19414.9 | 11473.6 | 11473.6 | -40.9% | -40.9% |
| 63 | copy | simt_only | fp32 | 1x4096 | 4 | 1L+1S | 5638.7 | 3017.2 | 3017.2 | -46.5% | -46.5% |
| 64 | copy | simt_only | fp32 | 32x128 | 1 | 1L+1S | 19506.4 | 11473.6 | 11473.6 | -41.2% | -41.2% |
| 65 | copy | simt_only | fp32 | 32x128 | 4 | 1L+1S | 5620.7 | 3017.2 | 3017.2 | -46.3% | -46.3% |
| 66 | copy | simt_only | fp32 | 64x64 | 1 | 1L+1S | 21862.1 | 11473.6 | 11473.6 | -47.5% | -47.5% |
| 67 | copy | simt_only | fp32 | 64x64 | 4 | 1L+1S | 5622.6 | 3017.2 | 3017.2 | -46.3% | -46.3% |

### 7.2 100-case 实测 vs 改前/改后预测（含 route）

口径：

- **实测**：`b695c198a` 基线**编译安装后**跑出的 `benchmark.profiler.average_us`，且要求 `measurement_metrics.costmodel_target == npu_profiler_kernel_duration_average_us`；device-event 退化值不计；
- **改前预测**：同一基线二进制的 `selected_route_elapsed_prediction.predicted_microseconds`；
- **改后预测**：最终版本（含本次连续访存标定）同一字段；
- **route**：`effective_decision_kind` + `effective_runtime_factor`。

覆盖与两处必须的环境条件：

| 项 | 结果 |
|---|---|
| 100 例中有实测 | **95** |
| 三项齐全（实测+两版预测）| **95**（5 例编译不过）|
| 基线必须正式 `pip install` | 解包 wheel + `PYTHONPATH` 只能测到 30/100，装进环境后 91/100 |
| `gather_dot*`、`fbgemm*` 需 CANN 的 `acl` | 补 `PYTHONPATH=/usr/local/Ascend/cann-9.1.0/python/site-packages` 后 4 例可测（改前/改后预测均补齐）|
| 剩余 5 例编译不过 | `histogram_*` 4 例 `hivm.hir.custom` 无法 legalize；`pytest_atomic_add_f32_32x32_c2` host module lowering 失败 |

| 指标 | 改前 | 改后 |
|---|---:|---:|
| 中位绝对误差 | 5.10% | **7.26%** |
| 平均绝对误差 | 17.23% | **19.09%** |
| <10% 比例 | 72/95 | **58/95** |

### 7.3 原本达标、改动后超 10% 的 16 例及原因

这 16 例在基线版本下误差 <10%，换成本次改动后 ≥10%。共同点是：**改动删掉了 `te > 4`、`<128 B`、宽 warp store 三个认领守卫**，于是中等尺寸的连续访存从“解析式回落”变成“由标定分段定价”，而分段值在这个区间明显偏贵。

| 用例 | route | 实测us | 改前 | 改后 | 改前误差 | 改后误差 |
|---|---|---:|---:|---:|---:|---:|
| pytest_indirect_load_add_b8 | SIMT+F1 | 1.237 | 1.234 | 1.367 | -0.3% | +10.5% |
| pytest_dot_bf16_16x32x16 | SIMD+F1 | 1.200 | 1.246 | 1.346 | +3.8% | +12.2% |
| pytest_max_dim1_f32_m64_n8 | SIMD+F1 | 1.302 | 1.397 | 1.461 | +7.3% | +12.2% |
| pytest_max_dim0_f32_64x8 | SIMD+F1 | 1.266 | 1.371 | 1.436 | +8.3% | +13.5% |
| pytest_view_f32_2x256x16 | SIMD+F1 | 1.259 | 1.304 | 1.436 | +3.6% | +14.1% |
| pytest_index_put_n64_d32_m8 | SIMT+F1 | 1.165 | 1.264 | 1.332 | +8.5% | +14.4% |
| cumsum_i32_w128_reverse | SIMT+F2 | 16.395 | 16.756 | 18.921 | +2.2% | +15.4% |
| pytest_indirect_index_maxmin_b8 | SIMT+F1 | 1.214 | 1.213 | 1.405 | -0.1% | +15.7% |
| pytest_indirect_index_and_b8 | SIMT+F1 | 1.200 | 1.213 | 1.405 | +1.1% | +17.1% |
| atomic_discrete_i32_p1_b256 | SIMT+F1 | 1.542 | 1.527 | 1.809 | -1.0% | +17.3% |
| pytest_exp_f32_2x4096x8 | SIMD+F1 | 4.331 | 4.596 | 5.086 | +6.1% | +17.4% |
| pytest_mul_reduce_f16_16x1x1x1x1 | SIMD+F1 | 0.991 | 1.067 | 1.166 | +7.6% | +17.6% |
| pytest_strided_scatter_f32_s16 | SIMT+F1 | 4.281 | 3.979 | 5.105 | -7.0% | +19.3% |
| pytest_discrete_mask_4way_f16_8x8 | SIMT+F1 | 1.001 | 1.029 | 1.243 | +2.8% | +24.2% |
| pytest_indirect_index_div_b8 | SIMT+F1 | 1.247 | 1.245 | 1.624 | -0.2% | +30.3% |
| cumsum_i32_w256 | SIMT+F2 | 14.092 | 15.144 | 19.787 | +7.5% | +40.4% |

原因分两类，都能在 stage 明细上看到（`resource_system_cycles`）：

**① 中等尺寸连续访存被分段高估（主因）**

以 `cumsum_i32_w256` 为例，`stage_5_continuous_tile_memory` 的 SIMT 实现：

| | load_per_iteration | stage total | 整核预测 |
|---|---:|---:|---:|
| 改前（解析式回落）| 33.5 cycles | 535.7 cycles | 15.144 us |
| 改后（标定分段）| 216.1 cycles | 4567.2 cycles | 19.787 us |
| 倍数 | **6.5x** | **8.5x** | +30% |

实测 14.092 us 说明回落值（33.5）才是对的；分段给 216 cycles 相当于 131 ns 做完 1 KiB，比机器实际能力慢约一倍。
`cumsum_i32_w128_reverse` 同理，且 route 是 SIMT+F2，误差被 factor 放大一倍。

**② 小算子被固定增量顶过 10% 线**

`pytest_indirect_load_add_b8`（实测 1.237 us）只是两个 stage 各多了几十 cycles：

| stage | 改前 | 改后 | 增量 |
|---|---:|---:|---:|
| stage_0_continuous_tile_memory | 211.5 | 305.2 | +93.7 cycles (+0.057 us) |
| stage_4_continuous_tile_store | 91.4 | 157.1 | +65.6 cycles (+0.040 us) |

总共 +0.133 us。对 1 us 级算子就是 +10.5%，对 10 us 级算子只有 +1%。所以这 16 例里 11 例是 <1.5 us 的小算子，误差放大主要来自分母太小。

**④ 与基线标定范围的结构性差异（这 14 例的共同根因）**

基线的 profile 里**没有** `memory.contiguous_ldst`，它的连续访存标定在 `stage_resources.continuous_memory`，而且只覆盖一个点：

```
操作数 = 1，字节 = 1024，dtype = fp32，factor = 1/2/4
```

其 description 明确写着 “Curves are consumed only inside the measured byte interval”。落在区间外的形状走**回落公式**（`uncovered_bytes / bytes_per_cycle`）。

本 PR 新增了 `memory.contiguous_ldst`，覆盖从几十字节到 64 KiB 的连续访存并优先于回落。所以这 16 例里 14 例的连续访存 stage 变贵，不是“把标定数据改坏了”，而是**接管了基线压根没建模、交给回落兜底的区间**。

以 `cumsum_i32_w256` 的 1 KiB SIMT load 为例：

| 来源 | 每迭代代价 | 折合 |
|---|---:|---:|
| 基线回落（带宽口径）| 33.5 cycles | 20 ns |
| 本次标定分段 | 216.1 cycles | 131 ns |
| 实测反推 | ≈33 cycles | ≈20 ns |

两者的差异不在“测得准不准”，而在**口径**：我们的分段值来自**独立 kernel** 的 min-of-N 实测，含完整的启动与等待；而算子里这条 load 处在循环中，多次访存彼此重叠，有效单价接近带宽极限。基线回落用的正是带宽口径，所以在这种循环里反而对。

这解释了为什么“单 op 误差 <10%”与“100-case 变差”会同时成立：**标定在独立 kernel 上是对的，放进循环体里就偏贵**。

短结论：

- 没有修改任何已测数据点，是**扩充**（新增 266 个小访问点，另 16 个大访问点因越界已撤）；
- 重新拟合的是 SIMD 三次式，并新增 18 个 SIMT 小访问段 + 4 张 te≥8 的表；
- 对 100-case 有影响，影响机制是**把回落区间换成了独立 kernel 口径的实测值**；循环体里这些值偏贵 2~6 倍。

**③ 顺带发现的缺陷**

表插值在定义域边缘会给出**负值**：`elem=256 eb=4 warps=1 te=8.0` 时 `store.eb4` 返回 `-120.2 cycles`。当前该值使预测偏小而非偏大，但属于边界处理错误，应加下界钳制。

#### 逐例明细

| # | 用例 | route | 实测us | 改前预测 | 改后预测 | 改前误差 | 改后误差 | 改后是否PASS |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | muls_add_n2097152 | SIMD+F1 | 5.875 | 6.733 | 6.735 | +14.6% | +14.6% | FAIL |
| 2 | muls_add_n1048576 | SIMD+F1 | 3.781 | 4.734 | 4.734 | +25.2% | +25.2% | FAIL |
| 3 | memory_stream_l4_n2097152 | SIMD+F1 | 9.184 | 10.369 | 10.371 | +12.9% | +12.9% | FAIL |
| 4 | memory_stream_l2_n1048576 | SIMD+F1 | 3.890 | 4.730 | 4.729 | +21.6% | +21.6% | FAIL |
| 5 | cumsum_i32_w256 | SIMT+F2 | 14.092 | 15.149 | 19.628 | +7.5% | +39.3% | FAIL |
| 6 | cumsum_i32_w128_reverse | SIMD+F1 | 16.395 | 16.756 | 18.921 | +2.2% | +15.4% | FAIL |
| 7 | gather_dot_f16_k16 | SIMT+F1 | 7.878 | 8.059 | 8.038 | +2.3% | +2.0% | PASS |
| 8 | gather_dot_f16_k16_p112 | SIMT+F1 | 5.102 | 5.230 | 5.221 | +2.5% | +2.3% | PASS |
| 9 | masked_copy_r16_l64_v64_s64_k8 | SIMD+F1 | 2.486 | 3.115 | 2.945 | +25.3% | +18.5% | FAIL |
| 10 | masked_copy_r64_l128_v128_s256_k12 | SIMD+F1 | 7.028 | 7.668 | 7.669 | +9.1% | +9.1% | PASS |
| 11 | grid_stride_gather_p224 | SIMT+F1 | 3.339 | 4.625 | 4.760 | +38.5% | +42.6% | FAIL |
| 12 | grid_stride_dot_p224 | SIMD+F1 | 5.253 | 4.985 | 4.948 | -5.1% | -5.8% | PASS |
| 13 | grid_stride_dot_p56 | SIMT+F1 | 3.124 | 3.205 | 3.400 | +2.6% | +8.8% | PASS |
| 14 | grid_stride_dot_p112 | SIMT+F1 | 3.955 | 3.892 | 4.171 | -1.6% | +5.5% | PASS |
| 15 | grid_stride_dot_p336 | SIMD+F1 | 5.679 | 5.418 | 5.354 | -4.6% | -5.7% | PASS |
| 16 | grid_stride_dot_p448 | SIMD+F1 | 6.114 | 5.851 | 5.761 | -4.3% | -5.8% | PASS |
| 17 | pytest_indirect_load_add_b8 | SIMT+F1 | 1.237 | 1.233 | 1.367 | -0.3% | +10.5% | FAIL |
| 18 | pytest_indirect_index_div_b8 | SIMT+F1 | 1.247 | 1.245 | 1.624 | -0.2% | +30.2% | FAIL |
| 19 | pytest_indirect_index_and_b8 | SIMT+F1 | 1.200 | 1.213 | 1.405 | +1.1% | +17.1% | FAIL |
| 20 | pytest_indirect_index_maxmin_b8 | SIMT+F1 | 1.214 | 1.213 | 1.405 | -0.1% | +15.7% | FAIL |
| 21 | pytest_indirect_load_2d_min_m4_n16_k8 | SIMT+F1 | 1.213 | 1.255 | 1.255 | +3.5% | +3.5% | PASS |
| 22 | pytest_indirect_load_2d_grid_m4_n16_k8 | SIMT+F1 | 1.741 | 1.574 | 1.573 | -9.6% | -9.6% | PASS |
| 23 | pytest_index_select_n64_d32_m8 | SIMD+F1 | 1.163 | 1.084 | 1.166 | -6.8% | +0.3% | PASS |
| 24 | pytest_index_put_n64_d32_m8 | MIXED+F1 | 1.165 | 1.264 | 1.332 | +8.5% | +14.3% | FAIL |
| 25 | pytest_indirect_store_b8 | SIMT+F1 | 1.070 | 1.162 | 1.162 | +8.6% | +8.6% | PASS |
| 26 | pytest_gather_dot_f16_k16_p56 | SIMT+F1 | 3.725 | 3.818 | 3.813 | +2.5% | +2.4% | PASS |
| 27 | pytest_vector_add_n98432 | SIMD+F1 | 1.661 | 2.864 | 3.065 | +72.4% | +84.5% | FAIL |
| 28 | pytest_permute_x64_y32 | SIMD+F1 | 2.337 | 2.332 | 2.422 | -0.2% | +3.6% | PASS |
| 29 | pytest_sigmoid_f32_2x4096x8 | SIMD+F1 | 4.988 | 4.634 | 4.633 | -7.1% | -7.1% | PASS |
| 30 | pytest_rms_norm_f16_m128_n128 | SIMD+F1 | 5.401 | 6.179 | 6.288 | +14.4% | +16.4% | FAIL |
| 31 | pytest_layer_norm_f16_m128_n128 | SIMT+F1 | 3.100 | 3.460 | 5.449 | +11.6% | +75.8% | FAIL |
| 32 | pytest_insert_slice_n1024_s32 | SIMD+F1 | 1.136 | 1.078 | 1.195 | -5.1% | +5.2% | PASS |
| 33 | pytest_atomic_add_f32_32x32_c2 | SIMD+F1 | - | - | 1.932 | - | - | - |
| 34 | pytest_exp_f32_2x4096x8 | SIMD+F1 | 4.331 | 4.595 | 5.086 | +6.1% | +17.4% | FAIL |
| 35 | pytest_log_f32_n781 | SIMD+F1 | 1.070 | 1.057 | 1.141 | -1.2% | +6.6% | PASS |
| 36 | pytest_relu_f32_2x4096x8 | SIMD+F1 | 4.483 | 4.626 | 4.628 | +3.2% | +3.2% | PASS |
| 37 | pytest_where_f32_2x1024x8 | SIMD+F1 | 1.983 | 1.959 | 1.959 | -1.2% | -1.2% | PASS |
| 38 | pytest_broadcast_bf16_2x4x8 | SIMD+F1 | 0.974 | 0.956 | 0.973 | -1.8% | -0.1% | PASS |
| 39 | pytest_max_dim1_f32_m64_n8 | SIMD+F1 | 1.302 | 1.397 | 1.461 | +7.3% | +12.2% | FAIL |
| 40 | pytest_mean_dim1_f32_m57_n3 | SIMD+F1 | 1.147 | 1.166 | 1.231 | +1.7% | +7.3% | PASS |
| 41 | pytest_mul_reduce_f16_16x1x1x1x1 | SIMD+F1 | 0.991 | 1.066 | 1.166 | +7.6% | +17.7% | FAIL |
| 42 | pytest_cumprod_f32_n1024 | SIMT+F1 | 1.840 | 1.820 | 1.756 | -1.1% | -4.6% | PASS |
| 43 | pytest_cummax_f32_16x32_dim1 | SIMT+F1 | 1.277 | 1.214 | 1.324 | -4.9% | +3.7% | PASS |
| 44 | pytest_atomic_max_i32_32x32_c2 | SIMD+F1 | 1.364 | 1.446 | 1.463 | +6.0% | +7.3% | PASS |
| 45 | atomic_continuous_i32_p1_b256 | SIMD+F1 | 1.087 | 1.009 | 1.026 | -7.2% | -5.6% | PASS |
| 46 | atomic_continuous_i32_p56_b256 | SIMD+F1 | 1.524 | 2.306 | 2.323 | +51.3% | +52.4% | FAIL |
| 47 | atomic_discrete_i32_p1_b256 | SIMT+F1 | 1.542 | 1.527 | 1.809 | -1.0% | +17.3% | FAIL |
| 48 | atomic_discrete_i32_p56_b256 | SIMT+F1 | 2.177 | 2.824 | 3.105 | +29.7% | +42.6% | FAIL |
| 49 | rowred_r16_c128_k16 | SIMT+F1 | 27.972 | 15.888 | 17.455 | -43.2% | -37.6% | FAIL |
| 50 | rowred_r64_c256_k22 | SIMD+F1 | 100.547 | 92.805 | 92.769 | -7.7% | -7.7% | PASS |
| 51 | softmax_power2 | SIMD+F1 | 50.284 | 65.218 | 65.200 | +29.7% | +29.7% | FAIL |
| 52 | softmax_original | SIMD+F1 | 33.998 | 31.244 | 34.349 | -8.1% | +1.0% | PASS |
| 53 | fze_default | SIMT+F2 | 2895.190 | 2840.181 | 2846.596 | -1.9% | -1.7% | PASS |
| 54 | solve_tril_b1_t1024_h32_bt64 | MIXED+F4 | 181.820 | 178.911 | 178.894 | -1.6% | -1.6% | PASS |
| 55 | solve_tril_b1_t1024_h64_bt64 | MIXED+F4 | 322.356 | 324.290 | 324.141 | +0.6% | +0.6% | PASS |
| 56 | solve_tril_b8_t1024_h32_bt64 | MIXED+F4 | 1502.790 | 1424.645 | 1424.071 | -5.2% | -5.2% | PASS |
| 57 | solve_tril_b16_t1024_h64_bt64 | MIXED+F4 | 5941.681 | 5882.264 | 5885.014 | -1.0% | -1.0% | PASS |
| 58 | solve_tril_b1_t8192_h64_bt64 | MIXED+F4 | 2980.461 | 2899.989 | 2899.816 | -2.7% | -2.7% | PASS |
| 59 | solve_tril_b8_t8192_h32_bt64 | MIXED+F4 | 11852.023 | 11840.171 | 11840.531 | -0.1% | -0.1% | PASS |
| 60 | solve_tril_b16_t8192_h64_bt64 | MIXED+F4 | 47722.393 | 47531.503 | 47550.853 | -0.4% | -0.4% | PASS |
| 61 | solve_tril_b4_t131072_h32_bt64 | MIXED+F4 | 94993.324 | 95183.311 | 95172.210 | +0.2% | +0.2% | PASS |
| 62 | masked_reduction_r32_l150_v256_s150_k12 | SIMD+F1 | 46.285 | 43.924 | 43.913 | -5.1% | -5.1% | PASS |
| 63 | masked_reduction_r64_l150_v256_s150_k22 | SIMD+F1 | 99.298 | 92.943 | 92.947 | -6.4% | -6.4% | PASS |
| 64 | masked_softmax_r32_l150_v256_s150_k12 | SIMD+F1 | 16.113 | 16.999 | 16.995 | +5.5% | +5.5% | PASS |
| 65 | masked_softmax_r80_l150_v150_s150_k18 | SIMD+F1 | 33.685 | 31.260 | 34.348 | -7.2% | +2.0% | PASS |
| 66 | pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512 | SIMT+F2 | 7.759 | 10.482 | 10.482 | +35.1% | +35.1% | FAIL |
| 67 | pytest_sum_low_f32_64x8192 | SIMD+F1 | 46.633 | 1.539 | 1.599 | -96.7% | -96.6% | FAIL |
| 68 | pytest_sum_high_f32_64x8192 | SIMD+F1 | 4.055 | 25.222 | 25.133 | +522.0% | +519.8% | FAIL |
| 69 | histogram_i32_r8192_b256_uniform | MIXED+F1 | - | - | 85.366 | - | - | - |
| 70 | histogram_i32_r8192_b256_single | MIXED+F1 | - | - | 85.366 | - | - | - |
| 71 | histogram_i32_r8192_b256_radix | MIXED+F1 | - | - | 85.366 | - | - | - |
| 72 | pytest_discrete_mask_4way_f16_8x8 | SIMT+F1 | 1.001 | 1.029 | 1.243 | +2.8% | +24.2% | FAIL |
| 73 | pytest_reduce_sum_epilogue_f32_8x8x512 | SIMD+F1 | 2.758 | 2.380 | 2.380 | -13.7% | -13.7% | FAIL |
| 74 | pytest_max_dim0_f32_64x8 | SIMD+F1 | 1.266 | 1.371 | 1.436 | +8.3% | +13.4% | FAIL |
| 75 | pytest_argmax_f32_37x125_dim1 | SIMT+F1 | 22.168 | 22.478 | 21.399 | +1.4% | -3.5% | PASS |
| 76 | pytest_gather_after_reduce_f32_16x1024 | SIMD+F1 | 4.646 | 2.328 | 2.427 | -49.9% | -47.8% | FAIL |
| 77 | pytest_view_f32_2x256x16 | SIMD+F1 | 1.259 | 1.304 | 1.436 | +3.6% | +14.1% | FAIL |
| 78 | pytest_permute_reshape_fused_f32 | SIMD+F1 | 6.302 | 6.428 | 6.431 | +2.0% | +2.0% | PASS |
| 79 | pytest_nearest_resize_i32_1x4x64x64 | SIMT+F1 | 1.868 | 2.346 | 2.989 | +25.6% | +60.0% | FAIL |
| 80 | pytest_sort_f32_256x16 | SIMD+F1 | 53.261 | 51.610 | 51.727 | -3.1% | -2.9% | PASS |
| 81 | pytest_topk_f32_16x64_k4 | SIMT+F1 | 1.596 | 1.588 | 1.588 | -0.5% | -0.5% | PASS |
| 82 | pytest_multi_input_scan_f32_2x5x10 | SIMT+F1 | 1.711 | 1.571 | 1.571 | -8.2% | -8.2% | PASS |
| 83 | pytest_cummin_f32_3x5x7_dim1 | SIMD+F1 | 1.157 | 1.160 | 1.160 | +0.3% | +0.3% | PASS |
| 84 | pytest_histogram_i32_2048_bins2 | MIXED+F1 | - | - | 3.113 | - | - | - |
| 85 | pytest_scalar_reduce_store_f32_n16 | SIMD+F1 | 1.490 | 1.559 | 1.576 | +4.6% | +5.8% | PASS |
| 86 | pytest_conditional_load_f32_n32 | SIMT+F1 | 1.110 | 1.118 | 1.117 | +0.7% | +0.6% | PASS |
| 87 | pytest_dot_bf16_16x32x16 | SIMD+F1 | 1.200 | 1.246 | 1.346 | +3.8% | +12.2% | FAIL |
| 88 | pytest_fused_softmax_f32_128x257 | SIMD+F1 | 2.858 | 2.335 | 2.380 | -18.3% | -16.7% | FAIL |
| 89 | pytest_add_f32_2x4096x8 | SIMD+F1 | 4.409 | 4.629 | 4.628 | +5.0% | +5.0% | PASS |
| 90 | pytest_sqrt_f32_n781 | SIMD+F1 | 1.025 | 1.023 | 1.107 | -0.2% | +8.0% | PASS |
| 91 | pytest_strided_scatter_f32_s16 | SIMT+F1 | 4.281 | 3.981 | 5.105 | -7.0% | +19.2% | FAIL |
| 92 | pytest_block_ptr_strided_f32_4x8_s32x4 | SIMT+F1 | 1.094 | 1.091 | 1.091 | -0.3% | -0.3% | PASS |
| 93 | pytest_cos_f32_n781 | SIMD+F1 | 1.526 | 1.561 | 1.645 | +2.3% | +7.8% | PASS |
| 94 | pytest_mul_f32_2x4096x8 | SIMD+F1 | 4.442 | 4.629 | 4.628 | +4.2% | +4.2% | PASS |
| 95 | pytest_associative_scan_max_f32_128x4x16_dim2 | SIMT+F1 | 5.069 | 5.414 | 5.415 | +6.8% | +6.8% | PASS |
| 96 | pytest_div_f32_2x4096x8 | SIMD+F1 | 4.408 | 4.628 | 4.628 | +5.0% | +5.0% | PASS |
| 97 | pytest_f2i32_f32_n3 | SIMD+F1 | 0.990 | 0.958 | 1.041 | -3.2% | +5.2% | PASS |
| 98 | pytest_chunk_cumsum_f32_b4_t2048_h8_bt128 | SIMT+F2 | 3.288 | 3.870 | 3.869 | +17.7% | +17.7% | FAIL |
| 99 | pytest_maximum_f32_2x4096x8 | SIMD+F1 | 5.624 | 0.208 | 5.673 | -96.3% | +0.9% | PASS |
| 100 | pytest_strided_compaction_f32_s16 | SIMD+F1 | 3.203 | 0.064 | 2.916 | -98.0% | -9.0% | PASS |

### 7.4 100-case：以「剔除坏提交」为基线的 A/B

本节与 7.3 使用同一批用例，但**实测列改用官方基准表**（`simdsimt_costmodel_calibration_20260914.md` 的 `profiler 实测` 列，100 例齐全），不再用本地重测值；预测仍来自下面两个版本：

| 列 | 来自哪个版本 |
|---|---|
| 改前预测 | `d4667dab9`（基线 `b695c198a` + 回退 `6cc785f94f`）|
| 改后预测 | `39fb5ae63`（同上 + 本次连续访存标定）|
| route | 取**改前**版本的判定；若改后不同，写成 `改前 → 改后` 并标 ⚠ |

之所以换成这个基线：`6cc785f94f` 把「循环携带重叠」的建模方式改掉了，`pytest_gather_after_reduce_f32_16x1024` 在它之下预测 2.329 µs、实测 4.684 µs，而回退后预测 4.745 µs。也就是说 7.3 里的「改前」本身带着这个回归，不能当作健康基线。

| 指标 | 改前（`d4667dab9`）| 改后（`39fb5ae63`）|
|---|---:|---:|
| 中位绝对误差 | **3.60%** | 5.47% |
| 平均绝对误差 | 4.43% | 7.57% |
| <10% 比例 | **94/100** | 77/100 |

100 例中 **4 例的 route 发生变化**，已在表中用 ⚠ 标出：

| 用例 | route 变化 |
|---|---|
| cumsum_i32_w128_reverse | SIMT+F2 → **SIMD+F1** |
| grid_stride_dot_p112 | SIMT+F1 → **SIMD+F1** |
| pytest_index_put_n64_d32_m8 | SIMT+F1 → **MIXED+F1** |
| pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512 | SIMT+F4 → **SIMT+F2** |

按官方实测，**17 例**从 <10% 掉到 ≥10%，这就是要消除的清单：pytest_layer_norm_f16_m128_n128、pytest_nearest_resize_i32_1x4x64x64、cumsum_i32_w256、pytest_discrete_mask_4way_f16_8x8、pytest_indirect_index_div_b8、pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512、atomic_discrete_i32_p1_b256、pytest_vector_add_n98432、pytest_exp_f32_2x4096x8、pytest_mul_reduce_f16_16x1x1x1x1、histogram_i32_r8192_b256_single、pytest_max_dim0_f32_64x8、cumsum_i32_w128_reverse、histogram_i32_r8192_b256_radix、pytest_view_f32_2x256x16、atomic_discrete_i32_p56_b256、pytest_indirect_index_and_b8。

| # | 用例 | route | 实测us | 改前预测 | 改后预测 | 改前误差 | 改后误差 | 改后是否PASS |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | muls_add_n2097152 | SIMD+F1 | 7.025 | 6.735 | 6.735 | -4.1% | -4.1% | PASS |
| 2 | muls_add_n1048576 | SIMD+F1 | 5.033 | 4.734 | 4.734 | -5.9% | -5.9% | PASS |
| 3 | memory_stream_l4_n2097152 | SIMD+F1 | 10.231 | 10.371 | 10.371 | +1.4% | +1.4% | PASS |
| 4 | memory_stream_l2_n1048576 | SIMD+F1 | 4.706 | 4.729 | 4.729 | +0.5% | +0.5% | PASS |
| 5 | cumsum_i32_w256 | SIMT+F2 | 14.941 | 15.144 | 19.628 | +1.4% | +31.4% | FAIL |
| 6 | cumsum_i32_w128_reverse | SIMT+F2 → SIMD+F1 ⚠ | 16.686 | 16.756 | 18.921 | +0.4% | +13.4% | FAIL |
| 7 | gather_dot_f16_k16 | SIMT+F1 | 9.279 | 8.455 | 8.589 | -8.9% | -7.4% | PASS |
| 8 | gather_dot_f16_k16_p112 | SIMT+F1 | 6.184 | 5.774 | 5.879 | -6.6% | -4.9% | PASS |
| 9 | masked_copy_r16_l64_v64_s64_k8 | SIMD+F1 | 2.952 | 3.116 | 2.945 | +5.6% | -0.2% | PASS |
| 10 | masked_copy_r64_l128_v128_s256_k12 | SIMD+F1 | 7.368 | 7.669 | 7.669 | +4.1% | +4.1% | PASS |
| 11 | grid_stride_gather_p224 | SIMT+F1 | 4.684 | 4.626 | 4.760 | -1.2% | +1.6% | PASS |
| 12 | grid_stride_dot_p224 | SIMD+F1 | 5.076 | 5.530 | 5.493 | +8.9% | +8.2% | PASS |
| 13 | grid_stride_dot_p56 | SIMT+F1 | 4.007 | 3.990 | 4.186 | -0.4% | +4.5% | PASS |
| 14 | grid_stride_dot_p112 | SIMT+F1 → SIMD+F1 ⚠ | 4.989 | 4.941 | 5.087 | -1.0% | +2.0% | PASS |
| 15 | grid_stride_dot_p336 | SIMD+F1 | 5.608 | 5.963 | 5.900 | +6.3% | +5.2% | PASS |
| 16 | grid_stride_dot_p448 | SIMD+F1 | 5.883 | 6.397 | 6.307 | +8.7% | +7.2% | PASS |
| 17 | pytest_indirect_load_add_b8 | SIMT+F1 | 1.210 | 1.215 | 1.215 | +0.4% | +0.4% | PASS |
| 18 | pytest_indirect_index_div_b8 | SIMT+F1 | 1.233 | 1.263 | 1.509 | +2.4% | +22.4% | FAIL |
| 19 | pytest_indirect_index_and_b8 | SIMT+F1 | 1.172 | 1.231 | 1.290 | +5.0% | +10.1% | FAIL |
| 20 | pytest_indirect_index_maxmin_b8 | SIMT+F1 | 1.186 | 1.231 | 1.290 | +3.8% | +8.8% | PASS |
| 21 | pytest_indirect_load_2d_min_m4_n16_k8 | SIMT+F1 | 1.225 | 1.255 | 1.255 | +2.4% | +2.4% | PASS |
| 22 | pytest_indirect_load_2d_grid_m4_n16_k8 | SIMT+F1 | 1.715 | 1.573 | 1.573 | -8.3% | -8.3% | PASS |
| 23 | pytest_index_select_n64_d32_m8 | SIMD+F1 | 1.126 | 1.084 | 1.166 | -3.7% | +3.6% | PASS |
| 24 | pytest_index_put_n64_d32_m8 | SIMT+F1 → MIXED+F1 ⚠ | 1.307 | 1.264 | 1.332 | -3.3% | +1.9% | PASS |
| 25 | pytest_indirect_store_b8 | SIMT+F1 | 1.068 | 1.162 | 1.162 | +8.8% | +8.8% | PASS |
| 26 | pytest_gather_dot_f16_k16_p56 | SIMT+F1 | 4.753 | 4.433 | 4.524 | -6.7% | -4.8% | PASS |
| 27 | pytest_vector_add_n98432 | SIMD+F1 | 2.623 | 2.864 | 3.065 | +9.2% | +16.9% | FAIL |
| 28 | pytest_permute_x64_y32 | SIMD+F1 | 2.324 | 2.333 | 2.422 | +0.4% | +4.2% | PASS |
| 29 | pytest_sigmoid_f32_2x4096x8 | SIMD+F1 | 4.909 | 4.633 | 4.633 | -5.6% | -5.6% | PASS |
| 30 | pytest_rms_norm_f16_m128_n128 | SIMD+F1 | 6.131 | 6.611 | 6.611 | +7.8% | +7.8% | PASS |
| 31 | pytest_layer_norm_f16_m128_n128 | SIMT+F1 | 3.631 | 3.461 | 5.449 | -4.7% | +50.1% | FAIL |
| 32 | pytest_insert_slice_n1024_s32 | SIMD+F1 | 1.132 | 1.078 | 1.195 | -4.8% | +5.6% | PASS |
| 33 | pytest_atomic_add_f32_32x32_c2 | SIMD+F1 | 1.765 | 1.849 | 1.932 | +4.8% | +9.5% | PASS |
| 34 | pytest_exp_f32_2x4096x8 | SIMD+F1 | 4.353 | 4.596 | 5.086 | +5.6% | +16.8% | FAIL |
| 35 | pytest_log_f32_n781 | SIMD+F1 | 1.055 | 1.057 | 1.141 | +0.2% | +8.2% | PASS |
| 36 | pytest_relu_f32_2x4096x8 | SIMD+F1 | 4.361 | 4.628 | 4.628 | +6.1% | +6.1% | PASS |
| 37 | pytest_where_f32_2x1024x8 | SIMD+F1 | 1.938 | 1.959 | 1.959 | +1.1% | +1.1% | PASS |
| 38 | pytest_broadcast_bf16_2x4x8 | SIMD+F1 | 0.958 | 0.956 | 0.973 | -0.2% | +1.6% | PASS |
| 39 | pytest_max_dim1_f32_m64_n8 | SIMD+F1 | 1.270 | 1.397 | 1.461 | +10.0% | +15.0% | FAIL |
| 40 | pytest_mean_dim1_f32_m57_n3 | SIMD+F1 | 1.160 | 1.166 | 1.231 | +0.5% | +6.1% | PASS |
| 41 | pytest_mul_reduce_f16_16x1x1x1x1 | SIMD+F1 | 1.002 | 1.067 | 1.166 | +6.5% | +16.4% | FAIL |
| 42 | pytest_cumprod_f32_n1024 | SIMT+F1 | 1.787 | 1.820 | 1.756 | +1.8% | -1.7% | PASS |
| 43 | pytest_cummax_f32_16x32_dim1 | SIMT+F1 | 1.294 | 1.214 | 1.324 | -6.2% | +2.3% | PASS |
| 44 | pytest_atomic_max_i32_32x32_c2 | SIMD+F1 | 1.356 | 1.446 | 1.463 | +6.6% | +7.9% | PASS |
| 45 | atomic_continuous_i32_p1_b256 | SIMD+F1 | 1.077 | 1.009 | 1.026 | -6.3% | -4.7% | PASS |
| 46 | atomic_continuous_i32_p56_b256 | SIMD+F1 | 2.389 | 2.306 | 2.323 | -3.5% | -2.8% | PASS |
| 47 | atomic_discrete_i32_p1_b256 | SIMT+F1 | 1.538 | 1.527 | 1.809 | -0.7% | +17.6% | FAIL |
| 48 | atomic_discrete_i32_p56_b256 | SIMT+F1 | 2.807 | 2.824 | 3.105 | +0.6% | +10.6% | FAIL |
| 49 | rowred_r16_c128_k16 | SIMT+F1 | 23.994 | 15.880 | 17.455 | -33.8% | -27.3% | FAIL |
| 50 | rowred_r64_c256_k22 | SIMD+F1 | 103.389 | 92.769 | 92.769 | -10.3% | -10.3% | FAIL |
| 51 | softmax_power2 | SIMD+F1 | 68.194 | 65.200 | 65.200 | -4.4% | -4.4% | PASS |
| 52 | softmax_original | SIMD+F1 | 33.076 | 31.261 | 34.349 | -5.5% | +3.8% | PASS |
| 53 | fze_default | SIMT+F2 | 2838.693 | 2840.585 | 2846.596 | +0.1% | +0.3% | PASS |
| 54 | solve_tril_b1_t1024_h32_bt64 | MIXED+F4 | 189.258 | 178.894 | 178.894 | -5.5% | -5.5% | PASS |
| 55 | solve_tril_b1_t1024_h64_bt64 | MIXED+F4 | 335.352 | 324.141 | 324.141 | -3.3% | -3.3% | PASS |
| 56 | solve_tril_b8_t1024_h32_bt64 | MIXED+F4 | 1466.227 | 1424.071 | 1424.071 | -2.9% | -2.9% | PASS |
| 57 | solve_tril_b16_t1024_h64_bt64 | MIXED+F4 | 5797.191 | 5885.014 | 5885.014 | +1.5% | +1.5% | PASS |
| 58 | solve_tril_b1_t8192_h64_bt64 | MIXED+F4 | 2914.962 | 2899.816 | 2899.816 | -0.5% | -0.5% | PASS |
| 59 | solve_tril_b8_t8192_h32_bt64 | MIXED+F4 | 11585.062 | 11840.531 | 11840.531 | +2.2% | +2.2% | PASS |
| 60 | solve_tril_b16_t8192_h64_bt64 | MIXED+F4 | 46327.622 | 47550.853 | 47550.853 | +2.6% | +2.6% | PASS |
| 61 | solve_tril_b4_t131072_h32_bt64 | MIXED+F4 | 92658.205 | 95172.210 | 95172.210 | +2.7% | +2.7% | PASS |
| 62 | masked_reduction_r32_l150_v256_s150_k12 | SIMD+F1 | 46.448 | 43.913 | 43.913 | -5.5% | -5.5% | PASS |
| 63 | masked_reduction_r64_l150_v256_s150_k22 | SIMD+F1 | 101.026 | 92.947 | 92.947 | -8.0% | -8.0% | PASS |
| 64 | masked_softmax_r32_l150_v256_s150_k12 | SIMD+F1 | 16.442 | 16.995 | 16.995 | +3.4% | +3.4% | PASS |
| 65 | masked_softmax_r80_l150_v150_s150_k18 | SIMD+F1 | 32.937 | 31.260 | 34.348 | -5.1% | +4.3% | PASS |
| 66 | pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512 | SIMT+F4 → SIMT+F2 ⚠ | 8.764 | 8.259 | 10.482 | -5.8% | +19.6% | FAIL |
| 67 | pytest_sum_low_f32_64x8192 | SIMD+F1 | 46.643 | 46.435 | 46.499 | -0.4% | -0.3% | PASS |
| 68 | pytest_sum_high_f32_64x8192 | SIMD+F1 | 4.481 | 3.961 | 3.961 | -11.6% | -11.6% | FAIL |
| 69 | histogram_i32_r8192_b256_uniform | MIXED+F1 | 78.776 | 73.133 | 85.366 | -7.2% | +8.4% | PASS |
| 70 | histogram_i32_r8192_b256_single | MIXED+F1 | 74.499 | 73.133 | 85.366 | -1.8% | +14.6% | FAIL |
| 71 | histogram_i32_r8192_b256_radix | MIXED+F1 | 75.547 | 73.133 | 85.366 | -3.2% | +13.0% | FAIL |
| 72 | pytest_discrete_mask_4way_f16_8x8 | SIMT+F1 | 1.009 | 1.029 | 1.243 | +2.0% | +23.2% | FAIL |
| 73 | pytest_reduce_sum_epilogue_f32_8x8x512 | SIMD+F1 | 2.884 | 2.380 | 2.380 | -17.5% | -17.5% | FAIL |
| 74 | pytest_max_dim0_f32_64x8 | SIMD+F1 | 1.257 | 1.371 | 1.436 | +9.1% | +14.2% | FAIL |
| 75 | pytest_argmax_f32_37x125_dim1 | SIMT+F1 | 22.185 | 22.483 | 21.399 | +1.3% | -3.5% | PASS |
| 76 | pytest_gather_after_reduce_f32_16x1024 | SIMD+F1 | 4.603 | 4.745 | 4.826 | +3.1% | +4.8% | PASS |
| 77 | pytest_view_f32_2x256x16 | SIMD+F1 | 1.279 | 1.304 | 1.436 | +2.0% | +12.3% | FAIL |
| 78 | pytest_permute_reshape_fused_f32 | SIMD+F1 | 6.498 | 6.431 | 6.431 | -1.0% | -1.0% | PASS |
| 79 | pytest_nearest_resize_i32_1x4x64x64 | SIMT+F1 | 2.226 | 2.346 | 2.989 | +5.4% | +34.3% | FAIL |
| 80 | pytest_sort_f32_256x16 | SIMD+F1 | 51.837 | 51.612 | 51.727 | -0.4% | -0.2% | PASS |
| 81 | pytest_topk_f32_16x64_k4 | SIMT+F1 | 1.616 | 1.588 | 1.588 | -1.7% | -1.7% | PASS |
| 82 | pytest_multi_input_scan_f32_2x5x10 | SIMT+F1 | 1.588 | 1.571 | 1.571 | -1.1% | -1.1% | PASS |
| 83 | pytest_cummin_f32_3x5x7_dim1 | SIMD+F1 | 1.166 | 1.160 | 1.160 | -0.5% | -0.5% | PASS |
| 84 | pytest_histogram_i32_2048_bins2 | MIXED+F1 | 2.902 | 3.028 | 3.113 | +4.3% | +7.3% | PASS |
| 85 | pytest_scalar_reduce_store_f32_n16 | SIMD+F1 | 1.523 | 1.559 | 1.576 | +2.4% | +3.5% | PASS |
| 86 | pytest_conditional_load_f32_n32 | SIMT+F1 | 1.091 | 1.117 | 1.117 | +2.4% | +2.4% | PASS |
| 87 | pytest_dot_bf16_16x32x16 | SIMD+F1 | 1.249 | 1.246 | 1.346 | -0.2% | +7.8% | PASS |
| 88 | pytest_fused_softmax_f32_128x257 | SIMD+F1 | 3.202 | 3.006 | 3.191 | -6.1% | -0.3% | PASS |
| 89 | pytest_add_f32_2x4096x8 | SIMD+F1 | 4.365 | 4.628 | 4.628 | +6.0% | +6.0% | PASS |
| 90 | pytest_sqrt_f32_n781 | SIMD+F1 | 1.039 | 1.023 | 1.107 | -1.5% | +6.5% | PASS |
| 91 | pytest_strided_scatter_f32_s16 | SIMT+F1 | 4.446 | 3.979 | 5.105 | -10.5% | +14.8% | FAIL |
| 92 | pytest_block_ptr_strided_f32_4x8_s32x4 | SIMT+F1 | 1.076 | 1.091 | 1.091 | +1.4% | +1.4% | PASS |
| 93 | pytest_cos_f32_n781 | SIMD+F1 | 1.515 | 1.561 | 1.645 | +3.0% | +8.6% | PASS |
| 94 | pytest_mul_f32_2x4096x8 | SIMD+F1 | 4.373 | 4.628 | 4.628 | +5.8% | +5.8% | PASS |
| 95 | pytest_associative_scan_max_f32_128x4x16_dim2 | SIMT+F1 | 5.064 | 5.415 | 5.415 | +6.9% | +6.9% | PASS |
| 96 | pytest_div_f32_2x4096x8 | SIMD+F1 | 4.406 | 4.628 | 4.628 | +5.0% | +5.0% | PASS |
| 97 | pytest_f2i32_f32_n3 | SIMD+F1 | 0.968 | 0.958 | 1.041 | -1.0% | +7.5% | PASS |
| 98 | pytest_chunk_cumsum_f32_b4_t2048_h8_bt128 | SIMT+F2 | 3.598 | 3.869 | 3.869 | +7.5% | +7.5% | PASS |
| 99 | pytest_maximum_f32_2x4096x8 | SIMD+F1 | 5.712 | 5.673 | 5.673 | -0.7% | -0.7% | PASS |
| 100 | pytest_strided_compaction_f32_s16 | SIMD+F1 | 2.974 | 2.916 | 2.916 | -2.0% | -2.0% | PASS |

### 7.5 公式合并 + 带宽口径后的 100-case 预测

本节把两件事同时落地后重测：

| 改动 | 内容 |
|---|---|
| 公式合并 | 43 条 → **16 条**（14 SIMT + 2 SIMD），新增 `mask` / `log2eb` / `warpRounds` / `i8wRounds` / `halfwRounds` 五个特征 |
| 带宽口径接入 | 无循环携带依赖的 stage 改走 `bw.*` 共 15 条吞吐曲线，形如 `max(a0 + a1/u, b·KiB)` |

两列的来源：

| 列 | 来源 |
|---|---|
| 基线预测 | `b695c198a` + 回退 `6cc785f94f`，**不含任何连续访存改动**（profile 无 `contiguous_ldst`）|
| 新预测 | 同上 + 公式合并 + 带宽口径 |
| route | 取新版的判定 |
| 实测us | 官方基准表（与 7.3 / 7.4 同一列）|

| 指标 | 基线 | 新版 |
|---|---:|---:|
| 中位绝对误差 | **4.71%** | 8.09% |
| <10% 比例 | **87/100** | 57/100 |

**结论：这一版比基线差**，中位误差从 4.71% 升到 8.09%，达标从 87 例掉到 57 例。其中 **34 例**从 <10% 掉到 ≥10%，恶化最严重的是 SIMD 大块 elementwise：

| 用例 | 实测us | 基线误差 | 新误差 |
|---|---:|---:|---:|
| pytest_relu_f32_2x4096x8 | 4.361 | +6.1% | +123.5% |
| pytest_add_f32_2x4096x8 | 4.365 | +6.0% | +123.3% |
| pytest_mul_f32_2x4096x8 | 4.373 | +5.8% | +122.9% |
| pytest_div_f32_2x4096x8 | 4.406 | +5.0% | +121.2% |
| memory_stream_l4_n2097152 | 10.231 | +1.4% | +109.4% |
| pytest_sigmoid_f32_2x4096x8 | 4.909 | -5.6% | +98.7% |
| pytest_maximum_f32_2x4096x8 | 5.712 | -0.7% | +88.9% |
| pytest_where_f32_2x1024x8 | 1.938 | +1.1% | +88.3% |
| pytest_exp_f32_2x4096x8 | 4.353 | +5.6% | +75.4% |
| muls_add_n2097152 | 7.025 | -4.1% | +71.2% |

## 7.5.1 失败原因分析

| 现象 | 判断 |
|---|---|
| `2x4096x8` 一类（单次访存 16 KiB）普遍 +70%~+124% | 新的 SIMD 公式 `66.38 + 0.005412·by + 5.80·blk_part + 73.06·burst` 在 **16 KiB 处约为基线的 2 倍**；该公式的标定点覆盖到 16 KB，但真实 kernel 的 tile 正好落在上界，外推误差最大 |
| 加 `realBy ≤ 8 KiB` 守卫后数值完全未变 | 说明这些用例的访问本就 **≤ 8 KiB**（`2x4096x8` 是每程序 4096 元素 × 4 B = 16 KiB，但分摊到多次访问），问题在**延迟公式本身**而非带宽分支 |
| 基线在这些用例上 +1%~+6% | 基线走的是通用回退曲线（按字节折算带宽），恰好接近实测；我们的闭式公式反而更差 |

也就是说：**这次性能回归的主因不是带宽口径，而是把 SIMD 的延迟公式换成了文档版**（§3.2 那两条）。文档版在它自己的 133 个标定点上是 6.97%，但那些点里没有 `2x4096x8` 这类「多 stage 同形访问 + 大 tile」的组合。

## 7.5.2 下一步

| 项 | 做法 |
|---|---|
| 1 | 把 SIMD 延时公式换回旧的 `one+by`（或对 `by` 加分段），先在 100-case 上把 SIMD 大块拉回来 |
| 2 | 用 `ASCEND_MODEL_NO_CONTIG=1` 做 A/B，逐 stage 定位是哪一类访问被高估 |
| 3 | 带宽口径本身（`bw.*`）在这 100 例里没有造成回归，可保留 |

| # | 用例 | route | 实测us | 基线预测 | 新预测 | 基线误差 | 新误差 | 新是否PASS |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | muls_add_n2097152 | all_simd+F1 | 7.025 | 6.735 | 12.023 | -4.1% | +71.2% | FAIL |
| 2 | muls_add_n1048576 | all_simd+F1 | 5.033 | 4.734 | 7.450 | -5.9% | +48.0% | FAIL |
| 3 | memory_stream_l4_n2097152 | all_simd+F1 | 10.231 | 10.371 | 21.420 | +1.4% | +109.4% | FAIL |
| 4 | memory_stream_l2_n1048576 | all_simd+F1 | 4.706 | 4.729 | 7.445 | +0.5% | +58.2% | FAIL |
| 5 | cumsum_i32_w256 | all_simt_only+F2 | 14.941 | 15.144 | 20.209 | +1.4% | +35.3% | FAIL |
| 6 | cumsum_i32_w128_reverse | all_simd+F1 | 16.686 | 16.756 | 18.921 | +0.4% | +13.4% | FAIL |
| 7 | gather_dot_f16_k16 | all_simt_only+F1 | 9.279 | 8.455 | 8.603 | -8.9% | -7.3% | PASS |
| 8 | gather_dot_f16_k16_p112 | all_simt_only+F1 | 6.184 | 5.774 | 5.886 | -6.6% | -4.8% | PASS |
| 9 | masked_copy_r16_l64_v64_s64_k8 | all_simd+F1 | 2.952 | 3.116 | 3.582 | +5.5% | +21.3% | FAIL |
| 10 | masked_copy_r64_l128_v128_s256_k12 | all_simd+F1 | 7.368 | 7.669 | 7.669 | +4.1% | +4.1% | PASS |
| 11 | grid_stride_gather_p224 | all_simt_only+F1 | 4.684 | 4.626 | 4.787 | -1.2% | +2.2% | PASS |
| 12 | grid_stride_dot_p224 | all_simd+F1 | 5.076 | 5.530 | 5.877 | +8.9% | +15.8% | FAIL |
| 13 | grid_stride_dot_p56 | all_simt_only+F1 | 4.007 | 3.990 | 4.131 | -0.4% | +3.1% | PASS |
| 14 | grid_stride_dot_p112 | all_simt_only+F1 | 4.989 | 4.941 | 5.105 | -1.0% | +2.3% | PASS |
| 15 | grid_stride_dot_p336 | all_simd+F1 | 5.608 | 5.963 | 6.481 | +6.3% | +15.6% | FAIL |
| 16 | grid_stride_dot_p448 | all_simd+F1 | 5.883 | 6.397 | 7.086 | +8.7% | +20.4% | FAIL |
| 17 | pytest_indirect_load_add_b8 | all_simt_only+F1 | 1.210 | 1.215 | 1.215 | +0.4% | +0.4% | PASS |
| 18 | pytest_indirect_index_div_b8 | all_simt_only+F1 | 1.233 | 1.263 | 1.480 | +2.4% | +20.0% | FAIL |
| 19 | pytest_indirect_index_and_b8 | all_simt_only+F1 | 1.172 | 1.231 | 1.293 | +5.0% | +10.3% | FAIL |
| 20 | pytest_indirect_index_maxmin_b8 | all_simt_only+F1 | 1.186 | 1.231 | 1.293 | +3.8% | +9.1% | PASS |
| 21 | pytest_indirect_load_2d_min_m4_n16_k8 | all_simt_only+F1 | 1.225 | 1.255 | 1.255 | +2.5% | +2.5% | PASS |
| 22 | pytest_indirect_load_2d_grid_m4_n16_k8 | all_simt_only+F1 | 1.715 | 1.573 | 1.573 | -8.3% | -8.3% | PASS |
| 23 | pytest_index_select_n64_d32_m8 | all_simd+F1 | 1.126 | 1.084 | 1.138 | -3.7% | +1.1% | PASS |
| 24 | pytest_index_put_n64_d32_m8 | mixed_simd_simt+F1 | 1.307 | 1.264 | 1.311 | -3.3% | +0.3% | PASS |
| 25 | pytest_indirect_store_b8 | all_simt_only+F1 | 1.068 | 1.162 | 1.162 | +8.8% | +8.8% | PASS |
| 26 | pytest_gather_dot_f16_k16_p56 | all_simt_only+F1 | 4.753 | 4.433 | 4.528 | -6.7% | -4.7% | PASS |
| 27 | pytest_vector_add_n98432 | all_simd+F1 | 2.623 | 2.864 | 3.045 | +9.2% | +16.1% | FAIL |
| 28 | pytest_permute_x64_y32 | all_simd+F1 | 2.324 | 2.333 | 2.427 | +0.4% | +4.4% | PASS |
| 29 | pytest_sigmoid_f32_2x4096x8 | all_simd+F1 | 4.909 | 4.633 | 9.752 | -5.6% | +98.7% | FAIL |
| 30 | pytest_rms_norm_f16_m128_n128 | all_simd+F1 | 6.131 | 6.611 | 7.542 | +7.8% | +23.0% | FAIL |
| 31 | pytest_layer_norm_f16_m128_n128 | all_simt_only+F1 | 3.631 | 3.461 | 5.641 | -4.7% | +55.3% | FAIL |
| 32 | pytest_insert_slice_n1024_s32 | all_simd+F1 | 1.132 | 1.078 | 1.185 | -4.8% | +4.7% | PASS |
| 33 | pytest_atomic_add_f32_32x32_c2 | all_simd+F1 | 1.765 | 1.849 | 1.909 | +4.7% | +8.2% | PASS |
| 34 | pytest_exp_f32_2x4096x8 | all_simd+F1 | 4.353 | 4.596 | 7.633 | +5.6% | +75.4% | FAIL |
| 35 | pytest_log_f32_n781 | all_simd+F1 | 1.055 | 1.057 | 1.131 | +0.2% | +7.2% | PASS |
| 36 | pytest_relu_f32_2x4096x8 | all_simd+F1 | 4.361 | 4.628 | 9.747 | +6.1% | +123.5% | FAIL |
| 37 | pytest_where_f32_2x1024x8 | all_simd+F1 | 1.938 | 1.959 | 3.649 | +1.1% | +88.3% | FAIL |
| 38 | pytest_broadcast_bf16_2x4x8 | all_simd+F1 | 0.958 | 0.956 | 0.961 | -0.2% | +0.3% | PASS |
| 39 | pytest_max_dim1_f32_m64_n8 | all_simd+F1 | 1.270 | 1.397 | 1.443 | +10.0% | +13.6% | FAIL |
| 40 | pytest_mean_dim1_f32_m57_n3 | all_simd+F1 | 1.160 | 1.166 | 1.213 | +0.5% | +4.5% | PASS |
| 41 | pytest_mul_reduce_f16_16x1x1x1x1 | all_simd+F1 | 1.002 | 1.067 | 1.123 | +6.5% | +12.0% | FAIL |
| 42 | pytest_cumprod_f32_n1024 | all_simt_only+F1 | 1.787 | 1.820 | 1.756 | +1.8% | -1.8% | PASS |
| 43 | pytest_cummax_f32_16x32_dim1 | all_simt_only+F1 | 1.294 | 1.214 | 1.313 | -6.2% | +1.5% | PASS |
| 44 | pytest_atomic_max_i32_32x32_c2 | all_simd+F1 | 1.356 | 1.446 | 1.455 | +6.6% | +7.3% | PASS |
| 45 | atomic_continuous_i32_p1_b256 | all_simd+F1 | 1.077 | 1.009 | 1.016 | -6.3% | -5.6% | PASS |
| 46 | atomic_continuous_i32_p56_b256 | all_simd+F1 | 2.389 | 2.306 | 2.313 | -3.5% | -3.2% | PASS |
| 47 | atomic_discrete_i32_p1_b256 | all_simt_only+F1 | 1.538 | 1.527 | 1.862 | -0.7% | +21.0% | FAIL |
| 48 | atomic_discrete_i32_p56_b256 | all_simt_only+F1 | 2.807 | 2.824 | 3.158 | +0.6% | +12.5% | FAIL |
| 49 | rowred_r16_c128_k16 | all_simt_only+F1 | 23.994 | 15.880 | 14.445 | -33.8% | -39.8% | FAIL |
| 50 | rowred_r64_c256_k22 | all_simd+F1 | 103.389 | 92.769 | 92.769 | -10.3% | -10.3% | FAIL |
| 51 | softmax_power2 | all_simd+F1 | 68.194 | 65.200 | 65.200 | -4.4% | -4.4% | PASS |
| 52 | softmax_original | all_simd+F1 | 33.076 | 31.261 | 41.611 | -5.5% | +25.8% | FAIL |
| 53 | fze_default | all_simt_only+F2 | 2838.693 | 2840.585 | 2846.636 | +0.1% | +0.3% | PASS |
| 54 | solve_tril_b1_t1024_h32_bt64 | all_simt_only+F4 | 189.258 | 123.754 | 197.394 | -34.6% | +4.3% | PASS |
| 55 | solve_tril_b1_t1024_h64_bt64 | all_simt_only+F4 | 335.352 | 211.694 | 341.145 | -36.9% | +1.7% | PASS |
| 56 | solve_tril_b8_t1024_h32_bt64 | all_simt_only+F4 | 1466.227 | 772.701 | 1310.113 | -47.3% | -10.6% | FAIL |
| 57 | solve_tril_b16_t1024_h64_bt64 | all_simt_only+F4 | 5797.191 | 2983.361 | 5096.297 | -48.5% | -12.1% | FAIL |
| 58 | solve_tril_b1_t8192_h64_bt64 | all_simt_only+F4 | 2914.962 | 1509.588 | 2552.760 | -48.2% | -12.4% | FAIL |
| 59 | solve_tril_b8_t8192_h32_bt64 | all_simt_only+F4 | 11585.062 | 5964.278 | 10211.861 | -48.5% | -11.9% | FAIL |
| 60 | solve_tril_b16_t8192_h64_bt64 | all_simt_only+F4 | 46327.622 | 23749.669 | 40482.129 | -48.7% | -12.6% | FAIL |
| 61 | solve_tril_b4_t131072_h32_bt64 | all_simt_only+F4 | 92658.205 | 47496.894 | 81425.850 | -48.7% | -12.1% | FAIL |
| 62 | masked_reduction_r32_l150_v256_s150_k12 | all_simd+F1 | 46.448 | 43.913 | 43.913 | -5.5% | -5.5% | PASS |
| 63 | masked_reduction_r64_l150_v256_s150_k22 | all_simd+F1 | 101.026 | 92.947 | 92.947 | -8.0% | -8.0% | PASS |
| 64 | masked_softmax_r32_l150_v256_s150_k12 | all_simd+F1 | 16.442 | 16.995 | 16.995 | +3.4% | +3.4% | PASS |
| 65 | masked_softmax_r80_l150_v150_s150_k18 | all_simd+F1 | 32.937 | 31.260 | 41.610 | -5.1% | +26.3% | FAIL |
| 66 | pytest_fbgemm_rowwise_quant_t256_d1024_e8_v512 | all_simt_only+F2 | 8.764 | 8.259 | 10.528 | -5.8% | +20.1% | FAIL |
| 67 | pytest_sum_low_f32_64x8192 | all_simd+F1 | 46.643 | 46.435 | 46.481 | -0.4% | -0.3% | PASS |
| 68 | pytest_sum_high_f32_64x8192 | all_simd+F1 | 4.481 | 3.961 | 3.961 | -11.6% | -11.6% | FAIL |
| 69 | histogram_i32_r8192_b256_uniform | mixed_simd_simt+F1 | 78.776 | 73.133 | 81.486 | -7.2% | +3.4% | PASS |
| 70 | histogram_i32_r8192_b256_single | mixed_simd_simt+F1 | 74.499 | 73.133 | 81.486 | -1.8% | +9.4% | PASS |
| 71 | histogram_i32_r8192_b256_radix | mixed_simd_simt+F1 | 75.547 | 73.133 | 81.486 | -3.2% | +7.9% | PASS |
| 72 | pytest_discrete_mask_4way_f16_8x8 | all_simd+F1 | 1.009 | 1.029 | 1.038 | +2.0% | +2.8% | PASS |
| 73 | pytest_reduce_sum_epilogue_f32_8x8x512 | all_simd+F1 | 2.884 | 2.380 | 3.148 | -17.5% | +9.2% | PASS |
| 74 | pytest_max_dim0_f32_64x8 | all_simd+F1 | 1.257 | 1.371 | 1.418 | +9.1% | +12.8% | FAIL |
| 75 | pytest_argmax_f32_37x125_dim1 | all_simt_only+F1 | 22.185 | 22.483 | 21.459 | +1.3% | -3.3% | PASS |
| 76 | pytest_gather_after_reduce_f32_16x1024 | all_simd+F1 | 4.603 | 4.745 | 4.822 | +3.1% | +4.8% | PASS |
| 77 | pytest_view_f32_2x256x16 | all_simd+F1 | 1.279 | 1.304 | 1.494 | +2.0% | +16.8% | FAIL |
| 78 | pytest_permute_reshape_fused_f32 | all_simd+F1 | 6.498 | 6.431 | 11.042 | -1.0% | +69.9% | FAIL |
| 79 | pytest_nearest_resize_i32_1x4x64x64 | all_simt_only+F1 | 2.226 | 2.346 | 2.984 | +5.4% | +34.0% | FAIL |
| 80 | pytest_sort_f32_256x16 | all_simd+F1 | 51.837 | 51.612 | 51.750 | -0.4% | -0.2% | PASS |
| 81 | pytest_topk_f32_16x64_k4 | all_simt_only+F1 | 1.616 | 1.588 | 1.588 | -1.7% | -1.7% | PASS |
| 82 | pytest_multi_input_scan_f32_2x5x10 | all_simt_only+F1 | 1.588 | 1.571 | 1.571 | -1.1% | -1.1% | PASS |
| 83 | pytest_cummin_f32_3x5x7_dim1 | all_simd+F1 | 1.166 | 1.160 | 1.160 | -0.5% | -0.5% | PASS |
| 84 | pytest_histogram_i32_2048_bins2 | mixed_simd_simt+F1 | 2.902 | 3.028 | 3.121 | +4.3% | +7.5% | PASS |
| 85 | pytest_scalar_reduce_store_f32_n16 | all_simd+F1 | 1.523 | 1.559 | 1.572 | +2.4% | +3.2% | PASS |
| 86 | pytest_conditional_load_f32_n32 | all_simt_only+F1 | 1.091 | 1.117 | 1.176 | +2.4% | +7.8% | PASS |
| 87 | pytest_dot_bf16_16x32x16 | all_simd+F1 | 1.249 | 1.246 | 1.310 | -0.3% | +4.9% | PASS |
| 88 | pytest_fused_softmax_f32_128x257 | all_simd+F1 | 3.202 | 3.006 | 3.364 | -6.1% | +5.1% | PASS |
| 89 | pytest_add_f32_2x4096x8 | all_simd+F1 | 4.365 | 4.628 | 9.748 | +6.0% | +123.3% | FAIL |
| 90 | pytest_sqrt_f32_n781 | all_simd+F1 | 1.039 | 1.023 | 1.097 | -1.5% | +5.5% | PASS |
| 91 | pytest_strided_scatter_f32_s16 | all_simt_only+F1 | 4.446 | 3.979 | 4.036 | -10.5% | -9.2% | PASS |
| 92 | pytest_block_ptr_strided_f32_4x8_s32x4 | all_simt_only+F1 | 1.076 | 1.091 | 1.091 | +1.4% | +1.4% | PASS |
| 93 | pytest_cos_f32_n781 | all_simd+F1 | 1.515 | 1.561 | 1.635 | +3.1% | +7.9% | PASS |
| 94 | pytest_mul_f32_2x4096x8 | all_simd+F1 | 4.373 | 4.628 | 9.748 | +5.8% | +122.9% | FAIL |
| 95 | pytest_associative_scan_max_f32_128x4x16_dim2 | all_simt_only+F1 | 5.064 | 5.415 | 5.415 | +6.9% | +6.9% | PASS |
| 96 | pytest_div_f32_2x4096x8 | all_simd+F1 | 4.406 | 4.628 | 9.748 | +5.0% | +121.2% | FAIL |
| 97 | pytest_f2i32_f32_n3 | all_simd+F1 | 0.968 | 0.958 | 1.010 | -1.0% | +4.3% | PASS |
| 98 | pytest_chunk_cumsum_f32_b4_t2048_h8_bt128 | all_simt_only+F2 | 3.598 | 3.869 | 3.869 | +7.5% | +7.5% | PASS |
| 99 | pytest_maximum_f32_2x4096x8 | all_simd+F1 | 5.712 | 5.673 | 10.793 | -0.7% | +88.9% | FAIL |
| 100 | pytest_strided_compaction_f32_s16 | all_simd+F1 | 2.974 | 2.916 | 2.916 | -1.9% | -1.9% | PASS |

### 7.6 组内访存条数：N 条 load + 1 条 store

目的：我们的闭式公式按“一条孤立访存”标定，需要知道**同一轮迭代里放多条连续访存**时硬件如何计价。

#### 7.4.1 kernel 核心代码

```python
@triton.jit
def nloads_kernel(a, out, N: tl.constexpr, WINMASK: tl.constexpr,
                  NLOADS: tl.constexpr, ITERS: tl.constexpr):
    off = tl.arange(0, N)
    acc = tl.zeros((N,), a.dtype.element_ty)
    for i in range(ITERS):
        base = (i & WINMASK) * N
        for k in tl.static_range(NLOADS):
            acc += tl.load(a + base + k * N + off)   # NLOADS 条连续 load
    tl.store(out + off, acc)                          # 循环结束后 1 次 store
```

- 用 `+` 累加，与标定探针和 `add3` 同风格；dtype 用 fp32，与整核测试一致。
- store 只在循环结束后执行一次，摊到 2048 轮后约为 0，所以这个 kernel 实测的**就是 load 的代价**。
- 单一变量：只改 `NLOADS`。grid 都是 1。

#### 7.4.2 实测结果（fp32，每次 load 16 KB，2048 轮，取 3 次最小）

| load 条数 | 周期/迭代 | 边际成本（每多一条）| 有效带宽 | 公式预测 | 公式误差 |
|---:|---:|---:|---:|---:|---:|
| 1 | 246.6 | — | 66 B/周期 | 255.8 | **+3.7%** |
| 2 | 460.9 | +214.3 | 76 B/周期 | 511.6 | −9.9% |
| 4 | 656.1 | **+97.6/条** | 167 B/周期 | 1023.2 | **−35.9%** |
| 8 | 2049.0 | +348/条 | — | 2046.4 | 溢出，不可比 |

公式预测 = `simd.load` 在 16384 B 处的值 255.8 周期 × 条数。

#### 7.4.3 结论

1. **单条 load 的公式是准的**（+3.7%），说明标定本身没有问题。
2. **多条 load 不是线性累加**：4 条的总代价只有 1 条的 2.7 倍，边际成本从 214 掉到 98 周期/条，有效带宽从 66 涨到 167 B/周期——硬件会把同方向的连续 load 流水起来。
3. 所以用“单条公式 × 条数”估多访问流会**系统性高估**（2 条 −10%，4 条 −36%）。这是当前模型只认领单条流、多访问流交回组曲线的依据。
4. 8 条时骤降到 2049 周期不是带宽问题，而是 8 × 16 KB 活跃数据超出片上容量导致溢出，任何线性模型都覆盖不了，需要单独判据。

#### 7.4.4 顺带验证：1-D 与 2-D 偏移是否等价

只改偏移写法，其余完全相同（`copy` kernel，fp16，2048 轮）：

| mode | 1-D `(4096,)` | 2-D `(1,4096)` | 差 |
|---|---:|---:|---:|
| simd | 219.9 周期 | 221.6 周期 | +0.8% |
| simt_only | 1005.4 周期 | 1004.3 周期 | −0.1% |

二维写法不是误差来源，后续实验统一用 1-D。

#### 7.4.5 为什么 copy 误差大、而单条 load 误差小

两者测的不是同一件事：

| kernel | 每轮内容 | 公式预测（周期/轮）| 实测（周期/轮）| 误差 |
|---|---|---:|---:|---:|
| N-loads，N=1 | 1 load，store 摊到 2048 轮后≈0 | 255.8 | 246.6 | **+3.7%** |
| copy（表格 6.2 第 48 行）| 1 load + 1 store | 293.7 | 528.7 | **−44.4%** |

N-loads 实验里 store 被摊掉了，只验证了 **load 公式**；copy 的误差全部来自 **load 与 store 的合成**。

把实测拆开看（fp32，16 KB/方向）：

- 实测 store-only = 305.4 周期，实测 copy = 438.7 周期（DCE-safe 对照）；
- 实测 load 边际 = 169.7 周期；
- copy ≈ load + store（246.6 + 305.4 = 552，实测 528.7），即**这一尺寸下两条流基本串行**。

而小尺寸下并非如此（fp16 2 KB：max(L,S)=161.7，copy=184.1，只多 14%）。所以重叠程度**随尺寸变化**：

| 每次方向字节数 | copy / max(L,S) | 含义 |
|---:|---:|---|
| 2 KB（fp16 1024）| 1.14 | 大部分重叠 |
| 8 KB（fp16 4096）| 1.16 | 大部分重叠 |
| 16 KB（fp32 4096）| 1.80 ~ 2.07 | 几乎串行 |

现在模型里 `k = 0.15` 是**固定值**，只对小块成立；16 KB 处真实值接近 1.0，所以 copy 被低估 44%。

**可执行的修法**：把 `k` 改成随字节数递增的函数（例如按 16 KB 归一的分段线性），用上表这类数据点拟合，而不是继续用常数。

### 7.7 单方向验证（把另一侧摊掉）

说明：本节表格里的 `N` 是**每次访问的元素数**（`--cols`），迭代数 `ITERS` 固定 2048；`NACC` 是每轮的访存条数。

思路：要验证 load 公式，就把 store 挪到循环外摊掉；要验证 store 公式，就让循环里只有 store、没有 load。这样每轮只剩一个方向的访存，公式和实测才可比。

#### 7.5.1 kernel 核心代码

```python
@triton.jit
def load_only(a, out, N: tl.constexpr, WINMASK: tl.constexpr,
              NACC: tl.constexpr, ITERS: tl.constexpr):
    off = tl.arange(0, N)
    acc = tl.zeros((N,), a.dtype.element_ty)
    for i in range(ITERS):
        base = (i & WINMASK) * N
        for k in tl.static_range(NACC):
            acc += tl.load(a + base + k * N + off)   # 只有 load
    tl.store(out + off, acc)                          # 循环外：摊掉

@triton.jit
def store_only(out, N: tl.constexpr, WINMASK: tl.constexpr,
               NACC: tl.constexpr, ITERS: tl.constexpr):
    off = tl.arange(0, N)
    v = tl.full((N,), 1, out.dtype.element_ty)
    for i in range(ITERS):
        base = (i & WINMASK) * N
        for k in tl.static_range(NACC):
            tl.store(out + base + k * N + off, v)    # 只有 store
```

#### 7.5.2 整核口径：动态迭代下的实测 vs 预测（42 点网格，进行中）

口径（按最新要求）：

- **被测 kernel 用动态 `for`**：`iters` 为运行时参数（`@triton.jit(do_not_specialize=['iters'])`），调用时传 16；
- **实测 = 整核时间**：按该点要求的 route 强制 `compile_mode`（SIMD 点用 `simd`，SIMT 点用 `simt_only`），用 `torch_npu.profiler` 抓 `kernel_details.csv` 的 `Duration(us)`，取 28 次中的最小值；
- **预测 = 整核时间**：先用 `compile_mode='simd_simt'` 编译一次拿到 route 报告，读 `selected_route_elapsed_prediction.predicted_microseconds`（同一份 JSON 里还带 `candidate_elapsed_costs`，可看各 route 的整核时间）；
- 探针每个配置单独起进程，避免同一进程内重复启动 profiler 失败。

#### 动态迭代下预测必然偏低（这是模型的已知边界，不是标定问题）

`StagePartitioner.cpp` 的 `getLoopTripCount` 对循环次数的处理顺序是：静态常量 → 符号表达式（循环界为 kernel 参数的仿射函数）→ 上界 → **都推不出来就按 1 次计**，并记 `unknown_trip_count_loop_count`。动态 `iters` 属于最后一种，模型看不到 16 次迭代，只按 1 次算，所以整核预测系统性偏低。

这在此前 100-case 上也得到印证：96 例里 17 例含“次数未知”的 stage，这 17 例中位误差 16.4%、仅 5 例 <10%；其余 74 例中位 6.7%、48 例 <10%。

#### 已测数据（12/42）

| dir | dtype | el | 预测us | 实测us | 误差 |
|---|---|---:|---:|---:|---:|
| load | fp16 | 127 | 1.041 | 1.632 | -36.2% |
| store | fp16 | 127 | 0.602 | 2.030 | -70.3% |
| load | fp16 | 1000 | 1.066 | 1.891 | -43.6% |
| store | fp16 | 1000 | 0.614 | 2.063 | -70.2% |
| load | fp16 | 1024 | 1.066 | 1.881 | -43.3% |
| store | fp16 | 1024 | 0.614 | 2.037 | -69.9% |
| load | fp16 | 4096 | 1.152 | 2.342 | -50.8% |
| store | fp16 | 4096 | 0.653 | 2.315 | -71.8% |
| load | fp16 | 16384 | 1.669 | 5.150 | -67.6% |
| store | fp16 | 16384 | 0.860 | 4.325 | -80.1% |
| load | fp32 | 127 | 1.045 | 1.761 | -40.7% |
| store | fp32 | 127 | 0.604 | 2.023 | -70.1% |

形态很清楚：el 从 127 涨到 16384，实测从 1.6 us 涨到 5.2 us，而预测几乎不动（1.04→1.67），误差从 −36% 恶化到 −68%（store 更差，−70%→−80%）。这正是“按 1 次迭代计价”的直接特征。

#### 待补

fp32 其余点与 SIMT 22 点为超时/失败，正在用更长超时重跑；补齐后本表会更新为 42 行。

#### 7.5.3 每条访问多条访存（NACC>1）：load 会流水，store 不会（关键不对称）

| dir | dtype | N | 条数 | 实测/条 | 公式/条 | 误差 | 实测总 | 公式总 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| load | fp32 | 1024 | 2 | 81.7 | 146.1 | -44.1% | 163.4 | 292.2 |
| load | fp32 | 1024 | 4 | 55.9 | 146.1 | -61.8% | 223.5 | 584.4 |
| load | fp32 | 4096 | 2 | 196.7 | 255.8 | -23.1% | 393.5 | 511.7 |
| load | fp32 | 4096 | 4 | 164.3 | 255.8 | -35.8% | 657.3 | 1023.4 |
| store | fp32 | 1024 | 2 | 174.4 | 166.1 | +5.0% | 348.8 | 332.3 |
| store | fp32 | 1024 | 4 | 167.2 | 166.1 | +0.6% | 668.7 | 664.6 |
| store | fp32 | 4096 | 2 | 253.7 | 252.6 | +0.5% | 507.4 | 505.1 |
| store | fp32 | 4096 | 4 | 253.7 | 252.6 | +0.4% | 1014.7 | 1010.2 |

**结论：**

- **store 是严格线性**：每条 166.1 / 252.6 周期，2 条 4 条都几乎不变（误差 ≤5%）。乘条数完全正确。
- **load 会流水**：单条 146.1，2 条降到 81.7/条，4 条降到 55.9/条 —— 多了 2.6 倍带宽。乘条数会**高估 24%~62%**。
- 这解释了为什么模型只认领单条流：单条公式对 load 是准的，但“×条数”这一步对 load 不成立、对 store 成立。

#### 7.5.4 对模型的直接含义

| 场景 | 当前做法 | 正确做法 |
|---|---|---|
| SIMD 单条 load / store | 闭式公式 | ✅ 已验证，中位 5% |
| SIMD 多条 load | 不认领，交回组曲线 | 需要 `T(n)` 曲线（8.3 表就是起点）|
| SIMD 多条 store | 不认领 | 可以直接线性相加（已验证 ≤5%）|
| SIMT 任意 | 用同一套公式 | ❌ 需要重做：溢出判据 + warp 建模 |

### 7.8 寄存器压力与溢出判定

#### 7.6.1 架构给出的寄存器预算

`04-ascend950-architecture-compendium/02-hardware-architecture.md` §2.7：SIMT Register File 每个 AIV 128 KB，每线程寄存器数由线程数决定。

| 线程数 | 每线程寄存器 | 字节预算 | 对应 warp |
|---:|---:|---:|---|
| 1~256 | 127 | 508 B | W ≤ 8 |
| 257~512 | 64 | 256 B | W 9~16 |
| 513~1024 | 32 | 128 B | W 17~32 |
| 1025~2048 | 16 | 64 B | W 33~64 |

代码里的 `regBytes()` 原先写的是 `W<=8→508, W==16→256, W==32→128, else→64`，于是 W=9~15 与 W=17~31 都掉进 64 B（应为 256 / 128）。已按线程数分档修正。

#### 7.6.2 report 里的实际寄存器压力

报告字段 `logical_stages[].whole_simt_register_pressure.peak_live_32bit_units_per_thread` 给出每线程峰值活跃寄存器数（32-bit 单位）。实测其规律为 `4 × te + 3`：

| n | te | peak_live | 预算 | ρ = live/预算 | 实测（load-only）| 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 35 | 127 | 0.28 | 332.9 | +5.1% |
| 512 | 16 | 67 | 127 | 0.53 | 413.8 | +7.0% |
| 1024 | 32 | 131 | 127 | **1.03** | 464 / 495 | +24% / +26% |
| 4096 | 128 | 515 | 127 | **4.06** | 6482 | **−82.6%** |

#### 7.6.3 结论：溢出判据用 ρ，不要用固定 te

- **ρ ≤ 1（不溢出）**：误差 5~7%，公式可用。
- **ρ 略大于 1**（te=32，ρ=1.03）：误差 24%，已明显劣化。
- **ρ ≫ 1**（te=128，ρ=4.06）：误差 −83%，实测是公式的 5.7 倍，是溢出而非带宽。

所以正确的分界是 **`4·te + 3 > reg_budget(W)`**（即 ρ > 1），不是固定的 `te > 8`。这与“先把不溢出的建模好、溢出的单独建模”的思路一致。

#### 7.6.4 尚未完成：LDG/STG 与 LDK/STK 的指令级确认

想直接看指令类型来确认溢出，三条路都试过：

| 方法 | 结果 |
|---|---|
| `llvm-objdump -d --triple=hiipu64`（后端自己用的那条）| 认得出 `elf64-hiipu`，但每条指令都是 `<not available>` |
| CANN 自带 `llvm-objdump` / `asc_dumper` | 同样解不出 |
| 编译器 `[BISHENG] ... Stack size ... Used register number` | 后端把 bishengir 的 stdout/stderr 吞掉，`TRITON_ALWAYS_COMPILE=1` 也拿不到 |

目前用 `peak_live_32bit_units_per_thread` 作为溢出证据，它与实测劣化点完全对应。

## 8. 附录：历史过程与已废止结论

以下内容保留结论、删除过程，避免与第 4 节的最终公式混用。

### 8.1 设备漂移与双峰（影响测量方法，仍然有效）

- 同一配置跨进程会出现双峰，差 1.38~2 倍；参考核归一化、固定偏移分配都消不掉；
- 因此测量口径定为：**同一配置跑 5 个独立进程取最小值**，重复性 0.9%；
- 单次运行为准则不可信，任何标定点都必须是 min-of-N。

**影响范围（重要）**：同一个 `value + step` 也出现在 `probe_mask_kind.py`（第 51 行），
而形态 B（带掩码）的扫描脚本 `run_formb_v2_sweep.sh` 正是调用它。因此：

| 形态 | 数据来源 | 是否受影响 | 现状 |
|---|---|---|---|
| A（无掩码）| `probe_block_pure.py` | 是 | 已重测 221 个配置并重拟合 |
| B（带掩码）| `probe_mask_kind.py` | **是** | **尚未重测**，其 store 段仍带这个偏差 |

按形态 A 的量级推断，形态 B 的偏差同样集中在小 `bytes`、随 `bytes` 增大而衰减；
但**未经实测确认，不应假定等于 11%**。

### 8.2 探针缺陷：store 循环里混入了向量加法（已修）

早期 store 探针写的是 `tl.store(..., value + step)`，把一次向量加塞进了访存循环。同一探针只改这一行做 A/B：

| 方向 | 配置 | `value+step` | `value` | 差异 |
|---|---|---:|---:|---:|
| load（对照）| fp32 el=8 W=64 | 155.09 | 155.14 | +0.0% |
| store | fp16 el=2 W=64 | 155.15 | 116.35 | −25.0% |
| store | fp32 el=8 W=1 | 30.28 | 26.04 | −14.0% |
| store | int8 el=64 W=64 | 126.66 | 126.76 | +0.1% |
| store | fp32 el=2048 W=4 | 234.29 | 235.79 | +0.6% |

结论：加法只在小 store 上显著；**SIMT store 系统性偏高约 11%，SIMD 侧不受影响**（三来源数值一致到 0.02%）。
小 store（`by <= 256 B`）已重测 221 个配置并重拟合（见 4.2 节）。

### 8.3 已废止的结论

| 旧结论 | 现状 |
|---|---|
| SIMD 公式用 `by` 三次项（4 系数）| 三次项是为在 10 点子集上凑 10% 门槛加的；全量 97 点上收益 6.82%→6.78%。记录在此，不再作为依据 |
| 8.2 用单次访存（k-marginal）对比模型访存分量 | 该口径只能验拟合，不能验整核；整核口径见 7.5.2 |
| 靠「认领范围」把 8.2 压到 <10% | 认领范围扩大反而使 100-case 变差（7.3 节），根因是循环内访存的重叠未建模 |
| 「基线没有连续访存标定」| 错。基线在 `stage_resources.continuous_memory`，只覆盖单点（1 操作 / 1024 B / fp32 / factor 1,2,4）|

### 8.4 仍未解决

- 循环内多次访存的重叠（软件流水）没有建模，模型按「单次延迟 × 次数」线性累加，导致循环体多的算子被高估；
- 动态 `iters`（运行时循环界）时模型按 1 次计价，整核预测系统性偏低 36%~80%（7.5.2 节数据）。

## 附录 A：数据与脚本

| 路径 | 内容 |
|---|---|
| `raw/raw_pure_unified.jsonl` | 形态 A（755 点）|
| `raw/raw_formb_v2.jsonl` | 形态 B（722 点）|
| `raw/raw_simd_npow2_A.jsonl` / `_B.jsonl` | SIMD 非 2 幂（268 / 136 点）|
| `raw/formb_v2_health.log` | 采集期间 7 次健康锚点 |
| `fitted/SIMD_CONTIGUOUS.json` | SIMD 两条公式（覆盖 2 幂与非 2 幂）|
| `fitted/FINAL_SPEC.json` | SIMT A 的 ③⑤ |
| `fitted/FORMB_FINAL.json` | SIMT B 的 4 + 6 段（当前采用，尚未接入 assemble_spec）|
| `fitted/archive/FORMB_V2_PHYSICAL.json` | SIMT B 的旧 4 + 8 段（缺 int8 大块覆盖，已归档）|
| `fitted/SIMD_NPOW2_FIX.json` | SIMD 偏块修正的早期单项拟合（已被 SIMD_CONTIGUOUS 取代）|
| `scripts/probe_block_pure.py` | 形态 A 探针 |
| `scripts/probe_mask_kind.py` | 形态 B 探针（`--dynamic 1`）|
| `scripts/probe_speed.py` | 卡健康检查（90~110 ns/iter 为健康）|
| `scripts/run_formb_v2_sweep.sh` | 形态 B 722 点扫描 |
| `scripts/fit_formb_physical.py` | 形态 B 拟合 |
| `scripts/probe_tile2d.py` | 二维验证探针（同时测 2D 与 1D）|
| `raw/raw_tile2d_verify.jsonl` | 二维验证数据（240 条 / 80 配置）|
| `raw/raw_bimodal_addr.jsonl` | 双峰溯源：15 组不同地址偏移的重复测量 |
| `raw/raw_stability.jsonl` | 稳定性检验：3 档 × 6 轮 |
| `scripts/probe_bimodal.py` | 双峰溯源探针（记录设备指针与耗时）|
| `fitted/README.md` | fitted 目录的状态登记表（当前/已归档）|
