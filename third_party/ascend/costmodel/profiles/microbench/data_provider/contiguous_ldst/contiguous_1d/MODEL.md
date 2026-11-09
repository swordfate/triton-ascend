# 连续访存成本模型（`stride == 1`）

时间单位：cycle（1 µs = 1000 cycle）。旧版记录见 [`_archive/_contiguous_summary_v1.md`](_archive/_contiguous_summary_v1.md)。

适用范围：**连续访存**（`stride == 1`），一维 `(e,)` 与二维 `(R,C)`（行跨距 = 列数）通用——
二维直接展平、按 `n = R·C` 代入即可，见 §5.1。

---

## 一、两种输入形态

分类依据：**TTIR 里有没有 `mask`**。

### 1.1 形态 A：无掩码

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

### 1.2 形态 B：带掩码

[01-vector-add.py:51-61](third_party/ascend/unittest/autotune_ut/01-vector-add.py#L51-L61)：

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements                     # n_elements 是运行期参数
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(output_ptr + offsets, x + y, mask=mask)
```

TTIR 里是 `tt.splat %n_elements` → `cmpi slt`，**有效元素数不可知**。
CostModel 能拿到：块宽、dtype、`W`、方向、**「有掩码」这一位**。走 §3.3。

### 1.3 两形态对照

| | 形态 A | 形态 B |
|---|---|---|
| TTIR | 无 `mask` | 有 `mask`，有效数不可知 |
| 公式 | §3.2 | §3.3 |
| 数据 | `raw/raw_pure_unified.jsonl`（755 点）<br>+ `raw/raw_simd_npow2_A.jsonl`（268 点，SIMD 非 2 幂）| `raw/raw_formb_v2.jsonl`（722 点）<br>+ `raw/raw_simd_npow2_B.jsonl`（136 点，SIMD）|

> **掩码按块宽计价**。实测 634 对同块宽样本：有效元素数减半只省 13%（比值中位 0.87、p90 1.01），
> 所以块宽是一阶正确变量，再细分有效数没有收益。

---

## 二、标定数据

### 2.1 测量方法

- **k-marginal**：同一 kernel 跑 `iters ∈ {2048, 8192}` 两次，`cost = (t(8192) − t(2048)) / 6144`；
- `grid = 1`，每程序在 `k_low` 个块的窗口内轮转；
- 单次 `spread > 5%` 丢弃；
- **整批采集**：同一轮标定的所有点必须在一次连续作业里跑完（见 §2.3 的跨时段漂移）；
- **每次采集前必须跑 `scripts/probe_speed.py`**：本例实测 65.6~66.3 ns/iter 为健康，>150 不可用。
  降级卡能让数据整体慢 3~20 倍，而单次测量内部完全自洽，事后无法识别。

### 2.2 总体测试范围

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

### 2.3 数据质量

| 项 | 值 |
|---|---|
| 形态 B 采集时的卡健康值 | 65.6~66.3 ns/iter，**7 次锚点波动 ±0.5%** |
| spread 中位 | 形态 B **0.14%**；SIMD 非 2 幂 **0.026%（A）/ 0.035%（B）** |
| 失败点 | 形态 B 6 点、SIMD 非 2 幂 3 点，全是寄存器/UB 溢出到无法编译的极端组合 |
| **跨时段漂移** | 同一配置在同一时段内**极稳**（B=2048 四次全 104.3；B=4096 四次 149.1~149.2；B=8192 四次 271.2~271.7，波动 ≤0.2%），但**绝对水平会随卡的状态漂移**：B=8192 在两小时内曾从 ~329 变到 ~272（−17%），B=4096 也出现过 149 / 194（+30%）交替。**所以必须整批采集、批内比较，绝不能跨天拼接数据**；§2.1 的健康锚点只能发现严重降级，发现不了这种 20~30% 的漂移 |

---

## 三、公式

### 3.1 记号

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

### 3.2 形态 A 的公式

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

#### SIMT store（4 段，324 点，**8.47%**）

| 段 | n | 公式 |
|---|---:|---|
| `te<1` | 189 | `55.76 + 1.346·W − 21.95·[n<32] + 1.035·[n<32]·W − 10.24·ln(1+W)` |
| `te>=1,int8` | 28 | `12.67 + 0.1611·K1 + 0.01761·K2 − 0.08649·bytes·ln(1+te) − 6.069·W + 17.48·lines` |
| `te>=1,wide,W<=4` | 59 | `28.79 + 0.01172·bytes·ln(1+te) + 0.009181·bytes·ln(1+W) − 0.01882·bytes − 0.004040·K2` |
| `te>=1,wide,W>4` | 48 | `8.92 + 0.008493·K1 + 2.570·W − 0.01761·bytes + 0.004675·bytes·ln(1+te)` |

### 3.3 形态 B 的公式

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

### 3.4 汇总

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

## 四、物理解释

### 4.1 有硬件出处的量

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

### 4.2 四条实测物理规律

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

### 4.3 非 2 幂块宽的底层机制

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

### 4.4 纯拟合项

| 项 | 出现在 | 说明 |
|---|---|---|
| `K1 = bytes·kof(lines/W)` | SIMT A ③⑤ | 把「每 warp 的 line 数」压成 1/2/4 三档，分档数本身是拟合的 |
| `K2 = bytes·2^min(2,⌊log₂(te/16)⌋)` | SIMT A ⑤ | 同上，按 `te/16` 分档 |
| `W²` / `ln(1+W)` | SIMT A ③⑤ | 地板随 `W` 的增长形式，二次/对数都只是拟合 |
| `lines·W`、`LDG·⌈W/4⌉` | SIMT B | 交互项，有量纲意义但系数无独立出处 |

### 4.5 一条必须写明的偏差

SIMD 的掩码**不是无条件免费**——取决于是否真的掩掉了东西：

| KEEP（块宽 4096）| 4096（无掩码）| 4095 | 4064 | 3072 | 2048 |
|---|---:|---:|---:|---:|---:|
| cost | **2682** | 6338 | 6420 | 5990 | 5343 |

只掩 1 个元素就已 2.36 倍——**掩码是开关，不是比例**。
§3.3 说「动态 mask 免费」是因为那次实验里 `M = B`（实际全 true），开关没触发；
真实 kernel 的**最后一块**是部分 true，会触发 → 按全 true 计价会**低估 2.3 倍**，按 `1/N_block` 摊薄。

## 五、适用条件

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

### 5.1 二维 tile 直接展平，套用一维公式

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

## 六、已知弱点

| 项 | 数据量 | MAPE | 说明 |
|---|---:|---:|---|
| SIMD load int64 | 32 | 12.6% | 误差集中在最大块（16384 元素 = 128 KB）|
| SIMT B load `l>32_nospill` | 50 | 17.9% | 段内 `W` 跨 1~64、成本跨 50 倍。只用物理量重搜（4~6 项）最好 17.9%/16.4%（后者 LOO-CV 反而更差），**已到物理极限** |
| SIMT B store `narrow, 8<lines≤32` | 26 | 18.0% | 最弱段；该档只有 26 点（块宽必须 2 的幂，`lines` 取值稀疏）。改用 `schedR/fill/⌈n/32⌉/(lines/⌈W/4⌉)` 四个物理量后从 22.4% 降到 18.0% |
| SIMT A ③ 段内 `W≤4` 的子片 | 8 | 25.0% | 该段总共 19 点，`W≤4` 只占 8 点，点数太少 |

**族级 MAPE 都在 10% 以内**：SIMD 6.97% / 5.24%、SIMT A 7.73% / 8.47%、SIMT B 9.47% / 9.03%。
段不要求各自 <10%。

### 试过但否定的方向

| 方向 | 结果 |
|---|---|
| 段内用 `max(串行项, 带宽项)` 瓶颈形式 | load 35%、store 38~47%，远差于线性求和 |
| 不分段单条公式 | load 15~18%、store 22~24% |
| 3~5 段替代 6 段 | 11.6~14.3%，超过 10% |
| 用 `LDG·eb`、`te·eb²` 等 dtype 缩放特征救弱段 | 17.92% vs 18.04%，无改善 |

---

## 附：数据与脚本

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

