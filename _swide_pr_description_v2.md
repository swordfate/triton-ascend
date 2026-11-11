# [costmodel](feat) add StridedWide direct load/store costmodel

> 本 PR 交付：StridedWide 直接访存的 **30 个闭式公式**（4 个 SIMD 共享槽位 + 10 个 SIMT 槽位 +
> 16 个 masked SIMT 槽位），以及配套 profile、C++ 选择/认领逻辑、单元测试和整核守护用例。

---

## 一、背景

### 1.1 现有 StridedWide 曲线的使用范围窄

`StridedWide` 指这样一类访问：**rank-2、恰有一个 unit-stride 的内层轴、行内连续段
`run >= 32 B`、行距 `pitch > run`（行与行之间有 gap）、inner/row-major、静态仿射地址
（strides 是编译期常量）**。

现有代码对 StridedWide 只有 `layout_memory` 指纹曲线可用，profile 里一共只有 **6 条 key**，
全部是同一组几何：

`masked=1 / fp32 / 16×16 / row 64 B / stride 8192 B / warps 4,8,16 / factor 1,2,4`

它只覆盖 fp32、rows=16、run=64 B、gap=8128 B 这一种形状，warps 只覆盖 4/8/16；
int8/fp16、rows=2/4/8/64、run=32–2048 B、gap=2–8192 B、无 mask、warps 1–64 都不在曲线内，
换一个 run/gap/dtype 就查不到。**这族访存没有按自身几何做闭式标定。**

### 1.2 本 PR 扩展了什么

新增 **30 个闭式槽位**，覆盖范围明确为：

- `{SIMD, SIMT} × {load, store} × {无 mask, 动态全 true mask}` 全覆盖；
- SIMD：rows 2–256；SIMT：rows 2 / 8 / 64；masked 另含 rows=16；
- run（行内连续字节）32–2048 B；SIMT load 限 run ≤ 128 B；
- gap 2–8192 B；rows=16 且 run ≤ 128 B 时，SIMD（有/无 mask）与 masked SIMT 扩展到 16 KB 行距；
- dtype int8 / float16 / float32；SIMT warps 1–64；
- 每次迭代只认领一条不间断流（`count == 1`）。

槽位选择是访问几何的纯函数（秩、unit-stride 轴、run、gap、dtype 宽度、rows、warp 数、mask
状态），**不读算子名**；run/gap 在实测端点闭区间内按公式插值，端点外不认领。

实现落点：`StridedWideCostModel.{h,cpp}`（30 个槽位、特征、选择与求值）、
`StageCostModels.cpp` 的 `mapStridedWideMemoryCompletion`（认领顺序在 `layout_memory` 分组曲线之前，
认领后 `storeLayoutService` 让位，诊断字段 `swide_load/store_completion_per_iteration`）、
profile 的 `stage_resources.{simd,simt}.memory.strided_wide_completion`。

---

## 二、标定公式汇总与拟合误差

### 2.1 标定方法与配置

| 项 | 做法 |
|---|---|
| 被测对象 | 单次 StridedWide load/store（rotate-loop 内核，循环体里只放这一次访问） |
| 量什么 | K-marginal 斜率：每轮做 `K` 次该访问，扫 `K = 1/2/4/8`，取 `dT/dK` |
| 计时器 | profiler 的 kernel duration（`kernel_duration_average_us`） |
| 拟合 | 加权最小二乘，权重 `1/T`；每个槽位独立拟合，不外推、不共享系数 |
| 落点统计 | 4601 个标定点（SIMD 1145 + SIMT unmasked 1051 + masked 2405）；16 KB 扩展点并入 masked rows=16 重拟合 |

测试范围：

| 项 | 范围 |
|---|---|
| dtype | int8 / float16 / float32（`eb` = 1 / 2 / 4 B） |
| 访问形态 | rank-2、inner/row-major、恰有一个 unit-stride 内层轴；行内连续 `run`，行距 `pitch = run + gap` |
| run（行内连续字节） | 32 / 64 / 128 / 256 / 512 / 1024 / 2048 B（SIMT load ≤ 128 B） |
| gap（行距空洞） | 2 / 32 / 64 / 128 / 256 / 512 / 1024 / 2048 / 8192 B；rows=16 且 run ≤ 128 B 时，SIMD（有/无 mask）与 masked SIMT 另含 16256 / 16320 / 16384 B |
| rows | SIMD 2–256；SIMT 2 / 8 / 64；masked 另含 16 |
| num_warps | 1 / 2 / 4 / 8 / 16 / 32 / 64（仅 SIMT） |
| compile_mode | simd / simt |
| mask | 无 mask + 动态全 true mask（`activeElementRatio == 1.0`） |
| 访问条数 | 每次迭代一条不间断流（`count == 1`） |
| 额外验证点 | SIMD 16 KB 行距：masked / unmasked × store / load 各 27 点（不参与拟合） |

### 2.2 公式列表与拟合误差

表中「估计 cost」为槽位内逐点预测的均值、「平均实测」为逐点实测均值（单位 µs），
「平均相对误差」为逐点 `|pred/meas-1|` 的均值（按点数加权）。为控制篇幅：

- 按 6 个模式/掩码大类汇总；公式栏用**文字说明 rows 范围 / W 分段和每段项数**，不展开符号，
  完整特征项与系数见 profile；
- SIMD 的 masked 与 unmasked 共用槽位；SIMT 的 masked 用独立槽位。

合并/外推拟合实验：把 rows=16 与 rows=64 合成一条后，masked 总误差 10.7–13.0%，不达标；
把 rows 作为连续因子（n / nbytes / span / bytes_per_lane / seg_per_warp / rows_over_W /
rows_x_iters 等 + L/mx/S 地址集统计）拟合，in-sample 仍有 14.6–21.4%，**留一行验证 50–170%**，
说明 rows 不满足外推条件；rows=2 与 rows=8 合并后 unmasked load 总误差 10.09%（比保留两段的
9.64% 更差）。补充的 rows=4/16/32 pilot（72 点）也证实：rows=4 可由 rows2/8 插值
（store −3%~−5%），但 rows=16/32 用相邻锚点插值误差 +20%~+130%，row 方向不是平滑趋势。

项数裁剪：交付版按分组裁剪项数——SIMD load 每段 6 项、SIMD store 13/16 项、
SIMT unmask load 5–14 项、store 6–8 项、mask load 4–10 项、mask store 4–8 项；
模式级 MAPE 仍全部 <10%。

| # | 场景 | 公式 | 拟合case数 | 估计 cost (us) | 平均实测 (us) | 平均相对误差 | 拟合误差<10% |
|---|---|---|---:|---:|---:|---:|---|
| 1 | SIMD load（masked + unmasked 共用槽位） | 按 rows≤4 / rows>4 两段；每段一条线性公式（各 6 项） | 551 | 0.1706 | 0.1746 | 6.44% | PASS |
| 2 | SIMD store（masked + unmasked 共用槽位） | 按 rows≤2 / rows>2 两段；每段一条线性公式（13 / 16 项） | 594 | 0.2622 | 0.2707 | 8.16% | PASS |
| 3 | SIMT unmask load | rows=2、rows=8 各一条；rows=64 再按 W≤2 / 3≤W≤8 / W>8 三档（共 5 条，5–14 项） | 463 | 0.2331 | 0.2641 | 9.64% | PASS |
| 4 | SIMT unmask store | rows=2、rows=8 各一条；rows=64 再按 W 三档（共 5 条，6–8 项） | 588 | 0.0903 | 0.1023 | 9.52% | PASS |
| 5 | SIMT mask load | rows=2、rows=8 各一条；rows=16、rows=64 再按 W 三档（共 8 条，4–10 项） | 956 | 0.2389 | 0.2798 | 9.36% | PASS |
| 6 | SIMT mask store | rows=2、rows=8 各一条；rows=16、rows=64 再按 W 三档（共 8 条，4–8 项） | 1449 | 0.0847 | 0.0913 | 9.73% | PASS |
| — | **合计 / 加权** | 30 个槽位 | 4601 | 0.1656 | 0.1824 | 9.02% | PASS |

注：rows 方向经合并 / 连续因子拟合以及 rows=4/16/32 pilot 验证均**不可外推**
（留一行验证 50–170%），因此只按实测锚点 2 / 8 / 16 / 64 分段；row 区间化列为遗留问题（§四.4）。

### 2.3 适用条件（不满足则不认领）

必须同时满足：

1. 布局类 = `StridedWide`（rank-2、恰有一个 unit-stride 内层轴、inner/row-major、静态仿射地址）；
2. `run` 32–2048 B；`gap` 在所属槽位的实测端点内（常规 2–8192 B；rows=16 且 run ≤ 128 B 时 2–16384 B）；
3. dtype 字节数 ∈ {1, 2, 4}；span ≤ 512 KB；
4. mask 只接受动态全 true（`activeElementRatio == 1.0`）；静态可证明的部分 mask 不认领；
5. 每次迭代一条不间断流（`count == 1`）；
6. `rows` / `W` 落在对应槽位的分段内（SIMD rows 2–256；SIMT rows 2/8/64，masked 另含 16）。

---

## 三、回归验证

### 3.1 100-case 中受影响的 case

100-case 中 92 例与本 PR 前逐位一致；以下 8 例为 `solve_tril_*`（4 例 h32 + 4 例 h64），
全部由本机制接管，误差均在 10% 内：

| case | 实测(us) | 本PR前预测(us) | 本PR前误差 | 本PR后预测(us) | 本PR后误差 | 合入本PR后是否超过10% |
|---|---:|---:|---:|---:|---:|---|
| `solve_tril_b16_t1024_h64_bt64` | 5797.191 | 5885.0140 | +1.51% | 5840.1740 | +0.74% | 否 |
| `solve_tril_b16_t8192_h64_bt64` | 46327.622 | 47550.8530 | +2.64% | 47192.5940 | +1.87% | 否 |
| `solve_tril_b1_t1024_h32_bt64` | 189.258 | 178.8940 | -5.48% | 177.5050 | -6.21% | 否 |
| `solve_tril_b1_t1024_h64_bt64` | 335.352 | 324.1410 | -3.34% | 321.3090 | -4.19% | 否 |
| `solve_tril_b1_t8192_h64_bt64` | 2914.962 | 2899.8160 | -0.52% | 2877.3960 | -1.29% | 否 |
| `solve_tril_b4_t131072_h32_bt64` | 92658.205 | 95172.2100 | +2.71% | 94488.1750 | +1.97% | 否 |
| `solve_tril_b8_t1024_h32_bt64` | 1466.227 | 1424.0710 | -2.88% | 1413.3320 | -3.61% | 否 |
| `solve_tril_b8_t8192_h32_bt64` | 11585.062 | 11840.5310 | +2.21% | 11754.9810 | +1.47% | 否 |

### 3.2 整核守护用例（StridedWide 几何，16 例）

从标定配置池（rows/run/gap/mask）构造单流 copy 整核：8 条 `simd`（costmodel auto 选路即
`simd`，实测/预测取同一次 auto 运行）+ 8 条强制 `simt_only`（这批单流探针的 auto 选路始终是
simd，故强制该路线，F 见 mode 列）。`simt_only` 的整核预测 = auto route JSON 中该 factor 的
`all_simt_factor_routes[F].total_system_cycles`（每个 stage 走 simt 的资源价）+ 同一 route 的
边界/global-load 额外项。**16/16 整核误差 <10%**（0.33%–9.43%）：

| case | 强制mode | 实测 | 本PR后预测 | 预测误差 | 整核误差是否<10% |
|---|---|---:|---:|---:|---|
| `swide_m_r2_run128_gap128` | simd | 4.091 | 4.078 | -0.33% | 是 |
| `swide_m_r16_run64_gap8192` | simd | 3.206 | 2.920 | -8.94% | 是 |
| `swide_m_r2_run128_gap512` | simd | 4.036 | 3.903 | -3.29% | 是 |
| `swide_m_r2_run128_gap8192` | simd | 3.986 | 3.901 | -2.13% | 是 |
| `swide_m_r8_run128_gap2048` | simd | 3.355 | 3.239 | -3.46% | 是 |
| `swide_u_r16_run128_gap16384` | simd | 3.166 | 2.867 | -9.43% | 是 |
| `swide_u_r32_run128_gap128` | simd | 3.511 | 3.230 | -8.01% | 是 |
| `swide_u_r8_run128_gap128` | simd | 3.823 | 3.466 | -9.35% | 是 |
| `swide_m_r64_run128_gap2048` | simt_only (F1) | 10.812 | 11.311 | +4.61% | 是 |
| `swide_m_r64_run128_gap8192` | simt_only (F1) | 10.734 | 11.569 | +7.78% | 是 |
| `swide_u_r64_run128_gap2048` | simt_only (F1) | 10.116 | 9.970 | -1.45% | 是 |
| `swide_u_r16_run64_gap32` | simt_only (F1) | 3.683 | 3.718 | +0.94% | 是 |
| `swide_u_r2_run128_gap32` | simt_only (F2) | 5.618 | 5.461 | -2.79% | 是 |
| `swide_u_r2_run128_gap512` | simt_only (F2) | 5.530 | 5.461 | -1.24% | 是 |
| `swide_m_r16_run64_gap128` | simt_only (F1) | 4.202 | 4.130 | -1.72% | 是 |
| `swide_m_r2_run128_gap512` | simt_only (F2) | 5.563 | 5.862 | +5.37% | 是 |

## 四、遗留问题

1. **已知弱段**：SIMT `rows=8`（load 13.1%）、`rows=64, W≤2`（load 18.6%、store 11.5%）、
   `rows=64, 3≤W≤8`（10–13%），masked `rows=16, W≤2` store 15.0%；模式级加权仍 <10%
   （load 9.64%、store 9.52%）。`run=32`（duty≈50%）与 `gap=2` 的边界点单点误差可达 ±40–75%。
2. **不在本机制认领范围**：静态可证明的部分 mask；outer/列主序（从未标定）；`count>1` 的多流访问
   （标定的是单流稳态单价）。
3. **探索性整核观测**：非选中 SIMT 路线的整核误差可达 −29%~+230%，以及 factor 合法性搜索缺口，
   属既有模型/后续项，与本机制的认领逻辑无关。
4. **row 区间化（后续）**：本 PR 只按实测锚点 rows=2/8/16/64 分段；pilot 显示 rows=4 可插值，
但 rows=16/32 用相邻锚点误差 +20%~+130%，row 方向非平滑。要支持中间 rows 需逐行补测后单独
拟合，不能直接外推。
