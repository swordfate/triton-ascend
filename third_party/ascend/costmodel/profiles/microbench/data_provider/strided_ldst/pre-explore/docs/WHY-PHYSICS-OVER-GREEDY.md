# 为什么最终放弃数据驱动的贪心拟合，改用物理第一性建模

> 面向读者：接手 strided load/store CostModel 标定、或需要判断"该不该上自动特征选择"的人。
> 目录：`third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/docs/`。
> 本文所有数字都来自仓库里现存的实测产物（`../results/final/MODEL_*.json`、`PHYS*.json`、`REPORT_*.md`），
> 逐条给出**文件 + 字段名**，可逐条复核（索引见 §7）；少量只能间接推断的地方会写明"推断"。

---

## 0. 一句话结论

贪心/自动特征选择路线**没有失败在"精度不够"**：它把总体 in-sample MAPE 做到了 9.91%，
甚至能在单个公式上用 100 多项把误差压到 4.38%。它失败在**三个不可修复的地方**：

1. **它选出来的量是错的**：它的主力自变量之一是 128 B cache line，而这个 128 B 是**我们自己设的假设**
   （写死在 `scripts/build_v4_numwarps_dataset.py:32` 的 `LINE_BYTES = 128`），对 MTE2 路径实测不成立。
2. **它在正确的物理自变量上完全看不见**：跳变的真实自变量是 `n * min(R, 32)`，
   而为这类跳变准备的 **15 种惩罚项在第一轮全被压到 0**（因为用的是 R 或 C 的一次项）。
3. **它无法解释、无法迁移**：项是"基特征 x regime 指示"的机械乘积，项数按公式从 6 到 113 任意分化，
   同一个量在不同公式里含义不同；换一颗芯片必须从头重跑，学不到任何硬件事实。

物理第一性路线用 **10 个公式、每个 3-8 项、全部 < 10%、加权 7.22%** 达到目标，
并且顺带**标定出了 6 个可写进 CostModel 的硬件量**（32 B sector、64 B burst、256 B 冲突粒度、
1 KB 观察窗口、`k ≈ 2.18 ns/warp` 的 SIMT 发射瓶颈、`n*min(R,32)` 分段发射量）。
这些量是贪心路线永远给不出来的。

---

## 1. 三条路线正面对比

| 维度 | A 数据驱动贪心 | B <=10 交叉项 + 剔除规则 | C 物理第一性（最终采用） |
|---|---|---|---|
| 代表文件 | `MODEL_MODSEL.json`、`MODEL_REGIME_V2/V3.json` | `MODEL_10CROSSED_DOMAIN.json` | `PHYS3_*.json` + `PHYS2_wide_simtload.json` + `PHYS2_simt_store.json` |
| 公式形式 | `T = Σ cᵢ · 基特征 · [修饰因子]`；修饰因子 = dtype 指示 / num_warps 指示 / 前向选出的 regime 标志（`C<8`、`n>=256`、`L>16` …） | 每项是"变量 x 变量"的乘积，贪心选 <=10 项，外加 domain 剔除规则 | `T = 常数 + Σ cᵢ · (物理量)`，允许 `max()` 瓶颈结构；不用 log/指数/幂次 |
| 每个公式的项数 | **6 - 113**（MODSEL 实测）；堆到 regime 14 标志时 **27 - 209**；15 标志时 **28 - 217** | 9 - 11 | **3 - 8** |
| 总体 in-sample MAPE | **11.29%**（MODSEL）；堆到 209 项 **9.91%**；217 项 **9.80%** | **15.58%**（`_overall_kept`，且要靠剔除数据） | **7.22%**（10 式加权，5118 个保留实测点，逐式全部 < 10%） |
| 需要剔除数据才能成立 | 不需要（但它靠 regime 标志代替剔除） | **需要**（`exclusion_rule` + `coverage` + `mape_kept`） | 需要，但每条门槛都有物理理由，且**逐条实测过"加了没改善就不加"** |
| 每一项能否解释 | 不能 | 不能 | 能（项名即物理量：搬运字节、BIU 命令数、32 B sector 门槛、行 descriptor…） |
| 能否迁移到新硬件 | 不能（项名与系数都是数据拟合出来的） | 不能 | 能（换硬件只需重标定 6 个硬件量） |
| 是否学到硬件事实 | 否 | 否 | **是（6 个，见 §4）** |

---

## 2. 贪心路线的完整成绩单（先把它的好话说清楚）

### 2.1 它确实把总体误差压到了 10% 附近

| 模型 | 每式项数（实测） | 总 in-sample MAPE（文件自带 `_overall`） |
|---|---:|---:|
| `MODEL_MODSEL.json`（dtype / num_warps / regime 指示逐位选型） | 6 - 113 | **11.29%** |
| `MODEL_REGIME_FINAL.json`（6 个 regime 标志） | 109 | 11.27% |
| `MODEL_REGIME_V2.json`（**14 个** regime 标志，`_regime_flags` 实测 14 个） | 27 - 209 | **9.91%** |
| `MODEL_REGIME_V3.json`（**15 个** regime 标志） | 28 - 217 | **9.80%** |
| `MODEL_PRUNED.json`（剪枝） | 13 - 97 | 10.87% |
| `MODEL_10CROSSED.json`（<=10 交叉项） | 9 - 11 | 16.57% |
| `MODEL_10CROSSED_DOMAIN.json`（<=10 交叉项 + 剔除规则） | 9 - 11 | **15.58%**（`_overall_kept`） |

**"项数换误差"的曲线是连续可用的**：`PRUNE_CURVE.json` 的 `overall` 实测
4 项 34.23% -> 8 项 31.9% -> 12 项 28.33% -> 16 项 25.14% -> 24 项 20.0% -> 32 项 17.36% -> 
48 项 14.79% -> 64 项 12.53% -> **96 项 10.86%**。几乎没有平台期 —— 这本身就是"过拟合式下降"的指纹：
如果每多一项都在抓真实结构，曲线应该出现明显的肘部然后走平。

### 2.2 逐 slot 对照：物理式在 7/10 个 slot 上更好，而且项数少一到两个数量级

下表取自 `REPORT_v5_physics_scope.md` 的汇总表（"贪心对照"列），实测值：

| slot | 物理式项数 | 物理式 MAPE | 贪心项数 | 贪心 MAPE | 谁更好 |
|---|---:|---:|---:|---:|---|
| `simd_load_wide_r1` | 3 | **1.22%** | 6 | 3.46% | 物理 |
| `simd_load_wide_r2` | 7 | **4.48%** | 17 | 17.10% | 物理（项数 1/2.4，误差 1/3.8） |
| `simd_load_gather_r1` | 6 | **8.70%** | 78 | 12.14% | 物理（项数 1/13） |
| `simd_load_gather_r2` | 7 | **8.44%** | 73 | 11.08% | 物理（项数 1/10） |
| `simt_load_r1` | 7 | **4.95%** | 89 | 7.50% | 物理（项数 1/13） |
| `simt_load_r2` | 6 | **6.34%** | 81 | 7.24% | 物理（项数 1/13） |
| `simt_store_r2` | 8 | **9.64%** | 49 | 17.75% | 物理 |
| `simd_store_r1` | 5 | 5.33% | 105 | **4.38%** | 贪心（多 21 倍项数换 0.95 个点） |
| `simd_store_r2` | 4 | 8.12% | 113 | **5.63%** | 贪心（多 28 倍项数换 2.49 个点） |
| `simt_store_r1` | 7 | 6.27% | 113 | **5.47%** | 贪心（多 16 倍项数换 0.80 个点） |

> **必须说明的口径差异**：这一列的贪心 MAPE 是在**该 slot 的全量点**上算的，物理 MAPE 是在**物理适用范围**内算的
> （保留率 38%-100%）。所以这张表**不是严格同 scope 比较**，它只能说明"量级和项数"的差异。
> 严格同 scope 的比较见 §2.3。

### 2.3 严格同 scope 的比较（这才是公平的胜负）

物理重构的记录文件里留下了同一批点、同一套候选池下两边的成绩，逐条实测：

| 槽 / 数据子集 | 贪心（同 scope） | 物理式（同 scope） | 来源 |
|---|---|---|---|
| `simd_load_gather_r2`，438 点 | 8 项 **8.94%**（CV 9.16%） | **7 项 8.44%**（CV 8.70%） | `PHYS3_simd_load_gather_r2.json` -> `comparison_128b_removal.e_reference_unconstrained_greedy` |
| `simd_load_wide_r2`，120 点 | 76 特征贪心 17 项 **12.39%** | **7 项 4.48%** | `PHYS3_simd_load_wide_r2.json` -> `comparison.our_greedy_76feat_17_terms` / `this_model_MTE_only` |
| `simd_load_gather_r2`（第 1 轮口径），678 点 | - | 8 项 13.18% | `round1_reference`（第 1 轮物理式） |
| `simd_store_r2`，325 点，严格禁止 128 B 量 | 允许 128 B 的贪心式 8 项 **9.52%** | 全 MTE 重构 **7 项 8.44%**（同槽） | `PHYS3_simd_load_gather_r2.json` -> `a_original_128B` vs `d_mte_restructured_all_positive` |
| `simd_store_r2`，325 点，严格 128-free 上限 | 9.5 - 10.5%（9 项 / 6 项） | 现状 8.62%（第 2 轮）/ **8.12%**（最终） | `PHYS3_simd_store_r2.json` -> `strict_128_free_alternatives.conclusion` |
| `simd_store_r2`，325 点，粒度消融 | `1+nbytes`（2 项）54.52%；`1+nbytes+mx64+S64`（64 B burst）18.42%；`mx32+S32` 23.65%；`mx256+S256` 21.22% | 最终 4 项 **8.12%** | `PHYS3_simd_store_r2.json` -> `ablation_128b_removed.ladder` |

**结论**：在"正确的物理自变量"上（MTE 的 32 B / 64 B / 256 B / 1 KB），物理式用**更少的项**取得**更低的误差**，
并且交叉验证误差与 in-sample 几乎相同（8.44% vs CV 8.70%；8.12% vs CV 8.24%），说明它没有靠记数据。
贪心唯一赢的地方是 store 槽的 in-sample MAPE，代价是 20-30 倍的项数。

---

## 3. 五个致命问题（为什么最终不用它）

### 3.1 它选中的 128 B 量是**我们自己的错误假设**

这是最硬的一条。128 B 从来不是实测出来的硬件事实，而是写死在特征生成脚本里的常量：

```python
# scripts/build_v4_numwarps_dataset.py 第 32 / 35 行（原文）
LINE_BYTES = 128          # <- 我方自己设的 cache line 假设
LANES_PER_WARP = 32
ELEM_BYTES = 4
PAGE_BYTES = 4096         # <- 同样是 CPU 概念，不是这颗芯片的实测粒度
```

贪心路线把这个 128 B 当作核心自变量：它的基特征里直接出现 `minL128`、`le_L128xL`、`le_L128xN`、
`line_elems_max`（每条 128 B line 内最多元素数）等 128 B 量（实测于 `MODEL_MODSEL.json` 的 `base` 列表：
`simt_load_r1` 用 `minL128`/`minL64`/`minL32`，`simt_load_r2` 用 `le_L128xL`，`simt_store_r1`/`simt_store_r2`/`simd_store_r1` 用 `line_elems_max`）。

后来对 MTE2 路径的独立扫描给出了完全不同的粒度：

| 量 | 实测结论 | 证据 |
|---|---|---|
| 搬运粒度（MTE burst） | **64 B**（= 2 个 32 B sector） | `PHYS3_simd_load_gather_r2.json` -> `burst_grain_scan`：三个独立扫描都在 B=64 取最小（27.68% / 14.29% / 14.17%）；`conclusion` 原文："三个独立扫描都在 B=64 取最小；64B = 2 个 32B memory sector，与已验证的 32B sector 硬门槛自洽" |
| 冲突粒度 | **256 B**（不是 128 B） | `PHYSICS_METHOD.md` §4："扫描 B∈{16..4096} 得最优 B=256"；`burst_grain_scan.cross_row_share_grain`："64B 是搬运单位，256B 是冲突单位" |
| 观察窗口 | **1 KB** | `PHYS3_simd_load_wide_r2.json` -> `burst_scan_branchA_window`：256->6.34%、512->4.27%、**1024->3.19%**、2048->5.91% |
| 128 B | **SIMT/LSU 路径的 cacheline，不属于 MTE** | `burst_grain_scan.note` 原文："扫描中显式跳过 B=128：128B 是 SIMT/LSU 路径的 cacheline，不属于 MTE" |

**最能说明问题的是这一段对照**（`PHYS3_simd_load_gather_r2.json` -> `comparison_128b_removal`，同一批 438 点）：

| 模型 | 项数 | MAPE | 关键事实 |
|---|---:|---:|---|
| (a) 原来的 128 B 版 | 8 | 9.52% | 8 项里有 **4 项依赖 128 B**（`pairs128`、`multi128`、`wrow128`，`share1024` 也按 128 B 的 8 倍粒度定义） |
| (b) 只把粒度换成 64 B，结构不动 | 8 | 13.22% | 恶化 3.7 个点；`pairs64`、`share64` 出现**负系数** —— 说明原结构是围着 128 B 量搭起来的，换粒度后结构本身失效 |
| (c) 每族各自最优 MTE 粒度 | 8 | 11.08% | 仍有负系数 |
| (d) 按 MTE 物理重构（全系数非负） | **7** | **8.44%** | **比原 128 B 公式少 1 项、误差低 1.08 个点，且无负系数** |

也就是说：128 B **不是"差不多能用"的量，而是把整个模型结构带偏的量**。
一个"自动特征选择"系统选出了一个由我方错误假设制造出来的自变量，并围绕它搭出了带负系数的结构 —— 
这正是"没有物理约束的拟合无法自我纠错"的定义。

### 3.2 regime 标志是前向选出来的：同数据选特征、又同数据拟合

`MODEL_MODSEL` / `MODEL_REGIME_*` 的形式是

```
T = Σ cᵢ · 基特征ᵢ · [修饰因子ᵢ]
修饰因子 ∈ {1, dtype 指示, num_warps 指示, regime 标志}
regime 标志例：C<8、C<16、C<32、C<64、n<64、n>=128、n>=256、R>8、R>16、L>4、L>16、L>64、mwl>2、mwl>8、ISB 小
```

这些标志**不是物理门槛，是前向搜索在同一条数据上挑出来的切分点**（`MODEL_REGIME_V2.json` 的 `_regime_flags` 实测 14 个，
`MODEL_REGIME_V3.json` 15 个）。用同一批数据既决定"在哪里切"又拟合"切完之后的系数"，
没有任何留出集或嵌套交叉验证来约束这个过程，因此**误差里必然含有一部分是记住噪声换来的**。

过拟合的直接铁证就在文件里：

| 现象 | 实测 | 来源 |
|---|---|---|
| 某个 slot 的 in-sample MAPE 精确为 **0.00%** | `simd_load_wide_r1` 在 `MODEL_REGIME.json` 里 MAPE = 0.00（91 项） | `MODEL_REGIME.json` |
| 同一个 slot 在 `MODEL_REGIME_FINAL.json` 里仍是 0.00% | 109 项 | `MODEL_REGIME_FINAL.json` |
| 项数按公式任意分化 | 6 / 17 / 78 / 73 / 105 / 113 / 89 / 81 / 113 / 49（`MODEL_MODSEL` 十个 slot 实测）；regime 版最大到 217 | `MODEL_MODSEL.json`、`MODEL_REGIME_V3.json` |
| 加项没有平台期 | 4 项 34.23% 一路降到 96 项 10.86%，几乎单调 | `PRUNE_CURVE.json` -> `overall` |

**"每个公式项数差异巨大且任意"本身就是症状**：真实硬件的代价机制在全域内是同一套，
如果模型抓到的是机制，十个 slot 的项数应该收敛到同量级；实测是 6 到 113，差 19 倍。
而物理式十个 slot 是 3-8 项，差不到 3 倍。

### 3.3 它在**正确的物理自变量**上完全看不见：`n * min(R, 32)` 与"15 种惩罚项全被压到 0"

这是压倒性的一条证据。实测（`REPORT_PHYS2_simt_store.md` 第 53 行起，n=256、ISB=16、f32、W=4）：

| (R,C) | (4,64) | (8,32) | (16,16) | (32,8) |
|---|---:|---:|---:|---:|
| 128 B line 数 `L128` | 32 | 32 | 32 | 32 |
| `lin_tot`（行内元素总段数） | 32 | 32 | 32 | 32 |
| `seg_tot`（行段总数） | 8 | 8 | 16 | 32 |
| **实测 T (ns)** | **42** | **95** | **235** | **594** |

四个形状的 `L128` 与 `lin_tot` **完全相同**，T 却差 **14 倍**（42 -> 594 ns）。
`seg_tot` 虽有 8/8/16/32 的变化，但把它当自变量（或加各种 hinge）时同样没用 —— 见下一段。

> 报告原文（`REPORT_PHYS2_simt_store.md:53`）：
> **"第一轮 15 种跳变惩罚全被压到 0，因为用的自变量是 R 或 C 的一次项。"**

也就是说：为解释这个跳变，第一轮**主动造了 15 种惩罚项**（R 的一次项、C 的一次项、各种 `min/max` 包裹、
指示器乘积），而 NNLS/WLS 把它们的系数**全部压到 0** —— 数据用"零系数"明确否决了这套自变量。
换成 `n * min(R, 32)` 之后，**单这一项**就把该子集从 26.5% 降到 14.9%（同报告）。

这是贪心路线最深层的缺陷：**它的候选池是人手写的，池子里没有 `n*min(R,32)`，它就永远找不到。**
而池子里有的（`minL128`、`seg_tot`、`bucket_*`）都是"看起来像硬件量"的统计量，不是物理量。
物理第一性方法之所以能找到，是因为它先问"一个元素按行主序落在不同行段里，每个元素是否要随行段单独发射一次"，
再据此**构造**自变量，而不是在现成特征里搜。

### 3.4 项没有物理意义 -> 无法解释、无法迁移、无法维护

看 `MODEL_MODSEL.json` 里真实选出来的项名（实测）：

```
simd_load_wide_r1 : ov_L_pages*1, ov_L_pages*smISB, ov_L_pages*e1, ov_L_pages*Lgt4, ov_L_pages*mwlg2 ...
simd_load_gather_r2: mn_L_cshortN*mwlg8, mn_L_cshortN*smISB, mn_L_cshortN*Clt32, ...
simt_load_r1     : minL64*W64, minL64*W32, minL64*Wgt1, minL64*mwlg8, minL64*Clt16, minL64*wide, ...
simt_store_r2    : mx_L_W*e4, mx_L_W*Rgt8, mx_L_W*Clt32, ...
```

这些项**没有物理过程可讲**：`minL64*W64` 不是"每 64 条 line 乘 64 个 warp 的什么代价"，
只是"基特征 x 指示变量"的机械乘积。后果是具体的：

1. **无法解释**：CostModel 的消费者（C++ / profile 侧）拿到公式后问"这一项代表什么"，答不出来。
   物理式的项名 `[C*eb<32]*L64` 可以直接回答："burst 不足区制下，每个 64 B 搬运块 37.6 ns"。
2. **无法迁移**：换一颗芯片（或同一芯片换 lowering 路径），128 B 指示、`Clt32` 标志、113 个系数全部要重跑；
   物理式换芯片只需重标定 6 个硬件量（32 B / 64 B / 256 B / 1 KB / k / sector 数）。
3. **无法维护**：同一个量在不同公式里是不同项。`line_elems_max` 在 `simd_store_r1` 是 105 项里的第 1 项，
   在 `simt_store_r2` 里根本不是主项 —— 因为它不是硬件量，只是某一批数据里的相关量。

### 3.5 误差不是噪声，是结构（而且被项数掩盖了）

- `MODEL_FINAL_V7.json`（`run_final_pipeline.py` 的贪心末端产物）在 `simt_store_r2` 上仍有 **25.98%**，
  而同一 slot 物理式在适用范围内是 **9.64%**（8 项）。
- 第 1 轮物理式对 `simt_store_r1` 报 18.95% 时，报告的第一句诊断是：
  "r1 的误差**不集中在任何子集**（dtype 16-30%、W 13-27%、ISB 4-45%），是**公式形式错**而不是数据脏"
  （`REPORT_PHYS2_simt_store.md` §1）。换成三瓶颈 `max()` 形式后降到 6.27%。
  贪心不会告诉你"形式错"，它只会继续加项把误差摊平。
- 物理重构过程中反复确认的一条：**最大的单一误差来源是测量稳定性而不是模型项**：
  `target_spread > 0.2`（同一配置重复测量相对离散度 > 20%，双稳态区）的点只占约 5%，却承载约 **24% 的总误差质量**
  （`REPORT_v5_physics_scope.md` §A）。这种"误差质量分布"的判断，只有先有物理分组才看得出来。

---

## 4. 物理第一性路线换来了什么：6 个可写进 CostModel 的硬件量

这是放弃贪心的**收益侧**。这些量全部由独立扫描或白盒证据确认，不是拟合出来的系数：

| 硬件量 | 值 | 证据（文件 / 字段） | 在哪些最终公式里出现 |
|---|---|---|---|
| **memory sector 硬门槛** | **32 B** | `PHYS_simd_load_wide_r2.json` -> `notes`：222 个点**精确**在 `src = 32 B` 处分成两族；`src < 32 B` 时"编译器无法使用连续块搬运 + extract_slice lowering，退化为逐元素 strided 命令路径，**单价高 10-30 倍**" | `simd_load_wide_r2`（门限 + `R*[rb<32]`）、`simd_load_gather_r2`（`[C*eb<32]` 双区制）、`simt_store_r2`（`C*eb>=32` 门槛） |
| **MTE burst 粒度** | **64 B**（= 2 个 32 B sector） | `PHYS3_simd_load_gather_r2.json` -> `burst_grain_scan`：三个独立扫描都最优在 B=64；`wide_r2` 的 `burst_scan_branchB` 也在 64 B 处从 16.98% 骤降到 8.52%（96 B 为 8.32%，差异 0.2 个点、33 个点，判定为同量级） | `simd_load_wide_r2`（`nblk64`，54.43 ns/burst）、`simd_load_gather_r1/r2`（`mx64`/`L64`/`wrow64`）、`simt_store_r2` 消融对照 |
| **冲突粒度** | **256 B** | `PHYSICS_METHOD.md` §4："扫描 B∈{16..4096} 得最优 B=256"；白盒 `biu_send_count = n` | `simd_load_gather_r2`（`multi256` 跨行共享、`L256` 冲突块） |
| **元素拥挤观察窗口** | **1 KB**（是观察窗口，**不是**硬件阈值） | `PHYS3_simd_load_wide_r2.json` -> `burst_scan_branchA_window`：1024 最优（3.19%）| `simd_load_wide_r2`（`rmax1k`）、`simd_load_gather_r1/r2`（`sq1024`/`w1024`）、`simt_load_r2`（`conc1k`） |
| **SIMT 双瓶颈 + 发射单价** | `T = max(k·W, 访存)`，**k ≈ 2.18 ns/warp** | `PHYS2_simt_store.json` -> `notes`："W>=16 时 T 与元素数无关（实测 W=16/32/64 -> 34/68/140 ns），k=2.18 ns/warp"；最终 `simt_store_r1` 系数 2.18151 | `simt_store_r1`、`simt_load_r1/r2`、`simt_store_r2` 的 `max()` 分支 |
| **分段发射量** | **`n · min(R, 32)`**（在 warp 宽度 32 处饱和） | `REPORT_PHYS2_simt_store.md` §3 的 (R,C) 跳变表；T 峰值在 R=32，R=64 回落到 818 < 1171 | `simt_store_r2` 的 `n*min(R,32)` 分支（单这一项把子集 26.5% -> 14.9%） |

### 4.1 诚实的两条反向记录

物理第一性不是"永远更准"，有两条必须写清楚：

**(1) 128 B 并没有被完全赶出去，它在 SIMT/LSU 路径上是对的，而且在 store 槽上以"从数据估出的 sector 数"形式回来了。**

- `simt_store_r1`/`simt_store_r2` 的 line 吞吐分支用的是 **`L128`**：
  报告实测"把 256 B 块模型搬过来重算几何后，L256 版本全量 MAPE=15.8%，L128 版本 6.3%"
  （`REPORT_PHYS2_simt_store.md` §2）。原因是 SIMT 路径走的是 LSU/DCache，128 B 是它的 line 粒度。
- `simd_store_r2`（MTE3 路径）用的写合并窗口 = **4 个 32 B sector = 128 B**，但它的得出方式是
  "对 sector 数 k 扫描，k=4 尖锐最优"（k=3 -> 11.47%、**k=4 -> 8.12%**、k=5 -> 11.64%、k=8 -> 24.49%，`window_scan`），
  文件里专门声明："**这不是把 128 藏起来**：它是从数据估计出来的 sector 数（4），单位是 32 B memory sector，
  机制是部分写 RMW 的写合并窗口，不是 SIMT LSU 的 cacheline"（`PHYS3_simd_store_r2.json` -> `derived`）。
- 并且实测了它的代价：**严格 128-free 的天花板约 9.5-10.5%**（9 项 / 6 项），
  差于现状 8.62%，也差于最终 8.12%（`strict_128_free_alternatives.conclusion`）。

**这条区别很重要**：贪心的错不在于"用了 128 这个数"，而在于**把这个数当成了预设的硬件粒度**（写死在 `LINE_BYTES = 128`），
并围绕它搭结构。物理路线的做法是：先从 32 B sector 出发定义量，再让数据决定"几个 sector"——
数字可能落在 128，但它此刻是一个**待估参数**，不是假设。

**(2) 物理式有时会主动放弃更低的 MAPE。**

`PHYS3_simd_load_gather_r1.json` -> `variants` 里，6 项物理式的最优是 **8.70%**；
而把第二拥挤尺度放到 4 KB 能到 **8.50%**（本批最低）。最终**没有采用 4 KB**，因为 4 KB 是 CPU page 概念、
不是这颗芯片的实测粒度，且 4 KB 附近是双稳态而不是干净阈值（`PHYSICS_METHOD.md` §4："两次测量同配置差 2.6 倍"）。
同样，允许 128 B 的 6 项版本能到 **8.00%**，也被弃用（128 B 不属于 MTE）。
**放弃 0.2-0.7 个 MAPE 点，换取"每一项都能解释、且量都是实测粒度"** —— 这就是物理第一性的取舍准则，
也是贪心路线在结构上做不到的事（它的目标函数里只有误差，没有可解释性）。

**(3) 第 1 轮物理式确实输给过贪心。**

`simd_load_gather_r1` 第 1 轮：物理式 6 项 **20.18%**，贪心 78 项 **12.14%**（`PHYS_mine.json`、`REPORT_v4_physics.md`）。
物理式是在第 3 轮引入 64 B burst / 1 KB 窗口 / 命令队列项之后才反超到 8.70% 的。
所以准确的表述是：**在正确的物理量被找到之前，贪心是更强的基线；找到之后，物理式以更少项数胜出。**

---

## 5. 公平地说：贪心不是一无是处

它在这次标定里至少有三个正面作用，而且是关键作用：

### 5.1 它是"物理式还没找到自变量"阶段的唯一有效基线

第 1 轮物理式在 5 个 slot 上的总体 MAPE 是 21.46%（`MODEL_PHYS.json`），
而同一时期贪心已经在 11.29%（`MODEL_MODSEL.json`）。没有贪心的成绩压着，
"物理式还不够好、必须继续找自变量"这个判断就没有参照物。

### 5.2 它的失败给出了关键线索

| 贪心选中/依赖的量 | 当时看不懂 | 后来的物理落地 |
|---|---|---|
| **`bucket_pairs_32k16`**（32 KB / 16-bucket 地址冲突对数）—— 它是 `MODEL_10TERM_FINAL.json` 里 `simt_store_r2` 的**第一个基特征**，也在 `MODEL_MODSEL.json` 的 `simd_store_r2`/`simt_store_r1`/`simt_store_r2` 基特征里 | 一个纯粹的"分桶冲突计数"，没有物理故事 | **指向了地址-通道冲突**：后来的独立扫描确认冲突粒度是 **256 B**，并落地为 `multi256`（跨行共享 256 B 块，36.79 ns/块）与 `L256` |
| **`line_elems_max`**（每条 128 B line 内最多元素数） | 设成 128 B 是我方假设 | 在 **SIMT/LSU** 路径上 128 B line 确实是对的（`L128` 成为 `simt_store_r1` 最强项，eb=1/2/4/8 -> 4.77/4.73/1.45/1.94 ns/line） |
| **`minL128` / `minL64` / `minL32` 三兄弟同时入选** | 为什么同一族三个刻度都要 | 提示存在**分段/门槛结构**，后来以 32 B sector 门槛 + 64 B burst + 1 KB 窗口的"多刻度"形式被证实 |
| **`iters` / `mwl` / `cmd` 大量入选** | 只能当统计量用 | 分别落地为"每 warp 循环延迟 `14.92 + 9.19*iters`"、"warp 波次 line 请求"、"每元素一条 BIU 命令（白盒 `biu_send_count = n`）" |

一句话：**贪心是一台不错的"相关性探测器"，它反复指向的变量值得用物理方法重新解释一遍。**
它的问题在于把"相关"直接当成了"机制"。

### 5.3 它给出了"纯拟合能到哪"的上限，从而证明了下限

`PRUNE_CURVE.json` 的实测曲线（96 项 10.86%）与 `MODEL_DELIVER.json`（261 项/式，9.95%）、
`MODEL_FINAL_V3.json`（最多 1353 项，4.07%）一起画出了"项数 -> 误差"的天花板。
这条曲线的意义是**反证**：如果连 1353 项、带全指示因子的模型都只能到 4.07%（且完全不可解释），
那么"用 3-8 项做到 7.22% 且每项可解释"就不是妥协，而是更优解。

---

## 6. 判据：什么时候可以用贪心，什么时候必须用物理

| 情形 | 建议 | 理由 |
|---|---|---|
| 需要把公式交付给 C++ / profile 侧消费、需要解释每一项 | **物理第一性** | 贪心的项名无法回答"这代表什么"（§3.4） |
| 目标是跨硬件 / 跨 lowering 路径迁移 | **物理第一性** | 迁移成本 = 重标定 6 个硬件量 vs 重跑全部特征与系数（§4） |
| 数据网格里有"看似硬件量、实则我方假设"的常量（如 `LINE_BYTES`、`PAGE_BYTES`） | **先做物理第一性，或至少把这些量标成待验证** | 否则拟合会把错误假设固化进结构（§3.1） |
| 存在阈值/区制行为（跳变、饱和、`max()` 瓶颈） | **物理第一性** | 分段线性/指示变量会掩盖机制；正确自变量可能是乘积量 `n*min(R,32)`（§3.3） |
| 只需要一个"够用的排序/回归基线"，且不打算解释 | 贪心可用 | 但要报告项数与 in-sample/留出误差的差距，并给嵌套交叉验证 |
| 探索阶段，还不确定哪些量相关 | **贪心可用作探测器**，但结论必须回到物理解释 | §5.2：它的入选名单是有价值的线索 |

**必须遵守的两条硬规则**（这次标定用血换来的）：

1. **特征选择必须放进交叉验证折内**。用同一批数据选 regime 标志又拟合系数，会得到 0.00% 这种数字（§3.2）。
2. **每个自变量都要能回答"它是哪条物理过程的哪个量"**。回答不出来就先别放进池子 —— 
   池子里没有的东西，贪心永远找不到（§3.3）。

---

## 7. 证据索引（可逐条复核）

| 结论 | 文件 | 字段 / 位置 |
|---|---|---|
| 贪心项数 6-113、总体 11.29% | [`../results/final/MODEL_MODSEL.json`](../results/final/MODEL_MODSEL.json) | `_overall`、各 slot `n_terms` / `base` / `modifiers` / `terms` |
| 14 / 15 个 regime 标志、9.91% / 9.80% | [`MODEL_REGIME_V2.json`](../results/final/MODEL_REGIME_V2.json)、[`MODEL_REGIME_V3.json`](../results/final/MODEL_REGIME_V3.json) | `_regime_flags`、`_overall` |
| 某 slot in-sample MAPE = 0.00%（过拟合） | [`MODEL_REGIME.json`](../results/final/MODEL_REGIME.json) | `simd_load_wide_r1.mape` |
| <=10 交叉项 + 剔除规则 = 15.58% | [`MODEL_10CROSSED_DOMAIN.json`](../results/final/MODEL_10CROSSED_DOMAIN.json) | `_overall_kept` |
| 项数-误差曲线（4 项 34.23% -> 96 项 10.86%） | [`PRUNE_CURVE.json`](../results/final/PRUNE_CURVE.json) | `overall` |
| `LINE_BYTES = 128` / `PAGE_BYTES = 4096` 是我方假设 | [`../scripts/build_v4_numwarps_dataset.py`](../scripts/build_v4_numwarps_dataset.py) | 第 32 / 35 行 |
| 128 B 对 MTE 不成立；64 B burst；256 B 冲突粒度 | [`../docs/PHYSICS_METHOD.md`](PHYSICS_METHOD.md) | §4「已确认的物理发现」表 |
| gather_r2：128B 版 9.52%(8项) vs MTE 重构 8.44%(7项) | [`PHYS3_simd_load_gather_r2.json`](../results/final/PHYS3_simd_load_gather_r2.json) | `comparison_128b_removal` |
| gather_r2：同 scope 贪心 8.94%(8项) | 同上 | `comparison_128b_removal.e_reference_unconstrained_greedy` |
| wide_r2：贪心 76 特征 17 项 12.39% vs 物理 7 项 4.48% | [`PHYS3_simd_load_wide_r2.json`](../results/final/PHYS3_simd_load_wide_r2.json) | `comparison` |
| 32 B 门槛：`src<32` 退化，单价高 10-30 倍 | [`PHYS_simd_load_wide_r2.json`](../results/final/PHYS_simd_load_wide_r2.json) | `notes` |
| 严格 128-free 天花板 9.5-10.5% | [`PHYS3_simd_store_r2.json`](../results/final/PHYS3_simd_store_r2.json) | `strict_128_free_alternatives.conclusion` |
| 写合并窗口 k=4 sector（=128 B）是估出来的 | 同上 | `derived`、`window_scan` |
| 粒度消融阶梯（1+nbytes 54.52% -> mx64 18.42%） | 同上 | `ablation_128b_removed.ladder` |
| 15 种跳变惩罚全被压到 0；`n*min(R,32)` | [`REPORT_PHYS2_simt_store.md`](../results/final/REPORT_PHYS2_simt_store.md) | 第 53 行起 + §3 跳变表 |
| `k = 2.18 ns/warp`；W>=16 与元素数无关 | [`PHYS2_simt_store.json`](../results/final/PHYS2_simt_store.json) | `simt_store_r1.notes` |
| 物理式主动放弃 4 KB（8.50%）与 128 B（8.00%）更低 MAPE | [`PHYS3_simd_load_gather_r1.json`](../results/final/PHYS3_simd_load_gather_r1.json) | `variants` |
| 第 1 轮物理 20.18% 输给贪心 78 项 12.14% | [`PHYS_mine.json`](../results/final/PHYS_mine.json)、[`REPORT_v4_physics.md`](../results/final/REPORT_v4_physics.md) | 汇总表 |
| 逐 slot 物理 vs 贪心对照表 | [`REPORT_v5_physics_scope.md`](../results/final/REPORT_v5_physics_scope.md) | 汇总表（第 9-21 行） |
| 最终 10 式（3-8 项、全部 <10%、加权 7.22%） | [`../README.md`](../README.md) | 总表 |
| 误差质量集中在 5% 的双稳态点（24%） | [`REPORT_v5_physics_scope.md`](../results/final/REPORT_v5_physics_scope.md) | §A |

---

## 8. 一张图总结

```
贪心路线（MODSEL / REGIME）
  起点：221 个现成特征（含我方假设的 128 B 量）
  手段：前向选修饰因子 + WLS/NNLS，同数据选特征又拟合
  终点：6-113 项/式，总体 11.29%（堆到 209 项 9.91%）
  产物：一堆无物理意义的乘积项 + 一个 0.00% 的 slot
  问题：自变量的错（128 B）、机制看不见（n*min(R,32)）、不可解释、不可迁移

        |  贪心的失败给出线索（bucket_pairs_32k16 -> 通道冲突）
        |  物理式在找到正确自变量前确实更差（20.18% vs 12.14%）
        v

物理第一性路线（最终交付）
  起点：物理过程（搬运字节 / BIU 命令数 / sector 门槛 / warp 发射 / 冲突排队）
  手段：先用受控扫描找结构（T 对单变量的二维表），再构造自变量，残差有结构就回到第一步
  终点：3-8 项/式，总体 7.22%，逐式 <10%，CV 与 in-sample 一致
  产物：10 条可解释公式 + 6 个硬件量（32 B / 64 B / 256 B / 1 KB / k=2.18 ns每warp / n*min(R,32)）
  代价：主动放弃 0.2-0.7 个 MAPE 点（4 KB、128 B 版本更低但物理不成立）
```

