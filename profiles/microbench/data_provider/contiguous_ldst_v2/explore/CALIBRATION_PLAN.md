# 连续访存重标定计划（CALIBRATION_PLAN v1，2026-11-12）

> 依据：`EXPERIMENTS.md` 全部探索结论（E1~E11）。
> 目标：为新分类（EXPERIMENTS.md §17 决策树）的四族公式提供 MAPE<10% 的
> 标定数据，全部采用**同会话 probe/ctl 成对差分**口径（模式免疫），
> 替代现有 FormS/segment/bw 三族的混杂状态。

## 0. 目标公式（拟合的最终形态）

```
公式A（单发+组合并）  T(N) = first + (N−1)×marginal      N ≤ Q(≈16, 队列深度)
                      T(N) = N × c_transfer              N > Q
公式B（循环）         T(K) = S_loop + K × m_round
residency 系数        simd: warm/cold 两套参数
                      simt: hot(DCache命中)/miss 两套参数（warm≡cold）
S 修正（仅 simt）     T += S × penalty(dtype, bytes)      （E10: 142ns@4KB fp32）
```

- 公式A 的 N=1 点 = 单发（FormS 语义），N≤Q 段覆盖组摊销（E3c/E3e 的
  亚线性），N>Q 段覆盖传输受限（E7 收敛）。
- 公式B 覆盖 chain/indpt/store 三种循环体——simd 已证轮价对 body 不敏感
  （E1 88 ≈ E6 102），simt 待重标确认。
- store 侧单独成族（store 无 L2 悬崖、近线性、膝点 32→64MB，E1/E4-C）。

## 1. 参数格（要拟合的完整矩阵）

| 维度 | 取值 | 数量 |
|---|---|---|
| 引擎 | simd / simt_only | 2 |
| dtype | int8 / fp16 / fp32 / int64 | 4 |
| bytes/实例 | 256B / 1KB / 4KB / 8KB / 16KB（部分格子）| ≤5 |
| residency | simd: warm/cold；simt: hot/miss | 2 |
| W（simt 专属） | 1/2/4/8/16/32/64 | 7（仅 simt）|
| mask | FormA（无）/FormB（动态全 true）| 2（单发族先做，其余后补）|

全矩阵规模：simd A 族 4×5×2=40 格 ×（first+marginal+c_transfer）；
simt 同格 × W 轴。**分里程碑执行，不一次跑全**（见 §5）。

## 2. 可复用的已有实测（不用重测）

| 数据源 | 位置 | 提供什么 | 复用条件 |
|---|---|---|---|
| FormS v5 | `forms/raw/forms_single_shot_950pr_v5.jsonl` | warm N=1 单发全 dtype（simd 合并 + simt 8 族），ABAB 差分中位 | 直接作为公式A `first` 的 warm 值（E1 复现 149≈146.4 已验证）|
| E1 load/store 单发 | `results/e1_fp32_c2.jsonl`（tag ss_*） | fp32 4KB warm/cold 单发（simd 149/447、store 77/172；simt 370/324、227/240）| ✓ |
| E3c U sweep | `results/e1b_e3c_c2.jsonl` | warm 组边际（S=1）：simd ~30-50、simt ~75-133（U≤8）| fp32 4KB ✓ |
| E10 S 扫描 | `results/e10_c2.jsonl` | S 罚系数（simt 142ns@4KB fp32；simd 不敏感）| fp32 ✓，其它 dtype 待补 |
| E6 loop indpt | `results/e6_c3.jsonl` | warm 循环轮价（simd ~102、simt 慢态 ~980 含流式 store）| warm ✓ |
| E1 循环 K 扫描 | `results/e1_fp32_c2.jsonl` | chain/store 循环 warm/cold 轮价曲线（fp32：88↔167 等）| fp32 ✓ |
| E4-C 足迹悬崖 | `results/e4c_c2.jsonl` | 聚合层常数（L2 悬崖 92→249、store 膝点、DRAM ~0.9TB/s）| route 层直接用 |
| raw_pure_unified | 老标定 `raw/raw_pure_unified.jsonl` | grid=1 acc-chain warm 轮价（全 dtype×W）| 作为 simt/simd 公式B warm `m_round` 候选，抽查 2-3 点与 E1 对齐后可用 |
| small_group usweep | 老标定 `forms/raw/small_group_usweep_950pr.jsonl` | ≤256B 小 N 组摊销 | 小 bytes 格子复用 |

**不可复用 / 需重测**：
- `bw_raw/BW_SIMT.jsonl`：双峰污染（63/187 组 >1.8×）→ simt 吞吐作废重测；
- E6 simt 循环：慢态混入流式 store，不可拆 → 重测（拆 load-only/store-only）；
- `te8_stream_paired_950pr.jsonl`：映射到新分类的对应关系未梳理 → 暂缓；
- FormS 的 simd te≥8 表（`ldst.*.eb*`）：当时口径（singleShotContext 门）
  与新分类不同 → 用 E1/E11 结论重判（单发 warm 即 first，与 te 无关）。

## 3. 待测清单（新探针）

| # | 探针 | 测什么 | 格子 |
|---|---|---|---|
| P1 | `cal_a_group.py --direction load` | 公式A：N∈{1,2,4,8,16,32,64} × warm/cold，流式地址，S=1（求和收尾）| simd/simt × dtype × bytes |
| P2 | `cal_a_group.py --direction store` | store 族：first/marginal/c_transfer，warm/cold | 同上 |
| P3 | `cal_b_loop.py --body chain` | 公式B chain：K∈{2..256} × warm/cold | 全 dtype（fp32 可复用 E1 部分）|
| P4 | `cal_b_loop.py --body indpt` | 公式B indpt（load+store 体）× warm/cold | E6 只有 warm fp32 → 补 cold/其它 |
| P5 | `cal_b_loop.py --body store` | store 循环 × warm/cold | E1 fp32 有 → 其它 dtype |
| P6 | `cal_s_penalty.py` | S 罚其它 dtype（fp16/int8/int64）| simt |
| P7 | 队列深度边界 | U=16 点（simd 两段断点）| P1 加一个 U 即可 |
| P8 | simt W 轴 | W∈{1,2,4,8,16,32,64}（公式A/B 的 simt 格子）| 随 P1-P5 的 --warps |

## 4. 测量协议（所有探针强制）

1. 同会话 ABAB 成对差分（probe/ctl 同引擎同模式）→ 模式免疫；
2. 每 12-16 对 × 3 个独立会话，取 cv≤0.4 会话的最小差分中位；
3. 会话参考门：simd 参考越界 ±10% 弃会话；simt 加参考 kernel 门；
4. 流式地址（base+step*BLOCK，不用 window 复读）；residency 由足迹控制：
   warm = 区域 ≤32MB 复用；cold = 旋转 >320MB 从不重访；
5. U/K 全运行时参数或一次生成模块（少编译）；dump 逐 launch 时序备聚类。

## 5. 里程碑

| 阶段 | 内容 | 验证 |
|---|---|---|
| M0 | pilot：P1/P3 各跑 fp32 4KB 少量点，对照 E1/E3c/E6 期望值 | 数量级+方向一致 |
| M1 | simd warm 全 dtype/bytes（P1/P2/P3/P4/P5 warm 段）| 每族 MAPE<10% |
| M2 | simd cold 全格子（P1-P5 cold 段）| 同上 |
| M3 | simt 重标（差分免疫协议，全格子）| 同上 + 状态聚类报告 |
| M4 | S 罚其它 dtype（P6）+ U=16 边界（P7）| 曲线平滑 |
| M5 | 集成 `estimateContiguousCompletionNs` 新分发 | 单元对照 |
| M6 | 100-case compile-only vs **3050 基线（88/100, median 3.43%）** | ≥ 基线才算重构有效 |

## 6. 期望值速查（pilot 对照用，均为 warm/cold 成对差分 ns）

| 点 | 期望 | 出处 |
|---|---|---|
| simd 单发 4KB warm/cold | ~149 / ~447 | E1 ss |
| simd 组边际 4KB（U≤8, S=1）| ~30-50 | E3c/E3e |
| simd U=32 每实例 | ~120-125（传输受限收敛）| E7 |
| simd 循环轮价 chain/indpt warm | ~88-92 / ~102 | E1/E6 |
| simd 循环 cold 轮价 | ~167 | E1 |
| simt 单发 4KB | ~270-370 | E1/E3c |
| simt 组边际（S=1，U≤8）| ~75-133 | E3e v1 |
| simt S 罚 | ~142ns/store | E10 |
| simt 循环 chain warm | ~230 | E1 |
| simt 循环 cold | ~230（对源不敏感）| E1 |
| simt 同块复读（hot）| ~79 | E1 |
| 今日环境态 | simt 全程慢态（~415-490ns/对），simd 正常 | E8/E9 |
