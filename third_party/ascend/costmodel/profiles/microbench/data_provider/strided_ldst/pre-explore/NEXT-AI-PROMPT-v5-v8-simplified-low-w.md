# 接力提示词：v8 SIMT strided load 最终公式 → CostModel/C++ 集成

> 给下一个 AI：这不是重新拟合任务。请基于下面已经提交到
> `feature/strided-load-store-costmodel-v4` 的数据、脚本、公式和误差，
> 完成 **CostModel facts / StageCostModels / profile / 测试** 的接入；
> route-level 验证通过前，不要改正式 profile 默认系数或启用 C++ 分支。

---

## 0. 当前现场

- 本地 worktree：
  `/Users/weijianchen/Documents/2026/triton-ascend-worktrees/strided-costmodel-stride3`
- 分支：`feature/strided-load-store-costmodel-v4`
- base：`feature/strided-load-store-costmodel`（tip `5c368b591`）
- pre-explore 根目录：
  `third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore`
- 本轮已完成：
  - 1288 case v4 SIMT strided-load dataset；
  - 旧 v7 baseline 在 v4 全量上的误差；
  - v8 简化 + low-warp 增强公式的 LOBO/LOWO、分组误差、model JSON；
  - `results/model_v4_simple/` 与 `results/model_v4_simple_p25/` 两套产物；
  - `predict_strided_load_v8.py --verify` 已通过；
  - profile / C++ 尚未改动，只在 README §4 给出建议映射。

不要连接 `ascend-950pr-63`、不要重跑板卡测量；本轮只剩代码集成与测试。

---

## 1. 数据和覆盖

完整数据：

- `results/v4_retest_full_23.json`
  - 主 retest + stride 20/24/40/48/96/192 缺失 retest 合并；
  - 5-allocation Event-slope retest。
- `results/model_v4_numwarps/dataset.csv`
  - 1288 行唯一 case；
  - block：4, 8, 16, 32, 64, 128, 256, 512；
  - num_warps：1, 2, 4, 8, 16, 32, 64；
  - stride：1..12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256，共 23 个。
- `results/model_v4_numwarps/flagged_discordant.csv`
  - flagged 110 行：spread>1.4 单独 97 行、spread+dropped 7 行、
    仅 dropped 4 行、samples<3+dropped 2 行；
  - 主模型不删除 flagged；`target_p25_ns` 与 exclude-flagged 作鲁棒性对照。

清理规则：

- 丢弃 `< 120 ns` 的 allocation 样本和非正样本；
- `target_ns` = 清洗后 median；`target_p25_ns` = 下四分位；
- 保留 `target_min_ns / target_max_ns / target_spread / n_samples / n_dropped`。

---

## 2. 最新脚本

在 pre-explore 根目录下：

```text
# 数据构建 / 合并
scripts/merge_v4_retest.py
scripts/build_v4_numwarps_dataset.py

# v7 baseline
scripts/evaluate_strided_load_v7_on_v4.py
scripts/predict_strided_load_v7_simple.py

# v8 最终拟合
scripts/fit_strided_model_v8.py
scripts/predict_strided_load_v8.py
scripts/compare_strided_load_v7_v8_on_v4.py

# 本轮测量相关（已归档，无需重跑）
scripts/simt_frequency_retry_probe_aligned.py
scripts/syscnt_event_compare_aligned.py
scripts/retest_strided_simt_cases.py
scripts/run_v4_aligned_23.sh
scripts/run_v4_retest_missing.sh
scripts/run_v4_followup.sh
```

标准复现命令：

```bash
cd third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore

# 从已合并 retest 重建 dataset（不重跑板卡）
python3 scripts/build_v4_numwarps_dataset.py \
  --out results/model_v4_numwarps/dataset.csv \
  --retest results/v4_retest_full_23.json

# v7 baseline
python3 scripts/evaluate_strided_load_v7_on_v4.py

# v8 最终模型：强制选择简化 + low-warp 方案
python3 scripts/fit_strided_model_v8.py \
  --dataset results/model_v4_numwarps/dataset.csv \
  --outdir results/model_v4_simple \
  --target-col target_ns --alphas 1.0 1.5 2.0 \
  --chosen-spec v8_l32_t32_d128_p32_lowW4 --chosen-alpha 2.0

# p25 鲁棒对照
python3 scripts/fit_strided_model_v8.py \
  --dataset results/model_v4_numwarps/dataset.csv \
  --outdir results/model_v4_simple_p25 \
  --target-col target_p25_ns --alphas 1.0 1.5 2.0 \
  --chosen-spec v8_l32_t32_d128_p32_lowW4 --chosen-alpha 2.0

# predictor 自校验
python3 scripts/predict_strided_load_v8.py --verify
python3 scripts/predict_strided_load_v8.py \
  --model results/model_v4_simple_p25/model_v4_simple.json \
  --errors results/model_v4_simple_p25/errors_all.csv --verify
```

说明：`fit_strided_model_v8.py` 的 auto selection 会选到一个 CV 略好的
`lowW8 + WL3072` 7 项候选；本轮最终按“简化优先”固定为 6 项
`v8_l32_t32_d128_p32_lowW4`（no WL、dup cap128、lowW4），
模型 JSON 的 `selection.mode = "forced"` 会记录这一点。

---

## 3. 最终公式

事实定义：

```text
W      = num_warps
E      = block / (32 * W)
dup    = max(0, 1/E - 1)
L      = block                            if stride*4 >= 128
         floor(((block-1)*stride*4)/128)+1 otherwise   # distinct 128B lines
span   = (block - 1) * stride * 4
cross  = 1 if span >= 4096 else 0
```

最终主模型（raw ns / iteration，`target_ns` median，alpha=2.0）：

```text
T_simt_v8(ns/iteration)
  = 146.7722948208
  + 2.1261657441 * min(L, 32)
  + 0.2922359462 * max(0, L - 32)
  + 0.0062859052 * dup * L * min(L, 128)
  + 8.8352974780 * cross * min(L, 32)
  + 0.4598654937 * max(0, 4 - W) * L
```

`target_p25_ns` 鲁棒对照：

```text
T_simt_v8_p25(ns/iteration)
  = 145.9622878460
  + 2.1615075998 * min(L, 32)
  + 0.3012240610 * max(0, L - 32)
  + 0.0061428318 * dup * L * min(L, 128)
  + 8.6925758737 * cross * min(L, 32)
  + 0.4574158752 * max(0, 4 - W) * L
```

term 物理解释：

| term | 含义 |
|---|---|
| 常数 | 一次 SIMT rotate iteration 的固定 LDG 启动/控制、首条 line fill 下限 |
| `min(L,32)` | 前 32 条 128B line 的 fill/queue 工作量；32 条 ≈ 一个 4KB page 的 line 数 |
| `max(0,L-32)` | 进入 overlap 后的 line 边际；成本约为前段的 1/7 |
| `dup*L*min(L,128)` | underfilled warp 复制后的 line 请求数；复制压力在 128 条 line（16KB）后饱和 |
| `cross*min(L,32)` | 跨 4KB page 的 page-walk/TLB 额外成本，最多按一个 page 的 line 数收费 |
| `max(0,4-W)*L` | W<4 时少于 4 个 warp 组无法覆盖 line fill latency，每个缺失 warp、每条 line 的额外暴露 |

注意：`K`、单独的 `dup`、`max(0,W*L-3072)` 均未进入最终公式。
`max(0,W*L-3072)` 在完整 v8 中系数只有 0.0212 ns，去掉后 CV MAPE 和
CV max 都变好；`K`/`dup` 被 NNLS 压成 0。

---

## 4. 误差摘要

1288 行全量 in-sample（`target_ns`）：

| 指标 | v7 旧 | v8 最终 | v8 p25 |
|---|---:|---:|---:|
| MAPE | 33.25% | **15.23%** | **14.44%** |
| p50 | 18.17% | 9.73% | 9.50% |
| p90 | 69.44% | 37.55% | 35.85% |
| p95 | 84.41% | 59.55% | 56.13% |
| max | 1253.94% | 94.28% | 83.79% |
| bias | +6.82% | −5.39% | −4.72% |
| RMSE | 82.82% | 23.21% | 21.72% |

CV pooled（out-of-fold prediction）：

| 口径 | MAPE | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|
| v8 target_ns LOBO | 16.08% | 10.26% | 40.15% | 59.87% | 94.27% |
| v8 target_ns LOWO | 16.51% | 10.50% | 49.28% | 63.18% | 94.28% |
| v8 target_p25 LOBO | 15.29% | 10.15% | 37.63% | 57.32% | 84.02% |
| v8 target_p25 LOWO | 15.64% | 10.45% | 44.54% | 59.45% | 83.76% |

分组重点（v7 → v8）：

- block8：MAPE 68.4% → 14.2%；block512：43.0% → 25.3%；
- W32：46.2% → 12.9%；W64：63.1% → 20.7%；
- stride192：92.7% → 16.3%；stride256：105.3% → 30.7%；
- W1：30.1% → 19.4%；W2：32.0% → 22.7%。

完整文件：

```text
results/model_v4_simple/model_v4_simple.json
results/model_v4_simple/metrics_summary.json/.csv
results/model_v4_simple/errors_all.csv
results/model_v4_simple/group_metrics_by_block.csv
results/model_v4_simple/group_metrics_by_num_warps.csv
results/model_v4_simple/group_metrics_by_stride.csv
results/model_v4_simple/cv_candidates.csv
results/model_v4_simple/cv_lobo_folds.json
results/model_v4_simple/cv_lowo_folds.json
results/model_v4_simple/v7_vs_v8_group_metrics.csv
results/model_v4_numwarps/v7_eval/
```

已知 limitation：

- W1/W2、小 block 高 stride，以及 b512 单 warp 大 span 的
  bank/set contention tail 仍未建模；这些点 target 偶发 800–3000 ns，
  模型只给 150–600 ns；
- 试过的 2KB/8-bank、32KB/16-group、page_count、128B set proxy
  等特征在 alpha=2 NNLS 下全部为 0；需要更精细的真实 set/sector 事实；
- `max(0,4-W)` 的阈值 4 是经验值；当前 W 网格没有 3/6/12，
  建议后续补点验证，或在 route-level 验证里确认；
- 当前事实仍限制在 aligned base、32-bit element、正 stride、无 mask。

---

## 5. 融入 C++ / profile 的具体方案（待 route-level 验证后启用）

### 5.1 `StridedMemoryProfile` 字段

建议在 `StageCostModels.h::StridedMemoryProfile` 中增加/调整：

```cpp
double simtLoadInterceptNs = 0.0;            // 保留现有字段
double simtLoadMinLine32Ns = 0.0;            // 新增
double simtLoadTailLine32Ns = 0.0;           // 新增
double simtLoadDupLMinL128Ns = 0.0;          // 新增，替代旧 dup_line 语义
double simtLoadCrossMinL32Ns = 0.0;          // 新增
double simtLoadLowWarp4LNs = 0.0;            // 新增
```

旧的 `simtLoadLineNs` / `simtLoadDupLineNs` / `simtLoadWarpLineOverflowNs`
在最终公式中不再使用；建议置 0 并在 schema 中标记 deprecated，
但不要在没有兼容层的情况下直接删除旧 JSON 字段。

### 5.2 `StageCostModels.cpp::stridedSimtLoadCostNs()`

```cpp
const double E = elements / (32.0 * warps);
const double dup = std::max(0.0, 1.0 / std::max(E, 1e-12) - 1.0);
const double lines = stridedLineCount(elements, strideBytes);
const double cross = ((elements - 1.0) * strideBytes) >= 4096.0 ? 1.0 : 0.0;
const double lowWarp4 = std::max(0.0, 4.0 - warps);
return cal.simtLoadInterceptNs
     + cal.simtLoadMinLine32Ns * std::min(lines, 32.0)
     + cal.simtLoadTailLine32Ns * std::max(0.0, lines - 32.0)
     + cal.simtLoadDupLMinL128Ns *
           dup * lines * std::min(lines, 128.0)
     + cal.simtLoadCrossMinL32Ns * cross * std::min(lines, 32.0)
     + cal.simtLoadLowWarp4LNs * lowWarp4 * lines;
```

### 5.3 profile JSON / schema

建议在 `david_v100_simd_simt_v1.json` 的 `simt.stage_resources.strided_memory` 中：

- schema version：13 → **14**；
- profile version：v28 → **v29**；
- 新增字段：
  ```json
  "simt_load_min_line32_ns": 2.1261657441,
  "simt_load_tail_line32_ns": 0.2922359462,
  "simt_load_dup_l_min_l128_ns": 0.0062859052,
  "simt_load_cross_min_l32_ns": 8.8352974780,
  "simt_load_low_warp4_l_ns": 0.4598654937
  ```
- `simt_load_intercept_ns` 更新为 146.7722948208；
- `nanoseconds_to_system_cycles = 0.9889` 不变；
- SIMD load / store 和 `stridedSimtStoreCostNs()` 本轮不动。

### 5.4 测试建议

必须补 C++/profile UT：

1. `StridedMemoryProfile::isValid`：新字段缺失/负数/非有限值报错；
2. `stridedSimtLoadCostNs` golden test：
   - W=1/2/4/8/32/64；
   - block=32/128/512；
   - stride=1/4/32/256；
   - 覆盖 L≤32、L>32、cross=0/1、dup=0/>0；
3. `stridedLineCount` / `dup` / `lowWarp4` 在 fractional per-iteration
   elements 下的行为；
4. 检测路径：确认 `detectStridedMemoryAccess()` 产生的
   `StridedMemoryAccess` 与公式输入一致；
5. 防 double-count：确认 `mapWorkload()` 从 legacy direct
   bytes/warp-instruction 中扣除的仍是
   `ceil(elements/32)`，且新 lowW 项不会再次叠加 legacy issue；
6. schema/profile version bump 后跑 `predict_strided_load_v8.py --verify`
   和 `simd_simt_profile_schema.json` 校验。

### 5.5 route-level 验证

在启用 profile/C++ 默认值前，至少做：

- 对所有 v7/v8 差异大的真实 workload（b8/s256、W1/W2、b512 大 stride）
  跑 route report，比较 selected route 和 predicted cycles；
- 与旧 v7 profile 做 A/B，确认没有 route collapse；
- 对 lowW4 阈值补 W=3/6/12 板卡/仿真点，或证明 route 结果对该阈值不敏感。

---

## 6. 你要交付什么

1. C++ 新字段 + `stridedSimtLoadCostNs()` 新公式；
2. profile schema v14 / profile v29 的新 `strided_memory` 字段；
3. UT 覆盖 5.4 的 golden/边界/检测路径；
4. README 或对应设计文档更新，说明最终公式与 deprecated 字段；
5. route-level 验证报告，说明是否启用默认值；
6. 不要直接删除旧字段，先保留兼容路径。
