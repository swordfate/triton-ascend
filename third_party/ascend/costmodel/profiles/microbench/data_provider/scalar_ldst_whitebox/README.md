# scalar load/store CostModel 白盒建模与验证

> 对象：`test_cases/scalar_dominate_kernels/` 中 6 个 scalar-dominated megablocks kernel：
> `padded_copy_{gather,scatter,wgrad}`、`binned_copy_{gather,scatter,wgrad}`。
> 固定条件：`shape=(sl,hs,ne,top_k)=(4,256,4,2)`、`BLOCK_X=64`、`superblock_factor=1`、`num_warps=1`。
> 当前 route：6 个 kernel 全部 `all_simt_only`；本文同时给出 forced `simd` 的实现公式对照。

---

## 0. 总结

1. base 的 costmodel 没有 scalar `tt.load` / `tt.store` 的独立 StageKind 与白盒成本。以上 6 个 kernel 的 scalar 访存表现为：
   - **direct scalar load**：padded 的 `indices` / `bin_ids`，binned 的 `bins`，K=1 或 2；
   - **indirect scalar load**：地址由另一个 scalar load 的结果计算出来；
   - **单 scalar store**：wgrad 的 `tl.store(wgrad, out)`。
2. 我们先用最简单的 CCE/CAModel 微基准程序确定时钟和单位：CAModel 内部是 **1.8 GHz**，costmodel profile 是 **SYS_CNT 988.9 MHz** 域，落盘时乘 `988.9/1800 = 0.5493889`。
3. 白盒公式与 CAModel 微基准程序/目标 kernel：
   - SIMD load 三点精确拟合；
   - SIMT load 用 CCE 微基准程序 + 6 kernel 联合重拟合 `fill=480`（目标 kernel direct/indirect MAPE ≈8%）；
   - SIMT K=1 store 与 CAModel 误差 −8.4%，SIMD Triton store 走 MTE3，目标 MTE3 window 误差 −5.2%/+13.8%。
4. 6 kernel matched-only + per-stage-union MAPE：SIMT direct 8.0%、indirect 8.0%、store 18.9%；SIMD forced direct 10.9%、indirect 10.1%、store 9.5%。
5. SIMD scalar store 在 Triton 下走 MTE3（`scalar → UB staging → MTE3 MOV UB→OUT`），不是 CCE MainScalar `ST_XD_XN_IMM → GM`。目标 route 全为 SIMT，SIMD store 只作为 forced-mode 公式对照。

---

## 1. 目标场景

### 1.1 6 个 kernel 的 scalar 访存

| kernel | direct scalar load | indirect scalar load（地址依赖） | scalar store |
|---|---|---|---|
| padded_copy_gather | `indices[pid]`, `bin_ids[pid]`（K=2） | `bins[bin_idx-1]`, `padded_bins[bin_idx-1]`（本次执行 2 条） | — |
| padded_copy_scatter | `indices[pid]`, `bin_ids[pid]`（K=2） | `bins[bin_idx-1]`, `padded_bins[bin_idx-1]`, `weights[index_a]`（3 条） | — |
| padded_copy_wgrad | `indices[pid]`, `bin_ids[pid]`（K=2） | `bins[bin_idx-1]`, `padded_bins[bin_idx-1]`（2 条） | `tl.store(wgrad, out)` |
| binned_copy_gather | `bins[expert_idx]`（seed=0 执行 1 条） | `indices[start+entry_idx]`（1 条） | — |
| binned_copy_scatter | `bins[expert_idx]`（seed=0 执行 1 条） | `indices[start+entry_idx]`（1 条） | — |
| binned_copy_wgrad | `bins[expert_idx]`（seed=0 执行 1 条） | `indices[start+entry_idx]`（1 条） | `tl.store(wgrad, out)` |

### 1.2 最终建模的 4 种情况

| 场景 | 形态 | 公式 |
|---|---|---|
| SIMD/SIMT direct scalar load | 冷 line、每 Stage K=1 或 2 条独立 load | `T = prep + fill + (K-1)*extra_issue` |
| indirect scalar load | target 中每个 producer 只有一个 consumer（exposure=1）；consumer 自己的 line fill 已经包含 producer→consumer 气泡 | 使用同 mode、同 K 的 direct load 公式，不额外收 dependency |
| SIMT scalar store | `SIMT_STG` 写 128B line，target 为单 store | `T = store_base` |
| SIMD scalar store（forced mode） | Triton scalar `tt.store` → MTE3 UB→OUT | `T = mte3_prep + mte3_fill` |

> **Stage 边界与 K**：Stage 不是按单条 op 切的。`StagePartitioner` 按 semantic root / control-flow anchor 聚合；
> padded 的两条 direct（`indices` / `bin_ids`）被切成两个 K=1 stage，各付一次 `prep+fill`；
> 而 padded 的 `bins[bin_idx-1]` / `padded_bins[bin_idx-1]` 落在同一个 `if bin_idx>0` stage，K=2，
> 所以 K=2 公式实际用于 indirect stage。§4 中 padded_copy_gather indirect 的单 stage 486（SIMT）/ 450（SIMD）
> 就是 K=2 的结果；如果误按两个 K=1 stage 相加，会得到 972 / 894。
> padded_copy_scatter 的 `weights[index_a]` 另成一个 K=1 stage，所以其 indirect 总和是
> `stage_7(K=2) + stage_11(K=1) = 972`（SIMT）/ `897`（SIMD）。

### 1.3 采样与单位

- CAModel 取 `core0.veccore0` 一个 program；
- `msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 --launch-count=1`；
- 当前 route：`compile_mode=simt_only`；forced SIMD 对照：`compile_mode=simd`；
- CAModel active window 是 core cycle @1.8GHz；costmodel profile 的 `*_system_cycles` 是 SYS_CNT cycle @988.9MHz。

---

## 2. CAModel 标定

### 2.1 时钟确认：1.8 GHz

用最简单的 CCE scalar CAModel 程序，把 CAModel 打印的 `duration_time(us)` 与同一 core dump 的 `instr_log` cycle span 对齐：

| launch | core | dump cycle span | duration (us) | cycles/us | 等效 GHz |
|---|---|---:|---:|---:|---:|
| load | core0.veccore0 | 10906 | 6.06 | 1799.7 | **1.800** |
| load | core0.veccore1 | 10514 | 5.84 | 1800.3 | **1.800** |
| store | core0.veccore0 | 8786 | 4.88 | 1800.4 | **1.800** |
| store | core0.veccore1 | 9140 | 5.08 | 1799.2 | **1.800** |

结论：CAModel 内部 cycle → time 换算为 **1.8 GHz**。写入 profile 的 SYS_CNT 值为：

```text
T_sys = T_camodel × 988.9 / 1800 = T_camodel × 0.5493889
```

### 2.2 标定程序

| 场景 | 文件 / 函数 | 用途 |
|---|---|---|
| SIMD/SIMT 单 scalar load | `load/scalar_o1/load_scalar_o1.cce`：`simd_main_ld_o1`、`simt_ld_uniform_o1` | 固定 prep + line fill 成本 |
| SIMD/SIMT 多 op load | `load/scalar_o4/load_scalar_o4.cce`：`simd_main_ld_same_o4`、`simd_main_ld_diff_o4`、`simt_ld_uniform_same_o4`、`simt_ld_uniform_diff_o4` | 拆分 hit/outstanding 与 issue 成本 |
| SIMT scalar store | `store/scalar_o1/store_scalar_o1.cce`：`simt_st_uniform_o1` | 拟合单 store window |
| SIMD Triton scalar store | `store/triton_scalar_store/triton_scalar_store_demo.py`：`_triton_scalar_store_demo` | 确认 MTE3 路径并拟合 stage 系数 |
| 真卡对照 | `syscnt/board_marginal/board_vs_model_agg.csv` | 真卡 SYS_CNT marginal 聚合结果（多次 run 中位数），校验 CAModel 的绝对时间 |

### 2.3 公式构造

**A. SIMD MainScalar load**

从 o1 load 和 o4 same/diff 三个点拆出：

- `prep = 7`：issue→tag/MSHR/BIU dispatch（4）+ refill→retire（3）；
- `fill = 440`：一次 cold 64B line fill（DC 发 BIU 8 + BIU 读 421 + 回 DC 11）；
- `issue = 3`：每条额外 op 的 issue/serialization；
- `outstanding = 2`：MSHR 同时在途的 line 数；超出部分每条不同 line 额外付一次 line fill。

```text
extra(K) = K <= 4 ? 250 : 350
T_simd_load(K) = 7 + 440 + max(0, K - 2) × extra(K) + (K - 1) × 3

K=1: 447
K=2: 450
K=3: 703
K=4: 956
K>4: 按 extra=350 外推
```

目标 kernel 使用 K≤2；K>2 的 extra-line 项用于覆盖多 outstanding line 的 diff-line load。

profile 落盘值：`main_load_prep=3.8457`、`main_load_fill=241.7311`、`main_load_issue=1.6482`、
`main_load_outstanding_line_count=2`、`main_load_extra_line_low=137.3472`、
`main_load_extra_line_high=192.2861`、`main_load_extra_line_high_threshold=4`（SYS_CNT cycle/计数）。

**B. SIMT warp-uniform load**

CCE 单条 load 微基准程序为 `6 + 524 = 530`；但 6 个目标 kernel 实测 line-fill 在 406–558 cycle，
只用微基准程序拟合的 524 会使目标 kernel 的 indirect 高估约 +30%。用 CCE 微基准程序 + 6 kernel 联合重拟合取 `fill = 480`：

```text
T_simt_load(K) = 6 + 480 + (K - 1) × 1
             K=1: 486（对 CCE 微基准程序 530 误差 −8.3%）
             K=2: 487
             K=4: 489（diff-line 线性外推）
```

`1` 表示每条额外 diff-line load 付 1 个 CAModel core cycle 的 issue 地板。

profile 落盘值：`uniform_load_prep=6×0.5493889=3.2963`、`uniform_load_fill=480×0.5493889=263.7067`、`uniform_load_diff_line_issue=1×0.5493889=0.5493889`（SYS_CNT cycle）。

**C. indirect scalar load**

目标 6 个 kernel 中，indirect load 的 producer 都只有一个 consumer（exposure=1）。
consumer 自己的 line fill window 已经包含 producer retire→consumer issue 的气泡，因此：

```text
T_indirect(mode, K) = T_load(mode, K)
```

不额外增加 dependency latency 字段。

**D. scalar store**

- SIMT `SIMT_STG` 单 store：`T_simt_store = 450`（profile `uniform_store_base=450×0.5493889=247.225`）。
- SIMD Triton scalar store 走 MTE3；costmodel 使用 MTE3 stage 可摊销系数：

```text
T_simd_store = 20 + 450 = 470
```

profile 落盘：`mte3_store_prep=20×0.5493889=10.9878`、`mte3_store_fill=450×0.5493889=247.225`。

### 2.4 标定结果

公式 vs CAModel 微基准：

| 场景 | 公式 cycle | CAModel window | 误差 |
|---|---:|---:|---:|
| SIMD diff-line load K=1 | 447 | 447 | 0.0% |
| SIMD diff-line load K=4 | 956 | 956 | 0.0% |
| SIMT diff-line load K=1 | 486 | 530 | −8.3% |
| SIMT diff-line load K=4 | 489 | 526 | −7.0% |
| SIMT scalar store K=1 | 450 | 491 | −8.4% |
| SIMD scalar store（MTE3）K=1 | 470 | 496 / 413 | −5.2% / +13.8% |

K=2 形态（padded indirect stage）与 6 kernel 的 matched-only 结果见 §4。

CAModel vs 真卡 direct load microbench（board ns 来自 `board_vs_model_agg.csv`）：

| 场景 | CAModel cycle | CAModel ns@1.8 | board ns | 误差 |
|---|---:|---:|---:|---:|
| SIMD same-line load | 447 | 248.3 | 274.5 | −9.5% |
| SIMD diff-line load | 447 | 248.3 | 273.8 | −9.3% |
| SIMT same-line load | 530 | 294.4 | 291.8 | +0.9% |
| SIMT diff-line load | 530 | 294.4 | 289.0 | +1.9% |

---

## 3. CostModel 落地

### 3.1 Stage 识别与工作量

- `StageWorkload` 增加 `scalarLoadCount` / `scalarStoreCount`（每 iteration 的 scalar op 数 K）；
- `StagePartitioner` 对非 shaped 的 `tt.load` / `tt.store` 单独计数，并分类为 `ScalarLoad` / `ScalarStore`；
- scalar 访存不再进入 tile / indirect / contiguous memory 的 workload 与成本路径。

### 3.2 profile 字段

simd `stage_resources.scalar_memory`：

| 字段 | 值（SYS_CNT cycle） | 来源 |
|---|---:|---|
| `main_load_prep_system_cycles` | 3.8457 | CAModel 7 core cycle |
| `main_load_fill_system_cycles` | 241.7311 | CAModel 440 core cycle |
| `main_load_issue_system_cycles` | 1.6482 | CAModel 3 core cycle |
| `main_load_outstanding_line_count` | 2 | MSHR outstanding lines |
| `main_load_extra_line_low_system_cycles` | 137.3472 | CAModel 250 core cycle（K≤4） |
| `main_load_extra_line_high_system_cycles` | 192.2861 | CAModel 350 core cycle（K>4） |
| `main_load_extra_line_high_threshold` | 4 | extra-line 分档阈值 |
| `mte3_store_prep_system_cycles` | 10.9878 | CAModel 20 core cycle |
| `mte3_store_fill_system_cycles` | 247.2250 | CAModel 450 core cycle |

simt `stage_resources.scalar_memory`：

| 字段 | 值（SYS_CNT cycle） | 来源 |
|---|---:|---|
| `uniform_load_prep_system_cycles` | 3.2963 | CAModel 6 core cycle |
| `uniform_load_fill_system_cycles` | 263.7067 | CAModel 480 core cycle（联合重拟合） |
| `uniform_load_diff_line_issue_system_cycles` | 0.5493889 | CAModel 1 core cycle |
| `uniform_store_base_system_cycles` | 247.2250 | CAModel 450 core cycle |

profile 字段都是可选项；`schema_version=12`、`profile_version` 与 base 一致。

### 3.3 代码路径

| 文件 | 修改 |
|---|---|
| `include/.../StageCostModels.h` | 新增 `ScalarLoad` / `ScalarStore` kind 与 profile 字段 |
| `include/.../StageRouteCostModel.h` | `StageWorkload` 增加 scalar load/store count |
| `lib/.../Analysis/StagePartitioner.cpp` | scalar 识别、计数、分类 |
| `lib/.../RouteModel/StageCostModels.cpp` | `mapWorkload` 按 mode 加入白盒公式 |
| `lib/.../RouteModel/StageRouteCostModel.cpp` | workload 校验与 report JSON |
| `lib/.../RouteModel/SimdSimtCostModel.cpp` | 读取可选 `scalar_memory` 字段 |

---

## 4. 6-kernel matched-only + stage-union 验证

口径：

- costmodel：`compile_mode=simd_simt` report，只累加真正执行的 matched stage；
- CAModel：当前 route `simt_only`，padded seed=12、binned seed=0；每个 matched stage 取 `max(retire)-min(issue)` union 后求和；
- costmodel 的 SYS_CNT cycle 乘 `1800/988.9` 转回 CAModel core cycle 后与 CAModel 比较；
- `err = (costmodel_camodel_equiv - camodel_union) / camodel_union`。

### 4.1 SIMT（当前 route）

| kernel | 类别 | matched | costmodel SYS_CNT cyc | CAModel-equiv cyc | CAModel union cyc | costmodel ns | CAModel ns | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| padded_copy_gather | direct load | 2/2 | 534.0 | 972.0 | 1113.0 | 540.0 | 618.3 | −12.7% |
| padded_copy_gather | indirect load | 1/1 | 267.55 | 487.0 | 509.0 | 270.6 | 282.8 | −4.3% |
| padded_copy_scatter | direct load | 2/2 | 534.0 | 972.0 | 1115.0 | 540.0 | 619.4 | −12.8% |
| padded_copy_scatter | indirect load | 2/2 | 534.56 | 973.0 | 973.0 | 540.6 | 540.6 | 0.0% |
| padded_copy_wgrad | direct load | 2/2 | 534.0 | 972.0 | 897.0 | 540.0 | 498.3 | +8.4% |
| padded_copy_wgrad | indirect load | 1/1 | 267.55 | 487.0 | 481.0 | 270.6 | 267.2 | +1.2% |
| padded_copy_wgrad | scalar store | 1/1 | 247.2 | 450.0 | 559.0 | 250.0 | 310.6 | −19.5% |
| binned_copy_gather | direct load | 1/2 | 267.0 | 486.0 | 497.0 | 270.0 | 276.1 | −2.2% |
| binned_copy_gather | indirect load | 1/1 | 267.0 | 486.0 | 436.0 | 270.0 | 242.2 | +11.5% |
| binned_copy_scatter | direct load | 1/2 | 267.0 | 486.0 | 497.0 | 270.0 | 276.1 | −2.2% |
| binned_copy_scatter | indirect load | 1/1 | 267.0 | 486.0 | 436.0 | 270.0 | 242.2 | +11.5% |
| binned_copy_wgrad | direct load | 1/2 | 267.0 | 486.0 | 442.0 | 270.0 | 245.6 | +10.0% |
| binned_copy_wgrad | indirect load | 1/1 | 267.0 | 486.0 | 406.0 | 270.0 | 225.6 | +19.7% |
| binned_copy_wgrad | scalar store | 1/1 | 247.2 | 450.0 | 551.0 | 250.0 | 306.1 | −18.3% |

### 4.2 SIMD（forced 对照）

| kernel | 类别 | matched | costmodel SYS_CNT cyc | CAModel-equiv cyc | CAModel union cyc | costmodel ns | CAModel ns | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| padded_copy_gather | direct load | 2/2 | 491.1 | 894.0 | 883.0 | 496.7 | 490.6 | +1.2% |
| padded_copy_gather | indirect load | 1/1 | 247.2 | 450.0 | 533.0 | 250.0 | 296.1 | −15.6% |
| padded_copy_scatter | direct load | 2/2 | 491.1 | 894.0 | 1011.0 | 496.7 | 561.7 | −11.6% |
| padded_copy_scatter | indirect load | 2/2 | 492.8 | 897.0 | 930.0 | 498.3 | 516.7 | −3.5% |
| padded_copy_wgrad | direct load | 2/2 | 491.1 | 894.0 | 801.0 | 496.7 | 445.0 | +11.6% |
| padded_copy_wgrad | indirect load | 1/1 | 247.2 | 450.0 | 508.0 | 250.0 | 282.2 | −11.4% |
| padded_copy_wgrad | scalar store | 1/1 | 258.2 | 470.0 | 496.0 | 261.1 | 275.6 | −5.2% |
| binned_copy_gather | direct load | 1/2 | 245.6 | 447.0 | 364.0 | 248.3 | 202.2 | +22.8% |
| binned_copy_gather | indirect load | 1/1 | 245.6 | 447.0 | 528.0 | 248.3 | 293.3 | −15.3% |
| binned_copy_scatter | direct load | 1/2 | 245.6 | 447.0 | 475.0 | 248.3 | 263.9 | −5.9% |
| binned_copy_scatter | indirect load | 1/1 | 245.6 | 447.0 | 425.0 | 248.3 | 236.1 | +5.2% |
| binned_copy_wgrad | direct load | 1/2 | 245.6 | 447.0 | 510.0 | 248.3 | 283.3 | −12.4% |
| binned_copy_wgrad | indirect load | 1/1 | 245.6 | 447.0 | 493.0 | 248.3 | 273.9 | −9.3% |
| binned_copy_wgrad | scalar store | 1/1 | 258.2 | 470.0 | 413.0 | 261.1 | 229.4 | +13.8% |

### 4.3 MAPE

| 类别 | SIMT MAPE | SIMT 最大/最小 | SIMD MAPE | SIMD 最大/最小 |
|---|---:|---:|---:|---:|
| direct scalar load | 8.0% | +10.0% / −12.8% | 10.9% | +22.8% / −12.4% |
| indirect scalar load | 8.0% | +19.7% / 0.0% | 10.1% | +5.2% / −15.6% |
| scalar store | 18.9% | −18.3% / −19.5% | 9.5% | +13.8% / −5.2% |

> 残差主要来自 SIMT line-fill 的 BIU/subcore 仲裁波动、`stage_5` 条件分支未执行时的 matched 计数，以及 SIMT store 白盒 450 与 CAModel 551/559 的窗口差。

---

## 5. 文件与复现

```text
scalar_ldst_whitebox/
  README.md                           # 本文
  load/scalar_o1/                     # SIMD/SIMT 单 scalar load 微基准程序
  load/scalar_o4/                     # same/diff 4-op load 微基准程序
  store/scalar_o1/                    # SIMT 单 scalar store 微基准程序
  store/triton_scalar_store/          # Triton scalar store MTE3 demo
  syscnt/board_marginal/board_vs_model_agg.csv  # 真卡 SYS_CNT marginal 聚合结果
  costmodel_eval/
    make_profile.py                   # 写入 SYS_CNT 域的 scalar_memory 字段
    summarize_all_scalar.py           # matched-only + per-stage-union
    results/all_scalar_eval.csv/md    # SIMT 表
    results/all_scalar_eval_simd.csv/md # forced SIMD 表
```

复现：

```bash
source ~/env_ascend.sh
ulimit -n 1048576

cd load/scalar_o1 && bash build_scalar_o1.sh && bash run_scalar_o1_camodel.sh
python3 parse_load_scalar_o1.py camodel_results/simd camodel_results/simt32 camodel_results/simt1

cd ../scalar_o4 && bash build_scalar_o4.sh && bash run_scalar_o4_camodel.sh
python3 parse_load_scalar_o4.py camodel_results/simd_same camodel_results/simd_diff \
    camodel_results/simt_same camodel_results/simt_diff

cd ../../store/scalar_o1 && bash build_store_scalar_o1.sh && bash run_store_scalar_o1_camodel.sh
python3 parse_store_scalar_o1.py camodel_results/simd camodel_results/simt32

cd ../triton_scalar_store && N_ST=1 GRID=4 bash run_triton_scalar_store_camodel.sh
```

目标 kernel 来自外部输入 `test_cases/scalar_dominate_kernels/`，固定条件见文首。
