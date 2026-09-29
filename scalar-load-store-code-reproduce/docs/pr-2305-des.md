# scalar load/store CostModel 白盒建模与验证

> 当前 commit：`c1f401da6`（已把 latest 2 commits squash 成 1 个：SIMT load issue 项、SIMT store 改名 subsequent、K=16 store 外推复核）。


## 1. 目标场景

### 1.1 建模的 4 种情况

| 场景 | 形态 |
|---|---|
| SIMD scalar load | 每 Stage 1 或多条独立 load（diff-line） |
| SIMT scalar load | 每 Stage 1 或多条独立 load（diff-line） |
| SIMD scalar store | Triton scalar `tt.store` → MTE3 UB→OUT |
| SIMT scalar store | `SIMT_STG` 写 128B line，target 为单 store |

---

## 2. 标定微基准与公式建模

load 场景下 CAModel cycle 与真卡 microbench 的误差约 ±10%，因此下面保持 CAModel 白盒的 cycle 形式，用 2026-09-27 板级 diff-line 多 ptr K-sweep（K=1/2/4/8）重标系数；store 的 board 可见窗口差异较大，直接按同一批 K-sweep 的 K=1/K=8 端点重标。第 4 节给出实测误差。

### 2.1 CAModel 时钟确认

用最简单的 CCE scalar CAModel 程序，把 CAModel 打印的 `duration_time(us)` 与 `instr_log` cycle span 对齐：

| launch | core | dump cycle span | duration (us) | cycles/us | 等效 GHz |
|---|---|---:|---:|---:|---:|
| load | core0.veccore0 | 10906 | 6.06 | 1799.7 | 1.800 |
| store | core0.veccore0 | 8786 | 4.88 | 1800.4 | 1.800 |

结论：CAModel 内部 cycle → time 换算为 **1.8 GHz**（物理卡计算时钟是 1.65 GHz）。profile 的 `*_system_cycles` 是 SYS_CNT 域（988.9 MHz），所以写盘时：

```text
T_sys = T_camodel × 988.9 / 1800 = T_camodel × 0.5493889
```

### 2.2 标定微基准程序

| 场景 | 文件 / 函数 | 用途 |
|---|---|---|
| SIMD/SIMT 单 scalar load | `load/scalar_o1/load_scalar_o1.cce`：`simd_main_ld_o1`、`simt_ld_uniform_o1` | 固定 prep + line fill 成本 |
| SIMD/SIMT 多 op load | `load/scalar_o4/load_scalar_o4.cce`：`simd_main_ld_same_o4`、`simd_main_ld_diff_o4`、`simt_ld_uniform_same_o4`、`simt_ld_uniform_diff_o4` | 拆分 hit/outstanding 与 issue 成本 |
| SIMT scalar store | `store/scalar_o1/store_scalar_o1.cce`：`simt_st_uniform_o1` | 拟合单 store window |
| SIMD Triton scalar store | `store/triton_scalar_store/triton_scalar_store_demo.py`：`_triton_scalar_store_demo` | 确认 MTE3 路径并拟合 stage 系数 |

板级重标数据：`triton_probe/profiler_pair_probe.py` + `triton_probe/results/{simd,simt}_{load,store}.json`；K=1/2/4/8，`measured_cycle = measured_us × 1000 × 1.8`。

2026-09-28 store retest：`triton_probe/store_probe_variants.py`（matched sink baseline + readback/barrier 变体）。
旧 store probe 的 target 没有 baseline 的 sink store，导致 K=2/K=4 负值；修正后：
SIMT store 用 matched-sink `Duration(us)_delta`，SIMD store 用 `aiv_mte3_time(us)_delta`
（异步 MTE3，kernel Duration 看不到完成）。这里的 **ScalarStore resource 表示资源占用/吞吐，不含完成/可见延迟**；
readback/barrier 变体只用于暴露 producer→consumer 可见性，归 dependency/synchronization 语义。

### 2.3 公式构造（2026-09-27 重标）

**A. SIMD MainScalar load**

```text
T_simd_load(K) = prep + fill + max(0, K - outstanding) * extra + (K - 1) * issue

prep = 7, fill = 263.9, outstanding = 1, extra = 278.74, issue = 1
K=1: 270.9, K=2: 550.6, K=4: 1110.1, K=8: 2229.1
```

- `prep + fill = 270.9` 锚定 K=1；
- K=8 端点给出 `extra = (2222.1 - 270.9) / 7 - 1 = 278.74`；
- K=2/K=4 检查点误差 −6.0% / −9.9%；
- `outstanding=1`：板级窗口下第 2 条不同 line 已付完整额外 line 成本。

profile 落盘值：`main_load_prep=3.8457`、`main_load_fill=144.9837`、`main_load_issue=0.5494`、
`main_load_outstanding_line_count=1`、`main_load_extra_line_low=high=153.1382`、
`main_load_extra_line_high_threshold=4`（SYS_CNT cycle）。

**B. SIMT warp-uniform load（threshold 形式）**

```text
T_simt_load(K) = prep + fill + max(0, K - threshold) * extra + (K - 1) * issue

prep = 6, fill = 306.75, threshold = 2, extra = 74.775, issue = 1
K=1: 312.8, K=2: 313.8, K=4: 465.3, K=8: 768.4
```

- K=1/K=2 基本持平，共用 base `prep + fill = 312.75`；
- 超过 threshold=2 后每条额外 128B line 的可见边际为 74.775 cycle；
- 每条额外 op 再加 issue=1（与 SIMD MainScalar load 的 issue 项对齐）；
- K=8 锚定，K=4 检查误差 −13.2%，全部点 ≤20%。

profile 落盘值：`uniform_load_prep=3.2963`、`uniform_load_fill=168.525`、
`uniform_load_diff_line_threshold=2`、`uniform_load_diff_line_extra=41.0806`、
`uniform_load_issue=0.5494`（SYS_CNT cycle）。

**C. scalar store（2026-09-28 matched-sink retest，K=8 异常已复核）**

旧 2026-09-27 store probe 的 target 缺少 baseline 的 sink `tl.store`，导致 K=2/K=4 出现负值。
2026-09-28 重测把 target/baseline 的 sink 对齐，并按指标选择公式：

```text
SIMD scalar store（aiv_mte3_time 窗口）:
  K == 1 : T = fill
  K >= 2 : T = fill + activation + (K - 2) * subsequent
  fill = 11.7, activation = 95.4, subsequent = 4.5
  K=1/2/4/8 = 11.7 / 107.1 / 116.1 / 134.1 cycle

SIMT scalar store（matched-sink Duration 窗口）:
  T = base + (K - 1) * subsequent
  base = 30.4, subsequent = 12.348
  K=1/2/4/8 = 30.4 / 42.7 / 67.4 / 116.8 cycle
```

- **K=8 异常复核**：原 run SIMT K=8=229 ns，三个 round 稳定在 222/236 ns；
  但同一脚本 fresh 重跑三次为 66.5 / 57.5 / 61.5 ns，matched K=8 探针 61.5 ns。
  因此判定原 run 是运行状态异常，不是 probe 程序问题；本版 K=8 使用 fresh 中位数 61.5 ns。
- SIMD store 是异步 MTE3：kernel `Duration` 看不到完成；必须用 `aiv_mte3_time`。
  该窗口衡量的是 **MTE3 pipe active 资源占用**，不含 scalar→UB staging、等 flag、write ack 完成语义。
  单条 store 很小，第 2 条触发写路径 activation，之后每条额外 store 约 4.5 cycle。
  目标 wgrad 的 store 结果不被消费，因此资源占用语义与目标场景一致；强制可见的 readback/barrier 成本
  （SIMD K=1 ~96 ns、SIMT K=1 ~78.5 ns）另有同步/dependency 语义，不叠加到 store resource。
- SIMT store 用 matched-sink `Duration`，跨 run 中位数拟合线性 base+subsequent；
  K=1/2/4/8 最大相对误差 ±5.3%。
- SIMD store K=1/2/4/8 最大相对误差 −6.2%。

profile 落盘值：`mte3_scalar_store_fill=6.4279`、
`mte3_scalar_store_serial=52.4117`、`mte3_scalar_store_subsequent=2.4723`；
`uniform_store_base=16.7014`、`uniform_store_subsequent=6.7839`（SYS_CNT cycle）。

---

## 3. costmodel 代码修改

### 3.1 Stage 识别与工作量

- `StageWorkload` 增加 `scalarLoadCount` / `scalarStoreCount`（每 iteration 的 scalar op 数 K）；
- `StagePartitioner` 对非 shaped 的 `tt.load` / `tt.store` 单独计数，并分类为 `ScalarLoad` / `ScalarStore`；
- scalar 访存不再进入 contiguous memory 等错误的 workload 与成本计算路径。

### 3.2 profile 字段

以下值为 raw cycle × 0.5493889（= 988.9/1800）后的 SYS_CNT cycle。

simd `stage_resources.scalar_memory`：

| 字段 | 值（SYS_CNT cycle） | 来源（raw cycle） |
|---|---:|---|
| `main_load_prep_system_cycles` | 3.8457 | 7 |
| `main_load_fill_system_cycles` | 144.9837 | 263.9 |
| `main_load_issue_system_cycles` | 0.5494 | 1 |
| `main_load_outstanding_line_count` | 1 | 1 |
| `main_load_extra_line_low_system_cycles` | 153.1382 | 278.74 |
| `main_load_extra_line_high_system_cycles` | 153.1382 | 278.74 |
| `main_load_extra_line_high_threshold` | 4 | 4 |
| `mte3_scalar_store_fill_system_cycles` | 6.4279 | 11.7 |
| `mte3_scalar_store_serial_system_cycles` | 52.4117 | 95.4 |
| `mte3_scalar_store_subsequent_system_cycles` | 2.4723 | 4.5 |

simt `stage_resources.scalar_memory`：

| 字段 | 值（SYS_CNT cycle） | 来源（raw cycle） |
|---|---:|---|
| `uniform_load_prep_system_cycles` | 3.2963 | 6 |
| `uniform_load_fill_system_cycles` | 168.525 | 306.75 |
| `uniform_load_diff_line_threshold` | 2 | 2 |
| `uniform_load_diff_line_extra_system_cycles` | 41.0806 | 74.775 |
| `uniform_load_issue_system_cycles` | 0.5494 | 1 |
| `uniform_store_base_system_cycles` | 16.7014 | 30.4 |
| `uniform_store_subsequent_system_cycles` | 6.7839 | 12.348 |

### 3.3 costmodel 代码修改路径

| 文件 | 修改 |
|---|---|
| `include/.../StageCostModels.h` | 新增 `ScalarLoad` / `ScalarStore` kind 与 profile 字段，含 SIMT load `threshold`、MTE3 store `subsequent` |
| `include/.../StageRouteCostModel.h` | `StageWorkload` 增加 scalar load/store count |
| `lib/.../Analysis/StagePartitioner.cpp` | scalar 识别、计数、分类 |
| `lib/.../RouteModel/StageCostModels.cpp` | `mapWorkload` 按 mode 加入白盒公式；SIMT load 使用 threshold + issue，SIMD store 使用 piecewise activation，SIMT store 线性 base+subsequent |
| `lib/.../RouteModel/StageRouteCostModel.cpp` | workload 校验与 report JSON |
| `lib/.../RouteModel/SimdSimtCostModel.cpp` | 读取可选 `scalar_memory` 字段，含 `uniform_load_diff_line_threshold`、`uniform_load_issue`、MTE3 `subsequent` |

---

## 4. 与实测微基准的误差

> **2026-09-27/28 多 ptr K-sweep + store retest**：8 个独立 ptr tensor、手写展开 K 条 `ptr + pid`、K=1/2/4/8。
> load 与 SIMT store 用 `Duration(us)_delta`；SIMD store 用 `aiv_mte3_time(us)_delta`（异步 MTE3，Duration 无信号）。
> store 已改为 matched-sink probe；原 SIMT K=8=229 ns 经 fresh 重跑（66.5/57.5/61.5 ns）判定为运行状态异常并排除，
> 本表 K=8 用 fresh 中位数；load 与 store 共 16 个点全部 <20%。
> 另做了 K=16 store 外推复核，见 §4.1；target 只用 K=1 store，K=16 不参与系数重拟合。

| mode | kind | K | 指标 | 实测 ns | 实测 cycle | 公式 cycle | 误差 |
|---|---|---:|---|---:|---:|---:|---:|
| simd | load | 1 | Duration | 150.5 | 270.9 | 270.9 | +0.0% |
| simd | load | 2 | Duration | 287.5 | 517.5 | 550.6 | −6.0% |
| simd | load | 4 | Duration | 555.5 | 999.9 | 1110.1 | −9.9% |
| simd | load | 8 | Duration | 1234.5 | 2222.1 | 2229.1 | −0.3% |
| simt_only | load | 1 | Duration | 177.0 | 318.6 | 312.8 | +1.9% |
| simt_only | load | 2 | Duration | 170.5 | 306.9 | 313.8 | −2.2% |
| simt_only | load | 4 | Duration | 224.5 | 404.1 | 465.3 | −13.2% |
| simt_only | load | 8 | Duration | 423.0 | 761.4 | 768.4 | −0.9% |
| simd | store | 1 | aiv_mte3_time | 6.5 | 11.7 | 11.7 | +0.0% |
| simd | store | 2 | aiv_mte3_time | 59.5 | 107.1 | 107.1 | +0.0% |
| simd | store | 4 | aiv_mte3_time | 60.5 | 108.9 | 116.1 | −6.2% |
| simd | store | 8 | aiv_mte3_time | 74.5 | 134.1 | 134.1 | +0.0% |
| simt_only | store | 1 | matched Duration | 16.0 | 28.8 | 30.4 | −5.3% |
| simt_only | store | 2 | matched Duration | 25.0 | 45.0 | 42.7 | +5.3% |
| simt_only | store | 4 | matched Duration | 37.0 | 66.6 | 67.4 | −1.3% |
| simt_only | store | 8 | matched Duration | 61.5 | 110.7 | 116.8 | −5.3% |

> **SIMD store 口径（已决）**：SIMD scalar store 用 `aiv_mte3_time`（MTE3 pipe active）表示
> **资源占用/吞吐**，不是完成/可见延迟。目标 6 kernel 中只有 wgrad 的 `tl.store(wgrad,out)`，结果不被消费，
> 与资源占用语义匹配。强制可见 readback/barrier 变体（SIMD K=1 ~96 ns、SIMT K=1 ~78.5 ns）测的是
> producer→consumer 可见性/同步，应走 dependency/synchronization/scope-handoff 类成本；
> 当前不把 completion floor 加到 store resource 上，避免对 target 的最后一个 dead store 双重收费。
> 若未来 kernel 读回/依赖 store 结果，需要另行增加 completion/visibility 项。
> 证据见 `triton_probe/store_recheck_results/sync_recheck_variant12.json` 与 `HANDOFF.md` §4。

### 4.1 K=16 store 外推复核（2026-09-28）

用同一 matched-sink probe 扩展到 16 个 ptr 后做三组独立 run（NPU1，matched-sink variant 0）：

| mode | 指标 | run1 | run2 | run3 | 中位数 | 公式 cycle | 误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| SIMD scalar store | `aiv_mte3_time(us)_delta` | 0.124 | 0.120 | 0.122 | 0.122 us / 219.6 cyc | 170.1 | **+29.1%** |
| SIMT scalar store | matched-sink `Duration(us)_delta` | 0.150 | 0.159 | 0.157 | 0.157 us / 282.6 cyc | 215.6 | **+31.1%** |

- **方向确认**：K=16 时 SIMT（157 ns）确实高于 SIMD MTE3 active（122 ns）；K=8 时 SIMT（61.5 ns）低于 SIMD（74.5 ns），
  交叉约在 K=12–14。`subsequent_simt=12.348 > subsequent_simd=4.5` 不是符号错误。
- 但两者指标域不同：SIMD 是 MTE3 pipe active 资源占用；SIMT matched-sink Duration 含 SIMT_STG 写/响应窗口，
  不能直接横向比较 subsequent 数值。
- K=16 超出 K≤8 拟合范围，两 mode 都比公式高约 +29%/+31%；target 只有 K=1 store，因此不重拟合 target-used 系数。

> 原始 JSON：`triton_probe/k16_results/k16_run1.json`、`k16_repeat1.json`、`k16_repeat2.json`；扩展探针：`triton_probe/store_probe_variants_k16.py`。

### UT:
之前 https://github.com/triton-lang/triton-ascend/pull/1989 的提交中已经提交了 scalar 主导的算子 UT：
`third_party/ascend/unittest/pytest_ut/test_scalar_dominate_costmodel_cases.py`
