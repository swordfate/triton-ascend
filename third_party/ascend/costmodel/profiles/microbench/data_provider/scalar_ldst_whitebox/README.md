# scalar load/store CostModel 白盒建模与验证

> **本 PR 代码改动统计（相对 base `1f2666afb`）**
> - 核心 costmodel 代码：6 files，+182 / −11
> - profile + schema：2 files，+83 / −2
> - CAModel/CCE 微基准与标定资产：27 files，+1981 / −0
> - 合计：35 files，+2246 / −13
> 核心代码约占新增行数的 8.1%，其余为可复现的标定微基准程序和结果表。

## 1. 目标场景

### 1.1 建模的 4 种情况

| 场景 | 形态 |
|---|---|
| SIMD/SIMT scalar load | 冷 line、每 Stage K=1 或 2 条独立 load |
| SIMT scalar store | `SIMT_STG` 写 128B line，target 为单 store |
| SIMD scalar store | Triton scalar `tt.store` → MTE3 UB→OUT |

---

## 2. 标定微基准与公式建模

在scalar场景下，camodel给出的cycle和sys_cnt实测的cycle误差<10%，还是比较准的；因此，下面打算基于camodel的白盒指令进行建模，最后在实际算子上测试建模误差；

### 2.1 CAModel 时钟确认

用最简单的 CCE scalar CAModel 程序，把 CAModel 打印的 `duration_time(us)` 与 `instr_log` cycle span 对齐：

| launch | core | dump cycle span | duration (us) | cycles/us | 等效 GHz |
|---|---|---:|---:|---:|---:|
| load | core0.veccore0 | 10906 | 6.06 | 1799.7 | 1.800 |
| store | core0.veccore0 | 8786 | 4.88 | 1800.4 | 1.800 |

结论：本次 CAModel 运行时使用的 soc-version 是 Ascend950PR_9599, simulator 使用的内部 cycle → time 换算为 1.8 GHz（虽然本机物理卡的使用的计算时钟是 1.65GHz）。因此白盒建模时从 CAModel 中采集的时钟周期数在写入 profile json 时需要转化：`T_sys = T_camodel × 988.9 / 1800 = T_camodel × 0.5493889`
### 2.2 标定微基准程序

| 场景 | 文件 / 函数 | 用途 |
|---|---|---|
| SIMD/SIMT 单 scalar load | `load/scalar_o1/load_scalar_o1.cce`：`simd_main_ld_o1`、`simt_ld_uniform_o1` | 固定 prep + line fill 成本 |
| SIMD/SIMT 多 op load | `load/scalar_o4/load_scalar_o4.cce`：`simd_main_ld_same_o4`、`simd_main_ld_diff_o4`、`simt_ld_uniform_same_o4`、`simt_ld_uniform_diff_o4` | 拆分 hit/outstanding 与 issue 成本 |
| SIMT scalar store | `store/scalar_o1/store_scalar_o1.cce`：`simt_st_uniform_o1` | 拟合单 store window |
| SIMD Triton scalar store | `store/triton_scalar_store/triton_scalar_store_demo.py`：`_triton_scalar_store_demo` | 确认 MTE3 路径并拟合 stage 系数 |

### 2.3 公式构造

**A. SIMD MainScalar load**

从 o1 load 和 o4 same/diff 三个点拆出：

- `prep = 7`：issue→tag/MSHR/BIU dispatch（4）+ refill→retire（3）；
- `fill = 440`：一次 cold 64B line fill（DC 发 BIU 8 + BIU 读 421 + 回 DC 11）；
- `issue = 3`：每条额外 op 的 issue/serialization；
- `outstanding = 2`：MSHR 同时在途的 line 数；超出部分每条不同 line 额外付一次 line fill。

```text
extra(K) = K <= 4 ? 250 : 350
T_simd_load(K) = 7 + 440 
                + max(0, K - 2) × extra(K) // K>2 的 op 要付出的从GM load cycle
                + (K - 1) × 3              // K>1 的 op issue/serialization cycle

具体计算结果：K=1: 447、K=2: 450、K=3: 703、K=4: 956、K>4: 按 extra=350 外推
```

profile 落盘值：`main_load_prep=3.8457`、`main_load_fill=241.7311`、`main_load_issue=1.6482`、
`main_load_outstanding_line_count=2`、`main_load_extra_line_low=137.3472`、
`main_load_extra_line_high=192.2861`、`main_load_extra_line_high_threshold=4`（SYS_CNT cycle/计数）。

**B. SIMT warp-uniform load**

CCE 单条 load 微基准程序 + 6 kernel 联合拟合取 `fill = 480`（填充到 DCache 是 128B）：

```text
T_simt_load(K) = 6 + 480 + (K - 1) × 1

具体计算结果：K=1: 486、K=2: 487、K=4: 489（diff-line 线性外推）
```

`1` 表示不同 128B line 的额外 load 只付 LSU issue 地板。

profile 落盘值：`uniform_load_prep=6×0.5493889=3.2963`、`uniform_load_fill=480×0.5493889=263.7067`、`uniform_load_diff_line_issue=1×0.5493889=0.54939`。

**C. scalar store**

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

---

## 3. costmodel 代码修改

### 3.1 Stage 识别与工作量

- `StageWorkload` 增加 `scalarLoadCount` / `scalarStoreCount`（每 iteration 的 scalar op 数 K）；
- `StagePartitioner` 对非 shaped 的 `tt.load` / `tt.store` 单独计数，并分类为 `ScalarLoad` / `ScalarStore`；
- scalar 访存不再进入 contiguous memory 等错误的 workload 与成本计算路径。

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
| `uniform_load_diff_line_issue_system_cycles` | 0.54939 | CAModel 1 core cycle |
| `uniform_store_base_system_cycles` | 247.2250 | CAModel 450 core cycle |


### 3.3 costmodel 代码修改路径

| 文件 | 修改 |
|---|---|
| `include/.../StageCostModels.h` | 新增 `ScalarLoad` / `ScalarStore` kind 与 profile 字段 |
| `include/.../StageRouteCostModel.h` | `StageWorkload` 增加 scalar load/store count |
| `lib/.../Analysis/StagePartitioner.cpp` | scalar 识别、计数、分类 |
| `lib/.../RouteModel/StageCostModels.cpp` | `mapWorkload` 按 mode 加入白盒公式 |
| `lib/.../RouteModel/StageRouteCostModel.cpp` | workload 校验与 report JSON |
| `lib/.../RouteModel/SimdSimtCostModel.cpp` | 读取可选 `scalar_memory` 字段 |

---

## 4. scalar ld/st 在实际算子中的误差

### 4.1 SIMT

| kernel | 类别 | costmodel ns | CAModel ns | err |
|---|---|---:|---:|---:|
| padded_copy_gather | scalar load | 540.0 | 618.3 | −12.7% |
| padded_copy_scatter | scalar load | 540.0 | 619.4 | −12.8% |
| padded_copy_wgrad | scalar load | 540.0 | 498.3 | +8.4% |
| padded_copy_wgrad | scalar store | 250.0 | 310.6 | −19.5% |
| binned_copy_gather | scalar load | 270.0 | 276.1 | −2.2% |
| binned_copy_scatter | scalar load | 270.0 | 276.1 | −2.2% |
| binned_copy_wgrad | scalar load | 270.0 | 245.6 | +10.0% |
| binned_copy_wgrad | scalar store | 250.0 | 306.1 | −18.3% |

### 4.2 SIMD

| kernel | 类别 | costmodel ns | CAModel ns | err |
|---|---|---:|---:|---:|
| padded_copy_gather | scalar load | 496.7 | 490.6 | +1.2% |
| padded_copy_scatter | scalar load | 496.7 | 561.7 | −11.6% |
| padded_copy_wgrad | scalar load | 496.7 | 445.0 | +11.6% |
| padded_copy_wgrad | scalar store | 261.1 | 275.6 | −5.2% |
| binned_copy_gather | scalar load | 248.3 | 202.2 | +22.8% |
| binned_copy_scatter | scalar load | 248.3 | 263.9 | −5.9% |
| binned_copy_wgrad | scalar load | 248.3 | 283.3 | −12.4% |
| binned_copy_wgrad | scalar store | 261.1 | 229.4 | +13.8% |

### 4.3 MAPE（平均误差统计）

| 类别 | **SIMT MAPE** | SIMT 最大/最小误差 | **SIMD MAPE** | SIMD 最大/最小误差 |
|---|---:|---:|---:|---:|
|  scalar load | 8.0% | +10.0% / −12.8% | 10.9% | +22.8% / −12.4% |
| scalar store | 18.9% | −18.3% / −19.5% | 9.5% | +13.8% / −5.2% |


---

## 5. 文件与复现

```text
scalar_ldst_whitebox/
  README.md                           # 本文
  load/scalar_o1/                     # SIMD/SIMT 单 scalar load 微基准程序
  load/scalar_o4/                     # same/diff 4-op load 微基准程序
  store/scalar_o1/                    # SIMT 单 scalar store 微基准程序
  store/triton_scalar_store/          # Triton scalar store MTE3 demo
```

