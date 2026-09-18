# Scalar load/store white-box calibration + CAModel verification (padded family)

> 目录名：`scalar_load_store_whitebox`（原 `scalar_load_whitebox` 已重命名；load 与 store 的标定、集成和 CAModel 验证统一放在本目录）。

本目录是 scalar load/store 白盒建模的标定、costmodel 集成与 CAModel 验证归档：

- **标定**：`scalar_load_probe.cce` / `scalar_load_diff_probe.cce` 标定 SIMD
  MainScalar 与 SIMT warp-uniform scalar load；store 公式参数沿用
  `SCALAR-MODEL.md` 的 CAModel 标定结果，并由 profile/schema 固化；
- **集成**：indirect exposure 去重、MainScalar / SIMT scalar store 白盒公式
  已进入 costmodel（load exposure `e6cf5c41a`/`accf0ea679`，store 白盒
  `fb205f0be`/`743e1bffb4`）；
- **验证**：用 tiny `npu_padded_copy_{gather,scatter,wgrad}` 的 costmodel
  report 对 CAModel active window 复核；脚本、report、结果 CSV、关键 dump
  都在 `verification/`。

覆盖范围：

1. `SIMD MainScalar` scalar load（same-line / diff-line / indirect）；
2. `SIMT warp-uniform` scalar load（same-line / diff-line / fan-out exposure）；
3. `SIMD MainScalar` / `SIMT warp-uniform` scalar store 白盒公式；
4. 三个 padded kernel 的 CAModel 验证与 profile 参数 sweep。

> 不包含 strided / contig 等无关轮次；只归档本轮标定、集成和验证直接使用的代码与数据。

> 旧的 cheap-hash CCE 探针测的是“每个 op 换一条 64B line”的随机地址循环，
> 属于 diff-line/warm-throughput 口径；新的 `scalar_load_probe.cce` 和
> `scalar_load_diff_probe.cce` 给出 CAModel active-cycle 口径。两者分别对应当前
> cost model 的 legacy fallback 和结构化 diff-line 分支。

---

## 1. 标定 kernel

### 1.1 same-line：`scalar_load_probe.cce`

| kernel | 说明 |
|---|---|
| `simd_main_ld_o1/o2/o4/o8` | MainScalar GM load，全部落在同一条 64B line |
| `simt_ld_uniform_o1/o2/o4/o8` | 1 warp，32 lane 同地址，全部落在同一条 128B line |

### 1.2 same vs diff：`scalar_load_diff_probe.cce`

Round 6 的 same-line / diff-line 对照（文件从
`63-workspace/10-scalar-camodel-bench/scalar_bench6.cce` 归档）：

```text
simd_main_ld_same_o8
simd_main_ld_diff_o1/o2/o4/o8      # 每条 load 间隔 64B，各占一条 line
simt_ld_uniform_same_o8
simt_ld_uniform_diff_o1/o2/o4/o8   # 每条 LDG 间隔 128B，各占一条 line
```

### 1.3 legacy cheap-hash：`../scalar_ldst/`

| 文件 | 用途 |
|---|---|
| `scalar_ldst/simd_scalar_gm_memory.cce` | MainScalar cheap-hash random 64B load/store |
| `scalar_ldst/simt_scalar_gm_memory.cce` | SIMT warp-uniform random 64B load/store |
| `scalar_ldst/simd_scalar_gm_dep.cce` | MainScalar indirect dependency |
| `scalar_ldst/simt_scalar_gm_dep.cce` | SIMT indirect dependency |
| `scalar_ldst/README.md` | 端到端标定/填 profile 流程和拟合系数 |

### 1.4 运行

同目录的 `scalar_load_probe_host.cpp` 和 `scalar_load_probe_runner.sh` 使用
`aclrtBinaryLoad/GetFunction + aclrtLaunchKernelWithHostArgs`，可以直接跑以上
kernel（修改 `.o` 和 kernel 名称列表即可）。

```bash
source ~/env_ascend.sh
INC=~/AscendNPU-IR-triton/bishengir/lib/Template/include
ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" scalar_load_probe.cce -o scalar_load_probe.o
g++ -O2 scalar_load_probe_host.cpp -o scalar_load_probe_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl

ulimit -n 1048576
msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=8 --timeout=10 \
  ./scalar_load_probe_runner.sh scalar_load_probe.o \
  simd_main_ld_o1 simd_main_ld_o2 simd_main_ld_o4 simd_main_ld_o8 \
  simt_ld_uniform_o1 simt_ld_uniform_o2 simt_ld_uniform_o4 simt_ld_uniform_o8
```

`scalar_load_diff_probe.cce` 的 kernel 更多，按 HANDOFF 提示建议分批跑。

---

## 2. 白盒公式

`K` = 一个 Stage iteration 内 scalar load 的条数；
`U` = 这些 load 落在多少条不同 cache line 上（`1 <= U <= K`；
无法证明 same-line 时默认 `U = K`）。

### 2.1 SIMD MainScalar load（64B line）

```text
T_main_load(K, U)
  = C_MAIN_PREP + C_MAIN_FILL
    + max(0, U - M_MAIN_LOAD) * C_MAIN_LINE(U)
    + (K - U) * C_MAIN_HIT
    + (K - 1) * C_MAIN_ISSUE

C_MAIN_LINE(U) = C_MAIN_LINE_LO   (U <= 4)
                 C_MAIN_LINE_HI   (U >  4)
```

| 参数 | 值 | profile 字段 |
|---|---:|---|
| `C_MAIN_PREP` | 30 | `main_load_prep_system_cycles` |
| `C_MAIN_FILL` | 450 | `main_load_fill_system_cycles` |
| `C_MAIN_HIT` | 4 | `main_load_hit_system_cycles` |
| `C_MAIN_ISSUE` | 1 | `main_load_issue_system_cycles` |
| `M_MAIN_LOAD` | 2 | `main_load_outstanding_line_count` |
| `C_MAIN_LINE_LO` | 250 | `main_load_extra_line_low_system_cycles` |
| `C_MAIN_LINE_HI` | 350 | `main_load_extra_line_high_system_cycles` |
| high threshold | 4 | `main_load_extra_line_high_threshold` |

same-line 是 `U=1` 的特例：`30 + 450 + (K-1)*(4+1)`。

### 2.2 SIMT warp-uniform load（128B line）

```text
same-line: T = C_SIMT_PREP + C_SIMT_FILL + (K-1) * C_SIMT_SAME_LINE_SERIAL
diff-line: T = C_SIMT_PREP + C_SIMT_FILL + (K-1) * C_SIMT_DIFF_LINE_ISSUE
```

| 参数 | 值 | profile 字段 |
|---|---:|---|
| `C_SIMT_PREP` | 50 | `uniform_load_prep_system_cycles` |
| `C_SIMT_FILL` | 500 | `uniform_load_fill_system_cycles` |
| `C_SIMT_SAME_LINE_SERIAL` | 450 | `uniform_load_same_line_serial_system_cycles` |
| `C_SIMT_DIFF_LINE_ISSUE` | 2 | `uniform_load_diff_line_issue_system_cycles` |

> CAModel 里 same-line 重复 LDG 不给 hit/merge，所以同 line 第 2 条起按 450
> cycles/条串行；真卡 SYS_CNT marginal 表明真卡 DCache 会 hit，后续校准需要用
> `63-workspace/10-scalar-camodel-bench/board_syscnt/BOARD-VS-MODEL.md` 的
> board-visible 系数替换本分支参数。

### 2.3 legacy scalar_ldst fallback

当 profile 未提供结构化白盒字段时，保留旧探针的拟合：

```text
cycles = a + b*warps + c*ops + d*warps*ops
throughput = warps*ops / cycles
cost = ops / throughput + load_latency
```

Legacy 系数（SYS_CNT 域，warm/runtime-loop）：

```text
SIMT load_cycles_fit  = [65.523189, 2.048849, 7.893464, 0.374104]
SIMT store_cycles_fit = [24.769761, -1.298297, 1.818695, 1.676496]
SIMD cheap-hash load throughput = 0.01104935915300334
SIMD load direct latency        = 12.790396391
```

这些数值写在 `data_provider/scalar_ldst/README.md`，profile 里的
`load_cycles_fit` / `store_cycles_fit` 字段由 `SimdSimtCostModel.cpp` 读取。
口径说明：legacy fit 是 warm/runtime-loop 的相对吞吐口径，结构化白盒字段是
cold-line CAModel active-cycle 口径，二者只在缺少结构化字段时才互换使用。

---

## 3. cost model 集成

### 3.1 workload / feature

- `StageWorkload::scalarLoadCount`：动态 load 条数 `K`。
- `StageWorkload::directScalarLoadCount` / `indirectScalarLoadCount`：
  地址链是否依赖另一个 scalar load（legacy `scalar_ldst` 口径）。
  `indirectScalarLoadCount` 是 **consumer 侧**计数（fan-out 有几条 load
  就数几条），只用于报告和兼容回退。
- `StageWorkload::indirectScalarLoadExposureCount` /
  `indirectScalarStoreExposureCount`：**producer 侧**去重后的依赖暴露次数。
  同一个 producer load 喂给同一 Stage 内多条 consumer load 时只记 1；
  串行链 `P -> C1 -> C2` 记 2。cost model 用这个字段收费。
- `StageWorkload::indirectScalarStoreCount`：地址依赖 scalar load 的 store
  条数（legacy consumer 侧 dependency 口径）。
- `StageWorkload::scalarLoadUniqueLines`：保守默认 = `scalarLoadCount`；
  只有 `StageModelFeatures::scalarLoadsShareLine == true` 时才降为 1。
- `StageModelFeatures::hasScalarIndirectLoad` /
  `hasScalarIndirectStore`：存在 indirect scalar edge 时，
  cost model 额外加
  `indirect_*_exposure_count * scalarIndirectDependencyLatency`；
  exposure count 为 0（未分析/手工构造 workload）时回退 legacy
  `indirect_*_count`，保持旧 UT/兼容路径。
- `StageModelFeatures::scalarLoadsShareLine`：
  - 单条 scalar load 恒为 true（只有一条 line，没有歧义）；
  - 多条 load 时，只有 IR 能证明“同一 base pointer + 常量偏移落在同一个 64B line”
    且不在 loop 内时才为 true；
  - 否则 false，cost model 走 diff-line 分支。

### 3.2 选择逻辑

```text
if scalarLoadCount == 0: return
if scalarLoadsShareLine:
    SIMD -> main load formula with U=1
    SIMT -> uniform same-line formula
else:
    SIMD -> main load formula with U = min(K, scalarLoadUniqueLines)（默认 K）
    SIMT -> uniform diff-line formula
```

结构化字段缺失时回退到 legacy scalar_ldst 拟合/吞吐；SIMT store 也回退到
`store_cycles_fit`。因此在完整 profile 下，same-line、diff-line 和间接依赖
都有明确公式：白盒结构化公式覆盖 line/fill/hit，legacy 字段覆盖缺失字段时的
warm/runtime-loop 兼容路径。

### 3.3 profile 字段

`profiles/simd_simt/david_v100_simd_simt_v1.json`：

- `simd.stage_resources.scalar_memory`：
  `main_load_{prep,fill,hit,issue}_system_cycles`
  `main_load_outstanding_line_count`
  `main_load_extra_line_{low,high}_system_cycles`
  `main_load_extra_line_high_threshold`
  `load_instructions_per_system_cycle` / `load_latency_system_cycles`
  （legacy diff-line warm fallback）
- `simt.stage_resources.scalar_memory`：
  `uniform_load_{prep,fill}_system_cycles`
  `uniform_load_same_line_serial_system_cycles`
  `uniform_load_diff_line_issue_system_cycles`
  `load_cycles_fit` / `store_cycles_fit`
  （legacy uniform random fallback）

---

## 4. 验证数据

### 4.1 same-line（`scalar_load_probe.cce`，CAModel active cycles）

| K | MainScalar 公式(U=1) | MainScalar 实测 | SIMT 公式(U=1) | SIMT 实测 |
|---:|---:|---:|---:|---:|
| 1 | 480 | 447 | 550 | 530 |
| 2 | 485 | 448 | 1000 | 905 |
| 4 | 495 | 537 | 1900 | 1906 |
| 8 | 515 | 531 | 3700 | 3629 |

### 4.2 diff-line（`scalar_load_diff_probe.cce`，CAModel active cycles）

| K | MainScalar 公式(U=K) | MainScalar 实测 | SIMT 公式(diff) | SIMT 实测 |
|---:|---:|---:|---:|---:|
| 1 | 480 | 482 | 550 | 553 |
| 2 | 481 | 466 | 552 | 522 |
| 4 | 981 | 978 | 556 | 563 |
| 8 | 2383 | 2476 | 564 | 572 |

> 这些是 CAModel 单 warp、cold-line、active-window 数据；真实 kernel 的
> cache 状态、warp 数、L2 命中率不同，board 校准见
> `63-workspace/10-scalar-camodel-bench/board_syscnt/BOARD-VS-MODEL.md`。

### 4.3 间接 scalar dependency（legacy `scalar_ldst`）

地址链依赖另一条 scalar load 的 load/store（如 `index_a = tl.load(indices + start + entry_idx)`，
`start` 来自 `bins` 的 scalar load）在 `StageModelFeatures` 中标记
`hasScalarIndirectLoad/Store`，cost model 在 line/fill/hit 成本之外增加：

```text
indirect_scalar_load_exposure_count  * scalarIndirectDependencyLatency
indirect_scalar_store_exposure_count * scalarIndirectDependencyLatency
```

`*_exposure_count` 是 producer 侧去重计数：一个 producer load 喂给同 Stage 内
多条 consumer load 时只收一次（fan-out dedupe），串行链仍按每条 edge 收费。
exposure count 为 0 时回退到 legacy consumer 侧的
`indirect_scalar_*_count * scalarIndirectDependencyLatency`。

profile 字段：`indirect_dependency_latency_system_cycles`（SIMD 1.95、SIMT 65.4，
来自 legacy `scalar_ldst/simd_scalar_gm_dep` 与 `simt_scalar_gm_dep`）。这部分
不替代 same-line/diff-line 白盒公式，而是叠加在它之上的 dependency edge 成本。
latency 本身的标定口径仍需后续用目标 kernel + 真卡 SYS_CNT 复核。

### 4.4 MainScalar / SIMT store 白盒模型（2026-09-17 集成）

`ScalarStore` stage 在 profile 提供结构化 store 字段时，不再走 legacy
`store_cycles_fit` / provisional scalar-pipe rate，而是使用 `SCALAR-MODEL.md`
的 store 公式：

```text
SIMD MainScalar store (U = distinct 64B lines):
  U == 1:  30 + 450 + (K - 1) * 2
  U >= 2:  30 + 450 + extra(U) + (K - 1) * 2
           extra(2)=70, extra(3)=78,
           extra(U>=4)=85 + (U-4)*106

SIMT warp-uniform store:
  same-line (K >= 2): 555 + (K - 1) * 480
  first store / diff-line: 450 + (K - 1) * 20
```

profile 字段：
`main_store_{prep,fill,issue}_system_cycles`、
`main_store_extra_line_{2,3,4,step}_system_cycles`、
`uniform_store_{same_line_base,same_line_serial,diff_line_base,diff_line_issue}_system_cycles`。
K=1 的单条 scalar store 没有重复同 line 访问，SIMT 侧固定走 first-store
（diff-line）分支；same-line 只用于 K>=2 且 IR 证明同 64B/128B line 的 store。
`scalarStoreUniqueLines` / `scalarStoresShareLine` 与 load 侧同名的同 line 证明对应。

---

## 5. CAModel 验证：三个 padded kernel（2026-09-16/17）

### 5.1 固定条件

| 项 | 值 |
|---|---|
| kernel | `npu_padded_copy_gather` / `npu_padded_copy_scatter` / `npu_padded_copy_scatter_wgrad_camodel` |
| shape | `sl=4, hs=2, ne=2, top_k=1` |
| grid | `(4,)` |
| autotune config | 截断到第一个：`BLOCK_X=64, superblock_factor=1, num_warps=1` |
| kernel 参数 | `NUM_COLUMNS=2, TOP_K=1`；gather `A_TO_B=True, SCALE=False`；scatter `A_TO_B=False, SCALE=True`；wgrad 独立入口 |
| hints | `logical_program_count_hint=4`；`physical_vector_core_count_hint=56` |
| 输入 | CPU 构造后 `.to("npu")`；`top_expert=ones(4)`，所有 program 走 `bin_idx>0` |
| costmodel | `TRITON_ASCEND_COMPILE_MODE=simd_simt`, `TRITON_ASCEND_AUTO_SIMT_SCOPE=report` |
| CAModel | `TRITON_ASCEND_COMPILE_MODE=simt_only|simd`, `TRITON_ASCEND_AUTO_SIMT_SCOPE=off`；`msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 --launch-count=1 --kernel-name=<prefix>` |

### 5.2 脚本与命令

以下脚本从原始运行目录 `~/stage_vs_camodel_20260917/` **原样归档** 到
`verification/`（脚本内 `OUT` 仍指向原运行目录；直接在归档目录运行时请先改 `OUT`
或建立同名软链接）：

| 脚本 | 作用 |
|---|---|
| `tiny_padded_suite_runner.py` | 三个 padded kernel 的固定 tiny shape/config runner（costmodel/CAModel 共用） |
| `run_suite_costmodel.sh` | 单 kernel costmodel report（`simd_simt` + `report`） |
| `run_suite_camodel.sh` | 单 kernel CAModel（`simt_only` 或 `simd`） |
| `run_suite_all.sh` | 批量跑 scatter/wgrad 的两种模式 |
| `parse_suite_loads.py` | 从 OPPROF dump 解析 SIMT_LDG / MainScalar load 窗口 |
| `parse_suite_stores.py` | 从 OPPROF dump 解析 SIMT_STG / MTE3 store 窗口 |
| `make_three_kernel_table.py` | 汇总 load+store 对比为 `three_kernel_compare.csv` |
| `make_three_kernel_store_table.py` | 生成 `three_kernel_store_compare.csv` |
| `inspect_store_stages.py` | 打印三个 report 里的 ScalarStore/tile store stage |
| `make_candidate_profiles.py` | 生成 profile 参数候选 |
| `eval_candidates.py` / `write_candidate_mape.py` | 用同一套 CAModel 窗口评估候选参数 |
| `run_candidate_costmodel.sh` / `run_candidates.sh` | 候选 profile 的 costmodel report 批跑 |

实际使用的命令：

```bash
# wheel：源码 commit fb205f0be / server 743e1bffb4，构建后安装
bash ~/build_triton_whitebox_wheel.sh
pip install ~/triton-whitebox-latest/dist/triton_ascend-*.whl --force-reinstall --no-deps

# 1) costmodel report
bash run_suite_costmodel.sh padded_copy_gather
bash run_suite_costmodel.sh padded_copy_scatter
bash run_suite_costmodel.sh padded_copy_wgrad

# 2) CAModel：每个 kernel 跑 simt_only 和 simd
bash run_suite_camodel.sh padded_copy_gather simt_only
bash run_suite_camodel.sh padded_copy_gather simd
bash run_suite_camodel.sh padded_copy_scatter simt_only
bash run_suite_camodel.sh padded_copy_scatter simd
bash run_suite_camodel.sh padded_copy_wgrad simt_only
bash run_suite_camodel.sh padded_copy_wgrad simd

# 3) 解析窗口并生成对比表
python3 parse_suite_loads.py <OPPROF>/dump simt
python3 parse_suite_stores.py <OPPROF>/dump simt
python3 make_three_kernel_store_table.py
python3 make_three_kernel_table.py
python3 inspect_store_stages.py

# 4) 参数 sweep
bash run_candidates.sh
python3 eval_candidates.py
python3 write_candidate_mape.py
```

### 5.3 Load 验证结论

完整数据：`verification/results/three_kernel_compare.csv`。

- SIMT 当前 profile：**MAPE 18.1%，bias +17.5%**；误差集中在 fan-out
  `stage_7`（+13.1%～+48.7%）和 scatter 的 `stage_11` weights load（+40.6%）。
- SIMD 当前 profile：**MAPE 13.5%，bias +9.4%**；`stage_7` pair 在 −7.2%～+0.8%。
- 单条 direct K=1：SIMT −1.6%～+15.8%，SIMD −9.6%～+29.6%（SIMD 的 LDP
  融合点参考性弱）。
- 结论：fan-out exposure 去重已经生效；残余高估主要来自
  `base(K=2) + exposure × latency` 的简单相加，以及 SIMD LDP 把一对 load
  合成一条指令。

### 5.4 ScalarStore 验证结论

wgrad `stage_22_scalar_store`（完整数据：`three_kernel_store_compare.csv`）：

| mode | model base | CAModel | base err | 加 indirect store exposure 后 |
|---|---:|---:|---:|---:|
| SIMT | 450 | 420（`SIMT_STG` id867） | **+7.1%** | 515.4，err **+22.7%** |
| SIMD | 480 | 436（MTE3 UB→OUT id1258） | **+10.1%** | 481.95，err **+10.5%** |

- ScalarStore 白盒 base 已经可用（SIMT +7.1%、SIMD +10.1%）。
- SIMT total 的 +22.7% 来自现有 `indirect_scalar_store_exposure × 65.4`：
  wgrad store 的地址依赖在 loop/reduction 里早被隐藏，当前模型仍收费；
  这是 dependency 暴露口径问题，不是 store base 公式的问题。

### 5.5 tile store 参考（不在 ScalarStore 公式内）

gather/scatter 的 for-loop store 是 shaped tile store，stage 为
`IndirectGatherMemory`，当前仍走 indirect transaction 模型：

| mode | kernel | stage | costmodel store resource | CAModel | err |
|---|---|---:|---:|---:|---:|
| SIMT | gather | stage_15 | 4 | 458 | −99.1% |
| SIMD | gather | stage_15 | 16 | 1171 | −98.6% |
| SIMT | scatter | stage_17 | 4 | 525 | −99.2% |
| SIMD | scatter | stage_17 | 16 | 1185 | −98.6% |

结论：tile store 与 ScalarStore 不在同一口径，需要单独建模，不能直接套本节公式。

### 5.6 profile 参数 sweep

脚本：`make_candidate_profiles.py` / `run_candidates.sh` / `eval_candidates.py` /
`write_candidate_mape.py`；结果 `verification/results/candidate_mape.csv`。

- 三个 tiny kernel 上，`combined_470_48_430_10`（SIMT fill 500→470、
  indirect 65.4→48；SIMD fill 450→430、indirect 1.95→10）把 avg MAPE
  15.8% → 12.6%，但属于目标 kernel 定向调参；
- 把 dedicated white-box probe 一起拟合后：SIMD 当前参数已接近联合最优
  （9.5% vs 9.7%），不建议改；SIMT 联合最优仍把 indirect latency 推向 0，
  与 legacy serial-chain dependency probe 冲突，不能直接采用；
- 因此默认 profile 不改，先保留 sweep 结论作为后续结构性修法（区分
  serial-chain 与 fan-out exposure、单独处理 LDP/tile store）的依据。

### 5.7 归档文件

| 路径 | 内容 |
|---|---|
| `verification/reports/costmodel_padded_copy_{gather,scatter,wgrad}.json` | 三个 kernel 的最终 costmodel report（store 白盒 + exposure fix） |
| `verification/results/three_kernel_compare.csv` | load + store 汇总对比 |
| `verification/results/three_kernel_store_compare.csv` | ScalarStore / tile store store 明细 |
| `verification/results/candidate_mape.csv` | profile sweep MAPE 表 |
| `verification/camodel_dumps/` | core0 CAModel 关键 dump（lsu/dc/instr/queue），用于窗口复核 |
| `verification/*.sh`、`verification/*.py` | 上述运行、解析、汇总、sweep 脚本 |

---

## 6. 适用边界

1. load 有 same-line / diff-line / indirect 三类结构化口径；store 已有
   MainScalar / SIMT 白盒公式，缺结构化字段时回退 legacy `store_cycles_fit`
   / scalar-pipe provisional rate；tile store（`IndirectGatherMemory`）仍走
   indirect transaction 模型，不在本节的 scalar store 公式范围内。
2. same-line 证明是保守的 IR 静态证明；证不出就一定走 diff-line。
3. 多 warp / 多核 / L2 state 未纳入。
4. 结构化参数来自 CAModel active cycles，board 绝对周期需用 SYS_CNT 或
   triton profiler 二次校准。
