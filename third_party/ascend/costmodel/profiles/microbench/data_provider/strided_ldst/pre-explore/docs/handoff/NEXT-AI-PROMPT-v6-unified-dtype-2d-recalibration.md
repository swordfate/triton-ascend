# 接力提示词：统一口径重测 + 公式收敛（位宽作为因子，6–8 个公式）

> ⚠ **脚本现状注记（后加，非原文）**：本文引用到的下面这些脚本**已在脚本清理中删除**（正文已就地加「（**已删除**）」标注）：
> `board_launch.sh`（2 处）、`fit_dtype_model.py`、`fit_2d_model.py`、`predict_dtype_model.py`、`predict_2d_model.py`、
> `refit_no_stride1.py`、`summarize_calibration.py`、`board_launch_camodel2d.sh`、`run_phaseA_measure.sh`、
> `run_phaseB_campaign.sh`、`run_phaseAB_campaign.sh`（后三个是 `run_dtype_calibration.sh` 的冗余包装，
> 直接用 `PASS=all_a` / `PASS=all_b` 即可），以及**要求新写、但最终没有落地成独立文件**的
> `dtype_stride_rotate512_probe.py`、`stride2d_rotate512_probe.py`、`dtype_stride_single_shot_probe.py`、`stride2d_single_shot_probe.py`
> （512 MB 口径后来由 `run_rotate512_campaign.sh` 复用原探针实现；单发口径由 `frequency_retry_probe.py` 与 `simd_one_load_syscnt.py` 承担）。
>
> 推导记录保留在 [`../agent-logs/`](../agent-logs/)（11 份 `p3_*.txt`）与 git 历史里
> （`git show HEAD:pre-explore/scripts/<名>` 可查看，`git checkout HEAD -- <路径>` 可恢复）。
> 本文其余脚本（`stride_feature_reference.py`、`dtype_stride_path_check.py`、`stride2d_path_check.py`、
> `dtype_stride_measure_probe.py`、`stride2d_measure_probe.py`、`build_dtype_dataset.py`、`build_2d_dataset.py`、
> `check_acceptance.py`、`run_dtype_calibration.sh`、`analyze_strided_camodel.py`、`analyze_stride2d_camodel.py`、
> `board_env.sh` 等）**都还在**；CAModel 的驱动实际位于 `scripts/camodel/`（如 `scripts/camodel/run_camodel.sh`）。
> `env_ascend.sh` / `conda.sh` 是**板卡上的 CANN 文件**，不在本仓库。现存脚本索引见 [`../../scripts/INDEX.md`](../../scripts/INDEX.md)。

> 给下一个 AI：这是一个**重测 + 重构拟合**任务，不是从零开始，也**不是集成任务**。
>
> **七条最重要的约束：**
>
> **(1) 起点** 是 `feature/strided-load-store-costmodel` 分支的
> `third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/` 工作区。
> 那里已经有一整套 1D/4B、多 dtype、2D 的标定与建模成果（含脚本、数据、公式、误差、验证）。
>
> **(2) 本轮只做"重测 + 建模 + 汇总"。** 不要改 CostModel 的 C++、不要改 profile/schema、
> 不要改 UT、不要改后端 lowering。集成留给再下一轮（你只写"接入建议"）。
>
> **(3) 口径要对齐，但不是「换成 cold」这么简单。** 四个探针核对下来：
> **只有 SIMD load 是单发**；v8（SIMT load）/ store / template **都是 rotate-loop 斜率**，
> 与今天的 dtype/2D **方法相同**，唯一差异是**旋转窗口：今天 64 MB，历史 512 MB**。
> 所以：**主轮**只需把窗口对齐到 512 MB（`--rotate-step 8192 --rotate-mask 16383`），
> 就能覆盖 5/6 条公式；**附带轮**用单发（`simd_one_load_syscnt.py` 那套）对齐 SIMD load，
> 并产出「单发/斜率」比值。两轮同网格。还要做 64 MB vs 512 MB 的窗口对照实验。详见 §6.1。
> **(4) 公式数量上限 6 个（必要时 8 个）。**
> `6 = {load, store} × {simd, pure-SIMT, template}`；
> 只有当数据证明确实需要时，才把 SIMD 的 `stride == 2`（wide）拆出来 → 最多 8 个。
> **绝对不要 per-dtype 公式**（今天有 17 个，太多了）。位宽必须作为**因子**融进公式，
> 所有 dtype 的数据 **放在一起拟合**。
>
> **(5) 每一个交付的模型都必须能用一条默认参数命令逐位复现**，并且
> `predict_*.py --verify` 逐点复算 `max |predict - errors.pred_ns| < 1e-6`。
>
> **(6) 只做 in-sample**，不做交叉验证 / holdout（上一轮已确认这个口径）。
>
> **(7) 收尾必交「逐公式汇总」**（§6.3）：每个公式都要有
> **逐项解释 → 数据规模与构造原则（怎么乘出来的、扣掉几个无效点）→ 两类误差 + 最差 case 归因**。
>
> 本文件里 ✅ 的"现状"是上一轮从代码与实测数据里逐条核实过的，可以直接当事实用；
> ⚠️ 的地方要求你自己再验一遍再下结论。

---

## 0. 这件事的目的

triton-ascend 的 **CostModel for AutoScope** 要在编译期给一个 Triton kernel 估算
**SIMD / pure-SIMT / simt-template** 三条 lowering 的 cycle 成本，从而选择最优 route。

其中 **shaped strided`tl.load` / `tl.store`**（有空洞的访存）在三条 lowering 上是
三种完全不同的机制，必须**分别标定、分别建模**：

| 路径 | 触发条件 | 机制 |
|---|---|---|
| **SIMD** | `compile_mode=simd` | MTE2 wide 事务 / per-element gather（MTE3 同理） |
| **pure SIMT** | `compile_mode=simt_only` | `SIMT_LDG/STG` + 128B DCache line |
| **simt-template** | `compile_mode=simd_simt_template` + 内轴静态非 2 次幂 stride >= 3 | `triton_stride_load/store`（固定 1024 线程的 scalar 循环） |

最终要做到：**任意 (元素位宽, rank, 形状, stride) 的 strided 访存都能算出一个 ns/iteration 的代价**，
并且公式数量足够少（6–8 个）以便维护与接入。

---

## 1. 环境

### 1.1 板卡

```bash
ssh ascend-950pr-63                 # 用户 c00946898，共享板卡
source ~/env_ascend.sh              # CANN + bishengir PATH
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope         # 主运行环境（python 3.11）
export ASCEND_RT_VISIBLE_DEVICES=0
ulimit -n 1048576                   # 不设会 "Too many open files"
```

- ⛔ **普通测量不要设** `PYTHONPATH=$HOME/sb64_latest_env`（会破坏 CANN 的 acl 检测）。
  统一的板卡包装脚本是 `scripts/board_env.sh`。
- ✅ **只有 CAModel（`msopprof simulator`）需要** `PYTHONPATH=$HOME/sb64_latest_env`，
  见 `scripts/run_camodel.sh` / `run_stride2d_camodel.sh`。
- 共享板卡：跑前看 `uptime` 与 `npu-smi info`；witness / 频率门 / spread 门一道都不要放宽。

### 1.2 板卡上的工作副本

```text
~/strided-template-calibration-v2/
├── scripts/      # 与本地 pre-explore/scripts/ 同步
├── results/      # 数据（部分）
└── runs/         # CAModel 的 OPPROF 输出（本地只归档 parsed json/csv）
```

同步与启动：

```bash
# 同步（⛔ 绝不要在 bash 脚本正在运行时覆盖它 —— bash 是增量读取，会读到错位内容而崩）
scp <local files> ascend-950pr-63:~/strided-template-calibration-v2/scripts/

# 长任务用 tmux（不要用裸 nohup ... &）
ssh -n ascend-950pr-63 "bash ~/strided-template-calibration-v2/scripts/board_launch.sh（**已删除**） <session> <PASS>"
ssh -n ascend-950pr-63 "tmux ls; tail -5 ~/strided-template-calibration-v2/<session>_outer.log"
```

### 1.3 本地仓库

```text
D:\work_wj\triton-ascend
branch: feature/strided-template-calibration-v2  （= fork-63/feature/strided-load-store-costmodel）
```

---

## 2. 已经做了什么（代码清单）

主战场：`pre-explore/`（下面所有路径都相对它）。

### 2.1 特征参考实现（唯一权威）

- `scripts/stride_feature_reference.py` ✅
  三条路径**全部特征**的精确整数语义实现，**已参数化 `element_bytes`**（非 4B 元素可用）。
  含 `--verify`：与 `StageCostModels.cpp` 的浮点镜像逐值比对（9888 样本，0 mismatch）
  + 与旧的 `template_stride_features.py` 在 v2 网格上比对（2960 值，0 mismatch）。
  **所有 fitter / predictor / dataset builder 都必须用它**，不要自己再实现一份。

### 2.2 路径检查（compile-only，零成本）

- `scripts/dtype_stride_path_check.py` ✅ dtype × stride × mode
- `scripts/stride2d_path_check.py` ✅ rank-2 形态 × mode（含 `axis_class` 分类）

### 2.3 测量探针

- `scripts/dtype_stride_measure_probe.py` ✅ 阶段 A（dtype × stride），**rotate-loop 斜率法**
- `scripts/stride2d_measure_probe.py` ✅ 阶段 B（2D），同上
- ⚠️ **旧探针（cold 口径）**：`scripts/simd_one_load_syscnt.py`（SIMD 单发 syscnt 延迟，
  **每个 rep 一块全新 buffer**，并减去结构匹配的 no-load 内核基线）、
  `scripts/board_event_probe.py`、`scripts/frequency_retry_probe.py`、
  `scripts/store_measure_probe.py`、`scripts/strided_load_probe.py`。
  **这是你重测时要参考的脚本。**

### 2.4 数据集构建 + 拟合 + 预测器

- `scripts/build_dtype_dataset.py` / `build_2d_dataset.py` ✅ 断言无 stride==1 / 无 contiguous
- `scripts/fit_dtype_model.py（**已删除**）` ✅ 方案 A / E1 / E2 / E3 / B / C / D 的对比
- `scripts/fit_2d_model.py（**已删除**）` ✅ 方案 A_1d / B / C / W / W2 / D
- `scripts/predict_dtype_model.py（**已删除**）` / `predict_2d_model.py（**已删除**）` ✅ 参考预测器 + `--verify`
- `scripts/refit_no_stride1.py（**已删除**）` ✅ 阶段 0（去 stride==1 后重拟）
- `scripts/summarize_calibration.py（**已删除**）` ✅ 两类误差统计
- `scripts/check_acceptance.py` ✅ 一键验收自查（`--with-refit`）

### 2.5 CAModel 白盒

- `scripts/strided_load_probe.py` + `run_camodel.sh` + `analyze_strided_camodel.py` ✅ 1D
- `scripts/stride2d_camodel_probe.py` + `run_stride2d_camodel.sh` +
  `analyze_stride2d_camodel.py` + `board_launch_camodel2d.sh（**已删除**）` ✅ 2D

### 2.6 驱动

`scripts/run_dtype_calibration.sh`（`PASS=pathlite|simd|simt|template|2dpath|2d|all_a|all_b`）、
`run_phaseA_measure.sh（**已删除**）`、`run_phaseB_campaign.sh（**已删除**）`、`run_phaseAB_campaign.sh（**已删除**）`、`board_launch.sh（**已删除**）`。

---

## 3. 结果放在哪里

### 3.1 主文档

`pre-explore/README.md`（3000+ 行）—— **先读这个**，章节：

| 章节 | 内容 |
|---|---|
| §0 | 标定数据总览（全部实测数据的分类、规模、**结果路径**） |
| §1–§6 | 早期 1D/4B 标定（load / store / template v2 / v1） |
| §7 | 阶段 0：去 stride==1 的审计与重拟 |
| §8 | 阶段 A：多数据类型标定与建模 |
| §9 | 阶段 B：2D 标定与建模 |
| §10 | 接入建议书（只写文档，未实施） |
| §11 | 逐公式汇总（17 个公式，每个都有语义 / 数据配置 / 两类误差） |
| §12 | 验收清单自查 |
| §13 | 2D 白盒探索（CAModel 证据） |
| §1.2.1 / §1.2.1.1 | **数据规模表 + 口径一致性矩阵**（必读） |

### 3.2 数据与产物

```text
pre-explore/results/
├── DATA_INVENTORY.md                  # §0 的独立副本
├── refit_no_stride1_v1/               # 阶段 0：去 stride==1 的三条路径重拟（16 文件）
├── dtype_calibration_v1/              # 阶段 A + B
│   ├── dataset.csv                    # 阶段 A：88 行（dtype）
│   ├── dataset_2d.csv                 # 阶段 B：128 行（rank-2）
│   ├── dataset_meta.json / dataset_2d_meta.json
│   ├── model_dtype_v1.json / model_2d_v1.json
│   ├── scheme_comparison.{json,md} / scheme_comparison_2d.json
│   ├── errors_*.csv / errors_2d_*.csv / metrics_summary*.csv / summary_stats.json
│   ├── raw/                           # 48 个原始测量 JSON（你重测时写新目录，别覆盖）
│   ├── ir_evidence/                   # path check JSON + 日志 + 197 个 TTAdapter/.npubin 证据
│   └── camodel_2d/                    # rank-2 CAModel 的 parsed_2d.json/csv + run.log
└── model_v2/ model_v3_numwarps/ model_v4_numwarps/ model_store_v1/
    model_template_stride_v1/ model_template_stride_v2/   # 历史数据（**不要覆盖**）
```

---

## 4. 目前的拟合结果（in-sample MAPE）

### 4.1 阶段 0：1D/4B，过滤 stride==1 后重拟（零新增测量）

| branch | n | 旧系数在同批数据 | 新 | 备注 |
|---|---:|---:|---:|---|
| store SIMT（profile 在用） | 616 | 13.34% | **12.23%** | p90 30.0→28.0 |
| store SIMD gather（在用） | 88 | 10.78% | **10.43%** | p90 26.2→20.4 |
| SIMT v8 load（在用） | 1232 | 15.25% | 15.31% | 基本不变 |
| SIMD load gather | 504 | 5.71% | **5.68%** | |
| SIMD load wide（只剩 stride==2） | 12 | 6.38% | **6.37%** | stride_minus_1 不可辨识 |
| store SIMD wide | 0 | — | — | **分支不可达，建议删除** |

### 4.2 阶段 A：多数据类型（今天的数据：rotate-loop 斜率，窗口 64 MB）

| path | op | n | A_reuse | A_reuse(f32 only) | E2 位宽因子 | B 混合重拟 | D per-dtype |
|---|---|---:|---:|---:|---:|---:|---:|
| simd | load | 21 | 132.87% | **171.79%（口径错配）** | 36.74% | **17.61%** | 14.99% |
| simd | store | 21 | **6.36%** | 5.09% | 15.54% | 33.31% | 4.18% |
| simt_only | load | 5 | 30.00% | — | 17.84% | 2.52% | 1.84% |
| simt_only | store | 5 | 31.22% | — | **22.63%** | 22.96% | 1.61% |
| template | load | 18 | 17.17% | — | 12.88% | 6.08% | 1.81% |
| template | store | 18 | 15.18% | — | 10.66% | **9.47%** | 3.95% |

> `D_per_dtype` 一律**不采用**（每 dtype 只有 2–7 点，却有 4–6 个系数，是过拟合）。

### 4.3 阶段 B：2D（rank-2，两轴都有空洞 + R==1，同样 64 MB 窗口）

| path | op | n | A_1d_transfer | B | C | W2（白盒驱动） | D |
|---|---|---:|---:|---:|---:|---:|---:|
| simd | load | 24 | 460.98% | 56.40% | 41.14% | **13.91%** | 51.74% |
| simd | store | 24 | 6.12% | **6.09%** | 6.09% | 16.45% | 3.37% |
| simt_only | load | 24 | 32.77% | **8.45%** | 8.45% | 9.75% | 4.02% |
| simt_only | store | 24 | 36.99% | **32.36%** | 32.60% | 36.46% | 18.61% |
| template | load | 16 | 23.05% | 5.81% | **5.61%** | — | 2.88% |
| template | store | 16 | 37.06% | 3.01% | **2.79%** | — | 1.54% |

数据量：阶段 A **88/88 valid**、阶段 B **128/128 valid**，全部 correct；
跨 pass 相对离散度 median 0.95% / 0.69%（测量本身很稳，误差基本都是模型误差）。

---

## 5. 已核实的关键结论（不要重复推翻，除非你有新数据）

1. **template 的分派判据是元素域，与 dtype 无关** ✅（80/80 + 114/114，mismatches=0）：
   `stride<=1 bail / ==2 bail / 2 次幂>=4 bail / 其余静态非 2 次幂>=3 进 template`。
   **外轴 stride 完全不参与判断**——所以 2D 里"内轴非 2 次幂 >=3"的形态也能进 template。
2. **SIMD load 的 wide 分支就是 `stride == 2`**（元素域）✅：
   所有 `stride==2` 的行落在 67–83 ns，与元素宽度无关（1B/2B/4B/8B 都一样）；
   所有 `stride>=3` 的行落在 206–538 ns。**现有 C++ 的 `strideElements <= 2` 判据是对的。**
3. **SIMD store 的现有公式已经跨位宽可用（6.36%）** ✅，
   `line_request_size = min(128, (128//strideBytes)*4)` 里那个 C++ 硬编码 `* 4.0`
   **已经把位宽效应吸收进去了**，**不要"修正"它**（改成真实位宽会把 6.36% 弄到 40.38%）。
4. **位宽不能作为加性项**：把 `element_bytes` 当加项时 NNLS 直接压成 0（E3 ≈ B）。
   位宽是**乘性**的（同 block 下元素越窄、命令越多，代价越大）。
5. **2D 白盒（CAModel）结论** ✅：SIMD 的 rank-2 strided load 退化成
   **每元素一条 4B BIU command**（`biu_send_count == R*C`，B/cmd 恒为 4.0）；
   线数（L）与 MTE2 command 数都**不是**代价驱动量（L 涨 8 倍、窗口只涨 2.2 倍；
   MTE2 push 从 1 变成 R，窗口几乎不变）。**内轴 stride == 2 便宜约 6 倍** →
   2D 也需要按内轴 stride 做 wide/gather 划分（W2 方案：41.14% → 13.91%）。
6. ⚠️ **口径警告（最重要，前两版提示词在此处表述有误）**：
   - **只有 SIMD load 的现有模型是「单发」**（每 rep 一块全新小 buffer + 扣 no-load 基线）；
   - **v8 SIMT load / store / template 都是 rotate-loop 斜率**，与今天的 dtype/2D **方法相同**；
   - 唯一的差异是**旋转窗口：今天 64 MB（按字节预算，四种位宽一致），历史 512 MB**（`step 8192 × mask 16383`）；
   - 所以「今天的是 warm、以前的是 cold」这个二分法**不成立**：
     v8/store/template 的 512 MB 窗口同样每迭代 cache miss；
     真正的区别在**单发延迟 vs 流水化吞吐**（同一点相差 4–5 倍：263.5 ns vs 68.9 ns）；
   - 需要做的对齐是**把窗口改回 512 MB**（主轮，覆盖 5/6 条公式）+ **补一轮单发**
     （附带轮，对齐 SIMD load），并做 64 MB vs 512 MB 的窗口对照实验。详见 §6.1。
---

## 6. 你的任务

### 6.1 阶段 R：对齐口径，重测 dtype 与 2D

#### 6.1.1 ⚠️ 先把「口径」这件事说准（本节最重要的一段）

前两版提示词把口径简化成「cold vs warm 缓存」并且数字也没核准，**都是不准确的**。
核对四个探针源码 + 今天 raw JSON 里记录的 `buffer` 字段后，实际情况是：

| 探针 | 测量方式 | buffer 分配 | **旋转窗口** | target |
|---|---|---|---|---|
| SIMD load（`model_v2`）<br>`simd_one_load_syscnt.py` | **单发**，inline SYS_CNT 读**一次** load 的延迟 | **每个 rep 一块全新小 buffer**（各只读一次） | 无（单发） | `simd_single_cold` |
| SIMT load v8<br>`simt_frequency_retry_probe_aligned.py` | **rotate-loop 斜率**（iters 2000/8000 线性回归） | 一块复用 | **512 MB**<br>`step 8192 × mask 16383 = 1.34e8 元素 × 4B` | `simt_rotate_aligned_event_median` |
| store（`model_store_v1`）<br>`store_measure_probe.py` | **rotate-loop 斜率** | 一块复用 | **512 MB**（同上） | `store_rotate_min_observed_ns_per_iter` |
| template v2<br>`template_stride_load_probe.py` | **rotate-loop 斜率** | 一块复用 | **512 MB**（同上） | `min_across_probe_passes` |
| 今天 Phase A / B<br>`dtype_stride_measure_probe.py` / `stride2d_measure_probe.py` | **rotate-loop 斜率** | 一块复用 | **64 MB**（按**字节**预算：`budget = (1<<25)//elem_bytes`，所以四种位宽都是 64 MB） | `rotate_loop_min_event_slope_ns_per_iter` |

**三条必须记住的事实：**

1. **v8（SIMT load）、store、template 都不是「单发」——它们和今天的 dtype/2D 用的是
   同一种方法：rotate-loop 斜率。** 唯一用单发的是 **SIMD load**。
2. 「cold」这个词有两个含义，不要混：
   - **含义 A：每次访问是否 cache miss（不靠缓存）** —— v8/store/template **是** cold
     （窗口 512 MB >> cache，每迭代读的都是新 line）；
   - **含义 B：是否是单发延迟测量** —— v8/store/template **不是**
     （它们是流水化后的边际吞吐）。
   所以「v8 是 cold 的」这句话，在含义 A 下成立，在含义 B 下不成立。
3. **今天的数据与 v8/store/template 的唯一方法差异是「旋转窗口」：
   64 MB vs 512 MB，差 8 倍。**
   64 MB 是否部分落进 L2（从而让今天的成本偏低）**需要用实测确认**——
   这是今天这批数据**唯一真正需要对齐的地方**，不需要换测量方法。

#### 6.1.2 决策：分两轮，各对齐一半公式

| 路径 | 现有 profile 系数的口径 | 对齐它需要什么 |
|---|---|---|
| SIMT load / store、template load / store（**4 条公式**） | rotate-loop 斜率，**512 MB 窗口** | **主轮**：沿用今天的探针，**只把窗口改成 `step 8192 × mask 16383`** |
| SIMD store（**1 条公式**） | rotate-loop 斜率，512 MB | 同上 |
| SIMD load（**1 条公式**） | **单发** | **附带轮**：单发探针（每 rep 新 buffer + no-load 基线） |

**推荐的执行方案（两轮同网格，编译可复用）：**

- **主轮（必做，对齐 5/6 条公式）**：把今天 dtype 与 2D 探针的**旋转窗口从 64 MB 改成 512 MB**
  （`--rotate-step 8192 --rotate-mask 16383`），其余（斜率法、四道门、网格、字段）**全部不变**。
  这样得到的数据与 v8 / store / template **同口径**，可以直接用来评估与重拟那 5 条公式。
- **附带轮（建议做，对齐 SIMD load）**：按 `simd_one_load_syscnt.py` 的单发方式，
  对**同一批格点**再测一遍（每 rep 一块全新 buffer + 结构匹配的 no-load 基线）。
  这样：
  - SIMD load 的公式也能用同一批数据校准；
  - 可以得到**「单发 / 斜率」逐点比值**——这本身是一个可交付的结论
    （上一轮已知同一点相差 4–5 倍，见下），还能让 CostModel 未来按
    「访问是否在循环里」选择系数。
- **必须回答的一个问题**：窗口 64 MB vs 512 MB 到底有没有影响？
  挑 3–5 个代表性格点把两种窗口都测一遍，报出差异。**如果没差异**，
  说明今天已有数据在含义 A 上本来就是 cold 的，重测规模可以大幅缩减；
  **如果有差异**，说明窗口必须对齐，且今天的历史结论需要复核。

#### 6.1.3 一个物理风险，必须如实评估并写进结论

CostModel 要预测的是**一次 dynamic access 在真实 kernel 里的代价**。
如果这个访问在循环里反复执行，实际更接近**流水化吞吐（斜率）**而不是**单发延迟**。
用单发延迟去预测循环内访问，可能**系统性高估**——上一轮实测同一点相差 **4–5 倍**
（block 64 / stride 2：单发 263.5 ns vs 斜率 68.9 ns）。

所以：**主轮（斜率）应当是 CostModel 的主口径**，单发数据作为 SIMD load 的兼容口径与对照。
如果你的分析认为应该反过来，**如实报告理由**，不要为了让数字好看而默默挑一个口径。

#### 6.1.4 做法

1. **主轮**：直接改今天两个探针的窗口默认值（或加命令行参数），
   新写 `scripts/dtype_stride_rotate512_probe.py（**已删除**）` 与 `scripts/stride2d_rotate512_probe.py（**已删除**）`，
   （或者给现有探针加 `--rotate-step/--rotate-mask` 参数并保留旧默认值，二选一，说明清楚）。
   要求：
   - 斜率法 / iters / reps / attempts / 四道门（ALU witness 15% / Aicore 频率 >= 1500 MHz /
     spread <= 0.40 / 数值正确性）**全部不变，一道都不许放宽**；
   - 每个 case 跑 2 个 pass，dataset 取全部 valid 的**最小** target；
   - 断言 `stride == 1` 不进数据集；`contig` 只进 path check。
2. **附带轮（单发）**：参考 `scripts/simd_one_load_syscnt.py`（每 rep 一块全新 buffer，
   减去结构匹配的 no-load 内核基线），新写 `scripts/dtype_stride_single_shot_probe.py（**已删除**）`
   与 `scripts/stride2d_single_shot_probe.py（**已删除**）`。要求：
   - **每个 rep 一块全新 buffer**（`min(nbuf, reps)` 块，各只读一次）；
   - ⚠️ **先把 no-load 基线的抖动统计出来再决定是否相减**——上一轮该基线在 250–380 cycle
     之间抖，减法噪声可能比某些 case 的整个代价还大；抖动不可接受时改用「多次取最小」或报总量；
   - ⚠️ load 的 sink buffer 用 **float32**（上一轮 int8 sink 被截断，造成 correctness 假失败）。
3. **网格与今天逐格对齐**（block / stride / 形态 / 形状都一样），并保证
   **每条路径都有 f32（4 字节）行**——这是分离「口径差异」与「位宽效应」的唯一锚点。
   （对 SIMT / template 来说，这是**新增**测量；历史数据里它们的 f32 是斜率口径的。）
4. 成本：编译约 **40 s / 个 (path, dtype, block, stride) 组合**，是硬约束；先算总组合数再定网格。
   可以只对**关键格点**做（每条路径 × 每种位宽 × 2 个 block × 3 个 stride 起步），
   但**必须覆盖 f32 锚点 + 至少 3 种其它位宽**，并且**包含 6.1.2 里那个窗口对照实验**。
5. 产物写到**新目录**，**不要覆盖** `results/dtype_calibration_v1/`：
   - `pre-explore/results/dtype_calibration_v2_rotate512/`（主轮）
   - `pre-explore/results/dtype_calibration_v2_single_shot/`（附带轮）
6. 必须回答（写进 README）：
   - **窗口 64 MB vs 512 MB 的差异**（逐格点，代表性子集即可）；
   - **单发 / 斜率的逐点比值**，按路径 / 位宽 / block 分组；
   - 主轮口径下位宽因子的形状（`k(1B) : k(2B) : k(4B) : k(8B)` 的比值）；
   - 现有六条公式的系数在新数据上的误差（脱节程度的量化）；
   - 两轮数据能否用**同一个**位宽因子；如果不能，差在哪。


### 6.2 阶段 F：公式收敛到 **6 个（最多 8 个）**

**目标**：

`~text
6 个 = { load, store } × { SIMD, pure-SIMT, simt-template }
8 个 = 上面 6 个 + SIMD 的 wide/gather 各拆一个（仅当数据证明确实需要）
`~

**硬性要求**：

1. **位宽作为因子融进公式**，所有 dtype 的数据**放在一起拟合**；**禁止** per-dtype 公式。
2. **什么叫「把位宽做成因子」——两种可行做法 + 一种已被证伪的做法：**

   **(a) 把自变量从「元素域」换到「字节域」（首选）**

   `text
   元素域（会按 dtype 分裂）:  T = c0 + c1*(stride - 1) + ...
       对 f32 ：stride=2 表示相邻元素隔 8 字节
       对 int8：stride=2 表示相邻元素隔 2 字节
       —— 同一个 (stride-1)=1 对应完全不同的物理间距，一个公式盖不住。

   字节域（一个公式盖住所有位宽）:
       strideBytes = strideElements * elementBytes
       T = c0 + c1*(strideBytes - elementBytes) + ...
       —— f32 s2 与 int8 s8 都是 8 字节间距，进入同一项。
   `

   这就是「位宽融入公式」的本质：**不是额外加一个 dtype 参数，而是把自变量的单位换成字节**。
   判断依据已经有了：上一轮实测 **SIMD store 在同一个字节间距下，四个位宽的代价一致到 0.5% 以内**
   —— 说明物理量就是字节间距。请把每条公式里**所有与地址间距有关的量**都检查一遍是否已在字节域
   （例如 `stride_minus_1` 是元素域；`line_request_size` 已经是字节域）。

   **(b) 显式乘性因子（次选，用于结构上无法完全字节化的地方）**

   `text
   T = k(elementBytes) * f(字节域特征)
   k = 1 张 4 行小表（1B / 2B / 4B / 8B 各一个标量）
   `

   **(c) 已被证伪：加性项。** 即 `T = f(原有特征) + c * elementBytes` 这种多加一项的做法。
   ⚠️ 上一轮实测**加性项会被 NNLS 直接压成 0**（`element_bytes` 系数 = 0，加了等于没加）。
   原因是位宽改变的是**整体尺度**：同样 block 下元素越窄 → 命令条数越多 → 代价整体放大，
   这是**乘法关系**，不是「额外加一笔固定开销」。
   结论：**位宽要么进字节域特征，要么做乘性因子，不要做加项。**
3. **拆分规则要用数据决定，不要先验假设**：
   - SIMD load：有证据表明 `stride == 2` 是一个独立的便宜分支（**元素域**判据，
     同 stride 下位宽无关：f16 s2 / f32 s2 / i64 s2 都是 67–83 ns），
     所以 **SIMD load 可能必须拆成 2 个**（wide / gather）→ 那就是 8 个公式；
   - SIMD store：目前证据表明**不需要**按字节分界（现有公式已跨位宽可用，6.36%），
     但 **cold 口径下要重新验证**；
   - 2D：白盒显示需要按**内轴 stride** 划分（`IS <= 2` vs `IS > 2`）。
     请明确它是「同一公式的第二个划分维度」还是「要占额外公式名额」，并给出建议。
4. **每个公式都要给出**：
   - 公式原文（全部系数实际数值，0 项也要显式写出并说明为什么是 0）+ **逐项物理含义表**
     （找不到物理解释的项要如实说明）；
   - 适用域（位宽 / rank / 形状 / stride / 对齐 / mask）；
   - **不可辨识项**要标注（例如某特征在该网格上恒为常数、与截距共线）；
   - in-sample 误差：`n / MAPE / p50 / p90 / p95 / max / bias / RMSE`；
   - **分组误差**（按 dtype / block / stride / 2D 形态）+ **最差 top-15 case 清单**与归因；
   - 常数项不为 0、系数非负、无 `log`、无减法项。
5. **拟合口径**：raw target 域加权 NNLS（`min Σ w_i (c0 + Σ c_j f_ij - T_i)^2, c_j >= 0`），
   `w_i = T_i^-alpha`，`alpha` 在 `{1.0, 1.5, 2.0}` 里按**最小 in-sample MAPE** 自动选，
   并列取最小值；**交付的 model JSON 必须能用一条默认参数命令逐位复现**。
6. **防止某个 dtype 主导拟合**：报告按 dtype 分组的误差与样本数；
   某个位宽只有 2–3 个点时明确标注它是**弱约束**，不要让它拉动全局系数。
   这是判断「公式是否被某个位宽主导」的主要手段。


### 6.3 阶段 S：逐公式汇总（收尾必交，覆盖**每一个**公式）

公式定稿后，在 README 新增一章，对**每一个公式**给出下面三块。
**不许只给一张总表了事——每个公式都要有自己的三块。**

#### 6.3.1 公式与逐项解释

- **公式原文**，含**全部系数的实际数值**（不要省略 0 项；0 项要显式写出来并说明为什么是 0）；
- **逐项表**：`| 项 | 单位 | 物理含义 | 为什么会让时间增加 |`；
  每一项都必须有可解释的物理含义；如果某个项是为拟合而加、找不到物理解释，**如实说明**；
- **适用域**：位宽 / rank / 形状 / stride 范围 / 对齐 / mask 条件；
- **已知退化或不可辨识的项**要标注（例如某特征在本网格上恒为常数、与截距共线）。

#### 6.3.2 数据规模与**构造原则**（不要只写「有效点数 n = xxx」）

必须写清**这些点是怎么乘出来的**，至少包含：

| 维度 | 要写什么 |
|---|---|
| 元素量 | block（2D 写 R×C），列出**取了哪些值、一共几个** |
| 数据类型 / 位宽 | 哪些 dtype；**每个位宽各几个点** |
| stride | 取了哪些值；哪些是 non-power-of-two；**是否含 stride==1 的 bail 对照** |
| num_warps | 取了哪些值（SIMD 要说明与 W 无关） |
| rank / 形态 | 1D 还是 2D；2D 的 axis_class、OS / IS 的取法 |
| 采样配置 | 每个 case 的 iters / reps / attempts / pass 数；target 的统计口径 |
| **乘积与扣减** | **乘起来的总格点数 = A × B × C × D**，然后**去掉多少个无效点**，invalid 必须**按原因归因**（spread / witness / 频率 / correctness / 编译失败） |
| 覆盖矩阵 | 哪些 (dtype, block, stride) 有数据、哪些缺、哪些是**弱约束区**（有效点很少） |
| 路径 | dataset.csv / raw/*.json / ir_evidence/ 的实际路径 |

#### 6.3.3 实测误差与最差 case 归因

- **两类误差分开报，不要混为一谈**：
  1. **实测不确定度**：同一 case 在多 pass / 多样本之间的相对离散度
     （定义要写清，如 `(max-min)/min` 或 spread 的定义），报 median / p90 / max，按 path × dtype 分组；
     目的是让读者判断「公式 15% 的误差」在「测量本身抖动 10%」的前提下意味着什么；
  2. **公式预测误差**：in-sample 的 `n / MAPE / p50 / p90 / p95 / max / bias / RMSE`。
- **分组误差**：按 dtype / block / stride / 2D 形态分组各报一遍；
- **最差 case 清单（top 10–20）**：逐行给出
  `(path, dtype, rank/形状, block, stride, num_warps, target_ns, pred_ns, 相对误差)`；
- **必须回答「误差大的通常是哪些 case」**，并把原因至少分成三类：
  (a) **模型缺项**——该形态没有被特征覆盖；
  (b) **测量本身不稳**——该 case 的实测离散度也大；
  (c) **弱约束区**——该 (dtype, block, stride) 的有效样本数很少。
- **一页纸总表**（`| path | op | 位宽 | rank/形状 | 公式来源 | n | MAPE | p90 | max | 建议是否采用 |`）
  放在三块之前或之后，并写清：哪些公式建议采用、哪些建议保留观察、哪些不覆盖，以及理由。

## 7. 通用方法论（照抄现有流程）

### 7.1 compile-only path check
参考 `scripts/dtype_stride_path_check.py` / `stride2d_path_check.py`：
只编译不发射，断言 TTAdapter 里是否出现 `call @triton_stride_load/store`；
**必须包含 stride==1 / 2 / 2 次幂作为 bail 对照**；验收 **mismatches = 0**；每 case 存 asm 作证据。

### 7.2 CAModel 白盒（不需要物理卡）
- 1D：`strided_load_probe.py` + `run_camodel.sh` + `analyze_strided_camodel.py`
- 2D：`stride2d_camodel_probe.py` + `run_stride2d_camodel.sh` + `analyze_stride2d_camodel.py`
   （异步：跑完在板卡上直接分析，只把 `parsed_*.json/csv` 与 `run.log` 拉回本地，
   OPPROF 原始 dump 很大，别进 git）
- 看：MTE2/MTE3 的 command 粒度与条数、BIU 的 send/recv 条数与字节数（**B/cmd 是关键量**）、
  DC TagRam 的 line 有效字节、SIMT LSU 的 `SIMT_LDG/STG` 条数、issue/retire overlap。

### 7.3 测量协议（沿用今天探针里的实现，不要自己发明）
- rotate loop / 单发（按阶段选择），iters / reps / attempts 与今天**逐格对齐**；
- 每次测量前跑 ALU busy burst 保持 boost；
- attempt 后 ALU witness（15%）+ `npu-smi` Aicore 频率（>=1500 MHz）+ spread（<=0.40），
  三者全过才算 valid；
- dataset 对同一 key 取全部 valid+correct 样本的**最小** target；
- **绝不允许**为了"保数据"手工放宽任一阈值；invalid 的归因要如实统计。

### 7.4 拟合
- 数据集先断言 `stride == 1` 行数为 0（contiguous 不属于 strided 模型）；
- 加权 NNLS（Lawson-Hanson），`c >= 0`，无 `log`，无减法项；
- 只做 in-sample；
- 报 `n / MAPE / p50 / p90 / p95 / max / bias / RMSE` + 分组误差 + 最差 case 归因。

### 7.5 特征可实施性
- 每个特征给出**精确整数语义**（`floor`/`mod`/`ceil` 边界写清）；
- 用 `stride_feature_reference.py` 的 Python 参考实现（按将来 C++ 会怎么写来写）；
- 做**数值等价性验证**（对 C++ 镜像 + 对旧特征模块），要求 **0 mismatch**。

---

## 8. 交付物与验收

### 8.1 交付物

```text
pre-explore/
├── README.md                              # 追加新章节：阶段 R（重测）/ 阶段 F（公式收敛）/ 汇总
├── scripts/                               # 新探针 / 新 dataset builder / 新 fitter / 新 predictor
└── results/
    ├── dtype_calibration_v2_cold/         # 阶段 R 的 cold 数据（不覆盖 v1）
    │   ├── dataset.csv / dataset_2d.csv   #   断言：无 stride==1、无 contiguous
    │   ├── raw/*.json
    │   └── ir_evidence/
    └── unified_formulas_v1/               # 阶段 F 的 6–8 个公式
        ├── model_unified_v1.json
        ├── errors_*.csv / metrics_summary.* / summary_stats.json
        └── plots/
```

### 8.2 验收标准（缺一不可）

- [ ] 新数据全部为**统一口径**，target_kind 只有 1 个取值（自查：扫 `raw/*.json`）
- [ ] **每条路径都有 f32（4 字节）锚点**，且与其它位宽同口径
- [ ] 交付 dataset 中 `stride == 1` 行数为 0（自动断言）
- [ ] path check **mismatches = 0**，含 stride==1/2/2 次幂 bail 对照
- [ ] **公式总数 <= 8**，且**没有任何 per-dtype 公式**（自查：列一遍公式清单）
- [ ] 每个公式都给出：系数实际值 / 逐项物理含义 / 适用域 / 不可辨识项 /
      in-sample 误差 + 分组误差 + 最差 case 归因
- [ ] 位宽确实是**因子**（在公式里显式出现，或已被字节域特征吸收），
      且按 dtype 分组的误差不存在「只有某一个位宽准、其余全崩」的情况
- [ ] `predict_*.py --verify` 逐点复算 `max |predict - errors.pred_ns| < 1e-6` → VERIFY OK
- [ ] 默认参数命令能**逐位复现**交付的 model JSON（跑两次哈希一致）
- [ ] 特征参考实现 `--verify` 通过（0 mismatch）
- [ ] 只做 in-sample，没有交叉验证 / holdout
- [ ] 如实记录 valid/invalid 比例与归因，**没有手工放宽任何阈值**
- [ ] **§6.3 逐公式汇总完成**：每个公式都有 6.3.1 逐项解释 /
      **6.3.2 数据规模与构造原则**（写清 block × dtype × stride × num_warps 怎么乘出来的、
      扣掉几个无效点、归因是什么）/ 6.3.3 两类误差 + **最差 case 清单与「误差大的是哪些 case」的归因**
- [ ] 明确写出**未覆盖的域**（misalignment / mask / dynamic stride / 负 stride / rank>=4 …）
- [ ] `git diff --stat` 证明**没有改**禁改路径
      （`costmodel/include|lib/AscendModel`、`profiles/simd_simt`、`ascend/unittest`、
      `ascend/lib`、任何 `CMakeLists.txt`）

---

## 9. 上一轮踩过的坑（请勿重犯）

1. **口径混用**（最严重）。上一轮有一版结论"现有系数完全不能跨 dtype 迁移（SIMD load 132.87%）"
   是**错的**——旧 SIMD load 模型是 cold 单发口径，而新数据是 warm 斜率口径，
   **旧系数只在 f32 行上就已经差 171.79%**。教训：**比较之前先对齐 target 定义**，
   并且**每条路径都要留一个同口径的 4 字节锚点**。
2. **特征集映射错**。上一轮把 store 的 `A_reuse` 映射到了 load 的 gather 特征集，
   导致 store 的迁移性被低估（142.87% → 修正后 6.36%）。
   教训：每个分支用**它自己 profile 里那组特征**，别想当然。
3. **不可达分支被当成可达**。store 的 `stride == 1` wide 分支在 C++ 里永远不可达
   （CostModel 拒绝 `stride <= 1`），但上一轮一度把 `stride == 2` 的行送进去。
   教训：先确认分支的**可达域**再喂数据。
4. **"修正"了不该修正的硬编码**。`stridedLineRequestSize()` 里的 `* 4.0` 看着像 bug，
   但实测它是**对的**（该量是"每行命令数 × 4B"，已吸收位宽）；改成真实位宽反而更差。
   教训：**先量，再改**；把"看起来对"写进建议书之前先用数据验证。
5. **探针 bug 造成假失败**。load 探针原先把 `tl.sum(acc)`（float32）写进与元素同类型的
   输出缓冲，int8 下被截断 → correctness 假失败。教训：**sink buffer 用 float32**，
   并且每个 dtype 都要真跑一次 correctness。
6. **bash 脚本被覆盖导致运行中崩**。`scp` 覆盖了正在运行的 bash 脚本，
   bash 是增量读取的，会读到错位内容（报 `es: command not found` 之类）。
   教训：**杀 tmux 会话之后再同步脚本**。
7. **"默认命令 == 交付内容"**。交付的 model JSON 必须能用一条默认参数命令逐位复现，
   并加自动断言（`check_acceptance.py --with-refit` 已实现，照用）。
8. **文档一次改全**。上一轮改了结论但没同步建议书，导致 §10 里还留着过时的
   "把 store 改成字节域分界"的建议。改完请全文扫一遍旧结论。

---

## 10. 参考材料（按重要性）

1. `pre-explore/README.md` —— **先读这个**，尤其 §0（数据总览）、§1.2.1.1（口径一致性矩阵）、
   §8（阶段 A）、§9（阶段 B）、§11（逐公式汇总）、§13（2D 白盒）。
2. `pre-explore/results/DATA_INVENTORY.md` —— 全部数据 + 路径。
3. `pre-explore/scripts/stride_feature_reference.py` —— 特征唯一权威实现（含 `--verify`）。
4. `pre-explore/scripts/simd_one_load_syscnt.py` —— **cold 口径旧探针**，你重测的模板。
5. `pre-explore/scripts/dtype_stride_measure_probe.py` / `stride2d_measure_probe.py` ——
   今天的 warm 探针（网格、门控、字段都照它对齐）。
6. `pre-explore/results/dtype_calibration_v1/` —— 今天的全部产物。
7. `pre-explore/results/dtype_calibration_v1/camodel_2d/` —— 2D 白盒证据。
8. CostModel 源码（**只读**）：`third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp`、
   `SimdSimtCostModel.cpp`、`third_party/ascend/lib/TritonToLinalg/StridedLoadStoreRewrite.cpp`。

---

## 11. 每个阶段结束时的汇报清单

1. **阶段 R**：cold 口径的网格 / 有效数 / 与 warm 数据的逐格对照 / 位宽因子形状 /
   旧模型在同口径下的误差 / 决策点（cold 是否取代 warm）的结论与理由。
2. **阶段 F**：公式清单（几个？分别覆盖什么？）+ 每个公式的三块内容（§6.3）+
   按 dtype 分组的误差、以及是否被某个位宽主导的判断 + 与今天 17 个公式的误差对照。
3. **阶段 S**：逐公式汇总（§6.3）——**每个公式**的逐项解释、**数据规模与构造原则**
   （乘出来的总格点数与扣减/归因）、两类误差、最差 case 清单与「误差大的是哪些 case」的结论。
4. **汇总**：一页纸总表（path / op / 位宽 / 公式来源 / n / MAPE / p90 / max / 是否采用）。
5. **接入建议**（只写文档）：需要哪些新 facts 字段、profile 字段、schema 是否 bump、
   `isValid()` 数组长度、需要重跑的 UT 清单、commit 拆分。
6. **自查证据**：`git diff --stat` 证明没碰禁改文件；`check_acceptance.py` 全绿。
7. **数据复用与新增清单**：哪些点复用了已有数据（注明来源）、哪些是新测的、
   有没有重测过任何旧点（若有，写明理由）。
8. **残余风险与未覆盖域**。
