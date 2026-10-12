# `pre-explore/results/` 数据目录清单（提交 2 版）

> **本文件对应提交 2**（"只精简数据"的那次提交）。
> **原始 campaign 测量数据已不在本提交里** —— 它们完整保存在提交 1 `5539c4831`
> （`docs(costmodel): add the full strided load/store calibration workbench`）。
> 需要原始 raw JSON 的人请 `git checkout 5539c4831 -- <路径>` 或直接切到该提交。
>
> 本文由**实际扫描目录**生成（`os.walk` + 逐个 JSON/CSV 解析），所有文件数、字节数、行数、列数、MAPE 都是实测值。
> 扫描根：`third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/results/`。
> 路径若未特别说明，均相对该根目录。生成日期：提交 2 工作区（HEAD = `5539c4831`，工作区为该提交 + 数据精简）。

**配套文档**：

| 文档 | 说明 |
|---|---|
| [`DATA_INVENTORY.md`](DATA_INVENTORY.md) | 按"标定阶段"逐条索引"哪一组测量 -> 哪个结果文件"。**注意：它描述的是提交 1 的数据布局**，其中被 `results/<campaign>/` 引用的路径需到提交 1 查 |
| [`README.md`](README.md)（本文件） | 目录级清单：现存文件、数据血缘、模型/报告索引 |
| [`../README.md`](../README.md) | 最终物理模型报告（10 条公式总表 + 逐项物理含义） |
| [`../docs/WHY-PHYSICS-OVER-GREEDY.md`](../docs/WHY-PHYSICS-OVER-GREEDY.md) | 为什么最终放弃数据驱动贪心、改用物理第一性建模 |

---

## 0. 总览

### 0.1 本提交（提交 2）的 `results/` 实测

| 项 | 值 |
|---|---|
| 顶级条目 | 13（1 个 `final/` 目录 + 12 个根目录散落文件） |
| 文件总数 | **111 个**（`final/` 99 + 根目录 12） |
| 占用空间 | **38.2 MB**（40 041 498 B 实测） |
| 其中 `final/` | 99 个文件 / 38.1 MB |
| 板卡环境 | `ssh ascend-950pr-63`（用户 `c00946898`，环境 `wj_autoscope`，256 核） |
| 板上工作目录 | `~/strided-template-calibration-v2/` |
| 统一入口 | [`../scripts/board_env.sh`](../scripts/board_env.sh)（所有 `board_*.sh` 先 source 它） |

### 0.2 提交 1 -> 提交 2 的删减（实测，三种口径）

| 口径 | 文件数 | 大小 | 说明 |
|---|---:|---:|---|
| 提交 1 **工作区**实测（旧版本文件 `os.walk` 记录） | 2381 | 168.3 MB（176 498 999 B） | 含 `.gitignore` 忽略、从未提交的 592 个文件 |
| 提交 1 **git 跟踪**（`git ls-tree -r HEAD -- pre-explore/results`） | **1789** | **138.6 MB**（145 328 894 B） | 真正进了提交 1 的部分 |
| 提交 2（本提交，工作区实测） | **111** | **38.2 MB**（40 041 498 B） | 只保留最终数据集 / 模型 / 报告 |
| 提交 2 相对提交 1 的 git 跟踪量 | **-1678 个文件（-93.8%）** | **-100.4 MB（-72.4%）** | `git status` 里 1678 条 `D`（145 328 894 B -> 40 041 498 B） |

**为什么工作区和 git 跟踪差 592 个文件**：提交 1 的 `pre-explore/.gitignore` 明确忽略了三类中间产物；
这 592 个文件当时在工作区存在，但从未进入提交 1（占 29.7 MB）：

| 被忽略的目录 | 文件数 | 忽略规则（`.gitignore`） |
|---|---:|---|
| `dtype_calibration_v1/raw/`、`measure_asm/` | 303 | `results/dtype_calibration_v*/raw/`、`.../measure_asm/` |
| `dtype_calibration_v2_rotate512/raw/` | 83 | 同上 |
| `dtype_calibration_v4_final/raw/` | 21 | 同上 |
| `merged/`（v4+v5 合并快照） | 185 | `results/merged/` |
| **合计** | **592** | `.gitignore` 注释：canonical merged raw 在 `results/final/raw/`，per-round 副本冗余 |

> 换句话说：**提交 1 里"能追溯的原始数据"= `results/final/raw/` 的 231 个 JSON（29 MB）**，
> 它是 v4/v5/v8/v9 四代 raw 的合并视图；各代 campaign 自己的 `raw/` 副本按设计未提交。

### 0.3 提交 2 删了什么（`git status` 逐类实测）

| 删除对象 | 文件数 | 说明 |
|---|---:|---|
| 37 个 campaign / 中间产物目录（整个目录树） | 1397 | 见 §8 的说明（1397 + 231 + 44 + 6 = 1678） |
| `results/final/raw/` | **231** | 29 MB，是 campaign 原始数据在 `final/` 下的合并副本 |
| `results/final/_*`（拟合中间状态） | **44** | `_p2c_*`、`_p3_*`、`_simt_*`、`_phys3_*`、`_greedy_*` 等推导过程状态 |
| `results/final/fit_v5/` + `results/final/fit_newonly/` | 6 | 统一 6 式在"全量"与"new-only"上的对照产物 |
| **合计删除** | **1678** | **脚本一个都没删（`pre-explore/scripts/` 623 个文件全部保留）** |

---

## 1. 数据血缘

### 1.1 一句话链路（含实际脚本名）

```
  [板上] 探针 Python（scripts/*_measure_probe.py、*_probe.py、camodel/*.py）
      │  经 scripts/board_env.sh 设置 asc 环境后在 ascend-950pr-63 上运行
      │  写 results/<campaign>/raw/*.json        <<< 提交 1 独有，本提交已删
      │  顶层结构：{kind, op, path, dtype, element_bytes, cases:[...], buffer, frequency_check}
      │  kind ∈ {dtype_stride_measure(1D), stride2d_measure(2D), dtype_calibration_dataset, stride2d_dataset}
      ▼
  [特征化] build_*.py ── 统一走 scripts/stride_feature_reference.py 的特征命名空间
      │  build_dtype_dataset.py    -> <campaign>/dataset.csv     (rank-1)
      │  build_2d_dataset.py       -> <campaign>/dataset_2d.csv  (rank-2)
      │  build_store_dataset.py    -> model_store_v1/dataset.csv
      │  build_template_stride_dataset.py -> model_template_stride_v*/dataset.csv
      │  build_unified_dataset.py  -> final/dataset_unified_v*.csv   (rank1+rank2 同一命名空间)
      │  build_v{2,3,4}_numwarps_dataset.py -> model_v{2,3,4}*/dataset.csv
      │  run_final_pipeline.py     -> final/dataset_new_only_v*.csv   (过滤 source 前缀)
      ▼
  [拟合] fit_*.py ── 加权最小二乘（权重 1/T，即最小化相对误差 = MAPE）、NNLS、
      │              贪心前向+后向消除、max() 结构、regime 指示展开
      │  写 results/final/MODEL_*.json、fit_*.json、model_v*.json
      │  物理重构回合写 results/final/PHYS*.json（PHYS_ = 第 1 轮，PHYS2_ = 第 2 轮，PHYS3_ = 第 3 轮）
      ▼
  [报告] gen_report*.py ── gen_report_v5.py 按 PHYS3 > PHYS2 > PHYS 优先级合并成最终报告
      │  写 results/final/REPORT_*.md
      ▼
  [文档] scripts/_readme{1..4}.py -> pre-explore/README.md + pre-explore/README_data.json
```

**一句话总结血缘**：

```
campaign 原始 json（提交 1） -> build_*.py -> dataset csv -> fit_*.py
    -> MODEL_*.json / PHYS*.json -> gen_report*.py -> REPORT_*.md -> _readme4.py -> pre-explore/README.md
```

### 1.2 各阶段的产物今天在哪里

| 阶段 | 产物 | 本提交（提交 2） | 提交 1 `5539c4831` |
|---|---|---|---|
| 板上测量 | `results/<campaign>/raw/*.json` | **已删** | 有：37 个目录；`final/raw/` 231 个 JSON 是能用 git 追溯的那一份 |
| 特征化 | `final/dataset_*.csv` | **有**（最终数据集，13 个 CSV） | 有（同左） |
| 拟合（统计线） | `final/MODEL_*.json`、`fit_*.json`、`model_v*.json` | **有**（60 个） | 有（+ 44 个 `_*` 中间状态） |
| 拟合（物理线，最终交付） | `final/PHYS*.json` | **有**（16 个） | 有 |
| 报告 | `final/REPORT_*.md` | **有**（7 个） | 有 |
| 脚本 | `pre-explore/scripts/**` | **全在**（623 个文件：604 顶层 + 19 camodel） | 有 |

**端到端重跑命令（唯一一条真正打通全链路的脚本）**

```bash
# scripts/run_final_pipeline.py 是唯一把 raw -> dataset -> new-only -> 拟合串起来的脚本
python3 scripts/run_final_pipeline.py --build \
    --unified pre-explore/results/final/dataset_unified_v5.csv
#  steps:
#   1) build_2d_dataset.py  --raw-dir results/expand2d/raw   --out-dir results/_build/build_c1
#   2) build_2d_dataset.py  --raw-dir results/expand2d_b/raw --out-dir results/_build/build_c2
#   3) build_unified_dataset.py --out results/final/dataset_unified_v5.csv --exclude-template \
#        --v1-1d results/final/dataset.csv  --v1-2d results/final/dataset_2d.csv \
#        --v3-2d results/_build/build_c1/dataset_2d.csv --v4-2d results/_build/build_c2/dataset_2d.csv
#   4) 过滤 source.startswith(v1_/v2_/v3_/v4_) -> results/final/dataset_new_only_v5.csv
#   5) 逐 slot 贪心拟合（<=10 项）-> results/final/MODEL_FINAL_V7.json
```

> **注意**：`run_final_pipeline.py` 的**第 1、2、3 步输入全部在提交 1**（`expand2d/raw`、`expand2d_b/raw`、`_build/`）。
> 在提交 2 上只能从 `--unified` 之后的步骤起步；想从 raw 重跑请用提交 1。
> 另外：`MODEL_FINAL_V7.json` 是**统计选型链**的末端，**不是**最终交付的物理公式；
> 最终 10 条公式来自 `final/PHYS3_*.json` 与 `final/PHYS2_*.json`（见 §5）。

---

## 2. `results/` 根目录现存 12 个文件

| 文件 | 大小 | 说明 |
|---|---:|---|
| `README.md` | 本文件 | 目录级清单（提交 2 版） |
| `DATA_INVENTORY.md` | 9.0 KB | 按标定阶段 A/B 的逐条数据索引。**描述的是提交 1 的布局** |
| `v4_retest_full_23.json` | 6.95 MB | "23 组配置全重测"的合并结果（顶层 `{results:[...]}`），用于把早期不对齐的 v4 数据替换掉 |
| `_phys_r2_table.json` | 116 KB | 第 2 轮物理重构的 222 行分组表 |
| `_phys_r2_models.json` | 2.8 KB | 第 2 轮 12 个候选模型（M1..M12）的 MAPE 与系数 |
| `_phys_r2_final.json` | <1 KB | 分组终选 |
| `_phys_r2_greedy.json` / `_phys_r2_greedy2.json` | <1 KB 各 | 第 2 轮贪心对照结果 |
| `_phys_r2_lib.json` | <1 KB | 特征名表 |
| `_p3_store_r1_final.json` | 764 B | 第 3 轮 store_r1 的 4 组候选（`old/new5/new6/val`）对比 |
| `strided_load_summary.csv` | 1.2 KB | CAModel 白盒汇总表（run/mode/block/stride/useful_bytes/span_bytes/task_cycles/BIU cmds/DC reqs…） |
| `strided_load_summary.md` | 1.0 KB | 上表的 Markdown 版 |

> 根目录这 7 个 `_*.json` 是**刻意保留**的：它们是第 2/3 轮物理重构的候选模型与终选记录，
> 体积很小但能证明"最终公式不是唯一候选、是比出来的"。
> 两份 `strided_load_summary.*` 出自 `scripts/make_summary.py`（**该脚本已在后续的脚本清理中删除**；
> 推导记录见 [`../docs/agent-logs/`](../docs/agent-logs/)，需要时用 `git show HEAD:pre-explore/scripts/make_summary.py` 取回），
> 其数据源 `results/block{32,64,128}_s3/` 与 `results/store_camodel_b{32,64,128}/` **已删（见提交 1）**。

---

## 3. `final/dataset_*.csv`：13 个 CSV + 2 个 meta（实测行/列）

行数为**数据行数**（不含表头）。列名只摘前几列，完整 59 列释义见 §3.2。

### 3.1 全部 dataset 文件

| 文件 | 行数 | 列数 | 生成脚本 | 用途 |
|---|---:|---:|---|---|
| `dataset.csv` | 2560 | 51 | [`build_dtype_dataset.py`](../scripts/build_dtype_dataset.py) | rank-1（1D）主数据集，v4 代 raw 的特征化结果 |
| `dataset_2d.csv` | 3432 | 48 | [`build_2d_dataset.py`](../scripts/build_2d_dataset.py) | rank-2（2D tile）主数据集 |
| `dataset_unified.csv` | 8016 | 43 | [`build_unified_dataset.py`](../scripts/build_unified_dataset.py) | 第一次统一：rank1+rank2 重特征化到同一命名空间 |
| `dataset_unified_dedup.csv` | 7895 | 43 | 同上 + 去重 | 去重版（-121 行） |
| `dataset_unified_v2.csv` | 8016 | 56 | 同上 | v2：+ 2D 行/列/间距特征 |
| `dataset_unified_v3.csv` | 8016 | 59 | 同上 | v3：+ `valid` / `quality` / `target_spread` |
| `dataset_unified_v4.csv` | 9456 | 59 | 同上 | v4：并入 v8/v9 代 raw |
| **`dataset_unified_v5.csv`** | **12040** | **59** | 同上 | **全量统一数据集**（并入 expand2d / expand2d_b 两轮 2D 扩形）；用于"跨口径对照" |
| `dataset_new_only.csv` | 5918 | 43 | [`run_final_pipeline.py`](../scripts/run_final_pipeline.py) | "只看本轮新测"视图（过滤 `source` 前缀） |
| `dataset_new_only_v2.csv` | 6039 | 56 | 同上 | 同上，v2 列集 |
| `dataset_new_only_v3.csv` | 6039 | 59 | 同上 | 同上，v3 列集 |
| `dataset_new_only_v4.csv` | 7479 | 59 | 同上 | 同上，并入 v8/v9 |
| **`dataset_new_only_v5.csv`** | **10063** | **59** | 同上 | **最终拟合用数据集**（物理公式全部在它上面拟合，取 `valid=1` 的 9541 行） |
| `dataset_meta.json` | 21.9 KB | — | `build_dtype_dataset.py` | 元数据：`kind=dtype_calibration_dataset`、total 2560 / valid 2371 / invalid 189、coverage、`feature_source` |
| `dataset_2d_meta.json` | 11.0 KB | — | `build_2d_dataset.py` | 元数据：`kind=stride2d_dataset`、total 3524 / valid 3262 / invalid 262、`scope`、coverage |

> 口径说明：**14 个 `dataset_*` 条目 = 13 个 CSV + `dataset_meta.json`**；若把 `dataset_2d_meta.json` 也算进来则是 15 个 `dataset*` 文件。
> 这里按"CSV 数据文件"计为 13，按"dataset 前缀文件全算"计为 15，两个数都对，取决于是否把 `dataset_2d_meta.json` 计入。

### 3.2 59 列的组成（`dataset_unified_v5.csv` / `dataset_new_only_v5.csv`）

所有特征由 [`stride_feature_reference.py`](../scripts/stride_feature_reference.py) 唯一实现（`dataset_meta.json` 的 `feature_source` 也这么记），
原则是**一律用字节域或纯计数，绝不用元素域 stride**，这样 dtype 宽度以"因子"而非"分支"进入公式。

| 列组 | 列名 | 含义 |
|---|---|---|
| **实验条件（11 列）** | `op` / `path` / `dtype` / `element_bytes` | load,store / simd,simt_only / int8,float16,float32,int64 -> 1/2/4/8 |
| | `num_warps` / `rank` / `elements` / `inner_stride_bytes` | W 1..64 / 1 或 2 / 元素总数 n / ISB = stride x eb |
| | `source` / `domain` | 数据来源（v1_1d,v1_2d,v2_1d,v3_2d,v4_2d,hist_*）/ rotate 或 single_shot |
| **目标值（4 列）** | `target_ns` | 板上 Event 测得的每次迭代耗时（ns），同一配置多次 pass 取 **min** |
| | `target_kind` / `valid` / `quality`, `target_spread` | 口径标记 / 四道门（witness+频率+spread+数值正确性）/ 重复测量离散度（>0.2 视为双稳区） |
| **地址表基本量（8 列）** | `f_elements`, `f_L` | 元素数；distinct 128 B line 数 |
| | `f_line_elems_max/min`, `f_line_density` | 单条 128 B line 内最多/最少/平均元素数 |
| | `f_span_bytes`, `f_page_cross`, `f_cross` | 首末字节跨度；跨 4 KB 页数；跨页标志 |
| **warp 行为（4 列）** | `f_mean_warp_lines` / `f_min_warp_lines` / `f_max_warp_lines` / `f_iters_per_thread` | 每 warp 触碰 line 数的均值/最小/最大；每线程循环次数 |
| **热点/冲突（4 列）** | `f_bucket_pairs_2k8` / `f_bucket_worst_2k8` / `f_bucket_pairs_32k16` / `f_bucket_worst_32k16` | 2 KB/8-bucket 与 32 KB/16-bucket 粒度下的命令冲突对数与最拥塞 bucket |
| **命令与事务（8 列）** | `f_K`, `f_K_stg`, `f_dup`, `f_n_cmd_block` | 指令条数（LDG/STG）；重复请求 line 数；per-element BIU 命令数 |
| | `f_line_request_size`, `f_line_request_size_cpp4`, `f_tail_elems`, `f_rank` | 每条 line 请求字节量（真实位宽 / C++ 语义 4 B 每元素）；`max(0,n-64)`；rank 数值副本 |
| **分段/阈值指示（8 列）** | `f_min_L_32`, `f_max_L_minus_32`, `f_dupL_minL_128`, `f_cross_minL_32`, `f_lowW_4_L` | `min(L,32)`、`max(0,L-32)`、`dup x L x min(L,128)`、跨页 x `min(L,32)`、`max(0,4-W) x L` |
| | `f_inner_stride_bytes`, `f_row_gap_bytes` | 内层跨步字节；行间空隙字节 |
| **2D 专有（9 列）** | `f_rows`, `f_cols`, `f_row_span_bytes`, `f_row_pitch_bytes`, `f_row_lines`, `f_gap_lines`, `f_comb_lines`, `f_comb_overlap`, `f_rows_share_line`, `f_row_segments`, `f_gap_per_line`, `f_row_span_div_gap`, `f_tail_row_elems` | R、C、行字节跨度、行间距、行内 line 数、空隙 line 数、行列 line 并集、重叠量、是否共享 line、行被切几段、每 line 空隙、跨度/空隙比、末行余量 |

字段分布（`dataset_unified_v5.csv` 实测）：

| 维度 | 分布 |
|---|---|
| `op` | load 5790 / store 6250 |
| `path` | simt_only 8812 / simd 3228 |
| `dtype` | float32 4254 / int8 2775 / float16 2729 / int64 2282 |
| `rank` | 1 -> 4584 / 2 -> 7456 |
| `num_warps` | 1:6530、2:594、4:2540、8:594、16:594、32:594、64:594 |
| `valid` | 1 -> 11518 / 0 -> 522（new-only 中 1 -> 9541 / 0 -> 522） |
| `domain` | rotate 11599 / single_shot 441 |

### 3.3 `dataset_unified_v5` vs `dataset_new_only_v5`：为什么两个文件

`build_unified_dataset.py` 把 5 类来源拼进一张表，行首 `source` 列标明出处：

| `source` | 行数 | 含义 |
|---|---:|---|
| `v1_2d` | 3392 | 本轮 `final/dataset_2d.csv`（rank-2） |
| `v1_1d` | 2184 | 本轮 `final/dataset.csv`（rank-1） |
| `v3_2d` | 2112 | `_build/build_c1`（`expand2d/raw`，见提交 1） |
| `v4_2d` | 1952 | `_build/build_c2`（`expand2d_b/raw`，见提交 1） |
| `hist_v4` | 992 | **历史复用**：`model_v4_numwarps` / `model_v4_simple` 的数据（见提交 1） |
| `hist_store` | 544 | **历史复用**：`model_store_v1/dataset.csv`（见提交 1） |
| `hist_v2_single_shot` | 441 | **历史复用**：`model_v2`（`domain=single_shot`，见提交 1） |
| `v2_1d` | 423 | `dtype_calibration_v2_rotate512` 的 1D 行（见提交 1） |

`dataset_new_only_v5.csv` = 只保留 `source ∈ {v1_1d, v1_2d, v2_1d, v3_2d, v4_2d}` 的 10063 行（丢掉 3 类 `hist_*` 共 1977 行）。
**为什么拟合要用 new-only**：`hist_v2_single_shot` 的 target 口径不同（单次发射而非 rotate-loop slope），混进来会污染标定；
所以全量 unified 只用于"跨口径对照"，真正拟合一律用 new-only。

---

## 4. `final/MODEL_*.json`：41 个大写 `MODEL_*` + 4 个 `model_v*` = **45 个模型文件**

口径：除注明外，全部 slot 级模型的方法都是 **加权最小二乘，权重 = 1/T**（等价于直接最小化 MAPE）；
`MAPE` 列为该 slot 的 **in-sample** MAPE（%）；`项数` 为该式的项数（含截距，按各文件 `n_terms` / 列表长度实测）；
`总 MAPE` 取各文件自带的 `_overall`（按 n 加权）。`-` 表示该文件没有这一字段（例如只存 `base`+`coefs`、或只存结构）。

### 4.1 早期整模型（4 个）

| 文件 | slot | 项数 | slot MAPE | 总 MAPE | 方法 / 备注 |
|---|---|---:|---|---:|---|
| `model_v9.json` | 5 | 4-7 | 16.7-34.2 | 23.50 | 最早整模型；1/T 相对加权 + 非负线性，无 log/exp/幂；数据集 `dataset_new_only.csv`(5918 行) |
| `model_v15_final.json` | 5 | 231 | 9.50 / 22.86 / 16.19 / 9.95 / 15.83 | 14.54 | 全 231 特征整模型（不做任何选型） |
| `model_v16_backward.json` | 5 | 214-223 | 9.27 / 22.74 / 16.02 … | 14.42 | 后向消除版；1/T 加权 WLS，字节域原始量线性项 |
| `model_v17_dedup_unc.json` | 5 | 157 | 见文件 | 14.65 | 特征去重 + 不确定性处理版 |

### 4.2 指示因子展开线（项数爆炸 -> 揭示"堆指示器"的极限）

| 文件 | slot | 项数 | slot MAPE | 总 MAPE | 方法 / 备注 |
|---|---|---:|---|---:|---|
| `MODEL_FINAL.json` | 5 | 5-6 | 13.85-26.26 | 17.41 | 每特征 x dtype 指示（eb in 1/2/4/8），系数全非负 -> 每项都是成本 |
| `MODEL_FINAL_V2.json` | 5 | 181-521 | 5.15-15.42 | 7.23 | 再加 num_warps / rank 指示因子（严重过参数化） |
| `MODEL_FINAL_V3.json` | 5 | 547-1353 | 1.17-5.34 | 4.07 | 同上（**项数爆炸，已废弃**：最多 1353 项/式，比 `MODEL_FINAL` 的 17.41% 低了 13 个点，但完全不可解释；见 `WHY-PHYSICS-OVER-GREEDY.md` §5.3） |
| `MODEL_DELIVER.json` | 5 | 261 | 6.78-20.65 | 9.95 | "交付版"：>140 项/式，全部基础特征乘 dtype/num_warps/rank 指示因子 |
| `MODEL_INDICATOR_FULL.json` | 9 | 73 | 6.87-20.95 | - | 指示器全展开（9 个 slot） |
| `MODEL_COMPACT.json` | 5 | 13 | 9.79-20.30 | 14.90 | 每 slot 恰 12 项（+截距），权重 1/T |
| `MODEL_COMPACT_12.json` | 5 | 13 | 9.79-20.30 | 14.90 | 同上；`_note` 记录拟合在 5686 个 probe-valid 点上 |
| `MODEL_V22.json` | 5 | 222-339 | 5.97-18.96 | - | 早期整模型（只存 mape / n / base / mods / n_terms） |
| `MODEL_V23.json` | 5 | 222-573 | 5.00-13.93 | - | 早期整模型 |

### 4.3 分 rank + <=10 项线（`run_final_pipeline.py` 的产物族）

| 文件 | slot | 项数 | slot MAPE | 总 MAPE | 方法 / 备注 |
|---|---|---:|---|---:|---|
| `MODEL_FINAL_V4.json` | 10 | 8-10 | 0.78-17.35 | 11.68 | 按 rank 拆成 10 个 slot，每 slot <=10 项 |
| `MODEL_FINAL_V5.json` | 10 | 8-10 | 0.78-17.35 | 11.68 | 同 V4（仅重跑确认，数值完全相同） |
| `MODEL_FINAL_V6.json` | 10 | 8-10 | 0.73-22.19 | 14.17 | 同 V4，数据换成含 `hist_*` 的更大集 |
| `MODEL_FINAL_V7.json` | 10 | 8-10 | 0.74-25.98 | 16.31 | **`run_final_pipeline.py` 的末端产物**：贪心前向+后向消除，<=10 项；`simt_store_r2` 仍有 25.98% |
| `MODEL_FINAL_10TERM.json` | 10 | 8-10 | 4.08-21.96 | 14.79 | 固定 <=10 项 |
| `MODEL_10TERM_FINAL.json` | 10 | 11 | - | - | 固定 10 项，只存 `base`+`coefs` |
| `MODEL_BACKELIM10.json` | 10 | 11 | 1.36-21.36 | 15.24 | 后向消除到 11 项 |
| `MODEL_EXHAUST5.json` | 10 | 6 | 1.46-24.30 | 16.97 | 穷举到 6 项 |
| `MODEL_RANK_SPLIT.json` | 10 | 11 | 0.87-19.84 | 13.06 | 按 rank 分治，11 项 |
| `MODEL_RANKSPLIT_NNLS.json` | 10 | 26-37 | 1.13-22.00 | - | 同上但用 **NNLS**（非负最小二乘） |
| `MODEL_TOPK30.json` | 10 | 2-11 | 4.03-29.20 | - | 从 top-30 候选里选（K 记在文件里） |
| `MODEL_BESTOF.json` | 10 | 11-73 | 0.57-20.95 | - | 逐 slot 取各方案最优（无项数约束；`simd_load_wide_r1` 用 73 项换 0.57%） |
| `MODEL_HYBRID.json` | 10 | 11-73 | 0.74-20.95 | - | 逐 slot 混选（在 FINAL_V7 与 BESTOF 之间按 slot 取优） |
| `MODEL_HYBRID2.json` | 10 | 11-73 | 0.73-22.46 | - | 同上，换数据后重跑 |

### 4.4 交叉项 / 指示器 / regime 线（**数据驱动贪心路线的完整记录**）

| 文件 | slot | 项数 | slot MAPE | 总 MAPE | 方法 / 备注 |
|---|---|---:|---|---:|---|
| `MODEL_10CROSSED.json` | 10 | 9-11 | 1.31-24.33 | 16.57 | 每项为"变量 x 变量"的交叉乘积，贪心选 <=10 项 |
| `MODEL_10CROSSED_DOMAIN.json` | 10 | 9-11 | 1.23-24.08 | 15.58（`_overall_kept`） | 同上 + `domain` 剔除规则；带 `n_kept` / `coverage` / `rule` / `frontier` |
| `MODEL_MODSEL.json` | 10 | **6-113** | 3.46-17.75 | **11.29** | **按公式逐位选型**：dtype 指示 / num_warps 指示 / 前向选出的 regime 标志；对应 `REPORT_v1_modsel.md` |
| `MODEL_REGIME.json` | 10 | 11-91 | 0.00-19.57 | - | regime 标志 + 全交叉项；`simd_load_wide_r1` MAPE **0.00%**（典型过拟合） |
| `MODEL_REGIME_FINAL.json` | 10 | 109 | 0.00-16.32 | 11.27 | regime 6 标志（`nge256,nlt64,mwlg2,Lgt16,Rgt8,Clt8`） |
| `MODEL_REGIME_V2.json` | 10 | 27-209 | 3.47-15.30 | **9.91** | regime **14 标志**（`_regime_flags` 实测 14 个），`_n_base=8` |
| `MODEL_REGIME_V3.json` | 10 | 28-217 | 3.48-15.17 | **9.80** | regime **15 标志**；对应 `REPORT_v3_10crossed.md` 一族 |
| `MODEL_PRUNED.json` | 10 | 13-97 | 3.40-15.89 | 10.87 | 剪枝到 12-96 项；曲线见 `PRUNE_CURVE.json` |
| `MODEL_DOMAIN.json` | 10 | 2-11 | - | 18.60（kept）/ 18.27（all） | 带 `exclusion_rule` / `coverage` / `mape_kept`，剔除不适应点后重拟 |

> **这一族是 [`../docs/WHY-PHYSICS-OVER-GREEDY.md`](../docs/WHY-PHYSICS-OVER-GREEDY.md) 的证据来源**：
> 项数从 6 到 113 按公式任意分化、总体 in-sample MAPE 最好 9.80-11.29%，且 `MODEL_REGIME.json` 里出现 0.00% 的 slot ——
> 这些都是"在同数据上既选特征又拟合"的过拟合特征。

### 4.5 `max()` 结构线

| 文件 | slot | 项数 | slot MAPE | 总 MAPE | 方法 / 备注 |
|---|---|---:|---|---:|---|
| `MODEL_MAXLINEAR.json` | 10 | - | 0.78-17.18 | 11.81 | 线性项 + `max()` 包裹（记 `linear_features`/`linear_coefs`/`max_qty`/`max_coef`） |
| `MODEL_MAXPLUS.json` | 10 | - | 4.20-37.23 | 20.09 | `max()` 结构搜索（`K` + `branches`） |
| `MODEL_MAXPLUS2.json` | 10 | - | 0.78-17.35 | 11.68 | 同上修正版 |
| `MODEL_MAXSTRUCT.json` | 7 | - | - | - | `max()` 结构，显式记录 `mape_max`/`mape_linear`/`a_ns_per_warp`/`warp_dominated`；**证明 W>=16 时发射成本与元素数无关** |
| `MODEL_PHYSMAX.json` | 10 | - | 0.78-17.35 | 11.68 | 物理式 + `max()` |

### 4.6 统计线里的物理尝试 + 专项（4 个）

| 文件 | slot | 项数 | slot MAPE | 总 MAPE | 方法 / 备注 |
|---|---|---:|---|---:|---|
| `MODEL_PHYS.json` | 5 | 7 | 12.74-28.83 | 21.46 | 纯物理式但**不分 rank**（5 slot），第 1 轮物理尝试 |
| `MODEL_PHYS_SMALL.json` | 10 | 4-9 | 1.85-32.63 | 21.95 | 物理式压缩到 4-9 项；例：`simd_load_wide_r1 = 66.4+0.6032*L+3.619*maxL32+0.0111*n+10.8*spanperN-1.025*eb` |
| `MODEL_GATHER.json` | 4 | - | - | - | SIMD load gather 专项 |
| `MODEL_GATHER2.json` | 4 | - | - | - | 同上改进版 |

> **`MODEL_FINAL_V7.json` 不是最终交付**：它由统计贪心选出，`simt_store_r2` 仍有约 26% 误差。
> 最终交付的是 §5 的 10 条物理公式。

---

## 5. `final/PHYS*.json`：16 个（**最终交付物**）

每个文件结构为 `{<slot>: {n, mape, n_terms, formula, terms:[{name, coef, meaning}], derived, scope, notes, ...}, ...}`。
方法：**由硬件物理过程直接构造项**（搬运字节、BIU 命令数、同块冲突、warp 发射开销），不做 log/指数/幂次，
系数用加权最小二乘（权重 1/T）或 NNLS 定出；`max()` 结构的式额外做交替分配 / 坐标下降、多随机起点取最优。

### 5.1 三轮谱系

| 轮次 | 文件 | slot 数 | 说明 |
|---|---|---:|---|
| 第 1 轮（`PHYS_*`） | `PHYS_mine.json` | 2 | 最早的手工物理式（wide 式尚无 K 饱和项） |
| | `PHYS_simd_load_wide_r2.json` / `PHYS_simd_load_gather_r2.json` | 1 / 1 | 第 1 轮 2D |
| | `PHYS_simd_store.json` / `PHYS_simt_store.json` / `PHYS_simt_load.json` | 2 / 2 / 2 | 第 1 轮 store / SIMT store / SIMT load |
| 第 2 轮（`PHYS2_*`） | `PHYS2_wide_simtload.json` | 4 | wide_r1/r2 + simt_load_r1/r2 |
| | `PHYS2_simt_store.json` | 2 | simt_store_r1/r2 |
| | `PHYS2_gather.json` / `PHYS2_simd_store.json` | 2 / 2 | 已被第 3 轮版本取代 |
| 第 3 轮（`PHYS3_*`） | `PHYS3_simd_load_wide_r1` / `_wide_r2` / `_gather_r1` / `_gather_r2` / `_store_r1` / `_store_r2` | 各 1 | **最终 6 个 slot 的定稿**，每个文件只装一个 slot |

### 5.2 全部 16 个文件的实测 MAPE / 项数

| 文件 | slot | n | MAPE | 项数 | 5 折 / 留一 |
|---|---|---:|---:|---:|---:|
| `PHYS_mine.json` | simd_load_wide_r1 | 32 | 4.11 | 2 | - |
| | simd_load_gather_r1 | 284 | 20.18 | 7 | - |
| `PHYS_simd_load_wide_r2.json` | simd_load_wide_r2 | 222 | 13.10 | 8 | - |
| `PHYS_simd_load_gather_r2.json` | simd_load_gather_r2 | 678 | 13.18 | 8 | - |
| `PHYS_simd_store.json` | simd_store_r1 / r2 | 291 / 851 | 9.69 / 25.26 | 6 / 5 | - |
| `PHYS_simt_store.json` | simt_store_r1 / r2 | 962 / 3224 | 18.95 / 25.24 | 8 / 8 | - |
| `PHYS_simt_load.json` | simt_load_r1 / r2 | 855 / 1666 | 10.22 / 10.79 | 8 / 9 | - |
| `PHYS2_wide_simtload.json` | simd_load_wide_r1 / r2、simt_load_r1 / r2 | 21 / 120 / 506 / 1211 | **1.73** / 5.79 / **4.95** / **6.34** | 2 / 8 / 7 / 6 | - |
| `PHYS2_gather.json` | simd_load_gather_r1 / r2 | 136 / 438 | 9.54 / 9.52 | 5 / 8 | - |
| `PHYS2_simd_store.json` | simd_store_r1 / r2 | 189 / 325 | 5.33 / 8.62 | 5 / 4 | 5.55 / 8.73 |
| `PHYS2_simt_store.json` | simt_store_r1 / r2 | 962 / 1210 | **6.27** / **9.64** | 7 / 8 | - |
| `PHYS3_simd_load_wide_r1.json` | simd_load_wide_r1 | 21 | **1.22** | 3 | - |
| `PHYS3_simd_load_wide_r2.json` | simd_load_wide_r2 | 120 | **4.48** | 7 | - |
| `PHYS3_simd_load_gather_r1.json` | simd_load_gather_r1 | 136 | **8.70** | 6 | 9.19（留一） |
| `PHYS3_simd_load_gather_r2.json` | simd_load_gather_r2 | 438 | **8.44** | 7 | 8.70 |
| `PHYS3_simd_store_r1.json` | simd_store_r1 | 189 | **5.33** | 5 | 5.55 |
| `PHYS3_simd_store_r2.json` | simd_store_r2 | 325 | **8.12** | 4 | 8.24 |

### 5.3 最终 10 条公式 = 6 + 4

```
PHYS3_simd_load_wide_r1 + PHYS3_simd_load_wide_r2
+ PHYS3_simd_load_gather_r1 + PHYS3_simd_load_gather_r2
+ PHYS3_simd_store_r1 + PHYS3_simd_store_r2
+ PHYS2_wide_simtload (simt_load_r1 / simt_load_r2)
+ PHYS2_simt_store    (simt_store_r1 / simt_store_r2)
```

与 [`../README.md`](../README.md) 的总表逐行 MAPE 完全对齐（1.22 / 8.70 / 5.33 / 4.95 / 6.27 / 4.48 / 8.44 / 8.12 / 6.34 / 9.64），
加权平均 **7.22%**（5118 个保留实测点）。

写出这些文件的脚本（**⚠ 下面这一整组写回脚本都已在脚本清理中删除**，`PHYS*.json` 现在是**冻结交付物**）：

| 文件 | 原写出脚本（**均已删除**） |
|---|---|
| `PHYS3_simd_load_wide_r1.json` | `_p3_w1_write.py`、`_w1.py`、`_fixrate.py`、`_clean3.py` |
| `PHYS3_simd_load_gather_r1.json` | `_p3_g1_write.py`、`_p3_g1_patch.py`、`_p3_g1_verify.py` |
| `PHYS3_simd_store_r1/r2.json` | `p3_store_r1_write.py`、`phys3_write3.py`、`write_phys2_store.py` |
| `PHYS2_wide_simtload.json` | `phys2_final_report.py` |
| `PHYS2_simt_store.json` | `p2_r2c_write.py`、`p2_final_write.py` |
| `PHYS2_gather.json` | `p2_writejson.py`、`p2_verify2.py` |
| 全部 PHYS 文件的项名中文化 | `_ren.py` |

> **这些脚本的推导记录仍在**：[`../docs/agent-logs/`](../docs/agent-logs/) 的 11 份 `p3_*.txt`，
> 以及 git 历史（`git show HEAD:pre-explore/scripts/p3_feat.py`、`git checkout HEAD -- <路径>` 可恢复）。
> **后果**：报告链仍然是通的（`gen_report_v5.py` / `gen_report_concise.py` 直接读这些 JSON），
> 但**无法用现存脚本重新生成 `PHYS*_*.json`**。要重跑物理拟合轮次，先从 git 取回对应写回脚本，
> 或改走 `run_final_pipeline.py` 的贪心路线（产出 `MODEL_FINAL_V7.json`，不是 PHYS 族）。

---

## 6. `final/fit_*.json`（11 个）+ 辅助 json（5 个）

### 6.1 `fit_*.json`：逐轮搜索的中间状态（11 个）

| 文件 | 大小 | slot 数 | 说明 |
|---|---:|---:|---|
| `fit_v9_greedy.json` | 2.7 KB | 5 | 最早一轮贪心 |
| `fit_v10_exhaustive.json` | 2.5 KB | 5 | 穷举 |
| `fit_v11_stepwise.json` | 4.1 KB | 5 | 逐步回归 |
| `fit_v12_big.json` | 4.2 KB | 5 | 放大候选池 |
| `fit_v13.json` | 898 B | 5 | 收敛轮 |
| `fit_v14_crossed.json` | 5.2 KB | 5 | 交叉项 |
| `fit_v15_Rcross.json` | 663 B | 5 | 行数 R 相关交叉 |
| `fit_v18_indicator.json` | 943 B | 5 | 指示器 |
| `fit_v19_indicators.json` | 887 B | 5 | 指示器（多组） |
| `fit_v20_crosscand.json` | 1.4 KB | 5 | 交叉候选 |
| `fit_v21.json` | 1.5 KB | 5 | 最后一轮统计搜索 |

> 这 11 个文件是**搜索轨迹**，不是交付物；它们记录了统计路线从 5 个 slot 走到 10 个 slot、
> 从纯线性走到交叉项/指示器的过程。四条 `model_v*` 见 §4.1。

### 6.2 辅助 json（5 个）

| 文件 | 大小 | 内容 |
|---|---:|---|
| `COMPOSITION.json` | 7.8 KB | 10 个 slot 的数据构成：每个 slot 的 n、shapes 分布、`inner_stride_bytes` 分布、dtypes 分布 |
| `FEATURE_MEANING.json` | 7.4 KB | 特征中文释义表（`L` = 触及的 128B line 数；`n` = 元素数 = BIU 命令数；`iters` = ceil(n/(32W)) 等） |
| `DOMAIN_CURVE.json` | 32.8 KB | 各 slot 的 domain 收敛曲线：每条剔除规则下的 `coverage` / `mape` / `n` |
| `PRUNE_CURVE.json` | 3.0 KB | 剪枝曲线：`per_slot` + `overall`（4 项 34.23% -> 8 项 31.9 -> 16 项 25.14 -> 32 项 17.36 -> **96 项 10.86%**） |
| `features_dedup.json` | 2.0 KB | 156 个特征的去重后列表（`L, Lf, n, eb, ebL, ebn, nW, iters, minL32, maxL32, minL128, W, lowW, rowgap, cmd, ...`） |

---

## 7. `final/REPORT_*.md`：7 个（版本谱系）

| 文件 | 大小 | 行数 | 版本 | 内容 | 生成脚本 |
|---|---:|---:|---|---|---|
| `REPORT_v1_modsel.md` | 85 KB | 1751 | **v1：MODSEL（逐位选型）** | 每个公式在候选池里逐位选型（dtype 指示、num_warps 指示、regime 标志），项数 6-113，附"换成哪个公式收益多少"的对照 | `gen_reports.py`（**已删除**，见 §5 脚注） |
| `REPORT_v2_regime.md` | 175 KB | 3525 | **v2：REGIME（全指示器展开）** | 每式用全部 14 个 regime 标志 + 4 个 dtype 指示 + 7 个 num_warps 指示，项数爆炸但 in-sample 最小 | `gen_reports.py`（**已删除**，见 §5 脚注） |
| `REPORT_v3_10crossed.md` | 20 KB | 529 | **v3：10 CROSSED** | 每式 <=10 项，且每项是"变量 x 变量"的乘积，用贪心挑选 | `gen_reports.py`（**已删除**，见 §5 脚注） |
| `REPORT_v4_physics.md` | 35 KB | 581 | **v4：PHYSICS（第 1 轮物理重构）** | 每个式由物理过程重建；不用 log/指数/幂次；加权最小二乘（1/T）。**表里带"贪心对照"列** | [`gen_report_phys.py`](../scripts/gen_report_phys.py)（合并全部 `PHYS_*.json`） |
| `REPORT_v5_physics_scope.md` | 57 KB | 737 | **v5：PHYSICS + 真实算子范围（详细版）** | 在 v4 基础上把每个式限定在真实 AI 算子取值范围内；逐条给出剔除理由 + 保留率 + 误差 | [`gen_report_v5.py`](../scripts/gen_report_v5.py)（按 `PHYS3 > PHYS2 > PHYS` 合并 + `COMPOSITION.json`） |
| `REPORT_v5_physics_scope_concise_des.md` | 35 KB | 553 | **v5（精简中文版）：一页读懂模型** | 每个式的物理直构、派生量定义、公式与各项物理意义、误差；只保留最终留存集 | [`gen_report_concise.py`](../scripts/gen_report_concise.py) |
| `REPORT_PHYS2_simt_store.md` | 4.7 KB | 87 | **第 2 轮：simt_store 专项** | simt_store_r1/r2 单独一轮回归；含"第一轮 15 种跳变惩罚全被压到 0"与 `n*min(R,32)` 的关键发现 | 第 2 轮 p2_r2c 系列脚本 |

**版本谱系一句话**：

```
v1 MODSEL（逐位选型，6-113 项）
  -> v2 REGIME（全指示器展开，项数爆炸但 in-sample 最小）
  -> v3 10 CROSSED（<=10 项交叉乘积）
  -> v4 PHYSICS（第 1 轮物理重构，附贪心对照）
  -> v5 PHYSICS + 真实算子范围（最终：10 式全部 <10%，加权 7.22%）
     + v5 精简中文版（一页读懂）
     + PHYS2_simt_store 专项（第 2 轮，含跳变自变量发现）
```

**报告与 README 的关系**：`pre-explore/README.md`（10 条公式总表 + PASS/FAIL 判定）是这些报告的**摘要**，
其数据源是 `pre-explore/README_data.json`，由 `_readme.py` / `_readme2.py` / 
`_readme3.py` / [`_readme4.py`](../scripts/_readme4.py) 生成（`_readme.py` / `_readme2.py` / `_readme3.py` 是它的三个早期版本，**均已删除**，推导记录见 [`../docs/agent-logs/`](../docs/agent-logs/) 与 git 历史），正文引用 `REPORT_v5_physics_scope_concise_des.md`。
因此改报告后应重跑 `_readme4.py`，而不是手改 `pre-explore/README.md`。

---

## 8. 本提交已没有的东西（见提交 1 `5539c4831`）

| 已删除 | 数量 | 内容 | 为什么能删 |
|---|---:|---|---|
| 37 个 campaign / 中间产物目录 | 1434 文件 | `dtype_calibration_v1`、`v2_rotate512`、`v4_final`、`v4raw`、`v5raw`、`v8raw`、`v9raw`、`expand2d`、`expand2d_b`、`dir23`、`wide12`、`board_v2_*`、`model_store_v1`、`model_template_stride_v1/v2`、`model_v2/v3_numwarps/v4_numwarps/v4_simple/v4_simple_p25/v7_simple`、`refit_no_stride1_v1`、`unified_formulas_v1/v2`、`_build`、`_tmpbuild`、`block*_s3`、`store_camodel_b*` | 全部已被特征化成 `final/dataset_*.csv`；原始证据可从提交 1 取回 |
| `results/final/raw/` | 231 文件 / 29 MB | v4/v5/v8/v9 四代原始 JSON 的合并视图（文件名带 `vN__` 前缀） | 同上；**这是能用 git 追溯的原始数据** |
| `results/final/_*` | 44 文件 | `_p2c_*`、`_p3_*`、`_simt_*`、`_phys3_*`、`_greedy_*` 等拟合中间状态 | 推导轨迹在提交 1 与 [`../docs/agent-logs/`](../docs/agent-logs/) 里都有 |
| `results/final/fit_v5/`、`fit_newonly/` | 6 文件 | 统一 6 式在全量 / new-only 上的对照产物 | 结论已在 §4/§7 记录 |

**脚本在提交 2 里一个都没删**：`pre-explore/scripts/` 当时共 623 个文件（顶层 604 = 96 个 `.sh` + 507 个 `.py` + `INDEX.md`；`camodel/` 子目录 19 个 = 13 `.sh` + 6 `.py`）。

> **后续更新（脚本清理，提交 2 之后）**：又做了一轮「只保留能直接跑出全量实测数据的必要脚本」的清理，
> 删掉 **518 个**子代理推导的中间产物 —— `_p2_*`/`p2_*` 154、`_p3_*`/`p3_*` 75、一次性 `_*.py` 66、
> `phys_*` 53、`phys3_*` 26、实验性 `fit_*` 33、一次性诊断 `.sh` 57；
> 并从 git **恢复了被依赖闭包误删的** `board_env.sh` / `board_env_latest.sh`（所有板端调用的环境包装器）。
> 现存 **106 个脚本**（顶层 49 `.py` + 47 `.sh`，`camodel/` 3 `.py` + 7 `.sh`）。
> 索引见 [`../scripts/INDEX.md`](../scripts/INDEX.md)（含完整复现调用链与「测量批次 campaign」术语表）；
> 被删脚本的推导记录见 [`../docs/agent-logs/`](../docs/agent-logs/)（11 份 `p3_*.txt`）与 git 历史。

另外，目录里从来没有（也不要再找）：

- `.npubin` 编译产物（约 17.7 MB）
- `.pkl` / `.npy` 特征缓存（约 8.3 MB）
- `.pyc` / `__pycache__`
- 嵌套的 `third_party/` 副本

---

## 9. 血缘速查表

| 产物 | 由谁生成 | 由谁消费 | 本提交是否还在 |
|---|---|---|---|
| `<campaign>/raw/*.json` | 板上探针 `scripts/*_probe.py`（`board_*.sh` 驱动） | `build_*.py` | **否（提交 1）** |
| `<campaign>/dataset.csv`、`dataset_2d.csv` | `build_dtype_dataset.py` / `build_2d_dataset.py` | `build_unified_dataset.py`、各 `fit_*.py` | **否（提交 1）** |
| `model_store_v1/dataset.csv` | `build_store_dataset.py` | `fit_strided_store_v1.py`（**已删除**）（-> `hist_store` 进 unified） | **否（提交 1）** |
| `model_template_stride_v2/dataset.csv` | `build_template_stride_dataset.py` | `fit_template_stride_v2.py`（**已删除**；**不进最终 10 式**，被 `--exclude-template` 排除） | **否（提交 1）** |
| `_build/build_c{1,2}/dataset_2d.csv` | `run_final_pipeline.py` -> `build_2d_dataset.py` | `build_unified_dataset.py` | **否（提交 1）** |
| `final/dataset_unified_v5.csv` | `build_unified_dataset.py` | `fit_unified6.py`（**已删除**）、跨口径对照 | **是** |
| `final/dataset_new_only_v5.csv` | `run_final_pipeline.py`（从 unified_v5 过滤 source） | **全部物理拟合脚本**（`phys2b_lib` / `phys3_write3.py` / `write_phys2_store.py` / `p3_verify.py` …，**这一族均已删除**，见 §5 脚注） | **是** |
| `final/MODEL_*.json` | `fit_v9..v21`、`fit_final_v4/v5/v6/v7`、`fit_10crossed*`、`fit_regime*`、`fit_max*`、`fit_modsel`、`fit_prune`、`run_final_pipeline.py`（**除 `fit_v9.py` / `fit_v9_pool.py` / `fit_v11_stepwise.py` / `run_final_pipeline.py` 外均已删除**） | `gen_reports.py`（**已删除**） / `gen_report_phys.py` / `gen_report_v5.py` | **是** |
| `final/PHYS3_*.json` / `final/PHYS2_*.json` | `_p3_w1_write.py` / `_p3_g1_write.py` / `p3_store_r1_write.py` / `phys2_final_report.py` / `p2_r2c_write.py`（**均已删除**，见 §5 脚注） | `gen_report_v5.py`（按 PHYS3 > PHYS2 > PHYS 合并） | **是** |
| `final/REPORT_v*.md` | `gen_reports.py`（**已删除**） / `gen_report_phys.py` / `gen_report_v5.py` / `gen_report_concise.py` | `_readme4.py` -> `pre-explore/README.md` + `README_data.json`（`_readme.py`/`_readme2.py`/`_readme3.py` **已删除**） | **是** |

---

## 10. 想复现全流程怎么办

| 你的目标 | 用哪个提交 | 起步命令 |
|---|---|---|
| 从板上 raw JSON 重跑特征化 + 拟合 + 报告 | **提交 1 `5539c4831`** | `git checkout 5539c4831` 后照 [`../../README.md`](../../README.md) 第二~六节做 |
| 只想用最终数据集重拟 / 校验公式 | **提交 2（本提交）** | 直接用 `final/dataset_new_only_v5.csv`（valid=1 的 9541 行）+ [`../scripts/fit_v9.py`](../scripts/fit_v9.py) 的 `load()` / `mape()` |
| 只想看结论 | 任意提交 | [`../README.md`](../README.md)（10 式总表）+ [`final/REPORT_v5_physics_scope_concise_des.md`](final/REPORT_v5_physics_scope_concise_des.md) |
| 想知道为什么不用贪心 | 任意提交 | [`../docs/WHY-PHYSICS-OVER-GREEDY.md`](../docs/WHY-PHYSICS-OVER-GREEDY.md) |

取回提交 1 的某个目录示例：

```bash
# 只取回 final/raw/ 的 231 个原始 JSON（29 MB）
git checkout 5539c4831 -- third_party/ascend/costmodel/profiles/microbench/data_provider/
                              strided_ldst/pre-explore/results/final/raw/
# 或者不落到工作区，直接看：
git show 5539c4831:third_party/.../strided_ldst/pre-explore/results/final/raw/v5__load_simd_float32_W1_fix1.json
```


