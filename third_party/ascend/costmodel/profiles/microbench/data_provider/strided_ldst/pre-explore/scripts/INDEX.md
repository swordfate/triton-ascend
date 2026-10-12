# pre-explore/scripts/ 脚本索引（现存 106 个）

> **本文件是重写版。** 旧版（原 491 行）逐个列出了清理前的 **622 个脚本**；其中 **518 个已被真删**
> （不是移动），旧索引里绝大多数条目因此指向不存在的文件。本版由**实际扫描目录**后重新生成，
> 只覆盖现存的 **106 个脚本**。
>
> 扫描方式（可复现）：对 `pre-explore/scripts/` 递归列举 `*.py` / `*.sh`，
> 得到 **107 个文件 = 106 个脚本（52 个 `.py` + 54 个 `.sh`）+ 本 `INDEX.md`**。

---

## 0. 这份索引覆盖什么，不覆盖什么

### 0.1 现状（实测）

| 位置 | `.py` | `.sh` | 小计 |
|---|---:|---:|---:|
| `scripts/` 顶层 | 49 | 47 | 96 |
| `scripts/camodel/` | 3 | 7 | 10 |
| **合计** | **52** | **54** | **106** |

清理前是 **622 个脚本**（513 `.py` + 109 `.sh`）。本次删除 **518 个**（461 `.py` + 57 `.sh`，约 1.6 MB），
另从 git 恢复 **2 个**（`board_env.sh`、`board_env_latest.sh`，见 §0.2）。

> **算式**：622 − 518 + 2 = **106**。

### 0.2 被删掉的 518 个脚本原本是什么

它们**不是**一次性垃圾，而是**子代理逐步推导物理模型时的中间产物**，按血缘分：

| 族 | 数量 | 原本干什么 | 为什么能删 |
|---|---:|---|---|
| `_p2_*` / `p2_*` | 154 | 第二轮物理拟合的逐步推进（`p2_step1..29`、`_p2_step1..28`、`_p2_final1..5`、`p2_lib.py`、`_p2_lib.py`）；负责 `simt_store_r1/r2`、`simd_store_*` 槽 | 结论已固化进 `results/final/PHYS2_*.json` 与 `PHYS3_*.json` |
| `_p3_*` / `p3_*` | 75 | 第三轮（MTE 粒度重建、128B-free）的 9 个 stage：`_p3_step1..13`、`_p3_search{,2,3}`、`_p3_hypo{,2}`、`_p3_polish`、`_p3_verify`、`_p3_write`、`_p3_w1_*`、`p3_feat.py`、`p3_step1..29` | 与 `docs/agent-logs/p3_*.txt` 的 11 份日志一一对应 |
| 一次性 `_*.py` | 66 | `_fit1..14`、`_probe1..13`、`_exh*`、`_chk1..4`、`_addr`、`_occ`、`_warp`、`_geom`、`_loglog`、`_noise`、`_raw*`、`_clean3`、`_fixrate`、`_ren`、`_w1*` 等单点诊断/草稿 | 没有下游依赖，结论已写进文档与 JSON |
| `phys_*` | 53 | 纯物理优先（physics-first）的 simt load 建模链：`phys_simt_step2..9`、`phys_simt_load.py`、`phys_simt_writejson.py`、`phys_r2_step1..17`、`phys2b_lib/feat`、`phys2d_lib` | 同上 |
| `phys3_*` | 26 | 128B-free 的 `simd_store_r2` 建模链（`phys3_lib.py`、`phys3_s2_step1..6`） | 同上 |
| 实验性 `fit_*` | 33 | `fit_final_v4/v5/v6/v7`、`fit_maxlinear`、`fit_maxplus*`、`fit_regime`、`fit_modsel`、`fit_prune`、`fit_10crossed*`、`fit_domain*`、`fit_exh5`、`fit_back10`、`fit_why`、`fit_unified6`、`fit_2d_model`、`fit_dtype_model` 等结构/搜索变体 | 结果留在 `results/final/MODEL_*.json`（45 个模型文件）|
| 一次性诊断 `.sh` | 57 | `board_launch_*`（17 个 tmux 启动器）、`board_progress/status/eta/watchdog/asm 巡检/一次性修补` 等运维脚本 | 生命周期已结束；启动方式可直接用 `nohup` / `tmux` 一行替代（见 §8.7）|

**推导记录仍然保留**，两处：

1. `pre-explore/docs/agent-logs/` —— 11 份原始推导日志：
   `p3_explore2.txt`、`p3_final1.txt`、`p3_final2.txt`、`p3_hypo.txt`、`p3_hypo2.txt`、
   `p3_polish.txt`、`p3_search.txt`、`p3_search2.txt`、`p3_search3.txt`、`p3_verify.txt`、`p3_write.txt`。
2. **git 历史**：清理尚未提交，`git ls-tree -r HEAD -- pre-explore/scripts` 仍是 **623 个文件**。
   任何被删脚本都能取回：
   `git show HEAD:<路径>` 查看，`git checkout HEAD -- <路径>` 恢复。
   例：`git checkout HEAD -- pre-explore/scripts/p2_lib.py`。

> **本次已从 git 恢复 2 个文件**：`scripts/board_env.sh` 与 `scripts/board_env_latest.sh`。
> 它们不是探索产物，而是**所有板端调用的环境包装器**（`source ~/env_ascend.sh` + `conda activate wj_autoscope`）；
> 依赖闭包只沿 `import` 递归，漏掉了这种「被 `bash` 调用」的依赖。若不恢复，**现存 42 个板端 `.sh` 全部跑不起来**。

### 0.3 标记约定

| 标记 | 含义 |
|---|---|
| **★必需** | 复现 `results/final/` 的最终数据集 / 公式 / 报告所必需，删了链路就断 |
| **○可复用** | 这一轮已经跑完，但换参数、补测、重跑该支线还要用它 |
| **△历史** | 已被后续脚本取代，保留只为可追溯 |

板上测量环境（所有板端脚本共用）：

* `ssh ascend-950pr-63`，用户 `c00946898`，conda 环境 `wj_autoscope`，**256 核**
* 板端工作目录 `~/strided-template-calibration-v2/`（脚本里简称 `$PRE`）
* 板端调用链固定为 `bash scripts/board_env.sh python scripts/<probe>.py <args>`：
  探针在板上生成 Triton kernel、计时、判正确性，然后把结果写成 `results/<campaign>/raw/*.json`
* 仓库里的 `scripts/` 会整体同步到板上同名目录，所以脚本里的相对路径都是相对 `pre-explore/` 的
* 现存 `.sh` 中有 **40 个**用 `board_env.sh`，**2 个**（store 探针）用 `board_env_latest.sh`

---

## 1. 总览

| # | 类别 | 数量 | 代表脚本 | 一句话作用 |
|---|---|---:|---|---|
| 1 | 探针 | 15 | `dtype_stride_measure_probe.py`、`stride2d_measure_probe.py` | 板上生成 kernel、计时、校验、写 raw JSON |
| 2 | 路径检查 | 4 | `dtype_stride_path_check.py`、`stride2d_path_check.py` | 只编译不执行，确认走了哪条 lowering |
| 3 | 数据集构建 | 8 | `build_dtype_dataset.py`、`build_unified_dataset.py` | raw JSON → 拟合用 CSV |
| 4 | 拟合与库 | 14 | `fit_v9.py`、`fit_v11_stepwise.py`、`run_final_pipeline.py` | 特征池 + 加权 NNLS + 端到端流水线 |
| 5 | 报告生成 | 6 | `gen_report_v5.py`、`_readme4.py` | 公式 JSON → markdown 报告 / 顶层 README |
| 6 | 板上驱动 | 47 | `run_final_campaign.sh`、`run_dtype_calibration.sh` | 循环调用探针跑完一整片测量网格 |
| 7 | CAModel | 12 | `camodel/analyze_strided_camodel.py` | 解析 simulator dump，导出白盒事实 |
| | **合计** | **106** | | |

---

## 2. 完整复现调用链

### 2.1 板上：驱动 `.sh` → 探针 → raw JSON

| 阶段 | 板端命令 | 调用的脚本 | 结果落哪 |
|---|---|---|---|
| 环境包装（所有板端调用都要套） | `bash scripts/board_env.sh python <probe> ...` | `board_env.sh`（store 用 `board_env_latest.sh`）| —（只做 source + conda activate + ulimit）|
| 1D 探针（单次测量） | `bash scripts/board_env.sh python scripts/dtype_stride_measure_probe.py --op load\|store --path simd\|simt_only\|template --dtype ... --blocks ... --strides ... --num-warps ... --out ...` | `dtype_stride_measure_probe.py` | `results/<campaign>/raw/*.json`（顶层 `{kind,op,path,dtype,element_bytes,cases:[...],buffer,frequency_check}`）|
| 2D 探针（单次测量） | `bash scripts/board_env.sh python scripts/stride2d_measure_probe.py --shapes 4x16 --row-strides ... --inner-strides ...` | `stride2d_measure_probe.py` | 同上，`kind=stride2d_measure` |
| 编译期路径证据 | `PASS=path` / `PASS=2dpath` / 模板 path_check | `dtype_stride_path_check.py`、`stride2d_path_check.py`、`template_stride_path_check.py` | `<campaign>/measure_asm/*.ttir` + path_check JSON |
| 批量测量（一个 PASS 一片网格） | `PASS=core1d bash scripts/run_final_campaign.sh`<br>`PASS=simd bash scripts/run_dtype_calibration.sh` | 见 §8 的 47 个驱动 | `results/<campaign>/raw/*.json` + `measure_asm/` |
| 脱离终端长跑 | `setsid nohup bash -c "PASS=<p> bash scripts/run_final_campaign.sh > runs/<p>.log 2>&1" &` | 无（原来由已删的 `board_launch_*.sh` 做）| 日志 `runs/<p>.log` |

### 2.2 离线：raw JSON → dataset CSV → 公式 → 报告

| # | 命令（在 `pre-explore/` 下执行）| 调用的脚本 | 产物 |
|---|---|---|---|
| 1 | `python scripts/build_dtype_dataset.py --raw-dir results/dtype_calibration_v1/raw --out-dir results/dtype_calibration_v1` | `build_dtype_dataset.py` | `dataset.csv`（2560 行 × 51 列）+ `dataset_meta.json` |
| 2 | `python scripts/build_dtype_dataset.py --raw-dir results/dtype_calibration_v2_rotate512/raw --out-dir results/dtype_calibration_v2_rotate512` | 同上 | 第二轮 1D 数据集（`source=v2_1d`）|
| 3 | `python scripts/build_2d_dataset.py --raw-dir results/expand2d/raw --out-dir results/_build/build_c1` | `build_2d_dataset.py` | `dataset_2d.csv`（3432 行 × 48 列）|
| 4 | `python scripts/build_2d_dataset.py --raw-dir results/expand2d_b/raw --out-dir results/_build/build_c2` | 同上 | 第二个 2D 数据集（`source=v4_2d`）|
| 5 | `python scripts/build_store_dataset.py` | `build_store_dataset.py` | `results/model_store_v1/dataset.csv`（`hist_store`，544 行）|
| 6 | `python scripts/build_unified_dataset.py --out results/final/dataset_unified_v5.csv --exclude-template --v1-1d results/final/dataset.csv --v1-2d results/final/dataset_2d.csv --v3-2d results/_build/build_c1/dataset_2d.csv --v4-2d results/_build/build_c2/dataset_2d.csv` | `build_unified_dataset.py`（内部用 `stride_feature_reference.py`）| `final/dataset_unified_v5.csv`（8016 行 × 43 列）|
| 7 | `python scripts/run_final_pipeline.py --build` | `run_final_pipeline.py` → 子进程调 `build_2d_dataset.py` / `build_unified_dataset.py` | 重建 2D + unified，并派生 `final/dataset_new_only_v5.csv` |
| 8 | `python scripts/run_final_pipeline.py` | `run_final_pipeline.py` + `fit_v9.py` + `fit_v9_pool.py` + `fit_v11_stepwise.py` | `final/MODEL_FINAL_V7.json`（10 式，每式 ≤10 项）|
| 9 | `python scripts/check_acceptance.py` | `check_acceptance.py` | PASS/FAIL 验收清单 |
| 10 | `python scripts/_eval.py` | `_eval.py` | `_eval.json`（逐点复算 10 式并与 JSON 记录的 mape 对拍）|
| 11 | `python scripts/_tbl.py` | `_tbl.py` | `_readme_tbl.json`（n / est_us / meas_us / mape / PASS 总表）|
| 12 | `Copy-Item _readme_tbl.json _tbl.json` | —（**已知坑**：两个文件名不一致）| `_tbl.json` |
| 13 | `python scripts/_readme4.py` | `_readme4.py` + `README_data.json` | `pre-explore/README.md` |
| 14 | `python scripts/gen_report_concise.py` | `gen_report_concise.py` | `results/final/REPORT_v5_physics_scope_concise_des.md` |
| 15 | `python scripts/gen_report_v5.py` | `gen_report_v5.py` | `results/final/REPORT_v5_physics_scope.md`（按 `PHYS3 > PHYS2 > PHYS` 合并）|

### 2.3 一行速记

```
[板上] run_*.sh ──► board_env.sh + <probe>.py ──► results/<campaign>/raw/*.json
                                                   results/<campaign>/measure_asm/*.ttir
[离线] build_dtype_dataset.py / build_2d_dataset.py  ──► <campaign>/dataset*.csv
       build_store_dataset.py                        ──► model_store_v1/dataset.csv
       build_template_stride_dataset.py              ──► model_template_stride_v2/dataset.csv
       build_unified_dataset.py                      ──► final/dataset_unified_v5.csv
       run_final_pipeline.py --build                 ──► final/dataset_new_only_v5.csv
       run_final_pipeline.py                         ──► final/MODEL_FINAL_V7.json
       _eval.py ─► _tbl.py ─► _readme4.py            ──► pre-explore/README.md
       gen_report_v5.py / gen_report_concise.py      ──► final/REPORT_v5_*.md
```

### 2.4 ⚠ 链路上**已经断开**的一环

`results/final/` 的 **16 个 `PHYS*.json`**（`PHYS_*` 第 1 轮 / `PHYS2_*` 第 2 轮 / `PHYS3_*` 第 3 轮，即最终 10 条公式的真实载体）
的**写回脚本已删除**：`_p3_w1_write.py`、`_p3_g1_write.py`、`p3_store_r1_write.py`、`phys3_write3.py`、
`write_phys2_store.py`、`phys2_final_report.py`、`p2_r2c_write.py`、`p2_writejson.py`、`write_phys.py`、`_ren.py` 等。

后果与现状：

* `gen_report_v5.py` / `gen_report_concise.py` / `_tbl.py` / `gen_report_phys.py` **仍然能读**这些 JSON 出报告 → 报告链是通的；
* 但**无法用现存脚本重新拟合出 `PHYS*_*.json`** → 它们是**冻结交付物**；
* 若要重跑物理拟合轮次，需从 git 取回对应写回脚本（见 §0.2 的 `git checkout HEAD --`），或改用 `run_final_pipeline.py` 的贪心路线（产出 `MODEL_FINAL_V7.json`，不是 PHYS 族）。

同样地，`fit_unified6.py`、`fit_dtype_model.py`、`fit_2d_model.py`、`summarize_calibration.py`、`make_v7_plots.py`、`refit_no_stride1.py`
等脚本已删除，`pre-explore/results/README.md` 里指向它们的历史说明见该文件内的「已删除脚本」标注。

---

## 3. 探针（15）

探针是**唯一直接跟板子打交道的一层**：板端 kernel 生成、计时、正确性校验、写 JSON。

| 脚本 | 行数 | 作用 | 谁调用它 | 必需？ |
|---|---:|---|---|---|
| `dtype_stride_measure_probe.py` | 406 | **1D（rank-1）主探针**：一次 rotate-loop 迭代做一次 shaped strided load/store，取 Event 斜率（ns/迭代）。参数 `--op` / `--path` / `--dtype` / `--blocks` / `--strides` / `--num-warps` / `--rotate-step` / `--rotate-mask` | 11 个现存 `.sh` | **★必需**（`dataset.csv` 的唯一来源）|
| `stride2d_measure_probe.py` | 391 | **2D（rank-2）主探针**：tile 地址集 `base + i*rowpitch + j*ISB`，`OS = rs*C*cs` | 13 个现存 `.sh` | **★必需**（`dataset_2d.csv` 的唯一来源）|
| `store_measure_probe.py` | 284 | store 目标探针：短 kernel Event 斜率 + boost 见证（长 kernel 会把 AICore 推出 boost） | `run_store_measure_simd.sh` / `_simt.sh` | **★必需**（`hist_store` → unified）|
| `template_stride_load_probe.py` | 310 | `triton_stride_load` 模板路径（1D rank1），rotate-loop Event 斜率 | `run_template_stride_measure_load.sh` | **★必需**（模板数据集）|
| `template_stride_store_probe.py` | 301 | `triton_stride_store` 模板路径（1D rank1） | `run_template_stride_measure_store.sh` | **★必需**（模板数据集）|
| `frequency_retry_probe.py` | 182 | v2 频率感知重试探针（SIMD，SYS_CNT）：查 Aicore 频率、throttle 时预热、异常时重试 | `run_v2_simd_dense.sh`、`run_v2_more_blocks.sh` | ○可复用 |
| `simt_frequency_retry_probe.py` | 173 | 同上，SIMT 分支（Event 斜率） | `run_v2_simt_dense.sh`、`run_simt_numwarps_{sweep,dense}.sh` | ○可复用 |
| `simt_frequency_retry_probe_aligned.py` | 173 | **与上一行只差一行 import**（`syscnt_event_compare` → `_aligned`）；服务页对齐的 v4 测量 | `run_v4_aligned_23.sh` | ○可复用（有意冻结的副本）|
| `simd_one_load_syscnt.py` | 130 | SIMD 单次 strided load 的 Triton inline `SYS_CNT` 计时原语 | 被 `frequency_retry_probe.py` import | ○可复用（库）|
| `syscnt_event_compare.py` | 231 | 交叉验证 `MOV $0, SYS_CNT` 与 `torch.npu.Event` 的计时一致性；提供 `loop_noasm` / `run_case` / `SYS_CNT_MHZ` | 被 `simt_frequency_retry_probe.py` import | ○可复用（库）|
| `syscnt_event_compare_aligned.py` | 231 | 同上（与上一份 **sha256 相同**），供页对齐支线 | 被 `simt_frequency_retry_probe_aligned.py` / `retest_strided_simt_cases.py` import | ○可复用（库）|
| `retest_strided_simt_cases.py` | 123 | 用**独立 buffer 分配**重测指定 SIMT case（5 次分配同持，滤掉瞬时离群） | `run_v4_retest_missing.sh` | ○可复用 |
| `precompile_store_probe.py` | 78 | 并行预热 store 探针的 kernel 目录，避免首次编译计入计时 | **无 `.sh` 调用**（手工执行）| ○可复用 |
| `board_time_probe.py` | 226 | 早期直接在板上测 strided load（`strided_load_probe.py` 的真卡对照） | 无 | △历史 |
| `board_event_probe.py` | 154 | Event 计时与探针的交叉验证 | 无 | △历史 |

---

## 4. 路径检查（4）

**只编译、不执行**，回答「这段 strided 访问实际走了哪条 lowering」——是模型分档（wide / gather / template / bail）的证据。

| 脚本 | 行数 | 作用 | 谁调用它 | 必需？ |
|---|---:|---|---|---|
| `dtype_stride_path_check.py` | 257 | dtype × 静态 stride × lowering mode 的编译期路径检查（`--dtypes` / `--strides` / `--modes` / `--asm-limit`）| `run_dtype_calibration.sh PASS=path` / `pathlite` | **★必需** |
| `stride2d_path_check.py` | 221 | rank-2 tile 的编译期路径检查（按 `--shapes` 遍历 form）| `run_dtype_calibration.sh PASS=2dpath` | **★必需** |
| `template_stride_path_check.py` | 94 | 模板 load/store dispatch 的编译期边界检查（源码依据 `StridedLoadStoreRewrite.cpp`）| `run_template_stride_calibration_v2.sh` | **★必需**（模板支线）|
| `mixed_route_stride_probe.py` | 122 | 编一个**真 costmodel 控制**的 strided kernel（`compile_mode="simd_simt"`、`auto_simt_scope_mode="auto"`），检查它实际走哪条 route；不需要 CAModel | 手工 | ○可复用 |

---

## 5. 数据集构建（8）

统一套路：读 `results/<campaign>/raw/*.json` → 对**所有数值正确的 pass 取中位数** → 每行带
`target_min_ns` / `target_max_ns` / `target_spread` / `quality` → 写 CSV。spread 高只是**质量标记**，不是丢点的理由。

| 脚本 | 行数 | 产出 | 必需？ |
|---|---:|---|---|
| `build_dtype_dataset.py` | 252 | `<campaign>/dataset.csv`（rank-1，51 列）→ `final/dataset.csv`（2560 行，valid 2371）| **★必需** |
| `build_2d_dataset.py` | 224 | `<campaign>/dataset_2d.csv`（rank-2，48 列）→ `final/dataset_2d.csv`（3432 行）| **★必需** |
| `build_unified_dataset.py` | 288 | `final/dataset_unified_v5.csv`（8016 行 × 43 列）；按 `source` 列拼接 v1_1d / v1_2d / v2_1d / v3_2d / v4_2d / hist_* | **★必需** |
| `build_store_dataset.py` | 201 | `results/model_store_v1/dataset.csv`（`hist_store`，544 行）| **★必需**（进 unified）|
| `build_template_stride_dataset.py` | 99 | `model_template_stride_v2/dataset.csv`（模板支线）| ○可复用（被 `--exclude-template` 排除在最终 10 式之外）|
| `build_v4_numwarps_dataset.py` | 291 | `model_v4_numwarps/dataset.csv`（`hist_v4`，992 行；页对齐测量）| ○可复用（被 `run_v4_followup.sh` 调用）|
| `build_v2_dataset.py` | 164 | `model_v2/dataset.csv`（`hist_v2_single_shot`，441 行）| △历史 |
| `build_v3_numwarps_dataset.py` | 163 | `model_v3_numwarps/dataset.csv`（SIMT 粗扫 `num_warps`）| △历史 |

> 端到端建集入口是 `run_final_pipeline.py --build`（见 §6），它内部以子进程调 `build_2d_dataset.py` 与 `build_unified_dataset.py`。

---

## 6. 拟合与库（14）

共同点：`import fit_v9` 拿 `load` / `mape` / `nn`，用 `fit_v9_pool.POOL` 里的物理特征名做前向 / 后向选择，
**加权最小二乘（权重 `1/T`，等价于最小化相对误差 = MAPE）**，只允许线性形式与 `max()`，**禁止 log / 幂 / 指数**。

### 6.1 端到端入口

| 脚本 | 行数 | 作用 | 必需？ |
|---|---:|---|---|
| `run_final_pipeline.py` | 197 | **唯一把 raw → dataset → new-only → 拟合串起来的脚本**。`--build` 建集，无参运行时逐 slot 贪心选 ≤10 项（线性或 `max(a*W, 线性)`），产出 `final/MODEL_FINAL_V7.json` | **★必需** |

### 6.2 拟合骨架与特征库

| 脚本 | 行数 | 作用 | 被谁 import | 必需？ |
|---|---:|---|---|---|
| `fit_v9.py` | 124 | **字节域探索骨架**：`load()` 读 CSV、`mape()` 指标、`nn()` = Lawson-Hanson NNLS（纯 numpy，无 scipy）。模型可以是「纯线性」或「`max(a*W, 线性)`」，系数非负 | `fit_v10_exhaustive`、`fit_v11_stepwise`、`_eval`、`_tbl`、`gen_report_concise`、`run_final_pipeline` | **★必需** |
| `fit_v9_pool.py` | 81 | 物理特征池 `POOL`（特征名 → 计算函数）：`L`（128B 行数）、`pages`、`span`、`eb`、`ebL`、`perlane`、`rowgap`、`cmd`、`lpe`、`pagecross`、`hL32/hL128/hL512/hL2048/hL8192`、`hP1/hP4/hP16/hP64` 等 | 同上 | **★必需** |
| `fit_v10_exhaustive.py` | 141 | 两阶段搜索：先按单特征排名取前 N，再枚举 1..4 项子集（截距恒在） | 被 `fit_v11_stepwise.py` import | **★必需**（库）|
| `fit_v11_stepwise.py` | 388 | 前向-后向逐步选择（特征可以被换出），非负相对加权最小二乘，最多 10 项 | `run_final_pipeline.py` | **★必需** |
| `stride_feature_reference.py` | 489 | **所有 strided 特征的精确整数语义参考实现**——特征定义的唯一真相，C++ 移植照它写。`--verify` 自检 | `build_dtype_dataset`、`build_2d_dataset`、`build_unified_dataset`、`camodel/analyze_stride2d_camodel.py` | **★必需** |
| `template_stride_features.py` | 139 | 模板路径的半白盒特征（直接由 NPUIR 模板源码推导） | `build_template_stride_dataset.py` | ○可复用 |
| `template_stride_common.py` | 171 | **三个模板探针 + 模板 path_check 的公共工具**：Event 计时、斜率拟合、四道物理门、asm 落盘 | `dtype_stride_measure_probe`、`stride2d_measure_probe`、`dtype_stride_path_check`、`stride2d_path_check`、三个 `template_*` | **★必需** |

### 6.3 预测器与支线拟合

| 脚本 | 行数 | 作用 | 必需？ |
|---|---:|---|---|
| `predict_strided_load_v7_simple.py` | 276 | v7-simple 的**参考实现 + 自检**：把公式系数硬编码，在数据集上重算 cost 与实测对拍 | ○可复用（被 `evaluate_strided_load_v7_on_v4.py` 复用）|
| `fit_strided_model_v8.py` | 733 | v8 半白盒 SIMT strided-load 模型的拟合（v4 数据集） | ○可复用（被 `run_v4_followup.sh` 调用）|
| `evaluate_strided_load_v7_on_v4.py` | 83 | 在 v4 数据集上评估既有 v7 模型，输出分组误差表到 `model_v4_numwarps/v7_eval/` | ○可复用 |

### 6.4 校验与工具

| 脚本 | 行数 | 作用 | 必需？ |
|---|---:|---|---|
| `check_acceptance.py` | 166 | 本轮标定的**验收自检**：逐公式 MAPE、覆盖率、阈值，打印 PASS/FAIL 清单；`--with-refit` 可重拟合 | **★必需** |
| `merge_v4_retest.py` | 45 | 合并主测与缺 stride 补测的 JSON（key 必须不相交，输出按 (W, block, stride) 排序） | ○可复用（被 `run_v4_followup.sh` 调用）|
| `board_integrity.py` | 21 | 板端完整性巡检：统计某 campaign `raw/*.json` 的完成数 / case 数 / valid 数 / spread 中位数与 p90（目录写死为 `v10_expand2d`，换目录要改第 4 行）| ○可复用 |

---

## 7. 报告生成（6）

| 脚本 | 行数 | 作用 | 产物 | 必需？ |
|---|---:|---|---|---|
| `_eval.py` | 134 | 逐点复算 10 个公式（含 2D 几何重算），与 `PHYS3_*.json` 里记录的 `mape` 对拍；不一致会打印 `*** MISMATCH ***` | `_eval.json` | **★必需** |
| `_tbl.py` | 62 | 汇总总表（n / est_us / meas_us / mape / PASS），并内含**适用范围门槛 `SCOPE`**（最终 5118 个点的来源）| `_readme_tbl.json` | **★必需** |
| `_readme4.py` | 95 | **当前 `pre-explore/README.md` 的生成器**：读 `_tbl.json` + `README_data.json` 拼 markdown | `pre-explore/README.md` | **★必需** |
| `gen_report_v5.py` | 196 | **当前物理报告生成器**：汇总所有 `PHYS*_*.json`（按 `PHYS3 > PHYS2 > PHYS` 合并 + `COMPOSITION.json`）| `results/final/REPORT_v5_physics_scope.md` | **★必需** |
| `gen_report_concise.py` | 213 | 中文精简版：只留最终保留集，不讲剔除过程 | `results/final/REPORT_v5_physics_scope_concise_des.md` | **★必需** |
| `gen_report_phys.py` | 147 | 上一版：把所有 `PHYS_*.json` 汇总成 v4 物理报告 | `results/final/REPORT_v4_physics.md` | ○可复用 |

> **报告链的已知坑**：`_tbl.py` 写出的文件名是 `_readme_tbl.json`，而 `_readme4.py` 读的是 `_tbl.json`，
> 中间必须手工 `Copy-Item` / `cp` 一次，否则报告链在中间断掉。

---

## 8. 板上驱动（47 个 `.sh`）

所有驱动的共同形态：定义 `m1d()` / `m2d()` 这类局部函数，函数体固定是
「设 `REPS` / `MAX_ATTEMPTS` → `board_env.sh python <probe>` → 输出文件名由 block/stride 签名唯一化」。
**判读这些脚本的关键是最后一个参数 `--out` / `$OUT`，它决定结果落在哪个 campaign 目录。**

### 8.1 环境包装（2）

| 脚本 | 作用 | 必需？ |
|---|---|---|
| `board_env.sh` | `source ~/env_ascend.sh` + `source conda.sh` + `conda activate wj_autoscope` + `ulimit -n 1048576` + `exec "$@"`；**40 个现存 `.sh` 的板端入口** | **★必需** |
| `board_env_latest.sh` | 同上，但把 `$HOME/sb64_latest_env` 前置到 `PYTHONPATH`（store 探针用）| **★必需** |

### 8.2 主线测量驱动（6）

| 脚本 | 行数 | 写到的 campaign | 作用 | 必需？ |
|---|---:|---|---|---|
| `run_dtype_calibration.sh` | 219 | `dtype_calibration_v1` | **Phase A/B 主驱动**，按 `PASS=` 派发：`path` / `pathlite` / `simd` / `simt` / `template` / `2dpath` / `2d` / `all_a` / `all_b`；网格是**字节间隙交叉设计**（同一 byte gap 由多个 dtype 实现）| **★必需** |
| `run_final_campaign.sh` | 88 | `dtype_calibration_v4_final` | FINAL 轮，5 个 PASS：`core1d`(2048) / `wsweep1d`(1152) / `core2d`(6912) / `wsweep2d`(1296) / `n1`(1008) | **★必需** |
| `run_final_delta.sh` | 236 | `dtype_calibration_v4_final` | FINAL 的 DELTA：只补 `v4_final` 里还没有的格点 | ○可复用 |
| `run_final2.sh` | 58 | `dtype_calibration_v5_final` | FINAL-v2（缩减 2D：core2d 180→120 form），并串起 `run_core1d_remainder.sh` | ○可复用 |
| `run_core1d_remainder.sh` | 49 | `dtype_calibration_v4_final` | core1d 补跑；**有文件名冲突 bug，会丢结果（23 个里丢 9 个）** | △历史（勿用）|
| `run_core1d_remainder2.sh` | 52 | `dtype_calibration_v4_final` | 同上，但输出名加 block/stride 签名，修掉丢结果问题 | ○可复用 |

### 8.3 补点 / 补洞 / 重测批次（11）

| 脚本 | 行数 | 写到的 campaign | 作用 | 必需？ |
|---|---:|---|---|---|
| `run_rotate512_campaign.sh` | 112 | `dtype_calibration_v2_rotate512` | **Phase R**：用历史 512 MB rotation window（`step=8192 mask=16383`）重跑同一探针，使新旧数据集可逐格对比（`PASS=rot` / `rot2d` / `stepiso`）| ○可复用 |
| `run_balance_campaign.sh` | 88 | 同上 | Phase F 补齐 1：均衡各元素宽度的点数（历史数据 f32-only，int8/f16/int64 每槽只有 2-8 点）| ○可复用 |
| `run_simdfill_campaign.sh` | 61 | 同上 | Phase F 补齐 2：SIMD load 是最稀疏的槽（历史 SIMD 数据是 single-shot）| ○可复用 |
| `run_gapfill_campaign.sh` | 58 | 同上 | Phase F 补齐 3：补掉 SIMT f16=0 点、SIMT int8/int64 极少等覆盖洞 | ○可复用 |
| `run_gapfill_final.sh` | 67 | `dtype_calibration_v8_gapfill` | 最后一轮补洞，使每个 (slot, dtype, W) 达到目标网格（缺口是精确矩形，只测缺的）| ○可复用 |
| `run_delta_campaign.sh` | 56 | `dtype_calibration_v3_delta` | 只测尚未测过的格点 | △历史 |
| `run_d2_campaign.sh` | 58 | `dtype_calibration_v3_delta` | (row_stride, col_stride) 参数化的 2D 批次：`PASS=d2main` = 9 shape × 6 rs × 8 cs = 432 case/(op,path,dtype) → 6912；`PASS=d2n1` = 18 → 288 | △历史 |
| `run_fullgrid_campaign.sh` | 53 | `dtype_calibration_v3_fullgrid` | 1D 全网格：8 elements × 23 strides = 184 cell/(slot,dtype) × 4 slot × 4 dtype = **2944 cell**（`PASS=simd` / `simt`）| △历史 |
| `run_int8fix.sh` | 42 | `dtype_calibration_v6_int8fix` | 修 2D int8 load 溢出（旧探针把 load 结果写进 element dtype，tile ≥43 元素就溢出，448 个 case 被误判 wrong）后重测该切片 | ○可复用 |
| `run_resample.sh` | 99 | `dtype_calibration_v7_resample` | 重测 spread > 0.40 的噪声点（79% 重复测量一致、12% 稳定噪声、9% 瞬时）| ○可复用 |
| `run_wextend.sh` | 57 | `dtype_calibration_v9_wsweep_full` | **W 扩展**：给选定 `num_warps` 与 W=4 同等的全网格（1D 8×10=80、2D 6×4×5=120 per (op,dtype)）；`W` 为环境变量，`W=2` 复现已删的 `run_wextend_w2.sh` | ○可复用 |

### 8.4 v2 / v4 频率感知与页对齐支线（8）

| 脚本 | 行数 | 写到的 campaign | 作用 | 必需？ |
|---|---:|---|---|---|
| `run_v2_simd_dense.sh` | 34 | `board_v2_simd` | v2 协议：频率感知的密集 SIMD 单 load | △历史 |
| `run_v2_simt_dense.sh` | 29 | `board_v2_simt` | v2 协议：SIMT Event 斜率 | △历史 |
| `run_v2_more_blocks.sh` | 47 | `board_v2_simd_more_blocks` / `board_v2_simt_more_blocks` | v2 补中间 / 大 block（原黑盒面只有 5 个 block 尺寸）| △历史 |
| `run_simt_numwarps_sweep.sh` | 34 | `board_v2_simt_warps` | SIMT `num_warps` 粗扫（224 个粗 case）| △历史 |
| `run_simt_numwarps_dense.sh` | 35 | 同上 | 关键 `num_warps`（32 / 64）密集测 | △历史 |
| `run_v4_aligned_23.sh` | 27 | `v4_aligned_23`（+ `v4_aligned_dev1`）| 物理 NPU1（`ASCEND_RT_VISIBLE_DEVICES=1`）上补 6 个大 stride × W=1/2/4/8/16 | △历史 |
| `run_v4_retest_missing.sh` | 15 | `v4_retest_23_missing.json` | 240 个缺失大 stride case 的 **5 次独立分配**重测 | ○可复用 |
| `run_v4_followup.sh` | 49 | `v4_retest_all.json` / `v4_retest_full_23.json` | 等 `v4_retest_all` 结束 → 补跑 240 case → `merge_v4_retest.py` → `build_v4_numwarps_dataset.py` → `evaluate_strided_load_v7_on_v4.py` → `fit_strided_model_v8.py` | ○可复用 |

### 8.5 store / 模板支线（5）

| 脚本 | 行数 | 写到的 campaign | 作用 | 必需？ |
|---|---:|---|---|---|
| `run_store_measure_simd.sh` | 19 | `model_store_v1/raw` | SIMD store 目标网格（频率门控探针，走 `board_env_latest.sh`）| **★必需** |
| `run_store_measure_simt.sh` | 21 | `model_store_v1/raw` | SIMT store 目标网格（`num_warps` 1..64）| **★必需** |
| `run_template_stride_calibration_v2.sh` | 57 | `model_template_stride_v2` | 模板路径 v2 全量标定（path_check + load + store 三条命令串起来）| ○可复用 |
| `run_template_stride_measure_load.sh` | 26 | `model_template_stride_v2/raw` | rank-1 模板 load 测量（rotate loop 的 Event 斜率）| ○可复用 |
| `run_template_stride_measure_store.sh` | 26 | 同上 | rank-1 模板 store 测量 | ○可复用 |

### 8.6 定向单点实验 `board_*.sh`（15）

这些**不是批量网格**，而是「针对某几个点做对照实验」的一次性脚本，**规模几十个 case、几十分钟**。

| 脚本 | 结果目录 | 作用 | 必需？ |
|---|---|---|---|
| `board_expand2d.sh` | `dtype_calibration_v10_expand2d/raw` | 扩展 2D shape 网格，聚焦误差最大的槽（→ `source=v3_2d`）| **★必需** |
| `board_expand2d_b.sh` | `dtype_calibration_v11_expand2d_b/raw` | 缩减补充批次（A: R 扫描，B: W 扫描），限时 22:00 前跑完（→ `source=v4_2d`）| **★必需** |
| `board_wide12.sh` | `dtype_calibration_v12_wide12/raw` | SIMD wide 分支 stride=1 vs 2 的**等 span 对照**（分离搬运成本与抽取成本）| ○可复用 |
| `board_wide12b.sh` | `dtype_calibration_v12_wide12/raw` | 同上；用 `sed` 从主探针派生 `_wide12_probe.py`（只放宽 `stride < 2` 断言），分两阶段跑 | ○可复用 |
| `board_span4k.sh` | `dtype_calibration_v12_wide12/raw2` | 恒定约 **4 KB 跨度**下的受控对照（8 组 dtype×block×stride）| ○可复用 |
| `board_dir23.sh` | `dtype_calibration_v13_dir23/raw` | D2: int64 连续 store（stride=1）及配套 stride=2；D3: 大内 stride 用**小窗口 vs 大窗口**（8192/16383 ↔ 65536/65535）对照 | ○可复用 |
| `board_one1688.sh` | 就地 `--out` | 单点 1.688 复测（2D 探针）| ○可复用 |
| `board_verify1688.sh` | 就地 `--out` | 1.688 的单点验证（1D + 2D 两个探针）| ○可复用 |
| `board_repro2d.sh` | 就地 `--out` | 2D 单点可复现性复核 | ○可复用 |
| `board_reverify.sh` | 就地 `--out` | **严格复验 4x16 vs 8x8**：多种 `rotate-mask` / `rotate-step` / `reps`，看结论是否稳定 | ○可复用 |
| `board_rot2d_test.sh` | 就地 `--out` | rotation window 对 2D 结论的影响 | ○可复用 |
| `board_steptest.sh` | 就地 `--out` | **决定性测试**：rotate STEP（单位=元素）是否在混淆 dtype 对比——固定其它只动 `--rotate-step` | ○可复用 |
| `board_simt_thresh.sh` | 就地 `--out` | 找断点：int8 / W=4 / n=64，ISB 从 2 扫到 20 | ○可复用 |
| `board_test_shapes.sh` | 就地 `--out` | 若干 shape 的快速可用性验证（打印 valid / cases）| ○可复用 |
| `board_analyze_simtcm.sh` | `runs/simt_camodel/w2`、`w16` | 调 `analyze_strided_camodel.py` 解析两个 CAModel run，再用内嵌 Python 打印 stride / L128 / SIMT_LDG / DC hit-miss / BIU send 表（**属 CAModel 分析，见 §9**）| ○可复用 |

> 注：`board_wide12b.sh` / `board_dir23.sh` / `board_span4k.sh` 里的 `Sed` 会**覆盖重写** `scripts/_wide12_probe.py` / `scripts/_dir23_probe.py` ——
> 这两个文件是**生成物**，不是手写脚本；本次清理把它们删掉**不影响**上述驱动（运行时自动重建）。

### 8.7 已删除的 tmux 启动器，怎么替代

原本有 17 个 `board_launch_*.sh`（把驱动丢进 detached tmux 会话）。现已全部删除，等价做法：

```bash
# 后台跑一个 PASS（推荐）
setsid nohup bash -c "PASS=core1d bash scripts/run_final_campaign.sh > runs/core1d.log 2>&1" &

# 或者 tmux（多 PASS 循环）
tmux new-session -d -s final -c "$PRE" \
  "for p in core1d wsweep1d core2d wsweep2d n1; do PASS=$p bash $PRE/scripts/run_final_campaign.sh > $PRE/final_$p.log 2>&1; done; echo ALLDONE > $PRE/final_done.txt"
```

> 原 `board_launch_*.sh` 默认的 session 名 / PASS 列表 / runner / 日志前缀 / done 文件名，
> 记载在旧版 `INDEX.md` 的 §5.3 与 §2.2（可从 git 取回：`git show HEAD:pre-explore/scripts/INDEX.md`）。
> 进度监控（原 `board_progress.sh` / `board_eta.sh` / `board_watchdog.sh`）现由 `board_integrity.py` 与手工 `ps` + 数 `raw/*.json` 替代。

---

## 9. CAModel 白盒（12）

**这一层是整套物理模型的证据来源**：README 里「`biu_send_count = n`」「64 B = 2 个 32 B sector = MTE 一次 burst」
这类论断都出自这里。CAModel 是 simulator（`msopprof`），跑在板上但不占真卡。

| 脚本 | 行数 | 作用 | 必需？ |
|---|---:|---|---|
| `camodel/analyze_strided_camodel.py` | 499 | 解析 strided-**load** 的 CAModel dump，导出白盒事实（MTE2 队列、AIV vector 窗口、SIMT LSU/DC TagRam、`biu_send_count` / `biu_cmds` / `B per cmd`）；被 `analyze_stride2d_camodel.py` 与 `board_analyze_simtcm.sh` 复用 | **★必需** |
| `camodel/analyze_stride2d_camodel.py` | 127 | 复用上面的解析器，post-join rank-2 的 2D 白盒事实，输出 `parsed.json` / `parsed.csv` | **★必需** |
| `camodel/stride2d_camodel_probe.py` | 141 | rank-2 strided `tl.load` 的 CAModel 探针 | ○可复用 |
| `strided_load_probe.py` | 183 | 最早的「单 op strided load」探针（顶层，只服务 CAModel）| ○可复用 |
| `strided_store_probe.py` | 177 | 最早的「单 op shaped strided store」探针（顶层，只服务 CAModel）| ○可复用 |
| `camodel/run_camodel.sh` | 61 | CAModel 驱动（load）：建 `runs/<时间戳>/`、拷探针、调 simulator（`--launch-count` / `--timeout=2880`）| ○可复用 |
| `camodel/run_store_camodel.sh` | 63 | CAModel 驱动（store，多一个 `NUM_WARPS`）| ○可复用 |
| `camodel/run_stride2d_camodel.sh` | 45 | rank-2 strided load 的 CAModel 驱动 | ○可复用 |
| `camodel/run_simt_camodel.sh` | 42 | 纯 SIMT 分支的 CAModel 白盒（专攻半白盒公式误差最大的 regime）| ○可复用 |
| `camodel/board_camodel2.sh` | 12 | 板上案例：f16 / ISB=128 → IS=64 元素；4x16 rowpitch 4096B、8x8 rowpitch 2048B（CAModel 用 f32 等价元素数）| ○可复用 |
| `camodel/board_cam_wide2d.sh` | 10 | wide 分支的 2D 扫描（`4x16 8x8` × inner `1 2 3 6`），跑完立刻出分析 | ○可复用 |
| `camodel/board_camodel_an.sh` | 7 | 取最新的 `runs/shapes_*` 做一次分析 | ○可复用 |

> `board_analyze_simtcm.sh`（顶层）也属于这一层，为避免重复计数列在 §8.6。

---

## 10. 术语：测量批次（campaign）

### 10.1 这个词在这里是什么意思

本文档与 `results/` 目录名里的 **"campaign"** 指**一轮批量板上测量**，不是「运动 / 战役」的字面义。具体含义：

* **一个测量网格**：一个 `.sh` 驱动（或一个 `PASS` 分支）覆盖一片固定的 (op, path, dtype, block, stride, shape, num_warps) 组合；
* **由一个 `.sh` 启动**：`PASS=<分支> bash scripts/run_<批次名>.sh`，脚本内部循环调用探针；
* **规模：几十到几千个 case**（单点定向实验几十个；`run_final_campaign.sh` 一个 PASS 就上千）；
* **耗时：几十分钟到几小时**（SIMT 网格约 25 s/case，几千 case 就是十几小时，通常拆成多个 PASS 或分夜跑）；
* **产物落在一个同名目录**：`results/<campaign>/raw/*.json`（+ `measure_asm/*.ttir`、可选 `dataset*.csv`）。

在脚本里它对应的只是 `OUT` / `OUT_DIR` 环境变量和 `results/<campaign>/raw/` 这个路径约定；
`campaign` 这个词**只出现在文档与目录名里**，源码里没有这个标识符。

### 10.2 历史上各批次一览（32 个）

下表由现存 `.sh` 的 `OUT` / `results/...` 路径 + `pre-explore/results/README.md` 整理。
「规模」一列凡是写「约」的都是从脚本注释里的网格算式读出，不是实测计数。

| # | campaign | 启动脚本 | 规模 | 说明 |
|---:|---|---|---|---|
| 1 | `dtype_calibration_v1` | `run_dtype_calibration.sh`（`PASS=path\|pathlite\|simd\|simt\|template`）| 1D 主网格；产 `dataset.csv` **2560 行**（valid 2371）| **Phase A** 基线批次，字节间隙交叉设计 |
| 2 | `dtype_calibration_v2_rotate512` | `run_rotate512_campaign.sh`（`rot` / `rot2d` / `stepiso`）| 同 v1 网格的子集，512 MB rotation window；`source=v2_1d` **423 行** | **Phase R**：把新旧数据统一到历史窗口 |
| 3 | 同上（补点） | `run_balance_campaign.sh` | 5 block × 8 stride = 40 case/(op,path,dtype) | **Phase F-1**：均衡元素宽度点数 |
| 4 | 同上（补点） | `run_simdfill_campaign.sh` | SIMD load 稀疏槽，1D + 2D | **Phase F-2** |
| 5 | 同上（补点） | `run_gapfill_campaign.sh` | 单遍；补 SIMT f16 = 0 点等洞 | **Phase F-3** |
| 6 | `dtype_calibration_v3_fullgrid` | `run_fullgrid_campaign.sh`（`simd` / `simt`）| **2944 cell**（184 cell/(slot,dtype) × 4 slot × 4 dtype）| 1D 全网格 |
| 7 | `dtype_calibration_v3_delta` | `run_delta_campaign.sh` | 只测未测格点 | DELTA |
| 8 | `dtype_calibration_v3_delta` | `run_d2_campaign.sh`（`d2main` / `d2n1`）| `d2main` 432 case/(op,path,dtype) → **6912**；`d2n1` 18 → 288 | (row_stride, col_stride) 参数化 2D |
| 9 | `dtype_calibration_v4_final` | `run_final_campaign.sh`（5 个 PASS）| core1d **2048** + wsweep1d **1152** + core2d **6912** + wsweep2d **1296** + n1 **1008** ≈ **12416 case** | **FINAL 主批次**（含 SIMT `num_warps` 扫描）|
| 10 | `dtype_calibration_v4_final` | `run_final_delta.sh` | 只补 v4_final 缺失格点 | FINAL DELTA |
| 11 | `dtype_calibration_v4_final` | `run_core1d_remainder.sh` / `run_core1d_remainder2.sh` | 23 个 invocation | core1d 补跑（v1 丢 9 个结果，用 v2）|
| 12 | `dtype_calibration_v5_final` | `run_final2.sh`（`fix1` / `core2d` / `wsweep2d` / `n1`）| core2d 120 form（原 180）| FINAL-v2（缩减 2D）|
| 13 | `dtype_calibration_v6_int8fix` | `run_int8fix.sh` | **448 case**（2D int8 load 全部）| 修溢出后重测 |
| 14 | `dtype_calibration_v7_resample` | `run_resample.sh` | spread > 0.40 的全部点，更多样本 + 更多 attempt | 噪声点重测 |
| 15 | `dtype_calibration_v8_gapfill` | `run_gapfill_final.sh` | 精确矩形缺口 | 最后一轮补洞 |
| 16 | `dtype_calibration_v9_wsweep_full` | `run_wextend.sh`（`W=` 环境变量）| 1D 8×10 = **80** + 2D 6×4×5 = **120** per (op, dtype) | W 扩展全网格 |
| 17 | `dtype_calibration_v10_expand2d` | `board_expand2d.sh` | 扩展 2D shape 网格（几十 case）| 聚焦弱槽 → `source=v3_2d` |
| 18 | `dtype_calibration_v11_expand2d_b` | `board_expand2d_b.sh` | R 扫描 + W 扫描（缩减版）| → `source=v4_2d` **1952 行** |
| 19 | `dtype_calibration_v12_wide12` | `board_wide12.sh` / `board_wide12b.sh`（`raw/`）+ `board_span4k.sh`（`raw2/`）| 4 dtype × 2 op × 2 stride 的等 span 对照 + 8 组 4 KB 跨度对照 | wide 分支定向实验 |
| 20 | `dtype_calibration_v13_dir23` | `board_dir23.sh` | D2 int64 连续 store；D3 大 stride 小/大窗对照 | 定向实验 |
| 21 | `board_v2_simd` | `run_v2_simd_dense.sh` | 频率感知密集 SIMD 单 load | v2 协议 |
| 22 | `board_v2_simt` | `run_v2_simt_dense.sh` | 频率感知密集 SIMT Event 斜率 | v2 协议 |
| 23 | `board_v2_simd_more_blocks` / `board_v2_simt_more_blocks` | `run_v2_more_blocks.sh` | 补中间 / 大 block | v2 补充 |
| 24 | `board_v2_simt_warps` | `run_simt_numwarps_sweep.sh` / `run_simt_numwarps_dense.sh` | 粗扫 **224** 个 case + W=32/64 密集 + **29** 个高重复重测 | `num_warps` 专项 |
| 25 | `model_store_v1` | `run_store_measure_simd.sh` / `run_store_measure_simt.sh` | store 目标网格；`hist_store` **544 行** | 进 unified |
| 26 | `model_template_stride_v1` / `v2` | `run_template_stride_calibration_v2.sh` / `run_template_stride_measure_load.sh` / `_store.sh` | v2 共 **764 个文件** | 模板路径支线（被 `--exclude-template` 排除）|
| 27 | `model_v2`（`domain=single_shot`）| `run_v2_*.sh` → `build_v2_dataset.py` | `hist_v2_single_shot` **441 行** | 历史复用 |
| 28 | `model_v3_numwarps` | `build_v3_numwarps_dataset.py` | 224 粗 case + 密集 + 29 重测 | 历史复用 |
| 29 | `model_v4_numwarps` / `model_v4_simple` | `build_v4_numwarps_dataset.py` | `hist_v4` **992 行** | 历史复用 |
| 30 | `v4_aligned_23` / `v4_aligned_dev1` | `run_v4_aligned_23.sh` | 6 大 stride × W=1/2/4/8/16，物理 NPU1 | 页对齐测量 |
| 31 | `v4_retest_23_missing` / `v4_retest_all` / `v4_retest_full_23` | `run_v4_retest_missing.sh` / `run_v4_followup.sh` | **240** 个缺失 case × **5 次独立分配**；合并后 `v4_retest_full_23.json` **6.95 MB** | 5-allocation 重测 |
| 32 | `runs/` 下的 CAModel run 目录（`runs/<时间戳>/`、`runs_store/`、`runs/shapes_*`、`runs/is64_*`、`runs/wide2d_*`、`runs/simt_camodel/w*`）| `camodel/run_*.sh` / `camodel/board_*.sh` | 每个 run 十几到几十个 case（`--launch-count` 计数）| simulator 白盒，不占真卡 |

### 10.3 总体量级（来自 `pre-explore/results/README.md` 的实测）

| 口径 | 文件数 | 大小 |
|---|---:|---:|
| 提交 1 的 `pre-explore/results/` **工作区**实测 | 2381 | 168.3 MB |
| 其中被 `.gitignore` 忽略、从未提交的（各 campaign 自己的 `raw/` 副本）| 592 | ≈ 29.7 MB |
| 提交 1 **git 跟踪** | 1789 | 138.6 MB |
| 提交 2（当前工作区，只留最终数据集 / 模型 / 报告）| **111** | **38.2 MB** |

即：**37 个 campaign / 中间产物目录**（1434 个文件）已被删除，它们的内容全部已被特征化进 `results/final/` 的 CSV；
能追溯的原始数据是 `results/final/raw/` 的 **231 个 JSON（29 MB）**——它是 v4/v5/v8/v9 四代 raw 的合并视图。

---

## 11. 维护提醒

1. **`board_env.sh` / `board_env_latest.sh` 是本次从 git 恢复的**（清理时被依赖闭包漏掉）。
   任何「只保留必要脚本」的清理都必须把 `bash scripts/xxx.sh` 这种**非 import 依赖**算进去——
   现存 42 个板端 `.sh` 全部依赖它们，删掉等于整条板端链路不可运行。
2. **不要用 `run_core1d_remainder.sh`（v1）**：它因输出文件名冲突静默丢结果（23 个里丢 9 个）。用 `run_core1d_remainder2.sh`。
3. **`_wide12_probe.py` / `_dir23_probe.py` 是生成物**，由 `board_wide12b.sh` / `board_dir23.sh` / `board_span4k.sh` 里的 `sed` 从 `dtype_stride_measure_probe.py` 覆盖写出。它们已被删除且**不需要**手动重建。
4. **特征定义的唯一真相是 `stride_feature_reference.py`**（精确整数语义）。任何拟合脚本、C++ 移植里的特征计算都必须与它一致。
5. **`_tbl.py` 与 `_readme4.py` 的文件名不匹配**：前者写 `_readme_tbl.json`，后者读 `_tbl.json`，中间要手工复制一次。
6. **`board_integrity.py` 的结果目录写死**在文件第 4 行（`results/dtype_calibration_v10_expand2d/raw`），换 campaign 要改。
7. **`run_wextend.sh` 的 `W` 环境变量**取代了已删的 `run_wextend_w2.sh`：`W=2 bash scripts/run_wextend.sh` 即历史 W=2 批次。
8. **`PHYS*_*.json` 的写回脚本已删除**（见 §2.4）：报告链仍可读，但物理拟合轮次无法用现存脚本重跑。需要时从 git 取回。
9. 本索引对应的脚本快照 = **106 个脚本**（52 `.py` + 54 `.sh`），与 §0.1 的实测一致。
   改动本目录任何被 import 的库（§6.2）前，先确认下游影响面。

---

*本文件由实际扫描 `pre-explore/scripts/` 后重写，覆盖现存 106 个脚本；*
*清理前的 622 个脚本的完整分类索引仍在 git 历史里：`git show HEAD:pre-explore/scripts/INDEX.md`。*
