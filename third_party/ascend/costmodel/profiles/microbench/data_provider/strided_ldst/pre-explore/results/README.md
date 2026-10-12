# `pre-explore/results/` 数据目录清单

> 本文由**实际扫描目录**生成（`os.walk` + `stat` + 逐个 JSON/CSV 解析），所有文件数、字节数、行数、MAPE 都是实测值，不是估计。
> 扫描根：`third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/results/`。
> 路径若未特别说明，均相对该根目录。

**配套文档**：[`DATA_INVENTORY.md`](DATA_INVENTORY.md) 是按「标定阶段」逐条索引「哪一组测量 → 哪个结果文件」的细表；
本文是**目录级**清单（campaign / 数据集 / 模型 / 报告 / 血缘），两者互补，不重复。

---

## 0. 总览

| 项 | 值 |
|---|---|
| 顶级条目 | 49（37 个 campaign/结果子目录 + 12 个根目录散落文件/文档） |
| 文件总数 | **2381 个**（`os.walk` 实测） |
| 占用空间 | **168.3 MB**（176 498 999 B 实测） |
| 最大的三块 | `final/` 60.2 MB、`merged/` 21.4 MB、`v5raw/` 14.0 MB |
| 板上环境 | `ssh ascend-950pr-63`（用户 `c00946898`，环境 `wj_autoscope`，256 核） |
| 板上工作目录 | `~/strided-template-calibration-v2/` |
| 统一入口 | [`../scripts/board_env.sh`](../scripts/board_env.sh)（所有 `board_*.sh` 先 source 它） |

---

## 1. 数据血缘（一句话链路 + 实际脚本）

```
  [板上] 探针 Python（scripts/*_measure_probe.py、*_probe.py）
      │  经 scripts/board_env.sh 设置 asc 环境后在 ascend-950pr-63 上运行
      │  写 results/<campaign>/raw/*.json
      │  顶层结构：{kind, op, path, dtype, element_bytes, cases:[...], buffer, frequency_check}
      │  kind ∈ {dtype_stride_measure(1D), stride2d_measure(2D), dtype_calibration_dataset, stride2d_dataset}
      ▼
  [特征化] build_*.py  ── 用 scripts/stride_feature_reference.py 统一算特征
      │  build_dtype_dataset.py        → <campaign>/dataset.csv        (rank-1)
      │  build_2d_dataset.py           → <campaign>/dataset_2d.csv     (rank-2)
      │  build_store_dataset.py        → model_store_v1/dataset.csv
      │  build_template_stride_dataset.py → model_template_stride_v*/dataset.csv
      │  build_unified_dataset.py      → final/dataset_unified_v*.csv  (rank1+rank2 同一套特征命名空间)
      │  build_v{2,3,4}_numwarps_dataset.py → model_v{2,3,4}*/dataset.csv
      ▼
  [拟合] fit_*.py  ── 加权最小二乘（权重 1/T，即最小化相对误差=MAPE）、NNLS、贪心前向+后向消除、max() 结构
      │  写 results/final/MODEL_*.json、fit_*.json、model_v*.json
      │  物理重构回合写 results/final/PHYS*.json（PHYS_ = 第 1 轮，PHYS2_ = 第 2 轮，PHYS3_ = 第 3 轮）
      ▼
  [报告] gen_report*.py  ── gen_report_v5.py 按 PHYS3 > PHYS2 > PHYS 优先级合并成最终报告
      │  写 results/final/REPORT_*.md
      ▼
  [文档] scripts/_readme{1..4}.py  →  pre-explore/README.md + pre-explore/README_data.json
```

**端到端重跑命令（唯一一条真正打通全链路的脚本）**

```bash
# scripts/run_final_pipeline.py 是唯一把 raw → dataset → new-only → 拟合串起来的脚本
python3 scripts/run_final_pipeline.py --build \
    --unified pre-explore/results/final/dataset_unified_v5.csv
#  steps:
#   1) build_2d_dataset.py  --raw-dir results/expand2d/raw   --out-dir results/_build/build_c1
#   2) build_2d_dataset.py  --raw-dir results/expand2d_b/raw --out-dir results/_build/build_c2
#   3) build_unified_dataset.py --out results/final/dataset_unified_v5.csv --exclude-template \
#        --v1-1d results/final/dataset.csv  --v1-2d results/final/dataset_2d.csv \
#        --v3-2d results/_build/build_c1/dataset_2d.csv --v4-2d results/_build/build_c2/dataset_2d.csv
#   4) 过滤 source.startswith(v1_/v2_/v3_/v4_) → results/final/dataset_new_only_v5.csv
#   5) 逐 slot 贪心拟合（≤10 项，可选 max(a*num_warps, linear)）→ results/final/MODEL_FINAL_V7.json
```

> 注意：`run_final_pipeline.py` 生成的 `MODEL_FINAL_V7.json` 是**统计选型链**的末端，**不是**最终交付的物理公式；
> 最终 10 条公式来自物理重构回合写出的 `final/PHYS3_*.json` 与 `final/PHYS2_*.json`（见 §4）。

---

## 2. Campaign 子目录清单

### 2.1 板上原始测量 campaign（raw JSON 的产地）

| 目录 | 文件数 | 大小 | 是什么 campaign（实测网格） | 对应脚本 |
|---|---:|---:|---|---|
| `dtype_calibration_v1/` | 500 | 3.28 MB | **阶段 A+B 基线**。`raw/` 48 个 JSON；op=load,store；path=simd,simt_only,template；dtype=int8,float16,float32,int64（+bf16/int32 仅编译）；block=16..1024；stride=2..32；num_warps=1,4,32；2D shape=8x16,16x32,1x64,1x256。另含 `measure_asm/` 255 个逐 case 汇编/IR 转储（`.ttir/.source/.bcmlir/.mlirbc/.ttadapter`）、`ir_evidence/` 8 个 path-check、`camodel_2d/` 5 个白盒 | [`run_dtype_calibration.sh`](../scripts/run_dtype_calibration.sh)、`run_phaseA_measure.sh`、`run_phaseB_campaign.sh`、`board_launch_all.sh`；探针 `dtype_stride_measure_probe.py` / `stride2d_measure_probe.py`；构建 `build_dtype_dataset.py` / `build_2d_dataset.py` |
| `dtype_calibration_v2_rotate512/` | 85 | 2.11 MB | **rotate-512 重测**（target 口径改为「512 次迭代 rotate-loop 的最小 event slope」）。84 个 JSON = 34 个 2D + 49 个 1D + 1 个 dataset meta。block=16..512；stride=2..64；num_warps=1；dtype 四种；2D shape=8x16,16x32,1x64,1x256。`dataset.csv` 427 行 | [`run_rotate512_campaign.sh`](../scripts/run_rotate512_campaign.sh)、`run_balance_campaign.sh`、`run_gapfill_campaign.sh`、`run_simdfill_campaign.sh`（四者 `OUT` 都默认指向该目录）；构建 `build_v2_dataset.py` |
| `dtype_calibration_v4_final/` | 25 | 3.41 MB | **1D core 补扫（core1d remainder）**。`raw/` 21 个 JSON；这是 1D 网格最宽的一批：block=4..512；stride=2..256；num_warps=1,2,4,8,16,32,64；dtype 四种；path=simd,simt_only；2D shape=4x1,4x4,4x8,4x16,8x8,8x16,16x1,16x16。`dataset.csv` 76 行、`dataset_2d.csv` 128 行 | [`run_final_campaign.sh`](../scripts/run_final_campaign.sh)、`run_final_delta.sh`、`run_core1d_remainder.sh`、`run_core1d_remainder2.sh`（`OUT` 默认 = 本目录）、`board_n1_status.sh`、`board_progress.sh` |
| `v4raw/` | 69 | 6.01 MB | **v4 代原始测量**：17 个 `stride2d_measure` + 52 个 `dtype_stride_measure`。block=4..512；stride=2..256；W=1..64；dtype 四种；2D shape=4x1..16x16 | 由 `board_launch_*.sh` / `run_v4_*.sh` 系列产出。目录名在合并进 `final/raw/` 后缩短为 `v4raw`，脚本里已不再出现该短名 |
| `v5raw/` | 112 | 14.03 MB | **v5 代**：72 个 2D + 40 个 1D。2D 全面扩形：shape 覆盖 4x4..16x16，inner_stride=2..16，row_stride=2..256。文件名后缀 `_core` / `_wsweep` / `_fix1` / `_fix2_<blocks>_<strides>` / `_n1` | 同上（v5 代 runner） |
| `v8raw/` | 34 | 2.87 MB | **v8 代 gap 补扫**：8 个 2D + 26 个 1D，文件名后缀 `_gap_<row_strides>_<inner_strides>`；row_stride=16..512，inner_stride=2..64，block=4..512，W=1..64 | 同上（v8 代 runner，含 `board_launch_gapfill_final.sh`） |
| `v9raw/` | 16 | 6.06 MB | **v9 代 wfull 全扫描**：8 个 2D + 8 个 1D，path 全部 `simt_only`，W=1，block=4..512 × stride=2..256 全交叉（文件名 `_wfull_`） | 同上（v9 代 runner） |
| `expand2d/` | 20 | 10.97 MB | **2D 扩形（c1）**：`raw/` 20 个 `stride2d_measure`，op=load,store；path=simd,simt_only；dtype 四种；W=1,4；R×C 覆盖 1x16,1x32,1x64,2x8..2x64,4x8..4x32,8x4..8x32,16x4..16x32,32x8,32x16 | [`board_expand2d.sh`](../scripts/board_expand2d.sh)（脚本内旧名 `dtype_calibration_v10_expand2d`）；被 [`run_final_pipeline.py`](../scripts/run_final_pipeline.py) 当作 `--build` 的 c1 输入 |
| `expand2d_b/` | 22 | 8.93 MB | **2D 扩形（c2，纯 SIMT 补充）**：`raw/` 22 个，path 全为 `simt_only`；W=1,2,4,8,16,32,64；R×C 覆盖 2x8..64x8、2x16..8x16、2x64,2x128,4x64,8x64 等。后缀 `_A_rs`（row-stride 扫描）/ `_B_ws`（warp 扫描） | [`board_expand2d_b.sh`](../scripts/board_expand2d_b.sh)（旧名 `dtype_calibration_v11_expand2d_b`）；`run_final_pipeline.py` 的 c2 输入 |
| `dir23/` | 5 | 0.23 MB | **定向补充窗口探针**：`d2_store_simd_int64_W1`、`d3_load_simd_f32_W1_win{big,small}`、`d3_store_simt_f32_W4_win{big,small}`。op=load,store；path=simd,simt_only；dtype=float32,int64；block=4..1024；stride=1,2,32,128,512,1024；W=1,4 | [`board_dir23.sh`](../scripts/board_dir23.sh)（旧名 `dtype_calibration_v13_dir23/raw`） |
| `wide12/` | 7 | 0.57 MB | **SIMD wide 分支 stride∈{1,2} 复核**：1D，path=simd，dtype 四种，block=4..1024，stride=1,2，W=1，load+store。文件名 `1d_<op>_simd_<dtype>_W1_s1_2.json` | [`board_wide12.sh`](../scripts/board_wide12.sh) / [`board_wide12b.sh`](../scripts/board_wide12b.sh)（旧名 `dtype_calibration_v12_wide12/raw`）、[`board_span4k.sh`](../scripts/board_span4k.sh)（写 `raw2/`）、`_p2_write.py` |
| `board_v2_simd/` | 1 | 0.17 MB | **最早一代 Event 频率重试**（无特征，只有 cycles/ns）。220 个 case，block=16,32,64,128,256 × stride=1..256（44 个取值） | [`run_v2_simd_dense.sh`](../scripts/run_v2_simd_dense.sh)、[`frequency_retry_probe.py`](../scripts/frequency_retry_probe.py) |
| `board_v2_simd_more_blocks/` | 1 | 0.24 MB | 同上，补非 2 次幂 block：8,24,48,96,192,384,512（308 case） | [`run_v2_more_blocks.sh`](../scripts/run_v2_more_blocks.sh) |
| `board_v2_simt/` | 1 | 0.12 MB | 纯 SIMT 版：115 case，path=simt_only，block=16..256，stride=1..256 | [`run_v2_simt_dense.sh`](../scripts/run_v2_simt_dense.sh)、`run_simt_numwarps_sweep.sh`；被 `build_v3_numwarps_dataset.py` 读取 |
| `board_v2_simt_more_blocks/` | 1 | 0.07 MB | 纯 SIMT 补 block=4,8,512（69 case） | [`run_v2_more_blocks.sh`](../scripts/run_v2_more_blocks.sh) |
| `board_v2_simt_warps/` | 1 | 0.38 MB | **num_warps 扫描**：344 case，block=32,64,128,256 × stride=1..256 × W=1,2,4,8,16,32,64 | [`run_simt_numwarps_sweep.sh`](../scripts/run_simt_numwarps_sweep.sh)、[`run_simt_numwarps_dense.sh`](../scripts/run_simt_numwarps_dense.sh)（`OUT_DIR` 默认 = 本目录）、[`simt_frequency_retry_probe.py`](../scripts/simt_frequency_retry_probe.py) |
| `model_store_v1/raw/` | 3 | 1.26 MB | **store 专项原始测量**：`board_store_simd.json`(179 KB)、`board_store_simd_wide_more.json`(13 KB)、`board_store_simt.json`(991 KB)。mode=simd,simt_only；block=8..512（12 值）；stride=1..256（23 值）；W=1..64 | [`run_store_measure_simd.sh`](../scripts/run_store_measure_simd.sh)、[`run_store_measure_simt.sh`](../scripts/run_store_measure_simt.sh)；探针 [`store_measure_probe.py`](../scripts/store_measure_probe.py) / [`strided_store_probe.py`](../scripts/strided_store_probe.py) |
| `model_template_stride_v1/` | 31 | 1.67 MB | **Triton 模板路径（`compile_mode=simd_simt`）1D stride 标定 v1**：f32，block=16..2048（6 值）× 19 个非 2 次幂 stride（3..255）× W=1..8。`dataset.csv` 190 行 | [`fit_template_stride_v1.py`](../scripts/fit_template_stride_v1.py)、[`predict_template_stride_v1.py`](../scripts/predict_template_stride_v1.py)、[`make_template_stride_plots.py`](../scripts/make_template_stride_plots.py) |
| `model_template_stride_v2/` | 764 | 9.82 MB | **模板路径 v2（取代 v1）**：f32，block=4..2048 × 37 个非 2 次幂 stride（3..255）× W=32。`raw/` 4 个 JSON（load/store × pass1/pass2，共 1206/1480 case）+ 4 个 log；`ir_evidence/measure_asm/` 740 个汇编证据 + `template_stride_path_check_v2.json`(673 KB)。`dataset.csv` 740 行 | [`run_template_stride_calibration_v2.sh`](../scripts/run_template_stride_calibration_v2.sh)、`run_template_stride_measure_{load,store}.sh`、[`build_template_stride_dataset.py`](../scripts/build_template_stride_dataset.py)、[`fit_template_stride_v2.py`](../scripts/fit_template_stride_v2.py)、[`template_stride_path_check.py`](../scripts/template_stride_path_check.py) |

**关键事实**：`final/raw/` 的 231 个 JSON = `v4raw`(69) + `v5raw`(112) + `v8raw`(34) + `v9raw`(16)，文件名统一加了 `vN__` 前缀（逐个核对前缀计数完全吻合：`{v4:69, v5:112, v8:34, v9:16}`）。
`merged/raw/` 的 181 个 JSON = `v4raw`(69) + `v5raw`(112)，是 v8/v9 加入之前的中间合并快照。

### 2.2 CAModel 白盒证据目录（对照真卡实测的硬件计数器）

| 目录 | 文件数 | 大小 | 内容 | 对应脚本 |
|---|---:|---:|---|---|
| `block32_s3/` | 4 | 0.02 MB | load 白盒单点：block=32、stride=3、W=1、f32。`parsed.json`/`parsed.csv`（AIV/MTE2/BIU 计数器）+ `launch_manifest.json` + `run.log` | [`make_summary.py`](../scripts/make_summary.py)；探针在 `scripts/camodel/`（`board_cam_survey.py`、`board_cam_deep.sh`、`board_launch_camodel2d.sh`） |
| `block64_s3/` | 4 | 0.02 MB | 同上，block=64 | 同上 |
| `block128_s3/` | 4 | 0.02 MB | 同上，block=128 | 同上 |
| `store_camodel_b32/` | 3 | 0.08 MB | store 白盒：block=32 × stride=1,2,4,8,16,32,64,128,256 × simd/simt_only。`parsed.json`/`parsed.csv` + `launch_manifest.json` | [`camodel/run_store_camodel.sh`](../scripts/camodel/run_store_camodel.sh)、[`camodel/summarize_strided_store_camodel.py`](../scripts/camodel/summarize_strided_store_camodel.py)、[`make_store_plots.py`](../scripts/make_store_plots.py) |
| `store_camodel_b64/` | 3 | 0.08 MB | 同上，block=64 | 同上 |
| `store_camodel_b128/` | 3 | 0.08 MB | 同上，block=128 | 同上 |
| `strided_load_summary.csv`（根） | 1 | 1.2 KB | 上面 block{32,64,128}_s3 的白盒汇总表（run/mode/block/stride/useful_bytes/span_bytes/task_cycles/BIU cmds/DC reqs…） | [`make_summary.py`](../scripts/make_summary.py) |
| `strided_load_summary.md`（根） | 1 | 1.0 KB | 上表的 Markdown 版 | 同上 |

### 2.3 数据集 / 模型结果目录（中间产物）

| 目录 | 文件数 | 大小 | 是什么 | 对应脚本 |
|---|---:|---:|---|---|
| `model_v2/` | 1 | 0.07 MB | 只有 `dataset.csv`（v2 代 1D f32 load 数据集）。**历史数据集，未重跑** | [`build_v2_dataset.py`](../scripts/build_v2_dataset.py) |
| `model_v3_numwarps/` | 1 | 0.06 MB | 只有 `dataset.csv`（num_warps 扫描数据集） | [`build_v3_numwarps_dataset.py`](../scripts/build_v3_numwarps_dataset.py)（同时读 `board_v2_simt/`、`board_v2_simt_warps/`） |
| `model_v4_numwarps/` | 7 | 0.42 MB | `dataset.csv` + 6 个指标 csv；`metrics_summary.json` = {mape_pct, p50, p90, p95, max, bias, rmse, n} | [`build_v4_numwarps_dataset.py`](../scripts/build_v4_numwarps_dataset.py)、[`fit_strided_model_v8.py`](../scripts/fit_strided_model_v8.py)、[`compare_strided_load_v7_v8_on_v4.py`](../scripts/compare_strided_load_v7_v8_on_v4.py) |
| `model_v4_simple/` | 11 | 0.47 MB | v4 简化模型：`model_v4_simple.json`（coefficients / coefficients_ns / LOBO / LOWO 折 / in_sample）+ `metrics_summary.{csv,json}` + `errors_*.csv` + 分组指标 | [`fit_strided_model_v8.py`](../scripts/fit_strided_model_v8.py)、[`predict_strided_load_v8.py`](../scripts/predict_strided_load_v8.py) |
| `model_v4_simple_p25/` | 10 | 0.46 MB | v4_simple 的 p25 分位变体：额外含 `cv_candidates.csv`、`cv_lobo_folds.json`、`cv_lowo_folds.json`、按 block/num_warps/stride 的分组指标 | 无脚本硬编码引用（历史变体） |
| `model_v7_simple/` | 11 | 0.67 MB | v7 纯 SIMT load 模型：`model_v7_simple.json` 分 `simt`/`simd_wide`/`simd_gather`(±`_target`) 四支 + `plots/` 4 张 png + `errors_*.csv` | [`fit_strided_model_v7_simple.py`](../scripts/fit_strided_model_v7_simple.py)、[`predict_strided_load_v7_simple.py`](../scripts/predict_strided_load_v7_simple.py)、[`make_v7_plots.py`](../scripts/make_v7_plots.py)、[`evaluate_strided_load_v7_on_v4.py`](../scripts/evaluate_strided_load_v7_on_v4.py) |
| `model_store_v1/` | 15 | 1.94 MB | store 专项标定 v1：`dataset.csv` 744 行（mode/block/stride/num_warps + 30 个特征列）+ `model_store_v1.json` + `metrics_summary.{csv,json}` + `errors_{all,simd_gather,simd_wide,simt}.csv` + `plots/` 4 张 png | [`build_store_dataset.py`](../scripts/build_store_dataset.py)、[`fit_strided_store_v1.py`](../scripts/fit_strided_store_v1.py)、[`predict_strided_store_v1.py`](../scripts/predict_strided_store_v1.py)、[`make_store_plots.py`](../scripts/make_store_plots.py) |
| `refit_no_stride1_v1/` | 16 | 0.74 MB | **阶段 0「零新增测量」重拟**：丢掉所有 `stride==1` 行后重拟 v4_simple / v7_simple / store_v1。`load/` 9 个文件、`store/` 4 个文件，另有 `loading_audit.csv`、`old_vs_new.csv`、`metrics_summary.json` | [`refit_no_stride1.py`](../scripts/refit_no_stride1.py) |
| `unified_formulas_v1/` | 5 | 0.99 MB | **第一次「统一 6 式」尝试**：`dataset_unified.csv` 3638 行/43 列；`model_unified6.json`（`version=unified_v1`，`formula_count=7`，含 `template_load`/`template_store`）；`metrics_summary_unified6.csv`、`worst_cases_unified6.csv`、`FORMULA_SUMMARY.md`(28 KB) | [`fit_unified_model.py`](../scripts/fit_unified_model.py)、[`fit_unified6.py`](../scripts/fit_unified6.py)、[`summarize_unified6.py`](../scripts/summarize_unified6.py) |
| `unified_formulas_v2/` | 5 | 0.79 MB | 同上但 `--exclude-template` 后只剩 5 式：`dataset_unified.csv` 2935 行；`model_unified6.json`（`formula_count=5`）；`FORMULA_SUMMARY.md`(23 KB) | [`fit_unified6.py`](../scripts/fit_unified6.py)、[`summarize_unified6.py`](../scripts/summarize_unified6.py)（其 `--dir` 默认值就是本目录） |
| `_build/build_c1/` | 2 | 0.83 MB | 由 `expand2d/raw` 建出的 2D 数据集（`dataset_2d.csv` 830 KB + meta）；`run_final_pipeline.py --build` 的中间产物 | [`run_final_pipeline.py`](../scripts/run_final_pipeline.py) → [`build_2d_dataset.py`](../scripts/build_2d_dataset.py) |
| `_build/build_c2/` | 2 | 0.67 MB | 由 `expand2d_b/raw` 建出 | 同上 |
| `_tmpbuild/` | 2 | 0.05 MB | 一次性草稿：`dataset_2d.csv`(51544 B) 与 `dataset_2d_meta.json`(4313 B) 与 `dtype_calibration_v4_final/` 下同名文件**字节数完全一致**，即该目录的临时副本 | 无脚本引用 |
| `merged/` | 185 | 21.39 MB | **v4+v5 合并快照**：`raw/` 181 个 JSON；`dataset.csv` 1520 行/51 列；`dataset_2d.csv` 2352 行/48 列（meta 记 89 个 raw 文件、2628 case、2215 valid）；+ `dataset_meta.json`、`dataset_2d_meta.json` | 无脚本硬编码引用（v8/v9 加入前的中间快照） |
| `final/` | 380 | 60.22 MB | **最终汇合目录**（下详 §3–§5） | 见 §3–§5 |

### 2.4 `final/` 内部结构

| 子项 | 文件数 | 说明 |
|---|---:|---|
| `final/raw/` | 231 | v4/v5/v8/v9 四代原始 JSON，文件名加 `vN__` 前缀（= v4raw+v5raw+v8raw+v9raw） |
| `final/dataset*.csv` + `*_meta.json` | 15 | 见 §3 |
| `final/MODEL_*.json` | 41 | 见 §4.1 |
| `final/PHYS*.json` | 16 | 见 §4.2（**最终 10 条公式的载体**） |
| `final/fit_*.json` | 11 | `fit_v9_greedy`、`fit_v10_exhaustive`、`fit_v11_stepwise`、`fit_v12_big`、`fit_v13`、`fit_v14_crossed`、`fit_v15_Rcross`、`fit_v18_indicator`、`fit_v19_indicators`、`fit_v20_crosscand`、`fit_v21` —— 逐轮搜索的中间状态 |
| `final/model_v{9,15,16,17}*.json` | 4 | 早期整模型（v9 / v15 全特征 / v16 后向消除 / v17 去重+不确定性） |
| `final/_*.json` / `_p3_feats.csv` | 44 | `_p2c_*`、`_p3_*`、`_simt_*`、`_phys3_*`、`_greedy_*` 等子代理推导过程的中间状态 |
| `final/REPORT_*.md` | 7 | 见 §5 |
| `final/fit_v5/` | 3 | `fit_unified6.py` 在**含历史复用行的全量 unified** 上的产物：`model_unified6.json` + `metrics_summary_unified6.csv` + `worst_cases_unified6.csv`（除 `simd_load_gather` 外各 slot n 都更大，如 `simd_load_wide` n=141） |
| `final/fit_newonly/` | 3 | 同一脚本在 **`dataset_new_only_v5.csv`**（只含本轮新测行）上的产物（`simd_load_wide` n=132） |
| 其他 5 个 | 5 | `COMPOSITION.json`（10 个 slot 的数据构成）、`FEATURE_MEANING.json`（特征中文释义）、`DOMAIN_CURVE.json`（各 slot 的 domain 收敛曲线）、`PRUNE_CURVE.json`（剪枝曲线）、`features_dedup.json`（156 个特征去重表） |

### 2.5 根目录散落文件

| 文件 | 大小 | 说明 | 对应脚本 |
|---|---:|---|---|
| `v4_retest_full_23.json` | 7.19 MB | 「23 组配置全重测」的合并结果（顶层 `{results:[...]}`）。用于把早期不对齐的 v4 数据替换掉 | [`merge_v4_retest.py`](../scripts/merge_v4_retest.py)、[`retest_strided_simt_cases.py`](../scripts/retest_strided_simt_cases.py)、`run_v4_retest_missing.sh`、`run_v4_aligned_23.sh`（脚本内写作 `results/v4_aligned_23`） |
| `_phys_r2_table.json` | 116 KB | 第 2 轮物理重构的 222 行分组表 | `phys_r2_step*.py` |
| `_phys_r2_models.json` | 2.8 KB | 第 2 轮 12 个候选模型（M1..M12）的 MAPE 与系数 | 同上 |
| `_phys_r2_final.json` / `_phys_r2_greedy{,2}.json` / `_phys_r2_lib.json` | <1 KB 各 | 分组终选、特征名表、贪心结果 | 同上 |
| `_p3_store_r1_final.json` | 764 B | 第 3 轮 store_r1 的 4 组候选（`old/new5/new6/val`）对比 | `p3_store_r1_*.py` |
| `strided_load_summary.csv` / `.md` | 1.2 / 1.0 KB | CAModel 白盒汇总表（见 §2.2） | [`make_summary.py`](../scripts/make_summary.py) |
| `DATA_INVENTORY.md` | 9.0 KB | 按标定阶段 A/B 的逐条数据索引（**不是本文件**，两者互补） | — |

---

## 3. 最终数据集：`final/dataset_*.csv`

### 3.1 全部 dataset csv（实测行数）

| 文件 | 行数 | 列数 | 生成脚本 | 说明 |
|---|---:|---:|---|---|
| `final/dataset.csv` | 2560 | 51 | [`build_dtype_dataset.py`](../scripts/build_dtype_dataset.py) | rank-1（1D）主数据集 |
| `final/dataset_2d.csv` | 3432 | 48 | [`build_2d_dataset.py`](../scripts/build_2d_dataset.py) | rank-2（2D tile）主数据集 |
| `final/dataset_unified.csv` | 8016 | 43 | [`build_unified_dataset.py`](../scripts/build_unified_dataset.py) | 第一次统一（rank1+rank2 重特征化到同一命名空间） |
| `final/dataset_unified_dedup.csv` | 7895 | 43 | 同上 + 去重 | 去重版 |
| `final/dataset_unified_v2.csv` | 8016 | 56 | 同上 | v2：+ 2D 行/列/间距特征 |
| `final/dataset_unified_v3.csv` | 8016 | 59 | 同上 | v3：+ `valid`/`quality`/`target_spread` |
| `final/dataset_unified_v4.csv` | 9456 | 59 | 同上 | v4：并入 v8/v9 代 raw |
| **`final/dataset_unified_v5.csv`** | **12040** | **59** | 同上 | **最终统一数据集**（并入 expand2d / expand2d_b 两轮 2D 扩形） |
| `final/dataset_new_only.csv` | 5918 | 43 | `run_final_pipeline.py`（过滤 `source` 前缀） | 「只看本轮新测」视图 |
| `final/dataset_new_only_v2.csv` | 6039 | 56 | 同上 | |
| `final/dataset_new_only_v3.csv` | 6039 | 59 | 同上 | |
| `final/dataset_new_only_v4.csv` | 7479 | 59 | 同上 | |
| **`final/dataset_new_only_v5.csv`** | **10063** | **59** | 同上 | **最终拟合用数据集**（物理公式全部在它上面拟合） |

另有各 campaign 自己的 `dataset.csv` / `dataset_2d.csv`（见 §2.1、§2.3），以及 `final/dataset_meta.json`（kind=dtype_calibration_dataset，total 2560 / valid 2371）与 `final/dataset_2d_meta.json`（kind=stride2d_dataset，total 3524 / valid 3262，raw_files 89 个）。

### 3.2 `dataset_unified_v5.csv` vs `dataset_new_only_v5.csv`：为什么两个文件

`build_unified_dataset.py` 把 5 类来源拼进一张表，行首 `source` 列标明出处：

| `source` | 行数 | 含义 |
|---|---:|---|
| `v1_2d` | 3392 | 本轮 `final/dataset_2d.csv`（rank-2） |
| `v1_1d` | 2184 | 本轮 `final/dataset.csv`（rank-1） |
| `v3_2d` | 2112 | `_build/build_c1`（`expand2d/raw`） |
| `v4_2d` | 1952 | `_build/build_c2`（`expand2d_b/raw`） |
| `hist_v4` | 992 | **历史复用**：`model_v4_numwarps`/`model_v4_simple` 的数据 |
| `hist_store` | 544 | **历史复用**：`model_store_v1/dataset.csv` |
| `hist_v2_single_shot` | 441 | **历史复用**：`model_v2`（`domain=single_shot`） |
| `v2_1d` | 423 | `dtype_calibration_v2_rotate512` 的 1D 行 |

`dataset_new_only_v5.csv` = 只保留 `source ∈ {v1_1d, v1_2d, v2_1d, v3_2d, v4_2d}` 的 10063 行（丢掉 3 类 `hist_*` 共 1977 行）。
**为什么拟合要用 new-only**：`hist_v2_single_shot` 的 target 口径不同（`domain=single_shot`，单次发射而非 rotate-loop slope），混进来会污染标定；
所以全量 unified 只用于「跨口径对照」，真正拟合一律用 new-only。

其余字段分布（`dataset_unified_v5.csv` 实测）：

| 维度 | 分布 |
|---|---|
| `op` | load 5790 / store 6250 |
| `path` | simt_only 8812 / simd 3228 |
| `dtype` | float32 4254 / int8 2775 / float16 2729 / int64 2282 |
| `rank` | 1 → 4584 / 2 → 7456 |
| `num_warps` | 1:6530、2:594、4:2540、8:594、16:594、32:594、64:594 |
| `valid` | 1 → 11518 / 0 → 522（new-only 中 1 → 9541 / 0 → 522） |
| `domain` | rotate 11599 / single_shot 441 |

### 3.3 `dataset_unified_v5.csv` 的 59 列含义

所有特征都由 [`stride_feature_reference.py`](../scripts/stride_feature_reference.py) 唯一实现（`dataset_meta.json` 的 `feature_source` 字段也这么记），
原则是**一律用字节域或纯计数，绝不用元素域 stride**，这样 dtype 宽度以「因子」而非「分支」进入公式。

| 列组 | 列名 | 含义 |
|---|---|---|
| **实验条件（11 列）** | `op` | `load` / `store` |
| | `path` | `simd` / `simt_only`（template 路径已被 `--exclude-template` 排除） |
| | `dtype`, `element_bytes` | `int8/float16/float32/int64` → 1/2/4/8 |
| | `num_warps` | warp 数 1..64 |
| | `rank` | 1（1D）或 2（2D tile） |
| | `elements` | 本次访问的元素总数 n（1D = block；2D = R×C） |
| | `inner_stride_bytes` | 内层跨步字节数 ISB = stride × eb |
| | `source` | 数据来源（见 §3.2） |
| | `domain` | `rotate`（旋转缓冲循环，主口径）/ `single_shot` |
| **目标值（4 列）** | `target_ns` | 板上 Event 测得的每次迭代耗时（ns），同一配置多次 pass 取 **min**（竞争只会抬高测量） |
| | `target_kind` | `rotate_loop_min_event_slope_ns_per_iter` 等口径标记 |
| | `valid` | 1 = 通过 witness+频率+spread+数值正确性四道门 |
| | `quality`, `target_spread` | 反复测量的离散度，`spread` 大于 0.2 视为双稳区 |
| **地址表基本量（8 列）** | `f_elements`, `f_L` | 元素数；distinct 128 B line 数 |
| | `f_line_elems_max/min` | 单条 128 B line 内最多/最少元素数（line 占用密度上下界） |
| | `f_line_density` | 每 line 平均元素数 |
| | `f_span_bytes` | 首字节到末字节的连续字节跨度 |
| | `f_page_cross`, `f_cross` | 跨 4 KB 页数；跨页标志 |
| **warp 行为（4 列）** | `f_mean_warp_lines` / `f_min_warp_lines` / `f_max_warp_lines` | 每个 warp 平均/最少/最多触碰的 distinct line 数（warp 内发散度） |
| | `f_iters_per_thread` | 每线程循环次数（元素多于线程数时 >1） |
| **热点/冲突（4 列）** | `f_bucket_pairs_2k8` / `f_bucket_worst_2k8` | 2 KB / 8-bucket 粒度：bucket 内命令冲突对数、最拥塞 bucket 的命令数 |
| | `f_bucket_pairs_32k16` / `f_bucket_worst_32k16` | 同上，32 KB / 16-bucket 大尺度 |
| **命令与事务（8 列）** | `f_K`, `f_K_stg` | 指令条数（LDG / STG，每 32 lane 一条） |
| | `f_dup` | 重复请求的 line 数（warp 欠填导致同一 line 被重复要） |
| | `f_n_cmd_block` | per-element BIU 命令数（SIMD gather 下 = 元素数） |
| | `f_line_request_size`, `f_line_request_size_cpp4` | 每条 128 B line 发起的请求字节量（真实位宽 / C++ 语义硬编码 4 B 每元素） |
| | `f_tail_elems` | `max(0, elements-64)` 尾部余量 |
| | `f_rank` | rank 的数值副本（供拟合使用） |
| **分段/阈值指示（8 列）** | `f_min_L_32`, `f_max_L_minus_32` | `min(L,32)` 与 `max(0, L-32)`：line 数 32 条前后的两段线性区 |
| | `f_dupL_minL_128` | `dup × L × min(L,128)`：warp 欠填 × line 数 |
| | `f_cross_minL_32` | 跨页标志 × `min(L,32)` |
| | `f_lowW_4_L` | `max(0, 4-W) × L`：低 warp 数惩罚 |
| | `f_inner_stride_bytes`, `f_row_gap_bytes` | 内层跨步字节、行间空隙字节 |
| **2D 专有（9 列）** | `f_rows`, `f_cols` | R、C |
| | `f_row_span_bytes`, `f_row_pitch_bytes` | 一行占的字节跨度、行间距（pitch） |
| | `f_row_lines`, `f_gap_lines` | 落在行内的 line 数、落在空隙里的 line 数 |
| | `f_comb_lines`, `f_comb_overlap` | 行列 line 的并集大小、重叠量 |
| | `f_rows_share_line`, `f_row_segments`, `f_gap_per_line`, `f_row_span_div_gap`, `f_tail_row_elems` | 相邻行是否共享 line、行被切成几段、每 line 空隙、跨度/空隙比、末行余量 |

### 3.4 谁在用这两个最终数据集

| 数据集 | 使用者 | 用途 |
|---|---|---|
| `dataset_new_only_v5.csv` | `phys2b_lib.get_slots()`（被 [`phys2_final_report.py`](../scripts/phys2_final_report.py) 调用） | 第 2 轮物理拟合的数据入口 |
| | [`write_phys2_store.py`](../scripts/write_phys2_store.py)、[`phys3_write3.py`](../scripts/phys3_write3.py)、[`_write_phys_store.py`](../scripts/_write_phys_store.py) | 写 `PHYS2_*` / `PHYS3_*` 公式文件 |
| | [`p3_verify.py`](../scripts/p3_verify.py)、`phys3_cols.py`、`_p2_head.py`、`_p3_hdr.py`、`_z_hdr.py` | 校验与列清单核对 |
| `dataset_unified_v5.csv` | [`fit_unified6.py`](../scripts/fit_unified6.py)（→ `final/fit_v5/`） | 全量（含 hist_*）对照拟合 |
| | `final/fit_newonly/` | 同一脚本在 new-only 上的对照 |
| | [`summarize_unified6.py`](../scripts/summarize_unified6.py) | 生成 `metrics_summary_unified6.csv` / `worst_cases_unified6.csv` / `FORMULA_SUMMARY.md` |

---

## 4. 模型结果文件

### 4.1 `final/MODEL_*.json`（41 个，统计选型链）

除特别注明外，所有 slot 级模型的方法都是 **加权最小二乘，权重 = 1/T（等价于直接最小化 MAPE）**；
`MAPE` 列是该 slot 的 in-sample MAPE（%），`项数` 是该项式的项数（不含截距时按原文件计），`n` 是拟合点数。
`_overall` 是文件里自带的按 n 加权的总 MAPE。

| 文件 | 结构 | 方法 | 总 MAPE | 数据 / n 摘要 |
|---|---|---|---:|---|
| `MODEL_FINAL.json` | 5 slot（不分 rank） | 每特征乘 dtype 指示（eb），全系数非负 | 17.41 | n=136/796/847/2177/2083；5–6 项 |
| `MODEL_FINAL_V2.json` | 5 slot | 同上 + dtype/num_warps/rank 指示因子 | 7.23 | 181–520 项（严重过参数化） |
| `MODEL_FINAL_V3.json` | 5 slot | 同上 | 4.07 | 295–1058 项（**项数爆炸，已废弃**） |
| `MODEL_FINAL_V4.json` | 10 slot（分 rank） | 每 slot ≤10 项 | 11.68 | 32/133/284/432/291/491/855/1086/962/1120 |
| `MODEL_FINAL_V5.json` | 10 slot | 同 V4（V5 仅重跑确认） | 11.68 | 与 V4 完全相同 |
| `MODEL_FINAL_V6.json` | 10 slot | 同 V4，数据换成含 hist_* 的更大集 | 14.17 | n 涨到 246/750/943/1762/3416 |
| `MODEL_FINAL_V7.json` | 10 slot | **`run_final_pipeline.py` 的产物**：贪心前向+后向消除，≤10 项，可选 `max(a*num_warps, linear)` | 16.31 | 由 new-only 拟合；`simd_load_wide_r1` MAPE 0.74、`simt_store_r2` 25.98 |
| `MODEL_FINAL_10TERM.json` / `MODEL_10TERM_FINAL.json` | 10 slot | 固定 10 项（后者只存 `base`+`coefs`） | 14.79 | — |
| `MODEL_BESTOF.json` | 10 slot | 逐 slot 取各方案最优（无项数约束） | 无 | `simd_load_wide_r1` 0.57 / `simt_store_r2` 20.95 |
| `MODEL_BACKELIM10.json` | 10 slot | 后向消除到 11 项 | 15.24 | — |
| `MODEL_EXHAUST5.json` | 10 slot | 穷举到 6 项 | 16.97 | — |
| `MODEL_TOPK30.json` | 10 slot | 从 top-30 候选里选 | 无 | 2–11 项 |
| `MODEL_RANK_SPLIT.json` | 10 slot | 按 rank 分治，11 项 | 13.06 | — |
| `MODEL_RANKSPLIT_NNLS.json` | 10 slot | **NNLS**（非负最小二乘），31 项 | 无 | — |
| `MODEL_10CROSSED.json` | 10 slot | 每项为「变量 × 变量」交叉乘积，≤10 项 | 16.57 | — |
| `MODEL_10CROSSED_DOMAIN.json` | 10 slot | 同上 + domain 过滤 | 15.58 | — |
| `MODEL_MAXLINEAR.json` | 10 slot | 线性 + `max()` 包裹 | 11.81 | — |
| `MODEL_MAXPLUS.json` / `MODEL_MAXPLUS2.json` | 10 slot | `max()` 结构搜索（2 = 修正版） | 20.09 / 11.68 | — |
| `MODEL_MAXSTRUCT.json` | 7 slot | `max()` 结构，显式记录 `mape_max`/`mape_linear`/`a_ns_per_warp`/`warp_dominated` | 无 | 证明 num_warps≥16 时发射成本与元素数无关 |
| `MODEL_PHYSMAX.json` | 10 slot | 物理式 + `max()` | 11.68 | — |
| `MODEL_PHYS.json` | 5 slot | 纯物理式（每 slot 7 项） | 21.46 | — |
| `MODEL_PHYS_SMALL.json` | 10 slot | 物理式压缩到 4–9 项 | 21.95 | 例：`simd_load_wide_r1` = `66.4+0.6032*L+3.619*maxL32+0.0111*n+10.8*spanperN-1.025*eb` |
| `MODEL_MODSEL.json` | 10 slot | **按公式逐位选型**：dtype 指示 / num_warps 指示 / regime 标志，每式 5–111 项 | 11.29 | `REPORT_v1_modsel.md` 对应模型 |
| `MODEL_REGIME.json` | 10 slot | regime 标志 + 全交叉项（91 项） | 无 | `simd_load_wide_r1` MAPE 0.0（过拟合） |
| `MODEL_REGIME_FINAL.json` | 10 slot | regime 6 标志（`nge256,nlt64,mwlg2,Lgt16,Rgt8,Clt8`），109 项 | 11.27 | — |
| `MODEL_REGIME_V2.json` | 10 slot | regime 14 标志，183 项 | 9.91 | — |
| `MODEL_REGIME_V3.json` | 10 slot | regime 15 标志，17–216 项 | 9.80 | `REPORT_v3_10crossed.md` 对应模型 |
| `MODEL_PRUNED.json` | 10 slot | 剪枝到 12–96 项 | 10.87 | `PRUNE_CURVE.json` |
| `MODEL_COMPACT.json` / `MODEL_COMPACT_12.json` | 5 slot | 每 slot 恰 12 项，权重 1/T，拟合 5686 个 probe-valid 点 | 14.90 | 12-term-per-slot 压缩版 |
| `MODEL_DELIVER.json` | 5 slot | 交付版：>140 项/式，全部基础特征乘 dtype/num_warps/rank 指示因子 | 9.95 | — |
| `MODEL_INDICATOR_FULL.json` | 9 slot | 指示器全展开，73 项 | 无 | — |
| `MODEL_GATHER.json` / `MODEL_GATHER2.json` | 单式 | SIMD load gather 专项 | 11.32 / 5.34 | n=748 |
| `MODEL_DOMAIN.json` | 10 slot | 带 `exclusion_rule` / `coverage` / `mape_kept`，剔除不适应点后重拟 | 18.60（kept） | 每 slot 有 `n` 与 `n_kept` |
| `MODEL_V22.json` / `MODEL_V23.json` | 5 slot | 早期整模型（只存 `mape/n/base/n_terms`） | 无 | `simd_load_wide` 7.05 / `simt_load` 4.998 |

> **`MODEL_FINAL_V7.json` 不是最终交付**：它由统计贪心选出，`simt_store_r2` 仍有 25.98% 误差。
> 最终交付的是 §4.2 的 10 条物理公式。

### 4.2 `final/PHYS*.json`（16 个，物理公式，**最终交付**）

每个文件结构为 `{<slot>: {n, mape, n_terms, formula, terms:[{name, coef, meaning}], ...}, _meta: {...}}`。
方法：**由硬件物理过程直接构造项（搬运字节、BIU 命令数、同块冲突、warp 发射开销），不做 log/指数/幂次，系数用加权最小二乘（权重 1/T）或 NNLS 定出**；
`max()` 结构的式额外做交替分配 / 坐标下降、多随机起点取最优。

| 文件 | slot | MAPE | 留一/5 折 | 项数 | 公式（截断） |
|---|---|---:|---:|---:|---|
| `PHYS3_simd_load_wide_r1.json` | `simd_load_wide_r1` | **1.22** | — | 3 | `T = a + b*span_bytes - c*max(0, span_bytes - K)`，a=65.1033, b=0.00847947, c=0.00478897, K=1536 |
| `PHYS3_simd_load_wide_r2.json` | `simd_load_wide_r2` | **4.48** | — | 7 | 分支 A/B 的行粒度 + 元素粒度组合 |
| `PHYS3_simd_load_gather_r1.json` | `simd_load_gather_r1` | **8.70** | 9.19 | 6 | `T = a + b*n + c*(n*eb) + d*mx64 + e*mx512 + f*sq1024` |
| `PHYS3_simd_load_gather_r2.json` | `simd_load_gather_r2` | **8.44** | 8.70 | 7 | 块粒度 + 元素粒度 + 拥挤度 |
| `PHYS3_simd_store_r1.json` | `simd_store_r1` | **5.33** | 5.55 | 5 | `T = a + b*nbytes + c*mx128 + d*S64 + e*S512` |
| `PHYS3_simd_store_r2.json` | `simd_store_r2` | **8.12** | 8.24 | 4 | `T = a + b*Q4 + c*n*Q4 + d*max(0, n-128)`，Q4 = min(C, 128/ISB) |
| `PHYS2_wide_simtload.json` | `simd_load_wide_r1/r2`, `simt_load_r1/r2` | 1.73 / 5.79 / **4.95** / **6.34** | — | 2/8/7/6 | `simt_load` 用 `max(a*W, b*W*iters*mwl, c + d*min(mwl,8) + ...)` 三段结构；wide 两式已被 PHYS3 取代 |
| `PHYS2_simt_store.json` | `simt_store_r1/r2` | **6.27** / **9.64** | 6.32 / 9.69 | 7 / 8 | `simt_store_r1` = `max(2.18151*W, 14.9242+9.19181*iters, 4.77174*L128*[eb=1]+...)` |
| `PHYS2_gather.json` | `simd_load_gather_r1/r2` | 9.54 / 9.52 | 9.85 / 9.72 | 5 / 8 | 已被 PHYS3 版本取代 |
| `PHYS2_simd_store.json` | `simd_store_r1/r2` | 5.33 / 8.62 | 5.55 / 8.73 | 5 / 4 | r2 已被 PHYS3 版本取代 |
| `PHYS_simd_store.json` | `simd_store_r1/r2` | 9.69 / 25.26 | — | 6 / 5 | 第 1 轮（NNLS，部分系数被归零） |
| `PHYS_simt_store.json` | `simt_store_r1/r2` | 18.95 / 25.24 | — | 8 / 8 | 第 1 轮 SIMT store（第 2 轮降到 6.27 / 9.64） |
| `PHYS_simt_load.json` | `simt_load_r1/r2` | 10.22 / 10.79 | — | 8 / 9 | 第 1 轮 SIMT load |
| `PHYS_simd_load_wide_r2.json` | `simd_load_wide_r2` | 13.10 | — | 8 | 第 1 轮 2D（第 3 轮降到 4.48） |
| `PHYS_simd_load_gather_r2.json` | `simd_load_gather_r2` | 13.18 | — | 8 | 第 1 轮 2D（第 3 轮降到 8.44） |
| `PHYS_mine.json` | `simd_load_wide_r1`, `simd_load_gather_r1` | 4.11 / 20.18 | — | 2 / 7 | 最早的手工物理式（wide 式尚无 K 饱和项） |

**最终 10 条公式 = `PHYS3_simd_load_wide_r1/r2` + `PHYS3_simd_load_gather_r1/r2` + `PHYS3_simd_store_r1/r2` + `PHYS2_wide_simtload`(simt_load_r1/r2) + `PHYS2_simt_store`(simt_store_r1/r2)**，
与 [`../README.md`](../README.md) 的总表逐行 MAPE 完全对齐（1.22 / 8.70 / 5.33 / 4.95 / 6.27 / 4.48 / 8.44 / 8.12 / 6.34 / 9.64）。

写出这些文件的脚本：

| 文件 | 写出脚本 |
|---|---|
| `PHYS3_simd_load_wide_r1.json` | [`_p3_w1_write.py`](../scripts/_p3_w1_write.py)、[`_w1.py`](../scripts/_w1.py)、[`_fixrate.py`](../scripts/_fixrate.py)、[`_clean3.py`](../scripts/_clean3.py) |
| `PHYS3_simd_load_gather_r1.json` | [`_p3_g1_write.py`](../scripts/_p3_g1_write.py)、[`_p3_g1_patch.py`](../scripts/_p3_g1_patch.py)、[`_p3_g1_verify.py`](../scripts/_p3_g1_verify.py) |
| `PHYS3_simd_store_r1/r2.json` | [`p3_store_r1_write.py`](../scripts/p3_store_r1_write.py)、[`phys3_write3.py`](../scripts/phys3_write3.py)、[`write_phys2_store.py`](../scripts/write_phys2_store.py) |
| `PHYS2_wide_simtload.json` | [`phys2_final_report.py`](../scripts/phys2_final_report.py) |
| `PHYS2_simt_store.json` | [`p2_r2c_write.py`](../scripts/p2_r2c_write.py)、[`p2_final_write.py`](../scripts/p2_final_write.py) |
| `PHYS2_gather.json` | [`p2_writejson.py`](../scripts/p2_writejson.py)、[`p2_verify2.py`](../scripts/p2_verify2.py) |
| 全部 PHYS 文件的项名中文化 | [`_ren.py`](../scripts/_ren.py) |

### 4.3 其他模型文件

| 文件 | 说明 |
|---|---|
| `final/model_v15_final.json` | 全 231 特征整模型，5 slot，MAPE 9.50/22.86/16.19/9.95/15.83 |
| `final/model_v16_backward.json` | 后向消除版（204–223 项） |
| `final/model_v17_dedup_unc.json` | 特征去重 + 不确定性版（157 项） |
| `final/model_v9.json` | 最早整模型（4–7 项，MAPE 16.7–34.2） |
| `final/fit_v5/model_unified6.json` | `version=unified_v1`，`branches` 结构（按 slot 存公式），全量 unified 拟合 |
| `final/fit_newonly/model_unified6.json` | 同上，new-only 拟合 |
| `unified_formulas_v1/model_unified6.json` | `formula_count=7`（含 template 两式），数据集 3638 行 |
| `unified_formulas_v2/model_unified6.json` | `formula_count=5`（`--exclude-template`），数据集 2935 行 |
| `model_store_v1/model_store_v1.json` | store 专项 v1（744 行数据集） |
| `model_template_stride_v1/model_template_stride_{load,store}_v1.json` | 模板路径 v1（190 行） |
| `model_template_stride_v2/model_template_stride_{load,store}_v2.json` | 模板路径 v2（740 行） |
| `model_v4_simple/model_v4_simple.json`、`model_v4_simple_p25/model_v4_simple.json`、`model_v7_simple/model_v7_simple.json` | 1D load 的历史模型（含 LOBO/LOWO 交叉验证折） |
| `refit_no_stride1_v1/load/model_v4_simple_refit.json`、`.../model_v7_simple_refit.json`、`.../store/model_store_v1_refit.json` | 丢掉 stride==1 后的重拟版 |

---

## 5. 报告：`final/REPORT_*.md`

| 文件 | 大小 | 版本 | 内容 | 生成脚本 |
|---|---:|---|---|---|
| `REPORT_v1_modsel.md` | 85 KB | **版本 1：MODSEL** | 每个公式在**候选池**里逐位选型（dtype 指示、num_warps 指示、regime 标志），项数 6–113，附「换成哪个公式收益多少」的对照 | [`gen_reports.py`](../scripts/gen_reports.py) |
| `REPORT_v2_regime.md` | 175 KB | **版本 2：REGIME** | 每个式子用**全部 14 个 regime 标志 + 4 个 dtype 指示 + 7 个 num_warps 指示**共 26 个指示变量乘基础项，项数爆炸但 in-sample 最小 | [`gen_reports.py`](../scripts/gen_reports.py) |
| `REPORT_v3_10crossed.md` | 20 KB | **版本 3：10 CROSSED** | 每式 ≤10 项，且每项是「变量 × 变量」的乘积，用贪心挑选 | [`gen_reports.py`](../scripts/gen_reports.py) |
| `REPORT_v4_physics.md` | 35 KB | **版本 4：PHYSICS** | 每个式由物理过程重建（搬运字节数、BIU 命令数、同类型冲突、warp 发射开销）；不用 max/min 乘积和复杂项，不用 log/指数/幂次；加权最小二乘（权重 1/T） | [`gen_report_phys.py`](../scripts/gen_report_phys.py)（合并全部 `PHYS_*.json`） |
| `REPORT_v5_physics_scope.md` | 57 KB | **版本 5：PHYSICS + 真实算子范围** | 在 v4 基础上把每个式**限定在真实 AI 算子的取值范围内**；剔除「真实算子不会用」或「不稳定」的点 | [`gen_report_v5.py`](../scripts/gen_report_v5.py)（按 `PHYS3 > PHYS2 > PHYS` 优先级合并 + `COMPOSITION.json`） |
| `REPORT_v5_physics_scope_concise_des.md` | 35 KB | **版本 5（精简中文版）：一页读懂模型** | 每个式的物理直构（含搬运字节、BIU、边界接口单元的乘积）、派生量定义、公式与各项物理意义、误差；只保留最终留存集，不讲剔除故事 | [`gen_report_concise.py`](../scripts/gen_report_concise.py) |
| `REPORT_PHYS2_simt_store.md` | 4.7 KB | **第 2 轮：simt_store_r1/r2 单独一轮回归** | 数据 `results/final/dataset_new_only_v5.csv`（valid=1），结果 `results/final/PHYS2_simt_store.json` | 第 2 轮 p2_r2c 系列脚本（脚本里没有硬编码该文件名，由子代理直接写出） |

报告与 README 的关系：`pre-explore/README.md`（10 条公式总表 + PASS/FAIL 判定）是这些报告的**摘要**，其数据源是 `pre-explore/README_data.json`，由 [`_readme.py`](../scripts/_readme.py)/[`_readme2.py`](../scripts/_readme2.py)/[`_readme3.py`](../scripts/_readme3.py)/[`_readme4.py`](../scripts/_readme4.py) 生成，正文里引用 `REPORT_v5_physics_scope_concise_des.md`。

另有 `unified_formulas_v1/FORMULA_SUMMARY.md`(28 KB) 与 `unified_formulas_v2/FORMULA_SUMMARY.md`(23 KB)，是 [`summarize_unified6.py`](../scripts/summarize_unified6.py) 生成的**统一 6 式**逐公式说明（含每个特征的中文释义、量纲、物理含义），是理解 `f_*` 特征最直接的文档。

---

## 6. 血缘速查表

| 产物 | 由谁生成 | 由谁消费 |
|---|---|---|
| `<campaign>/raw/*.json` | 板上探针 `scripts/*_probe.py`（`board_*.sh` 驱动） | `build_*.py` |
| `<campaign>/dataset.csv`、`dataset_2d.csv` | `build_dtype_dataset.py` / `build_2d_dataset.py` | `build_unified_dataset.py`、各 `fit_*.py` |
| `model_store_v1/dataset.csv` | `build_store_dataset.py` | `fit_strided_store_v1.py`（→ `hist_store` 进 unified） |
| `model_template_stride_v2/dataset.csv` | `build_template_stride_dataset.py` | `fit_template_stride_v2.py`（**不进最终 10 式**，被 `--exclude-template` 排除） |
| `_build/build_c{1,2}/dataset_2d.csv` | `run_final_pipeline.py` → `build_2d_dataset.py` | `build_unified_dataset.py` |
| **`final/dataset_unified_v5.csv`** | `build_unified_dataset.py` | `fit_unified6.py`（→ `final/fit_v5/`） |
| **`final/dataset_new_only_v5.csv`** | `run_final_pipeline.py`（从 unified_v5 过滤 source） | 全部物理拟合脚本（`phys2b_lib` / `phys3_write3.py` / `write_phys2_store.py` / `p3_verify.py` …） |
| `final/MODEL_*.json` | `fit_v9..v21`、`fit_final_v4/v5/v6`、`fit_10crossed*`、`fit_regime*`、`fit_max*`、`fit_modsel`、`fit_prune`、`run_final_pipeline.py` | `gen_reports.py` / `gen_report_phys.py` / `gen_report_v5.py` |
| **`final/PHYS3_*.json` / `final/PHYS2_*.json`** | `_p3_w1_write.py` / `_p3_g1_write.py` / `p3_store_r1_write.py` / `phys2_final_report.py` / `p2_r2c_write.py` | `gen_report_v5.py`（按 PHYS3>PHYS2>PHYS 合并） |
| `final/REPORT_v*.md` | `gen_reports.py` / `gen_report_phys.py` / `gen_report_v5.py` / `gen_report_concise.py` | `_readme*.py` → `pre-explore/README.md` + `README_data.json` |

---

## 7. 目录里没有的东西（已删除，不要再找）

- `.npubin` 编译产物（约 17.7 MB）
- `.pkl` / `.npy` 特征缓存（约 8.3 MB）
- `.pyc` / `__pycache__`
- 嵌套的 `third_party/` 副本

另外：`results/final/` 里 `_` 开头的 44 个文件（`_p2c_*`、`_p3_*`、`_simt_*`、`_phys3_*`）是子代理推导过程的中间状态，**不是交付物**，但保留了完整可复现的搜索轨迹；
推导过程记录另见 [`../docs/agent-logs/`](../docs/agent-logs/)（12 个 `p3_*.txt`）。
