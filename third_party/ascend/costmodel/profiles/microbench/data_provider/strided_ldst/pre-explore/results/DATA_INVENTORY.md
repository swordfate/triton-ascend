
## 0. 标定数据总览（全部实测数据的分类、规模与结果路径）

> 本表是**唯一的数据索引**：每一行 = 一组同质实测数据。
> 路径均相对 `third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore/`。
> "有效点"= 通过 witness + 频率 + spread + 数值正确性四道门、并进入数据集的行数。

### 0.1 阶段 0 —— 1D/4B 三条路径（零新增测量，只过滤 stride==1 后重拟）

| # | 路径 | dtype | 维度 | 网格（block × stride × W） | 有效点 | target 口径 | 数据结果路径 |
|---|---|---|---|---|---:|---|---|
| 0-1 | SIMD load wide | f32 | rank1 | block 8–512 × stride 2 | 12 | ns（syscnt） | `results/refit_no_stride1_v1/load/errors_simd_wide.csv` |
| 0-2 | SIMD load gather | f32 | rank1 | block 8–512 × stride 3–256 | 504 | ns | `results/refit_no_stride1_v1/load/errors_simd_gather.csv` |
| 0-3 | **pure SIMT v8 load** | f32 | rank1 | block 4–512 × stride 2–256 × W 1–64 | 1232 | ns/iteration | `results/refit_no_stride1_v1/load/dataset_simt_v8.csv`、`.../errors_simt_v8.csv` |
| 0-4 | **store SIMD gather** | f32 | rank1 | block 32–256 × stride 2–256 | 88 | ns/iteration | `results/refit_no_stride1_v1/store/dataset.csv`、`.../errors_all.csv` |
| 0-5 | **store SIMT** | f32 | rank1 | block 32–256 × stride 2–256 × W 1–64 | 616 | ns/iteration | 同 0-4 |
| 0-6 | 重拟原始来源（未改动） | f32 | rank1 | — | 712 / 1288 / 744 | — | `results/model_v2/dataset.csv`、`results/model_v4_numwarps/dataset.csv`、`results/model_store_v1/dataset.csv` |
| 0-7 | 审计表 / old-vs-new 对照 | — | — | — | — | — | `results/refit_no_stride1_v1/loading_audit.csv`、`.../old_vs_new.csv` |
| — | `pure SIMT v7 load` | — | — | — | **已删除** | — | 被 v8 取代，本轮起不再保留公式 |
| — | `store SIMD wide（stride==1）` | — | — | — | **已删除** | — | stride==1 属连续访问，不属于 strided 模型 |

### 0.2 阶段 0（续）—— template 路径的 1D/4B 基线（v2 标定，**本轮未重跑**）

| # | 路径 | dtype | 维度 | 网格（block × stride × W） | 有效点 | target 口径 | 数据结果路径 |
|---|---|---|---|---|---:|---|---|
| 0-8 | template load | f32 | rank1 | block 4–2048 × 37 个非 2 次幂 stride × W32 | 311 | ns/iteration | `results/model_template_stride_v2/dataset.csv`（`errors_load.csv`） |
| 0-9 | template store | f32 | rank1 | 同上 | 320 | ns/iteration | `results/model_template_stride_v2/dataset.csv`（`errors_store.csv`） |
| 0-10 | template 原始测量（4 JSON，两 pass） | f32 | rank1 | 同上 | 1206/1480 | — | `results/model_template_stride_v2/raw/*.json` |
| 0-11 | template v1（历史，已被 v2 取代） | f32 | rank1 | block 16–2048 × 19 stride × W1 | 190 | ns/iteration | `results/model_template_stride_v1/dataset.csv` |

### 0.3 阶段 A —— 多数据类型（2026-10-10 新增，全部真卡 Event）

| # | 路径 | dtype | 元素字节 | 维度 | 网格（block × stride × W） | 字节间距 | 有效点 | target 口径 | 数据结果路径 |
|---|---|---|---|---|---:|---:|---:|---|---|
| A-1 | SIMD load | int8 | 1 | rank1 | block 64/256 × stride 4/8/12/16/32 × W1 | 4–32B | 7 | ns/iteration | `results/dtype_calibration_v1/raw/load_simd_int8_pass{1,2}.json` |
| A-2 | SIMD load | float16 | 2 | rank1 | block 64 × stride 2/4/6/8/16 × W1 | 4–32B | 5 | ns/iteration | `.../raw/load_simd_float16_pass{1,2}.json` |
| A-3 | SIMD load | float32 | 4 | rank1 | block 64/256 × stride 2/3/4/8 × W1 | 8–32B | 7 | ns/iteration | `.../raw/load_simd_float32_pass{1,2}.json` |
| A-4 | SIMD load | int64 | 8 | rank1 | block 64 × stride 2/4 × W1 | 16–32B | 2 | ns/iteration | `.../raw/load_simd_int64_pass{1,2}.json` |
| A-5 | SIMD store | int8 / float16 / float32 / int64 | 1/2/4/8 | rank1 | 同 A-1..A-4 | 4–32B | 7/5/7/2 | ns/iteration | `.../raw/store_simd_{int8,float16,float32,int64}_pass{1,2}.json` |
| A-6 | pure SIMT load | int8 | 1 | rank1 | block 256 × stride 4/8/16 × W 1/4 | 4–16B | 3 | ns/iteration | `.../raw/load_simt_only_int8_pass{1,2}.json` |
| A-7 | pure SIMT load | int64 | 8 | rank1 | block 256 × stride 2/4 × W 1/4 | 16–32B | 2 | ns/iteration | `.../raw/load_simt_only_int64_pass{1,2}.json` |
| A-8 | pure SIMT store | int8 / int64 | 1/8 | rank1 | 同 A-6/A-7 | 4–32B | 3/2 | ns/iteration | `.../raw/store_simt_only_{int8,int64}_pass{1,2}.json` |
| A-9 | template load | int8 | 1 | rank1 | block 16/256/1024 × stride 3/9 × W32 | 3–9B | 6 | ns/iteration | `.../raw/load_template_int8_pass{1,2}.json` |
| A-10 | template load | float16 | 2 | rank1 | block 16/256/1024 × stride 3/9 × W32 | 6–18B | 6 | ns/iteration | `.../raw/load_template_float16_pass{1,2}.json` |
| A-11 | template load | int64 | 8 | rank1 | block 16/256/1024 × stride 3/9 × W32 | 24–72B | 6 | ns/iteration | `.../raw/load_template_int64_pass{1,2}.json` |
| A-12 | template store | int8 / float16 / int64 | 1/2/8 | rank1 | 同 A-9..A-11 | 3–72B | 6/6/6 | ns/iteration | `.../raw/store_template_{int8,float16,int64}_pass{1,2}.json` |
| A-13 | **合并数据集** | 全 4 种 | 1/2/4/8 | rank1 | — | — | **88** | ns/iteration | `results/dtype_calibration_v1/dataset.csv`、`dataset_meta.json` |
| A-14 | dtype × stride path check | 6 种（含 bf16/int32） | 1/2/4/8 | rank1 | block 64 × 10 stride | — | 80+32 | compile-only | `.../ir_evidence/dtype_stride_path_check_{template,modes}.json` |

### 0.4 阶段 B —— 2D（两轴都 strided）+ R=1 的 rank-2（2026-10-10 新增）

形态记法：`both_sK_tight` = `IS=K, OS=C*K`；`both_sK_gap` = `IS=K, OS=C*K+7`。

| # | 路径 | dtype | 维度 / axis_class | 形状 | 形态 | 有效点 | target 口径 | 数据结果路径 |
|---|---|---|---|---|---:|---|---|---|
| B-1 | SIMD load | f32 | rank2 **both_strided** | 8x16, 16x32 | both_s2/s3/s5 × tight/gap | 12 | ns/iteration | `results/dtype_calibration_v1/raw/2d_load_simd_float32_pass{1,2}.json` |
| B-2 | SIMD load | f32 | rank2 **singleton_row_strided（R=1）** | 1x64, 1x256 | 同上 | 12 | ns/iteration | 同 B-1 |
| B-3 | SIMD store | f32 | rank2 both / R=1 | 8x16,16x32 / 1x64,1x256 | 同上 | 12+12 | ns/iteration | `.../raw/2d_store_simd_float32_pass{1,2}.json` |
| B-4 | pure SIMT load | f32 | rank2 both / R=1 | 同上 | 同上 | 12+12 | ns/iteration | `.../raw/2d_load_simt_only_float32_pass{1,2}.json` |
| B-5 | pure SIMT store | f32 | rank2 both / R=1 | 同上 | 同上 | 12+12 | ns/iteration | `.../raw/2d_store_simt_only_float32_pass{1,2}.json` |
| B-6 | template load | f32 | rank2 both / R=1 | 同上 | **both_s3/s5 × tight/gap**（s2 不触发模板） | 8+8 | ns/iteration | `.../raw/2d_load_template_float32_pass{1,2}.json` |
| B-7 | template store | f32 | rank2 both / R=1 | 同上 | 同上 | 8+8 | ns/iteration | `.../raw/2d_store_template_float32_pass{1,2}.json` |
| B-8 | **合并数据集** | f32 | rank2 | 4 形状 | 6 形态（模板 4 形态） | **128** | ns/iteration | `results/dtype_calibration_v1/dataset_2d.csv`、`dataset_2d_meta.json` |
| B-9 | 2D path check | f32 | rank2 全 5 类 axis_class | 1x64, 8x16 | 10 形态 × 3 mode × 2 op | 114 | compile-only | `.../ir_evidence/stride2d_path_check.json` |

### 0.5 CAModel 白盒证据（1D/4B，历史既有；阶段 B 的 2D 白盒见 §13）

| # | 内容 | dtype | 维度 | 网格 | 结果路径 |
|---|---|---|---|---|---|
| C-1 | SIMD load stride=3 白盒（AIV/MTE2/BIU 解析） | f32 | rank1 | block 32/64/128 × stride 3 | `results/block32_s3/parsed.{json,csv}`、`results/block64_s3/`、`results/block128_s3/` |
| C-2 | store 白盒（MTE3/LSU/DC/BWIF） | f32 | rank1 | block 32/64/128 × stride 1/2/4/8/16/32/64/128/256 × simd/simt | `results/store_camodel_b32/parsed.csv`、`store_camodel_b64/`、`store_camodel_b128/` |
| C-3 | 白盒汇总 | f32 | rank1 | 同上 | `results/strided_load_summary.{csv,md}` |
| C-4 | 板卡原始 Event（v2 SIMD/SIMT） | f32 | rank1 | block/stride/W 扫描 | `results/board_v2_simd/`、`board_v2_simd_more_blocks/`、`board_v2_simt/`、`board_v2_simt_more_blocks/`、`board_v2_simt_warps/` |

### 0.6 汇总：每一轮的规模

| 轮次 | 新增测量 case | valid | correct | 说明 |
|---|---:|---:|---:|---|
| 阶段 0（本轮） | **0** | — | — | 只对已有 dataset.csv 过滤 + 重拟 |
| 阶段 A（本轮） | 402（合并后 88 个唯一 case） | 398 | 402 | 两 pass 重复计入 402；int8 load 因探针 bug 修复后重测一次 |
| 阶段 B（本轮） | 128（唯一 case，无重复 pass 数据点） | 128 | 128 | 每 (form, shape) 1 个 case，两 pass 取 min |
| 阶段 A+B 合计唯一 case | **216** | **216** | **216** | 100% valid |

> 复用的历史数据：`model_v2` / `model_v3_numwarps` / `model_v4_numwarps` / `model_store_v1` /
> `model_template_stride_v1/v2` 及其 `raw/`，以及 §0.5 的 CAModel 证据。**均未重跑。**

---
