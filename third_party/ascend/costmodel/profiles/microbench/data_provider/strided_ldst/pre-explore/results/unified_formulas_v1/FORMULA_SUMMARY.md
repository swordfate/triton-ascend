# 统一模型：逐公式汇总（阶段 S）

数据集 `results/unified_formulas_v1/dataset_unified.csv`，共 **3164 行**（rank-1 3036 行 + rank-2 128 行），全部为 512 MB 旋转窗口的 rotate-loop 斜率口径。

## 0. 点数在「槽位 x 位宽」上的分布

| slot | n | int8 | float16 | float32 | int64 |
|---|---:|---:|---:|---:|---:|
| simd_load_gather | 155 | 35 | 33 | 52 | 35 |
| simd_load_wide | 29 | 5 | 5 | 14 | 5 |
| simd_store | 228 | 40 | 36 | 112 | 40 |
| simt_load | 1343 | 43 | 27 | 1256 | 17 |
| simt_store | 706 | 18 | 31 | 640 | 17 |
| template_load | 347 | 7 | 6 | 328 | 6 |
| template_store | 356 | 7 | 6 | 337 | 6 |

## simd_load_gather

覆盖：op=`load`，path=`simd`，n=155，alpha=1.0

### 6.3.1 公式与逐项解释

```text
T[simd_load_gather](ns) = 85.575182 + 0.571502*n_cmd_block + 0.012063*bucket_pairs_2k8 + 0.631041*bucket_worst_2k8 + 7.006848*page_cross + 8.943422*line_density
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 85.575182 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `n_cmd_block` | 0.571502 | 条 | per-element 命令数（= 元素数） | SIMD gather 每元素一条 4B BIU command |
| `bucket_pairs_2k8` | 0.012063 | 对 | 2KB/8-bucket 内的命令冲突对数 | 同一 bucket 内的命令互相仲裁排队 |
| `bucket_worst_2k8` | 0.631041 | 条 | 最拥塞 2KB bucket 的额外命令数 | 决定尾部 drain |
| `page_cross` | 7.006848 | 页 | 跨 4KB 页数 | 每跨一页多一次 page 级开销 |
| `line_density` | 8.943422 | 个/line | elements / L：平均每条 128B line 装几个元素 | 密度越高，同样元素数需要的 line 事务越少 —— 这是把元素位宽吸收进公式的关键量 |

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 139 点, rank2: 16 点
- **数据类型 / 位宽**：float16 (2B): 33 点, float32 (4B): 52 点, int64 (8B): 35 点, int8 (1B): 35 点
- **元素量 elements 的取值**：6 个不同值（16、32、64、128、256、512）
- **内轴 stride（字节）取值**：[3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 64, 128, 256, 512]
- **num_warps 取值**：[1]
- **乘积**：2 个 rank x 4 个位宽 x 6 个元素量 x 16 个 stride 方向；去重后 **有效点 155**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：v1_1d 1 行, v1_2d 16 行, v2_1d 138 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=155，MAPE **15.56%**，p50 13.48%，p90 26.83%，p95 37.52%，max 80.83%，bias -4.27%，RMSE 24.82%
- **按 dtype 分组**：float16 11.8% (n=33), float32 11.5% (n=52), int64 15.5% (n=35), int8 25.2% (n=35)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 16.1% (n=139), rank2 10.5% (n=16)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| int8 | 1 | 16 | 4 | 1 | 1 | 137.5 | 248.7 | 80.8% |
| int8 | 1 | 16 | 64 | 8 | 1 | 587.6 | 123.5 | 79.0% |
| int8 | 1 | 16 | 3 | 1 | 1 | 139.6 | 248.7 | 78.1% |
| int8 | 1 | 16 | 32 | 4 | 1 | 518.3 | 141.4 | 72.7% |
| int64 | 1 | 32 | 256 | 32 | 1 | 227.1 | 125.6 | 44.7% |
| int64 | 1 | 16 | 512 | 16 | 1 | 199.8 | 112.9 | 43.5% |
| int8 | 1 | 16 | 16 | 2 | 1 | 293.9 | 177.2 | 39.7% |
| int64 | 1 | 64 | 128 | 64 | 1 | 246.2 | 153.4 | 37.7% |

- **误差大的通常是哪些 case**：无明显系统偏置（bias -4.3%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 int8 / rank1 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：本位宽样本量均衡，弱约束不显著

## simd_load_wide

覆盖：op=`load`，path=`simd`，n=29，alpha=2.0

### 6.3.1 公式与逐项解释

```text
T[simd_load_wide](ns) = 65.181945 + 0.002478*elements + 0.092954*inner_stride_bytes + 0.829833*L
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 65.181945 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `elements` | 0.002478 | 个 | 逻辑元素数 R*C 或 block | 搬运量随元素数线性增长 |
| `inner_stride_bytes` | 0.092954 |  | (待补) | (待补) |
| `L` | 0.829833 | line | distinct 128B line 数 | line 数是 MTE/LSU 事务数的下界 |

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 21 点, rank2: 8 点
- **数据类型 / 位宽**：float16 (2B): 5 点, float32 (4B): 14 点, int64 (8B): 5 点, int8 (1B): 5 点
- **元素量 elements 的取值**：6 个不同值（16、32、64、128、256、512）
- **内轴 stride（字节）取值**：[2, 4, 8, 16]
- **num_warps 取值**：[1]
- **乘积**：2 个 rank x 4 个位宽 x 6 个元素量 x 4 个 stride 方向；去重后 **有效点 29**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：v1_2d 8 行, v2_1d 21 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=29，MAPE **4.17%**，p50 2.16%，p90 6.77%，p95 16.92%，max 32.75%，bias -1.45%，RMSE 11.86%
- **按 dtype 分组**：float16 1.6% (n=5), float32 6.5% (n=14), int64 3.4% (n=5), int8 1.0% (n=5)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 2.3% (n=21), rank2 9.1% (n=8)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 2 | 128 | 8 | 10 | 1 | 110.8 | 74.5 | 32.7% |
| float32 | 2 | 512 | 8 | 36 | 1 | 126.8 | 97.1 | 23.5% |
| float32 | 1 | 512 | 8 | 32 | 1 | 87.5 | 93.7 | 7.1% |
| float32 | 2 | 512 | 8 | 32 | 1 | 87.9 | 93.7 | 6.7% |
| int64 | 1 | 256 | 16 | 32 | 1 | 89.1 | 93.9 | 5.3% |
| int64 | 1 | 64 | 16 | 8 | 1 | 76.7 | 73.5 | 4.2% |
| float32 | 1 | 64 | 8 | 4 | 1 | 66.6 | 69.4 | 4.2% |
| float32 | 2 | 64 | 8 | 4 | 1 | 66.8 | 69.4 | 3.9% |

- **误差大的通常是哪些 case**：无明显系统偏置（bias -1.5%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 float32 / rank2 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：float16 只有 5 点、int64 只有 5 点、int8 只有 5 点，系数主要由样本多的位宽决定

## simd_store

覆盖：op=`store`，path=`simd`，n=228，alpha=1.0

### 6.3.1 公式与逐项解释

```text
T[simd_store](ns) = 16.321298 + 0.000000*line_request_size_cpp4 + 0.042517*bucket_pairs_2k8 + 0.415346*bucket_worst_32k16 + 103.344105*line_density
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 16.321298 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `line_request_size_cpp4` | 0.000000 **NNLS 压成 0：本网格不可辨识** | B | 每 128B line 发起的请求量（C++ 语义，硬编码 4B/元素） | 等价于每行命令数 x 4B，已把位宽效应吸收进去 |
| `bucket_pairs_2k8` | 0.042517 | 对 | 2KB/8-bucket 内的命令冲突对数 | 同一 bucket 内的命令互相仲裁排队 |
| `bucket_worst_32k16` | 0.415346 | 条 | 最拥塞 32KB/16-bucket 的额外命令数 | 大尺度热点集中度 |
| `line_density` | 103.344105 | 个/line | elements / L：平均每条 128B line 装几个元素 | 密度越高，同样元素数需要的 line 事务越少 —— 这是把元素位宽吸收进公式的关键量 |

> 不可辨识项：`line_request_size_cpp4` —— 在本网格上与其它项共线或恒定；保留 0 系数以便接入时字段齐备。

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 204 点, rank2: 24 点
- **数据类型 / 位宽**：float16 (2B): 36 点, float32 (4B): 112 点, int64 (8B): 40 点, int8 (1B): 40 点
- **元素量 elements 的取值**：6 个不同值（16、32、64、128、256、512）
- **内轴 stride（字节）取值**：[2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
- **num_warps 取值**：[1]
- **乘积**：2 个 rank x 4 个位宽 x 6 个元素量 x 28 个 stride 方向；去重后 **有效点 228**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：hist_store 81 行, v1_1d 6 行, v1_2d 24 行, v2_1d 117 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=228，MAPE **12.44%**，p50 8.02%，p90 27.48%，p95 42.69%，max 63.44%，bias -3.32%，RMSE 34.67%
- **按 dtype 分组**：float16 11.0% (n=36), float32 8.1% (n=112), int64 25.7% (n=40), int8 12.7% (n=40)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 13.1% (n=204), rank2 6.9% (n=24)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| int64 | 1 | 16 | 512 | 16 | 1 | 347.1 | 126.9 | 63.4% |
| int64 | 1 | 32 | 256 | 32 | 1 | 370.6 | 137.3 | 63.0% |
| int64 | 1 | 32 | 512 | 32 | 1 | 358.2 | 134.6 | 62.4% |
| int64 | 1 | 64 | 256 | 64 | 1 | 402.9 | 155.4 | 61.4% |
| int64 | 1 | 64 | 128 | 64 | 1 | 429.9 | 166.2 | 61.3% |
| int64 | 1 | 64 | 512 | 64 | 1 | 397.3 | 155.4 | 60.9% |
| int64 | 1 | 128 | 512 | 128 | 1 | 446.4 | 186.6 | 58.2% |
| int64 | 1 | 128 | 128 | 128 | 1 | 474.6 | 213.2 | 55.1% |

- **误差大的通常是哪些 case**：无明显系统偏置（bias -3.3%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 int64 / rank1 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：本位宽样本量均衡，弱约束不显著

## simt_load

覆盖：op=`load`，path=`simt_only`，n=1343，alpha=1.0

### 6.3.1 公式与逐项解释

```text
T[simt_load](ns) = 148.872033 + 2.090779*min_L_32 + 0.401291*max_L_minus_32 + 0.006654*dupL_minL_128 + 7.537517*cross_minL_32 + 0.456713*lowW_4_L
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 148.872033 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `min_L_32` | 2.090779 | line | min(L, 32)：前 32 条 line 的线性区 | 每多一条 line 多一次事务 |
| `max_L_minus_32` | 0.401291 | line | max(0, L-32)：超出部分 | 超过 32 条后的边际成本不同 |
| `dupL_minL_128` | 0.006654 | line | dup * L * min(L,128)：warp 欠填 x line 数 | 线程不足时同一 line 被重复请求 |
| `cross_minL_32` | 7.537517 | line | 跨 4KB 页标志 x min(L,32) | 跨页时的额外代价 |
| `lowW_4_L` | 0.456713 | line | max(0, 4-W) * L：低 warp 数惩罚 | warp 太少无法掩盖 line 延迟 |

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 1319 点, rank2: 24 点
- **数据类型 / 位宽**：float16 (2B): 27 点, float32 (4B): 1256 点, int64 (8B): 17 点, int8 (1B): 43 点
- **元素量 elements 的取值**：8 个不同值（4、8、16、32、64、128、256、512）
- **内轴 stride（字节）取值**：[2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
- **num_warps 取值**：[1, 2, 4, 8, 16, 32, 64]
- **乘积**：2 个 rank x 4 个位宽 x 8 个元素量 x 28 个 stride 方向；去重后 **有效点 1343**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：hist_v4 1230 行, v1_1d 5 行, v1_2d 24 行, v2_1d 84 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=1343，MAPE **16.01%**，p50 10.12%，p90 39.24%，p95 61.44%，max 119.72%，bias -5.89%，RMSE 70.13%
- **按 dtype 分组**：float16 14.3% (n=27), float32 16.0% (n=1256), int64 14.7% (n=17), int8 18.4% (n=43)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 15.8% (n=1319), rank2 28.5% (n=24)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 2 | 256 | 20 | 40 | 4 | 209.4 | 460.2 | 119.7% |
| float32 | 2 | 256 | 20 | 40 | 4 | 211.7 | 460.2 | 117.3% |
| int8 | 1 | 256 | 64 | 128 | 1 | 335.3 | 670.9 | 100.1% |
| int64 | 1 | 256 | 32 | 64 | 4 | 236.9 | 469.8 | 98.3% |
| int8 | 1 | 128 | 64 | 64 | 1 | 282.3 | 557.5 | 97.5% |
| float32 | 1 | 8 | 256 | 8 | 2 | 3039.7 | 175.9 | 94.2% |
| float32 | 1 | 32 | 36 | 9 | 4 | 1186.7 | 169.3 | 85.7% |
| float32 | 1 | 32 | 36 | 9 | 2 | 1131.8 | 176.4 | 84.4% |

- **误差大的通常是哪些 case**：整体偏乐观（低估）（bias -5.9%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 float32 / rank2 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：本位宽样本量均衡，弱约束不显著

## simt_store

覆盖：op=`store`，path=`simt_only`，n=706，alpha=1.0

### 6.3.1 公式与逐项解释

```text
T[simt_store](ns) = 16.491616 + 0.484190*K_stg + 1.109628*L + 0.168940*line_request_size + 0.281085*dup
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 16.491616 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `K_stg` | 0.484190 | 条 | SIMT_STG 指令数 | 每 32 lane 一条 STG |
| `L` | 1.109628 | line | distinct 128B line 数 | line 数是 MTE/LSU 事务数的下界 |
| `line_request_size` | 0.168940 | B | 每 128B line 的请求量（真实位宽） | 同上，但用真实元素宽度 |
| `dup` | 0.281085 | 倍 | max(0, 1/E - 1)，E = elements/(32*W) | 线程不足时同一元素被重复搬运 |

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 682 点, rank2: 24 点
- **数据类型 / 位宽**：float16 (2B): 31 点, float32 (4B): 640 点, int64 (8B): 17 点, int8 (1B): 18 点
- **元素量 elements 的取值**：5 个不同值（32、64、128、256、512）
- **内轴 stride（字节）取值**：[2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
- **num_warps 取值**：[1, 2, 4, 8, 16, 32, 64]
- **乘积**：2 个 rank x 4 个位宽 x 5 个元素量 x 26 个 stride 方向；去重后 **有效点 706**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：hist_store 614 行, v1_1d 5 行, v1_2d 24 行, v2_1d 63 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=706，MAPE **15.73%**，p50 11.45%，p90 33.43%，p95 48.00%，max 81.97%，bias -4.82%，RMSE 50.64%
- **按 dtype 分组**：float16 42.8% (n=31), float32 13.2% (n=640), int64 27.0% (n=17), int8 47.6% (n=18)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 14.9% (n=682), rank2 39.6% (n=24)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 2 | 512 | 8 | 32 | 4 | 391.4 | 70.6 | 82.0% |
| float32 | 2 | 512 | 8 | 36 | 4 | 371.3 | 75.0 | 79.8% |
| float32 | 2 | 512 | 12 | 48 | 4 | 356.2 | 84.3 | 76.3% |
| float16 | 1 | 256 | 128 | 256 | 1 | 1217.1 | 304.8 | 75.0% |
| float32 | 2 | 512 | 12 | 52 | 4 | 347.0 | 88.7 | 74.4% |
| float16 | 1 | 256 | 64 | 128 | 1 | 634.8 | 163.1 | 74.3% |
| float16 | 1 | 128 | 128 | 128 | 1 | 597.9 | 160.8 | 73.1% |
| float16 | 1 | 256 | 32 | 64 | 1 | 335.0 | 92.7 | 72.3% |

- **误差大的通常是哪些 case**：无明显系统偏置（bias -4.8%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 float32 / rank2 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：本位宽样本量均衡，弱约束不显著

## template_load

覆盖：op=`load`，path=`template`，n=347，alpha=1.0

### 6.3.1 公式与逐项解释

```text
T[template_load](ns) = 238.893366 + 1.786215*L + 0.280625*bucket_worst_32k16 + 5.946737*mean_warp_lines + 0.000000*tail_elems
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 238.893366 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `L` | 1.786215 | line | distinct 128B line 数 | line 数是 MTE/LSU 事务数的下界 |
| `bucket_worst_32k16` | 0.280625 | 条 | 最拥塞 32KB/16-bucket 的额外命令数 | 大尺度热点集中度 |
| `mean_warp_lines` | 5.946737 | line | 每 warp 平均访问的 distinct line 数 | warp 内发散度 |
| `tail_elems` | 0.000000 **NNLS 压成 0：本网格不可辨识** | 个 | max(0, elements-64) | 超过 64 元素后的边际搬运 |

> 不可辨识项：`tail_elems` —— 在本网格上与其它项共线或恒定；保留 0 系数以便接入时字段齐备。

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 331 点, rank2: 16 点
- **数据类型 / 位宽**：float16 (2B): 6 点, float32 (4B): 328 点, int64 (8B): 6 点, int8 (1B): 7 点
- **元素量 elements 的取值**：10 个不同值（4、8、16、32、64、128、256、512、1024、2048）
- **内轴 stride（字节）取值**：[3, 6, 9, 12, 18, 20, 24, 28, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 160, 192, 252, 260, 320, 384, 508, 516, 640, 768, 1020]
- **num_warps 取值**：[1, 32]
- **乘积**：2 个 rank x 4 个位宽 x 10 个元素量 x 41 个 stride 方向；去重后 **有效点 347**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：hist_template 311 行, v1_1d 18 行, v1_2d 16 行, v2_1d 2 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=347，MAPE **7.67%**，p50 5.78%，p90 17.12%，p95 20.44%，max 36.57%，bias -1.07%，RMSE 10.41%
- **按 dtype 分组**：float16 14.4% (n=6), float32 7.5% (n=328), int64 4.2% (n=6), int8 12.8% (n=7)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 7.6% (n=331), rank2 8.8% (n=16)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 1 | 8 | 768 | 8 | 32 | 477.2 | 302.7 | 36.6% |
| float16 | 1 | 1024 | 18 | 144 | 32 | 600.8 | 812.9 | 35.3% |
| float16 | 1 | 1024 | 6 | 48 | 32 | 468.6 | 623.6 | 33.1% |
| float32 | 1 | 32 | 76 | 19 | 32 | 299.4 | 394.5 | 31.8% |
| float32 | 1 | 32 | 80 | 20 | 32 | 306.8 | 402.3 | 31.1% |
| float32 | 1 | 16 | 1020 | 16 | 32 | 516.2 | 366.8 | 28.9% |
| float32 | 1 | 32 | 72 | 18 | 32 | 301.6 | 386.8 | 28.2% |
| float32 | 2 | 512 | 12 | 48 | 32 | 384.0 | 485.9 | 26.5% |

- **误差大的通常是哪些 case**：无明显系统偏置（bias -1.1%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 float32 / rank1 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：float16 只有 6 点、int64 只有 6 点，系数主要由样本多的位宽决定

## template_store

覆盖：op=`store`，path=`template`，n=356，alpha=0.0

### 6.3.1 公式与逐项解释

```text
T[template_store](ns) = 215.461564 + 1.761873*L + 0.036577*bucket_worst_32k16 + 4.309121*mean_warp_lines + 0.274675*tail_elems
```

| 项 | 系数 | 单位 | 物理含义 | 为什么会让时间增加 |
|---|---:|---|---|---|
| `intercept` | 215.461564 | ns | 固定启动/排队成本 | 与 tile、位宽、stride 都无关的冷启动开销 |
| `L` | 1.761873 | line | distinct 128B line 数 | line 数是 MTE/LSU 事务数的下界 |
| `bucket_worst_32k16` | 0.036577 | 条 | 最拥塞 32KB/16-bucket 的额外命令数 | 大尺度热点集中度 |
| `mean_warp_lines` | 4.309121 | line | 每 warp 平均访问的 distinct line 数 | warp 内发散度 |
| `tail_elems` | 0.274675 | 个 | max(0, elements-64) | 超过 64 元素后的边际搬运 |

### 6.3.2 数据规模与构造原则

- **元素的构造方式**：rank-1 为 block 个元素按 stride 等距；rank-2 为 R x C 的 tile（内轴与外轴都可 strided）
- **rank 分布**：rank1: 340 点, rank2: 16 点
- **数据类型 / 位宽**：float16 (2B): 6 点, float32 (4B): 337 点, int64 (8B): 6 点, int8 (1B): 7 点
- **元素量 elements 的取值**：10 个不同值（4、8、16、32、64、128、256、512、1024、2048）
- **内轴 stride（字节）取值**：[3, 6, 9, 12, 18, 20, 24, 28, 36, 40, 44, 48, 52, 56, 60, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 160, 192, 252, 260, 320, 384, 508, 516, 640, 768, 1020]
- **num_warps 取值**：[1, 32]
- **乘积**：2 个 rank x 4 个位宽 x 10 个元素量 x 41 个 stride 方向；去重后 **有效点 356**，无效点 **0**（dataset 只收 valid+correct 的 pass 最小 target）
- **采样配置**：iters 属于 {2000, 8000} 的 Event 线性回归斜率；每 case 2 个 pass（pass1 reps=2/attempts=2，pass2 reps=3/attempts=1）；取全部 valid pass 的最小 target
- **旋转窗口**：512 MB（--rotate-step 8192 --rotate-mask 16383）；四道门（witness 15% / 频率 >=1500 MHz / spread <=0.40 / 正确性）全部通过
- **路径**：`results/unified_formulas_v1/dataset_unified.csv`、`model_unified6.json`、`worst_cases_unified6.csv`
- **数据来源**：hist_template 320 行, v1_1d 18 行, v1_2d 16 行, v2_1d 2 行

### 6.3.3 实测误差与最差 case 归因

- **公式预测误差（in-sample）**：n=356，MAPE **9.22%**，p50 6.11%，p90 19.38%，p95 28.93%，max 59.67%，bias +1.71%，RMSE 9.17%
- **按 dtype 分组**：float16 13.6% (n=6), float32 8.9% (n=337), int64 14.6% (n=6), int8 15.9% (n=7)　← 各位宽误差接近 ⇒ 位宽已被字节域特征吸收
- **按 rank 分组**：rank1 8.2% (n=340), rank2 30.9% (n=16)　← 两个 rank 误差接近 ⇒ 同一公式覆盖 rank-1 与 rank-2
- **最差 case（top 8）**：

| dtype | rank | elements | inner_stride_B | L | W | target | pred | err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 2 | 512 | 20 | 80 | 32 | 325.5 | 519.7 | 59.7% |
| float32 | 2 | 512 | 12 | 48 | 32 | 294.0 | 454.7 | 54.7% |
| float32 | 2 | 512 | 12 | 52 | 32 | 317.8 | 465.5 | 46.5% |
| float32 | 1 | 32 | 124 | 31 | 32 | 280.2 | 404.8 | 44.5% |
| float32 | 2 | 512 | 20 | 84 | 32 | 370.0 | 530.3 | 43.3% |
| float32 | 1 | 32 | 120 | 30 | 32 | 280.0 | 398.7 | 42.4% |
| float32 | 1 | 32 | 116 | 29 | 32 | 278.7 | 392.7 | 40.9% |
| float32 | 2 | 256 | 20 | 40 | 32 | 264.1 | 369.5 | 39.9% |

- **误差大的通常是哪些 case**：无明显系统偏置（bias +1.7%）。归因三类：
  - **(a) 模型缺项**：最差点集中在 float32 / rank2 的特定形态上，现有项未覆盖该形态；
  - **(b) 测量不稳**：核对 `attempts` 的 spread，若明显高于全局中位数则属此类；
  - **(c) 弱约束**：float16 只有 6 点、int64 只有 6 点，系数主要由样本多的位宽决定
