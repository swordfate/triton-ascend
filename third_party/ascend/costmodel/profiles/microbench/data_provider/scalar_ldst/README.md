# Scalar Load/Store StageKind：端到端实现与验证指南

> 2026-09-16 说明：本目录的 cheap-hash 探针与 `a+b*warps+c*ops+d*warps*ops`
> 拟合是 **legacy diff-line / warm-runtime-loop** 口径，用来覆盖“每个 op 换一条
> 64B line、L2 已热身”的随机地址场景。cost model 里的 same-line / CAModel
> diff-line 结构化白盒模型见
> [`../scalar_load_store_whitebox/README.md`](../scalar_load_store_whitebox/README.md)。
> 两条路径都保留：IR 能证明 same-line 时用白盒同 line 公式，证不出时默认走
> 结构化 diff-line；当 profile 没有结构化字段时回退到本目录的 legacy 拟合。

本文档面向新接手的人，按顺序说明：

1. 环境准备；
2. 代码中如何加入 ScalarLoad/ScalarStore StageKind；
3. 如何跑标定脚本并得到 JSON 数据；
4. 如何把数据填入 cost model profile；
5. 如何用 demo 验证 SIMD/SIMT 预测。

---

## 1. 环境信息

### 本地代码路径

```text
/Users/weijianchen/Documents/2026/triton-ascend
```

### 服务器路径

```text
/home/c00946898/triton-ascend
```

### SSH 登录

```bash
ssh ascend-950pr-63
```

实际用户：

```text
c00946898
```

### 激活服务器环境

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
ulimit -n 1048576
```

如果跑 cost model / CAModel，还需要按具体章节设置 `COSTMODEL_LOG_LEVEL` 等。

### 关键目录

```text
third_party/ascend/costmodel/
  include/AscendModel/RouteModel/StageCostModels.h
  include/AscendModel/RouteModel/StageRouteCostModel.h
  lib/AscendModel/Analysis/StagePartitioner.cpp
  lib/AscendModel/RouteModel/StageCostModels.cpp
  lib/AscendModel/RouteModel/SimdSimtCostModel.cpp
  profiles/microbench/ascend_davidv100_v1.json
  profiles/simd_simt/david_v100_simd_simt_v1.json
  profiles/microbench/data_provider/scalar_ldst/
```

---

## 2. 代码修改路径：如何加入 ScalarLoad/ScalarStore StageKind

### 2.1 增加 StageCostModelKind

文件：

```text
third_party/ascend/costmodel/include/AscendModel/RouteModel/StageCostModels.h
```

在枚举中增加：

```cpp
ScalarLoad,
ScalarStore,
```

### 2.2 增加 workload 与 feature

文件：

```text
third_party/ascend/costmodel/include/AscendModel/RouteModel/StageRouteCostModel.h
```

在 `StageWorkload` 中增加：

```cpp
double scalarLoadCount = 0.0;
double directScalarLoadCount = 0.0;
double indirectScalarLoadCount = 0.0;
double scalarStoreCount = 0.0;
double indirectScalarStoreCount = 0.0;
```

在 `StageModelFeatures` 中增加：

```cpp
bool hasScalarLoad = false;
bool hasScalarStore = false;
bool hasScalarIndirectLoad = false;
bool hasScalarIndirectStore = false;
```

### 2.3 Partitioner 识别 scalar load/store

文件：

```text
third_party/ascend/costmodel/lib/AscendModel/Analysis/StagePartitioner.cpp
```

在 workload 累加逻辑中：

- 非 shaped `tt.load` 计入 `scalarLoadCount`；
- load 地址如果依赖另一个 scalar load，则计入 `indirectScalarLoadCount`
  （consumer 侧计数，用于报告/兼容回退）；
- 同一 Stage 内按 producer load 去重得到
  `indirectScalarLoadExposureCount`：fan-out 只记 1，串行链按 edge 记；
- 非 shaped `tt.store` 计入 `scalarStoreCount`；
- store 地址如果依赖 scalar load，则计入 `indirectScalarStoreCount`，
  同理得到 `indirectScalarStoreExposureCount`。

同时在 Stage 分类时，如果 operation tree 只包含 scalar load/store，则返回：

```cpp
StageCostModelKind::ScalarLoad
StageCostModelKind::ScalarStore
```

### 2.4 mapWorkload 公式

文件：

```text
third_party/ascend/costmodel/lib/AscendModel/RouteModel/StageCostModels.cpp
```

在 `mapWorkload()` 中增加 scalar memory 计算：

```text
scalarMemory +=
    scalarLoadCount / scalarLoadThroughput
    + scalarLoadLatency
    + indirectScalarLoadExposureCount * scalarIndirectDependencyLatency

scalarMemory +=
    scalarStoreCount / scalarStoreThroughput
    + scalarStoreLatency
    + indirectScalarStoreExposureCount * scalarIndirectDependencyLatency
```

> `*_exposure_count` 为 0（未分析/手工 workload）时回退到 legacy
> `indirectScalar*Count`，保持旧行为；正常 partitioner 分析出的 Stage 使用
> producer 侧去重计数，fan-out 不会再被重复收费。
>
> 2026-09-17 起，profile 提供结构化 store 字段时，store 的 throughput/latency
> legacy 路径被白盒公式替换：
>
> ```text
> SIMD MainScalar store:
>   U == 1: 30 + 450 + (K-1)*2
>   U >= 2: 30 + 450 + extra(U) + (K-1)*2
>           extra(2)=70, extra(3)=78, extra(U>=4)=85+(U-4)*106
> SIMT warp-uniform store:
>   same-line (K >= 2): 555 + (K-1)*480
>   first store / diff-line: 450 + (K-1)*20
> ```
>
> 字段与语义见 `scalar_load_store_whitebox/README.md` §4.4；缺字段时仍回退上面的
> legacy store fit。

对于 SIMT，`scalarLoadThroughput` 由 `warps × ops` 解析拟合得到：

```text
aggregate_cycles = a + b*warps + c*ops + d*warps*ops
throughput = warps * ops / aggregate_cycles
```

### 2.5 Profile 读取与 JSON 字段

文件：

```text
third_party/ascend/costmodel/lib/AscendModel/RouteModel/SimdSimtCostModel.cpp
```

`readStageResources()` 从 JSON 的 `scalar_memory` 中读取：

- SIMD scalar load/store throughput/latency；
- SIMT scalar load/store throughput/latency；
- SIMT `load_cycles_fit` / `store_cycles_fit`；
- indirect dependency latency。

### 2.6 JSON / schema

需要同步：

```text
third_party/ascend/costmodel/profiles/microbench/ascend_davidv100_v1.json
third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json
third_party/ascend/costmodel/profiles/simd_simt/simd_simt_profile_schema.json
```

---

## 3. 标定流程与完整命令

### 3.1 编译 CCE 探针

在服务器 scalar_ldst 目录：

```bash
cd ~/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst

source ~/env_ascend.sh
INC=/home/c00946898/AscendNPU-IR/bishengir/lib/Template/include
TK="$ASCEND_TOOLKIT_HOME"

for name in simd_scalar_gm_memory simt_scalar_gm_memory simd_scalar_gm_dep simt_scalar_gm_dep; do
  ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 -I"$INC" "$name.cce" -o "$name.o"
  g++ -O2 "${name}_host.cpp" -o "${name}_host" \
      -I"$TK/x86_64-linux/pkg_inc" -I"$TK/include" -L"$TK/lib64" -lruntime -lascendcl
done
```

### 3.2 跑 SIMD/SIMT CCE throughput

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
./simd_scalar_gm_memory_host
./simt_scalar_gm_memory_host
```

实测输出（2026-09-07）：

```text
SIMD MainScalar scalar GM memory probe
ops,load_cycles,load_scalar_instr_per_cycle,store_cycles,store_scalar_instr_per_cycle
1,11.944499,0.083721,8.843343,0.113079
2,19.497640,0.102577,16.462077,0.121491
4,30.563965,0.130873,26.670166,0.149980
8,51.956543,0.153975,48.288574,0.165671
```

```text
SIMT uniform scalar GM memory probe
== per-warp uniform (same address across lanes) ==
warps,ops,load_cycles,load_warp_instr_per_cycle,store_cycles,store_warp_instr_per_cycle
1,1,58.888428,0.016981,53.948079,0.018536
1,2,65.655273,0.030462,40.592936,0.049270
1,4,84.696777,0.047227,44.241536,0.090413
1,8,131.913330,0.060646,59.387858,0.134708
1,16,173.758382,0.092082,73.931071,0.216418
1,32,345.528158,0.092612,106.047852,0.301751
2,1,59.590983,0.033562,54.526611,0.036679
2,2,75.227865,0.053172,40.602458,0.098516
2,4,101.610921,0.078732,45.445964,0.176033
2,8,151.546305,0.105578,66.665609,0.240004
2,16,215.874756,0.148234,74.980387,0.426778
2,32,370.327637,0.172820,120.518555,0.531039
4,1,69.578776,0.057489,55.141276,0.072541
4,2,93.732503,0.085349,40.598307,0.197053
4,4,124.379232,0.128639,45.139893,0.354454
4,8,194.614095,0.164428,62.237305,0.514161
4,16,220.792725,0.289865,104.413737,0.612946
4,32,427.432943,0.299462,276.744141,0.462521
8,1,88.743571,0.090147,58.774414,0.136114
8,2,114.559896,0.139665,43.081624,0.371388
8,4,163.388916,0.195852,53.604655,0.596963
8,8,195.747640,0.326952,97.602946,0.655718
8,16,237.452067,0.539056,216.419596,0.591444
8,32,390.783447,0.655094,720.027018,0.355542
16,1,113.308512,0.141207,77.524984,0.206385
16,2,158.032633,0.202490,70.408122,0.454493
16,4,165.442220,0.386842,101.571859,0.630096
16,8,198.756999,0.644002,182.834310,0.700087
16,16,299.819092,0.853848,402.774740,0.635591
16,32,451.328532,1.134429,981.665934,0.521562
32,1,164.918620,0.194035,135.723633,0.235773
32,2,159.206217,0.401994,118.858154,0.538457
32,4,181.816895,0.704005,193.855794,0.660285
32,8,264.299072,0.968600,359.788330,0.711529
32,16,484.775065,1.056160,802.485758,0.638018
32,32,809.549479,1.264901,1757.795817,0.582548
```

### 3.3 跑 Triton SIMD scalar store marginal calibration

```bash
cd ~/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst

source ~/env_ascend.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=0

python3 calibrate_triton_simd_scalar_store.py --grid 4096 --reps 30
```

脚本输出：

```text
ops,store_marginal_cycles_per_op
...
```

这里的 `marginal cycles` 是指：

```text
(time(额外做 N 个 store) - time(baseline)) / N
```

即排除了固定 launch/最终 store 后，每个额外 scalar store 的平均 cycles。

> 当前 SIMD scalar store 使用该脚本的稳定 N=32 结果约 `1/79.3`。

### 3.4 SIMT `store_throughput_cap=0.08` 的来源

该值不是 marginal microbenchmark 的直接输出，而是从 SIMT store-only 真机总耗时反推的保守上限：

```text
SIMT store-only（grid=2048, 32+1 stores）≈ 0.017 ms

per-program/core cycles ≈ 0.017 * 988900 * 56 / 2048
                        ≈ 459 cycles

每个 program 约 33 次 store：
459 / 33 ≈ 14 cycles/store

等效 throughput ≈ 1 / 14 ≈ 0.071 store/cycle
```

为了留一点余量，取：

```text
store_throughput_cap = 0.08 store/cycle
                    ≈ 12.5 cycles/store
```

它用于限制 SIMT scalar store throughput，避免 CCE 解析拟合把 SIMT F4 的 scalar store 估得太便宜。

### 3.5 从输出填 JSON

#### SIMD scalar load

从 `simd_scalar_gm_memory_host` 的输出做线性回归：

```text
load_cycles ≈ intercept + slope * ops
```

填入：

```text
profiles/microbench/ascend_davidv100_v1.json
```

```json
"simd.scalar_gm.load.throughput": 1 / slope,
"simd.scalar_gm.load.direct_latency": intercept
```

#### SIMD scalar store

Triton SIMD scalar store 实际走 MTE3 路径，不能用 CCE 直接 scalar store 代替。
从 `calibrate_triton_simd_scalar_store.py` 得到 marginal store cycles 后：

```json
"simd.scalar_gm.store.throughput": 1 / store_marginal_cycles_per_op
```

当前值约为：

```json
"simd.scalar_gm.store.throughput": 0.0125
```

#### SIMT scalar load/store

从 `simt_scalar_gm_memory_host` 输出 `warps × ops` 矩阵，对 aggregate cycles 做解析拟合：

```text
aggregate_cycles = a + b*warps + c*ops + d*warps*ops
```

填入：

```text
profiles/simd_simt/david_v100_simd_simt_v1.json
```

当前值：

```json
"load_cycles_fit": [65.523189, 2.048849, 7.893464, 0.374104],
"store_cycles_fit": [24.769761, -1.298297, 1.818695, 1.676496]
```

---

## 4. Demo 验证

### 4.1 实测 SIMD/SIMT 耗时

```bash
cd ~/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst/demo

python3 demo_scalar_scenarios.py --grid 2048 --n 32
```

实测输出（2026-09-07）：

```text
case,mode,sf,median_ms
ldst,simd,1,0.161763
ldst,simt_only,1,0.029605
ldst,simt_only,4,0.020660
ldst,simt_only,16,0.020213
ldst,simt_only,64,0.023915
compute,simd,1,0.061708
compute,simt_only,1,0.040231
compute,simt_only,4,0.021113
compute,simt_only,16,0.020214
compute,simt_only,64,0.019999
control,simd,1,0.062161
control,simt_only,1,0.021653
control,simt_only,4,0.018803
control,simt_only,16,0.020259
control,simt_only,64,0.020330
```

### 4.2 Cost model 预测比值

```bash
cd ~/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst

python3 check_pure_scalar_costmodel_ratios.py
```

实测输出（2026-09-07，含 `store_throughput_cap=0.08`）：

```text
case,effective,selected_sb,all_simd,all_simt_only,ratio_simd_over_simt
ldst,all_simt_only,4,110960.5,8300.5,13.368
compute,all_simt_only,4,8375.4,3232.8,2.591
control,all_simt_only,4,12090.0,5051.6,2.393
```

### 4.3 实测比值 vs cost model 预测比值

统一实验参数：

- `num_warps=1`；
- `grid=2048`；
- `ops` 对 ldst 指每个 program 的 `N_LD/N_ST`；
- `compute/control` 的 `ops` 指循环展开次数 `N`；
- F1/F4 是 `superblock_factor`，不是 warp 数；
- SIMT 实际 cost model 中的 `effectiveWarps = num_warps * superblockFactor`。

预测值来源：

- 4.2/4.3/4.4 的“预测”列是直接运行 `check_pure_scalar_costmodel_ratios.py`，读取 route JSON 中 `all_simd / all_simt_only` cycles 得到的；
- 4.5 的“含 cap”预测列不是直接 route report，而是根据 trace 中 scalarMemory 增量和 wave 数近似折算的，具体见 4.5。

默认配置 `grid=2048, n=32`：

| case | 实测 SIMD/SIMT F1 | 预测 SIMD/SIMT F1 | 实测 SIMD/SIMT F4 | 预测 SIMD/SIMT F4 |
|---|---:|---:|---:|---:|
| ldst | 5.464 | 3.171 | 7.830 | 13.368 |
| compute | 1.534 | 0.805 | 2.923 | 2.591 |
| control | 2.871 | 1.001 | 3.306 | 2.393 |

### 4.4 高/低 grid 与高/低 ops 矩阵

所有行均使用 `num_warps=1`；没有单独列出 warp 列，因为 warp 数固定为 1。预测列为直接跑 cost model route report 得到的 `all_simd / all_simt_only` cycles 比值。

| grid | ops | case | 实测 SIMD/SIMT 比值 (F1) | 预测 SIMD/SIMT 比值 (F1) | 实测 SIMD/SIMT 比值 (F4) | 预测 SIMD/SIMT 比值 (F4) |
|---|---|---|---:|---:|---:|---:|
| 128 | 4 | ldst | 1.455 | 1.457 | 1.299 | 5.962 |
| 128 | 4 | compute | 0.054 | 0.452 | 0.797 | 1.667 |
| 128 | 4 | control | 0.496 | 0.477 | 1.144 | 1.676 |
| 128 | 32 | ldst | 1.287 | 4.403 | 0.821 | 18.077 |
| 128 | 32 | compute | 1.052 | 0.805 | 1.004 | 2.137 |
| 128 | 32 | control | 1.540 | 1.001 | 1.025 | 1.962 |
| 2048 | 4 | ldst | 5.364 | 1.457 | 5.446 | 7.353 |
| 2048 | 4 | compute | 3.326 | 0.452 | 3.268 | 2.056 |
| 2048 | 4 | control | 3.203 | 0.477 | 3.525 | 2.067 |
| 2048 | 32 | ldst | 5.464 | 4.403 | 7.830 | 22.296 |
| 2048 | 32 | compute | 1.534 | 0.805 | 2.923 | 2.635 |
| 2048 | 32 | control | 2.871 | 1.001 | 3.306 | 2.420 |

> 表中所有数值都是 `SIMD 时间 / SIMT 时间` 或 `all_simd cycles / all_simt cycles` 的**比值**，不是时间本身。

> 低 grid 下 fixed overhead 占比高，`compute` 的 F1 实测噪声很大，例如 `128/n4` 的 0.054 属于异常值，不建议用于精确校准。

> 注意：4.4 表中的“预测”列仍来自加入 cap 前；加入 cap 后的默认 `n=32` 数值见 4.3 与 4.5。

说明：

- `ldst`：F1 方向已正确，F4 模型明显高估 SIMT 收益；
- `compute`：F4 接近，F1 仍偏低/方向偏保守；
- `control`：F4 接近，F1 仍偏低；
- cost model 当前都能选出 `all_simt_only`，但数值比值仍需要后续继续校准 scalar compute/control 和 F4 以上 superblock。

### 4.5 load-only / store-only / load+store 拆分对比

以下数据来自 `grid=2048, n=32, num_warps=1` 的真机测量。

预测列说明：

- F1/F4 是 `superblock_factor`；
- 直接跑 cost model 得到的原始预测是加入 cap 前的值；
- “含 cap”预测列是在 `store_throughput_cap=0.08` 下，按 trace 中 scalarMemory 增量与 wave 数近似折算的，不是完整重编后的 route report 数值。

| 场景 | 实测 SIMD/SIMT F1 | 预测 SIMD/SIMT F1（含 cap） | 实测 SIMD/SIMT F4 | 预测 SIMD/SIMT F4（含 cap） |
|---|---:|---:|---:|---:|
| load-only | ~2.3x | 0.70x | ~3.7x | 3.75x |
| store-only | ~9.1x | 4.61x | ~8.9x | 15.67x |
| load+store | ~5.9x | 3.17x | ~8.8x | 13.37x |

预测列说明：

- 这些值是重装包含 `store_throughput_cap=0.08` 的 wheel 后，直接跑 cost model route report 得到的；
- load-only 的 F4 预测为 3.75x；
- store-only / load+store 的 F4 预测为 15.67x / 13.37x；
- 它们已经比 cap 前的 32.52x / 22.30x 明显下降。

---

## 5. 当前状态

- SIMD scalar store 已改用 Triton 标定，修复了 scalar-heavy `ldst` 场景 F1 反向；
- SIMT scalar load/store 使用解析拟合；
- `compute/control` 的幅度仍可能偏低，需要后续单独标定 scalar compute/control；
- 当前 cost model superblock factor 只支持到 F4，尚未覆盖 F8/F16/F32/F64。

---

## 6. 相关脚本

```text
calibrate_triton_simd_scalar_store.py
check_pure_scalar_costmodel_ratios.py
demo/demo_scalar_scenarios.py
```
