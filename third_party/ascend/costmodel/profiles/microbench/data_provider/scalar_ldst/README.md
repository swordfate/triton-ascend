# Scalar Load/Store StageKind：端到端实现与验证指南

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
- load 地址如果依赖另一个 scalar load，则计入 `indirectScalarLoadCount`；
- 非 shaped `tt.store` 计入 `scalarStoreCount`；
- store 地址如果依赖 scalar load，则计入 `indirectScalarStoreCount`。

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
    + indirectScalarLoadCount * scalarIndirectDependencyLatency

scalarMemory +=
    scalarStoreCount / scalarStoreThroughput
    + scalarStoreLatency
    + indirectScalarStoreCount * scalarIndirectDependencyLatency
```

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
export ASCEND_RT_VISIBLE_DEVICES=0
./simd_scalar_gm_memory_host
./simt_scalar_gm_memory_host
```

#### SIMD 标量访存探针（cheap-hash random-64B）

实测输出：

```text
SIMD MainScalar scalar GM memory probe
ops,load_cycles,load_scalar_instr_per_cycle,store_cycles,store_scalar_instr_per_cycle
1,104.866943,0.009536,21.190592,0.047191
2,193.928711,0.010313,39.886719,0.050142
4,371.850098,0.010757,79.782878,0.050136
8,738.060628,0.010839,159.674072,0.050102
```

**SIMD scalar load 填入 JSON**：对 `load_cycles ≈ intercept + slope*ops` 做有截距线性拟合，得到：

```json
"simd.scalar_gm.load.throughput": 0.01104935915300334,
"simd.scalar_gm.load.direct_latency": 12.790396391
```

#### SIMT 标量访存探针（cheap-hash random-64B）

- per-warp uniform；
- 地址由 cheap hash 直接算出，不在计时循环里读 slot table；
- `ops` 只测 1、2、4、8。

实测输出：

```text
SIMT uniform random-spaced scalar GM memory probe
== per-warp uniform random (same address across lanes) ==
warps,ops,load_cycles,load_warp_instr_per_cycle,store_cycles,store_warp_instr_per_cycle
1,1,173.342773,0.005769,63.655518,0.015710
1,2,316.322266,0.006323,87.291178,0.022912
1,4,601.640381,0.006648,155.776855,0.025678
1,8,1171.205892,0.006831,264.277913,0.030271
2,1,174.088623,0.011488,65.457520,0.030554
2,2,317.762858,0.012588,88.487386,0.045204
2,4,604.132080,0.013242,156.981283,0.050961
2,8,1176.513346,0.013600,266.070312,0.060134
4,1,175.353190,0.022811,69.085286,0.057899
4,2,319.990072,0.025001,89.691813,0.089194
4,4,608.904785,0.026277,159.394287,0.100380
4,8,1188.355713,0.026928,269.716309,0.118643
8,1,179.493490,0.044570,76.960531,0.103949
8,2,324.679199,0.049279,97.565999,0.163992
8,4,618.665690,0.051724,168.353109,0.190077
8,8,1206.100749,0.053064,282.167969,0.226815
16,1,191.816325,0.083413,110.289551,0.145073
16,2,335.764486,0.095305,116.830892,0.273900
16,4,635.876139,0.100649,222.042562,0.288233
16,8,1236.040365,0.103556,341.773356,0.374517
32,1,239.337972,0.133702,188.275635,0.169964
32,2,359.071533,0.178237,212.399495,0.301319
32,4,678.888590,0.188543,354.471354,0.361101
32,8,1298.982340,0.197077,581.064860,0.440570
```

**SIMT scalar load/store 填入 JSON**：对 `cycles = a + b*warps + c*ops + d*warps*ops` 做线性拟合，得到：

```json
"load_cycles_fit": [25.905301708, 1.244144672, 143.099973692, 0.336592091],
"store_cycles_fit": [27.479909884, 2.694579448, 25.260166838, 0.927104867]
```

#### CCE 基本对比（SIMD vs SIMT，cheap-hash random-64B）

说明：SIMD 是单 scalar pipe 的 cycles；SIMT 是 per-warp uniform 的 aggregate cycles，`warps` 表示同时跑的 warp 数。

| ops | SIMD load cycles | SIMT load cycles (warps=1) | SIMT load cycles (warps=4) |
|---:|---:|---:|---:|
| 1 | 104.867 | 173.343 | 175.353 |
| 2 | 193.929 | 316.322 | 319.990 |
| 4 | 371.850 | 601.640 | 608.905 |
| 8 | 738.061 | 1171.206 | 1188.356 |

观察：`warps=1` 和 `warps=4` 的 aggregate load cycles 很接近；这里不列 store，因为 SIMD scalar store 以 Triton MTE3 marginal calibration 为准，CCE store 数据不代表最终 SIMD store 成本。

### 3.3 跑 SIMD/SIMT dependency probe（cheap-hash random-64B）

```bash
./simd_scalar_gm_dep_host
./simt_scalar_gm_dep_host
```

> 修正说明：早期版本 independent 模式把 host 预填的 int pattern 当 float 读，denormal 浮点可能让 independent 变慢，从而出现“dependent 比 independent 还快”的假象。现在 independent 和 dependent 都按 `int` 读取同一份 pattern，比较口径一致。

SIMD dep 实测输出：

```text
SIMD MainScalar scalar GM dependency probe
ops,independent_cycles,dependent_cycles,extra_per_edge_cycles
1,102.454590,103.049561,0.594971
2,191.682048,193.765544,1.041748
4,369.895101,375.791016,1.473979
8,726.197428,739.270996,1.634196
```

SIMT dep 实测输出（节选）：

```text
SIMT uniform scalar GM dependency probe
warps,ops,independent_cycles,dependent_cycles,extra_per_edge_cycles
1,1,149.686361,149.650146,-0.036214
1,2,269.781250,269.588623,-0.096313
1,4,426.936523,524.784587,24.462016
1,8,543.735921,1001.719401,57.247935
32,1,194.498128,194.700521,0.202393
32,2,304.786947,302.849528,-0.968709
32,4,476.713542,573.535807,24.205566
32,8,670.808350,1089.130127,52.290222
```

#### CCE dependency 基本对比（SIMD vs SIMT）

单位：cycles per dependency edge；`edge = (dependent - independent) / (ops - 1)`，`ops=1` 没有依赖链，不参与统计。

| ops | SIMD edge cycles | SIMT F1 edge cycles (warps=1) | SIMT F1 edge cycles (warps=4) | SIMT F1 edge cycles (warps=32) |
|---:|---:|---:|---:|---:|
| 4 | 1.965 | 32.616 | 32.048 | 32.274 |
| 8 | 1.868 | 65.426 | 65.378 | 59.760 |

观察：

- SIMD 真实 dependency edge 约 `1.9 cycles/edge`；
- SIMT 在 `ops=4/8` 下给出约 `32 ~ 65 cycles/edge`，明显高于 SIMD；
- `ops=1` 没有 dependency edge，`ops=2` 只有一条边、fixed overhead 占比过高，二者不参与 edge latency 统计。
- `ops=8` 的 per-edge 值约为 `60 ~ 65 cycles/edge`，当前 JSON 使用 `65.4 cycles/edge` 作为 SIMT representative。

JSON 已按代表性值更新：

```text
simd.indirect_dependency_latency_system_cycles = 1.95
simt.indirect_dependency_latency_system_cycles = 65.4
```

### 3.4 跑 Triton SIMD scalar store marginal calibration

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

**SIMD scalar store 填入 JSON 的方法**：Triton SIMD scalar store 实际走 MTE3 路径，不能用 CCE 直接 scalar store 代替。从 `calibrate_triton_simd_scalar_store.py` 得到 marginal store cycles 后：

```json
"simd.scalar_gm.store.throughput": 0.0125
```

---

## 4. Demo 验证

### 4.1 怎么跑

真机 marginal 测量使用：

```bash
cd .../scalar_ldst/demo
source ~/env_ascend.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/tmp/triton_cache_scalar_ldst

for ops in 1 2 4; do
  python3 demo_scalar_bench.py --ops $ops --grids 2048 4096 8192 16384 --reps 10
done

# uniform dependency load / load+store
for ops in 1 2 4; do
  python3 demo_scalar_dep_bench.py --ops $ops --grids 2048 4096 8192 16384 --reps 20
done
```

cost model 测量使用：

```bash
cd .../scalar_ldst
for ops in 1 2 4; do
  python3 check_scalar_ldst_split_ratios.py --grid 2048 --ops $ops --model-only
  python3 check_scalar_dep_costmodel_ratios.py --grid 2048 --ops $ops
done
```

说明：

- 实测单位为 marginal cycles/op；Triton demo 使用 host 侧随机 offset，作为真机对照；CCE 标定使用 cheap-hash random-64B；
- `F1/F4` 指 SIMT `superblock_factor`；

### 4.2 总表（Triton random 读，grid=2048, num_warps=1）

> 本文所有场景默认都是 **per-warp uniform**：同一 warp 内所有 lane 使用同一个地址；除非特别说明。

| 场景 | ops | 实测 SIMD | 实测 SIMT F1 | 实测 SIMT F4 | 实测 SIMD/SIMT F1 | 实测 SIMD/SIMT F4 | costmodel SIMD/SIMT F1 | costmodel SIMD/SIMT F4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform load-only | 1 | 139.671 | 134.799 | 24.929 | 1.04 | 5.60 | 0.603 | 3.698 |
| uniform load-only | 2 | 53.061 | 84.630 | 25.362 | 0.63 | 2.09 | 0.619 | 4.453 |
| uniform load-only | 4 | 35.894 | 80.268 | 19.805 | 0.45 | 1.81 | 0.630 | 5.449 |
| uniform store-only | 1 | 75.514 | 40.517 | 12.405 | 1.86 | 6.09 | 0.779 | 3.767 |
| uniform store-only | 2 | 60.634 | 34.094 | 12.399 | 1.78 | 4.89 | 0.998 | 5.046 |
| uniform store-only | 4 | 61.323 | 34.569 | 11.963 | 1.77 | 5.13 | 1.323 | 7.295 |
| uniform load+store | 1 | 51.845 | 67.079 | 19.566 | 0.77 | 2.65 | 0.714 | 4.654 |
| uniform load+store | 2 | 39.130 | 54.608 | 16.324 | 0.72 | 2.40 | 0.805 | 6.087 |
| uniform load+store | 4 | 51.973 | 52.781 | 16.174 | 0.98 | 3.21 | 0.886 | 7.901 |
| uniform dep-load | 1 | 38.076 | 126.397 | 30.301 | 0.30 | 1.26 | 0.541 | 3.654 |
| uniform dep-load | 2 | 70.230 | 119.901 | 32.533 | 0.59 | 2.16 | 0.518 | 3.589 |
| uniform dep-load | 4 | 57.729 | 110.449 | 30.574 | 0.52 | 1.89 | 0.494 | 3.581 |
| uniform dep-load+store | 1 | 163.459 | 139.878 | 30.424 | 1.17 | 5.37 | 0.565 | 3.661 |
| uniform dep-load+store | 2 | 123.345 | 122.211 | 32.136 | 1.01 | 3.84 | 0.555 | 3.648 |
| uniform dep-load+store | 4 | 100.605 | 117.115 | 33.368 | 0.86 | 3.02 | 0.547 | 3.637 |

**load+store 的 load/store 次数比例**：

- `scalar_ldst_kernel` 在 `ldst` 场景下使用 `N_LD=ops`、`N_ST=ops`，所以 timed region 内 **load : store = 1 : 1**；
- 每个 program 额外还有一个 `tl.store(out + pid, s)` 的 final store，但它不在 marginal 差分的 per-op 分母里；
- 实测的 `cycles/op` 用 `(cycles_N - cycles_0) / (2*ops)` 归一化，因此某一格表示“**平均一次 scalar load + 一次 scalar store**”的成本。

**uniform dependency 场景**：

- `uniform dep-load`：Triton kernel 用 `cur = load(next + cur)` 走随机置换链，每个 iteration 一次 uniform dependent scalar load；地址仍是 random（precomputed random permutation），不是顺序；
- `uniform dep-load+store`：在同一依赖链上再执行 `store(data + cur, ...)`，即 dependent scalar load + dependent scalar store；
- 实测使用 `demo/demo_scalar_dep_bench.py`，cost model 使用 `check_scalar_dep_costmodel_ratios.py`；
- 使用 4 个 grid 和 `reps=20` 重新测量后，dep 场景所有点均为正值；
- F1 方向与实测一致：dep-load 下 SIMD 比 SIMT F1 快；dep-load+store 在 ops>=2 后 SIMT F1 略快；
- F4 的实测收益和 costmodel 预测仍有偏差，尤其 dep-load 的实测 SIMD/F4 比值明显低于模型。

### 4.3 结论

- uniform load-only：random 读下实测 SIMD 仍比 SIMT F1 快；costmodel 在 ops>=2 后也开始认为 SIMD 更快，但 F4 仍高估 SIMT F4。
- uniform store-only / load+store：random 读下 SIMT F4 优势明显，costmodel 方向基本正确但幅度仍偏低。
- uniform dependency load/load+store：F1 方向与实测一致；增加 `reps` 和 grid 点后 dep 场景不再出现无效点；但 F4 的实测收益和 costmodel 预测仍有偏差。
- 总体：当前 uniform direct load/store/load+store 能建模方向，但绝对比值还不能精确复现实测；uniform dependency 场景已加入 4.2 总表并单独标记。
## 5. 相关脚本

```text
calibrate_triton_simd_scalar_store.py
check_pure_scalar_costmodel_ratios.py
check_scalar_ldst_split_ratios.py   # 支持 --model-only
check_scalar_dep_costmodel_ratios.py
demo/demo_scalar_bench.py
demo/demo_scalar_dep_bench.py
simt_scalar_gm_memory.cce            # cheap-hash random-64B 版本
simt_scalar_gm_memory_host.cpp       # ops 最多到 8
```

---

