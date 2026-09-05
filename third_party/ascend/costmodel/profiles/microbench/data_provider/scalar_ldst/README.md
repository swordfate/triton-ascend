# Scalar Load/Store CostModel 标定与复现

本文档描述 scalar load/store 的 CCE 标定、CAModel 指令周期分析，以及如何把结果填入 cost model JSON。

> 旧版 `wall_summary.csv` / `parse_wall.py` / `parse_camodel_scalar.py` 方法已废弃。
> 它把 wall-time 斜率直接当 throughput，并把 dependent chain 斜率直接当 dependency latency，存在 double counting。
> 当前以本目录下的 CCE 探针和 `data_provider/camodel/README.md` 的 per-unit/span 方法为准。

---

## 0. 如何一步步进入 ScalarLoad / ScalarStore StageKind

1. **增加枚举**
   `include/AscendModel/RouteModel/StageCostModels.h` 的 `StageCostModelKind` 中增加：
   ```cpp
   ScalarLoad,
   ScalarStore,
   ```

2. **增加 workload / feature 字段**
   `include/AscendModel/RouteModel/StageRouteCostModel.h` 的 `StageWorkload` 增加：
   ```cpp
   double scalarLoadCount = 0.0;
   double directScalarLoadCount = 0.0;
   double indirectScalarLoadCount = 0.0;
   double scalarStoreCount = 0.0;
   double indirectScalarStoreCount = 0.0;
   ```
   `StageModelFeatures` 增加：
   ```cpp
   bool hasScalarLoad = false;
   bool hasScalarStore = false;
   bool hasScalarIndirectLoad = false;
   bool hasScalarIndirectStore = false;
   ```

3. **在 Partitioner 中识别**
   `lib/AscendModel/Analysis/StagePartitioner.cpp` 的 `accumulateOneOperation()` 中：
   - 非 shaped `tt.load` 计入 `scalarLoadCount`；
   - 若 load 地址依赖另一个 scalar load，计入 `indirectScalarLoadCount`；
   - 非 shaped `tt.store` 计入 `scalarStoreCount`；
   - 若 store 地址依赖另一个 scalar load，计入 `indirectScalarStoreCount`。
   同时 `StageFeatureAnalysis` 设置对应 `hasScalar*` feature。

4. **分类**
   `classifySemanticRoot()` 中，如果 operation tree 含 scalar load/store，返回
   `StageCostModelKind::ScalarLoad` / `ScalarStore`。

5. **增加 formula**
   `lib/AscendModel/RouteModel/StageCostModels.cpp` 的 `mapWorkload()` 中：
   ```text
   scalarMemory += scalarLoadCount / scalarLoadInstructionsPerCycle
                + scalarLoadLatencyCycles
                + indirectScalarLoadCount * scalarIndirectDependencyLatencyCycles
   scalarMemory += scalarStoreCount / scalarStoreInstructionsPerCycle
                + scalarStoreLatencyCycles
                + indirectScalarStoreCount * scalarIndirectDependencyLatencyCycles
   ```

6. **增加 profile 字段**
   `StageModeProfile` 增加 scalar load/store throughput/latency 与 indirect dependency 字段；
   `SimdSimtCostModel.cpp` 从 `scalar_memory` JSON 读取。

7. **同步 JSON / schema / 单测**
   - `profiles/microbench/ascend_davidv100_v1.json`
   - `profiles/simd_simt/david_v100_simd_simt_v1.json`
   - `profiles/simd_simt/simd_simt_profile_schema.json`
   - `unittest/costmodel_ut/SimdSimtCostModelTest.cpp`

---

## 1. 目录内容

```text
scalar_ldst/
  simd_scalar_gm_memory.cce          # SIMD MainScalar scalar GM load/store throughput
  simd_scalar_gm_memory_host.cpp
  simt_scalar_gm_memory.cce          # SIMT per-warp uniform scalar GM load/store throughput
  simt_scalar_gm_memory_host.cpp
  simd_scalar_gm_dep.cce             # SIMD scalar load-to-load dependency
  simd_scalar_gm_dep_host.cpp
  simt_scalar_gm_dep.cce             # SIMT uniform scalar load-to-load dependency
  simt_scalar_gm_dep_host.cpp
  demo/
    demo_scalar_route.py             # 真实 scalar-heavy / mixed kernel 路由与时间 demo
    demo_scalar_cycle_accuracy.py    # scalar formula vs CCE cycle 精度 demo
```

## 2. 运行环境

在服务器上执行：

```bash
ssh ascend-950pr-63
source ~/env_ascend.sh
export ASCEND_RT_VISIBLE_DEVICES=1   # 优先用 1 号卡；不可用再改 0
```

编译所有探针：

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

## 3. 运行 throughput 探针

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
./simd_scalar_gm_memory_host
./simt_scalar_gm_memory_host
```

`simd_scalar_gm_memory_host` 输出示例（实测）：

```text
ops,load_cycles,load_scalar_instr_per_cycle,store_cycles,store_scalar_instr_per_cycle
1,11.928955,0.083830,8.747396,0.114320
2,19.502035,0.102553,16.480387,0.121356
4,30.562907,0.130878,26.680176,0.149924
8,51.973714,0.153924,48.229329,0.165874
```

`simt_scalar_gm_memory_host` 输出 32 warps、1 op/warp 的关键行（实测）：

```text
32,1,154.549642,0.207053,96.994303,0.329916
```

取值规则：

- SIMD throughput 取 `ops=8` 的 `scalar_instr_per_cycle`
- SIMD per-stage latency：
  - load latency ≈ `cycles(ops=1) - 1 / load_rate`
  - store latency ≈ `cycles(ops=1) - 1 / store_rate`
- SIMT throughput 取 `32 warps, ops=1` 的 `warp_instr_per_cycle`

当前填入 JSON 的实测值：

```text
simd.scalar.load.throughput        = 0.177913   # 由 ops=1,2,4,8 线性回归 slope 换算
simd.scalar.load.direct_latency    = 7.414187   # 线性回归 intercept
simd.scalar.store.throughput       = 0.181104
simd.scalar.store.direct_latency   = 4.327973
simt.scalar_gm.load.throughput     = 0.207053
simt.scalar_gm.store.throughput    = 0.329916
```

## 4. 运行 dependency 探针

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
./simd_scalar_gm_dep_host
./simt_scalar_gm_dep_host
```

`simd_scalar_gm_dep_host` 实测：

```text
ops,independent_cycles,dependent_cycles,extra_per_edge_cycles
1,10.116618,10.304118,0.187500
2,17.819743,17.580811,-0.119466
4,31.713053,32.122884,0.102458
8,49.857910,61.207764,1.418732
```

SIMD scalar indirect dependency 取 `ops=8` 的 `extra_per_edge_cycles`，当前填入：

```text
SIMD scalar indirect dependency = 1.65 SYS_CNT/edge   # dependent slope - independent slope
```

`simt_scalar_gm_dep_host` 实测（32 warps 行）：

```text
32,8,266.053060,376.023031,13.746246
```

SIMT scalar indirect dependency 取 32 warps、ops=8 的 `extra_per_edge_cycles`，当前填入：

```text
SIMT scalar indirect dependency = 14.0 SYS_CNT/edge
```

## 5. 从 CAModel 分析指令周期

如果使用 CAModel 而不是真机 CCE，请遵循：

```text
third_party/ascend/costmodel/profiles/microbench/data_provider/camodel/README.md
```

要点：

1. 不要递归累加所有 `instr_exe.csv`；
2. 按 `core/veccore` per-unit 解析，并记录 `span`；
3. SIMT scalar load/store 要识别 `SIMT_LDG/LDS/STG/STS` 和 RVEC pipe，不能只看 `SCALAR`；
4. simulator cycle 转 SYS_CNT 使用：
   ```text
   sys_cnt_cycles = simulator_cycles * simulator_mhz / sys_cnt_mhz
   ```
5. instruction cycles 用于 throughput/pipe 分析，dependency stall 应结合 span/wall 或 CCE dependent-chain slope。

## 5.1 CAModel Triton scalar cases（初步对照）

新增脚本：

```text
camodel_validate/
  scalar_camodel_cases.py      # 多个 scalar load/store Triton case
  run_camodel_compare.py       # 自动跑几个 case 并解析 CAModel block duration
```

运行：

```bash
cd .../scalar_ldst/camodel_validate
# 先真卡确认 kernel 可跑
python3 scalar_camodel_cases.py direct_load_4 --grid 2

# CAModel 对照（需要 msprof + simulator 环境）
python3 run_camodel_compare.py
```

已跑结果（grid=2，CAModel block duration，simulator cycles）：

```text
case              core0    core1
direct_load_1     2156     2256
direct_load_4     2620     2698
direct_store_4    3677     3777
dep_load_4        2691     2682
```

注意：

- 这是 **整个 Triton kernel block 的 CAModel duration**，包含 scalar 以外的启动/地址/控制开销；
- 不能直接和 CCE 微基准的单条 scalar load/store cycle 画等号；
- 从 marginal delta 看：
  - `direct_load_1 -> direct_load_4` 多 3 个 direct load，约 151 sim cycles/load；
  - `direct_load_1 -> direct_store_4` 多 3 个 store，约 507 sim cycles/store；
  - `direct_load_1 -> dep_load_4` 多 4 个 dependent load，约 120 sim cycles/dep-load；
- 若要与公式比较，应先做 simulator cycle -> SYS_CNT 换算，并说明这里包含的是完整 kernel 开销，不是纯 scalarMemory resource。

## 5.2 SIMD vs SIMT 的 CAModel 指令级对比与打分验证

新增脚本：

```text
camodel_validate/
  analyze_camodel_simd_simt.py   # 解析 SIMD instr_exe 与 SIMT primary dump
  score_scalar_cases.py          # 输出 cost model 对 SIMD/SIMT 的 total cycle 打分
```

使用方式：

```bash
# 1) 分别跑 SIMD 和 SIMT 的 CAModel case
# SIMD:  run_camodel_compare.sh
# SIMT:  run_camodel_compare_simt.sh

# 2) 解析两套 OPPROF
python3 analyze_camodel_simd_simt.py \
    --simd-dir results_simd --simt-dir results_simt \
    --cases direct_load_1,direct_load_4,direct_store_4,dep_load_4

# 3) 查看 cost model 打分
python3 score_scalar_cases.py
```

实测结果：

```text
case              CAModel block avg (simd)   CAModel block avg (simt)   cost model all_simd   cost model all_simt
direct_load_1     2206                       2866                       26.68                 155.36
direct_load_4     2659                       2890                       60.60                 172.10
direct_store_4    3727                       2884                       50.76                 165.95
dep_load_4        2687                       3033                       98.59                 221.68
```

CAModel 指令级汇总：

```text
case              SIMD scalar_pipe_cycles   SIMT active_span_cycles   SIMT memory_ops
direct_load_1     4802                      2379                      24
direct_load_4     7551                      2427                      42
direct_store_4    4826                      2415                      42
dep_load_4        6793                      2696                      48
```

初步结论：

- 在当前 `grid=2, num_warps=1` 的低占用场景下，CAModel block 显示：
  - direct load：SIMD 更快；
  - direct store：SIMT 更快；
  - dependent load：SIMD 更快。
- cost model 全量打分目前全部选 SIMD，说明：
  - direct load / dependent load 的趋势与 CAModel 一致；
  - direct store 场景可能低估了 SIMT 在低 grid 下的 store 优势。
- 如果只用 scalarMemory 公式看，direct store 的 SIMT 相对优势会出现，但全量 cost model 的 SIMT setup 成本把优势吃掉了。
- 因此后续若要提高 SIMD/SIMT 区分度，需要重点检查 SIMT setup/launch 成本以及低 warp 占用下的 SIMT scalar store 参数。


> 注意：上面这些 tiny scalar-only case 是“把 scalar load/store 单独拆出来”的微验证，
> 不代表真实 binned/scalar-heavy kernel 的 route 结论。
> 真实 binned 例子（如 padded_copy_gather、binned_copy_gather）还包含 vector tile 循环 / recurrence / 大量 logical program，
> 这些负载会把 SIMT setup 摊薄，cost model 仍可能选择 all_simt_only。
> 因此 tiny scalar-only 的 all_simd 结果不应外推到 binned 场景。

## 5.4 纯 scalar load/store 的高 grid 实测对比

前面的 tiny scalar-only CAModel 例子 grid 太小，SIMT setup 没有被摊薄，所以容易得出 all_simd。
为了看真实趋势，这里用同一个“只有 scalar load/store”的 Triton kernel，但把 grid 提高到 512，
在真机上分别跑 SIMD 和 SIMT-only，再和 cost model score 比较。

新增脚本：

```text
demo/compare_scalar_only_runtime.py
```

运行：

```bash
cd .../scalar_ldst/demo
source ~/env_ascend.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
python3 compare_scalar_only_runtime.py --grid 512
```

结果：

```text
case              score_simd   score_simt   measured_simd_ms   measured_simt_ms   score_winner   measured_winner   consistent
dep_load_4        985.922      768.538      0.022686           0.019614           simt           simt              True
dep_load_8        1709.080     1051.495     0.027600           0.021258           simt           simt              True
direct_load_1     266.764      524.582      0.023405           0.019883           simd           simt              False
direct_load_4     605.987      595.049      0.037323           0.018337           simt           simt              True
direct_load_8     985.856      689.006      0.032872           0.021095           simt           simt              True
direct_store_1    266.764      524.582      0.020136           0.019342           simd           simt              False
direct_store_4    507.624      569.862      0.029620           0.020909           simd           simt              False
direct_store_8    828.771      630.235      0.042444           0.019338           simt           simt              True
```

结论：

- 在较高 grid 下，纯 scalar load/store 的实测通常 SIMT-only 更快；
- cost model 在 load/dep 数量较大时能正确选择 SIMT；
- 对数量很少的 `direct_load_1` / `direct_store_1` / `direct_store_4`，cost model 仍偏向 SIMD，但实测 SIMT 更快；
- 说明当前模型在小 scalar 数量时可能高估 SIMT setup 或低估 SIMT scalar store 优势；
- 这个纯 scalar 高 grid 结果与前面的 tiny CAModel 小例子不同，更适合用来观察 SIMD/SIMT 趋势。

## 5.5 Megablocks 六个 scalar-heavy 测试 kernel

对应 kernel：

```text
padded_copy_gather
padded_copy_scatter
padded_copy_wgrad
binned_copy_gather
binned_copy_scatter
binned_copy_wgrad
```

它们属于：

- 高 logical program 数
- 高 scalar index load 占比
- 同时包含 vector tile copy / loop / reduction 等非 scalar 负载
- 不是“纯 scalar-only”，也不是“低 grid tiny scalar”

在远程实际跑 cost model 路由：

```text
padded_copy_gather    -> all_simt_only  PASS
padded_copy_scatter   -> all_simt_only  PASS
padded_copy_wgrad     -> all_simt_only  PASS
binned_copy_gather    -> all_simt_only  PASS
binned_copy_scatter   -> all_simt_only  PASS
binned_copy_wgrad     -> all_simt_only  PASS
```

结论：这类“高 scalar 占比 + 有 vector 主体 + 高 grid”的 Megablocks kernel，cost model 能稳定预测 `all_simt_only`，和测试预期一致。

## 5.6 当前场景能力总结

| 场景 | 典型 case | 真实谁快 | 模型判谁 | 是否一致 | 原因 |
|---|---|---:|---:|---:|---|
| 高 grid + 纯 scalar + scalar 多 | direct_load_4/8, dep_load_4/8, store_8 | SIMT | SIMT | ✅ | scalar 多，SIMT per-op 优势盖过 setup |
| 高 grid + scalar 很少 | direct_load_1, store_1, store_4 | SIMT | SIMD | ❌ | 模型对 SIMT setup 摊销不足，小 scalar 量时偏向 SIMD |
| 低 grid + load/dep | tiny direct_load_1/4, dep_load_4 | SIMD | SIMD | ✅ | 低占用下 SIMD scalar path 更划算 |
| 低 grid + store-heavy | tiny direct_store_4 | SIMT | SIMD | ❌ | 模型 SIMT setup 吃掉 store 优势 |
| 高 scalar + vector 主体 + 高 grid | Megablocks 六 kernel | 测试预期 SIMT | SIMT | ✅ | vector/recurrence 主体让 SIMT 优势明显 |

主要待解决问题：

- SIMT setup 在“小 scalar 工作量”时摊销不足；
- 需要进一步把 SIMT setup 建模成随 scalar workload / occupancy 摊销，而不是在小 scalar 场景也收完整 setup。

## 6. 更新 JSON

修改：

```text
profiles/microbench/ascend_davidv100_v1.json
profiles/simd_simt/david_v100_simd_simt_v1.json
```

当前所有数值都来自本目录 CCE 探针的真机输出，不是 provisional。

## 6.1 Vector core 数量换算

本机真卡：

```python
driver.active.utils.get_device_properties(torch.npu.current_device())
# {'num_vectorcore': 56}
```

如果 CAModel/simulator 显示 32 或 64 个 vector core，不要直接把真卡的 56 当作 simulator 的
`physical_vector_core_count_hint`。

Cost model 的换算关系：

```text
runtime_physical_program_count = ceil(logical_program_count / superblock_factor)
runtime_wave_count = ceil(runtime_physical_program_count / physical_vector_core_count)
total_system_cycles *= runtime_wave_count
```

因此：

- 对比真卡测量时，使用 `physical_vector_core_count_hint = 56`
- 对比 simulator 时，使用 simulator 实际 vector core 数（例如 32 或 64）
- 如果要把 simulator 结果折算到真卡，请按 wave_count 比例换算，而不是直接比绝对 cycle

## 7. 验证

```bash
cd ~/triton-ascend
python3 -m json.tool \
  third_party/ascend/costmodel/profiles/microbench/ascend_davidv100_v1.json
python3 -m json.tool \
  third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json
```

手动运行 demo：

```bash
cd ~/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst

# 不需要 NPU：查看 scalar formula 与 CCE cycle 的精度
python3 demo/demo_scalar_cycle_accuracy.py

# 需要 NPU/已安装 wheel：跑 scalar-heavy / mixed kernel 的 route 与实测时间
source ~/env_ascend.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
python3 demo/demo_scalar_route.py
```

C++ 单测：

```bash
cd ~/triton-ascend/build-ut
source ~/env_ascend.sh
ninja third_party/ascend/unittest/costmodel_ut/SimdSimtCostModel
./third_party/ascend/unittest/costmodel_ut/SimdSimtCostModel
```

预期全部通过。
