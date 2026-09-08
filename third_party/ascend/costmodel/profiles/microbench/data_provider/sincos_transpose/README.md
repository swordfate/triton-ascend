# sin / cos / transpose cost model 标定与验证

## 1. 问题是什么

cost model 之前没有 `sin` / `cos` / `transpose` 的显式成本项：

- TTIR 里的 `math.sin` / `math.cos` / `tt.trans` 都会落到 `generic.issue`；
- workload 里看不到 `f32.sin` / `f32.cos` / `f32.trans`；
- 在简单 load → op → store 形态下，它们会和 store 一起落到
  `continuous_tile_store`，只按 storeBytes / issueElements 计费；
- 实测 transpose 的 SIMD/SIMT 耗时接近，但旧模型预测 SIMD 远好于 SIMT，
  说明没有显式成本项会导致 route 选择失真。

NPUIR 侧确认：

- `sin` / `cos` 会被展开成 range reduction + polynomial / select，不是单条硬件指令；
- `transpose` 会出现 `linalg.transpose → hivm.hir.vtranspose`，部分场景再被
  `FuseTransposeIntoLoad` 吸收成 strided load/store。

## 2. 代码做了什么修改

### 2.1 StagePartitioner 识别新算子

`third_party/ascend/costmodel/lib/AscendModel/Analysis/StagePartitioner.cpp`：

```cpp
.Cases("math.sin", "tt.sin", "f32.sin")
.Cases("math.cos", "tt.cos", "f32.cos")
.Cases("tt.trans", "linalg.transpose", "f32.trans")
```

### 2.2 RouteModel 加载 rate

`third_party/ascend/costmodel/lib/AscendModel/RouteModel/SimdSimtCostModel.cpp`：

SIMD / SIMT 的 op 列表都加入：

```cpp
"f32.sin", "f32.cos", "f32.trans"
```

### 2.3 profile JSON 增加 factor

`third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json`：

| mode | op | factor |
|---|---|---:|
| SIMD | f32.sin | 15.0 |
| SIMD | f32.cos | 15.0 |
| SIMD | f32.trans | 6.0 |
| SIMT | f32.sin | 15.0 |
| SIMT | f32.cos | 20.0 |
| SIMT | f32.trans | 33.0 |

对应 JSON 字段：

```text
simd.ops.f32.sin.factor
simd.ops.f32.cos.factor
simd.ops.f32.trans.factor
simt.ops.f32.sin.factor
simt.ops.f32.cos.factor
simt.ops.f32.trans.factor
```

### 2.4 回归测试

`third_party/ascend/unittest/pytest_ut/test_sincos_costmodel_cases.py`：

- Triton sin/cos kernel 数值正确；
- route report 里出现 `f32.sin` / `f32.cos`；
- `unmodeled_cost_terms == []`。

## 3. 怎么标定的

标定方法：**同一份 Triton kernel**，只切换 `compile_mode='simd'` / `'simt_only'`，
扫描重复次数后做线性回归，得到 marginal slope，再和同一 kernel 下 `add` 比较。

### 3.1 sin / cos

脚本：

```bash
cd third_party/ascend/costmodel/profiles/microbench/data_provider/sincos_transpose

python3 bench_arith_triton.py \
  --modes simd simt_only \
  --ops add sin cos \
  --ops-list 0 64 128 256 512 \
  --reps 10
```

拟合结果：

| mode | op | slope (ms/op) | 相对 add | profile factor |
|---|---:|---:|---:|---:|
| SIMD | add | 0.000251 | 1.00 | - |
| SIMD | sin | 0.003705 | 14.76 | 15.0 |
| SIMD | cos | 0.003694 | 14.72 | 15.0 |
| SIMT | add | 0.001430 | 1.00 | - |
| SIMT | sin | 0.021139 | 14.78 | 15.0 |
| SIMT | cos | 0.028001 | 19.58 | 20.0 |

CAModel SIMD 辅助观察：BLOCK=1024 时 sin/add compute cycles 约为
19877/1784 ≈ 11.14，和上表 14.8 同量级，因此 SIMD factor 取 15.0。

### 3.2 transpose

脚本：

```bash
python3 bench_transpose_triton.py \
  --modes simd simt_only \
  --ops add trans \
  --ops-list 0 32 64 128 256 512 \
  --reps 20
```

拟合结果：

| mode | op | slope (ms/op) | 相对 add | profile factor |
|---|---:|---:|---:|---:|
| SIMD | add | 0.000222 | 1.00 | - |
| SIMD | trans | 0.001260 | 5.67 | 6.0 |
| SIMT | add | 0.000026 | 1.00 | - |
| SIMT | trans | 0.000876 | 33.3 | 33.0 |

说明：

- SIMD ratio 比较稳定，取 6.0；
- SIMT add baseline 偏小，33.3 不够稳定，先作为 effective factor；
- transpose 的成本与是否被 `FuseTransposeIntoLoad` 吸收有关，后续可继续细分。

### 3.3 SIMT F1/F2/F4 标定对照

为了确认 SIMT 标定是否需要按 `superblock_factor` 分开存 factor，补跑了
F2/F4 的 marginal sweep。命令与 F1 相同，只是额外传：

```bash
python3 bench_arith_triton.py   --modes simt_only --superblock-factor 2 ...
python3 bench_arith_triton.py   --modes simt_only --superblock-factor 4 ...
python3 bench_transpose_triton.py --modes simt_only --superblock-factor 2 ...
python3 bench_transpose_triton.py --modes simt_only --superblock-factor 4 ...
```

sin / cos 结果：

| mode | op | F1 ratio vs add | F2 ratio vs add | F4 ratio vs add |
|---|---|---:|---:|---:|
| SIMT | sin | 14.78 | 15.21 | 15.21 |
| SIMT | cos | 19.58 | 19.44 | 19.41 |

结论：sin / cos 的 marginal ratio 在不同 F 下基本不变，不需要在 JSON 中
为 F2/F4 单独存 factor。

transpose 结果：

| mode | op | F1 ratio vs add | F2 ratio vs add | F4 ratio vs add |
|---|---|---:|---:|---:|
| SIMT | trans | 33.3 | 10.44 | 10.44 |

transpose 的 F2/F4 ratio 和 F1 差异明显。当前先不增加 per-F JSON factor，
原因是：

- cost model 已经有通用的 `applySuperBlock()`，会在打分时用 F 对
  latency-sensitive / compute 部分分别处理；
- 现有 profile 的其他标定也没有“每个 op 一组 F1/F2/F4 factor”的结构；
- transpose 本身在 NPUIR 下更偏 data movement / layout，而不是纯 compute，
  直接加 per-F factor 可能会跟 superblock 公式重复计费。

后续建议：先用 route 验证确认 cost model 对 F1/F2/F4 的排序是否和实测一致；
如果仍然偏差明显，再考虑给 profile 增加类似
`factor_by_superblock` 的字段，或者把 transpose 建模成 latency-sensitive
而不是 compute。

## 4. 怎么验证修改有作用

### 4.1 单算子 partition 探针

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_ASCEND_AUTO_SIMT_SCOPE=auto
export TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=~/costmodel_test_output/partition.json

cd third_party/ascend/costmodel/profiles/microbench/data_provider/sincos_transpose
python3 partition_probe.py sin
python3 partition_probe.py cos
python3 partition_probe.py trans
```

sin 实际输出：

```text
op: sin
effective: all_simd
unmodeled: []
stage_3_continuous_tile_memory continuous_tile_memory {'generic.issue': 22552}
stage_4_continuous_tile_store continuous_tile_store {'f32.sin': 4096, 'generic.issue': 8192}
```

transpose 实际输出：

```text
effective: all_simd
unmodeled: []
stage_3_continuous_tile_memory continuous_tile_memory {'generic.issue': 3424}
stage_4_continuous_tile_store continuous_tile_store {'f32.trans': 1024, 'generic.issue': 3232}
```

说明新算子已经被识别成 `f32.sin` / `f32.cos` / `f32.trans`，并进入
`continuous_tile_store` 的 workload。

### 4.2 单算子 route 比值验证

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=$(pwd)/cache/route_ratio

python3 validate_route_ratio.py sin
python3 validate_route_ratio.py cos
python3 validate_route_ratio.py trans
```

每次只跑一个算子，只触发一次 cost model。

sin / cos / transpose 结果（每个算子只运行一次 cost model）：

| op | measured SIMD ms | measured SIMT ms | measured SIMD/SIMT | all_simd score | all_simt score | predicted SIMD/SIMT |
|---|---:|---:|---:|---:|---:|---:|
| sin | 0.020264 | 0.051084 | 0.3967 | 493.14 | 2299.88 | 0.2144 |
| cos | 0.019803 | 0.051962 | 0.3811 | 493.14 | 2445.13 | 0.2017 |
| trans | 0.033196 | 0.021679 | 1.5313 | 110.81 | 815.82 | 0.1358 |

transpose 加入 `f32.trans` 前后对比：

- `all_simd` score 从约 43.5 变成约 110.8；
- predicted ratio 从 0.0755 变成 0.1358；
- 虽然和实测仍有差距，但 transpose 已经不再只是 `generic.issue`。

说明：以上 SIMT 实测使用 `superblock_factor=1`，F2/F4 的 latency
hiding / wave 收益由 cost model 的 SuperBlock 逻辑在打分时处理。

### 4.3 回归测试

```bash
cd ~/triton-ascend
python3 -m pytest -q \
  third_party/ascend/unittest/pytest_ut/test_sincos_costmodel_cases.py
```

服务器结果：

```text
2 passed, 18 warnings
```

## 5. 文件说明

```text
sincos_transpose/
├── README.md
├── bench_arith_triton.py       # sin/cos/add marginal 标定
├── bench_transpose_triton.py   # transpose/add marginal 标定
├── partition_probe.py          # 单算子 stage 归属探针
└── validate_route_ratio.py     # 单算子 SIMD/SIMT 预测 vs 实测比值验证
```
