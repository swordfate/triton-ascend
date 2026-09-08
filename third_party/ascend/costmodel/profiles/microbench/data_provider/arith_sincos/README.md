# Triton sin/cos 标定与验证

目标：给 autoscope cost model 增加 `f32.sin` / `f32.cos` 的可用成本，
并验证当前 factor 的合理性。

## 目录结构

```text
arith_sincos/
├── README.md                          # 本说明
├── bench_arith_triton.py              # Triton 吞吐/斜率标定脚本
├── partition_probe.py              # 单次 sin/cos 查看 stage 归属的探针
├── validate_sincos_route_ratio.py     # SIMD/SIMT 预测比值 vs 实测比值验证
├── results/
│   ├── results_triton_sweep.csv       # ops 扫描原始 median 时间
│   └── route_ratio_results.csv        # SIMD/SIMT 路由比值验证结果
└── notes/
    ├── camodel_simd_notes.md          # CAModel SIMD 观察
    └── npuir_full_ir_notes.md         # NPUIR 全量 IR 展开观察
```

## 当前使用的 factor

来自 `results/results_triton_sweep.csv` 对 `ops=64..512` 做线性斜率。
标定使用**同一份 Triton kernel**，只切换 `compile_mode='simd'` / `'simt_only'`，
让后端自动决定底层是否 Taylor 展开/如何向量化。

| mode | op | 相对 add 的实测斜率比 | profile 中 factor |
|---|---|---|---|
| SIMD | sin | 14.76 | 15.0 |
| SIMD | cos | 14.72 | 15.0 |
| SIMT | sin | 14.78 | 15.0 |
| SIMT | cos | 19.58 | 20.0 |

这些 factor 已经写入：

```text
third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json
```

并已让 `f32.sin` / `f32.cos` 不再落到 `generic.issue`。


## 实测 marginal slope（同一份 kernel）

`results/results_triton_sweep.csv` 中 `ops=64..512` 线性回归得到：

| mode | op | slope (ms/op) | 相对 add |
|---|---:|---:|---:|
| SIMD | add | 0.000251 | 1.00 |
| SIMD | sin | 0.003705 | 14.76 |
| SIMD | cos | 0.003694 | 14.72 |
| SIMT | add | 0.001430 | 1.00 |
| SIMT | sin | 0.021139 | 14.78 |
| SIMT | cos | 0.028001 | 19.58 |

对应的 profile factor：

| mode | op | factor |
|---|---|---|
| SIMD | sin | 15.0 |
| SIMD | cos | 15.0 |
| SIMT | sin | 15.0 |
| SIMT | cos | 20.0 |

## 单算子 partition 探针

只跑一次 cost model，适合确认 `f32.sin` / `f32.cos` 落在哪个 stage：

```bash
cd third_party/ascend/costmodel/profiles/microbench/data_provider/arith_sincos

# sin
python3 partition_probe.py sin

# cos
python3 partition_probe.py cos
```

也支持 `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP`：

```bash
export TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=~/costmodel_test_output/sincos/partition_sin.json
python3 partition_probe.py sin
```


## 运行吞吐标定

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/home/c00946898/triton-ascend/build/lib.linux-x86_64-cpython-311:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
export TRITON_BACKENDS_IN_TREE=1

python3 bench_arith_triton.py \
  --modes simd simt_only \
  --ops add sin cos \
  --ops-list 0 64 128 256 512 \
  --reps 10
```

## 运行 SIMD/SIMT 路由比值验证

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/home/c00946898/triton-ascend/build/lib.linux-x86_64-cpython-311:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
export TRITON_BACKENDS_IN_TREE=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=$(pwd)/cache/sincos_route_validation

python3 validate_sincos_route_ratio.py
```

最近结果见 `results/route_ratio_results.csv`：

| case | N | measured SIMD ms | measured SIMT ms | measured SIMD/SIMT | predicted all_simd/all_simt | effective |
|---|---:|---:|---:|---:|---:|---|
| add | 4096 | 0.025490 | 0.020756 | 1.2281 | 0.0969 | all_simd |
| add | 16384 | 0.021893 | 0.026237 | 0.8344 | 0.1002 | all_simd |
| sin | 4096 | 0.018802 | 0.053351 | 0.3524 | 0.1628 | all_simd |
| sin | 16384 | 0.020632 | 0.091911 | 0.2245 | 0.1669 | all_simd |
| cos | 4096 | 0.018597 | 0.054840 | 0.3391 | 0.1384 | all_simd |
| cos | 16384 | 0.034797 | 0.099494 | 0.3497 | 0.1412 | all_simd |

解读：

- 方向基本正确：sin/cos 的实测和预测都倾向 `all_simd`；
- 数值上还不是完全精准，当前模型对简单 elementwise kernel 的 SIMD 优势估计偏强；
- add baseline 也有偏差，说明一部分偏差来自 memory/launch overhead 与 cost model
  资源分之间的口径差异，不全是 sin/cos factor 的问题。

## 结论

- sin/cos 已经从 `generic.issue` 变成可打分的 `f32.sin` / `f32.cos`；
- 当前 factor 仍属于低置信度，适合用于让模型“先认识 sin/cos”；
- 如果后续要更准，需要更多 shape、更多 SIMT CAModel，以及结合
  `notes/npuir_full_ir_notes.md` 里看到的实际 polynomial 展开做进一步校准。
