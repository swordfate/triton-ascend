# Scalar Load/Store Calibration (All-Triton)

本目录当前只保留 **全部使用 Triton 内核** 的 scalar load/store 标定，不再使用 CCE 探针。

## 1. 环境

```bash
ssh ascend-950pr-63
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=0
ulimit -n 1048576
```

## 2. 标定脚本

```text
calibrate_triton_scalar_memory.py
```

核心方法：

- 同一份 Triton kernel 分别用 `compile_mode=simd` 和 `compile_mode=simt_only` 编译；
- SIMT 扫 `superblock_factor = 1/2/4`；
- 对每个 `(variant, ops, mode, factor)` 做 `N=0` / `N=ops` paired 差分；
- 地址由 host 预生成成 64B 间隔的 random slot，不在计时 kernel 里做随机计算；
- 用 grid sweep `2048/4096/8192/16384` 拟合 marginal cycles/program；
- 除以 `ops` 得到 marginal cycles per scalar op；
- 使用 median + Theil-Sen，降低异常点影响。

运行：

```bash
python3 calibrate_triton_scalar_memory.py \
    --reps 30 \
    --ops 1 2 4 \
    --grids 2048 4096 8192 16384
```

## 3. reps=30 实测结果

单位：cycles per scalar op。

| variant | ops | SIMD | SIMT F1 | SIMT F2 | SIMT F4 |
|---|---:|---:|---:|---:|---:|
| load | 1 | invalid | 137.537 | 69.119 | 26.439 |
| load | 2 | 41.241 | 85.181 | 48.891 | 23.134 |
| load | 4 | 29.980 | 79.090 | 43.648 | 23.228 |
| store | 1 | 28.109 | 34.832 | 24.383 | 7.706 |
| store | 2 | 52.612 | 36.603 | 21.029 | 8.083 |
| store | 4 | 58.491 | 35.771 | 21.519 | 10.706 |

观察：

- SIMD load 的 per-op 成本随 ops 增大明显下降，但固定开销仍然存在；
- SIMT 随 `superblock_factor` 增大明显变快；
- load 场景下 SIMD 在 ops=1/2/4 仍优于 SIMT F1/F2，F4 接近或优于 SIMD；
- store 场景下 SIMT F4 明显优于 SIMD。

## 4. 如何从实测结果填入 JSON

### 4.1 SIMD scalar load / store

对每个 variant 用总 cycles 对 ops 做线性拟合：

```text
total_cycles ≈ intercept + slope * ops
```

其中：

```text
total_cycles = cycles_per_op * ops
```

本次结果：

```text
load:  intercept ≈ 45.044, slope ≈ 18.719000   (ops=1 为 invalid，只用 ops=2/4 拟合)
store: intercept ≈ -36.261, slope ≈ 68.011429
```

填入：

```text
profiles/microbench/ascend_davidv100_v1.json
```

```json
"simd.scalar_gm.load.throughput": 0.05342165713980443,
"simd.scalar_gm.load.direct_latency": 45.044,
"simd.scalar_gm.store.throughput": 0.014703411191396398,
"simd.scalar_gm.store.direct_latency": 0.0
```

说明：store 的截距为负，不能直接作为 latency，因此 direct_latency 取 0。

### 4.2 SIMT scalar load / store

对每个 variant 使用：

```text
aggregate_cycles = a + b*warps + c*ops + d*warps*ops
aggregate_cycles = cycles_per_op * ops
```

其中 `warps = num_warps * superblock_factor`。本次只使用 `num_warps=1`，所以 `warps = 1/2/4`。

拟合结果：

```json
"load_cycles_fit": [
  78.3365,
  -19.520071429,
  68.1695,
  -12.148887755
],
"store_cycles_fit": [
  3.8915,
  -2.147214286,
  40.336857143,
  -7.483081633
]
```

填入：

```text
profiles/simd_simt/david_v100_simd_simt_v1.json
```

## 5. 结论

- scalar load/store 标定现在完全基于 Triton 实测，不再维护 CCE 探针数据。
- reps=30 下整体稳定；SIMD load ops=1 仍出现负值，因此 SIMD load fit 只用 ops=2/4。SIMT F2/F4 收益清晰，尤其 store 场景。
- load 场景 SIMD 仍有优势，说明“SIMT 一定比 SIMD 快”不成立，取决于 ops、F factor 和地址模式。
