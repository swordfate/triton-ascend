# transpose cost model 验证与材料

目标：确认 transpose 在 Triton-Ascend / NPUIR 下是否真的“底下做了替换”，
并用 Triton demo 对比 cost model 预测的 SIMD/SIMT 比值与实测比值。

## 文件说明

- `validate_transpose_route_ratio.py`
  验证脚本：同一个 `tl.trans` kernel 分别跑显式 `simd` / `simt_only`，
  再读 auto cost model report 中的 `all_simd_score` / `all_simt_score`。
- `bench_transpose_triton.py`
  同一份 tile kernel 的 marginal 标定脚本：对 `add` / `trans` 扫描次数，
  分别跑 `simd` / `simt_only`。
- `partition_probe.py`
  查看 StagePartitioner 把 `tt.trans` 放到哪个 stage 的探针脚本。
- `route_ratio_results.csv`
  SIMD/SIMT 路由比值验证结果。
- `results/transpose_marginal.csv`
  transpose marginal 标定原始输出。
- `npuir_full_ir_notes.md`
  NPUIR 全量 IR 观察：`linalg.transpose` → `hivm.hir.vtranspose` →
  在部分场景会被 `FuseTransposeIntoLoad` 变成 strided load/store。

## 路由比值验证

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH=/home/c00946898/triton-ascend/build/lib.linux-x86_64-cpython-311:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
export TRITON_BACKENDS_IN_TREE=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=$(pwd)/cache/transpose_route_validation

cd third_party/ascend/costmodel/profiles/microbench/data_provider/transpose
python3 validate_transpose_route_ratio.py
```

### 最近结果

| case | M | N | measured SIMD ms | measured SIMT ms | measured SIMD/SIMT | predicted all_simd/all_simt | effective |
|---|---:|---:|---:|---:|---:|---:|---|
| trans_64x64 | 64 | 64 | 0.023930 | 0.022379 | 1.0693 | 0.0755 | all_simd |
| trans_128x128 | 128 | 128 | 0.022413 | 0.018734 | 1.1964 | 0.0755 | all_simd |

## Marginal 标定

同一份 tile kernel，只切换 mode，扫描 `ops` 后取 marginal slope。

```bash
source ~/env_ascend.sh
source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope
export ASCEND_RT_VISIBLE_DEVICES=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR=$(pwd)/cache/transpose_marginal

cd third_party/ascend/costmodel/profiles/microbench/data_provider/transpose
python3 bench_transpose_triton.py \
  --modes simd simt_only \
  --ops add trans \
  --ops-list 0 32 64 128 256 512 \
  --reps 20
```


实测 marginal slope（`results/transpose_marginal.csv`）：

| mode | op | slope (ms/op) | 相对 add |
|---|---:|---:|---:|
| SIMD | add | 0.000222 | 1.00 |
| SIMD | trans | 0.001260 | 5.67 |
| SIMT | add | 0.000026 | 1.00 |
| SIMT | trans | 0.000876 | 33.3 |


当前用于 profile 的 factor：

| mode | trans/add 初步比 | profile factor |
|---|---|---|
| SIMD | 5.67 | 6.0 |
| SIMT | 33.3（不稳定） | 33.0 |

对应 JSON：

```json
"f32.trans": {
  "relative_to": "f32.add",
  "factor": 6.0
}
```

SIMT 使用 `33.0`，确保 SIMD/SIMT 分开考虑，避免 cost model 选错 route。

## 结论

- transpose 已作为 `f32.trans` 接入 cost model；
- 但这个 factor 是 effective / context-dependent，不是稳定硬件吞吐；
- 后续如果遇到明确被 `FuseTransposeIntoLoad` 吸收成 strided load/store 的场景，
  可能需要单独再调。
