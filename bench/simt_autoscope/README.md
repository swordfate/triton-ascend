#!/usr/bin/env python3
# SIMT Auto-Scope 数据采集与残差分析

这套脚本用于在昇腾 A5 上采集代表性 Triton kernel 的三条路由延迟，并定位
当前 costmodel 中哪个评分组件误差最大。

## 文件

- `run_triton_benchmark.py`：跑内置代表性 kernel，输出 JSONL；
- `analyze_residuals.py`：读取 JSONL，输出“组件可疑度排序”；
- `microbench_simd_memory.py`：SIMD memory 微基准（contiguous/stride/gather/masked）；
- `microbench_simd_components.py`：SIMD compute/dot 微基准（add/mul/div/exp/cmp/select + matmul）；
- `docs/simt-costmodel-dataset-plan.md`：完整计划。

## 在 A5 上运行

前置条件：
- 已安装 triton-ascend（当前仓库），且 `python -c "import torch, triton"` 成功；
- `torch` 带 NPU 支持（`torch.npu` 可用）。

### 1. 采集数据

```bash
cd /path/to/triton-ascend

python bench/simt_autoscope/run_triton_benchmark.py \
    --case all --route all \
    --out results/simt_autoscope_bench.jsonl \
    --report-dir results/reports \
    --warmup 10 --reps 50
```

如果想先快速验证一条：

```bash
python bench/simt_autoscope/run_triton_benchmark.py \
    --case rowwise_reduce_masked --route simd \
    --out results/simt_autoscope_bench.jsonl
```

说明：
- `--route all` 会依次跑 `simd`、`simt_only`、`simd_simt_report`；
- 每条路由在独立子进程中运行，避免 `TRITON_ASCEND_*` 环境变量被缓存；
- `simd_simt_report` 会额外设置 `TRITON_ASCEND_AUTO_SIMT_SCOPE=report`，
  C++ costmodel 报告写入 `results/reports/<case>_simd_simt_report.jsonl`，
  同时把最后一行的 report JSON 内联到输出 JSONL 中。

### 2. 残差分析

```bash
python bench/simt_autoscope/analyze_residuals.py \
    --input results/simt_autoscope_bench.jsonl
```

脚本会输出：
1. 每个 case 的 `measured_ratio = latency(simd)/latency(simt_only)` 和
   `predicted_ratio = raw_all_simd / raw_all_simt_only`；
2. 每个 case 的组件占比；
3. 可疑组件排序：哪个组件的占比变化最能解释预测比和实测比的残差。

### 3. 如何根据结果决定下一步 cce benchmark

`analyze_residuals.py` 输出中，排名靠前的组件就是当前误差最可疑的地方。例如：

- 如果 `simt_predicate_share` 排名最高，就去重测 predicate rate；
- 如果 `simd_memory_share` 排名最高，就去重测 SIMD MTE2/MTE3 带宽；
- 如果 `simt_memory_share` 排名最高，就去扩展 `simt_gm_memory.cce` 的 stride/gather sweep。

注意：Triton 样本只能定位到“哪个评分组件”可疑；该组件的底层 rate 需要再用
cce benchmark 按 `docs/simt-costmodel-dataset-plan.md` 第 6 节的 sweep 矩阵重测。

## 当前内置 kernel 覆盖

| case | 近似目标算子 |
|---|---|
| `elementwise_silu_mul` | SGLang `silu_mul_static_tensorwise_quant...` / elementwise 类 |
| `rowwise_reduce_masked` | FBGEMM masked rowwise reduction / `_count_expert_num_tokens` |
| `indirect_elementwise` | SGLang `deepep_compute_src2dst...` / `deepgemm_compute_src2dst...` |
| `block_matmul` | VLLM `_w8a8_*` / Liger `matmul_kernel` / `array_jagged_bmm_kernel` |
| `single_block_cumsum` | FlagGems `cumsum` / FBGEMM `fused_padding_cumsum...` |

后续把外部仓库的真实 kernel 接入时，只需在 `run_triton_benchmark.py` 中增加
对应的 case 构建函数，保持 `(kernel, grid, args, launch_kwargs, meta)` 接口。
