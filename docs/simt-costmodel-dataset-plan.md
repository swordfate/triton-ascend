# SIMT Auto-Scope 数据集与打分模型优化计划

> 目标：用代表性 Triton 算子 + 昇腾 A5 实测延迟构建数据集，替代当前
> `david_v100_simd_simt_v1.json` 中“单点 cce 微基准 + 小样本 Event 乘数”的
> 粗糙标定方式，最终让 SIMT auto-scope 能自动找出真正有用的 mixed/simt 路由。

## 1. 当前模型的问题（简要）

- 解析分数由固定 rate + 手工 penalty 组成，rate 是单点微基准测量；
- `simtPredicate` 用 `maskRankSum * ceil(maxNumel/32) / 0.038`，在 ROPE 类 kernel
  上会贡献 80% 的估算时间，明显虚高；
- `simtShuffle` 用整核 `maxNumel`，小 reduce 会被其他大 tensor 放大；
- SIMD memory 的 `202.25 B/cycle` 是 legacy 种子，未实测；
- coverage 只有 6 个域，每个域只有一组 route multiplier。

## 2. 数据分层

### 第一层：CCE 微基准 → rate 函数

把当前标量 rate 升级为 `rate = f(pattern_features)`。

### 第二层：Triton 算子 → 路由打分模型

用代表性 Triton kernel 在 A5 上测三条路由的实际 Event 延迟，得到：

```text
(TTIR 特征, shape, num_warps, num_stages, 路由) -> 实测延迟
```

## 3. 目标算子清单

完整位置和分类见下文，克隆路径在 `/Users/weijianchen/Documents/2026/`：

- FBGEMM
- VLLM
- SGLang
- LigerKernel
- FlagGems

### A. GEMM / Matmul（P0）

| 算子 | 文件 | 主要特征 |
|---|---|---|
| `_w8a8_block_int8_matmul` | VLLM/vllm/model_executor/layers/quantization/utils/int8_utils.py | `tl.dot` + block scale |
| `_w8a8_triton_block_scaled_mm` | VLLM/vllm/model_executor/layers/quantization/utils/fp8_utils.py | `tl.dot` + block scale |
| `fused_moe_kernel` | VLLM/vllm/model_executor/layers/fused_moe/fused_moe.py | grouped GEMM + activation |
| `matmul_kernel` | LigerKernel/src/liger_kernel/ops/experimental/mm_int8int2.py | int8xint2 GEMM |
| `array_jagged_bmm_kernel` | FBGEMM/fbgemm_gpu/fbgemm_gpu/sll/triton/triton_jagged_bmm_jagged_out.py | jagged BMM + offset load |

### B. Attention / MLA / RoPE（P0）

| 算子 | 文件 | 主要特征 |
|---|---|---|
| `_fwd_kernel_stage1` | VLLM/vllm/v1/attention/ops/triton_decode_attention.py | decode attention, dot+mask+reduce |
| `_fwd_grouped_kernel_stage1` | 同上 | grouped attention |
| `_fwd_grouped_kernel_stage1_rope` | SGLang/python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py | MLA + rope + indirect KV |
| `_rope_padded_kernel` | FBGEMM/fbgemm_gpu/experimental/gen_ai/test/kv_cache/rope_padded.py | rope + padding |
| `_mask_fwd_kernel` | LigerKernel/src/liger_kernel/ops/multi_token_attention.py | 因果 mask |

### C. Elementwise / Quantization（P1）

| 算子 | 文件 | 主要特征 |
|---|---|---|
| `_kernel_dequantize_mx4` | FBGEMM/fbgemm_gpu/fbgemm_gpu/triton/quantize.py | pack/unpack + lookup |
| `_kernel_silu_quantize_mx4_unpack` | FBGEMM/fbgemm_gpu/experimental/gemm/triton_gemm/fp4_quantize.py | silu + mx4 pack |
| `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` | SGLang/python/sglang/srt/layers/moe/ep_moe/kernels.py | 动态 loop + quant |
| `_bias_kernel` | VLLM/vllm/v1/worker/gpu/sample/logit_bias.py | logits bias |
| `sample_recovered_tokens_kernel` | VLLM/vllm/v1/sample/rejection_sampler.py | 逐 token 分支 |

### D. Reduction / Scan / Cumsum（P1）

| 算子 | 文件 | 主要特征 |
|---|---|---|
| `_count_expert_num_tokens` | VLLM/vllm/model_executor/layers/fused_moe/utils.py | reduction 计数 |
| `compute_seg_indptr_triton_kernel` | SGLang/python/sglang/srt/layers/moe/ep_moe/kernels.py | 二分查找边界 |
| `scan_part_sum_kernel` 等 | FlagGems/src/flag_gems/runtime/backend/_ascend/ops/cumsum.py | 分段 cumsum |
| `fused_padding_cumsum_and_segmented_arange_kernel` | FBGEMM/fbgemm_gpu/experimental/gemm/triton_gemm/fp4_quantize.py | cumsum + 二分查找 |

### E. Indirect / Scatter / Masked Select（P1）

| 算子 | 文件 | 主要特征 |
|---|---|---|
| `deepep_compute_src2dst_triton_kernel` | SGLang/python/sglang/srt/layers/moe/ep_moe/kernels.py | loaded src_id scatter |
| `deepgemm_compute_src2dst_triton_kernel` | 同上 | 多级 indirect |
| `masked_select` | FlagGems/src/flag_gems/experimental_ops/masked_select.py | count + prefix + scatter |

### F. Normalization / Conv / LoRA（P2）

| 算子 | 文件 | 主要特征 |
|---|---|---|
| `instance_norm` | FlagGems/src/flag_gems/fused/instance_norm.py | 行归约 + rsqrt |
| `_causal_conv1d_fwd_kernel` | VLLM/vllm/model_executor/layers/mamba/ops/causal_conv1d.py | 有状态循环 |
| `_lora_expand_kernel` | VLLM/vllm/lora/ops/triton_ops/lora_expand_op.py | gather + 累加 |

## 4. 三条路由采集协议

对每个 Triton kernel 的每组 shape，至少跑 3 条路由：

| 路由 | 环境变量 | 说明 |
|---|---|---|
| SIMD | `TRITON_ASCEND_COMPILE_MODE=simd` | 纯 SIMD 基线 |
| SIMT-only | `TRITON_ASCEND_COMPILE_MODE=simt_only` | 纯 SIMT 基线 |
| SIMD/SIMT 报告 | `TRITON_ASCEND_COMPILE_MODE=simd_simt` + `TRITON_ASCEND_AUTO_SIMT_SCOPE=report` | 采集 costmodel report，不改变路由 |

每条记录包含：
- case 名、shape、dtype、num_warps、num_stages；
- 三条路由的实测 Event 延迟（warmup + reps，取 median）；
- `simd_simt` 路由的 `report_json`（内含 features、breakdown、candidate_costs）。

## 5. 如何定位残差最大的 rate

Triton 样本跑完后，不能直接从 Triton 样本“读出” cce rate，但可以定位到
**哪个评分组件**误差最大。步骤：

1. 对每个 case，计算实测速度比：
   ```text
   measured_ratio_simd_over_simt = latency(simd) / latency(simt_only)
   ```
2. 从 report 中取出模型预测的三条路由分数，计算预测比：
   ```text
   predicted_ratio_simd_over_simt = candidate_costs.all_simd / candidate_costs.all_simt_only
   ```
3. 对比同一 case 的 `measured_ratio` 和 `predicted_ratio`：
   - 模型高估 SIMD / 低估 SIMT：说明 SIMD memory/compute/penalty 或 SIMT compute/memory/shuffle/predicate 有偏；
   - 不同 shape 下 `measured_ratio` 变化但 `predicted_ratio` 不变：说明控制该组件随 shape
     变化的 rate 是常数导致欠拟合。
4. 用 `breakdown` 的组件值做归因：
   - `simd_roofline_system_cycles`（memory）
   - `compute_only.simd / compute_only.simt`
   - `simt_execution.shuffle_system_cycles`
   - `simt_execution.predicate_system_cycles`
   - `simt_execution.program_issue_scale`
   - `event_route_calibration.raw_candidate_costs` 和 `candidate_costs`
5. 对每个组件构造 `measured_latency ~ sum(component_j)` 的线性/岭回归，
   拟合系数与 profile 中隐含系数偏差最大的组件，就是最需要重测/重拟合的 rate。

`bench/simt_autoscope/analyze_residuals.py` 会自动完成 1-4 步，并输出
每个组件的“实际拟合系数 vs 当前 profile 隐含系数”的排序。

## 6. CCE sweep 矩阵

按目标算子特征决定 sweep 维度，详见脚本目录下 README。当前最高优先级：

1. SIMD GM/vector memory（MTE2/MTE3，替换 legacy 202.25）
2. SIMT GM load/store（contiguous / strided / gather / masked）
3. SIMT predicate / masked execution（当前最可疑）
4. SIMT shuffle（reduce 树 / 部分 reduce / 依赖链）
5. SIMT/SIMD ALU（op、dtype、ILP、warp 数）
6. launch/setup/transition

每个 rate 都拟合成 `rate = f(pattern_features)`，不直接存单点常数。
