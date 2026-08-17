# SIMT Auto-Scope 数据集与打分模型优化计划

> 目标：用代表性 Triton 算子 + 昇腾 A5 实测延迟构建数据集，替代当前
> `david_v100_simd_simt_v1.json` 中“单点 cce 微基准 + 小样本 Event 乘数”的
> 粗糙标定方式，最终让 SIMT auto-scope 能自动找出真正有用的 mixed/simt 路由。

## 0. 路线图与当前进度

> 当前处于 **Step 4：A5 采集 rate 训练数据**。已完成 Step 1-3，待完成 Step 5-8。

| Step | 内容 | 状态 |
|---|---|---|
| 1 | 读懂现有 SIMT auto-scope costmodel 的 anchor 识别、特征抽取、三路线打分、gate、scope 物化链路 | ✅ 已完成 |
| 2 | 设计需要重测的 rate、影响 rate 的 feature、以及 5 个代表性 Triton 内置 kernel | ✅ 已完成 |
| 3 | 编写测量工具：`run_triton_benchmark.py`、`analyze_residuals.py`、`microbench_simd_memory.py`、`microbench_simd_components.py`、`simt_predicate.cce`、`simt_gm_memory_pattern.cce`，并恢复 baseline `data_provider/` cce probe | ✅ 已完成，已推送 `kx_simt_costmodel` |
| 4 | 在 A5 上运行测量工具，采集 rate 训练数据 | ◀◀◀ **当前在这里** |
| 5 | 根据 A5 数据拟合 `rate = f(pattern_features)`，替换 profile JSON 中写死的单点 rate | 待完成 |
| 6 | 将拟合结果接入 C++ costmodel / profile JSON，使运行时按 TTIR 特征在线计算 rate | 待完成 |
| 7 | 重跑 5 个代表性 Triton kernel，检查 `predicted_ratio` 与 `measured_ratio` 是否接近 | 待完成 |
| 8 | 在 `auto` 模式下做真实路由决策验证 | 待完成（最终目标） |

### Step 4 需要在 A5 采集的清单

1. `simt_predicate_host` 的输出（重测 SIMT predicate rate）；
2. `simt_gm_memory_pattern_host` 的输出（重测 SIMT GM load/store rate）；
3. `ascend_results/simd_memory_microbench.jsonl`（重测 SIMD memory 行为）；
4. `ascend_results/simd_components_microbench.jsonl`（重测 SIMD compute / dot）。

### Step 1-3 已完成的具体内容

- 已理清三路线评分公式：`all_simd = (setup + payload*8)*(1+penalty)`，
  `all_simt_only = setup + payload*8`，`mixed = setup_fallback + 8*(regular_payload*(1+residual_penalty) + anchor_payload)`；
- 已确认 `simtPredicate` 的 `maskRankSum * ceil(maxNumel/32) / 0.038` 是
  ROPE 类 kernel 中 SIMT 分数虚高的主要原因；
- 已确认 `simdMemory` 的 `202.25 B/cycle` 对 indirect/gather 完全失效；
- 已确认 `simdDot` 和 `simtDot` 在当前 matmul 上两侧都有明显偏差；
- 已把第一批 5 case 的实测诊断写入本文档第 7 节；
- 已产出测量代码和文档：
  - `bench/simt_autoscope/run_triton_benchmark.py`
  - `bench/simt_autoscope/analyze_residuals.py`
  - `bench/simt_autoscope/microbench_simd_memory.py`
  - `bench/simt_autoscope/microbench_simd_components.py`
  - `third_party/ascend/costmodel/profiles/microbench/data_provider/simt_predicate.cce`
  - `third_party/ascend/costmodel/profiles/microbench/data_provider/simt_gm_memory_pattern.cce`

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
5. 当前实现是：对每个组件，计算 `component_share`（组件占 `issue_payload`
   的比例）与 `residual = predicted_ratio - measured_ratio` 的 **Pearson 相关系数**，
   按 `|corr|` 排序。相关性最强的组件，就是最需要重测/重拟合的 rate。

   这是小样本阶段（当前 5 个 case）的稳健做法；当 A5 样本达到约 20 个以上后，
   会升级为 `measured_latency ~ sum(component_j)` 的线性/岭回归，直接拟合
   每个组件在线公式中的系数。

`bench/simt_autoscope/analyze_residuals.py` 自动完成以上计算，并输出组件可疑度排序。

## 6. CCE sweep 矩阵

按目标算子特征决定 sweep 维度，详见脚本目录下 README。当前最高优先级：

1. SIMD GM/vector memory（MTE2/MTE3，替换 legacy 202.25）
2. SIMT GM load/store（contiguous / strided / gather / masked）
3. SIMT predicate / masked execution（当前最可疑）
4. SIMT shuffle（reduce 树 / 部分 reduce / 依赖链）
5. SIMT/SIMD ALU（op、dtype、ILP、warp 数）
6. launch/setup/transition

每个 rate 都拟合成 `rate = f(pattern_features)`，不直接存单点常数。


## 7. 第一批实测诊断（2026-08-17）

数据文件：`ascend_results/simt_autoscope_bench.jsonl`（A5 实测，内置 5 case）。

### 7.1 模型预测比 vs 实测比

| case | 实测 SIMD ms | 实测 SIMT ms | 实测 simd/simt | 模型 raw simd/simt | 模型 calibrated simd/simt |
|---|---|---|---|---|---|
| block_matmul | 0.0167 | 0.0150 | 1.111 | 0.103 | 0.103 |
| elementwise_silu_mul | 0.0229 | 0.0321 | 0.713 | 0.036 | 0.036 |
| indirect_elementwise | 0.5048 | 0.0140 | 36.087 | 0.012 | 6.461 |
| rowwise_reduce_masked | 0.0233 | 0.0131 | 1.781 | 0.028 | 0.028 |
| single_block_cumsum | 0.0610 | 0.0133 | 4.577 | 0.008 | 0.008 |

结论：raw 公式系统性低估 SIMD / 高估 SIMT；domain multiplier 只能修正
indirect_elementwise 的一部分，且修正后仍远低于实测比。

### 7.2 组件占比暴露的问题

`simt_predicate_share` 在所有带 bounds mask 的 case 中占 SIMT payload 的
84.6%~87.3%。例如 `single_block_cumsum` 没有任何业务 mask，仅
`offs < n_elements` 一个边界 mask 就贡献了：

```text
predicate_warp_instructions = 3 * ceil(4096/32) = 384
predicate_system_cycles = 384 / 0.038 = 10105
simt_issue_payload = 11822   // predicate 占 85.5%
```

这正是 `_fwd_grouped_kernel_stage1_rope` 中 predicate 成本虚高的同一个问题。

### 7.3 主要结论

1. **最优先修 `simtPredicate`**：当前 `maskRankSum * maxNumel/32 / 0.038`
   把边界 mask 当成主要成本，必须改为按“实际被 mask 的 warp 指令数”计算，
   并重测 predicate rate。
2. **SIMD memory 对 indirect 完全失效**：`indirect_elementwise` 实测 SIMD
   0.5048 ms，raw 公式只用 40.5 cycles 估计 memory，偏差约 1000 倍。
   202.25 B/cycle 的 legacy 带宽必须替换为按访问模式区分的 SIMD memory
   模型。
3. **dot 模型两侧都有偏**：`block_matmul` 中 SIMD dot 被低估约 5 倍，
   SIMT dot 被高估约 2 倍。需要按 M/N/K 重测 dot 吞吐。

### 7.4 下一步 cce / 微基准

按优先级：

1. `simt_predicate.cce`：masked add / predicated select / masked load，
   扫描 active_ratio 和 warp 数；
2. `microbench_simd_memory.py`：Triton SIMD memory contiguous/stride/gather/masked；
3. `simt_gm_memory_pattern.cce`：SIMT GM load/store 的 contiguous/stride/gather；
4. dot 微基准：SIMD cube 和 SIMT scalar FMA 分 M/N/K 重测。

### 7.5 dot 低估/高估倍数的计算依据

`block_matmul` 实测与模型 raw 分数如下（SYS_CNT 频率按 microbench 配置
988.9 MHz，cycle = latency_ms * 988900）：

```text
measured_simd_cycles = 0.0167 ms * 988900 = 16514 cycles
model_raw_simd       = 3125.6 cycles
=> SIMD 侧模型低估约 16514 / 3125.6 = 5.28x

measured_simt_cycles = 0.0150 ms * 988900 = 14833 cycles
model_raw_simt       = 30418.1 cycles
=> SIMT 侧模型高估约 30418.1 / 14833 = 2.05x
```

再看 `block_matmul` 的 breakdown：

```text
simd_dot = 256 cycles, simd_issue_payload = 257.8  => dot 占 payload 99%
simt_dot = 3782 cycles, simt_issue_payload = 3784.6 => dot 占 payload 99.9%
```

所以这次 matmul 的模型误差几乎全部来自 dot 公式，因此可近似归因为：
**SIMD dot 被低估约 5 倍，SIMT dot 被高估约 2 倍。**

### 7.6 从 analyze_residuals 输出到 Step 4 测量清单的推理链

`analyze_residuals.py` 输出组件占比表和可疑度排序后，按以下推理选择重测对象：

| 观测到的证据 | 推理 | 选择的测量工具 |
|---|---|---|
| 4 个 case 中 `simt_predicate_share` 高达 84.6%~87.3%；`single_block_cumsum` 仅一个边界 mask 就按 `3 * ceil(4096/32) = 384` 个 predicate warp 指令计费 | 当前 predicate 指令数公式和 0.038 rate 都不合理，必须重测 masked/predicated 执行成本 | `simt_predicate.cce` |
| `indirect_elementwise` 实测 SIMD 0.5048 ms，raw 公式的 SIMD memory 仅 40.5 cycles；`simd_memory_share` 与残差相关性 -0.641 | SIMD memory 的 202.25 B/cycle 对 irregular/gather 完全失效 | `microbench_simd_memory.py` |
| SIMT memory 当前只有 contiguous 的 0.176/0.129 两个单点；indirect case 的 SIMT 实际很快但公式只用顺序带宽估计 | SIMT GM rate 需要按 contiguous/stride/gather 重测 | `simt_gm_memory_pattern.cce` |
| `block_matmul` 中 dot 占 SIMD/SIMT payload 均超过 99%，且模型两侧偏差方向相反 | `simd.ops.*`、`simd.dot.*`、`simt.dot.*` 都需要实测 | `microbench_simd_components.py` |
| `run_triton_benchmark.py` 的 5 个内置 kernel 是第一步诊断数据来源 | 它们负责产生组件占比表，不是测量工具本身 | 已在第一批采集完成 |

### 7.7 `simt.gm.load=0.176 / store=0.129` 为什么说是顺序访存

证据在微基准 profile 和 cce 源码里。

`ascend_davidv100_v1.json` 中这两个 measurement 的描述是：

```text
simt.gm.load.throughput:
  "Effective source GM-load rate for the stated 32-warp sequential rotating
   runtime-loop workload ..."

simt.gm.store.throughput:
  "Effective source GM-store rate for the stated 32-warp sequential rotating
   runtime-loop workload ..."
```

对应 `simt_gm_memory.cce` 的地址生成是：

```cpp
int base = (i & 4095) * threads * 8 + tid;   // 顺序旋转地址
x0 = gm[base + threads * 0];
x1 = gm[base + threads * 1];
...
```

所以它们只能代表 **contiguous/sequential** 访存，不能代表 strided 或 gather。

### 7.8 把现有 penalty 乘到 memory/dot 上会变准吗

用第一批 5 case 数据验证，结论是：**只靠现有 penalty 乘法不够，必须换 rate 函数。**

#### 7.8.1 indirect SIMD：penalty 根本没触发

`indirect_elementwise` 的 report 显示：

```text
irregular_density = 0
irregular_addressing = 0
penalty_ratio = 0.088   // 只有 mask_materialization 0.088
```

为什么 `irregular_density=0`？因为当前 irregular 特征来自
`laneDependentPointerOps`（rank>1 的 pointer 代理），而我们的 gather 是
rank-1 索引，`laneDependentPointerOps=0`。但同一 report 里
`loaded_index_dependent_memory_ops=1` 其实已经识别出了 indirect。

因此即使把 `irregular_addressing` 从“乘整个 SIMD analytical”改成“只乘
memory”，这个 case 也不会有任何变化——因为 penalty 项是 0。

数值上：

```text
measured_simd_cycles = 0.5048 ms * 988900 = 499239 cycles
raw_all_simd         = 375.6 cycles
measured / raw       = 1329x
```

要把 memory 从当前 `40.5 cycles` 抬到实测水平，需要：

```text
implied_payload = (499239 / 1.088 - 21.2) / 8 = 57354 cycles
implied_simd_mte2_rate ≈ 8192 / 57354 = 0.143 B/cycle
```

而当前写死的是 `202.25 B/cycle`。这不是 `min(0.5, irregularDensity*0.8)`
这种量级的惩罚能解决的，必须按 `loaded_index_dependent_memory_ops` 等特征
重新定义 SIMD memory rate。

#### 7.8.2 dot：现有 penalty 已含但仍然差 5.3x

`block_matmul` 的 dotFlops=524288，超过 `tiny_dot_flops_max=16384`，所以
`tiny_dot` penalty 没触发；它实际触发的是 irregular penalty 0.5：

```text
irregular_density = 1
penalty_ratio = 0.5
raw_all_simd = (21.2 + 8*257.8) * (1 + 0.5) = 3125.6 cycles
measured_simd_cycles = 0.01672 ms * 988900 = 16536 cycles
measured / raw = 5.29x
```

也就是说，**当前公式已经把 irregular penalty 乘进去了，仍然低估 5.29 倍**。
SIMT 侧没有 dot penalty：

```text
raw_all_simt = 30418.1 cycles
measured_simt_cycles = 0.01505 ms * 988900 = 14878 cycles
raw / measured = 2.04x
```

所以正确做法不是继续加大惩罚系数，而是分别重测：
- SIMD dot 的有效 flops/cycle（按 M/N/K 分档）；
- SIMT dot 的有效 flops/cycle（按 M/N/K 分档）。

### 7.9 四个测量文件的 feature 设计逻辑

`analyze_residuals` 的结论指向“某个组件不可信”后，需要进一步拆解该组件
由哪些 feature 决定。四个测量文件就是按这个逻辑设计的。

#### 1) `simt_predicate.cce`

- 目标组件：`simt_predicate_cycles = predicate_warp_instructions / predicate_rate`
- 当前公式的两处问题：
  1. `predicate_warp_instructions = maskRankSum * ceil(maxNumel/32)` 不合理；
  2. `predicate_rate = 0.038` 是单个 workload 的旧数据。
- 需要测量的 feature：
  - `mode`：无 mask / bounds-mask / predicated select / masked load，对应
    Triton 中 `mask = offs < n`、`tl.where(cond, a, b)`、masked `tl.load` 等模式；
  - `active_lanes`：mask 激活比例，对应 TTIR 中 mask 的 true 比例；
  - `warps`：warp 数，对应 `num_warps`。
- 输出 target：`cycles_per_iter`，后续拟合成 `predicate_warp_instructions` 和
  `predicate_rate` 的 feature 函数。

#### 2) `simt_gm_memory_pattern.cce`

- 目标组件：`simt_load_cycles = load_warp_instructions / simt_load_rate`
  和 `simt_store_cycles = store_warp_instructions / simt_store_rate`。
- 当前公式问题：`simt_load_rate=0.176`、`simt_store_rate=0.129` 只有顺序
  访存单点。
- 需要测量的 feature：
  - `pattern`：contiguous / strided / gather，对应 TTIR 中
    `loaded_index_dependent_memory_ops`、`lane_dependent_pointer_ops` 的不同取值；
  - `stride`：stride 大小，对应 pointer/访问步长；
  - `mode`：load / store；
  - `warps`：对应 `num_warps`。
- 输出 target：`bytes_per_cycle` 和 `warp_instructions_per_cycle`。

#### 3) `microbench_simd_memory.py`

- 目标组件：`simd_memory = max(load_bytes / mte2_rate, store_bytes / mte3_rate)`。
- 当前公式问题：`mte2_rate = mte3_rate = 202.25` 是 legacy 单点，且对
  irregular 完全不成立。
- 需要测量的 feature：
  - `pattern`：contiguous / strided / gather / masked，对应 TTIR 中
    `loaded_index_dependent_memory_ops`、`lane_dependent_pointer_ops` 和 mask 特征；
  - `stride`：访问步长；
  - `n`：tensor 元素数，控制 working set。
- 输出 target：`bytes_per_second`，后续换算成 `bytes_per_cycle`。

#### 4) `microbench_simd_components.py`

- 目标组件：`simd_compute_cycles = Σ ceil(elements/vector_width)/op_rate*factor`
  和 `simd_dot_cycles = dot_setup + dot_flops / simd_dot_flops_per_cycle`。
- 当前公式问题：`simd.ops.*` 中只有 `f32.add` 是实测，其余 op 用固定
  factor；`simd.dot.*` 是 legacy 种子。
- 需要测量的 feature：
  - `op`：add / mul / div / exp / cmp / select，对应 TTIR op 类型；
  - `dtype`：f32 / f16；
  - `n`：元素数；
  - dot 的 `M/N/K`：对应 `tt.dot` 的 shape。
- 输出 target：`elements_per_second` 和 `flops_per_second`，后续换算为
  `vector_instructions_per_cycle` 和 `flops_per_cycle`。

### 7.10 四个测量脚本的目标值与特征含义

| 脚本 | 目标值（被拟合变量） | 特征 | 特征含义 |
|---|---|---|---|
| `simt_predicate.cce` | `warp_instructions_per_cycle`（由 `cycles_per_iter` 换算） | `mode` | 0 无 mask 基线；1 bounds-mask add；2 predicated select；3 masked GM load |
| | | `active_lanes` | 活跃 lane 数：32/24/16/8/4/1，对应 mask 激活比例 |
| | | `warps` | warp 数：1/2/4/8/16/32，对应 `num_warps` |
| `simt_gm_memory_pattern.cce` | `bytes_per_cycle`、`warp_instructions_per_cycle` | `mode` | 0 load；1 store |
| | | `pattern` | 0 contiguous；1 strided；2 gather |
| | | `stride` | strided 模式下的步长：1/2/4/8/16 |
| | | `warps` | warp 数：1/2/4/8/16/32 |
| `microbench_simd_memory.py` | `bytes_per_second`（后续换算 `bytes_per_cycle`） | `pattern` | contiguous / strided / gather / masked |
| | | `stride` | strided 模式下的步长：2/4/8/16 |
| | | `n` | 元素数：1M / 4M，控制 working set 大小 |
| `microbench_simd_components.py` | elementwise：`elements_per_second`；dot：`flops_per_second` | `op` | add / mul / div / exp / cmp / select，对应 TTIR op 类型 |
| | | `dtype` | f32 / f16 |
| | | `n` | elementwise 元素数：1M / 4M |
| | | `M,N,K` | dot shape：(128,128,64) / (256,256,128) / (512,512,128) |

这些目标值最终会换算成 profile 中对应的 rate 或 flops_per_cycle，再按特征
拟合成在线可查的 `rate = f(features)`。
