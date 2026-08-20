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
| 4 | 在 A5 上运行测量工具，采集 rate 训练数据 | ✅ 已完成（数据在 `ascend_results/`） |
| 5 | 根据 A5 数据拟合 `rate = f(pattern_features)`，替换 profile JSON 中写死的单点 rate | ◀◀◀ **当前在这里** |
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

### Step 4 实测数据摘要（A5，SYS_CNT=988.9 MHz）

#### SIMD memory（Triton，compile_mode=simd）

| pattern | n=4M 读 B/cycle | 初步拟合 |
|---|---|---|
| contiguous | 1250.8 | 固定基准 |
| strided s=2/4/8/16 | 16.5 / 11.7 / 8.0 / 4.3 | `rate ≈ 27.0 * stride^-0.63` |
| gather | 2.3 | 固定惩罚 |
| masked(50%) | 1204.6 | 接近 contiguous |

结论：当前 `202.25 B/cycle` 只对 contiguous/masked 大致可用，对 strided/gather
严重高估；gather 实际只有约 2.3 B/cycle。

#### SIMD compute/dot（Triton，compile_mode=simd）

- elementwise f32（load+op+store，1M）：有效 vector instr/cycle 约
  `add=1.47`，`mul/div/exp/cmp/select=1.15~1.17`；
- dot 实测 flops/cycle：`128/128/64=129`，`256/256/128=1023`，
  `512/512/128=4134`；三个 shape 延迟几乎都是 0.0164 ms，说明存在
  **约 16 μs 的固定 overhead**，当前 `dot.setup=128 cycles` 无法覆盖小 matmul。

#### SIMT predicate（cce，32 warps）

| 模式 | 有效 warp instr/cycle |
|---|---|
| 无 mask add | 0.206 |
| bounds-mask add | 0.235 |
| predicated select | 0.185 |
| masked GM load | 0.160 |

结论：当前 `simtPredicateRate=0.038` 约比实测 masked add 悲观 6 倍。

#### SIMT GM memory pattern（cce，32 warps）

| pattern | load warp instr/cycle | store warp instr/cycle |
|---|---|---|
| contiguous | 0.400 | 0.464 |
| stride=2 | 0.223 | 0.272 |
| stride=4 | 0.159 | 0.145 |
| stride=8 | 0.080 | 0.074 |
| stride=16 | 0.040 | 0.037 |
| gather | 0.020 | 0.021 |

结论：当前 `simt.gm.load=0.176` / `store=0.129` 只代表“近 contiguous”场景；
gather 场景实际只有约 0.020 warp instr/cycle，需要按 pattern 查表。

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


## 8. 优化存档（截至 v6）

> v6 后 5 个内置 case 的 raw ratio 与实测 ratio 基本对齐。以下优化都是有效
> 且已经落入代码的，后续优化以本节为基线。

### 8.1 数据与工具基础

| 提交 | 内容 | 作用 |
|---|---|---|
| `74594b2d1` | 新增 `run_triton_benchmark.py` / `analyze_residuals.py` | 三路由 5-case 采集与残差定位 |
| `e5776d678` | Triton kernel 移到模块顶层 | 修复 `tl is not defined` |
| `ae239d667` | 修复 report 扁平字段读取 | 残差分析能读到 breakdown |
| `2b996f317` | 恢复 baseline cce probe 并新增 predicate / GM pattern probe | cce 层训练数据 |
| `d58c4f49d` | 微基准输出 per-CTA 指标 | 对齐 TTIR block-local 口径 |
| `de5fcf6ed` | 扩展 SIMD memory / dot sweep 尺寸 | 拟合 size-dependent rate |

### 8.2 公式与 rate 有效优化

| 提交 | 优化 | 解决的问题 |
|---|---|---|
| `460faaee7` | 惩罚按组件归位：irregular→memory、tiny_dot→dot、mask/reduction/loop/control/rank1→compute | 修复 `(1+Pstruct)` 整体乘导致的内存/compute 错误 |
| `bc8e0a4ab` | SIMT dot 改用 cube：`flops_per_cycle` 141→4096 | `block_matmul` 中 SIMT dot 被高估 14 倍 |
| `bb24c6ac0` | 新增 scan 专用成本：SIMD O(n)，SIMT 固定 | `single_block_cumsum` 预测从 0.010 对齐到 5.0 |
| `7fc20efd3` | 修复 SIMT scan 未加入 payload | 补上后 cumsum 完全对齐 |
| `0dfd996bd` | 从 JIT 把 launch grid 传入 C++ | 让 per-CTA TTIR 特征能换算 whole-program 工作量 |
| `658835097` | 区分 kernel-level rate 与 CTA-level rate | SIMT GM cce rate 不再被 grid 错误放大 |
| `f13a203c6` | `program_issue_scale=1.0`，切换实际-cycle 口径；增加 min kernel cycle floor；domain multiplier 暂置 1.0 | 消除 8 倍错位，4/5 case 对齐 |
| `375835355` | SIMD memory 拆分 contiguous/gather load bytes | `indirect_elementwise` SIMD 分数 923k→466k，v6 对齐 |
| `ef3f7628a` | 修复 mixed 分区幽灵 gather（regular gather 只计超出 anchor load bytes 的部分）；gather 字节估计改用 anchor walk 精确 loadBytes；新增 i32.bitop/min/rem/bitcast 分类与 profile 占位 rate；用 walk 级计数器替换死代码 unclassified 检测 | 外部算子第一批诊断：count_expert mixed 18×→~1×；causal_conv1d SIMD gather 字节 4× 通胀消除；silu_quantize_mx4 位运算不再隐形 |

### 8.3 当前 v6 基准

| case | 实测 ratio | v6 raw ratio |
|---|---|---|
| block_matmul | ~1.106 | ~0.993 |
| elementwise_silu_mul | ~0.864 | ~0.951 |
| rowwise_reduce_masked | ~0.984 | ~0.880 |
| single_block_cumsum | ~4.904 | ~4.811 |
| indirect_elementwise | ~44.6 | ~45 左右（SIMD 分数 466k） |

#### v7 重验（2026-08-19，P0 修复后，数据 `ascend_results/simt_autoscope_bench_v7.jsonl`）

| case | 实测 ratio | v7 raw ratio |
|---|---|---|
| block_matmul | 0.999 | 0.993 |
| elementwise_silu_mul | 1.113 | 0.951 |
| rowwise_reduce_masked | 0.957 | 0.880 |
| single_block_cumsum | 5.099 | 4.811 |
| indirect_elementwise | 39.576 | 37.327（SIMD 466593，与 v6 一致） |

结论：5-case 模型输出与 v6 完全一致（P0 修复对内置 case 零影响，符合预期）；
P0-1 使 indirect 的 mixed 从 v5 的 463838（幽灵 gather）降到 10129，实测
mixed ≈13889 cycles，误差 0.73×。注意两次运行实测环境漂移大（elementwise
simd 0.0229→0.0145ms、simt 0.0321→0.0130ms），v6/v7 实测不可直接比；
elementwise 实测比在 0.86~1.11 间波动，不宜作校准锚点。

## 8.5 设计考量：为什么这样做，而不是那样做

### 1) 为什么 SIMD memory / dot 用 Triton 微基准，而 SIMT GM / shuffle / predicate 用 cce

- **SIMD 侧**：旧 cce probe 依赖 `RegBase/VecUtils.h` 里的 `VectorReg` /
  `CREATE_MASK_BY_SIZE` 旧 API，而当前 AscendNPU-IR 已删除该头文件并重写了
  SIMD intrinsic。要让旧 cce probe 编译，必须把 NPU-IR 回退到 `d4405acb`，
  或手工把 probe 移植到新 `Vector/VecUtils.h`。两者成本都高，而且本地没有
  ccec 工具链，无法验证。
  同时，costmodel 的输入本来就是 Triton 生成的 TTIR；用 Triton 微基准去测
  SIMD memory/dot 更接近真实 codegen，并自动适配当前 NPU-IR。
- **SIMT 侧**：cce probe 只依赖 SIMT 指令，不依赖旧 SIMD vector API，能直接
  编译；而且 cce 的 slope-over-iters 方法能把 launch/barrier 开销从
  instruction rate 中剥离，得到单 AIV 的 ALU/LSU/shuffle/predicate rate。
  这类“纯指令级 rate”用 Triton kernel 很难隔离，因为 Triton kernel 一定包含
  循环、mask 和固定开销。

备选方案：
- SIMD 侧也可以把旧 cce probe 移植到新 SIMD API，或使用双 INC 指向旧 NPU-IR；
  后续如果需要隔离“纯 ALU rate”（不含 load/store），再做这件事。
- SIMT 侧也可以用 Triton 微基准，但后续需要做额外处理，把 kernel 固定开销
  和指令 rate 分离。

### 2) 为什么把 `program_issue_scale` 设成 1.0

原来的 8.0 是 legacy 经验比例。引入 grid 和 whole-program work 后，如果继续
保留 8.0，所有 whole-kernel rate 都会被多乘一个 8，组件单位难以解释。

改成 1.0 后，所有组件都是实际 cycle 口径，拟合 rate 时可以直接和微基准实测
cycle 比较，避免“payload 单位”带来的换算错误。

备选方案：保留 8.0，但把每个 rate 先换算成 payload 单位。这样公式不用改，
但每次更新 rate 都要额外换算，容易再次出现“total/per-CTA”和“8 倍”混合错误。

### 3) 为什么暂时把 domain multiplier 置 1.0

当前阶段是验证解析公式本身，不应该让旧的 domain 常数把公式误差掩盖掉。
multiplier 保留在报告中，但不参与这次验证。

备选方案：不置 1，直接重拟合 multiplier。但那样会把公式误差吸收进 multiplier，
我们就无法知道公式哪里还不准。

### 4) 为什么要把 launch grid 传进 C++

TTIR 特征是 block-local 的，而 SIMD memory / dot 的 rate 是 whole-kernel 测的。
没有 grid，就无法把 `loadBytes_per_cta` 换算成 `totalLoadBytes`。这是 v1-v4
中 SIMD 被系统性低估的根因。

备选方案：把 whole-kernel rate 换算成 per-CTA rate。但 per-CTA rate 随 grid /
资源饱和变化，不是常数；grid 传递是更稳定的方案。

### 5) 为什么 SIMT GM rate 不乘 grid，而 SIMD memory / dot 要乘

- `simt_gm_memory_pattern.cce` 测的是单 AIV（单 CTA）的 rate；
- `microbench_simd_memory.py` / `microbench_simd_components.py` 测的是整个
  kernel 的 total rate。
两者口径不同：CTA 级 rate 应直接作用于 per-CTA 工作量；kernel 级 rate 应作用
于 `per-CTA × grid` 的总工作量。

### 6) 为什么 SIMD memory 要拆 contiguous / gather bytes

`indirect_elementwise` 这类 kernel 有两次 load：
一次连续读索引，一次按索引 gather 数据。若全部按 gather rate 计，SIMD 会被
高估。当前特征里只有 `loadedIndexDependentMemoryOps`，没有字节级区分，所以
用 `ops × maxTensorNumel × elementBytes` 估算 gather 字节数，剩余部分按
contiguous 处理。

备选方案：在特征抽取阶段就为每个 `tt.load` 标注是否为 loaded-index-dependent，
并单独累加 gather bytes；这是更精确的长期方案。

### 7) 为什么加入 min kernel cycle floor

小 kernel 的实测 Event 延迟有一大部分是固定开销（launch/wave/control），而
当前模型没有显式建模它。用 `max(analytical, floor)` 是工程上的简单近似。

备选方案：分别测 SIMD/SIMT 的 device-side launch overhead，放进 `setup` 或
单独加一个 `fixedOverhead` 项；这比 floor 更有解释性，后续应替换。

### 8) 为什么 scan 只测单 block

单 block scan 没有跨 CTA carry，模型简单可解释。多 block scan 需要 carry 传播
和 grid 依赖，当前缺少可靠的多 block scan microbench。

备选方案：扩展 `microbench_scan.py` 支持多 block + carry，再拟合 grid 相关的
scan 模型。
## 9. 距离最终目标还需要优化/验证的点

1. **外部 25 个真实算子验证**
   目前只在 5 个内置 case 上对齐。需要把 FBGEMM / VLLM / SGLang / LigerKernel /
   FlagGems 中梳理出的目标算子逐个接入 benchmark，检查泛化性。

2. **dot 模型仍需更细标定**
   - 目前 `small_kernel_min_cycles=16200` 是基于 3 个 SIMD shape 的 floor；
   - SIMT dot 只有 `128x128x64` 一个成功点，`256x256x128` 会 NPU 507035；
   - 需要更多成功 shape 拟合 `dotCycles = max(setup + flops/rate, floor)`。

3. **SIMT GM rate 应 warp 数自适应**
   cce 数据已有 1/2/4/8/16/32 warps 的 rate，但 C++ 仍只用 32-warp 单点。
   下一步把 `simt_load/store_rate = f(pattern, num_warps)` 接入。

4. **SIMD strided memory 尚未接入**
   已测出 stride=2/4/8/16 的 rate，但 C++ 目前只区分 contiguous/gather，
   还没有从 TTIR 特征估计 stride 并查表。

5. **predicate 指令数仍较粗**
   目前用 `maskTensorOps * ceil(maxNumel/32)`，还没用 `active_ratio`、
   `mode` 等 cce 特征。`simt_predicate.cce` 的数据可以支持更细的拟合。

6. **shuffle 模型只覆盖单 warp 树**
   当前 `ceil(reduceElements/32)*5` 没有区分 reduce axis 长度和跨 warp reduce。

7. **mixed 路由的 transition/setup 仍是旧 fallback**
   `mixed_setup_fallback` 仍来自空 VF harness，`mixedBoundaryCycles` 仍为 0。
   需要测真实 SIMD↔SIMT 切换成本。

8. **domain coverage 与 multiplier 需要重校**
   现在 multiplier 临时置 1.0。公式稳定后需要：
   - 重测/扩展 coverage 边界；
   - 重新拟合 multiplier，或证明其可移除。

9. **auto 模式端到端验证**
   目前主要验证 report 模式 raw ratio。最终要在 `auto` 模式下确认：
   - mixed scope 物化正确；
   - 实际路由延迟与模型选择一致；
   - 没有因 legality/confidence/gain gate 误拒。

10. **更多 shape/grid 的回归集**
    每个目标算子建议至少覆盖 3 个 shape × 3 条路由，并记录 grid、num_warps、
    num_stages，作为长期回归集，防止后续调参回退。

## 10. 外部真实算子第一批诊断（2026-08-19）

> 用 FBGEMM / VLLM / SGLang 的 5 个真实算子跑三条路由实测 + report JSON
> （基准提交 3b145028a，JSON/TTIR/ttadapter 在 `ascend_results/`），逐个派发
> 诊断得到的问题分类与修复清单。实测延迟单位 μs，cycle 换算
> SYS_CNT=988.9 MHz（cycles = μs × 988.9）。模型名次按 `decision_kind`
> 口径（candidate_costs 中数值最低但不可选的路线不算名次）。

### 10.1 结论表

| case | 实测名次 | 模型名次 | 模型/实测误差 | 主因 |
|---|---|---|---|---|
| silu_mul_quant (SGLang) | simt < mixed < simd | simd 最优（全错） | 三条全 floor（11000/12500/12723）；simd 低估 1.5×、simt 高估 3.7× | floor 不对称（11000<12500 与真实 simt<simd 相反）+ 循环零缩放使 analytical 被 floor 压住 + 计算索引不判 indirect |
| compute_seg_indptr (SGLang) | mixed < simd < simt | simd 最优（真最优 mixed 被排除） | 三条全 floor，比实测高 3.8-5.5× | `scf.while` 不支持 → anchor=0 → mixed inapplicable |
| silu_quantize_mx4 (FBGEMM) | simt < mixed < simd | simd 最优（全错） | SIMD 低估 7.4×（实测 82.3μs vs 模型 11μs） | 位运算完全漏计 + 死代码 bug |
| causal_conv1d (VLLM) | mixed < simd < simt | 决策 mixed（candidate_costs 里 simt_only 数值最低但不可选） | simd 160×、simt 2.75×、mixed 37× 高估 | gather 字节三重通胀 + 2.27 B/cycle 对局部性不敏感 |
| count_expert_num_tokens (VLLM) | mixed < simt < simd | 决策 mixed（simt_only 数值最低但不可选） | simd 2.2×、mixed 18.3× 高估（修复后 ~1×） | mixed 分区幽灵 gather |

### 10.2 问题分类

**A. 明确 bug（已修，见 §8.2 存档）**
- A1 mixed 分区幽灵 gather：旧公式 `regularGatherLoadBytes = regularTotal ×
  gatherRatio`，其中 gatherRatio 是**全 kernel** 的 gather 占比。anchor 分区
  已把 loaded-index-dependent load 全部移给 SIMT 段后，regular（SIMD）段仍按
  全局比例再分摊一次 gather 字节，按 2.27 B/cycle 计费——这些 gather 在
  regular 段里根本不存在，是被 anchor 付过一次账的"幽灵"。
  count_expert_num_tokens 的数字：每 CTA load 8192B = 4096B 连续（topk_ids）
  + 4096B gather（expert_map），gatherRatio=0.5。anchor 拿走全部 4096B gather
  后 regular 只剩 4096B 连续，旧代码却再计 2048B/CTA gather → 115k cycles
  （占 mixed 的 96.6%），mixed = 119487 vs 实测 6513（18.3×）。
  修复：regular 的 gather 只计超出 anchor 名下的部分：
  `anchorGather = min(totalGather, anchor.loadBytes×grid)`，
  `regularGather = max(0, totalGather − anchorGather)`。修复后 mixed ≈6089，
  与实测误差 0.94×。
- A2 gather 字节三重通胀：`loadedIndexDependentMemoryOps × maxNumel ×
  maxElementBits/8` 用了 i64 索引的 64bit、含 store ops、按满 numel
  （61440 vs 实际 15360，4×），且 > totalLoad 时 gatherRatio=1.0 全部按
  gather 计费（causal SIMD 160× 的来源之一）。修复：改用 anchor walk 计出的
  精确 loadBytes，fallback 时 bit 宽 cap 32。
  注意字节通胀与速率错误是**两层独立问题**：字节修复后 causal SIMD 仍约
  440k cycles vs 实测 4657（~100×），剩余误差来自 gather rate 本身对局部性
  不敏感（见 D）。
- A3 位运算漏计 + 死代码：`classifyWeightedOp` 无 and/or/xor/shl/shr/min/rem/
  bitcast（mx4 pack 核心 29 op 全隐形，实测 SIMD 缺口 ~70k cycles ≈ 1.2
  cycles/element-op 标量化量级）；且 `unclassifiedScalarOps = scalarOps −
  classifiedScalarOps ≡ 0` 是死代码，"unclassified arithmetic ops" 永不触发。
  修复：新增 bitop/min/rem/bitcast 类 + profile 占位 rate（SIMD 按标量化
  1 elem/cycle，待微基准）+ 用 walk 中真实登记未分类 op 计数。

**B. anchor 识别范围 < 实际混编范围**
- silu_mul：索引由 divsi/muli 计算得出（非 loaded index）→ 判 contiguous、
  anchors=0；ttadapter 实际 `mix_simd_simt` + 2 个 `@triton_indirect_load`。
- seg_indptr：`scf.while` loop-carried 索引，`pointerDependsOnLoadedIndex`
  只处理 `scf.for` 块参数 → anchors=0；实际整个二分循环 AIV 化，无模板标记。
- silu_quantize_mx4：同 silu_mul，load 实际走 `@triton_indirect_load`。
- 需扩展：① runtime 计算索引间接访存；② `scf.while` carried 依赖。

**C. min_kernel_cycles floor 不泛化（4/5 case 中招）**
- 11000/12500 按内置 5-case 拟合；seg_indptr 实测仅 2000-3300 cycles（floor
  高 3-5×）；silu_mul 三条全 floor → 排序退化为常数比较 → 无条件选 SIMD。
- floor 不仅绝对值错，还制造错误相对排序。应替换为实测 launch overhead 的
  可解释 `fixedOverhead` 项（§8.5.7 备选方案）。

**D. gather rate 2.27 缺局部性分档（causal 主因）**

间接访存不是 gather/非 gather 二值，而是局部性谱系（实测数据）：

| 模式 | 形态 | SIMD 有效带宽 | SIMT warp instr/cycle |
|---|---|---|---|
| contiguous | 相邻 lane 访问相邻地址 | 1250 B/cycle | 0.400 |
| strided s=2/4/8/16 | 固定步长 | 16.5/11.7/8.0/4.3 | 0.223/0.159/0.080/0.040 |
| **blocked-gather** | **行内连续（宽 W）+ 行偏移随机** | **未测（causal 实测 ≈268）** | 未测 |
| gather/random | 每 lane 独立随机地址 | 2.3 | 0.020 |

blocked-gather 是中间档：行内 W 个元素 stride-1（cache line 全利用），只有
行起点 data-dependent，硬件行内 coalesce → 有效带宽接近 contiguous。当前
C++ 只有 contiguous/gather 两档，blocked-gather 掉进 gather 档。

causal_conv1d 属 blocked-gather 的硬证据：实测 SIMD 全 kernel 仅 4657
cycles，而 load 总量 1.25MB 若按 2.27 B/cycle 计，光 load 一项就 550k
cycles——比整个 kernel 实测长 118 倍，物理上不可能，故有效速率必 ≈268
B/cycle。TTIR 结构佐证（`_causal_conv1d_fwd_kernel.ttir.txt:71-73`）：
`x_ptrs = tt.addptr(x_base(标量 data-dependent 偏移), make_range(256))`，
即 256 宽 stride-1 连续行，仅行起点依赖 loaded index。

count_expert 的 expert_map（仅 516B，全缓存命中，≈4.83 B/cycle）说明小
working-set 全缓存场景还需要另一个方向的分档。

修复方向：TTIR 侧用秩分解特征识别 W（pointer = `broadcast(base[行维]) +
broadcast(offset[连续维])`，仅小维 data-dependent），rate 按 W 查表/拟合；
微基准扫 W=32/128/512 + 随机行偏移。

**E. SIMT 组件仍粗**
- predicate 虚高：silu_quantize_mx4 中 predicate=8964 cycles > 实测全核 4450。
- gather/contiguous 二选一太粗：causal SIMT 侧改用 contiguous rate 后误差
  仅 +1.3%（真实访问是"行连续"）。
- binary search 串行依赖标量 load 延迟（~200 cycles/轮）无模型。注意：这只
  影响 simt_only 的绝对估值（实测 3.36μs vs 模型 floor 12500，且实测
  simt_only 比 simd 慢 1.65× 的方向模型对不上量级）；mixed 实际未 SIMT 化
  （见 B），此缺口优先级低。

**F. gate/校准链路**
- count_expert：covered 但 `has_unknown_trip_count` → `selection_score_invalid`
  → auto 模式回退最慢 SIMD，丢 16.7×（模型排名其实对）。
- mixed transition 只收 setup fallback 223 cycles，实测 ~2000。
- dynamic_loop_elementwise 域的 m_simd=67.52 在 profile source 文案里有实测
  推导但未填值（验证阶段有意置 1）。

### 10.3 优化优先级

- **P0（明确 bug）**：A1 mixed 幽灵 gather；A2 gather 字节估计；A3 位运算
  分类 + 死代码。
- **P1（特征/公式扩展）**：B anchor 扩展（计算索引 + scf.while）；D blocked-
  gather 分档；C floor → fixedOverhead；E predicate 指令数细化；unknown-trip
  循环 trip 代理。
- **P2（校准/验证）**：mixed transition 实测；domain multiplier 重校 + gate
  覆盖；auto 模式端到端。

### 10.4 需要的微基准

| 微基准 | 目的 | 状态 |
|---|---|---|
| SIMD i32/i16 位运算吞吐（and/or/xor/shl/shr/bitcast，扫 n） | 测标量化长度，校准 A3 占位 rate | `microbench_simd_bitops.py` 已写 |
| blocked-gather：行内 stride-1 + 行偏移随机，扫行宽 32/128/512 | 拟合 gather 局部性分档（D） | `microbench_blocked_gather.py` 已写 |
| 动态界循环 per-trip 成本（trip=1/10/100/1000，间接 load tile） | unknown-trip 的 trip 代理 | `microbench_dynamic_loop.py` 已写 |
| SIMD strided 形态（rope interleave / column / strided store / general） | strided 特征识别 + `strided_power_fit` 接入 | `microbench_simd_strided_patterns.py` 已写 |
| SIMD/SIMT device launch overhead | 替换 floor（C）→ `fixedOverhead(grid, warps)` | `microbench_launch_overhead.py` 已写 |
| mixed transition 方向性实测 | 替换 setup fallback 223 | 暂不需要新脚本：外部 case 已有数据（mx4 差值 ≈2000、count_expert 被吸收），先修好 anchor 成本再用残差重估 |

camodel 侧：predicate 的 mode/warps 查表数据已齐（4 mode × 6 warps），等
特征侧能区分 mask 类型后接入，无需新测。

### 10.5 决策链路关卡全景（2026-08-19 复核）

按代码顺序（`SelectSimdSimtCostModel.cpp` + `SimdSimtCostModel.cpp`）：

**① 候选合法性**（cpp:2397-2407）
- all_simd legal：`kernelLowerability.allSimd == Native`（无 anchor 时默认 Native）。
- all_simt_only legal：`compileOn91095 && !hasExplicitScope && allSimtOnly ==
  Native`。有 anchor 时 anchor 分析将其降为 `BackendConditional`
  （SimtAnchorAnalysis.cpp:304-309，"纯 SIMT 需整 kernel 后端验证"）。
- mixed legal：`!hasExplicitScope && materializable && mixed == Native`。
- 提升条款（cpp:2452-2458）：BackendConditional 的 simt_only 仅在「covered
  && 该域 `all_simt_only_validated=true`」时升为可选；域
  `mixed_simd_simt_validated=false` 则 mixed 降为不可选。
- 设置原因：`compileOn91095` 表示目标是否支持 SIMT 物化；`hasExplicitScope`
  防覆盖用户手写 scope；validated 标志要求纯 SIMT 路线在该域有实测正确性
  证据才放行。
- 评价：方向合理，但过保守——count_expert 因 uncovered（unknown_loop_trip_
  count）拿不到提升，simt_only 被排除，丢 16.7× 收益；且 validated 是域级
  二元标志，域外同构 kernel 一律不享。

**② Coverage 闸**（cpp:2419-2427, 2462-2469）
- 7 个特征域分类；未覆盖且 auto 模式 → 提前返回、`selection_score_invalid`、
  分数为 null；report 模式（`scoreOutsideCalibrationCoverage=!autoMode`）继续
  算分。
- 设置原因：分数是域内校准的，域外不让 auto 决策。
- 评价：auto 模式合理；report 模式放行是诊断设计，合理。

**③ Unsupported 条款**（cpp:2494-2528）
- gather/histogram/atomic 核心成本未校准、未分类算术 op → 进 unsupported →
  ranking_confidence=none。
- 设置原因：有不知道成本的工作就不应自信排序。
- 评价：合理，但死代码 bug（A3）曾使它形同虚设，已修。

**④ 置信度闸**（cpp:3091-3111）
- rankingConfidence 取各组件 profile confidence 最小值（当前全是 "low"）；
  低于 `minimum_confidence_for_decision`（profile policy 设 "low"）才拒。
- 评价：形同虚设——所有 rate 都是 low、最小要求也是 low，恒过；实际决定权
  全在 coverage/unsupported。

**⑤ Gain/margin 闸**（cpp:3074-3118）
- 非 all_simd 决策要求 `decisionAdvantage > max(64, baseline×0.10)`，否则
  auto 模式安全回退 all_simd（Select:163-172）。
- 设置原因：切换路线有正确性/验证风险，要求 ≥10% 或 64 cycles 优势。
- 评价：合理的安全边际；64 的绝对 floor 相对真实成本（万级 cycles）偏小，
  实际由 10% 生效。

**⑥ 应用环节**（Select:135-207）
- 仅 auto && gatePassed && actionSupported 生效；gate 失败（非 gain 原因）→
  `backend_default`，回退给 legacy 后端启发式（大概率 SIMD），**不是
  all_simd**。
- 评价：与⑤"safe baseline"哲学不一致——gate 失败时应保持已验证基线而非
  回到旧启发式。count_expert 的生产隐患所在。

**⑦ 物化**（Select:203-207）
- effective==mixed 才调 `materializeSimtAnchorPlan`。

### 10.6 P0 修复后外部算子重验（2026-08-19，ef3f7628a + 526533d6c）

实测延迟沿用 §10.1 同组（μs）；模型分数来自
`ascend_results/test_*_ef3f7628a.json`。

| case | 修复前模型 | 修复后模型 | 实测（cycles） | 判定 |
|---|---|---|---|---|
| silu_mul_quant | 11000/12500/12723 | 不变 | 16243/3422/3868 | 符合预期（P1 遗留：floor + 计算索引 anchor） |
| compute_seg_indptr | 11000/12500/12723 | 不变 | 2007/3323/1899 | 符合预期（排序本就正确，仅 floor 绝对值） |
| silu_quantize_mx4 | 11000/12500/12723，simd 最优 | **27252/12500/27475，决策 simt_only ✓** | 81386/4450/6428 | 排序翻转正确；simd 绝对值仍低估 3×（占位 factor 96 待微基准）；unsupported 曾报 1 条 `math.exp2`，已分类归入 exp（526533d6c） |
| causal_conv1d | 746165/17216/162894 | **590314/17216/17532** | 4657/6250/4414 | 字节+幽灵 gather 修复生效（simd −21%、mixed −89%）；剩余 simd 127×、mixed 4× 为 P1 速率分档遗留 |
| count_expert_num_tokens | 235094/12500/119487 | 235094/12500/**6060** | 108563/7147/6513 | **mixed 0.93× 完美命中**；simd 2.17× 为 gather rate + unknown-trip 遗留 |

关键结论：
1. P0-1（幽灵 gather）由 count_expert 完全验证（119487→6060，实测 6513）；
2. P0-2（字节通胀）由 causal 部分验证（simd 746k→590k），剩余误差全在
   gather rate 局部性（P1）；
3. P0-3 安全网首次生效并抓到真实缺口（math.exp2），且位运算计费把 mx4 的
   排序翻转成实测一致；
4. 下一批修复建议顺序：D blocked-gather 分档（causal 主因）→ 位运算微基准
   校准 factor → floor → fixedOverhead。

### 10.7 微基准分析与落地（2026-08-20，4 组数据）

数据：`ascend_results/{blocked_gather,launch_overhead,bitops,dynamic_loop}_{simd,simt}_microbench.jsonl`
+ 扁平命名的 `ttir_*` 文件。

**blocked-gather（已落地 ebd22b699）**
- SIMD read 速率 W 曲线：W≤128 严格线性 2.272×W（r²>0.9999），W=256→507、
  W=512→1204（≈contiguous cap）；causal_conv1d（W=256）用 507 重算 SIMD
  ≈5k cycles，与实测 4657 吻合。
- W 识别：loaded-index-dependent load 指针链上、非 loaded 侧的
  `make_range end` 取 max（causal 的 256 是连续维、2 是行数维；纯 gather 如
  indirect_elementwise 链在 loaded 边界即断 → W=0 → 常数不变，5-case 安全）。
- SIMT 侧微基准 warp/cyc 口径与 cce 不可比（SIMT 路由可能向量化 load），
  SIMT 分档暂缓，需要 cce 补测 blocked 档。

**launch overhead（待落地，需决策）**
- SIMD：grid≤1024 平台 ≈10800 cycles，4096 翻倍 → cores_per_wave≈2048，
  拟合 `10668×ceil(grid/2048)+146`；SIMT 全段平 ≈11920（无爬坡）。
- 5-case（launch-inclusive 测法）可安全删除 floor（fixedOH≤实测恒成立）；
  但 seg_indptr（kernel-only 测法，实测 2007 < 任何 base）会超 5.4×，且删
  floor 后其 simd/simt 排序会翻——根因是 simt 标量串行 load 欠模（E 类），
  需先修 E 或保留小 floor 过渡。mixed 是否加 base 有矛盾：count_expert
  mixed 6060≈实测 6513 说明无 base 才对。建议 profile 加
  `fixed_overhead_scope` 开关（launch-inclusive/kernel-only）。

**bitops（已落地 ebd22b699）**
- **96× 标量化假设被证伪**：实测 factor SIMD bitop≈0.87、min≈0.88、
  rem≈1.47、bitcast≈0.92（16M 档，ALU 上界）；SIMT ≈1.1。profile 改为
  1.0/1.0/1.5/1.0，confidence medium。
- mx4 验证：新 factor 下 SIMD 分回 9096→floor 11000，**排序会再翻错**——
  说明 mx4 的 72k 残差不是 bitop 吞吐，是循环依赖链/控制开销；trip 代理
  落地前 mx4 排序依赖 factor 96 的"歪打正着"，需与 trip 代理一起回归。

**dynamic loop（trip 代理依据，待落地）**
- 纯连续 silu loop：SIMD 75 cyc/trip vs SIMT 415（SIMD 反而便宜）；
  **计算索引（divsi）loop：SIMD 8456 cyc/trip vs SIMT 416——SIMD 崩坏
  110×，0.51 cyc/elem**。silu_mul 的 5243 cyc gap ≈ 0.6 个微基准 trip，
  量级自洽。
- trip 代理高价值场景是"SIMD × 计算索引 loop"；纯连续 loop 收益小。
- 方案：JIT 传 tensor numel（同 launch_grid 链路）→ C++ 按 offset 结构算
  per-CTA trips；需同步扩展 `pointerDependsOnLoadedIndex` 识别 divsi 计算
  索引，否则 per-trip 成本分档失效。

### 10.8 trip 代理落地与 fixedOverhead 搁置（2026-08-20）

**trip 代理（已落地 13c8df933）**
- 链路：`jit.py` 把最大 tensor numel 塞入 `launch_numel`（与 launch_grid 同
  链路）→ `compiler.py`/`triton_ascend.cc`/`Passes.td`/pass 选项 →
  `analyzeSimdSimtFeatures`。
- 估计：仅对 runtime-bound 且 body 内 load/store 指针依赖循环块参数的
  scf.for 生效；trip = ceil(numel/bodyMaxNumel)，body 依赖 program_id 时
  再除以 grid（per-CTA 分区语义）。
- 安全性：5-case 无 unknown loop → 零影响；causal 的 segment loop 地址是
  常数（状态累积）→ 不估；count_expert body 用 pid → ÷grid → 1 → 不变；
  mx4 numel≈bodyNumel → 1 → 不变。唯一实质变化是 silu_mul 类 tile 循环。

**fixedOverhead 搁置（launch_overhead 数据自相矛盾）**
- 微基准（背靠背 N 次 launch 均摊）测得最小 SIMT kernel ≈11920 cycles 全
  grid 平，SIMD ≈10800 + grid>1024 翻倍（cores/wave≈2048）。
- 但 silu_mul/seg_indptr/mx4 的实测 SIMT 只有 3422/3323/4450 cycles——
  **比最小空 kernel 还快 3 倍**，物理矛盾。解释：这些 kernel 的 simt_only
  走了 whole-body-void-SIMT-scope inline 快速路径（compiler.py），空 kernel
  与 causal/count_expert 走 runtime-loop 序言慢路径。SIMT 固定开销是双模的，
  单常数 fixedOverhead 会继续高估快速路径 3.5×。
- 下一步确认测量：`microbench_launch_overhead_v2.py`（be6a07ef6）扫
  kernel 形态（scalar/vector/silu/indirect）× grid × num_warps，且同时
  测 back-to-back 与 single-launch 两种口径；拟合
  `fixedOverhead(shape_class, grid, num_warps, route)`，shape_class 映射到
  C++ 特征（hasIndirectMemory→indirect、tensor load→vector/silu、
  否则 scalar）。数据回来前保留 min_kernel_cycles floor。

### 10.9 launch overhead v2 分析与外部 harness 疑点（2026-08-20）

**v2 数据（`launch_overhead_v2_{simd,simt}.jsonl`，back-to-back，cycles）**
- SIMD scalar：12900 平（grid≤1024），4096 翻倍（+8500/波，cores/wave≈2048）；
  vector/silu：13800 平、无爬坡（tensor kernel 不翻倍）；indirect 工作主导。
- SIMT：全 shape 12800-14000 平、无爬坡；大 grid 低 warps 是工作主导。
- single-launch ≈ back-to-back + ~8500（launch gap 被背靠背摊掉）。

**系统性矛盾**
- 外部 harness 的 5 个算子 SIMT 实测 3323-7147 全部 < SIMT 最小 launch 12700；
  seg_indptr simd 2007 / causal simd 4657 < SIMD 最小 12257——物理不可能
  （launch-inclusive 口径）。
- 5-case harness 的 SIMT 实测 11755-15000 全部 ≥ 下限，与微基准自洽。
- 结论：外部 harness 疑似 kernel-only 计时（Event 未包 launch gap 或缺 sync），
  "SIMT 快速路径 3.4μs"很可能是测量伪影。
- 决定性测试：用 single-launch 口径重测 seg_indptr（见本文件外的复测说明）。
  若变 ~13μs → 旧数据作废、floor 保留（已 ≈ 实测最小开销）；若仍 3.36μs →
  按 codegen 路径分档实现 fixedOverhead。
- 另注：现有 floor 已 ≈ v2 实测最小开销，且 dot/scan 组件级 floor 本身含
  launch 成分，直接替换为加性 fixedOverhead 会双重计费，需先剥离组件
  floor 的 launch 部分。
