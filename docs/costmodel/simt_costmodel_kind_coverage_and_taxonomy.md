# StageCostModelKind 覆盖分析与分类重组

> 配套文档：《simt_costmodel_structured_design-v2.md》（8 族 20 类定义见其 2.6 节；注：v1 设计文档已废弃删除）
>
> 本文内容：
> 1. 现状 8 族 20 类的逻辑复盘（含每个类的判定依据）
> 2. 现状分类"逻辑性不够"的三个根因
> 3. 分类方法论：三个公理 + 一条判定树
> 4. 28 个目标算子（5 仓库）的逐算子覆盖分析：未纳入已有范式的代码段
> 5. 未覆盖模式汇总
> 6. 重组方案：7 族新体系，每族每类的划分逻辑、合并与新增依据
> 7. 与现状的对照迁移表 + 落地建议

---

## 1. 目标与结论摘要

本文回答三个问题：

1. **现状的 8 个语义族 / 20 个 Kind 各自是什么意思、判定依据是什么**（第 2 节）。
2. **目标真实算子（FBGEMM / VLLM / SGLang / LigerKernel / FlagGems，共 28 个 kernel）中有哪些代码段不属于已有范式**，具体到"哪个算子哪段代码、为什么"（第 4、5 节）。
3. **现有分类是否合理、怎么重组更有逻辑层次**（第 6 节）。

核心结论：

- 现状 8 族 20 类的**骨架方向正确**（模式与语义正交、强结构优先），但存在三个逻辑缺陷：
  - **分类轴不一致**：有的族按"主导资源"分（Memory/Cube），有的按"依赖结构"分（Recurrence/Independent），有的按"调度来源"分（Dispatch），三个轴混在同一层枚举里；
  - **粒度不均匀**：Scalar 族 6 个 Kind 共享同一成本公式，Continuous 族 4 个 Kind 共享同一公式，按"主成本公式"标准应当合并；
  - **整族缺席**：Scan（前缀和）、Coordination（原子/屏障/跨 CTA 收尾）、间接写（scatter store/置换表）、位操作与寄存器重排、Ragged——这些在 28 个真实算子中高频出现，20 类里完全没有轴。
- 重组建议：**族 = 主成本公式形状（7 个），类 = 族内主导资源/模式形态，结构修饰降级为属性 flag**，分类顺序按"约束强度"固定为一条判定树（第 6 节）。

---

## 2. 现状：8 个语义族 / 20 个 Kind 的逻辑复盘

现状定义（来自设计文档 3.5.2）：`StageCostModelKind` 回答"这段代码在做什么"，
不指定 SIMD/SIMT；与 `StageMode` 正交，构成 Registry 键 `(StageMode, StageCostModelKind)`。

| 语义族 | 族的隐含划分轴 | Kind | 判定依据（识别什么结构） |
|---|---|---|---|
| **Dispatch** | 调度来源（编译器注入） | `auto_blockify_dispatch` | V1 为每个物理 program 生成的 PID/chunk/边界 setup |
| | | `auto_blockify_loop` | V1 生成的 logical-program 聚合循环（非算法循环） |
| **Scalar** | 标量工作的产出物 | `scalar_issue` | 普通标量发射块（无分支/访存/向量） |
| | | `scalar_control` | early return、条件选择、控制转移主导 |
| | | `scalar_math` | 标量算术/SFU（`1/max`、`rsqrt` 等） |
| | | `index_generation` | tile offset/stride/div-rem/地址索引生成 |
| | | `predicate_mask` | 非循环的 compare、边界 mask 生成与应用 |
| | | `loop_predicate` | 随算法循环迭代变化的谓词、退出条件、backedge |
| **Continuous Memory** | 访存形态（可证明连续） | `continuous_tile_memory` | layout 合并后可证明连续的 tile load |
| | | `continuous_tile_store` | 连续 tile store |
| | | `continuous_short_load` | 连续但量小、启动延迟不可忽略的 load |
| | | `cache_policy_store` | 带 cache modifier（`.cg` 等）的 store |
| **Indirect Memory** | 访存形态（地址依赖运行时） | `indirect_scalar_memory` | 数据相关地址的标量 load/store |
| | | `indirect_gather_memory` | 多 lane 地址离散的 gather/scatter |
| **Independent Pipeline** | 依赖结构（无跨迭代依赖） | `independent_pipelined_loop` | 无真实 loop-carried 数据依赖且可重叠的循环 |
| **Recurrence / Reduction** | 依赖结构（跨 lane/跨迭代） | `loop_carried_recurrence` | 第 i 轮消费第 i-1 轮数据（pointer induction 除外） |
| | | `rowwise_reduction` | 沿维度的 tree/serial/shuffle 归约 |
| **Cube / Tiny Cube** | 计算类别（tensor core） | `cube_roofline` | 规则 `tt.dot`，有效工作量覆盖 setup |
| | | `tiny_cube_roofline` | 小 shape/不完整 tile，setup 与 underfill 不可忽略 |
| **Conversion / Pack** | 计算类别（值空间转换） | `conversion_pack` | dtype convert、quantize、pack/unpack 主导 |

现状分类优先级（设计文档 2.6）：V1 provenance → recurrence/reduction → dot → indirect
memory → continuous memory → conversion/pack → scalar/control。

---

## 3. 现状分类"逻辑性不够"的三个根因

### 3.1 根因一：分类轴混用，同一层枚举里存在三个互不正交的轴

看现状的"族"：

| 族 | 实际使用的划分轴 |
|---|---|
| Dispatch | **调度来源**（V1 注入 vs 手工） |
| Scalar / Continuous / Indirect / Cube / Conversion | **主导资源 / 工作类别**（标量、访存、Cube、转换） |
| Independent Pipeline / Recurrence / Reduction | **依赖结构**（无跨迭代依赖 / 有跨迭代依赖 / 跨 lane 合并） |

同一层上同时用三个轴分类，必然产生边界模糊：
- `independent_pipelined_loop`（依赖结构轴）与 `continuous_tile_memory`（资源轴）——一个"独立循环 + 连续 load"的 Stage 该归谁？
- `loop_carried_recurrence` 与 `scalar_math`——一个"循环携带的标量更新"该归谁？
- 现状文档 3.5.6 的解法是"分类优先级"和"requires_split"，但优先级本身没有逻辑推导，是拍脑袋的顺序。

**逻辑上必须修正为：分类优先级由"约束强度"推导，而不是经验顺序。**

### 3.2 根因二：粒度不均匀——Kind 与成本公式脱钩

设计文档 3.5.6 明确：Scalar 族全部 6 个 Kind 在 SIMD/SIMT 两侧都映射到**同一个公式**
（`C_scalar+C_compute+C_predicate+C_control+C_issue+C_spill`）；Continuous 族的
`continuous_short_load`、`cache_policy_store` 也共享连续访存公式，只是参数不同。

按设计文档自己的原则——"**一个候选同时命中两个需要不同主公式的强结构才 requires_split**"——
反过来说，**两个 Kind 若共享同一个主成本公式，就不构成独立 Kind**。差异应降级为参数/属性：

| 现状 | 实为 | 处置 |
|---|---|---|
| `scalar_issue` / `scalar_math` / `index_generation` | 同一标量公式的三个标签 | 合并，差异降为产出物 sub-label |
| `continuous_tile_memory` / `continuous_short_load` | 同一访存公式 + tile 大小阈值 | 合并 + `short_tile` flag |
| `continuous_tile_store` / `cache_policy_store` | 同一访存公式 + cache modifier | 合并 + `cache_policy` flag（并补 load 侧） |
| `cube_roofline` / `tiny_cube_roofline` | 同一 Cube 公式 + underfill 连续量 | 合并 + `underfill_ratio` flag |

### 3.3 根因三：整族缺席——四个真实世界高频语义没有轴

28 个算子比对后确认，以下语义在现有 20 类中**没有任何归属**（第 5 节给出全部证据）：

| 缺失语义 | 出现频次（目标算子） | 典型证据 |
|---|---|---|
| **Scan / 前缀和**（含跨 block carry） | 12+ 处 | FlagGems cumsum 全家族、masked_select、FBGEMM fused_padding_cumsum、SGLang seg_indptr（兄弟 kernel 用 `tl.cumsum`） |
| **Coordination**（原子计数、屏障、last-CTA 收尾、设备→主机交接） | 5+ 处 | FlagGems masked_select L76-86、SGLang ep_moe `_fwd_kernel_ep_scatter_2` L692、FBGEMM `tl.debug_barrier()` L2798 |
| **间接写 / 索引表**（scatter store、置换表构建、LUT 查表、页表间接、指针追逐） | 10+ 处 | FlagGems masked_select L26、SGLang deepep/deepgemm src2dst、FBGEMM dequantize L552、VLLM decode_attention 页表 |
| **位操作 / 寄存器重排 / RNG** | 6+ 处 | LigerKernel int2 unpack、FBGEMM mx4 量化核心、`tl.interleave`/`tl.split`/`tl.join`、`tl.randint4x` |
| **Ragged / 动态形状**（运行时 per-segment 边界、动态 trip count、环绕寻址） | 8+ 处 | FBGEMM array_jagged_bmm、VLLM lora_expand、SGLang silu（运行时 `num_tokens`） |

---

## 4. 分类方法论：三个公理 + 一条判定树

### 4.1 公理（重组的逻辑基础）

> **公理 A（公式轴）**：两个 Kind 存在，当且仅当二者在主成本公式形状上可区分
> （串/并行约束不同、资源流水不同、依赖类型不同）。公式相同则合并为一个 Kind，
> 差异降级为 `StageModelFeatures` 属性 flag。
>
> **公理 B（穷尽互斥）**：分类是"一次只问一个问题"的判定树，每个 Stage 恰好命中一个
> Kind；命中两个强结构的 Stage 返回 `requires_split`。
>
> **公理 C（正交性）**：三个维度互不越界——
> - 语义族/类 = 主公式形状与主导模式（本文第 6 节）；
> - `StageMode` = SIMD / SIMT（已有）；
> - 结构修饰 = 属性 flag（cache policy、动态 trip count、精度、underfill、ragged
>   边界、角色、语义填充……），只影响合法性判定与公式参数，不进入枚举。

### 4.2 判定树（由约束强度推导的分类顺序）

分类顺序不是经验表，而是按"**该结构对关键路径的约束强度**"排序——
约束越强（越无法被并行/隐藏）越先判定：

```
1. 是否有跨 program / 跨 CTA 协作原语（atomic / barrier / 合作计数）？
       是 -> Coordination 族
       （全局语义不能被 Stage 内局部公式覆盖，最强约束）
2. 是否主要是调度结构（V1 循环 / 手工持久化 / 角色分发）？
       是 -> Dispatch 族
3. 是否存在跨 lane / 跨迭代合并（reduce / scan / recurrence / carry / 流式归一化）？
       是 -> Dataflow 族
       （决定公式是 critical_path×N、log 步还是 tree，先于资源判定）
4. 关键路径上主导资源是访存还是计算？
       访存主导 -> 按地址可证明性 -> Memory 族（连续 / 间接 / ragged）
       计算主导 -> 是否 tt.dot -> Matrix 族；否则 -> Compute 族
5. 兜底：上述都不主导，主要是 head/setup 工作 -> Index / Scalar / Control 族
```

该顺序与现状文档 2.6 的差异：**把 Coordination 提到最前（现状完全没有）、
把 Dataflow（含 scan）提到 dot 之前**。理由：目标算子中原子/扫描/置换表决定
关键路径的 Stage，若排在 dot 之后会被错误归类到 Cube 公式。

---

## 5. 28 个目标算子的逐算子覆盖分析

> 仓库与算子清单见任务说明；以下按仓库给出"不属于已有 20 个 Kind 的代码段"。
> 行号为分析当日（2026-08-21）所读源码行号。
> "部分命中"指现状 Kind 能描述其中一部分，但整体无法用单个 Kind 表达（需要 arbitrary split）。

### 5.1 FBGEMM

#### 5.1.1 `_kernel_dequantize_mx4`（fbgemm_gpu/triton/quantize.py:481）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L548-549 | `low_mx4 = a & MX4_BIT_MASK; high_mx4 = (a >> 4) & MX4_BIT_MASK` | 无 dtype 转换的纯位域提取，且结果喂**地址**（LUT 索引），不是 `conversion_pack` 的值空间转换，也不是 `index_generation`（非地址算术） | 位操作 `bit_unpack` |
| L552-553 | `low_fp32 = tl.load(mx4_lookup_table + low_mx4)` | 机制上是 gather，但这是 16 项均匀小表、每 program 同表、L1 驻留、可广播的查表；`indirect_gather_memory` 假设大地址空间离散 gather，成本画像完全不同 | 索引表 `lookup_table_gather` |
| L567 | `scale = tl.exp2(exp.to(tl.float64)).to(tl.float32)` | fp64 超越函数在硬件上模拟/吞吐骤降（注释自述 "This might be slow"），`scalar_math` 隐含 fp32 SFU | 超越函数（精度 flag） |
| L572 | `scaled_fp32 = tl.interleave(scaled_low_fp32, scaled_high_fp32)` | 寄存器内 lane 交织，非 dtype 转换、非索引、非 reshape | 寄存器重排 `in_register_rearrange` |
| L530-537 + L572 | 跳过共享指数槽的 offset 构造 + 写侧 interleave | 打包布局的双向映射（地址端+寄存器端共同反转 MX4 交错存储），会被强制拆成两个 Stage | 打包布局映射 |

#### 5.1.2 `_kernel_silu_quantize_mx4_unpack`（experimental/gemm/triton_gemm/fp4_quantize.py:401）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L589-614 | `philox_4x_offset = tl.split(tl.reshape(input_offset, [...])); a_4x,b_4x,c_4x,d_4x = tl.randint4x(group_rand_bits, philox_4x_offset, n_rounds=7)` | **Philox 伪随机数生成**（随机舍入用），20 类中 RNG 完全缺席 | `random_number_generation` |
| L620-672 | 符号/指数/尾数提取、subnormal 分支、舍入、溢出修正、`\|` 重构（约 20 个 shift/mask/where） | bit-serial 浮点格式转换，与廉价 dtype cast 成本差异巨大；`conversion_pack` 会吞掉但失去区分度 | 位操作 `bitmanip_convert` |
| L589-614, L675-677 | `low_mx4, high_mx4 = tl.split(tl.reshape(mx4_value, [(GROUP_LOAD*GROUP_SIZE)//2, 2]))` | `tl.split`/`tl.join`/`reshape(can_reorder=True)` 是寄存器内 lane 重排 | `in_register_rearrange` |
| L534-562 | `tl.max(tl.abs(a_groups), axis=1)` + reshape→broadcast 除→reshape 回 | 隐藏的寄存器布局往返（每个往返一次重排）未建模 | 寄存器重排（间接） |

#### 5.1.3 `fused_single_block_kernel`（fp4_quantize.py:2739）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L2761, L2787 | `cumsum = tl.cumsum(m_sizes, axis=0)` | **scan/前缀和**：等长输出、lane 间 log 步依赖；`rowwise_reduction` 是塌缩到标量，`loop_carried_recurrence` 是跨迭代，scan 是单语句内跨 lane | `rowwise_scan` |
| L2806-2824 | `for _ in range(64): mid=(left+right)//2; mid_val=tl.load(size_cumulative_ptr+mid,...); left=tl.where(mid_val<=row_idx, mid+1, left)` | 迭代内 load 地址依赖上一轮比较结果（dependent gather+compare+select 循环）；不是一次性 index 生成，不是数据表 gather | 搜索类索引生成 |
| L2798 | `tl.debug_barrier()` | 跨 CTA 显式同步，20 类无 inter-program sync | `inter_program_sync` |
| L2800-2801 | `for start in range(0, N, BLOCK_SIZE*NUM_BLOCKS)`（256-block 持久化） | 手工持久化 grid-stride 调度，非 V1 注入 | `persistent_grid_stride` |
| L2754-2766 | 每 block 冗余重复计算同一 cumsum | 用"重复计算广播"代替跨核同步的设计语义 | 协调（compute-broadcast 变体） |
| L2775 | `if pid == 0:`（单 program 特化） | program 角色分工，非数据分支 | `program_role_dispatch` |

#### 5.1.4 `fused_padding_cumsum_and_segmented_arange_kernel`（fp4_quantize.py:2872）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L2897 | `cumsum = tl.cumsum(padded_sizes, axis=0)` | scan（同 5.1.3） | `rowwise_scan` |
| L2915-2924 | `for _ in range(32): ...`（注释 "32 iterations for binary search"） | 二分搜索（32/64 次迭代为编译期调参，证明是独立算法结构） | 搜索类索引生成 |
| L2886-2908 | `if pid == 0:` 只有首 block 算 padded cumsum | pid==0 特化 / straggler | `program_role_dispatch` |

#### 5.1.5 `array_jagged_bmm_kernel`（sll/triton/triton_jagged_bmm_jagged_out.py:15）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L39-53 | `batch_offset_am = tl.load(a_offsets_ptr + pid_batch); batch_K = tl.load(b_offsets_ptr+pid_batch+1) - batch_offset_bk; batch_M = tl.minimum(batch_M, max_seq_len)` | 基址、tile 范围、合法性全部来自**运行时 per-segment offset 对**，空 segment 需 early exit；不是静态 affine（`index_generation`），不是数据离散 gather | `jagged_ragged_memory` |
| L64-65 | `offs_am = (pid_m*BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)) % batch_M` | **运行时模数环绕寻址**：tile 是变长 segment 的环形切片，跨 batch lane 别名需 store mask 抑制 | ragged（环绕子模式） |
| L79-88 | `for k in range(0, tl.cdiv(batch_K, BLOCK_SIZE_K)):` + `mask=offs_k[None,:] < batch_K - k*BLOCK_SIZE_K` | 循环 trip count 与 mask 依赖**运行时标量**，动态 trip count 破坏 `independent_pipelined_loop` 的静态界假设 | 动态 trip count（属性） |
| L86 `tl.dot` | 本身 → `cube_roofline` 覆盖良好 | — | — |

#### 5.1.6 `_rope_padded_kernel`（experimental/gen_ai/test/kv_cache/rope_padded.py:53）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L196-200 | `freqs = seq_pos * pow(theta, powers/(-dim)); sines = tl.sin(freqs); cosines = tl.cos(freqs)` | **寄存器内合成三角函数**（pow+sin+cos 链，运行时指数），不是预计算表 gather；`scalar_math` 的 SFU 假设是标量单次 | 超越函数合成 |
| L142-143, 182-190, 211-219 | `is_q = head_idx < k_start; is_v = head_idx >= v_start` → 三路分别读/写 out_q / cache_k / cache_v | **程序角色分发**：分支条件来自 program_id 范围、全 program 均匀、角色互斥、访问不同张量；`scalar_control` 只描述数据分支 | `program_role_dispatch` |
| L135-158 | `query_pos = ... + tl.load(seqstartq+...); cache_pos = end_of_batch_elt_cache - (end_query_pos - query_pos)` | ragged 批量 + **反向 segment 写**（写入 cache 区最后几格） | `jagged_ragged_memory`（反向写子模式） |

### 5.2 VLLM

#### 5.2.1 `_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1`（v1/attention/ops/triton_decode_attention.py:59 / 249）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L114-121, L321-328 | `kv_page_number = tl.load(Req_to_tokens + ... + offs_n // PAGE_SIZE); kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE` | **两级页表间接**：先 load 页表项再派生页内连续块；页内连续可合并 transaction，页间离散——`indirect_gather_memory` 不表达"页内连续"结构 | `page_table_indirection` |
| L132（仅 stage1） | `qk = tl.sum(q[None, :] * k, 1)` | **无 tl.dot 的手工外积+归约**（head dim 小、num_warps=1/2）；`tiny_cube_roofline` 预设 tt.dot，此段无 tensor core | `manual_dot_reduction` |
| L151-158, L372-378 | `n_e_max = tl.maximum(tl.max(qk,0), e_max); re_scale = tl.exp(e_max-n_e_max); acc *= re_scale; acc += tl.sum(p[:,None]*v,0); e_sum = e_sum*re_scale + tl.sum(p,0)` | **流式归一化（online softmax）**：循环携带 e_max/e_sum/acc + 跨迭代 rescale + exp，是"recurrence+reduction+SFU"融合体；单独分给 `rowwise_reduction` 或 `loop_carried_recurrence` 都会丢失 rescale-exp 依赖 | `online_softmax` |
| L180-183 | `e_max + tl.log(e_sum)` 作为标量随 tile 存出（LSE） | LSE 导出与 tile 并行，属 online_softmax 附属 | 同 `online_softmax` |

#### 5.2.2 `sample_recovered_tokens_kernel`（v1/sample/rejection_sampler.py:773）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L822 | `recovered_id = tl.argmax(prob / q, axis=-1)` | **arg-reduction**：带索引追踪的 compare-select 归约，与 sum/max 的公式（无索引维护）不同 | arg 归约（Dataflow 子类） |
| L785-787 | `start_idx = ... tl.load(cu_num_draft_tokens_ptr + req_idx - 1)` | 每请求区间来自 cumsum 数组 → ragged 区间 | ragged 边界（属性） |

#### 5.2.3 `_bias_kernel`（v1/worker/gpu/sample/logit_bias.py:148）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L184-189 | `allowed_token_ids = tl.load(allowed_token_ids_ptr + req_state_idx*stride + block); logits = tl.load(logits_ptr + batch_idx*logits_stride + allowed_token_ids, mask=mask)` | **允许集索引表**：数据相关离散 load（允许集合）→ 后续 scattered 回写；"索引表掩蔽+双向 scatter"无单 Kind | 索引表 + 间接写 |

#### 5.2.4 `_causal_conv1d_fwd_kernel`（model_executor/layers/mamba/ops/causal_conv1d.py:16）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L411-452 | `for idx_token in range(segment_len): acc = acc_preload; for j in tl.static_range(KERNEL_WIDTH): acc += matrix_x * matrix_w; col0 = col1; col1 = matrix_x` | **滑窗移位寄存器递推**：逐 token 串行 + 宽度 K 的窗口状态跨迭代携带；`loop_carried_recurrence` 部分命中但状态是"窗口+stencil"而非单值 | 滑窗递推（Dataflow 子类） |
| L151-307 | conv_state shift-left 搬移 + `tl.where` 合并 + 写回（L270-280） | **状态缓存维护**（shift+merge+store 一体），且有 L234/269/383 三处 `tl.debug_barrier()`（注释明说编译器 bug 需要 barrier） | `inter_program_sync` + 状态缓存维护 |
| L316-338 | `cache_modifier=".ca"` 的 load | 只有 `cache_policy_store`，**load 侧 cache policy 缺失** | `cache_policy_load`（属性） |

#### 5.2.5 `fused_moe_kernel`（model_executor/layers/fused_moe/fused_moe.py:312）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L398-447 | `offs_token = tl.load(sorted_token_ids_ptr + offs_token_id); off_experts = tl.load(expert_ids_ptr + pid_m)`（-1 → 写零 early return） | **按 expert 分组的 token 路由**：token 由排序表给定、expert 由查表给定、块边界依赖运行时数据；静态 `index_generation` 与数据 gather 都不描述"分组连续区间" | 分组路由（ragged/索引表） |
| L449 | `offs_bn = (pid_n*BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N).to(tl.int64)) % N` | 环绕寻址（同 5.1.5） | 环绕（属性） |
| L518 | `accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]` | **dot 循环内嵌 blockwise 反量化**（非独立 Stage 的 dequant） | 量化 dot（`quantized_dot` flag） |
| L539-558 | `accumulator = accumulator * a_scale * b_scale; accumulator += bias[None,:]; accumulator *= moe_weight[:, None]` | **归约→广播统计→逐元素仿射**的 dequant+bias+路由权重尾缀 | `broadcast_affine_epilogue` |

#### 5.2.6 `fused_moe_kernel_gptq_awq`（fused_moe.py:79）

| 代码段 | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| 反量化部分 | `qweight` 位打包拆解（int4 高低位）+ zero-point/scale 反量化 | 位拆解 + 操作数路径反量化（同 LigerKernel） | `bit_unpack` + `quantized_dot` |

#### 5.2.7 `_count_expert_num_tokens`（model_executor/layers/fused_moe/utils.py:32）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L55-60 | `has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0); acc += has_curr_expert; tl.store(..., tl.sum(acc))` | **分段计数/histogram**：每 expert 全量遍历 O(E×N) 以**刻意规避 atomic**（避免乱序）；既无 atomic 类也无 histogram 类 | 分段计数（Coordination/Dataflow 边缘） |

#### 5.2.8 `_w8a8_block_int8_matmul`（quantization/utils/int8_utils.py:255）
#### 5.2.9 `_w8a8_triton_block_scaled_mm`（quantization/utils/fp8_utils.py:1050）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L321 / L1116 | `accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]` | dot + broadcast scale 融合（量化 GEMM 标准形态）；grouped-M swizzle/输出 mask 由现有类覆盖 | `quantized_dot`（flag） |

#### 5.2.10 `_lora_expand_kernel`（lora/ops/triton_ops/lora_expand_op.py:19 + kernel_utils.py:111）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L85-92 | `lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx); ram = tl.load(cta_lora_seq_indices + offset_m)` | A 矩阵 M 行由排序后的 token 索引表给定（GDC 风格按 lora_id 分组）——**运行时行 gather**，非规则 2D block | 分组路由 / ragged |
| L76 | `curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)` | 每 slice 变长 N + 每 LoRA 变长 M → 动态形状 | ragged（属性） |

### 5.3 SGLang

#### 5.3.1 `_fwd_grouped_kernel_stage1_rope`（attention/triton_ops/rocm_mla_decode_rope.py:45）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L152-165 | `pos = tl.load(positions + cur_batch*stride_positions_b); cos = tl.load(cos_sin_cache + pos*stride + offs_rotary, mask=mask_rotary)`，`offs_rotary = tl.arange(0,BLOCK_R) % (rotary_dim//2)` | **标量基址 + 重复 lane 偏移**的广播式表 gather（每 rotary 索引被 load 两次）；非 per-lane 离散 gather、非连续 load | `lookup_table_gather`（标量基址变体） |
| L167-179 | `off_q_pe_rot` 由 mod/where 构造的编译期旋转索引二次 reload 同一缓冲区 | **编译期索引置换重载**（用 L2 重载模拟寄存器 permute）；`indirect_gather_memory` 假设数据相关地址 | `in_register_rearrange`（permute-reload 变体） |
| L263-271 | `n_e_max = tl.maximum(tl.max(qk,1), e_max); re_scale = tl.exp(e_max-n_e_max); p = tl.exp(qk-n_e_max[:,None]); acc += tl.dot(p.to(v.dtype), v); e_sum = e_sum*re_scale + tl.sum(p,1)` | online softmax（同 5.2.1） | `online_softmax` |
| L208-209 | `if split_kv_end > split_kv_start: for start_n in range(split_kv_start, split_kv_end, BLOCK_N):` | 循环界依赖**运行时标量**（kv_indptr）→ 动态 trip count | 动态 trip count（属性） |
| L295-306 | per-head LSE 标量写 `Att_Out + stride_mid_oh`（strided 短 store） | `continuous_short_load` 只有 load 侧，无 short store | short store（属性） |

#### 5.3.2 `compute_seg_indptr_triton_kernel`（moe/ep_moe/kernels.py:183）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L188-195 | `while low <= high: mid = (low+high)//2; if tl.load(reorder_topk_ids + mid) > expert_id_minus_1: high = mid-1 ...` | 语义是 **exclusive scan / segmented offset**（兄弟 kernel `_fwd_kernel_ep_scatter_1` L625 即 `tl.cumsum(tokens_per_expert) - tokens_per_expert`），实现是**数据相关 while 循环**：trip count 依赖输入、无 lane、不可流水；现有所有循环 Kind 都假设 `tl.range` 静态/启动时常量界 | `rowwise_scan`（语义）+ 数据相关标量循环 |

#### 5.3.3 `deepep_compute_src2dst_triton_kernel`（ep_moe/kernels.py:149）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L155-157 | `src_id = tl.load(reorder_ids + dst_id, mask=mask); tl.store(src2dst + src_id, dst_id - num_invalid, mask=mask)` | **逆置换 scatter store**：写的是索引值（构建 remap 表），正确性依赖 `reorder_ids` 是置换（无重复 lane ⇒ 无需原子）——唯一性不变量无 Kind 可表达 | `permutation_remap` |

#### 5.3.4 `deepgemm_compute_src2dst_triton_kernel`（ep_moe/kernels.py:980）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L992-995 | `src_id = tl.load(reorder_ids + dst_id); expert_id = tl.load(topk_ids + src_id, mask=src_id < num_toks); expert_dst_start = tl.load(seg_indptr + expert_id, mask=expert_id >= 0)` | **多跳数据相关标量间接链（pointer chasing）**：每跳 load 地址依赖上一跳值，延迟串成本与单跳 `indirect_scalar_memory` 不同；mask 守卫的是**地址合法性**（哨兵）而非值边界 | 指针追逐（间接深度 flag） |
| L996-997 | `dst_id = expert_id * m_max + expert_dst_offset; tl.store(src2dst + src_id, dst_id, mask=mask)` | 数据相关边界（seg_indptr）上的 block-padded 偏移 + 索引 scatter | `permutation_remap` + 打包偏移 |

#### 5.3.5 `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`（ep_moe/kernels.py:433）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L444-458 | `num_tokens = tl.load(num_tokens_tensor_ptr); numel = num_tokens * intermediate_size; for id in tl.range(start_idx, numel, step, num_stages=NUM_STAGES): ids = id + tl.arange(0,BLOCK_SIZE); token_ids = ids // intermediate_size; offs = ids + token_ids * intermediate_size` | 循环界来自**运行时 tensor**；指针表达式含 `//` 为非 affine → 即使内存近连续，"可证明连续"也失败 | 动态 trip count + 非 affine 寻址（属性） |
| L459-462 | `gate = tl.load(gate_ptr + offs, ...).to(tl.float32); output = gate / (1 + tl.exp(-gate)) * up * scale; tl.store(output_ptr + ids, output.to(OutDtype), mask=mask)` | 主导成本是 **tile 级逐元素超越函数（silu）** + 融合 scale+cast；`scalar_math` 定义标量，`conversion_pack` 只覆盖 cast | `vector_sfu` + 融合激活量化 |

### 5.4 LigerKernel

#### 5.4.1 `matmul_kernel`（ops/experimental/mm_int8int2.py:232）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L322-325 | `mask = 3 << (2 * i); b = (b_uint8 & mask) >> (2 * i)`（外层 `for i in range(4)` 编译期展开的位平面循环） | 纯整数位域提取（无 dtype 转换），产物是 dot 操作数；`conversion_pack` 是值空间转换 | `bit_unpack` |
| L327-329 | `accumulator += tl.dot(a, (b.to(tl.int8) - tensor_full), out_dtype=tl.int32)` | **dot 操作数路径内联偏置反量化**（pack 时 +1 的偏置在 MMA 输入表达式内修正），非独立 dequant 块 | `quantized_dot`（flag） |
| L256-258 | `phy_pid = tl.program_id(axis=0); for pid in range(phy_pid, logic_grids, phy_grids):` | 手工持久化 grid-stride 调度（与 V1 的 `auto_blockify_*` 不同来源） | `persistent_grid_stride` |
| L279-280 | `offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M` | 用环绕**代替谓词**（边界重复读而非 mask 关闭），与 `predicate_mask` 策略相反 | 环绕寻址（属性） |

#### 5.4.2 `_mask_fwd_kernel`（ops/multi_token_attention.py:17）

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L38-39 | `future = col_idx[None, :] > row_idx[:, None]; mask_load = in_bounds & ~future` | **从 tile 坐标关系合成的结构掩码**（因果/三角）：broadcast compare 两个索引向量 + bool 组合；`predicate_mask` 是"对数据 lane 的 compare"，这是"索引向量间生成几何掩码" | `predicate_mask`（index-derived 变体） |
| L40 | `out = tl.load(base + offs, mask=mask_load, other=mask_val, cache_modifier=".ca")` | masked 位置的 `other=` 是**语义输出值**（-1e9/0，下游 softmax 消费），mask 应用与 load 融合为一 op，且 masked lane 仍被 store（L41 只用 in_bounds） | 语义填充 load（属性） |
| L40-41 | `.ca` load / `.cs` store | load 侧 cache policy 缺失（同 5.2.4） | `cache_policy_load`（属性） |

### 5.5 FlagGems

#### 5.5.1 `cumsum`（ops/cumsum.py）——scan 家族全集

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L58, L128, L334, L361, L387, L437 | `result = tl.cumsum(inp_vals, axis=0)` 等 | **rowwise scan**：等长输出、lane 间 log 步（Hillis-Steele）依赖；`rowwise_reduction` 是塌缩、`loop_carried_recurrence` 是跨迭代 | `rowwise_scan` |
| L85-90, L166-171, L354-356, L362, L437 | `last_part_sum_via_sum = tl.load(partial_sum_ptrs + pid - 1); final_vals = out_vals + last_part_sum_via_sum`；`tile_scan = prefix + tl.cumsum(x, 0); prefix += tl.sum(x, 0)` | **跨 block/CTA 的 scan carry**：携带值是另一个 CTA 在先前 kernel launch 算出的 block 前缀，通过全局内存通信；不是 per-lane i-1 递推 | `scan_carry_propagation` |
| L186-189 | host 递归 `scan_then_fan`（partial_sum 自身递归重扫至 part_num<2） | 递归多 kernel 发射的 scan 分解，Dispatch 族（单 kernel 调度）不覆盖 | 多发射流水（协调属性） |

#### 5.5.2 `masked_select`（ops/masked_select.py）——scan + atomic + scatter 全集

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L24 | `out_offsets = tl.cumsum(mask_ints, axis=0) - 1` | **exclusive scan**（掩码→偏移压缩，stream compaction 惯用法） | `rowwise_scan` |
| L26, L117, L125 | `tl.store(out_ptr + out_offsets, inp, mask=(offsets < N) & mask)` | **scan 派生的间接 scatter store**（gather 的写侧镜像） | `indirect_scatter_store` |
| L76-86 | `count = tl.atomic_add(counter_ptr, 1, sem="acq_rel"); np = tl.num_programs(0); if count == np - 1: ... pre_sums = tl.cumsum(part_sums, 0); tl.store(part_sums_ptr + np, final_sum)` | **arrival-counter 跨 CTA 合作屏障** + last-CTA 收尾 scan + 设备→主机哨兵交接（host 读 `part_sums[-1]` 分配输出）；20 类中 atomic 完全缺席 | `atomic_counter` / `last_cta_scan` |
| L114 | `out_ptr += advance`（上一 block 的 count 移动写基址） | **指针级 scan carry** | `scan_carry_propagation`（指针变体） |

#### 5.5.3 `instance_norm`（fused/instance_norm.py）——normalize epilogue 全集

| 代码段（行号） | 代码 | 为什么不属于已有范式 | 缺失语义 |
|---|---|---|---|
| L67, L121, L208/216, L262 | `out = (x - m) * rstd * w + b`（persistent/多行/loop/running-stats 四变体） | **reduction→广播统计→逐元素 scale-shift** 的 normalize epilogue；`rowwise_reduction` 止步于统计、`scalar_math` 只支持标量 | `broadcast_affine_epilogue` |
| L162-163 | `new_m = m + (x - m) / (step + 1); new_s = s + (x - new_m) * (x - m)` + L180-181 `final_m = tl.sum(m * cnt) / N` | **Welford 在线递推**（数值稳定两状态更新律）+ 跨 lane 统计合并归约 | `loop_carried_recurrence`（welford 变体） |
| L206, L213 | `tl.load(in_ptr + ..., eviction_policy="evict_first")` | load 侧 eviction policy（同 5.2.4） | `cache_policy_load`（属性） |
| L99 | `c_offsets = m_offsets % C`（TILE_M 宽向量取模） | 向量化取模索引（per-channel 参数查表） | 向量索引（属性） |
| L307-308 | `new_running_mean = (1 - momentum) * running_mean + momentum * new_mean / B` | 动量 EMA 混合（running stats 更新） | broadcast-affine 变体 |

---

## 6. 未覆盖模式汇总（缺失语义 → 证据一览）

| # | 缺失语义族 | 缺失 Kind（建议名） | 关键证据（kernel:line） | 现状部分命中（为何不够） |
|---|---|---|---|---|
| 1 | Scan | `rowwise_scan` | FlagGems cumsum:58/128/334/361/387/437、masked_select:24；FBGEMM fused_single_block:2761/2787、fused_padding:2897；SGLang seg_indptr（兄弟 kernel L625） | `rowwise_reduction`（塌缩 vs 等长前缀，公式不同） |
| 2 | Scan | `scan_carry_propagation` | FlagGems cumsum:85-90/166-171/354-356/362/437、masked_select:114 | `loop_carried_recurrence`（跨 CTA vs 跨迭代） |
| 3 | Coordination | `atomic_counter` / `last_cta_scan` | FlagGems masked_select:76-86；SGLang `_fwd_kernel_ep_scatter_2`:692 | 无（atomic 完全缺席） |
| 4 | Coordination | `inter_program_sync` | FBGEMM fused_single_block:2798；VLLM causal_conv1d:234/269/383 | 无 |
| 5 | 间接写 | `indirect_scatter_store` | FlagGems masked_select:26/117/125；VLLM logit_bias:189 | `indirect_gather_memory`（只建模 load 侧） |
| 6 | 索引表 | `permutation_remap` | SGLang deepep:155-157、deepgemm:996-997 | `indirect_gather_memory`（写索引值+唯一性不变量） |
| 7 | 索引表 | `lookup_table_gather` | FBGEMM dequantize:552-553；SGLang MLA rope:152-165 | `indirect_gather_memory`（小均匀表可广播，成本不同） |
| 8 | 索引表 | `page_table_indirection` | VLLM decode_attention:114-121/321-328 | `indirect_gather_memory`（页内连续可合并） |
| 9 | 索引表 | `pointer_chasing`（多跳间接链） | SGLang deepgemm:992-995 | `indirect_scalar_memory`（链式延迟串） |
| 10 | 位操作 | `bit_unpack` / `bitmanip_convert` | LigerKernel mm_int8int2:322-325；FBGEMM dequantize:548、silu_quantize:620-672 | `conversion_pack`（无 dtype 转换的 ALU 位操作） |
| 11 | 寄存器重排 | `in_register_rearrange` | FBGEMM dequantize:572、silu_quantize:589-614/675-677；SGLang MLA:167-179 | 无 |
| 12 | 随机数 | `random_number_generation` | FBGEMM silu_quantize:589-614（Philox/randint4x） | 无 |
| 13 | 超越函数 | `vector_sfu`（+fp64/合成 flag） | SGLang silu:459-462；FBGEMM rope_padded:196-200、dequantize:567 | `scalar_math`（定义为标量） |
| 14 | 搜索 | `search_based_index_generation` | FBGEMM fused_padding:2915-2924、fused_single_block:2806-2824；SGLang compute_seg_indptr:188-195 | `index_generation`+`indirect_gather_memory`+`scalar_control`（需三者拼接） |
| 15 | 调度 | `persistent_grid_stride` | LigerKernel mm_int8int2:256-258；FBGEMM fused_single_block:2800-2801 | `auto_blockify_loop`（V1 注入 vs 手工） |
| 16 | 调度 | `program_role_dispatch` | FBGEMM rope_padded:142/182-190/211-219、fused_single_block:2775、fused_padding:2886 | `scalar_control`（pid 角色 vs 数据分支） |
| 17 | Ragged | `jagged_ragged_memory` | FBGEMM array_jagged_bmm:39-88、rope_padded:135-158；VLLM lora_expand:76/85-92；SGLang silu:444-458 | `index_generation`+`indirect_gather_memory`+`predicate_mask`（需三者拼接） |
| 18 | 数据流 | `online_softmax` | VLLM decode_attention:151-158/372-378；SGLang MLA:263-271 | `rowwise_reduction`+`loop_carried_recurrence`+`scalar_math`（三者拼接） |
| 19 | 数据流 | `broadcast_affine_epilogue` | FlagGems instance_norm:67/121/216/262、cumsum:388/456/509；VLLM fused_moe:539-558、w8a8:321 | `rowwise_reduction`（止步于统计）+`scalar_math`（标量） |
| 20 | Matrix | `manual_dot_reduction` / `argmax_reduction` | VLLM decode_attention:132、sample_recovered:822 | `tiny_cube_roofline`（预设 tt.dot） |
| 21 | 属性（不造 Kind） | 动态 trip count / cache_policy_load / underfill / 环绕寻址 / fp64 / 语义填充 / 短 store | 见 5.x 各表 | 现状以 Kind 或完全缺失表达 |

---

## 7. 重组方案：7 族新体系与划分逻辑

### 7.1 总览

新体系遵循第 4 节公理：**族 = 主成本公式形状（互斥）→ 类 = 族内主导模式形态 →
结构修饰 = 属性 flag（正交）**。净效果：Scalar 6→5、Continuous 4→2、Cube 2→1
（约合并 6 类），新增 17 类，总约 30 类；但每条类现在都由"公式差异"或"参数差异
不可降级"支撑，不再存在"同公式多标签"。

```
StageCostModelKind
├── 1. Dispatch / Scheduling 族   （公式：C_setup + N_iter×C_body，控制开销为主）
├── 2. Index / Scalar / Control 族（公式：main-scalar / issue / predicate 串行关键路径）
├── 3. Memory 族                  （公式：MTE transaction / DCache transaction / 延迟）
├── 4. Matrix / Dot 族            （公式：Cube setup + MMAD 吞吐 + underfill）
├── 5. Compute / ALU 族           （公式：EXU / ASU / 寄存器重排 / 转换指令吞吐）
├── 6. Dataflow 族                （公式：critical_path×N / log 步 / tree 深度 / 跨 block carry）
└── 7. Coordination 族            （公式：原子竞争 / 屏障 / 收尾串行）
```

### 7.2 每族每类的划分逻辑

> 每族给出一句话划分轴（族间互斥依据），每类给出"识别结构 + 与同族其它类的区分点"。

#### 族 1：Dispatch / Scheduling —— 划分轴：调度结构的"来源与形状"

| 类 | 识别结构 | 与同族其它类的区分 |
|---|---|---|
| `auto_blockify_dispatch` | V1 注入的每物理核 PID/chunk/边界 setup | 来源=编译器注入；无循环 |
| `auto_blockify_loop` | V1 注入的 logical-program 聚合循环 | 来源=编译器注入；有循环 |
| `persistent_grid_stride` | **手工**持久化 grid-stride 循环（`for pid in range(phy_pid, logic_grids, phy_grids)`） | 来源=kernel 手写；有循环 |
| `program_role_dispatch` | program_id 分区的互斥角色分支（q/k/v、pid==0 特化） | 来源=kernel 手写；无循环，是分支 |

合并依据：`auto_blockify_dispatch` 与 `auto_blockify_loop` 公式相同，可进一步合并为
一个 Kind + Phase 角色 flag；保留两个是为了 V1 provenance 追踪。新增 `persistent_grid_stride`
与 `program_role_dispatch`：前者决定**是否可套 AutoBlockify/SuperBlock 包装**，后者决定
**分支的 uniformity（全 program 均匀）与角色间负载均衡**——都是主公式之外的独立判定。

#### 族 2：Index / Scalar / Control —— 划分轴：标量工作的"产出物"

| 类 | 识别结构（产出物） | 与同族其它类的区分 |
|---|---|---|
| `index_generation` | 产出地址/offset（`pid*BLOCK+arange`、div-rem、pointer induction） | 产出=地址；可能影响 MTE 发射槽 |
| `scalar_math` | 产出计算值（标量算术/SFU；含 fp64 flag） | 产出=值；SFU 延迟 |
| `predicate_mask` | 产出谓词/mask（含 index-derived 结构掩码、语义填充 flag） | 产出=掩码；compare+select 计数 |
| `scalar_control` | 产出控制流（early return、分支、数据相关标量循环） | 产出=控制转移；分支/backedge 代价 |
| `loop_predicate` | 循环内变化的谓词/退出条件 | 产出=掩码+控制；依赖 trip count |

合并依据（相对现状）：`scalar_issue` 与 `scalar_math`/`index_generation` 公式相同，
并入后两者（兜底语义由"无显著产出"识别）。`predicate_mask` 吸收"坐标合成三角掩码"
（成本是 broadcast compare，参数不同但公式同族）。`scalar_control` 吸收
"数据相关标量循环"（while 二分搜索的 trip count 无法静态化，作为循环类 flag）。

#### 族 3：Memory —— 划分轴：地址的"可证明性质 × 形态"

这是最清晰的轴，先问"地址能否静态证明"，再问"间接形态"：

```
地址可静态证明连续？ ──是──> continuous_tile_load / continuous_tile_store
        │                          （+ short_tile / cache_policy / 语义填充 flag）
        └─否（依赖运行时）──> 形态？
              ├─ 单标量基址（可链式）      -> indirect_scalar_memory（+ 链深度 flag）
              ├─ 多 lane 离散 load        -> indirect_gather_memory
              ├─ 多 lane 离散 store       -> indirect_scatter_store
              ├─ 小均匀表查表             -> lookup_table_gather
              ├─ 两级页表（页内连续）      -> page_table_indirection
              └─ per-segment 运行时边界   -> jagged_ragged_memory
```

| 类 | 区分点（与最近邻） |
|---|---|
| `continuous_tile_load/store` | 合并原 4 类：`short_tile`、`cache_policy`（load/store 统一）、语义填充为 flag |
| `indirect_scalar_memory` | 1 个基址标量；多跳链（pointer chasing）以"链深度" flag 表达（延迟串 = 深度×单跳延迟） |
| `indirect_gather_memory` | 多 lane 离散 load；地址来自数据、无唯一性要求 |
| `indirect_scatter_store` | 多 lane 离散 store；写合并/冲突与 gather 公式不同（store 延迟不可按 load 隐藏） |
| `lookup_table_gather` | 小均匀表（≤64 项）、跨 program 同表、L1 驻留、lane 地址可重复——按可广播 LUT 计费 |
| `page_table_indirection` | 两级间接，页内偏移连续 → transaction 部分合并 |
| `jagged_ragged_memory` | per-segment 运行时边界；变长 tile、空段 early exit、环绕寻址、动态 trip count 一并携带 |

#### 族 4：Matrix / Dot —— 划分轴：dot 的"实现路径"

| 类 | 识别结构 | 区分点 |
|---|---|---|
| `cube_roofline` | 规则 `tt.dot`（合并原 tiny_cube：`underfill_ratio` 为连续 flag） | 走 Cube 流水；`quantized_dot` flag 吸收"操作数/累加路径内联反量化"（公式仍是 Cube，仅多一个广播乘） |
| `manual_dot_reduction` | 无 tt.dot 的外积+sum / argmax 类索引归约 | 走 EXU+归约，无 Cube setup |

#### 族 5：Compute / ALU —— 划分轴：计算类别对应的"资源流水"

| 类 | 识别结构 | 区分点 |
|---|---|---|
| `vector_sfu` | tile 级逐元素超越函数（silu/sigmoid/tanh/exp/pow 链）；fp64、trig 合成为 flag | SFU/ASU 流水；超越函数链深度影响延迟 |
| `bit_unpack` | 整数位域提取（shift/mask/OR），无 dtype 转换 | 整数 ALU 流水；每元素 op 数显著（mx4/int2 约 10-20 ops） |
| `dtype_conversion` | dtype cast/quantize/pack 的值空间转换（原 `conversion_pack`） | 转换流水；与 bit_unpack 区分：有无 dtype 语义变化 |
| `in_register_rearrange` | `tl.interleave`/`tl.split`/`tl.join`/transpose/permute-reload | 寄存器/shuffle 流水；独立于 EXU 计费（不重复计入 C_shuffle） |
| `random_number_generation` | `tl.randint4x`/Philox 及种子派生 | 专用 RNG/整数流水；多轮 rounds 成本 |

#### 族 6：Dataflow —— 划分轴：合并的"输出形状 × 依赖跨越范围"

```
合并的输出形状：
  塌缩 (N→1)        -> rowwise_reduction（tree 深度）
  等长前缀 (N→N)     -> rowwise_scan（log 步）
  跨迭代串行         -> loop_carried_recurrence（critical_path×N；welford 为 flag）
依赖跨越范围：
  跨 block/CTA       -> scan_carry_propagation（跨 CTA 全局内存携带）
  流式 + 归约 + SFU  -> online_softmax（flash-attention 惯用法）
  归约→广播统计→仿射 -> broadcast_affine_epilogue（normalize/dequant/bias 尾缀）
```

| 类 | 区分点 |
|---|---|
| `rowwise_reduction` | 塌缩；tree/shuffle 深度 |
| `rowwise_scan` | 等长前缀；log 步跨 lane 依赖——与 reduction 公式根本不同 |
| `loop_carried_recurrence` | 跨迭代；Welford 双状态更新为 flag |
| `scan_carry_propagation` | 携带值来自**其它 CTA/先前 launch**（全局内存通信），与"本迭代前值"不同 |
| `online_softmax` | reduction+recurrence+exp 的固定三步融合体，识别即用专用公式 |
| `broadcast_affine_epilogue` | 归约结果广播到整 tile 的逐元素仿射（含反量化 scale、bias、路由权重、EMA 混合） |

#### 族 7：Coordination —— 划分轴：协作"机制"

| 类 | 识别结构 | 区分点 |
|---|---|---|
| `atomic_counter` | arrival counter（`tl.atomic_add(counter, 1, sem="acq_rel")`）+ last-CTA 收尾（`last_cta_scan` 为 flag） | 计数器语义；竞争可忽略、收尾串行 |
| `atomic_scatter_accumulate` | 数据相关地址原子累加（histogram/槽分配/计数） | 地址竞争/乱序不可忽略 |
| `inter_program_sync` | `tl.debug_barrier()` 等显式跨 CTA 屏障 | 同步等待成本 |

### 7.3 分类优先级（更新）

按 4.2 判定树（约束强度序）：

```
Coordination（全局语义，最强）
→ Dispatch / Scheduling（程序结构）
→ Dataflow（scan/recurrence/carry/reduction，决定公式串并行）
→ Memory（访存形态）与 Matrix（dot）/ Compute（计算类别）——三者由"关键路径主导资源"互斥
→ Index / Scalar / Control（兜底 head/setup）
```

相对现状的变更：Coordination 提至最前（新增轴）；Dataflow 中 scan 提到 dot 之前
（目标算子中 scan 主导关键路径的 Stage 会被现状误归 Cube/连续访存）；scalar 兜底不变。

### 7.4 属性 flag 清单（进入 `StageModelFeatures`，不进入枚举）

| 属性 | 影响 |
|---|---|
| `cache_policy`（load/store 统一：`.ca/.cg/.cs/.cv`、eviction_policy） | 合法性与 transaction 参数 |
| `short_tile`（tile 大小低于启动延迟阈值） | 固定启动延迟项 |
| `dynamic_trip_count`（循环界依赖运行时标量/tensor） | 取消 `independent_pipelined_loop` 静态界合法性 |
| `ragged_bounds` / `wrap_index` | 连续/间接公式参数 |
| `underfill_ratio`（Cube） | Cube 利用率 |
| `fp64_sfu` / `transcendental_chain` | SFU 延迟/吞吐降级 |
| `semantic_fill`（masked load 的 other= 为语义常数） | 无效 lane 仍产生写流量 |
| `quantized_dot`（dot 路径内联反量化） | Cube 公式 + 广播乘 |
| `pointer_chasing_depth` | 间接延迟串 = 深度 × 单跳延迟 |
| `program_role`（pid 分区角色） | 负载均衡/分支 uniformity |
| `welford` / `last_cta_scan` | 递推/计数器公式参数 |

---

## 8. 与现状的对照迁移表

| 现状 Kind | 处置 | 新归属 |
|---|---|---|
| `auto_blockify_dispatch` | 保留 | 族 1 |
| `auto_blockify_loop` | 保留（可与 dispatch 合并+flag） | 族 1 |
| `scalar_issue` | **合并**（公式同 scalar_math/index_generation） | 族 2 `scalar_math` / `index_generation` |
| `scalar_control` | 保留（吸收数据相关标量循环） | 族 2 |
| `scalar_math` | 保留（吸收 vector_sfu 的标量特例；+fp64 flag） | 族 2 / 族 5 `vector_sfu` |
| `index_generation` | 保留 | 族 2 |
| `predicate_mask` | 保留（+index-derived / 语义填充 flag） | 族 2 |
| `loop_predicate` | 保留 | 族 2 |
| `continuous_tile_memory` | 保留（+short/cache flag） | 族 3 |
| `continuous_tile_store` | 保留（+cache flag；补 load 侧） | 族 3 |
| `continuous_short_load` | **合并** → flag `short_tile` | 族 3 |
| `cache_policy_store` | **合并** → flag `cache_policy` | 族 3 |
| `indirect_scalar_memory` | 保留（+pointer_chasing flag） | 族 3 |
| `indirect_gather_memory` | 保留 | 族 3 |
| `independent_pipelined_loop` | 保留（+dynamic_trip_count flag；与 Dataflow 互斥判定） | 族 6 边缘 / 族 3-5 的循环形态 |
| `loop_carried_recurrence` | 保留（+welford flag） | 族 6 |
| `rowwise_reduction` | 保留 | 族 6 |
| `cube_roofline` | 保留（+underfill/quantized flag） | 族 4 |
| `tiny_cube_roofline` | **合并** → flag `underfill_ratio` | 族 4 |
| `conversion_pack` | 更名 `dtype_conversion`（位操作剥离） | 族 5 |
| **（新增）** | 族 1：`persistent_grid_stride`、`program_role_dispatch` | 族 1 |
| **（新增）** | 族 3：`indirect_scatter_store`、`lookup_table_gather`、`page_table_indirection`、`jagged_ragged_memory` | 族 3 |
| **（新增）** | 族 4：`manual_dot_reduction` | 族 4 |
| **（新增）** | 族 5：`vector_sfu`、`bit_unpack`、`in_register_rearrange`、`random_number_generation` | 族 5 |
| **（新增）** | 族 6：`rowwise_scan`、`scan_carry_propagation`、`online_softmax`、`broadcast_affine_epilogue` | 族 6 |
| **（新增）** | 族 7：`atomic_counter`、`atomic_scatter_accumulate`、`inter_program_sync` | 族 7 |

汇总：合并 6 类（scalar_issue、continuous_short_load、cache_policy_store、
tiny_cube_roofline、auto_blockify_dispatch/loop 可并）、更名 1 类、新增 17 类、
新增 1 族（Coordination）、扩展 3 族（Dispatch、Memory、Dataflow）、属性 flag 11 项。

---

## 9. 落地建议

1. **枚举层面**：`StageCostModelKind` 按 7 族组织（C++ 枚举或带族前缀的常量），
   属性全部进 `StageModelFeatures`/`StageWorkload`，不进枚举。
2. **Registry 层面**：允许"一类多 Kind"（同一具体模型类服务公式相同的多个 Kind，
   如 Memory 族可共用一个 transaction 基类 + 形态参数），维持文档"约 8 组语义模型
   每模式"的目标量级。
3. **StageKindClassifier**：实现 4.2 判定树（Coordination → Dispatch → Dataflow →
   Memory/Matrix/Compute → Scalar），把判定从"经验优先级"改为"约束强度推导"；
   命中两个强结构返回 `requires_split`（如"scan + gather"拆开）。
4. **StageModeLegalityAnalysis**：新增属性相关合法性（`dynamic_trip_count` 取消
   pipelined 静态界、`cache_policy` 的 load/store 对称、原子语义在 SIMD/SIMT scope
   内外的顺序安全）。
5. **补充验证用例**：建议用本文第 5 节证据密度最高的三类补 profile——
   scan（FlagGems cumsum 多 pass）、Coordination（masked_select 三 kernel 流水）、
   Ragged（FBGEMM array_jagged_bmm）。