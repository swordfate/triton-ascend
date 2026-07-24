# TTAdapter (Linalg) → NPUIR 转换详解

## 0. 先修知识：这个转换的本质

TTAdapter 仍然是**标准 MLIR 方言**（linalg, memref, arith, tensor, scf），不包含任何硬件信息。NPUIR 则是**硬件绑定**的表示——每条 `hivm.hir.*` 指令直接对应 Ascend NPU 的一条硬件指令。

转换由两部分完成：

1. **Triton Adapter 的 DynamicCVPipeline**（本仓库）——插入同步原语、分配 CUBE/VECTOR 核、auto-blockify
2. **bishengir-compile**（外部编译器）——最终 lowering 到 hivm.hir 指令、地址空间标注、UB buffer 分配、GraphSyncSolver 插入同步

NPUIR 文件头的 `// IR Dump After GraphSyncSolver (hivm-graph-sync-solver)` 说明这是 bishengir 最终同步求解器之后的产物。触发 debug 输出的方式是 `TRITON_DEBUG=1` 或 `@triton.jit(debug=True)`（见 `compiler.py:687-688`）。

---

## A. 函数签名变化

### TTAdapter

```mlir
func.func @add_kernel(
  %arg0: memref<?xi8>,                               // sync lock (无地址空间)
  %arg1: memref<?xi8>,                               // workspace (无地址空间)
  %x_ptr: memref<?xf32> {tt.divisibility = 16, tt.tensor_kind = 0},
  %y_ptr: memref<?xf32> {tt.divisibility = 16, tt.tensor_kind = 0},
  %output_ptr: memref<?xf32> {tt.divisibility = 16, tt.tensor_kind = 1},
  %n_elements: i32 {tt.divisibility = 16},
  %arg6: i32, %arg7: i32, %arg8: i32,                // num_programs[0,1,2]
  %arg9: i32, %arg10: i32, %arg11: i32)              // program_id[0,1,2]
attributes {SyncBlockLockArgIdx = 0, WorkspaceArgIdx = 1, global_kernel = "local",
            mix_mode = "aiv", parallel_mode = "simd"}
```

### NPUIR

```mlir
func.func @add_kernel(
  %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
  %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
  %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>},
  %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16, tt.tensor_kind = 0},
  %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16, tt.tensor_kind = 0},
  %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16, tt.tensor_kind = 1},
  %arg6: i32 {tt.divisibility = 16},
  %arg7: i32, %arg8: i32, %arg9: i32)
attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
            func_dyn_memref_args = dense<[false, true, true, true, true, true,
            false, false, false, false]> : vector<10xi1>,
            hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
            hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned,
            mix_mode = "aiv", parallel_mode = "simd"}
```

### 并排对比

```
位置  TTAdapter                                     NPUIR
──────────────────────────────────────────────────────────────────────────────────
 0    %arg0: memref<?xi8>          (sync lock)      %arg0: i64                     (ffts_base_address) ← 新增
 1    %arg1: memref<?xi8>          (workspace)      %arg1: memref<?xi8, gm>        (sync_block_lock)   ← 原来 arg0
 2    —                                             %arg2: memref<?xi8, gm>        (workspace)         ← 原来 arg1
 3    x_ptr:   memref<?xf32>                        %arg3: memref<?xf32, gm>       (x_ptr)
 4    y_ptr:   memref<?xf32>                        %arg4: memref<?xf32, gm>       (y_ptr)
 5    output:  memref<?xf32>                        %arg5: memref<?xf32, gm>       (output_ptr)
 6    n_elem:  i32                                  %arg6: i32                     (n_elements)
 7    %arg6:   i32              (num_programs[0])   %arg7: i32                     (num_programs[0])
 8    %arg7:   i32              (num_programs[1])   %arg8: i32                     (num_programs[1])
 9    %arg8:   i32              (num_programs[2])   %arg9: i32                     (num_programs[2])
10    %arg9:   i32              (program_id[0])      —                                          ← 删除
11    %arg10:  i32              (program_id[1])      —                                          ← 删除
12    %arg11:  i32              (program_id[2])      —                                          ← 删除
```

净变化：**+1 (ffts), -3 (program_id)** → 12 → 10 个参数。

### 逐项解读

| # | 变化 | 推导 | 含义 |
|---|------|------|------|
| A1 | **插入** `%arg0: i64 {ffts_base_address}` | FFTS（Fast Forwarding Transaction System）是 Ascend 芯片上 AI Core 之间的高速消息传递机制。bishengir 编译器在所有 device kernel **最前面**插入这个参数，作为硬件 FFTS 寄存器的基地址。这是编译器的**通用行为**，跟 kernel 是否使用 FFTS 无关 | sync lock 没有变成 ffts——从 `hacc.arg_type` 枚举可以验证，`ffts_base_address`、`sync_block_lock`、`workspace` 是三个不同的 enum 值 |
| A2 | `%arg0`（sync lock）→ `%arg1` + `gm` + `hacc.arg_type` | sync lock 被 ffts 往后挤了一位，并且补充了地址空间和硬件参数类型标记 |
| A3 | `%arg1`（workspace）→ `%arg2` + `gm` + `hacc.arg_type` | workspace 后移一位并补充标记 |
| A4 | 所有数据 memref 都加了 `#hivm.address_space<gm>` | bishengir 编译器分析 memref 的使用模式，标注每个 memref 所在的内存空间。`gm` = Global Memory（HBM/DDR），`ub` = Unified Buffer（片上 SRAM）。后续 `hivm.hir.load`/`store` 需要知道源和目标才能生成正确的 DMA 描述符 |
| A5 | `%arg9`~`%arg11` (program_id[0,1,2]) **删除** | 函数体内出现了 `hivm.hir.get_block_idx`，它取代了 program_id 参数 | 在 wave 模型中，program_id 不再是固定的函数参数——每个物理核跑多轮，每轮的 program_id 是 `wave_index * 40 + local_block_idx` 计算出来的 |
| A6 | 新增 `func_dyn_memref_args` | `dense<[false, true, true, true, true, true, false, false, false, false]>` | 标记哪些参数是动态大小的 memref（true=动态，false=静态）。arg1~arg5 是动态 memref |
| A7 | 新增 `hacc.entry` + `hacc.function_kind<DEVICE>` | hacc = Huawei Ascend Compute Compiler | 标记这是设备端入口函数（而非 host 函数），后端编译器据此决定调用约定 |
| A8 | 新增 `hivm.func_core_type<AIV>` | AIV = Ascend Intelligent Vector | 明确告诉硬件调度器：这个 kernel 只在 AIV core 上跑 |
| A9 | 新增 `hivm.storage_aligned` | bishengir 在 UB 上分配 buffer 时做了地址对齐（通常 32B 或 64B） | 对齐的地址能让 DMA 和向量指令更高效 |

---

## B. Wave Loop：从 SPMD 到物理波次调度

这是 NPUIR 与 TTAdapter **最根本的结构差异**。

### TTAdapter（无循环，直接计算）

```mlir
%block_start_0 = arith.muli %arg9, %block_start : i32      // program_id[0] * 1024
// 直接使用 program_id 参数，假设每个 program 对应一个物理核
```

### NPUIR（出现 wave loop + 硬件 block_idx）

```mlir
// ① 计算总 block 数
%0 = arith.muli %arg7, %arg8 : i32           // num_programs[0] * num_programs[1]
%1 = arith.muli %0, %arg9 : i32              // * num_programs[2] = 总 program 数
annotation.mark %1 {logical_block_num} : i32  // 标记总逻辑块数

// ② 计算 wave 数
%c40_i32 = arith.constant 40 : i32
%2 = arith.ceildivsi %1, %c40_i32 : i32       // ceil(总块数 / 物理核数) = wave数

// ③ 读取硬件 block 索引（取代 program_id 参数）
%3 = hivm.hir.get_block_idx -> i64            // 硬件指令：当前物理核编号 [0..39]
%4 = arith.trunci %3 : i64 to i32

// ④ Wave loop
scf.for %arg10 = %c0_i32 to %2 step %c1_i32  : i32 {
    // ⑤ 计算本 wave 的逻辑 block_id
    %17 = arith.muli %arg10, %c40_i32 : i32      // wave * 40
    %18 = arith.addi %17, %4 : i32               // + 物理核编号 = 全局 block_id
    %19 = arith.minsi %18, %1 : i32              // 对于最后一波，cap到总块数
    // ⑥ 从 block_id 反推 program_id
    %20 = arith.divsi %19, %5 : i32              // block_id / (num_z * num_y)
    %21 = arith.remsi %20, %arg7 : i32           // % num_x → program_id[0]
    // ...
```

### 逐项解读

| 步骤 | 变化 | 推导 | 含义 |
|------|------|------|------|
| B1 | 计算 `total_blocks = num_x * num_y * num_z` | `arg7*arg8*arg9`，其中 arg7/arg8/arg9 是 num_programs[0,1,2] | vecadd 只用 1D grid（num_programs[0]=97），所以实际 `total_blocks = 97 * 1 * 1 = 97` |
| B2 | `logical_block_num` annotation | AutoBlockify pass（`AutoBlockify.cpp:205`）计算的结果 | 告诉后端 scheduler 这个 kernel 逻辑上共有多少个 block，用于 wave 调度 |
| B3 | `ceildivsi(total, 40)` = wave 数 | 40 = AIV 物理核数（910B4 上有 40 个 AIV core） | vecadd 的 97 个 block 分配到 40 个核上，需要 `ceil(97/40) = 3` 波 |
| B4 | `hivm.hir.get_block_idx` → 硬件指令 | 直接映射到 NPU 的 block_idx 寄存器 | 取代了 `%arg9` (program_id 参数)。每个物理核启动时，硬件自动填入该核的编号 [0..39] |
| B5 | `wave * 40 + local_idx` = 全局 block_id | 例如第 2 波的 5 号核处理 block_id = 2×40+5 = 85 | wave 模型的核心：物理核编号固定为 [0..39]，通过 wave 循环覆盖所有逻辑 block |
| B6 | `minsi(block_id, total_blocks)` | 最后一波可能不满 40 个。vecadd：wave 2 只有 97-80=17 个有效 block | 防止访问越界 |

### 为什么需要 Wave Loop？

TTAdapter 假设每个 program **恰好对应一个物理核**——这是 SPMD 的抽象。但真实硬件只有 40 个 AIV Core，却有 97 个 program。所以必须把 97 个逻辑 program **折叠成 3 波**，每波每个物理核处理 1 个 program。就像餐厅只有 40 个厨师但有 97 桌客人——分 3 轮上菜。wave loop 就是"轮次"的调度器。

**代码依据**：`compiler.py:672-673`

```python
if _is_auto_map_parallel_blocks_enabled() and not metadata.get("has_auto_blockify_blacklist_op", False):
    _compile_option_list += ["--enable-auto-blockify-loop"]
```

---

## C. UB Buffer 分配：`memref.alloc()` → `hivm.hir.pointer_cast()`

### TTAdapter

```mlir
%x_3 = memref.alloc() : memref<1024xf32>        // 在"某处"分配
%y_14 = memref.alloc() : memref<1024xf32>
```

### NPUIR

```mlir
// 在循环开头，每次迭代重新计算（因为 double buffering）：
%14 = hivm.hir.pointer_cast(%c0_i64, %c8192_i64)
    : memref<1024xf32, #hivm.address_space<ub>>
annotation.mark %14 {hivm.multi_buffer = 2 : i32}

%15 = hivm.hir.pointer_cast(%c4096_i64, %c12288_i64)
    : memref<1024xf32, #hivm.address_space<ub>>
annotation.mark %15 {hivm.multi_buffer = 2 : i32}

%16 = hivm.hir.pointer_cast(%c0_i64, %c8192_i64)
    : memref<1024xf32, #hivm.address_space<ub>>
annotation.mark %16 {hivm.multi_buffer = 2 : i32}
```

### 逐项解读

| 变化 | 推导 | 含义 |
|------|------|------|
| `memref.alloc()` → `hivm.hir.pointer_cast(base, bound)` | 在 NPU 上没有"malloc"——UB（Unified Buffer）是一块固定大小的片上 SRAM。`pointer_cast` 把**整数偏移量**映射到 UB 的物理地址 | `%c0_i64` 是起始偏移，`%c8192_i64` 是结束偏移。即"在 UB 的 [0, 8192) 字节范围内分配 1024 个 f32" |
| 三块 buffer 共享 UB 空间 | x_buffer [0, 8192), y_buffer [4096, 12288), out_buffer [0, 8192)。注意 x 和 out 的偏移范围相同 [0, 8192)——它们**分时复用**同一块 UB 区域 | load x → add → store out 是串行的，x load 结束后 out store 才会用到这块区域，所以可以安全复用 |
| `#hivm.address_space<ub>` | 明确了内存空间是 UB（片上 SRAM），区别于 GM（Global Memory/HBM） | 后端 DMA 引擎需要知道搬出/搬入的目标地址空间 |
| `hivm.multi_buffer = 2` | 虽然物理上只有一份，但通过 event 交替机制实现逻辑上的 double buffering | 见下节 D.2 |

---

## D. 同步原语：整个 NPU pipeline 的骨架

这是 TTAdapter → NPUIR 中**最复杂的变化**。TTAdapter 没有任何同步指令——数据依赖是隐式的（通过 SSA use-def chain）。NPUIR 必须把每个隐式依赖转化为显式的硬件同步。

先理解 NPU 的流水线模型：

```
MTE2 (DMA Load) → VECTOR (计算) → MTE3 (DMA Store)
     ↕ set_flag/wait_flag 同步 ↕
```

`set_flag[src_pipe, dst_pipe, event_id]` 和 `wait_flag[src_pipe, dst_pipe, event_id]` 是流水线间的信号量机制。

### D.1 函数入口的初始化 set_flag

```mlir
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
```

在函数入口，MTE3（Store Pipe）向 MTE2（Load Pipe）**预设了 EVENT_ID0 和 EVENT_ID1 为 ready**。这相当于"许可"第一个 wave 的 load 可以开始。如果没有这个初始化，第一个 wait_flag 会永远阻塞。

### D.2 Double-Buffering Event 选择器

```mlir
// loop 内部开头：
%11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
%12 = arith.index_cast %11 : index to i1
%13 = arith.select %12, %c0_i64_0, %c1_i64 : i64
```

这是编译器自动生成的"交替选择器"模板公式。代入 `s1=0, s2=1` 后：

```
((s0 - 0) floordiv 1) mod 2 = s0 mod 2
```

**就是判断 wave 索引是偶数还是奇数。** 为什么写这么复杂？因为这是编译器自动生成的模板——对于任意循环 `for i = lb to ub step st`，`(i - lb) floordiv st` 给出的是"第几个迭代"，然后 `mod 2` 做奇偶交替。在这个特例中 lb=0, st=1 所以退化为 `s0 mod 2`。

`arith.select %12, %c0_i64_0, %c1_i64` 语义是 `条件 ? 真值 : 假值`：

| Wave | `%arg10` | `mod 2` | `%12` (i1) | 选哪个 | `%13` |
|------|----------|---------|------------|--------|-------|
| 0    | 0        | 0       | **false**  | 假值 = 1 | **1** (EVENT_ID1) |
| 1    | 1        | 1       | **true**   | 真值 = 0 | **0** (EVENT_ID0) |
| 2    | 2        | 0       | **false**  | 假值 = 1 | **1** (EVENT_ID1) |

`%13` 是一个 event 选择器，在 load 前和 store 后使用：

```mlir
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]    // 等上一波的 store
// ...
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]     // 通知下一波的 load
```

**完整的 double-buffering 时间线**：

```
         Wave 0              Wave 1              Wave 2
         (EVENT_ID1)         (EVENT_ID0)         (EVENT_ID1)
         ─────────           ─────────           ─────────
load x:  wait EVENT_ID1  │  wait EVENT_ID0  │  wait EVENT_ID1
         (入口已预设✓)     │  (等 Wave 0 █)   │  (等 Wave 1 █)
         ↓                │  ↓               │  ↓
vadd:    compute          │  compute          │  compute
         ↓                │  ↓               │  ↓
store:   set EVENT_ID1 ───┼→ (释放给 Wave 2)  │
                          │  set EVENT_ID0 ───┼→ (释放给 Wave 3)
```

Wave 0 和 Wave 2 都用 EVENT_ID1。Wave 0 store 完后 set EVENT_ID1 → Wave 2 的 load 才能通过 wait EVENT_ID1。Wave 1 用 EVENT_ID0，独立于 0 和 2，形成流水线重叠：

```
时间 →  Wave0: [load x] [load y] [vadd] [store]
             Wave1:    [load x] [load y] [vadd] [store]
                  Wave2:       [load x] [load y] [vadd] [store]
```

### D.3 mask padding 分支的同步

```mlir
// x fill (NPUIR 第 55-61 行)
scf.if %27 {
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]     // ① MTE3告诉V：store完了
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]    // ② V等MTE3信号
  hivm.hir.vbrc ins(%cst : f32) outs(%16 : ...ub>)           // ③ 向量广播填充0
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]     // ④ V告诉MTE2：fill完了
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]    // ⑤ MTE2等V信号
} {hivm.unlikely_condition}

// y fill (NPUIR 第 67-72 行) — 略有不同
scf.if %27 {
  hivm.hir.pipe_barrier[<PIPE_V>]                             // ① V pipe内部barrier
  hivm.hir.vbrc ins(%cst : f32) outs(%15 : ...ub>)           // ②
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]     // ③
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]    // ④
}
```

x 的 fill 用 `wait_flag[MTE3, V]` 等待——因为 out buffer (`%16`) 和上一波的 store 共用 UB 区域。y 的 fill 用 `pipe_barrier[PIPE_V]`——因为 y buffer 地址独立，只需要确保 V pipe 内部前序操作完成。

与 TTAdapter 中同一位置的代码对比：

```mlir
// TTAdapter：只有 linalg.fill，无同步
scf.if %x_9 {
  linalg.fill ins(%y : f32) outs(%x_3 : memref<1024xf32>)
} {hivm.unlikely_condition}
```

**为什么 TTAdapter 不需要同步？** 因为 TTAdapter 是 SPMD 抽象——假设单核单 program，没有并发。而 NPUIR 是真实的并行流水线——MTE2/V/MTE3 并行跑，必须显式同步。

### D.4 Load 前的 Event 等待

```mlir
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]           // 等上一个 wave 的 store
hivm.hir.load ins(..gm..) outs(..ub..)                        // x data load
// ...
hivm.hir.load ins(..gm..) outs(..ub..)                        // y data load
```

第一个 load (x) 前等待 `%13`（根据 wave 奇偶选择 EVENT_ID0 或 EVENT_ID1）。这是 double buffering 的关键：wave N 的 load 必须等 wave N-2 的 store 释放 buffer。

### D.5 计算 (vadd) 前的同步

```mlir
hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]     // MTE2 说：x load 完了
hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]    // V 等 MTE2 的信号
hivm.hir.vadd ins(%16, %15 : ...ub, ...ub) outs(%14 : ...ub)
hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]     // V 说：计算完了
```

V pipe 在启动 vadd 前，必须等 MTE2 pipe 把 x 和 y 都加载完。`wait_flag[MTE2, V, EVENT_ID0]` 确保了数据就绪。

### D.6 Store 前的同步

```mlir
hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]    // MTE3 等 V 完成
hivm.hir.pipe_barrier[<PIPE_MTE3>]                          // MTE3 内部barrier
hivm.hir.store ins(..ub..) outs(..gm..)
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]           // store完成→通知下一波
```

- `wait_flag[V, MTE3, EVENT_ID0]` — store 前等计算产出
- `pipe_barrier[<PIPE_MTE3>]` — MTE3 pipe 内部 barrier，确保上一次 store 的 DMA 传输完成
- `set_flag[MTE3, MTE2, %13]` — 本轮 store 完成后，通知下一波：buffer 可以重用了

### D.7 函数退出前的收尾同步

```mlir
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
hivm.hir.pipe_barrier[<PIPE_ALL>]
return
```

等待两个 event 通道都完成（所有 wave 结束），然后 barrier 所有 pipe（确保所有 DMA 传完、所有 store 落地），最后才能 return。

---

## E. Op Lowering：标准方言 → 硬件指令

| TTAdapter Op | NPUIR Op | 含义 |
|---|---|---|
| `linalg.fill ins(%y) outs(%mem)` | `hivm.hir.vbrc ins(%cst) outs(%mem)` | Vector Broadcast——向量单元广播一个标量到整个 buffer |
| `memref.copy %src, %dst` (GM→UB) | `hivm.hir.load ins(%src) outs(%dst) pad_mode=<PadValue> pad_value=%cst left_padding_num=%c0 init_out_buffer=false ...` | DMA 从 GM 加载到 UB。`pad_mode=<PadValue>` 表示如果源数据少于目标大小，用 `pad_value` 填满 |
| `arith.addf %x, %y : tensor<1024xf32>` | `hivm.hir.vadd ins(%x, %y : ...ub, ...ub) outs(%out : ...ub)` | Vector Add——向量单元逐元素加法。注意操作数**不再是 tensor，而是 ub memref** |
| `bufferization.materialize_in_destination` + `tensor.extract_slice` | `hivm.hir.store ins(%ub) outs(%gm)` | DMA 从 UB 存储到 GM。NPU 上的 store 指令本身就支持 subview（通过 DMA 描述符的 offset 和 size），不需要拆成 slice + store |

---

## F. 地址空间：从无到有

| 位置 | TTAdapter | NPUIR |
|------|-----------|-------|
| 函数参数 | `memref<?xf32>` | `memref<?xf32, #hivm.address_space<gm>>` |
| UB buffer | `memref<1024xf32>` | `memref<1024xf32, #hivm.address_space<ub>>` |

bishengir 编译器分析每个 memref 的访问模式：
- 从函数参数来的 → GM（Global Memory = HBM/DDR，芯片外）
- `pointer_cast` 创建的 → UB（Unified Buffer，片上 SRAM）

地址空间标记不是装饰——DMA 引擎根据源和目标的地址空间组合选择正确的硬件通道。

---

## G. 其他变化

| # | 变化 | 说明 |
|---|------|------|
| G1 | 新增 `hivm.hir.set_mask_norm`（循环体开头） | 硬件指令：设置向量 mask 为 normal 模式（全部元素参与运算）。在 TTAdapter 中 mask 通过 `scf.if` + subview 处理，但在 NPU 上还需要硬件 mask 寄存器配合 |
| G2 | `arith.maxsi`/`arith.minsi` → `affine.max`/`affine.min` | affine dialect 更适合表示 index 范围运算，bishengir 做了规范化转换 |
| G3 | `bufferization.to_tensor` 全部消失 | NPU 层面没有 tensor 的概念——所有数据都在 memref 中。tensor → memref 的边界在 NPUIR 中已不存在 |
| G4 | `func.return loc(#loc)` → `return`（无 loc） | 调试位置信息在最终 hardware IR 中被剥离 |
| G5 | 新增常量 `c12288_i64`, `c4096_i64`, `c8192_i64` 用于指针范围计算 | UB 大小和 buffer 偏移量。`8192 = 1024*8` 字节，`4096 = 1024*4` 字节（f32）= y buffer 起始位置 |

---

## 完整转换映射汇总

```
TTAdapter                                NPUIR                                 转换者
─────────────────────────────────────────────────────────────────────────────────────
memref<?xf32>                           memref<?xf32, #hivm.address_space<gm>>  bishengir: 地址空间推导
memref.alloc()                          hivm.hir.pointer_cast(offset, bound)    bishengir: UB allocator
                                          + hivm.multi_buffer annotation
%arg9 (program_id[0])                   hivm.hir.get_block_idx                  AutoBlockify pass
                                          + wave*40+local → block_id 计算
(无循环)                                  scf.for (wave loop)                     AutoBlockify + bishengir
(无同步)                                  set_flag / wait_flag / pipe_barrier     bishengir: GraphSyncSolver
linalg.fill                             hivm.hir.vbrc                           bishengir: vector lowering
memref.copy(GM→UB)                      hivm.hir.load (with pad attrs)           bishengir: DMA lowering
arith.addf                              hivm.hir.vadd                           bishengir: vector lowering
materialize_in_destination(UB→GM)       hivm.hir.store                          bishengir: DMA lowering
(无 sync lock handler)                  %arg0: i64 {ffts_base_address}           bishengir: FFTS register setup
(无 mask 硬件指令)                        hivm.hir.set_mask_norm                  bishengir: 硬件初始化
arith.maxsi / arith.minsi               affine.max / affine.min                 bishengir: 规范化
bufferization.to_tensor                 (消失)                                   bishengir: tensor→memref统一
tensor.extract_slice + subview          (合并进 hivm.hir.load/store 的             bishengir: DMA描述符
                                           offset/size)
(无)                                    hacc.arg_type annotations               bishengir + hacc
```

## 总结：这个转换的"为什么"

如果 TTIR → TTAdapter 是把 **Triton 编程模型拆解为标准 MLIR**，那么 TTAdapter → NPUIR 就是把**标准 MLIR 映射到硬件指令集**。

核心哲学：

1. **地址空间显式化**：每块内存必须标记 GM 还是 UB——DMA 引擎需要知道搬入/搬出的物理位置。

2. **同步显式化**：SSA 的隐式依赖被替换为 `set_flag`/`wait_flag`——因为真实的 NPU 流水线是并行乱序执行的，只有显式 barrier 才能保证正确性。

3. **SPMD → Wave 循环**：97 个 program 不可能真分配 97 个物理核。`get_block_idx` + `scf.for` wave loop 让 40 个物理核通过循环覆盖所有逻辑任务。

4. **Double Buffering via Event ID**：通过交替使用 EVENT_ID0/EVENT_ID1，相邻 wave 的 load 和 store 可以并行——wave N 在做计算时，wave N+1 的 load 和 wave N-1 的 store 可以同时进行。

## 与 costmodel 的关系

NPUIR 中的四类显式化对 costmodel 精度的影响：

| 显式化 | PipelineScheduler (TTIR) | HIVMAnalysis (NPUIR) | 精度提升 |
|--------|--------------------------|---------------------|---------|
| Wave Loop | `ceil(N/M)` 计算，正确 | 显式 IR 结构，等价 | 无 |
| 地址空间 | 从 op 语义推断，正确 | 显式标注 | 无 |
| **同步指令** | **固定 `7500 * numIters`** | **逐条 set_flag/wait_flag/pipe_barrier 建模** | **最大** |
| **Double Buffer** | **未建模** | **multiBufferSlots + buffer slot 状态追踪** | **大** |

详见 `hivm-analysis-deep-dive.md` 中 HIVMAnalysis 如何利用这些信息做精确的性能评估。
