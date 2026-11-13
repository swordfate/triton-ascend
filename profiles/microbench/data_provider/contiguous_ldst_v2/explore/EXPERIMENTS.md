# 探索实验总目录（EXPERIMENTS.md）

> 用途：回看每个实验"做了什么、参数含义、脚本在哪、结果在哪、回答了什么问题"。
> 路径约定：本地 = `D:\work_wj\contiguous-final\_explore\`，服务器 = `/root/work/explore_contig/`
>（两边 scripts/ 同名同步；results/ 定期 scp 回本地）。
> 跑法统一：`bash run_exp.sh <script.py> <卡号> <输出.jsonl> [脚本参数]`
>（run_exp.sh 负责 source CANN、选卡、落盘到 results/）。

## 0. 公共设施

| 文件 | 作用 |
|---|---|
| `scripts/common.py` | 测量框架：`prof_session`（ABAB 交替发射 + profiler 抓 Duration(us)）、`pair_diff_ns`（逐对差分中位数，与 FormS 标定同口径）、`stats_us`（min/q1/med/q3）、`health_check`（4KB store + 56 程序流水 load 跨卡基准，>1.3× 最优卡拒跑——防"看起来稳定实际延迟高"的卡）、`JsonlOut`（JSONL 落盘，每条带卡号/时间戳）|
| `scripts/run_exp.sh` | 服务器侧启动包装（CANN 环境、ASCEND_RT_VISIBLE_DEVICES、ulimit）|
| `scripts/analyze_e1_e3.py` | 把 E1/E3 的 JSONL 转成每轮边际表（diff_med_ns ÷ K / 边际递增）|
| `results/health.jsonl` | 历次健康门记录 |

测量方法学（所有实验通用）：一次会话内 A(探针)/B(对照) 交替发射 12–16 对，
逐对相减取中位数 = "这条访存的增量成本"；B 的选择：测 load 时 B=4KB store-only，
测 store 时 B=1 元素 scalar store。绝对值（probe.med_us）只在回答"整 kernel 总时间"
时使用。

## 1. E1：单发 vs 串行依赖 for + L2 三态（第一轮，卡 2）

- 脚本：`scripts/e1_single_vs_serial.py`
- kernel：`e1_load_once`（无循环单发）；`e1_load_loop`（`acc += load(src + boff +
  (step & winmask)*BLOCK + lane)`，iters/winmask/boff 均为运行时参数，
  do_not_specialize）；store 侧对称的 `e1_store_loop/once`。
- 参数含义：K=循环轮数 {1,2,4,8,16,32,64,128}；winmask 控制"每轮读哪块"——
  0=同块复读（same）、K−1=逐轮换块（warm，足迹 K×4KB 留在 L2）；
  boff 在 >128MB（L2 容量）范围轮转（cold，每次 launch 都读新行）。
  fp32 / BLOCK=1024(4KB) / grid=1 / simd+simt_only(W=4)。
- 结果：`results/e1_fp32_c2.jsonl`（104 点）
- 回答：单发≠K=1 循环（+~100ns 脚手架）；L2 冷/热是 3.0×（simd 单发 447 vs 149）；
  K 扫描的每轮边际曲线（warm 244→88，cold 679→166.8）；simd 的 warm 稳态 88ns
  与 FormS 公式 146ns(4KB) 同域；simt 的分界在"同块复读 79ns vs 换块 230ns"
  （SIMT DCache），warm≈cold。
- 已知缺陷：单发对照误用了 K=1 循环（后被 E1b 修正）；load/store 表曾被分析脚本
  键冲突混排（已修，按 kind 分组重出）。

## 2. E1b：K=1 循环比单发贵的 100ns 拆解（第二轮，卡 2）

- 脚本：`scripts/e1b_e3c_followup.py`（--skip-e1b 可只跑 E3c 部分）
- 变体（全 warm、4KB、grid=1、simd，各 3 会话取 min）：
  - a `e1b_once`：单发（137）
  - b `e1b_loop_acc`：运行时 for K=1 + acc（237）
  - c `e1b_loop_noacc`：运行时 for K=1，循环体内 load→store，无 acc（156）
  - d `e1b_static_acc`：`tl.static_range(1)`，循环折叠、仅剩 addf（134）
- 同时 dump 三个 kernel 的 TTIR 并逐 op 计数（JSONL `e1b_ttir` 行：
  scf_for/muli/andi/addf 等），证明结构差异与数字对应。
- 结论：19ns=运行时循环控制+地址计算（c−a）；add 本身免费（d−a≈0）；
  81ns=acc 跨循环边界（b−c，store 在循环外依赖 loop-carried phi）。
  950PR profile 无 `control_flow` 块 → 这些当前完全没计价。
- 结果：`results/e1b_e3c_c2.jsonl`（simd 部分；simt 的 a–d 分解尚未跑，见遗留）

## 3. E3：多条同形 load/store 的并发与依赖（第一轮，卡 3）

- 脚本：`scripts/e3_multiload.py`
- kernel：`e3_sl`（U=constexpr∈{1,2,3,4,6,8} 条 4KB load，CHAIN 开关：
  0=全部累加到 acc，1=乘法链 c=c*x_j，单 store）；`e3_st`（U 条独立 store）；
  对照=4KB store / tiny store。grid=1 为主，grid=56 补独立组。
- 结果：`results/e3_c3.jsonl`
- 回答：simd load 亚线性重叠（第 2..8 条边际 16–55ns，8 条总价 390 vs 串行 1080）；
  store 几乎线性（每条 +100~130ns）；grid=56 时 U 几乎不影响总时间。
  simt grid=56 组噪声过大作废。
- 已知缺陷（E3c 修正）：acc 消费把"独立"假设污染了；simt 结论被 E3c 反转。

## 4. E3c：真独立变量 load + SIMT 重测（第二轮，卡 2）

- 脚本：`scripts/e1b_e3c_followup.py`
- kernel：脚本运行时**生成真实 .py 模块**（`results/_gen_e3c.py`，triton @jit
  要求函数定义在可 import 的文件里，exec 生成不行）再 import。形如：
  ```python
  def e3c_ind_u8(src, dst, BLOCK):
      lane = tl.arange(0, BLOCK)
      x0 = tl.load(src + 0*BLOCK + lane)   # 每条 load 独立变量
      ...
      x7 = tl.load(src + 7*BLOCK + lane)
      tl.store(dst + 0*BLOCK + lane, x0 + 1)  # 每条 store 消费自己的 x
      ...
  ```
  对照 `e3c_ctl_u8`：同样的 U 条 store，但用 `tl.full` 常量替代 load
  → store 逐位抵消，diff 只剩 U 条 load。chain 变体沿用乘法链（单 store）。
- 方法升级：每点 3 个独立会话取 min（防慢会话）；JSONL 里存每个会话的
  probe/ctl/diff + `diff_med_ns_min`。
- 结果：`results/e1b_e3c_c2.jsonl`
- 回答：simd ind/chain 都亚线性（边际 ~30-50ns/条）——维持；
  **simt ind 完全线性 ~277ns/条（每条付全价）；simt chain 亚线性（~90-166ns/条）**
  ——SIMT 的多 load 价格取决于消费/活跃值结构。机制候选：①独立消费下 U 个
  结果块同时活跃（寄存器/DCache refill 队列）②每条 store 依赖自己的 load，
  SIMT 单 LSU 路径上 store 阻塞后续 load；SIMD 因 MTE2/MTE3 是独立队列
  （架构手册：不同指令序列可并行）所以 store 不阻塞 load。
  **待 camodel 指令轨迹裁决**；判别实验 E3e（ind 变体改单 store）见遗留。

## 5. E3b：stage 划分（同一指针多 load）（第一轮，卡 3）

- 脚本：`scripts/e3b_stage_split.py`
- 方法：compile-only，`compile_mode="simd_simt"` + 环境变量
  `TRITON_ASCEND_AUTO_SIMT_SCOPE=auto` + `TRITON_ASCEND_AUTO_SIMT_SCOPE_DUMP=<path>`
  + `TRITON_ASCEND_COSTMODEL_DETAILED_REPORT=1`（**三个条件缺一不可**，
  mode=off 会清空 dump），解析 route.json 的 stage_model。
- kernel：`e3_sl`（同 E3，直线型）与 `e3_loop`（循环体 U 条 load）。
- 结果：`results/e3b_c3.jsonl` + `results/e3b_*.json`（完整 route dump）
- 回答：直线型 U 条 load（同指针）→ 合进 1 个 `continuous_tile_memory` stage，
  每加一条 +54.85ns（U≥2 严格线性）；循环体 → 1 个 `loop_carried_recurrence`。
  附带发现：新连续访存通道不更新 `formula_evidence` 字段，归因不能只看它。

## 6. E3d：stage 分裂由指针决定（第二轮，卡 3）

- 脚本：`scripts/e3d_stage_split_pointers.py`
- 变体：v0=vector_add 原样（mask+pid+两指针+grid=97+harness 同款 kwargs，
  含 `logical_program_count_hint`）→ 复现 2 个 memory stage（161.6×2）+V1 壳；
  v1=同指针两块（grid=1）→ 1 个 stage（136.4，own 9 ops）；
  v2=两指针（无 mask/pid）→ **2 个 stage**（182.9+146.5，x 组 4 ops / y 组 3 ops）；
  v3=v2 去 hints；v4=v2+pid grid=97——都分裂 → V1/mask/pid/hints 均非原因。
- 机制（代码位）：`StageBoundaryAnalysis::analyze`（StagePartitioner.cpp:3081）
  相邻根合并 `mergePlainStage` 需要双方 anchorGroup<0 且（同 kind+schedule 或
  支持型+数据依赖）；`isSupportingSemanticKind`（:~3010）只认
  scalar/vector/index/predicate 族——**ContinuousTileMemory 不是支持型**；
  `hasDataDependencyOnStage`（:3015）要求 candidate 的某个操作数由 stage 已有
  op 产生。同指针时 y 侧地址计算与 x 侧共享 lane/基址链 → 依赖成立 → 合并；
  换指针后在指针切换处断开。**修正**：普通连续 load 不是 SIMT anchor
  （`SimtAnchorAnalysis.cpp analyzeAnchor` 只认 gather/histogram/scan/atomic/
  triangular/loaded-index/unstructured），第一轮"anchor 组决定分裂"的说法作废。
  精确到每条 IR 的边界落点还需 dump TTIR 对照（遗留）。
- 结果：`results/e3d_c3.jsonl` + `results/e3d_v*.json`
- 计价影响：多 tensor elementwise 的相邻 load 被切成多个 stage、各自按单发
  全价计价，多算 1.7–2.4×（182.9+146.5=329 vs 合并 136 vs 实测 ~190ns）。

## 7. E4：grid / L2 悬崖 / 复用（第一轮，卡 2+卡 3）

- 脚本：`scripts/e4_l2_grid.py`（--parts A|B|C 选择部分）
- Part A（grid 扫描，单发 copy）：`e4_copy1[(P,)]`——P∈{1,2,4,8,16,32,56,112}
  个程序各自拷贝自己的 4KB 块；warm（boff=0）vs cold（boff 在 320MB 轮转）。
  注意 diff 混含 P 条 load+store（对照只减 1 条 store），只用于相对比较。
- Part B（复用）：`e4_same`（56 程序读写同一块）vs `e4_copy1`（离散块），
  总流量相同。
- Part C（足迹跨 L2）：grid=56 流水 kernel，winmask 控制每程序窗口，
  总足迹 {8,32,64,96,128,160,224,288}MB 横跨 L2=128MB；K∈{2048,4096}
  两点斜率 = 每轮边际 ns（与 probe_block_pure 同口径）。
- 结果：`results/e4_c2.jsonl`（A/B simd + C simd）、`results/e4c_c2.jsonl`
  （C 双引擎，修了 kernel 名子串匹配 bug 后重跑）、`results/e4ab_simt_c3.jsonl`
  （A/B simt，卡 3）
- 回答：单发冷/热 4.4×(simd)/3.2×(simt)；load 的 L2 悬崖严丝合缝落在
  96→128MB（92→249 ns/round，2.7×），store 膝点更早（32→64MB）幅度更小
  （~110→~225）；同块 vs 离散在 224KB 量级无差；发现 10× 慢 launch 瞬态
  （B 部分 q3 6.9–9.2µs vs 中位 0.64µs；A simt P2-cold 一会话 cv=39.7）。

## 8. E5：100-case 形态普查（本地/服务器，无 NPU）

- 脚本：`scripts/e5_census.py`
- 方法：读 `generalization_matrix_v140_all_100.json` → AST 抽每个 driver 的
  `@triton.jit` kernel → 统计 load/store 条数、mask、循环、grid-stride、
  地址形态（1-D/2-D/block_ptr）、gather。
- 结果：`results/e5_census.jsonl`
- 回答：≥2 条 load 的 case 82/100；masked 67/100；循环/grid-stride 75/100；
  2-D 地址 60/100；indirect 30/100；block_ptr 12/100。
  缺陷：按 driver 计数（多 kernel 文件会抬高计数），方向性结论可靠、
  精确数字偏大。

## 9. 标定口径考证（无新脚本，读历史提交）

- 对象：`probe_forms_single_shot.py`（ABAB 16 对差分、固定地址 L2-warm、grid=1）、
  `probe_block_pure.py`（grid=1 acc 链，K∈{2048,8192} 斜率，min-of-4）、
  `probe_bw_throughput.py`（grid=56、UNROLL 独立组、足迹按 64MB 反推压在 L2 内、
  k∈{k,2k,4k} 三点回归）。squash3 分支 `BW_CALIBRATION.md` 记载 L2=128MB。
- 结论：三族全是差分/边际量（"探针墙钟"说法修正），但**全部 L2 热态**，
  没有任何冷态覆盖。

## 10. 遗留实验清单

| 项 | 目的 | 脚本状态 |
|---|---|---|
| camodel 指令轨迹 ×3 | ①E1b 的 81ns acc 跨边界是什么指令；②E3e v2"store 位置无关"的指令级原因（后端重排 or DCache 写路径争用）；③E3d v1/v2 边界落在哪条 IR 上 | 未写 |
| E1b simt 补齐 | a–d 分解目前只有 simd（--skip-e1b 跳过了）| 脚本就绪，重跑不加 flag |
| E4-C simt + int8/fp16 | 悬崖形状的引擎/dtype 矩阵补全（simt 是否也有 DRAM 悬崖未测）| 脚本就绪，改参数 |
| 4KB group-marginal 标定 | 方案B 退款曲线（E3/E3c 已有 ~30-50ns/条，需按格子入库）| 未写 |
| 串行守护大轮数（v8 §6.3）| §3.2 公式在 K 几百轮时是否进 ±10% | 未写 |
| ~~E3e 判别实验~~ | ✅ 已完成，结论见 §11（store 数量是串行化主因，活跃值被否决）| `scripts/e3e_store_structure.py` |

## 11. E3e：SIMT 多 load 线性/亚线性的判决实验（第三轮，卡 2）

- 脚本：`scripts/e3e_store_structure.py`（生成模块 `results/_gen_e3e.py`）
- 三个变体（全 warm、4KB fp32、grid=1、每点 3 会话取 min；对照做同样的
  加法/store 但用 tl.full 常量替代 load → 加法和 store 全部抵消，diff 只剩
  U 条 load）：
  - v1_1store：U 条 load 进 U 个变量，**最后一条 store** 汇总（U 个活跃块、1 条 store）
  - v2_endstore：U 条 load 全部先发，**U 条 store 全部放在 loads 之后**
    （**解释修正（用户指出）**：v2 仍线性不能证明"硬件位置无关"——
    编译器会重排，源码顺序≠最终指令序；它只证明**源码位置**不影响
    编译产物。位置是否在硬件级可重叠，需 camodel 看最终指令序）
  - v3_interleave：每条 load 后跟一条 store（=E3c ind 参照）
- 结果（diff_med_ns_min，simd / simt_only）：

| 变体 | U=1 | U=2 | U=4 | U=8 | 每条边际 |
|---|---:|---:|---:|---:|---|
| v1_1store simd | 172 | 189 | 260 | 400 | ~17-35 亚线性 |
| v2_endstore simd | 171 | 203 | 270 | 427 | ~32-39 亚线性 |
| v3_inter simd | 169 | 204 | 266 | (270)⚠ | 亚线性（U8 行为坏会话，E3c 同构=430 可信）|
| v1_1store simt | 270 | 401 | 550 | **1081** | ~75-133 **亚线性** |
| v2_endstore simt | 260 | 579 | 1111 | **2190** | ~266-360 **线性** |
| v3_inter simt | 266 | 570 | 1142 | **2197** | ~277 **线性**（复现 E3c）|

- **判决**：
  - v1（U 个活跃块 + 1 store）**亚线性**，且与 chain 几乎逐点相同
    （U8: 1081 vs E3c chain 1086）→ **H2（活跃值/寄存器压力）被否决**；
    "chain vs indpt"是假变量，真变量是 **store 条数**（chain 恰好只有 1 条 store）。
  - v2（U 条 store 全部挪到 loads 之后）**仍线性** → store 的**位置**无关，
    **数量**才是关键：SIMT 上每条依赖 store 的开销无法与 load 流重叠，
    无论它排在哪。
  - simd 三种 store 结构全部亚线性（MTE2/MTE3 独立队列，store 不占 load 通道）。
- 结论（修正 E3c 的表述）：SIMT 多 load 价格 = f(U 条 load, **S 条依赖 store**)；
  S≥1 时每条 store 付 ~全价且串行化整条流，S=0 时 load 亚线性重叠。
  分类学上 (a)(b) 的 "chain/indpt" 子类对 simt 应改为按 **store 数 S** 区分。
- 遗留：v2 位置无关的指令级原因（后端重排 or DCache 写路径争用）→ camodel。

## 12. 标定分类提案 v2（讨论稿）+ 方案B 待办

> 用户提案：按 (循环结构 × 访存内容) 分类，替代 单发/串行/并行。
> (a) 无 for / static for（static_range 前端展开，TTIR 与直线型同构）：
>     (i) 多 load chain；(ii) 多 load indpt；(iii) store；
> (b) 动态 for（运行时 trip count）：同样三分。

**评估：采纳，附两处修正 + 两条必须补的轴。**

修正 1（E3e）：(i)/(ii) 的"chain/indpt"（消费依赖）对 simd 无差别、对 simt
的真变量是 **伴随 store 条数 S**——子类应改为 (i) 多 load（参数 U，simt 另需 S）
(ii) store（参数 S，近线性）。
修正 2（E3d）：**同指针/不同指针**必须进分类——它决定 U 是否落在同一 stage
（groupCount 生效）还是被切成多个 stage 各按单发计价（多算 1.7-2.4×）。
补轴：residency（simd 的 L2 冷热 / simt 的 DCache 命中）与 K（仅 (b)）。
单发不再是独立家族，而是 (a) 且 U=1、S=1 的格子。

各格子实测行为速查（grid=1、4KB fp32）：

| 格子 | simd | simt | 机制 |
|---|---|---|---|
| (a) 多 load，S=0 | 亚线性 ~30-50/条 | 亚线性 ~75-133/条 | load 背靠背，延迟重叠 |
| (a) 多 load，S=U | 亚线性 ~30-50/条 | **线性 ~277/条** | simt: 依赖 store 串行化；simd: MTE2/3 分离 |
| (a) store | 近线性 +100~130/条 | 近线性 +110~155/条 | store 路径不重叠（双引擎）|
| (b) chain（acc 链）| m(K) 曲线：warm 244→88、cold 679→167 | warm≈cold≈230（DCache miss 主导）；同块 79 | 每轮边际随 K 摊薄；simd 看源（L2/DRAM），simt 看 DCache |
| (b) indpt（流水）| = bw 族域；悬崖在 128MB L2 | 未测（E4-C 只有 simd）| 吞吐域 |
| (b) store 循环 | warm 88→101、cold→158 | warm 125→54 | E1 store 表 |

**待办顺序**：
1. 巩固标定（当前优先）：按上表格子补齐——尤其 (b) 的 m(K) 曲线族入库、
   (a) 的 4KB group-marginal、simt 的 E4-C 足迹扫描、E1b simt 补齐；
2. **方案B（stage 照旧切，route 层跨 stage 兄弟 load 群退款）**——标定巩固后再动；
   基于 subLineLoadClaims 模式扩展认领条件（直线型区域、相邻 load-stage、
   同 elements/eb/mask、方向=load），退款曲线=group-marginal；
   挂开关 + 100-case A/B（盯 cumprod/index_put 擦边）；
   注意 vector_add 教训：load 侧修正必须与 store/固定侧同时动；
3. camodel 裁决（E1b 81ns、E3e v2 位置无关、E3d 边界精确落点）。

## 13. E6：循环 vs 直线型（burst）的本质区别（第三轮，卡 3）

- 脚本：`scripts/e6_loop_vs_burst.py`（生成模块 `results/_gen_e6.py`）
- 动机：用户质疑"运行期 for 为什么要单独建模"。本实验用**真实流式地址**
  （base+step*BLOCK，不用 window 复读）把**同一访存模式**分别写成
  动态循环（K∈{1..128} 运行时 iters）和直线型（U∈{1,2,4,8} 交错
  load/store），对比每实例边际。足迹 K×8KB ≤ 1MB « L2 → warm。
- 结果（diff_med_ns_min；simd / simt）：

| K / U | loop_ind simd | loop_ind simt | burst simd | burst simt |
|---|---:|---:|---:|---:|
| 1 | 188 | 767⚠ | 160 | 240 |
| 2 | 278 | 1695 | 198 | 552 |
| 4 | 498 | 3746 | 271 | 1126 |
| 8 | (None) | 7644 | (None) | 2242 |
| 16 | 1713 | 15408 | — | — |
| 32 | 3344 | 31021 | — | — |
| 64 | 6601 | 62209 | — | — |
| 128 | 13047 | 124637 | — | — |
| **稳态每实例** | **~102ns/轮** | **~980ns/轮** | **~37ns/条** | **~277ns/条** |

- **判决：(a) burst 与 (b) loop 是本质不同的执行上下文，必须分开建模。**
  - simd：同样的 4KB 独立 load+store，直线型每实例 ~37ns（8 条同时在飞，
    MTE2 队列深 16 吃满），循环每轮 ~102ns（每轮约一次发射在飞）——**2.7×**。
  - simt：burst ~277ns/条 vs loop ~980ns/轮——**3.5×**（此数混入流式 store；
    E1 拆分：load-only 循环 ~230、store-only 循环 ~60，合起来 980 超加性
    → simt 的 load 流与 store 流互相干扰，与 E3e H1 一致）。
  - 机制：直线型代码的所有访问在同一指令流里、队列深度内全部在飞；
    循环受每轮发射节流（in-order + 回边 + body 依赖），**每轮约一轮在飞**。
    交叉验证：E4-C grid=56 左侧 92ns/轮 ≈ grid=1 循环轮 102ns——
    **bw 族"多程序聚合吞吐"就是 56 个程序各一轮在飞的叠加**，
    单程序循环价与满核聚合价自洽。
  - simd 循环轮对 body 内容不敏感（load-only 88 / store-only ~100 /
    load+store 102）——循环轮价是上下文常数，per-instance 内容价只在
    burst 里成立。
- ⚠ 未解：simt loop K=1 = 767ns，比同构直线型（240）高 3.2×——simt 的
  循环脚手架远大于 simd（E1b simd 仅 ~100ns）；camodel 待查。
- **分类学结论**：用户的 (a)/(b) 划分成立，但表述应改为
  **上下文（burst=直线/已展开 vs loop=真循环）**——判定标准是
  "TTIR 里是不是真 scf.for"（static_range 前端展开→直线型→burst；
  constexpr range 若未被后端展开仍是 scf.for→loop）。统一形式：
  `T = S(loop) + N × m(context, residency, U/S, …)`，
  N=U（burst，显式 ops）或 K（loop，3050 管道输入）。

## 14. E7：同一份工作（N 对 load+store），只变代码形状（第三轮，卡 3）

- 脚本：`scripts/e7_same_work.py`
- 设计：固定总访问数 N∈{8,32}（4KB 对），四种形状：A=动态循环 K=N（每轮 1 对）、
  C=直线 N 对、D=static_range(N)（前端展开）、E=动态循环每轮 static 展开 4 对。
  对照做同样的 store（常量替代 load）。
- **方法注意**：A 的对照（运行时循环 store kernel）自身波动巨大（cv 1.3~72），
  成对差分全部作废；改用**绝对 kernel 中位数**对比（同工作、同发射批次、
  只变形状，无需对照）。这也暴露：**simt 的循环 kernel 存在双峰状态**（见下）。

### 绝对 kernel 中位数（µs，3 会话）

| 形状 | simd N=8 | simd N=32 | simt N=8 | simt N=32 |
|---|---:|---:|---:|---:|
| A loop1（动态循环） | **1.38** | **3.81** | 1.78-1.89 | **4.64 / 13.15**⚠ |
| C burst（直线） | 1.60 | 4.01-4.04 | 3.90 | 13.55 |
| D static_range（展开） | 1.58-1.60 | 4.01-4.04 | 3.91 | 13.53 |
| E loopu4（每轮 4 对） | — | 4.03-4.05 | — | 13.19-13.38 |

- **D ≡ C**（三组引擎/N 全部一致）：static_range 展开后与直线型完全同价
  ——用户"static loop 归入无 for"的判断再次确认。
- **simd N=32：A=C=D=E（差 <5%）**——同一份工作，循环与直线**同价**。
- **simt：C/D/E 一致（13.2-13.6µs），A 双峰**：两次会话 4.64µs（比 burst 快
  2.9×）、一次 13.15µs（与 burst 同）。simt kernel 存在**会话级双峰状态**
  （同一 kernel、同一卡、相邻会话 2.8× 差异）。

### 对 E6 结论的撤回与修正

E6 的"循环比直线贵 2.7×(simd)/3.5×(simt)"是**外推假象**：
- burst 的 ~37ns/条边际只在 **U≤8（≈MTE2 队列深度 16 内）** 成立；
  外推到 U=32 错了——实测 U32 直线型 ~125ns/条，**与循环轮收敛**。
  机制：直线型的并发优势被 MTE2 队列深度封顶（≤16 笔在飞），超出部分
  退化为传输受限，与循环的"每轮一笔"殊途同归。
- simt 侧的"3.5×"混入了 E6 的流式 store 对照失配与双峰状态，不可信。
- **正确表述**：直线型相对循环的并发优势**存在但被队列深度封顶**——
  只影响每个程序**前 ~8-16 笔**（≤~64KB）；超过之后两种形状收敛到
  同一个传输受限价（simd ~100-125ns/4KB·实例）。
- 对分类学 v2 的修订：顶层不再是"burst vs loop 两种价位"，而是
  **N（=U×K 总实例数）决定是否进入队列深度内的组摊销区**；
  循环的脚手架 S（simd ~100ns 固定）仍在。
  真正稳定成立的因子：residency（L2/DCache）、S（simt store 数）、
  dtype/bytes、以及 N 是否 ≤ 队列深度。

### ⚠ 新发现：simt 会话级双峰（本轮最重要的副产品）

同一 simt kernel（E7 A_loop1_N32）在**同一张卡**的相邻 profiler 会话中
中位数 4.64µs 与 13.15µs（2.8×），C/D/E 稳定在 13.2-13.6。回看历史：
- v8 §6.4 记载的 bw 标定双峰（同几何窗口热 26ns vs 冷 73~442ns，
  63/187 组 >1.8× 散布）很可能就是同一现象；
- E4-A simt P2-cold 会话 cv=39.7、E4-B 的 10× 慢 launch 长尾同源；
- **在查清双峰机制（DVFS/频率耦合/其他租户/电源状态？）之前，
  所有 simt 循环/流水的跨会话绝对值都不可信**——这可能是历史上
  simt 标定反复不准（bw.simt MAPE 26.9%）的根因。
- 立即行动项：长跑单 kernel（200+ launch 连续打印）刻画状态转换；
  探索固定状态的手段（预热时长/锁频/隔离）。

## 15. PR 3050 与本探索的关系（2026-11-12 查证）

PR 3050（triton-ascend）：**把"launch 绑定的运行时标量"接入循环 trip count 分析**——
`SelectSimdSimtCostModelPass` 已有机制把 capability JSON 的
`runtime_scalar_values=[{argument_index, value}]` 写成函数参数属性
`ascend.costmodel.runtime_value`；PR 在 `LoopWorkloadAnalysis` 新增
`launchBoundInteger`：动态 `scf.for` 的 lower/upper/step 若是带该属性的标量
参数，则解析出 `launchBoundTripCount`（workloadKind="exact_launch_bound"），
`workloadTripCount` 用它；另触 `StagePartitioner::hasKnownAtMostOneIteration`
与 SimdSimtCostModel 一处。

对本探索的意义：**这正是 (b) 类格子"K 轴"的输入管道**——我们证明了 m(K)
必须进模型（E1 曲线），3050 让 K 在编译期可知（host 绑定真实值）。
但它只是管道：profile 里还没有任何按 K 取值的曲线族可查（m(K) 待标定入库），
且 K 已知后选择哪个格子仍需新分类器。衔接核对项：capability JSON 的
runtime_scalar_values 由 host 生成，合入后需确认 100-case harness 是否把
真实 trip count 绑进来（harness 目前只传 logical_program_count_hint）。

## 16. E8：simt 双峰状态刻划（第四轮，卡 3 + 卡 2）

- 脚本：`scripts/e8_bimodal.py`（每 launch 逐条落盘的时序版测量；
  两个小坑已修：--modes 默认值须为 list（nargs="+" 迭代字符串得 's'）、
  汇总 print 格式串）
- 设计：一个 profiler 会话内固定顺序交替发射
  [simt_loop(K=32 流式 load+store 循环), simd_ref(4KB store), simt_line(8 对直线),
  simd_ref] × 60 = 240 launch，逐 launch 落盘时序；卡 3、卡 2 各 3 会话。
- 结果：
  - **会话内极稳**：simt_loop 卡3 s1 n=117 med=13.27µs（±2%）；卡2 三会话
    med=13.31/13.31/13.31（±1.5%）——无任何会话内状态翻转；
  - **跨卡可复现**：卡 3 与卡 2 慢态逐位一致（13.27 vs 13.31µs）；
  - **simd 免疫**：simd_ref 恒 0.8-1.1µs，与 simt 状态无关；
  - **慢态是常态**：E8 两卡全部会话均慢态（loop ~415ns/对、line ~493ns/对）；
    E7 快态（145ns/对）、E3c/E1 中态（230-275ns/对）未在 E8 发射模式下复现；
  - simt_line（直线 8 对）≈ simt_loop（32 轮）每对同价（493 vs 415）——
    形状（直线 vs 循环）不改变 simt 每实例价（又一次印证 E7）。
- **结论**：不是每 launch 随机噪声，是**会话级、跨卡可复现、simd 免疫、
  只打击 simt GM 访存路径**的模式差，幅度 ~2-3×（145 / 230-275 / 415-490
  ns/对至少三档）。触发器未知，**最大嫌疑是邻接 kernel 的发射模式**
  （E7 快态会话里 simt_loop 与同形 store-loop 控制交替；E8 慢态里与
  simd_ref/simt_line 交替）。
- 下一步（E9）：同一 simt loop kernel 分别与 ①同形 store 循环 ②simd_ref
  ③无邻核 交替，各 3 会话。若邻核决定模式 → 快态是 kernel 间发射重叠假象，
  标定以"干净会话慢态"为真值；若非 → 查时间/租户。
- 护栏协议（trigger 查明前）：每个 simt 测点会话内交替测量 simt 参考
  kernel，参考越界 ±10% 弃会话（E8 证明会话内稳定，gate 有效）。

### E9 判决（同日追加，`scripts/e9_trigger.py`）

单进程内交错 7 个会话（P1=store-loop 邻核 / P2=simd_ref 邻核 各 3 次 +
P3=无邻核 1 次，时间上不可分）：

```
P1_storeloop_nb med=13.315 / 13.306 / 13.326 µs   ← E7 快态的同款发射模式
P2_simdref_nb   med=13.321 / 13.329 / 13.308
P3_alone        med=13.213
（全部 415ns/对，±1%，逐 launch 稳定）
```

- **邻核假设被否决**：E7 快态的同款交替模式今天不复现快态。
- 结合"同一进程内 E7 的 A_loop1 三会话 4.6/13.2/4.6 翻转、而 C/D/E 同时段
  稳定慢态"：模式是**时变的环境状态**（嫌疑：主机/设备功耗时钟管理或其他
  容器租户；npu-smi 损坏无法直接观察租户），且可能与 kernel 类型交互。
- **对策升级**（比单靠会话门更强）：
  1. simt 标定一律用**同会话 probe/ctl 成对差分**——两者同处一个模式，
     模式倍率在差分中抵消（E3c/E3e 的 simt 结论因此可信）；
  2. 每次 launch 落盘时序，按状态聚类取主导态中位（防会话中途翻转，
     E7 A 的 cv>1 即此原因）；
  3. 会话级 simt 参考 kernel 门保留。
- 绝对值口径：当前可复现真值 = 慢态（loop/line/burst ≈ 415-490ns/对）；
  快/中态待环境复现后再定归。

## 21. E12：三卡并行同会话 A/B——cold 漂移 + 卡 1 病态实锤（第五轮）

- 脚本：`scripts/e12_cold_ab.py`（同会话背靠背 A/B + 逐 launch 落盘；
  两处小坑：--pairs 与公共参数重复、session 打印变量名）
- 起因：同一"cold acc-chain 循环（K=64）"一天内出现三个值——
  E1 上午（卡2）~167、calB 晚上（卡3）~87、E1 脚本重跑（卡3，晚 10 分钟）~243。
  用户提出换卡验证。
- 设计：卡 1/2/3 **同时刻并行**，每卡 6 会话（simt×3 + simd×3），
  会话内背靠背交替 [e1_load_loop, cb_chain]（两份只有 `& winmask` 差异的
  同构 cold 循环），K=64。
- 结果（kernel 绝对中位 µs，K=64 cold）：

| 卡 | simt e1/cb | simd e1/cb | 判定 |
|---|---|---|---|
| 1 | **283.2 / 283.7** | **118.7 / 117.9** | **病态：比卡 2/3 慢 ~19×/18×，但会话内 min-max ±3% "看起来极稳"** |
| 2 | 15.5-15.7 / 15.5-15.7 | 6.24-6.35 / 6.24-6.33 | 正常 |
| 3 | 15.1-15.6 / 15.1-15.6 | 6.25-6.34 / 6.25-6.34 | 正常，与卡 2 逐位一致 |

- **结论四条**：
  1. **卡因素实锤**：卡 1 当前整档慢 ~19×（两引擎同步慢），且会话内
     波动 <3%——不换卡根本看不出来。E12 用了 --skip-health，健康门
     （4KB store / 流水 load 跨卡对比）本可当场拦下。
  2. **卡 2 ≡ 卡 3**（逐位一致）：健康卡之间无差异。
  3. **两个 kernel（e1 版 / calb 版）同会话差 <2%**：243-vs-87 不是
     kernel 结构差异。
  4. **cold 有时间维模式**：同一卡 3 在 ~40 分钟内 87 → 243ns/轮
     （calB 22:05 vs E12 22:4x），加上上午 167——cold 至少三档
     （87/167/243），而 warm 全天稳定 88-93。**cold 是自带环境模式的
     测量对象，warm 不是**。
- **协议升级（标定强制）**：
  1. 健康门不可跳过（删除所有 --skip-health 的正式使用）；
  2. 健康门增加 **cold 流式参考**（当前只有 warm 流水 + 4KB store）；
  3. cold 数据按"参考归一 + 多时段重测"处理，单次 cold 绝对值不入库；
  4. 卡 1 当前禁用，恢复需复测健康门。

## 17. 分类提案 v3（最终建议）——"到底怎么划分"

**你的 (a)/(b) 大框架保留，维度重新定义（全部有实验背书）：**

- 维度 1 **引擎**（simd/simt）：机制不同（MTE2/3 队列 vs 单 LSU+DCache），永远分列。
- 维度 2 **访存内容**：(i) 纯 load 组（U 条，S=0）；(ii) load+依赖 store（S≥1）；
  (iii) 纯 store。**chain/indpt 从一级维度降级**：E3e v1 证明消费结构不影响
  load 发射（v1 与 chain 逐点重合）；simd 循环内 chain(88)≈indpt(102)。
  simt 的一级变量是 **S（store 条数）**（E3e：S=0 亚线性、S=U 线性）。
- 维度 3 **N = 实例总数**（burst 的 U；循环的 K）：N≤~16（队列深度）为组摊销区
  （首条全价+边际）；N>16 为传输受限区（simd warm ~100-125ns/4KB 实例，线性）。
  E7 证明 burst 与 loop 在 N=32 同价——**N 是唯一携带价位信息的数量维度**，
  循环 = "让 N 变大 + 固定脚手架 S_loop（simd ~100ns，E1b）"。
- 维度 4 **residency**：simd warm/cold（128MB L2 悬崖 92→249ns/轮，E4-C）；
  simt DCache 命中/缺失（79 vs 230）。
- 维度 5 现有轴：dtype/bytes/mask/W。
- 维度 6（识别器输入，非公式轴）**同指针/异指针**（E3d：决定 U 是否合进同一
  stage；异指针多算 1.7-2.4×）。

| 格子（引擎×内容×N 区）| 公式 | 已有数据 | 缺口 |
|---|---|---|---|
| simd 单发 (N=1) | FormS | ✓ warm | cold 单发 |
| simd load 组 (S=0, N≤16) | 组摊销 first+marginal | E3c/E3e warm | cold；4KB 口径入库 |
| simd load+store (N≤16) | 线性（E3e v2/v3）| 部分 | 专项确认 |
| simd 任意 (N>16) | N × c_transfer(warm/cold) | E6/E7 warm；E4-C 悬崖 | cold 曲线 |
| simd store 组/循环 | 近线性+膝点 | E1/E4-C | 膝点细化 |
| simt 全部 | **冻结** | E1/E3c/E3e | **双峰解决后按同表重标** |

探针组织原则：K/U 运行时参数（一次编译扫全曲线）、流式真实地址、
residency 由足迹控制（window 复读只留给单发）、每点 3 会话取 min、
simd 参考门 + simt 参考 kernel 会话门。

## 18. E10：S 扫描——S 的中间值怎么定价（第五轮，卡 2）

- 脚本：`scripts/e10_s_sweep.py`
- 构造（U=8 条 load 固定，S = 前 S 条 load 各配一条依赖 store，
  其余 load 求和进一条收尾 store；对照做完全相同的 store/求和但用常量）：
  diff 恰好 = 8 条 load。
- 结果（diff_med_ns_min，今日慢态口径）：

| S | 0 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| **simt** | 1069 | 1244 | 1378 | 1593 | 2202 |
| **simd** | 394 | 393 | 386 | 396 | 422 |

- **simt：T(S) ≈ 组价(U=8) + S × ~142ns，线性内插成立**（S=8 预测
  1069+8×142=2205 vs 实测 2202）；只需标定每格子的 store 罚系数，
  不必逐 S 标定。
- **simd：对 S 完全不敏感**（394→422 在会话噪声内）——MTE2/MTE3 独立队列，
  store 不占 load 通道（E3e v2/v3 的机制再次印证）。
- 分类 v4 修订：维度"内容修正"（第四层）的 simt S 因子定价公式确定为
  `T(U,S) = load_group(U) + S × penalty(dtype,bytes)`。

### 分类 v4 增补（回应"static for 归属 / 三档 residency / grid 结论"）

- **static for 归属**：分类看 TTIR。`tl.static_range` 前端展开 → TTIR 无
  scf.for → 第二层②（组）；`for i in range(8)`（constexpr）在 TTIR 里
  **仍是 scf.for** → 默认落 ③（循环族，含脚手架）。坑：若后端对小 constexpr
  K 做了展开，循环族价会多算脚手架——需 dump 一个 constexpr-range kernel
  的最终 ISA 验证（待办）。
- **三档 residency 映射（按引擎两套）**：
  simd：hot≡warm（MTE 直连 L2：同块复读 91-112 ≈ L2 流 88-92）→ 只需
  warm/cold 两档；simt：hot（DCache 命中，同块复读 79）与 warm（换块但
  L2 内 230）有 2.9× 差 → 需要 hot 档；simt warm≈cold（230≈230）→
  合并为 miss 档。判定规则（TTIR 可算）：hot=同一 (ptr,off) 每 K 轮复读；
  warm=launch 总足迹 ≤ 128MB；cold=超过。
- **grid 结论**：per-program 访存价与 grid 无关（E4-A 摊薄是墙钟假象；
  E4-C grid=56 92ns/轮 ≈ grid=1 88-102；E3 的 U 在 grid=56 不敏感）。
  grid 只有两个间接作用，都在 route 层：①residency 判定输入
  （launch 总足迹 = P × 每程序足迹 vs 128MB）；②聚合带宽 max
  （总足迹 > L2 时 wall ≥ 总字节 / DRAM 带宽 ~0.9TB/s）。

## 19. E11：constexpr `range(8)` 在最终 ISA 里是循环还是展开？（第五轮）

- 脚本：`scripts/e11_static_range_isa.py`（dump asm 全部键
  [ttir/ttadapter/source(CCE)/bcmlir/npubin] 并逐键数循环标记）
- 结果（simd）：

| kernel | TTIR | ttadapter | source(CCE) | bcmlir | 判定 |
|---|---|---|---|---|---|
| `tl.static_range(8)` | 0 | 0 | 0 | 0 | **全程展开 → 组/burst 格子** |
| `for i in range(8)`（constexpr）| **1** | **1** | **1** | **1** | **全程保留循环 → 循环族** |
| `for i in range(iters)`（运行时）| 1 | 1 | 1 | 1 | 循环族 |

- **结论（static_range 待办闭环）**：即使 trip count 是 constexpr，
  `range` 也**不会被后端展开**，最终 CCE 里就是循环——分类规则定为：
  `tl.static_range` → 组格子；一切 `range`（constexpr 或运行时）→ 循环族。
  建模含义：`range(8)` 与 `tl.static_range(8)` 是**两种价**（小 N 时循环
  每实例 ~2× 于展开，见 E6/E7 的 N≤16 区间）。

## 20. 3050 新基线（进行中）

- 上游 main（fdaacb7bf4）**不含** 3050（grep launchBoundTripCount=0）；
- 新基线取 **PR3050 head `229fd60064`**（"Keep launch-bound loops accurately
  priced after unified analysis"，基座 = d4667dab9+#2884，与我们基线对齐）；
- 已建 worktree `/root/work/ta-3050`，构建脚本 `/root/work/build3050.sh`
  （=build37.sh 改路径）；首败原因：`AscendNPU-IR` submodule 未初始化
  （worktree 不继承主 checkout 的 submodule 内容），处理中；
- 构建成功后：安装 wheel → runpath8 式 8 分片跑 100-case compile-only →
  与 `simdsimt_costmodel_calibration_20260914.md` §1.0.1 实测表对比，
  摸清新基线；对比完恢复安装 calib wheel（dist/ 里有 git04bcfe6f）。

### 20.1 结果（2026-11-12 晚，wheel git2e9efcac）

管线：fetch 上游分支 → worktree `/root/work/ta-3050` → 干净重建
（第一次增量构建因跨 commit 切换残留旧对象导致 libtriton.so 未定义符号
——**切 commit 后必须 rm -rf build 全量重建**，ccache 会兜住未变文件，
实测 ~10 分钟）→ 安装 → `run3050.sh`（runpath8_card123 改输出根
/tmp/RUN3050）8 分片 → 对比脚本解析 §1.0.1 表（注意 case 名带反引号、
千分位逗号；**该分支 route.json 的 candidate_elapsed_costs 单位是 ns**）。

**新基线（feature/simd-simt-compile-mode @ 2e9efcacb8）vs §1.0.1 实测：**

| 指标 | 新基线 3050 | 老基线 6232f64a4（v8 参照）| calib 分支（v8）|
|---|---|---|---|
| <10% | **88/100** | 88/100 | 89/100 |
| median | **3.43%** | 4.10% | 3.40% |
| mean | 5.30% | 5.66% | 5.52% |
| p90 | 10.49% | — | 11.6% |
| FAIL 数 | 12 | 12 | 11 |

- FAIL 12 项：rowred_r16(-33.8%)/rowred_r64(-10.3%)、gather_dot×3
  (-14.8/-17.3/-21.7%)、grid_stride_dot_p56/p112(-20.0/-22.0)、
  sum_high(-11.6%)、cumprod(+12.4%)、strided_scatter(-10.5%)、
  reduce_sum_epilogue(-17.5%)、associative_scan(+32.9%)——与我们已知的
  结构性 FAIL 高度重合；
- 抽查 vector_add(2.864)/reduce_sum_epilogue(2.380)/rowred_r16(15.880)
  与 v8 参照基线**逐位一致** → #2443/#2983/#3108/#3050 对这些 case 的
  预测无影响；
- **值得注意**：本分支包含 #2443（我们在 d4667dab9 revert 的
  loop-overlap model），整体 median 反而略好于 6232f64a4——当年 revert
  的理由需要结合 #2983/#3050 的后续修复重新评估；
- calib wheel（git04bcfe6f）已恢复安装。
