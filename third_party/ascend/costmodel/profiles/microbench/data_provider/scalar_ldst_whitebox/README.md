# scalar load/store 白盒报告

> 目标 6 个 scalar-dominated kernel 的针对性建模与验证请看 [`README-targeted-v1.md`](README-targeted-v1.md)。
> 2026-09-19 更新：default profile 的 SIMT `uniform_load_fill_system_cycles` 由 524 联合重拟合为 **480**；下文旧表中出现 524/530 的 SIMT load 数值均为历史值，最新目标场景结果见 [`README-targeted-v1.md`](README-targeted-v1.md)。

> **汇总版（先看这个）：[README-tidy.md](README-tidy.md)**

> 范围：目前包含 **单 scalar load op**（SIMD MainScalar / SIMT 32T / SIMT 1T）、**4-op scalar load**（same-line / diff-line，SIMD / SIMT 32T）、
> **单 scalar store op**（SIMD / SIMT 32T）、**4-op scalar store**（same-line / diff-line，SIMD / SIMT 32T）、
> **scalar add compute**（SIMD MainScalar / SIMT 32T）、**load → vector compute**（1op / 4op，SIMD / SIMT），
> 以及 **Triton SIMD scalar store 的 MTE3 链路**（§5.4）与 **costmodel 公式调整 / 6-kernel 打分验证**（§11）。
> 详细结论见各节；完整原始 OPPROF 都存在 Mac `63-workspace/10-scalar-camodel-bench/camodel_full_outputs/`，不入 git。
>
> 数据全部来自 CAModel simulator，不是真卡。表中的 cycle 都是 **指令级 active window（issue → retire）**，
> 不是整个 kernel 的时长；kernel 启动/收尾（如 SIMD 的 `LDP` 取参数）都不计入。
>
> **真卡 SYS_CNT 对照**：见 [`README-tidy.md`](README-tidy.md) §1.2。§1.2 统一换算到 ns：
> CAModel 用 §1.1 的 cycle / 1.8；真卡用 `get_sys_cnt()` 写回 `out` 的 marginal 值。
> 结论：SIMD MainScalar load o1/same-line o4、SIMT uniform load o1/diff-line o4 与实测在 ≤~20% 内；
> SIMT same-line o4 和所有 store 是已定位的窗口问题（CAModel 不 merge / SYS_CNT 只看到 write-buffer），
> 不能把这两类的差值当成模型整体不准。单发 `*_syscnt.cce` 是 latency 口径，归档在 `syscnt/` 作边界证据。

---

## 0. 结论速览

| case | total cycles | prep-ish* | line/BIU | return + retire | 关键微指令差异 |
|---|---:|---:|---:|---:|---|
| SIMD MainScalar `LD_XD_XN_IMM` | **447** | 4 | 440 | 3 | 64B line、MSHR merge 入口、write-back 不在 load 路径 |
| SIMT uniform 32T `SIMT_LDG` | **530** | 21 | 477 | 32 | `thread_count=32`，`GSU_I2_OUT target_size=128` |
| SIMT uniform 1T `SIMT_LDG` | **483** | 21 | 430 | 32 | `thread_count=1`，`GSU_I2_OUT target_size=4` |


\* prep-ish = issue → tag/MSHR/BIU dispatch；SIMD 为 4，SIMT 为 21。

相同点：三者都是 cold-line、一次 128B/64B line fill；SIMT 两条都是**一条** `SIMT_LDG`、DC 一条 `size:4`
请求、BIU 一次 `size:128` line read。

差异点（微指令结构层面）：32T → 1T 的稳定差异只有两处硬件可见字段：
`thread_count 32 → 1`、`GSU_I2_OUT target_size 128 → 4`；
issue/tag/BIU send/MROB/GSU 回传/retire 阶段完全一致（6 / 15 / 18 / 11 / 3）。

注意：本版 SIMT 32T 用 round4 canonical dump，1T 用 later combined run（早期另一次孤立 run 曾得到 fill=384，
为避免把 run 波动误读成“1T 更快”，统一采用后面的 430）。两组 BIU `send_rd_cmd → recv_biu_data` 分别为 477 / 430 cycles，
但两条指令对 BIU 的请求本身相同（同地址、都是 1 条 128B line read）；
`GSU_I2_OUT` 发生在 BIU 数据回来之后，属于 UB 回传路径，不能用来解释 BIU 返回时间的差。
477 vs 430 目前只作为观察值记录，可能来自双 subcore 共享 BIU 的仲裁/调度差异，
不能归因于线程数；要做结论需要重复/控制实验。

> 4-op load 见 §4：SIMD same-line 复用 MSHR/DCache；SIMT same-line 在当前 CAModel 不 merge/hit，4 次 line read 串行；diff-line 可并发。

**store 速览**（详见 §5/§6）：

| case | window | 关键结构 |
|---|---:|---|
| SIMD MainScalar store o1 | **478** | write-allocate 64B line fill 440 + dirty refill 后 retire；writeback 在 retire 之后异步 |
| SIMT uniform 32T store o1 | **491** | 无 write-allocate read；STG → UB staging → BIU 128B line write，retire 等 write ack/data rsp |
| SIMD same-line o4 | **531** | 4 条 store 合并到同 1 个 STB entry（sub_entry_size 1→4），1 次 write-allocate fill，4 条一起 retire |
| SIMD diff-line o4 | **557** | 4 条 64B line 各自 write-allocate；read/refill 有重叠，retire 1697 / 1716 / 1763 / 1801 |
| SIMT 32T same-line o4 | **1914** | 当前 CAModel 同 128B line STG 不 merge，4 次 BIU write round-trip 串行 |
| SIMT 32T diff-line o4 | **577** | 4 条 128B line 的 write 可并发，window 只比单条略大 |

> **Triton SIMD scalar store** 不走 CCE MainScalar store：Triton 编译器经 UB staging + MTE3 写 OUT。
> CAModel 单 program 短链路 1079 cyc = 599.4 ns@1.8G；真卡 marginal 81 ns/store（grid=4096, N=32）。
> costmodel 现在使用 MTE3 白盒窗口 `T = 20 + 450 + (K-1)*480`（单条约 470 cyc），不再使用 board marginal throughput；详见 §11.4 / §11.5。

**scalar add 速览**（详见 §7）：

| case | 指令 | 关键能力 |
|---|---|---|
| SIMD MainScalar | `ADD dtype:F32` | burst 2 ADD/cycle；单条 push→retire 6 cycles；ILP16 含 loop 1.606 ADD/cycle |
| SIMT 32T | `SIMT_FADD` | burst 1 warp-FADD/cycle = 32 lane-add/cycle；ILP16 含 loop 0.425 warp-FADD/cycle（13.6 lane-add/cycle）；单条 `exec_time=8`、`stallCyc` 1–2；loop gap ~23 cycles |
| SIMT uniform add | `SIMT_FADD`（full mask） | 32 lane 各自重复执行同一条 add；写各自 slot 时结果一致（7+1=8）；写同一地址时只保留一个 lane 的 add（o1=1、o4=4），不是 32 lane 累加 |

---

## 1. SIMD MainScalar：单条 scalar load

**来源**：`simd_main_ld_o1`，`OPPROF_20260914164450_XVDORZJCYOPYOLET/simd_main_ld_o1/0/dump`
（round 4 canonical，`scalar_bench4.cce` 第 10 行；本目录 `camodel_results/simd/` 只保留了相关 dump）。

> 命名说明：round 6 的 same-line 单 op 名为 `simd_main_ld_same_o8`（同一核里放 8 条）；单条 op 的 same-line 就是 `simd_main_ld_o1`，本报告用它。

指令是 `LD_XD_XN_IMM dtype:B32, PC=0x10d0e050`，`execTime=0x1bf=447`，`dcacheHit=0`。

### 1.1 事件时间线

| time | 事件 | 说明 |
|---:|---|---|
| 1371 | `Push Instr LD_XD_XN_IMM id=20` | 进入 scalar issue queue |
| 1372 | `calc_req_addr` + `lookup_tag is MISS` | aligned 64B line，DCache miss |
| 1374 | `push req to mshr entry 0` + `mshr_req_create_rd_req` | 分配 MSHR，生成读请求 |
| 1375 | `send_rd_biu_req req_id=1418` | 发给 BIU |
| 1383 | BIU `send_rd_cmd size:64 gid=1418` | 真正向内存发 64B line read |
| 1804 | BIU `recv_biu_data gid=1418` | line 数据回到 BIU |
| 1815 | `recv_rd_biu_rsp` + `refill data to cache_ram` + `retire_mshr_instr` | 回填 DCache、wakeup |
| 1818 | `RETIRE INSTR LD_XD_XN_IMM id=20` | scalar load retire |

### 1.2 阶段 cycle

| 阶段 | cycles | 事件差 |
|---|---:|---|
| issue → tag lookup | 1 | 1371 → 1372 |
| tag → MSHR push | 2 | 1372 → 1374 |
| MSHR push → BIU req | 1 | 1374 → 1375 |
| BIU req → DC response/refill | **440** | 1375 → 1815 |
| refill → scalar retire | 3 | 1815 → 1818 |
| **total** | **447** | 1371 → 1818 |

440 的 BIU 内部再拆（用 BIU log，gid=1418）：

| BIU 子阶段 | cycles |
|---|---:|
| DC `send_rd_biu_req` → BIU `send_rd_cmd` | 8 |
| BIU `send_rd_cmd` → `recv_biu_data` | 421 |
| `recv_biu_data` → DC `recv_rd_biu_rsp` | 11 |
| 小计 | **440** |

### 1.3 结论

- MainScalar load 的成本几乎全是一次 **64B line fill**（440/447）；issue+tag+MSHR+BIU dispatch 只有 4 cycles，
  refill→retire 只有 3 cycles。
- 同一条 64B line 的第 2 条及以后访问会 hit 或用 MSHR merge，不应按 447 cycles/op 线性计。
- 本 case 只测 load；store 的 write-allocate + dirty writeback 不在本轮范围。
---

## 2. SIMT warp-uniform：单条 scalar load

**32T 来源**：`simt_ld_uniform_o1`（`dim3{32,1,1}`），
`OPPROF_20260914164619_LASNLOTCZBUQPVUQ/simt_ld_uniform_o1/0/dump`（round 4 canonical）。
**1T 来源**：`simt_ld_uniform_o1_t1`（`dim3{1,1,1}`），
`OPPROF_20260917161008_EBOYHJAJTBDKZHHN/simt_ld_uniform_o1_t1/0/dump`（2026-09-17 later combined run，
同进程先跑 simd + simt32；源文件见 `load_scalar_o1.cce`）。

两条用的是同一个 `__simt_vf__` core body，只改 `dim3`；地址都是 `gm[0] = 0x164b6da00`，32 lane 同地址。
LDG 后的 `SIMT_STS`（把结果写 UB）不计入本节 active window。

### 2.1 32 线程

`SIMT_LDG isaId=121`，total = 2472 - 1942 = **530 cycles**。

| time | 事件 |
|---:|---|
| 1942 | `ISSUE_INSTR SIMT_LDG id=121` |
| 1948 | DC `TagRam C6, size:4, tagRst:INVALID` |
| 1963 | BIU `send_rd_cmd size:128 gid=2052` |
| 2440 | BIU `recv_biu_data size:128 gid=2052` |
| 2458 | `UBITF SIMT_LDG thread_count=32` / `MROB_BUF_ALLOC` |
| 2459 | `OP_SPLIT dst_req_cnt=1` / `GSU_ARB_GRANT split_size=4 thread_count=32` |
| 2461 | `GSU_CLC` / `GSU_I2_OUT size=8 target_size=128` |
| 2469 | `UBITF_SEND_RSP DC_MROB_RD` |
| 2472 | `RETIRE_INSTR SIMT_LDG id=121` |

阶段 cycle：

| 阶段 | cycles |
|---|---:|
| issue → DC tag | 6 |
| DC tag → BIU `send_rd_cmd` | 15 |
| BIU `send_rd_cmd` → `recv_biu_data` | **477** |
| BIU data → UBITF MROB | 18 |
| MROB → `UBITF_SEND_RSP` | 11 |
| `UBITF_SEND_RSP` → LSU retire | 3 |
| **total** | **530** |

### 2.2 1 线程

`SIMT_LDG isaId=390`，total = 6366 - 5883 = **483 cycles**。

| time | 事件 |
|---:|---|
| 5883 | `ISSUE_INSTR SIMT_LDG id=390` |
| 5889 | DC `TagRam C6, size:4, tagRst:INVALID` |
| 5904 | BIU `send_rd_cmd size:128 gid=2703` |
| 6334 | BIU `recv_biu_data size:128 gid=2703` |
| 6352 | `UBITF SIMT_LDG thread_count=1` / `MROB_BUF_ALLOC` |
| 6353 | `OP_SPLIT dst_req_cnt=1` / `GSU_ARB_GRANT split_size=4 thread_count=1` |
| 6355 | `GSU_CLC` / `GSU_I2_OUT size=8 target_size=4` |
| 6363 | `UBITF_SEND_RSP DC_MROB_RD` |
| 6366 | `RETIRE_INSTR SIMT_LDG id=390` |

阶段 cycle：

| 阶段 | cycles |
|---|---:|
| issue → DC tag | 6 |
| DC tag → BIU `send_rd_cmd` | 15 |
| BIU `send_rd_cmd` → `recv_biu_data` | **430** |
| BIU data → UBITF MROB | 18 |
| MROB → `UBITF_SEND_RSP` | 11 |
| `UBITF_SEND_RSP` → LSU retire | 3 |
| **total** | **483** |

### 2.3 32T vs 1T：微指令层面的区别

相同的部分：

- 一条 `SIMT_LDG`；
- DC `size:4`、`tagRst:INVALID`，两条都只产生 **1 个** DC request；
- BIU 都是一次 `send_rd_cmd size:128`（同地址、同一条 128B line）；
- UBITF 都是 `DC_MROB_RD` → `OP_SPLIT dst_req_cnt=1` → `GSU_ARB_GRANT` → `GSU_CLC` → `UBITF_SEND_RSP`；
- 后段 18 + 11 + 3 cycles 完全一致。

稳定的差异只有两处字段：

| 字段 | 32T | 1T | 含义 |
|---|---:|---:|---|
| UBITF `thread_count` | 32 | 1 | 一个 warp 内参与该 LDG 的线程数 |
| `GSU_I2_OUT target_size` | 128 | 4 | GSU→UB 访问要服务的数据量（32 lane × 4B vs 1 lane × 4B） |

BIU `send_rd_cmd → recv_biu_data` 本次为 477 / 430 cycles；但这不是上面两个字段直接决定的，
原因见下面的数据通路说明。

#### 数据通路：BIU 和 GSU_I2_OUT 分别在哪一段

- **BIU 段**：`core0.biu.brif.log.dump` 的 `send_rd_cmd` → `recv_biu_data`。这里发生的是
  SIMT DC 向 GM/L2 发 128B line read，以及 line 数据返回到 BIU/DC 侧；32T 和 1T 这段请求完全相同。
- **UB 回传段**：BIU 数据回来后，UB 日志（`core0.veccore0.ub.dump`，以 32T 为例）显示：
  - `DC_BHU_WR` 把 128B line 写到 UB staging（本次 `addr=0x32000`）；
  - `DC_MROB_RD` / `GSU_I2_OUT` 产生 UB 读请求，`UB_RD_REQ` / `UB_RD_DONE` 把数据经 RWDB
    送回 SIMT 寄存器回写路径；
  - 最后 `UBITF_SEND_RSP` → LSU retire。

  也就是说：**`GSU_I2_OUT` 是 GSU 输出到 UB 接口（UBITF/UB）的访问**，不是写 DCache，
  也不是直接的寄存器写端口。对 `SIMT_LDG` 它是“从 UB staging 取数据供寄存器写回”；
  对显式 `SIMT_LDS` / `SIMT_STS`，client 分别是 `LSU_BPSQ_RD` / `LSU_BPSQ_WR`，
  同样走 GSU→UBITF→UB，不走 DCache。
- **结论**：`GSU_I2_OUT` 发生在 BIU 数据回来之后，不能反过来决定 BIU 返回周期。
  BIU 477 vs 430 更可能是两个 subcore 共享 BIU 的仲裁/调度差异；同一个 32T kernel 在不同 run
  的 `veccore0` fill 已观察到 432 / 477 / 505 三个值，1T 也观察到 384 / 430，随 sibling core 服务顺序变化。
  微指令结构差异（`thread_count` + `target_size`）稳定，但 fill 绝对值不能当成线程数的函数。

> 注意：SIMT mix kernel 在 CAModel 中会同时在两个 AIV subcore（`core0.veccore0/1`）上执行，两者共享 BIU。
> 本报告 32T 数据来自 round4 canonical run，1T 数据来自 later combined run，两者不在同一个 run 内对齐；
> 固定看 `core0.veccore0` 即可，绝对 fill 周期必须重复测量或固定仲裁顺序后再比较。

---

## 3. 横向对比

| 项目 | SIMD MainScalar | SIMT 32T | SIMT 1T |
|---|---:|---:|---:|
| 执行域 | MainScalar / PIPE_S | SIMT LSU | SIMT LSU |
| 指令 | `LD_XD_XN_IMM` | `SIMT_LDG` | `SIMT_LDG` |
| line 粒度 | 64B | 128B | 128B |
| DC request | 1 × size 4 | 1 × size 4 | 1 × size 4 |
| 总 active cycles | 447 | 530 | 483 |
| issue+tag+dispatch | 4 | 21 | 21 |
| line/fill | 440 | 477 | 430 |
| 回传+retire | 3 | 32 | 32 |

- SIMD 是标量通路，`issue+tag+MSHR` 只有 4 cycles，fill 占绝对主体。
- SIMT 多出 DC/UBITF/GSU 的固定回传链（约 32 cycles）与更长的 issue→BIU 路径（21 cycles）。
- 32T 与 1T 的稳定差异只有 `thread_count` 和 `GSU_I2_OUT target_size`；二者都发生在 GM line
  回来之后，**不改变 BIU 请求**。表中 `line/fill` 的 477/430 是两组 dump 的观察值，
  不能解释成 `target_size` 拉低了 BIU 时间。

---

## 4. 多 op load：4-op（same-line / diff-line）

> 4 条 scalar load op，cold line；分 same-line / diff-line，各跑 SIMD MainScalar 与 SIMT warp-uniform 32T。
> 每个 case 单独一个 `msopprof --launch-count=1` run；完整 OPPROF 保留在
> `63-workspace/10-scalar-camodel-bench/camodel_full_outputs/load_scalar_o4/`（不入 git）。
> 分析固定 `core0.veccore0`；`core0.veccore1` 跑同一 kernel，作为 sibling control。

### 4.1 数据来源与窗口

| case | kernel | OPPROF | 地址（float 偏移） | window |
|---|---|---|---:|---:|
| SIMD same | `simd_main_ld_same_o4` | `OPPROF_20260917162015_NGAPCYZPGKUMLOEM` | p[0..3] | **493** |
| SIMD diff | `simd_main_ld_diff_o4` | `OPPROF_20260917162032_ILLLBOBHPWDFHTRL` | p[0],p[16],p[32],p[48] | **956** |
| SIMT 32T same | `simt_ld_uniform_same_o4` | `OPPROF_20260917162049_ABEWLUBKOSHHJMPZ` | p[0..3] | **1923** |
| SIMT 32T diff | `simt_ld_uniform_diff_o4` | `OPPROF_20260917162109_NSCGDLULOFUCSTOT` | p[0],p[32],p[64],p[96] | **526** |

> window = `core0.veccore0` 第一条 load issue → 最后一条 load retire。

### 4.2 SIMD MainScalar 4-op

#### 4.2.1 same-line：p[0..3]（一条 64B line）

| time | 事件 |
|---:|---|
| 1702 | LD#1 p[0] push |
| 1703 | LD#1 calc_req_addr + tag MISS；LD#2 p[1] push |
| 1704 | LD#2 calc_req_addr + tag MISS（同一 64B line） |
| 1705 | LD#1 `push req to mshr` entry0 |
| 1706 | LD#2 `push req to mshr` entry0（merge）；`send_rd_biu_req gid=2464` |
| 1714 | BIU `send_rd_cmd size:64 gid=2464` |
| 2175 | BIU `recv_biu_data gid=2464` |
| 2186 | DC `recv_rd_biu_rsp` + refill + `retire_mshr_instr`（LD#1/#2 同时唤醒） |
| 2189 / 2190 | LD#1 / LD#2 retire（execTime **487 / 487**） |
| 2190 / 2191 | LD#3 p[2] / LD#4 p[3] push |
| 2191 / 2192 | LD#3 / LD#4 `lookup_tag HIT` |
| 2194 / 2195 | LD#3 / LD#4 retire（execTime **4 / 4**） |

阶段（LD#1/#2 共用一次 line fill）：

| 段 | cycles |
|---|---:|
| LD#1 issue → tag/MSHR/BIU dispatch | 1702→1706 = 4 |
| DC `send_rd_biu_req` → BIU `send_rd_cmd` | 1706→1714 = 8 |
| BIU line read | 1714→2175 = 461 |
| BIU data → DC refill | 2175→2186 = 11 |
| refill → LD#1 / LD#2 retire | 2186→2189 / 2190 = 3 / 4 |
| LD#3 / LD#4 hit → retire | 4 / 4 |
| **window** | **493** |

微指令层结论：

- 4 条 load 落在同一 64B line：只产生 **1 个 MSHR entry**、**1 次 `send_rd_biu_req`**、**1 次 BIU 64B read**。
- 前两条 miss 合并到同一 MSHR，refill 后一起 retire；后两条 `lookup_tag HIT`，各 4 cycles。
- 因此 same-line 多 op 的成本主要是第一次 line fill，不应按 `N × 单 op` 线性累计。

#### 4.2.2 diff-line：p[0], p[16], p[32], p[48]（四条 64B line）

每条 op 各占一条 64B line；受 MSHR outstanding 限制，自然分成两波。

**Wave 1（p[0], p[16]）**

| time | 事件 |
|---:|---|
| 1367 / 1368 | LD#1 p[0] / LD#2 p[16] push |
| 1368 / 1369 | LD#1 / LD#2 calc_req_addr + tag MISS |
| 1370 / 1371 | LD#1 MSHR entry0 / LD#2 MSHR entry1 |
| 1371 / 1372 | LD#1 `send_rd_biu_req gid=1406` / LD#2 `send_rd_biu_req gid=1407` |
| 1379 / 1381 | BIU read gid1406 (p[0]) / gid1407 (p[16]) |
| 1777 | BIU `recv_biu_data gid1407` |
| 1788 / 1791 | p[16] refill / retire（execTime **423**） |
| 1800 | BIU `recv_biu_data gid1406` |
| 1811 / 1814 | p[0] refill / retire（execTime **447**） |

**Wave 2（p[32], p[48]）**

| time | 事件 |
|---:|---|
| 1814 / 1815 | LD#3 p[32] / LD#4 p[48] push |
| 1815 / 1816 | LD#3 / LD#4 calc_req_addr + tag MISS |
| 1817 / 1818 | LD#3 MSHR entry0 / LD#4 MSHR entry1 |
| 1818 / 1819 | LD#3 `send_rd_biu_req gid=1412` / LD#4 `send_rd_biu_req gid=1413` |
| 1826 / 1827 | BIU read gid1412 (p[32]) / gid1413 (p[48]) |
| 2234 | BIU `recv_biu_data gid1413` |
| 2245 / 2248 | p[48] refill / retire（execTime **433**） |
| 2309 | BIU `recv_biu_data gid1412` |
| 2320 / 2323 | p[32] refill / retire（execTime **509**） |

阶段汇总：

| 段 | cycles |
|---|---:|
| Wave1：first issue → wave1 last retire | 1367→1814 = 447 |
| Wave2：first issue → wave2 last retire | 1814→2323 = 509 |
| **window（first issue → last retire）** | **956** |

微指令层结论：

- 4 条 op 命中 4 条不同 64B line，全部 miss；4 次 `mshr_req_create_rd_req` + 4 次 BIU 64B read。
- MSHR entry0/entry1 两个 outstanding 被两条 op 占满，因此形成两波；同波内两次 line fill 可 overlap，第二波等第一波 retire 后开始。
- 无 DCache hit。

### 4.3 SIMT warp-uniform 4-op（32T）

#### 4.3.1 same-line：p[0..3]（一条 128B line）

| op | issue | DC tag | tagRst | BIU send | BIU recv | fill | MROB | UBITF rsp | retire | latency |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| #1 (id131) | 1930 | 1936 | INVALID | 1951 | 2447 | 496 | 2465 | 2476 | 2479 | 549 |
| #2 (id132) | 1932 | 1938 | INVALID | 2469 | 2871 | 402 | 2889 | 2900 | 2903 | 971 |
| #3 (id133) | 1934 | 1940 | INVALID | 2893 | 3359 | 466 | 3377 | 3388 | 3391 | 1457 |
| #4 (id134) | 1954 | 1960 | INVALID | 3381 | 3821 | 440 | 3839 | 3850 | 3853 | 1899 |

window = 1930 → 3853 = **1923**。

微指令层结论：

- 4 条 `SIMT_LDG` 全部对同一条 128B line 产生 DC `TagRam ... tagRst:INVALID`。
- 每条 LDG 各产生一次 `send_rd_biu_req` 和一次 BIU `send_rd_cmd size:128`；
  当前 CAModel 对同 line 的重复 SIMT uniform LDG **不 merge / 不 hit**。
- BIU 串行返回 4 次 line data；后续 LDG 在前一条数据/UB 回传完成后才继续，
  单条 latency 逐条累加到 549 → 971 → 1457 → 1899。

#### 4.3.2 diff-line：p[0], p[32], p[64], p[96]（四条 128B line）

| op | issue | DC tag | addr | BIU send | BIU recv | fill | MROB | UBITF rsp | retire | latency |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| #1 | 2135 | 2141 | 0x164b6da00 | 2156 | 2558 | 402 | 2576 | 2587 | 2590 | 455 |
| #2 | 2137 | 2143 | 0x164b6da80 | 2158 | 2615 | 457 | 2633 | 2644 | 2647 | 510 |
| #3 | 2139 | 2145 | 0x164b6db00 | 2160 | 2628 | 468 | 2646 | 2657 | 2660 | 521 |
| #4 | 2141 | 2147 | 0x164b6db80 | 2162 | 2629 | 467 | 2647 | 2658 | 2661 | 520 |

window = 2135 → 2661 = **526**。

微指令层结论：

- 4 条 LDG 各命中不同 128B line，DC 4 条 `TagRam INVALID`；BIU 4 条 `send_rd_cmd size:128` 在 6 cycles 内背靠背发出。
- 4 条 line fill 可以并发 outstanding，数据集中返回，4 条 LDG 的 retire 也集中在 2590–2661。
- 与 same-line 相反：这里没有同 line 串行，窗口只比单条 SIMT LDG 略大。

### 4.4 4-op 对比

| case | unique line | BIU line read | DC hit | window | 主要结构 |
|---|---:|---:|---:|---:|---|
| SIMD same | 1×64B | 1 | 2 | 493 | MSHR merge + hit |
| SIMD diff | 4×64B | 4 | 0 | 956 | 2 outstanding，两波 |
| SIMT 32T same | 1×128B | 4 | 0（tag INVALID×4） | 1923 | 当前 CAModel 不 merge/hit，4 次 line read 串行 |
| SIMT 32T diff | 4×128B | 4 | 0 | 526 | 4 条 line fill 并发 |

> 说明：`GSU_I2_OUT` / UB 回传发生在 BIU 数据回来之后，不参与 BIU line read 的发起；
> 本节的 `fill` 只表示 BIU `send_rd_cmd → recv_biu_data`。
>
> ⚠️ `SIMD diff = 956` 是 **round-6 原始 `simd_main_ld_diff_o4` 的 codegen**：4 条 load 被排成 2 波、每波 2 条
> MSHR outstanding。给同一个 body 加 `get_sys_cnt`/barrier 重编译成 `load_scalar_o4_syscnt.cce` 后，
> 4 条 load 背靠背 issue，CAModel dump 中是 4 个 MSHR entry 同时 outstanding，guest syscnt ≈528。
> 两者不是同一个 schedule，不能把 956 和 528 当成矛盾；详见 [`README-tidy.md`](README-tidy.md) §1.2。
> 补做的 pre/post barrier 拆分实验也表明：**post-load `pipe_barrier` 是把两波调度变成 4 outstanding 的原因**；
> 去掉 post 后 CAModel 回到 955≈956，但真卡 t1 不再保证在 load 完成后读（677），所以当前 cce 不删这个 barrier。

---

## 5. 单 scalar store op

> 单 op store 只有一条 line，不存在 same/diff 分类。每个 case 单独一个 `msopprof --launch-count=1` run；
> 完整 OPPROF 在 `63-workspace/10-scalar-camodel-bench/camodel_full_outputs/store_scalar_o1/`（不入 git）。
> 分析固定 `core0.veccore0`。

### 5.1 SIMD MainScalar：单条 store

**来源**：`simd_main_st_o1`，`OPPROF_20260917165308_FXOUAAMJQSCSPCON/dump`（`store_scalar_o1.cce`）。
目标 op 是 `ST_XD_XN_IMM B32, PC=0x10d0d130, id=62`（p[0]）；`out[0] = 0` 的 sink store 不计入。

| time | 事件 |
|---:|---|
| 1225 | target `ST_XD_XN_IMM id=62` push |
| 1226 | `calc_req_addr` + `lookup_tag MISS`（64B line） |
| 1228 | `push req to stb entry0`（write-allocate 入口） |
| 1260 | `stb_create_rd_req`（store miss 先读 line） |
| 1261 | `send_rd_biu_req req_id=1455` |
| 1269 | BIU `send_rd_cmd size:64 gid=1455` |
| 1690 | BIU `recv_biu_data gid=1455` |
| 1701 | DC `recv_rd_biu_rsp(stb)` + `refill data ... is_dirty:1` + `retire_stb_instr` |
| 1703 | `RETIRE INSTR ST_XD_XN_IMM id=62`（`execTime=478`） |

阶段 cycle：

| 段 | cycles |
|---|---:|
| issue → tag MISS | 1 |
| tag → STB push | 2 |
| STB push → `stb_create_rd_req` | 32 |
| create read req → `send_rd_biu_req` | 1 |
| `send_rd_biu_req` → `recv_rd_biu_rsp(stb)`（write-allocate line read） | **440** |
| `recv_rd_biu_rsp` → retire | 2 |
| **window** | **478** |

440 的 BIU 内部：DC `send_rd_biu_req` 1261 → BIU `send_rd_cmd` 1269 = 8；BIU `send_rd_cmd` 1269 → `recv_biu_data` 1690 = 421；BIU recv 1690 → DC `recv_rd_biu_rsp` 1701 = 11。

**dirty writeback 在 retire 之后异步发生**：`create_write_data_to_ddr_req` 1704 → BIU `recv_su_wr_cmd` 1707 / `send_aw_cmd` 1708 / `recv_wr_data` 1712 / `write_store_buf` 1715 → `recv_wack` 2187 → `send_wr_data` 2192 → `send_data_rsp` 2234 → DC `recv_wr_biu_rsp` 2235。它不进入 store op 的 issue→retire window。

### 5.2 SIMT warp-uniform 32T：单条 store

**来源**：`simt_st_uniform_o1`（`dim3{32,1,1}`），`OPPROF_20260917165322_ZCLLVHKVDNSHXBQS/dump`。
目标 `SIMT_STG isaId=121`，total = 2204 - 1713 = **491 cycles**。

| time | 事件 |
|---:|---|
| 1713 | `ISSUE_INSTR SIMT_STG id=121` |
| 1719 | DC `TagRam C6, cmd:WRITE, size:4, tagRst:INVALID, dispType:ST_NCA` |
| 1719 | UBITF `SIMT_STG thread_count=32` / `BPSQ_ALLOC` |
| 1723 | `GSU_I2_OUT size:8 target_size:128` |
| 1727 | `UBITF_SEND_RSP DC_BPSQ_WR` |
| 1736 | BIU `recv_simt_wr_cmd port=vdcache0 addr=0x164b6da00 gid=2491` |
| 1737 | BIU `send_aw_cmd wr_len:128` |
| 1749 | BIU `recv_wr_data size:128` |
| 1750 | BIU `write_store_buf` |
| 2144 | BIU `recv_wack` |
| 2149 | BIU `send_wr_data size:128` |
| 2186 | BIU `recv_brsp` |
| 2201 | BIU `send_data_rsp` |
| 2204 | `RETIRE_INSTR SIMT_STG id=121` |

阶段 cycle：

| 段 | cycles |
|---|---:|
| issue → DC tag | 6 |
| tag → `UBITF_SEND_RSP`（UB staging/BPSQ/GSU 路径） | 8 |
| UBITF rsp → BIU `recv_simt_wr_cmd` | 9 |
| BIU write cmd → `write_store_buf` | 14 |
| `write_store_buf` → `recv_wack` | **394** |
| `recv_wack` → `send_data_rsp` | 57 = 5 + 52 |
| `send_data_rsp` → LSU retire | 3 |
| **window** | **491** |

微指令层结论：

- SIMT STG 对 GM 是**直接 write**：BIU 侧是 `recv_simt_wr_cmd` → `send_aw_cmd` → `recv_wr_data` → `write_store_buf`，没有 MainScalar store 那种 `stb_create_rd_req` write-allocate read。
- retire 等到 BIU `recv_wack` + `send_wr_data` + `recv_brsp` + `send_data_rsp` 完成；主要等待在 `write_store_buf → recv_wack`（394 cycles）。
- `GSU_I2_OUT target_size=128` 是 UB staging 侧数据量，不是 GM write 的请求结构。

### 5.3 单 op store 对比

| 项目 | SIMD MainScalar | SIMT 32T |
|---|---:|---:|
| 指令 | `ST_XD_XN_IMM` | `SIMT_STG` |
| line/写路径 | 64B write-allocate + dirty writeback | 128B 直接 write |
| issue → tag / UBITF rsp | 1 + 2 | 6 + 8 |
| STB / UB staging + BIU 准备 | 32 + 1 | 9 + 14 |
| line fill / write 完成 | **440** | **394 + 57** |
| retire 前 | 2 | 3 |
| window | **478** | **491** |

两种 store 的 window 接近；差别在实现路径：SIMD 先读整条 64B line 再标 dirty；SIMT 直接写 128B line，但 retire 要等 write response。

### 5.4 Triton SIMD scalar store：MTE3 链路（2026-09-18）

> §5.1/§5.2 是 CCE 直接写 GM 的 store。Triton 代码里的 scalar store **不走 MainScalar `ST_XD_XN_IMM` 直写 GM**：
> 标量值先由 scalar 指令写 UB staging，再由 MTE3 `MOV_SRC_TO_DST_ALIGNv2 Src:UB,Dst:OUT` 搬出去，
> 且前面还要等一个 VF/VEC flag（本 demo 里 `+1.0` 被降成 5 条 RV 指令的 VF）。

**Demo**：`store/triton_scalar_store/triton_scalar_store_demo.py`（服务器副本 `~/triton_scalar_store_demo/`）；
编译选项 `compile_mode=simd`、`num_warps=1`、`superblock_factor=1`、grid=4、N_ST=1。
CAModel dump：`OPPROF_20260918123720_IGLWXPHGPCWLMSZN/dump`（core0.veccore0, block0）。

事件时间线（绝对 cycle）：

| time | 事件 |
|---:|---|
| 6519 | block start |
| 7760 | scalar `ST_XD_XN_IMM B32 accessUb:1 XN=0x80000` push（scalar 值写 UB staging） |
| 7772 | 该 store retire（`execTime=12`）；`SET_FLAG PIPE:SCALAR→VEC` |
| 7773 | VEC `WAIT_FLAG` release |
| 7780 | `PUSH_PB` 启动 VF（5 条 RV：`RV_VLDI` / `RV_VADDS` / `RV_VSTI` 等） |
| 7781 / 7785 | MTE3 `WAIT_FLAG` push / `MOV_SRC_TO_DST_ALIGNv2` push |
| 8328 | VF retire（`vf_execute_time=546`）；`SET_FLAG PIPE:VEC→MTE3`；MTE3 WAIT release |
| 8329 | MTE3 `MOV` pop 开始执行 |
| 8335 | BIU `recv_mte_wr_cmd addr=0x164b8d800 size:4 gid=1989` |
| 8336 | BIU `send_aw_cmd wr_len:4` |
| 8337 | BIU `send_wr_cmd_rsp` |
| 8358 | BIU `recv_wr_data size:4` |
| 8359 | BIU `write_store_buf` |
| 8788 | BIU `recv_wack` |
| 8789 / 8793 | `ict_wr_data_ready` / `send_wr_data size:4` |
| 8822 | BIU `recv_brsp` |
| 8837 | BIU `send_data_rsp` |
| 8839 | MTE3 `MOV` retire |
| 8840…8851 | 后续 scalar 指令 / done |

阶段 cycle：

| 段 | cycles | 说明 |
|---|---:|---|
| scalar 值 → UB staging | 12 | `ST_XD_XN_IMM accessUb:1` push→retire |
| 等 VF / VEC flag（MTE3 前同步空窗） | 556 | 7772 → 8328；其中 VF `vf_execute_time=546` |
| MTE3 `MOV` push → retire | **1054** | 7785 → 8839；前 544 在等 VEC flag，后 510 是 UB→BIU/OUT 写 |
| ↳ BIU 建链路 | 24 | 8335→8359 |
| ↳ 等写完成 ack | **429** | 8359 → `recv_wack` 8788 |
| ↳ ack→data rsp | 49 | 8788→8837 |
| ↳ data rsp→MTE3 retire | 2 | 8837→8839 |
| **整条短链路（scalar store push → MTE3 retire）** | **1079** | 7760 → 8839 |

N_ST=4（同 grid=4）：MTE3 `MOV` 之间各有一条 `BAR PIPE:ALL`，因此 retire 串行：
`(7381,7898)=517`、`(7899,8410)=511`、`(8411,8894)=483`、`(8895,9359)=464`；
从第一条 scalar UB staging push（6791）到最后一条 MTE3 retire（9359）总 window **2568**，
后 3 条平均 ~487 cycles/store。BIU 侧 4 条 write 的 aw/wr 可以交错 outstanding，但 `BAR` 把 retire 串起来了。

**和真卡 ns 对比**：

- CAModel 单条短链路：`1079 cycles / 1.8 GHz = 599.4 ns`；MTE3 push→retire `1054/1.8 = 585.6 ns`；
  其中等 `recv_wack` 一段 `429/1.8 = 238.3 ns`。
- 真卡：`scalar_ldst/calibrate_triton_simd_scalar_store.py --grid 4096 --reps 20` 的 marginal
  （1 tick ≈ 1 ns）：N_ST=1 **92.4 ns**、N_ST=8 213.2 ns、N_ST=16 141.0 ns、N_ST=32 **81.0 ns**
  （旧 README 记 79.3，同量级稳定）。
- ratio：`599.4 / 81.0 ≈ 7.4×`；即使对比单发 92.4 ns 也有 6.5×。

**结论**：

- Triton SIMD scalar store 的链路是
  `scalar → UB staging → VF/同步 → MTE3 MOV UB→OUT → BIU write ack`；与 CCE `ST_XD_XN_IMM` 直写 GM 完全不同。
- 本 demo 的标量 `+1.0` 被编译器降成 VF，所以固定段里包含 546-cycle VF；如果 store value 直接来自
  scalar load/无算术，VF/同步段可能消失或变短，但 MTE3 `MOV → BIU write ack` 段仍然存在。
- CAModel 的 1079 cycles 是 **单 program 冷链路 latency**；N>1 时 `BAR PIPE:ALL` 强制串行。
  真卡 marginal 是 grid=4096、56 核并行下的 **吞吐摊销**，多 program 的 MTE3/BIU write 可以交错隐藏。
- 因此 **CAModel MTE3 window 不能直接当 cost model 系数**；cost model 里 SIMD scalar store
  应使用真卡 marginal（或按真卡口径单独建模），CAModel 只用于拆 MTE3 链路的固定阶段。

**无算术补测（2026-09-18，见 `store/triton_scalar_store/README.md`）**：

- `const`：`tl.store(out_ptr + pid, 1.0)`（无 cast、无算术）
- `const0`：`tl.store(out_ptr, 1.0)`（grid=1，连 `addptr` 都没有）

两个版本的 TTIR 都只有标量 `tt.store`，但 TTAdapter 仍降成 `tensor<1xf32>` +
`bufferization.materialize_in_destination` → `memref<1xf32>`；CAModel 里仍然是：

```text
SCALAR ST_XD_XN_IMM accessUb:1, accessDdr:0     # 写 UB
SET_FLAG PIPE:SCALAR -> MTE3
MTE3  MOV_SRC_TO_DST_ALIGNv2 Src:UB, Dst:OUT    # 写 GM
```

两个版本的 `instr_log` 中都没有 `ST_XD_XN_IMM accessDdr:1`。所以 **MTE3 是本后端 scalar
`tt.store` lowering 的固有路径，不是 `+1.0` 的 VF 造成的**；无算术时 VF/同步段消失或变短，
但 `MTE3 MOV → BIU write ack` 仍然存在。

---

> **2026-09-18 代码修订**：§5.4 的 board marginal 只用于说明真卡吞吐与 CAModel 冷链路的区别；
> costmodel 现已改用 MTE3 白盒 `T = 20 + 450 + (K-1)*480`，`main_store_*` 已删除。当前 6-kernel
> store 误差 -18.3%/-19.5%，见 §11.4/§11.5。

## 6. 多 op store：4-op（same-line / diff-line）

> 4 条 scalar store op，cold line；same-line / diff-line 各跑 SIMD MainScalar 与 SIMT warp-uniform 32T。
> 完整 OPPROF 在 `63-workspace/10-scalar-camodel-bench/camodel_full_outputs/store_scalar_o4/`（不入 git）。

### 6.1 SIMD MainScalar 4-op

#### 6.1.1 same-line：p[0..3]（一条 64B line）

`OPPROF_20260917165342_ISRQYJEIKLNEIRYD`；window = **531**（1843 → 2374）。

| op | PC | issue | retire | execTime |
|---|---|---:|---:|---:|
| #1 | 0x10d0d250 | 1843 | 2371 | 528 |
| #2 | 0x10d0d25c | 1845 | 2372 | 527 |
| #3 | 0x10d0d268 | 1847 | 2373 | 526 |
| #4 | 0x10d0d274 | 1849 | 2374 | 525 |

微指令层：

- 4 条 `ST_XD_XN_IMM` 都 hit 同一条 64B line；各 `lookup_tag MISS` 后进入 **同一个 `stb_main_entry_id:0`**，`stb_sub_entry_size` 依次为 1/2/3/4。
- 只产生 **1 次** `stb_create_rd_req`（1878）和 **1 次** `send_rd_biu_req req_id=2500`（1879）；BIU `send_rd_cmd gid=2500` 于 1887 发出，`recv_biu_data` 2359 回来，DC 2369 dirty refill 后 **4 条 store 一起 `retire_stb_instr`**（2369），随后 2371–2374 retire。
- dirty writeback（`create_write_data_to_ddr_req` 2389 起）在 retire 之后异步。

阶段：

| 段 | cycles |
|---|---:|
| first issue → tag MISS | 1 |
| tag → first STB push | 3 |
| STB push → single `stb_create_rd_req` | 32 |
| create read req → `send_rd_biu_req` | 1 |
| `send_rd_biu_req` → `recv_rd_biu_rsp(stb)` | 490 |
| dirty refill → last retire | 5 |
| **window** | **531** |

#### 6.1.2 diff-line：p[0], p[16], p[32], p[48]（四条 64B line）

`OPPROF_20260917165356_ZRHFISCLTBZZDCVJ`；window = **557**（1244 → 1801）。

| op | line addr | PC | issue | retire | execTime | STB entry | read req | write-allocate rsp |
|---|---|---|---:|---:|---:|---:|---:|---:|
| #1 | 0x164b6da00 | 0x10d0d2e4 | 1244 | 1697 | 453 | 0 | 1464 | 1695 |
| #4 | 0x164b6dac0 | 0x10d0d308 | 1250 | 1716 | 466 | 3 | 1471 | 1714 |
| #2 | 0x164b6da40 | 0x10d0d2f0 | 1246 | 1763 | 517 | 1 | 1466 | 1761 |
| #3 | 0x164b6da80 | 0x10d0d2fc | 1248 | 1801 | 553 | 2 | 1468 | 1799 |

微指令层：

- 4 条 store 各落一条不同 64B line，各自 `push req to stb` entry0/1/2/3，各自 `stb_create_rd_req` + `send_rd_biu_req`（write-allocate）。
- 4 次 read 在 1279–1287 内发出；BIU/refill 有重叠，retire 顺序不是程序顺序：1697 / 1716 / 1763 / 1801。
- 没有 DCache hit；window 557，不是单条 478 × 4。
- dirty writeback 在 1802–1804 起继续异步执行。

#### 6.1.3 SIMD 4-op 对比

| case | unique line | STB entries | write-allocate reads | window |
|---|---:|---:|---:|---:|
| same | 1×64B | 1（4 条 merge） | 1 | **531** |
| diff | 4×64B | 4 | 4 | **557** |

### 6.2 SIMT warp-uniform 32T 4-op

#### 6.2.1 same-line：p[0..3]（一条 128B line）

`OPPROF_20260917165412_ENUDUKKDMPEZUQJL`；window = **1914**（1828 → 3742）。

| op | issue | retire | latency |
|---|---:|---:|---:|
| #1 (id121) | 1828 | 2301 | 473 |
| #2 (id123) | 1831 | 2697 | 866 |
| #3 (id124) | 1833 | 3236 | 1403 |
| #4 (id125) | 1835 | 3742 | 1907 |

微指令层：

- 4 条 `SIMT_STG` 都写同一条 128B line；DC TagRam 4 条 `cmd:WRITE, size:4, tagRst:INVALID`。
- UBITF 4 条 STG 都产生 `DC_BPSQ_WR` → `GSU_I2_OUT target_size=128`。
- BIU 侧 `recv_simt_wr_cmd` / `send_aw_cmd` / `recv_wr_data` / `write_store_buf` 在 `vdcache0` 上按 op 串行出现（1851、2307、2703、3242），每条 op 的 LSU retire 等待自己的 write ack/data rsp（2301、2697、3236、3742）。
- 当前 CAModel **不 merge** 同一 128B line 的多个 SIMT STG；4 次 write round-trip 串行，window ≈ 4 × 单条。

#### 6.2.2 diff-line：p[0], p[32], p[64], p[96]（四条 128B line）

`OPPROF_20260917165428_HKXOPGVTZUJXFFKN`；window = **577**（2050 → 2627）。

| op | line addr | issue | retire | latency |
|---|---|---:|---:|---:|
| #1 (id121) | 0x164b6da00 | 2050 | 2558 | 508 |
| #2 (id123) | 0x164b6da80 | 2053 | 2525 | 472 |
| #3 (id124) | 0x164b6db00 | 2055 | 2627 | 572 |
| #4 (id125) | 0x164b6db80 | 2057 | 2560 | 503 |

微指令层：

- 4 条 STG 各写一条不同 128B line；DC TagRam 4 条 WRITE INVALID，地址 0x164b6da00 / a80 / b00 / b80。
- BIU 对 4 条 line 的 `recv_simt_wr_cmd` / `send_aw_cmd` 在 2073–2094 内背靠背发出，write 可以并发 outstanding；后续 ack/data rsp 集中返回，retire 集中在 2525–2627。
- window 577，只比单条 store 略大；没有 same-line 的 4 次串行。

#### 6.2.3 SIMT 4-op 对比

| case | unique line | BIU write | merge | window |
|---|---:|---:|---|---:|
| same | 1×128B | 4 次（每 op 一次） | ❌ 当前 CAModel 不 merge | **1914** |
| diff | 4×128B | 4 次，可并发 | ❌ 但 write 可 overlap | **577** |

### 6.3 4-op store 总对比

| case | unique line | 关键结构 | window |
|---|---|---:|---:|
| SIMD same | 1×64B | STB entry0 merge 4 条，1 次 write-allocate fill | 531 |
| SIMD diff | 4×64B | 4 STB entry，4 次 write-allocate，refill 重叠 | 557 |
| SIMT same | 1×128B | 4 次 BIU 128B write 串行，不 merge | 1914 |
| SIMT diff | 4×128B | 4 次 BIU 128B write 并发 | 577 |

---

## 7. scalar add compute：SIMD MainScalar vs SIMT 32T 能力对比

> 本轮从 scalar load/store 扩到 scalar compute 的 FP32 `add`。
> 方法参考 `tput.cce`：不断增加每轮独立累加链数量（ILP），观察 burst issue 能力和 loop 形态下的持续吞吐。
> SIMD 走 MainScalar 标量 `ADD`，SIMT 走 32-thread warp `SIMT_FADD`；不做 1-thread SIMT。
> cycle 都是 CAModel 指令级 issue 统计，不是真卡。

### 7.1 probe 与窗口

`compute/scalar_add/scalar_add.cce`：

```text
simd_add_ilp1/2/4/8  循环，每轮 1/2/4/8 条独立 ADD（显式累加器）
simd_add_ilp16       循环，每轮 16 条独立 ADD（数组 + #pragma unroll）
simt_add_ilp1/2/4/8  循环，每轮 1/2/4/8 条独立 SIMT_FADD
simt_add_ilp16       循环，每轮 16 条独立 SIMT_FADD

simt_add_uniform_ilp16      循环，每轮 16 条独立 SIMT_FADD；所有 lane 用同一个 uniform 初值/加数
simt_add_uniform_same_addr  32 lane load/add/store 同一个 GM 地址（iters=1，语义探针）
```

非 uniform kernel 用 `iters=64`、`--launch-count=1` 单独跑 CAModel；
uniform 吞吐用 `iters=64`（`simt_add_uniform_ilp16`，与 §7.5 的 ILP16 横向对比）；
`simt_add_uniform_same_addr` 用 `iters=1` 做同地址写语义观察；不引入依赖链场景。
表里只统计**循环体内的 hot PC**：

```text
loop ADD 数 = ILP × iters = ILP × 64
例如 ILP16 = 16 × 64 = 1024 条
```

循环前后的初始化/最终汇总会产生少量非 hot add，表中单独列成“循环外”，不参与吞吐除法。

### 7.2 SIMD MainScalar：`ADD dtype:F32`

`instr_log.dump` 里的 hot 指令：

```text
ADD  dtype:F32, XD:X1=..., XN:X3=..., XM:X1=...
```

`ccu.scalar_issque.dump` 里：

- `Push Instr` / `pop Instr` 同 cycle、`queue.size=0`；
- 单条 ADD 的 push→retire 是 **6 cycles**；
- loop control 是标量指令 `SUB_IMM / ZEROEXT / CMP / JUMPC`，与 ADD 交错执行。

ILP16 一轮 16 条 hot ADD 的 push 时刻：

```text
1620, 1620, 1621, 1621, 1622, 1622, 1623, 1623,
1624, 1624, 1625, 1625, 1626, 1626, 1627, 1627
```

这说明：

- **burst 内 2 条 ADD/cycle**：同 cycle 可以 push 两条不同 PC 的 ADD；
- 一轮 16 条 ADD 占 8 个 busy cycle；
- 最后一条 1627 → 下一轮第一条 1630，中间有 2 个 idle cycle；
- 一轮周期 = 8 busy + 2 idle = 10 cycles，所以含 loop 的整段吞吐是 16 / 10 = 1.6 ADD/cycle。

ILP sweep：

| kernel | loop ADD 数 | 循环外 ADD | loop span | cycles/loop ADD | loop ADD/cycle | issue 间隔 min/avg/max |
|---|---:|---:|---:|---:|---:|---|
| ilp1（单累加器） | 64 | 1 | 252 | 4.000 | 0.250 | 4 / 4 / 4 |
| ilp2 | 128 | 3 | 256 | 2.016 | 0.496 | 0 / 2.016 / 8 |
| ilp4 | 256 | 7 | 307 | 1.204 | 0.831 | 0 / 1.204 / 4 |
| ilp8 | 512 | 15 | 381 | 0.746 | 1.341 | 0 / 0.746 / 3 |
| ilp16 | 1024 | 17 | 637 | 0.623 | **1.606** | 0 / 0.623 / 3 |

表的除法口径：

```text
cycles/loop ADD = loop span / (loop ADD 数 - 1)
loop ADD/cycle = (loop ADD 数 - 1) / loop span
```

`loop span` 是第一条 hot ADD 到最后一条 hot ADD 的 cycle 数，**已经把每轮之间的 idle loop cycle 包含在内**，
所以表里的 `cycles/loop ADD` 不是纯 burst 值。

关键 cycle：

```text
burst 能力：2 ADD/cycle
一轮 ILP16：8 busy + 2 idle = 10 cycles / 16 ADD = 1.606 ADD/cycle
单条 ADD：push→retire 6 cycles
```

### 7.3 SIMT uniform add：32 个 lane 同值

`simt_add_uniform_same_addr`：32 lane 都 load 同一个 GM 地址、各自加 `k`、再 store 回同一地址；
host 用 `out[0]` 检查最终值。这里只看 uniform 语义，不引入依赖链场景。

host 实测值（iters=1）：

| kernel | iters | out0 | 说明 |
|---|---:|---:|---|
| `simt_add_uniform_same_addr` | 1 | **1** | 32 lane 都执行 add；最终只保留一个 lane 的 +1，不是 32 |

CAModel 指令层面：

```text
SIMT_LDG  ... thread_count 32, target_size 128
SIMT_FADD ... [execMask:ffffffff] [stallCyc: 2] [exec_time: 8]
SIMT_STG  ... thread_count 32, BPSQ merge, target_size 128, DC TagRam 同地址 HIT
```

关键时间点：`SIMT_LDG` issue 2055 / retire 2510 → `SIMT_FADD` 2540 → `SIMT_STG` issue 2557 / retire 3141。

字段口径：

- `exec_time=8`：**单条 `SIMT_FADD` 的执行时长/延迟**（占 EXU/写回的时间），不是“每 8 cycle 只能发一条”。
  ILP16 里 16 条独立 FADD 的 push 时间连续为 1625…1640，间隔 1 cycle；流水线把 8-cycle 执行重叠起来，
  所以 burst 吞吐是 **1 warp-FADD/cycle**。`exec_time=8` 和“1 条/cycle”不矛盾：前者是单条延迟，后者是发射吞吐。
- `stallCyc`：发射阶段因操作数/记分牌等待而停的 cycle 数，不是执行时长；这里单条 FADD 是 2，
  §7.4 的 ILP16 独立链中位数为 1。
- `execMask=ffffffff`：32 lane 全部有效；uniform 不会把 32 lane 合并成 1 次。

结论：

1. **会重复做**：32 个 lane 各自在自己的 lane register 上执行同一条 `SIMT_FADD`，`execMask=ffffffff`；
   不是硬件把 32 个 lane 合并成一个 32 输入的 add。
2. **每 lane 结果相同且正确**：32 lane 输入相同、执行同一条 FADD，各自 lane register 得到相同结果；
   写各自的 `out[lane]` 时 32 份一致。
3. **同地址写不会自动累加**：32 lane 都 store 同一个地址时，最终只保留一个 lane 的 +1（`out0=1`，不是 32）；
   所以“32 线程对一个数字做 add”不是 cross-lane reduction，需要显式 reduction 或只让一个 lane 执行。

> uniform 的 ILP16 吞吐对比见 §7.5。

### 7.4 SIMT 32T：`SIMT_FADD`

`instr_log.dump` 里的字段：

```text
SIMT_FADD [PEX:7|P] [Rm:4|S] [Rn:5|R] [Rd:5|R]
          [waitBitMask:0] [stallCyc:1/2] [exec_time: 8]
```

- 一条 `SIMT_FADD` 覆盖 **32 个 lane**；
- `exec_time=8`、`waitBitMask=0`；`stallCyc` 在 ILP1 为 2，ILP2–16 为 1；
- 独立 FADD 在 burst 里的 push 间隔是 **1 cycle/FADD**。

ILP16 一轮 16 条 FADD 的 push 时刻：

```text
1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632,
1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640
```

即 **burst 内 1 条 warp FADD/cycle**。

单 warp 的 loop control 无法隐藏，`simt_add_ilp1` 一个迭代的指令序列：

```text
1858 SIMT_IADD_I  (exec_time 8, stallCyc 1)
1860 SIMT_FADD    (exec_time 8, stallCyc 2)
1861 SIMT_ISETP   (exec_time 7, stallCyc 6)
1865 SIMT_BRANCH  (exec_time 5, stallCyc 5)
1881 SIMT_IADD_I  ...                 # 下一轮
```

每轮结束后有 **约 23 cycle 的 control gap**；ILP 只能摊薄、不能消除：

| kernel | loop FADD 数 | 循环外 FADD | loop span | cycles/warp-FADD | warp-FADD/cycle | issue 间隔 min/avg/max | exec_time |
|---|---:|---:|---:|---:|---:|---|---:|
| ilp1（单累加器） | 64 | 0 | 1449 | 23.000 | 0.043 | 23 / 23 / 23 | 8 |
| ilp2 | 128 | 1 | 1519 | 11.961 | 0.084 | 1 / 11.961 / 29 | 8 |
| ilp4 | 256 | 3 | 1641 | 6.435 | 0.155 | 1 / 6.435 / 23 | 8 |
| ilp8 | 512 | 7 | 1897 | 3.712 | 0.269 | 1 / 3.712 / 23 | 8 |
| ilp16 | 1024 | 16 | 2409 | 2.355 | **0.425** | 1 / 2.355 / 23 | 8 |

表的除法口径与 SIMD 相同，只统计 loop 内 FADD：

```text
cycles/warp-FADD = loop span / (loop FADD 数 - 1)
warp-FADD/cycle = (loop FADD 数 - 1) / loop span
```

ILP16 一轮 16 条 FADD：15 个 burst cycle + 23 个 idle loop cycle ≈ 38 cycles，
整段实测 `2409 / 1023 = 2.355 cycles/FADD`。

关键 cycle：

```text
burst 能力：1 warp FADD/cycle = 32 lane-add/cycle
一轮 ILP16：约 38 cycles / 16 FADD = 2.355 cycles/FADD（含 23 cycle gap）
单条 FADD：exec_time 8 cycles
```

### 7.5 能力对比

| 指标 | SIMD MainScalar | SIMT 32T | SIMT 32T uniform |
|---|---:|---:|---:|
| 指令 | `ADD dtype:F32` | `SIMT_FADD` | `SIMT_FADD`（full `execMask`） |
| 一条指令处理 | 1 个 scalar | 32 个 lane | 32 lane，但 32 lane 同值；硬件不合并，31 lane 冗余 |
| burst issue | **2 条指令/cycle** | **1 条指令/cycle** | **1 条指令/cycle** |
| burst 元素吞吐 | **2 scalar add/cycle** | **32 lane-add/cycle** | 32 lane-add/cycle；**有用 uniform-add 只有 1 个/cycle** |
| loop gap / 轮 | 2 个 idle cycle | 23 个 idle cycle | 23 个 idle cycle（实测 issue interval max = 23） |
| ILP16 含 loop 实测 | 1.606 scalar add/cycle | 0.425 warp-FADD/cycle = **13.6 lane-add/cycle** | **0.425 warp-FADD/cycle = 13.6 lane-add/cycle**；按有用 uniform add 算是 **0.425/cycle** |
| 达到 burst 峰值 | ~80% | ~42% | ~42%（`simt_add_uniform_ilp16` 与非 uniform ILP16 完全相同） |

uniform vs SIMD scalar（按“有用 uniform add”计）：

- ILP16 含 loop：SIMD 1.606 scalar add/cycle vs SIMT uniform 0.425 warp-FADD/cycle
  ⇒ **SIMD 快约 3.78×**（0.425 已经是有用 uniform-add/cycle）。
- 独立 burst：SIMD 2 个 scalar add/cycle vs SIMT uniform 1 个 warp-FADD/cycle
  ⇒ **SIMD 快 2×**。
- SIMT 的 32 lane/指令优势只有在 32 lane 数据不同、且每个 lane 结果都需要时才能兑现；
  uniform 数据下不会合并，31 lane 是冗余计算。

结论：

1. **按指令吞吐**：SIMD MainScalar 更强，同 cycle 可发 2 条标量 ADD；SIMT 只能 1 条 `SIMT_FADD`。
2. **按每个 scalar lane 的吞吐**：SIMT 更强——一条 FADD 同时处理 32 个 lane；
   - burst 理想值 `32 / 2 = 16×` SIMD；
   - 当前 ILP16 + 单 warp loop 形态 `13.6 / 1.606 ≈ 8.5×` SIMD。
3. SIMT 的主要损失是单 warp 藏不住的 **23 cycle loop gap**；SIMD 的固定 gap 只有 2 个 idle cycle，
   所以更接近自己的 burst 上限。
4. 两条路的差距来自“每指令处理多少元素”，不是单纯时钟或 ALU 快慢：
   SIMD 是真正的标量通路，SIMT FADD 是按 warp 执行的 32-lane 指令。
5. **uniform 特殊结论**：uniform 不会让 SIMT 合并 lane，所以按有用结果算，SIMD MainScalar
   反而更快（ILP16 含 loop 约 3.78×、burst 2×）。

---

## 8. load → vector compute：1op / 4op（SIMD vs SIMT，2026-09-17）

> 场景：先做 1 / 4 条 scalar GM load，再做 32-lane vector compute；两边使用各自的计算通路。
> SIMD：MainScalar `LD_XD_XN_IMM` → 编译器把 scalar staging 到 UB（`ST_XD_XN_IMM`）→ `BAR` → `PUSH_PB` / VF（`RV_VLDI` + `RV_VBR` + `RV_VADD` + `RV_VSTI`）。
> SIMT：`SIMT_LDS`（UB 输入）→ `SIMT_LDG`（32 lane uniform scalar load）→ `SIMT_FADD`（寄存器直接算）→ `SIMT_STS`。
> 数据来自 CAModel；r1 / r2 两轮独立 run 的总 window 完全一致。

### 8.1 测量口径

- SIMD：第一条 scalar `LD_XD_XN_IMM` 的 issue（scalar issue queue）→ VF record（vector function 完成）。
- SIMT：第一条 `SIMT_LDG` 的 issue → 最后一条 `SIMT_STS` retire。
- 两者都包含：输入 UB load（SIMT `SIMT_LDS` / SIMD VF 内 `RV_VLDI`）、scalar load、依赖等待、计算、结果写 UB（SIMT `SIMT_STS` / SIMD VF 内 `RV_VSTI`）。
- 不算：kernel 的 `out[0]=...` sink store 和前后 prologue/epilogue。
- vector tile = 32 个 FP32（1 warp；SIMD 用 mask=32 的 vector op）；每个 case 单独 `msopprof --launch-count=1`。

### 8.2 总 window（cycles）

| case | SIMD | SIMT | 谁快 |
|---|---:|---:|---|
| o1（1 scalar + 1 次 32-lane add） | **1261** | **514** | SIMT ≈ 2.45× |
| o4 same-line（p[0..3]，1 条 line） | **1256** | **1973** | SIMD ≈ 1.57× |
| o4 diff-line（4 条 line） | **2412** | **553** | SIMT ≈ 4.36× |

> same-line SIMT 的 1973 主要来自当前 CAModel 对同 128B line uniform LDG 不 merge / hit；真卡需复核。

### 8.3 SIMD 分阶段

| case | scalar load 窗口 | scalar→UB store 窗口 | VF `PUSH_PB` 时刻 | VF record 时刻 | first RV 时刻 | last RV 时刻 | window |
|---|---|---:|---:|---:|---:|---:|---:|
| o1 | 1432→1844（412） | 1843→2122（279） | 2131 | 2693 | 2659 | 2686 | 1261 |
| o4 same | 1430→1930（500；487 + 4/5/4 hit） | 1916→2197（278/275/271/268） | 2208 | 2686 | 2643 | 2679 | 1256 |
| o4 diff | 1375→3227（1852；422/432/464/533 waves） | 1796→3263（280/37/37/37） | 3274 | 3787 | 3744 | 3780 | 2412 |

微指令与要点：

- `LD_XD_XN_IMM B32`：64B line；same-line 有 MSHR merge / hit，diff-line 被 MSHR outstanding 拆成近乎串行的 waves。
- 编译器为了 `vbr` 自动插入 `ST_XD_XN_IMM`，把 scalar 写到 UB staging（地址 ~`0x208e90`）。
- VF 从 `PUSH_PB` 到第一条 RV 指令之间有 **435–528 cycles** 空窗。`rvec.icache1.dump` 显示这是 **vector icache 第一次 line fill**（`send Read biu Req 2138` → `recv Read biu Rsp 2650` → `IFU send instr to IDU 2658`）。
- 真正 RV 指令窗口只有 **27–36 cycles**（`RV_VLDI` / `RV_VBR` / `RV_VADD` / `RV_VSTI` / `RV_SEND`）。
- 结论：单次 VF 的入口/取指固定成本远大于向量运算本身。

### 8.4 SIMT 分阶段

| case | LDS issue→retire | LDG issue→retire | FADD 时刻（exec_time=8） | STS issue→retire（结果写 UB） | window |
|---|---:|---|---|---|---:|
| o1 | 2182→2198（16） | 2183→2666（483） | 2678 | 2688→2697（9） | 514 |
| o4 same | 1834→1850（16） | 1835/1837/1839/1841 → 2296/2775/3247/3777（串行 1942） | 2308/2787/3259/3789 | 3799→3808（9） | 1973 |
| o4 diff | 1838→1854（16） | 1839/1841/1843/1845 → 2300/2351/2352/2353（overlap 514） | 2312/2363/2370/2373 | 2383→2392（9） | 553 |

微指令与要点：

- `SIMT_LDS` 从 UB 读 per-lane 输入，隐藏在 LDG 延迟里；
- `SIMT_LDG`：32 lane 同地址 uniform scalar load（128B line）；
- `SIMT_FADD`：`exec_time=8`，一条指令覆盖 32 lane；
- `SIMT_STS`：结果写回 UB。
- same-line 4 条 LDG 在当前 CAModel 下串行 4 次 line read；diff-line 4 条 line fill 可 overlap，所以 4op diff 只比 1op 大 ~40 cycles。

### 8.5 结论

1. **单条 scalar + 32-lane vector compute：SIMT 明显快**（514 vs 1261）。SIMD 慢在 scalar→UB staging（279）和 VF 冷取指/入口（~530），这两块约 800 cycles，占 window 的 2/3。
2. **4op same-line：SIMD 快**（1256 vs 1973）。MainScalar 对同一条 64B line 有 MSHR merge + hit，4 条 load 只付一次 fill；SIMT uniform same-line 在当前 CAModel 不 merge / hit，这点需要真卡复核。
3. **4op diff-line：SIMT 明显快**（553 vs 2412）。SIMT 的 4 条 128B line 能并发；SIMD 的 4 条 64B line 被 MSHR / outstanding 拆成近乎串行的 waves，而且还要付 VF 冷取指。
4. **SIMD VF 冷取指可以摊销**：如果 VF 代码已在 vicache、或 VF 内有足够多向量指令（loop / 大 tile），435–528 cycles 的入口成本会被摊薄；本次测的是“每个 Stage 一次新 VF”的偏 worst case。
5. 绝对 cycle 仍是 CAModel 口径，不能直接作为 cost model 系数；下一步需要补 warm / loop 对照和真卡 SYS_CNT 复核。

### 8.6 口径补记（VF 术语 / window 展开 / SIMT 向量语义 / DCE）

#### 8.6.1 VF 相关时间点

- `PUSH_PB`：scalar 侧把 VF 描述符推给 vector 队列的指令。
- `VF record`：CAModel 在 VF 结束时打的 `VF addr:... vf_execute_time:...` 日志；`vf_execute_time` 是
  **整个 VF 的时长**（vector icache fetch/dispatch + 所有 RV 指令），不是纯 vector add 的时间。
- `first RV` / `last RV`：VF 内第一条 / 最后一条 RV 指令的执行时间。
- 以 `simd_lc_o1` 为例：`PUSH_PB` 2131 → `first RV` 2659（中间 528 cycles 是 VF 入口 + vicache line fill）→
  RV 操作窗口 2659→2686 = 27 cycles（`RV_VADD` 在 2674）→ VF record 2693（`vf_execute_time=561`）。
- 所以“vector 计算时间”应看 `RV_VADD` 或 `first RV→last RV`；`vf_execute_time` 主要是 VF 入口/取指 + RV 指令总窗口。
- `simd_lc_o4_same` 的 4 条 `RV_VADD` 时刻为 2658 / 2661 / 2664 / 2667，4 条共 9 cycles；
  `first RV→last RV` 36 cycles；VF record 2686、`vf_execute_time=477`，其中 435 cycles 是 `PUSH_PB→first RV` 的入口/取指。

#### 8.6.2 window 展开

- SIMD：`window = VF record time − first LD_XD_XN_IMM push`
  - o1 = 2693 − 1432 = **1261** = load 412 + scalar→UB store 279 + (store retire→PB 9) + VF 562 − 1 cycle overlap。
  - o4 same = 2686 − 1430 = **1256**；o4 diff = 3787 − 1375 = **2412**。
- SIMT：`window = last SIMT_STS retire − first SIMT_LDG issue`
  - o1 = 2697 − 2183 = **514** = LDG 483（issue→retire）+ 12（LDG retire→FADD）+ FADD/依赖/STS 19；
    `SIMT_LDS` 的 16 cycles 藏在 LDG 延迟里。
  - o4 same = 3808 − 1835 = **1973**；o4 diff = 2392 − 1839 = **553**。
- `SIMT_STS` 是 32 lane 的结果写 UB（结果 store）；SIMD 对应的 store 是 VF 内的 `RV_VSTI`。
  两边都算在 window 里，只是 SIMD 的 store 混在 VF 里，没有单独一行。
- o4 的分段事件见 8.3 / 8.4 表；这些段之间会 overlap（例如 scalar load 和 scalar→UB store、
  多条 LDG 的 line fill），所以 window 只按“首末时间戳之差”算，不能把各行相加。

#### 8.6.3 32-lane 向量语义（公平性）

两边做的逐元素运算相同：

```text
SIMD: for i in 0..31: ub_out[i] = ub_in[i] + s0
SIMT: lane i:        ub_out[i] = ub_in[i] + s0
```

| 环节 | SIMD | SIMT |
|---|---|---|
| 输入向量 | `RV_VLDI`：UB → vector register，mask=32 | `SIMT_LDS`：UB → 32 lane registers，`thread_count=32` |
| scalar 来源 | MainScalar load → UB staging → `RV_VBR` 广播成 vector | MainScalar uniform load → 每条 lane 寄存器同一个值 |
| 计算指令 | `RV_VADD`（vector-vector） | `SIMT_FADD`（`execMask=ffffffff`，一条 warp 指令覆盖 32 lane） |
| 结果写回 | `RV_VSTI`：vector register → UB，32 lane | `SIMT_STS`：32 lane registers → UB，`thread_count=32` |

- SIMD：`vlds` 把 32 个元素装进 vector register，`vbr` 把 scalar broadcast 成 vector，`vadd` 做向量-向量加；
- SIMT：`SIMT_LDS` 一次读 32 lane，`SIMT_FADD` 的 `execMask=ffffffff` 覆盖 32 lane，每条 lane 的 `s0` 相同，
  所以本质就是“32 个元素分别加同一个 scalar”；
- 两者不是“SIMD 加向量、SIMT 加标量”，而是同一条 elementwise 运算在两种执行通路上的实现：
  SIMD 需要显式 `vbr` 把 scalar 变成 vector operand，SIMT FADD 直接吃 lane 寄存器里的 scalar。
- 当前 tile = 32 个元素：SIMD 用 mask=32 的 64-lane vector op，SIMT 用 1 warp。若要看满宽度，
  可以另跑一个 tile=64 的对照（SIMT 侧做 2 次 32-lane 迭代）。

#### 8.6.4 关于“vector 结果没被消费，会不会 DCE”

- SIMD 的 vector 结果通过 `vsts`/`RV_VSTI` 写到 UB `ub_out`；即使后面没有 reader，本版 ccec 仍保留了
  `RV_VADD` 和 `RV_VSTI`（每个 SIMD case 的 `core0.veccore0.instr_log.dump` 都能看到）；
  SIMT 侧的 `SIMT_FADD` / `SIMT_STS` 同样在 dump 里。
- 可用以下命令检查：

```bash
grep -E "RV_VADD|RV_VSTI" camodel_results/simd_lc_o1/core0.veccore0.instr_log.dump
grep -E "SIMT_FADD|SIMT_STS" camodel_results/simt_lc_o1/core0.veccore0.instr_log.dump
```

- 这个担心是成立的：如果后续换编译器 / 优化等级，DCE 风险需要重新检查；必要时加消费点
  （例如在 window 之后把 `ub_out[0]` 读回 `out[1]`，或用 volatile UB 指针）再重跑校验。
  本轮另外跑过一个加 readback 的临时变体（未入库）校验，`RV_VADD`/`RV_VSTI` 和 `SIMT_FADD`/`SIMT_STS` 仍然存在。

### 8.7 待办（下一轮）

- 补 warm VF / VF 内 loop 对照，分离“VF 入口固定成本”和“稳态向量吞吐”。
- 用真卡 SYS_CNT 复核 same-line SIMT uniform LDG 是否 merge / hit。
- 决定哪些 absolute cycle 要换成真卡 SYS_CNT 口径。

---

## 9. 文件与复现

目录：

```text
scalar_ldst_whitebox/
  README.md                          # 本报告
  load/scalar_o1/
    load_scalar_o1.cce               # SIMD + SIMT 32T + SIMT 1T（同一 core body）
    load_scalar_o1_host.cpp          # 新 aclrt API host
    load_scalar_o1_runner.sh         # HAL/runtime 指向 dav_3510 camodel 的 wrapper
    build_scalar_o1.sh               # ccec + g++ 构建
    run_scalar_o1_camodel.sh         # 三个 kernel 各跑一个 msopprof 进程
    parse_load_scalar_o1.py          # 从 dump 提取事件/阶段 cycle
    stages_load_scalar_o1.json       # parser 输出
    camodel_results/
      simd/                          # SIMD canonical round-4 dumps
      simt32/                        # canonical round-4 dumps
      simt1/                         # 2026-09-17 新跑 dumps
  load/scalar_o4/
    load_scalar_o4.cce               # SIMD SIMT 各 same/diff 4-op
    load_scalar_o4_host.cpp
    load_scalar_o4_runner.sh
    build_scalar_o4.sh
    run_scalar_o4_camodel.sh
    parse_load_scalar_o4.py
    stages_load_scalar_o4.json
    camodel_results/
      simd_same/                     # 1x64B line
      simd_diff/                     # 4x64B line
      simt_same/                     # 1x128B line
      simt_diff/                     # 4x128B line
  store/scalar_o1/
    store_scalar_o1.cce              # SIMD MainScalar + SIMT 32T 单 store
    store_scalar_o1_host.cpp
    store_scalar_o1_runner.sh
    build_store_scalar_o1.sh
    run_store_scalar_o1_camodel.sh
    parse_store_scalar_o1.py
    stages_store_scalar_o1.json
    camodel_results/{simd,simt32}/
  store/scalar_o4/
    store_scalar_o4.cce              # SIMD/SIMT same/diff 4-op store
    store_scalar_o4_host.cpp
    store_scalar_o4_runner.sh
    build_store_scalar_o4.sh
    run_store_scalar_o4_camodel.sh
    parse_store_scalar_o4.py
    stages_store_scalar_o4.json
    camodel_results/{simd_same,simd_diff,simt_same,simt_diff}/
  store/triton_scalar_store/
    triton_scalar_store_demo.py       # Triton SIMD scalar store demo (MTE3 path)
    run_triton_scalar_store_camodel.sh # msopprof CAModel runner (N_ST/GRID env)
  compute/scalar_add/
    scalar_add.cce                   # SIMD/SIMT ILP 1/2/4/8/16 + SIMT uniform add
    scalar_add_host.cpp              # tput-style args host
    scalar_add_runner.sh             # HAL/runtime 指向 dav_3510 camodel 的 wrapper
    build_scalar_add.sh
    run_scalar_add_camodel.sh        # ILP case ITERS=64；uniform ILP16=64，same_addr 语义 iters=1
    parse_scalar_add.py              # 提取 ADD/SIMT_FADD issue 统计
    stages_scalar_add.json           # parser 输出
    camodel_results/
      simd_add_ilp1/2/4/8/16/
      simt_add_ilp1/2/4/8/16/
      simt_add_uniform_ilp16/
      simt_add_uniform_same_addr_o1/
  load_vec_compute/scalar_o1_o4/
    load_vec_compute.cce             # SIMD/SIMT × o1/o4_same/o4_diff 六个 kernel
    load_vec_compute_host.cpp
    load_vec_compute_runner.sh
    build_load_vec_compute.sh
    run_load_vec_compute_camodel.sh
    parse_load_vec_compute.py
    stages_load_vec_compute.json     # r1 parser 输出
    stages_load_vec_compute_r2.json  # r2 重复性检查
    camodel_results/{simd_lc_o1,simd_lc_o4_same,simd_lc_o4_diff,simt_lc_o1,simt_lc_o4_same,simt_lc_o4_diff}/
  syscnt/                            # 真卡 get_sys_cnt() 探针的公共 host/runner/结果（见 §10）
    syscnt_host.cpp / syscnt_rep_host.cpp / syscnt_runner.sh
    build_syscnt.sh / run_board_syscnt.sh / run_camodel_syscnt.sh
    parse_syscnt_results.py
    results/{board,camodel_ind_all,camodel_sequence}/
  costmodel_eval/                    # §11 costmodel 公式调整 / 6-kernel 打分验证脚本与结果
    run_scalar_dominate_one.py
    run_costmodel_one*.sh / run_camodel_one.sh / run_all_*.sh
    make_tuned_profile.py / compare_scalar_loads.py / summarize_direct_scalar.py
    results/direct_scalar_eval.csv
```

运行：

```bash
source ~/env_ascend.sh
ulimit -n 1048576

# 单 op
cd load/scalar_o1
bash build_scalar_o1.sh
bash run_scalar_o1_camodel.sh
python3 parse_load_scalar_o1.py camodel_results/simd camodel_results/simt32 camodel_results/simt1

# 4-op load
cd ../scalar_o4
bash build_scalar_o4.sh
bash run_scalar_o4_camodel.sh
python3 parse_load_scalar_o4.py camodel_results/simd_same camodel_results/simd_diff \
  camodel_results/simt_same camodel_results/simt_diff

# 单 op store
cd ../../store/scalar_o1
bash build_store_scalar_o1.sh
bash run_store_scalar_o1_camodel.sh
python3 parse_store_scalar_o1.py camodel_results/simd camodel_results/simt32

# 4-op store
cd ../scalar_o4
bash build_store_scalar_o4.sh
bash run_store_scalar_o4_camodel.sh
python3 parse_store_scalar_o4.py camodel_results/simd_same camodel_results/simd_diff \
  camodel_results/simt_same camodel_results/simt_diff

# Triton SIMD scalar store (MTE3)
cd ../triton_scalar_store
N_ST=1 GRID=4 bash run_triton_scalar_store_camodel.sh

# scalar add compute
cd ../../compute/scalar_add
bash build_scalar_add.sh
bash run_scalar_add_camodel.sh
python3 parse_scalar_add.py camodel_results/* > stages_scalar_add.json

# load -> vector compute
cd ../../load_vec_compute/scalar_o1_o4
bash build_load_vec_compute.sh
# 共享服务器建议用 tmux 挂后台，参考主指南 §10.2
tmux new-session -d -s lvc -c "$PWD" "bash run_load_vec_compute_camodel.sh > run.log 2>&1"
python3 parse_load_vec_compute.py camodel_results/simd_lc_o1 camodel_results/simd_lc_o4_same \
  camodel_results/simd_lc_o4_diff camodel_results/simt_lc_o1 camodel_results/simt_lc_o4_same \
  camodel_results/simt_lc_o4_diff > stages_load_vec_compute.json
```

> 所有 `run_*_camodel.sh` 都是每个 case 一个 `--launch-count=1` run，避免 kernel 间状态互相影响。
> `camodel_results/` 只保留分析用到的 dump；完整 `OPPROF_*` 可由脚本重新生成，load/store/compute/load_vec_compute 的完整
> OPPROF 另存到 `63-workspace/10-scalar-camodel-bench/camodel_full_outputs/`（含 `load_scalar_o4/`、`store_scalar_o1/`、
> `store_scalar_o4/`、`compute_scalar_add/`、`load_vec_compute/`）。

## 10. 真卡 SYS_CNT 对照

`syscnt/` 目录提供与上面 CAModel case 对应的真卡 `get_sys_cnt()` 探针：

```bash
cd syscnt
bash build_syscnt.sh          # 编译 <case>/<case>_syscnt.cce + 公共 host
bash run_board_syscnt.sh      # 真卡单发探针：每个 kernel 重复 10 次、每次先 memset 16 MiB GM
bash run_camodel_syscnt.sh    # CAModel 对照（边界证据，不是主表的方法）
python3 parse_syscnt_results.py > syscnt_compare.txt
```

- **主表口径**（`README-tidy.md` §1.2）：CAModel 用 §1.1 cycle/1.8 换算 ns；真卡用
  `syscnt/board_marginal/board_scalar_marginal.cce` 的 marginal `sys_cnt` 斜率（结果 `board_vs_model_agg.csv`）；
  偏差大的 load diff o4 / store case 还列了去掉 post barrier、加 `dcci` / readback 的调整值；
- **单发探针**：`load/scalar_o1/load_scalar_o1_syscnt.cce`、`load/scalar_o4/load_scalar_o4_syscnt.cce`、
  `store/scalar_o1/store_scalar_o1_syscnt.cce`、`store/scalar_o4/store_scalar_o4_syscnt.cce`、
  `compute/scalar_add/scalar_add_syscnt.cce`、`load_vec_compute/scalar_o1_o4/load_vec_compute_syscnt.cce`；
  原始 log 在 `syscnt/results/{board,camodel_ind_all,camodel_sequence,board_extra,camodel_extra}/`；
- 单发探针测到的是 **单次 latency 窗口**（前后 barrier + launch），和 §1.2 的 marginal 不是同一口径；
  它用于观察 store sync（dcci/readback）和 SIMT AIC/AIV 边界，不直接对 §1.1/1.8 主表；
- VF 场景（SIMT、load+vec）用 AIC 的 sys_cnt 外层包装测不到 AIV 指令窗口（板级值会掉到 nop 量级），
  相关 `<case>_syscnt.cce` 仅作为边界证据。这不影响 `simt_shuffle.cce` / `simt_memory.cce` 的
  **K=100→500 循环吞吐斜率**用法（固定偏差相消，用于相对排序，不是单 VF latency）。


---

## 11. costmodel 白盒公式调整与 6-kernel 验证（2026-09-18）

### 11.1 调整后的公式参数（历史 board-marginal 版本；已被 §11.4/§11.5 白盒版取代）

> 2026-09-18 修订：本节表格记录的是上一版“SIMD store 用真卡 board marginal 145.8 cyc/store”的方案。
> 当前代码已改为 MTE3 白盒 `mte3_store_prep=20 / fill=450 / serial=480`，删除 `main_store_*`，
> SIMT store 也改回白盒窗口 `555/480/450/20`；当前 6-kernel 结果以 §11.4/§11.5 为准。


load 以 §1–§6 的 CAModel 测试结果为准；store 以 §5.4 的 MTE3 实测和 README-tidy §1.2 的真卡 marginal 为准，
`profiles/simd_simt/david_v100_simd_simt_v1.json` 中 `scalar_memory` 参数调整如下：

| 参数 | 旧值 | 新值 | 说明 |
|---|---:|---:|---|
| `main_load_prep_system_cycles` | 30 | **7** | 7 + 440 = 447 = o1 window；含 refill→retire 3 |
| `main_load_fill_system_cycles` | 450 | **440** | 白盒 BIU fill |
| `main_load_hit_system_cycles` | 4 | **37/3 = 12.333** | 与 issue=3 一起拟合 o4 same 493 |
| `main_load_issue_system_cycles` | 1 | **3** | o4 diff 956 的 fit |
| `uniform_load_prep_system_cycles` | 50 | **6** | issue→DC tag；2026-09-19 后与 fill 一起使用 |
| `uniform_load_fill_system_cycles` | 500 | **480（2026-09-19 joint refit）** | probe-only 原为 524（6+524=530 = o1 window）；为覆盖目标 kernel 406–558 fill，联合拟合为 480（base 486），见 [`README-targeted-v1.md`](README-targeted-v1.md) §3.2 B |
| `uniform_load_same_line_serial_system_cycles` | 450 | **(1923-530)/3 = 464.333** | o4 same 1923 |
| `uniform_load_diff_line_issue_system_cycles` | 2 | **0.001** | diff o4 = 486（实测 526，−7.6%）；必须 >0 才走 structured 分支 |
| `uniform_store_same_line_base_system_cycles` | 555 | **297** | §5.4 发现 Triton SIMD store 走 MTE3；SIMT store 改用真卡 marginal ns ×1.8 |
| `uniform_store_same_line_serial_system_cycles` | 480 | **(320-297)/3 = 7.667** | board o4 same 177.8 ns → 320 cyc |
| `uniform_store_diff_line_base_system_cycles` | 450 | **297** | board o1 165.0 ns → 297 cyc |
| `uniform_store_diff_line_issue_system_cycles` | 20 | **(404.5-297)/3 = 35.833** | board o4 diff 224.7 ns → 404.5 cyc |
| `simd.store_instructions_per_system_cycle` | 1.0 | **1/145.8 = 0.006859** | Triton SIMD store 走 MTE3：board marginal 81 ns → 145.8 cyc @1.8G |
| `simd.main_store_fill_system_cycles` | 450 | **0** | 停用 CCE MainScalar store 结构化分支（Triton 不走这条） |

> SIMD store 不再用 §5.1/§6.1 的 MainScalar write-allocate 公式：C++ 里把
> `main_store_fill_system_cycles` 置 0 即走 `scalar_memory` 的 fallback throughput 分支，
> 每 store 145.8 cycles = 81 ns@1.8G，与 §5.4 的真卡 MTE3 marginal 对齐。
> 代码里 SIMD scalar store 的公式就是 `T(K) = K × 145.8 cycles`；§5.4 的 MTE3 阶段拆解
> 只用于白盒解释，不会逐项相加。
> SIMT store 的 ground truth 也改成 README-tidy §1.2 的真卡 marginal：165.0/177.8/224.7 ns。

调整后公式对 README 测试点的预测：

| case | 公式预测 | README 实测 | err |
|---|---:|---:|---:|
| SIMD MainScalar load o1 | 447 | 447 | 0% |
| SIMD MainScalar o4 same | 493 | 493 | 0% |
| SIMD MainScalar o4 diff | 956 | 956 | 0% |
| SIMT warp-uniform load o1 | 486 | 530 | −8.3% |
| SIMT warp-uniform o4 same | 1879 | 1923 | −2.3% |
| SIMT warp-uniform o4 diff | 486 | 526 | −7.6% |
| SIMD MTE3 scalar store | 145.8 cyc = 81 ns | board marginal 81.0 ns | 0% |
| SIMT warp-uniform store o1 | 297 cyc = 165.0 ns | board 165.0 ns | 0% |
| SIMT warp-uniform store o4 same | 320 cyc = 177.8 ns | board 177.8 ns | 0% |
| SIMT warp-uniform store o4 diff | 404.5 cyc = 224.7 ns | board 224.7 ns | 0% |

> SIMD MainScalar store o1/o4 的 CAModel cycle（478/531/557）现在只作为
> CCE 直写路径的参考，不再作为 Triton costmodel 的 store 系数。
>
> SIMT load 的 524→480 是 2026-09-19 joint refit：probe 三条从 0% 变成 −8.3%/−2.3%/−7.6%，
> 目标 6 kernel direct/indirect MAPE 从 10.2%/16.2% 降到 8.0%/8.1%，route 仍 all_simt_only。
>
> §11.3 的 6-kernel load 打分只依赖上面的 load 参数；store 参数是在同一轮里按
> §5.4/§1.2 另行改的，不影响 load 的预测值。

### 11.2 6-kernel 验证方法（历史 random-input direct-only 口径）

> 本节方法已被 §11.4/§11.5 的 matched-only + stage-union 口径取代；保留用于对照。


固定 shape / config：

```text
shape = (sl=4, hs=256, ne=4, top_k=2)
BLOCK_X=64, superblock_factor=1, num_warps=1
logical_program_count_hint = grid (6 kernels: 8 programs)
```

6 个 kernel：

```text
padded_copy_gather / padded_copy_scatter / padded_copy_wgrad
binned_copy_gather / binned_copy_scatter / binned_copy_wgrad
```

- costmodel：`compile_mode=simd_simt`、`AUTO_SIMT_SCOPE=report`，分别用 baseline profile 与
  §11.1 tuned profile；
- CAModel：每个 kernel 分别跑 `compile_mode=simd` 和 `simt_only` 的 `msopprof simulator`，
  每个 case 单独 `--launch-count=1`；
- **direct scalar load 对比**：所有 6 个 kernel 的 ScalarLoad stage 序列都是
  `两个 K=1 direct` + `若干 indirect`。CAModel 取一个 program（`core0.veccore0`）按顺序匹配 load 指令 window：
  - SIMD：`LD_*/LDP_*`，要求 `accessDdr=1`、`accessUb=0`、地址 `>=0x1_0000_0000`
    （排除 prologue/UB/compiler scratch）；
  - SIMT：按 issue 顺序的 `SIMT_LDG`；
  - 分支未执行、没配上的 stage 记 unmatched，不再计入 matched 预测/实测，单独列 `unmatched_pred_cycles`；
  - 验证结果同时保留 `direct_pred_static`，表示“预测侧把全部 direct stage 都算上”的静态 worst-case 值，便于区分公式误差与控制流口径。

脚本与结果：`costmodel_eval/`（`run_*`、`compare_scalar_loads.py`、`summarize_direct_scalar.py`），
结果 CSV `costmodel_eval/results/direct_scalar_eval.csv`；原始 report / OPPROF 在服务器
`~/scalar_dominate_eval_20260918/out/`。

### 11.3 结果（历史，含 static 口径）

> 本节是 random-input、direct-only 的旧结果，包含已弃用的 static worst-case 列；当前结论见 §11.4/§11.5。


**tuned profile（§11.1）：direct scalar load matched 误差**

| kernel | mode | 预测(matched) | 静态 worst-case 预测 | 实测(matched) | matched sum err | matched MAPE | coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| padded_copy_gather | SIMD | 894 | 894 | 932 | -4.1% | 4.0% | 2/2 |
| padded_copy_gather | SIMT | 1060 | 1060 | 1113 | -4.8% | 4.8% | 2/2 |
| padded_copy_scatter | SIMD | 894 | 894 | 1112 | -19.6% | 17.4% | 2/2 |
| padded_copy_scatter | SIMT | 1060 | 1060 | 1115 | -4.9% | 4.9% | 2/2 |
| padded_copy_wgrad | SIMD | 894 | 894 | 807 | +10.8% | 10.8% | 2/2 |
| padded_copy_wgrad | SIMT | 1060 | 1060 | 897 | +18.2% | 19.8% | 2/2 |
| binned_copy_gather | SIMD | 894 | 894 | 892 | +0.2% | 19.1% | 2/2 |
| binned_copy_gather | SIMT | 530 | **1060** | 497 | +6.6% | 6.6% | **1/2** |
| binned_copy_scatter | SIMD | 894 | 894 | 985 | -9.2% | 9.1% | 2/2 |
| binned_copy_scatter | SIMT | 1060 | 1060 | 933 | +13.6% | 14.1% | 2/2 |
| binned_copy_wgrad | SIMD | 894 | 894 | 951 | -6.0% | 6.9% | 2/2 |
| binned_copy_wgrad | SIMT | 1060 | 1060 | 879 | +20.6% | 20.6% | 2/2 |

汇总（6 kernels）：

| profile | mode | matched-direct MAPE | sum-direct matched MAPE | sum-direct static worst-case MAPE |
|---|---|---:|---:|---:|
| baseline | SIMD | 11.3% | 7.8% | 7.8% |
| baseline | SIMT | 13.8% | 13.1% | 31.6% |
| **tuned** | **SIMD** | **11.2%** | **8.3%** | **8.3%** |
| **tuned** | **SIMT** | **12.3%** | **11.5%** | **29.2%** |

解读：

1. 调整后的参数与 §1–§6 的确定性 CAModel 测试点完全对齐；在 6 个真实 kernel 上，
   **direct scalar load matched MAPE ≈ 11%–12%**（SIMD 11.2%、SIMT 12.3%，按 12/11 个 matched load 对聚合），
   说明 K=1 白盒公式在真实 Triton codegen 下仍可用。
2. 相比 baseline，tuned 对 SIMT direct load 有改善（13.8% → 12.3%），SIMD 持平（11.3% → 11.2%）；
   真实 kernel 的剩余误差主要来自地址模式、依赖链、编译器把 direct load 与 prologue/其他 op 合并，
   不是单 op 公式本身。
3. `sum-direct` 有两个口径：
   - **matched 口径**：预测/实测都只加配对上的 stage；这是修掉 `binned_copy_gather SIMT +113%` 假象后的口径，tuned SIMT 为 11.5%；
   - **static worst-case 口径**：预测侧仍把两条 direct load 都加进去（包括 `scf.if` 里本次没执行的 stage），tuned SIMT 为 29.2%，`binned_copy_gather SIMT` 的 +113.3% 只属于这个口径。
   `binned_copy_gather` 的 `if expert_idx > 0` 条件在采样 program（`expert_idx==0`）下不成立，只执行了无条件 load；
   当前 `StageWorkloadAnalysis` 静态累计两条 stage，没有建模分支执行概率。验证侧已改为 matched 口径；
   如果要把 route cost 也改成期望值，需要后续把 launch grid shape / 分支概率喂给 costmodel，而不是继续调 `prep/fill/serial`。
4. indirect stage 的预测误差更大（padded scatter SIMD +114%、padded gather SIMT −39% 等）：
   当前模型按 dependency latency × exposure 收费，没有区分实际是否执行、是否同 line hit；
   这些 case 已在 §4/§6 标为需要进一步结构化建模，本轮不动。

### 11.4 当前结论（2026-09-18 代码修订后）

- **store 白盒完成**：
  - Triton SIMD scalar store 改为 MTE3 白盒 `T = prep(20) + fill(450) + (K-1)*serial(480)`；
    代码里删除 `main_store_*` 字段、`mainScalarStoreCycles()` 及相关 schema/UT，不再走 CCE MainScalar write-allocate 公式。
  - SIMT scalar store 保持白盒 `T = 555 + (K-1)*480`（same-line）/ `T = 450 + (K-1)*20`（first-store/diff-line）；
    单条 store 走 first-store。
- **indirect dependency latency 改为只对额外边收费**：
  `resources.load += max(0, exposure - 1) * latency`。第一条 producer→consumer edge 已包含在 consumer
  自己的 line cost 里；这样修掉了 binned SIMT 单条 shallow indirect 被多收 65.4 cycles 的问题。
- 6 kernel 在 fixed shape 下 tuned route 全部变成 **`all_simt_only`**（SIMD MTE3 store 470 vs SIMT 450，
  且 indirect 首边不再收费）。
- 验证口径改为 **matched-only + per-stage union**（§11.5）：direct 最大误差 19.9%、indirect 最大 30.5%、
  store 最大 -19.5%，全部 <50%。
- 残留误差主要来自：单条 SIMT 指令 window 的 run/subcore BIU 仲裁波动（436–558 cycle 量级）、
  `if expert_idx > 0` 未执行 stage 的 control-flow 差异（matched 口径不计）、以及 CAModel 冷链路与公式拟合差。

### 11.5 6-kernel matched-only + stage-union 全量验证（2026-09-18）

代码/profile 修订：

```text
profile (simd scalar_memory)：mte3_store_prep=20, mte3_store_fill=450, mte3_store_serial=480
                             (删除 main_store_* 字段)
profile (simt scalar_memory)：uniform_store_same_line_base=555, same_line_serial=480,
                             diff_line_base=450, diff_line_issue=20
indirect dependency：只对 exposure > 1 的额外边收费
UT：过滤后 scalar 测试 + 全部 41 tests passed
wheel：新代码已 build/install，本文数字用新 wheel 生成的 round4 report
```

Route：6 kernel 全部 `all_simt_only`，所以 CAModel 全部跑 `compile_mode=simt_only`。

固定 shape/config 和 seed：

- `shape=(4,256,4,2)`、`BLOCK_X=64`、`superblock_factor=1`、`num_warps=1`；
- padded kernel `seed=12`（`bin_idx>0`）；binned kernel `seed=0`（`bins[0]>0`）；
- 取 `core0.veccore0` 一个 program；SIMT 按 DC request `size<128` 过滤 scalar，排除 128B vector tile load。

验证口径：

- **costmodel matched**：只累加真正执行的 stage（`if expert_idx>0` 未执行的 `stage_5` 不计）；
- **CAModel stage-union**：每个 matched stage 取其 instructions 的 `max(retire)-min(issue)`，再对 matched stage 求和；
  重叠 instruction 只算一次，不同 stage 也不混成一个 union（避免跨 stage 空档放大）；
- 不再列 static worst-case。

| kernel | mode | 类别 | costmodel matched | CAModel stage-union | 误差 | matched stages | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| padded_copy_gather | SIMT | 直接标量 load | 1060 | 1113 | -4.8% | 2/2 | 两条 direct 各自一个 window，求和 |
| padded_copy_gather | SIMT | 间接标量 load | 530.0 | 509 | +4.1% | 1/1 | stage_7 两条 bins/padded_bins 的 union |
| padded_copy_scatter | SIMT | 直接标量 load | 1060 | 1115 | -4.9% | 2/2 | 同上 |
| padded_copy_scatter | SIMT | 间接标量 load | 1060 | 973 | +8.9% | 2/2 | stage_7 union 458 + stage_11 weights 515 |
| padded_copy_wgrad | SIMT | 直接标量 load | 1060 | 897 | +18.2% | 2/2 | 同上 |
| padded_copy_wgrad | SIMT | 间接标量 load | 530.0 | 481 | +10.2% | 1/1 | stage_7 union |
| padded_copy_wgrad | SIMT | 标量 store | 450 | 559 | -19.5% | 1/1 | CAModel SIMT_STG |
| binned_copy_gather | SIMT | 直接标量 load | 530 | 497 | +6.6% | 1/2 | `stage_5`（if expert_idx>0）未执行，matched 只算 `stage_6` |
| binned_copy_gather | SIMT | 间接标量 load | 530 | 436 | +21.6% | 1/1 | 单条 index_*；首边 latency 已不再收费 |
| binned_copy_scatter | SIMT | 直接标量 load | 530 | 497 | +6.6% | 1/2 | 同上 |
| binned_copy_scatter | SIMT | 间接标量 load | 530 | 436 | +21.6% | 1/1 | 同上 |
| binned_copy_wgrad | SIMT | 直接标量 load | 530 | 442 | +19.9% | 1/2 | 同上 |
| binned_copy_wgrad | SIMT | 间接标量 load | 530 | 406 | +30.5% | 1/1 | 同上 |
| binned_copy_wgrad | SIMT | 标量 store | 450 | 551 | -18.3% | 1/1 | CAModel SIMT_STG |

误差汇总：

| 类别 | MAPE | 最大误差 | 说明 |
|---|---:|---:|---|
| direct scalar load | ~10.2% | +19.9% / -4.9% | binned/padded wgrad 偏大，其余 ≤6.6% |
| indirect scalar load | ~16.2% | +30.5% / +4.1% | binned wgrad 最大；padded 4.1%~10.2% |
| scalar store | ~18.9% | -18.3% / -19.5% | 白盒 SIMT store 450 vs CAModel 551/559 |

**为什么现在没有 >50% 项：**

1. **store**：不再是 MTE3 冷链路 vs 真卡 marginal 的 7.4× 口径差；改成同一条 SIMT_STG 白盒窗口，
   `450` vs `551/559` 只是 -18% 量级的拟合差。
2. **binned SIMT indirect**：首边 65.4 cycles 已不再收费，530 vs 436/406 的剩余差距来自
   CAModel 单条 window 的核间 BIU 仲裁/服务先后（同一 kernel 不同 run 可见 436~558 的波动），
   以及可能的 warm line 复用；公式已去掉系统性 overcharge，剩下的是测量口径波动。
3. **control-flow**：`stage_5` 在采样 program 不执行；static 口径已从表格移除，matched 口径只比较真正执行的
   `stage_6`，所以不会再出现 +113% 假象；但 `StageWorkloadAnalysis` 静态 workload 仍会把 `stage_5` 计入 route cost，
   这是另一层 control-flow 可执行性建模问题。
4. **direct**：2 条 direct load 是独立 stage，按各自 window 求和；不再使用跨 stage union，也不求和重叠 instruction。

脚本与结果：

```text
costmodel_eval/
  summarize_all_scalar.py       # matched-only + per-stage-union; --report-dir 指向 round4 report
  results/all_scalar_eval.csv   # 上表原始数据
  results/all_scalar_eval.md    # 上表
```

服务器：

```text
~/scalar_dominate_eval_round4/out/costmodel_<k>.json   # 新代码/profile report
~/scalar_dominate_eval_seeded/out/<k>_simt_only.run.log # seeded CAModel（padded seed=12 / binned seed=0）
```
