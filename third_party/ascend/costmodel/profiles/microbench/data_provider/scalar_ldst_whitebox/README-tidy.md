# scalar load/store/compute 白盒对比总览

> **目的**：把 SIMD MainScalar 与 SIMT warp 在 scalar **load / store / compute / load+vec compute** 的各种 case 下
> 放到同一口径比较，回答“谁快、快多少”。
>
> **口径**：
> - load / store / load+vec compute：用**指令级 active window（cycles，越小越快）**；具体起止点见每个 case 的说明。
> - scalar compute：用 **issue 吞吐（越大越快）**；SIMD 按 scalar add/cycle，SIMT 按 warp-FADD/cycle 或 lane-add/cycle。
> - 所有数字都是 **CAModel simulator**，不是真卡。绝对 cycle 不能直接当 cost model 系数。
> - **§1.2 单独给出真卡 `get_sys_cnt()` 对照**：SYS_CNT 域与 active-window cycle 域不同，不能把 1.1 的 cycle 直接除以 1.8 GHz 再和 SYS_CNT 比。
>
> **细节**：本文件只放汇总和关键阶段表；完整事件时间线、逐条 dump 说明见同目录 [`README.md`](README.md)。

---

## 1. 总表

### 1.1 总表（load / store / load+vec compute / uniform compute）

| 类别 | case | 指标 | SIMD | SIMT 32T | SIMT 1T | 更快 | 倍数 |
|---|---|---|---:|---:|---:|---|---|
| load | 单 scalar load o1 | cycles | **447** | 530 | 483 | SIMD | vs 32T **1.19×**；vs 1T **1.08×** |
| load | 4-op same-line（p[0..3]） | cycles | **493** | 1923 | — | SIMD | **3.90×** |
| load | 4-op diff-line | cycles | 956 | **526** | — | SIMT 32T | **1.82×** |
| store | 单 scalar store o1 | cycles | **478** | 491 | — | ≈ 打平 | SIMD 1.03× |
| store | 4-op same-line（p[0..3]） | cycles | **531** | 1914 | — | SIMD | **3.60×** |
| store | 4-op diff-line | cycles | **557** | 577 | — | ≈ 打平 | SIMD 1.04× |
| load+vec compute | 1op（1 scalar + 32-lane add） | cycles | 1261 | **514** | — | SIMT 32T | **2.45×** |
| load+vec compute | 4op same-line | cycles | **1256** | 1973 | — | SIMD | **1.57×** |
| load+vec compute | 4op diff-line | cycles | 2412 | **553** | — | SIMT 32T | **4.36×** |
| compute（uniform） | burst | useful add/cycle | **2 useful add/cycle** | 1 useful add/cycle（31 lane 冗余） | — | SIMD | **2×** |
| compute（uniform） | ILP16 + loop | useful add/cycle | **1.606 useful add/cycle** | 0.425 useful add/cycle（31 lane 冗余） | — | SIMD | **3.78×** |

> “倍数” = 慢的一方 / 快的一方。
>
> 重要限定：
> - SIMT 32T same-line 的 1923 / 1973 / 1914 都来自**当前 CAModel 对同 128B line 的 uniform LDG/STG 不 merge / hit**；
>   真卡可能 merge，这一列需要真卡复核。
> - load o1 的 SIMD 447 是 round-4 canonical；SIMT 32T 530、1T 483 是后续 run。1T 的 BIU fill（430）和 32T（477）
>   是两组 dump 的观察值，不能归因于 `thread_count`。
> - `4-op diff-line` 的 956 是 round-6 原始 `simd_main_ld_diff_o4` 的 codegen：2 波 × 2 outstanding；
>   `load_scalar_o4_syscnt.cce` 重新编译后可以 4 outstanding，CAModel guest syscnt 只有 ~528，见 §1.2 的专门说明。
>
> `compute（uniform）` 两行的数字是**有用结果吞吐**：SIMT 的 32 lane 对同一个值重复执行，当前 CAModel 不合并，
> 所以 31 lane 是冗余；非 uniform 的指令 / lane 两种口径见 §4 细节。

### 1.2 CAModel vs 真卡 SYS_CNT（统一 ns）

> **换算口径**：本节所有数字都是 ns。
> - CAModel 侧：直接用 §1.1 的 cycle，`model_ns = cycle / 1.8`（CAModel 1.8 GHz）。
> - 真卡侧：`sys_cnt` tick；板卡校准 997.7 MHz，`ns = tick × 1000 / 997.7`，下文按 **1 tick ≈ 1 ns** 处理。
> - cost model 预测的是 **一段 stage 的总时间**，所以实测取 **marginal / 一段 K 次操作的 `t1-t0`**，不是单次冷启动 latency。
> - `*_syscnt.cce` 单发版（前后 `pipe_barrier`）测到的是 latency；本节主表用 marginal，单发值只放在文末做边界说明。

**实测方法：sys_cnt 写到 out**

```cpp
t0 = get_sys_cnt();
<repeat K ops / reps 次 target sequence>
pipe_barrier(PIPE_ALL);
t1 = get_sys_cnt();
out[0] = t1 - t0;          // host 读回；再用 K/reps 两点斜率扣 nop
```

- host 对每个 kernel 跑两个规模（如 K=1→K=4 或 reps=4→16），用斜率消掉固定启动开销；
- 地址隔离、扣 nop（`m_loop_nop` / `m_loop_simt_nop`）；
- 这条链的源码/结果已归档在 `syscnt/board_marginal/`：
  `board_scalar_marginal.cce` / `board_scalar_marginal_host.cpp`，原始聚合值 `board_vs_model_agg.csv`（`board_ns` 列）。
- 新加的 `syscnt/` 单发探针也把 `t1-t0` 写 `out[0]`，但它是 latency 口径，不能直接替换这里的 marginal。

**主表（model_ns = §1.1 cycle / 1.8；真卡 = marginal ns）**

| 类别 | case | §1.1 CAModel cycle | CAModel ns (/1.8) | 真卡 marginal ns | err |
|---|---|---:|---:|---:|---:|
| load | SIMD MainScalar o1（同 64B line） | 447 | 248.3 | 274.5 | **+10.5%** |
| load | SIMD o4 same-line（p[0..3]） | 493 | 273.9 | 281.9 | **+2.9%** |
| load | SIMD o4 diff-line（4×64B） | 956 | 531.1 | 717.9 | +35.2% |
| load | SIMT 32T uniform o1（同 128B line） | 530 | 294.4 | 291.8 | **−0.9%** |
| load | SIMT 32T uniform o4 diff-line（4×128B） | 526 | 292.2 | 348.8 | **+19.4%** |
| load | SIMT 32T uniform o4 same-line | 1923 | 1068.3 | 292.2 | **−72.6%** |
| store | SIMD MainScalar o1（同 64B line） | 478 | 265.6 | 31.9 | **−88.0%** |
| store | SIMD o4 same-line | 531 | 295.0 | 46.1 | **−84.4%** |
| store | SIMD o4 diff-line | 557 | 309.4 | 177.8 | **−42.5%** |
| store | SIMT 32T uniform o1 | 491 | 272.8 | 165.0 | **−39.5%** |
| store | SIMT 32T uniform o4 same-line | 1914 | 1063.3 | 177.8 | **−83.3%** |
| store | SIMT 32T uniform o4 diff-line | 577 | 320.6 | 224.7 | **−29.9%** |
| compute | SIMD MainScalar ILP16（每 16 ADD） | 10 | 5.56 | 10.9 gross / 8.5 net | +96% / +53% |

> 数据来源：load/store 行 = `syscnt/board_marginal/board_vs_model_agg.csv` 的 `board_ns`；
> compute 行 = 本次 `syscnt/results/board_extra/scalar_add_syscnt__*` 的 gross slope（扣 nop 见正文）。

**结论**

1. **cost model 主要覆盖的 load case 是和实测接近的**：
   SIMD MainScalar o1 +10.5%、same-line o4 +2.9%；SIMT uniform o1 −0.9%、o4 diff-line +19.4%。
   这些点在 1.8 GHz 换算下误差 ≤~20%，说明这部分 CAModel 建模可用于 cost model。
2. **已知模型偏差**：
   - SIMT 32T **same-line uniform o4**：CAModel 当前对同 128B line 不 merge/hit，模型 1068 ns，
     真卡只有 292 ns（−72.6%）。这是 CAModel 本身在这类地址模式下的已知问题，同 line K>1 的 SIMT cost 不能直接用。
   - **所有 store**：CAModel window 是 retire/写分配，真卡 SYS_CNT 窗口只到 write-buffer/DCache 接受，
     所以差 −30%~−88%。要接真实 store 需要单独 board-visible 模型（或明确用 dcci 定义窗口，见 `syscnt/` dcci 变体）。
   - SIMD diff-line o4 load +35.2%：真卡多个 write/read line 可以 overlap，模型偏保守。
     去掉 post barrier（单发 +27.5%）或 asm clobber（+14.4%）能靠近一些，但会破坏 `t1` 完成语义，
     只作为诊断，不替换 marginal 主表。
3. **compute ILP16**：模型 5.56 ns/16 ADD，真卡 gross 10.9 ns（扣 nop 后 8.5 ns）。
   1.8 GHz 换算在这块对不上，需要单独确认 ALU clock / loop 口径，暂时不作为“已对齐”点。
4. **load+vec compute / SIMT VF**：`get_sys_cnt()` 在 AIC 侧，包 `__VEC_SCOPE__` / `async_invoke`
   测不到 AIV 指令窗口（单发探针会出现 nop 量级的假结果），因此本节不把它们列入对照；
   其 VF 内部仍以 CAModel dump 的 active window 为准。

**偏差大的 case：去掉 post barrier / 加 dcci 后的实测对比**

> 下面全是**单发探针的 latency 口径**（单位 ns），只用来判断调整方向；主表仍是 marginal 口径。
> 目标 case：`SIMD load diff-line o4` 和 store 类。

`SIMD load diff-line o4`（model = 956/1.8 = 531.1 ns），改 barrier 组合：

| barrier 变体 | 真卡 median ns | err |
|---|---:|---:|
| pre + post（当前 `*_syscnt.cce`） | 381 | **−28.3%** |
| only post（去掉 pre） | 756 | +42.3% |
| only pre（**去掉 post**） | 677 | **+27.5%** |
| 无 barrier | 1014.5 | +91.0% |
| asm memory clobber + barrier | 607.5 | **+14.4%** |

- 去掉 post barrier 可以把单发值从 381 拉高到 677，更接近 model 531（+27% vs −28%），
  但此时 `t1` 不保证在操作全部完成后读取，主表用它会引入“测早了”的假值。
- asm clobber 变体最接近（+14%），但改变了代码 schedule，也只能作为诊断，不是主表口径。

store 类（model = §1.1/1.8；实测 = 单发 median ns）：

| case | model ns | 不调整 | + dcci | + dcci + readback |
|---|---:|---:|---:|---:|
| SIMD st o1 | 265.6 | 451 | 638 | 1043 |
| SIMD st same-line o4 | 295.0 | 482 | 599 | — |
| SIMD st diff-line o4 | 309.4 | 430 | 1179 | — |
| SIMT 32T st o1 | 272.8 | 286 | 400 | — |
| SIMT 32T st same-line o4 | 1063.3（模型不 merge） | 352.5 | 641.5 | — |
| SIMT 32T st diff-line o4 | 320.6 | 556.5 | 643 | — |

- `dcci` 确实把真卡从“只到 write-buffer”推到了 writeback，但有的 case 直接 overshoot：
  SIMD diff-line o4 从 430 → 1179（model 309），SIMT o1 从接近的 286 → 400。
- 没有一个统一 sync 能让所有 store 都对上 `model/1.8`；**store 的差异是 CAModel retire 口径 vs
  真卡 SYS_CNT 可见窗口的口径差，不是靠单一 barrier/dcci 能修平的**。
- `SIMT 32T same-line o4`（−72.6%）则和实测调整无关：真卡会 merge/hit，CAModel 当前不 merge/hit，
  必须改模型侧；调实测量只会离模型更远。

> **单发 `*_syscnt.cce` 的 latency 值**：例如 SIMD ld o1 真卡 459 ns（`syscnt/results/board/`），
> 比模型 248 ns 大很多，因为它是“单次冷 line + 前后 barrier + launch”的 latency 窗口，
> 不是 cost model 的 stage marginal 口径。单发数据用于观察 barrier/store sync/SIMT AIC-AIV 边界，
> **不要直接和 §1.1/1.8 的主表数字混用**。

### 1.3 一句话结论

- **单条 scalar load**：SIMD 最快（447），但领先 SIMT 只有 8%–19%；SIMT 多出的主要是 DC/UBITF/GSU 固定回传链。
- **多 op same-line**：SIMD 明显快。MainScalar 对同 64B line 有 MSHR merge + hit / STB merge；SIMT 当前 CAModel 对同 128B line 不 merge。
- **多 op diff-line**：SIMT 快（load 1.82×；load+compute 4.36×）。4 条 128B line 可以并发；SIMD 的 4 条 64B line 被 outstanding 拆成 waves。
- **Triton SIMD scalar store（MTE3）**：链路 1079 cyc=599.4 ns（CAModel/1.8G），真卡 marginal N=32 约 81 ns，差 ~7.4×；Triton store 不能用 CAModel window 直接当系数，见 §3.3。
- **单个 store**：两边接近（478 vs 491）。SIMD 是 write-allocate + dirty writeback；SIMT 是直接 128B write，但 retire 等 write response。
- **scalar compute（uniform，总表口径）**：SIMD 更快——burst 2 useful add/cycle vs 1，ILP16 3.78×；非 uniform 的 burst / lane 吞吐能力见 §4 细节。
- **load+vec compute**：SIMD 多出 scalar→UB staging + VF 冷入口/取指；o1 / diff-line 下 SIMT 更快，same-line 下 SIMD 因 load merge 更快。

---

## 2. Load 细节

### 2.1 单 scalar load：阶段 cycle

| 阶段 | SIMD MainScalar | SIMT 32T | SIMT 1T |
|---|---:|---:|---:|
| issue → tag / DC tag | 1 | 6 | 6 |
| tag → MSHR/BIU dispatch | 3（2 + 1） | 15 | 15 |
| line / BIU fill | 440 | 477 | 430 |
| 回传 + retire | 3 | 32（18 + 11 + 3） | 32 |
| **window** | **447** | **530** | **483** |

来源与窗口：

| case | 指令 | OPPROF | window |
|---|---|---|---:|
| SIMD | `LD_XD_XN_IMM` | `OPPROF_20260914164450_XVDORZJCYOPYOLET` | 首个 LD push → retire |
| SIMT 32T | `SIMT_LDG` | `OPPROF_20260914164619_LASNLOTCZBUQPVUQ` | LDG issue → retire |
| SIMT 1T | `SIMT_LDG`（dim3=1） | `OPPROF_20260917161008_EBOYHJAJTBDKZHHN` | LDG issue → retire |

微指令结构：

- SIMD：`LD_XD_XN_IMM` → `lookup_tag MISS` → `push req to mshr` → `send_rd_biu_req` → BIU `send_rd_cmd size:64` → refill → retire。
- SIMT 32T / 1T：`SIMT_LDG` → DC `TagRam size:4 INVALID` → BIU `send_rd_cmd size:128` → `UBITF DC_MROB_RD` / `GSU_I2_OUT` → `UBITF_SEND_RSP` → retire。
- 32T 与 1T 的稳定差异只有 `thread_count=32/1` 和 `GSU_I2_OUT target_size=128/4`，其它阶段 cycle 相同；`GSU_I2_OUT` 发生在 BIU 数据回来之后，不能用它解释 BIU fill 差。

### 2.2 4-op scalar load

| case | unique line | BIU line read | DC hit | window | 主要结构 |
|---|---:|---:|---:|---:|---|
| SIMD same | 1×64B | 1 | 2 | **493** | 第 1/2 条 MSHR merge，一次 fill；后 2 条 hit（各 4 cycles） |
| SIMD diff | 4×64B | 4 | 0 | **956** | MSHR 2 outstanding，分两波；第 2 波等第 1 波 |
| SIMT 32T same | 1×128B | 4 | 0 | **1923** | 当前 CAModel 同 line 不 merge/hit，4 次 line read 串行 |
| SIMT 32T diff | 4×128B | 4 | 0 | **526** | 4 条 line fill 并发，window 只比单条略大 |

> window = `core0.veccore0` 第一条 load issue → 最后一条 load retire。

---

### 2.3 公式调整与 6-kernel 打分（2026-09-18）

以 §2 测试点为准，把 profile 调整为：

```text
SIMD MainScalar load : prep=7, fill=440, hit=37/3, issue=3
                       -> o1 447, o4-same 493, o4-diff 956
SIMT warp-uniform    : prep=6, fill=524, same_serial=464.333, diff_issue=0.001
                       -> o1 530, o4-same 1923, o4-diff 530 (实测 526)
SIMT store           : 改用真卡 marginal（ns×1.8）：base=297, same_serial=7.667,
                       diff_issue=35.833 -> o1 165.0 ns, o4-same 177.8 ns,
                       o4-diff 224.7 ns
SIMD Triton store    : MTE3 white-box T = 20 + 450 + (K-1)*480
                       (单条约 470 cyc；不再用 board marginal throughput)
```

6 个 scalar-dominated kernel 在 `(sl,hs,ne,top_k)=(4,256,4,2)`、
`BLOCK_X=64 / superblock_factor=1 / num_warps=1` 下的历史 direct-only 口径：

> 下表是旧 random-input / static 口径，保留参考；当前 matched-only + stage-union 结果见下一节。

| profile | SIMD matched MAPE | SIMT matched MAPE | 备注 |
|---|---:|---:|---|
| baseline | 11.3% | 13.8% | 旧 profile |
| tuned | **11.2%** | **12.3%** | §11 调整值 |

成本模型代码修订（2026-09-18）后，6 个 kernel 的 route 全部变成 `all_simt_only`；CAModel 用 seeded
`simt_only` 跑 matched-only + per-stage-union 对比，完整表见 `README.md` §11.5。摘要：

| 类别 | 误差范围 | 说明 |
|---|---|---|
| direct scalar load | MAPE ~10.2%，-4.9% ~ +19.9% | padded/binned wgrad 最大；其余 ≤6.6% |
| indirect scalar load | MAPE ~16.2%，+4.1% ~ +30.5% | binned wgrad 最大；首边 dependency latency 已不再收费 |
| scalar store | MAPE ~18.9%，-18.3% ~ -19.5% | SIMT_STG 白盒 450 vs CAModel 551/559 |

- **不再有 >50% 项**：store 改成同口径 SIMT 白盒；binned SIMT indirect 去掉首边 65.4 overcharge；
  static worst-case 已不列入表格。
- `stage_5`（`if expert_idx > 0`）在采样 program 不执行；matched 只算真正执行的 `stage_6`，
  不再出现 static 口径的 +113% 假象。
- 代码/profile 修订：SIMD scalar store 改为 MTE3 白盒 `prep20 + fill450 + (K-1)*serial480`，
  删除 `main_store_*`；indirect dependency latency 只对 `exposure > 1` 的额外边收费；41/41 UT passed。

---

## 3. Store 细节

### 3.1 单 scalar store

| 项目 | SIMD MainScalar | SIMT 32T |
|---|---:|---:|
| 指令 | `ST_XD_XN_IMM` | `SIMT_STG` |
| line / 写路径 | 64B write-allocate + dirty writeback | 128B 直接 write |
| issue → tag / UBITF rsp | 1 + 2 | 6 + 8 |
| STB / UB staging + BIU 准备 | 32 + 1 | 9 + 14 |
| line fill / write 完成 | **440** | **394 + 57** |
| retire 前 | 2 | 3 |
| **window** | **478** | **491** |

结论：window 接近；SIMD 先读整条 64B line 再标 dirty，writeback 在 retire 之后异步；
SIMT 直接写 128B line，但 retire 要等 `recv_wack` + data rsp。

### 3.2 4-op scalar store

| case | unique line | 关键结构 | window |
|---|---|---:|---:|
| SIMD same | 1×64B | 4 条合并到同一 STB entry（`sub_entry_size` 1→4），1 次 write-allocate fill | **531** |
| SIMD diff | 4×64B | 4 个 STB entry，4 次 write-allocate；refill 有重叠 | **557** |
| SIMT 32T same | 1×128B | 当前 CAModel 不 merge，4 次 BIU 128B write 串行 | **1914** |
| SIMT 32T diff | 4×128B | 4 次 BIU 128B write 可并发 | **577** |

### 3.3 Triton SIMD scalar store（MTE3，2026-09-18）

> 前面 §3.1/§3.2 是 CCE 直写 GM。Triton scalar store 走
> `scalar → UB staging → VF/同步 → MTE3 MOV UB→OUT → BIU write ack`。

| 项 | CAModel（block0 latency） | 真卡 marginal（grid=4096） |
|---|---:|---:|
| SIMT UB staging | 12 | — |
| 等 VF/VEC flag | 556（VF 546） | — |
| MTE3 `MOV` push→retire | **1054** | — |
| BIU 建链路→`write_store_buf` | 24 | — |
| 等 `recv_wack` | **429** | — |
| ack→data rsp→retire | 51 | — |
| 整条短链路 | **1079 cyc = 599.4 ns**（/1.8G） | N=1 92.4 ns；N=32 **81.0 ns** |

- N_ST=4 时 MTE3 `MOV` 之间有 `BAR PIPE:ALL`，每条 ~487 cyc，串行；总 window 2568。
- 结论（2026-09-18 修订）：MTE3 链路本身可用 CAModel window 建模，不再用真卡 marginal throughput 混合口径。
  代码采用白盒 `T = 20 + 450 + (K-1) * 480`（MTE3 `MOV` 单条冷链路约 470 cyc），
  `main_store_*` 字段和 CCE MainScalar write-allocate 公式已删除；seeded 6-kernel store 误差 -18.3%/-19.5%。
  早期 board marginal 81 ns 只保留作真卡吞吐参考，不再是 cost model 系数。

---

## 4. scalar compute 细节

`compute/scalar_add/scalar_add.cce`：

- SIMD `ADD dtype:F32`：单条 push→retire **6 cycles**；burst **2 ADD/cycle**；ILP16 一轮 8 busy + 2 idle = 10 cycles / 16 ADD ⇒ **1.606 ADD/cycle**。
- SIMT `SIMT_FADD`：单条 `exec_time=8`、`stallCyc` 1–2；burst **1 warp-FADD/cycle = 32 lane-add/cycle**；
  单 warp loop 有 **~23 cycle gap**，ILP16 一轮 ≈ 38 cycles / 16 FADD ⇒ **0.425 warp-FADD/cycle = 13.6 lane-add/cycle**。
- SIMT uniform：32 lane 各自重复同一条 add，`execMask=ffffffff`；写各自 slot 结果一致（7+1=8），
  写同一地址只保留一个 lane 的 add（o1=1、o4=4），不是 32 lane 累加。

| 指标 | SIMD MainScalar | SIMT 32T | SIMT 32T uniform |
|---|---:|---:|---:|
| 一条指令处理 | 1 scalar | 32 lane | 32 lane（同值） |
| burst issue | 2 条/cycle | 1 条/cycle | 1 条/cycle |
| burst 元素吞吐 | 2 scalar add/cycle | 32 lane-add/cycle | 32 lane-add/cycle（有用 1/cycle） |
| loop gap | 2 cycles | 23 cycles | 23 cycles |
| ILP16 含 loop | 1.606 scalar add/cycle | 0.425 warp-FADD/cycle（13.6 lane-add/cycle） | 0.425 warp-FADD/cycle（有用 0.425/cycle） |

---

## 5. load → vector compute 细节

### 5.1 总 window / 谁快

| case | SIMD | SIMT 32T | 更快 | 倍数 |
|---|---:|---:|---|---:|
| 1op（1 scalar + 32-lane add） | 1261 | **514** | SIMT | **2.45×** |
| 4op same-line | **1256** | 1973 | SIMD | **1.57×** |
| 4op diff-line | 2412 | **553** | SIMT | **4.36×** |

窗口定义：

- SIMD：`first LD_XD_XN_IMM push → VF record`。
- SIMT：`first SIMT_LDG issue → last SIMT_STS retire`。
- tile = 32 FP32；SIMD 用 mask=32 的 vector op，SIMT 用 1 warp；`out[0]` sink store 不计入。

### 5.2 SIMD 阶段数据

| case | scalar load | scalar→UB store | VF `PUSH_PB` 时刻 | VF record 时刻 | first RV 时刻 | last RV 时刻 | window |
|---|---|---:|---:|---:|---:|---:|---:|
| 1op | 1432→1844（412） | 1843→2122（279） | 2131 | 2693 | 2659 | 2686 | 1261 |
| 4op same | 1430→1930（500） | 1916→2197（278/275/271/268） | 2208 | 2686 | 2643 | 2679 | 1256 |
| 4op diff | 1375→3227（1852） | 1796→3263（280/37/37/37） | 3274 | 3787 | 3744 | 3780 | 2412 |

关键：

- 编译器为 `vbr` 自动插入 `ST_XD_XN_IMM`，把 scalar 写到 UB staging；这是 SIMD 的额外一段。
- VF `PUSH_PB → first RV` 有 **435–528 cycles** 空窗；dump 显示是 vector icache 第一次 line fill：
  `send Read biu Req 2138 → recv Read biu Rsp 2650 → IFU send instr to IDU 2658`。
- 真正的 RV 指令窗口只有 **27–36 cycles**；`simd_lc_o4_same` 的 4 条 `RV_VADD` 时刻是 2658/2661/2664/2667，4 条共 9 cycles。
- `vf_execute_time` 是整个 VF（入口/取指 + 所有 RV 指令）的时间，不是单条 vector add 的时间。

### 5.3 SIMT 阶段数据

| case | LDS | LDG issue→retire | FADD 时刻（exec_time=8） | STS issue→retire（结果写 UB） | window |
|---|---|---|---:|---|---:|
| 1op | 2182→2198（16） | 2183→2666（483） | 2678 | 2688→2697（9） | 514 |
| 4op same | 1834→1850（16） | 1835/1837/1839/1841 → 2296/2775/3247/3777（1942） | 2308/2787/3259/3789 | 3799→3808（9） | 1973 |
| 4op diff | 1838→1854（16） | 1839/1841/1843/1845 → 2300/2351/2352/2353（514） | 2312/2363/2370/2373 | 2383→2392（9） | 553 |

> 表里的 FADD 数字是**绝对时间戳**，不是周期；`exec_time=8` 是单条 FADD 的执行延迟。
> `SIMT_STS` 是 32-lane 结果写 UB，已经算在 window 里；SIMD 对应的结果 store 是 VF 里的 `RV_VSTI`，
> 混在 VF 时间里，所以 SIMD 表没有单独 STS 行。

### 5.4 公平性：SIMD / SIMT 都是 32 元素向量 + scalar

两边逐元素运算相同：

```text
SIMD: for i in 0..31: ub_out[i] = ub_in[i] + s0
SIMT: lane i:        ub_out[i] = ub_in[i] + s0
```

| 环节 | SIMD | SIMT |
|---|---|---|
| 输入向量 | `RV_VLDI`：UB → vector register，mask=32 | `SIMT_LDS`：UB → 32 lane registers，`thread_count=32` |
| scalar 来源 | 标量 load → UB staging → `RV_VBR` 广播 | uniform load → 每条 lane 同一个寄存器值 |
| 计算指令 | `RV_VADD`（vector-vector） | `SIMT_FADD`（`execMask=ffffffff`，一条 warp 指令 32 lane） |
| 结果写回 | `RV_VSTI`（32 lane） | `SIMT_STS`（32 lane） |

所以这不是“SIMD 加向量、SIMT 加标量”；SIMT 代码里每个 lane 是标量，但 `SIMT_FADD` 是 32-lane 的 warp 指令。
当前 tile = 32 元素；若要比较满宽度，可另跑 tile=64 对照。

### 5.5 DCE 风险提示

- SIMD 结果通过 `vsts`/`RV_VSTI` 写到 UB `ub_out`；本版 ccec 没有被 DCE，dump 里能看到 `RV_VADD` / `RV_VSTI`。
- SIMT 侧同样能看到 `SIMT_FADD` / `SIMT_STS`。
- 检查命令：

```bash
grep -E "RV_VADD|RV_VSTI" camodel_results/simd_lc_o1/core0.veccore0.instr_log.dump
grep -E "SIMT_FADD|SIMT_STS" camodel_results/simt_lc_o1/core0.veccore0.instr_log.dump
```

- 换编译器 / 优化等级后建议重新检查；必要时加消费点或用 volatile UB 指针。

---

## 6. 使用注意

1. 本文件是**汇总层**；每个 case 的完整事件时间线、微指令解释和源数据路径在 [`README.md`](README.md) §1–§8。
2. CAModel 不是真卡；**真卡 SYS_CNT 对照见 §1.2（统一 ns）**。SIMD MainScalar load o1/same-line o4、
   SIMT uniform load o1/diff-line o4 与实测在 ≤~20% 内；SIMT same-line o4（模型不 merge）和所有 store
   （CAModel retire vs 真卡 write-buffer）是已定位的窗口问题，不能当成模型整体不准。
3. cost model 使用这些数据时，优先用“结构/趋势”（merge、hit、并发、固定入口成本），不要直接抄绝对 cycle。
4. 真卡 SYS_CNT 对照的两类数据：
   - **主表 marginal**：`syscnt/board_marginal/board_scalar_marginal.cce` + `board_vs_model_agg.csv`；
   - **单发 latency / 边界证据**：每个 case 的 `<case>_syscnt.cce` 在 `load/scalar_o1`、`load/scalar_o4`、
     `store/scalar_o1`、`store/scalar_o4`、`compute/scalar_add`、`load_vec_compute/scalar_o1_o4`；
     `build_syscnt.sh` 编译，`run_board_syscnt.sh`/`run_camodel_syscnt.sh` 跑，`parse_syscnt_results.py` 汇总；
     原始结果在 `syscnt/results/{board,camodel_ind_all,camodel_sequence,board_extra,camodel_extra}/`。
5. 下一轮建议：
   - tile=64 的 load+vec compute 对照；
   - SIMD VF 内 loop / warm vicache 对照；
   - VF 场景的 AIV 侧计时手段（当前 AIC `get_sys_cnt()` 外层包装测不到 VF 指令窗口，
     compute / load+vec 的 `<case>_syscnt.cce` 已作为边界证据归档）。
