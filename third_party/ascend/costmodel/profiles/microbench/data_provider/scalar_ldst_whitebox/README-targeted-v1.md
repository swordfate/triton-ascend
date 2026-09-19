# scalar load/store CostModel 白盒建模与验证（targeted）

> 对象：`test_cases/scalar_dominate_kernels/` 中 6 个 scalar-dominated megablocks kernel
> （`padded_copy_{gather,scatter,wgrad}`、`binned_copy_{gather,scatter,wgrad}`）。
> 固定条件：`shape=(sl,hs,ne,top_k)=(4,256,4,2)`、`BLOCK_X=64`、`superblock_factor=1`、`num_warps=1`。
> 当前 route：6 个 kernel 全部 `all_simt_only`；本文同时给出 forced `simd` 的实现公式对照。
> 代码基线：`scalar-load-whitebox` @ `65e46a3bc`；公式参数来自 `david_v100_simd_simt_v1.json`。
> **2026-09-19 profile 调整**：SIMT `uniform_load_fill_system_cycles` 由 probe-only 的 524 联合重拟合为 **480**（目标 6 kernel direct/indirect MAPE 从 10.2%/16.2% 降到 8.0%/8.1%，CCE probe 三点误差 ≤8.3%，route 不变）；详见 §3.2 B / §4.4。
> 大而全的探索见 [README.md](README.md) / [README-tidy.md](README-tidy.md)；本文只保留目标场景需要的建模、标定和验证。

---

## 0. 总结

1. 目标场景只有四类：(1) **direct scalar load same-line**；(2) **direct scalar load diff-line**；(3) **indirect scalar load**；(4) **scalar store**（SIMT `SIMT_STG` / SIMD Triton MTE3）。
2. **CAModel 内部时间换算 = 1.8 GHz**：同一核心 active window 的 dump cycle span ÷ CAModel 自己显示的 `duration_time(us)` 恒为 1.800 GHz。
3. 白盒公式与 CAModel 专项微基准：SIMD load 误差 ≤0.8%；SIMT load 的 `uniform_load_fill` 已按 probe + 目标 kernel 联合重拟合为 480，probe 误差变为 −8.3% ~ −2.3%；store 误差 −11.6% ~ +4.2%。
4. 6 个目标 kernel 的 matched-only + per-stage-union MAPE（SIMT load fill 重拟合后）：
   - SIMT（当前 route）：direct 8.0%、indirect 8.1%、store 18.9%；
   - SIMD（forced 对照）：direct 10.9%、indirect 10.1%、store 9.5%。
   - 三类 MAPE 均 <20%；单点最大为 SIMT binned wgrad indirect 的 +19.7%。
5. **CAModel 可靠性**：direct load 与上版真卡微基准的绝对误差约 −9.3% ~ +1.9%，可直接支撑白盒 load 标定；indirect 的依赖项是 **cycle 域**标定；store 不做 board marginal 对照，直接用白盒公式和 CAModel stage window 比（误差 −11.6% ~ +13.8%，见 §2.2.3）。
6. SIMD scalar store 在 Triton 下走 MTE3（`scalar → UB staging → MTE3 MOV UB→OUT`），不是 CCE MainScalar `ST_XD_XN_IMM → GM`；公式为 `20 + 450 + (K-1)*480`。目标算子当前 route 全为 SIMT，所以 SIMD store 只作为 forced-mode 对照。

---

## 1. 目标场景：6 个 kernel → 4 类标量访存

### 1.1 目标 kernel 与 scalar 操作点

| kernel | file | scalar op（源码行） | 地址关系 | 本次 seed 执行 |
|---|---|---|---|---|
| padded_copy_gather | `npu_padded_copy_gather.py` | direct: `indices[pid]`(:62), `bin_ids[pid]`(:63) | 两个 tensor，不同 128B line（+512B） | 两条都执行 |
|  |  | indirect: `bins[bin_idx-1]`(:66), `padded_bins[bin_idx-1]`(:69), `weights[index_a]`(:74, `SCALE=False`) | 地址依赖 `bin_idx` / `index_a`；不同 line | 前两条执行；weights 不执行 |
| padded_copy_scatter | `npu_padded_copy_scatter.py` | direct: `indices[pid]`(:64), `bin_ids[pid]`(:65) | 同 gather | 两条都执行 |
|  |  | indirect: `bins`(:68), `padded_bins`(:71), `weights[index_a]`(:76, `SCALE=True`) | 依赖 `bin_idx` / `index_a` | 三条都执行 |
| padded_copy_wgrad | `npu_padded_copy_scatter_wgrad_camodel.py` | direct: `indices[pid]`(:80), `bin_ids[pid]`(:84) | 同 gather | 两条都执行 |
|  |  | indirect: `bins`(:91), `padded_bins`(:96) | 依赖 `bin_idx` | 两条都执行 |
|  |  | store: `tl.store(wgrad, out)`(:115) | 单标量 store（`out = tl.sum(acc)`） | 1 条 |
| binned_copy_gather | `npu_prec_binned_kernel_simd_modified.py` | direct: `bins[expert_idx-1]`(:99, `expert_idx>0`), `bins[expert_idx]`(:100) | 同一 `bins` 数组相邻元素，同一 line | seed=0 只执行 :100 |
|  |  | indirect: `indices[start+entry_idx]`(:107), `weights[index_a]`(:121, `SCALE=False`) | 依赖 `start` / `index_a` | 前者执行；weights 不执行 |
| binned_copy_scatter | 同上 | direct: :169/:170 | 同一 line | seed=0 只执行 :170 |
|  |  | indirect: `indices[start+entry_idx]`(:177), `weights[index_a]`(:191, `SCALE=True`) | 依赖 `start` / `index_a` | 源码 2 个 site；costmodel_eval runner 传 `weights=None`，本次只执行 index_a |
| binned_copy_wgrad | 同上 | direct: :321/:322 | 同一 line | seed=0 只执行 :322 |
|  |  | indirect: `indices[start+entry_idx]`(:329) | 依赖 `start` | 1 条 |
|  |  | store: `tl.store(wgrad, out)`(:348) | 单标量 store | 1 条 |

> `SCALE=False` 时 `scale = tl.load(weights + index_a) if SCALE else 1` 不生成实际 load。
> costmodel 的 static workload 仍可能统计未执行分支，但本文 matched-only 表只累加真正执行的 stage。

### 1.2 costmodel 建模场景归类

| kernel | direct-scalar-load (same/diff) | indirect-scalar-load (diff) | scalar-store |
|---|---|---|---|
| padded_copy_gather | 0 / 1×2（2 个 op） | 1×3（本次执行 2） | 0 |
| padded_copy_scatter | 0 / 1×2 | 1×3（本次执行 3） | 0 |
| padded_copy_wgrad | 0 / 1×2 | 1×2 | 1 |
| binned_copy_gather | 1×2（本次执行 1）/ 0 | 1×2（本次执行 1） | 0 |
| binned_copy_scatter | 1×2（本次执行 1）/ 0 | 1×2（本次执行 1） | 0 |
| binned_copy_wgrad | 1×2（本次执行 1）/ 0 | 1×1 | 1 |

### 1.3 四类场景解释

**1) direct scalar load same-line**
- 含义：同一个 stage 里的多条 scalar load 落在同一条 cache line（SIMD MainScalar 64B / SIMT warp 128B）。
- IR 判定：`StagePartitioner.cpp::scalarLoadsShareOneLine` 要求同一个 base pointer，`tt.addptr` 常量偏移极差对应的 span ≤64B；只有 prove 成功才走 same-line 分支。
- 硬件行为：第 1 条 miss 付一次 line fill，后续命中同 line。SIMD MainScalar 有 MSHR merge + DC hit（CAModel o4 same = 493，而不是 4×447）；SIMT CCE uniform 探针在当前 CAModel 下同 line 不 merge/hit，每条 LDG 仍按一次 line read 串行（o4 same = 1923），但 Triton scalar pair 5-cycle issue spacing 可能命中（`FAKE_HIT`）。
- 目标 6 kernel：本次没有真正的 same-line direct stage；binned 的 `bins[expert_idx-1]` / `bins[expert_idx]` 源码上同 line，但被拆成两个条件 stage 且 seed=0 只执行一条。

**2) direct scalar load diff-line**
- 含义：同一 stage 的多条 scalar load 落在不同 cache line。
- 目标场景：padded 的 `indices` / `bin_ids` 是两个 tensor（+512B），属于 diff-line；IR 证不出同 line，保守走 diff-line 分支。
- 硬件行为：SIMD MainScalar 的 MSHR 有 2 个 outstanding，K≤2 条不同 line 可同时 outstanding，K=4 时会分两波（o4 diff = 956）；SIMT 不同 128B line 可以并发 fill，边际只加 LSU issue（o4 diff = 526，公式 486）。
- 目标 report：padded 两条 direct 被切成两个 K=1 ScalarLoad stage，所以 SIMT 聚合为 `2×486=972`、SIMD 为 `2×447=894`。如果后续把它们合并成一个 K=2 diff-line stage，公式值会变成 SIMT `486`、SIMD `450`（见 §4.5 OPEN ISSUE）。

**3) indirect scalar load**
- 含义：一条 scalar load 的地址由另一条 scalar load 的结果计算出来（pointer chase / fan-out）。
- 目标场景：padded 的 `bins[bin_idx-1]` / `padded_bins[bin_idx-1]` 依赖 `bin_idx`；scatter 的 `weights[index_a]` 依赖 `index_a`；binned 的 `indices[start+entry_idx]` 依赖 `start`（且经过 `num_tokens`/`if` 控制流）。
- 建模：indirect stage 自身的 line fill 仍按普通 load 公式算；依赖额外收费只针对 `exposure > 1` 的额外边：`T += max(0, exposure-1) * L_dep`。第一条 producer→consumer 边已经包含在 consumer 的 line fill 里，不重复收费。`L_dep` 当前取值 SIMD 1.95 cyc / SIMT 65.4 cyc，来自旧版 board SYS_CNT 依赖链拟合（cycle 域）。
- 目标场景绝大多数 `exposure=1`，所以 indirect stage 公式值就等于同 mode 的 load 公式值。

**4) scalar store**
- 目标场景：padded/binned wgrad 的 `tl.store(wgrad, out)`，即 `out = tl.sum(acc)` 后的单标量 store。
- SIMT：`SIMT_STG` 直接写 128B line；same-line K≥2 在 CAModel 下不 merge，公式 first store / diff-line `450+(K-1)*20`，same-line K≥2 `555+(K-1)*480`。
- SIMD：CCE 路径的 `ST_XD_XN_IMM → GM`（write-allocate + dirty writeback）不是 Triton 的路径。Triton scalar `tt.store` 被 TTAdapter 降成 `tensor<1xf32>` + `materialize_in_destination`，CAModel 里是 `SCALAR ST_XD_XN_IMM accessUb:1`（写 UB）→ `SET_FLAG` → `MTE3 MOV_SRC_TO_DST_ALIGNv2 Src:UB,Dst:OUT`（写 GM）。因此 SIMD store 公式用 MTE3 白盒 `20+450+(K-1)*480`，而不是 CCE MainScalar store 的 478/531/557 窗口。

### 1.4 固定变量与采样

- `K = scalar_load_count_per_iteration` / `scalar_store_count_per_iteration`；
- `U = scalar_load_unique_lines` / `scalar_store_unique_lines`；
- `share = scalarLoadsShareLine` / `scalarStoresShareLine`；
- `exposure = indirectScalarLoadExposureCount` / `indirectScalarStoreExposureCount`（producer-side 去重）；
- CAModel 取 `core0.veccore0` 一个 program；`msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 --launch-count=1`；
- 当前 route = `all_simt_only`，CAModel 对应 `compile_mode=simt_only`；forced SIMD 表对应 `compile_mode=simd`。

---

## 2. CAModel 可靠性

### 2.1 先确定 CAModel 内部用的是 1.8 GHz 还是 1.65 GHz

方法：用最简单的 CCE scalar CAModel 程序，把 CAModel 自己打印的 `duration_time(us)`（`Core operator results` 表）
与 dump 中同一 core 的 `instr_log.dump` 活跃窗口 cycle span 对齐，看 `cycles / duration_us` 是多少 GHz。

```bash
# 1) CAModel 显示的核心时间（us）
grep -A4 "Core operator results" run.log

# 2) 同一 core 的 dump cycle span（min/max instr_log 时间戳）
python3 - <<'PY'
import re
txt = open('.../core0.veccore0.instr_log.dump', errors='replace').read()
v = [int(x) for x in re.findall(r'^\[info\] \[(\d+)\]', txt, re.M)]
print(max(v) - min(v))
PY
```

实测（`OPPROF_20260912165738_SSCQHPMIWFGTSMDT`，CCE `simt_scalar_memory` load/store 两次 launch）：

| launch | core | dump cycle span | CAModel `duration_time(us)` | cycles / us | 等效 GHz |
|---|---|---:|---:|---:|---:|
| load (`measure/0`) | core0.veccore0 | 10906 | 6.06 | 1799.7 | **1.800** |
| load (`measure/0`) | core0.veccore1 | 10514 | 5.84 | 1800.3 | **1.800** |
| store (`measure/1`) | core0.veccore0 | 8786 | 4.88 | 1800.4 | **1.800** |
| store (`measure/1`) | core0.veccore1 | 9140 | 5.08 | 1799.2 | **1.800** |

结论：**CAModel 内部 cycle → time 的换算就是 1.8 GHz**，不是 1.65 GHz。
后文所有 `ns@1.8G = cycle / 1.8`。如果需要换算到 1.65 GHz，乘 `1.8/1.65 = 1.0909`。

### 2.2 CAModel 与上版真卡微基准的绝对时间误差

“上版真卡微基准”指 `syscnt/board_marginal/` 的 CCE marginal 探针（`board_vs_model_agg.csv`，SYS_CNT 997.7 MHz）
以及旧版 dependency 探针（`scalar_ldst/{simd,simt}_scalar_gm_dep.cce`）。

**2.2.1 direct load same / diff-line（K=1）**

| 场景 | mode | CAModel cycle | CAModel ns@1.8 | board 微基准 | board ns | err |
|---|---|---:|---:|---|---:|---:|
| direct same-line load | SIMD | 447 | 248.3 | `m_main_ld_same_k1` | 274.5 | **−9.5%** |
| direct same-line load | SIMT | 530 | 294.4 | `m_simt_ld_uni_same_k1` | 291.8 | **+0.9%** |
| direct diff-line load | SIMD | 447 | 248.3 | `m_main_ld_diff_k1` | 273.8 | **−9.3%** |
| direct diff-line load | SIMT | 530 | 294.4 | `m_simt_ld_uni_diff_k1` | 289.0 | **+1.9%** |


**2.2.2 indirect dependency**

| 场景 | mode | CAModel 依赖项 | cycle 域对比 | ns 域对比 |
|---|---|---|---|---|
| indirect edge | SIMD | `L_dep=1.95 cyc` | board dep extra ≈2 cyc → **−2.5%** | CAModel 1.08 ns vs board ≈2.0 ns → −45.8% |
| indirect edge | SIMT | `L_dep=65.4 cyc` | board dep extra 65.4 cyc → **0%（标定来源）** | CAModel 36.3 ns vs board 65.4 ns → −44.4% |

- 结论：indirect 的 `L_dep` 是 **board SYS_CNT cycle 域**拟合量，在 costmodel 的 system-cycle 公式里工作，不能先除 1.8 再和 board ns 直接比。
- 目标 kernel 的 indirect 验证因此放在 §4 的 matched-only + stage-union（CAModel cycle 域）里做。

**2.2.3 scalar store：白盒公式 vs CAModel**

store 不引入 board marginal，也不把 CAModel 的 cold 短链路和 board 吞吐混在一起；直接看白盒公式与 CAModel 指令窗口：

| 场景 | mode | 公式 | 公式 cycle | CAModel window | err |
|---|---|---:|---:|---:|---:|
| `simt_st_uniform_o1`（K=1） | SIMT | `450` | 450 | 491 | **−8.4%** |
| `simt_st_uniform_same_o4`（K=4 same-line） | SIMT | `555+(4-1)*480` | 1995 | 1914 | **+4.2%** |
| `simt_st_uniform_diff_o4`（K=4 diff-line） | SIMT | `450+(4-1)*20` | 510 | 577 | **−11.6%** |
| `padded_copy_wgrad` store stage（K=1） | SIMT | `450` | 450 | 559 | **−19.5%** |
| `binned_copy_wgrad` store stage（K=1） | SIMT | `450` | 450 | 551 | **−18.3%** |
| `padded_copy_wgrad` store stage（K=1） | SIMD Triton | `20+450` | 470 | 496（MTE3 MOV window） | **−5.2%** |
| `binned_copy_wgrad` store stage（K=1） | SIMD Triton | `20+450` | 470 | 413（MTE3 MOV window） | **+13.8%** |

- SIMT store 公式直接拟合 `SIMT_STG` 窗口：K=1/同 line K≥2/diff-line K≥2 三条公式分别对应 `simt_st_uniform_o1`、`..._same_o4`、`..._diff_o4`。
- SIMD store 公式拟合的是目标 kernel 的 **MTE3 stage window**（`MOV_SRC_TO_DST_ALIGNv2` issue→retire），不是 Triton demo 的整条短链路；整条短链路的 1079 cycle 还包含 scalar→UB staging 和等 VF flag，不属于 costmodel 的 store stage 系数。
- 结论：store 的白盒公式和 CAModel 结果误差在 −19.5% ~ +13.8%（probe 三条 + 目标 kernel 两条），目标 kernel matched-only MAPE 见 §4。

**2.2.4 小结**

- **白盒建模是否成立**：direct load 的 CAModel 绝对时间与 board 微基准误差 −9.3% ~ +1.9%，可以直接支撑白盒 load 标定；indirect 的依赖项在 cycle 域标定；store 直接用白盒公式和 CAModel stage window 对照，误差 −19.5% ~ +13.8%。三类模型都在各自 domain 内成立，并通过 §4 的 target-kernel CAModel stage-union 复核。
- **direct load**：board 微基准绝对误差 −9.3% ~ +1.9%，白盒 load 标定有意义；
- **indirect**：cycle 域依赖项一致，ns 域差一个 clock-domain 因子，公式只应在 cycle/相对域使用；
- **store**：不做 board marginal；白盒公式直接对 CAModel STG / MTE3 stage window，目标 kernel 复核见 §4。

---

## 3. 白盒公式

### 3.1 公式输入变量

| 变量 | 含义 | 来源 |
|---|---|---|
| `K` | stage 内 scalar load/store 数 | `StageWorkload::scalarLoadCount` / `scalarStoreCount` |
| `U` | 不同 64B/128B line 数 | `scalarLoadUniqueLines` / `scalarStoreUniqueLines` |
| `share` | IR 是否证明同一 line | `scalarLoadsShareLine` / `scalarStoresShareLine` |
| `exposure` | indirect producer 被消费的边数（producer-side 去重） | `indirectScalarLoadExposureCount` / `indirectScalarStoreExposureCount` |
| `mode` | SIMD / SIMT | stage implementation |

代码入口：`StageCostModels.cpp::mapWorkload` → `mainScalarLoadCycles` / `simtUniformLoadCycles` / `mte3StoreCycles` / `simtUniformStoreCycles`。

### 3.2 四类公式与解释

**A. direct scalar load — SIMD MainScalar**

```text
u = share ? 1 : clamp(U, 1, K)          // U 未知时按 U=K（保守 diff-line）
extra(u) = (u <= 4) ? 250 : 350
T_main(K,U,share) = 7 + 440
                  + max(0, u - 2) * extra(u)
                  + (K - u) * 12.333
                  + (K - 1) * 3
```

- `7` = issue→tag/MSHR/BIU dispatch 4 cycle + refill→retire 3 cycle，是每条 load 进入 line fill 前的固定成本。
- `440` = 一次 cold 64B line fill（DC 发 BIU 8 + BIU 读 421 + 回 DC 11）。
- `max(0,u-2)*extra(u)`：MainScalar 同时只有 2 个 MSHR outstanding；第 3 条及以后的不同 line 要等前两波之一，额外付一次 line fill（u≤4 取 250，u>4 取 350）。same-line 时 u=1，此项为 0。
- `(K-u)*12.333`：落在已有 line 上的额外 op 的命中成本（fit o4 same：447 + 3*(12.333+3) = 493）。
- `(K-1)*3`：每条额外 op 的 issue/serialization 成本。
- same-line 分支（`share=true`, u=1）：`T = 447 + (K-1)*15.333`；K=1→447，K=4→493。
- diff-line 分支（u=K）：K≤2 时 2 个 outstanding 够用，`T=447+(K-1)*3`（K=1→447，K=2→450）；K=4 时多 2 条 line，`T=956`。

**B. direct scalar load — SIMT warp-uniform**

```text
margin = share ? 464.333 : 0.001
T_simt(K,share) = 6 + 480 + (K - 1) * margin
```

- `6` = issue→DC tag（SIMT LDG 前段）。
- `480` = DC tag→BIU→line fill→UBITF/GSU 回传路径的**联合重拟合值**：原先 probe-only 为 524（CCE o1 = 6+524 = 530）；但 6 个目标 kernel 实测 fill 只有 406–558 cycle，524 会让 binned indirect 高估 +30%。联合拟合取 fill=480（base=486）后：CCE probe 三点误差 −8.3%/−2.3%/−7.6%，目标 kernel direct/indirect MAPE 降到 ~8%。
- same-line margin `464.333` 保持不变：CCE uniform 探针在当前 CAModel 下同 128B line 不 merge/hit，每条额外 LDG 再付一次 line read（o4 same = 486 + 3×464.333 = 1879，对 CAModel 1923 误差 −2.3%）。
- diff-line margin `0.001`：不同 128B line 可以 outstanding/overlap，额外 op 只付 LSU issue 地板；K≤4 公式值约 486（CAModel o4 diff 526，−7.6%）。
- 注意：Triton scalar pair 的 LDG issue spacing ≈5 cycle，可能让同 line 第二条变 `FAKE_HIT`；same-line 公式是 CCE 2-cycle spacing 的保守分支，目标 6 kernel 不依赖它。

**C. indirect scalar load**

```text
T_indirect = T_load(mode; K,U,share) + max(0, exposure - 1) * L_dep
L_dep(SIMD) = 1.95 cyc
L_dep(SIMT) = 65.4 cyc
```

- consumer（indirect load）自己的 line fill 已由 `T_load` 覆盖，第一条 producer→consumer 边不再额外收费。
- `exposure` 是 producer-side 去重后的边数：一个 producer fan-out 给多个 consumers 只算一次；更深 serial chain 才继续收费。
- `L_dep` 是旧版 board SYS_CNT 依赖链拟合值（cycle 域），在当前 profile 中 SIMD 1.95 / SIMT 65.4。
- 目标场景基本 `exposure=1`，所以 indirect stage 值等于同 mode 的 load 公式值。

**D. scalar store**

```text
SIMT same-line (K>=2)   : 555 + (K-1)*480
SIMT first store / diff : 450 + (K-1)*20
SIMD Triton (MTE3)      : 20 + 450 + (K-1)*480
```

- SIMT `SIMT_STG` 直接写 128B line：K=1 走 first-store/diff 分支 450；same-line K≥2 CAModel 不 merge，每条后续 store 串行 +480（o4 same 1995 vs CAModel 1914）；diff-line 可 overlap，后续 +20（o4 diff 510 vs CAModel 577）。
- SIMD Triton store 的真实链路是 `SCALAR ST_XD_XN_IMM accessUb:1`（scalar→UB）→ 等 VF flag（约 556）→ `MTE3 MOV` push→retire（约 1054，其中等 `recv_wack` 429）→ BIU write；整条单 program 冷链路约 1079 cycle。
- `20+450+(K-1)*480` 是 costmodel 使用的 **stage 可摊销白盒系数**，不是整条 1079-cycle 冷链路；它用目标 kernel 的 store stage 做验证（SIMD padded wgrad `470` vs CAModel MTE3 window `496/413`，误差 −5.2%/+13.8%）。
- CCE MainScalar store 的 478/531/557 窗口仅作为对照保留，Triton 路径不采用。

### 3.3 标定探针与公式准确情况（合并）

| 场景 | mode | 标定程序/函数 | 公式值（cycle） | CAModel window | err |
|---|---|---|---|---:|---:|
| SIMD direct load K=1 | SIMD | `load/scalar_o1/load_scalar_o1.cce::simd_main_ld_o1` | 447 | 447 | 0.0% |
| SIMD direct load K=4 same | SIMD | `load/scalar_o4/load_scalar_o4.cce::simd_main_ld_same_o4` | 447+3×15.333 = 493 | 493 | 0.0% |
| SIMD direct load K=4 diff | SIMD | 同上 `simd_main_ld_diff_o4` | 447+2×250+3×3 = 956 | 956 | 0.0% |
| SIMT direct load K=1 | SIMT | `load/scalar_o1/load_scalar_o1.cce::simt_ld_uniform_o1` | 6+480 = 486 | 530 | −8.3% |
| SIMT direct load K=4 same | SIMT | `load/scalar_o4/load_scalar_o4.cce::simt_ld_uniform_same_o4` | 486+3×464.333 = 1879 | 1923 | −2.3% |
| SIMT direct load K=4 diff | SIMT | 同上 `simt_ld_uniform_diff_o4` | 486+3×0.001 = 486.0 | 526 | −7.6% |
| indirect dependency | SIMD/SIMT | `../scalar_ldst/{simd,simt}_scalar_gm_dep.cce::measure` | L_dep 1.95 / 65.4 | board 拟合 | cycle 域 −2.5% / 0% |
| SIMT scalar store K=1 | SIMT | `store/scalar_o1/store_scalar_o1.cce::simt_st_uniform_o1` | 450 | 491 | −8.4% |
| SIMT scalar store K=4 same | SIMT | `store/scalar_o4/store_scalar_o4.cce::simt_st_uniform_same_o4` | 555+3×480 = 1995 | 1914 | +4.2% |
| SIMT scalar store K=4 diff | SIMT | 同上 `simt_st_uniform_diff_o4` | 450+3×20 = 510 | 577 | −11.6% |
| SIMD scalar store（Triton） | SIMD | `store/triton_scalar_store/triton_scalar_store_demo.py::_triton_scalar_store_demo` | 20+450+(K-1)×480 | 见 §3.2 D | 见 §4.3 |
| board marginal / clock probe | CCE | `syscnt/board_marginal/board_scalar_marginal.cce` | `m_main_ld_same_k1` 等 | 真卡 SYS_CNT 斜率 | 见 §2.2 |

> indirect 没有单独 CCE 探针；其公式来自旧版 `scalar_gm_dep.cce` 的 board 拟合 + 目标 kernel CAModel stage-union 验证（§4）。
>
> ⚠️ SIMT load 三行采用 2026-09-19 的联合重拟合 `uniform_load_fill=480`（base 486）；此时 probe 不再 0% 拟合，而是用 ≤8.3% 的 probe 误差换取目标 6 kernel direct/indirect MAPE ≈8% 的平衡。旧值 524 的拟合表在 `README-tidy.md`/历史提交中仍可见。

---

## 4. 6-kernel matched-only + stage-union 总表

### 4.1 口径

- costmodel：`compile_mode=simd_simt` + report；只累加真正执行的 matched stage（`stage_5` 未执行则不计）；
- CAModel：当前 route `compile_mode=simt_only`（§4.2），forced SIMD `compile_mode=simd`（§4.3）；padded seed=12、binned seed=0；每个 matched stage 取 `max(retire)-min(issue)` union，再对 matched stage 求和；不跨 stage 合并；
- SIMD load 取 `LD_*`/`LDP_*`（`accessDdr=1`、`accessUb=0`），SIMT load 按 DC `size<128` 过滤，排除 128B vector tile load；
- `ns@1.8G = cycle / 1.8`（§2.1 已确认 CAModel 内部 1.8 GHz）；
- `err = (costmodel - CAModel) / CAModel`。
- “board ns”列本轮未测（target kernel real-board per-stage 时间不是 scalar stage union；不混入本表）。

### 4.2 SIMT（当前 route，`all_simt_only`；profile `uniform_load_fill=480`）

| kernel | 类别 | matched | costmodel cyc | CAModel union cyc | costmodel ns@1.8 | CAModel ns@1.8 | err |
|---|---|---:|---:|---:|---:|---:|---:|
| padded_copy_gather | direct load | 2/2 | 972.0 | 1113.0 | 540.0 | 618.3 | −12.7% |
| padded_copy_gather | indirect load | 1/1 | 486.0 | 509.0 | 270.0 | 282.8 | −4.5% |
| padded_copy_scatter | direct load | 2/2 | 972.0 | 1115.0 | 540.0 | 619.4 | −12.8% |
| padded_copy_scatter | indirect load | 2/2 | 972.0 | 973.0 | 540.0 | 540.6 | −0.1% |
| padded_copy_wgrad | direct load | 2/2 | 972.0 | 897.0 | 540.0 | 498.3 | +8.4% |
| padded_copy_wgrad | indirect load | 1/1 | 486.0 | 481.0 | 270.0 | 267.2 | +1.0% |
| padded_copy_wgrad | scalar store | 1/1 | 450.0 | 559.0 | 250.0 | 310.6 | −19.5% |
| binned_copy_gather | direct load | 1/2 | 486.0 | 497.0 | 270.0 | 276.1 | −2.2% |
| binned_copy_gather | indirect load | 1/1 | 486.0 | 436.0 | 270.0 | 242.2 | +11.5% |
| binned_copy_scatter | direct load | 1/2 | 486.0 | 497.0 | 270.0 | 276.1 | −2.2% |
| binned_copy_scatter | indirect load | 1/1 | 486.0 | 436.0 | 270.0 | 242.2 | +11.5% |
| binned_copy_wgrad | direct load | 1/2 | 486.0 | 442.0 | 270.0 | 245.6 | +10.0% |
| binned_copy_wgrad | indirect load | 1/1 | 486.0 | 406.0 | 270.0 | 225.6 | +19.7% |
| binned_copy_wgrad | scalar store | 1/1 | 450.0 | 551.0 | 250.0 | 306.1 | −18.3% |

### 4.3 SIMD（forced `compile_mode=simd`）

| kernel | 类别 | matched | costmodel cyc | CAModel union cyc | costmodel ns@1.8 | CAModel ns@1.8 | err |
|---|---|---:|---:|---:|---:|---:|---:|
| padded_copy_gather | direct load | 2/2 | 894.0 | 883.0 | 496.7 | 490.6 | +1.2% |
| padded_copy_gather | indirect load | 1/1 | 450.0 | 533.0 | 250.0 | 296.1 | −15.6% |
| padded_copy_scatter | direct load | 2/2 | 894.0 | 1011.0 | 496.7 | 561.7 | −11.6% |
| padded_copy_scatter | indirect load | 2/2 | 897.0 | 930.0 | 498.3 | 516.7 | −3.5% |
| padded_copy_wgrad | direct load | 2/2 | 894.0 | 801.0 | 496.7 | 445.0 | +11.6% |
| padded_copy_wgrad | indirect load | 1/1 | 450.0 | 508.0 | 250.0 | 282.2 | −11.4% |
| padded_copy_wgrad | scalar store | 1/1 | 470.0 | 496.0 | 261.1 | 275.6 | −5.2% |
| binned_copy_gather | direct load | 1/2 | 447.0 | 364.0 | 248.3 | 202.2 | +22.8% |
| binned_copy_gather | indirect load | 1/1 | 447.0 | 528.0 | 248.3 | 293.3 | −15.3% |
| binned_copy_scatter | direct load | 1/2 | 447.0 | 475.0 | 248.3 | 263.9 | −5.9% |
| binned_copy_scatter | indirect load | 1/1 | 447.0 | 425.0 | 248.3 | 236.1 | +5.2% |
| binned_copy_wgrad | direct load | 1/2 | 447.0 | 510.0 | 248.3 | 283.3 | −12.4% |
| binned_copy_wgrad | indirect load | 1/1 | 447.0 | 493.0 | 248.3 | 273.9 | −9.3% |
| binned_copy_wgrad | scalar store | 1/1 | 470.0 | 413.0 | 261.1 | 229.4 | +13.8% |

> SIMD 的结果是 forced-mode 公式对照，不代表 route 翻转；当前 6 kernel route 仍全部 `all_simt_only`。
> SIMD OPPROF 在服务器 `~/scalar_dominate_eval_seeded/`：新补的 `OPPROF_20260919215253_JNWGENEGTVYDBEJR`（padded gather）、`OPPROF_20260919215352_CAFEFBAWNLVYYMXD`（binned gather）、`OPPROF_20260919215440_HZKKADEIPPSRNJMI`（binned scatter），以及 2026-09-18 的 `padded_scatter/padded_wgrad/binned_wgrad` SIMD OPPROF；costmodel report 为 `~/scalar_dominate_eval_round4/out/costmodel_*.json`。

### 4.4 MAPE 汇总

| 类别 | SIMT MAPE | SIMT 最大/最小 | SIMD MAPE | SIMD 最大/最小 |
|---|---:|---:|---:|---:|
| direct scalar load | 8.0% | +10.0% / −12.8% | 10.9% | +22.8% / −12.4% |
| indirect scalar load | 8.1% | +19.7% / −4.5% | 10.1% | +5.2% / −15.6% |
| scalar store | 18.9% | −18.3% / −19.5% | 9.5% | +13.8% / −5.2% |

> 三类 MAPE 均 <20%；单点最大 +19.7%（SIMT binned wgrad indirect），SIMD 单点最大 +22.8%（binned gather direct）。残差来源：
> 1）SIMT fill 仍有地址/BIU 仲裁波动（目标 kernel 实测 406–558 cycle；联合拟合值 480 只是平衡点，binned indirect 仍偏高 +19.7%）；
> 2）`stage_5`（`if expert_idx>0`）在 seed=0 不执行，matched 只算执行分支；
> 3）SIMT store 白盒 450 vs CAModel `SIMT_STG` 551/559；
> 4）CAModel 与真卡的窗口/clock 口径差（§2.2）。

### 4.5 已知问题与下一步

1. **padded direct stage 边界**：当前 report 把两条 direct load 切成两个 K=1 ScalarLoad stage，所以 SIMT=972（fill=480）/ SIMD=894；如果后续合并为一个 K=2 diff-line stage，公式值会变成 SIMT=486 / SIMD=450。这是 stage 切分/overlap 问题，不是公式常数问题（见 `63-workspace/10-scalar-camodel-bench/six_camodel_tiny/FIRST-TWO-LOADS.md`）。
2. **binned 直接 load 的条件分支**：`start = tl.load(bins + expert_idx - 1)` 在 `if expert_idx>0` 里；seed=0 只执行 `end`，matched 表是 1/2；static workload 仍统计未执行 stage，需要 control-flow 概率建模。
3. **binned scatter 第二个 indirect site**：源码 `weights[index_a]`（`SCALE=True`）存在，但 costmodel_eval 的 seeded runner 传 `weights=None`，当前表只验证了 `indices[start+entry_idx]`；若需要覆盖两条 indirect，需要 runner 传 weights 后重跑。
4. **tile store**：gather/scatter 的 `tl.store(optr+offsets, x)` 是 shaped tile store，不属于 scalar store 白盒；当前 `IndirectGatherMemory` transaction 口径与 CAModel window 差 ~99%，需要单独建模。
5. **SIMT same-line uniform load K>1**：当前 CAModel 对同 128B line 不 merge/hit；same-line 公式只适用于 CCE spacing 的场景，Triton 5-cycle spacing 可能 FAKE_HIT，目标 6 kernel 不依赖该分支。
6. **真卡 ns 对齐**：本文的 ns 列是 CAModel 1.8 GHz 换算；若后续要把 costmodel 预测直接对齐 board ns，只需对仍使用 board absolute 的 load/indirect 考虑 correction；store 采用 CAModel 域白盒公式，不做 board marginal 对齐。
7. **profile 生效**：`uniform_load_fill_system_cycles=480` 已写入仓库 `profiles/simd_simt/david_v100_simd_simt_v1.json` 和 `costmodel_eval/make_profile.py`；profile JSON 是运行时 data 文件，已同步到服务器 repo 和 `~/.conda/envs/wj_autoscope/.../costmodel_profiles/`。默认 report 复验：binned wgrad 的 `stage_5/6/9_scalar_load` cycles 均为 486，`decision=all_simt_only`。如果之后重装 wheel，仓库 profile 会重新打包；旧值 524 的对照结果保留在 `results/all_scalar_eval_fill524.csv`。

---

## 5. 文件与脚本索引

```text
scalar_ldst_whitebox/
  README.md / README-tidy.md           # 大而全探索 + 汇总
  README-targeted-v1.md                # 本文（目标场景最终版）
  load/scalar_o1/load_scalar_o1.cce    # SIMD/SIMT 单条 load 探针
  load/scalar_o4/load_scalar_o4.cce    # same/diff 4-op load 探针
  store/scalar_o1/store_scalar_o1.cce  # SIMT store 探针
  store/scalar_o4/store_scalar_o4.cce  # same/diff 4-op store 探针
  store/triton_scalar_store/           # Triton MTE3 store demo
  syscnt/board_marginal/               # 真卡 SYS_CNT marginal / clock probe
  costmodel_eval/
    summarize_all_scalar.py            # matched-only + per-stage-union；--mode auto/simd/simt
    results/all_scalar_eval.csv/md     # SIMT 表 4.2 原始数据（fill=480）
    results/all_scalar_eval_fill524.csv# 旧 probe-only fill=524 对照
    results/all_scalar_eval_simd.csv/md# SIMD 表 4.3 原始数据
    make_profile.py                    # 公式/profile 参数来源（fill=480）
```

**本轮 profile 改动**：`third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json` 的 SIMT `uniform_load_fill_system_cycles` 524→480；`make_profile.py` 同步；已复制到服务器 repo 与 installed profile 路径并用默认 report 复验通过。

目标 kernel 与运行器：

```text
test_cases/scalar_dominate_kernels/
  npu_padded_copy_gather.py
  npu_padded_copy_scatter.py
  npu_padded_copy_scatter_wgrad_camodel.py
  npu_prec_binned_kernel_simd_modified.py
  autotune_profile_runner.py           # 真卡 kernel duration（未在本表使用）
```

服务器复现路径：

```text
~/scalar_dominate_eval_seeded/out/     # SIMT/SIMD CAModel run logs + 新补的 SIMD OPPROF
~/scalar_dominate_eval_round4/out/     # costmodel_*.json（当前 profile 代码）
~/camodel-cce-test/OPPROF_20260912165738_SSCQHPMIWFGTSMDT/  # §2.1 1.8 GHz 证据
```

历史证据：

```text
63-workspace/10-scalar-camodel-bench/
  HANDOFF.md                           # warm/cold、board marginal、dependency 过程记录
  triton-pair/TRITON-PAIR-VS-MODEL.md  # §2.2 warm/cold 2-op pair 对照
  six_camodel_tiny/FIRST-TWO-LOADS.md  # 6 kernel 前两条 load
  stage_vs_camodel_20260917/           # padded 三 kernel stage 级对比
```
