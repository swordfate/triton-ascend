# Diff-line scalar load/store：cycle 域半白盒公式与实测误差

> 数据：2026-09-27 multi-ptr K-sweep（load）+ 2026-09-28 store 重测
> （`triton_probe/store_probe_variants.py`，matched-sink baseline + readback/sync 变体）。
> 实测转 cycle：`measured_cycle = measured_us × 1000 × 1.8`；误差：`err = (measured − pred) / pred`。
> 指标选择：load 用 `Duration(us)_delta`；SIMT store 用 **matched-sink** `Duration(us)_delta`；
> SIMD scalar store 是异步 MTE3，kernel Duration 没有信号，改用 `aiv_mte3_time(us)_delta`。
> 语义约定：`ScalarStore` 表示 store 资源占用/吞吐，**不含完成/可见延迟**；强制 readback/barrier 变体只作为边界证据。
> **K=8 异常**：原 store 重测 run 的 SIMT K=8=229 ns 三次都异常偏高（222/236 ns），
> 但同一脚本的三次 fresh 重跑得到 66.5/57.5/61.5 ns，matched K=8 探针 61.5 ns；
> 确认是运行状态异常，不是程序逻辑问题。本版 K=8 用 fresh run 中位数，其余 K 用跨 run 中位数。

## 0. Store 重测结论（2026-09-28）

1. **旧 K=2/K=4 的“噪声/负值”主要是 baseline 不对齐**：旧 target `store_k` 没有 baseline
   `arith_k` 的 sink `tl.store(out_ptr, acc)`，`target - baseline` 少减了一次 store。
2. 给 target 补上同一个 sink 后，SIMT store 的 matched-sink `Duration_delta` 恢复单调：
   K=1/2/4/8 的跨 run 中位数为 **16 / 25 / 37 / 61.5 ns**（28.8 / 45.0 / 66.6 / 110.7 cycle）。
3. **原 K=8=229 ns 是运行状态异常**：原 run 三个 round 稳定在 222/236 ns，
   但事后用同一脚本 fresh 重跑三次得到 66.5 / 57.5 / 61.5 ns，matched K=8 探针 61.5 ns；
   程序与 baseline 逻辑不变，说明是运行时异常状态（不可复现），本版排除该 run 的 K=8，
   K=1/2/4 仍参与跨 run 中位数。
4. SIMD scalar store 走 MTE3，是异步写：matched sink 后 kernel `Duration_delta` 仍接近 0；
   改用 `aiv_mte3_time_delta` 后跨 run 中位数为 **6.5 / 59.5 / 60.5 / 74.5 ns**
   （11.7 / 107.1 / 108.9 / 134.1 cycle）。
5. readback / barrier+readback 变体（v1/v2）会让 SIMD Duration / MTE3 变成 ~90–110 ns 的平线，
   这是“强制 store 可见/同步”的成本，不是 store 的边际成本；建模只使用 matched-sink/异步 pipe 数据。

## 1. 公式（cycle 域，半白盒）

### 1.1 SIMD diff-line scalar load

```text
T_simd_load(K) = prep + fill + max(0, K - outstanding) * extra + (K - 1) * issue
prep = 7, fill = 263.9, outstanding = 1, extra = 278.74, issue = 1
```

- K=1 锚定 `prep + fill = 270.9`；K=8 锚定 extra；K=2/K=4 检查 −6.0% / −9.9%。

### 1.2 SIMT diff-line scalar load

```text
T_simt_load(K) = prep + fill + max(0, K - threshold) * extra + (K - 1) * issue
prep = 6, fill = 306.75, threshold = 2, extra = 74.775, issue = 1
```

- K=1/K=2 共用 base 312.75；K=4/K=8 检查 −13.2% / −0.9%。

### 1.3 SIMD scalar store（异步 MTE3，用 `aiv_mte3_time` 标定）

```text
K == 1 : T = fill
K >= 2 : T = fill + activation + (K - 2) * subsequent

fill = 11.7, activation = 95.4, subsequent = 4.5
K=1/2/4/8 预测：11.7 / 107.1 / 116.1 / 134.1 cycle
```

- 物理含义：单条 scalar store 的 MTE3 active time 很小（~12 cycle）；
  第 2 条 store 触发 MTE3 line 写路径的 activation（~95 cycle）；
  之后每条额外 store 的边际约 4.5 cycle。
- 语义边界：这是 **MTE3 pipe active 资源占用**，不含 scalar→UB staging、等 VF flag 和 write ack 完成；
  强制可见/readback 的额外成本见 §3.5 与 `HANDOFF.md` §4。
- 拟合目标：`aiv_mte3_time_delta` 跨 run 中位数 11.7 / 107.1 / 108.9 / 134.1 cycle，
  最大误差 −6.2%。
- profile 落盘：`fill=6.4279`、`serial=52.4117`、
  `subsequent=2.4723`（SYS_CNT cycle）。

### 1.4 SIMT diff-line scalar store（matched-sink Duration）

```text
T_simt_store(K) = base + (K - 1) * subsequent

base = 30.4, subsequent = 12.348
K=1/2/4/8 预测：30.4 / 42.7 / 67.4 / 116.8 cycle
```

- 拟合目标：跨 run matched-sink `Duration_delta` 中位数 28.8 / 45.0 / 66.6 / 110.7 cycle，
  最大误差 ±5.3%。
- 物理含义：单条 store 可见成本约 30 cycle；每条额外 store 的 write-buffer/issue
  边际约 12.3 cycle。
- profile 落盘：`uniform_store_base=16.7014`、`subsequent=6.7839`（SYS_CNT cycle）。

## 2. 实测 vs 公式

| mode | kind | K | 指标 | 实测 us | 实测 cycle | 公式 cycle | 误差 |
|---|---|---:|---|---:|---:|---:|---:|
| simd | load | 1 | Duration | 0.1505 | 270.9 | 270.9 | +0.0% |
| simd | load | 2 | Duration | 0.2875 | 517.5 | 550.6 | −6.0% |
| simd | load | 4 | Duration | 0.5555 | 999.9 | 1110.1 | −9.9% |
| simd | load | 8 | Duration | 1.2345 | 2222.1 | 2229.1 | −0.3% |
| simt_only | load | 1 | Duration | 0.1770 | 318.6 | 312.8 | +1.9% |
| simt_only | load | 2 | Duration | 0.1705 | 306.9 | 313.8 | −2.2% |
| simt_only | load | 4 | Duration | 0.2245 | 404.1 | 465.3 | −13.2% |
| simt_only | load | 8 | Duration | 0.4230 | 761.4 | 768.4 | −0.9% |
| simd | store | 1 | aiv_mte3_time | 0.0065 | 11.7 | 11.7 | +0.0% |
| simd | store | 2 | aiv_mte3_time | 0.0595 | 107.1 | 107.1 | +0.0% |
| simd | store | 4 | aiv_mte3_time | 0.0605 | 108.9 | 116.1 | −6.2% |
| simd | store | 8 | aiv_mte3_time | 0.0745 | 134.1 | 134.1 | +0.0% |
| simt_only | store | 1 | matched Duration | 0.0160 | 28.8 | 30.4 | −5.3% |
| simt_only | store | 2 | matched Duration | 0.0250 | 45.0 | 42.7 | +5.3% |
| simt_only | store | 4 | matched Duration | 0.0370 | 66.6 | 67.4 | −1.3% |
| simt_only | store | 8 | matched Duration | 0.0615 | 110.7 | 116.8 | −5.3% |

误差汇总：

| 类别 | 可用点 | 最大相对误差 |
|---|---|---:|
| SIMD load | K=1/2/4/8 | −9.9% |
| SIMT load | K=1/2/4/8 | −13.2% |
| SIMD store（aiv_mte3_time） | K=1/2/4/8 | −6.2% |
| SIMT store（matched Duration） | K=1/2/4/8 | ±5.3% |

## 3. K=16 store 复核（2026-09-28）

用与 §1.3/§1.4 相同的 matched-sink probe（扩展到 16 个独立 ptr，variant 0）做三组独立 run：

| mode | 指标 | run1 | run2 | run3 | 中位数 | 公式 cycle | 误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| SIMD scalar store | `aiv_mte3_time(us)_delta` | 0.124 us | 0.120 us | 0.122 us | 0.122 us / 219.6 cyc | 170.1 | **+29.1%** |
| SIMT scalar store | matched-sink `Duration(us)_delta` | 0.150 us | 0.159 us | 0.157 us | 0.157 us / 282.6 cyc | 215.6 | **+31.1%** |

结论：

1. **方向确认**：K=16 时按各自指标，SIMT store 的中位数 157 ns 确实高于 SIMD MTE3 的 122 ns；
   K≤8 时 SIMT（如 K=8 61.5 ns）快于 SIMD（74.5 ns），交叉点在 K≈12–14。
   因此 `subsequent_simt=12.348 > subsequent_simd=4.5` 不是符号或拟合错误。
2. **不要跨指标比较 subsequent**：SIMD 的 subsequent 是 MTE3 pipe active（异步资源占用）边际；
   SIMT 的 subsequent 是 matched-sink `Duration`（含 SIMT_STG write/response 窗口）边际，两者不是同一资源域。
3. K=16 超出 K≤8 拟合范围：本次两 mode 都比公式高约 +29%/+31%（同 run 的 K=8 也有约 +9%~+24% 的环境漂移），
   而 target 只用 K=1 store，因此**不重拟合 target-used 系数**；K=16 只作为“方向/外推边界”证据记录。

> 原始 JSON：`triton_probe/k16_results/k16_run1.json`、`k16_repeat1.json`、`k16_repeat2.json`；扩展探针：`triton_probe/store_probe_variants_k16.py`。

## 4. 使用边界

0. **指标选择**：
   - load：`Duration(us)_delta`，SIMD/SIMT 都给出稳定边际；
   - SIMT store：matched-sink `Duration(us)_delta`；
   - SIMD store：`aiv_mte3_time(us)_delta`，表示 MTE3 pipe active 资源占用（不含 scalar→UB、等 flag、write ack 完成语义）；
   - readback/barrier 变体只能用于暴露异步完成/可见性，不进入 store resource 公式。
1. formula cycle 是 raw CAModel/core cycle（1.8 GHz）；profile 落盘乘 `988.9/1800 = 0.5493889`。
2. 旧 2026-09-27 store K=2/K=4 的负值/噪声是 probe baseline 不对齐导致的，已有修正；
   §0 / §1.3 / §1.4 的结果以 2026-09-28 retest 为准。
3. SIMT K=8 的原 229 ns run 已判定为运行状态异常并排除；fresh 重跑与 matched 探针一致。
4. indirect / same-line / 非连续地址不在本轮公式范围内。
5. **SIMD store 语义（已决）**：`ScalarStore` 表示 store 资源占用/吞吐，**不表示完成/可见延迟**。
   - 目标 6 kernel 中只有 wgrad 的 `tl.store(wgrad, out)`，且 store 结果不被消费；用 `aiv_mte3_time`
     的资源占用窗口与该场景一致。
   - readback/barrier 变体测到的是 producer→consumer 可见性/同步（SIMD K=1 ~96 ns、SIMT K=1 ~78.5 ns，
     见 `store_recheck_results/sync_recheck_variant12.json`），应归入 dependency/synchronization/scope-handoff，
     而不是加到 store resource 上；否则会对 target 的最后一个 dead store 双重收费。
   - 若未来 kernel 读回/依赖 store 结果，需要单独增加 completion/visibility 项，不能复用当前 SIMD store 系数。

## 5. 文件
```text
triton_probe/profiler_pair_probe.py            # 2026-09-27 multi-ptr load/store
triton_probe/store_probe_variants.py           # 2026-09-28 store matched-sink / sync variants
triton_probe/run_store_variants.sh             # tmux 运行脚本
triton_probe/store_sync_results/               # 第一轮 retest json/csv/log（含异常 K=8=229 ns）
triton_probe/store_recheck_results/            # fresh 重跑：SIMT/SIMD K=8 异常复核
triton_probe/summarize_board_k_sweep.py        # 生成下面的对比表
triton_probe/results/scalar_k_board_summary.csv/.md
服务器原始数据：
  ~/scalar_store_sync_probe/results/           # 第一轮 matched-sink/sync 变体
  ~/store_recheck_all{,2}/                     # SIMT fresh rerun
  ~/store_recheck_all_simd{,2}/                # SIMD fresh rerun
  ~/store_recheck_k8_matched/                  # matched K=8 复核
```
