# te≥8 SIMT 流式访存标定 v2（2026-11-08，Ascend 950PR，2 号卡）

替换 v1 的 4 张 (te, by) 双线性结点表（load.eb2 / load.eb4 / store.eb2 / store.eb4）。

## v1 的测量缺陷（"阶梯"的来源）

v1 探针（`scripts/probe_block_pure.py`）的 load 循环体是 `acc += load(...)`：

- **累加器自身就是 te 宽的寄存器向量**。寄存器预算随线程数变化
  （≤256 线程 127 regs、512→64、1024→32、2048→16），acc+加载临时量
  在 te×eb ≥ 128B（W≤8）即破预算，循环每轮迭代都 spill——v1 表中
  te=64→128 的 5-10 倍"阶梯"和 te=64 行的非单调都是**探针的寄存器压力，
  不是访存系统的属性**；
- 双趟斜率还把逐元素 add 的吞吐（47-104ns/op）折进了"访存"成本。

## v2 方法：配对对照

- A：`acc += tl.load(src + base + (step & MASK)·BLOCK + lane)`（同 v1）
- B：`acc += value + tl.load(scalars + (step & 7))`——同形累加器 + 不可提升
  （LICM/闭式化都挡住）的每迭代标量工作，仅去掉块加载
- **cost = slope(A) − slope(B)**，单 profiler 会话 [A,B]×4 × 双趟（1024→4096）

store 无累加器，直接斜率（背靠背 store 环维持的正是排空率语义）。

## 拟合域与公式（8/8 族 4.6-12.5% MAPE）

拟合剔除（公式从干净区光滑外推，实测点保留在 raw 供对照）：
- load 的双内核溢出格：`te·eb/2 > budget(W)`（budget: W≤8→120, 16→60, 32→28）
- W=64 全部（16-reg 边际区，4-reg 向量也挤不下）
- int8 te≥64 / fp16 te=64（字节路径悬崖/边际压力，属另一机制，外推低估已记录）

| 族 | MAPE | 族 | MAPE |
|---|---:|---|---:|
| load.eb1 | 11.5% | store.eb1 | 12.5% |
| load.eb2 | 8.5% | store.eb2 | 5.6% |
| load.eb4 | 4.6% | store.eb4 | 7.8% |
| load.eb8 | 4.8% | store.eb8 | 5.2% |

覆盖从旧表的 2 个 dtype 扩到全部 4 个（eb1/eb8 旧版为拒绝→分析式回落）。
干净区物理：`ns ≈ 150 + 0.0165·by + 小的 te/W 项`（fp32），即流水型 load =
延迟地板 + LSU 字节吞吐，与旧表干净区相差约一个 B 斜率（47-104ns 的加法开销）。

- 复现链：`scripts/probe_te8_stream.py`（采集）→ `scripts/fit_te8.py`（重拟合）
- profile 键：`contiguous_ldst.forms.te8_formulas`（先查 `<dir>.eb<n>` 再查合并 `<dir>`；
  C++ 同时保留旧结点表解析作旧 profile 回退）

## 2026-11-08 状态更新：公式已拟合，撤出激活 profile

配对对照修复了 v1 探针的累加器伪影（本文档上文），8/8 族 4.6-12.5% MAPE 成立。
但 100-case 验证显示：流式斜率对**循环稳态**正确，对语料 kernel 的实际访问
（含启动、依赖链、与散射 store 交错）系统性偏低——rowred −38.8→−49.2、
layer_norm/cumprod/fbgemm 恶化。`forms.te8_formulas` 已从激活 profile 移除
（te≥8 回落段公式），数据/脚本/C++ 支持保留：C++ 对旧 profile 的结点表与
te8_formulas 均可回退解析。重新启用前需带 kernel 上下文的重标定
（依赖链长度、与其它访存的交错），或明确其适用域为纯流式循环体。
