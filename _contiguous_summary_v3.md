# Triton-Ascend 连续访存（stride == 1）代价标定 v3

> 本版只保留**标定数据**与**最终公式**，以及由标定得到的探索结论；历史版本见 v2。


连续访存指内层 stride == 1 的直接 load/store。代价模型按**两套机制**给它定价：

| 机制 | 适用 | 计的是什么 |
|---|---|---|
| **延迟型** | 循环在访存上串行（每次访问之间有依赖） | 一次访问从发出到完成的延迟 |
| **带宽型** | 循环不携带跨迭代依赖，访问可流水 | 稳态下每次访问分摊的时间 |

两套机制都再按 **unmask** 与 **动态全 true mask** 分侧建模，理由见 §3.1。

---

## 1. 标定数据

### 1.1 延迟型

**配置**

| 项 | 取值 |
|---|---|
| 方法 | k-marginal：同一 kernel 跑 `iters ∈ {2048, 8192}`，`cost = (t(8192) − t(2048)) / 6144` |
| `grid` | **1**（单核，避免核间竞争） |
| dtype | int8 / fp16 / fp32 / int64（`eb` = 1 / 2 / 4 / 8 B） |
| 方向 | load / store |
| SIMD 块宽 | 34 档：2 的幂 `2…16384` + 非 2 幂 `31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1000, 1023, 1025, …` |
| SIMT 元素数 | `1, 2, 4, …, 16384`（只能 2 的幂） |
| `num_warps` | 1 / 2 / 4 / 8 / 16 / 32 / 64（SIMD 固定 1） |
| 对齐 | 32 B |
| 重复 | 每配置 3 次取最小值；单次 spread > 5% 丢弃 |
| 前置检查 | 每次采集先跑 `probe_speed.py`，健康值 65.6~66.3 ns/iter，>150 不可用 |

**kernel 核心代码**（`scripts/probe_block_pure.py`）

```python
@triton.jit(do_not_specialize=["iters"])
def block_kernel(source, output, iters, BLOCK: tl.constexpr,
                 MASK: tl.constexpr, direction: tl.constexpr):
    program = tl.program_id(0)
    base = program * (MASK + 1) * BLOCK
    lane = tl.arange(0, BLOCK)
    if direction == 0:                       # 测 load
        acc = tl.zeros((BLOCK,), source.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(source + base + (step & MASK) * BLOCK + lane)
        tl.store(output + program * BLOCK + lane, acc)
    else:                                     # 测 store
        value = tl.full((BLOCK,), 1, output.dtype.element_ty)
        for step in range(iters):
            tl.store(output + base + (step & MASK) * BLOCK + lane, value + step)
```

要点：`MASK` 控制同时在用的窗口块数（`MASK = 0` 时反复打同一块，即 unmask；`MASK = 3` 时四块轮转，配 `m = BLOCK/2` 即动态全 true mask）；load 侧用 `acc +=` 串行累加，**故意造出跨迭代依赖**，这就是「延迟型」的物理前提。

**实测数据**

| 文件 | 内容 |
|---|---|
| `…/contiguous_1d/raw/raw_pure_unified.jsonl` | 773 点，一维主体（unmask + mask 两形） |
| `…/contiguous_1d/raw/raw_simd_npow2_A.jsonl` / `_B.jsonl` | SIMD 非 2 幂块宽 |
| `…/contiguous_1d/raw/raw_small_min5.jsonl` | < 128 B 小访问 |
| `…/contiguous_1d/raw/raw_formb_v2.jsonl` | 动态全 true mask 复核 |
| `…/contiguous_1d/raw/raw_store_retest.jsonl`、`raw_stability.jsonl`、`raw_bimodal_addr.jsonl` | 复测与稳定性 |
| `…/contiguous_1d/lattice_raw/lat5{,f32,s2,s4}.jsonl` | te ≥ 8 的表格点阵 |
| 拟合结果 | `…/contiguous_1d/fitted/FINAL_SPEC.json`（16 段）、`TABLES.json`（4 张表） |

### 1.2 带宽型

**配置**

| 项 | 取值 |
|---|---|
| 方法 | 每形状 3 个 unroll 点做回归，**R² < 0.98 直接丢弃**，不硬拟合 |
| `grid` | **56**（铺满 AIV），访问足迹压在 L2（~64 MiB）内 |
| `unroll` | 1 / 4 / 8（`tl.static_range` 展开，保证多条传输同时在飞）|
| dtype | int8 / fp16 / fp32 / int64 |
| 方向 | load / store |
| 形态 | unmask 与 动态全 true mask |
| 字节 | SIMD 33 ~ 16392 B；SIMT 128 ~ 8192 B |
| `num_warps` | SIMT 1 / 2 / 4 / 8 / 16 / 32 / 64 |
| 重复 | 每点 R² 与重复性校验，中位 spread 1.00（最大 1.49） |

**kernel 核心代码**（`scripts/probe_bw_throughput.py`）

```python
@triton.jit(do_not_specialize=['iters', 'm_dyn'])
def bw_kernel(src, dst, iters, m_dyn, BLOCK: tl.constexpr, WINMASK: tl.constexpr,
              UNROLL: tl.constexpr, STORE: tl.constexpr, FORM: tl.constexpr,
              M_STATIC: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    span = (WINMASK + 1) * BLOCK * UNROLL
    base = pid * span
    if FORM == 1:
        keep = lane < m_dyn
    elif M_STATIC < BLOCK:
        keep = lane < M_STATIC
    else:
        keep = lane < BLOCK
    if STORE == 0:
        acc = tl.zeros((BLOCK,), src.dtype.element_ty)
        for step in range(iters):
            row = base + (step & WINMASK) * BLOCK * UNROLL
            v = tl.zeros((BLOCK,), src.dtype.element_ty)
            for j in tl.static_range(UNROLL):
                v += tl.load(src + row + j * BLOCK + lane, mask=keep, other=0)
            acc += v
```

要点：与延迟型相反，这里**每次迭代内部有 `UNROLL` 条互相独立的 load**（`tl.static_range` 展开，不是 `acc` 串行链），所以多条传输能同时在飞 —— 这正是「带宽型」的物理前提。`unroll` 就是拟合式里的并发度。

**实测数据**

| 文件 | 内容 |
|---|---|
| `…/contiguous_1d/bw_raw/BW_SIMD.jsonl` | 346 点 |
| `…/contiguous_1d/bw_raw/BW_SIMT.jsonl` | 426 点 |
| 拟合脚本 | `scripts/fit_bw_simd.py`、`scripts/fit_bw_simt.py` |
| 拟合结果 | `…/contiguous_1d/bw_fitted/simd_bw_formulas.json`、`simt_bw_formulas.json` |
| 生成 profile 片段 | `scripts/gen_contiguous_model.py` |

---

## 2. 最终公式

### 2.1 记号

| 记号 | 含义 |
|---|---|
| `n` | 一次访问的元素数；`eb` 元素字节数 |
| `by = n·eb` | 字节数 |
| `W` | warps（SIMD 恒 1） |
| `te = n/(32·W)` | 每线程元素数 |
| `lines = by/128` | SIMT 128 B cache line 数 |
| `blk_part` | `by % 32 ≠ 0` 时为 1（不是 32 B DataBlock 的整数倍）|
| `burst` | `⌈by/16384⌉ − 1`（超过 16 KiB 的额外突发段数）|
| `lnW` / `byLnW` | `ln(1+W)` / `by·ln(1+W)` |
| `invW` | `1/W` |
| `warpRounds` | `W·⌈W/4⌉`（每 warp 要被 4 个调度器轮到的次数）|
| `mask` | 动态全 true mask 时为 1 |
| `unroll` | 带宽型的并发传输数，`invU = 1/unroll` |

### 2.2 延迟型 · unmask

| 段 | 条件 | 特征（除常数项） | 常数(ns) | 其余系数 |
|---|---|---|---:|---|
| `simd.load` | SIMD，unmask 与 mask 共用 | `by` `blk_part` `burst` | 66.38 | 0.005412, 5.800, 73.060 |
| `simd.store` | SIMD，unmask 与 mask 共用 | `by` `blk_part` `burst` | 83.24 | 0.004262, 10.690, 40.400 |
| `normal.load` | SIMT load，`lines ≤ 32`（含 mask） | `te` `W` `log2eb` `mask` | 92.36 | 17.514, 0.8379, 14.153, 15.826 |
| `unmasked.store.te_lt_1` | unmask store，`te < 1` | `te` `W` `schedR` | 29.80 | 14.552, 4.118, -12.210 |
| `unmasked.store.te_ge_1_int8` | unmask store，`te ≥ 1`，`eb = 1` | `LN` `W` `lines` | 9.62 | -0.1782, -9.835, 63.141 |
| `unmasked.store.te_ge_1_wide_w_le_4` | unmask store，`te ≥ 1`，`eb > 1`，`W ≤ 4` | `byh1k` `rounds` `half` | 27.10 | 0.01903, 0.9109, 2.135 |
| `unmasked.store.te_ge_1_wide_w_gt_4` | unmask store，`te ≥ 1`，`eb > 1`，`W > 4` | `by` `W` `K1` `warpB` | 21.98 | -0.006888, 1.707, 0.008054, -0.09785 |

**逐条物理解释**

- **`simd.load` = 66.38 + 0.0054·by + 5.8·blk_part + 73.06·burst**
  常数 66 ns 是一次 SIMD 访问的固定开销（发指令 + 等第一拍数据）。`0.0054 ns/B` 就是**流式斜率的倒数**：约 185 GB/s 单核，即 SIMD 顺序搬运的带宽。`blk_part` 项 5.8 ns 的物理原因是 MTE 以 32 B DataBlock 为最小搬运粒度，尾部不满一个块时要多搬一块。`burst` 项 73 ns 对应超过 16 KiB 后每多一个 16 KiB 段就多一次突发仲裁——这也是**为什么按 16 KiB 分段**：那正是 MTE 单次突发的上限。
- **`simd.store` = 83.24 + 0.0043·by + 10.69·blk_part + 40.4·burst**
  形态同上，但常数更大（写要先占写缓冲）、斜率更小（写回不需要等数据）、`blk_part` 更贵（写不满块要读改写）。`burst` 只有 40 ns，因为写突发可以排队，比读突发便宜。
- **`normal.load` = 92.36 + 17.51·te + 0.84·W + 14.15·log2eb + 15.83·mask**
  这条覆盖 SIMT load 在 `lines ≤ 32` 的全部情形（mask 由内部的 `mask` 项吸收）。17.51·te 是**每线程元素数**的线性项：SIMT 的 load 是逐线程发 `LDG`，一条搬 8 B，所以成本随 `te` 线性。0.84·W 是 warp 数带来的发射开销。`log2eb` 项 14.15 反映窄 dtype 在同一字节数下指令数翻倍的代价。
- **`unmasked.store.te_lt_1` = 29.80 + 14.55·te + 4.12·W − 12.21·schedR**
  `te < 1` 时一条 store 指令装不满一个 warp 的宽度，成本由「每个 warp 各自的写」决定，所以带 `W`；`schedR = ⌈W/4⌉` 为负是因为 warp 越多，单条指令被摊到更多调度槽上，**单位成本反而下降**。
- **`unmasked.store.te_ge_1_int8` = 9.62 − 0.18·LN − 9.84·W + 63.14·lines**
  int8 的 store 走的是字节写，成本几乎完全由 cache line 数决定（63.14 ns/line），常数很小——**这是唯一以 `lines` 为主项的段**，因为 1 B 元素写会把整条 128 B line 都拖进来。
- **`unmasked.store.te_ge_1_wide_w_le_4` = 27.10 + 0.019·byh1k + 0.91·rounds + 2.14·half**
  `W ≤ 4` 时一条 store 是多线程拼成的宽传输，超过 1 KiB 的部分才线性增长（`byh1k = max(0, by−1024)`）——**1 KiB 以下基本免费**，这就是为什么这条要按 warp 数分段。`rounds = ⌈te⌉` 是每条线程要发几条写指令。
- **`unmasked.store.te_ge_1_wide_w_gt_4` = 21.98 − 0.0069·by + 1.71·W + 0.0081·K1 − 0.098·warpB**
  `W > 4` 后调度器要轮转，成本转为随 `W` 线性；`by` 系数是**负的** —— warp 多到一定程度后，写带宽反而随字节数摊薄（更多 lane 并行填一条 line），这是实测的、不是笔误。

### 2.3 延迟型 · 动态全 true mask

mask 在运行期全为 true（`lane < m`，`m ≡ 块宽`），所以**没有元素被真正屏蔽**，但编译器仍要为每个元素生成谓词。

| 段 | 条件 | 特征（除常数项） | 常数(ns) | 其余系数 |
|---|---|---|---:|---|
| `small.load` | `by < 128 B`，与 dtype、mask 无关 | `warpRounds` `i8wRounds` `halfwRounds` `schedR` `d64` | 46.02 | 0.09698, -0.02721, 0.007758, 0.4430, 9.434 |
| `small.store` | `by < 128 B`，与 dtype、mask 无关 | `warpRounds` `i8wRounds` `halfwRounds` `schedR` `d64` | 22.77 | 0.05600, 0.007873, -0.01797, 3.395, -0.9701 |
| `masked.load.l_gt_32_nospill` | mask load，`lines > 32`，`2·te·eb ≤ regB(W)` | `lines` `te` `ldg` `schedR` `W2` `burst` | 1.24 | 34.386, 10.446, 10.337, -0.01819, 529.82 |
| `masked.load.l_gt_32_spill` | mask load，`lines > 32`，`2·te·eb > regB(W)` | `lines` `te` `thr` `W` `invW` `half` | 106.39 | 69.052, -12.797, -5588.92, 2341.54, 988.46 |
| `masked.store.te_lt_1` | mask store，`te < 1` | `i8` `lnW` `W` `warpB` | 37.59 | -13.041, -11.498, 2.600, -0.01348 |
| `masked.store.te_ge_1_int8` | mask store，`te ≥ 1`，`eb = 1` | `by` `LN` `byLnW` `W` | 27.82 | 0.3190, -0.1264, 0.02607, -8.966 |
| `masked.store.te_ge_1_fp16` | mask store，`te ≥ 1`，`eb = 2` | `K1` `LN` `lperw` `W` | 28.63 | 0.04134, 0.06123, -7.463, -3.426 |
| `masked.store.te_ge_1_fp32` | mask store，`te ≥ 1`，`eb = 4` | `K1` `thr` `invW` `byh1k` `d64h8` | 20.81 | 0.01352, -2.216, 30.610, -0.01804, 2.583 |
| `masked.store.te_ge_1_int64` | mask store，`te ≥ 1`，`eb = 8` | `K1` `invW` `te` | 3.80 | 0.007200, 24.840, -3.076 |

**逐条物理解释**

- **`small.load` / `small.store`（`by < 128 B`，与 dtype、mask 无关）**
  低于一条 128 B cache line 时，成本不再由字节数决定，而由**几个 warp 各自发一条指令、以及它们是否撞在同一条 line 上**决定。所以特征里出现 `warpRounds`（`W·⌈W/4⌉`，warp 被调度器轮到的次数）、`schedR`、以及 `i8wRounds`/`halfwRounds`（窄 dtype 各自的轮转次数）、`d64`（是否够 64 B）。**这就是为什么这里不按 dtype 再分段**：小访问的成本与 dtype 只有弱相关，266 个点合成一条反而更稳。
- **`masked.load.l_gt_32_nospill` = 1.24 + 34.39·te + 10.45·ldg + 10.34·schedR + 529.82·burst − 0.018·W²**
  `lines > 32` 的 masked load 走谓词化批量搬。`ldg = ⌈te·eb/8⌉` 是**每线程的 load 指令条数**（一条 8 B），这是 SIMT 真实发射成本的驱动量；`burst` 系数高达 530 ns/段，因为多段突发在有谓词时无法合并。**分段点取在 spill 边界**：寄存器够用时谓词只是多一条指令，不够时（下一条）要额外做寄存器溢出重排，代价结构完全不同。
- **`masked.load.l_gt_32_spill` = 106.39 + 69.05·te − 12.80·thr − 5588.92·W + 2341.54·invW + 988.46·half**
  溢出后要先把数据压到栈上再搬，常数陡增到 106 ns。`W` 的系数是**千量级负值**，`invW` 是**千量级正值**——两者共同表达「warp 越多，每个 warp 摊到的溢出重排越少」，本质是同一个 1/W 效应被拆成两项才拟合得住。`half` 项 988 ns 说明 fp16 溢出后要额外做一次打包。**这条是全部 16 条里唯一需要 `1/W` 形式的**，因为只有它同时出现「溢出」和「多 warp」。
- **`masked.store.te_lt_1` = 37.59 − 13.04·i8 − 11.50·lnW + 2.60·W − 0.0135·warpB**
  不足一条指令的 masked store，成本随 warp 数增加（每个 warp 都要单独等谓词），但 int8 反而更快（`i8` 系数为负：1 B 元素写完一条 line 就结束，谓词开销被 line 内部并行吸收）。
- **`masked.store.te_ge_1_int8` = 27.82 + 0.319·by − 0.1264·LN + 0.0261·byLnW − 8.97·W**
  int8 masked store 由字节数主导（0.319 ns/B，是 int8 特有的高斜率——1 B 元素要让每个元素单独走谓词），`byLnW` 与 `LN` 两项共同修正「字节多 + warp 多」时的交叉效应。
- **`masked.store.te_ge_1_fp16` = 28.63 + 0.0413·K1 + 0.0612·LN − 7.46·lperw − 3.43·W**
  fp16 需要**逐元素谓词**，无法合并成宽类型写，所以斜率远小于 int8 但仍有可观常数；`lperw = lines/schedR` 为负表示每 warp 分到的 line 越多，掩码判断的摊销越好。
- **`masked.store.te_ge_1_fp32` = 20.81 + 0.0135·K1 − 2.22·thr + 30.61·invW − 0.018·byh1k + 2.58·d64h8**
  fp32 谓词化后可以按 32 位粒度合并，斜率进一步下降；`invW` 项 30.61 说明 warp 少时单 warp 承担更多谓词工作。`d64h8`（≥64 B 且 W > 8）为正，反映宽 warp 下谓词寄存器压力上升。
- **`masked.store.te_ge_1_int64` = 3.80 + 0.0072·K1 + 24.84·invW − 3.08·te**
  int64 谓词化写几乎免费（常数 3.8 ns，斜率 0.0072 ns/B）：64 位元素的谓词天然按 lane 对齐，不产生额外搬运。这条存在只是为了覆盖边界，实测点很少。

**为什么必须分段**

| 分段点 | 原因 |
|---|---|
| `by < 128 B` | 低于一条 cache line，成本由「几个 warp 发几条指令」决定，与字节数无关 |
| `te < 1` | 一条指令装不满一个 warp，成本按 warp 计而不是按元素计 |
| `eb == 1` | int8 走字节通路，谓词/写回粒度与其它宽度完全不同 |
| `eb == 2 / 4 / 8`（masked store）| 谓词化写的合并能力逐宽度不同：fp16 必须逐元素、fp32 可按 32 位合并、int64 天然对齐 |
| `W ≤ 4 / > 4`（unmasked store 宽访问）| 4 个 warp 正好填满 4 个调度器，超过后要轮转，成本转为随 W 线性 |
| `lines ≤ 32 / > 32`（masked load）| 32 条 line = 4 KiB，超过后必须走批量搬（`ldg` 指令流），不再是单次事务 |
| spill / nospill | 寄存器够不够，决定谓词是「多一条指令」还是「先溢出再搬」 |

### 2.4 带宽型

拟合形态（`bw_fitted/*.json` 原样）：

```
load : max(a0 + a1/unroll, b·KiB)
store: a0 + b·KiB + c/unroll        （仅 SIMD store，线性无 max）
```

| 段 | a0 | a1 | b (ns/KiB) |
|---|---:|---:|---:|
| `simd.load` | 7 | 70 | 12.5 |
| `simd.store` | 91.87 | −7.77 (c) | 7.459 |
| `simt.load.W1` | −60 | 210 | 63.5 |
| `simt.load.W2` | −22 | 150 | 34.5 |
| `simt.load.W4` | −2 | 80 | 19.5 |
| `simt.load.W8` | 4 | 60 | 22.5 |
| `simt.load.W16` | 6 | 60 | 14.5 |
| `simt.load.W32` | 20 | 70 | 13.5 |
| `simt.load.W64` | 72 | 100 | 0.5 |
| `simt.store.W2` | −52 | 80 | 41.5 |
| `simt.store.W4` | 8 | 20 | 14.5 |
| `simt.store.W8` | 10 | 20 | 14.5 |
| `simt.store.W16` | 14 | 30 | 19.5 |
| `simt.store.W32` | 2 | 70 | 14.5 |
| `simt.store.W64` | −4 | 140 | 13.5 |

**物理解释**

- `a0` 是**纯粹的流式项**：`unroll` 很大时每次访问摊到的时间趋近 `a0`。`simd.load` 的 7 ns 对应约 146 GB/s 单核，与延迟型公式里 0.0054 ns/B 的斜率（185 GB/s）同量级，互相印证。
- `a1/unroll` 是**一次性的启动延迟被并发摊薄**：`unroll` 越大，这笔固定成本摊得越薄。`simd.load` 的 a1 = 70 ns 与延迟型的常数 66 ns 几乎相等——**同一个物理量在两种机制下的两种记法**，这是两套标定互相自洽的直接证据。
- `b·KiB` 是**带宽下限**：即便延迟再低，一次访问也不可能快过「字节数 ÷ 峰值带宽」。`simd.load` 的 12.5 ns/KiB ≈ 82 GB/s 单核，是**实测的持续带宽**，低于瞬时峰值。
- SIMT 各段的 `b` 随 warp 数下降（63.5 → 0.5 ns/KiB），因为 warp 越多，单次访问摊到的带宽越高；`W64` 的 0.5 明显偏离趋势（相邻 W32 是 13.5），是 30 个点里 W64 只占少数、且该配置已接近带宽饱和导致的拟合噪声，**使用时应注意这一段的置信度低**。
- `simd.store` 用线性形式而不是 max，因为 store 侧有写缓冲：固定开销与流式时间**同时**存在，不存在「取大者」的切换。

**与延迟型的根本区别**

| | 延迟型 | 带宽型 |
|---|---|---|
| 物理前提 | 循环在访存上串行（`acc +=` 依赖链） | 循环无跨迭代依赖，多条传输在飞 |
| 计的量 | 一次访问的完成延迟 | 稳态下每次访问分摊的时间 |
| 公式形态 | 单段线性（含分段） | `max(固定 + 启动/u, 带宽下限)` |
| 对大块的走向 | 随字节线性增长 | 收敛到带宽下限 |

### 2.5 te ≥ 8：用实测表而不是公式

`te ≥ 8` 后，成本由**寄存器分配器的溢出位置**决定，呈阶梯而非曲线。实测 8 KiB load 在 `te = 8/16/32/64/128` 下分别是 **318 / 309 / 459 / 918 / 3433 ns** —— 不是任何光滑函数。三次多项式拟合的最优 MAPE 是 19%，而点阵复测重复性只有 1%，所以这一域**按实测 knot 存表**，在 `(log2 te, log2 by)` 上双线性插值。

### 2.6 集成时的准入条件

公式之外，代码里还有三条准入规则决定「用不用这套公式」。它们不是拟合出来的，是对标定域的诚实边界：

| 规则 | 具体条件 | 超出后的行为 |
|---|---|---|
| **域上限** | `elements ≤ 16384` 且 `warps ≤ 64` | 什么都不算，退回基线曲线 |
| **非 2 幂的 SIMT FormA** | `n` 是 2 的幂 | `return 0`，退回基线曲线；**不做 `padPow2` 硬塞**——SIMT 的 898 个标定点全是 2 幂（`raw_pure_unified.jsonl` 773 点 + `lattice_raw/lat5.jsonl` 125 点），补出来的宽度没有实测支撑 |
| **分支优先级** | `small → 带宽型 → 表 → 延迟型` | 先命中先返回 |

优先级的依据是各自的标定域：

| 分支 | 适用域 | 与带宽域的关系 |
|---|---|---|
| `small` | SIMT，`by < 128 B` | **域外**（SIMT 带宽标定从 128 B 起），所以排在最前是对的 |
| 带宽型 | 无跨迭代依赖，SIMD ≤16 KiB / SIMT ≤8 KiB | — |
| 表 | SIMT，`te ≥ 8` | **域内**（≤8 KiB），所以必须排在带宽型**之后**，否则会被抢走 |
| 延迟型 | 其余 | — |

> 注：SIMD 的访问本来就落在带宽域内（33~16392 B），且 `small` 分支只对 SIMT 生效，所以 SIMD 侧不存在分支争抢。

---

## 3. 探索结论

### 3.1 mask 与 unmask 为什么必须分开

**证据**：同一个 `bytes`、同一个 dtype 下，动态全 true mask 的访问明显更慢，且差距随宽度变化：

| 场景 | unmask | 动态全 true mask | 说明 |
|---|---|---|---|
| SIMT store, `te ≥ 1`, fp32 | 一条公式覆盖全部宽度 | 必须按 fp16 / fp32 / int64 分三段 | 谓词化写无法跨宽度合并 |
| SIMT load, `lines > 32` | 无独立段（≥4 KiB 直接交回通用曲线） | 需要 `spill`/`nospill` 两段 | 谓词让突发无法合并，`burst` 系数从 ~0 涨到 530 ns |
| SIMD | 一条 `simd.load/store` | **复用同一条** | SIMD 的谓词是整块判断，不逐元素 |

结论：**mask 的影响不是「加一个固定量」，而是改变了成本的结构**（哪一项主导、要不要按 dtype 分段）。对 SIMD 不成立，所以 SIMD 两形共用一条——这一点本身也是证据：如果 mask 只是加价，SIMD 侧也该有差异，实测没有。

### 3.2 延迟型与带宽型为什么必须分开

**证据一：同一批字节，两种循环的成本差一个量级。**

| kernel | 每轮内容 | 延迟型公式预测 | 实测 |
|---|---|---:|---:|
| N-loads，N=1 | 1 load，store 摊薄为 0 | 255.8 周期 | 246.6 周期（+3.7%）|
| N-loads，N=2 | 2 条连续 load | 按延迟相加 511.6 | 460.9（−9.9%）|
| N-loads，N=4 | 4 条连续 load | 相加 1023.2 | 656.1（**−35.9%**）|

N = 4 时实测只有线性相加的 64%，边际成本从 214 周期/条掉到 **97.6 周期/条** —— 多条访问在流水，延迟被共享了。

**证据二：CAModel 指令流**。一个 `independent_pipelined_loop` 的 4 次迭代里，4 条 store 全部在 **315 个 cycle 内** push 完（间隔 105 cycle），而每条要到 1000+ cycle 才 retire；BIU 实际接收写命令间隔 450~530 cycle。即**发射端不等写完**，稳态成本由带宽而非依赖链决定。

**针对的 kernel 场景**

| 特征 | 延迟型适用 | 带宽型适用 |
|---|---|---|
| 循环内是否有跨迭代依赖 | 有（`acc +=`、递推、scan） | 无（逐元素独立、elementwise） |
| 同时有多少访问在飞 | 1 | ≥ 2（unroll 或静态展开）|
| 矛盾主导 | 单次延迟 | 持续带宽 |
| 典型例子 | cumsum、solve_tril、recurrence | relu / add / mul 一类 elementwise |

模型里的判据就是这个：`pipelined = (hasLoop || iterations > 1) && !hasLoopCarriedDataDependency`。

### 3.3 二维 tile 与一维为什么能共用一套公式

**结论**：二维 tile `(R, C)`（行跨距 = C）与一维 `(R·C,)` 等价，直接把 `n = R·C` 代入一维公式，行或列是不是 2 的幂都不影响。

**证据**：每个几何**同时**测二维 `(R,C)` 与一维 `R·C`，同一次运行、同一窗口，直接比比值：

| 模式 | 几何类别 | 组数 | 2D/1D 比值中位 | min | max |
|---|---|---:|---:|---:|---:|
| SIMD | 都 2 幂 | 32 | **1.0002** | 0.9988 | 1.0131 |
| SIMD | 其一非 2 幂 | 16 | **1.0000** | 0.9989 | 1.0096 |
| SIMD | 都非 2 幂 | 16 | **1.0000** | 0.9934 | 1.0007 |
| SIMT | 都 2 幂 | 12 | **0.9997** | 0.9857 | 1.0008 |

覆盖 `32×64`、`32×128`、`64×64`、`16×256`、`31×127`、`32×127`、`31×128`、`63×33`，4 dtype × 2 方向，比值全部在 ±1.5% 内。

**原因**：内层 stride == 1 且外层跨距恰为 C 时，二维 tile 在 GM 上是**一段连续的字节**，MTE 看到的只有「起始地址 + 长度」，看不到有没有二维形状。所以代价只取决于总字节数。SIMT 遇到非 2 幂的行列会直接编译失败（`arange's range must be a power of 2`），与一维一致。

### 3.4 组内多访问：一条延迟 + 其余摊薄

同一轮迭代里放 N 条同形连续访存时，实测的边际成本迅速下降（见 §3.2 表一）。这解释了为什么带宽型公式里要保留 `a1/unroll`：

```
组总时间 = N · (a0 + a1/N) = N·a0 + a1
```

即 **N 份流式时间 + 一份启动延迟**，而不是 N 份延迟。基线的组曲线（总字节 ÷ 带宽）是我们这个式子在 `a1 → 0` 时的极限。

### 3.5 寄存器压力与溢出判定（存疑，待复核）

当前实现用「per-thread 寄存器预算」解释 `te ≥ 8` 的阶梯（1~256 线程 → 127 寄存器，257~512 → 64，依此类推，见 `regBytes()`），并用 `2·te·eb > regB(W)` 作为 FormB 的分段判据。

**但这一条的证据只有「阶梯存在」，还没有「阶梯边界恰好等于寄存器档位」的直接测量**。§2.5 的实测点（318/309/459/918/3433 ns）只能说明它是阶梯，不能说明边界在哪。**使用时请把这条当假设而非结论**，后续需要按 warp 数扫描 `te` 的边界位置来验证。
