# 非 2 的幂尾轴：机制、证据、复现

本文回答一个问题：**为什么尾轴元素数不是 2 的幂时，连续访存会慢 40~90%？**
结论来自三层证据（ttadapter IR / hivm.opt IR / CAModel 逐指令），并给出了可复现的命令。
文中每个数字都标了出处；推断与实测分开写。

---

## 0. 一句话结论

尾轴不是 2 的幂 → 编译器把 UB 缓冲的行宽补到下一个 2 的幂 → 掩码不再可折叠 →
**整块 UB 缓冲要先被广播清零，且这次广播与 DMA 之间有跨流水同步（串行化）**，
同时 DMA 从「1 个 16384 B 的 burst」变成「32 个 508 B 的 burst」。

代价分解（都有实测支撑）：

    多付的 cycle ≈ 0.039 cyc/补齐元素  ×  补齐后的元素数      ← 主项，约占 3/4
                 + 2 cyc/burst          ×  burst 数           ← 次项，约占 1/4

**主项是填充与它的跨流水同步，不是搬运本身；次项才是 burst 碎片。**

---

## 1. 现象：三个可直接对照的数

同一 kernel（32 行 fp32、simd、带 extent 掩码、GRID=1），只改尾轴长度：

| 几何 | 实测 cyc/op | 每 KB | 相对对齐 |
|---|---:|---:|---:|
| 32×128（2 的幂） | 360.8 | 22.6 | — |
| 32×127（列尾） | 577.8 | 36.4 | **+61%** |
| 32×120（列尾，32 B 对齐） | 552.6 | 36.8 | **+63%** |
| 32×160（列尾，**128 B 完全对齐**） | 856.9 | 42.8 | **+89%** |
| 32×256（2 的幂） | 719.5 | 22.5 | — |
| 31×128（行尾） | 509.8 | 32.9 | **+46%** |

数据出处：`raw/raw_align_grid.jsonl`（每几何 3 个重复），归约脚本 `fit/fit_final.py`。

**注意 C=160**：它的行是 640 B = 5×128 B，128 B 线完全对齐，却比 C=127 **更慢**。
所以触发条件不是「字节对齐」，而是「元素数是不是 2 的幂」。

---

## 2. 触发条件与 lowering 的对应

| 几何 | 缓冲行宽 | 目的 stride | subview 数 | 拷贝能否折叠成 1D |
|---|---:|---:|---:|---|
| C=128（2 的幂） | 128 | identity | **0** | 能 |
| C=127 | 128（补 1） | strided[128,1] | 3 | 不能 |
| C=160 | **256**（补 96） | strided[256,1] | 3 | 不能 |
| C=256（2 的幂） | 256 | identity | **0** | 能 |
| R=31（行尾，C=128） | 128（行不补） | strided[128,1] | 3 | 能（stride==行长度） |
| R=33（行尾，C=128） | 128（但 rpad=64 行） | strided[128,1] | 3 | 能 |

关键区分：**列尾破坏 stride==行长度，行尾不破坏**。这决定了后文的成本分解。

---

## 3. 三层证据，逐层说明能看什么、看不到什么

### 3.1 ttadapter（能看结构，不能看指令）

生成：

    python3 scripts/probe_alignment_ir.py --rows 32 --cols 127 --dtype fp32 \
        --mode simd --masked 1 --dump-dir out/alignir/c127

产物：`out/alignir/c127/kernel.ttadapter`。C=127 的关键三行：

    %value_1  = memref.alloc() : memref<32x128xf32>                    ← 缓冲行宽 128（真实 127）
    %value_2  = memref.subview %value_0[0,0] [32,127] [1,1]             ← subview ①
    %value_3  = memref.subview %value_1[0,0] [32,127] [1,1]             ← subview ②，目的 stride 128
    memref.copy %value_2, %value_3

**坑**：`memref.copy` 那一行本身**不包含** subview 的信息。subview 数是靠
`grep -c memref.subview` 数的（C=127 是 3，C=128 是 0）。

**这一层看不到**：DMA 用几条指令、每条搬多少字节。

### 3.2 hivm.opt（能看 op 形态与流水同步，仍看不到指令）

生成（**必须带 --target，否则默认 Ascend910B1**，下一步 hivmc 会 legalize 失败）：

    ~/bishengir-install/bin/bishengir-compile \
        <TRITON_CACHE>/.../align_kernel.bcmlir \
        --target=Ascend950PR_9579 --save-temps=/tmp/savetemps_c127

产物：`/tmp/savetemps_c127/module.hivm.opt.mlir`。这是 **convert-to-hivm + hivm 优化之后、
交给 hivmc 之前**的那一份（bishengir-compile 自己会打印 `Executing: hivmc …/module.hivm.opt.mlir`）。

C=128 只有 3 个 op：

    %collapse = memref.collapse_shape %src [[0,1]] : → memref<4096xf32, strided<[1]>>
    hivm.hir.load  ins(%collapse) outs(%ub) eviction_policy = <EvictFirst>
    hivm.hir.store ins(%ub) outs(%collapse)

C=127 多出四处：

    %0 = hivm.hir.pointer_cast : memref<32x128xf32, ub>              ← 缓冲被补
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse(%ub))               ← 整块广播 0
    hivm.hir.set_flag [PIPE_V, PIPE_MTE2, EVENT_ID0]                 ← 跨流水同步
    hivm.hir.wait_flag[PIPE_V, PIPE_MTE2, EVENT_ID0]
    hivm.hir.load … pad_mode = <PadValue> pad_value = 0              ← load 改 pad 模式

**这一层看不到**：`hivm.hir.load` 展开成几条机器指令。

### 3.3 CAModel（能看到 burst 参数——这才是答案）

跑法见 §6。产物：`~/align_camodel2/<case>/OPPROF_*/dump/core0.mte_log.dump`。

C=128：

    MTE2 instr_id: 22, burst_num: 1,  burst = 16384, src_stride: 16384, dst_stride: 16384, pad: (0,0)

C=127：

    MTE2 instr_id: 22, burst_num: 32, burst = 508, src_stride: 508, dst_stride: 512, pad: (0,0)
    MTE3 instr_id: 24, burst_num: 32, burst = 508, src_stride: 512, dst_stride: 508, pad: (0,0)

结论：

* **指令数没变**（都是 1 条 MTE2 + 1 条 MTE3）；
* **变的是同一条指令内部的 burst 结构**：`1 × 16384 B` → `32 × 508 B`；
* 目的 stride 512 ≠ 源 stride 508，那 4 B 就是 padding。

总线层另外能看到碎片：C=128 的请求只有 512 B 与 128 B 两种；C=127 出现 **66 条 124 B**
（508 B 落在 128 B 线上被切开后剩下的尾巴）。

---

## 4. 成本分解：两条独立路径给出同一个系数

把「多付」按**补齐后的元素数**折算：

| 几何 | 实测 | 按对齐速率应付 | 多付 | 补齐后缓冲 | 多付/补齐元素 |
|---|---:|---:|---:|---:|---:|
| 32×128 对齐 | 360.8 | 360.8 | — | 无 fill | — |
| **31×128 行尾** | 509.8 | 350.3 | ≈160 | 4096 | **0.039** |
| 32×127 列尾 | 577.8 | 358.8 | ≈218 | 4096 | 0.053 |
| 32×160 列尾 | 856.9 | 452.0 | ≈405 | 8192 | 0.049 |
| 33×128 行尾 | 771.2 | 382.0 | ≈389 | 8192 | 0.047 |

取第一行（**没有 stride 失配**，所以多付的部分只能来自填充与同步）：

    31×128 行尾：160 / 4096 = 0.039 cyc/补齐元素

而拟合（完全不看 IR、只看成本曲线的 `fit/fit_families.py`）给出的谓词项是
**0.0400 cyc/元素**。两条路径独立得出同一个数。

按 0.039 拆开剩下三个几何：

| 几何 | 多付 | 填充部分 | 余下（burst） | burst 数 |
|---|---:|---:|---:|---:|
| 31×128（无 stride 失配） | 160 | **160** | **0** | 1（推断） |
| 32×127 | 218 | 160 | **58** | 32 |
| 32×160 | 405 | 320 | **85** | 32 |
| 33×128 | 389 | 320 | **69** | 32 |

余下部分 = 58~85 拍，而三者的 burst 数都是 32 → **约 2 cyc/burst**。

### 为什么行尾（R=31）也慢

R=31 的 stride 没有失配（`strided<[128,1]>`，行 stride == 行长度），拷贝仍是 1D 连续，
但它有掩码（rpad=32 > R=31），于是同样要 `vbrc` 整块清零 + 跨流水同步。**它的 160 拍全部来自这一项**。

这正好从反面证明：**填充/同步是主项（约 3/4），burst 碎片是次项（约 1/4）**。

---

## 5. store 为什么也碎片化却不慢

C=127 的 **store 同样**退化成 `burst_num: 32, burst = 508`，但实测 store 的每 KB 成本
在所有几何上都是 12.0~15.5，列尾 120 甚至是 14.2（对齐 14.3）。

也就是说：**碎片化在读写两侧都发生，但只有读方向（MTE2）对 burst 长度敏感**。
所以「这个现象只影响 load」的准确含义是「只有读方向的成本跟着走」，
而不是「写方向的 lowering 没有变化」。

---

## 6. 可运行复现

前置：`source ~/env_ascend.sh`；只用 `~/bishengir-install/bin/hivmc-a5`，**不要用 CANN 里的 hivmc**。

### 6.1 IR（不需要设备，最快）

    python3 scripts/probe_alignment_ir.py --rows 32 --cols 128 --dtype fp32 \
        --mode simd --masked 1 --dump-dir out/alignir/c128
    python3 scripts/probe_alignment_ir.py --rows 32 --cols 127 --dtype fp32 \
        --mode simd --masked 1 --dump-dir out/alignir/c127
    grep -c memref.subview out/alignir/c128/kernel.ttadapter   # 0
    grep -c memref.subview out/alignir/c127/kernel.ttadapter   # 3

### 6.2 hivm.opt（需要 --target）

    ~/bishengir-install/bin/bishengir-compile \
        <TRITON_CACHE_DIR>/<hash>/align_kernel.bcmlir \
        --target=Ascend950PR_9579 --save-temps=/tmp/savetemps_c127
    diff /tmp/savetemps_c128/module.hivm.opt.mlir /tmp/savetemps_c127/module.hivm.opt.mlir

### 6.3 CAModel（能拿到 burst 参数）

    bash scripts/run_align_camodel.sh 32 127

然后读：

    OPPROF=$(ls -d ~/align_camodel/<tag>/OPPROF_* | head -1)
    grep -E "burst_num|MOV_SRC_TO_DST" $OPPROF/dump/core0.mte_log.dump
    grep -oE "size: [0-9]+" $OPPROF/dump/core0.mte_log.dump | sort | uniq -c | sort -rn

逐指令 cycle 在 `$OPPROF/simulator/core0.veccore0/core0.veccore0_instr_exe.csv` 的 `cycles` 列。

---

## 7. 坑（按踩到的顺序）

| 坑 | 现象 | 处置 |
|---|---|---|
| **污染 LD_LIBRARY_PATH** | 把 `SIMLIB/BISHLIB` 塞进 `LD_LIBRARY_PATH` 后，Triton 的 ascend driver 现编 `npu_utils` 扩展失败（它的编译命令里带 `/data/pri/cann-9.1.0`、`/usr/local/lib/python3.11/...` 这类别的机器的硬编码路径）。**kernel 根本不会在仿真器里执行**，但会产出 1.2 GB 的 `*_dump`——**全是空文件**，只有 80 字节的 summary 表头 | 用 `env_ascend.sh` 的干净环境，不要注入 SIMLIB/BISHLIB；判断是否真跑了：日志里要有 `ALIGN_RUN_OK` 和 `Profiling data parse finished` |
| 忘记 `--target` | `bishengir-compile` 默认 `Ascend910B1`，`hivmc-a5` 报 `failed to legalize operation 'hivm.hir.load'` | 显式 `--target=Ascend950PR_9579` |
| 用错 hivmc | CANN 里的 `hivmc` 与本项目工具链不同源 | 只用 `~/bishengir-install/bin/hivmc-a5`（可软链成 `hivmc` 放 PATH 前部，让 driver 带全参数调它） |
| 期待 hivmc 打印 | `hivmc … -o -` 在 stdout 上 **0 字节** | 要指令级信息只能走 CAModel 或反汇编 npubin |
| 数 subview 数错 | `memref.copy` 那行不含 subview 信息 | `grep -c memref.subview` |
| `pgrep -f` 自匹配 | 等待器自己的命令行含被搜字符串，循环永不退出（表现成「任务卡死」，实际任务已完成） | 用日志哨兵或行数条件；或 `[p]attern` 写法 |
| PowerShell 吃 `$(...)` | `ssh host 'for i in $(seq 1 400)'` 传到远端变成空列表，等待器立刻退出 | 用 bash 花括号 `{1..400}` |
| 两点 OUTER 阶梯 | 只有 2 个 trip count 时**无法检测非线性**，而实测确实存在（fp16 store 4096 B：64 轮 38 µs、256 轮 305 µs），会静默污染斜率 | 每几何至少 3 个 OUTER 点 |
| 每操作成本归一化 | 用 `/ group / waves`（`waves = 4/factor`）会把每个 `factor>1` 的点放大 F 倍，伪装成「成本 ∝ F」 | 除以「每核每 step 的操作数」= `group × logical_programs / vector_cores` |

---

## 8. 还没验证的部分

| 项 | 现状 |
|---|---|
| R=31 行尾的 burst 结构 | **推断**为 `burst_num: 1`（因为 stride 未失配），尚未跑 CAModel 实测 |
| SIMT 两个方向的尾轴代价 | **未做干净隔离**：align 网格里的 SIMT 点落在每线程溢出区，离散度大，最终模型未合并尾巴项 |
| 「2 cyc/burst」 | 由 4 个几何的算术反推，未直接读 CAModel 的 MTE cycle 列 |
| `pad_mode` 对 DMA 的影响 | 只看到属性变化（`EvictFirst` → `PadValue`），未量化 |

要闭合第 1、3 项，跑一次 `bash scripts/run_align_camodel.sh 32 128` 与 `31 128` 的对照即可。
