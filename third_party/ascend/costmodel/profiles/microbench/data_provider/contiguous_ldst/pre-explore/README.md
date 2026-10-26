# 非对齐（非 2 的幂尾轴）机制探索

本分支**只**包含「尾轴不是 2 的幂时为什么慢」的取证与复现：三层 IR、CAModel 逐指令、成本分解。
**模型与系数不在这里**——那些在 `feature/contiguous-calibration`（已含 4 条闭式与拟合脚本）。

基线提交与主分支相同（`feature/simd-simt-compile-mode` 的 `e87bec9a3`），可单独 checkout 复现。

## 结论

尾轴元素数不是 2 的幂 → 编译器把 UB 缓冲行宽补到下一个 2 的幂 → 掩码不再可折叠 → 多出两件事：

1. **整块缓冲先被广播清零**（`hivm.hir.vbrc`），且它与 DMA 之间有 `PIPE_V→PIPE_MTE2` 跨流水同步（串行化）；
2. DMA 从 `1 个 16384 B 的 burst` 变成 `32 个 508 B 的 burst`。

    多付 cycle ≈ 0.039 cyc/补齐元素 × 补齐后元素数    ← 主项，约 3/4
                + 2 cyc/burst × burst 数              ← 次项，约 1/4

三个关键实测（`raw/raw_align_grid.jsonl`，每几何 3 重复）：

| 几何 | cyc/op | cyc/KB | vs 对齐 |
|---|---:|---:|---:|
| 32×128（2 的幂） | 360.8 | 22.6 | — |
| 32×127（列尾） | 577.8 | 36.4 | +61% |
| 32×160（列尾，**128 B 完全对齐**） | 856.9 | 42.8 | +89% |
| 31×128（行尾） | 509.8 | 32.9 | +46% |

**C=160 是判据的分水岭**：行 640 B = 5×128 B，字节上完美对齐，却比 C=127 更慢 →
触发条件是「元素数是不是 2 的幂」，不是字节对齐。

## 文件

| 路径 | 作用 |
|---|---|
| `fit/ALIGNMENT.md` | **完整机制文档**：三层证据逐字对比、成本分解、坑、复现命令、未验证项 |
| `scripts/probe_alignment_ir.py` | 只编译不运行，dump `ttir`/`ttadapter`，统计 `memref.subview` 数 |
| `scripts/probe_alignment_sweep.py` | 二维 tile + **运行期** real_rows/real_cols 的计时探针（尾轴/行数） |
| `scripts/align_camodel.py` | CAModel 用的最小 kernel（一个 program，带 extent 掩码） |
| `scripts/run_align_camodel.sh` | 把上面那个 kernel 送进 `msprof op simulator` 并打印 burst 参数 |
| `scripts/make_alignment_grid.py` / `sweeps/align_grid.json` | 36 个尾轴/行尾几何的定义 |
| `scripts/run_points.py` | 网格驱动（支持 `--resume`） |
| `raw/raw_align_grid.jsonl` | 216 次 launch 的冻结原始数据 |

## 复现

    source ~/env_ascend.sh

    # 1) 结构：不需要设备
    python3 scripts/probe_alignment_ir.py --rows 32 --cols 127 --dtype fp32 \
        --mode simd --masked 1 --dump-dir out/ir_c127
    grep -c memref.subview out/ir_c127/kernel.ttadapter      # 3（对齐时是 0）

    # 2) op 形态：编译到 hivm.opt（必须带 --target）
    ~/bishengir-install/bin/bishengir-compile <CACHE>/<hash>/align_kernel.bcmlir \
        --target=Ascend950PR_9579 --save-temps=/tmp/st_c127

    # 3) burst 参数：CAModel，约 4 分钟一次
    bash scripts/run_align_camodel.sh 32 128     # burst_num: 1,  burst = 16384
    bash scripts/run_align_camodel.sh 32 127     # burst_num: 32, burst = 508

## 三个必踩的坑

1. **不要把 `SIMLIB`/`BISHLIB` 塞进 `LD_LIBRARY_PATH`**。Triton 的 ascend driver 初始化时要现编一个
   `npu_utils` 扩展，其编译命令里带别的机器的硬编码路径；污染后编译失败，**kernel 根本不会在仿真器里执行**，
   但 `msprof` 照样写出 ~1 GB 的 `*_dump`——**全是空文件**。判据只有一个：日志里出现
   `ALIGN_RUN_OK` 与 `Profiling data parse finished`。
2. **忘 `--target` 会让 hivmc 失败**：默认 `Ascend910B1`，`hivmc-a5` 报
   `failed to legalize operation 'hivm.hir.load'`。必须 `--target=Ascend950PR_9579`。
3. **只用 `~/bishengir-install/bin/hivmc-a5`**，不要用 CANN 里的 `hivmc`（不同源）。
   另外 `hivmc … -o -` 在 stdout 上是 **0 字节**，指望它打印指令级信息会白等。

其余坑（`pgrep -f` 自匹配、PowerShell 抢先展开 `$(...)`、两点 OUTER 阶梯无法检测非线性、
每操作成本的归一化写错）见 `fit/ALIGNMENT.md` §7。

## 未验证

| 项 | 现状 |
|---|---|
| R=31 行尾的 burst 结构 | 推断为 `burst_num: 1`（stride 未失配），未跑 CAModel 实测 |
| SIMT 两个方向的尾轴代价 | align 网格里的 SIMT 点落在每线程溢出区，离散度大，**未做干净隔离** |
| `pad_mode` 的实际作用 | 只见属性从 `EvictFirst` 变 `PadValue`，未量化 |
