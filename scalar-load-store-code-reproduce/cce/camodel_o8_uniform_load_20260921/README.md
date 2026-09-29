# CCE scalar o8 load：SIMD / SIMT CAModel 完整结果（2026-09-21 重跑）

## 内容

本目录是 `scalar_bench6.cce` 中 4 个 o8 scalar load 函数的完整 CAModel 结果：

| 函数 | 形态 |
|---|---|
| `simd_main_ld_same_o8` | SIMD MainScalar，8 条 load 落在同一 64B line |
| `simd_main_ld_diff_o8` | SIMD MainScalar，8 条 load 落在 8 条不同 64B line |
| `simt_ld_uniform_same_o8` | SIMT warp-uniform，8 条 LDG 访问同一 128B line |
| `simt_ld_uniform_diff_o8` | SIMT warp-uniform，8 条 LDG 访问 8 条不同 128B line |

## 运行命令

服务器：`ascend-950pr-63`，CANN 9.1.0，`msopprof simulator`。

```bash
source ~/env_ascend.sh
ulimit -n 1048576

cd ~/o8_uniform_load_camodel_20260921
msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=4 --timeout=30 \
  ./scalar_bench_runner.sh scalar_bench6.o \
  simd_main_ld_same_o8 simd_main_ld_diff_o8 \
  simt_ld_uniform_same_o8 simt_ld_uniform_diff_o8
```

- `scalar_bench6.cce` / `scalar_bench_host.cpp` / `scalar_bench_runner.sh` 为本次运行源码与 wrapper；
- `scalar_bench6.o` 与 `scalar_bench_host` 二进制未入库，需要用目录内源码重新编译；
- 编译命令参考 `63-workspace/10-scalar-camodel-bench/scalar_bench_camodel_same_vs_diffline_20260915` 或服务器 `~/scalar-bench-exp/`。

## 结果文件

```text
OPPROF_20260921142340_AMYOQYQSBJNJIITP.tar.gz
  OPPROF_20260921142340_AMYOQYQSBJNJIITP/
    simd_main_ld_same_o8/0/{dump,simulator}/
    simd_main_ld_diff_o8/0/{dump,simulator}/
    simt_ld_uniform_same_o8/0/{dump,simulator}/
    simt_ld_uniform_diff_o8/0/{dump,simulator}/
```

解压：

```bash
tar -xzf OPPROF_20260921142340_AMYOQYQSBJNJIITP.tar.gz
```

- `run.log`：msopprof 完整 stdout/stderr；
- `analysis.json`：用 `analyze_scalar_bench.py` 对四个函数提取的 SIMT LSU、SIMT DC、MainScalar 指令统计；
- `SHA256SUMS`：本目录文件校验和。

## 实测摘要

本次运行（2026-09-21，`core0.veccore0`）：

| 函数 | 活跃窗口 cycle | 关键观察 |
|---|---:|---|
| `simd_main_ld_same_o8` | **531** | 8 条 `LD_XD_XN_IMM`；前 2 条 miss，后 6 条 dcache hit（`execTime=4`） |
| `simd_main_ld_diff_o8` | **2828** | 8 条 `LD_XD_XN_IMM`，8 条不同 64B line，全部 miss；MSHR 排队后逐波完成 |
| `simt_ld_uniform_same_o8` | **3652** | 8 条 `SIMT_LDG`；DC 8 个 size=4 请求，同 line 不 merge，串行返回 |
| `simt_ld_uniform_diff_o8` | **569** | 8 条 `SIMT_LDG`；不同 128B line 可并发，窗口只比单条略大 |

- `msopprof` 汇总：`Model RUN TIME: 26837.5 ms`、`Total tick: 15957`、`All task success`；
- CAModel 内部 cycle → time 为 1.8 GHz；`ns = cycle / 1.8`。

## 校验和

见 `SHA256SUMS`。其中 OPPROF 包：

```text
6f9e3190a5b433db13d87638941a5634a6b0d0cedfd9c5c18bce3310dfebc741  OPPROF_20260921142340_AMYOQYQSBJNJIITP.tar.gz
```
