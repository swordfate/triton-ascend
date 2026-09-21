# SIMD scalar o8 load：去掉 ADD 的 diff-line 变体（2026-09-21）

## 内容

本目录是 `simd_ld_o8_variants.cce` 的完整 CAModel 结果，用于对比：

- `simd_main_ld_diff_o8_sum`：原形态，8 条不同 64B line load，最后 `out[0] = a0+...+a7`；
- `simd_main_ld_diff_o8_nc`：去掉 ADD，直接用多个 GM 地址做 8 条 volatile load；
- 同 line 版本 `same_o8_sum` / `same_o8_nc` 作为对照。

`_nc` 的核心代码：

```cpp
volatile __gm__ DT* p = gm;
(void)p[0 * 16]; (void)p[1 * 16]; ... (void)p[7 * 16];
out[0] = 0;
```

## 运行命令

```bash
source ~/env_ascend.sh
ulimit -n 1048576

msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=4 --timeout=30 \
  ./scalar_bench_runner.sh simd_ld_o8_variants.o \
  simd_main_ld_same_o8_sum simd_main_ld_same_o8_nc \
  simd_main_ld_diff_o8_sum simd_main_ld_diff_o8_nc
```

## 结果（3 次独立 run 完全一致）

活跃窗口 = `core0.veccore0` 第一条 `LD_XD_XN_IMM` issue → 最后一条 retire。

| 函数 | 活跃窗口 cycle | 现象 |
|---|---:|---|
| `simd_main_ld_same_o8_sum` | **469** | 前 2 条 miss/merge，后 6 条 dcache hit |
| `simd_main_ld_same_o8_nc` | **476** | 只有第 1 条 miss，后 7 条 hit |
| `simd_main_ld_diff_o8_sum` | **2747** | 8 条不同 64B line，全部 miss |
| `simd_main_ld_diff_o8_nc` | **3014** | 8 条不同 64B line，全部 miss，比 sum 慢约 **+9.7%** |

`simd_main_ld_diff_o8_nc` 的 rep1 issue 时间戳：

```text
10537, 10931, 11376, 11747, 12168, 12661, 13030, 13031
issue gaps ≈ 394, 445, 371, 421, 493, 369, 1
```

## 结论

- 去掉 ADD 计算后，8 条不同 GM 地址的 load 没有变快，反而略慢；
- issue 间隔仍在约 370–500 cycle 量级，说明 SIMD diff-line o8 的时间由 MSHR / BIU line-fill 调度决定，不是最后的加法链；
- same-line 版本也基本不变（469 → 476），只是 miss/hit 分布从 2 miss + 6 hit 变成 1 miss + 7 hit。

完整 OPPROF 包含四个函数：

```text
OPPROF_20260921202737_YIUKZEMVRNTGVPOM/
  simd_main_ld_same_o8_sum/0/{dump,simulator}/
  simd_main_ld_same_o8_nc/0/{dump,simulator}/
  simd_main_ld_diff_o8_sum/0/{dump,simulator}/
  simd_main_ld_diff_o8_nc/0/{dump,simulator}/
```

## 文件

```text
simd_ld_o8_variants.cce
OPPROF_20260921202737_YIUKZEMVRNTGVPOM.tar.gz
run.log
scalar_bench_runner.sh
scalar_bench_host.cpp
SHA256SUMS
```

解压：

```bash
tar -xzf OPPROF_20260921202737_YIUKZEMVRNTGVPOM.tar.gz
```

服务器复用三次 run：

```text
~/simd_ld_o8_nc_camodel_20260921/
  OPPROF_20260921202737_YIUKZEMVRNTGVPOM/
  rep2/OPPROF_20260921202827_UESSLZGEXOYCLWIF/
  rep3/OPPROF_20260921202859_ZSJHWKRMNOQUFYAR/
```
