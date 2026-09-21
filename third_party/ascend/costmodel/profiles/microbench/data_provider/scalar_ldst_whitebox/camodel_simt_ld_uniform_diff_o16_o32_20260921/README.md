# SIMT uniform diff-line o4 / o16 / o32：CAModel 结果（2026-09-21）

## 探针

`simt_ld_uniform_diff_o16_o32.cce` 包含三个函数：

```cpp
simt_ld_uniform_diff_o4    // p[0*32]..p[3*32]
simt_ld_uniform_diff_o16   // p[0*32]..p[15*32]
simt_ld_uniform_diff_o32   // p[0*32]..p[31*32]
```

每个 op 的 32 个 lane 访问同一地址；相邻 op 相隔 32 个 float（128B），即每个 op 落在不同的 128B line：

```cpp
volatile __gm__ DT* p = gm;
DT a0 = p[0 * 32]; ... DT aN = p[(N-1) * 32];
sink[lane] = (DT)(a0 + ... + aN);
```

## 运行命令

```bash
source ~/env_ascend.sh
ulimit -n 1048576

msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=3 --timeout=30 \
  ./scalar_bench_runner.sh simt_ld_uniform_diff_o16_o32.o \
  simt_ld_uniform_diff_o4 simt_ld_uniform_diff_o16 simt_ld_uniform_diff_o32
```

## 结果（两次独立 run 完全一致）

活跃窗口 = `core0.veccore0` 第一条 `SIMT_LDG` issue → 最后一条 `SIMT_LDG` retire。

| 函数 | LDG 数 | distinct 128B line | 活跃窗口 cycle |
|---|---:|---:|---:|
| `simt_ld_uniform_diff_o4` | 4 | 4 | **570** |
| `simt_ld_uniform_diff_o16` | 16 | 16 | **579** |
| `simt_ld_uniform_diff_o32` | 32 | 32 | **2735** |

### o4 / o16：所有 line fill 可以 overlap

- o4 issue：`1829, 1848, 1850, 1852`，last retire `2399`；
- o16 issue：`3810, 3812, ..., 3840`，全部 2 cycle 间隔背靠背发出，last retire `4389`；
- o16 的 16 个不同 128B line 请求可以同时在途，窗口只比 o4 大 9 cycle。

### o32：在途队列饱和后出现 LSU 反压

o32 issue 时间戳：

```text
5506, 5508, ..., 5548,   # 前 22 条，2-cycle spacing
6114,                    # gap 566
6621, 6624, 6627, 6630, 6633,
7214, 7217, 7220,
7702                     # gap 482
```

- 前 22 条 LDG 仍能背靠背发出；
- 第 23 条开始出现 482–581 cycle 的 issue gap，说明 SIMT LSU/DC/BIU 在途队列已饱和；
- last retire `8241`，窗口 **2735 cycle**，约 o4/o16 的 **4.8×**；
- DC 侧 32 个请求各 size=4，对应 32 条不同 128B line。

## 结论

- SIMT uniform diff-line 的 line fill 在没有超过在途限制时可以完全 overlap：o16 的窗口仍只有 ~579 cycle；
- 在途能力不是无限的，o32 超过阈值后 LSU issue 被反压，窗口跳到 ~2735 cycle；
- 因此 cost model 不能把 SIMT diff-line 每个 op 都按一个固定小边际线性外推，需要区分“在途可容纳”和“队列饱和”两段行为。

> 额外说明：如果把 `sum` 完全去掉、每条 load 改成独立的 `(void)p[i]` volatile 语句，CAModel 会按 volatile 顺序一条条串行发出（issue gap ~450–570 cycle），不能用来比较队列行为；本目录保留原 `sum` 形式。

## 文件

```text
simt_ld_uniform_diff_o16_o32.cce
OPPROF_20260921203051_WGYLVNSHLAOWSMVY.tar.gz   # run1 完整 OPPROF
OPPROF_20260921203306_PVWVHHDVSPRDECWJ.tar.gz   # run2 完整 OPPROF
run.log
scalar_bench_runner.sh
scalar_bench_host.cpp
SHA256SUMS
```

解压：

```bash
tar -xzf OPPROF_20260921203051_WGYLVNSHLAOWSMVY.tar.gz
tar -xzf OPPROF_20260921203306_PVWVHHDVSPRDECWJ.tar.gz
```

完整结果服务器路径：

```text
~/simt_ld_uniform_diff_o32_camodel_20260921/
  OPPROF_20260921203051_WGYLVNSHLAOWSMVY/
  rep2/OPPROF_20260921203306_PVWVHHDVSPRDECWJ/
```
