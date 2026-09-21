# SIMD MainScalar diff-line o8：volatile vs 非 volatile 调度（2026-09-21）

## 内容

`simd_ld_diff_o8_sched.cce` 包含两个函数，地址完全相同：`p[0*16]..p[7*16]`，8 条不同 64B line。

```cpp
// v: 原 probe 形式，volatile 指针
volatile __gm__ DT* p = gm;
DT a0 = p[0 * 16]; ... DT a7 = p[7 * 16];
out[0] = (long long)(a0 + ... + a7);

// nv: 去掉 volatile，让编译器自由调度 8 条 load
__gm__ DT* p = gm;
DT a0 = p[0 * 16]; ... DT a7 = p[7 * 16];
out[0] = (long long)(a0 + ... + a7);
```

## 运行命令

```bash
source ~/env_ascend.sh
ulimit -n 1048576

msopprof simulator --soc-version=Ascend950PR_9599 --core-id=0 \
  --launch-count=2 --timeout=30 \
  ./scalar_bench_runner.sh simd_ld_diff_o8_sched.o \
  simd_main_ld_diff_o8_v simd_main_ld_diff_o8_nv
```

## 结果（两次独立 run 完全一致）

活跃窗口 = `core0.veccore0` 第一条 `LD_XD_XN_IMM` issue → 最后一条 retire。

| 函数 | 活跃窗口 cycle | 最大同时 outstanding load |
|---|---:|---:|
| `simd_main_ld_diff_o8_v`（volatile） | **2912** | 2 |
| `simd_main_ld_diff_o8_nv`（非 volatile） | **2667** | 2 |

- 非 volatile 比 volatile 窗口短约 **8.4%**（2912 → 2667）；
- 但两种写法都只有 **2 个 load 同时在途**，没有出现第 3 个。

### MSHR 占用

非 volatile 版本的 MSHR push：

```text
6157: load #1 -> mshr_main_entry_id:0
6158: load #2 -> mshr_main_entry_id:1
6624: load #3 -> mshr_main_entry_id:0
7010: load #4 -> mshr_main_entry_id:0
7410: load #5 -> mshr_main_entry_id:0
7791: load #6 -> mshr_main_entry_id:0
8298: load #7 -> mshr_main_entry_id:0
8299: load #8 -> mshr_main_entry_id:1
```

- 前两条 load 背靠背发出，占用 entry 0/1；
- 第 3 条及以后要等前面的 line fill 返回、entry 释放后才继续 issue；
- 只有最后两条再次同时占用 entry 0/1；
- 没有任何一条 load 使用 entry 2，说明 MainScalar 的 outstanding 上限仍是 2。

## 结论

- 把 8 条 load 写成非 volatile、允许编译器自由调度，确实能让窗口稍微变短（2912 → 2667），但**不能把并行 load 数从 2 提升到 3 或更多**；
- CAModel 的 MainScalar 只有 2 个 MSHR entry，超过 2 条的不同 line load 只能排队；
- 因此 `simd_main_ld_diff_o8` 的耗时不是由 volatile/ADD 造成的，而是由 2-entry MSHR + line fill 周转决定。

## 文件

```text
simd_ld_diff_o8_sched.cce
OPPROF_20260921204239_OQLJCLLOMKMYUCPZ.tar.gz   # run1 完整 OPPROF
OPPROF_20260921204326_PFQUTCQSOHWZTFNT.tar.gz   # run2 完整 OPPROF
run.log
scalar_bench_runner.sh
scalar_bench_host.cpp
SHA256SUMS
```

解压：

```bash
tar -xzf OPPROF_20260921204239_OQLJCLLOMKMYUCPZ.tar.gz
tar -xzf OPPROF_20260921204326_PFQUTCQSOHWZTFNT.tar.gz
```

服务器路径：

```text
~/simd_ld_diff_o8_sched_camodel_20260921/
  OPPROF_20260921204239_OQLJCLLOMKMYUCPZ/
  rep2/OPPROF_20260921204326_PFQUTCQSOHWZTFNT/
```
