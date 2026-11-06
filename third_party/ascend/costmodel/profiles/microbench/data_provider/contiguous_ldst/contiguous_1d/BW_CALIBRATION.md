# 带宽型（吞吐口径）连续访存标定

## 1. 这份数据解决什么问题

延迟口径的标定（`raw/`、`fitted/` 那一套）用 `acc += tl.load(...)` 的循环，
每轮必须等上一轮的加法完成，迭代被依赖链串起来，测到的是一次访存的**延迟**。

真实 kernel 的循环体是 `load -> 计算 -> store`：第 i+1 轮的 load 不与第 i 轮产生
依赖，可以和上一轮的计算重叠。用 CAModel 的总线日志可以直接看到这一点：

```
第 1 轮 8 条 128 B 读        : t = 21220 .. 21243
第 2 轮第一条读              : t = 21734
第 1 轮第一条写              : t = 22587
```

第 2 轮的 load 比第 1 轮的 store 早 851 个时间单位，说明访存确实是流水的。
这种情况下起作用的是**吞吐**，不是延迟；用延迟值去乘迭代数会系统性高估。

所以本次另做一组吞吐口径的标定：打断依赖链、按满核并发、用 k-marginal 测每轮增量。

## 2. 测量方法

脚本：`scripts/probe_bw_throughput.py`

```python
for step in range(iters):              # 运行时循环，做 k-marginal
    row = base + (step & WINMASK) * BLOCK * UNROLL
    v = tl.zeros((BLOCK,), dt)
    for j in tl.static_range(UNROLL):  # 静态展开：U 条互不依赖的访存
        v += tl.load(src + row + j * BLOCK + lane, mask=keep, other=0)
    acc += v                           # 每轮只加一次，不形成逐条依赖
```

| 项 | 取值 | 理由 |
|---|---|---|
| grid | 56（满核）| cost model 算的是“和其他 55 个核同时跑”的单程序代价 |
| 地址窗口 | 按字节数反推，使总足迹 ≈ 64 MB | 落在 128 MB L2 内，匹配 100-case 里 MB 级 kernel 的真实状态 |
| 求斜率 | k 取 {k, 2k, 4k} 三点线性回归 | 两点差分噪声大（曾出现负值）；三点回归 R² 中位 1.0000 |
| 判废 | R² < 0.98 或斜率 ≤ 0 | 自动剔除测量噪声点 |
| 形态 A | 无 mask | 与延迟口径一致 |
| 形态 B | `mask = lane < M`，M 运行时传入且 M == BLOCK | 动态全 true，编译器证不出 |
| 非 2 幂 | SIMD 下直接用真实长度 `tl.arange(0, N)`，**不填充也不加 mask** | 已实测 781/1000/33 均可编译运行 |

## 3. 数据

| 文件 | 点数 | 说明 |
|---|---:|---|
| `bw_raw/BW_SIMD.jsonl` | 346 | SIMD，2 幂 + 非 2 幂，A/B 形态，四种 dtype |
| `bw_raw/BW_SIMT.jsonl` | 426 | SIMT，仅 2 幂，warp 1/2/4/8/16/32/64 |

每条记录的字段：`mode, direction, dtype, bytes, nreal, block, form, warps, grid,`
`unroll, window_blocks, footprint_mb, per_iter_ns, per_acc_ns, mlp_bytes, r2, source`。

清洗规则（提交的数据已按此过滤）：

| 规则 | 原因 |
|---|---|
| `block == nreal` | 早期探针把非 2 幂填充到 2 的幂再加掩码，产物偏慢 2.5~3 倍，已全部剔除 |
| `r2 >= 0.999` | 剔除回归线性度差的点 |
| SIMT 只用 `source in {sm, retest}` | 早期 SIMT 批次 repeats=4，128 B 小尺寸偏高 2.5~3 倍 |
| SIMT 拟合只用 `te < 8` | 每线程元素数超过 8 后进入寄存器溢出区，是步进而非曲线 |

## 4. 拟合结果

### 4.1 SIMD（两个方向各一条，MAPE 均 < 10%）

$$\text{perAccess}_{\text{load}} = \max\left(7 + \frac{70}{u},\ 12.5 \times \text{KiB}\right)$$

$$\text{perAccess}_{\text{store}} = 91.9 + 7.46 \times \text{KiB} - \frac{7.8}{u}$$

| 段 | n | MAPE | 系数 |
|---|---:|---:|---|
| simd.load | 175 | **8.6%** | a0=7, a1=70, b=12.5 |
| simd.store | 171 | **7.11%** | a0=91.87, b=7.46, c=-7.77 |

| 项 | 物理含义 |
|---|---|
| `a0` | 每次访问的固定下限 |
| `a1/u` | 单次访存延迟被 u 条并发访存摊薄（u = 同一轮内的同形访存条数）|
| `b·KiB` | 流式传输项，load 12.5 ns/KiB ≈ 82 GB/s/程序，store 7.46 ns/KiB ≈ 137 GB/s/程序 |

两条公式同时覆盖 2 幂与非 2 幂、形态 A/B、int8/fp16/fp32/int64，不需要再分段。

### 4.2 SIMT（按 warp 分段，同形）

$$\text{perAccess} = \max\left(a_0 + \frac{a_1}{u},\ b \times \text{KiB}\right)$$

| 段 | n | MAPE | a0 | a1 | b |
|---|---:|---:|---:|---:|---:|
| `simt.load.W1` | 4 | 14.73% | -60 | 210 | 63.5 |
| `simt.load.W2` | 14 | 33.33% | -22 | 150 | 34.5 |
| `simt.load.W4` | 23 | 31.94% | -2 | 80 | 19.5 |
| `simt.load.W8` | 22 | 26.99% | 4 | 60 | 22.5 |
| `simt.load.W16` | 28 | 31.93% | 6 | 60 | 14.5 |
| `simt.load.W32` | 29 | 24.80% | 20 | 70 | 13.5 |
| `simt.load.W64` | 30 | 26.86% | 72 | 100 | 0.5 |
| `simt.store.W2` | 10 | 27.15% | -52 | 80 | 41.5 |
| `simt.store.W4` | 12 | 19.07% | 8 | 20 | 14.5 |
| `simt.store.W8` | 20 | 26.68% | 10 | 20 | 14.5 |
| `simt.store.W16` | 21 | 25.36% | 14 | 30 | 19.5 |
| `simt.store.W32` | 25 | 27.14% | 2 | 70 | 14.5 |
| `simt.store.W64` | 28 | 35.14% | -4 | 140 | 13.5 |

13 段中位 26.99%，范围 14.73%~35.14%。**未达到 10% 的目标。**

## 5. SIMT 未达标的原因（已排除测量噪声）

| 检查项 | 结果 |
|---|---|
| 同配置连测 3 次 | 极差中位 1.00，最大 1.49 → 测量可重复 |
| 单点回归 R² | 中位 1.00000，最小 0.9834 → 单点质量好 |
| 每段点数 | 4~30 个，W1 只有 4 个 |
| 曲面非单调 | W16 在 4 KiB 折算 93 GB/s，8 KiB 只有 41 GB/s |

即误差来自**点太少 + 曲面本身不单调**，不是公式形式或测量质量。

## 6. 复现

```bash
source ~/env_ascend.sh; source /data/miniconda3/etc/profile.d/conda.sh
conda activate wj_autoscope; ulimit -n 1048576

# 单点测量（例：SIMD load 1 KiB，unroll 4）
python3 scripts/probe_bw_throughput.py --mode simd --direction load --dtype fp32 \
    --bytes 1024 --unroll 4 --form a --repeats 6 --manifest out.jsonl

# 拟合（读本目录的 bw_raw/）
python3 scripts/fit_bw_simd.py bw_raw/BW_SIMD.jsonl
python3 scripts/fit_bw_simt.py bw_raw/BW_SIMT.jsonl
```

## 7. 踩过的坑

| 坑 | 现象 | 结论 |
|---|---|---|
| 同一台机上多进程共用一个 trace 目录 | `CANN Profiling data does not exist`，73/97 失败 | 每个进程用独立 `tempfile.mkdtemp()` 目录，之后同卡 2 进程正常 |
| 非 2 幂用“填充到 2 幂 + 掩码”实现 | 测出比同字节 2 幂慢 2.5~3 倍，误判为硬件行为 | SIMD 直接 `tl.arange(0, N)` 即可，无需掩码 |
| `nacc=1`（单条访存）测带宽 | 聚合带宽算出 1224 GB/s，超过峰值 | 至少 2 条独立访存在途才有带宽意义；本数据用 `unroll` 提供 |
| 两点差分求斜率 | 出现负值（hi < lo） | 改三点回归 + R² 判废 |
| 足迹未控制 | 58 MB 时测得 1609 GB/s，940 MB 时只有 162 GB/s | 128 MB L2；窗口按足迹 ≈ 64 MB 反推 |
| SIMT repeats=4 | 128 B 小尺寸偏高 2.5~3 倍 | 统一 repeats=8 |

