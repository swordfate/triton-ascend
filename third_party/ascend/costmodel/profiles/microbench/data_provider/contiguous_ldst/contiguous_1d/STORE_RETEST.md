# store 重测与拟合链路状态（2026-11-04）

## 1. 为什么要重测 store

`scripts/probe_block_pure.py` 的 store 分支写的是：

```python
for step in range(iters):
    tl.store(output + base + (step & MASK) * BLOCK + lane, value + step)
```

即每次 store 前多一条 `value + step` 的向量加法。用小 store 做 A/B（同一探针、同一 profiler 通道、只改这一行）：

| 方向 | dtype | el | W | `value+step` | `value` | 差异 |
|---|---|---:|---:|---:|---:|---:|
| load（对照）| fp32 | 8 | 64 | 155.09 | 155.14 | +0.0% |
| store | fp16 | 2 | 64 | 155.15 | 116.35 | −25.0% |
| store | fp32 | 8 | 1 | 30.28 | 26.04 | −14.0% |
| store | fp32 | 8 | 64 | 155.10 | 116.33 | −25.0% |
| store | int8 | 64 | 1 | 25.47 | 22.41 | −12.0% |
| store | int8 | 64 | 64 | 126.66 | 126.76 | +0.1% |
| store | fp32 | 2048 | 4 | 234.29 | 235.79 | +0.6% |

结论：那次加法是 ALU 开销，只在小 store 上显著（12~25%），大 store 可忽略（<1%）。load 分支两版一致，说明装置本身没问题。
常量版没有被优化掉：代价仍随规模单调增长（26 → 116 → 235 ns）。

## 2. 重测范围与结果

- 覆盖 `by <= 256 B` 的 store 配置，共 **221 个**（SIMT 196、SIMD 25），两张卡并行跑完；
- 与老数据重叠 201 个配置，新值系统性偏低：**中位 −9.1%、平均 −10.6%、最大 −93.5%**；

| 字节档 | 配置数 | 中位差异 | 最大差异 |
|---|---:|---:|---:|
| by ≤ 32 | 126 | −11.6% | −61.4% |
| 32 < by ≤ 128 | 52 | −5.6% | −93.5% |
| 128 < by ≤ 256 | 23 | −8.5% | −24.1% |

数据：`raw/raw_store_retest.jsonl`（221 条，`variant=const`）。探针：`scripts/probe_variant.py`（`--variant step|const`）。

`by > 256 B` 的 store 数据未重测——A/B 显示该档加法影响 <1%。

## 3. 拟合链路现状（阻塞点）

想用新数据重拟合时发现链路已跑不通：

1. `fit/assemble_spec.py` 读 `fitted/SIMD_CONTIGUOUS.json` 时取 `v['a'] / v['b'] / v['c'] / v['d']`，
   而该文件现在的键是 `features / coefficients / mape / max_err / n` —— **schema 不匹配**；
2. `fitted/SIMD_CONTIGUOUS.json` 声明的数据源 `lattice_raw/simd_min5.jsonl` **不在仓库里**；
3. 因此 `assemble_spec.py` 目前无法复现 profile 里的 `simd.load` / `simd.store`。

## 4. 关于 SIMD 公式的项数

现有 load 公式是 `one + by + by2 + by3` 四项。用同一批 10 个 load 点比较：

| 形式 | MAPE | 最大误差 |
|---|---:|---:|
| `one+by` | 11.89% | 49.2% |
| `one+by+by2` | 4.49% | 11.25% |
| `one+by+by2+by3` | 2.46% | 5.69% |

三次项是为了把最差点从 11.25% 压到 10% 以下——**不是覆盖扩展带来的**。

但用完整 SIMD 扫描（97 load 点、100 store 点，31 B ~ 128 KiB）重拟合：

| 形式 | load MAPE | load max | store MAPE | store max |
|---|---:|---:|---:|---:|
| `one+by` | 9.12% | 55.6% | 9.31% | 28.6% |
| `one+by+by2` | 6.82% | 37.6% | 6.39% | 24.8% |
| `one+by+by2+by3` | 6.78% | 37.6% | 58.9% | 99.4% |

即：在全量数据上三次项几乎无收益（6.82% → 6.78%），对 store 甚至发散。说明它是在 10 点子集上为凑 10% 门槛加的项。

## 5. 下一步

1. 修 `assemble_spec.py` 的 schema（或恢复 `lattice_raw/simd_min5.jsonl`），让链路可复现；
2. 用 `raw_store_retest.jsonl` 替换 by≤256 的 store 值，重拟合 SIMT store 段与 SIMD store 公式；
3. SIMD 侧改用全量扫描点重拟合，并按第 4 节结论决定是否去掉三次项；
4. 重新生成 profile、重编、跑 100-case 验证。
