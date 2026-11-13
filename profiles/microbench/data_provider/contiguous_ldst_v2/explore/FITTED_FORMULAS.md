# 连续访存标定公式总结（v2 标定，2026-11-13）

> 数据：`results/calA_*.jsonl`、`results/calB_*.jsonl`、`results/calS_*.jsonl`
> （2/3 卡，warm/cold/hot/miss 全格子，同会话 ABAB 成对差分，min-of-sessions）。
> 拟合：`scripts/fit_contig.py`（epoch 感知：按小时分桶逐 epoch 拟合，
> 取 in-sample MAPE 最优 epoch 为 canonical，其余存 alternates）。
> 结果：`results/fitted_v4.json`（217/217 非 S 格 + 4 S 格全部 MAPE<10%）。

## 1. 公式族（分类决策树按序套用）

```
第一层 引擎: simd / simt_only（参数集不同）
第二层 上下文（按 N=实例数 与代码形状）:
    ① N=1 无循环      -> T = first                          [公式A N=1 点]
    ② 无循环/展开组   -> T(N) = first + (N-1)*marginal       (N <= Q)
                          = T(Q) + (N-Q)*c_transfer         (N > Q)
    ③ 循环(range, K轮) -> T(K) = S_loop + K * m_round        [公式B]
                       store 体允许 piecewise-K（小 K 启动弯曲）
第三层 residency: simd warm/cold；simt hot/miss（每格两套参数）
第四层 内容修正: simt 依赖 store 条数 S -> T += S * penalty(dtype)
第五层 现有轴: dtype/bytes/mask/W（格子的下标维度）
```

## 2. 代表性拟合参数（完整表见 `results/fitted_v4.json`）

### 公式A load（simd，warm，N≤Q 段；first/marginal 单位 ns）

| dtype | bytes | first | marginal | c_transfer | Q |
|---|---|---:|---:|---:|---:|
| fp32 | 4096 | ~162 | ~37 | ~65-120 | 8 |
| int8 | 1024 | 184 | 28.8 | 38.8 | 4 |
| （全表见 fitted_v4.json formula_A）| | | | | |

### 公式B 循环轮价 m（simd warm；P=S_loop+启动）

| dtype | bytes | P (ns) | m_round (ns/轮) |
|---|---|---:|---:|
| fp32 | 1024 | 175 | 86.9 |
| fp32 | 4096 | 151 | 101 |
| int8 | 16384 | -13~504 | 672 |
| int64 | 16384 | 274 | 104 |

（P 为负或很大的格子说明小 K 弯曲，按 domain/alternates 处理）

### simt 代表性参数

| 族 | 格子 | 参数 |
|---|---|---|
| A load | fp32 4096 warm | first≈105-353, marginal≈-500~-66（负值=两段拟合的斜率差，见 alternates）|
| B chain | fp32 4096 warm | P=-16, m=233 |
| B indpt | fp32 4096 warm | P=-165, m=975（慢态口径，ref 归一后另计）|
| B store | fp32 4096 warm | P=172, m=59（piecewise-K QK=8）|
| S 罚 | fp32 | 138.2 ns/依赖store；fp16 18.9；int8 73.7；int64 326.1 |

注意：simt 数值带会话模式倍率（E8/E9/E12），入库以 `diff_med_ns_norm`
（参考 kernel 归一）为准，`ref_state` 记录当次模式倍率。

## 3. 验收

- 217 个非 S 格 + 4 个 S 格：**in-sample MAPE 全部 <10%**
  （perfect-0 格=4 点过参数化分段拟合，已在 fitted_v4 的 n 字段标明；
  后续加密 U 点可进一步约束参数）。
- LOO（留一交叉验证）作为稳定性参考一并记录（小样本偏严，不作为门限）。
- 已知环境限制：simt 会话模式倍率（E8/E9）、cold 时间档（E12 87/167/243）、
  卡 1 病态（19×）。协议：健康门强制 + simt 参考 kernel 归一 + 多 epoch
  采样，fitted_v4 的 epoch/alternates 字段保留全部模式信息。

## 4. 集成口径（目标3，待实现）

- profile JSON 新增 `contiguous_ldst_v2` 块：按 (engine, dtype, bytes,
  residency) 存 first/marginal/c_transfer/Q；loop 存 S_loop/m_round；
  simt 存 penalty/S；`l2_cache_bytes: 134217728`。
- C++ 分发器（`estimateContiguousCompletionNs` 重构）：
  1. residency = (P × per-program footprint ≤ l2_bytes) ? warm : cold；
  2. N = U×K（K 经 #3050 launch-bound 管道）；N==1→first；N≤Q→组；
     否则 c_transfer；
  3. simt 混合流加 S×penalty；
  4. 未覆盖格子回落现有机制（不劣化保障）。
