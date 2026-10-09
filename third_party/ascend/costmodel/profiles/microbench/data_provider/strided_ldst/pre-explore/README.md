# Strided `tl.load` / `tl.store` CostModel：白盒探索与实测重拟合

本目录记录 A5 上 shaped strided `tl.load` 的 CostModel 研究结果，只保留当前有效结论。
所有 Python / Shell 脚本都在 `scripts/` 下。

---

## 1. Strided `tl.load`

### 1.1 白盒模型探索

#### 1.1.1 写了什么脚本

| 脚本 | 作用 |
|---|---|
| `scripts/strided_load_probe.py` | 最小 Triton 探针：一个 shaped `tl.load`，地址为 `base + tl.arange(0, BLOCK) * STRIDE`；支持强 SIMD / 强 SIMT、BLOCK 和 stride 扫描，输出 launch manifest 与编译产物 |
| `scripts/run_camodel.sh` | `msopprof simulator` 驱动：在 Ascend950PR CAModel 上按 case 顺序采集 OPPROF |
| `scripts/analyze_strided_camodel.py` | 解析 CAModel dump：AIV `instr_log`、MTE2 issue/retire、SIMT LSU、SIMT DC TagRam、BIU send/recv，输出 `parsed.json` / `parsed.csv` |
| `scripts/analyze_simt_ldg_windows.py` | 按 `SIMT_LDG` 统计 issue/retire，检查多 warp 的 overlap |
| `scripts/extract_evidence.py` | 从具体 case 提取原始指令 / cache / BIU 片段，便于审计 |
| `scripts/make_summary.py`、`scripts/summarize_parsed.py` | 汇总不同 BLOCK 的 CAModel 结果并打印表格 |

#### 1.1.2 从 CAModel 分析出了什么

##### SIMD：`tt.load` 走 MTE2 `MOV_SRC_TO_DST_ALIGNv2` GM→UB

- shaped strided load 先由 MTE2 搬进 UB，算子在 VF 中消费 UB；**不走 AIV 64B DCache**。
- `stride=1/2` 时 MTE2 能用宽事务：
  - BLOCK=32：stride1 = 1×128B，stride2 = 1×256B；
  - BLOCK=128：stride1 = 1×512B，stride2 = 2×512B。
- `stride>=3`（元素间隔 ≥12B）时退化为 **每个元素一条 4B BIU read**：
  - 此前代表性表格只放了 `stride=1/2/4`，看起来像从 2 直接跳到 4；
    补测 `stride=3` 后确认它已经在 gather 路径上：
  - BLOCK=32 是 32×4B，BLOCK=64 是 64×4B，BLOCK=128 是 128×4B。
- 因此 SIMD 侧的关键不是 128B line 数，而是 MTE2 能否宽事务、是否退化成
  per-element gather；后面拟合出的 SIMD wide / gather 两条路径即对应这里。

代表性 BLOCK=32 数据（raw CAModel cycle；`active = pre + load + post`）：

| stride | active | pre | load | post | load 内 `issue→BIU` / `BIU fill` / `BIU→ret` | BIU command |
|---:|---:|---:|---:|---:|---|---|
| 1 | 1503 | 424 | 446 | 633 | 20 / 403 / 23 | 1×128B |
| 2 | 1699 | 464 | 550 | 685 | 20 / 504 / 26 | 1×256B |
| 3 | 1593 | 425 | 571 | 597 | 20 / 529 / 22 | 32×4B |
| 4 | 1642 | 458 | 586 | 598 | 20 / 543 / 23 | 32×4B |
| 16 | 1552 | 419 | 588 | 545 | 20 / 545 / 23 | 32×4B |
| 256 | 1803 | 556 | 580 | 667 | 20 / 534 / 26 | 32×4B |

BLOCK 扫描进一步给出 gather 分支的线性关系：

```text
SIMD gather (stride>=3), BLOCK=32/64/128:
load_active ≈ 554 + 1.0 * BLOCK
```

##### SIMT：`tt.load` 走 `SIMT_LDG` + 128B line DCache

- shaped strided load 降成一条或多条 `SIMT_LDG`，经 SIMT DCache 读 128B line。
- DC 会把同一个 128B line 内多个 lane 的请求合并，`TagRam.size` 是该 line 的有效字节；
  但 BIU 仍按整 128B line 读取。
- BLOCK=32、f32 的 line / DC 形态：

| stride | distinct 128B line | 每 line 有效字节 | DC request size | line 利用率 | BIU line 放大 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 128 | 128 | 100% | 1× |
| 2 | 2 | 64 | 64 | 50% | 2× |
| 3 | 3 | 40–44 | 40 / 44 | 33.3% | 3× |
| 4 | 4 | 32 | 32 | 25% | 4× |
| 8 | 8 | 16 | 16 | 12.5% | 8× |
| 16 | 16 | 8 | 8 | 6.25% | 16× |
| 32+ | 32 | 4 | 4 | 3.125% | 32× |

- `stride=3→256` 时 line 数从 3 增到 32，但单条 `SIMT_LDG` 的 load 窗口只在约
  483–654 cycle 之间（stride=3: 546，stride=4: 483，stride=32: 615，stride=256: 654）：
  BIU line fill 高度 overlap，**不能按“每条 line 一次完整 miss latency”收费**。
- 多 warp（BLOCK=128）时，单条 LDG 自身约 1.1k cycle，但 4 条 LDG 的总窗口只有约
  1.17k cycle：stage-level 成本不能按 `K × 单条 latency` 线性放大。

##### 补测：`stride=2` 与 `stride=4` 之间的 `stride=3`

为消除“从 2 直接跳到 4”的抽样缺口，补跑 `stride=3`、BLOCK=32/64/128、
SIMD/SIMT 各一遍。结论是：`stride=3` 在 SIMD 侧已经是 per-element gather，
在 SIMT 侧按 128B line 合并（BLOCK=32 为 3 条 line），所以模型分界应是 `stride>=3`。

| BLOCK | mode | active | pre | load | post | load 内 issue→BIU / BIU fill / BIU→ret | BIU command | DC / lines |
|---:|---|---:|---:|---:|---:|---|---|---|
| 32 | SIMD | 1593 | 425 | 571 | 597 | 20 / 529 / 22 | 32×4B | — |
| 64 | SIMD | 1602 | 425 | 618 | 559 | 20 / 569 / 29 | 64×4B | — |
| 128 | SIMD | 1793 | 425 | 682 | 686 | 20 / 635 / 27 | 128×4B | — |
| 32 | SIMT | 2276 | 1105 | 546 | 625 | 23 / 491 / 32 | 3×128B | 3 lines；DC size 44×2 + 40×1 |
| 64 | SIMT | 2216 | 1190 | 464 | 562 | 23 / 407 / 34 | 6×128B | 6 lines；DC size 44×4 + 40×2 |
| 128 | SIMT | 2312 | 1174 | 571 | 567 | 23 / 515 / 33 | 12×128B | 12 lines；DC size 44×8 + 40×4 |

BLOCK=64/128 的 SIMD gather 与已有关系 `554 + 1.0 × BLOCK` 完全一致
（618、682）；BLOCK=32 的 571 与 586 差 15 cycle，SIMT BLOCK=64 的 464
也低于 BLOCK=32 的 546，属于 CAModel 单次 run / 地址相关的波动。
本轮补测只用于确认 `stride=3` 的路径和量级，不作为新的拟合点。

其中 SIMT `stride=3` 每个 128B line 放 10–11 个 f32，所以 line 内有效字节是
40B 或 44B；BIU 仍然读整 128B。补测产物在：

- `results/block32_s3/{launch_manifest.json,parsed.json,parsed.csv,run.log}`
- `results/block64_s3/{launch_manifest.json,parsed.json,parsed.csv,run.log}`
- `results/block128_s3/{launch_manifest.json,parsed.json,parsed.csv,run.log}`
- `results/strided_load_summary.{csv,md}`：由 `python3 scripts/make_summary.py` 生成

复现命令：

```bash
cd <pre-explore>
BLOCK=32 STRIDES="3" MODES="simd simt_only" bash scripts/run_camodel.sh
python3 scripts/analyze_strided_camodel.py \
  results/block32_s3 -o results/block32_s3/parsed.json \
  --output-csv results/block32_s3/parsed.csv
# BLOCK=64/128 同理
```

##### 指令路径示意

```text
SIMD:  tl.load -> MTE2 MOV_SRC_TO_DST_ALIGNv2 (GM->UB) -> VF reduce/use
                 stride=1/2: wide transaction
                 stride>=3: BLOCK x 4B gather command

SIMT:  tl.load -> SIMT_LDG -> SIMT 128B DCache -> BIU 128B line read
                 DC 合并同一 line 内 lane，但 line 利用率随 stride 下降
```

##### SIMT_template：mixed local SIMT 模板路径

- `compile_mode=simd_simt` 的 Route Model 选中 local SIMT scope 后，rank1..3、
  静态非 2 次幂 `stride>=3` 的 `tt.load/store` 会走
  `ascend.stride_load/store -> triton_stride_load/store` 模板调用。
- 它既不是纯 SIMD，也不是 `simt_only` 的 pure-SIMT；模板内部固定
  1024 threads，每个 thread 以 scalar GM 访问循环处理元素。
- 本轮白盒触发条件、半白盒特征、实测数据、拟合公式和误差见 §4。
- 与 pure-SIMT 的关键区别：pure-SIMT 走 `SIMT_LDG` + 128B DCache；
  template 路径的线程/loop 结构和成本不同，不能复用 v8 pure-SIMT 系数。

---

### 1.2 实测性能与数据重新拟合

#### 1.2.1 写了什么脚本

| 脚本 | 作用 |
|---|---|
| `scripts/simd_one_load_syscnt.py` | SIMD single cold load 的真卡 syscnt / Event 测量 |
| `scripts/frequency_retry_probe.py` | 频率感知的 SIMD single-load 测量：记录调频、no-load 参考、异常重试 |
| `scripts/simt_frequency_retry_probe.py` | SIMT rotate-loop Event-slope 测量（旧 v2/v3 sweep） |
| `scripts/simt_frequency_retry_probe_aligned.py` | v4 page-aligned SIMT rotate-loop Event-slope 测量：`stride=1..12` dense + 稀疏 stride |
| `scripts/syscnt_event_compare.py`、`scripts/syscnt_event_compare_aligned.py` | Event 单发 / loop 口径实现（旧版 / v4 aligned 版） |
| `scripts/retest_strided_simt_cases.py` | v4 5-allocation retest 驱动 |
| `scripts/merge_v4_retest.py` | 合并主 retest 和缺失 stride retest，生成 `results/v4_retest_full_23.json` |
| `scripts/build_v4_numwarps_dataset.py` | v4 dataset 构建：120ns floor、median/p25、flagged 规则，输出 1288 行 `results/model_v4_numwarps/dataset.csv` |
| `scripts/evaluate_strided_load_v7_on_v4.py` | 在 1288 行 v4 dataset 上评估旧 v7 SIMT load predictor，输出 by block / W / stride 误差 |
| `scripts/compare_strided_load_v7_v8_on_v4.py` | 合并 v7/v8 逐点误差，输出 `v7_vs_v8_group_metrics.csv` |
| `scripts/fit_strided_model_v8.py` | v8 半白盒公式拟合：raw 域 NNLS、LOBO/LOWO、分组误差、模型 JSON/errors 输出 |
| `scripts/predict_strided_load_v8.py` | v8 predictor + `--verify`，直接从 model JSON 读取 terms/coeff |
| `scripts/fit_strided_model_v7_simple.py` | v7 拟合脚本（历史基线） |
| `scripts/predict_strided_load_v7_simple.py` | v7 predictor，内置 `--verify` 逐点复算模型与误差表 |
| `scripts/run_v4_aligned_23.sh`、`scripts/run_v4_retest_missing.sh`、`scripts/run_v4_followup.sh` | v4 批量 driver |

数据规模：

数据规模：

| 场景 | 点数 | block | stride | num_warps | target |
|---|---:|---|---|---|---|
| SIMD wide | 24 | 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 1, 2 | 1（SIMD 与 num_warps 无关） | ns |
| SIMD gather | 504 | 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 3..32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256 | 1（SIMD 与 num_warps 无关） | ns |
| SIMT（旧 v2/v3 baseline） | 344 | 32, 64, 128, 256 | 1..12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256 | 1, 2, 4, 8, 16, 32, 64 | ns/iteration |
| SIMT（v4 全量，本版最终） | **1288** | **4, 8, 16, 32, 64, 128, 256, 512** | **1..12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256（23 个）** | **1, 2, 4, 8, 16, 32, 64（7 个）** | ns/iteration |

v4 SIMT 数据是 8 blocks × 7 num_warps × 23 strides 的唯一 case 网格，来自
`results/v4_retest_full_23.json`（主 retest + stride 20/24/40/48/96/192 缺失 retest），
每个 case 有 5 次独立 allocation 的 Event-slope 样本。数据清理规则：

- 丢弃 `< 120 ns` 的 allocation 样本（实测合法 target 的物理下限约 130 ns）；
- 丢弃非正样本；
- `target_ns` = 清洗后样本 median；`target_p25_ns` = 清洗后样本下四分位；
- 保留 `target_min_ns / target_max_ns / target_spread / n_samples / n_dropped`；
- `target_spread > 1.4`、`n_samples < 3` 或 `n_dropped > 0` 的行写入
  `results/model_v4_numwarps/flagged_discordant.csv`，只标记，不直接删除。

1288 行里 flagged 共 110 行：其中 `target_spread > 1.4` 97 行，
同时有 dropped 样本 7 行，仅 dropped 4 行，`n_samples < 3` 2 行。
本版主模型使用全部 1288 行 + `target_ns`（median）；`target_p25_ns` 和
exclude-flagged（1178 行）作为鲁棒性对照。exclude-flagged 时 in-sample MAPE
从 16.18% 降到 14.85%、max 从 94.46% 降到 80.92%，说明部分 tail 来自
allocation-spread 较大的 flagged 行；但主模型仍保留全部 1288 行，不直接删除。

拟合目标函数（全部在 raw target 域求解）：

```text
min Σ w_i (c0 + Σ c_j f_ij - T_i)^2,   c_j >= 0
```

`w_i = T_i^-alpha`，`alpha ∈ {1.0, 1.5, 2.0}` 在候选选择中比较。最终主模型
`alpha = 2.0`（偏相对误差，避免大 target 主导），公式本身不再出现任何
`log()` 或 `T_i^-alpha` 项。

#### 1.2.2 拟合公式和解释

模拟结果分三条路径。

##### SIMD wide（stride=1/2）

```text
stride_minus_1 = stride - 1
large_tile     = 1 if block > 64 else 0
tail_elems     = max(0, block - 64)

T_wide(ns) = 271.3729778488
           + 40.7987266013 * stride_minus_1
           + 84.4296207308 * large_tile
           +  0.0705164064 * tail_elems
```

| 项 | 物理含义 | 为什么增加时间 |
|---|---|---|
| 常数 | 一次 cold-line MTE2 wide-load 的固定启动/issue/retire | 与 block、stride 无关的 cold 开销 |
| `stride-1` | stride=2 的 lane 地址步长变成 8B | 不再是 unit-stride，MTE2 宽事务拆分/节拍变差 |
| `block>64` | block 进入 large-tile 路径 | f32 block>64（>256B）后固定多付启动与排队成本 |
| `max(0, block-64)` | 超过 64 元素后的搬运量 | 每多一个 f32 元素多搬 4B，约 17.6 ps/byte |

##### SIMD gather（stride>=3）

```text
counts[k] = #{ i in [0, block) : ((i*stride*4)//2048) % 8 == k }
bank_pairs = Σ_k C(counts[k], 2)
bank_worst = max_k counts[k] - 1
page_cross = floor( ((block-1)*stride*4) / 4096 )

T_gather(ns) = 373.2980095452
             + 0.8912079754 * block
             + 0.0140662765 * bank_pairs
             + 1.3008673227 * bank_worst
             + 0.8766837200 * page_cross
```

| 项 | 物理含义 | 为什么增加时间 |
|---|---|---|
| 常数 | per-element MTE2 gather 路径的 cold 启动 | 与 tile 大小无关的固定成本 |
| `block` | 每个元素一条 4B command | command 数随 block 线性增加 |
| `bank_pairs` | 同一 2KB/8-bank bucket 的 command 冲突 pair | 冲突导致排队/仲裁 |
| `bank_worst` | 最拥塞 bucket 的额外 command 数 | 决定尾部 drain 时间 |
| `page_cross` | span 跨越的 4KB 页数 | 每跨一页多一次 page 级开销 |

##### SIMT v8 半白盒公式（v4 全量，最终版本）

半白盒事实：

```text
W      = num_warps
E      = block / (32 * W)                 # 每个 thread 的逻辑元素数
dup    = max(0, 1/E - 1)                  # underfilled warp 的复制倍数
L      = block                            if stride*4 >= 128
         floor(((block-1)*stride*4)/128)+1 otherwise  # distinct 128B lines
K      = W * max(1, ceil(block/(32*W)))    # SIMT_LDG warp instructions
span   = (block - 1) * stride * 4
cross  = 1 if span >= 4096 else 0         # 是否跨 4KB page
```

最终主模型（`target_ns`，median；`alpha = 2.0`）：

```text
T_simt_v8(ns/iteration)
  = 146.7723
  + 2.12617 * min(L, 32)
  + 0.292236 * max(0, L - 32)
  + 0.00628591 * dup * L * min(L, 128)
  + 8.83530 * cross * min(L, 32)
  + 0.459865 * max(0, 4 - W) * L
```

该版本是在完整 v8 6 项基础上的“简化 + 精度增强”：

- 删去 `max(0, W*L - 3072)`：完整 v8 中该项系数仅 0.0212 ns，
  在 1288 行上的 CV MAPE 贡献为负，且会让 LOBO max 从 94.3% 恶化到 149.6%；
- `dup` 饱和 cap 由 64 改为 128：项数和形式不变，但 CV MAPE 从 17.46% 降到 16.29%，
  CV max 从 149.6%/97.6% 降到 94.3%/94.3%；
- 增加低 warp 并行度项 `max(0, 4-W)*L`：将 W1 的 MAPE 从 23.4% 降到 19.4%，
  且 LOWO（留一 W 交叉验证）从 17.59% 降到 16.51%。

鲁棒 `target_p25_ns` 对照模型系数几乎一致：
`145.9623 / 2.16151 / 0.301224 / 0.00614283 / 8.69258 / 0.457416`，
说明公式对 allocation 选择和 target 统计口径不敏感。
`K` 和单独的 `dup` 项在候选 NNLS 中被压成 0，未进入最终公式。

| 项 | 物理含义 | 为什么增加时间 |
|---|---|---|
| 常数 146.7723 ns | 一次 SIMT rotate iteration 的固定 LDG 启动、控制、首个 128B line fill 下限 | 与 block/W/stride 无关的基础成本 |
| `min(L,32)` × 2.12617 | 前 32 条 distinct 128B line 的 fill/queue 工作量 | 32 条 128B line = 一个 4KB page 的 line 数；未饱和区间每条 line 线性计费 |
| `max(0,L-32)` × 0.292236 | 超过 32 条后的 line 工作量 | 进入 overlap 区间后边际成本降到前 32 条约 1/7 |
| `dup*L*min(L,128)` × 0.00628591 | underfilled warp 复制的 line 请求数，最多计到 128 条 line | warp underfill 时同一 tile 的 line 请求被复制；复制压力在 16KB line 工作量后饱和 |
| `cross*min(L,32)` × 8.83530 | 跨 4KB page 时，最多一个 page 的 line 请求承担 page-walk / TLB 额外成本 | 小 block 高 stride 只碰少量 line 时避免固定大惩罚；大 tile 跨页时按 page 内 line 数收费 |
| `max(0,4-W)*L` × 0.459865 | W<4 时每缺一个 warp、每条 line 的额外 latency 暴露 | 少于 4 个 warp 组无法覆盖 line fill 延迟；该阈值是经验参数，需更多 W 点验证 |

所有系数 `>= 0`；最终公式没有 `log()`，没有 `x^2/x^3` 之外的项。
`dup*L*min(L,C)` 仍是截断后的二次交互，而不是完整二次项。

#### 1.2.3 v7 baseline 与 v8 error 对比

**旧 v7 SIMT predictor 与最终 v8 在 1288 行 v4 dataset 上的全量 in-sample 误差：**

| 指标 | v7（旧） | v8（target_ns 主模型） | v8（target_p25 对照） |
|---|---:|---:|---:|
| n | 1288 | 1288 | 1288 |
| MAPE | 33.25% | **15.23%** | **14.44%** |
| p50 | 18.17% | **9.73%** | **9.50%** |
| p90 | 69.44% | 37.55% | 35.85% |
| p95 | 84.41% | 59.55% | 56.13% |
| max | 1253.94% | **94.28%** | **83.79%** |
| bias | +6.82% | −5.39% | −4.72% |
| RMSE | 82.82% | 23.21% | 21.72% |

**v8 交叉验证（v4 全量，pooled out-of-fold prediction）：**

| 口径 | MAPE | p50 | p90 | p95 | max | bias | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_ns, LOBO | 16.08% | 10.26% | 40.15% | 59.87% | 94.27% | −5.31% | 23.88% |
| target_ns, LOWO | 16.51% | 10.50% | 49.28% | 63.18% | 94.28% | −6.06% | 24.98% |
| target_p25, LOBO | 15.29% | 10.15% | 37.63% | 57.32% | 84.02% | −4.65% | 22.45% |
| target_p25, LOWO | 15.64% | 10.45% | 44.54% | 59.45% | 83.76% | −5.40% | 23.42% |
| exclude flagged, target_ns, LOBO (1178) | 14.80% | 9.98% | 35.93% | 54.20% | 76.61% | −4.22% | 21.31% |
| exclude flagged, target_ns, LOWO (1178) | 15.36% | 10.38% | 39.82% | 58.53% | 87.71% | −4.85% | 22.77% |

**按 block：** v7 最差是 b8（MAPE 68.4%，max 1253.9%）和 b512（43.0%，max 195.3%）； 最终 v8 分别降到 14.2%/94.3% 和 25.3%/75.1%。b64/b128 也降到约 10% 量级。

| block | v7 MAPE | v7 max | v8 MAPE | v8 max | v8 bias |
|---:|---:|---:|---:|---:|---:|
| 4 | 35.2% | 234.6% | 9.1% | 80.1% | +3.3% |
| 8 | 68.4% | 1253.9% | 14.2% | 94.3% | −3.0% |
| 16 | 31.6% | 255.4% | 16.8% | 82.3% | −14.0% |
| 32 | 24.4% | 85.4% | 18.2% | 85.9% | −14.4% |
| 64 | 16.2% | 59.5% | 9.8% | 67.0% | −4.0% |
| 128 | 20.9% | 60.1% | 10.9% | 64.9% | +4.7% |
| 256 | 26.2% | 73.8% | 17.5% | 61.6% | −3.8% |
| 512 | 43.0% | 195.3% | 25.3% | 75.1% | −12.0% |

**按 num_warps：** v7 在大 W 和小 W 都差（W1 30.1%、W2 32.0%、W32 46.2%、W64 63.1%）； 最终 v8 在 W1/W2 为 19.4%/22.7%，W8/W16 约 9.5%，W32/W64 为 12.9%/20.7%。

| W | v7 MAPE | v7 max | v8 MAPE | v8 max | v8 bias |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.1% | 83.7% | 19.4% | 75.1% | −11.3% |
| 2 | 32.0% | 94.3% | 22.7% | 94.3% | −13.2% |
| 4 | 19.5% | 165.7% | 11.9% | 85.9% | −6.7% |
| 8 | 16.8% | 338.9% | 9.5% | 62.8% | −0.7% |
| 16 | 24.9% | 663.1% | 9.5% | 67.1% | +0.8% |
| 32 | 46.2% | 1174.0% | 12.9% | 63.3% | +5.0% |
| 64 | 63.1% | 1253.9% | 20.7% | 76.0% | −11.6% |

**按 stride：** v7 在 stride192/256 分别为 92.7%/105.3%；最终 v8 降到 16.3%/30.7%， stride 1–128 的 MAPE 都在 11%–20% 区间。完整分组文件：

```text
results/model_v4_numwarps/v7_eval/v7_by_block.csv
results/model_v4_numwarps/v7_eval/v7_by_num_warps.csv
results/model_v4_numwarps/v7_eval/v7_by_stride.csv
results/model_v4_simple/group_metrics_by_block.csv
results/model_v4_simple/group_metrics_by_num_warps.csv
results/model_v4_simple/group_metrics_by_stride.csv
results/model_v4_simple/v7_vs_v8_group_metrics.csv
```

**误差解释与 limitation：**

- v7 的 `min(L,64)` 外推不足，`dupL` 无法表达 underfilled warp 的
  “复制倍数 × line 数”交互，导致 b8/s256 等点误差超过 1000%。
- 最终 v8 的三段核心是：分段 line 工作量、截断复制交互、page-cross 交互；
  低 warp 项把 W1/W2 的并行度缺口单独定价。
- 残余最大误差仍集中在 `W=1/2`、小 block 高 stride，以及 b512 单 warp 大 span
  的 bank/set contention tail；这些 target 偶发 800–3000 ns，而公式给 150–600 ns。
  当前 2KB/8-bank、32KB/bank、page_count 等粗粒度代理特征在 NNLS 中均被压成 0，
  说明需要更精细的真实 set/sector 冲突事实，而不是继续加同类 proxy。
- `max(0,4-W)` 的阈值 4 是经验参数：现有 W 网格是 1/2/4/8/16/32/64，
  没有 W=3/6/12 点；后续需要补点验证阈值和系数可迁移性。

##### SIMT_template load 实测

- 测量协议：rotate-loop Event slope，`ns/iteration`；stride 只取静态非 2 次幂。
- 半白盒模型（in-sample）：
  ```text
  T_load = 211.3874
         + 1.64952 * L
         + 0.38442 * bucket_worst_32k16
         + 7.91975 * mean_warp_lines
         + 0.39463 * tail_elems
  ```
- n=92，in-sample MAPE 8.39%，p50 5.96%，p90 20.10%，p95 27.80%，max 40.70%。
- template / pure-SIMT median ratio ≈1.92×（详见 §4.6）。
- 完整数据、公式、误差和图见 §4.4–§4.6。

指标口径（相对误差定义为 `pred / target - 1`）：

- **MAPE**：`mean(|pred/target - 1|)`；**p50**：绝对百分比误差中位数；**max**：最坏点。
- p90/p95/bias/RMSE 在 `metrics_summary.csv` 与每行 `errors_all.csv` 中可复算。

#### 1.2.4 产物、复现与 predictor

模型/数据文件：

```text
results/model_v4_numwarps/dataset.csv
results/model_v4_numwarps/flagged_discordant.csv
results/v4_retest_full_23.json
results/model_v4_simple/model_v4_simple.json
results/model_v4_simple/metrics_summary.json
results/model_v4_simple/metrics_summary.csv
results/model_v4_simple/errors_all.csv
results/model_v4_simple/group_metrics_by_block.csv
results/model_v4_simple/group_metrics_by_num_warps.csv
results/model_v4_simple/group_metrics_by_stride.csv
results/model_v4_simple/cv_candidates.csv
results/model_v4_simple/cv_lobo_folds.json
results/model_v4_simple/cv_lowo_folds.json
results/model_v4_simple/v7_vs_v8_group_metrics.csv
results/model_v4_simple_p25/model_v4_simple.json
results/model_v4_simple_p25/errors_all.csv
```

复现命令：

```bash
cd third_party/ascend/costmodel/profiles/microbench/data_provider/strided_ldst/pre-explore

# 1) 从已合并的 retest 重建 v4 dataset（不重跑板卡测量）
python3 scripts/build_v4_numwarps_dataset.py \
  --out results/model_v4_numwarps/dataset.csv \
  --retest results/v4_retest_full_23.json

# 2) 旧 v7 baseline 误差
python3 scripts/evaluate_strided_load_v7_on_v4.py

# 3) v7/v8 分组误差合并
python3 scripts/compare_strided_load_v7_v8_on_v4.py

# 4) v8 主模型（target_ns, median）
python3 scripts/fit_strided_model_v8.py \
  --dataset results/model_v4_numwarps/dataset.csv \
  --outdir results/model_v4_simple \
  --target-col target_ns --alphas 1.0 1.5 2.0

# 5) 鲁棒 target_p25 对照
python3 scripts/fit_strided_model_v8.py \
  --dataset results/model_v4_numwarps/dataset.csv \
  --outdir results/model_v4_simple_p25 \
  --target-col target_p25_ns --alphas 1.0 1.5 2.0

# 6) v8 predictor 自校验
python3 scripts/predict_strided_load_v8.py --verify
python3 scripts/predict_strided_load_v8.py \
  --model results/model_v4_simple_p25/model_v4_simple.json \
  --errors results/model_v4_simple_p25/errors_all.csv --verify
```

单点预测（可直接用 `model_v4_simple.json` 的系数代入）：

```text
T_simt_v8 见 1.2.2；输入为 (block, stride, num_warps)。
```

---

## 2. Strided `tl.store`：白盒探索与实测重拟合

> Store 与 load 分开建模。SIMD store 只有 `stride=1` 走宽 MTE3 事务；
> `stride>=2` 是 per-element gather 路径。所有脚本同样在 `scripts/` 下。

### 2.1 白盒模型探索

#### 2.1.1 写了什么脚本

| 脚本 | 作用 |
|---|---|
| `scripts/strided_store_probe.py` | 最小 shaped strided store 探针：`tl.store(out + arange*STRIDE, value)`，无 load；强 SIMD / 强 SIMT |
| `scripts/run_store_camodel.sh` | `msopprof simulator` 驱动，输出 launch manifest 与编译产物 |
| `scripts/analyze_strided_store_camodel.py` | 解析 MTE3 issue/retire、SIMT LSU、SIMT DC TagRam、BIU `bwif` send/ack 等 |
| `scripts/summarize_strided_store_camodel.py` | 汇总 BLOCK=32/64/128 的 CAModel 结果 |

矩阵：BLOCK `32/64/128` × stride `1 2 4 8 16 32 64 128 256` ×
`simd/simt_only`，共 54 cases。原始 OPPROF 保存在
`63-workspace/14-strided-ldst-pre-explore/strided_store_camodel_raw.tgz`；
parsed 结果在 `results/store_camodel_b32|b64|b128/`。

#### 2.1.2 CAModel 看到了什么

**SIMD store 走 MTE3 `MOV_SRC_TO_DST_ALIGNv2`（UB→GM）**

- `stride=1` 时是宽事务：BLOCK=32 = 1×128B，BLOCK=64 = 1×256B，BLOCK=128 = 1×512B。
- `stride>=2` 时直接退化为 **BLOCK 条 4B write command**，没有 load 里 stride=2 的
  256B 宽事务。因此 store 的 SIMD 分支不能复用 load 的 wide/gather 分界。

BLOCK=32 白盒 store active window（MTE3 push→retire，raw CAModel cycle）：

| stride | SIMD store active | SIMD BIU | SIMT store active | SIMT lines | TagRam WRITE size |
|---:|---:|---|---:|---:|---:|
| 1 | 978 | 1×128B | 415 | 1 | 128B |
| 2 | 988 | 32×4B | 505 | 2 | 64B |
| 4 | 1089 | 32×4B | 530 | 4 | 32B |
| 8 | 1126 | 32×4B | 584 | 8 | 16B |
| 16 | 1008 | 32×4B | 590 | 16 | 8B |
| 32 | 1019 | 32×4B | 639 | 32 | 4B |
| 64 | 992 | 32×4B | 649 | 32 | 4B |
| 128 | 1046 | 32×4B | 623 | 32 | 4B |
| 256 | 1099 | 32×4B | 640 | 32 | 4B |

**SIMT store 走 `SIMT_STG` + 128B line write**

- DC 按 128B line 合并，`TagRam` 的 WRITE size 随 stride 从 128B 降到 4B；
  BIU 仍按 128B line 写。
- 当前 dump 未观察到 write-allocate read。
- 多条 STG 的 issue/retire 明显 overlap；BLOCK=128 时 4 条 STG 的总窗口只有约 1.17k
  cycle，不能按 `K × 单条 latency` 放大。

#### 2.1.3 SIMT_template store

- mixed local SIMT scope 内、rank1..3、静态非 2 次幂 `stride>=3` 的 shaped
  `tt.store` 会走 `triton_stride_store` 模板，而不是 `SIMT_STG`。
- 模板 1D store 同样是固定 1024 threads，每个 thread 执行
  `dst[storeLower + i0*stride] = src[i0]` 的 scalar 写循环。
- 本轮白盒触发条件和实测协议见 §4.1–§4.3；拟合公式与 in-sample 误差见
  §4.4–§4.5。

### 2.2 实测性能与数据重新拟合

#### 2.2.1 target 口径

按任务要求先尝试了三种 SIMD single-cold-store target：

1. `torch_npu.profiler` store−no-store `Duration(us)` delta：
   单次 kernel duration ≈0.8µs、run 间抖动同量级，median delta 常为 0–0.1µs，无法分辨
   0.1–0.6µs 的 store。
2. `torch.npu.Event` 配对 batch delta：
   host launch/调度噪声远大于单 store 成本，batch 间 target 抖动数倍。
3. 一个很长 kernel 的 Event slope：
   部分 stride/W 形态会把 AICore 推离 boost，slope 在 2000/8000/32000 iters 间非物理跳变。

最终采用与 load SIMT 一致的 **rotate-loop Event 目标**：

```text
每个 case: 在 2000 和 8000 iteration 的 rotate store loop 上重复测量，
target = 所有重复里 min(event_time / iteration)   [ns/iteration]
```

- `min` 用来拒绝掉频/调度变慢的样本；所有 case 前后都有 ALU witness，
  并用独立的长 ALU kernel 做 in-flight `npu-smi` 频率检查（1650 MHz）。
- Event 窗口内不运行 `npu-smi`，避免 device lock 污染计时。
- target 语义是 **boost 下 rotate loop 的每次 shaped-store 吞吐成本**，
  不是 warm single-line replay；这一点在 dataset 和 model JSON 中明确记录。

#### 2.2.2 测量脚本、网格与数据量

| 脚本 | 作用 |
|---|---|
| `scripts/store_measure_probe.py` | 最终测量 probe：2000/8000-iter rotate loop、ALU ramp/witness、Event timing |
| `scripts/precompile_store_probe.py` | 并行预编译全部 (mode, block, stride, W) Triton 变体，避免测量时等待 JIT |
| `scripts/run_store_measure_simd.sh` | SIMD 网格 driver |
| `scripts/run_store_measure_simt.sh` | SIMT 网格 driver |
| `scripts/build_store_dataset.py` | 从 raw JSON 生成 `results/model_store_v1/dataset.csv` 和全部派生特征 |
| `scripts/fit_strided_store_v1.py` | raw-domain NNLS 拟合，输出 model/metrics/errors |
| `scripts/predict_strided_store_v1.py` | 最终 predictor + `--verify` |
| `scripts/make_store_plots.py` | parity / error histogram / error-vs-stride / error-vs-target 图 |

数据规模：

| 场景 | 点数 | block | stride | num_warps | target |
|---|---:|---|---|---|---|
| SIMD wide | 12 | 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 1 | 1（SIMD 与 num_warps 无关） | ns/iteration |
| SIMD gather | 88 | 32, 64, 128, 256 | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256 | 1（SIMD 与 num_warps 无关） | ns/iteration |
| SIMT | 644 | 32, 64, 128, 256 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256 | 1, 2, 4, 8, 16, 32, 64 | ns/iteration |

三个分支合计 744 cases，全部 valid；raw JSON：
`results/model_store_v1/raw/board_store_simd.json`、
`results/model_store_v1/raw/board_store_simd_wide_more.json` 与
`results/model_store_v1/raw/board_store_simt.json`。

#### 2.2.3 拟合公式与逐项解释

拟合目标函数（raw target 域，非负）：

```text
min Σ (c0 + Σ c_j f_j - T_i)^2,  c_j >= 0
```

求解器为 Lawson-Hanson NNLS；没有对 target 取 log，公式中没有次方/负数项。

##### SIMD wide（`stride == 1`）

```text
T_simd_wide(ns/iteration)
  = 83.6254
  + 0.02730 * block
```

| 项 | 单位 | 物理含义 | 为什么增加时间 |
|---|---|---|---|
| 常数 83.6254 | ns | MTE3 wide path 的固定 issue/启动/retire 开销 | 与 tile 大小无关的 cold 启动 |
| `block` | element | wide MTE3 一次搬运的 f32 元素数 | 每个元素多搬运 4B，约 0.0273ns/element |

##### SIMD gather（`stride >= 2`）

```text
T_simd_gather(ns/iteration)
  = 4.9432
  + 27.0195 * line_request_size
  + 0.04556 * bank_pairs
  + 0.45431 * worst_G32768
```

其中（aligned base 假设）：

```text
line_request_size = 4B                           if stride_bytes >= 128
                    min(128, (128 // stride_bytes) * 4)  otherwise
bank_pairs = Σ_b C(n_b, 2),
             n_b = #{ i : ((i*stride_bytes)//2048) % 8 == b }
worst_G32768 = max_g n_g - 1,
             n_g = #{ i : ((i*stride_bytes)//32768) % 16 == g }
```

| 项 | 单位 | 物理含义 | 为什么增加时间 |
|---|---|---|---|
| 常数 4.9432 | ns | per-element MTE3 gather path 固定启动 | 与地址形态无关的 cold 启动 |
| `line_request_size` | byte/128B-line | 同一 128B line 内有效的写入字节数，是地址密度 proxy | 同一 line 内 4B command 越密集，MTE3 小命令合并/同 sector 串行压力越大 |
| `bank_pairs` | pair | 落在同一 2KB（8-bank）bucket 的元素 pair 数 | 同 bank 命令排队/仲裁，pair 越多串行越重 |
| `worst_G32768` | element | 单个 32KB 区域内最多写入元素数减一 | 地址过度集中到同一 row/region 造成 bank/row 热点 |

##### SIMT（`SIMT_STG` + 128B line）

```text
E        = block / (32 * W)
K_stg    = W * max(1, ceil(block / (32 * W)))
L        = block                              if stride_bytes >= 128
           floor((block-1)*stride_bytes/128)+1 otherwise
line_request_size = 同 SIMD 定义

T_simt(ns/iteration)
  = 12.6818
  + 0.65082 * K_stg
  + 1.23424 * L
  + 0.15181 * line_request_size
```

| 项 | 单位 | 物理含义 | 为什么增加时间 |
|---|---|---|---|
| 常数 12.6818 | ns | SIMT loop 内一次 shaped store 的固定 issue/控制/首命令开销 | 与 shape 无关的基础成本 |
| `K_stg` | warp instruction | 一个 program 的 STG warp 指令数 | 每条 STG 多一次 LSU issue/retire |
| `L` | line | tile 覆盖的不同 128B line 数 | 每条 distinct line 增加 BIU write/drain 的边际时间 |
| `line_request_size` | byte/line | 每条 line 的有效写入字节数 | 同一 line 内有效元素越多，line 内小步长命令压力越大 |

所有系数 `>= 0`；推理时没有 `log()`，没有 `x^2/x^3`，没有减法项。
注意：所有 collision/line 派生特征都建立在 **aligned base** 假设上；
misalignment 专项仍按任务要求暂缓，未混入本轮拟合。

#### 2.2.4 全量 in-sample 误差

| 分支 | n | MAPE | p50 | max |
|---|---:|---:|---:|---:|
| SIMD wide | 12 | 3.44% | 3.10% | 7.46% |
| SIMD gather | 88 | 10.78% | 7.14% | 40.20% |
| SIMT | 644 | 13.68% | 10.36% | 64.56% |
| 合计 | 744 | 13.17% | 9.71% | 64.56% |

p90 / p95 / bias / RMSE 都在
`results/model_store_v1/metrics_summary.json/.csv` 中，README 不展开。

误差图（脚本 `scripts/make_store_plots.py`，输出目录
`results/model_store_v1/plots/`）：

- `store_parity_loglog.png`：三组 target vs prediction 的 log-log parity；
- `store_error_hist.png`：signed relative error 直方图；
- `store_error_vs_stride.png`：误差随 stride 散点；
- `store_error_vs_target.png`：误差随 target 散点。

#### 2.2.5 predictor 校验

```bash
python3 scripts/fit_strided_store_v1.py
python3 scripts/predict_strided_store_v1.py --verify
```

当前 `--verify` 结果：

```text
max |predict - errors.pred_ns| = 2.274e-13
max predictor relative diff   = 2.884e-16
max recomputed metric diff    = 1.776e-14 percentage points
coefficients non-negative, no power/log feature names
VERIFY OK
```

单点预测示例：

```bash
python3 scripts/predict_strided_store_v1.py simd 128 16
python3 scripts/predict_strided_store_v1.py simt 128 16 --num-warps 4
```

#### 2.2.6 SIMT_template store 实测

- 测量协议同 §4.3：rotate-loop Event slope，`ns/iteration`，只取静态非 2 次幂
  stride。
- 半白盒模型（in-sample）：
  ```text
  T_store = 92.1451
          + 1.91721 * L
          + 0.23650 * bucket_worst_32k16
          + 6.25512 * mean_warp_lines
          + 111.92716 * iters_per_thread
  ```
- n=90，in-sample MAPE 9.26%，p50 6.27%，p90 21.89%，p95 27.47%，max 37.42%。
- template / pure-SIMT median ratio ≈3.26×（小 block 中位可达 10× 左右）。
- 完整数据、公式、误差和图见 §4.4–§4.6。


## 3. 所有脚本在 scripts/

所有 Python / Shell 脚本统一放在 `scripts/` 下（load / store / 诊断脚本都在这里）。

### 3.1 Load / 公共脚本

```text
# CAModel 白盒
scripts/strided_load_probe.py
scripts/run_camodel.sh
scripts/analyze_strided_camodel.py
scripts/analyze_simt_ldg_windows.py
scripts/extract_evidence.py
scripts/make_summary.py

# 真卡测量与数据集（历史 v2/v3，SIMD 仍沿用）
scripts/simd_one_load_syscnt.py
scripts/frequency_retry_probe.py
scripts/simt_frequency_retry_probe.py
scripts/syscnt_event_compare.py
scripts/build_v2_dataset.py
scripts/build_v3_numwarps_dataset.py
scripts/run_v2_simd_dense.sh
scripts/run_v2_simt_dense.sh
scripts/run_v2_more_blocks.sh
scripts/run_simt_numwarps_sweep.sh
scripts/run_simt_numwarps_dense.sh

# v4 SIMT page-aligned 测量 / retest / dataset
scripts/simt_frequency_retry_probe_aligned.py
scripts/syscnt_event_compare_aligned.py
scripts/retest_strided_simt_cases.py
scripts/merge_v4_retest.py
scripts/build_v4_numwarps_dataset.py
scripts/run_v4_aligned_23.sh
scripts/run_v4_retest_missing.sh
scripts/run_v4_followup.sh

# baseline / 最终拟合
scripts/evaluate_strided_load_v7_on_v4.py
scripts/fit_strided_model_v7_simple.py
scripts/fit_strided_model_common.py
scripts/predict_strided_load_v7_simple.py
scripts/make_v7_plots.py
scripts/fit_strided_model_v8.py
scripts/compare_strided_load_v7_v8_on_v4.py
scripts/predict_strided_load_v8.py
```

### 3.2 Store 脚本

```text
scripts/strided_store_probe.py
scripts/run_store_camodel.sh
scripts/analyze_strided_store_camodel.py
scripts/summarize_strided_store_camodel.py
scripts/store_measure_probe.py
scripts/precompile_store_probe.py
scripts/run_store_measure_simd.sh
scripts/run_store_measure_simt.sh
scripts/build_store_dataset.py
scripts/fit_strided_store_v1.py
scripts/predict_strided_store_v1.py
scripts/make_store_plots.py
```
## 4. v8 到 profile / C++ 的最终集成状态

> 本节记录 `feature/strided-load-store-costmodel-v4` 在服务器上完成 route-level
> A/B 后的最终状态。**v8 现在是唯一 SIMT strided-load 公式，v7 公式、C++
> 字段和 profile/schema 字段均已删除。**

### 4.1 最终公式

最终 v8 主模型（`target_ns`）系数（raw ns/iteration）：

```text
simt_load_intercept_ns           = 146.7722948208243
simt_load_min_line32_ns          =   2.126165744146226
simt_load_tail_line32_ns         =   0.29223594621358456
simt_load_dup_l_min_l128_ns      =   0.00628590518702342
simt_load_cross_min_l32_ns       =   8.83529747800243
simt_load_low_warp4_l_ns         =   0.45986549366866875
```

`stridedSimtLoadCostNs()` 现在只保留：

```text
E       = elements / (32 * W)
dup     = max(0, 1/E - 1)
L       = stridedLineCount(elements, strideBytes)
cross   = ((elements-1)*strideBytes) >= 4096
lowW4   = max(0, 4 - W)

T_v8 = intercept
     + min_L_32     * min(L, 32)
     + tail_L_minus_32 * max(0, L-32)
     + dupL_minL_128 * dup*L*min(L,128)
     + cross_minL_32 * cross*min(L,32)
     + lowW_4_L      * lowW4*L
```

`StridedMemoryProfile` 中不再有 `simtLoadFormulaVersion`、`simtLoadLegacyInterceptNs`
以及旧的 `simtLoadDupLineNs` / `simtLoadLineNs` /
`simtLoadWarpLineOverflowNs` / `simtLoadDupCrossNs`。

### 4.2 profile / schema

- 正式 profile：
  `david-v100-simd-simt-20261009-v30-strided-memory-v8`
- schema version：15
- `simt_load_formula_version` 与全部 v7 旧字段已从 profile/schema 删除；
- `simt_load_intercept_ns` 和五个 v8 字段成为必需解析字段，profile parser
  对缺失 v8 字段 fail-fast；
- `nanoseconds_to_system_cycles = 0.9889` 不变；
- SIMD load/store、SIMT store 公式本轮未改动。

### 4.3 route-level A/B 结果（服务器 `ascend-950pr-63`）

A/B 使用同一份 C++ 代码、同一批报告 workload，只切换临时 profile 的
`simt_load_formula_version`（v7=1，v8=2）。结果如下（单位：SYS_CNT cycle）：

| case | v7 decision | v8 decision | v7 strided SIMT load | v8 strided SIMT load | v7 all_simt total | v8 all_simt total |
|---|---|---|---:|---:|---:|---:|
| b8_s256_w1 | all_simt_only | all_simt_only | 207.8 | 244.0 | 603.4 | 639.6 |
| b8_s256_w2 | all_simd | all_simt_only | 272.2 | 241.9 | 667.8 | 637.6 |
| b8_s256_w3 | all_simd | all_simt_only | 336.5 | 239.9 | 732.2 | 635.5 |
| b64_s32_w1 | all_simd | all_simd | 495.9 | 588.6 | 905.3 | 998.0 |
| b64_s32_w2 | all_simd | all_simd | 495.9 | 559.5 | 905.3 | 968.9 |
| b128_s64_w1 | all_simd | all_simd | 495.9 | 694.4 | 924.9 | 1123.4 |
| b128_s64_w2 | all_simd | all_simd | 495.9 | 636.2 | 924.9 | 1065.2 |
| b512_s256_w1 | all_simt_only | all_simd | 495.9 | 1329.2 | 1066.9 | 1900.2 |
| b512_s256_w2 | all_simt_only | all_simd | 495.9 | 1096.4 | 1066.9 | 1667.4 |
| b512_s256_w3 | all_simt_only | all_simt_only | 495.9 | 863.6 | 1066.9 | 1434.5 |
| b512_s256_w6 | all_simt_only | all_simt_only | 495.9 | 630.7 | 1066.9 | 1201.7 |
| b512_s256_w12 | all_simt_only | all_simt_only | 495.9 | 630.7 | 1418.5 | 1201.7 |
| st_b32_s3_w32_g8 | all_simt_only | all_simt_only | 270.3 | 206.3 | 480.5 | 416.6 |
| st_b64_s3_w32_g4 | all_simt_only | all_simt_only | 270.3 | 206.3 | 480.5 | 416.6 |
| st_b128_s3_w32_g2 | all_simt_only | all_simt_only | 270.3 | 206.3 | 480.5 | 416.6 |
| st_b256_s3_w32_g1 | all_simt_only | all_simt_only | 270.3 | 206.3 | 496.3 | 432.4 |
| st_b64_s3_w32_g16（envelope 小 tile） | all_simt_only | all_simt_only | 495.9 | 510.5 | 897.5 | 912.1 |

A/B 结论：

1. 没有出现 mixed route 崩塌；所有 workload 的 legal candidate 集合不变。
2. route 变化只出现在 v7 已知大误差域：
   - `b8/s256 W2/W3`：v8 把 underfilled warp 的复制 line 成本从近线性
     v7 改为低 W 项后，SIMT 变便宜，route 从 all_simd 变为 all_simt；
     W1 两种版本都是 all_simt，形态一致。
   - `b512/s256 W1/W2`：v8 修正了 v7 对 512 条 line、单/双 warp 大 span 的
     严重低估，SIMT strided load 从 495.9 升到 1096–1329，route 变为
     all_simd；W3/W6/W12 仍保持 all_simt，low-warp 阈值 4 在这些点上
     route-insensitive。
   - 其余 `b64/b128` 大 stride 和全部 stride=3 小 tile 保持原 route。
3. 小 tile route-boundary case `st_b64_s3_w32_g16`：
   - v7 all_simt total = 897.5，与 envelope 报告中的 897.489 对齐；
   - v8 all_simt total = 912.1（+1.6%），route 不变，说明 v8 没有在小 tile
     上引入 route collapse。
4. 因此 route-level A/B 通过，正式 profile 已启用 v8，v7 兼容路径已删除。
   删除 v7 后，用同一批 17 个 case、正式 v8-only profile 重跑 report，
   与删除前临时 v8 profile 的 candidate totals / strided Stage SIMT load
   逐项对比：`max candidate diff = 0`、`max strided-load diff = 0`，
   即删除 v7 没有改变 v8 打分。

### 4.4 UT / Python 验证

- C++ `SimdSimtCostModel`：48 tests passed（含 v8 golden、fact detection、
  profile parser fail-fast、double-count、isValid 边界）。
- C++ `CostModelPasses`：18 tests passed。
- `predict_strided_load_v8.py --verify`：VERIFY OK（target_ns 主模型，1288 行）。
- `predict_strided_load_v8.py --model results/model_v4_simple_p25/... --verify`：
  VERIFY OK（target_p25 对照模型）。

### 4.5 残余风险

- v8 仍是 aligned-base、32-bit element、正 stride、无 mask 的窄事实模型；
  misalignment / mask tail / runtime stride / non-4B dtype 未覆盖。
- `max(0, 4-W)` 阈值没有独立的 W=3/6/12 板卡拟合点；本轮 route A/B 只能
  证明 W=3/6/12（b512/s256）route 对阈值不敏感，不能证明公式绝对精度。
- b512 单/双 warp 大 span 的 bank/set contention tail 仍是已知高误差区；
  v8 把它从 v7 的低估改成更保守的高估，route 切换依据来自校准模型而非
  本轮新增板卡测量（任务禁止重跑测量）。
- route 变化（b8/s256 W2/W3、b512/s256 W1/W2）是基于 v8 校准结果的预期
  修正，但仍建议后续用 in-domain 板卡数据复查这些边界。

### 3.3 SIMT_template 脚本

```text
scripts/template_stride_common.py
scripts/template_stride_features.py
scripts/template_stride_load_probe.py
scripts/template_stride_store_probe.py
scripts/run_template_stride_measure_load.sh
scripts/run_template_stride_measure_store.sh
scripts/template_stride_path_check.py
scripts/build_template_stride_dataset.py
scripts/fit_template_stride_v1.py
scripts/predict_template_stride_v1.py
scripts/make_template_stride_plots.py
scripts/compare_template_vs_pure_simt.py
```

---

## 5. SIMT_template 路径：白盒探索与实测（`triton_stride_load/store`）

### 5.1 白盒探索：SIMT_template 路径怎么触发

后端源码：`StridedLoadStoreRewrite.cpp` / `TritonToLinalgPass.cpp`。

关键条件：

1. 只在 `compile_on_910_95=true` 的 950PR 目标上运行；
2. `simd_simt_template` 的 legacy 全局强制路径，或 `simd_simt` Route Model
   选中的 local SIMT `scope.scope`（`shouldUseSimtTemplate(op, false)`）；
3. rank 1..3、静态 shape、静态 last-axis stride：
   - `stride == 1` bail；
   - `stride == 2` bail（走 deinterleave）；
   - power-of-two `stride >= 4` bail（走 strided DMA）；
   - dynamic stride bail；
   - 只有 **静态非 2 次幂 stride >= 3** 进入
     `ascend.stride_load/store`，最终生成
     `call @triton_stride_load` / `call @triton_stride_store`。

**Python backend 的一个实测坑**：在本分支里只传
`compile_mode="simd_simt_template"` 时，`parallel_mode` 仍是默认 `"simd"`，
TTAdapter 最终落回 `memref.copy`。必须同时传：

```python
compile_mode = "simd_simt_template"
parallel_mode = "mix_simd_simt"
compile_on_910_95 = True
auto_simt_scope_mode = "off"
enable_auto_blockify = False
superblock_factor = 1
```

才能稳定复现 `call @triton_stride_load/store`。这与
`third_party/ascend/unittest/Conversion/950PR/TritonToLinalg/indirect_load_rewrite.mlir`
里 `triton-opt` 的 `force-simt-template=true` 路径等价。

IR 证据（compile-only，不 launch）：

- 脚本：`scripts/template_stride_path_check.py`
- 结果：`results/model_template_stride_v1/ir_evidence/template_stride_path_check.json`
- TTAdapter 片段：`results/model_template_stride_v1/ir_evidence/path_check_asm/*.ttadapter`
- 实测 case 的 TTAdapter：`results/model_template_stride_v1/ir_evidence/measure_asm/*.ttadapter`

关键校验（0 mismatch）：

- `stride=3/5/7/255`：包含 `call @triton_stride_load/store`；
- `stride=1/2/4/8/16/32/64/128/256`：不包含模板调用（保留 structured
  / `memref.copy` / deinterleave 路径）。

### 5.2 白盒探索：SIMT_template 半白盒特征

1D 模板源码：`SIMTStrideLoad.cpp` / `SIMTStrideStore.cpp`。

```text
STRIDE_LOAD_THREAD_NUM = STRIDE_STORE_THREAD_NUM = 1024

for (i0 = threadIdx.x; i0 < size; i0 += blockDim.x)
  load:  dst[i0 * dstStride0] = src[loadLower + i0 * stride]
  store: dst[storeLower + i0 * stride] = src[i0 * srcStride0]
```

在本轮 rank1、无 mask、`numel == BLOCK` 的探针里：

- `iters_per_thread = ceil(BLOCK / 1024)`；
- `active_threads = min(BLOCK, 1024)`，`active_warps = ceil(active_threads/32)`；
- 每个元素一次 scalar GM 访问，地址 `i * stride_bytes`（offset=0，base 对齐）；
- `loadSize == BLOCK`，不会跑 `simtStridePad1D`；
- 模板内部线程数固定 1024，与 Triton `num_warps` 无关（实测见 §4.4）。

`scripts/template_stride_features.py` 实现的候选特征：

| 特征 | 来源 |
|---|---|
| `L` | strided tile 覆盖的 distinct 128B line 数 |
| `line_elems_max/min` | 每个 128B line 内有效元素数 |
| `mean/min/max_warp_lines` | 一个 32-thread warp 访问涉及的 line 数 |
| `tail_elems` = `BLOCK % 1024` | 模板 1024-thread 循环尾部 |
| `stride_bytes` | `stride * 4` |
| `page_cross` | `floor(((BLOCK-1)*stride_bytes)/4096)` |
| `bucket_pairs_2k8` / `bucket_worst_2k8` | 2KB/8-bank 桶冲突 pair / worst |
| `bucket_worst_32k16` | 32KB/16 桶的最拥塞计数 |
| `num_warps_minus1` | control，验证模板线程数是否受外层影响 |

### 5.3 实测：SIMT_template 测量协议

脚本：

- `scripts/template_stride_load_probe.py`
- `scripts/template_stride_store_probe.py`
- `scripts/run_template_stride_measure_load.sh`
- `scripts/run_template_stride_measure_store.sh`

协议：

```text
rotate loop 一次 shaped strided load/store，
iters ∈ {2000, 8000}，每点重复 reps 次取最小 Event；
target = 对 (iters, min_event_ns) 做线性斜率 [ns/iteration]；
每次 measured launch 前跑 ALU busy burst 保持 boost；
case 后做 ALU witness；case 前后用 npu-smi 检查 Aicore Freq；
witness 不达标或 spread 过大的 attempt 标记 invalid。
```

矩阵（主拟合）：

- BLOCK：16、64、256、1024、2048（`tl.arange` 为 2 的幂）；
- STRIDE：3、5、6、7、9、11、13、15、17、21、25、31、40、48、63、96、
  129、192、255；
- 不含 stride=1/2/pow2（这些不会进入模板，只出现在 §4.1 的 boundary check）；
- load/store 分开测量；
- 每个 case 2 个 pass（reps=2/attempts=2 与 reps=3/attempts=1 各一轮），
  再加一批 in-sample 误差最差点的定向 rerun；dataset builder 对同一
  `(block,stride,num_warps)` 取所有 pass 中 valid+correct 的 **最小 target**。

`num_warps` control：

| path | block | stride | W=1 | W=2 | W=4 | W=8 | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| load | 32 | 3 | 251.4 | 254.2 | 251.2 | 252.7 | 1.2% |
| load | 32 | 17 | 300.3 | 302.8 | 301.4 | 304.7 | 1.5% |
| load | 1024 | 17 | 1478.2 | 1496.3 | 1487.7 | 1509.7 | 2.1% |
| load | 1024 | 255 | 2032.6 | 2053.7 | 2037.5 | 2115.9 | 4.1% |
| store | 32 | 3 | 225.0 | 226.9 | 222.3 | 223.3 | 2.1% |
| store | 1024 | 17 | 1522.4 | 1526.6 | 1500.8 | 1498.5 | 1.9% |
| store | 1024 | 255 | 2124.8 | 2116.1 | 2114.2 | 2121.5 | 0.5% |

结论：template target 与 `num_warps` 基本无关，模板内部 1024 threads 固定；
profile 拟合不需要 `num_warps` 项。

### 5.4 实测：SIMT_template 拟合公式

模型文件：

- `results/model_template_stride_v1/model_template_stride_load_v1.json`
- `results/model_template_stride_v1/model_template_stride_store_v1.json`

形式：

```text
T = intercept + Σ c_i * f_i,   c_i >= 0
target = ns/iteration (rotate-loop Event slope，min-of-reps/passes)
```

#### Load template

```text
T_load = 211.3874
       + 1.64952 * L
       + 0.38442 * bucket_worst_32k16
       + 7.91975 * mean_warp_lines
       + 0.39463 * tail_elems
```

| 项 | 物理含义 |
|---|---|
| intercept 211.4 ns | 单个 rotate iteration 内 template 调用固定开销 |
| `L` | distinct 128B line 工作量；每多一条 line 约 1.65 ns |
| `bucket_worst_32k16` | 32KB/16-bucket 最拥塞程度，捕捉地址 bank/region 热点 |
| `mean_warp_lines` | 每个 warp 的 line 数，捕捉 warp 级分散访问 |
| `tail_elems` | BLOCK 不是 1024 倍数时的尾线程/尾 wave |

#### Store template

```text
T_store = 92.1451
        + 1.91721 * L
        + 0.23650 * bucket_worst_32k16
        + 6.25512 * mean_warp_lines
        + 111.92716 * iters_per_thread
```

| 项 | 物理含义 |
|---|---|
| intercept 92.1 ns | template store 调用固定开销 |
| `L` | 写向 distinct 128B line 数 |
| `bucket_worst_32k16` | 32KB/16-bucket 写热点 |
| `mean_warp_lines` | warp 级写分散度 |
| `iters_per_thread` = `ceil(BLOCK/1024)` | 1024-thread 模板的内部 loop 次数；store 对该项最敏感 |

所有系数非负；无 log、无 x²/x³、无减法项。predictor：

```bash
python3 scripts/predict_template_stride_v1.py --verify
python3 scripts/predict_template_stride_v1.py load 256 3
python3 scripts/predict_template_stride_v1.py store 256 3
```

### 5.5 实测：SIMT_template in-sample 误差

本轮只报告 in-sample 拟合误差。

| path | n | MAPE | p50 | p90 | p95 | max | bias | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| load | 92 | 8.39% | 5.96% | 20.10% | 27.80% | 40.70% | 0.00% | 6.94% |
| store | 90 | 9.26% | 6.27% | 21.89% | 27.47% | 37.42% | -0.85% | 9.08% |

逐点误差：

```text
results/model_template_stride_v1/errors_load.csv
results/model_template_stride_v1/errors_store.csv
results/model_template_stride_v1/errors_all.csv
```

图：

```text
results/model_template_stride_v1/plots/parity_loglog.png
results/model_template_stride_v1/plots/error_hist.png
results/model_template_stride_v1/plots/error_vs_stride.png
results/model_template_stride_v1/plots/error_vs_block.png
```

结论：load / store in-sample MAPE 分别为 8.39% / 9.26%，满足 <10%–15%
目标。max 误差主要来自少数 small-block outlier 和共享板卡噪声，未被少量
非负特征完全吸收。

### 5.6 实测：SIMT_template 与 pure SIMT 对比

同一 `(block, stride, num_warps)` 上，分别调用现有 pure SIMT model 和 template v1：

```text
results/model_template_stride_v1/template_vs_pure_simt.csv
```

比较（template / pure-SIMT）：

| path | n | median ratio | mean ratio | min | max |
|---|---:|---:|---:|---:|---:|
| load | 92 | 1.92× | 2.79× | 1.20× | 7.82× |
| store | 90 | 3.26× | 5.15× | 1.73× | 11.69× |

按 block 看中位 ratio：

| block | load ratio | store ratio |
|---:|---:|---:|
| 16 | 1.82× | 10.63× |
| 64 | 1.41× | 6.33× |
| 256 | 1.85× | 2.88× |
| 1024 | 3.26× | 1.99× |
| 2048 | 5.47× | 1.82× |

结论：template 路径与 pure-SIMT **不能共用**系数，尤其 store 小 block
差一个数量级；load 在 1.2×–7.8× 内，也值得独立建模。

### 5.7 集成建议

本轮不改 C++，下一轮建议：

1. 在 `StageWorkload`/`MemoryAccessFacts` 中显式标记：
   ```text
   strided_template_load_elements
   strided_template_store_elements
   strided_template_rank
   strided_template_stride
   strided_template_is_static_nonpow2
   strided_template_path_eligible   # rank1..3 + local SIMT/fixed trigger + mask 条件
   ```
2. `mapWorkload()` 增加 `StridedLoweringPath`/`localScope` 维度：
   - `implementation.mode == SIMT && implementation.localScope` 且该 memory op
     满足静态非 2 次幂 stride 才用 template 公式；
   - mixed 的 `localScope=true` 不要落回现有 `simtLoad*/simtStore*` 公式；
   - whole-kernel `simt_only` 仍用 pure SIMT 公式。
3. profile 新增 `stage_resources.template_strided_memory` 字段（单位仍是
   SYS_CNT cycle，写入时用 profile 已有的 `nanoseconds_to_system_cycles` 乘 ns）：
   ```text
   simt_stride_template_load_intercept_ns
   simt_stride_template_load_L_ns
   simt_stride_template_load_bucket_worst_32k16_ns
   simt_stride_template_load_mean_warp_lines_ns
   simt_stride_template_load_tail_elems_ns

   simt_stride_template_store_intercept_ns
   simt_stride_template_store_L_ns
   simt_stride_template_store_bucket_worst_32k16_ns
   simt_stride_template_store_mean_warp_lines_ns
   simt_stride_template_store_iters_per_thread_ns
   ```
4. 测试：
   - synthetic `StageWorkload` + `StageImplementation{localScope=true}` 断言选
     template 系数；
   - mixed route + local SIMT scope + rank1 static stride=3 的 IR 测试断言
     `call @triton_stride_load/store`；
   - route report 中 pure-SIMT 与 template-SIMT 两条实现 cycle 可区分。

### 5.8 剩余风险

1. 实测在共享 NPU 上进行，witness 只锁 Aicore 频率，无法完全排除其他进程
   的 BIU/L2/HBM 干扰；已用多 pass + min target 抑制，但 store in-sample
   max 仍有 37.4%。
2. 本轮触发使用 `simd_simt_template` legacy 全局路径（没有 `scope.scope`）。
   真实 `simd_simt` mixed local scope 会额外包含 scope handoff/transition；
   该部分应由现有 scope transition cost 承担，template 公式只负责 loop 内
   per-iteration 模板体。
3. 只标定 rank1、unmasked、static non-pow2、aligned base；rank2/3、mask、
   dynamic stride、misalignment 未覆盖。
4. `iters_per_thread` 对 store 的系数来自当前 1 program 数据；多 program/多核
   SuperBlock 形态需要正交验证。
5. 若后续把 `parallel_mode=mix_simd_simt` 依赖改为 backend 自动设置，需重新
   确认 TTAdapter 仍含 `call @triton_stride_load/store`，否则本轮测量会静默
   退化成 SIMD memref.copy。

---

## 6. CostModel 接入状态（本轮实现）

> 本节记录 template v1 公式真正接入 `feature/strided-load-store-costmodel`
> v8 版本后的字段、判定和测试状态。

### 6.1 lowering path facts

`StridedMemoryAccess` 现在同时携带 template 判定和 v8 判定所需事实：

```text
rank
has_static_shape
masked
static_non_power_of_two_stride
strided_template_path_eligible
```

`detectStridedMemoryAccess()` 在原有 `base + arange * const_stride` 识别基础上：

- 从 shaped result/value 读取 rank 和 static shape；
- 识别 mask operand（load operand 1 / store operand 2）；
- 只有 `rank 1..3`、`hasStaticShape`、`unmasked`、4B element、
  static non-power-of-two `stride >= 3` 才置
  `strided_template_path_eligible=true`；
- v8 的 `elements/elementBytes/strideElements` 语义不变，pow2 / stride2
  继续走 v8；rank2/3 虽可被后端模板识别，但本轮公式未标定，因此当前
  `mapWorkload` 只在 rank1 时消费 template 公式，其余回退 v8。

### 6.2 mapWorkload 分支

`mapWorkload()` 现在接收 `implementation.localScope`：

| implementation | path |
|---|---|
| `mode=simd` | SIMD v8 structured/gather 公式 |
| `mode=simt` + `localScope=false` | v8 pure-SIMT 公式 |
| `mode=simt` + `localScope=true` + template eligible rank1 | template v1 公式 |
| `mode=simt` + `localScope=true` + 非 eligible | 回退 v8 pure-SIMT |

template 公式使用的特征在 C++ 中按与 Python 标定脚本相同的定义计算：

```text
L                  = distinct 128B line 数
mean_warp_lines    = 1024-thread 模板中每个 32-thread warp 平均 line 数
bucket_worst_32k16 = addr/32KB mod 16 后的最拥塞计数
tail_elems         = elements mod 1024
iters_per_thread   = ceil(elements/1024)
```

### 6.3 profile 字段

`simt.stage_resources.template_strided_memory` 新增 10 个系数（raw ns，
共用 `strided_memory.nanoseconds_to_system_cycles`）：

```text
simt_stride_template_load_intercept_ns
simt_stride_template_load_l_ns
simt_stride_template_load_bucket_worst_32k16_ns
simt_stride_template_load_mean_warp_lines_ns
simt_stride_template_load_tail_elems_ns
simt_stride_template_store_intercept_ns
simt_stride_template_store_l_ns
simt_stride_template_store_bucket_worst_32k16_ns
simt_stride_template_store_mean_warp_lines_ns
simt_stride_template_store_iters_per_thread_ns
```

字段在 `SimdSimtCostModel.cpp::readStageResources()` 中解析；
`StridedMemoryProfile::templateEnabled` 控制是否启用 template 分支。
`StridedMemoryProfile::isValid()` 验证新系数非负、有限，且
`templateEnabled` 必须建立在已启用的 `strided_memory` 转换系数上。

### 6.4 测试

- `scripts/predict_template_stride_v1.py --verify`：仍通过。
- `SimdSimtCostModel` UT：
  - `SimtLocalScopeUsesStrideTemplateLoad`
  - `SimtLocalScopeUsesStrideTemplateStore`
  - `SimtLocalScopeFallsBackForIneligibleStride`
  - `StridedFactDetectionMarksTemplateEligible`
  - `StridedTemplateProfileParserRejectsMissingField`
- IR 测试：
  `third_party/ascend/unittest/Conversion/General/TritonToLinalg/mixed_scope_stride_template.mlir`
  - `stride=3` load/store 在 model-controlled local SIMT scope 下生成
    `call @triton_stride_load/store`；
  - `stride=1/2/4/8/256` 控制不生成模板调用。


### 6.5 真实 route 观测

`scripts/mixed_route_stride_probe.py` 用真实 Python backend
（`compile_mode="simd_simt"`, `auto_simt_scope_mode="auto"`）跑 stride=3 的
单 load/sum kernel：

- CostModel 选择 `all_simt_only`；
- `mixed_simd_simt` 不 legal（该 kernel 没有 `tt.scan` / gather / atomic /
  triangular-solve 等 local SIMT anchor），所以不会进入 local SIMT scope；
- 因此该简单 kernel 正确走 v8 pure-SIMT，不产生 template 调用。

这与设计一致：template 只属于 mixed route 下的 local SIMT scope。
混合 route 的后端 lowering 由 §6.4 的 model-controlled IR 测试覆盖
（`ascend.simt_costmodel.effective="mixed_simd_simt"` + `scope.scope<simt>`）。

