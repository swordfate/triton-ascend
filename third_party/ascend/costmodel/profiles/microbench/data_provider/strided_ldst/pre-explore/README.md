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
- `stride>=4`（元素间隔 ≥16B）时退化为 **每个元素一条 4B BIU read**：
  - BLOCK=32 是 32×4B，BLOCK=128 是 128×4B。
- 因此 SIMD 侧的关键不是 128B line 数，而是 MTE2 能否宽事务、是否退化成
  per-element gather；后面拟合出的 SIMD wide / gather 两条路径即对应这里。

代表性 BLOCK=32 数据（raw CAModel cycle；`active = pre + load + post`）：

| stride | active | pre | load | post | load 内 `issue→BIU` / `BIU fill` / `BIU→ret` | BIU command |
|---:|---:|---:|---:|---:|---|---|
| 1 | 1503 | 424 | 446 | 633 | 20 / 403 / 23 | 1×128B |
| 2 | 1699 | 464 | 550 | 685 | 20 / 504 / 26 | 1×256B |
| 4 | 1642 | 458 | 586 | 598 | 20 / 543 / 23 | 32×4B |
| 16 | 1552 | 419 | 588 | 545 | 20 / 545 / 23 | 32×4B |
| 256 | 1803 | 556 | 580 | 667 | 20 / 534 / 26 | 32×4B |

BLOCK 扫描进一步给出 gather 分支的线性关系：

```text
SIMD gather (stride>=4), BLOCK=32/64/128:
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
| 4 | 4 | 32 | 32 | 25% | 4× |
| 8 | 8 | 16 | 16 | 12.5% | 8× |
| 16 | 16 | 8 | 8 | 6.25% | 16× |
| 32+ | 32 | 4 | 4 | 3.125% | 32× |

- `stride=4→256` 时 line 数从 4 增到 32，但单条 `SIMT_LDG` 的 load 窗口只从约 483
  增到 654 cycle：BIU line fill 高度 overlap，**不能按“每条 line 一次完整 miss latency”收费**。
- 多 warp（BLOCK=128）时，单条 LDG 自身约 1.1k cycle，但 4 条 LDG 的总窗口只有约
  1.17k cycle：stage-level 成本不能按 `K × 单条 latency` 线性放大。

##### 指令路径示意

```text
SIMD:  tl.load -> MTE2 MOV_SRC_TO_DST_ALIGNv2 (GM->UB) -> VF reduce/use
                 stride=1/2: wide transaction
                 stride>=4: BLOCK x 4B gather command

SIMT:  tl.load -> SIMT_LDG -> SIMT 128B DCache -> BIU 128B line read
                 DC 合并同一 line 内 lane，但 line 利用率随 stride 下降
```

---

### 1.2 实测性能与数据重新拟合

#### 1.2.1 写了什么脚本

| 脚本 | 作用 |
|---|---|
| `scripts/simd_one_load_syscnt.py` | SIMD single cold load 的真卡 syscnt / Event 测量 |
| `scripts/frequency_retry_probe.py` | 频率感知的 SIMD single-load 测量：记录调频、no-load 参考、异常重试 |
| `scripts/simt_frequency_retry_probe.py` | SIMT rotate-loop 的 Event-slope 测量，支持不同 `num_warps` 和 stride sweep |
| `scripts/syscnt_event_compare.py` | 对比单发 / loop Event 口径，导出成本模型 target |
| `scripts/board_event_probe.py`、`scripts/board_time_probe.py` | 真卡 Event / 时间探针 |
| `scripts/run_v2_simd_dense.sh`、`scripts/run_v2_simt_dense.sh` | SIMD / SIMT dense stride 测量驱动 |
| `scripts/run_v2_more_blocks.sh` | 补充 BLOCK 网格 |
| `scripts/run_simt_numwarps_sweep.sh`、`scripts/run_simt_numwarps_dense.sh` | SIMT `num_warps` coarse + dense sweep 驱动 |
| `scripts/build_v2_dataset.py` | 把 SIMD / SIMT 原始测量 JSON 整理成 `results/model_v2/dataset.csv` |
| `scripts/build_v3_numwarps_dataset.py` | 把 `num_warps` sweep 整理成 `results/model_v3_numwarps/dataset.csv`，并生成 `L`、`dup`、`cross_page` 等特征 |
| `scripts/fit_strided_model_v7_simple.py` | 当前最终拟合脚本：raw target 域 weighted NNLS；最终公式无 log、无次方、系数非负 |
| `scripts/fit_strided_model_common.py` | 拟合公共工具：NNLS、特征、误差 metrics（被 v7 脚本复用） |
| `scripts/predict_strided_load_v7_simple.py` | 当前最终 predictor，内置 `--verify` 逐点复算模型与误差表 |

数据规模：

| 场景 | 点数 | block | stride | num_warps | target |
|---|---:|---|---|---|---|
| SIMD wide | 24 | 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 1, 2 | 1（SIMD 与 num_warps 无关） | ns |
| SIMD gather | 504 | 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256 | 1（SIMD 与 num_warps 无关） | ns |
| SIMT | 344 | 32, 64, 128, 256 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256 | 1, 2, 4, 8, 16, 32, 64 | ns/iteration |

拟合目标函数：

```text
min Σ w_i (c0 + Σ c_j f_ij - T_i)^2,   c_j >= 0
```

- SIMD：`w_i = 1/T_i`；
- SIMT：`w_i = 1/T_i^1.5`；
- 全部在 raw target 域求解，不对 `T` 取 log。

关于 SIMT 权重的次方：`w_i` 只是拟合时的样本权重，不是最终预测公式的一部分。
加权目标项为 `w_i * (pred_i - T_i)^2`。取 `w_i = 1/T_i^alpha` 时：

```text
w_i * (pred_i - T_i)^2
  = T_i^(2-alpha) * (pred_i / T_i - 1)^2
```

- `alpha = 2` 等价于纯相对误差；
- `alpha = 0` 等价于普通绝对误差，大 target 会主导拟合；
- SIMT 取 `alpha = 1.5`，是在“绝对误差”和“纯相对误差”之间的折中，
  让较小 target 的点在 NNLS 中有更高权重，避免大 target 主导。

最终预测公式 `T_simt` 本身没有任何次方项，推理时也不会出现 `T_i^1.5`。

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

##### SIMT（4 terms，无次方）

```text
E      = block / (32 * W)
dup    = max(0, 1/E - 1)
span   = (block - 1) * stride * 4
cross  = 1 if span >= 4096 else 0
L      = block                              if stride*4 >= 128
         floor((block-1)*stride*4/128) + 1  otherwise    # distinct 128B lines
dupL   = dup * L

T_simt(ns/iter) = 112.7181224362
                +   0.2056513657 * dupL
                +   6.0747935132 * min(L, 64)
                +   0.1157497298 * max(0, W*L - 3072)
                +  14.6254882490 * dup * cross
```

| 项 | 物理含义 | 为什么增加时间 |
|---|---|---|
| 常数 | loop iteration 的固定 SIMT 启动/控制/首条 LDG | 与 shape 无关的基础成本 |
| `dupL` | underfilled warp 复制后的 line 请求数 | 每个额外复制的 line 增加一次 issue/drain，近似线性 |
| `min(L, 64)` | 饱和前的 line 工作量 | 前 64 条 distinct line 每条增加填充/排队成本；超过 64 条后 overlap 足够，不再线性计费 |
| `max(0, W*L - 3072)` | 总 warp-line 请求超过 3072 后的尾部 | 在途请求饱和后，只对超出门限的队尾工作量收费 |
| `dup*cross` | underfilled warp 复制 + 4KB 页跨越 | page 级路径与复制压力叠加时额外增加时间 |

所有系数均 `>= 0`，没有 `log()`，没有 `x^2/x^3`。

#### 1.2.3 与实测的误差表

全量 in-sample 误差：

| 场景 | n | MAPE | p50 | max |
|---|---:|---:|---:|---:|
| SIMD wide | 24 | 5.10% | 2.51% | 19.00% |
| SIMD gather | 504 | 5.71% | 4.14% | 76.43% |
| SIMT | 344 | 18.05% | 13.63% | 72.53% |

指标口径（相对误差定义为 `pred / target - 1`）：

- **MAPE**：平均绝对百分比误差，`mean(|pred / target - 1|)`；反映整体平均偏差。
- **p50**：绝对百分比误差的中位数；一半样本误差小于它。
- **max**：最大绝对百分比误差；直接看最坏点。

p90 / p95 / bias / RMSE 仍保留在 `metrics_summary.json/.csv` 中，README 不再展示。

误差可视化：

- 生成脚本：`scripts/make_v7_plots.py`
- 图片目录：`results/model_v7_simple/plots/`
  - `v7_parity.png`：三组 target vs prediction 的 log-log parity
  - `v7_error_hist.png`：三组相对误差分布直方图
  - `v7_error_vs_stride.png`：误差随 stride 的散点图
  - `v7_error_vs_target.png`：误差随 target 的散点图
- 重新生成：

  ```bash
  python3 scripts/make_v7_plots.py
  ```

逐点误差文件：

```text
results/model_v7_simple/errors_all.csv
results/model_v7_simple/errors_simd_wide.csv
results/model_v7_simple/errors_simd_gather.csv
results/model_v7_simple/errors_simt.csv
```

模型文件：

```text
results/model_v7_simple/model_v7_simple.json
results/model_v7_simple/metrics_summary.json
results/model_v7_simple/metrics_summary.csv
```

predictor 一致性校验：

```bash
python3 scripts/predict_strided_load_v7_simple.py --verify
```

当前校验结果：

```text
max |predict - errors.pred_ns| = 4.987e-07
max predictor relative diff    = 4.081e-09
max recomputed metric diff     = 1.103e-08 percentage points
VERIFY OK
```

单点预测示例：

```bash
python3 scripts/predict_strided_load_v7_simple.py simd 128 16
python3 scripts/predict_strided_load_v7_simple.py simt 128 16 4
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

# 真卡测量与数据集
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

# 最终拟合与预测
scripts/fit_strided_model_v7_simple.py
scripts/fit_strided_model_common.py
scripts/predict_strided_load_v7_simple.py
scripts/make_v7_plots.py
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
