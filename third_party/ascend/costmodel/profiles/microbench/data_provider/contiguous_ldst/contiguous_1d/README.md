# 连续访存（stride == 1）标定：数据、探针与拟合

本目录是 CostModel 连续访存模型的**全部可复现材料**。模型本体见 [`MODEL.md`](MODEL.md)，
集成到 CostModel 的代码见同分支的下一个提交（`RouteModel/Models/ContiguousLdStCostModel.*`）。

## 目录

| 路径 | 内容 |
|---|---|
| `MODEL.md` | 模型说明：两种形态、公式、物理含义、适用条件、已知弱点 |
| `scripts/` | 测量探针与代码生成器 |
| `fit/` | 拟合脚本（复现 `fitted/FINAL_SPEC.json` 与形态 B 的 6+4 段）|
| `raw/` | 实测原始数据（JSONL，每行一个测量点）|
| `fitted/` | 拟合产物 |

## 测量方法（所有数据同一口径）

- **k-marginal**：同一 kernel 跑 `iters ∈ {2048, 8192}` 两次，`cost = (t(8192) − t(2048)) / 6144`（ns/次）；
- `grid = 1`，每程序在 `k_low = 2048` 个块的窗口内轮转（用 `step & 2047` 实现，避免取模）；
- 计时用 **CANN profiler** 取 `kernel_details.csv` 的 kernel 持续时间最小值；
  torch event 在 NPU 上不可用（同一个 copy 会量成 0.4 µs）；
- 采集前必须跑 `scripts/probe_speed.py`：健康 ~66 ns/iter，>150 不可用；
- **整批采集、批内比较**：批内波动 ≤0.2%，但跨时段绝对水平会漂 20~30%（见 `raw/raw_bimodal_addr.jsonl`、
  `raw/raw_stability.jsonl`），绝不能跨天拼接数据。

## 数据文件

| 文件 | 点数 | 形态 | 模式 | dtype | 尺寸范围 | 含非 2 幂 | W |
|---|---:|---|---|---|---|---|---|
| `raw/raw_pure_unified.jsonl` | 755 | A | SIMD+SIMT | 4 种 | 元素数 1~16384 | ❌ | 1~64 |
| `raw/raw_formb_v2.jsonl` | 722 | B | SIMT | 4 种 | 块宽 2~8192 | ❌ | 1~64 |
| `raw/raw_simd_npow2_A.jsonl` | 268 | A | SIMD | 4 种 | 块宽 31~16384（34 档）| ✅ 189 点 | 1 |
| `raw/raw_simd_npow2_B.jsonl` | 136 | B | SIMD | int8,fp32 | 块宽 31~16384（34 档）| ✅ 96 点 | 1 |
| `raw/raw_tile2d_verify.jsonl` | 240/80 组 | A | SIMD+SIMT | 4 种 | 8 种二维几何 | ✅ | 1,4 |
| `raw/raw_whole_kernel.jsonl` | 68 | A | SIMD+SIMT | 2 种 | 9 种几何 | ✅ 4 种 | 1,4 |

SIMT 的块宽只能是 2 的幂（语言限制，见 `MODEL.md` §4.3），所以 SIMT 侧没有非 2 幂数据。

## 复现步骤

```bash
# 1) 采集形态 A（2 的幂块宽）
python3 scripts/probe_block_pure.py --mode simd --direction load --dtype fp32 \
        --block 4096 --warps 1 --k-low 2048 --k-high 8192 --manifest raw/raw_simd_npow2_A.jsonl

# 2) 采集形态 B（带运行期 mask）
python3 scripts/probe_mask_kind.py --mode simt_only --direction load --dtype fp32 \
        --block 1024 --valid 1024 --dynamic 1 --warps 4 \
        --k-low 2048 --k-high 8192 --manifest raw/raw_formb_v2.jsonl

# 3) 整核验证（copy / add3）
python3 scripts/validate_whole_kernel.py --manifest raw/raw_whole_kernel.jsonl \
        --kind copy --mode simd --dtype fp32 --rows 1 --cols 4096 --warps 1 --iters 2048

# 4) 重拟合 → 生成 fitted/FINAL_SPEC.json
python3 fit/fit_bottleneck.py && python3 fit/fit_physical.py \
  && python3 fit/fit_tail_with_warps.py && python3 fit/assemble_spec.py

# 5) 生成 C++ 系数表（集成用）
python3 scripts/gen_contiguous_model.py
```

## 拟合产物

| 文件 | 内容 |
|---|---|
| `fitted/FINAL_SPEC.json` | 形态 A 的 SIMD + SIMT 公式（由 `fit/assemble_spec.py` 生成）|
| `fitted/SIMD_CONTIGUOUS.json` | SIMD 两条公式（4 系数，覆盖 2 次幂与非 2 次幂）|
| `fitted/FORMB_FINAL.json` | 形态 B 的 SIMT 4（load）+ 6（store）段 |
| `fitted/{SIMD_JOINT,SIMT_SEGMENTS,SIMT_TAIL_SEGMENTS,STAGED_FIT,LOAD_BOTTLENECK}.json` | `assemble_spec.py` 的输入 |

## 结果摘要

| 公式 | 段 | 系数 | 点 | MAPE |
|---|---:|---:|---:|---:|
| SIMD load | 1 | 4 | 133 | 6.97% |
| SIMD store | 1 | 4 | 135 | 5.24% |
| SIMT A load | 3 | 18 | 251 | 7.73% |
| SIMT A store | 4 | 21 | 324 | 8.47% |
| SIMT B load | 4 | 24 | 352 | 9.47% |
| SIMT B store | 6 | 34 | 361 | 9.03% |

合计 1 556 个实测点，六条公式全部 < 10%。
