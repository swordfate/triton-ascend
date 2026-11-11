# StridedWide 物理因子重标定（swide_phys）

背景、结果表与边界声明见仓库根 `_routemodel_contiguous_pricing.md` §10。
本目录只放**可复现产物**：拟合脚本、定稿系数、运行日志。

## 文件

| 文件 | 内容 |
|---|---|
| `refit_simt.py` | 全流程：加载数据（复用 `../pre-explore/scripts/` 旁路的 `strided-calib/03-featurize` 特征口径）→ 造特征（物理因子池）→ 分段 → 5 折 CV 选项数 + ridge + Nelder-Mead → 输出系数 |
| `phys_cap18.json` | **定稿系数**（四个槽位 × 各段：keys/coef/mape/n），C++/profile 侧即取自此文件 |
| `fit_log_cap18.txt` | 本次运行日志（逐段 MAPE + 逐系数） |

## 数据（不入 git，沿用 strided_ldst 的 raw 惯例）

原始实测在本机 `D:\work_wj\mask_vs_nomask\`：

| 文件 | 内容 | 过滤后 |
|---|---|---|
| `WIDE2_final.jsonl` | store 全量实测（13667 行） | 594 simd + 588 simt 点 |
| `LOADV1.jsonl` | load 全量实测（2124 行） | 551 simd + 463 simt 点 |

过滤口径：`valid && var==0 && spread<=0.2`，按 `(idx, var)` 去重——与早期 swide 标定逐位一致
（`refit_simt.py eval-baseline` 会先复核交付版 14.79%/12.40%，预期逐位复现）。

拟合脚本依赖 `D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize\{cmn,feats_v2}.py`
的**同一特征口径**（防止特征定义漂移）；脚本内以显式路径引用。

## 复现

```powershell
$PY = 'C:\Users\admin\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
& $PY swide_phys\refit_simt.py eval-baseline        # 交付版复核（逐位一致）
& $PY swide_phys\refit_simt.py fit-phys --cap 18    # 物理因子池四槽位（本表结果）
& $PY swide_phys\refit_simt.py oos --scheme rows3_W2 --cap 16   # dtype 留一 OOS
```

## 域边界（C++ 认领条件即据此设定）

实测覆盖：rows∈{2,8,64}、run_bytes∈{32,128(,2048 store)}、gap_bytes∈{2,32,128,2048,8192}、
W∈{1,2,4,8,16,32,64}、无 mask、inner（尾轴连续）、`run_bytes>=32`。
**认领必须校验上述域；域外让位给既有机制（layout_memory 指纹曲线 / 解析兜底）。**

## Masked（动态全 true mask）标定与集成

| 数据 | 来源/路径（不入 git） | 点数 |
|---|---|---|
| store masked | `WIDE2_final.jsonl` var=1（与 var=0 网格对等） | 6872（过滤后 store simt 966 / simd 979） |
| store masked rows=16 | `STORE_simt_rows16_v1.jsonl`（本机 `mask_vs_nomask/`；37 上 `mask_load/`） | 315 |
| load masked 主网格 | `LOADV1_v1.jsonl`（镜像 LOADV1，`probe_load.py --only-v1`） | 1034 有效 |
| load masked rows=16 | `LOAD_simt_rows16_v1.jsonl`（`probe_load.py --only-v1 --vars-from-case`） | 315 |

脚本：`fit_masked_slots.py`（拟合 16 个 masked 槽位）、`refit_mask.py`（共享/独立三路对比）、
`refit_mask_joint.py`（mask 因子方案验证：SIMT 不成立）、`gen_masked_profile.py`（profile/switch/UT
产物 + C++ 口径 parity，2069 点 0 偏差）。

结论：SIMD 与无 mask 共用槽位（比值中位 1.000）；SIMT 用独立 masked 槽位（store 9.73%、load 9.36%）。
mask 因子（联合 + is_masked/交互）在 SIMT 上不成立（联合→V1 11.6%，最差段 17–22%）。

## 加密 gap / 单流门控（2026-11-11）

| 数据 | 来源（不入 git） | 点数 |
|---|---|---|
| SIMD 加密 gap store/load | `DENSE_store.jsonl` / `DENSE_load.jsonl`（`cases_simd_densegap_v0.json`，fp32 gap∈{96..4096} × rows 2–256，f16 三点） | 356 / 356 |

脚本：`refit_simd_dense.py`（固定特征重估系数，旧/新/合并 MAPE）。
结论：原系数在加密点泛化良好（load 2.99/7.08%，store 2.62/5.96%）；`permute_reshape` 的偏差
来自"每迭代 2 条宽流按 2× 单流价求和"（实测单流 146ns vs 模型 148/151ns 仅差 1–3%）。
因此加入**单流门控**：只认领 count==1 且该方向宽访存总数==1 的访问。

## 区间插值与 16KB 行距扩展（2026-11-11）

- 准入采用**区间插值**：run 与 gap 只要落在该家族实测端点的闭区间内，就用公式算中间值；
  端点外一律回落（不外推）。常规 gap 端点为 [2, 8192] B；rows=16 的扩展标定见下。
- **16KB 行距扩展**：`solve_tril_*_h64` 的 block ptr 行距 = H×BT×4B = 16384 B（gap = 16320 B），
  超出原端点。已补测 rows=16（run {32,64,128} × gap {16256,16320,16384} × 3 dtype）：
  - masked SIMT（W 1–64）：store 14.99/9.09/7.05%、load 8.50/8.12/7.98%，重拟合后槽位
    store 9.73%、load 9.36%；
  - SIMD 共享槽位（masked 与 unmasked 各 27 点）：直接外推即可，masked store/load
    MAPE 4.36/4.88%、unmasked store/load 4.07/5.37%，单点最大 10.5%，无需重拟合。
  因此 rows=16 的 gap 端点放宽到 [2, 16384] B（仅限 run ≤ 128 B）：SIMD 有/无 mask 与
  masked SIMT 适用，其余家族仍为 8192 B。
