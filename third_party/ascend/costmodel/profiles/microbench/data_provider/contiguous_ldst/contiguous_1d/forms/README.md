# FormS 单发型标定（2026-11-07/08，Ascend 950PR，2 号卡）

单发（u=1、非流水）孤立访问的专用标定，是 for+延迟型/带宽型之外的第三条路线。

## v2（当前，2026-11-08）：全部表格 → 16 条物理公式

- `raw/forms_single_shot_950pr_v5.jsonl`：v5 重测（ABAB 逐对差分）。一个 profiler 会话内
  (A,B)×16 交错，逐对差分抵消块级漂移；四分位距做 cv 门限。940+ 点覆盖
  SIMT W{1..64}×by{64..16K，8 档}×{A,B}×{load,store}×4 dtype + SIMD te≥8。
  与 v4 相比消除了对照组漂移偏置（对照例：fp32 W1 by=256 load 244→134ns）。
- `raw/forms_single_shot_950pr.jsonl`：v4 原始数据（保留作对照与 v5 缺点回退）。
- `fitted/forms_spec.json`：v2 规格——**128 张实测表 → 16 条公式**：
  - SIMT load（8 族）：统一形状 `one+nI+lnW4+pwsq`（nI=每线程 16B 串行指令数，
    lnW4=行轮次/4 调度器，pwsq=字节吞吐/√W）；7/8 族 ≤15% MAPE（10.8–14.5%），
    load.B.int8 单独形状 21.1%（字节使能路径不连续：te≥16 且 W≥2 时进入慢路径）。
  - SIMT store（1 条，全 dtype 合并）：`max(issue, drain)`，issue=40ns+k·nI·max(1,W/4)，
    drain=35ns+按 W 的每 16KB 排空率；40ns 地板。by≥4096 段 ~20% MAPE，
    小 by 段为 ~50ns 平台（实测 19–107ns 离散，硬件写合并量子行为，如 W4/W8 在
    by=4096 比 W16 慢 3 倍且四种 dtype 完全复现）。
  - SIMD te≥8（4 条）：load.A 12.9% / load.B 8.8% / store.B 6.7% / store.A 25.9%
    （未掩码 SIMD store 本体近零，散差为主）。
  - SIMD te<8（3 条）：沿用 v1 公式不变。
- 复现链：`scripts/probe_forms_single_shot.py`（v5 ABAB 采集）→ `scripts/fit_forms.py`
  （从 raw 重拟合并重写 fitted spec）→ `scripts/forms_export.py`（v1 历史导出器，保留）。

## v1（2026-11-07，已被 v2 取代）

SIMD te<8 公式 + SIMD te≥8 实测表 16 张 + SIMT 实测表 112 张（(dir,form,dtype,W)→by→ns，
gapfill 补齐）。方法：固定地址 + 对照扣除，[A×10,B×10]×2 双会话 min。
v1→v2 的 100-case 对比见 `_contiguous_summary_v4.md`。
