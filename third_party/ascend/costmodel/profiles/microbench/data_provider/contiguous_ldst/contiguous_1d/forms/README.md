# FormS 单发型标定（2026-11-07，Ascend 950PR，2 号卡）

单发（u=1、非流水）孤立访问的专用标定，是 for+延迟型/带宽型之外的第三条路线。

- `raw/forms_single_shot_950pr.jsonl`：781 个最终数据点（ok/warn，双会话 min 的中位数；
  noisy/bad 与初版失败探针数据未保留）。格点：SIMT W{1..64}×by{64..16K}×{A,B}×{load,store}×4 dtype
  + SIMD pow2 + SIMD 非 2 幂 10 块宽。方法：固定地址 + 对照扣除（load=load kernel−常量 store 对照；
  store=常量满幅 store−单标量 store），单 profiler 会话交错 [A×10,B×10]×2 取双会话 min。
- `fitted/forms_spec.json`：最终规格——SIMD te<8 公式（load.A 10.5%/load.B 7.7%/store 6.3% MAPE，
  store A/B 按整块谓词规则合并并加 15ns 地板）+ SIMD te≥8 实测表 + SIMT 全量实测表
  （(dir,form,dtype,W)→by→ns，缺口 14 点已由 gapfill 补齐）。
- 复现链：`scripts/probe_forms_single_shot.py`（采集）→ `scripts/forms_gapfill.py`（表缺口补测）
  → `scripts/forms_export.py`（拟合+导出 spec）。

方法学演进（运行时 off 标量降级、轮转 TLB 混杂、交错单会话提速）与八轮拟合记录见
`_contiguous_summary_v3.md` §5。
