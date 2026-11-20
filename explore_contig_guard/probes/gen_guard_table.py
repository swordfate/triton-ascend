#!/usr/bin/env python3
"""Append the guard section to CONTIGUOUS_REVIEW_20261119.md:
final gate wheel (git56d9e023) guard predictions vs the 1118 doc's
historical measurements, 24 cases x 4 modes."""
import json

GT = "/tmp/GUARD_GATE/guard_table.json"
B4 = "/tmp/GUARD_4902/guard_table.json"
DOC = ("/root/work/triton-ascend/explore_contig_guard/"
       "CONTIGUOUS_REVIEW_20261119.md")

g = json.load(open(GT))
b4 = {(r["case"], r["combo"]): r for r in json.load(open(B4))}
by = {}
for r in g:
    by.setdefault(r["case"], []).append(r)

lines = []
lines.append("## 5. guard 用例终版回归（2026-11-20，准入门控后）")
lines.append("")
lines.append("口径：24 case × 4 mode（simd_f1 / simt_f1 / simt_f2 / simt_f4）= 96 组合；")
lines.append("实测 = 20261118 review §3.2 的历史 guard 实测（ns）；预测 = 终版 wheel")
lines.append("`git56d9e023` compile-only route dump（`guard_pred_gate.py`，卡 1）；")
lines.append("4902ae4a 基线对照列 = 同一脚本在 `git0868cdae` wheel 下的复测。")
lines.append("")
lines.append("**汇总**：终版 **26/96** 组合 PASS，四模式全 <10% 的 case **1/24**")
lines.append("（`A_copy_f32_4k_g256`：+6.3% / +6.7% / +4.5% / +9.1%）；基线 4902ae4a 仅")
lines.append("**10/96**、四模式全过 **0/24**。相对基线的提升来自直线域 simt 同形状组")
lines.append("认领（A 族 simt 从欠预测 −55~−77% 修到 ±10% 内）；1118 时代的 51/96、")
lines.append("7/24 是已退役的行表公式（E17）跑出的口径，不可比。")
lines.append("")
lines.append("已知缺口（与 §4 的未来工作一致，非本轮验收范围）：")
lines.append("1. 直线域 simt floor：f16/i8 小 dtype copy 的 simt_f1 被组解析 floor")
lines.append("   高估 +22~+36%（dtype 无关的 13775ns floor）；f32 4KB 档已准。")
lines.append("2. simt 循环域：认领已按准入纪律关闭（§4），B 族循环体回退到")
lines.append("   layout fallback，K-trip × 多程序形状高估 +40~+65%（与 4902 基线")
lines.append("   逐位一致，属基线自身行为）；simt 循环域的多程序重标定是独立的")
lines.append("   标定战役。")
lines.append("")
lines.append("| case | mode | 实测(ns) | 终版预测(ns) | 终版误差 | 终版 | 4902预测(ns) | 4902误差 |")
lines.append("|---|---|---:|---:|---:|:---:|---:|---:|")
for case in sorted(by):
    for r in sorted(by[case], key=lambda x: x["combo"]):
        b = b4.get((r["case"], r["combo"]))
        m = r["meas_ns"]
        p = r["pred_ns"]
        e = r["err_pct"]
        mark = "✅" if r["pass"] else "❌"
        if b and b["pred_ns"] is not None:
            b4s = "%.0f" % b["pred_ns"]
            b4e = "%+.1f%%" % b["err_pct"] if b["err_pct"] is not None else "-"
        else:
            b4s, b4e = "-", "-"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            case, r["combo"], "%.1f" % m if m else "-",
            "%.0f" % p if p else "-",
            "%+.1f%%" % e if e is not None else "-", mark, b4s, b4e))

with open(DOC, "a", encoding="utf-8") as f:
    f.write("\n\n" + "\n".join(lines) + "\n")
print("appended guard section,", len(lines), "lines")
