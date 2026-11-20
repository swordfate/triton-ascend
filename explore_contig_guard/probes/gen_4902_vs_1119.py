#!/usr/bin/env python3
"""Build _100case_4902ae4a_vs_1119.md:
  cols 1-4: #, case, 2e9efcac route, measured (from gridhint md)
  cols 5-7: 4902ae4a (git0868cdae) prediction/error/PASS
  cols 8-11: 1119 (gitc380f6da) route/prediction/error/PASS
PASS = |err| < 10% -> checkmark / cross; no-measurement rows -> 不可评."""
import glob
import json
import re

MD = "/tmp/gridhint.md"
RUN4902 = "/tmp/RUN_PATH"
RUN1119 = "/tmp/RUN_PATH_c380"
OUT = ("/root/work/triton-ascend/explore_contig_guard/"
       "_100case_4902ae4a_vs_1119.md")

NO_MEAS = {"histogram_i32_r8192_b256_uniform",
           "histogram_i32_r8192_b256_single",
           "histogram_i32_r8192_b256_radix",
           "pytest_histogram_i32_2048_bins2",
           "pytest_atomic_add_f32_32x32_c2"}
row_re = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*([0-9.]+|None)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|"
    r"\s*(\S+)\s*\|\s*([0-9.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|")

base_rows = []
for line in open(MD):
    m = row_re.match(line)
    if not m:
        continue
    name = m.group(2).strip()
    meas_s = m.group(4)
    try:
        berr = float(m.group(6).strip().replace("%", ""))
    except ValueError:
        berr = None  # dash rows (non-evaluable)
    base_rows.append((int(m.group(1)), name, m.group(3), meas_s,
                      float(m.group(5)), berr))


def load_run(root):
    out = {}
    for f in glob.glob(root + "/s*/*.route.json"):
        name = f.split("/")[-1].replace("__auto_f1.route.json", "")
        try:
            d = json.load(open(f))
        except Exception:
            continue
        route = d.get("effective_decision_kind") or "?"
        factor = d.get("selected_superblock_factor") or \
            d.get("effective_runtime_factor") or 1
        costs = d.get("candidate_elapsed_costs") or {}
        if route in costs:
            tag = route.replace("all_", "").replace("_only", "")
            out[name] = ("%s+F%s" % (tag, factor), costs[route] / 1000.0)
    return out


r4902 = load_run(RUN4902)
r1119 = load_run(RUN1119)


def cells(meas, entry):
    """-> (route, pred_s, err_s, mark); route '-' when entry missing."""
    if entry is None:
        return "-", "-", "-", "-"
    route, pred = entry
    err = (pred - meas) / meas * 100.0
    return route, "%.3f" % pred, "%+.2f%%" % err, (
        "✅" if abs(err) < 10 else "❌")


hdr = """# 100-case 三轮对照：基线 4902ae4a vs 1119 版本

口径（与前几轮完全一致）：

| 项 | 值 |
|---|---|
| harness | `/root/w/cc-harness/costmodel_generalization`，`run_generalization_matrix.py --disable-profiler`（compile-only，仅 route+预测），8 分片并行卡 1/2/3 轮转 |
| 实测(us) | 照抄 v140 历史实测表（gridhint 文档实测列，本轮未重测） |
| 4902ae4a 轮 | `triton_ascend-3.6.0.dev0+git0868cdae`（服务器 commit 0868cdaece = 本地 4902ae4a1 = 2e9efcacb8 + #3126；**不含任何连续访存公式**，2026-11-20 构建） |
| 1119 轮 | `triton_ascend-3.6.0.dev0+gitc380f6da`（1119 版本 = 连续访存全套：直线域同形状组 + 循环域 loopfit v6；E33 实验与 fix3 均已回退） |
| 预测(us) | 各轮 route.json `candidate_elapsed_costs[effective_decision_kind]`（ns→µs） |
| 误差 | (预测 − 实测) / 实测；PASS = \\|误差\\| < 10%（✅/❌） |

**汇总（95 例可评）**：4902ae4a **75** PASS（median 4.10%、p90 21.74%）；1119 **74** PASS（median 4.59%、p90 23.05%）。
4902ae4a 与 gridhint 文档的 2e9efcac 基线（75 PASS）零翻转——#3126 对这批 case 无跨线影响。
1119 相对 4902ae4a：改善 4（vector_add −33.8→−0.0、associative_scan +35.0→+8.6、cumprod +12.9→+2.4、max_dim1 +12.1→+9.2），
回归 5（ms_l4 +3.4→−22.4、muls_add_l2 −0.2→−11.5、fused_softmax −4.8→−15.4、permute_reshape −0.6→−38.3、strided_scatter −7.5→−23.0；均为 loopfit 时代已知项，100-case 之外的 5 例扫描域回归另见归因文档）。

## 全表（100 case）

| # | case | 2e9efcac route | 实测(us) | 4902预测 | 4902误差 | 4902 | 1119 route | 1119预测 | 1119误差 | 1119 |
|---:|---|---|---:|---:|---:|:---:|---|---:|---:|:---:|
"""

body = []
for idx, name, broute, meas_s, bpred, berr in sorted(base_rows):
    if name in NO_MEAS or meas_s == "None" or berr is None:
        e4 = r4902.get(name)
        e11 = r1119.get(name)
        p4 = "%.3f" % e4[1] if e4 else "-"
        p11 = "%.3f" % e11[1] if e11 else "-"
        route4 = e4[0] if e4 else "-"
        route11 = e11[0] if e11 else "-"
        body.append("| %d | `%s` | %s | None | %s | — | 不可评 | %s | %s | "
                    "— | 不可评 |" % (idx, name, broute, p4, route11, p11))
        continue
    meas = float(meas_s)
    route4, p4, e4, mk4 = cells(meas, r4902.get(name))
    if route4 == "-":
        route4 = "MISSING"
    route11, p11, e11, mk11 = cells(meas, r1119.get(name))
    if route11 == "-":
        route11 = "MISSING"
    body.append("| %d | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (idx, name, broute, meas_s, p4, e4, mk4, route11, p11,
                   e11, mk11))

open(OUT, "w").write(hdr + "\n".join(body) + "\n")
print("wrote", OUT, "rows:", len(body))
