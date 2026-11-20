#!/usr/bin/env python3
"""Build the final 100-case table for CONTIGUOUS_REVIEW_20261119.md:
cols: #, case, 2e9efcac route, measured (from the three-way doc) +
4902ae4a pred/err/mark + FINAL (git56d9e023) route/pred/err/mark."""
import glob
import json
import re

DOC = ("/root/work/triton-ascend/explore_contig_guard/"
       "_100case_4902ae4a_vs_1119.md")
RUNF = "/tmp/RUN_GATE2"
NO_MEAS = {"histogram_i32_r8192_b256_uniform",
           "histogram_i32_r8192_b256_single",
           "histogram_i32_r8192_b256_radix",
           "pytest_histogram_i32_2048_bins2",
           "pytest_atomic_add_f32_32x32_c2"}
row_re = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\S+)\s*\|\s*([0-9.]+|None)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|"
    r"\s*([^|]+)\|\s*([^|]+)\|")

base = {}
for line in open(DOC):
    m = row_re.match(line)
    if not m:
        continue
    name = m.group(2).strip()
    try:
        e4 = float(m.group(6).strip().replace("%", "").strip())
    except ValueError:
        e4 = None
    base[name] = {"idx": int(m.group(1)), "route2e": m.group(3),
                  "meas_s": m.group(4), "p4": m.group(5), "e4": e4}


def load_run(root):
    out = {}
    for f in glob.glob(root + "/s*/*.route.json"):
        name = f.split("/")[-1].replace("__auto_f1.route.json", "")
        try:
            d = json.load(open(f))
        except Exception:
            continue
        route = d.get("effective_decision_kind") or "?"
        costs = d.get("candidate_elapsed_costs") or {}
        if route in costs:
            tag = route.replace("all_", "").replace("_only", "")
            factor = d.get("selected_superblock_factor") or \
                d.get("effective_runtime_factor") or 1
            out[name] = ("%s+F%s" % (tag, factor), costs[route] / 1000.0)
    return out


fresh = load_run(RUNF)
hdr = """## 4. 100-case 终版回归（2026-11-20，准入门控后）

**终版 wheel = `triton_ascend-3.6.0.dev0+git56d9e023`**（HEAD = `56d9e02384`
= `c380f6da` + 循环域准入两项修正）：
1. `d92bc5a870` 循环域准入门控——loopfit 曲线标定域为 grid=1 热/无 mask/
   线性走址/无归约耦合的干净体（e27b_fit2）；携带 grid-stride shell、
   运行时部分掩码、归约/scan 耦合或 StridedWide 成员的循环体一概不予
   认领，交还 legacy 实测组曲线/跨步层（与基线 4902ae4a 完全同价）。
2. `56d9e02384` 关闭 simt 循环认领——simt loopfit 曲线为 drain-forced
   grid=1 标定，其全部 100-case 消费者欠预测 26-54%/trip
   （strided_scatter −26%、rowred_r16 −54%），且无任何正向贡献者。

**验收结果（95 例可评）**：终版 **79 PASS**，相对基线 4902ae4a（75 PASS）
**零新增 FAIL**（硬目标达成）；median 4.06% / p90 19.39%，均优于基线
（4.10% / 21.74%）。改善 4 例全部保持（vector_add −33.8→−0.0、max_dim1
+12.1→+9.2、cumprod +12.9→+2.4、associative_scan +35.0→+8.6，均来自
直线域/scan-stage 认领）。16 例 FAIL 全部为基线本身 FAIL 的 case
（归因见 `_100case_gt10_contig_analysis.md`：持久化 scan/softmax 结构、
gather 间接寻址、atomic 争用、归约结构——非连续访存标定所致）。

归因方法（Q4 指南 §6 stage 级做差）：5 例新增 FAIL 全部核实为循环域
kernel，且整核缺口 100% 落在被 v2 认领的单一循环 stage 上
（muls_add +28.2 ns/trip −20%、memory_stream_l4 +70.0 −31%（与 E33d
mask +11 + 跳址 +38 吻合）、fused_softmax +83.8 −23%、strided_scatter
−83.6 −26%、permute_reshape −204 −50%（32B 行 @384B 跨步成员被当作
连续计价））。loopfit 解析复刻（probes/loopfit_replica.py）与 stage
价逐位一致，证明缺口全部来自标定域外推。

| # | case | 2e9efcac route | 实测(us) | 4902预测 | 4902误差 | 4902 | 终版 route | 终版预测 | 终版误差 | 终版 |
|---:|---|---|---:|---:|---:|:---:|---|---:|---:|:---:|
"""

body = []
for name, b in sorted(base.items(), key=lambda kv: kv[1]["idx"]):
    if name in NO_MEAS or b["e4"] is None:
        e = fresh.get(name)
        p = "%.3f" % e[1] if e else "-"
        r = e[0] if e else "-"
        body.append("| %d | `%s` | %s | None | %s | — | 不可评 | %s | %s |"
                    " — | 不可评 |" % (b["idx"], name, b["route2e"],
                                       b["p4"], r, p))
        continue
    meas = float(b["meas_s"])
    e = fresh.get(name)
    if e is None:
        body.append("| %d | `%s` | %s | %s | %s | %s | %s | MISSING | - |"
                    " - | - |" % (b["idx"], name, b["route2e"],
                                  b["meas_s"], b["p4"],
                                  "%+.2f%%" % b["e4"],
                                  "✅" if abs(b["e4"]) < 10 else "❌"))
        continue
    route, pred = e
    err = (pred - meas) / meas * 100.0
    body.append("| %d | `%s` | %s | %s | %s | %s | %s | %s | %.3f | "
                "%+.2f%% | %s |" % (
                    b["idx"], name, b["route2e"], b["meas_s"], b["p4"],
                    "%+.2f%%" % b["e4"],
                    "✅" if abs(b["e4"]) < 10 else "❌",
                    route, pred, err,
                    "✅" if abs(err) < 10 else "❌"))

OUT = ("/root/work/triton-ascend/explore_contig_guard/"
       "_100case_final_gate_table.md")
open(OUT, "w").write(hdr + "\n".join(body) + "\n")
npass = sum(1 for line in body if line.endswith("| ✅ |"))
print("wrote", OUT, "rows:", len(body), "PASS:", npass)
