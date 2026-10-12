#!/usr/bin/env python3
"""Consolidate every PHYS_*.json into REPORT_v4_physics.md."""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results/final"
COMP = json.load(open(RES / "COMPOSITION.json", encoding="utf-8"))
Q = chr(96)
FENCE = Q * 3
EB = {"int8": 1, "float16": 2, "float32": 4, "int64": 8}
SLOT_DOC = {
 "simd_load_wide_r1": ("SIMD load \u5bbd\u5206\u652f\uff081D\uff09", "compile_mode=simd\uff0ctl.load\uff0c\u5143\u7d20\u57df stride <= 2\uff0crank-1\u3002CAModel \u767d\u76d2\uff1aBIU \u8fde\u7eed\u642c\u8fd0\u6574\u5757\u518d extract\uff0cbiu_cmds \u4ece n \u964d\u5230 1-9\u3002"),
 "simd_load_wide_r2": ("SIMD load \u5bbd\u5206\u652f\uff082D\uff09", "compile_mode=simd\uff0ctl.load\uff0c\u5185\u5c42\u5143\u7d20\u57df stride <= 2\uff0crank-2\u3002\u5df2\u5254\u9664 R=1 \u7684\u4f2a 2D \u5f62\u72b6\u3002"),
 "simd_load_gather_r1": ("SIMD load \u805a\u96c6\u5206\u652f\uff081D\uff09", "compile_mode=simd\uff0ctl.load\uff0c\u5143\u7d20\u57df stride > 2\uff0crank-1\u3002\u6bcf\u5143\u7d20\u4e00\u6761 BIU \u547d\u4ee4\uff08CAModel: biu_send_sizes={eb:n}\uff09\u3002"),
 "simd_load_gather_r2": ("SIMD load \u805a\u96c6\u5206\u652f\uff082D\uff09", "compile_mode=simd\uff0ctl.load\uff0c\u5185\u5c42\u5143\u7d20\u57df stride > 2\uff0crank-2\u3002\u5df2\u5254\u9664 R=1\u3002"),
 "simd_store_r1": ("SIMD store\uff081D\uff09", "compile_mode=simd\uff0ctl.store\uff0crank-1\uff0cMTE3 \u5199\u8def\u5f84\u3002"),
 "simd_store_r2": ("SIMD store\uff082D\uff09", "compile_mode=simd\uff0ctl.store\uff0crank-2\u3002\u5df2\u5254\u9664 R=1\u3002"),
 "simt_load_r1": ("\u7eaf SIMT load\uff081D\uff09", "compile_mode=simt_only\uff0ctl.load\uff0crank-1\uff0cLSU \u8def\u5f84\u3002"),
 "simt_load_r2": ("\u7eaf SIMT load\uff082D\uff09", "compile_mode=simt_only\uff0ctl.load\uff0crank-2\uff0cLSU \u8def\u5f84\u3002\u5df2\u5254\u9664 R=1\u3002"),
 "simt_store_r1": ("\u7eaf SIMT store\uff081D\uff09", "compile_mode=simt_only\uff0ctl.store\uff0crank-1\uff0cLSU \u8def\u5f84\u3002"),
 "simt_store_r2": ("\u7eaf SIMT store\uff082D\uff09", "compile_mode=simt_only\uff0ctl.store\uff0crank-2\uff0cLSU \u8def\u5f84\u3002\u5df2\u5254\u9664 R=1\u3002"),
}
ORDER = [s + "_r" + str(rk) for s in ("simd_load_wide","simd_load_gather","simd_store","simt_load","simt_store") for rk in (1,2)]
GREEDY = {"simd_load_wide_r1":(3.46,6),"simd_load_wide_r2":(17.10,17),"simd_load_gather_r1":(12.14,78),
          "simd_load_gather_r2":(11.08,73),"simd_store_r1":(4.38,105),"simd_store_r2":(5.63,113),
          "simt_load_r1":(7.50,89),"simt_load_r2":(7.24,81),"simt_store_r1":(5.47,113),"simt_store_r2":(17.75,49)}

def build():
    merged = {}
    for p in sorted(glob.glob(str(RES / "PHYS_*.json"))):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print("skip", p, e); continue
        for k, v in j.items():
            if k.startswith("_"): continue
            if isinstance(v, dict) and "mape" in v:
                merged[k] = v
    return merged

def composition(k):
    c = COMP.get(k)
    if not c: return ["\uff08\u65e0\u6784\u6210\u6570\u636e\uff09"]
    L = []
    L.append("- \u8def\u5f84/\u64cd\u4f5c/\u79e9\uff1a%s / %s / rank-%d" % (c["path"], c["op"], c["rank"]))
    L.append("- dtype\uff08%d \u79cd\uff09\uff1a%s" % (len(c["dtypes"]), "\u3001".join("%s(%dB)" % (x, EB[x]) for x in c["dtypes"])))
    L.append("- num_warps\uff08%d \u79cd\uff09\uff1a%s" % (len(c["num_warps"]), "\u3001".join(str(x) for x in c["num_warps"])))
    lab = "\u5f62\u72b6 (R\u00d7C)" if c["rank"] == 2 else "elements"
    ks = list(c["shapes"].keys())
    if c["rank"] == 1: ks = [x.split("x")[-1] for x in ks]
    L.append("- %s\uff08%d \u79cd\uff09\uff1a%s" % (lab, c["n_shapes"], "\u3001".join(ks)))
    L.append("- \u5185\u5c42\u6b65\u957f\uff08\u5b57\u8282\uff09\uff08%d \u79cd\uff09\uff1a%s" % (len(c["inner_stride_bytes"]), "\u3001".join(str(x) for x in c["inner_stride_bytes"])))
    L.append("- \u603b\u70b9\u6570\uff1a**%d**" % c["n"])
    return L

def main():
    m = build()
    keys = [k for k in ORDER if k in m] + [k for k in m if k not in ORDER]
    L = ["# \u7248\u672c 4\uff1aPHYSICS \u2014\u2014 \u7269\u7406\u7b2c\u4e00\u6027\u91cd\u5efa", "",
         "\u6bcf\u4e2a\u516c\u5f0f\u90fd\u4ece\u7269\u7406\u8fc7\u7a0b\u51fa\u53d1\u6784\u9020\uff08\u4f20\u8f93\u5b57\u8282\u6570\u3001BIU \u547d\u4ee4\u6570\u3001\u540c\u5757\u51b2\u7a81\u60e9\u7f5a\u3001warp \u53d1\u5c04\u5f00\u9500\u7b49\uff09\uff0c",
         "\u5141\u8bb8 max/min \u4e0e\u4e58\u79ef\u60e9\u7f5a\u9879\uff0c\u4e0d\u4f7f\u7528 log/\u6307\u6570/\u5e42\u6b21\u3002\u62df\u5408\u53e3\u5f84\uff1a\u52a0\u6743\u6700\u5c0f\u4e8c\u4e58\uff0c\u6743\u91cd 1/T\uff08\u5373 MAPE\uff09\u3002", "",
         "## \u6c47\u603b\u8868", "",
         "| \u516c\u5f0f | \u573a\u666f | \u70b9\u6570 | \u7269\u7406\u9879\u6570 | **\u7269\u7406 MAPE** | \u8d2a\u5fc3\u5bf9\u7167 | \u5dee\u5f02 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    TOT = 0.0; N = 0
    for k in keys:
        v = m[k]; rk = int(k.rsplit("_r", 1)[1]); sl = k.rsplit("_r", 1)[0]
        scen = SLOT_DOC.get(k, (sl, ""))[0]
        g = GREEDY.get(k)[0] if GREEDY.get(k) else None
        gt = GREEDY.get(k)[1] if GREEDY.get(k) else 0
        d = ("%+.2f" % (v["mape"] - g)) if g is not None else "-"
        L.append("| %s%s%s | %s | %d | %d | **%.2f%%** | %.2f%% (%d \u9879) | %s |" % (
            Q, k, Q, scen, v.get("n", 0), v.get("n_terms", 0), v["mape"],
            g if g is not None else 0, gt, d))
        TOT += v["mape"] * v.get("n", 0); N += v.get("n", 0)
    L.append("| | | **%d** | | **%.2f%%** | | |" % (N, TOT / N if N else 0))
    L += ["", "---", "", "## \u8de8\u516c\u5f0f\u7684\u5173\u952e\u7269\u7406\u53d1\u73b0", "",
      "### A. MTE \u6700\u5c0f burst = 32 \u5b57\u8282\u662f\u786c\u95e8\u69db", "",
      "`C * eb < 32`\uff08\u4e00\u884c\u7684\u6709\u6548\u8f7d\u8377\u88c5\u4e0d\u6ee1\u4e00\u4e2a\u6700\u5c0f burst\uff09\u65f6\uff0c\u7f16\u8bd1\u5668\u65e0\u6cd5\u7528\u300c\u8fde\u7eed\u5757\u642c\u8fd0 + extract_slice\u300d\uff0c",
      "\u9000\u5316\u4e3a\u9010\u5143\u7d20\u547d\u4ee4\u8def\u5f84\uff0c\u5355\u5143\u7d20\u4ee3\u4ef7\u9ad8 10\u201330 \u500d\u3002",
      "**\u4e24\u4e2a\u5b50\u4ee3\u7406\u72ec\u7acb\u53d1\u73b0\u4e86\u540c\u4e00\u95e8\u69db**\uff08`simd_load_wide_r2` \u7684 src=32B gate\u3001`simd_load_gather_r2` \u7684 C\u00b7eb=32B \u5206\u533a\uff09\u3002",
      "\u8fd9\u662f\u4e4b\u524d\u6240\u6709 `a + b\u00b7R\u00b7max(32,row_bytes)` \u578b\u516c\u5f0f\u5361\u5728 26\u201328% \u7684\u539f\u56e0\uff1a\u5b83\u4eec\u53ea\u63cf\u8ff0\u4e86\u5bbd\u8def\u5f84\u3002", "",
      "### B. \u51b2\u7a81\u7c92\u5ea6\u56e0\u573a\u666f\u800c\u5f02\uff08\u4e0d\u662f\u7edf\u4e00\u7684 128B\uff09", "",
      "| \u573a\u666f | \u6700\u4f18\u7c92\u5ea6 | \u6765\u6e90 |",
      "|---|---|---|",
      "| SIMD gather 1D | **256 B** | \u626b\u63cf B\u2208{16..4096} |",
      "| SIMD gather 2D | **128 B** | \u540c\u4e0a\uff0c2D \u68b3\u72b6\u5730\u5740\u96c6 |",
      "| SIMT load 1D | **256 B** | \u4e0e SIMD gather \u4e00\u81f4 |",
      "| SIMT load 2D | **1 KB** | conc1k |",
      "| SIMD store | **128 B** | \u5199\u56de\u8def\u5f84 RMW |",
      "| SIMT store 1D | **256 B** | \u5efa\u8bae\u7528 L256 \u800c\u975e L128 \u91cd\u7b97 |", "",
      "### C. \u62b5\u6263\u9879\uff08\u8d1f\u7cfb\u6570\uff09\u771f\u5b9e\u5b58\u5728", "",
      "\u540c\u4e00 line \u88ab\u591a\u884c\u547d\u4e2d\u65f6\u53ea\u53d6\u56de\u4e00\u6b21\uff0c\u6240\u4ee5\u51b2\u7a81\u60e9\u7f5a\u5e94\u88ab\u6298\u6263\u3002",
      "`simd_load_gather_r2` \u91cc `-71.63 \u00b7 [C\u00b7eb<32] \u00b7 multi@128`\uff08multi = \u88ab \u22652 \u884c\u547d\u4e2d\u7684 128B line \u6570\uff09\u4ef7\u503c 0.93 \u4e2a MAPE \u70b9\u3002", "",
      "### D. SIMT \u5185\u5b58\u4fa7\u4e8b\u52a1\u6570\u4fee\u6b63", "",
      "\u4e0d\u662f `iters \u00b7 mwl`\uff0c\u800c\u662f **`W \u00b7 iters \u00b7 mwl`**\uff08\u6240\u6709 W \u4e2a warp \u7684 line \u8bf7\u6c42\u90fd\u8ba1\u5165\uff0c\u5b9e\u6d4b \u22481.2 ns/\u8bf7\u6c42\uff09\u3002",
      "\u8fd9\u662f\u9ad8 W \u4e0b T \u7ee7\u7eed\u4e0a\u5347\u7684\u539f\u56e0\u3002", "",
      "### E. \u53cc\u74f6\u9888\u786e\u8ba4", "",
      "`T = max(k \u00b7 W, \u8bbf\u5b58\u4ee3\u4ef7)`\uff0c\u5b9e\u6d4b `k` = 1.54\u20132.43 ns/warp\uff08\u4e0e\u65e9\u671f 2.18 \u4e00\u81f4\uff09\u3002",
      "W\u226516 \u4e14 mwl\u22641 \u65f6 63 \u4e2a\u70b9\u5168\u843d\u5728 134\u2013162 ns\uff0c\u4e0e\u5143\u7d20\u6570\u65e0\u5173\u3002", "",
      "### F. \u88ab\u8bc1\u4f2a\u7684\u5047\u8bbe\uff08\u90fd\u662f\u539f\u6a21\u578b\u91cc\u81ea\u8bbe\u7684\uff09", "",
      "| \u5047\u8bbe | \u51fa\u5904 | \u5b9e\u6d4b\u7ed3\u8bba |",
      "|---|---|---|",
      "| **128B cache line** | `build_v4_numwarps_dataset.py:32` | \u274c \u5bf9 MTE2 \u4e0d\u9002\u7528\uff1b\u51b2\u7a81\u7c92\u5ea6\u968f\u573a\u666f\u4e3a 128B\u20131KB |",
      "| **4 KB page** | \u540c\u6587\u4ef6 :35 | \u274c CPU \u6982\u5ff5\uff1b4KB \u9644\u8fd1\u73b0\u8c61\u662f**\u53cc\u7a33\u6001**\uff0c\u4e0d\u662f\u5e72\u51c0\u9608\u503c |",
      "| **stride=1 \u5c5e\u4e8e wide \u5206\u652f** | \u7167\u642c C++ \u8c13\u8bcd | \u26a0\u63a2\u9488\u4e3b\u52a8\u62d2\u7edd\u6d4b stride=1\uff1bextract_slice \u4ee3\u4ef7\u5b9e\u6d4b \u2248 0 |", "",
      "### G. \u6570\u636e\u9677\u9631\uff08\u5f71\u54cd\u7ed3\u8bba\u53ef\u4fe1\u5ea6\uff09", "",
      "1. `simd_load_wide_r1` \u7684 32 \u70b9 stride \u5168\u662f 2\uff0cspan \u4e0e n\u00b7eb \u5b8c\u5168\u5171\u7ebf\uff0c**\u65e0\u6cd5\u5206\u79bb n \u4e0e eb**\u3002",
      "2. `simd_store` \u4e24\u4e2a\u69fd\u90fd**\u6ca1\u6709 ISB==eb \u7684\u8fde\u7eed\u5199\u5f62\u72b6**\uff0c\u7981\u6b62\u5916\u63a8\u56de\u8fde\u7eed\u5199\u3002",
      "3. `simd_load_gather_r2` \u5b50\u4ee3\u7406**\u590d\u73b0\u4e0d\u51fa 73 \u9879 = 11.08%**\uff08\u5b83\u7528 614 \u952e\u6c60\u9971\u548c\u5230 16.12%\uff0c624 \u952e\u5e76\u96c6\u5230 13.57% in-sample / 17.57% CV\uff09\u3002\u8be5\u5bf9\u7167\u503c\u9700\u6838\u5b9e\u3002",
      "4. \u63a2\u9488 rotate \u7a97\u53e3\u662f step=8192 \u5143\u7d20\uff1b\u5f53 `n\u00b7stride > 8192` \u65f6\u76f8\u90bb\u8fed\u4ee3\u5199\u5165\u533a\u57df\u91cd\u53e0\uff0c\u659c\u7387\u53ef\u80fd\u88ab\u7f13\u5b58\u547d\u4e2d\u538b\u4f4e\u3002",
      "5. \u5b9e\u6d4b\u5b58\u5728**\u53cc\u7a33\u6001**\uff1a\u540c\u4e00\u914d\u7f6e\u4e24\u6b21\u72ec\u7acb\u8fd0\u884c\u53ef\u5dee 2.6 \u500d\uff08\u5982 `float32 s1 block=1024`\uff09\u3002", "",
      "---", ""]
    for k in keys:
        v = m[k]; rk = int(k.rsplit("_r", 1)[1]); sl = k.rsplit("_r", 1)[0]
        scen, desc = SLOT_DOC.get(k, (sl, ""))
        L.append("## %s%s%s \u2014 %s" % (Q, k, Q, scen)); L.append("")
        L.append("### 1. \u76ee\u6807\u573a\u666f\u4e0e\u5b9e\u6d4b\u6761\u4ef6\u6784\u6210"); L.append("")
        L.append(desc); L.append("")
        L += composition(k); L.append("")
        if v.get("derived"):
            L.append("**\u6d3e\u751f\u7269\u7406\u91cf\uff1a**"); L.append("")
            for d0 in v["derived"]: L.append("- %s" % d0)
            L.append("")
        L.append("### 2. \u516c\u5f0f\u4e0e\u5404\u9879\u7269\u7406\u610f\u4e49"); L.append("")
        L.append(FENCE); L.append(v.get("formula", "")); L.append(FENCE); L.append("")
        L.append("| \u9879 | \u7cfb\u6570 | \u7269\u7406\u610f\u4e49 |")
        L.append("|---|---:|---|")
        for t in v.get("terms", []):
            L.append("| %s%s%s | %.6g | %s |" % (Q, t.get("name",""), Q, t.get("coef",0), t.get("meaning","")))
        L += ["", "### 3. in-sample \u62df\u5408\u8bef\u5dee", "",
              "- \u70b9\u6570\uff1a**%d**" % v.get("n",0),
              "- \u7269\u7406\u9879\u6570\uff1a**%d**" % v.get("n_terms",0),
              "- **MAPE\uff1a%.2f%%**" % v["mape"]]
        g = GREEDY.get(k)
        if g is not None:
            L.append("- \u8d2a\u5fc3\u5bf9\u7167\uff1a**%d \u9879 / %.2f%%**\uff08\u7269\u7406\u5f0f\u5dee\u5f02 %+.2f \u4e2a\u70b9\uff0c\u9879\u6570\u7ea6\u4e3a 1/%d\uff09" % (g[1], g[0], v["mape"] - g[0], max(1, round(g[1]/max(1, v.get("n_terms",1))))))
        if v.get("notes"):
            L += ["", "**\u7269\u7406\u6d1e\u5bdf\u4e0e\u5c40\u9650\uff1a**", "", v["notes"]]
        L += ["", "---", ""]
    out = RES / "REPORT_v4_physics.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote %s  (%d formulas, overall %.2f%%)" % (out.name, len(keys), TOT / N if N else 0))

if __name__ == "__main__":
    main()