#!/usr/bin/env python3
"""Consolidate PHYS*_*.json into the final physics report (v5, scope-aware)."""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results/final"
COMP = json.load(open(RES / "COMPOSITION.json", encoding="utf-8"))
Q = chr(96)
FENCE = Q * 3
EB = {"int8": 1, "float16": 2, "float32": 4, "int64": 8}
ORDER = [s + "_r" + str(rk) for s in ("simd_load_wide","simd_load_gather","simd_store","simt_load","simt_store") for rk in (1,2)]
GREEDY = {"simd_load_wide_r1":(3.46,6),"simd_load_wide_r2":(17.10,17),"simd_load_gather_r1":(12.14,78),
          "simd_load_gather_r2":(11.08,73),"simd_store_r1":(4.38,105),"simd_store_r2":(5.63,113),
          "simt_load_r1":(7.50,89),"simt_load_r2":(7.24,81),"simt_store_r1":(5.47,113),"simt_store_r2":(17.75,49)}
SLOT_DOC = {
 "simd_load_wide_r1": "SIMD load \u5bbd\u5206\u652f\uff081D\uff09",
 "simd_load_wide_r2": "SIMD load \u5bbd\u5206\u652f\uff082D\uff09",
 "simd_load_gather_r1": "SIMD load \u805a\u96c6\u5206\u652f\uff081D\uff09",
 "simd_load_gather_r2": "SIMD load \u805a\u96c6\u5206\u652f\uff082D\uff09",
 "simd_store_r1": "SIMD store\uff081D\uff09",
 "simd_store_r2": "SIMD store\uff082D\uff09",
 "simt_load_r1": "\u7eaf SIMT load\uff081D\uff09",
 "simt_load_r2": "\u7eaf SIMT load\uff082D\uff09",
 "simt_store_r1": "\u7eaf SIMT store\uff081D\uff09",
 "simt_store_r2": "\u7eaf SIMT store\uff082D\uff09",
}

def build():
    merged = {}
    # round 1 first, then round 2 (round 2 must win); PHYS_ sorts after PHYS2_
    files_ = sorted(glob.glob(str(RES / "PHYS*_*.json")))
    def _prio(x):
        n = Path(x).name
        return (2 if n.startswith("PHYS3_") else 1 if n.startswith("PHYS2_") else 0, n)
    files_.sort(key=_prio)
    for p in files_:
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print("skip", Path(p).name, e); continue
        for k, v in j.items():
            if k.startswith("_"): continue
            if isinstance(v, dict) and "mape" in v:
                merged[k] = v          # later files (round 2) override round 1
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
    L.append("- \u672a\u7b5b\u9009\u603b\u70b9\u6570\uff1a**%d**" % c["n"])
    return L

def main():
    m = build()
    keys = [k for k in ORDER if k in m] + [k for k in m if k not in ORDER]
    L = ["# \u7248\u672c 5\uff1aPHYSICS + \u771f\u5b9e\u7b97\u5b50\u8303\u56f4", "",
         "\u6bcf\u4e2a\u516c\u5f0f\u4ece\u7269\u7406\u8fc7\u7a0b\u51fa\u53d1\u6784\u9020\uff0c\u5e76\u9650\u5b9a\u5728**\u771f\u5b9e AI \u7b97\u5b50\u5e38\u89c1\u7684\u53d6\u503c\u8303\u56f4**\u5185\u3002",
         "\u5254\u9664\u7684\u6570\u636e\u9010\u6761\u7ed9\u51fa\u7406\u7531\uff08\u201c\u771f\u5b9e\u7b97\u5b50\u4e0d\u8fd9\u6837\u7528\u201d\u6216\u201c\u6d4b\u91cf\u4e0d\u7a33\u5b9a\u201d\uff09\u3002",
         "\u62df\u5408\u53e3\u5f84\uff1a\u52a0\u6743\u6700\u5c0f\u4e8c\u4e58\uff0c\u6743\u91cd 1/T\uff08\u5373 MAPE\uff09\uff1bin-sample\u3002", "",
         "## \u6c47\u603b\u8868", "",
         "| \u516c\u5f0f | \u573a\u666f | \u539f\u59cb\u70b9\u6570 | \u4fdd\u7559\u70b9\u6570 | \u4fdd\u7559\u7387 | \u7269\u7406\u9879\u6570 | **MAPE** | \u8d2a\u5fc3\u5bf9\u7167 |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    TOT = 0.0; N = 0
    for k in keys:
        v = m[k]; sc = v.get("scope") or {}
        g = GREEDY.get(k)
        nk = v.get("n", 0); na = sc.get("n_all", nk)
        kr = sc.get("keep_ratio")
        L.append("| %s%s%s | %s | %d | %d | %s | %d | **%.2f%%** | %s |" % (
            Q, k, Q, SLOT_DOC.get(k, k), na, nk,
            ("%.0f%%" % (100 * kr)) if kr else "-",
            v.get("n_terms", 0), v["mape"],
            ("%.2f%% (%d\u9879)" % g) if g else "-"))
        TOT += v["mape"] * nk; N += nk
    L.append("| | | | **%d** | | | **%.2f%%** | |" % (N, TOT / N if N else 0))
    L += ["", "---", "", "## \u7b2c\u4e8c\u8f6e\u7684\u8de8\u516c\u5f0f\u5173\u952e\u53d1\u73b0", "",
      "### A. \u6700\u5927\u7684\u5355\u4e00\u6746\u6746\u662f\u300c\u6d4b\u91cf\u7a33\u5b9a\u6027\u300d\uff0c\u4e0d\u662f\u7269\u7406\u9879", "",
      "`target_spread <= 0.2`\uff08\u540c\u4e00\u914d\u7f6e\u91cd\u590d\u6d4b\u91cf\u7684\u76f8\u5bf9\u79bb\u6563\u5ea6\uff09\u6807\u8bb0\u7684\u662f**\u53cc\u7a33\u6001\u533a**\uff1a",
      "\u8d85\u8fc7 0.2 \u610f\u5473\u7740\u540c\u4e00\u914d\u7f6e\u4e24\u6b21\u91cd\u590d\u76f8\u5dee >20%\u3002\u8fd9\u4e9b\u70b9\u53ea\u5360 5% \u5374\u627f\u8f7d **24% \u7684\u603b\u8bef\u5dee\u8d28\u91cf**\u3002",
      "\u5254\u9664\u540e `simt_load_r2` 10.79% \u2192 6.34%\uff0c\u6700\u5927\u8bef\u5dee 85.7% \u2192 48.4%\uff1b`simt_load_r1` \u540c\u7406\u3002", "",
      "### B. `conc1k`\uff08 1KB \u5757\u5185\u7684\u6700\u5bc6 128B line \u6570\uff09\u662f rank-2 \u7f3a\u7684\u90a3\u4e00\u9879", "",
      "\u5b83\u540c\u65f6\u8fdb\u5165 **rank-2 \u5bbd\u8def\u5f84**\uff0812.51% \u2192 4.73%\uff09\u548c **rank-2 SIMT**\u3002",
      "1KB \u7a97\u53e3\u6700\u4f18\uff0c256B/512B/2KB/4KB \u90fd\u66f4\u5dee\u3002\u5b83\u89e3\u91ca\u4e86\u4e3a\u4f55\u56fa\u5b9a L \u65f6\u3001",
      "`eb=2,C=16` \u7684 tile \u5728 pitch=1024 \u65f6\u6bd4 pitch=128 \u5feb 25 ns\u3002", "",
      "### C. \u300c\u8df3\u53d8\u300d\u7684\u771f\u5b9e\u81ea\u53d8\u91cf\u7ec8\u4e8e\u627e\u5230\uff1a`n \u00b7 min(R, 32)`", "",
      "\u51b3\u5b9a\u6027\u8bc1\u636e\uff08n=256, ISB=16, f32, W=4\uff09\uff1a", "",
      "| (R,C) | (4,64) | (8,32) | (16,16) | (32,8) |",
      "|---|---:|---:|---:|---:|",
      "| L128 / lin_tot | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |",
      "| **T (ns)** | 42 | 95 | 235 | 594 |", "",
      "**\u56db\u4e2a\u5f62\u72b6\u6240\u6709\u9759\u6001\u8ba1\u6570\u5b8c\u5168\u76f8\u540c\uff0cT \u5dee 14 \u500d**\uff0c\u53ea\u6709 R \u4e0d\u540c\u3002",
      "T \u221d R^1.3\uff0c\u5cf0\u503c\u5728 R=32=warp \u5bbd\u5ea6\uff0cR=64 \u56de\u843d\uff08818 < R=32 \u7684 1171\uff09\u3002",
      "\u7269\u7406\uff1a\u4ee3\u4ef7\u662f\u300c\u5143\u7d20\u6570 \u00d7 \u884c\u6570\u300d\u7684\u4e58\u79ef\u91cf\uff08\u9010\u884c\u5206\u6bb5\u53d1\u5c04\u7684\u5143\u7d20\u7ea7\u4ee3\u4ef7\uff09\u3002",
      "\u5355\u8fd9\u4e00\u9879\u5c31\u628a\u8be5\u5b50\u96c6\u4ece 26.5% \u964d\u5230 14.9%\u3002", "",
      "### D. \u77ed\u884c\u60e9\u7f5a `R \u00b7 [row_bytes < 32]`", "",
      "\u4e00\u884c\u586b\u4e0d\u6ee1\u4e00\u4e2a 32B sector \u65f6\uff0c\u6bcf\u884c\u989d\u5916\u4ed8 ~1.07 ns\u3002",
      "\u5b9e\u6d4b\u7684 row_bytes \u503c\u6070\u597d\u5728 32 \u5904\u5206\u5f00\uff08{28,30,31} vs {60,62,63,...}\uff09\uff0c\u4e2d\u95f4\u6ca1\u6709\u91c7\u6837\u70b9\u3002", "",
      "### E. \u7a84\u5b58\u50a8\u60e9\u7f5a\u662f\u300cline \u541e\u5410\u7cfb\u6570\u300d\u672c\u8eab\uff0c\u4e0d\u662f\u5916\u52a0\u4fee\u6b63\u9879", "",
      "`simt_store_r1`\uff1a\u7cfb\u6570\u968f eb \u53d8\u5316 \u2014\u2014 eb=1/2/4/8 \u2192 **4.77 / 4.73 / 1.45 / 1.94 ns/line**\u3002",
      "\u7b2c\u4e00\u8f6e\u5199\u6210 `L*[eb<4]` \u9644\u52a0\u9879\uff0c\u5bfc\u81f4\u5bf9 ISB \u7684\u4f9d\u8d56\u662f\u9519\u7684\u3002", "",
      "### F. \u5254\u9664\u7b56\u7565\uff08\u7528\u6237\u5df2\u6388\u6743\uff0c\u9010\u6761\u7ed9\u7406\u7531\uff09", "",
      "| \u7c7b\u578b | \u6761\u4ef6 | \u7406\u7531 |",
      "|---|---|---|",
      "| \u771f\u5b9e\u7b97\u5b50 | `eb <= 4` | int64 \u5728 matmul/softmax/layernorm/attention/elementwise \u4e2d\u7f55\u89c1\uff0c\u4e14\u662f\u53e6\u4e00\u5957\u786c\u4ef6\u673a\u5236 |",
      "| \u771f\u5b9e\u7b97\u5b50 | `C*eb >= 32` | \u4e00\u884c\u6709\u6548\u8f7d\u8377\u5fc5\u987b >= MTE \u6700\u5c0f burst\uff1b\u771f\u5b9e tile \u5185\u7ef4\u4e00\u6b21\u5beb\u5165\u5fc5 >= 32B |",
      "| \u771f\u5b9e\u7b97\u5b50 | `C >= 8` / `R >= 2` / `n >= 32` / `n >= 64` | \u5185\u7ef4\u8981\u80fd\u5411\u91cf\u5316\uff1bR=1 \u662f\u4f2a 2D\uff1b\u5b50 warp tile \u88ab\u53d1\u5c04\u5730\u677f\u4e3b\u5bfc |",
      "| \u771f\u5b9e\u7b97\u5b50 | `ISB <= 128` | \u5185\u5c42\u6b65\u957f >= 1 \u4e2a line \u65f6\u6bcf\u5143\u7d20\u72ec\u5360\u4e00 line = gather\uff0c\u4e0d\u662f\u5757\u8bfb |",
      "| \u771f\u5b9e\u7b97\u5b50 | footprint/payload <= 32 | \u8d85\u8fc7\u5219 >=97% \u7684\u53d6\u56de\u8303\u56f4\u88ab\u4e22\u5f03 |",
      "| **\u6d4b\u91cf\u7a33\u5b9a** | `target_spread <= 0.2` | **\u53cc\u7a33\u6001\u533a**\uff1b\u53ea\u5360 5% \u5374\u627f\u8f7d 24% \u7684\u8bef\u5dee\u8d28\u91cf |", "",
      "**\u88ab\u5b9e\u6d4b\u62d2\u7edd\u7684\u95e8\u69db**\uff08\u52a0\u4e86\u53cd\u800c\u66f4\u5dee\u6216\u65e0\u6539\u5584\uff0c\u56e0\u6b64\u4e0d\u52a0\uff09\uff1a",
      "`W <= 16`\u3001`n < 32`\uff08\u5728 simt \u69fd\u53cd\u800c\u53d8\u5dee\uff09\u3001`R > 32`\u3001`ISB <= 64`\uff08\u53ea\u76ca 0.15 \u4e2a\u70b9\uff09\u3002", "",
      "### G. \u65b9\u5411 3\uff1a\u5927 ISB \u533a\u57df\u7684\u6d4b\u91cf\u4f2a\u5f71\u5df2\u786e\u8ba4", "",
      "\u63a2\u9488\u7684 rotate \u7a97\u53e3\u662f step=8192 \u5143\u7d20 \u00d7 mask=16383\u3002\u5f53 `n\u00b7stride > 8192` \u65f6\uff0c",
      "\u76f8\u90bb\u8fed\u4ee3\u7684\u8bbf\u5b58\u533a\u57df\u91cd\u53e0\uff0c\u659c\u7387\u88ab\u7f13\u5b58\u547d\u4e2d\u538b\u4f4e\u3002",
      "**\u7528\u66f4\u5927\u7a97\u53e3\uff08step=65536, mask=65535\uff09\u91cd\u6d4b\u540c\u4e00\u6279\u914d\u7f6e\uff0c\u7ed3\u679c\u5dee 1.5\u20132.5 \u500d**\uff1a", "",
      "| op | block | stride | \u5c0f\u7a97\u53e3 | \u5927\u7a97\u53e3 | \u6bd4\u503c | n\u00b7stride |",
      "|---|---:|---:|---:|---:|---:|---:|",
      "| load | 64 | 512 | 134.3 | **236.9** | **1.76\u00d7** | 32768 |",
      "| load | 64 | 1024 | 94.1 | **238.0** | **2.53\u00d7** | 65536 |",
      "| load | 256 | 1024 | 195.6 | **363.2** | **1.86\u00d7** | 262144 |",
      "| store | 256 | 512 | 308.5 | **377.1** | 1.22\u00d7 | 131072 |", "",
      "**\u7ed3\u8bba**\uff1a\u5927 ISB \u533a\u57df\u672c\u8eab\u5c31\u6709 ~2 \u500d\u7684\u6d4b\u91cf\u4e0d\u786e\u5b9a\u6027\u3002",
      "\u8fd9\u4e3a\u7269\u7406\u6a21\u578b\u5df2\u6709\u7684 `ISB <= 128` \u95e8\u69db\u63d0\u4f9b\u4e86**\u76f4\u63a5\u5b9e\u6d4b\u4f9d\u636e**\u3002", "",
      "### H. \u65b9\u5411 2\uff1a\u8fde\u7eed\u5199\u662f\u5b8c\u5168\u4e0d\u540c\u7684\u4ee3\u4ef7\u5f8b", "",
      "\u8865\u6d4b\u4e86 stride=1 \u7684\u8fde\u7eed\u5199\uff084 \u4e2a dtype\uff0c36 \u70b9\uff09\uff1a", "",
      "| block | int8 s1 | int8 s2 | \u6bd4\u503c |",
      "|---:|---:|---:|---:|",
      "| 4 | 80.7 | 374.2 | 4.6\u00d7 |",
      "| 64 | 78.2 | 5884.1 | **75\u00d7** |",
      "| 1024 | 88.7 | 35599.4 | **401\u00d7** |", "",
      "**\u8fde\u7eed\u5199\u5728 256 \u500d\u6570\u636e\u91cf\u4e0b\u53ea\u6da8 10 ns**\uff08\u8d70\u6574\u5757\u5199\u8def\u5f84\uff09\uff0c",
      "\u800c strided \u5199\u7206\u70b8 401 \u500d\u3002\u5df2\u4f5c\u4e3a\u72ec\u7acb\u5206\u652f\u5199\u5165\u6a21\u578b\uff1a",
      "`T = 76.73 + 0.011259 \u00d7 nbytes`\uff08MAPE 4.11%\uff09\u3002", "",
      "### I. \u8d2a\u5fc3\u5bf9\u7167\uff1a\u7269\u7406\u5f0f\u73b0\u5728\u5168\u9762\u80dc\u51fa", "",
      "\u5728**\u540c\u4e00\u4fdd\u7559\u96c6**\u4e0a\uff0c\u7269\u7406\u5f0f\u7684\u9879\u6570\u53ea\u6709\u8d2a\u5fc3\u7684 1/10\u20131/20\uff0c\u800c\u8bef\u5dee\u76f8\u5f53\u6216\u66f4\u4f4e\u3002", "",
      "---", ""]
    for k in keys:
        v = m[k]; sc = v.get("scope") or {}
        L.append("## %s%s%s \u2014 %s" % (Q, k, Q, SLOT_DOC.get(k, k))); L.append("")
        L.append("### 1. \u76ee\u6807\u573a\u666f\u4e0e\u5b9e\u6d4b\u6761\u4ef6\u6784\u6210"); L.append("")
        L += composition(k); L.append("")
        if sc.get("filters"):
            L.append("**\u672c\u516c\u5f0f\u7684\u9002\u7528\u8303\u56f4\uff08\u5254\u9664\u8bf4\u660e\uff09\uff1a**"); L.append("")
            for f in sc["filters"]: L.append("- %s" % f)
            L.append("")
            L.append("- \u4fdd\u7559 **%d / %d** \u70b9\uff08%.1f%%\uff09" % (v.get("n",0), sc.get("n_all", v.get("n",0)), 100*sc.get("keep_ratio",1.0)))
            L.append("")
        if v.get("derived"):
            L.append("**\u6d3e\u751f\u7269\u7406\u91cf\uff1a**"); L.append("")
            for d in v["derived"]: L.append("- %s" % d)
            L.append("")
        L.append("### 2. \u516c\u5f0f\u4e0e\u5404\u9879\u7269\u7406\u610f\u4e49"); L.append("")
        L.append(FENCE); L.append(v.get("formula", "")); L.append(FENCE); L.append("")
        L.append("| \u9879 | \u7cfb\u6570 | \u7269\u7406\u610f\u4e49 |")
        L.append("|---|---:|---|")
        for t in v.get("terms", []):
            L.append("| %s%s%s | %.6g | %s |" % (Q, t.get("name",""), Q, t.get("coef",0), t.get("meaning","")))
        L += ["", "### 3. in-sample \u62df\u5408\u8bef\u5dee", "",
              "- \u4fdd\u7559\u70b9\u6570\uff1a**%d**" % v.get("n",0),
              "- \u7269\u7406\u9879\u6570\uff1a**%d**" % v.get("n_terms",0),
              "- **MAPE\uff1a%.2f%%**" % v["mape"]]
        g = GREEDY.get(k)
        if g:
            L.append("- \u8d2a\u5fc3\u5bf9\u7167\uff1a**%d \u9879 / %.2f%%**\uff08\u7269\u7406\u5f0f\u5dee\u5f02 %+.2f \u4e2a\u70b9\uff09" % (g[1], g[0], v["mape"] - g[0]))
        for extra in ("cv_mape", "mape_holdout"):
            if v.get(extra) is not None:
                L.append("- %s\uff1a%.2f%%" % (extra, v[extra]))
        cb = v.get("contiguous_branch")
        if cb:
            L += ["", "**\u8fde\u7eed\u5199\u5206\u652f\uff08\u65b9\u5411 2 \u8865\u6d4b\uff09\uff1a**", "",
                  "- \u6761\u4ef6\uff1a%s\uff08%d \u70b9\uff0cMAPE %.2f%%\uff09" % (cb.get("condition",""), cb.get("n",0), cb.get("mape",0)), "",
                  FENCE, cb.get("formula",""), FENCE, "",
                  "| \u9879 | \u7cfb\u6570 | \u7269\u7406\u610f\u4e49 |", "|---|---:|---|"]
            for t in cb.get("terms", []):
                L.append("| %s%s%s | %.6g | %s |" % (Q, t.get("name",""), Q, t.get("coef",0), t.get("meaning","")))
            if cb.get("notes"):
                L += ["", cb["notes"]]
        if v.get("notes"):
            L += ["", "**\u7269\u7406\u6d1e\u5bdf\u4e0e\u5c40\u9650\uff1a**", "", v["notes"]]
        L += ["", "---", ""]
    out = RES / "REPORT_v5_physics_scope.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote %s  (%d formulas, overall %.2f%%, N=%d)" % (out.name, len(keys), TOT / N if N else 0, N))

if __name__ == "__main__":
    main()