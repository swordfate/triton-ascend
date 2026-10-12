#!/usr/bin/env python3
"""Concise Chinese report: only the FINAL retained set, no exclusion story."""
from __future__ import annotations
import sys, json, glob, collections, io
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_v9 import load
RES = HERE.parent / "results/final"
Q = chr(96)
FENCE = Q * 3
I = lambda x: int(float(x))
CV = lambda r: float(r.get("f_cols") or r["n"]) if int(r["rank"]) == 2 else float(r["n"])
RV = lambda r: float(r.get("f_rows") or 0.0)
SP = lambda r: float(r.get("target_spread") or 0.0)
FP = lambda r: (((RV(r)-1)*float(r["f_row_pitch_bytes"]) + (CV(r)-1)*float(r["ISB"]) + float(r["eb"]))
                / max(1.0, RV(r)*CV(r)*float(r["eb"]))) if int(r["rank"]) == 2 else 1.0

# cumulative filters, in the order the estimators applied them
def F_eb4(r):     return float(r["eb"]) <= 4
def F_sp(r):      return SP(r) <= 0.2
def F_isb128(r):  return float(r["ISB"]) <= 128
def F_isb128s(r): return float(r["ISB"]) <= 128
def F_r2(r):      return (RV(r) >= 2) if int(r["rank"]) == 2 else True
def F_c4(r):      return (CV(r) >= 4) if int(r["rank"]) == 2 else True
def F_c8(r):      return (CV(r) >= 8) if int(r["rank"]) == 2 else True
def F_n32(r):     return I(r["n"]) >= 32
def F_n64(r):     return I(r["n"]) >= 64
def F_ceb32(r):   return (CV(r) * float(r["eb"]) >= 32) if int(r["rank"]) == 2 else True
def F_fp32(r):    return FP(r) <= 32

SCOPE = {
 "simd_load_wide_r1":  [(F_eb4,"eb<=4"), (F_sp,"spread<=0.2")],
 "simd_load_wide_r2":  [(F_eb4,"eb<=4"), (F_r2,"R>=2"), (F_c4,"C>=4"), (F_sp,"spread<=0.2")],
 "simd_load_gather_r1":[(F_n32,"n>=32"), (F_isb128s,"ISB<=128"), (F_sp,"spread<=0.2")],
 "simd_load_gather_r2":[(F_r2,"R>=2"), (F_sp,"spread<=0.2"), (F_fp32,"footprint/payload<=32")],
 "simd_store_r1":      [(F_eb4,"eb<=4"), (F_isb128,"ISB<=128")],
 "simd_store_r2":      [(F_r2,"R>=2"), (F_ceb32,"C*eb>=32"), (F_eb4,"eb<=4")],
 "simt_load_r1":       [(F_eb4,"eb<=4"), (F_isb128,"ISB<=128"), (F_sp,"spread<=0.2")],
 "simt_load_r2":       [(F_eb4,"eb<=4"), (F_r2,"R>=2"), (F_sp,"spread<=0.2")],
 "simt_store_r1":      [],
 "simt_store_r2":      [(F_r2,"R>=2"), (F_ceb32,"C*eb>=32"), (F_c8,"C>=8"), (F_n64,"n>=64")],
}
ORDER = [s + "_r" + str(rk) for s in ("simd_load_wide","simd_load_gather","simd_store","simt_load","simt_store") for rk in (1,2)]
SLOT = {"simd_load_wide":"SIMD load 宽分支","simd_load_gather":"SIMD load 聚集分支","simd_store":"SIMD store",
        "simt_load":"纯 SIMT load","simt_store":"纯 SIMT store"}
DTS = {"int8":"int8(1B)","float16":"float16(2B)","float32":"float32(4B)","int64":"int64(8B)"}

def raw_slot(rr0, sl, rk):
    if sl == "simd_load_wide":
        return [r for r in rr0 if r["path"]=="simd" and r["op"]=="load" and r["ISB"]<=2.0*r["eb"] and int(r["rank"])==rk]
    if sl == "simd_load_gather":
        return [r for r in rr0 if r["path"]=="simd" and r["op"]=="load" and r["ISB"]>2.0*r["eb"] and int(r["rank"])==rk]
    if sl == "simd_store":
        return [r for r in rr0 if r["path"]=="simd" and r["op"]=="store" and int(r["rank"])==rk]
    return [r for r in rr0 if r["path"]=="simt_only" and r["op"]==sl.split("_")[1] and int(r["rank"])==rk]

def apply_scope(key, rr):
    for f, _ in SCOPE.get(key, []):
        rr = [r for r in rr if f(r)]
    return rr

def fmt_counter(c, order=None):
    ks = order if order else sorted(c, key=lambda z: (z if isinstance(z,(int,float)) else 0))
    return "\u3001".join("%s(%d \u70b9)" % (k, c[k]) for k in ks)

def variables(key, rr):
    """From the measured knobs to the symbols used in the formula."""
    rk = int(key.rsplit("_r", 1)[1])
    L = []
    dt = collections.Counter(r["dtype"] for r in rr)
    dts = [d for d in ("int8","float16","float32","int64") if d in dt]
    L.append("- **dtype** \u2192 `eb` = \u5143\u7d20\u5b57\u8282\u6570\uff1a%s" %
             "\u3001".join("%s\u2192%d" % (d, {"int8":1,"float16":2,"float32":4,"int64":8}[d]) for d in dts))
    if rk == 1:
        ns = sorted({I(r["n"]) for r in rr})
        L.append("- **\u4e00\u6b21\u8bbf\u95ee\u7684\u5143\u7d20\u603b\u6570** \u2192 `n`\uff1a%s\uff08\u63a2\u9488\u9010\u6b21\u626b\u63cf\u7684\u5143\u7d20\u4e2a\u6570\uff09" % "\u3001".join(str(x) for x in ns))
    else:
        sh = collections.Counter("%dx%d" % (I(RV(r)), I(CV(r))) for r in rr)
        ks = sorted(sh, key=lambda z: (int(z.split("x")[0])*int(z.split("x")[1]), int(z.split("x")[0])))
        L.append("- **2D \u5f62\u72b6 (R\u00d7C)**\uff08%d \u79cd\uff09\uff1a%s" % (len(sh), "\u3001".join(ks)))
        L.append("- \u21d2 `R` = \u884c\u6570\u3001`C` = \u6bcf\u884c\u5143\u7d20\u6570\uff0c`n` = R \u00d7 C = \u5143\u7d20\u603b\u6570")
    strides = sorted({round(float(r["ISB"]) / float(r["eb"])) for r in rr})
    isbs = sorted({I(r["ISB"]) for r in rr})
    L.append("- **\u5143\u7d20\u57df\u6b65\u957f stride**\uff08\u76f8\u90bb\u4e24\u4e2a\u88ab\u8bbf\u95ee\u5143\u7d20\u4e4b\u95f4\u76f8\u9694\u7684\u5143\u7d20\u4e2a\u6570\uff09\uff1a%s" % "\u3001".join(str(x) for x in strides))
    L.append("- \u21d2 **`ISB` = stride \u00d7 eb** = \u5185\u5c42\u6b65\u957f\uff08\u5b57\u8282\uff09\uff1a%s" % "\u3001".join(str(x) for x in isbs))
    if rk == 2:
        ps = sorted({round(float(r["f_row_pitch_bytes"]) / float(r["eb"])) for r in rr})
        pb = sorted({I(r["f_row_pitch_bytes"]) for r in rr})
        L.append("- **\u884c\u95f4\u8ddd row_stride**\uff08\u5143\u7d20\u57df\uff09\uff1a%s" % "\u3001".join(str(x) for x in ps))
        L.append("- \u21d2 **`pitch` = row_stride \u00d7 eb** = \u884c\u95f4\u8ddd\uff08\u5b57\u8282\uff09\uff1a%s" % "\u3001".join(str(x) for x in pb))
    ws = sorted({I(r["W"]) for r in rr})
    L.append("- **num_warps** \u2192 `W`\uff1a%s" % "\u3001".join(str(x) for x in ws))
    L.append("- **\u603b\u70b9\u6570**\uff1a**%d**" % len(rr))
    return L
def composition(key, rr):
    rk = int(key.rsplit("_r",1)[1])
    L = []
    L.append("- **\u8def\u5f84 / \u64cd\u4f5c / \u79e9**\uff1a%s / %s / rank-%d" % (rr[0]["path"], rr[0]["op"], rk))
    dt = collections.Counter(r["dtype"] for r in rr)
    L.append("- **dtype**\uff1a%s" % fmt_counter(dt, [d for d in ("int8","float16","float32","int64") if d in dt]))
    ws = collections.Counter(I(r["W"]) for r in rr)
    L.append("- **num_warps**\uff1a%s" % fmt_counter(ws))
    if rk == 2:
        sh = collections.Counter("%dx%d" % (I(RV(r)), I(CV(r))) for r in rr)
        ks = sorted(sh, key=lambda z: (int(z.split("x")[0])*int(z.split("x")[1]), int(z.split("x")[0])))
    else:
        sh = collections.Counter(I(r["n"]) for r in rr)
        ks = sorted(sh)
    L.append("- **%s**\uff08%d \u79cd\uff09\uff1a%s" % ("\u5f62\u72b6 R\u00d7C" if rk==2 else "\u5143\u7d20\u6570", len(sh), "\u3001".join(str(k) for k in ks)))
    isb = collections.Counter(I(r["ISB"]) for r in rr)
    L.append("- **\u5185\u5c42\u6b65\u957f\uff08\u5b57\u8282\uff09**\uff08%d \u79cd\uff09\uff1a%s" % (len(isb), "\u3001".join(str(k) for k in sorted(isb))))
    L.append("- **\u603b\u70b9\u6570**\uff1a**%d**" % len(rr))
    return L

def main():
    rows = load(str(RES / "dataset_new_only_v5.csv"))
    rr0 = [r for r in rows if str(r.get("valid")) == "1"]
    m = {}
    files_ = sorted(glob.glob(str(RES / "PHYS*_*.json")))
    def _prio(x):
        n = Path(x).name
        return (2 if n.startswith("PHYS3_") else 1 if n.startswith("PHYS2_") else 0, n)
    files_.sort(key=_prio)
    for p in files_:
        j = json.load(io.open(p, encoding="utf-8"))
        for k, v in j.items():
            if not k.startswith("_") and isinstance(v, dict) and "mape" in v:
                m[k] = v
    keys = [k for k in ORDER if k in m]
    kept = {}
    for k in keys:
        sl, rk = k.rsplit("_r", 1)
        kept[k] = apply_scope(k, raw_slot(rr0, sl, int(rk)))
    L = ["# \u7248\u672c 5\uff08\u7cbe\u7b80\u63cf\u8ff0\u7248\uff09\uff1a\u7269\u7406\u7b2c\u4e00\u6027\u4ee3\u4ef7\u6a21\u578b", "",
         "\u6bcf\u4e2a\u516c\u5f0f\u7531\u7269\u7406\u8fc7\u7a0b\u76f4\u63a5\u6784\u9020\uff08\u4f20\u8f93\u5b57\u8282\u6570\u3001BIU\uff08\u603b\u7ebf\u63a5\u53e3\u5355\u5143\uff0c\u8d1f\u8d23\u4e0b\u53d1\u8bbf\u5b58\u547d\u4ee4\uff09\u547d\u4ee4\u6570\u3001\u540c\u5757\u51b2\u7a81\u3001warp\uff0832 \u4e2a\u7ebf\u7a0b\u4e00\u7ec4\uff09\u53d1\u5c04\u5f00\u9500\u7b49\uff09\uff0c",
         "\u4e0d\u4f7f\u7528 log/\u6307\u6570/\u5e42\u6b21\uff0c\u4e5f\u4e0d\u5806\u780c\u65e0\u7269\u7406\u610f\u4e49\u7684\u4e58\u79ef\u56e0\u5b50\u3002",
         "\u62df\u5408\u53e3\u5f84\uff1a\u52a0\u6743\u6700\u5c0f\u4e8c\u4e58\uff0c\u6743\u91cd `1/T`\uff08\u7b49\u4ef7\u4e8e\u6700\u5c0f\u5316\u76f8\u5bf9\u8bef\u5dee\uff0c\u5373 MAPE\uff09\uff0cin-sample\u3002", "",
         "## \u6c47\u603b\u8868", "",
         "| \u516c\u5f0f | \u573a\u666f | \u70b9\u6570 | \u7269\u7406\u9879\u6570 | **MAPE** |",
         "|---|---|---:|---:|---:|"]
    TOT = 0.0; N = 0
    for k in keys:
        v = m[k]; sl, rk = k.rsplit("_r", 1)
        L.append("| %s%s%s | %s\uff08rank-%s\uff09 | %d | %d | **%.2f%%** |" % (
            Q, k, Q, SLOT[sl], rk, len(kept[k]), v.get("n_terms", 0), v["mape"]))
        TOT += v["mape"] * len(kept[k]); N += len(kept[k])
    L.append("| | | **%d** | | **%.2f%%** |" % (N, TOT/N))
    for k in keys:
        v = m[k]; sl, rk = k.rsplit("_r", 1)
        L.append("## %s%s%s \u2014 %s\uff08rank-%s\uff09" % (Q, k, Q, SLOT[sl], rk)); L.append("")
        L.append("### 1. \u5b9e\u6d4b\u6761\u4ef6\u4e0e\u53d8\u91cf\u5b9a\u4e49"); L.append("")
        L.append("**\u5b9e\u6d4b\u6761\u4ef6**"); L.append("")
        L.append("- \u8def\u5f84 / \u64cd\u4f5c / \u79e9\uff1a%s / %s / rank-%s" % (kept[k][0]["path"], kept[k][0]["op"], rk))
        L.append("")
        L.append("**\u53d8\u91cf\uff08\u7531\u5b9e\u6d4b\u6761\u4ef6\u6362\u7b97\uff09**"); L.append("")
        L += variables(k, kept[k]); L.append("")
        if v.get("derived"):
            L.append("**\u6d3e\u751f\u7269\u7406\u91cf\uff1a**"); L.append("")
            for d in v["derived"]: L.append("- %s" % d)
            L.append("")
        L.append("### 2. \u516c\u5f0f\u4e0e\u5404\u9879\u7269\u7406\u610f\u4e49"); L.append("")
        L.append(FENCE); L.append(v.get("formula","")); L.append(FENCE); L.append("")
        L.append("| \u9879 | \u7cfb\u6570 | \u7269\u7406\u610f\u4e49 |"); L.append("|---|---:|---|")
        for t in v.get("terms", []):
            L.append("| %s%s%s | %.6g | %s |" % (Q, t.get("name",""), Q, t.get("coef",0), t.get("meaning","")))
        cb = v.get("contiguous_branch")
        if cb:
            L += ["", "**\u8fde\u7eed\u5199\u5206\u652f**\uff08\u6761\u4ef6\uff1a%s\uff0c%d \u70b9\uff0cMAPE %.2f%%\uff09\uff1a" % (cb.get("condition",""), cb.get("n",0), cb.get("mape",0)), "",
                  FENCE, cb.get("formula",""), FENCE, ""]
            for t in cb.get("terms", []):
                L.append("- %s%s%s = %.6g\uff1a%s" % (Q, t.get("name",""), Q, t.get("coef",0), t.get("meaning","")))
        L += ["", "### 3. in-sample \u62df\u5408\u8bef\u5dee", "",
              "- \u70b9\u6570\uff1a**%d**" % len(kept[k]),
              "- \u7269\u7406\u9879\u6570\uff1a**%d**" % v.get("n_terms", 0),
              "- **MAPE\uff1a%.2f%%**" % v["mape"]]
        for e in ("cv_mape",):
            if v.get(e) is not None: L.append("- 5 \u6298\u4ea4\u53c9\u9a8c\u8bc1\uff1a%.2f%%" % v[e])
        L += ["", "---", ""]
    out = RES / "REPORT_v5_physics_scope_concise_des.md"
    out.write_text("\n".join(L), encoding="utf-8")
    # --- \u9996\u6b21\u51fa\u73b0\u5904\u52a0\u62ec\u53f7\u8bf4\u660e\uff08\u57fa\u7840\u7f29\u5199 SIMD/SIMT/MTE/L2/MAPE/CV \u4e0d\u6807\uff09---
    import re as _re
    ANN = [
        (r"(?<![A-Za-z])(descriptor)(?![A-Za-z])(?!\uff08)", "descriptor\uff08MTE \u642c\u8fd0\u63cf\u8ff0\u7b26\uff09"),
        (r"(?<![A-Za-z])(burst)(?![A-Za-z])(?!\uff08)", "burst\uff08MTE \u4e00\u6b21\u8fde\u7eed\u642c\u8fd0\u7684\u5355\u4f4d\uff09"),
        (r"(?<![A-Za-z])(sector)(?![A-Za-z])(?!\uff08)", "sector\uff08\u5185\u5b58\u6700\u5c0f\u53ef\u5bfb\u5740\u6247\u533a\uff09"),
        (r"(?<![A-Za-z])(LSU)(?![A-Za-z])(?!\uff08)", "LSU\uff08\u7ebf\u7a0b\u7ea7\u8bbf\u5b58\u5355\u5143\uff09"),
        (r"(?<![A-Za-z])(RMW)(?![A-Za-z])(?!\uff08)", "RMW\uff08\u8bfb-\u6539-\u5199\uff1a\u53ea\u5199\u5757\u7684\u4e00\u90e8\u5206\u65f6\u9700\u5148\u8bfb\u56de\u6574\u5757\u3001\u4fee\u6539\u3001\u518d\u5199\u56de\uff09"),
        (r"(?<![A-Za-z])(ramp)(?![A-Za-z])(?!\uff08)", "ramp\uff08\u6d41\u6c34\u7ebf\u672a\u586b\u6ee1\u65f6\u7684\u4f4e\u901f\u7387\u6bb5\uff09"),
        (r"(?<![A-Za-z])(hinge)(?![A-Za-z])(?!\uff08)", "hinge\uff08max(0, x\u2212K) \u5f62\u5f0f\u7684\u6298\u7ebf\u9879\uff09"),
    ]
    txt = out.read_text(encoding="utf-8")
    txt = txt.replace("\uff08ramp\uff09", " ramp").replace("\u6bb5 ramp", "\u6bb5 ramp")
    txt = txt.replace("\u6bb5ramp", "\u6bb5 ramp")
    for pat, rep in ANN:
        _mm = _re.search(pat, txt)
        if _mm:
            _tail = txt[_mm.end():]
            if _tail.startswith(" "): _tail = _tail[1:]
            txt = txt[:_mm.start()] + rep + _tail
    # BIU \u9664\u4e86 intro \u91cc\u5df2\u6807\u6ce8\u5916\uff0c\u5982\u679c\u9996\u6b21\u51fa\u73b0\u5728\u6b63\u6587\u5219\u8865\u6807
    if "BIU\uff08" not in txt.split("\n")[2]:
        _mm = _re.search(r"(?<![A-Za-z])(BIU)(?![A-Za-z])(?!\uff08)", txt)
        if _mm: txt = txt[:_mm.start()] + "BIU\uff08\u603b\u7ebf\u63a5\u53e3\u5355\u5143\uff09" + txt[_mm.end():]
    out.write_text(txt, encoding="utf-8")
    print("wrote %s  (%d formulas, N=%d, overall %.2f%%)" % (out.name, len(keys), N, TOT/N))
    for k in keys:
        print("   %-22s kept=%-5d json_n=%-5s mape=%.2f" % (k, len(kept[k]), m[k].get("n"), m[k]["mape"]))

if __name__ == "__main__":
    main()