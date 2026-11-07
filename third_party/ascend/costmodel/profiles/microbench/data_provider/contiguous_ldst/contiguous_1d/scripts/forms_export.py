#!/usr/bin/env python3
"""Export FormS spec: SIMD formulas (A/B merged refit) + te>=8 tables + SIMT full tables."""
import json, math
import numpy as np

rows = [json.loads(l) for l in open("/tmp/ss_final.jsonl")]
gap = [json.loads(l) for l in open("/tmp/ss_gapfill.jsonl")]
for g in gap:
    if g.get("ns") is not None:
        rows.append(g)
ok = [r for r in rows if r.get("ns") is not None and r["ns"] > 0]
BYS = [64, 256, 1024, 4096, 16384]

def bpt(r): return r["by"] / (32.0 * r["warps"])

# --- SIMD merged A/B formulas (v3 §3.1 rule), te<8 domain
forms = {"description": "FormS single-shot (u==1, non-pipelined) pricing; "
         "SIMD A/B merged per whole-block-predicate rule; SIMT full tables; "
         "measured 2026-11-07 card2, fixed-address, control-subtraction."}
simd_f = {}
for d in ("load", "store"):
    fam = [r for r in ok if r["mode"] == "simd" and r["dir"] == d
           and r["by"]/(r["eb"]*32.0*r["warps"]) < 8 and r["ns"] >= 20]
    if d == "store":  # 小 store 全部 <20ns: 用全部正点
        fam = [r for r in ok if r["mode"] == "simd" and r["dir"] == d
               and r["by"]/(r["eb"]*32.0*r["warps"]) < 8]
    X = np.array([[1.0, bpt(r), math.log2(r["eb"])] for r in fam])
    y = np.array([r["ns"] for r in fam])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ape = np.abs(X @ coef - y) / y
    simd_f[d] = {"features": ["one", "bpt", "log2eb"],
                 "coefficients": [round(c, 3) for c in coef],
                 "n": len(fam), "mape": round(100*ape.mean(), 2),
                 "domain": "te<8, A/B merged"}
forms["simd_formulas"] = simd_f

# --- SIMD te>=8 tables: by -> ns, per direction, per dtype, A/B separate
simd_t = {}
for d in ("load", "store"):
    for mk, tag in ((0, "A"), (1, "B")):
        for dt, eb in (("int8",1),("fp16",2),("fp32",4),("int64",8)):
            fam = [r for r in ok if r["mode"]=="simd" and r["dir"]==d and int(r["masked"])==mk
                   and r["dtype"]==dt and r["by"]/(eb*32.0) >= 8]
            if not fam:
                continue
            pts = {}
            for r in fam:
                pts[r["by"]] = min(pts.get(r["by"], 1e9), r["ns"])
            simd_t["%s.%s.eb%d" % (d, tag, eb)] = dict(sorted(pts.items()))
forms["simd_tables_te8"] = simd_t

# --- SIMT full tables: (dir, form, dtype, W) -> {by: ns}
simt_t = {}
for d in ("load", "store"):
    for mk, tag in ((0, "A"), (1, "B")):
        for dt, eb in (("int8",1),("fp16",2),("fp32",4),("int64",8)):
            for W in (1,2,4,8,16,32,64):
                fam = [r for r in ok if r["mode"]=="simt_only" and r["dir"]==d
                       and int(r["masked"])==mk and r["dtype"]==dt and r["warps"]==W]
                pts = {}
                for r in fam:
                    pts[r["by"]] = min(pts.get(r["by"], 1e9), r["ns"])
                for b in BYS:  # gap fill: neighbor interpolation
                    if b not in pts:
                        cand = sorted(pts)
                        if cand:
                            pts[b] = pts[cand[0]] if b < cand[0] else pts[cand[-1]]
                if pts:
                    simt_t["%s.%s.%s.W%d" % (d, tag, dt, W)] = dict(sorted((k, round(v,1)) for k,v in pts.items()))
forms["simt_tables"] = simt_t

json.dump(forms, open("/tmp/forms_spec.json", "w"), indent=1)
print("simd formulas:", {k: (v["coefficients"], "MAPE", v["mape"], "n", v["n"]) for k, v in simd_f.items()})
print("simd te>=8 tables:", len(simd_t), "keys")
print("simt tables:", len(simt_t), "keys (expect 2dir*2form*4dt*7W=112)")
