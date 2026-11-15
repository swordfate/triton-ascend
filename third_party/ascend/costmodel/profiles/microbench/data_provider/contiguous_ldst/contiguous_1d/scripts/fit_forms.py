#!/usr/bin/env python3
"""Fit the FormS v2 formula spec from the raw calibration jsonl files.

Reads forms/raw/forms_single_shot_950pr{,_v5}.jsonl (v5 ABAB re-measurement
wins on key collisions), fits the 16 physics formulas and rewrites
forms/fitted/forms_spec.json in place.
"""
import json, math, os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.join(_HERE, "..", "forms", "raw", "forms_single_shot_950pr_v5.jsonl")
V4 = os.path.join(_HERE, "..", "forms", "raw", "forms_single_shot_950pr.jsonl")
SPEC = os.path.join(_HERE, "..", "forms", "fitted", "forms_spec.json")
EB={"int8":1,"fp16":2,"fp32":4,"int64":8}
by_key={}
for path in (V4,V5):
    try: fh=open(path)
    except OSError: continue
    for line in fh:
        r=json.loads(line)
        if r.get("status") not in ("ok","warn"): continue
        by_key[(r["mode"],r["warps"],r["dtype"],r["by"],r["masked"],r["dir"])]=r
rows=[]
for r in by_key.values():
    W,eb,by=r["warps"],EB[r["dtype"]],r["by"]
    bpt=by/(32.0*W); te=bpt/eb
    if r["mode"]=="simt_only" and te>64: continue
    if r["dir"]=="load" and r["ns"]<40: continue
    rows.append(dict(dir=r["dir"],form="B" if r["masked"] else "A",dtype=r["dtype"],mode=r["mode"],
        W=W,eb=eb,by=by,te=te,ns=float(r["ns"]),nI=max(1.0,math.ceil(bpt/16.0))))
F={"one":lambda r:1.0,"te":lambda r:r["te"],"teq":lambda r:max(0.0,r["te"]-1.0),
   "nI":lambda r:r["nI"],"bw":lambda r:r["by"]/256.0,"lines":lambda r:r["by"]/128.0,
   "lnW4":lambda r:(r["by"]/128.0)/max(1.0,r["W"]/4.0),
   "lnW8":lambda r:(r["by"]/128.0)/max(1.0,r["W"]/8.0),
   "pw":lambda r:(r["by"]/256.0)/r["W"],"pwsq":lambda r:(r["by"]/256.0)/math.sqrt(r["W"]),
   "l2w":lambda r:math.log2(r["W"]),"nIl2w":lambda r:r["nI"]*math.log2(r["W"]),
   "sqte":lambda r:math.sqrt(r["te"]),"log2eb":lambda r:math.log2(r["eb"]),
   "eb":lambda r:float(r["eb"]),"bpt":lambda r:r["te"]*r["eb"],
   "nImW":lambda r:r["nI"]*max(1.0,r["W"]/4.0)}
WS=[1,2,4,8,16,32,64]
for w in WS: F["wdr%d"%w]=(lambda w:(lambda r:(r["by"]/16384.0) if r["W"]==w else 0.0))(w)
simt=[r for r in rows if r["mode"]=="simt_only"]
simdt8=[r for r in rows if r["mode"]=="simd" and r["te"]>=8]
spec_in=json.load(open(SPEC))
out={"description":"FormS single-shot pricing v2 (2026-11-08): physics formulas from v5 ABAB-interleaved re-measurement (card2). "
      "128 measured tables -> 16 formulas. SIMT loads: canonical shape one+nI+lnW4+pwsq per (dir,form,dtype); "
      "load.B.int8 keeps its own shape (discontinuous byte-enable path). SIMT store: single all-dtype max(issue,drain) with per-W drain rates. "
      "SIMD te<8 formulas unchanged from v1.",
     "simd_formulas":spec_in["simd_formulas"],
     "simt_formulas":{},"simd_te8_formulas":{}}
report=[]
C2=["one","nI","lnW4","pwsq"]
for form in ("A","B"):
    for dt in ("int8","fp16","fp32","int64"):
        fam=[r for r in simt if r["dir"]=="load" and r["form"]==form and r["dtype"]==dt]
        keys=C2 if not (form=="B" and dt=="int8") else ["one","teq","pwsq","sqte"]
        X=np.array([[F[k](r) for k in keys] for r in fam]); y=np.array([r["ns"] for r in fam])
        coef,*_=np.linalg.lstsq(X,y,rcond=None)
        ape=np.abs(X@coef-y)/y
        key="load.%s.%s"%(form,dt)
        out["simt_formulas"][key]=dict(features=keys,coefficients=[round(c,3) for c in coef],
            n=len(fam),mape=round(100*ape.mean(),1),p95=round(100*np.percentile(ape,95),1),
            domain="simt, te<=64, W<=64")
        report.append(("simt "+key, keys, 100*ape.mean(), len(fam)))
# store: all-dtype, issue branch intercept pinned at 40ns floor
st=[r for r in simt if r["dir"]=="store"]
small=[r for r in st if r["by"]<=1024]
Xs=np.array([[F["nImW"](r)] for r in small]); ys=np.array([r["ns"] for r in small])-40.0
k=float(np.linalg.lstsq(Xs,ys,rcond=None)[0][0]); k=max(k,0.1)
ik=["one","nImW"]; ic=[40.0,k]
big=[r for r in st if r["by"]>=4096]
dk=["one"]+["wdr%d"%w for w in WS]
Xd=np.array([[F[c](r) for c in dk] for r in big]); yd=np.array([r["ns"] for r in big])
dc=np.linalg.lstsq(Xd,yd,rcond=None)[0]
def ev(r):
    a=ic[0]+ic[1]*F["nImW"](r)
    b=sum(c*F[c2](r) for c,c2 in zip(dc,dk))
    return max(40.0,a,b)
ape=np.array([abs(ev(r)-r["ns"])/max(r["ns"],40.0) for r in st])
apebig=np.array([abs(ev(r)-r["ns"])/max(r["ns"],40.0) for r in big])
out["simt_formulas"]["store"]=dict(features=ik,coefficients=[round(c,3) for c in ic],
    max_features=dk,max_coefficients=[round(c,3) for c in dc],
    n=len(st),mape=round(100*ape.mean(),1),mape_by_ge_4096=round(100*apebig.mean(),1),
    domain="simt store, all dtypes merged, te<=64, floor 40ns")
report.append(("simt store(all)", "max(one+nImW, one+wdr*)", 100*ape.mean(), len(st)))
# simd te>=8
for direction in ("load","store"):
    for form in ("A","B"):
        fam=[r for r in simdt8 if r["dir"]==direction and r["form"]==form]
        if len(fam)<6:
            report.append(("simd te8 %s.%s"%(direction,form),"SKIP few pts",0,len(fam))); continue
        best=None
        import itertools
        for combo in itertools.combinations([k for k in F if k not in ("one","wdr1","wdr2","wdr4","wdr8","wdr16","wdr32","wdr64","nImW","lnW4","lnW8","pw","pwsq","l2w","nIl2w")],3):
            keys=["one"]+list(combo)
            X=np.array([[F[c](r) for c in keys] for r in fam]); y=np.array([r["ns"] for r in fam])
            coef,*_=np.linalg.lstsq(X,y,rcond=None)
            ape=np.abs(X@coef-y)/y
            m=100*ape.mean()
            if best is None or m<best[0]: best=(m,keys,coef,100*np.percentile(ape,95))
        m,keys,coef,p95=best
        out["simd_te8_formulas"]["%s.%s"%(direction,form)]=dict(features=keys,
            coefficients=[round(c,3) for c in coef],n=len(fam),mape=round(m,1),p95=round(p95,1),
            domain="simd te>=8, floor: production clamps >=15ns")
        report.append(("simd te8 %s.%s"%(direction,form), keys, m, len(fam)))
json.dump(out,open(SPEC,"w"),indent=1)
print("%-24s %-34s %6s %5s %5s" % ("family","features","MAPE%","p95%","n"))
for name,keys,m,n in report:
    print("%-24s %-34s %6.1f %5s %5d" % (name,"+".join(keys) if isinstance(keys,list) else keys,m,"",n))
print("\nspec v2 written: %d simt formulas, %d simd te8 formulas, 3 simd te<8" % (len(out["simt_formulas"]),len(out["simd_te8_formulas"])))
