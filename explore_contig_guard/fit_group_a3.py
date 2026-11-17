#!/usr/bin/env python3
"""Group_a fit v3: StridedShort-style structural factors.
  first    = a0 + a1*E + a2*B        (E=elements, B=bytes)
  marginal = b0 + b1*E + b2*B
  cT = kc*marginal ; Q structural (as shipped)
Features tried: [1,E,B] and [1,W,B] (W=ceil(E/32)).  OLS per
(engine, direction, residency); dtype NOT a feature - carried by E vs B.
"""
import json, math, statistics, sys
P='/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json'
prof=json.load(open(P))
ELEM={'int8':1,'fp16':2,'fp32':4,'int64':8}
def cells_of(mode):
    ga=prof[mode]['memory']['contiguous_ldst_v2']['group_a']
    out={}
    for k,v in ga.items():
        parts=k.split('/')
        out.setdefault((parts[1],parts[4]),[]).append(
            (int(parts[3]), ELEM[parts[2]], v['first'], v['marginal']))
    return out
def ols(X, y):
    n=len(X[0])
    # normal equations with simple gaussian elimination
    A=[[sum(X[i][r]*X[i][c] for i in range(len(X))) for c in range(n)]+[sum(X[i][r]*y[i] for i in range(len(X)))] for r in range(n)]
    for col in range(n):
        piv=max(range(col,n),key=lambda r:abs(A[r][col]))
        A[col],A[piv]=A[piv],A[col]
        pv=A[col][col]
        if abs(pv)<1e-12: return None
        A[col]=[x/pv for x in A[col]]
        for r in range(n):
            if r!=col and A[r][col]!=0:
                f=A[r][col]
                A[r]=[x-f*A[col][i] for i,x in enumerate(A[r])]
    return [A[i][n] for i in range(n)]
def run(mode, direction, res, feat):
    cells=[c for c in cells_of(mode).get((direction,res),[])]
    if len(cells)<6: return None
    X=[]; yF=[]; yM=[]
    for (B,eb,F,M) in cells:
        E=B//eb; W=(E+31)//32
        x=[1.0]+([E,B] if feat=='EB' else [W,B])
        X.append(x); yF.append(F); yM.append(M)
    cf=ols(X,yF); cm=ols(X,yM)
    if not cf or not cm: return None
    errs={'F':[],'M':[]}
    for i,(B,eb,F,M) in enumerate(cells):
        E=B//eb; W=(E+31)//32
        x=X[i]
        Fe=sum(c*v for c,v in zip(cf,x)); Me=sum(c*v for c,v in zip(cm,x))
        errs['F'].append(abs(Fe-F)/max(F,1)*100); errs['M'].append(abs(Me-M)/max(M,1)*100)
    return cf,cm,errs,cells
for feat in ('EB','WB'):
    print('=== features',feat)
    tot={'F':[],'M':[]}; bad=0; tot_cells=0
    for mode in ('simd','simt'):
        for d in ('load','store'):
            for r in ('warm','cold'):
                res=run(mode,d,r,feat)
                if not res: continue
                cf,cm,errs,cells=res
                tot_cells+=len(cells)
                for k in ('F','M'): tot[k]+=errs[k]
                bad+=sum(1 for e in errs['F'] if e>30)+sum(1 for e in errs['M'] if e>30)
                print(' %s %s/%s: cf=%s cm=%s  medF=%.1f%% medM=%.1f%%' % (
                    mode,d,r,[round(x,4) for x in cf],[round(x,4) for x in cm],
                    statistics.median(errs['F']),statistics.median(errs['M'])))
    print(' TOTAL cells %d  median F %.1f%% M %.1f%%  cells>30%%: %d' % (
        tot_cells, statistics.median(tot['F']), statistics.median(tot['M']), bad))
