
import json, math, itertools
import numpy as np
BASE = r"D:\work_wj\triton-ascend\third_party\ascend\costmodel\profiles\microbench\data_provider\contiguous_ldst\pre-explore"
rows=[json.loads(l) for l in open(BASE+r"\raw\raw_formb_v2.jsonl") if l.strip()]
rows=[r for r in rows if r.get("status")=="ok" and r.get("spread",1)<=0.05]
ded={}
for r in rows: ded[(r["block"],r["direction"],r["dtype"],r["warps"])]=r
rows=list(ded.values())
EB={"int8":1,"fp16":2,"fp32":4,"int64":8}
REGB={1:508,2:508,4:508,8:508,16:256,32:128,64:64}
def F(r):
    B,W,eb=r["block"],r["warps"],EB[r["dtype"]]
    ept=B/(32.0*W); thr=ept*eb; by=B*eb
    spv=max(0.0,2*thr-REGB.get(W,128))/REGB.get(W,128)
    return dict(by=by, lines=by/128.0, thr=thr, ept=ept, rounds=math.ceil(ept),
                schedR=math.ceil(W/4), W2=W*W, W=W, invW=1.0/W,
                lin=1.0 if by>=128 else 0.0, spthr=spv*thr, splines=spv*(by/128.0),
                i8=1.0 if eb==1 else 0.0, half=1.0 if eb==2 else 0.0,
                fill=32*eb/256.0, part_by=by*(1.0 if by%128 else 0.0),
                issp=1.0 if 2*thr>REGB.get(W,128) else 0.0, dt=r["dtype"])
POOL=["by","lines","thr","rounds","schedR","W2","invW","lin","part_by","spthr","splines","fill"]
def nnls(X, yw, iters=4000):
    n=X.shape[1]; w=np.zeros(n); r=yw-X@w; coln=(X*X).sum(0); coln[coln==0]=1e-12
    for _ in range(iters):
        ch=False
        for j in range(n):
            if coln[j]<=0: continue
            delta=(X[:,j]@r)/coln[j]
            if delta>1e-12:
                w[j]+=delta; r-=delta*X[:,j]; ch=True
        if not ch: break
    return w
def fitseg(rs, maxk=5, constrained=True):
    Fs=[F(r) for r in rs]
    y=np.array([r["cost_per_operation"] for r in rs],float); wgt=1.0/y
    names=["one"]+POOL
    M=np.array([[1.0]+[f.get(p,0.0) for p in POOL] for f in Fs])
    best=(1e9,None)
    for k in range(1,maxk+1):
        for combo in itertools.combinations(range(len(names)),k):
            X=M[:,combo]
            if constrained:
                co=nnls(X*wgt[:,None], y*wgt)
            else:
                co,*_=np.linalg.lstsq(X*wgt[:,None], y*wgt, rcond=None)
            p=X@co
            if np.any(p<=0): continue
            m=np.mean(np.abs(p-y)/y)*100
            if m<best[0]: best=(m,([names[i] for i in combo],co))
    return best
L=lambda f:f["lines"]; S=lambda f:f["issp"]>0
segs_load=[("lines<=2",lambda f:L(f)<=2),
           ("2<l<=32",lambda f:2<L(f)<=32),
           ("l>32,nospill,W<=16",lambda f:L(f)>32 and not S(f) and f["W"]<=16),
           ("l>32,nospill,W>16",lambda f:L(f)>32 and not S(f) and f["W"]>16),
           ("l>32,spill",lambda f:L(f)>32 and S(f))]
segs_store=[("ept<1",lambda f:f["ept"]<1),
            ("ept>=1,int8,l<=8",lambda f:f["ept"]>=1 and f["i8"]>0 and f["lines"]<=8),
            ("ept>=1,int8,8<l<=32",lambda f:f["ept"]>=1 and f["i8"]>0 and 8<f["lines"]<=32),
            ("ept>=1,fp16,l<=8",lambda f:f["ept"]>=1 and f["half"]>0 and f["lines"]<=8),
            ("ept>=1,fp16,8<l<=32",lambda f:f["ept"]>=1 and f["half"]>0 and 8<f["lines"]<=32),
            ("ept>=1,fp16,l>32",lambda f:f["ept"]>=1 and f["half"]>0 and f["lines"]>32),
            ("ept>=1,eb>=4,W<=4",lambda f:f["ept"]>=1 and f["i8"]==0 and f["half"]==0 and f["W"]<=4),
            ("ept>=1,eb>=4,W>4",lambda f:f["ept"]>=1 and f["i8"]==0 and f["half"]==0 and f["W"]>4)]
out={"source":"raw/raw_formb_v2.jsonl","n_points":len(rows)}
for tag,constrained in (("nonneg",True),("free",False)):
 out[tag]={"segments":{}}
 for direction,segs in (("load",segs_load),("store",segs_store)):
    allp=[];ally=[];ncoef=0
    print("### %s  %s" % (tag,direction))
    for name,cond in segs:
        sub=[r for r in rows if r["direction"]==direction and cond(F(r))]
        if len(sub)<5: continue
        m,(feats,co)=fitseg(sub, constrained=constrained)
        Fs=[F(r) for r in sub]
        X=np.array([[1.0 if p=="one" else f.get(p,0.0) for p in feats] for f in Fs])
        allp.extend(X@co); ally.extend([r["cost_per_operation"] for r in sub]); ncoef+=len(feats)
        out[tag]["segments"]["%s|%s"%(direction,name)]={"feats":feats,"coefs":[float(x) for x in co],"n":len(sub),"mape":float(m)}
        print("  %-26s n=%-3d %5.2f%%  %s" % (name,len(sub),m,feats))
    allp=np.array(allp);ally=np.array(ally)
    whole=float(np.mean(np.abs(allp-ally)/ally)*100)
    out[tag]["segments"]["%s|TOTAL"%direction]={"n":len(ally),"coefs_count":ncoef,"mape":whole}
    print("  >>> %d 点, %d 系数, MAPE=%.2f%%" % (len(ally),ncoef,whole))
json.dump(out, open(BASE+r"\fitted\FORMB_V2_PHYSICAL.json","w"), indent=1, ensure_ascii=False)
print("saved")
