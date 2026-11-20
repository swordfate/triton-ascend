#!/usr/bin/env python3
"""Group_a fit v2: physically-motivated, dtype-stratified slopes.
  first(B)    = L + B*fc[dtype]
  marginal(B) = m0 + max(0, B-b0)*mc[dtype]      (b0=512)
  cT          = kc * marginal
  Q(B)        = clamp(Kq/B, 4, 32) warm; clamp(Kqc/B, 4, 16) cold
"""
import json, statistics, sys
P='/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json'
prof=json.load(open(P))
ELEM={'int8':1,'fp16':2,'fp32':4,'int64':8}
def cells_of(mode):
    ga=prof[mode]['memory']['contiguous_ldst_v2']['group_a']
    out={}
    for k,v in ga.items():
        parts=k.split('/')
        base=tuple(parts[:5]); warp=parts[5] if len(parts)>5 else 'base'
        out.setdefault((base[1],base[4],warp),[]).append(
            (int(base[3]), ELEM[base[2]], v['first'], v['marginal'], v['c_transfer'], v['Q']))
    return out
def med(xs): return statistics.median(xs) if xs else 0.0
def fit_group(cells, res):
    small=[c for c in cells if c[0]<=512]
    L=med([c[2] for c in small]) or med([c[2] for c in cells])
    m0=med([c[3] for c in small]) or min(c[3] for c in cells)
    fc={}; mc={}
    for eb in sorted(set(c[1] for c in cells)):
        big=[c for c in cells if c[1]==eb and c[0]>512]
        if big:
            fc[eb]=med([(c[2]-L)/c[0] for c in big])
            mc[eb]=med([(c[3]-m0)/(c[0]-512) for c in big])
    kc=med([c[4]/c[3] for c in cells if c[3]>0 and c[4]>0])
    return {'L':round(L,1),'m0':round(m0,1),'b0':512,'kc':round(kc,3),
            'Kq':65536 if res=='warm' else 32768,
            'fc':{str(k):round(v,5) for k,v in fc.items()},
            'mc':{str(k):round(v,5) for k,v in mc.items()}}
def ev(f,B,eb,res):
    first=f['L']+B*f['fc'].get(str(eb),0.0)
    m=f['m0']+max(0,B-512)*f['mc'].get(str(eb),0.0)
    c=m*f['kc']
    q=max(4,min(32 if res=='warm' else 16, round(f['Kq']/B)))
    return first,m,c,q
errs={'F':[],'M':[],'cT':[]}; worst=[]
fits={}
for mode in ('simd','simt'):
    fits[mode]={}
    for (d,r,w),cells in sorted(cells_of(mode).items()):
        if r!='warm' and r!='cold': continue
        f=fit_group(cells,r); fits[mode][f'{d}/{r}/{w}']=f
        for (B,eb,F,M,cT,Q) in sorted(cells):
            Fe,Me,Ce,Qe=ev(f,B,eb,r)
            eF=abs(Fe-F)/max(F,1)*100; eM=abs(Me-M)/max(M,1)*100
            eC=abs(Ce-cT)/max(cT,1)*100 if cT>0 else 0
            errs['F'].append(eF); errs['M'].append(eM); errs['cT'].append(eC)
            if eF>30 or eM>30: worst.append((mode,d,r,w,B,eb,round(eF),round(eM)))
        print('%s %s/%s/%s: %s' % (mode,d,r,w,f))
print('== median abs err: F %.1f%% M %.1f%% cT %.1f%%' % (statistics.median(errs['F']),statistics.median(errs['M']),statistics.median(errs['cT'])))
print('== max abs err:    F %.1f%% M %.1f%% cT %.1f%%' % (max(errs['F']),max(errs['M']),max(errs['cT'])))
print('== cells >30%% err: %d' % len(worst))
for w in worst: print('   ',w)
json.dump(fits, open('/root/work/explore_contig/group_a_fit2.json','w'), indent=1)
print('saved group_a_fit2.json; param counts per group: 2 shared + 2/dtype present + kc + Kq')
