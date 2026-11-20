#!/usr/bin/env python3
"""Fit compact parametric formulas over the group_a cells.
Model (per engine x direction x residency, simt also per warp-class):
  Q(B)     = clamp(65536/B, 4, 32)  warm;  clamp(32768/B, 4, 16)  cold
  marginal = m0 + max(0, B - b0) / bw(dtype)          b0=512 fixed
  cTrans   = marginal * kc
  first    = f0 + max(0, B - 4096) / fw(dtype)
Median-based robust estimation; report per-cell relative errors.
"""
import json, math, statistics, sys

P='/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json'
prof=json.load(open(P))
ELEM={'int8':1,'fp16':2,'fp32':4,'int64':8}

def cells_of(mode):
    ga=prof[mode]['memory']['contiguous_ldst_v2']['group_a']
    out={}
    for k,v in ga.items():
        parts=k.split('/')  # engine/dir/dtype/bucket/res[/wN]
        base=tuple(parts[:5])
        warp=parts[5] if len(parts)>5 else 'base'
        out.setdefault((base[1],base[4],warp),[]).append(
            (int(base[3]), ELEM[base[2]], v['first'], v['marginal'], v['c_transfer'], v['Q']))
    return out

def fit_group(cells):
    # cells: list of (B, eb, F, M, cT, Q)
    small=[c for c in cells if c[0]<=512]
    m0=statistics.median([c[3] for c in small]) if small else min(c[3] for c in cells)
    b0=512.0
    bw={}
    for eb in sorted(set(c[1] for c in cells)):
        pts=[c for c in cells if c[1]==eb and c[0]>b0 and c[3]>m0]
        if pts:
            bw[eb]=statistics.median([(c[0]-b0)/(c[3]-m0) for c in pts])
    kc=statistics.median([c[4]/c[3] for c in cells if c[3]>0 and c[4]>0]) if any(c[4]>0 for c in cells) else 0.0
    f0=statistics.median([c[2] for c in cells if c[0]<=4096])
    fw={}
    for eb in sorted(set(c[1] for c in cells)):
        pts=[c for c in cells if c[1]==eb and c[0]>4096 and c[2]>f0]
        if pts:
            fw[eb]=statistics.median([(c[0]-4096)/(c[2]-f0) for c in pts])
    return {'m0':round(m0,1),'b0':b0,'kc':round(kc,3),'f0':round(f0,1),
            'bw':{str(k):round(v,1) for k,v in bw.items()},
            'fw':{str(k):round(v,1) for k,v in fw.items()}}

def eval_group(f,B,eb):
    m=f['m0']+max(0.0,B-f['b0'])/f['bw'].get(str(eb), f['bw'][max(f['bw'],key=lambda k:int(k))] if f['bw'] else 1.0)
    c=m*f['kc']
    fwv=f['fw'].get(str(eb))
    first=f['f0']+ (max(0,B-4096)/fwv if fwv else 0.0)
    return first,m,c

def q_of(B,res):
    num=65536 if res=='warm' else 32768
    lo,hi=(4,32) if res=='warm' else (4,16)
    return max(lo,min(hi,round(num/B)))

allerr={'F':[],'M':[],'cT':[]}
fits={}
for mode in ('simd','simt'):
    fits[mode]={}
    for (direction,res,warp),cells in sorted(cells_of(mode).items()):
        f=fit_group(cells)
        fits[mode][f'{direction}/{res}/{warp}']=f
        print(f'== {mode} {direction}/{res}/{warp}: {f}')
        for (B,eb,F,M,cT,Q) in sorted(cells):
            Fe,Me,Ce=eval_group(f,B,eb)
            eF=(Fe-F)/F*100; eM=(Me-M)/M*100 if M>0 else 0
            eC=(Ce-cT)/cT*100 if cT>0 else 0
            allerr['F'].append(abs(eF)); allerr['M'].append(abs(eM)); allerr['cT'].append(abs(eC))
            flag=' *' if abs(eF)>30 or abs(eM)>30 else ''
            print(f'   B{B:>6} eb{eb}: F {F:7.1f}->{Fe:7.1f} ({eF:+6.1f}%)  M {M:7.1f}->{Me:7.1f} ({eM:+6.1f}%)  cT {cT:7.1f}->{Ce:7.1f} ({eC:+6.1f}%){flag}')
        sys.stdout.flush()
print('== summary: median abs err  F %.1f%%  M %.1f%%  cT %.1f%%' % (
    statistics.median(allerr['F']), statistics.median(allerr['M']), statistics.median(allerr['cT'])))
print('   max abs err    F %.1f%%  M %.1f%%  cT %.1f%%' % (
    max(allerr['F']), max(allerr['M']), max(allerr['cT'])))
json.dump(fits, open('/root/work/explore_contig/group_a_fit.json','w'), indent=1)
print('saved group_a_fit.json')
