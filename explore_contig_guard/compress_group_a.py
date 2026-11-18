#!/usr/bin/env python3
"""Compress group_a: drop Q/c_transfer from cells (structural formulas in
code), drop /w cells that duplicate their base cell, add group_a_ct_kc.
Q formulas (consumed by C++): store -> 8; load warm -> clamp(65536/B,4,32);
load cold -> clamp(32768/B,4,8).
"""
import json, statistics
from collections import defaultdict

def process(path, write):
    p=json.load(open(path))
    stats={}
    for mode in ('simd','simt'):
        ga=p[mode]['memory']['contiguous_ldst_v2']['group_a']
        # 1) drop /w cells identical to base (last seg w16 etc.)
        dropped_dup=0
        for k in list(ga):
            seg=k.split('/')[-1]
            if seg.startswith('w') and seg[1:].isdigit():
                base=k.rsplit('/',1)[0]
                if base in ga and ga[k]==ga[base]:
                    del ga[k]; dropped_dup+=1
        # 2) compute kc per (direction/residency) from remaining cells
        ratios=defaultdict(list)
        for k,v in ga.items():
            parts=k.split('/'); d=parts[1]; res=parts[4]
            if v['Q']>0 and v['c_transfer']>0 and v['marginal']>0:
                ratios[(d,res)].append(v['c_transfer']/v['marginal'])
        kc={}
        for d in ('load','store'):
            for r in ('warm','cold'):
                key=(d,r)
                kc['%s/%s'%key]=round(statistics.median(ratios[key]),3) if ratios[key] else 1.2
        # 3) strip Q / c_transfer from cells
        for k,v in ga.items():
            v.pop('Q',None); v.pop('c_transfer',None)
        p[mode]['memory']['contiguous_ldst_v2']['group_a_ct_kc']=kc
        stats[mode]=(dropped_dup,len(ga),kc)
    if write:
        json.dump(p,open(path,'w'),indent=1)
    return stats

for path,write in (('/root/work/triton-ascend/third_party/ascend/costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json',True),
                   ('/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json',True)):
    s=process(path,write)
    print(path.split('site-packages')[-1] if 'site-packages' in path else 'repo')
    for mode,(dd,n,kc) in s.items():
        print('  %s: dropped %d dup /w cells, %d cells remain, kc=%s'%(mode,dd,n,kc))
