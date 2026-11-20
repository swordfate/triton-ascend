#!/usr/bin/env python3
"""Per-position dtype merge feasibility: for each (engine, direction,
bucket, residency) with >=2 dtypes, compute spread of first & marginal.
Mergeable if max relative spread <= 15% for BOTH fields."""
import json, statistics
p=json.load(open('/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json'))
ELEM={'int8':1,'fp16':2,'fp32':4,'int64':8}
total_cells=0; merged_save=0; mergeable=[]; report=[]
for mode in ('simd','simt'):
    ga=p[mode]['memory']['contiguous_ldst_v2']['group_a']
    pos={}
    for k,v in ga.items():
        parts=k.split('/')
        if parts[-1].startswith('w'): continue
        pos.setdefault((parts[1],int(parts[3]),parts[4]),[]).append((parts[2],v['first'],v['marginal']))
    for (d,B,res),lst in sorted(pos.items()):
        if len(lst)<2: continue
        total_cells+=len(lst)
        fvals=[x[1] for x in lst]; mvals=[x[2] for x in lst]
        fmed=statistics.median(fvals); mmed=statistics.median(mvals)
        fspread=max(abs(f-fmed)/max(fmed,1) for f in fvals)*100
        mspread=max(abs(m-mmed)/max(mmed,1) for m in mvals)*100
        ok = fspread<=15 and mspread<=15
        dtypes='/'.join(x[0] for x in sorted(lst))
        report.append((mode,d,B,res,dtypes,round(fspread,1),round(mspread,1),ok))
        if ok:
            mergeable.append((mode,d,B,res,len(lst)))
            merged_save += (len(lst)-1)*2   # merged cell stores 2 numbers
print('%-5s %-6s %6s %-5s %-16s %7s %7s %s' % ('engine','dir','bytes','res','dtypes','Fspread%','Mspread%','mergeable'))
for r in report:
    print('%-5s %-6s %6d %-5s %-16s %7.1f %7.1f %s' % r)
print()
print('cells in multi-dtype positions: %d; mergeable positions: %d; numbers saved: %d' % (
    total_cells, len(mergeable), merged_save))
