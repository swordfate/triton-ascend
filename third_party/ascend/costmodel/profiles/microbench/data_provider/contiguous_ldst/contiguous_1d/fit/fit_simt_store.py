"""重拟合 SIMT 各段，写 fitted/SIMT_SEGMENTS.json。"""
import json, sys
sys.path.insert(0, r"D:\work_wj\contiguous-final\third_party\ascend\costmodel\profiles\microbench\data_provider\contiguous_ldst\contiguous_1d\fit")
ROOT = r"D:\work_wj\contiguous-final\third_party\ascend\costmodel\profiles\microbench\data_provider\contiguous_ldst\contiguous_1d"
from fit_physical import fit, greedy, add_extras, fit_pos, POOL, feats as pf
from assemble_spec import segment, feats as af
raw = [json.loads(l) for l in open(ROOT + r'\raw\raw_pure_unified.jsonl') if l.strip()]
groups = {}
for r in raw:
    if r["cost"] <= 0:
        continue
    if r['mode'] != 'simt_only': continue
    gs = af(r['el'], r['eb'], r['W'], False)
    if gs['te'] > 4.0:      # 只在认领域 te<=4 内拟合
        continue
    k = segment('simt_only', r['dir'], gs)
    row = dict(c=r['cost'], mode='simt_only', d=r['dir'], **pf(r['el'], r['eb'], r['W']))
    groups.setdefault(k, []).append(row)
out = {}
for k, rs in sorted(groups.items()):
    if len(rs) < 8:
        print('%-34s n=%3d skip' % (k, len(rs))); continue
    if 'wide,W<=4' in k:
        # 只含 rounds 的纯 te 形式表达不了字节增长，强制带 by（最大误差 34%->25%）
        ch = ['rounds', 'byh1k']
        co, m = fit(rs, ch)
    else:
        ch, co, m = greedy(rs, 4, POOL)
        ch, co, m = add_extras(rs, ch, target=8.0, cap=2)
    print('%-34s n=%3d  MAPE %5.2f%%  %s' % (k, len(rs), m, ''.join(' %+.4g*%s'%(c,f) for f,c in zip(ch, co[1:]))))
    out[k] = dict(feats=ch, coefs=[round(float(c),6) for c in co], mape=round(m,2), n=len(rs))
json.dump(out, open(ROOT + r'\fitted\SIMT_SEGMENTS.json','w'), indent=1, ensure_ascii=False)
print('写出 %d 段' % len(out))