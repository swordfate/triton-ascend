# -*- coding: utf-8 -*-
"""判定「mask 因子」方案：同段联合拟合 + is_masked（±交互项） vs 独立 masked 槽位。

目标：若联合拟合在 V0/V1 两侧都能 <10%，则用「一份公式 + mask 因子」；
否则用独立 masked 槽位。

用法:
  python refit_mask_joint.py            # store + load（用本地 LOADV1_v1.jsonl）
"""
import sys, os, json, io, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.path.insert(0, r'D:\work_wj\mask_vs_nomask')
sys.stdout.reconfigure(encoding='utf-8')

import refit_simt as R
from refit_simt import enrich, fit_segment, segments, predict_seg, design, mape
from feats_v2 import load_all_v2, build as fv2_build
from cmn import feats as cmn_feats

DATA = r'D:\work_wj\mask_vs_nomask'


def prep(rows):
    for r in rows:
        if '_f' not in r:
            r['_f'] = cmn_feats(r)
            fv2_build(r)
        enrich(r)
    return rows


def load_raw(path, want_var=None):
    out = []
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if want_var is not None and int(r.get('var', -1)) != want_var:
                continue
            if r.get('error') or not r.get('valid'):
                continue
            if (r.get('spread') or 1) > 0.2:
                continue
            out.append(r)
    return out


def add_mask_features(r, masked):
    f = r['_f']
    f['is_masked'] = 1.0 if masked else 0.0
    # 交互项（允许 mask 改变斜率）
    for k in ('nbytes', 'n', 'iters', 'rows_x_duty', 'bytes_per_lane', 'L128',
              'duty', 'total_lines', 'elem_per_line', 'mx128'):
        f['m_x_' + k] = f['is_masked'] * f.get(k, 0.0)
    r['_masked'] = masked


def subset_mape(part, model):
    errs = []
    for r in part:
        pred = max(predict_seg(r, model), 1e-9)
        errs.append(abs(pred - r['target_ns']) / r['target_ns'] * 100)
    return float(np.mean(errs)) if errs else float('nan')


def run_slot(name, v0, v1, interactions, cap=16):
    segs0 = dict(segments(v0, 'simdK' if 'simd' in name else 'rows3_W2'))
    segs1 = dict(segments(v1, 'simdK' if 'simd' in name else 'rows3_W2'))
    print('=' * 86)
    print('%s  V0=%d V1=%d  方案=%s' % (name, len(v0), len(v1),
                                       '联合+mask因子(含交互)' if interactions else '联合+mask指示项'))
    tot0 = tot1 = n0 = n1 = 0.0
    for seg in sorted(set(segs0) | set(segs1)):
        a = segs0.get(seg, [])
        b = segs1.get(seg, [])
        if not a or not b:
            print('  %-14s 数据缺失，跳过' % seg)
            continue
        pool = R.pool_of(a + b)
        extra = ['is_masked'] + (['m_x_' + k for k in ('nbytes', 'n', 'iters', 'rows_x_duty',
                                                       'bytes_per_lane', 'L128', 'duty',
                                                       'total_lines', 'elem_per_line', 'mx128')]
                                 if interactions else [])
        R.POOL_OVERRIDE = [k for k in pool if k not in extra] + extra
        mj = fit_segment(a + b, cap=cap, seed=8888)
        R.POOL_OVERRIDE = None
        e0 = subset_mape(a, mj); e1 = subset_mape(b, mj)
        picks = [k for k in mj['keys'] if k == 'is_masked' or k.startswith('m_x_')]
        print('  %-14s nV0=%-4d nV1=%-4d | 联合→V0 %5.2f%% 联合→V1 %5.2f%% | 项数 %2d | mask项 %s'
              % (seg, len(a), len(b), e0, e1, len(mj['keys']), picks or '未选中'))
        tot0 += e0 * len(a); n0 += len(a); tot1 += e1 * len(b); n1 += len(b)
    print('  => 联合→V0 %.2f%%  联合→V1 %.2f%%' % (tot0 / max(1, n0), tot1 / max(1, n1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interactions', action='store_true')
    ap.add_argument('--cap', type=int, default=16)
    args = ap.parse_args()

    print('### STORE')
    S0 = prep(load_raw(os.path.join(DATA, 'WIDE2_final.jsonl'), 0))
    S1 = prep(load_raw(os.path.join(DATA, 'WIDE2_final.jsonl'), 1))
    for r in S0: add_mask_features(r, False)
    for r in S1: add_mask_features(r, True)
    for path in ('simd', 'simt_only'):
        v0 = [r for r in S0 if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
        v1 = [r for r in S1 if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
        R.SEG_K = 2 if path == 'simd' else R.SEG_K
        run_slot('store|' + path, v0, v1, args.interactions, cap=args.cap)

    print()
    print('### LOAD')
    _, L0 = load_all_v2()
    L1 = load_raw(os.path.join(DATA, 'LOADV1_v1.jsonl'))
    L0 = prep(L0); L1 = prep(L1)
    for r in L0: add_mask_features(r, False)
    for r in L1: add_mask_features(r, True)
    for path in ('simd', 'simt_only'):
        v0 = [r for r in L0 if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
        v1 = [r for r in L1 if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
        R.SEG_K = 4 if path == 'simd' else R.SEG_K
        run_slot('load|' + path, v0, v1, args.interactions, cap=args.cap)


if __name__ == '__main__':
    main()
