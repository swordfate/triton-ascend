# -*- coding: utf-8 -*-
"""SIMD 槽位微调：保持现有 4 个 SIMD 槽位的特征项不变，用「原 V0 + 加密 gap 补测」
合并数据重新估计系数（加权 LS + Nelder-Mead MAPE 精修），并报告旧点/新点/合并 MAPE。

输出：out/simd_dense_refit.json  -> {slot: {keys, coef, mape_old, mape_new, mape_all, n_*}}

用法: python refit_simd_dense.py
"""
import sys, os, json, io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.path.insert(0, r'D:\work_wj\mask_vs_nomask')
sys.stdout.reconfigure(encoding='utf-8')

from feats_v2 import load_all_v2
from cmn import feats as cmn_feats
from feats_v2 import build as fv2_build
from refit_simt import enrich, design, fit_ls, mape, nm_polish

DATA = r'D:\work_wj\mask_vs_nomask'


def load_jsonl(path, want_var=0):
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
            if int(r.get('var', -1)) != want_var:
                continue
            if r.get('error') or not r.get('valid'):
                continue
            if (r.get('spread') or 1) > 0.2:
                continue
            out.append(r)
    return out


def prep(rows):
    for r in rows:
        if '_f' not in r:
            r['_f'] = cmn_feats(r)
            fv2_build(r)
        enrich(r)
    return rows


def fit_fixed(part, keys, seed=8888):
    X, y = design(part, keys)
    p = fit_ls(X, y)
    rng = np.random.default_rng(seed)
    bp, bm = nm_polish(X, y, p, iters=4000, rng=rng)
    for _ in range(2):
        p2, m2 = nm_polish(X, y, bp, iters=4000, rng=rng, jitter=0.5)
        if m2 < bm:
            bp, bm = p2, m2
    return bp, bm


def seg_of(r, op):
    rows = int(r['rows'])
    if op == 'load':
        return 'rows_le4' if rows <= 4 else 'rows_gt4'
    return 'rows_le2' if rows <= 2 else 'rows_gt2'


def main():
    prof = json.load(io.open(os.path.join(HERE, 'gen', 'profile_strided_wide_completion_v2.json'),
                             encoding='utf-8'))
    out = {}
    # 原有 V0 数据
    S0, L0 = load_all_v2()
    S0 = prep(S0)
    L0 = prep(L0)
    # 加密 gap 补测
    Sd = prep(load_jsonl(os.path.join(DATA, 'DENSE_store.jsonl')))
    Ld = prep(load_jsonl(os.path.join(DATA, 'DENSE_load.jsonl')))

    for op, old, dense in (('store', S0, Sd), ('load', L0, Ld)):
        old = [r for r in old if r['path'] == 'simd' and r['order'] == 'inner' and r['run_bytes'] >= 32]
        dense = [r for r in dense if r['path'] == 'simd' and r['order'] == 'inner' and r['run_bytes'] >= 32]
        print('== %s: old=%d dense=%d' % (op, len(old), len(dense)))
        for slot_name, entry in prof['simd'].items():
            if slot_name in ('description', 'evidence'):
                continue
            keys = entry['terms']
            old_part = [r for r in old if seg_of(r, op).replace('rows_', '') in slot_name]
            dense_part = [r for r in dense if seg_of(r, op).replace('rows_', '') in slot_name]
            # slot 名形如 simd_load_swide_rows_le4 / rows_gt4 / rows_le2 / rows_gt2
            if not old_part and not dense_part:
                continue
            part = old_part + dense_part
            coef, all_mape = fit_fixed(part, keys)
            def mape_of(sub):
                if not sub:
                    return float('nan')
                X, y = design(sub, keys)
                return mape(X, y, coef)
            m_old, m_new = mape_of(old_part), mape_of(dense_part)
            out[slot_name] = {
                'keys': keys,
                'coef': [float(x) for x in coef],
                'mape_old': m_old, 'mape_new': m_new, 'mape_all': all_mape,
                'n_old': len(old_part), 'n_new': len(dense_part),
            }
            print('  %-28s n_old=%-4d n_new=%-4d | old %5.2f%%  new %5.2f%%  all %5.2f%%'
                  % (slot_name, len(old_part), len(dense_part), m_old, m_new, all_mape))

    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    json.dump(out, io.open(os.path.join(HERE, 'out', 'simd_dense_refit.json'), 'w',
                           encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved out/simd_dense_refit.json')


if __name__ == '__main__':
    main()
