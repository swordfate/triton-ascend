# -*- coding: utf-8 -*-
"""拟合全部 masked 槽位（SIMT store/load：rows=2/8/64×W + rows=16）。

数据：
  store: WIDE2_final.jsonl var=1（rows 2/8/64×W 主网格）+ STORE_simt_rows16_v1.jsonl
  load : LOADV1_v1.jsonl（rows 2/8/64×W）+ LOAD_simt_rows16_v1.jsonl
输出：out/masked_slots.json  （每个槽位 keys/coef/intercept/mape/n）

用法: python fit_masked_slots.py [--cap 16]
"""
import sys, os, json, io, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.path.insert(0, r'D:\work_wj\mask_vs_nomask')
sys.stdout.reconfigure(encoding='utf-8')

DATA = os.environ.get('SWIDE_DATA', r'D:\work_wj\mask_vs_nomask')
sys.path.insert(0, DATA)
import refit_simt as R
from refit_simt import enrich, fit_segment, predict_seg
from feats_v2 import build as fv2_build
from cmn import feats as cmn_feats

V0_KEYS = None  # 统一用全池（与主标定一致）


def prep(rows):
    for r in rows:
        if '_f' not in r:
            r['_f'] = cmn_feats(r)
            fv2_build(r)
        enrich(r)
    return rows


def load_jsonl(path, want_var=None):
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


def seg_name(r, is_store):
    rows = int(r['rows']); W = r['_f']['W']
    if rows == 16:
        # rows=16 在 W 上有明显分段（W<=2/3-8/>8），细分后段加权 <10%
        if W <= 2: return 'rows16_wle2'
        if W <= 8: return 'rows16_w3_8'
        return 'rows16_wgt8'
    if rows == 2:
        return 'rows2'
    if rows == 8:
        return 'rows8'
    if rows == 64:
        if W <= 2: return 'rows64_wle2'
        if W <= 8: return 'rows64_w3_8'
        return 'rows64_wgt8'
    return None


def fit_slot(part, cap):
    m = fit_segment(part, cap=cap, seed=8888)
    errs = []
    for r in part:
        pred = max(predict_seg(r, m), 1e-9)
        errs.append(abs(pred - r['target_ns']) / r['target_ns'] * 100)
    m['n'] = len(part)
    m['mape'] = float(np.mean(errs))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap-store', type=int, default=8)
    ap.add_argument('--cap-load', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'masked_slots.json'))
    args = ap.parse_args()
    out = {}

    print('### STORE masked (SIMT)')
    st = prep(load_jsonl(os.path.join(DATA, 'WIDE2_final.jsonl'), 1))
    st += prep(load_jsonl(os.path.join(DATA, 'STORE_simt_rows16_v1.jsonl')))
    # 16KB 行距扩展（solve_tril h64：pitch 16384B → gap 16320B）
    st += prep(load_jsonl(os.path.join(DATA, 'STORE_rows16_16k.jsonl')))
    st = [r for r in st if r['path'] == 'simt_only' and r['order'] == 'inner' and r['run_bytes'] >= 32]
    groups = {}
    for r in st:
        g = seg_name(r, True)
        if g:
            groups.setdefault(g, []).append(r)
    for g in ('rows2', 'rows8', 'rows64_wle2', 'rows64_w3_8', 'rows64_wgt8',
              'rows16_wle2', 'rows16_w3_8', 'rows16_wgt8'):
        part = groups.get(g, [])
        if not part:
            print('  %-12s 无数据' % g); continue
        m = fit_slot(part, args.cap_store)
        out['simt_store_swide_masked_' + g] = m
        print('  %-12s n=%-4d %6.2f%%  %2d项' % (g, m['n'], m['mape'], len(m['keys'])))

    print('### LOAD masked (SIMT)')
    ld = prep(load_jsonl(os.path.join(DATA, 'LOADV1_v1.jsonl')))
    ld += prep(load_jsonl(os.path.join(DATA, 'LOAD_simt_rows16_v1.jsonl')))
    ld += prep(load_jsonl(os.path.join(DATA, 'LOAD_rows16_16k.jsonl')))
    ld = [r for r in ld if r['path'] == 'simt_only' and r['order'] == 'inner' and r['run_bytes'] >= 32]
    groups = {}
    for r in ld:
        g = seg_name(r, False)
        if g:
            groups.setdefault(g, []).append(r)
    for g in ('rows2', 'rows8', 'rows64_wle2', 'rows64_w3_8', 'rows64_wgt8',
              'rows16_wle2', 'rows16_w3_8', 'rows16_wgt8'):
        part = groups.get(g, [])
        if not part:
            print('  %-12s 无数据' % g); continue
        m = fit_slot(part, args.cap_load)
        out['simt_load_swide_masked_' + g] = m
        print('  %-12s n=%-4d %6.2f%%  %2d项' % (g, m['n'], m['mape'], len(m['keys'])))

    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    json.dump(out, io.open(args.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved', args.out)


if __name__ == '__main__':
    main()
