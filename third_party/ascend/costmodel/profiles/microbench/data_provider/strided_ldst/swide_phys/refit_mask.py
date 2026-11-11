# -*- coding: utf-8 -*-
"""Masked 对照拟合：判定「动态全 true mask 能否与无 mask 共享同一公式」。

数据：D:\\work_wj\\mask_vs_nomask\\WIDE2_final.jsonl（store，var=0/1 都有）
      + 待补：LOADV1_v1.jsonl（load masked，上板实验中）
口径：与主标定一致（valid、spread<=0.2、inner、run>=32），特征口径复用
      refit_simt.enrich + 本文件的分段定义。
输出：每个槽位在 V0 / V1 / 联合 三种拟合方式下的 MAPE 对比，供确定
      「共享公式」还是「独立 masked 槽位」。

用法:
  python refit_mask.py store            # store 对比（现有数据）
  python refit_mask.py load  <v1.jsonl> # load 对比（V0 用 LOADV1.jsonl, V1 用上板产物）
"""
import sys, os, json, io, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.path.insert(0, r'D:\work_wj\mask_vs_nomask')
sys.stdout.reconfigure(encoding='utf-8')

from feats_v2 import load_all_v2, build as fv2_build  # noqa: E402
from cmn import feats as cmn_feats  # noqa: E402
from refit_simt import enrich, fit_segment, segments, predict_seg  # noqa: E402

DATA = r'D:\work_wj\mask_vs_nomask'


def load_v1_store():
    out = []
    with io.open(os.path.join(DATA, 'WIDE2_final.jsonl'), encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if int(r.get('var', -1)) != 1:
                continue
            if r.get('error') or not r.get('valid'):
                continue
            if (r.get('spread') or 1) > 0.2:
                continue
            out.append(r)
    return out


def load_v0_store():
    S, _ = load_all_v2()
    return S


def seg_for(slot, r):
    """槽位 -> 段名（与主标定一致）。"""
    rows = r['rows']; W = r['_f']['W']
    if slot == 'store|simd':
        return 'rows<=2' if rows <= 2 else 'rows>2'
    if rows == 2: return 'rows=2'
    if rows == 8: return 'rows=8'
    if W <= 2: return 'rows=64,W<=2'
    if W <= 8: return 'rows=64,3-8'
    return 'rows=64,W>8'


def prep(rows):
    for r in rows:
        if '_f' not in r:
            r['_f'] = cmn_feats(r)
            fv2_build(r)
        enrich(r)
    return rows


def mape_of(seg, model):
    errs = []
    for r in seg:
        pred = max(predict_seg(r, model), 1e-9)
        errs.append(abs(pred - r['target_ns']) / r['target_ns'] * 100)
    return float(np.mean(errs)) if errs else float('nan')


def compare(slot_name, v0, v1, cap=16):
    print('=' * 78)
    print('%s  V0 n=%d  V1 n=%d' % (slot_name, len(v0), len(v1)))
    segs_v0 = dict(segments(v0, 'simdK' if 'simd' in slot_name else 'rows3_W2'))
    segs_v1 = dict(segments(v1, 'simdK' if 'simd' in slot_name else 'rows3_W2'))
    all_segs = sorted(set(segs_v0) | set(segs_v1))
    tot = {'v0_only': [0.0, 0], 'v1_only': [0.0, 0], 'joint_v0': [0.0, 0], 'joint_v1': [0.0, 0]}
    for seg_name in all_segs:
        a = segs_v0.get(seg_name, [])
        b = segs_v1.get(seg_name, [])
        if not a and not b:
            continue
        m0 = fit_segment(a, cap=cap, seed=8888) if a else None
        m1 = fit_segment(b, cap=cap, seed=8888) if b else None
        mj = fit_segment(a + b, cap=cap, seed=8888)
        print('  %-14s nV0=%-4d nV1=%-4d | V0独立 %5.2f%% | V1独立 %5.2f%% | '
              '联合→V0 %5.2f%% 联合→V1 %5.2f%% | 联合项数 %d' % (
                  seg_name, len(a), len(b),
                  mape_of(a, m0) if m0 else float('nan'),
                  mape_of(b, m1) if m1 else float('nan'),
                  mape_of(a, mj) if a else float('nan'),
                  mape_of(b, mj) if b else float('nan'),
                  len(mj['keys'])))
        if a and m0:
            tot['v0_only'][0] += mape_of(a, m0) * len(a); tot['v0_only'][1] += len(a)
        if b and m1:
            tot['v1_only'][0] += mape_of(b, m1) * len(b); tot['v1_only'][1] += len(b)
        if a:
            tot['joint_v0'][0] += mape_of(a, mj) * len(a); tot['joint_v0'][1] += len(a)
        if b:
            tot['joint_v1'][0] += mape_of(b, mj) * len(b); tot['joint_v1'][1] += len(b)
    print('  加权: V0独立 %.2f%% | V1独立 %.2f%% | 联合→V0 %.2f%% | 联合→V1 %.2f%%' % (
        tot['v0_only'][0] / max(1, tot['v0_only'][1]),
        tot['v1_only'][0] / max(1, tot['v1_only'][1]),
        tot['joint_v0'][0] / max(1, tot['joint_v0'][1]),
        tot['joint_v1'][0] / max(1, tot['joint_v1'][1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('op', choices=['store', 'load', 'fit-store-masked'])
    ap.add_argument('v1_jsonl', nargs='?', default='')
    ap.add_argument('--cap', type=int, default=16)
    args = ap.parse_args()
    if args.op == 'fit-store-masked':
        out = {}
        for path in ('simd', 'simt_only'):
            v1 = [r for r in prep(load_v1_store())
                  if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
            scheme = 'simdK' if path == 'simd' else 'rows3_W2'
            import refit_simt as R
            if path == 'simd':
                R.SEG_K = 2
            tot = 0.0; n = 0; rec = {}
            for name, part in segments(v1, scheme):
                m = fit_segment(part, cap=args.cap, seed=8888)
                m['seg'] = name; rec[name] = m
                tot += m['mape'] * len(part); n += len(part)
                print('  %-14s n=%-4d %6.2f%%  %2d项' % (name, len(part), m['mape'], len(m['keys'])))
            blend = tot / n
            print('  => %s masked 加权 %.2f%%' % (path, blend))
            out[path] = {'blend': blend, 'n': n, 'segments': rec}
        os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
        json.dump(out, io.open(os.path.join(HERE, 'out', 'masked_store.json'), 'w',
                               encoding='utf-8'), indent=1, ensure_ascii=False)
        print('saved out/masked_store.json')
        return
    if args.op == 'store':
        v0_all = prep(load_v0_store())
        v1_all = prep(load_v1_store())
        for path in ('simd', 'simt_only'):
            v0 = [r for r in v0_all if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
            v1 = [r for r in v1_all if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
            compare('store|' + path, v0, v1, cap=args.cap)
    else:
        from feats_v2 import load_all_v2
        _, L0 = load_all_v2()
        v0_all = prep(L0)
        if not args.v1_jsonl:
            print('need V1 jsonl path'); return
        v1_raw = []
        with io.open(args.v1_jsonl, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get('error') or not r.get('valid'):
                    continue
                if (r.get('spread') or 1) > 0.2:
                    continue
                v1_raw.append(r)
        v1_all = prep(v1_raw)
        for path in ('simd', 'simt_only'):
            v0 = [r for r in v0_all if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
            v1 = [r for r in v1_all if r['path'] == path and r['order'] == 'inner' and r['run_bytes'] >= 32]
            compare('load|' + path, v0, v1, cap=args.cap)


if __name__ == '__main__':
    main()
