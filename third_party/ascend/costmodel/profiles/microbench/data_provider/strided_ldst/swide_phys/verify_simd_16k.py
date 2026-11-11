# -*- coding: utf-8 -*-
"""验证 16KB 行距下 rows=16 的 SIMD 实测点能否被现有共享槽位解释（免重拟合）。

数据：{SIMD_STORE,SIMD_LOAD}_rows16_16k{{,_v0}}.jsonl
  （run {32,64,128} × gap {16256,16320,16384} × 3 dtype，masked/unmasked 各 27 点）
用法: python verify_simd_16k.py [--data-dir DIR] [--profile JSON]
"""
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:\work_wj\63-workspace\63-workspace\strided-calib\03-featurize')
sys.stdout.reconfigure(encoding='utf-8')

from cmn import feats as cmn_feats  # noqa: E402
from feats_v2 import build as fv2_build  # noqa: E402
from refit_simt import enrich, design  # noqa: E402


def load_jsonl(path):
    out = []
    for line in io.open(path, encoding='utf-8'):
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
        out.append(r)
    return out


def prep(rows):
    for r in rows:
        if '_f' not in r:
            r['_f'] = cmn_feats(r)
            fv2_build(r)
        enrich(r)
    return rows


def report(name, rows, entry):
    errs = []
    for r in sorted(rows, key=lambda x: (x['run_bytes'], x['gap_bytes'], x['dtype'])):
        X, _ = design([r], entry['terms'])
        pred = float(entry['intercept_system_cycles'] + X[0] @
                     np.array(entry['coefficient_system_cycles']))
        meas = float(r['target_ns'])
        errs.append((pred / meas - 1.0) * 100.0)
    if not errs:
        print('== %s: no valid points' % name)
        return
    a = np.abs(np.array(errs))
    print('== %-42s n=%2d  MAPE=%5.2f%%  max=%5.2f%%  p90=%5.2f%%  within10=%.0f%%'
          % (name, len(errs), a.mean(), a.max(), np.percentile(a, 90),
             100.0 * (a <= 10.0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=r'D:\work_wj\mask_vs_nomask')
    ap.add_argument('--profile', default=os.path.join(
        HERE, '..', '..', '..', '..', 'simd_simt', 'ascend_950pr_simd_simt_v1.json'))
    args = ap.parse_args()

    prof = json.load(io.open(args.profile, encoding='utf-8'))
    slots = prof['simd']['memory']['strided_wide_completion']
    runs = [
        ('masked store', 'SIMD_STORE_rows16_16k.jsonl', 'simd_store_swide_rows_gt2'),
        ('masked load', 'SIMD_LOAD_rows16_16k.jsonl', 'simd_load_swide_rows_gt4'),
        ('unmasked store', 'SIMD_STORE_rows16_16k_v0.jsonl', 'simd_store_swide_rows_gt2'),
        ('unmasked load', 'SIMD_LOAD_rows16_16k_v0.jsonl', 'simd_load_swide_rows_gt4'),
    ]
    for title, name, slot in runs:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            print('== %s: missing %s' % (title, path))
            continue
        report('%s / %s' % (title, slot), prep(load_jsonl(path)), slots[slot])


if __name__ == '__main__':
    main()
