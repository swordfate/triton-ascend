# -*- coding: utf-8 -*-
"""按分组 cap 重拟合未 mask 的 4 个槽位，输出 phys_cap_mixed.json。

分组 cap：SIMD load 6 / SIMD store 18 / SIMT unmask load 14 / store 8；
masked 槽位由 fit_masked_slots.py --cap-load 10 --cap-store 8 生成。
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SWIDE_DATA', r'D:\work_wj\mask_vs_nomask')
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
sys.stdout.reconfigure(encoding='utf-8')

from feats_v2 import load_all_v2  # noqa: E402
import refit_simt as R  # noqa: E402

CAPS = {('load', 'simd'): 6, ('store', 'simd'): 18,
        ('load', 'simt_only'): 14, ('store', 'simt_only'): 8}


def main():
    S0, L0 = load_all_v2()
    out = {}
    for slot in ('load|simt_only', 'store|simt_only', 'load|simd', 'store|simd'):
        op, path = slot.split('|')
        cap = CAPS[(op, path)]
        R.POOL_OVERRIDE = R.SIMD_PHYS if path == 'simd' else R.SIMT_PHYS
        dat = L0 if op == 'load' else S0
        ss = R.slot_points(dat, path)
        if path == 'simd':
            R.SEG_K = 4 if op == 'load' else 2
            scheme = 'simdK'
        else:
            scheme = 'rows3_W2'
        rec = {}
        tot = 0.0
        n = 0
        for name, part in R.segments(ss, scheme):
            m = R.fit_segment(part, cap=cap, seed=8888)
            m['seg'] = name
            rec[name] = m
            tot += m['mape'] * len(part)
            n += len(part)
            print('%-16s %-14s n=%-4d %6.2f%% %2d项' % (
                slot, name, len(part), m['mape'], len(m['keys'])))
        out[slot] = {'blend': tot / n, 'n': n, 'segments': rec}
        print('  => %-16s cap=%-2d blend %.2f%% n=%d' % (slot, cap, out[slot]['blend'], n))
    path = os.path.join(HERE, 'phys_cap_mixed.json')
    json.dump(out, io.open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved', path)


if __name__ == '__main__':
    main()
