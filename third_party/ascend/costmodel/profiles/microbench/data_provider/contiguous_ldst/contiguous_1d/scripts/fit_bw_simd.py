#!/usr/bin/env python3
"""Fit the throughput-regime SIMD formulas from BW_SIMD.jsonl.

Both directions share one shape across power-of-two and non-power-of-two
lengths, both access forms and all four element widths.
"""
import json, sys
from pathlib import Path
import numpy as np


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith('{'):
            rows.append(json.loads(line))
    return rows


def maxfit(rows, sig, a0r, a1r, br):
    y = np.array([r['per_acc_ns'] for r in rows])
    u = np.array([r['unroll'] for r in rows], dtype=float)
    s = np.array([sig(r) for r in rows], dtype=float)
    best = None
    for a0 in a0r:
        for a1 in a1r:
            base = a0 + a1 / u
            for b in br:
                pred = np.maximum(base, b * s)
                m = float(np.mean(np.abs((pred - y) / y)) * 100)
                if best is None or m < best[0]:
                    best = (m, a0, a1, b)
    return best


def linfit(rows, feats):
    X = np.array([[f(r) for f in feats] for r in rows])
    y = np.array([r['per_acc_ns'] for r in rows])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    m = float(np.mean(np.abs((X @ c - y) / y)) * 100)
    return m, c


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else 'BW_SIMD.jsonl')
    out = {'shape': {'load': 'max(a0 + a1/unroll, b*KiB)',
                     'store': 'a0 + b*KiB + c/unroll'}, 'segments': {}}
    for dr in ('load', 'store'):
        seg = [r for r in rows if r['direction'] == dr]
        if dr == 'load':
            m, a0, a1, b = maxfit(seg, lambda r: r['bytes'] / 1024.0,
                                  np.arange(0., 40., 1.),
                                  np.arange(0., 400., 5.),
                                  np.arange(1., 40., 0.5))
            out['segments']['simd.load'] = {'n': len(seg), 'mape': round(m, 2),
                                            'a0': float(a0), 'a1': float(a1), 'b': float(b)}
        else:
            m, c = linfit(seg, [lambda r: 1.0, lambda r: r['bytes'] / 1024.0,
                                lambda r: 1.0 / r['unroll']])
            out['segments']['simd.store'] = {'n': len(seg), 'mape': round(m, 2),
                                             'a0': float(c[0]), 'b': float(c[1]), 'c': float(c[2])}
    print(json.dumps(out, indent=1))
    Path('simd_bw_formulas.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
