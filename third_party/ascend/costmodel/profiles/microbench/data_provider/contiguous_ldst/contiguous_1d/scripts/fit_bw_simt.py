#!/usr/bin/env python3
"""Fit the per-warp SIMT throughput formulas from BW_SIMT.jsonl.

SIMT is segmented by warp count because the measured surface depends on it
non-monotonically; inside a segment one shape is used:
    perAccess = max(a0 + a1/unroll, b * KiB)

Only te < 8 points are fitted: past eight elements per thread the transfer
steps with register spill rather than curving, which the latency model
already covers with its tables.
"""
import json, sys, math
from pathlib import Path
import numpy as np


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith('{'):
            rows.append(json.loads(line))
    return rows


def fitmax(rows):
    y = np.array([r['per_acc_ns'] for r in rows])
    u = np.array([r['unroll'] for r in rows], dtype=float)
    kb = np.array([r['bytes'] / 1024.0 for r in rows], dtype=float)
    best = None
    for a0 in np.arange(-60., 120., 2.):
        for a1 in np.arange(0., 400., 10.):
            base = a0 + a1 / u
            for b in np.arange(0.5, 120., 1.):
                pred = np.maximum(base, b * kb)
                m = float(np.mean(np.abs((pred - y) / y)) * 100)
                if best is None or m < best[0]:
                    best = (m, a0, a1, b)
    return best


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else 'BW_SIMT.jsonl')
    kept = [r for r in rows if r.get('r2', 1.0) >= 0.999]
    segs = []
    for dr in ('load', 'store'):
        for w in (1, 2, 4, 8, 16, 32, 64):
            seg = [r for r in kept if r['direction'] == dr and r['warps'] == w
                   and r['nreal'] / (32.0 * w) < 8.0]
            if len(seg) < 4:
                continue
            m, a0, a1, b = fitmax(seg)
            segs.append({'segment': 'simt.%s.W%d' % (dr, w), 'n': len(seg),
                         'mape': round(m, 2), 'a0': float(a0), 'a1': float(a1),
                         'b': float(b)})
    out = {'shape': 'max(a0 + a1/unroll, b*KiB)', 'scope': 'te < 8', 'segments': segs}
    print(json.dumps(out, indent=1))
    Path('simt_bw_formulas.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
