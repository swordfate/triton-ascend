#!/usr/bin/env python3
"""fit_fixed_share.py - solve direct_fixed_share from guard B rows.

pred(s) = pred(0.3) + (s - 0.3) * A(geom)   where A from the direct-table
fit T(K)=A+K*m of the row geometry.  Solve per-row s*, report the median.
"""
import json
import re
import statistics
import sys
from pathlib import Path

PROF = json.load(open("/usr/local/python3.11.15/lib/python3.11/site-packages/"
                      "triton/_C/ascend/costmodel_profiles/simd_simt/"
                      "ascend_950pr_simd_simt_v1.json"))


def fit_A_m(eng, nl, ns, b, grid):
    rows = []
    for key, v in PROF[eng]["memory"]["contiguous_ldst_v2"]["direct"].items():
        m = re.match(r"n(\d+)_s(\d+)_b(\d+)_k(\d+)_g(\d+)$", key)
        if not m:
            continue
        if (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(5))) == (nl, ns, b, grid):
            rows.append((int(m.group(4)), v))
    rows.sort()
    if len(rows) < 2:
        return None, None
    slopes = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dK = rows[j][0] - rows[i][0]
            if dK > 0:
                slopes.append((rows[j][1] - rows[i][1]) / dK)
    slopes.sort()
    m = slopes[len(slopes) // 2]
    kc, tc = rows[len(rows) // 2]
    A = max(0.0, tc - kc * m)
    return A, m


def main():
    cases = [
        ("B_copy_f32_4k_k16_g56", 1, 1, 4096, 56, 16),
        ("B_copy_f32_4k_k64_g56", 1, 1, 4096, 56, 64),
        ("B_copy_f32_4k_k4_g8", 1, 1, 4096, 8, 4),
        ("B_add_f32_4k_k16_g56", 2, 1, 4096, 56, 16),
        ("B_load_f32_4k_k64_g56", 1, 0, 4096, 56, 64),
        ("B_store_f32_4k_k16_g56", 0, 1, 4096, 56, 16),
        ("B_copy_f16_2k_k16_g56", 1, 1, 2048, 56, 16),
        ("B_copy_i8_1k_k16_g56", 1, 1, 1024, 56, 16),
        ("B_store_i64_8k_k4_g56", 0, 1, 8192, 56, 4),
        ("B_copy_f32_4k_k8_g56", 1, 1, 4096, 56, 8),
        ("B_copy_f32_4k_k32_g56", 1, 1, 4096, 56, 32),
    ]
    table = json.load(open(sys.argv[1]))
    by_case = {}
    for r in table:
        by_case.setdefault(r["case"], {})[r["combo"]] = r
    out = []
    for (case, nl, ns, b, grid, K) in cases:
        row = by_case.get(case, {}).get("simd_f1")
        if not row or not row["pred_ns"] or not row["meas_ns"]:
            continue
        eng = "simd"
        A, m = fit_A_m(eng, nl, ns, b, grid)
        if A is None:
            continue
        # pred(s) = pred(0.3) + (s-0.3)*A; want pred = meas
        s_star = 0.3 + (row["meas_ns"] - row["pred_ns"]) / A
        out.append((case, row["pred_ns"], row["meas_ns"], A, m, s_star))
        print("%-26s pred=%.0f meas=%.0f A=%.0f m=%.1f s*=%.3f" %
              (case, row["pred_ns"], row["meas_ns"], A, m, s_star))
    if out:
        ss = sorted(x[5] for x in out)
        print("median s* = %.3f  (clamp to [0,1])" % ss[len(ss) // 2])


if __name__ == "__main__":
    main()
