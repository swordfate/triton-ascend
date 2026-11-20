#!/usr/bin/env python3
"""Python replica of loopFitPerIter (simd + simt) using the installed
profile JSON; cross-check against the fresh stage per-iter values for
the 5 new-FAIL cases and list all >10% loop-domain cases."""
import json
import math

P = ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/_C/"
     "ascend/costmodel_profiles/simd_simt/ascend_950pr_simd_simt_v1.json")
prof = json.load(open(P))


def curve_ns(curve, b):
    ks, vs = curve["b"], curve["ns"]
    if b <= ks[0]:
        return vs[0]
    if b >= ks[-1]:
        return vs[-1]
    for i in range(len(ks) - 1):
        if ks[i] <= b <= ks[i + 1]:
            w = (math.log2(b) - math.log2(ks[i])) / \
                (math.log2(ks[i + 1]) - math.log2(ks[i]))
            return vs[i] + w * (vs[i + 1] - vs[i])
    return vs[-1]


def lerp_map(m, n, b):
    if not m or n <= 0 or b <= 0:
        return 0.0
    mint = {int(k): v for k, v in m.items()}
    if n in mint:
        return curve_ns(mint[n], b)
    keys = sorted(mint.keys())
    import bisect
    i = bisect.bisect_left(keys, n)
    if i == len(keys):
        last = keys[-1]
        return curve_ns(mint[last], b) * n / last
    if i == 0:
        k0 = keys[0]
        return curve_ns(mint[k0], b) * n / k0
    prev, nxt = keys[i - 1], keys[i]
    w = (math.log2(n) - math.log2(prev)) / (math.log2(nxt) - math.log2(prev))
    return curve_ns(mint[prev], b) + w * (curve_ns(mint[nxt], b) - curve_ns(mint[prev], b))


def dom_ns(tbl, dom, ns):
    if not tbl:
        return 0.0
    keys = sorted(tbl.keys(), key=float)
    d = float(dom)
    if d <= float(keys[0]):
        return tbl[keys[0]][0][1]
    if d >= float(keys[-1]):
        return tbl[keys[-1]][-1][1]
    # find bracketing dom key, then lerp on nS within the pair list
    import bisect
    i = bisect.bisect_left([float(k) for k in keys], d)
    if i == 0:
        i = 1
    lo_k, hi_k = keys[i - 1], keys[i]
    lo, hi = tbl[lo_k], tbl[hi_k]
    lo = sorted(lo)
    hi = sorted(hi)
    n = ns

    def eval_pair(pairs, n):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        if n <= xs[0]:
            return ys[0]
        if n >= xs[-1]:
            return ys[-1]
        for j in range(len(xs) - 1):
            if xs[j] <= n <= xs[j + 1]:
                w = (n - xs[j]) / (xs[j + 1] - xs[j])
                return ys[j] + w * (ys[j + 1] - ys[j])
        return ys[-1]

    a = eval_pair(lo, n)
    b2 = eval_pair(hi, n)
    w = (d - float(lo_k)) / (float(hi_k) - float(lo_k)) if hi_k != lo_k else 0
    return a + w * (b2 - a)


def loop_fit_per_iter(eng, nL, lB, nS, sB, dom, tiny, redFed):
    f = prof[eng]["memory"]["contiguous_ldst_v2"]["loop_envelope_fit"]["dtypes"]["fp32"]
    step_l = {int(k): v for k, v in f["step_l"].items()}
    step_s = {int(k): v for k, v in f["step_s"].items()}
    step_t = {int(k): v for k, v in f["step_t"].items()}
    L = lerp_map(f["load_curves"], nL, lB) if nL > 0 else 0.0
    S = lerp_map(f["store_curves"], nS, sB) if nS > 0 else 0.0
    if f.get("additive"):
        per = L + S
        if nS > 0 and tiny:
            lbase = L
            if nL in step_l:
                lbase = max(lbase, step_l[nL])
            if nL in step_t:
                lbase = max(lbase, step_t[nL])
            per = lbase + dom_ns(f["chain"] if redFed else f["excess"], dom, nS)
        return per
    floor = 0.0
    if nL in step_l:
        floor = max(floor, step_l[nL])
    if nS in step_s:
        floor = max(floor, step_s[nS])
    if (nL + nS) in step_t:
        floor = max(floor, step_t[nL + nS])
    per = max(floor, max(L, S))
    if nL > 0 and nS > 0:
        per += f["c_l"] * lB / 128.0 + f["c_s"] * sB / 128.0
    if nS > 0 and tiny:
        lbase = max(floor, L)
        if nL in step_t:
            lbase = max(lbase, step_t[nL])
        per = lbase + dom_ns(f["chain"] if redFed else f["excess"], dom, nS)
    return per


# (case, eng, nL, lB, nS, sB, dom, tiny, redFed, fresh_stage_total, iters)
CASES = [
    ("muls_add_n1048576", "simd", 2, 8192, 1, 4096, 4096, False, False, 2122.7, 19),
    ("memory_stream_l4", "simd", 4, 16384, 1, 4096, 4096, False, False, 5710.1, 37),
    ("permute_reshape", "simd", 6, 12288, 2, 4096, 4096, False, False, 2413.5, 12),
    ("fused_softmax", "simd", 1, 2048, 1, 2048, 2048, False, True, 1106.6, 4),
    ("strided_scatter", "simt", 1, 1024, 1, 1024, 1024, False, False, 1901.5, 8),
]
for case, eng, nL, lB, nS, sB, dom, tiny, redFed, stotal, it in CASES:
    per = loop_fit_per_iter(eng, nL, lB, nS, sB, dom, tiny, redFed)
    print("%-20s eng=%s nL=%d lB=%d nS=%d sB=%d -> loopfit/iter=%7.1f  "
          "fresh_stage/iter=%7.1f (total %.1f/%d)" %
          (case, eng, nL, lB, nS, sB, per, stotal / it, stotal, it))
