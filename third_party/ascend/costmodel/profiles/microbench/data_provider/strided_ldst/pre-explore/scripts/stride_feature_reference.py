#!/usr/bin/env python3
"""Exact integer-semantics reference implementation of every strided feature.

This module is the single source of truth for the features used by the three
strided lowerings.  It is written the way the C++ side computes them
(StageCostModels.cpp), extended to non-4-byte element types, so that a future
C++ port can reproduce every value bit for bit.

C++ correspondence
------------------
  stridedLineCount(elements, strideBytes)
  stridedLineRequestSize(elements, strideBytes)   <-- NOTE: hardcodes * 4.0
  stridedBankFacts(elements, strideBytes, bucketBytes, bucketCount, ...)
  stridedTemplateMeanWarpLines(elements, strideBytes)
  stridedSimtLoadCostNs / stridedSimtStoreCostNs / stridedSimd*CostNs

Integer semantics, written explicitly (no numpy vectorisation, so the floor /
modulo boundaries are visible):

  lines(E, SB)          = 0                            if E <= 0
                        = E                            if SB >= 128
                        = (E-1)*SB // 128 + 1          otherwise
  line_req(E, SB, EB)   = EB                           if SB >= 128
                        = min(128, max(1, 128//SB)*EB) otherwise
  bank_pairs(E, SB, G, B) = sum_k c_k*(c_k-1)/2,  c_k = #{i: (i*SB)//G % B == k}
  bank_worst(E, SB, G, B) = max_k c_k - 1
  mean_warp_lines(E, SB)  = mean over 32-thread waves of the number of distinct
                            128B lines, waves taken over min(E, 1024) threads
  page_cross(E, SB)     = (E-1)*SB // 4096
  cross(E, SB)          = 1 if (E-1)*SB >= 4096 else 0
  E_per_thread          = E / (32*W)
  dup                   = max(0, 1/E_per_thread - 1)   (0 when E_per_thread <= 0)
  tail_elems            = E mod 1024
  iters_per_thread      = ceil(E / 1024)

Run the built-in self check with:

    python3 scripts/stride_feature_reference.py --verify
"""
from __future__ import annotations

import argparse
import math
from collections import OrderedDict

LINE_BYTES = 128
PAGE_BYTES = 4096
WARP = 32
TEMPLATE_THREADS = 1024


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def lines(elements: int, stride_bytes: int) -> int:
    """Distinct 128B lines touched by a 1-D strided tile (C++ stridedLineCount)."""
    elements = int(elements)
    stride_bytes = int(stride_bytes)
    if elements <= 0:
        return 0
    if stride_bytes >= LINE_BYTES:
        return elements
    return ((elements - 1) * stride_bytes) // LINE_BYTES + 1


def line_request_size(elements: int, stride_bytes: int, element_bytes: int) -> int:
    """Useful bytes carried by one 128B line.

    The C++ helper hardcodes 4 bytes per element; this version takes the real
    element size.  For element_bytes == 4 the two agree.
    """
    stride_bytes = int(stride_bytes)
    element_bytes = int(element_bytes)
    if stride_bytes >= LINE_BYTES:
        return element_bytes
    per_line = max(1, LINE_BYTES // stride_bytes)
    return min(LINE_BYTES, per_line * element_bytes)


def bank_facts(elements: int, stride_bytes: int, group_bytes: int,
               buckets: int):
    """(pairs, worst) of the bucket occupancy histogram."""
    elements = int(elements)
    stride_bytes = int(stride_bytes)
    if elements <= 0:
        return 0.0, 0.0
    counts = [0] * int(buckets)
    for i in range(elements):
        counts[((i * stride_bytes) // int(group_bytes)) % int(buckets)] += 1
    pairs = sum(c * (c - 1) // 2 for c in counts)
    worst = max(counts) - 1
    return float(pairs), float(worst)


def mean_warp_lines(elements: int, stride_bytes: int) -> float:
    """Mean distinct-line count per 32-thread wave of the 1024-thread template."""
    elements = int(elements)
    stride_bytes = int(stride_bytes)
    active = min(elements, TEMPLATE_THREADS)
    if active <= 0 or stride_bytes <= 0:
        return 0.0
    total = 0
    warps = 0
    for start in range(0, active, WARP):
        stop = min(start + WARP, active)
        previous = -1
        count = 0
        for i in range(start, stop):
            line = (i * stride_bytes) // LINE_BYTES
            if line != previous:
                count += 1
                previous = line
        total += count
        warps += 1
    return total / warps if warps else 0.0


def page_cross(elements: int, stride_bytes: int) -> int:
    if elements <= 0:
        return 0
    return ((elements - 1) * stride_bytes) // PAGE_BYTES


def cross_flag(elements: int, stride_bytes: int) -> float:
    return 1.0 if (elements - 1) * stride_bytes >= PAGE_BYTES else 0.0


def dup_factor(elements: int, num_warps: int) -> float:
    e = float(elements) / (WARP * float(num_warps))
    if e <= 0.0:
        return 0.0
    return max(0.0, 1.0 / e - 1.0)


def stg_instructions(elements: int, num_warps: int) -> float:
    warps = float(max(1, int(num_warps)))
    return warps * max(1.0, math.ceil(float(elements) / (WARP * warps)))


def ldg_instructions(elements: int, num_warps: int) -> float:
    return stg_instructions(elements, num_warps)


# ---------------------------------------------------------------------------
# per-path feature dicts
# ---------------------------------------------------------------------------

def simd_features(elements: int, stride_bytes: int, element_bytes: int,
                  bucket_group_bytes: int = 2048, buckets: int = 8):
    pairs, worst = bank_facts(elements, stride_bytes, bucket_group_bytes, buckets)
    return OrderedDict([
        ("intercept", 1.0),
        ("n_cmd_block", float(elements)),
        ("elements", float(elements)),
        ("stride_bytes", float(stride_bytes)),
        ("stride_minus_1", float(stride_bytes // element_bytes) - 1.0),
        ("large_tile_gt64", 1.0 if elements > 64 else 0.0),
        ("tail_elems_gt64", float(max(0, elements - 64))),
        ("bucket_pairs_g2048_b8", pairs),
        ("bucket_worst_g2048_b8", worst),
        ("page_cross_count", float(page_cross(elements, stride_bytes))),
        ("lines", float(lines(elements, stride_bytes))),
        ("line_request_size", float(line_request_size(elements, stride_bytes,
                                                      element_bytes))),
    ])


def simt_features(elements: int, stride_bytes: int, element_bytes: int,
                  num_warps: int):
    return OrderedDict([
        ("intercept", 1.0),
        ("K", ldg_instructions(elements, num_warps)),
        ("K_stg", stg_instructions(elements, num_warps)),
        ("W", float(num_warps)),
        ("L", float(lines(elements, stride_bytes))),
        ("dup", dup_factor(elements, num_warps)),
        ("cross", cross_flag(elements, stride_bytes)),
        ("page_count", float(page_cross(elements, stride_bytes))),
        ("line_request_size", float(line_request_size(elements, stride_bytes,
                                                     element_bytes))),
        ("min_L_32", float(min(lines(elements, stride_bytes), 32))),
        ("max_L_minus_32", float(max(0, lines(elements, stride_bytes) - 32))),
        ("min_L_64", float(min(lines(elements, stride_bytes), 64))),
        ("max_L_minus_64", float(max(0, lines(elements, stride_bytes) - 64))),
        ("dupL", dup_factor(elements, num_warps) * lines(elements, stride_bytes)),
        ("dupL_minL_128", dup_factor(elements, num_warps)
         * lines(elements, stride_bytes)
         * min(lines(elements, stride_bytes), 128)),
        ("cross_minL_32", cross_flag(elements, stride_bytes)
         * min(lines(elements, stride_bytes), 32)),
        ("lowW_4_L", max(0.0, 4.0 - num_warps) * lines(elements, stride_bytes)),
    ])


def template_features(elements: int, stride_bytes: int, element_bytes: int,
                      num_warps: int = 1):
    _, worst_32k16 = bank_facts(elements, stride_bytes, 32768, 16)
    return OrderedDict([
        ("intercept", 1.0),
        ("L", float(lines(elements, stride_bytes))),
        ("bucket_worst_32k16", worst_32k16),
        ("mean_warp_lines", mean_warp_lines(elements, stride_bytes)),
        ("tail_elems", float(int(elements) % TEMPLATE_THREADS)),
        ("iters_per_thread", float(math.ceil(int(elements) / TEMPLATE_THREADS))),
        ("active_threads", float(min(int(elements), TEMPLATE_THREADS))),
        ("line_request_size", float(line_request_size(elements, stride_bytes,
                                                      element_bytes))),
        ("page_cross", float(page_cross(elements, stride_bytes))),
        ("stride_bytes", float(stride_bytes)),
        ("elements", float(elements)),
    ])


FEATURE_SETS = {
    "simd": ["intercept", "n_cmd_block", "bucket_pairs_g2048_b8",
             "bucket_worst_g2048_b8", "page_cross_count"],
    "simd_wide": ["intercept", "stride_minus_1", "large_tile_gt64",
                  "tail_elems_gt64"],
    "simt_v8": ["intercept", "min_L_32", "max_L_minus_32", "dupL_minL_128",
                "cross_minL_32", "lowW_4_L"],
    "simt_v7": ["intercept", "dupL", "min_L_64", "max_WL_minus_3072", "dup_cross"],
    "simt_store": ["intercept", "K_stg", "L", "line_request_size"],
    "template_load": ["intercept", "L", "bucket_worst_32k16", "mean_warp_lines",
                      "tail_elems"],
    "template_store": ["intercept", "L", "bucket_worst_32k16", "mean_warp_lines",
                       "iters_per_thread"],
}


def design_row(feature_set: str, elements: int, stride_bytes: int,
               element_bytes: int, num_warps: int = 1):
    """One design-matrix row for the named feature set."""
    if feature_set in ("simd", "simd_wide"):
        f = simd_features(elements, stride_bytes, element_bytes)
    elif feature_set in ("simt_v8", "simt_v7", "simt_store"):
        f = simt_features(elements, stride_bytes, element_bytes, num_warps)
    elif feature_set in ("template_load", "template_store"):
        f = template_features(elements, stride_bytes, element_bytes, num_warps)
    else:
        raise KeyError(feature_set)
    return [f[name] for name in FEATURE_SETS[feature_set]]



# ---------------------------------------------------------------------------
# rank-2 (2D) tile features
# ---------------------------------------------------------------------------
#
# Address set of one rank-2 tile:
#
#     addr(r, c) = r * OS + c * IS        r in [0,R), c in [0,C)
#
# Contiguity of the whole tile:
#
#     fully contiguous  <=>  IS == 1 and OS == C
#     strided           <=>  anything else
#
# Everything below is computed from the explicit address list in row-major
# order, exactly as the hardware walks the flattened tile, so the 1-D line /
# warp-line quantities generalise without redefining them.

def tile_addresses(rows: int, cols: int, outer_stride: int,
                   inner_stride: int):
    return [r * int(outer_stride) + c * int(inner_stride)
            for r in range(int(rows)) for c in range(int(cols))]


def is_contiguous_tile(rows: int, cols: int, outer_stride: int,
                       inner_stride: int) -> bool:
    return int(inner_stride) == 1 and int(outer_stride) == int(cols)


def address_list_features(addrs, element_bytes: int, threads: int = 1024,
                          warp: int = 32):
    """Byte-domain features of an explicit element-address list."""
    byte_addrs = [int(a) * int(element_bytes) for a in addrs]
    elements = len(byte_addrs)
    feature = OrderedDict()
    feature["elements"] = float(elements)
    if elements == 0:
        for name in ("L", "line_elems_max", "line_elems_min", "line_density",
                     "mean_warp_lines", "min_warp_lines", "max_warp_lines",
                     "bucket_pairs_2k8", "bucket_worst_2k8",
                     "bucket_pairs_32k16", "bucket_worst_32k16",
                     "span_bytes", "page_cross", "cross"):
            feature[name] = 0.0
        return feature
    line_of = [b // LINE_BYTES for b in byte_addrs]
    counts = {}
    for line in line_of:
        counts[line] = counts.get(line, 0) + 1
    feature["L"] = float(len(counts))
    feature["line_elems_max"] = float(max(counts.values()))
    feature["line_elems_min"] = float(min(counts.values()))
    feature["line_density"] = elements / float(len(counts))

    active = min(elements, int(threads))
    per_warp = []
    for start in range(0, active, int(warp)):
        stop = min(start + int(warp), active)
        seen = []
        for i in range(start, stop):
            if not seen or seen[-1] != line_of[i]:
                seen.append(line_of[i])
        per_warp.append(len(set(line_of[start:stop])))
    feature["mean_warp_lines"] = (float(sum(per_warp)) / len(per_warp)
                                  if per_warp else 0.0)
    feature["min_warp_lines"] = float(min(per_warp)) if per_warp else 0.0
    feature["max_warp_lines"] = float(max(per_warp)) if per_warp else 0.0

    for group_bytes, buckets, tag in ((2048, 8, "2k8"), (32768, 16, "32k16")):
        hist = [0] * buckets
        for b in byte_addrs:
            hist[(b // group_bytes) % buckets] += 1
        feature[f"bucket_pairs_{tag}"] = float(
            sum(c * (c - 1) // 2 for c in hist))
        feature[f"bucket_worst_{tag}"] = float(max(hist) - 1)

    span = max(byte_addrs) - min(byte_addrs)
    feature["span_bytes"] = float(span)
    feature["page_cross"] = float(span // PAGE_BYTES)
    feature["cross"] = 1.0 if span >= PAGE_BYTES else 0.0
    return feature


def tile_features(rows: int, cols: int, outer_stride: int, inner_stride: int,
                  element_bytes: int):
    addrs = tile_addresses(rows, cols, outer_stride, inner_stride)
    feature = address_list_features(addrs, element_bytes)
    inner_span = (int(cols) - 1) * int(inner_stride)
    row_gap = int(outer_stride) - inner_span - 1
    feature["R"] = float(rows)
    feature["C"] = float(cols)
    feature["OS"] = float(outer_stride)
    feature["IS"] = float(inner_stride)
    feature["inner_stride_bytes"] = float(int(inner_stride) * element_bytes)
    feature["row_gap_elements"] = float(row_gap)
    feature["row_gap_bytes"] = float(row_gap * element_bytes)
    feature["contiguous"] = 1.0 if is_contiguous_tile(
        rows, cols, outer_stride, inner_stride) else 0.0
    feature["tile_bytes"] = float(int(rows) * int(cols) * element_bytes)
    feature["line_request_size"] = float(line_request_size(
        int(rows) * int(cols), int(inner_stride) * element_bytes, element_bytes))
    return feature


TILE_FEATURE_SETS = {
    "template_2d": ["intercept", "L", "bucket_worst_32k16", "mean_warp_lines",
                    "tail_elems_2d"],
    "simd_2d": ["intercept", "elements", "bucket_pairs_2k8",
                "bucket_worst_2k8", "page_cross"],
    "simt_2d": ["intercept", "K_2d", "L", "line_request_size"],
}


def tile_design_row(feature_set: str, rows: int, cols: int, outer_stride: int,
                    inner_stride: int, element_bytes: int, num_warps: int = 1):
    f = tile_features(rows, cols, outer_stride, inner_stride, element_bytes)
    elements = int(rows) * int(cols)
    values = {
        "intercept": 1.0,
        "elements": f["elements"],
        "L": f["L"],
        "bucket_worst_32k16": f["bucket_worst_32k16"],
        "mean_warp_lines": f["mean_warp_lines"],
        "bucket_pairs_2k8": f["bucket_pairs_2k8"],
        "bucket_worst_2k8": f["bucket_worst_2k8"],
        "page_cross": f["page_cross"],
        "line_request_size": f["line_request_size"],
        "tail_elems_2d": float(elements % TEMPLATE_THREADS),
        "iters_per_thread_2d": float(math.ceil(elements / TEMPLATE_THREADS)),
        "K_2d": float(max(1, int(num_warps))
                      * max(1.0, math.ceil(elements / (WARP * max(1, int(num_warps)))))),
        "row_gap_bytes": f["row_gap_bytes"],
        "inner_stride_bytes": f["inner_stride_bytes"],
    }
    return [values[name] for name in TILE_FEATURE_SETS[feature_set]]


# ---------------------------------------------------------------------------
# self check
# ---------------------------------------------------------------------------

def _verify_against_cpp_mirror(samples):
    """Cross-check the primitives against a literal transcription of the C++."""
    def cpp_lines(e, sb):
        if e <= 0:
            return 0.0
        if sb >= 128.0:
            return float(e)
        return math.floor(((e - 1.0) * sb) / 128.0) + 1.0

    def cpp_line_req(e, sb):
        if sb >= 128.0:
            return 4.0
        per_line = max(1.0, math.floor(128.0 / sb))
        return min(128.0, per_line * 4.0)

    def cpp_bank(e, sb, g, b):
        counts = [0] * b
        for i in range(int(max(0.0, e))):
            counts[int((i * sb) // g) % b] += 1
        pairs = sum(c * (c - 1) // 2 for c in counts)
        return float(pairs), float(max(counts) - 1)

    def cpp_mean_warp_lines(e, sb):
        active = int(min(max(0.0, e), 1024.0))
        if active <= 0 or sb <= 0.0:
            return 0.0
        total = 0.0
        warps = 0
        for start in range(0, active, 32):
            stop = min(start + 32, active)
            prev = -1
            count = 0
            for i in range(start, stop):
                line = int(math.floor(i * sb / 128.0))
                if line != prev:
                    count += 1
                    prev = line
            total += count
            warps += 1
        return total / warps if warps else 0.0

    bad = []
    for e, sb, eb in samples:
        if lines(e, sb) != cpp_lines(e, sb):
            bad.append(("lines", e, sb))
        if eb == 4:
            if line_request_size(e, sb, eb) != cpp_line_req(e, sb):
                bad.append(("line_request_size", e, sb))
        for g, b in ((2048, 8), (32768, 16)):
            if bank_facts(e, sb, g, b) != cpp_bank(e, sb, g, b):
                bad.append(("bank_facts", e, sb, g, b))
        if abs(mean_warp_lines(e, sb) - cpp_mean_warp_lines(e, sb)) > 1e-12:
            bad.append(("mean_warp_lines", e, sb))
    return bad


def verify():
    samples = []
    for e in list(range(1, 200)) + [256, 512, 1023, 1024, 1025, 2048, 4096]:
        for sb in list(range(1, 40)) + [63, 64, 65, 127, 128, 129, 255, 256, 512]:
            samples.append((e, sb, 4))
    bad = _verify_against_cpp_mirror(samples)
    print(f"cpp-mirror cross-check: {len(samples)} samples, {len(bad)} mismatches")
    for item in bad[:10]:
        print("  MISMATCH", item)

    # Cross-check against the committed v2 feature module on the f32 grid.
    import template_stride_features as tsf
    mismatch = 0
    checked = 0
    for block in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        for stride in [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21,
                       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 40, 48, 63, 65,
                       80, 96, 127, 129, 160, 192, 255]:
            ref = template_features(block, stride * 4, 4)
            got = tsf.base_features(block, stride)
            for name in ("iters_per_thread", "active_threads", "tail_elems", "L",
                         "mean_warp_lines", "bucket_worst_32k16", "page_cross",
                         "stride_bytes"):
                checked += 1
                if abs(float(ref[name]) - float(got[name])) > 1e-12:
                    mismatch += 1
                    if mismatch < 6:
                        print(f"  MISMATCH {name} block={block} stride={stride} "
                              f"ref={ref[name]} v2={got[name]}")
    print(f"template_stride_features cross-check: {checked} values, "
          f"{mismatch} mismatches")
    if bad or mismatch:
        raise SystemExit("FEATURE REFERENCE VERIFY FAILED")
    print("VERIFY OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify()
        return
    print("import this module, or run --verify")


if __name__ == "__main__":
    main()
