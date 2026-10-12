#!/usr/bin/env python3
"""Semi-white 1D rank1 template features.

These features are derived directly from the NPUIR template source:

  SIMTStrideLoad.cpp / SIMTStrideStore.cpp:
    * 1024 threads (``STRIDE_*_THREAD_NUM``);
    * each thread i0 loops ``i0 = tid; i0 < size; i0 += 1024`` and performs a
      scalar GM access at ``lower + i0 * stride``;
    * load/store size is ``min(numel, dst/src size)``.  For the unmasked probe
      numel == BLOCK, so no ``simtStridePad1D`` runs.

The feature functions therefore use logical element indices ``i`` in
``[0, block)``, address ``i * stride_bytes``.  Base alignment is fixed to a
128B/4KB boundary by the probe (offset 0), so line/page indices are exact.
"""
from __future__ import annotations

from collections import OrderedDict

ELEM_BYTES = 4
LINE_BYTES = 128
PAGE_BYTES = 4096
THREADS = 1024
WARP = 32


def _line_counts(block: int, stride: int):
    counts = {}
    for i in range(int(block)):
        line = (i * int(stride) * ELEM_BYTES) // LINE_BYTES
        counts[line] = counts.get(line, 0) + 1
    return counts


def distinct_lines(block: int, stride: int) -> int:
    return len(_line_counts(block, stride))


def line_elems(block: int, stride: int):
    counts = _line_counts(block, stride)
    if not counts:
        return 0, 0
    values = list(counts.values())
    return max(values), min(values)


def lines_per_warp(block: int, stride: int):
    """Line counts for each 32-thread warp/wave of one 1024-thread iteration."""
    active = min(int(block), THREADS)
    counts = []
    for start in range(0, active, WARP):
        stop = min(start + WARP, active)
        lines = set()
        for i in range(start, stop):
            lines.add((i * int(stride) * ELEM_BYTES) // LINE_BYTES)
        counts.append(len(lines))
    if not counts:
        return 0.0, 0.0, 0.0
    return (
        float(sum(counts) / len(counts)),
        float(min(counts)),
        float(max(counts)),
    )


def bucket_pair_features(block: int, stride: int, group_bytes: int, buckets: int):
    counts = [0] * buckets
    for i in range(int(block)):
        addr = i * int(stride) * ELEM_BYTES
        counts[(addr // group_bytes) % buckets] += 1
    pairs = sum(c * (c - 1) // 2 for c in counts)
    worst = max(counts) - 1
    return float(pairs), float(worst)


def page_cross_count(block: int, stride: int) -> float:
    span = max(0, (int(block) - 1) * int(stride) * ELEM_BYTES)
    return float(span // PAGE_BYTES)


def base_features(block: int, stride: int, num_warps: int = 1):
    """Return an ordered feature dict used by the template model.

    The feature names intentionally mirror the template-level work rather than
    any Triton operator name:
      iters_per_thread = ceil(block / 1024)
      active_threads   = min(block, 1024)
      L                = distinct 128B lines over all scalar accesses
      line_elems_max   = elements per densest 128B line (density proxy)
      page_cross       = 4KB boundary crossings of the strided span
    """
    block = int(block)
    stride = int(stride)
    stride_bytes = stride * ELEM_BYTES
    iters = (block + THREADS - 1) // THREADS
    active_threads = min(block, THREADS)
    active_warps = (active_threads + WARP - 1) // WARP
    tail_elems = block % THREADS
    lines = distinct_lines(block, stride)
    max_line_elems, min_line_elems = line_elems(block, stride)
    mean_warp_lines, min_warp_lines, max_warp_lines = lines_per_warp(block, stride)
    pairs_2k8, worst_2k8 = bucket_pair_features(block, stride, 2048, 8)
    pairs_32k16, worst_32k16 = bucket_pair_features(block, stride, 32768, 16)
    features = OrderedDict(
        [
            ("intercept", 1.0),
            ("iters_per_thread", float(iters)),
            ("active_threads", float(active_threads)),
            ("active_warps", float(active_warps)),
            ("tail_elems", float(tail_elems)),
            ("tail_lanes_gt0", 1.0 if tail_elems else 0.0),
            ("L", float(lines)),
            ("line_elems_max", float(max_line_elems)),
            ("line_elems_min", float(min_line_elems)),
            ("line_density", float(block) / float(lines) if lines else 0.0),
            ("mean_warp_lines", mean_warp_lines),
            ("min_warp_lines", min_warp_lines),
            ("max_warp_lines", max_warp_lines),
            ("stride_bytes", float(stride_bytes)),
            ("stride_gt_line", 1.0 if stride_bytes >= LINE_BYTES else 0.0),
            ("page_cross", page_cross_count(block, stride)),
            ("bucket_pairs_2k8", pairs_2k8),
            ("bucket_worst_2k8", worst_2k8),
            ("bucket_pairs_32k16", pairs_32k16),
            ("bucket_worst_32k16", worst_32k16),
            ("num_warps_minus1", float(int(num_warps) - 1)),
        ]
    )
    return features


FEATURE_ORDER = list(base_features(32, 3).keys())


def feature_vector(block, stride, num_warps=1):
    f = base_features(block, stride, num_warps)
    return [f[name] for name in FEATURE_ORDER]
