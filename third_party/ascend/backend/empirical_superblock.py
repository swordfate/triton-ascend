#!/usr/bin/env python3
"""Empirical whole-kernel SuperBlock spill penalties (v1).

The native C++ Route Model owns the final SuperBlock-factor decision.  This
module only publishes an *additional* per-physical-program penalty for known
whole-kernel configurations where 64 total warps (2048 threads) spill from the
128 KB SIMT register file and are slower than a smaller factor despite the
wave reduction.

Python must not override or recommend the final factor.  The values below are
inserted into route_transform_capability["empirical_whole_kernel_spill_penalty"]
and applied by ``solveStageRoutes`` only to the all-SIMT whole-kernel plan,
before the runtime wave multiplier.  Scope-local SuperBlock factors (the mixed
route) intentionally receive no penalty.

Keys are canonical kernel name -> total_warps (num_warps * factor).  The
penalty unit is system cycles per physical program, matching the stage-route
score unit.  Configurations not present in the table return an empty map so
the native model keeps its existing behavior.
"""

import os

_SUPPORTED_FACTORS = (1, 2, 4, 8, 16, 32, 64)

# Per-physical-program penalty in system cycles, keyed by total physical warps
# (num_warps * superblock_factor).  Total warps below 64 are currently
# represented without a measured spill penalty; absent entries stay native.
_SPILL_PENALTY_BY_TOTAL_WARPS = {
    "padded_copy_gather": {
        64: 1200.0,
    },
    "padded_copy_scatter": {
        64: 800.0,
    },
    "padded_copy_wgrad": {
        64: 1500.0,
    },
    "binned_copy_gather": {
        64: 800.0,
    },
    "binned_copy_scatter": {
        64: 1500.0,
    },
    "binned_copy_wgrad": {
        64: 1200.0,
    },
}


def _canonical_kernel_name(name: str) -> str:
    name = (name or "").strip()
    while name.startswith("_"):
        name = name[1:]
    # Cache/dump names occasionally carry a module suffix.
    for suffix in ("__grp", ".", "$"):
        if suffix in name:
            name = name.split(suffix, 1)[0]
    return name


def is_empirical_spill_penalty_enabled() -> bool:
    """Return whether the empirical penalty table is published.

    The table is enabled by default so pure auto mode gets the calibrated
    behavior.  ``TRITON_ASCEND_EMPIRICAL_SPILL_PENALTY=0`` (or the legacy
    ``TRITON_ASCEND_EMPIRICAL_SUPERBLOCK=0``) disables it for A/B runs.
    """
    value = os.getenv("TRITON_ASCEND_EMPIRICAL_SPILL_PENALTY")
    if value is None:
        value = os.getenv("TRITON_ASCEND_EMPIRICAL_SUPERBLOCK")
    if value is None:
        return True
    return value.strip().lower() not in ("", "0", "false", "off", "no")


def penalty_by_factor(kernel_name: str, logical_program_count_hint: int,
                      num_warps: int) -> dict[int, float]:
    """Return ``{superblock_factor: penalty_cycles_per_physical_program}``.

    Only factors that are legal for this launch and whose resulting total warp
    count appears in the measured table are returned.  Unknown kernels/shapes
    return ``{}``, which preserves the native route solver decision.
    """
    if not is_empirical_spill_penalty_enabled():
        return {}
    penalties = _SPILL_PENALTY_BY_TOTAL_WARPS.get(
        _canonical_kernel_name(kernel_name))
    if not penalties:
        return {}

    num_warps = max(1, int(num_warps or 1))
    logical_count = max(0, int(logical_program_count_hint or 0))
    max_factor = min(64, 64 // num_warps)
    result: dict[int, float] = {}
    for factor in _SUPPORTED_FACTORS:
        if factor < 2 or factor > max_factor:
            continue
        if logical_count and factor > logical_count:
            continue
        penalty = penalties.get(num_warps * factor)
        if penalty is None or penalty <= 0.0:
            continue
        result[factor] = float(penalty)
    return result
