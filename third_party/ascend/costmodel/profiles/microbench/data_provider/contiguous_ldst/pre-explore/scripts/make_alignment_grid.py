#!/usr/bin/env python3
"""Emit the non-aligned (row-tail) grid.

Two axes, chosen because the IR reconnaissance showed they are separate
triggers, not one:

  trailing extent C : when C stops being a power of two the lowered copy
                      destination gets a padded UB row (next power of two)
                      plus subviews - even when the row is byte-aligned, as
                      C=160 (640 B = 5 x 128 B) demonstrates;
  row count R       : when R stops being a power of two the destination row
                      stride is NOT padded, but subviews still appear.

C=120 (480 B, a multiple of 32) and C=160 (640 B, a multiple of 128) are the
two points that separate "byte alignment" from "power-of-two extent": if the
byte-alignment story were right both would behave like the aligned control,
and the measurement decides it.

fp32 only, and only factor 1: the alignment question is orthogonal to element
width and to SuperBlock grouping, and both are already characterised.
"""

import argparse
import json
from pathlib import Path

OUTERS = (64, 256)
PROGRAMS = 224
GROUPS = 4
UB_BUDGET = 200 * 1024
FOOTPRINT_BUDGET = 64 * 1024 * 1024
ELEMENT_BYTES = 4

COL_LADDER_ROWS = 32
COL_LADDER = (128, 127, 120, 160, 192, 256)
ROW_LADDER_COLS = 128
ROW_LADDER = (32, 31, 33, 64)


def pad(value):
    return 1 << max(0, (value - 1).bit_length())


def points():
    out = []

    def add(rows, cols, mode, factor, direction, tag):
        rpad, cpad = pad(rows), pad(cols)
        stride = cols * rpad
        for group in (2, 1):
            footprint = PROGRAMS * GROUPS * group * stride * ELEMENT_BYTES
            ub_bytes = 2 * stride * ELEMENT_BYTES
            if footprint <= FOOTPRINT_BUDGET and ub_bytes <= UB_BUDGET:
                for outer in OUTERS:
                    out.append({"tag": "%s_o%d" % (tag, outer),
                                "mode": mode, "factor": factor,
                                "direction": direction, "dtype": "fp32",
                                "real_rows": rows, "real_cols": cols,
                                "row_stride": cols,
                                "groups": GROUPS, "group": group,
                                "outer": outer,
                                "logical_programs": PROGRAMS, "grid": 1})
                return
        raise SystemExit("no admissible unroll for %s" % tag)

    for mode in ("simd", "simt_only"):
        for direction in ("load", "store"):
            for cols in COL_LADDER:
                add(COL_LADDER_ROWS, cols, mode, 1, direction,
                    "col_%s_%s_r%d_c%d" % (mode, direction, COL_LADDER_ROWS, cols))
            for rows in ROW_LADDER:
                if rows == COL_LADDER_ROWS:
                    continue          # same geometry as the C=128 control
                add(rows, ROW_LADDER_COLS, mode, 1, direction,
                    "row_%s_%s_r%d_c%d" % (mode, direction, rows, ROW_LADDER_COLS))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grid = points()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(grid, indent=2) + chr(10))
    geometries = {(p["mode"], p["direction"], p["real_rows"], p["real_cols"],
                   p["group"]) for p in grid}
    print("%d launches = %d geometries x %d OUTER points"
          % (len(grid), len(geometries), len(OUTERS)))


if __name__ == "__main__":
    main()
