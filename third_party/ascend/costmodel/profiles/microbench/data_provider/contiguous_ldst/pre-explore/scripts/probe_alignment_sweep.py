#!/usr/bin/env python3
"""Timing probe for rank-two contiguous tiles whose trailing extent is a tail.

This is the join between the two halves of the work: the IR reconnaissance
(probe_alignment_ir.py) showed that the lowered copy changes shape when the
trailing extent stops being a power of two, and this probe prices that change.

Shape convention.  The arange extents are RPAD x CPAD, both powers of two, so
the tile addresses a padded region; the real extents (real_rows, real_cols) are
runtime scalars, which makes the predicate dynamic and impossible to fold away.
ROW_STRIDE is the real row stride in elements and is a compile-time constant so
the backend can reason about the address pattern.

Three normalised forms are compared:
    aligned    real_cols == CPAD and real_rows == RPAD   (mask folds away)
    col_tail   real_cols <  CPAD                        (padded UB row)
    row_tail   real_rows <  RPAD, real_cols == CPAD      (no padding, extra ops)

Per-operation cost is the OUTER-ladder slope, exactly as in
probe_contiguous_sweep.py, so the two datasets are directly comparable.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

DTYPES = {"fp32": (torch.float32, tl.float32, 4, float("-inf")),
          "fp16": (torch.float16, tl.float16, 2, -65504.0),
          "int8": (torch.int8, tl.int8, 1, -128),
          "int64": (torch.int64, tl.int64, 8, -(2 ** 62))}


@triton.jit
def align_sweep_kernel(source, output, real_rows, real_cols,
                       RPAD: tl.constexpr, CPAD: tl.constexpr,
                       ROW_STRIDE: tl.constexpr, GROUPS: tl.constexpr,
                       GROUP: tl.constexpr, DIRECTION: tl.constexpr,
                       DTYPE: tl.constexpr, NEG_INF: tl.constexpr, OUTER,
                       MASKED: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, RPAD)[:, None]
    cols = tl.arange(0, CPAD)[None, :]
    tile = rows * ROW_STRIDE + cols
    mask = (rows < real_rows) & (cols < real_cols) if MASKED else None
    # Each operation owns ROW_STRIDE * RPAD addresses, padded or not.
    stride = ROW_STRIDE * RPAD
    window = pid * GROUPS * GROUP * stride
    if DIRECTION == 0:
        acc0 = tl.full((RPAD, CPAD), NEG_INF, DTYPE)
        acc1 = tl.full((RPAD, CPAD), NEG_INF, DTYPE)
        for step in range(OUTER):
            slot = step % GROUPS
            for j in tl.static_range(GROUP):
                offset = window + (slot * GROUP + j) * stride + tile
                if MASKED:
                    loaded = tl.load(source + offset, mask=mask, other=NEG_INF)
                else:
                    loaded = tl.load(source + offset)
                if j % 2 == 0:
                    acc0 = tl.maximum(acc0, loaded)
                else:
                    acc1 = tl.maximum(acc1, loaded)
        if MASKED:
            tl.store(output + pid * stride + tile, tl.maximum(acc0, acc1),
                     mask=mask)
        else:
            tl.store(output + pid * stride + tile, tl.maximum(acc0, acc1))
    else:
        value = (tl.load(source + pid * stride + tile, mask=mask, other=0)
                 if MASKED else tl.load(source + pid * stride + tile))
        for step in range(OUTER):
            slot = step % GROUPS
            for j in tl.static_range(GROUP):
                offset = window + (slot * GROUP + j) * stride + tile
                if MASKED:
                    tl.store(output + offset, value, mask=mask)
                else:
                    tl.store(output + offset, value)


def load_average():
    try:
        with open("/proc/loadavg") as stream:
            return [float(x) for x in stream.read().split()[:3]]
    except OSError:
        return None


def npubin_digest(cache_dir, stem):
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    return sorted({hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in root.rglob(stem + ".npubin")})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("simd", "simt_only"), required=True)
    parser.add_argument("--direction", choices=("load", "store"), required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    parser.add_argument("--real-rows", type=int, required=True)
    parser.add_argument("--real-cols", type=int, required=True)
    parser.add_argument("--row-stride", type=int, default=0,
                        help="elements between rows; 0 -> packed (= real-cols)")
    parser.add_argument("--num-warps", type=int, default=4,
                        help="warps per logical program; the concurrency sweep "
                             "crosses this with factor and unroll depth, so the "
                             "alignment grid has to accept it too")
    parser.add_argument("--masked", type=int, choices=(0, 1), default=1,
                        help="0 drops the extent predicate, which is only legal "
                             "when the real extents equal the padded ones; it is "
                             "the control that separates the predicate cost from "
                             "the tail cost on a two-dimensional tile")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--outer", type=int, default=128)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--logical-programs", type=int, default=0)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=8)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.mode == "simd" and args.factor != 1:
        parser.error("SIMD has no SuperBlock factor; use --factor=1")

    def pad(value):
        return 1 << max(0, (value - 1).bit_length())

    rpad, cpad = pad(args.real_rows), pad(args.real_cols)
    if not args.masked and (rpad != args.real_rows or cpad != args.real_cols):
        parser.error("--masked 0 requires the real extents to be powers of two: "
                     "without the predicate the tile would read past its rows")
    row_stride = args.row_stride or args.real_cols
    torch_dtype, tl_dtype, element_bytes, neg_inf = DTYPES[args.dtype]
    stride = row_stride * rpad
    torch_dtype = DTYPES[args.dtype][0]

    vector_cores = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    programs = args.logical_programs or vector_cores * 4
    programs //= args.grid
    window = programs * args.groups * args.group * stride
    footprint = window * element_bytes
    if footprint > 64 * 1024 * 1024:
        parser.error("footprint %.1f MB over the 64 MB budget" % (footprint / 1048576))
    ub_bytes = 2 * stride * element_bytes
    if ub_bytes > 200 * 1024:
        parser.error("working set %d B over the 200 KiB UB budget" % ub_bytes)

    torch.manual_seed(23)
    if args.direction == "store":
        source_elements, output_elements = programs * stride, window
    else:
        source_elements, output_elements = window, programs * stride
    source = torch.randint(0, 8, (source_elements, ), dtype=torch_dtype, device="npu")
    output = torch.full((output_elements, ), -7, dtype=torch_dtype, device="npu")

    options = {"compile_mode": args.mode, "num_warps": args.num_warps}
    if args.mode == "simt_only":
        options["superblock_factor"] = args.factor
    direction = 0 if args.direction == "load" else 1

    def launch(outer):
        return align_sweep_kernel[(programs, )](
            source, output, args.real_rows, args.real_cols, rpad, cpad,
            row_stride, args.groups, args.group, direction, tl_dtype, neg_inf,
            outer, args.masked, **options)

    binary = launch(args.outer)
    torch.npu.synchronize()

    correct = None
    span = args.groups * args.group
    # A packed tail does not cover rpad*cpad contiguous elements (the tile spans
    # (real_rows-1)*row_stride + real_cols at most), so the result cannot be
    # checked by reshaping.  Verify by gathering with the same address map,
    # on the CPU, over the first few programs and every lane of them.
    import numpy as np
    checked = min(programs, 16)
    rows_np = np.arange(rpad)[:, None]
    cols_np = np.arange(cpad)[None, :]
    tile_np = (rows_np * row_stride + cols_np).astype(np.int64)
    live_np = (rows_np < args.real_rows) & (cols_np < args.real_cols)
    # The kernel maps program p to a window of GROUPS*GROUP operations for the
    # source and to one operation-sized tile for the result (and the other way
    # round for a store), so the two bases differ by the span.
    src_take = checked * span * stride if args.direction == "load" else checked * stride
    out_take = checked * stride if args.direction == "load" else checked * span * stride
    src_np = source[:src_take].cpu().numpy()
    out_np = output[:out_take].cpu().numpy()
    for program in range(checked):
        src_base = program * span * stride if args.direction == "load" else program * stride
        out_base = program * stride if args.direction == "load" else program * span * stride
        # Lanes outside the real extent compute an address that may sit one past
        # the window (a packed tail's last padded column), so clamp before
        # gathering and only compare the live lanes.
        tile_safe = np.minimum(tile_np, stride - 1)
        if args.direction == "load":
            expected = np.full((rpad, cpad), neg_inf, dtype=src_np.dtype)
            for slot in range(span):
                values = src_np[src_base + slot * stride + tile_safe]
                expected = np.maximum(expected, values)
        else:
            expected = src_np[src_base + tile_safe]
        got = out_np[out_base + tile_safe]
        if not np.array_equal(got[live_np], expected[live_np]):
            raise AssertionError("mismatch in program %d" % program)
    correct = True

    for _ in range(args.warmup):
        launch(args.outer)
    torch.npu.synchronize()

    trace_dir = args.manifest.parent / ("trace_" + str(os.getpid()))
    load_before = load_average()
    started = time.time()
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=2, active=args.active),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as profiler:
        for _ in range(args.active + 3):
            launch(args.outer)
            profiler.step()
    torch.npu.synchronize()
    load_after = load_average()

    samples = []
    for path in sorted(trace_dir.rglob("kernel_details.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                # The trace can carry rows for other kernels (setup, memsets);
                # keep only ours, otherwise a 0.2 us row destroys the median.
                if not row.get("Duration(us)"):
                    continue
                if "align_sweep_kernel" not in (row.get("Name") or ""):
                    continue
                samples.append(float(row["Duration(us)"]))
    samples = samples[:args.active]
    if len(samples) < 5:
        raise RuntimeError("expected >=5 samples, got %d" % len(samples))

    cache_dir = os.environ.get("TRITON_CACHE_DIR",
                               str(Path.home() / ".triton" / "cache"))
    real_bytes = args.real_rows * args.real_cols * element_bytes
    record = {
        "mode": args.mode, "direction": args.direction, "dtype": args.dtype,
        "factor": args.factor, "masked": args.masked,
        "num_warps": args.num_warps,
        "real_rows": args.real_rows, "real_cols": args.real_cols,
        "row_stride_elements": row_stride,
        "rows_power_of_two": (args.real_rows & (args.real_rows - 1)) == 0,
        "cols_power_of_two": (args.real_cols & (args.real_cols - 1)) == 0,
        "packed": row_stride == args.real_cols,
        "row_bytes": args.real_cols * element_bytes,
        "row_bytes_mod_32": (args.real_cols * element_bytes) % 32,
        "row_bytes_mod_128": (args.real_cols * element_bytes) % 128,
        "real_bytes_per_operation": real_bytes,
        "real_elements_per_operation": args.real_rows * args.real_cols,
        "span_bytes_per_operation": stride * element_bytes,
        "groups": args.groups, "group": args.group, "outer": args.outer,
        "operations_per_program": args.outer * args.group,
        "logical_programs": programs, "grid": args.grid,
        "vector_cores": vector_cores,
        "physical_groups": math.ceil(programs / args.factor),
        "waves": math.ceil(math.ceil(programs / args.factor) / vector_cores),
        "footprint_bytes": footprint,
        "footprint_mb": round(footprint / 1048576, 4),
        "timer": "profiler.Duration",
        "sample_count": len(samples),
        "duration_us_min": min(samples),
        "duration_us_median": statistics.median(samples),
        "duration_us_mean": statistics.mean(samples),
        "spread": (max(samples) - min(samples)) / statistics.median(samples),
        "samples_us": samples,
        "load_before": load_before, "load_after": load_after,
        "wall_s": round(time.time() - started, 2),
        "correct": correct,
        "npubin_sha256": npubin_digest(cache_dir, "align_sweep_kernel"),
        "tag": args.tag, "repeat_index": args.repeat_index,
        "repeat_total": args.repeat,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": os.uname().nodename,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + chr(10))
    print(json.dumps({k: record[k] for k in (
        "mode", "direction", "dtype", "real_rows", "real_cols",
        "rows_power_of_two", "cols_power_of_two", "outer",
        "duration_us_min", "spread", "correct")}), flush=True)


if __name__ == "__main__":
    main()
