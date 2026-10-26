#!/usr/bin/env python3
"""Compile-only reconnaissance for the non-aligned / row-tail question.

The question this answers is structural, not a timing question: given a rank-2
tile whose trailing extent is not a multiple of the access granularity, does
the backend flatten the tile into one contiguous run, or does it keep per-row
bursts (and possibly a padded row)?  The answer decides whether the model may
price a row-tail tile from bytes alone or needs a separate family, so it has to
come from the lowered IR, not from a fit.

Shape convention: both axes are given as a real extent plus a padded arange
extent, because tl.arange can only build power-of-two lengths - which is
exactly the shape a real tail takes.  ROW_STRIDE is a compile-time constant so
the compiler can reason about it; a runtime stride would make the alignment
unprovable and the experiment would be about a different question.

This script never launches the kernel on the device.  It compiles, dumps ttir /
ttadapter / npuir, and reports what it found.
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

DTYPES = {"fp32": (torch.float32, tl.float32, 4),
          "fp16": (torch.float16, tl.float16, 2),
          "int8": (torch.int8, tl.int8, 1),
          "int64": (torch.int64, tl.int64, 8)}


@triton.jit
def align_kernel(source, output, R: tl.constexpr, C: tl.constexpr,
                 RPAD: tl.constexpr, CPAD: tl.constexpr,
                 ROW_STRIDE: tl.constexpr, DTYPE: tl.constexpr,
                 MASKED: tl.constexpr, DIRECTION: tl.constexpr):
    rows = tl.arange(0, RPAD)[:, None]
    cols = tl.arange(0, CPAD)[None, :]
    offsets = rows * ROW_STRIDE + cols
    mask = (rows < R) & (cols < C)
    # DIRECTION separates the two copies.  With only the fused load+store form a
    # dump shows a single memref.copy and it is easy to mistake which side it
    # belongs to, which is exactly the question "is the padded row a load thing
    # or a store thing".
    if DIRECTION == 0:                       # load only
        if MASKED:
            value = tl.load(source + offsets, mask=mask, other=0)
        else:
            value = tl.load(source + offsets)
        acc = tl.max(value)
        tl.store(output, acc)
    elif DIRECTION == 1:                     # store only
        if MASKED:
            tl.store(output + offsets, 0.0, mask=mask)
        else:
            tl.store(output + offsets, 0.0)
    else:                                    # both, the original shape
        if MASKED:
            value = tl.load(source + offsets, mask=mask, other=0)
            tl.store(output + offsets, value, mask=mask)
        else:
            value = tl.load(source + offsets)
            tl.store(output + offsets, value)


def ir_features(text):
    """Pull the memory-op structure out of a lowered IR dump."""
    features = {}
    for name in ("hivm.hir.copy", "hivm.hir.load", "hivm.hir.store",
                 "hivm.hir.pad", "hivm.hir.select", "hivm.hir.vsel",
                 "memref.cast", "hivm.hir.reinterpret_cast", "linalg.copy",
                 "hivm.hir.copy_1d", "hivm.hir.load_1d"):
        count = len(re.findall(re.escape(name), text))
        if count:
            features[name] = count
    # burst descriptors show up as named operands in the lowerings
    for key in ("nBurst", "lenBurst", "srcStride", "dstStride", "burst_cnt",
                "burst_len", "left_padding", "right_padding", "pad"):
        if key.lower() in text.lower():
            features["has:" + key] = True
    features["strided_layouts"] = len(re.findall(r"strided<\[", text))
    features["identity_layouts"] = len(re.findall(r"strided<\[1\]", text))
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True,
                        help="trailing extent in elements")
    parser.add_argument("--row-stride", type=int, default=0,
                        help="elements between rows; 0 -> packed (=cols)")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    parser.add_argument("--mode", choices=("simd", "simt_only"), default="simt_only")
    parser.add_argument("--direction", choices=("load", "store", "both"),
                        default="both",
                        help="which copy to lower; 'both' reproduces the original "
                             "fused kernel, the single-direction forms isolate it")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--masked", type=int, choices=(0, 1), default=1,
                        help="keep the extent mask (1) or drop it (0); dropping it "
                             "separates an unaligned extent from a predicate the "
                             "compiler cannot fold")
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    def pad(value):
        return 1 << max(0, (value - 1).bit_length())

    rpad, cpad = pad(args.rows), pad(args.cols)
    row_stride = args.row_stride or args.cols
    torch_dtype, tl_dtype, element_bytes = DTYPES[args.dtype]

    elements = max(rpad * cpad, (args.rows - 1) * row_stride + args.cols)
    torch.manual_seed(23)
    source = torch.randint(0, 8, (elements + 1, ), dtype=torch_dtype, device="npu")
    output = torch.zeros(elements + 1, dtype=torch_dtype, device="npu")

    options = {"compile_mode": args.mode, "num_warps": 4}
    if args.mode == "simt_only":
        options["superblock_factor"] = args.factor
    direction_of = {"load": 0, "store": 1, "both": 2}
    binary = align_kernel[(1, )](
        source, output, args.rows, args.cols, rpad, cpad, row_stride, tl_dtype,
        args.masked, direction_of[args.direction], **options)
    torch.npu.synchronize()

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    for key in ("ttir", "ttadapter", "npuir"):
        text = binary.asm.get(key)
        if isinstance(text, str):
            (args.dump_dir / ("kernel." + key)).write_text(text)
            texts[key] = text
    # This backend does not expose npuir through asm; it does write it next to
    # the device binary in the compile cache, and that is where the DMA burst
    # descriptors live.
    cache_dir = os.environ.get("TRITON_CACHE_DIR",
                               str(Path.home() / ".triton" / "cache"))
    npuirs = sorted(Path(cache_dir).rglob("align_kernel.npuir"),
                    key=lambda item: item.stat().st_mtime)
    if npuirs:
        text = npuirs[-1].read_text()
        (args.dump_dir / "kernel.npuir").write_text(text)
        texts["npuir"] = text

    row_bytes = args.cols * element_bytes
    report = {"rows": args.rows, "cols": args.cols,
              "row_bytes": row_bytes,
              "row_bytes_mod_32": row_bytes % 32,
              "row_bytes_mod_64": row_bytes % 64,
              "row_bytes_mod_128": row_bytes % 128,
              "row_stride_elements": row_stride,
              "packed": row_stride == args.cols,
              "dtype": args.dtype, "mode": args.mode, "factor": args.factor,
              "masked": args.masked, "direction": args.direction,
              "tile_bytes": args.rows * args.cols * element_bytes,
              "available_ir": sorted(texts),
              "features": {k: ir_features(v) for k, v in texts.items()}}
    print(json.dumps(report, indent=2)[:4000])
    if args.report:
        prior = []
        if args.report.exists():
            prior = json.loads(args.report.read_text())
        prior.append(report)
        args.report.write_text(json.dumps(prior, indent=2) + chr(10))


if __name__ == "__main__":
    main()
