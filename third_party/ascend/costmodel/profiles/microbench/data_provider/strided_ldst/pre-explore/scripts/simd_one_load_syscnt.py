#!/usr/bin/env python3
"""SIMD-only single strided load latency via Triton inline SYS_CNT.

The second timestamp asm takes the reduced value as an input operand, so the
`MOV ..., SYS_CNT` cannot be scheduled before the load/reduce chain leading to
that value.  A structurally matched no-load kernel measures the same reduce and
timestamp overhead; the difference isolates the GM load contribution.

Run on the real board; SIMD compile mode only (SIMT rejects the inline asm).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

SYS_CNT_MHZ = 999.92  # board counter, cross-checked against Event slope (period ~1.00008 ns)


@triton.jit
def _get_sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _get_sys_cnt_dep(dep):
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT\nADD.s64 $0, $0, $1",
        constraints="=l,l", args=[dep], dtype=tl.int64,
        is_pure=False, pack=1)


@triton.jit
def load_kernel(x_ptr, out_ptr, t0_ptr, t1_ptr,
                BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE
    t0 = _get_sys_cnt()
    v = tl.load(x_ptr + offs)
    s = tl.sum(v, axis=0)
    # The syscnt asm is side-effecting, so keep it ordered after the reduce in
    # the TTIR.  (A data-dependent asm input is rejected by the current
    # ConvertTritonIRToLinalgIR path.)
    t1 = _get_sys_cnt()
    o = tl.arange(0, 1)
    tl.store(out_ptr + o, s)
    tl.store(t0_ptr + o, t0)
    tl.store(t1_ptr + o, t1)


@triton.jit
def noload_kernel(out_ptr, t0_ptr, t1_ptr, BLOCK: tl.constexpr):
    v = tl.arange(0, BLOCK).to(tl.float32)
    t0 = _get_sys_cnt()
    s = tl.sum(v, axis=0)
    t1 = _get_sys_cnt()
    o = tl.arange(0, 1)
    tl.store(out_ptr + o, s)
    tl.store(t0_ptr + o, t0)
    tl.store(t1_ptr + o, t1)


def simd_opts():
    return {"num_warps": 1, "logical_program_count_hint": 1,
            "physical_vector_core_count_hint": 1, "superblock_factor": 1,
            "auto_simt_scope_mode": "off", "compile_mode": "simd",
            "enable_auto_blockify": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", type=int, default=[32])
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--nbuf", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("simd_one_load_results"))
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    opts = simd_opts()
    rows = []
    for block in args.blocks:
        out = torch.zeros(1, dtype=torch.float32, device="npu")
        t0 = torch.zeros(1, dtype=torch.int64, device="npu")
        t1 = torch.zeros(1, dtype=torch.int64, device="npu")
        noload_kernel.warmup(out, t0, t1, BLOCK=block, grid=(1, ), **opts)
        noload_kernel[(1, )](out, t0, t1, BLOCK=block, **opts)
        torch.npu.synchronize()
        base = int(t1.cpu().item()) - int(t0.cpu().item())
        for stride in args.strides:
            total = args.nbuf * block * max(stride, 1) + 4096
            xs = [(torch.arange(total, dtype=torch.float32) % 7.0).contiguous().to("npu")
                  for _ in range(min(args.nbuf, args.reps))]
            load_kernel.warmup(xs[0], out, t0, t1, BLOCK=block, STRIDE=stride,
                               grid=(1, ), **opts)
            vals_load, vals_noload = [], []
            for x in xs:
                load_kernel[(1, )](x, out, t0, t1, BLOCK=block, STRIDE=stride, **opts)
                torch.npu.synchronize()
                vals_load.append(int(t1.cpu().item()) - int(t0.cpu().item()))
                noload_kernel[(1, )](out, t0, t1, BLOCK=block, **opts)
                torch.npu.synchronize()
                vals_noload.append(int(t1.cpu().item()) - int(t0.cpu().item()))
            d = [a - b for a, b in zip(vals_load, vals_noload)]
            row = {"block": block, "stride": stride,
                   "load_cycles_median": statistics.median(vals_load),
                   "noload_cycles_median": statistics.median(vals_noload),
                   "delta_median": statistics.median(d),
                   "delta_min": min(d), "delta_max": max(d),
                   "base_noload_cycles": base}
            rows.append(row)
            print(f'simd b{block} s{stride}: load={row["load_cycles_median"]:.0f} '
                  f'noload={row["noload_cycles_median"]:.0f} delta={row["delta_median"]:.0f} '
                  f'cyc ({row["delta_median"]/SYS_CNT_MHZ*1000:.1f} ns)')
    (args.out / "simd_one_load_results.json").write_text(
        json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())
