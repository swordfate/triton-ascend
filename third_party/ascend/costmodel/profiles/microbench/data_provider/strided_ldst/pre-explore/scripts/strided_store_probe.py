#!/usr/bin/env python3
"""Single-op shaped strided ``tl.store`` probe for SIMD/SIMT pre-explore.

The kernel contains exactly one shaped ``tl.store``.  There is no load in the
kernel: the value vector is generated from ``tl.arange`` so the store cannot be
DCE'd (GM write is a side effect).  Addresses are
``base + pid*BLOCK*STRIDE + arange(0, BLOCK)*STRIDE``.

The script is meant to be launched under ``msopprof simulator``.  It writes a
launch manifest in launch order; the CAModel OPPROF parser can map each
``OPPROF_*/<kernel>/<index>`` directory back to (mode, stride, block, ...).

Typical use:
  python3 strided_store_probe.py --modes simd simt_only --strides 1 16 256
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def strided_store_kernel(out_ptr, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK) * STRIDE
    value = tl.arange(0, BLOCK).to(tl.float32) + 1.0
    tl.store(out_ptr + pid * BLOCK * STRIDE + offs, value)


def launch_options(mode: str, grid: int, num_warps: int):
    """Return kernel options that force one implementation mode."""
    common = {
        "num_warps": num_warps,
        "superblock_factor": 1,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": 1,
        "auto_simt_scope_mode": "off",
    }
    if mode == "simd":
        return {
            **common,
            "compile_mode": "simd",
            "enable_auto_blockify": False,
        }
    if mode == "simt_only":
        return {
            **common,
            "compile_mode": "simt_only",
            "enable_auto_blockify": True,
        }
    raise ValueError(f"unsupported mode: {mode}")


def save_asm(compiled, out_dir: Path, label: str):
    """Best-effort persistence of compiler IR / binary output for one config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    asm = getattr(compiled, "asm", None) or {}
    for key, value in asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        path = out_dir / f"{label}.{key}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value, encoding="utf-8", errors="replace")
        keys.append({"key": key, "path": path.name, "bytes": len(value)})
    metadata = getattr(compiled, "metadata", None)
    if metadata is not None:
        try:
            (out_dir / f"{label}.metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        except TypeError:
            pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["simd", "simt_only"])
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--num-warps", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=Path("launch_manifest.json"))
    ap.add_argument("--asm-dir", type=Path, default=Path("asm"))
    ap.add_argument("--warmup-only", action="store_true",
                    help="compile + dump asm but do not launch kernels")
    ap.add_argument("--sleep-between-launches", type=float, default=0.0)
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)

    rows = []
    asm_index = {}
    for mode in args.modes:
        opts = launch_options(mode, args.grid, args.num_warps)
        for stride in args.strides:
            # Separate output buffer per (mode, stride) means each case writes
            # fresh GM lines and cannot consume another case's warmed DCache.
            total = args.grid * args.block * max(1, stride) + 4096
            out = torch.zeros(total, dtype=torch.float32).to("npu")

            label = f"{mode}_b{args.block}_s{stride}"
            compiled = strided_store_kernel.warmup(
                out, BLOCK=args.block, STRIDE=stride, grid=(args.grid,), **opts
            )
            asm_index[label] = save_asm(compiled, args.asm_dir, label)
            if args.warmup_only:
                rows.append({
                    "order": len(rows),
                    "mode": mode,
                    "stride": stride,
                    "block": args.block,
                    "grid": args.grid,
                    "num_warps": args.num_warps,
                    "options": {k: v for k, v in opts.items()},
                    "kernel_symbol": getattr(compiled, "name", None),
                    "launched": False,
                    "out_ptr": hex(out.data_ptr()),
                    "out_bytes": int(out.numel()) * out.element_size(),
                    "used_bytes": args.grid * args.block * max(1, stride) * 4,
                })
                continue

            t0 = time.time()
            strided_store_kernel[(args.grid,)](
                out, BLOCK=args.block, STRIDE=stride, **opts
            )
            torch_npu.npu.synchronize()
            elapsed = time.time() - t0
            rows.append({
                "order": len(rows),
                "mode": mode,
                "stride": stride,
                "block": args.block,
                "grid": args.grid,
                "num_warps": args.num_warps,
                "options": {k: v for k, v in opts.items()},
                "kernel_symbol": getattr(compiled, "name", None),
                "launched": True,
                "host_launch_seconds": elapsed,
                "out_ptr": hex(out.data_ptr()),
                "out_bytes": int(out.numel()) * out.element_size(),
                "used_bytes": args.grid * args.block * max(1, stride) * 4,
            })
            if args.sleep_between_launches > 0:
                time.sleep(args.sleep_between_launches)

    payload = {
        "probe": "strided_store_probe",
        "block": args.block,
        "grid": args.grid,
        "num_warps": args.num_warps,
        "cases": rows,
        "asm_index": asm_index,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"WROTE {args.manifest} ({len(rows)} cases)", flush=True)
    if args.warmup_only:
        print("WARMUP_ONLY: no kernels launched", flush=True)
    else:
        print("LAUNCHED_ALL", flush=True)


if __name__ == "__main__":
    sys.exit(main())
