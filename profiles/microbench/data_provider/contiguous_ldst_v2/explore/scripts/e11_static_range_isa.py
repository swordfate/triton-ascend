"""E11: does a constexpr-trip-count `for i in range(8)` stay a loop in the
final ISA, or is it unrolled like tl.static_range?  (taxonomy: which kernels
fall into the burst/group cell vs the loop cell.)

Three kernels, identical bodies (8 x 4KB load+store, streaming):
  k_static  - tl.static_range(8)      (known: frontend unrolls, no scf.for in TTIR)
  k_cexpr   - for i in range(8), 8 is a Python int captured as constexpr
  k_runtime - for i in range(iters), iters passed at runtime = 8
Dump every available asm key for each and scan for loop remnants
(scf.for in IR keys; `for`/`br`/label patterns in the CCE 'source' key).

Usage (compile+launch once on any healthy card):
  bash run_exp.sh e11_static_range_isa.py 2 e11_c2.jsonl --skip-health
"""
import argparse
import json

import torch
import triton
import triton.language as tl

from common import JsonlOut, add_common_args, std_opts

BLOCK = 1024


@triton.jit
def k_static(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for j in tl.static_range(8):
        x = tl.load(src + j * BLOCK + lane)
        tl.store(dst + j * BLOCK + lane, x + 1)


@triton.jit
def k_cexpr(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for i in range(8):
        x = tl.load(src + i * BLOCK + lane)
        tl.store(dst + i * BLOCK + lane, x + 1)


@triton.jit(do_not_specialize=["iters"])
def k_runtime(src, dst, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for i in range(iters):
        x = tl.load(src + i * BLOCK + lane)
        tl.store(dst + i * BLOCK + lane, x + 1)


def scan(text):
    """Count loop-ish markers in an ISA/IR dump."""
    return {
        "scf.for": text.count("scf.for"),
        "cf.br": text.count("cf.br"),
        "for(": text.count("for(") + text.count("for ("),
        "#pragma": text.count("#pragma"),
        "br ": text.count("\nbr ") + text.count("\n br "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default=["simd"], nargs="+")
    args = ap.parse_args()
    out = JsonlOut(args.out)

    src = torch.randn(BLOCK * 16, device="npu")
    dst = torch.zeros(BLOCK * 16, device="npu")

    for mode in args.modes:
        opts = std_opts(mode, 4)
        handles = {
            "static": k_static[(1,)](src, dst, BLOCK, **opts),
            "cexpr": k_cexpr[(1,)](src, dst, BLOCK, **opts),
            "runtime": k_runtime[(1,)](src, dst, 8, BLOCK, **opts),
        }
        for kname, h in handles.items():
            keys = sorted(h.asm.keys())
            rec = {"exp": "e11", "kernel": kname, "mode": mode,
                   "asm_keys": keys, "scans": {}}
            for key in keys:
                text = h.asm.get(key)
                if isinstance(text, str) and len(text) > 0:
                    rec["scans"][key] = scan(text)
            out.write(**rec)
            print(kname, "keys:", keys, flush=True)
            for key, sc in rec["scans"].items():
                print("   %-10s %s" % (key, sc), flush=True)
    print("E11 DONE", flush=True)


if __name__ == "__main__":
    main()
