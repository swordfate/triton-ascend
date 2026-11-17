#!/usr/bin/env python3
"""E17h: physics-driven SIMT load calibration (unified-set phase 3b).

Hardware facts (04-ascend950-architecture-compendium):
- SIMT data path: GM -> SIMT DCache (128B lines, 32~128KB per AIV)
  -> thread registers; Warp Scheduler issues per-warp instructions.
- One SIMT warp instruction moves 32 x elt: fp32=128B (=1 DCache line,
  2x64B sub-units), fp16=64B (=0.5 line, 1 sub-unit).
- L2 = 128MB shared; SIMT register file 128KB/AIV (127 regs/thread at
  <=256 threads).

Discriminating axes (each isolates one physical factor):
- dtype fp32 vs fp16 at EQUAL BYTES: fp16 doubles warp instructions
  while keeping lines constant -> separates issue-bound vs line-bound.
- residency: dcache_hot (fixed block re-read per iteration, per-AIV
  working set = 16KB <= DCache) vs l2_stream (streaming, pool < L2)
  vs dram_stream (rotating pool > L2) -> line cost per source.

Cells: {fp32,fp16} x 16KB x {hot,l2,dram} + {fp32,fp16} x 4KB x hot.
Consumption: simple acc += load chain (no reduction); register budget
safe (4096 fp32 = 32 regs/thread; 8192 fp16 = 64 halves/thread).
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"
COMMON = """
@triton.jit
def _clock():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l,~{memory}", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""


def gen_kernel(tag, dtype, mode):
    """mode: hot (fixed block, DCache-resident) | stream (per-iteration
    fresh lines).  BLOCK is the element count; dtype fp32->fp16 switches
    elt width."""
    if mode == "hot":
        pre = "    p = pid * BLOCK + lane\n"
        loop_addr = ""
    else:
        pre = "    seg = ITERS * BLOCK\n"
        loop_addr = "        p = pid * seg + step * BLOCK + lane\n"
    dt = "tl.float32" if dtype == "fp32" else "tl.float16"
    return """
@triton.jit
def e17h_%s(src, out, tick_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
%s    acc = tl.zeros((BLOCK,), dtype=%s)
    t0 = _clock()
    for step in range(ITERS):
%s        acc += tl.load(src + p)
    t1 = _clock()
    tl.store(out + pid * BLOCK + lane, acc)
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (tag, pre, dt, loop_addr)


PATTERN = (
    r'tt\.elementwise_inline_asm "MOV \$0, SYS_CNT"'
    r' \{constraints = "=l,~\{memory\}", '
    r'packed_element = 1 : i32, pure = false\} -> i64'
)


def replace_markers(text):
    patched, count = re.subn(PATTERN, "ascend_dpx.clock64", text)
    if count < 1:
        for line in text.splitlines():
            if "elementwise_inline_asm" in line:
                print("TTIR ASM LINE:", line.strip()[:300])
        raise ValueError("clock marker not matched (count=%d)" % count)
    return patched


@contextmanager
def clock64_compilation():
    original_run = subprocess.run

    def run(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) > 1
                and Path(command[0]).name == "bishengir-compile"):
            source = Path(command[1])
            source.write_text(replace_markers(source.read_text()))
        return original_run(command, *args, **kwargs)

    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e17h_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    cells = []
    for dt, elems16k in (("fp32", 4096), ("fp16", 8192)):
        for mode in ("hot", "l2", "dram"):
            cells.append((dt, 16 * 1024, mode))
        cells.append((dt, 4 * 1024, "hot"))

    out = {"meta": {"grid": 56, "launches": 30, "ks": [16, 32],
                    "method": "clock64 slope K16/K32, simt_only; "
                              "hot=fixed block (DCache-resident), "
                              "l2=streaming pool<128MB, dram=rotating "
                              "pool>128MB"},
           "cells": {}}
    out_path = Path(args.out)

    for dt, bytes_per_iter, mode in cells:
        elems = bytes_per_iter // (2 if dt == "fp16" else 4)
        key = "%s_%dB_%s" % (dt, bytes_per_iter, mode)
        t0 = time.time()
        try:
            text = HEADER + COMMON + gen_kernel(key.replace(".", "_"),
                                                dt, mode)
            gen_path = Path("/root/work/explore_contig/_gen_e17h.py")
            gen_path.write_text(text, encoding="utf-8")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_gen_e17h", str(gen_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            seg_max = 56 * 32 * elems
            if mode == "dram":
                copies = max(8, (300 * 1024 * 1024) //
                             (seg_max * (2 if dt == "fp16" else 4)))
            else:
                copies = 1
            total = seg_max * copies
            if dt == "fp16":
                pool = torch.randn((total,), dtype=torch.float16,
                                   device="npu")
            else:
                pool = torch.randn((total,), dtype=torch.float32,
                                   device="npu")
            outbuf = torch.zeros((56 * elems,), dtype=pool.dtype,
                                 device="npu")
            tick = torch.zeros((112,), dtype=torch.int64, device="npu")
            kn = getattr(mod, "e17h_%s" % key.replace(".", "_"))

            def launch(iters, rot=0):
                buf = pool
                if mode == "dram":
                    buf = pool[(rot % copies) * seg_max:
                               (rot % copies) * seg_max + seg_max]
                with clock64_compilation():
                    kn[(56,)](buf, outbuf, tick, ITERS=iters, BLOCK=elems,
                              num_warps=4, compile_mode="simt_only",
                              auto_simt_scope_mode="off",
                              superblock_factor=1)

            launch(16)
            torch.npu.synchronize()

            def measure(iters):
                all_ns = []
                for i in range(30):
                    launch(iters, rot=i)
                    torch.npu.synchronize()
                    t = tick.cpu().reshape(56, 2)
                    ns = (t[:, 1] - t[:, 0]) * 1.00008
                    all_ns.append(statistics.median(ns.tolist()))
                return statistics.median(all_ns)

            w16 = measure(16)
            w32 = measure(32)
            slope = (w32 - w16) / 16
            instrs = elems // 32
            out["cells"][key] = {
                "whole16": round(w16, 1), "whole32": round(w32, 1),
                "slope": round(slope, 1),
                "per_warp_instr": round(slope / instrs, 3),
                "per_128b_line": round(slope / (bytes_per_iter / 128), 3)}
            print("%-22s slope=%8.1f  /instr=%6.2f  /line=%6.2f (%.0fs)"
                  % (key, slope, out["cells"][key]["per_warp_instr"],
                     out["cells"][key]["per_128b_line"], time.time() - t0),
                  flush=True)
            del pool
            torch.npu.empty_cache()
        except Exception as exc:  # noqa: BLE001
            out["cells"][key] = {"error": repr(exc)[:400]}
            print("%s ERROR %r" % (key, exc), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
