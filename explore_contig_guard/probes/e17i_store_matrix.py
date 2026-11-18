#!/usr/bin/env python3
"""E17i: SIMT store-side per-line cost matrix (completes E17h2).

E17h2 measured the load side: perLineCost(bytesPerIter, bits, residency)
with a 4KB issue penalty (11.3 vs 6.0 at 16KB, L2) and a 32KB jump.
Stores measured flat 6.1 ns/line (fp32, L2, all sizes) in E17g - this
probe fills the remaining store cells: {fp32, fp16} x {4,8,16,32}KB x
{l2, dram}, store-only bodies (WAW backpressure makes the window
capture store completion; no sink needed - E17g evidence).
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


def gen_kernel(tag, dtype):
    dt = "tl.float32" if dtype == "fp32" else "tl.float16"
    return """
@triton.jit
def e17i_%s(dst, tick_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = ITERS * BLOCK
    v = tl.zeros((BLOCK,), dtype=%s) + 1.0
    t0 = _clock()
    for step in range(ITERS):
        p = pid * seg + step * BLOCK + lane
        tl.store(dst + p, v)
    t1 = _clock()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (tag, dt)


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
                                     "e17i_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    cells = [(dt, kb * 1024, mode)
             for dt in ("fp32", "fp16")
             for kb in (4, 8, 16, 32)
             for mode in ("l2", "dram")]

    out = {"meta": {"grid": 56, "launches": 30, "ks": [16, 32],
                    "method": "clock64 slope K16/K32, simt_only, "
                              "store-only; l2=pool<128MB, "
                              "dram=rotating pool>128MB"},
           "cells": {}}
    out_path = Path(args.out)

    for dt, bytes_per_iter, mode in cells:
        elems = bytes_per_iter // (2 if dt == "fp16" else 4)
        key = "%s_%dB_%s" % (dt, bytes_per_iter, mode)
        t0 = time.time()
        try:
            text = HEADER + COMMON + gen_kernel(key.replace(".", "_"), dt)
            gen_path = Path("/root/work/explore_contig/_gen_e17i.py")
            gen_path.write_text(text, encoding="utf-8")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_gen_e17i", str(gen_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            seg_max = 56 * 32 * elems
            if mode == "dram":
                copies = max(8, (300 * 1024 * 1024) //
                             (seg_max * (2 if dt == "fp16" else 4)))
            else:
                copies = 1
            total = seg_max * copies
            pool = torch.zeros((total,), dtype=(torch.float16 if dt ==
                                                "fp16" else torch.float32),
                               device="npu")
            tick = torch.zeros((112,), dtype=torch.int64, device="npu")
            kn = getattr(mod, "e17i_%s" % key.replace(".", "_"))

            def launch(iters, rot=0):
                buf = pool
                if mode == "dram":
                    buf = pool[(rot % copies) * seg_max:
                               (rot % copies) * seg_max + seg_max]
                with clock64_compilation():
                    kn[(56,)](buf, tick, ITERS=iters, BLOCK=elems,
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
            lines = bytes_per_iter / 128.0
            out["cells"][key] = {
                "whole16": round(w16, 1), "whole32": round(w32, 1),
                "slope": round(slope, 1),
                "per_128b_line": round(slope / lines, 3)}
            print("%-22s slope=%8.1f  /line=%6.2f (%.0fs)"
                  % (key, slope, out["cells"][key]["per_128b_line"],
                     time.time() - t0), flush=True)
            del pool
            torch.npu.empty_cache()
        except Exception as exc:  # noqa: BLE001
            out["cells"][key] = {"error": repr(exc)[:400]}
            print("%s ERROR %r" % (key, exc), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
