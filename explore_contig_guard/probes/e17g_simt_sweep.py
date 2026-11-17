#!/usr/bin/env python3
"""E17g: SIMT loop-body envelope calibration (unified-set phase 3).

Measures the simt per-iteration cost of loop bodies as f(U loads x
S stores x per-access bytes) with the clock64 method (Q4 guide 6.2:
SYS_CNT markers in TTIR replaced by ascend_dpx.clock64 via a
bishengir-compile interception hook; 1 tick ~ 1 ns).

Cells: (U,S) in {1,2,4}^2 x per-access bytes {4KB,16KB}, plus
load-only (S=0) and store-only (U=0) references, K=16/32 constexpr
slope, grid=56, hot residency.  Fit target: perIter = max(load,store)
+ serial(S) shape for the simt tier-2 envelope.
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


def gen_body(tag, u, s):
    """u load chains + s store streams per iteration, K=ITERS constexpr.
    A tiny sink store + debug_barrier inside the window force load
    completion for s=0 bodies (loads are issued async in simt; without a
    consumer the window closes before data returns - Q4 guide 6.1)."""
    accs = "\n".join("    a%d = tl.zeros((BLOCK,), dtype=tl.float32)" % i
                     for i in range(u))
    loads = "\n".join(
        "        a%d += tl.load(src + p + %d * seg)" % (i, i)
        for i in range(u))
    red = " + ".join("a%d" % i for i in range(u)) if u else "0.0"
    st_body = "\n".join(
        "        tl.store(dst + p + %d * seg, v%d)"
        % (j, j) for j in range(s))
    sts_init = "\n".join(
        "    v%d = %s + 1.0" % (j, red if u else "tl.zeros((BLOCK,), "
                                              "dtype=tl.float32)")
        for j in range(s))
    sink_val = ("tl.sum(%s)" % red) if u else "0.0"
    return """
@triton.jit
def e17g_%s(src, dst, sink, tick_ptr, ITERS: tl.constexpr,
            BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = ITERS * BLOCK
%s
%s
    t0 = _clock()
    for step in range(ITERS):
        p = pid * seg + step * BLOCK + lane
%s
%s
    tl.store(sink + pid, %s)
    tl.debug_barrier()
    t1 = _clock()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (tag, accs, sts_init, loads, st_body, sink_val)


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
                                     "e17g_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    import torch  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    cells = []
    for u in (0, 1, 2, 4, 8):
        for s in (0, 1, 2, 4):
            if u == 0 and s == 0:
                continue
            for kb in (4, 16):
                cells.append((u, s, kb))
    # 32KB probes on the axis extremes for the curve fit
    cells += [(1, 0, 32), (4, 0, 32), (0, 1, 32), (0, 4, 32), (1, 1, 32),
              (8, 1, 32)]

    out = {"meta": {"grid": 56, "launches": 30, "ks": [16, 32],
                    "method": "clock64 slope K16/K32, simt_only, hot"},
           "cells": {}}
    out_path = Path(args.out)
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            out["cells"].update(prev.get("cells", {}))
            print("resumed %d cells" % len(out["cells"]), flush=True)
        except (ValueError, OSError):
            pass

    import triton  # noqa: E402

    for u, s, kb in cells:
        key = "u%d_s%d_%dk" % (u, s, kb)
        if key in out["cells"]:
            print("skip", key, flush=True)
            continue
        tag = key
        src_text = HEADER + COMMON + gen_body(tag, u, s)
        gen_path = Path("/root/work/explore_contig/_gen_e17g.py")
        gen_path.write_text(src_text, encoding="utf-8")
        import importlib.util
        spec = importlib.util.spec_from_file_location("_gen_e17g",
                                                      str(gen_path))
        mod = importlib.util.module_from_spec(spec)
        t0 = time.time()
        try:
            spec.loader.exec_module(mod)
            blk = 1024 * (kb // 4)
            seg_max = 56 * 32 * blk
            streams = max(u, s, 1)
            copies = max(4, min(8, int(2.0e9 / (streams * seg_max * 4))))
            total = streams * seg_max * copies
            pool = torch.randn((total,), dtype=torch.float32, device="npu")
            dstpool = torch.zeros((total,), dtype=torch.float32,
                                  device="npu") if s else None
            outbuf = torch.zeros((56,), dtype=torch.float32,
                                 device="npu") if u and not s else None
            tick = torch.zeros((112,), dtype=torch.int64, device="npu")
            sinkbuf = torch.zeros((56,), dtype=torch.float32, device="npu")
            kn = getattr(mod, "e17g_%s" % tag)

            def launch(iters):
                kw = dict(ITERS=iters, BLOCK=blk, num_warps=4,
                          compile_mode="simt_only",
                          auto_simt_scope_mode="off",
                          superblock_factor=1)
                if s:
                    with clock64_compilation():
                        kn[(56,)](pool, dstpool, sinkbuf, tick, **kw)
                else:
                    with clock64_compilation():
                        kn[(56,)](pool, outbuf, sinkbuf, tick, **kw)

            # first launch compiles (hook replaces markers)
            launch(16)
            torch.npu.synchronize()

            def measure(iters):
                all_ns = []
                for i in range(30):
                    launch(iters)
                    torch.npu.synchronize()
                    t = tick.cpu().reshape(56, 2)
                    ns = (t[:, 1] - t[:, 0]) * 1.00008
                    all_ns.append(statistics.median(ns.tolist()))
                return statistics.median(all_ns)

            w16 = measure(16)
            w32 = measure(32)
            out["cells"][key] = {
                "whole16": round(w16, 1), "whole32": round(w32, 1),
                "slope": round((w32 - w16) / 16, 1),
                "per_iter32": round(w32 / 32, 1)}
            print("%s slope=%.1f per_iter32=%.1f (%.0fs)" %
                  (key, out["cells"][key]["slope"],
                   out["cells"][key]["per_iter32"], time.time() - t0),
                  flush=True)
            del pool, dstpool
            torch.npu.empty_cache()
        except Exception as exc:  # noqa: BLE001 - record and continue
            out["cells"][key] = {"error": repr(exc)[:300]}
            print("%s ERROR %r" % (key, exc), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
