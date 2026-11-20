#!/usr/bin/env python3
"""E17f v3: stream-count x per-stream-transaction x residency calibration.

Two loop calibers per shape:
  d (dyn)   : runtime `iters` (scf.for, like real grid-stride kernels)
  c (cexpr) : constexpr `iters` (E17's original methodology)

Residency: hot = fixed buffers across launches; cold = rotate across a
pool (>= 8 copies, ~2 GB target).

Resumable: existing cells in --out are skipped; per-cell errors recorded
and the sweep continues.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""


def gen_ld(n, cexpr):
    accs = "\n".join("    a%d = tl.zeros((BLOCK,), dtype=tl.float32)" % s
                     for s in range(n))
    body = "\n".join(
        "        a%d += tl.load(src + p + %d * seg)" % (s, s)
        for s in range(n))
    red = " + ".join("a%d" % s for s in range(n))
    it = "ITERS: tl.constexpr" if cexpr else "iters"
    loops = "ITERS" if cexpr else "iters"
    return """
@triton.jit
def e17f_ld%d_%s(src, out, tick_ptr, %s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = %s * BLOCK
%s
    t0 = _sys_cnt()
    for step in range(%s):
        p = pid * seg + step * BLOCK + lane
%s
    t1 = _sys_cnt()
    tl.store(out + pid, tl.sum(%s, axis=0))
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (n, "c" if cexpr else "d", it, loops, accs, loops, body, red)


def gen_st(n, cexpr):
    init = "    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0"
    body = "\n".join(
        "        tl.store(dst + p + %d * seg, v)" % s for s in range(n))
    it = "ITERS: tl.constexpr" if cexpr else "iters"
    loops = "ITERS" if cexpr else "iters"
    return """
@triton.jit
def e17f_st%d_%s(dst, tick_ptr, %s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = %s * BLOCK
%s
    t0 = _sys_cnt()
    for step in range(%s):
        p = pid * seg + step * BLOCK + lane
%s
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (n, "c" if cexpr else "d", it, loops, init, loops, body)


def gen_mix(n, cexpr):
    accs = "\n".join("    a%d = tl.zeros((BLOCK,), dtype=tl.float32)" % s
                     for s in range(n))
    body = "\n".join(
        "        a%d += tl.load(src + p + %d * seg)" % (s, s)
        for s in range(n))
    red = " + ".join("a%d" % s for s in range(n))
    it = "ITERS: tl.constexpr" if cexpr else "iters"
    loops = "ITERS" if cexpr else "iters"
    return """
@triton.jit
def e17f_mix%d_%s(src, dst, tick_ptr, %s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    seg = %s * BLOCK
%s
    t0 = _sys_cnt()
    for step in range(%s):
        p = pid * seg + step * BLOCK + lane
%s
        tl.store(dst + p, %s + 1.0)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
""" % (n, "c" if cexpr else "d", it, loops, accs, loops, body, red)


GRID = 56
LAUNCHES = 30
KS = [16, 32]


def measure(launch, grid, iters):
    import torch
    tick = torch.zeros((grid * 2,), dtype=torch.int64, device="npu")
    for _ in range(5):
        launch(0, iters, tick)
    torch.npu.synchronize()
    all_ns = []
    for i in range(LAUNCHES):
        launch(i, iters, tick)
        torch.npu.synchronize()
        t = tick.cpu().reshape(grid, 2)
        ns = (t[:, 1] - t[:, 0]) * 1.00008
        all_ns.append(statistics.median(ns.tolist()))
    return statistics.median(all_ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=2)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e17f_results.json")
    ap.add_argument("--families", default="ld,st,mix")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    parts = [HEADER, COMMON]
    for n in (1, 2, 4, 8):
        parts.append(gen_ld(n, False))
        parts.append(gen_ld(n, True))
    for n in (1, 4):
        parts.append(gen_st(n, False))
        parts.append(gen_st(n, True))
    for n in (1, 2, 4, 8):
        parts.append(gen_mix(n, False))
        parts.append(gen_mix(n, True))
    gen_path = Path("/root/work/explore_contig/_gen_e17f.py")
    gen_path.write_text("".join(parts), encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e17f", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("module built", flush=True)

    blocks = {4: 1024, 8: 2048, 16: 4096, 32: 8192}
    all_families = [
        ("ld", [(n, kb) for n in (1, 2, 4, 8)
                for kb in (4, 8, 16, 32)]),
        ("st", [(n, kb) for n in (1, 4) for kb in (4, 8, 16, 32)]),
        ("mix", [(n, kb) for n in (1, 2, 4, 8) for kb in (4, 8, 16)]),
    ]
    want = set(args.families.split(","))

    out = {"meta": {"grid": GRID, "launches": LAUNCHES, "ks": KS,
                    "method": "SYS_CNT slope K16/K32; hot=fixed buffers; "
                              "cold=rotating pool"},
           "ld": {}, "st": {}, "mix": {}}
    out_path = Path(args.out)
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for k in ("ld", "st", "mix"):
                for kk, vv in prev.get(k, {}).items():
                    if "error" not in vv:
                        out[k][kk] = vv
            print("resumed: ld=%d st=%d mix=%d valid cells" %
                  (len(out["ld"]), len(out["st"]), len(out["mix"])),
                  flush=True)
        except (ValueError, OSError):
            pass

    for family, cells in all_families:
        if family not in want:
            continue
        print("== %s family ==" % family, flush=True)
        for n, kb in cells:
            key = "%dx%d" % (n, kb)
            if key in out[family]:
                print("skip %s %s (have)" % (family, key), flush=True)
                continue
            blk = blocks[kb]
            seg_max = GRID * max(KS) * blk
            bytes_per_copy = n * seg_max * 4
            pool_n = max(8, min(16, int(2.0e9 / bytes_per_copy)))
            total = n * seg_max * pool_n
            t0 = time.time()
            try:
                pool = torch.randn((total,), dtype=torch.float32,
                                   device="npu")
                dstpool = None
                if family == "mix":
                    dstpool = torch.zeros((total,), dtype=torch.float32,
                                          device="npu")
                cell = {"pool_copies": pool_n,
                        "copy_mb": round(bytes_per_copy / 1e6, 1)}
                for calib in ("d", "c"):
                    kn = getattr(mod, "e17f_%s%d_%s" % (family, n, calib))
                    ikw = "ITERS" if calib == "c" else "iters"
                    if family == "ld":
                        outbuf = torch.zeros((GRID,), dtype=torch.float32,
                                             device="npu")

                        def launch(ridx, iters, tick, pool=pool,
                                   outbuf=outbuf, kn=kn, n=n, seg=seg_max,
                                   blk=blk, pn=pool_n, ikw=ikw):
                            rot = ridx % pn
                            kn[(GRID,)](pool[rot * n * seg:], outbuf, tick,
                                        **{ikw: iters, "BLOCK": blk,
                                           "num_warps": 4,
                                           "compile_mode": "simd",
                                           "auto_simt_scope_mode": "off"})
                    elif family == "st":
                        def launch(ridx, iters, tick, pool=pool, kn=kn,
                                   n=n, seg=seg_max, blk=blk, pn=pool_n,
                                   ikw=ikw):
                            rot = ridx % pn
                            kn[(GRID,)](pool[rot * n * seg:], tick,
                                        **{ikw: iters, "BLOCK": blk,
                                           "num_warps": 4,
                                           "compile_mode": "simd",
                                           "auto_simt_scope_mode": "off"})
                    else:
                        def launch(ridx, iters, tick, pool=pool,
                                   dp=dstpool, kn=kn, n=n, seg=seg_max,
                                   blk=blk, pn=pool_n, ikw=ikw):
                            rot = ridx % pn
                            kn[(GRID,)](pool[rot * n * seg:],
                                        dp[rot * n * seg:], tick,
                                        **{ikw: iters, "BLOCK": blk,
                                           "num_warps": 4,
                                           "compile_mode": "simd",
                                           "auto_simt_scope_mode": "off"})

                    cell[calib] = {}
                    for tag, fixed in (("hot", True), ("cold", False)):
                        def g(ridx, iters, tick, launch=launch,
                              fixed=fixed):
                            launch(0 if fixed else ridx, iters, tick)
                        w16 = measure(g, GRID, 16)
                        w32 = measure(g, GRID, 32)
                        cell[calib][tag] = {
                            "whole16": round(w16, 1),
                            "whole32": round(w32, 1),
                            "slope_per_iter": round((w32 - w16) / 16, 1),
                            "per_iter32": round(w32 / 32, 1)}
                out[family][key] = cell
                print("%s %s pool=%2d | cexp hot %7.1f cold %7.1f | "
                      "dyn hot %7.1f cold %7.1f ns/iter (%.0fs)" %
                      (family, key, pool_n,
                       cell["c"]["hot"]["slope_per_iter"],
                       cell["c"]["cold"]["slope_per_iter"],
                       cell["d"]["hot"]["slope_per_iter"],
                       cell["d"]["cold"]["slope_per_iter"],
                       time.time() - t0), flush=True)
            except Exception as exc:  # noqa: BLE001 - keep sweeping
                out[family][key] = {"error": repr(exc)[:200]}
                print("%s %s ERROR %r" % (family, key, exc), flush=True)
            Path(args.out).write_text(json.dumps(out, indent=1))
            try:
                del pool, dstpool
            except NameError:
                pass
            torch.npu.empty_cache()
    print("saved", args.out)


main()
