#!/usr/bin/env python3
"""calF2 - gen-2 direct table probe: absolute profiler duration per geometry.

Fixes vs calF v1:
  * absolute kernel Duration (no pair-diff -> no negative simt values)
  * K swept {4, 16, 64}
  * load-only body keeps a tiny per-pid tail store (outside the loop)
  * no shared-output write serialization
Shapes: 1L(load loop + tiny tail), 1L+1S(copy), 2L+1S(add), 1S(store-only)
Chunks: {1024, 4096, 16384, 65536} B   Grids: {8, 56, 224}
Each row: whole-launch ns (median of N), per-program-equivalent = ns/waves.
"""
import argparse
import importlib.util
import statistics
from pathlib import Path

import torch

from common import (JsonlOut, add_common_args, gate_or_die, health_check,
                    std_opts)

EB = {"int8": 1, "fp16": 2, "fp32": 4}
TT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32}


def launch_opts(mode, warps):
    o = {"num_warps": warps,
         "compile_mode": "simt_only" if mode == "simt" else mode}
    if mode == "simt":
        o["superblock_factor"] = 1
    return o


def gen_module(path, shapes, chunks, dtype_key, ks, grids):
    eb = EB[dtype_key]
    parts = ["import triton\nimport triton.language as tl\n\n"]
    for nl, nst in shapes:
        for B in chunks:
            block = max(1, B // eb)
            # per-iteration per-tensor region in elements: one program's
            # i-th iteration covers nl*BLOCK src elements / nS*BLOCK dst
            maxb = max(nl, nst, 1) * block
            for k in ks:
                base = "cf2_%dl%ds_b%d_k%d" % (nl, nst, B, k)
                body = []
                for j in range(nl):
                    body.append(
                        "        x%d = tl.load(src + pid * iters * %d + "
                        "(step * %d + %d) * BLOCK + lane)"
                        % (j, maxb, nl, j))
                for j in range(nst):
                    rhs = "x0 + %d" % j if nl else \
                        "tl.full((BLOCK,), 1, dst.dtype.element_ty)"
                    body.append(
                        "        tl.store(dst + pid * iters * %d + "
                        "(step * %d + %d) * BLOCK + lane, %s)"
                        % (maxb, max(nst, 1), j, rhs))
                inner = "\n".join(body)
                for grid in grids:
                    # unique function per grid so profiler name filters
                    # never mix geometries in batched sessions
                    name = "%s_g%d" % (base, grid)
                    if nl and not nst:
                        # guard gc_lload body (acc + per-pid scalar sink)
                        text = (
                            "@triton.jit\n"
                            "def %s(src, dst, iters: tl.constexpr, "
                            "BLOCK: tl.constexpr):\n"
                            "    pid = tl.program_id(0)\n"
                            "    lane = tl.arange(0, BLOCK)\n"
                            "    base = pid * iters * BLOCK\n"
                            "    acc = tl.zeros((BLOCK,), src.dtype.element_ty)\n"
                            "    for step in range(iters):\n"
                            "        acc += tl.load(src + base + "
                            "step * BLOCK + lane)\n"
                            "    tl.store(dst + pid, tl.sum(acc, axis=0))\n"
                            % (name))
                    elif nst and not nl:
                        # guard gc_lstore body
                        text = (
                            "@triton.jit\n"
                            "def %s(src, dst, iters: tl.constexpr, "
                            "BLOCK: tl.constexpr):\n"
                            "    pid = tl.program_id(0)\n"
                            "    lane = tl.arange(0, BLOCK)\n"
                            "    base = pid * iters * BLOCK\n"
                            "    for step in range(iters):\n"
                            "        tl.store(dst + base + step * BLOCK + "
                            "lane, tl.full((BLOCK,), 1, "
                            "dst.dtype.element_ty))\n" % (name))
                    elif nl == 2 and nst == 1:
                        # guard gc_ladd body
                        text = (
                            "@triton.jit\n"
                            "def %s(src, src2, dst, iters: tl.constexpr, "
                            "BLOCK: tl.constexpr):\n"
                            "    pid = tl.program_id(0)\n"
                            "    lane = tl.arange(0, BLOCK)\n"
                            "    base = pid * iters * BLOCK\n"
                            "    for step in range(iters):\n"
                            "        x = tl.load(src + base + step * BLOCK "
                            "+ lane)\n"
                            "        y = tl.load(src2 + base + step * BLOCK "
                            "+ lane)\n"
                            "        tl.store(dst + base + step * BLOCK + "
                            "lane, x + y)\n" % (name))
                    else:
                        # guard gc_lcopy body (1L + 1S)
                        text = (
                            "@triton.jit\n"
                            "def %s(src, dst, iters: tl.constexpr, "
                            "BLOCK: tl.constexpr):\n"
                            "    pid = tl.program_id(0)\n"
                            "    lane = tl.arange(0, BLOCK)\n"
                            "    base = pid * iters * BLOCK\n"
                            "    for step in range(iters):\n"
                            "        x = tl.load(src + base + step * BLOCK "
                            "+ lane)\n"
                            "        tl.store(dst + base + step * BLOCK + "
                            "lane, x + 1)\n" % (name))
                    parts.append(text)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("_gen_cal_f2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default=["simd"], nargs="+")
    ap.add_argument("--dtypes", default=["fp32"], nargs="+")
    ap.add_argument("--shapes", nargs="+",
                    default=["1,0", "1,1", "2,1", "0,1"])
    ap.add_argument("--chunks", type=int, nargs="+",
                    default=[1024, 4096, 16384, 65536])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--grids", type=int, nargs="+", default=[8, 56, 224])
    ap.add_argument("--warps-simd", type=int, nargs="+", default=[4])
    ap.add_argument("--warps-simt", type=int, nargs="+", default=[1])
    ap.add_argument("--launches", type=int, default=24)
    ap.add_argument("--session-size", type=int, default=1)
    args = ap.parse_args()

    if args.health and not args.skip_health:
        rec = health_check(args.health)
        gate_or_die(rec, args.health)
        print("health:", rec)

    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    out = JsonlOut(args.out)
    import torch_npu  # noqa: F401
    import triton  # noqa: F401

    for mode in args.modes:
        warps_list = args.warps_simd if mode == "simd" else args.warps_simt
        for dtype_key in args.dtypes:
            eb = EB[dtype_key]
            ttype = TT[dtype_key]
            gen_path = str(Path(args.out).parent / "_gen_cal_f2.py")
            gen_module(gen_path, shapes, args.chunks, dtype_key, args.ks,
                       args.grids)
            mod = load_module(gen_path)
            for warps in warps_list:
                geos = []
                for (nl, nst) in shapes:
                    for B in args.chunks:
                        block = max(1, B // eb)
                        for K in args.ks:
                            for grid in args.grids:
                                geos.append((nl, nst, B, block, K, grid))
                max_elems = 0
                for (nl, nst, B, block, K, grid) in geos:
                    total = grid * max(nl, nst, 1) * K * block
                    max_elems = max(max_elems, total)
                src = torch.randn(max_elems, dtype=ttype, device="npu")
                src2 = torch.randn(max_elems, dtype=ttype, device="npu")
                dst = torch.zeros(max_elems, dtype=ttype, device="npu")
                launches = []
                for (nl, nst, B, block, K, grid) in geos:
                    kname = "%dl%ds_b%d_k%d_g%d" % (nl, nst, B, K, grid)
                    fn = getattr(mod, "cf2_" + kname)
                    kw = launch_opts(mode, warps)
                    if nl == 2 and nst == 1:
                        launches.append(
                            ((nl, nst, B, block, K, grid), kname,
                             lambda fn=fn, g=grid, K=K, b=block, kw=kw:
                             fn[(g,)](src, src2, dst, K, b, **kw)))
                    else:
                        launches.append(
                            ((nl, nst, B, block, K, grid), kname,
                             lambda fn=fn, g=grid, K=K, b=block, kw=kw:
                             fn[(g,)](src, dst, K, b, **kw)))
                ok_launches = []
                for entry in launches:
                    try:
                        entry[2]()
                        ok_launches.append(entry)
                    except Exception as exc:
                        geo, kname = entry[0], entry[1]
                        print("  COMPILE-FAIL %s: %s" % (kname, str(exc)[:90]),
                              flush=True)
                        out.write(kind="cf2_compile_fail", mode=mode,
                                  dtype=dtype_key, warps=warps,
                                  nl=geo[0], nst=geo[1], bytes=geo[2],
                                  k=geo[4], grid=geo[5])
                torch.npu.synchronize()
                launches = ok_launches
                import shutil, csv, tempfile
                import torch_npu.profiler as prof
                for start in range(0, len(launches), args.session_size):
                    batch = launches[start:start + args.session_size]
                    # outside warmup so the captured launches see the same
                    # steady state as the guard protocol (L2/DVFS ramp)
                    for _ in range(5):
                        for (_, _, fn) in batch:
                            fn()
                    torch.npu.synchronize()
                    trace = Path(tempfile.mkdtemp(prefix="cf2_"))
                    # harness-proven profiler config (profile_utils.py):
                    # Level1 + skip_first keeps exports stable session after
                    # session; op_statistic.csv gives per-kernel averages.
                    config = prof._ExperimentalConfig(
                        aic_metrics=prof.AiCMetrics.PipeUtilization,
                        profiler_level=prof.ProfilerLevel.Level1)
                    with prof.profile(
                            activities=[prof.ProfilerActivity.NPU],
                            record_shapes=False, profile_memory=False,
                            with_stack=False,
                            schedule=prof.schedule(wait=0, warmup=2,
                                                   active=20, repeat=1,
                                                   skip_first=2),
                            on_trace_ready=prof.tensorboard_trace_handler(
                                str(trace)),
                            experimental_config=config) as p:
                        for _ in range(28):
                            for (_, _, fn) in batch:
                                fn()
                            p.step()
                    torch.npu.synchronize()
                    rows = {}
                    for path in sorted(trace.rglob("op_statistic.csv")):
                        with path.open(newline="") as h:
                            for row in csv.DictReader(h):
                                name = row.get("OP Type") or ""
                                try:
                                    avg = float(row.get("Avg Time(us)"))
                                except (TypeError, ValueError):
                                    continue
                                for (_, kname, _) in batch:
                                    if ("cf2_" + kname) in name:
                                        rows.setdefault(kname, []).append(avg)
                    shutil.rmtree(trace, ignore_errors=True)
                    for (geo, kname, _) in batch:
                        nl, nst, B, block, K, grid = geo
                        durs = rows.get(kname, [])
                        if not durs:
                            print("  MISSING %s" % kname, flush=True)
                            continue
                        med = statistics.median(durs)
                        waves = max(1, (grid + 55) // 56)
                        out.write(kind="cf2", mode=mode, dtype=dtype_key,
                                  warps=warps, nl=nl, nst=nst, bytes=B,
                                  k=K, grid=grid, ns=round(med * 1000, 1),
                                  waves=waves,
                                  ns_prog=round(med * 1000 / waves, 1),
                                  n=len(durs), min_us=min(durs),
                                  us=med,
                                  src="op_statistic_avg20")
                    print("  batch %d-%d done" % (start, start + len(batch)),
                          flush=True)
    print("calF2 complete")


if __name__ == "__main__":
    main()
