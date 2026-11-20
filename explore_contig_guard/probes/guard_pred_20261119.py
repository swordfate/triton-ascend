#!/usr/bin/env python3
"""Guard prediction-only rerun for the 20261119 review: measured values
parsed from the 20261118 review doc's 3.2 table; predictions from the
route dumps of the current wheel (bench_guard_contig bodies)."""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/work/explore_contig")
import bench_guard_contig as B  # noqa: E402

DOC = ("/root/work/triton-ascend/explore_contig_guard/"
       "CONTIGUOUS_REVIEW_20261118.md")
OUT = Path("/tmp/GUARD_20261119/guard_table.json")

import torch  # noqa: E402
import torch_npu  # noqa: E402
torch.npu.set_device(0)

meas = {}
for line in open(DOC, encoding="utf-8"):
    m = re.match(r"\|\s*(\S+)\s*\|\s*(simd_f1|simt_f1|simt_f2|simt_f4)\s*"
                 r"\|\s*([\d.]+)\s*\|", line)
    if m:
        meas[(m.group(1), m.group(2))] = float(m.group(3))
print("measured combos parsed:", len(meas), file=sys.stderr)

out_path = OUT
out_path.parent.mkdir(parents=True, exist_ok=True)
gen_path = out_path.parent / "_gen_guard_contig.py"
gen_path.write_text(B.GEN, encoding="utf-8")
import importlib.util
sspec = importlib.util.spec_from_file_location("_gen_guard_contig",
                                               str(gen_path))
mod = importlib.util.module_from_spec(sspec)
sspec.loader.exec_module(mod)

results = []
names = list(B.CASES.keys())
for i, name in enumerate(names):
    cfg = B.CASES[name]
    kind, dt, block, K, grid = cfg
    if (name, "simd_f1") not in meas:
        continue
    src, src2, dst = B.build_case(mod, name, cfg)
    preds = {}
    try:
        dump4 = str(out_path.parent / (name + "_w4.route.json"))
        launch4 = {
            "scopy": lambda kw: mod.gc_scopy[(grid,)](
                src, dst, src.numel(), block, **kw),
            "sadd": lambda kw: mod.gc_sadd[(grid,)](
                src, src2, dst, src.numel(), block, **kw),
            "sload": lambda kw: mod.gc_sload[(grid,)](
                src, dst, src.numel(), block, **kw),
            "sstore": lambda kw: mod.gc_sstore[(grid,)](
                dst, dst.numel(), block, **kw),
            "lcopy": lambda kw: mod.gc_lcopy[(grid,)](
                src, dst, K, block, **kw),
            "ladd": lambda kw: mod.gc_ladd[(grid,)](
                src, src2, dst, K, block, **kw),
            "lload": lambda kw: mod.gc_lload[(grid,)](
                src, dst, K, block, **kw),
            "lstore": lambda kw: mod.gc_lstore[(grid,)](
                dst, K, block, **kw),
        }[kind]
        launch4(B.route_kwargs(grid, 4, dump4))
        torch.npu.synchronize()
        p4 = B.route_pred(dump4)
        preds["simd_f1"] = p4.get("all_simd")
        dump1 = str(out_path.parent / (name + "_w1.route.json"))
        launch1 = {
            "scopy": lambda kw: mod.gc_scopy[(grid,)](
                src, dst, src.numel(), block, **kw),
            "sadd": lambda kw: mod.gc_sadd[(grid,)](
                src, src2, dst, src.numel(), block, **kw),
            "sload": lambda kw: mod.gc_sload[(grid,)](
                src, dst, src.numel(), block, **kw),
            "sstore": lambda kw: mod.gc_sstore[(grid,)](
                dst, dst.numel(), block, **kw),
            "lcopy": lambda kw: mod.gc_lcopy[(grid,)](
                src, dst, K, block, **kw),
            "ladd": lambda kw: mod.gc_ladd[(grid,)](
                src, src2, dst, K, block, **kw),
            "lload": lambda kw: mod.gc_lload[(grid,)](
                src, dst, K, block, **kw),
            "lstore": lambda kw: mod.gc_lstore[(grid,)](
                dst, K, block, **kw),
        }[kind]
        launch1(B.route_kwargs(grid, 1, dump1))
        torch.npu.synchronize()
        p1 = B.route_pred(dump1)
        preds["simt_f1"] = p1.get("all_simt_only_f1") or \
            p1.get("all_simt_only")
        preds["simt_f2"] = p1.get("all_simt_only_f2")
        preds["simt_f4"] = p1.get("all_simt_only_f4")
    except Exception as e:
        print("PREDICT FAIL %s: %s" % (name, e), flush=True)
    for combo in ("simd_f1", "simt_f1", "simt_f2", "simt_f4"):
        m = meas.get((name, combo))
        p = preds.get(combo)
        err = (100.0 * (p - m) / m) if (p and m) else None
        results.append({"case": name, "combo": combo, "grid": grid, "K": K,
                        "meas_ns": m, "pred_ns": p, "err_pct": err,
                        "pass": bool(err is not None and abs(err) <= 10.0)})
    print("%2d/%d %s done" % (i + 1, len(names), name), flush=True)

out_path.write_text(json.dumps(results, indent=1))
n_pass = sum(1 for r in results if r["pass"])
n_eval = sum(1 for r in results if r["err_pct"] is not None)
print("GUARD: %d/%d PASS" % (n_pass, n_eval))
print("WROTE", OUT)
