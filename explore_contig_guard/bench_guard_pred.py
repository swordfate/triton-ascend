#!/usr/bin/env python3
"""bench_guard_pred.py - prediction-only guard: compile + ONE launch per
combo pair to dump route.json (no measurement loops), join predictions with
the TS8 measured table.  Card 2 friendly, no measurement contention."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/work/explore_contig")
import bench_guard_contig as B  # noqa: E402

TS8 = "/tmp/GUARD_CONTIG/guard_table_TS8.json"
OUT = "/tmp/GUARD_CONTIG/guard_table_hd14pred.json"

import os
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")
import torch  # noqa: E402
import torch_npu  # noqa: E402
torch.npu.set_device(0)

meas = {}
for r in json.load(open(TS8)):
    meas[(r["case"], r["combo"])] = r["meas_ns"]

out_path = Path(OUT)
out_path.parent.mkdir(parents=True, exist_ok=True)
gen_path = out_path.parent / "_gen_guard_contig.py"
gen_path.write_text(B.GEN, encoding="utf-8")
import importlib.util
sspec = importlib.util.spec_from_file_location("_gen_guard_contig", str(gen_path))
mod = importlib.util.module_from_spec(sspec)
sspec.loader.exec_module(mod)

results = []
names = list(B.CASES.keys())
for i, name in enumerate(names):
    cfg = B.CASES[name]
    kind, dt, block, K, grid = cfg
    src, src2, dst = B.build_case(mod, name, cfg)
    preds = {}
    try:
        dump4 = str(out_path.parent / (name + "_w4.route.json"))
        launch4 = {
            "scopy": lambda kw: mod.gc_scopy[(grid,)](src, dst, src.numel(), block, **kw),
            "sadd": lambda kw: mod.gc_sadd[(grid,)](src, src2, dst, src.numel(), block, **kw),
            "sload": lambda kw: mod.gc_sload[(grid,)](src, dst, src.numel(), block, **kw),
            "sstore": lambda kw: mod.gc_sstore[(grid,)](dst, dst.numel(), block, **kw),
            "lcopy": lambda kw: mod.gc_lcopy[(grid,)](src, dst, K, block, **kw),
            "ladd": lambda kw: mod.gc_ladd[(grid,)](src, src2, dst, K, block, **kw),
            "lload": lambda kw: mod.gc_lload[(grid,)](src, dst, K, block, **kw),
            "lstore": lambda kw: mod.gc_lstore[(grid,)](dst, K, block, **kw),
        }[kind]
        launch4(B.route_kwargs(grid, 4, dump4))
        torch.npu.synchronize()
        p4 = B.route_pred(dump4)
        preds["simd_f1"] = p4.get("all_simd")
        dump1 = str(out_path.parent / (name + "_w1.route.json"))
        launch1 = {
            "scopy": lambda kw: mod.gc_scopy[(grid,)](src, dst, src.numel(), block, **kw),
            "sadd": lambda kw: mod.gc_sadd[(grid,)](src, src2, dst, src.numel(), block, **kw),
            "sload": lambda kw: mod.gc_sload[(grid,)](src, dst, src.numel(), block, **kw),
            "sstore": lambda kw: mod.gc_sstore[(grid,)](dst, dst.numel(), block, **kw),
            "lcopy": lambda kw: mod.gc_lcopy[(grid,)](src, dst, K, block, **kw),
            "ladd": lambda kw: mod.gc_ladd[(grid,)](src, src2, dst, K, block, **kw),
            "lload": lambda kw: mod.gc_lload[(grid,)](src, dst, K, block, **kw),
            "lstore": lambda kw: mod.gc_lstore[(grid,)](dst, K, block, **kw),
        }[kind]
        launch1(B.route_kwargs(grid, 1, dump1))
        torch.npu.synchronize()
        p1 = B.route_pred(dump1)
        preds["simt_f1"] = p1.get("all_simt_only_f1") or p1.get("all_simt_only")
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
print("WROTE", OUT)
