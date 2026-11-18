#!/usr/bin/env python3
"""E20b: delta(P) calibration - the fixed-cost gap between measured
whole-kernel time and the current model at envelope-priced shapes.

For P in {2,4,8,14,28,56}: n = P*1024 (K=1 trip/program, grid=P).
  measured : torch_npu profiler median Duration of the real muls_add
             kernel (identical body to the bench)
  predicted: bench_muls_add_costmodel.py route.json candidate cost
  delta(P) = measured - predicted  (the A(P) shortfall to wire as the
             pricing-aware dual-table extra for envelope-claimed bodies)
"""
import argparse
import json
import statistics
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"

KERNEL = """
@triton.jit
def e20b_muls(x, y, out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n_elements
    v0 = tl.load(x + offset, mask=mask, other=0.0)
    v1 = tl.load(y + offset, mask=mask, other=0.0)
    tl.store(out + offset, v0 * 2.5 + v1, mask=mask)
"""


def profile_duration(mod, n, grid, blk=1024):
    import torch  # noqa: F811
    import torch_npu.profiler as prof
    x = torch.randn((n,), dtype=torch.float32, device="npu")
    y = torch.randn((n,), dtype=torch.float32, device="npu")
    o = torch.zeros((n,), dtype=torch.float32, device="npu")

    def launch():
        mod.e20b_muls[(grid,)](x, y, o, n, BLOCK=blk, num_warps=4,
                               compile_mode="simd",
                               auto_simt_scope_mode="off")

    for _ in range(10):
        launch()
    torch.npu.synchronize()
    trace = Path(tempfile.mkdtemp(prefix="e20b_"))
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(
                          str(trace))):
        torch.npu.synchronize()
        for _ in range(30):
            launch()
        torch.npu.synchronize()
    durs = []
    for tpath in sorted(trace.rglob("kernel_details.csv")):
        with tpath.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "e20b_muls" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]) * 1000)
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs)


import csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e20b_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    gen_path = Path("/root/work/explore_contig/_gen_e20b.py")
    gen_path.write_text(HEADER + KERNEL, encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e20b", str(gen_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = {"meta": {"method": "profiler median Duration (30 launches) vs "
                              "bench route.json prediction; K=1 bodies"},
           "points": {}}
    out_path = Path(args.out)

    bench = "/root/w/cc-harness/costmodel_generalization/" \
            "bench_muls_add_costmodel.py"
    for P in (2, 4, 8, 14, 28, 56):
        n = P * 1024
        t0 = time.time()
        rep = Path(tempfile.mkdtemp(prefix="e20brep_")) / "r.json"
        subprocess.run(
            ["python3", bench, "--route", "auto", "--factor", "1",
             "--n-elements", str(n), "--batch", "1", "--warmup", "2",
             "--repeats", "3", "--report", str(rep)],
            cwd="/root/w/cc-harness/costmodel_generalization",
            capture_output=True, timeout=600)
        pred = None
        try:
            s = rep.read_text(errors="ignore")
            d = json.JSONDecoder().raw_decode(s.strip())[0]
            kind = d.get("effective_decision_kind")
            pred = (d.get("candidate_elapsed_costs") or {}).get(
                kind, 0) / 1000.0
        except (ValueError, OSError) as exc:
            print("P=%d pred parse fail %r" % (P, exc), flush=True)
        dur = profile_duration(mod, n, P)
        delta = dur - pred if pred else None
        out["points"][str(P)] = {
            "duration_us": round(dur, 3), "pred_us": round(pred, 3),
            "delta_us": round(delta, 3) if delta else None}
        print("P=%-3d measured=%8.1f pred=%8.1f delta=%+8.1f ns (%.0fs)"
              % (P, dur, pred, delta, time.time() - t0), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
