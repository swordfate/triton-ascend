#!/usr/bin/env python3
"""remeasure simt n1_s0_b1024 g2 rows: 3 sessions x 12 launches, take the
min session median (guard convention), for k{4,8,16}."""
import json
import sys
from pathlib import Path
import os
import statistics
import shutil
import csv
import tempfile
sys.path.insert(0, "/root/work/explore_contig")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")
import torch  # noqa: E402
import torch_npu  # noqa: E402
from triton.runtime import driver  # noqa: E402

torch.npu.set_device(0)
genp = Path("/root/work/explore_contig/_gen_calN.py")
import importlib.util
genp = Path("/root/work/explore_contig/_gen_calN.py")
sspec = importlib.util.spec_from_file_location("_gen_calN", str(genp))
mod = importlib.util.module_from_spec(sspec)
sspec.loader.exec_module(mod)


def session(kn, args, grid, tag, warps, mode, launches=12):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="calNr_" + tag))

    def fn():
        kn[(grid,)](*args, num_warps=warps, compile_mode=mode,
                    auto_simt_scope_mode="off",
                    logical_program_count_hint=grid,
                    physical_vector_core_count_hint=56)
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(str(trace))):
        torch.npu.synchronize()
        for _ in range(launches):
            fn()
    torch.npu.synchronize()
    durs = []
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "cm_" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]))
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs) * 1000 if durs else float("nan")


new = {}
for K in (4, 8, 16):
    total = 2 * K * 256
    src = torch.randn((total,), dtype=torch.float32, device="npu")
    keep = torch.zeros((2,), dtype=torch.float32, device="npu")
    meds = [session(mod.cm_ldk, (src, keep, total, 256, K), 2,
                    "simt_g2_k%d_s%d" % (K, s), 4, "simt_only") for s in range(3)]
    best = min(meds)
    key = "n1_s0_b1024_k%d_g2_m0" % K
    new[key] = round(best, 1)
    print(key, ["%.1f" % m for m in meds], "-> min %.1f" % best)
    sys.stdout.flush()

rows = json.load(open("/root/work/explore_contig/calN_rows_simt.json"))
rows.update(new)
print("UPDATED")

