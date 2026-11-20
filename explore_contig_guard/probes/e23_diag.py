#!/usr/bin/env python3
"""E23 diagnostic: dump raw profiler kernel rows for A_rt(N=8) vs C_sl(8)."""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/root/work/triton-ascend/explore_contig_guard/probes")
from e23_static_vs_runtime import BLOCK, gen_module, load_module  # noqa: E402

import torch  # noqa: E402


def main():
    gen_module("/tmp/_g23.py")
    mod = load_module("/tmp/_g23.py")
    src = torch.randn(BLOCK * 1024, device="npu")
    dst = torch.zeros(BLOCK * 1024, device="npu")
    opts = {"compile_mode": "simd", "num_warps": 4}
    import torch_npu.profiler as prof
    fns = [lambda: mod.e23_rt[(1,)](src, dst, 8, BLOCK, **opts),
           lambda: mod.e23_sl_8[(1,)](src, dst, BLOCK, **opts)]
    trace = Path(tempfile.mkdtemp())
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=10),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for fn in fns:
            fn()
        torch.npu.synchronize()
        for _ in range(6):
            for fn in fns:
                fn()
        torch.npu.synchronize()
        for _ in range(5):
            for fn in fns:
                fn()
                p.step()
    torch.npu.synchronize()
    for path in sorted(trace.rglob("kernel_details.csv")):
        print("CSV", path)
        with path.open(newline="") as h:
            for r in csv.DictReader(h):
                print("  %-50s %s" % ((r.get("Name") or "")[:50],
                                      r.get("Duration(us)")))
    for path in sorted(trace.rglob("*.csv")):
        if "kernel_details" not in path.name:
            print("OTHER CSV", path.name)
    shutil.rmtree(trace, ignore_errors=True)


main()
