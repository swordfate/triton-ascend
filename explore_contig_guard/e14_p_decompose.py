#!/usr/bin/env python3
"""E14: decompose the E13 P-axis curve - where does T(P) growth come from?

Variants (BLOCK=1024 f32 = 4KB, simd, warm):
  ctl1      scalar store, grid=1            (reference floor)
  ctlP      scalar store, grid=P            (pure dispatch/prologue scaling)
  sumonly   x=full(BLOCK), sum, store 1elem (reduction chain, no load, no burst)
  ldmax     x=load 4KB, store max(x)        (load + cheap tree-consume, no sum-add chain)
  loadsum   x=load 4KB, sum, store 1elem    (= E13 probe)
Also prints the resolved AIV count (experiment 5 of the review).
"""
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")
import torch  # noqa: E402
import torch_npu  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime import driver  # noqa: E402

torch.npu.set_device(0)

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit
def e14_ctl1(dst):
    tl.store(dst + tl.program_id(0), 0.0)

@triton.jit
def e14_ctlP(dst):
    pid = tl.program_id(0)
    tl.store(dst + pid, 0.0)

@triton.jit
def e14_sumonly(dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.full((BLOCK,), 1.0, tl.float32)
    tl.store(dst + pid, tl.sum(x, 0) + 1)

@triton.jit
def e14_ldmax(src, dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    x = tl.load(src + pid * BLOCK + lane)
    tl.store(dst + pid, tl.max(x, 0) + 1)

@triton.jit
def e14_loadsum(src, dst, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    x = tl.load(src + pid * BLOCK + lane)
    tl.store(dst + pid, tl.sum(x, 0) + 1)
"""

PS = [1, 2, 4, 8, 16, 32, 56, 112, 224]


def prof_med(kn, args, grid, warps=4, launches=16):
    import torch_npu.profiler as prof
    from pathlib import Path
    import shutil
    import csv
    trace = Path(__import__("tempfile").mkdtemp(prefix="e14_"))

    def fn():
        kn[(grid,)](*args, num_warps=warps, compile_mode="simd",
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
                if "e14_" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]))
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs) * 1000 if durs else float("nan")


def main():
    print("AIV count:",
          driver.active.utils.get_device_properties(
              torch.npu.current_device()).get("num_vectorcore"))

    gen_path = Path("/root/work/explore_contig/_gen_e14.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e14", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    BLOCK = 1024
    src = torch.randn((224 * BLOCK,), dtype=torch.float32, device="npu")
    dst = torch.zeros((224 * BLOCK,), dtype=torch.float32, device="npu")

    table = {}
    for P in PS:
        row = {}
        row["ctl1"] = prof_med(mod.e14_ctl1, (dst,), 1) if P == 1 else None
        row["ctlP"] = prof_med(mod.e14_ctlP, (dst,), P)
        row["sumonly"] = prof_med(mod.e14_sumonly, (dst[:P],), P)
        row["ldmax"] = prof_med(mod.e14_ldmax, (src, dst[:P], BLOCK), P)
        row["loadsum"] = prof_med(mod.e14_loadsum, (src, dst[:P], BLOCK), P)
        table[P] = row
        print("P=%-4d ctl1=%s ctlP=%7.1f sumonly=%7.1f ldmax=%7.1f loadsum=%7.1f" %
              (P, ("%.1f" % row["ctl1"]) if row["ctl1"] else "  -  ",
               row["ctlP"], row["sumonly"], row["ldmax"], row["loadsum"]))
        sys.stdout.flush()

    print("\n== derived (ns) ==")
    hdr = "P     ctlP  sumonly  ldmax  loadsum | ld-ctl sum-ctl ldsum-ldmax ldsum-sumonly"
    print(hdr)
    for P in PS:
        r = table[P]
        c1 = table[1]["ctl1"] if P > 1 else r["ctl1"]
        print("%-5d %5.0f %7.0f %6.0f %7.0f | %6.0f %6.0f %10.0f %12.0f" % (
            P, r["ctlP"], r["sumonly"], r["ldmax"], r["loadsum"],
            r["ldmax"] - c1, r["sumonly"] - c1,
            r["ldmax"] - r["ctlP"], r["loadsum"] - r["sumonly"]))


main()
