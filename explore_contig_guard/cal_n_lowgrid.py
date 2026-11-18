#!/usr/bin/env python3
"""calN: low-grid rows (g{1,2,4}) for the b1024 families that strided
scatter/compaction need.  hint=2 stages sit below the lowest measured grid
(g8) and the tier-1 fit fell back to g8 rows priced for 8-way co-residency.

Families (mode x direction), grids {1,2,4}, K {4,8,16}, BLOCK=256 f32
(= 1024 B chunk = the real per-iteration chunk):
  n1_s0_b1024 (load-only, dynamic all-true mask - matches the real body)
  n0_s1_b1024 (store-only, dynamic all-true mask)
Rows = whole-kernel profiler median Duration (us -> ns), calJ convention
(grid <= 56: no wave division)."""
import json
import os
import statistics
import shutil
import csv
import tempfile
import sys
from pathlib import Path

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")
import torch  # noqa: E402
import torch_npu  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.npu.set_device(0)

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit(do_not_specialize=['total'])
def cm_ldk(src, keep, total, BLOCK: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = pid * K * BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(K):
        x = tl.load(src + base + k * BLOCK + offs, mask=base + k * BLOCK + offs < total)
        acc += x
    tl.store(keep + pid, tl.sum(acc))

@triton.jit(do_not_specialize=['total'])
def cm_stk(dst, total, BLOCK: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = pid * K * BLOCK
    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
    for k in range(K):
        tl.store(dst + base + k * BLOCK + offs, v, mask=base + k * BLOCK + offs < total)
"""


def session(kn, args, grid, tag, warps, mode, launches=12):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="calN_" + tag))

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
            p.step() if False else None
    torch.npu.synchronize()
    durs = []
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "cm_" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]))
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs) * 1000 if durs else float("nan")


def main():
    gen_path = Path("/tmp/_gen_calN.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_calN", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    for mode, mtag in (("simd", "simd"), ("simt_only", "simt")):
        rows = {}
        for direction in ("load", "store"):
            for grid in (1, 2, 4):
                for K in (4, 8, 16):
                    total = grid * K * 256
                    if direction == "load":
                        src = torch.randn((total,), dtype=torch.float32, device="npu")
                        keep = torch.zeros((grid,), dtype=torch.float32, device="npu")
                        kn, args, kname = mod.cm_ldk, (src, keep, total, 256, K), "cm_ldk"
                    else:
                        dst = torch.empty((total,), dtype=torch.float32, device="npu")
                        kn, args, kname = mod.cm_stk, (dst, total, 256, K), "cm_stk"
                    key = "n%s_s%d_b1024_k%d_g%d_m0" % (
                        0 if direction == "store" else 1,
                        1 if direction == "store" else 0, K, grid)
                    med = session(kn, args, grid,
                                  "%s_%s_g%d_k%d" % (mtag, direction, grid, K),
                                  4, mode)
                    rows[key] = round(med, 1)
                    print("%-34s %8.1f ns" % (key, med))
                    sys.stdout.flush()

        out = Path("/root/work/explore_contig/calN_rows_%s.json" % mtag)
        out.write_text(json.dumps(rows, indent=1))
        print("WROTE", out, len(rows))


main()
