#!/usr/bin/env python3
"""E15: do masked lanes fetch memory, or does HW move the full padded tile?

Kernel: per-program (ROWS x BLOCK) fp32 tile, mask = col < ACTIVE, row-sum.
Identical padded footprint in all variants; only ACTIVE changes.
  ACTIVE=512 : mask all-true (reference, ratio 1.0)
  ACTIVE=4   : ratio 1/128, padded 128KB, active 1KB per program
  ACTIVE=1   : ratio 1/512, padded 128KB, active 256B per program
If masked lanes fetch nothing -> time scales with ACTIVE (plus overheads).
If HW moves the padded tile and zero-fills -> all variants take the same time.
Store side: each program also masked-stores its row-sums (tiny, ACTIVE-independent).
"""
import csv
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "2")
import torch  # noqa: E402
import torch_npu  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.npu.set_device(0)

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit
def e15_rowsum(src, dst, ROWS: tl.constexpr, BLOCK: tl.constexpr,
               ACTIVE: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    m = cols < ACTIVE
    v = tl.load(src + pid * ROWS * BLOCK + rows[:, None] * BLOCK
                + cols[None, :], mask=m, other=0.0)
    s = tl.sum(v, axis=1)
    tl.store(dst + pid * ROWS + rows, s)

@triton.jit
def e15_rowfill(src, dst, ROWS: tl.constexpr, BLOCK: tl.constexpr,
                ACTIVE: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    m = cols < ACTIVE
    v = tl.load(src + pid * ROWS * BLOCK + rows[:, None] * BLOCK
                + cols[None, :], mask=m, other=0.0)
    tl.store(dst + pid * ROWS * BLOCK + rows[:, None] * BLOCK
             + cols[None, :], v + 1.0, mask=m)
"""


def measure(mod, kn_name, args, launches=24):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e15_"))
    kn = getattr(mod, kn_name)

    def fn():
        kn[(8,)](*args, num_warps=4, compile_mode="simd",
                 auto_simt_scope_mode="off", logical_program_count_hint=8,
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
    for tpath in sorted(trace.rglob("kernel_details.csv")):
        with tpath.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "e15_" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]))
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs) * 1000 if durs else float("nan")


def main():
    gen_path = Path("/root/work/explore_contig/_gen_e15.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e15", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    ROWS, BLOCK = 64, 512
    total = 8 * ROWS * BLOCK
    src = torch.randn((total,), dtype=torch.float32, device="npu")
    print("padded footprint per program: %d KB" % (ROWS * BLOCK * 4 // 1024))
    print("%-18s %10s %12s %12s" % ("variant", "ACTIVE", "active KB", "ns"))
    for kn_name in ("e15_rowsum", "e15_rowfill"):
        for active in (512, 4, 1):
            dst = torch.zeros((total,), dtype=torch.float32, device="npu")
            ns = measure(mod, kn_name,
                         (src, dst, ROWS, BLOCK, active))
            print("%-18s %10d %12.2f %12.1f" %
                  (kn_name, active, ROWS * active * 4 / 1024, ns))
            sys.stdout.flush()


main()
