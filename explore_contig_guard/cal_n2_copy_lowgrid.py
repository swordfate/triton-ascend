#!/usr/bin/env python3
"""calN2: low-grid rows (g{1,2,4}) for the n1_s1 copy family.

G-1b finding: plain copy loops (n1_s1) at hint 2..8 over-predict by
+17..+53% because the n1_s1 families only carry g8+ rows - the same G9
pattern scatter/compaction had, in an uncalibrated family.  This probe
measures the whole-kernel Duration of the copy body (dynamic all-true
mask, matching the real body) at g{1,2,4} x K{4,8,16} x chunk
{512B,1KB,2KB} (BLOCK 128/256/512 f32), simd and simt modes.
Rows = whole-kernel profiler median Duration (us -> ns).
"""
import csv
import json
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
@triton.jit(do_not_specialize=['total'])
def cm_cpy(src, dst, total, BLOCK: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = pid * K * BLOCK
    for k in range(K):
        idx = base + k * BLOCK + offs
        v = tl.load(src + idx, mask=idx < total)
        tl.store(dst + idx, v, mask=idx < total)
"""

GRIDS = [1, 2, 4]
KS = [4, 8, 16]
BLOCKS = [128, 256, 512]  # f32 -> 512B / 1KB / 2KB chunks


def session(kn, args, grid, tag, mode, launches=12):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="calN2_" + tag))

    def fn():
        kn[(grid,)](*args, num_warps=4, compile_mode=mode,
                    auto_simt_scope_mode="off",
                    logical_program_count_hint=grid,
                    physical_vector_core_count_hint=56)

    for _ in range(5):
        fn()
    torch.npu.synchronize()
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(
                          str(trace))):
        torch.npu.synchronize()
        for _ in range(launches):
            fn()
    torch.npu.synchronize()
    durs = []
    for tpath in sorted(trace.rglob("kernel_details.csv")):
        with tpath.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "cm_cpy" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]))
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs) * 1000.0 if durs else float("nan")


def main():
    gen_path = Path("/root/work/explore_contig/_gen_caln2.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_caln2", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    out = {"simd": {}, "simt": {}}
    for mode, mtag in (("simd", "simd"), ("simt", "simt")):
        for B in BLOCKS:
            for K in KS:
                total = 4 * K * B * 4  # covers the largest grid
                src = torch.randn((total,), dtype=torch.float32,
                                  device="npu")
                dst = torch.zeros((total,), dtype=torch.float32,
                                  device="npu")
                for g in GRIDS:
                    med = []
                    for sess in range(3):
                        med.append(session(mod.cm_cpy,
                                           (src, dst, g * K * B), g,
                                           f"{mtag}_b{B}_k{K}_g{g}", mode))
                    val = min(med) if mode == "simt" else statistics.median(med)
                    key = f"n1_s1_b{B}_k{K}_g{g}_m0"
                    out[mode][key] = round(val, 1)
                    print(mode, key, val)
                    sys.stdout.flush()
    Path("/root/work/explore_contig/calN2_rows.json").write_text(
        json.dumps(out, indent=1, sort_keys=True))
    print("saved calN2_rows.json")


main()
