#!/usr/bin/env python3
"""E17e: accurate small-K body cost via null-kernel subtraction.
For K in {1,2,4,8,16,32,64}, grid=56, profiler whole-kernel Duration:
  null  : same launch, loop body = 1 scalar add only (no memory)
  mixed : 4KB load + 4KB store per iteration (the tier-2 body)
  ld    : 4KB load per iteration (acc form)
  st    : 4KB store per iteration
body(K) = whole_variant(K) - whole_null(K)  -> pure memory-loop cost.
"""
import json
import statistics
import sys
from pathlib import Path

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit
def e17e_null(dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for step in range(iters):
        acc += 1.0
    tl.store(dst + pid, tl.sum(acc, axis=0))
    tl.store(tick_ptr + pid * 2, acc.sum())
    tl.store(tick_ptr + pid * 2 + 1, acc.sum())

@triton.jit
def e17e_mixed(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    for step in range(iters):
        v = tl.load(src + base + step * BLOCK + lane)
        tl.store(dst + base + step * BLOCK + lane, v + 1.0)
    tl.store(tick_ptr + pid * 2, v.sum() * 0)

@triton.jit
def e17e_ld(src, dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for step in range(iters):
        acc += tl.load(src + base + step * BLOCK + lane)
    tl.store(dst + pid, tl.sum(acc, axis=0))
    tl.store(tick_ptr + pid * 2, tl.sum(acc, axis=0) * 0)
    tl.store(tick_ptr + pid * 2 + 1, tl.sum(acc, axis=0) * 0)

@triton.jit
def e17e_st(dst, tick_ptr, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    v = tl.zeros((BLOCK,), dtype=tl.float32) + 1.0
    for step in range(iters):
        tl.store(dst + base + step * BLOCK + lane, v)
    tl.store(tick_ptr + pid * 2, tl.sum(v, axis=0) * 0)
    tl.store(tick_ptr + pid * 2 + 1, tl.sum(v, axis=0) * 0)
"""

GRID = 56
KS = [1, 2, 4, 8, 16, 32, 64]


def whole(mod, name, tensors, K, BLOCK=1024):
    import torch_npu.profiler as prof
    import shutil
    import tempfile
    import csv
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")
    kn = getattr(mod, name)

    def fn():
        kn[(GRID,)](*tensors, tick, iters=K, BLOCK=BLOCK, num_warps=4,
                    compile_mode="simd", auto_simt_scope_mode="off",
                    logical_program_count_hint=GRID,
                    physical_vector_core_count_hint=56)

    for _ in range(5):
        fn()
    torch.npu.synchronize()
    trace = Path(tempfile.mkdtemp(prefix="e17e_"))
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(
                          str(trace))):
        torch.npu.synchronize()
        for _ in range(30):
            fn()
    torch.npu.synchronize()
    durs = []
    for tpath in sorted(trace.rglob("kernel_details.csv")):
        with tpath.open(newline="", errors="ignore") as h:
            for row in csv.DictReader(h):
                if "e17e" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]) * 1000)
    shutil.rmtree(trace, ignore_errors=True)
    return statistics.median(durs)


def main():
    gen_path = Path("/root/work/explore_contig/_gen_e17e.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_e17e", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    n = GRID * max(KS) * 1024
    src = torch.randn((n,), dtype=torch.float32, device="npu")
    dst = torch.zeros((n,), dtype=torch.float32, device="npu")
    tick = torch.zeros((GRID * 2,), dtype=torch.int64, device="npu")

    variants = {
        "null": (dst, tick),
        "mixed": (src, dst),
        "ld": (src, dst),
        "st": (dst, tick),
    }
    out = {}
    table = {}
    for name, tensors in variants.items():
        table[name] = {}
        for K in KS:
            w = whole(mod, name, tensors, K)
            table[name][K] = round(w, 1)
        print("%-6s %s" % (name, table[name]))
        sys.stdout.flush()
    print("\n== body(K) = variant(K) - null(K), per-iteration ns ==")
    hdr = "K     " + "".join("%8s" % n for n in ("mixed", "ld", "st"))
    print(hdr)
    body_curves = {"mixed": {}, "ld": {}, "st": {}}
    for K in KS:
        cells = "  %-4d " % K
        for n in ("mixed", "ld", "st"):
            b = (table[n][K] - table["null"][K]) / K
            b = max(b, 0.0)
            body_curves[n][K] = round(b, 1)
            cells += "%8.1f" % b
        print(cells)
    out = {"null": table["null"], "body": body_curves,
           "raw": table}
    Path("/root/work/explore_contig/e17e_results.json").write_text(
        json.dumps(out, indent=1))
    print("saved e17e_results.json")


main()
