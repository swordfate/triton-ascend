#!/usr/bin/env python3
"""G-1: low-grid generalization probe.

Question (review doc N1): the direct row table only carries low-grid rows
(g1/g2/g4) for the b1024 n1_s0/n0_s1 families; all other families start at
g8.  Do hint<8 launches on OTHER byte families (b512/b2048/b4096) still
predict within tolerance, or do the g8 rows mis-price them?

Matrix: kernel in {compaction, scatter} x SUB in {128, 256, 512, 1024}
(chunk 512B/1KB/2KB/4KB) x grid in {2, 4, 8}.  SUB=256 is the method
control (b1024 HAS low-grid rows); grid=8 is the in-table control.
Prediction: costmodel report via auto_simt_scope_dump; measurement:
npu event timing, warmup 3, repeats 7 x batch 10, median.
"""
import json
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
def g1_compaction(inp, out, in_numel, out_numel, XBLOCK: tl.constexpr,
                  XBLOCK_SUB: tl.constexpr, STRIDE: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        src_index = index * STRIDE
        value = tl.load(inp + src_index, mask=src_index < in_numel)
        tl.store(out + index, value, mask=index < out_numel)

@triton.jit
def g1_scatter(inp, out, in_numel, out_numel, XBLOCK: tl.constexpr,
               XBLOCK_SUB: tl.constexpr, STRIDE: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        dst_index = index * STRIDE
        value = tl.load(inp + index, mask=index < in_numel)
        tl.store(out + dst_index, value, mask=dst_index < out_numel)
"""

K_ROUNDS = 8  # XBLOCK / XBLOCK_SUB


def run_variant(mod, kind, sub, grid, stride, report_path):
    xblock = sub * K_ROUNDS
    if kind == "compaction":
        in_numel = xblock * grid * stride
        out_numel = grid * xblock
        inp = torch.randn((in_numel,), dtype=torch.float32, device="npu")
        out = torch.empty((out_numel,), dtype=torch.float32, device="npu")
        expected = inp[::stride][:out_numel].contiguous()
        actual = out
    else:
        in_numel = xblock * grid
        out_numel = in_numel * stride
        inp = torch.randn((in_numel,), dtype=torch.float32, device="npu")
        out = torch.zeros((out_numel,), dtype=torch.float32, device="npu")
        expected = torch.zeros_like(out)
        expected[::stride] = inp
        actual = out

    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    kwargs = {
        "num_warps": 4,
        "compile_on_910_95": True,
        "compile_mode": "simd_simt",
        "superblock_factor": 1,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": int(props["num_vectorcore"]),
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(report_path),
    }
    kn = mod.g1_compaction if kind == "compaction" else mod.g1_scatter

    def launch():
        kn[(grid,)](inp, out, in_numel, out_numel, XBLOCK=xblock,
                    XBLOCK_SUB=sub, STRIDE=stride, **kwargs)

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for _ in range(3):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(7):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            launch()
        end.record()
        torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000.0 / 10)
    samples.sort()
    meas = statistics.median(samples)

    docs = []
    dec = json.JSONDecoder()
    s = report_path.read_text(errors="ignore")
    i = 0
    while i < len(s):
        rest = s[i:].lstrip()
        if not rest:
            break
        off = len(s) - i - len(rest)
        try:
            obj, end = dec.raw_decode(rest)
        except json.JSONDecodeError:
            break
        docs.append(obj)
        i = off + end
    kind_name, pred = "?", None
    if docs:
        r = docs[-1]
        kind_name = r.get("effective_decision_kind") or "?"
        p = (r.get("candidate_elapsed_costs") or {}).get(kind_name)
        if p is not None:
            pred = p / 1000.0
    err = (pred - meas) / meas * 100.0 if pred else float("nan")
    return kind_name, meas, pred, err


def main():
    gen_path = Path("/root/work/explore_contig/_gen_g1.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_g1", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    out_dir = Path("/tmp/G1_LOWGRID")
    out_dir.mkdir(exist_ok=True)
    print("%-12s %4s %5s %5s | %-14s %9s %9s %8s" %
          ("kernel", "SUB", "grid", "chunk", "decision", "meas_us",
           "pred_us", "err%"))
    rows = []
    for kind in ("compaction", "scatter"):
        for sub in (128, 256, 512, 1024):
            for grid in (2, 4, 8):
                rep = out_dir / f"{kind}_sub{sub}_g{grid}.route.json"
                rep.unlink(missing_ok=True)
                d, meas, pred, err = run_variant(
                    mod, kind, sub, grid, 16, rep)
                rows.append((kind, sub, grid, sub * 4, d, meas, pred, err))
                print("%-12s %4d %5d %5d | %-14s %9.3f %9.3f %+7.2f%%" %
                      (kind, sub, grid, sub * 4, d, meas, pred or -1, err))
                sys.stdout.flush()
    (out_dir / "summary.json").write_text(json.dumps(
        [{"kernel": r[0], "sub": r[1], "grid": r[2], "chunk_bytes": r[3],
          "decision": r[4], "meas_us": r[5], "pred_us": r[6], "err_pct": r[7]}
         for r in rows], indent=1))


main()
