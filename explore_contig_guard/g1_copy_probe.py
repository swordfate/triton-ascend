#!/usr/bin/env python3
"""G-1b: pure-contiguous copy at low grids / various chunk sizes.

Isolates the contiguous-v2 path from mixed-stream composition: a plain
load->store copy loop (family n1_s1) at SUB in {128,256,512} x grid in
{2,4,8}.  Prediction via auto_simt_scope_dump; measurement via profiler
device Duration (median of >=50 launches), same as the 100-case suite.
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
from triton.runtime import driver  # noqa: E402

torch.npu.set_device(0)

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit
def g1_copy(inp, out, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        v = tl.load(inp + index, mask=index < numel)
        tl.store(out + index, v, mask=index < numel)
"""

K_ROUNDS = 8


def main():
    gen_path = Path("/root/work/explore_contig/_gen_g1c.py")
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_g1c", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    out_dir = Path("/tmp/G1_LOWGRID")
    out_dir.mkdir(exist_ok=True)
    props = driver.active.utils.get_device_properties(
        torch.npu.current_device())
    print("%-8s %5s %5s %6s | %-14s %9s %9s %8s" %
          ("kernel", "SUB", "grid", "chunk", "decision", "meas_us",
           "pred_us", "err%"))
    rows = []
    for sub in (128, 256, 512):
        for grid in (2, 4, 8):
            xblock = sub * K_ROUNDS
            numel = xblock * grid
            rep = out_dir / f"copy_sub{sub}_g{grid}.route.json"
            rep.unlink(missing_ok=True)
            inp = torch.randn((numel,), dtype=torch.float32, device="npu")
            out = torch.empty((numel,), dtype=torch.float32, device="npu")
            kwargs = {
                "num_warps": 4, "compile_mode": "simd_simt",
                "superblock_factor": 1,
                "logical_program_count_hint": grid,
                "physical_vector_core_count_hint":
                    int(props["num_vectorcore"]),
                "auto_simt_scope_mode": "auto",
                "auto_simt_scope_dump": str(rep),
            }

            def launch():
                mod.g1_copy[(grid,)](inp, out, numel, XBLOCK=xblock,
                                     XBLOCK_SUB=sub, **kwargs)

            launch()
            torch.npu.synchronize()
            torch.testing.assert_close(out, inp, rtol=1e-4, atol=1e-4)
            for _ in range(3):
                launch()
            torch.npu.synchronize()
            import torch_npu.profiler as prof
            trace = Path(tempfile.mkdtemp(prefix="g1c_"))

            def batch():
                for _ in range(10):
                    launch()

            for _ in range(3):
                batch()
            torch.npu.synchronize()
            with prof.profile(activities=[prof.ProfilerActivity.NPU],
                              on_trace_ready=prof.tensorboard_trace_handler(
                                  str(trace))):
                torch.npu.synchronize()
                for _ in range(5):
                    batch()
            torch.npu.synchronize()
            durs = []
            for tpath in sorted(trace.rglob("kernel_details.csv")):
                with tpath.open(newline="", errors="ignore") as h:
                    for row in csv.DictReader(h):
                        if "g1_copy" in (row.get("Name") or ""):
                            durs.append(float(row["Duration(us)"]))
            shutil.rmtree(trace, ignore_errors=True)
            meas = statistics.median(durs) if durs else float("nan")
            dec = json.JSONDecoder()
            s = rep.read_text(errors="ignore")
            docs = []
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
            kind, pred = "?", None
            if docs:
                r = docs[-1]
                kind = r.get("effective_decision_kind") or "?"
                pv = (r.get("candidate_elapsed_costs") or {}).get(kind)
                if pv is not None:
                    pred = pv / 1000.0
            err = (pred - meas) / meas * 100.0 if pred else float("nan")
            rows.append({"kernel": "copy", "sub": sub, "grid": grid,
                         "chunk_bytes": sub * 4, "decision": kind,
                         "meas_us": meas, "pred_us": pred, "err_pct": err})
            print("%-8s %5d %5d %6d | %-14s %9.3f %9.3f %+7.2f%%" %
                  ("copy", sub, grid, sub * 4, kind, meas, pred or -1, err))
            sys.stdout.flush()
    with (out_dir / "summary_copy.json").open("w") as h:
        json.dump(rows, h, indent=1)


main()
