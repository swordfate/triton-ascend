#!/usr/bin/env python3
"""bench_guard_contig.py - v2 contiguous guard suite.

24 self-created contiguous load/store kernels (NOT from the 100-case set),
each forced-measured under 4 combos:
    simd F1 (num_warps=4)
    simt F1 / F2 / F4 (num_warps=1, superblock_factor)
Predictions come from the route model's own route.json dumps:
    candidate_elapsed_costs.all_simd          (auto compile @ warps=4)
    candidate_elapsed_costs_by_factor         (auto compile @ warps=1)
Whole-kernel prediction vs profiler median duration, target <=10%.
"""
import argparse
import json
import statistics
import shutil
import csv
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.runtime import driver

TT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32,
      "i64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "i64": 8}

HEADER = "import triton\nimport triton.language as tl\n\n"
GEN = HEADER + """
@triton.jit(do_not_specialize=['n'])
def gc_scopy(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)

@triton.jit(do_not_specialize=['n'])
def gc_sadd(src, src2, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    y = tl.load(src2 + offs, mask=mask)
    tl.store(dst + offs, x + y, mask=mask)

@triton.jit(do_not_specialize=['n'])
def gc_sload(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + pid, tl.sum(x, axis=0))

@triton.jit(do_not_specialize=['n'])
def gc_sstore(dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.full((BLOCK,), 1, dst.dtype.element_ty), mask=mask)

@triton.jit
def gc_lcopy(src, dst, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    for step in range(iters):
        x = tl.load(src + base + step * BLOCK + lane)
        tl.store(dst + base + step * BLOCK + lane, x + 1)

@triton.jit
def gc_ladd(src, src2, dst, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    for step in range(iters):
        x = tl.load(src + base + step * BLOCK + lane)
        y = tl.load(src2 + base + step * BLOCK + lane)
        tl.store(dst + base + step * BLOCK + lane, x + y)

@triton.jit
def gc_lload(src, dst, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + base + step * BLOCK + lane)
    tl.store(dst + pid, tl.sum(acc, axis=0))

@triton.jit
def gc_lstore(dst, iters: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = pid * iters * BLOCK
    for step in range(iters):
        tl.store(dst + base + step * BLOCK + lane,
                 tl.full((BLOCK,), 1, dst.dtype.element_ty))
"""

# name: (kind, dtype, BLOCK, K (0=straight), grid)
CASES = {
    "A_copy_f32_4k_g256":   ("scopy", "fp32", 1024, 0, 256),
    "A_copy_f32_4k_g64":    ("scopy", "fp32", 1024, 0, 64),
    "A_copy_f16_2k_g256":   ("scopy", "fp16", 1024, 0, 256),
    "A_copy_i8_1k_g256":    ("scopy", "int8", 1024, 0, 256),
    "A_copy_i64_8k_g256":   ("scopy", "i64",  1024, 0, 256),
    "A_add_f32_4k_g256":    ("sadd",  "fp32", 1024, 0, 256),
    "A_add_f16_2k_g128":    ("sadd",  "fp16", 1024, 0, 128),
    "A_load_f32_4k_g256":   ("sload", "fp32", 1024, 0, 256),
    "A_store_f32_4k_g256":  ("sstore", "fp32", 1024, 0, 256),
    "A_store_f16_2k_g256":  ("sstore", "fp16", 1024, 0, 256),
    "A_copy_f32_4k_g1024":  ("scopy", "fp32", 1024, 0, 1024),
    "A_add_f32_4k_g1024":   ("sadd",  "fp32", 1024, 0, 1024),
    "B_copy_f32_4k_k16_g56":  ("lcopy", "fp32", 1024, 16, 56),
    "B_copy_f32_4k_k64_g56":  ("lcopy", "fp32", 1024, 64, 56),
    "B_copy_f32_4k_k4_g8":    ("lcopy", "fp32", 1024, 4, 8),
    "B_add_f32_4k_k16_g56":   ("ladd",  "fp32", 1024, 16, 56),
    "B_add_f32_4k_k64_g224":  ("ladd",  "fp32", 1024, 64, 224),
    "B_load_f32_4k_k64_g56":  ("lload", "fp32", 1024, 64, 56),
    "B_store_f32_4k_k16_g56": ("lstore", "fp32", 1024, 16, 56),
    "B_copy_f16_2k_k16_g56":  ("lcopy", "fp16", 1024, 16, 56),
    "B_copy_i8_1k_k16_g56":   ("lcopy", "int8", 1024, 16, 56),
    "B_store_i64_8k_k4_g56":  ("lstore", "i64", 1024, 4, 56),
    "B_copy_f32_4k_k8_g56":   ("lcopy", "fp32", 1024, 8, 56),
    "B_copy_f32_4k_k32_g56":  ("lcopy", "fp32", 1024, 32, 56),
}


def route_kwargs(grid, warps, dump):
    vc = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    kw = {"num_warps": warps, "compile_on_910_95": True,
          "compile_mode": "simd_simt", "superblock_factor": 1,
          "logical_program_count_hint": grid,
          "physical_vector_core_count_hint": vc,
          "auto_simt_scope_mode": "auto"}
    if dump:
        kw["auto_simt_scope_dump"] = dump
    return kw


def meas_kwargs(mode, warps, factor):
    kw = {"num_warps": warps, "compile_on_910_95": True,
          "compile_mode": mode}
    if mode == "simt_only":
        kw["superblock_factor"] = factor
    return kw


def profile_median(fn, kernel_name, launches=28):
    import tempfile
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="guard_"))
    config = prof._ExperimentalConfig(
        aic_metrics=prof.AiCMetrics.PipeUtilization,
        profiler_level=prof.ProfilerLevel.Level1)
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False,
            schedule=prof.schedule(wait=0, warmup=2, active=20, repeat=1,
                                   skip_first=2),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace)),
            experimental_config=config) as p:
        for _ in range(launches):
            fn()
            p.step()
    torch.npu.synchronize()
    vals = []
    for path in sorted(trace.rglob("op_statistic.csv")):
        with path.open(newline="") as h:
            for row in csv.DictReader(h):
                if kernel_name in (row.get("OP Type") or ""):
                    try:
                        vals.append(float(row.get("Avg Time(us)")))
                    except (TypeError, ValueError):
                        pass
    shutil.rmtree(trace, ignore_errors=True)
    if not vals:
        return None, 0
    return statistics.median(vals) * 1000.0, len(vals)  # ns


def build_case(mod, name, cfg):
    kind, dt, block, K, grid = cfg
    ttype = TT[dt]
    if kind in ("scopy", "sadd", "sload"):
        n = grid * block
    elif kind == "sstore":
        n = grid * block
    else:
        n = grid * max(K, 1) * block
    src = torch.randn(n, dtype=torch.float32, device="npu").to(ttype)
    src2 = torch.randn(n, dtype=torch.float32, device="npu").to(ttype) \
        if kind in ("sadd", "ladd") else src
    dst = torch.zeros(n, dtype=ttype, device="npu")
    return src, src2, dst


def load_route_docs(path):
    """Parse possibly-concatenated JSON docs; return the last (freshest)."""
    import json as _json
    s = Path(path).read_text()
    dec = _json.JSONDecoder()
    docs = []
    i = 0
    while i < len(s):
        rest = s[i:].lstrip()
        if not rest:
            break
        skip = len(s) - i - len(rest)
        obj, end = dec.raw_decode(rest)
        docs.append(obj)
        i = i + skip + end
    return docs


def route_pred(path):
    best = {}
    for doc in load_route_docs(path):
        ce = doc.get("candidate_elapsed_costs", {}) or {}
        byf = doc.get("candidate_elapsed_costs_by_factor", {}) or {}
        for k, v in ce.items():
            if v is not None:
                best[k] = v
        for k, v in byf.items():
            if v is not None:
                best[k] = v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default="/tmp/GUARD_CONTIG/guard_table.json")
    ap.add_argument("--repeats", type=int, default=1,
                    help="profiler sessions per combo")
    args = ap.parse_args()
    names = args.cases or list(CASES.keys())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gen_path = out_path.parent / "_gen_guard_contig.py"
    gen_path.write_text(GEN, encoding="utf-8")
    import importlib.util
    sspec = importlib.util.spec_from_file_location("_gen_guard_contig", str(gen_path))
    mod = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(mod)

    results = []
    for name in names:
        cfg = CASES[name]
        kind, dt, block, K, grid = cfg
        src, src2, dst = build_case(mod, name, cfg)
        kn = "gc_" + kind
        # ---- launches per combo ----
        if kind == "scopy":
            launch = lambda kw: mod.gc_scopy[(grid,)](src, dst, src.numel(), block, **kw)
            expected = src.clone()
            check = lambda: torch.testing.assert_close(dst.float(), expected.float(), rtol=2e-2, atol=2e-2)
        elif kind == "sadd":
            launch = lambda kw: mod.gc_sadd[(grid,)](src, src2, dst, src.numel(), block, **kw)
            expected = (src.float() + src2.float()).to(dst.dtype)
            check = lambda: torch.testing.assert_close(dst.float(), expected.float(), rtol=2e-2, atol=2e-2)
        elif kind == "sload":
            launch = lambda kw: mod.gc_sload[(grid,)](src, dst, src.numel(), block, **kw)
            expected = src.float().reshape(grid, block).sum(1)
            check = lambda: torch.testing.assert_close(
                dst[:grid].float(), expected, rtol=2e-2, atol=2e-2)
        elif kind == "sstore":
            launch = lambda kw: mod.gc_sstore[(grid,)](dst, dst.numel(), block, **kw)
            expected = torch.ones_like(dst)
            check = lambda: torch.testing.assert_close(dst.float(), expected.float())
        elif kind == "lcopy":
            launch = lambda kw: mod.gc_lcopy[(grid,)](src, dst, K, block, **kw)
            expected = (src.float() + 1).to(dst.dtype)
            check = lambda: torch.testing.assert_close(dst.float(), expected.float(), rtol=2e-2, atol=2e-2)
        elif kind == "ladd":
            launch = lambda kw: mod.gc_ladd[(grid,)](src, src2, dst, K, block, **kw)
            expected = (src.float() + src2.float()).to(dst.dtype)
            check = lambda: torch.testing.assert_close(dst.float(), expected.float(), rtol=2e-2, atol=2e-2)
        elif kind == "lload":
            launch = lambda kw: mod.gc_lload[(grid,)](src, dst, K, block, **kw)
            expected = src.float().reshape(grid, K, block).sum((1, 2))
            check = lambda: torch.testing.assert_close(
                dst[:grid].float(), expected, rtol=2e-2, atol=2e-2)
        elif kind == "lstore":
            launch = lambda kw: mod.gc_lstore[(grid,)](dst, K, block, **kw)
            expected = torch.ones_like(dst)
            check = lambda: torch.testing.assert_close(dst.float(), expected.float())

        # ---- predictions: auto compiles at w4 (simd) and w1 (simt) ----
        preds = {}
        try:
            dump4 = str(out_path.parent / (name + "_w4.route.json"))
            launch(route_kwargs(grid, 4, dump4))
            p4 = route_pred(dump4)
            preds["simd_f1"] = p4.get("all_simd")
            dump1 = str(out_path.parent / (name + "_w1.route.json"))
            launch(route_kwargs(grid, 1, dump1))
            p1 = route_pred(dump1)
            preds["simt_f1"] = p1.get("all_simt_only_f1") or p1.get("all_simt_only")
            preds["simt_f2"] = p1.get("all_simt_only_f2")
            preds["simt_f4"] = p1.get("all_simt_only_f4")
        except Exception as e:
            print("  PREDICT FAIL %s: %s" % (name, e), flush=True)
        check(); torch.npu.synchronize()

        # ---- forced measurements ----
        combos = [("simd_f1", "simd", 4, 1),
                  ("simt_f1", "simt_only", 1, 1),
                  ("simt_f2", "simt_only", 1, 2),
                  ("simt_f4", "simt_only", 1, 4)]
        for cname, mode, warps, factor in combos:
            kw = meas_kwargs(mode, warps, factor)
            fn = lambda: launch(kw)
            for _ in range(5):
                fn()
            torch.npu.synchronize()
            med, cnt = profile_median(fn, kn)
            best = med
            for _ in range(args.repeats - 1):
                m2, _ = profile_median(fn, kn)
                if m2 and (not best or m2 < best):
                    best = m2
            pred = preds.get(cname)
            err = (100.0 * (pred - best) / best) if (pred and best) else None
            row = {"case": name, "combo": cname, "grid": grid, "K": K,
                   "block": block, "dtype": dt,
                   "pred_ns": round(pred, 1) if pred else None,
                   "meas_ns": round(best, 1) if best else None,
                   "err_pct": round(err, 2) if err is not None else None,
                   "pass": bool(err is not None and abs(err) <= 10.0)}
            results.append(row)
            print("  %-24s %-8s pred=%s meas=%s err=%s %s" % (
                name, cname,
                "%.0f" % pred if pred else "n/a",
                "%.0f" % best if best else "n/a",
                "%+.1f%%" % err if err is not None else "n/a",
                "PASS" if row["pass"] else "FAIL"), flush=True)
            json.dump(results, open(out_path, "w"), indent=1)
    npass = sum(1 for r in results if r["pass"])
    print("GUARD SUITE: %d/%d rows within 10%%" % (npass, len(results)))
    json.dump(results, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
