#!/usr/bin/env python3
"""FormS single-shot calibration v4 (2026-11-07).

Fixed-address (L2-warm after first launch, matching real multi-grid kernels).
One profiler session per config: [A*10, B*10, A*10, B*10]; the chronological
row split yields two per-name session minima -> cv gate.  ~6 s per point.
"""
import argparse, csv, json, tempfile, time
from pathlib import Path
import torch, torch_npu, triton
import triton.language as tl

@triton.jit
def ss_load(src, dst, BLOCK: tl.constexpr, MASKED: tl.constexpr, valid):
    lane = tl.arange(0, BLOCK)
    if MASKED:
        x = tl.load(src + lane, mask=lane < valid, other=0)
    else:
        x = tl.load(src + lane)
    tl.store(dst + lane, x)

@triton.jit
def ss_ctl(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))

@triton.jit
def ss_store(dst, BLOCK: tl.constexpr, MASKED: tl.constexpr, valid):
    lane = tl.arange(0, BLOCK)
    if MASKED:
        tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty), mask=lane < valid)
    else:
        tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))

@triton.jit
def ss_tiny(dst):
    tl.store(dst, 0)

def measure_pair(fA, nameA, fB, nameB, per_half=10):
    total = per_half * 4
    trace = Path(tempfile.mkdtemp())
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=total),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace))) as prof:
        for _ in range(per_half):
            fA(); prof.step()
        for _ in range(per_half):
            fB(); prof.step()
        for _ in range(per_half):
            fA(); prof.step()
        for _ in range(per_half):
            fB(); prof.step()
    torch.npu.synchronize()
    rows = []
    for p in sorted(trace.rglob("kernel_details.csv")):
        for row in csv.DictReader(p.open(newline="")):
            if row.get("Duration(us)"):
                rows.append((row.get("Name") or "", float(row["Duration(us)"])))
    if not rows:
        return None, None
    half = len(rows) // 2
    def mins(name):
        s1 = [d for (n, d) in rows[:half] if name in n]
        s2 = [d for (n, d) in rows[half:] if name in n]
        return (min(s1) if s1 else None, min(s2) if s2 else None)
    return mins(nameA), mins(nameB)

DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32, "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}

def pair_cost(mA, mB):
    vals = []
    for a, b in zip(mA, mB):
        if a is None or b is None:
            continue
        vals.append((a - b) * 1000.0)
    return vals

def run_point(mode, warps, dtype, block, masked, direction, big_s, big_d, out):
    opts = dict(compile_mode=mode, num_warps=warps)
    if mode == "simt_only":
        opts["superblock_factor"] = 1
    s = big_s[:block].contiguous()
    d = big_d[:block].contiguous()
    if direction == "load":
        fA = lambda: ss_load[(1,)](s, d, block, masked, block, **opts)
        fB = lambda: ss_ctl[(1,)](d, block, **opts)
        nB = "ss_ctl"
    else:
        fA = lambda: ss_store[(1,)](d, block, masked, block, **opts)
        fB = lambda: ss_tiny[(1,)](d, **opts)
        nB = "ss_tiny"
    for _ in range(6):
        fA(); fB()
    torch.npu.synchronize()
    rec = {"mode": mode, "warps": warps, "dtype": dtype, "block": block,
           "by": block * EB[dtype], "eb": EB[dtype], "masked": bool(masked),
           "dir": direction, "status": "ok"}
    mA, mB = None, None
    for attempt in range(2):
        mA, mB = measure_pair(fA, "ss_" + direction, fB, nB)
        if mA is None or mB is None:
            continue
        vals = pair_cost(mA, mB)
        if vals and min(vals) > 0:
            break
    if mA is None or mB is None:
        rec.update(status="bad", note="no-trace")
        out.write(json.dumps(rec) + "\n"); out.flush()
        return rec
    vals = pair_cost(mA, mB)
    if not vals or min(vals) <= 0:
        rec.update(status="bad", note="nonpositive", mins=[round(v,1) for v in vals])
    else:
        med = sum(vals) / len(vals)
        cv = (max(vals) - min(vals)) / max(1e-9, abs(med))
        rec.update(ns=round(med, 1), cv=round(cv, 4), mins=[round(v, 1) for v in vals])
        rec["status"] = "ok" if cv <= 0.12 else ("warn" if cv <= 0.30 else "noisy")
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dtypes", default="int8,fp16,fp32,int64")
    args = ap.parse_args()
    DTS = args.dtypes.split(",")
    BYS = [1024, 4096] if args.pilot else [64, 256, 1024, 4096, 16384]
    WS = [1, 4] if args.pilot else [1, 2, 4, 8, 16, 32, 64]
    NP2 = [48, 768] if args.pilot else [24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288]
    bufs = {}
    def big(dtype):
        if dtype not in bufs:
            n = 1 << 20
            if DT[dtype] == torch.int64:
                src = torch.full((n,), 3, dtype=torch.int64, device="npu")
            else:
                src = (torch.randn(n) * 7).to(DT[dtype]).to("npu")
            bufs[dtype] = (src, torch.zeros(n, dtype=DT[dtype], device="npu"))
        return bufs[dtype]
    out = open(args.out, "a")
    count, t0 = 0, time.time()
    for mode, warps in [("simt_only", w) for w in WS] + [("simd", 1)]:
        for dtype in DTS:
            for by in BYS:
                block = by // EB[dtype]
                if block < 1 or block > 16384:
                    continue
                s, d = big(dtype)
                for direction in ("load", "store"):
                    run_point(mode, warps, dtype, block, 0, direction, s, d, out)
                    run_point(mode, warps, dtype, block, 1, direction, s, d, out)
                    count += 2
                if count % 60 == 0:
                    print("... %d pts %.0fs" % (count, time.time() - t0), flush=True)
    for dtype in DTS:
        s, d = big(dtype)
        for block in NP2:
            for masked in (0, 1):
                for direction in ("load", "store"):
                    run_point("simd", 1, dtype, block, masked, direction, s, d, out)
                    count += 1
    print("DONE points=%d elapsed=%.0fs" % (count, time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
