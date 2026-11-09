#!/usr/bin/env python3
"""FormS single-shot calibration v5 (2026-11-08).

Improvement over v4: ABAB interleaved pairing (A,B,A,B,... x N pairs) inside
one profiler session; per-pair differences cancel block-level drift.  Fixed
address (L2-warm).  ~7 s per point.  Same kernels as v4 for comparability.
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

def measure_pair(fA, nameA, fB, nameB, pairs=16):
    total = pairs * 2
    trace = Path(tempfile.mkdtemp())
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=total),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace))) as prof:
        for _ in range(pairs):
            fA(); prof.step(); fB(); prof.step()
    torch.npu.synchronize()
    rows = []
    for p in sorted(trace.rglob("kernel_details.csv")):
        for row in csv.DictReader(p.open(newline="")):
            if row.get("Duration(us)"):
                rows.append((row.get("Name") or "", float(row["Duration(us)"])))
    if not rows:
        return None
    A = [d for (n, d) in rows if nameA in n]
    B = [d for (n, d) in rows if nameB in n]
    n = min(len(A), len(B))
    if n < pairs // 2:
        return None
    diffs = sorted((A[i] - B[i]) * 1000.0 for i in range(n))   # ns
    return diffs

DT = {"int8": torch.int8, "fp16": torch.float16, "fp32": torch.float32, "int64": torch.int64}
EB = {"int8": 1, "fp16": 2, "fp32": 4, "int64": 8}

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
           "dir": direction, "v": 5, "status": "ok"}
    diffs = None
    for attempt in range(2):
        diffs = measure_pair(fA, "ss_" + direction, fB, nB)
        if diffs is None:
            continue
        if diffs[len(diffs)//2] > 0:
            break
    if diffs is None:
        rec.update(status="bad", note="no-trace")
        out.write(json.dumps(rec) + "\n"); out.flush()
        return rec
    med = diffs[len(diffs)//2]
    q1 = diffs[len(diffs)//4]; q3 = diffs[(3*len(diffs))//4]
    if med <= 0:
        rec.update(status="bad", note="nonpositive", med=round(med,1))
    else:
        cv = (q3 - q1) / max(1e-9, med)
        rec.update(ns=round(med, 1), cv=round(cv, 4),
                   q=[round(q1,1), round(q3,1)], npairs=len(diffs))
        rec["status"] = "ok" if cv <= 0.20 else ("warn" if cv <= 0.40 else "noisy")
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtypes", default="int8,fp16,fp32,int64")
    ap.add_argument("--only", default="simt,simd", help="comma: simt,simd")
    args = ap.parse_args()
    DTS = args.dtypes.split(",")
    BYS = [64, 256, 512, 1024, 2048, 4096, 8192, 16384]
    WS = [1, 2, 4, 8, 16, 32, 64]
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
    todo = []
    if "simt" in args.only:
        for w in WS: todo.append(("simt_only", w))
    if "simd" in args.only:
        todo.append(("simd", 1))
    for mode, warps in todo:
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
                if count % 40 == 0:
                    print("... %d pts %.0fs" % (count, time.time() - t0), flush=True)
    print("DONE points=%d elapsed=%.0fs" % (count, time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
