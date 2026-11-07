#!/usr/bin/env python3
"""Fill 14 table gaps with per_half=30 precision (accept any positive min)."""
import csv, json, sys, tempfile
from pathlib import Path
import torch, torch_npu
sys.path.insert(0, "/root/work")
from ss_campaign4 import ss_store, ss_tiny, DT, EB

def measure_pair(fA, nameA, fB, nameB, per_half=30):
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

gaps = json.load(open("/tmp/ss_gaps.json"))
out = open("/tmp/ss_gapfill.jsonl", "w")
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

for g in gaps:
    opts = dict(compile_mode="simt_only", superblock_factor=1, num_warps=g["warps"])
    s_, d_ = big(g["dtype"])
    d = d_[:g["block"]].contiguous()
    mk = int(g["masked"])
    fA = lambda: ss_store[(1,)](d, g["block"], mk, g["block"], **opts)
    fB = lambda: ss_tiny[(1,)](d, **opts)
    for _ in range(6):
        fA(); fB()
    torch.npu.synchronize()
    vals = []
    for attempt in range(3):
        mA, mB = measure_pair(fA, "ss_store", fB, "ss_tiny")
        if mA is None or mB is None:
            continue
        for a, b in zip(mA, mB):
            if a is not None and b is not None:
                vals.append((a - b) * 1000.0)
        if vals and min(vals) > 0:
            break
    rec = dict(g)
    if vals:
        med = sorted(vals)[len(vals)//2]
        rec.update(ns=round(med, 1), n_sessions=len(vals),
                   status="ok" if vals and min(vals) > 0 else "low")
    else:
        rec.update(status="bad")
    out.write(json.dumps(rec) + "\n")
    out.flush()
    print(g["dir"], "AB"[mk], g["dtype"], "W%d" % g["warps"], "by%d" % g["by"],
          "->", rec.get("ns"), rec["status"], flush=True)
print("GAPFILL_DONE")
