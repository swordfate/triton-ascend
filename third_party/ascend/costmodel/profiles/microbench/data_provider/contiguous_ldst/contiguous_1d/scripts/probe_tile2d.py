#!/usr/bin/env python3
"""2-D contiguous tile probe: measures (R,C) and the equivalent flat 1-D block."""
import argparse, json, statistics
from pathlib import Path
import torch, torch_npu, triton
import triton.language as tl

DT = {"int8":torch.int8,"fp16":torch.float16,"fp32":torch.float32,"int64":torch.int64}
EB = {"int8":1,"fp16":2,"fp32":4,"int64":8}

@triton.jit(do_not_specialize=["iters"])
def tile2d(src, dst, iters, R: tl.constexpr, C: tl.constexpr, WINMASK: tl.constexpr, direction: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    off = r * C + c
    n: tl.constexpr = R * C
    if direction == 0:
        acc = tl.zeros((R, C), src.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(src + (step & WINMASK) * n + off)
        tl.store(dst + off, acc)
    else:
        v = tl.full((R, C), 1, dst.dtype.element_ty)
        for step in range(iters):
            tl.store(dst + (step & WINMASK) * n + off, v + step)

@triton.jit(do_not_specialize=["iters"])
def flat1d(src, dst, iters, N: tl.constexpr, WINMASK: tl.constexpr, direction: tl.constexpr):
    lane = tl.arange(0, N)
    if direction == 0:
        acc = tl.zeros((N,), src.dtype.element_ty)
        for step in range(iters):
            acc += tl.load(src + (step & WINMASK) * N + lane)
        tl.store(dst + lane, acc)
    else:
        v = tl.full((N,), 1, dst.dtype.element_ty)
        for step in range(iters):
            tl.store(dst + (step & WINMASK) * N + lane, v + step)

def timed(launch, k, reps):
    launch(k); torch.npu.synchronize()
    itf = triton.runtime.driver.active.get_device_interface()
    ev = [itf.Event(enable_timing=True) for _ in range(reps)]
    ev2 = [itf.Event(enable_timing=True) for _ in range(reps)]
    for a,b in zip(ev,ev2):
        a.record(); launch(k); b.record()
    itf.synchronize()
    s=[a.elapsed_time(b)*1000.0 for a,b in zip(ev,ev2)]
    return statistics.median(s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--mode",choices=("simd","simt_only"),required=True)
    ap.add_argument("--direction",choices=("load","store"),required=True)
    ap.add_argument("--dtype",required=True)
    ap.add_argument("--rows",type=int,required=True)
    ap.add_argument("--cols",type=int,required=True)
    ap.add_argument("--warps",type=int,default=4)
    ap.add_argument("--k-low",type=int,default=2048)
    ap.add_argument("--k-high",type=int,default=8192)
    ap.add_argument("--repeats",type=int,default=5)
    a=ap.parse_args()
    n=a.rows*a.cols
    window=a.k_low
    total=window*n+n
    dt=DT[a.dtype]; eb=EB[a.dtype]
    torch.manual_seed(23)
    src=(torch.rand(total,dtype=torch.float32)*7).to(dt).npu()
    dst=torch.zeros(total,dtype=dt,device="npu")
    direction=0 if a.direction=="load" else 1
    opt={"compile_mode":a.mode,"num_warps":a.warps}
    if a.mode=="simt_only": opt["superblock_factor"]=1
    def run2d(k):
        return tile2d[(1,)](src,dst,k,a.rows,a.cols,window-1,direction,**opt)
    def run1d(k):
        return flat1d[(1,)](src,dst,k,n,window-1,direction,**opt)
    out={"mode":a.mode,"direction":a.direction,"dtype":a.dtype,"rows":a.rows,"cols":a.cols,
         "n":n,"bytes":n*eb,"warps":a.warps}
    try:
        lo=timed(run2d,a.k_low,a.repeats); hi=timed(run2d,a.k_high,a.repeats)
        out["cost_2d"]=(hi-lo)*1000.0/(a.k_high-a.k_low)
    except Exception as e:
        out["cost_2d"]=None; out["err_2d"]=str(e)[:200]
    try:
        lo=timed(run1d,a.k_low,a.repeats); hi=timed(run1d,a.k_high,a.repeats)
        out["cost_1d"]=(hi-lo)*1000.0/(a.k_high-a.k_low)
    except Exception as e:
        out["cost_1d"]=None; out["err_1d"]=str(e)[:200]
    if out.get("cost_2d") and out.get("cost_1d"):
        out["ratio_2d_over_1d"]=out["cost_2d"]/out["cost_1d"]
    out["status"]="ok" if out.get("cost_2d") and out.get("cost_1d") else "fail"
    a.manifest.parent.mkdir(parents=True,exist_ok=True)
    with a.manifest.open("a") as h: h.write(json.dumps(out)+chr(10))
    print(json.dumps(out))
main()
