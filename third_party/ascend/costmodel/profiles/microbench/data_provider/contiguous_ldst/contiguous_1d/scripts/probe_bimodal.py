#!/usr/bin/env python3
"""双峰溯源：固定配置反复测量，记录 buffer 设备地址与耗时，看两者是否相关。"""
import argparse, json, statistics
from pathlib import Path
import torch, torch_npu, triton
import triton.language as tl

@triton.jit(do_not_specialize=["iters"])
def flat1d(src, dst, iters, N: tl.constexpr, WINMASK: tl.constexpr):
    lane = tl.arange(0, N)
    acc = tl.zeros((N,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + (step & WINMASK) * N + lane)
    tl.store(dst + lane, acc)

def timed(launch, k, reps):
    launch(k); torch.npu.synchronize()
    itf = triton.runtime.driver.active.get_device_interface()
    ev = [itf.Event(enable_timing=True) for _ in range(reps)]
    ev2 = [itf.Event(enable_timing=True) for _ in range(reps)]
    for a,b in zip(ev,ev2):
        a.record(); launch(k); b.record()
    itf.synchronize()
    return statistics.median([a.elapsed_time(b)*1000.0 for a,b in zip(ev,ev2)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--block",type=int,default=4096)
    ap.add_argument("--dtype",default="fp32")
    ap.add_argument("--pad-kb",type=int,default=0,help="先分配多少 KB 的占位张量，用来平移地址")
    ap.add_argument("--k-low",type=int,default=2048)
    ap.add_argument("--k-high",type=int,default=8192)
    ap.add_argument("--repeats",type=int,default=5)
    a=ap.parse_args()
    DT={"int8":torch.int8,"fp16":torch.float16,"fp32":torch.float32,"int64":torch.int64}
    dt=DT[a.dtype]; n=a.block; window=a.k_low; total=window*n+n
    keep=[]
    if a.pad_kb:
        keep.append(torch.zeros(a.pad_kb*1024//4, dtype=torch.float32, device="npu"))
    src=(torch.rand(total,dtype=torch.float32)*7).to(dt).npu()
    dst=torch.zeros(total,dtype=dt,device="npu")
    p_src, p_dst = src.data_ptr(), dst.data_ptr()
    def run(k): return flat1d[(1,)](src,dst,k,n,window-1,compile_mode="simd",num_warps=1)
    lo=timed(run,a.k_low,a.repeats); hi=timed(run,a.k_high,a.repeats)
    cost=(hi-lo)*1000.0/(a.k_high-a.k_low)
    rec={"pad_kb":a.pad_kb,"block":n,"dtype":a.dtype,"cost":cost,
         "src_ptr":p_src,"dst_ptr":p_dst,
         "src_mod_4k":p_src%4096,"src_mod_32k":p_src%32768,"src_mod_512":p_src%512,
         "lo_us":lo,"hi_us":hi,
         "lo_src_mod_512":p_src%512}
    a.manifest.parent.mkdir(parents=True,exist_ok=True)
    with a.manifest.open("a") as h: h.write(json.dumps(rec)+chr(10))
    print(json.dumps(rec))
    del keep
main()
