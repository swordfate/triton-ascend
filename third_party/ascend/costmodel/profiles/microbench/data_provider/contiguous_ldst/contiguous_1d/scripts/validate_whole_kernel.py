#!/usr/bin/env python3
"""整核验证 v2：iters 作 constexpr，TTIR 里 trip count 静态可知，CostModel 直接用 950PR 硬件配置。"""
import argparse, csv, json, re, shutil
from pathlib import Path
import torch, torch_npu, triton
import triton.language as tl
from triton.backends.ascend.runtime.costmodel_runtime import run_costmodel

DT = {"int8":torch.int8,"fp16":torch.float16,"fp32":torch.float32,"int64":torch.int64}
# 950PR 硬件配置：源码树里那份
HW = "/home/c00946898/triton-ascend/third_party/ascend/costmodel/configs/ascend_davidv100.json"

@triton.jit
def contig_kernel(a, b, out, RO: tl.constexpr, CO: tl.constexpr,
                  WINMASK: tl.constexpr, KIND: tl.constexpr, ITERS: tl.constexpr):
    if RO == 1:
        off = tl.arange(0, CO)
    else:
        r = tl.arange(0, RO)[:, None]
        c = tl.arange(0, CO)[None, :]
        off = r * CO + c
    n: tl.constexpr = RO * CO
    if KIND == 0:
        for i in range(ITERS):
            v = tl.load(a + (i & WINMASK) * n + off)
            tl.store(out + (i & WINMASK) * n + off, v)
    else:
        for i in range(ITERS):
            v = tl.load(a + (i & WINMASK) * n + off)
            w = tl.load(b + (i & WINMASK) * n + off)
            tl.store(out + (i & WINMASK) * n + off, v + w)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--kind",choices=("copy","add3"),required=True)
    ap.add_argument("--mode",choices=("simd","simt_only"),required=True)
    ap.add_argument("--dtype",required=True)
    ap.add_argument("--rows",type=int,required=True)
    ap.add_argument("--cols",type=int,required=True)
    ap.add_argument("--warps",type=int,required=True)
    ap.add_argument("--iters",type=int,default=2048)
    ap.add_argument("--window",type=int,default=2048)
    ap.add_argument("--repeats",type=int,default=5)
    ap.add_argument("--hw",default=HW)
    ap.add_argument("--skip-measure",action="store_true")
    a=ap.parse_args()
    EB={"int8":1,"fp16":2,"fp32":4,"int64":8}
    n=a.rows*a.cols; eb=EB[a.dtype]; w=a.window
    total=w*n; dt=DT[a.dtype]
    torch.manual_seed(23)
    src=(torch.rand(total,dtype=torch.float32)*7).to(dt).npu()
    src2=(torch.rand(total,dtype=torch.float32)*5).to(dt).npu()
    dst=torch.zeros(total,dtype=dt,device="npu")
    opt={"compile_mode":a.mode,"num_warps":a.warps}
    if a.mode=="simt_only": opt["superblock_factor"]=1
    kind=0 if a.kind=="copy" else 1
    tag="vk_%s_%s_%s_%dx%d_w%d_i%d" % (a.kind,a.mode,a.dtype,a.rows,a.cols,a.warps,a.iters)
    def launch():
        return contig_kernel[(1,)](src,src2,dst,a.rows,a.cols,w-1,kind,a.iters,**opt)
    rec=dict(kind=a.kind,mode=a.mode,dtype=a.dtype,rows=a.rows,cols=a.cols,n=n,
             bytes=n*eb,warps=a.warps,iters=a.iters,window=w,tag=tag,
             hw=Path(a.hw).name)
    try:
        h=launch(); torch.npu.synchronize()
        ttir=h.asm["ttir"]
        d=Path("/tmp/vk_ttir"); d.mkdir(parents=True,exist_ok=True)
        (d/(tag+".ttir")).write_text(ttir)
        rec["ttir_path"]=str(d/(tag+".ttir"))
    except Exception as e:
        rec["status"]="fail"; rec["err"]="compile: "+str(e)[:160]
        a.manifest.parent.mkdir(parents=True,exist_ok=True)
        with a.manifest.open("a") as f: f.write(json.dumps(rec)+chr(10))
        print(json.dumps(rec)); return
    try:
        args=["-ascend-perf-model"]
        if a.hw and Path(a.hw).exists(): args.append("hardware-config=%s"%a.hw)
        out=run_costmodel(ttir, args)
        txt = out if isinstance(out,str) else json.dumps(out)
        m=re.search(r"Estimated Time:\s+([0-9.]+)\s*us", txt or "")
        rec["costmodel_us"]=float(m.group(1)) if m else None
        if rec["costmodel_us"] is None: rec["cm_raw"]=(txt or "")[-500:]
    except Exception as e:
        rec["costmodel_us"]=None; rec["cm_err"]=str(e)[:200]
    if not a.skip_measure:
        try:
            trace=Path("/tmp/whole_kernel")/tag
            if trace.exists(): shutil.rmtree(trace)
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=1,warmup=1,active=a.repeats),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace))) as prof:
                for _ in range(a.repeats):
                    launch(); prof.step()
            torch.npu.synchronize()
            durs=[]
            for path in sorted(trace.rglob("kernel_details.csv")):
                with path.open(newline="") as st:
                    for row in csv.DictReader(st):
                        if row.get("Duration(us)") and (row.get("Name") or "").strip()=="contig_kernel":
                            durs.append(float(row["Duration(us)"]))
            if durs: rec["kernel_us"]=min(durs); rec["measured"]="ok"
            else: rec["measured"]="no rows"
        except Exception as e:
            rec["measured"]="err: "+str(e)[:120]
    if rec.get("costmodel_us") and rec.get("kernel_us"):
        rec["err_pct"]=(rec["costmodel_us"]-rec["kernel_us"])/rec["kernel_us"]*100
        rec["status"]="ok"
    else:
        rec["status"]="partial"
    a.manifest.parent.mkdir(parents=True,exist_ok=True)
    with a.manifest.open("a") as f: f.write(json.dumps(rec)+chr(10))
    print(json.dumps(rec))
main()
