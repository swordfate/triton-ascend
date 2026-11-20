#!/usr/bin/env python3
"""E33d: decompose the +71 ns/trip gap between the real memory_stream_l4
(271 ns/trip) and the clean probe M41 (199.5 ns/trip at g=56).

Variants (fp32, 4 loads + 1 store, e=1024 per stream):
  v0  contiguous, unmasked                  [= E33c M41, 199.5 anchor]
  v1  + mask on loads and store
  v2  + grid-stride addressing (jump 56*4KB per trip)
  v3  + both (the real kernel shape)
Sweep g {1, 56}, hot, K {16, 64} differencing.
"""
import argparse
import csv
import json
import shutil
import statistics
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

HEADER = "import triton\nimport triton.language as tl\n"

KERNEL_TMPL = """
@triton.jit(do_not_specialize=['iters', 'n_blocks'])
def e33d_%(tag)s(src1, src2, src3, src4, dst1, out, iters, n_blocks, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    for step in range(iters):
        block = pid + step * programs
%(addr)s
        mask = offsets < n_blocks * BLOCK
        v0 = tl.load(src1 + off0, mask=mask, other=0.0)
        v1 = tl.load(src2 + off1, mask=mask, other=0.0)
        v2 = tl.load(src3 + off2, mask=mask, other=0.0)
        v3 = tl.load(src4 + off3, mask=mask, other=0.0)
        acc0 += v0
        acc1 += v1 + v2 + v3
        tl.store(dst1 + %(doff)s, acc0 + acc1 + 1.0, mask=mask)
    tl.store(out + lane, (acc0 + acc1) * 0 + 1.0)
"""

ADDR = {
    "v0": """        off0 = pid * iters * BLOCK + step * BLOCK + lane
        off1 = off0 + NBLK
        off2 = off0 + 2 * NBLK
        off3 = off0 + 3 * NBLK
        offsets = off0
        doff = off0""",
    "v1": """        off0 = pid * iters * BLOCK + step * BLOCK + lane
        off1 = off0 + NBLK
        off2 = off0 + 2 * NBLK
        off3 = off0 + 3 * NBLK
        offsets = off0
        doff = off0""",
    "v2": """        off0 = (pid + step * programs) * BLOCK + lane
        off1 = off0 + NBLK
        off2 = off0 + 2 * NBLK
        off3 = off0 + 3 * NBLK
        offsets = off0
        doff = off0""",
    "v3": """        off0 = (pid + step * programs) * BLOCK + lane
        off1 = off0 + NBLK
        off2 = off0 + 2 * NBLK
        off3 = off0 + 3 * NBLK
        offsets = off0
        doff = off0""",
}
USEMASK = {"v0": False, "v1": True, "v2": False, "v3": True}
NBLK_DEF = "    NBLK = n_blocks * BLOCK\n"


def gen_module(path, e, tags):
    parts = [HEADER]
    for tag in tags:
        doff = "off0" if tag in ("v0", "v2") else "off0"
        body = KERNEL_TMPL % {"tag": tag, "addr": ADDR[tag], "doff": doff}
        body = body.replace("    for step in range(iters):",
                            NBLK_DEF + "    for step in range(iters):")
        if not USEMASK[tag]:
            body = body.replace(
                "        mask = offsets < n_blocks * BLOCK\n", "")
            body = body.replace("mask=mask, other=0.0", "")
            body = body.replace(", mask=mask)", ")")
        parts.append(body)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g33d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prof_med_us(fn, name_filter, pairs=6):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e33d_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=pairs),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        fn()
        torch.npu.synchronize()
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        for _ in range(pairs):
            fn()
            p.step()
    torch.npu.synchronize()
    durs = []
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for r in csv.DictReader(h):
                dur = r.get("Duration(us)")
                name = r.get("Name") or ""
                if dur and name_filter in name:
                    durs.append(float(dur))
    shutil.rmtree(trace, ignore_errors=True)
    return round(statistics.median(durs), 3) if durs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default=[1, 56], nargs="+", type=int)
    ap.add_argument("--out", default="/tmp/e33d.jsonl")
    args = ap.parse_args()
    e = 1024
    tags = ("v0", "v1", "v2", "v3")
    gen_path = "/tmp/_gen_e33d.py"
    gen_module(gen_path, e, tags)
    m = load_module(gen_path)
    opts = {"compile_mode": "simd", "num_warps": 4}

    NE = 56 * 64 * e * 4
    src = [torch.randn(NE, device="npu") for _ in range(4)]
    dst = torch.zeros(NE, device="npu")
    out = torch.zeros(e, device="npu")

    rows = []
    for tag in tags:
        for g in args.grids:
            d = {}
            for K in (16, 64):
                def fn(K=K, g=g, tag=tag):
                    # n_blocks = 56*64 so the mask stays all-true up to
                    # K=64 trips (we measure predication cost, not clip)
                    getattr(m, "e33d_%s" % tag)[(g,)](
                        src[0], src[1], src[2], src[3], dst, out, K, 3584,
                        e, **opts)
                d["D%d" % K] = prof_med_us(fn, "e33d_%s" % tag, pairs=6)
            slope = None
            if d.get("D16") is not None and d.get("D64"):
                slope = round((d["D64"] - d["D16"]) * 1000.0 / 48, 1)
            row = {"tag": tag, "e": e, "g": g, **d, "slope_ns": slope}
            rows.append(row)
            print("ROW", json.dumps(row), flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))
    print("E33D DONE", flush=True)


main()
