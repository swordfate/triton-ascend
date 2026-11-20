#!/usr/bin/env python3
"""E28b: simt mixed-body composition law (additive vs per-128B line tax).

Bodies (fp32, simt, grid=1, K=64, SYS_CNT windows via clock64 hook):
  loads-only anchors per nl in {1,2,4} (bytes {4KB,16KB})
  mixed (nl,ns,e) in {(1,1,1024),(2,1,1024),(2,2,1024),(1,2,1024),
                      (1,1,4096),(2,2,4096)}
  store-only anchors (1,1024),(1,4096)
Scalar readback after the window forces store drain (vector readback of a
just-stored region crashes the simt backend).
"""
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/work/triton-ascend/explore_contig_guard/probes")
from e27b_fit2 import clock64_compilation, replace_markers  # noqa: E402

NS_TO_TICK = 1.00008
ITERS = 64
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l,~{memory}", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""
HEADER = "import triton\nimport triton.language as tl\n\n" + COMMON

MIX = [(1, 1, 1024), (2, 1, 1024), (2, 2, 1024), (1, 2, 1024),
       (1, 1, 4096), (2, 2, 4096)]
LOADS = [(1, 1024), (2, 1024), (1, 4096), (2, 4096)]
STORES = [(1, 1024), (1, 4096)]


def gen_module(path):
    parts = [HEADER]
    for nl, e in LOADS:
        parts.append("""
@triton.jit
def e28b_L%d_%d(src1, src2, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %d)
    a0 = tl.zeros((%d,), dtype=tl.float32)
    a1 = tl.zeros((%d,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(iters):
        p = step * %d + lane
        a0 += tl.load(src1 + p)
        a1 += tl.load(src2 + p)
    t1 = _sys_cnt()
    tl.store(out + lane, a0 * 0 + a1 * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (nl, e, e, e, e, e))
    for ns, e in STORES:
        parts.append("""
@triton.jit
def e28b_S%d_%d(dst1, dst2, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %d)
    t0 = _sys_cnt()
    for step in range(iters):
        tl.store(dst1 + step * %d + lane, 1.0)
    t1 = _sys_cnt()
    r = tl.load(dst1 + lane)
    tl.store(out + lane, r * 0 + 1.0)
    tl.debug_barrier()
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (ns, e, e, e))
    for nl, ns, e in MIX:
        parts.append("""
@triton.jit
def e28b_M%d_%d_%d(src1, src2, dst1, dst2, out, tick_ptr, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, %d)
    a0 = tl.zeros((%d,), dtype=tl.float32)
    a1 = tl.zeros((%d,), dtype=tl.float32)
    t0 = _sys_cnt()
    for step in range(iters):
        p = step * %d + lane
        a0 += tl.load(src1 + p)
        a1 += tl.load(src2 + p)
        tl.store(dst1 + p, a0 + 1.0)
        tl.store(dst2 + p, a1 + 2.0)
    t1 = _sys_cnt()
    tl.store(out + lane, a0 * 0 + a1 * 0 + 1.0)
    tl.store(tick_ptr + 0, t0)
    tl.store(tick_ptr + 1, t1)
""" % (nl, ns, e, e, e, e, e))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e28b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def med_ns(fn, tick, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    xs = []
    for _ in range(reps):
        fn()
        torch.npu.synchronize()
        xs.append((int(tick[1].item()) - int(tick[0].item())) * NS_TO_TICK)
    return statistics.median(xs)


def main():
    gen_path = "/tmp/_gen_e28b.py"
    gen_module(gen_path)
    m = load_module(gen_path)

    src1 = torch.randn(16 * 1024, device="npu")
    src2 = torch.randn(16 * 1024, device="npu")
    dst1 = torch.zeros(16 * 1024, device="npu")
    dst2 = torch.zeros(16 * 1024, device="npu")
    out = torch.zeros(1024, device="npu")
    tick = torch.zeros(2, dtype=torch.int64, device="npu")
    opts = {"compile_mode": "simt_only", "num_warps": 4,
            "superblock_factor": 1}

    def meas(fn):
        with clock64_compilation():
            return med_ns(fn, tick, warmup=5, reps=30)

    res = {}
    for nl, e in LOADS:
        key = "L%d_%d" % (nl, e)
        res[key] = meas(lambda: getattr(m, "e28b_%s" % key)[(1,)](
            src1, src2, out, tick, ITERS, e, **opts))
    for ns, e in STORES:
        key = "S%d_%d" % (ns, e)
        res[key] = meas(lambda: getattr(m, "e28b_%s" % key)[(1,)](
            dst1, dst2, out, tick, ITERS, e, **opts))
    for nl, ns, e in MIX:
        key = "M%d_%d_%d" % (nl, ns, e)
        res[key] = meas(lambda: getattr(m, "e28b_%s" % key)[(1,)](
            src1, src2, dst1, dst2, out, tick, ITERS, e, **opts))
    print(json.dumps(res, indent=1))
    Path("/tmp/e28b_simt_mix.json").write_text(json.dumps(res, indent=1))
    print("E28B DONE", flush=True)


if __name__ == "__main__":
    main()
