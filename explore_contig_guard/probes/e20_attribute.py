#!/usr/bin/env python3
"""E20: attribute the 4 rows-lost cases - measure each real kernel body's
in-loop window and compare with its stage prediction.

Bodies (copied from the harness benches):
  muls1m : muls_add body, 2x4KB load + FMA + 4KB store, grid=56,
           K sweep {19,37,74} -> low-K drain curve
  permute: permute_reshape_fused body (2 shells x 12 x [2 contig 2KB
           loads + strided 32B load + strided 32B store]), grid=32
  softmax: fused_softmax body (2KB masked load 257/512 + exp/max/sum +
           2KB masked store), grid=32, 4 rows/program
  scatter: strided_scatter body (simt, 8 subs x [256B load + scatter
           store stride 16]), grid=2, clock64 timing

Stage predictions (norows): muls1m 2.160us/19it=113.7ns, permute
2.414us, softmax 1.173us/4rows, scatter 2.318us.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"
SIMD_COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""

MULS = """
@triton.jit
def e20_muls(x, y, out, tick_ptr, n_blocks, K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    t0 = _sys_cnt()
    for step in range(K):
        block_id = pid + step * nprog
        off = block_id * BLOCK + lane
        m = block_id < n_blocks
        v0 = tl.load(x + off, mask=m, other=0.0)
        v1 = tl.load(y + off, mask=m, other=0.0)
        tl.store(out + off, v0 * 2.5 + v1, mask=m)
    t1 = _sys_cnt()
    tl.store(tick_ptr + pid * 2, t0)
    tl.store(tick_ptr + pid * 2 + 1, t1)
"""

PERMUTE = """
@triton.jit
def e20_permute(a, b, c, out, tick_ptr, BLOCK1: tl.constexpr,
                BLOCK1_SUB: tl.constexpr, BLOCK2: tl.constexpr,
                S: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    sbase = tl.arange(0, BLOCK1_SUB)
    dbase = tl.arange(0, BLOCK2)
    t0 = _sys_cnt()
    for loop1 in range(BLOCK1 // BLOCK1_SUB):
        s = offset + loop1 * BLOCK1_SUB + sbase[:, None]
        for n in range(N):
            for loop2 in range(D // BLOCK2):
                d = loop2 * BLOCK2 + dbase[None, :]
                x = tl.load(a + 32768 * n + 8 * s + d)
                y = tl.load(b + 32768 * n + 8 * s + d)
                z = tl.load(c + 8 * n + d + 96 * s)
                tl.store(out + 8 * n + d + 96 * s, x + y + z)
    t1 = _sys_cnt()
    tl.store(tick_ptr + tl.program_id(0) * 2, t0)
    tl.store(tick_ptr + tl.program_id(0) * 2 + 1, t1)
"""

SOFTMAX = """
@triton.jit
def e20_softmax(out, inp, input_stride, output_stride, n_rows,
                n_cols, tick_ptr, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    t0 = _sys_cnt()
    for row in tl.range(row_start, n_rows, row_step):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        value = tl.load(inp + row * input_stride + cols, mask=mask,
                        other=-float("inf"))
        numerator = tl.exp(value - tl.max(value, axis=0))
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(out + row * output_stride + cols, result, mask=mask)
    t1 = _sys_cnt()
    tl.store(tick_ptr + tl.program_id(0) * 2, t0)
    tl.store(tick_ptr + tl.program_id(0) * 2 + 1, t1)
"""

SIMT_COMMON = """
@triton.jit
def _clock():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l,~{memory}", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""

SCATTER = """
@triton.jit
def e20_scatter(inp, out, in_numel, out_numel, tick_ptr,
                XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
                STRIDE: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    t0 = _clock()
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        dst_index = index * STRIDE
        value = tl.load(inp + index, mask=index < in_numel)
        tl.store(out + dst_index, value, mask=dst_index < out_numel)
    t1 = _clock()
    tl.store(tick_ptr + tl.program_id(0) * 2, t0)
    tl.store(tick_ptr + tl.program_id(0) * 2 + 1, t1)
"""

PATTERN = (
    r'tt\.elementwise_inline_asm "MOV \$0, SYS_CNT"'
    r' \{constraints = "=l,~\{memory\}", '
    r'packed_element = 1 : i32, pure = false\} -> i64'
)


def replace_markers(text):
    patched, count = re.subn(PATTERN, "ascend_dpx.clock64", text)
    if count < 1:
        raise ValueError("clock marker not matched")
    return patched


@contextmanager
def clock64_compilation():
    original_run = subprocess.run

    def run(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) > 1
                and Path(command[0]).name == "bishengir-compile"):
            source = Path(command[1])
            source.write_text(replace_markers(source.read_text()))
        return original_run(command, *args, **kwargs)

    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--out", default="/root/work/explore_contig/"
                                     "e20_results.json")
    args = ap.parse_args()

    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    import torch  # noqa: E402
    import triton  # noqa: E402
    import torch_npu  # noqa: E402
    torch_npu.npu.set_device(0)

    out = {"meta": {"launches": 30}, "cells": {}}
    out_path = Path(args.out)

    # ---- 1) muls_add K sweep ----
    text = HEADER + SIMD_COMMON + MULS
    gp = Path("/root/work/explore_contig/_gen_e20a.py")
    gp.write_text(text, encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e20a", str(gp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blk = 1024
    for K in (19, 37, 74):
        n_blocks = 56 * K
        x = torch.randn((n_blocks * blk,), dtype=torch.float32,
                        device="npu")
        y = torch.randn((n_blocks * blk,), dtype=torch.float32,
                        device="npu")
        o = torch.zeros((n_blocks * blk,), dtype=torch.float32,
                        device="npu")
        tick = torch.zeros((112,), dtype=torch.int64, device="npu")

        def launch():
            mod.e20_muls[(56,)](x, y, o, tick, n_blocks, K=K, BLOCK=blk,
                                num_warps=4, compile_mode="simd",
                                auto_simt_scope_mode="off")

        for _ in range(5):
            launch()
        torch.npu.synchronize()
        allns = []
        for i in range(30):
            launch()
            torch.npu.synchronize()
            t = tick.cpu()[:112].reshape(56, 2)
            ns = (t[:, 1] - t[:, 0]) * 1.00008
            allns.append(statistics.median(ns.tolist()))
        whole = statistics.median(allns)
        out["cells"]["muls_K%d" % K] = {"window_us": round(whole / 1000, 3),
                                        "per_iter": round(whole / K, 1)}
        print("muls K=%-3d window=%.3fus per_iter=%.1f" %
              (K, whole / 1000, whole / K), flush=True)
        out_path.write_text(json.dumps(out, indent=1))
        del x, y, o
        torch.npu.empty_cache()

    # ---- 2) permute ----
    text = HEADER + SIMD_COMMON + PERMUTE
    gp.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_gen_e20b", str(gp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = torch.randn((32768 * 12 + 65536,), dtype=torch.float32,
                    device="npu")
    b = torch.randn_like(a)
    cc = torch.randn((8 * 12 + 96 * 128 + 4096,), dtype=torch.float32,
                     device="npu")
    o = torch.zeros_like(cc)
    tick = torch.zeros((64,), dtype=torch.int64, device="npu")

    def launch_p():
        mod.e20_permute[(32,)](a, b, cc, o, tick, BLOCK1=128, BLOCK1_SUB=64,
                               BLOCK2=8, S=8, N=12, D=8, num_warps=4,
                               compile_mode="simd",
                               auto_simt_scope_mode="off")

    for _ in range(5):
        launch_p()
    torch.npu.synchronize()
    allns = []
    for i in range(30):
        launch_p()
        torch.npu.synchronize()
        t = tick.cpu()[:64].reshape(32, 2)
        ns = (t[:, 1] - t[:, 0]) * 1.00008
        allns.append(statistics.median(ns.tolist()))
    w = statistics.median(allns)
    out["cells"]["permute_window_us"] = round(w / 1000, 3)
    print("permute window=%.3fus" % (w / 1000), flush=True)
    out_path.write_text(json.dumps(out, indent=1))

    # ---- 3) fused_softmax ----
    text = HEADER + SIMD_COMMON + SOFTMAX
    gp.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_gen_e20c", str(gp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inp = torch.randn((128 * 512,), dtype=torch.float32, device="npu")
    oo = torch.zeros((128 * 512,), dtype=torch.float32, device="npu")
    tick = torch.zeros((64,), dtype=torch.int64, device="npu")

    def launch_s():
        mod.e20_softmax[(32,)](oo, inp, 512, 512, 128, 257, tick,
                               BLOCK_SIZE=512, num_warps=4,
                               compile_mode="simd",
                               auto_simt_scope_mode="off")

    for _ in range(5):
        launch_s()
    torch.npu.synchronize()
    allns = []
    for i in range(30):
        launch_s()
        torch.npu.synchronize()
        t = tick.cpu()[:64].reshape(32, 2)
        ns = (t[:, 1] - t[:, 0]) * 1.00008
        allns.append(statistics.median(ns.tolist()))
    w = statistics.median(allns)
    out["cells"]["softmax_window_us"] = round(w / 1000, 3)
    print("softmax window=%.3fus" % (w / 1000), flush=True)
    out_path.write_text(json.dumps(out, indent=1))

    # ---- 4) scatter (simt clock64) ----
    text = HEADER + SIMT_COMMON + SCATTER
    gp.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_gen_e20d", str(gp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    in_numel, stride, grid, xblock, sub = 4096, 16, 2, 2048, 256
    out_numel = in_numel * stride
    inp = torch.randn((in_numel,), dtype=torch.float32, device="npu")
    oo = torch.zeros((out_numel,), dtype=torch.float32, device="npu")
    tick = torch.zeros((4,), dtype=torch.int64, device="npu")

    def launch_x():
        with clock64_compilation():
            mod.e20_scatter[(grid,)](inp, oo, tick, in_numel=in_numel,
                                     out_numel=out_numel, XBLOCK=xblock,
                                     XBLOCK_SUB=sub, STRIDE=stride,
                                     num_warps=4, compile_mode="simt_only",
                                     auto_simt_scope_mode="off",
                                     superblock_factor=1)

    launch_x()
    torch.npu.synchronize()
    allns = []
    for i in range(30):
        launch_x()
        torch.npu.synchronize()
        t = tick.cpu()[:4].reshape(2, 2)
        ns = ((t[:, 1] - t[:, 0]) & ((1 << 64) - 1)).tolist()
        allns.append(statistics.median(ns))
    w = statistics.median(allns)
    out["cells"]["scatter_window_us"] = round(w / 1000, 3)
    print("scatter window=%.3fus (simt clock64, whole grid=2)" %
          (w / 1000), flush=True)
    out_path.write_text(json.dumps(out, indent=1))
    print("saved", args.out)


main()
