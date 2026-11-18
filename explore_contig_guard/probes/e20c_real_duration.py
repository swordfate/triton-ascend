#!/usr/bin/env python3
"""E20c: decisive split for muls_add_n1048576 (K=19, grid=56).
Profiler whole-kernel Duration of the real shape vs the SYS_CNT loop
window (1.766us measured) -> the true non-loop in-kernel cost."""
import statistics
import shutil
import tempfile
from pathlib import Path

HEADER = "import triton\nimport triton.language as tl\n\n"
COMMON = """
@triton.jit
def _sys_cnt():
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT", constraints="=l", args=[],
        dtype=tl.int64, is_pure=False, pack=1)
"""
KERN = """
@triton.jit
def e20c_muls(x, y, out, tick_ptr, n_blocks, K: tl.constexpr,
              BLOCK: tl.constexpr):
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


def main():
    import torch  # noqa: E402
    import torch_npu  # noqa: E402
    import torch_npu.profiler as prof  # noqa: E402
    torch_npu.npu.set_device(0)

    gp = Path(tempfile.mkdtemp(prefix="e20c_")) / "k.py"
    gp.write_text(HEADER + COMMON + KERN, encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_e20c", str(gp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    K, grid, blk = 19, 56, 1024
    n_blocks = grid * K
    x = torch.randn((n_blocks * blk,), dtype=torch.float32, device="npu")
    y = torch.randn((n_blocks * blk,), dtype=torch.float32, device="npu")
    o = torch.zeros((n_blocks * blk,), dtype=torch.float32, device="npu")
    tick = torch.zeros((112,), dtype=torch.int64, device="npu")

    def launch():
        mod.e20c_muls[(grid,)](x, y, o, tick, n_blocks, K=K, BLOCK=blk,
                               num_warps=4, compile_mode="simd",
                               auto_simt_scope_mode="off")

    for _ in range(10):
        launch()
    torch.npu.synchronize()
    trace = Path(tempfile.mkdtemp(prefix="e20cprof_"))
    windows = []
    with prof.profile(activities=[prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(
                          str(trace))):
        torch.npu.synchronize()
        for _ in range(30):
            launch()
            torch.npu.synchronize()
            t = tick.cpu()[:112].reshape(grid, 2)
            ns = (t[:, 1] - t[:, 0]) * 1.00008
            windows.append(statistics.median(ns.tolist()))
    torch.npu.synchronize()
    durs = []
    for tpath in sorted(trace.rglob("kernel_details.csv")):
        with tpath.open(newline="", errors="ignore") as h:
            for row in csv_DictReader(h):
                if "e20c_muls" in (row.get("Name") or ""):
                    durs.append(float(row["Duration(us)"]) * 1000)
    dur = statistics.median(durs)
    win = statistics.median(windows)
    print("K=19 grid=56: Duration=%.1f ns  SYS_CNT window=%.1f ns" %
          (dur, win))
    print("non-loop in-kernel = %.1f ns" % (dur - win))
    shutil.rmtree(trace, ignore_errors=True)


def csv_DictReader(h):
    import csv
    return csv.DictReader(h)


main()
