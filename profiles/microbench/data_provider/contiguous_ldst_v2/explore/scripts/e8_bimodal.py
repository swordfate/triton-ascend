"""E8: characterize the simt session-level bimodality (per-launch time series).

E7 found: the same simt loop kernel (E7_loop1 shape, N=32) measured
4.64 / 13.15 / 4.64 us medians in three adjacent profiler sessions on the same
card - a 2.8x state flip.  Historical bw calibration double-peaks (63/187
groups >1.8x) are probably the same phenomenon.  Before any simt loop/pipeline
calibration can be trusted we need to characterize the state:

  Q1  What does the per-launch time series look like inside one session -
      abrupt state flips, slow drift, or per-launch noise?
  Q2  Is the state global to the device (a simd reference kernel fluctuates
      in sync) or simt-specific (simd stays flat while simt flips)?
  Q3  Is it the loop structure, or does a plain simt straight-line kernel
      also flip?
  Q4  Does it persist across sessions (session 3 = fast again)?

Design: one profiler session, active=240 launches in fixed order:
  [simt_loop, simd_ref, simt_line, simd_ref] x 60
  simt_loop  - the E7 bimodal kernel (K=32 streaming load+store loop)
  simt_line  - simt straight-line 8 interleaved pairs (E3c ind shape)
  simd_ref   - a minimal simd 4KB store, expected flat; if it fluctuates in
               sync with simt, the state is device-global (clock/tenant);
               if flat, the state is simt-specific (LSU/DCache path).
All durations are dumped in launch order to JSONL (one row per launch) so the
transition pattern can be inspected directly.

Usage:
  bash run_exp.sh e8_bimodal.py 3 e8_c3.jsonl --sessions 3
"""
import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import JsonlOut, add_common_args, current_card, std_opts

BLOCK = 1024


@triton.jit(do_not_specialize=["iters"])
def e8_simt_loop(src, dst, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for step in range(iters):
        x = tl.load(src + step * BLOCK + lane)
        tl.store(dst + step * BLOCK + lane, x + 1)


@triton.jit
def e8_simt_line(src, dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for j in tl.static_range(8):
        x = tl.load(src + j * BLOCK + lane)
        tl.store(dst + j * BLOCK + lane, x + 1)


@triton.jit
def e8_simd_ref(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


def run_session(launchers, names, active, tag, out):
    """One profiler session; dump every launch in order."""
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e8_"))
    order = []
    for i in range(active):
        for li, fn in enumerate(launchers):
            order.append((i, li))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=active),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for fn in launchers:      # warmup
            fn()
        torch.npu.synchronize()
        for _ in range(30):       # long warmup so early launches settle
            for fn in launchers:
                fn()
        torch.npu.synchronize()
        idx = 0
        for i in range(active):
            for fn in launchers:
                fn()
                p.step()
                idx += 1
    torch.npu.synchronize()
    series = {name: [] for name in names}
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for row in csv.DictReader(h):
                dur = row.get("Duration(us)")
                name = row.get("Name") or ""
                if not dur:
                    continue
                for f in names:
                    if f in name:
                        series[f].append(round(float(dur), 3))
    shutil.rmtree(trace, ignore_errors=True)
    rec = {"tag": tag, "card": current_card(), "series": series}
    out.write(**rec)
    # quick textual summary
    for f in names:
        s = series[f]
        if not s:
            continue
        n = len(s)
        srt = sorted(s)
        med = srt[n // 2]
        p90 = srt[(9 * n) // 10]
        b = max(1, n // 6)
        buckets = [sorted(s[i * b:(i + 1) * b])[len(s[i * b:(i + 1) * b]) // 2]
                   for i in range(6)]
        print("%s %s: n=%d med=%.2f p90=%.2f min=%.2f max=%.2f seg=%s" % (
            tag, f, n, med, p90, min(s), max(s),
            " ".join("%.1f" % x for x in buckets)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default=["simt_only"], nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--active", type=int, default=240)
    args = ap.parse_args()

    out = JsonlOut(args.out)
    src = torch.randn(BLOCK * 1024, device="npu")
    dst = torch.zeros(BLOCK * 1024, device="npu")

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        launchers = [
            lambda: e8_simt_loop[(1,)](src, dst, 32, BLOCK, **opts),
            lambda: e8_simd_ref[(1,)](dst, BLOCK, **opts),
            lambda: e8_simt_line[(1,)](src, dst, BLOCK, **opts),
            lambda: e8_simd_ref[(1,)](dst, BLOCK, **opts),
        ]
        names = ["e8_simt_loop", "e8_simd_ref", "e8_simt_line"]
        for s in range(args.sessions):
            run_session(launchers, names, args.active,
                        "%s_s%d" % (mode, s), out)
    print("E8 DONE", flush=True)


if __name__ == "__main__":
    main()
