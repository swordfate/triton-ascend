"""E9: what triggers the simt fast/slow modes?  (follow-up to E8)

Known so far (E7/E8):
  simt_loop (K=32 streaming load+store loop) per-pair cost has been observed
  at three levels: fast ~145 ns (E7 A sessions 1/3), mid ~230-275 (E1/E3c),
  slow ~415 (E8, two cards, all sessions, +/-2% within session).
  simd is always stable.  Session-level mode, not per-launch noise.

Leading hypothesis H-N (neighbor pattern): the mode depends on which kernel
runs BETWEEN the probe launches.
  E7 fast sessions interleaved the probe with a same-shape STORE loop
  over the same buffers (its calibration control);
  E8 slow sessions interleaved simd_ref / simt_line instead.

E9 design (one process, sessions ALTERNATE so time-of-day cannot explain):
  s0 P1  probe x store-loop-neighbor   (E7 pattern; H-N predicts FAST)
  s1 P2  probe x simd_ref neighbor     (E8 pattern; H-N predicts SLOW)
  s2 P1  (repeat)
  s3 P2  (repeat)
  s4 P3  probe alone, no neighbor      (isolates "any neighbor" vs "store loop")
  s5 P1  (repeat)
  s6 P2  (repeat)

Each session: profiler, probe launches interleaved with its neighbor
(1:1) or alone; 120 probe launches; dump the probe per-launch series and
report median/quartiles.  Verdict rule: if P1 sessions are consistently fast
while interleaved-in-time P2 sessions are slow, the neighbor pattern is the
trigger and the fast state is an inter-launch overlap artifact; calibration
should then use the clean slow state as truth.  If P1 is slow too, H-N dies
and the trigger is temporal/environmental.

Usage:
  bash run_exp.sh e9_trigger.py 3 e9_c3.jsonl
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
def e9_loop(src, dst, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for step in range(iters):
        x = tl.load(src + step * BLOCK + lane)
        tl.store(dst + step * BLOCK + lane, x + 1)


@triton.jit(do_not_specialize=["iters"])
def e9_storeloop(dst, iters, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for step in range(iters):
        tl.store(dst + step * BLOCK + lane,
                 tl.full((BLOCK,), 1, dst.dtype.element_ty))


@triton.jit
def e9_simdref(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3, dst.dtype.element_ty))


def run_session(launchers, probe_idx, probe_name, active, tag, out, opts):
    import torch_npu.profiler as prof
    trace = Path(tempfile.mkdtemp(prefix="e9_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=active),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for fn in launchers:
            fn()
        torch.npu.synchronize()
        for _ in range(30):
            for fn in launchers:
                fn()
        torch.npu.synchronize()
        for _ in range(active):
            for fn in launchers:
                fn()
                p.step()
    torch.npu.synchronize()
    series = []
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for row in csv.DictReader(h):
                dur = row.get("Duration(us)")
                name = row.get("Name") or ""
                if dur and probe_name in name:
                    series.append(round(float(dur), 3))
    shutil.rmtree(trace, ignore_errors=True)
    n = len(series)
    srt = sorted(series)
    med = srt[n // 2] if n else None
    p10 = srt[n // 10] if n else None
    p90 = srt[(9 * n) // 10] if n else None
    rec = {"tag": tag, "card": current_card(), "n": n, "med": med,
           "p10": p10, "p90": p90, "min": min(series) if series else None,
           "max": max(series) if series else None, "series": series}
    out.write(**rec)
    print("%s: n=%d med=%s p10=%s p90=%s" % (tag, n, med, p10, p90),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions-per-pattern", type=int, default=3)
    ap.add_argument("--active", type=int, default=120)
    args = ap.parse_args()

    out = JsonlOut(args.out)
    src = torch.randn(BLOCK * 1024, device="npu")
    dst = torch.zeros(BLOCK * 1024, device="npu")
    opts = std_opts("simt_only", args.warps)

    probe = lambda: e9_loop[(1,)](src, dst, 32, BLOCK, **opts)
    storeloop = lambda: e9_storeloop[(1,)](dst, 32, BLOCK, **opts)
    simdref = lambda: e9_simdref[(1,)](dst, BLOCK, **opts)

    plan = []
    for _ in range(args.sessions_per_pattern):
        plan.append(("P1_storeloop_nb", [probe, storeloop], 0))
        plan.append(("P2_simdref_nb", [probe, simdref], 0))
    plan.append(("P3_alone", [probe], 0))

    for i, (tag, launchers, _) in enumerate(plan):
        run_session(launchers, 0, "e9_loop", args.active, tag, out, opts)
    print("E9 DONE", flush=True)


if __name__ == "__main__":
    main()
