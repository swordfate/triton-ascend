"""E12: e1_load_loop vs cal_b chain kernel, SAME session, both cold.

The 'cold' acc-chain loop has now been measured at 167 (E1, morning),
243-256 (E1 script rerun, tonight) and 87 ns/round (cal_b, tonight).
Same-session A/B of the two kernels separates "kernel difference" from
"session/environment state".

Also dumps both kernels' CCE ('source') loop bodies for a structural diff.

Usage: bash run_exp.sh e12_cold_ab.py 3 e12_c3.jsonl --skip-health
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import JsonlOut, add_common_args, current_card, std_opts

BLOCK = 1024


@triton.jit(do_not_specialize=["iters", "winmask", "boff"])
def e1_load_loop(src, dst, iters, winmask, boff, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + boff + (step & winmask) * BLOCK + lane)
    tl.store(dst + lane, acc)


@triton.jit(do_not_specialize=["iters", "boff"])
def cb_chain(src, dst, iters, boff, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + boff + step * BLOCK + lane)
    tl.store(dst + lane, acc)


@triton.jit
def ctl_1store(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 1, dst.dtype.element_ty))


def main():
    ap = argparse.ArgumentParser()
    ap = add_common_args(ap)
    ap.add_argument("--modes", default=["simt_only", "simd"], nargs="+")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=3)
    args = ap.parse_args()

    out = JsonlOut(args.out)
    n = (320 + 64) * 1024 * 1024 // 4
    src = torch.randn(n, device="npu")
    dst = torch.zeros(BLOCK * 2, device="npu")
    stride = (2 * 1024 * 1024) // 4
    nreg = (320 * 1024 * 1024 // 4) // stride
    ir = [0]

    def boff():
        v = (ir[0] % nreg) * stride
        ir[0] += 1
        return v

    for mode in args.modes:
        opts = std_opts(mode, args.warps)
        for s in range(args.sessions):
            # within one session: A/B pairs (e1-style vs calb-style), cold
            import torch_npu.profiler as prof
            import tempfile, shutil, csv
            trace = Path(tempfile.mkdtemp(prefix="e12_"))
            launchers = [
                lambda b=boff(): e1_load_loop[(1,)](src, dst, 64, 63, b,
                                                    BLOCK, **opts),
                lambda b=boff(): cb_chain[(1,)](src, dst, 64, b,
                                                BLOCK, **opts),
            ]
            names = ["e1_load_loop", "cb_chain"]
            with prof.profile(
                    activities=[prof.ProfilerActivity.NPU],
                    schedule=prof.schedule(wait=1, warmup=2, active=24),
                    on_trace_ready=prof.tensorboard_trace_handler(
                        str(trace))) as p:
                for fn in launchers:
                    fn()
                torch.npu.synchronize()
                for _ in range(6):
                    for fn in launchers:
                        fn()
                torch.npu.synchronize()
                for _ in range(args.pairs):
                    for fn in launchers:
                        fn()
                        p.step()
            torch.npu.synchronize()
            meds = {}
            for f in names:
                durs = []
                for path in sorted(trace.rglob("kernel_details.csv")):
                    with path.open(newline="") as h:
                        for row in csv.DictReader(h):
                            if f in (row.get("Name") or "") and \
                                    row.get("Duration(us)"):
                                durs.append(float(row["Duration(us)"]))
            # collect
            series = {f: [] for f in names}
            for path in sorted(trace.rglob("kernel_details.csv")):
                with path.open(newline="") as h:
                    for row in csv.DictReader(h):
                        dur = row.get("Duration(us)")
                        name = row.get("Name") or ""
                        if not dur:
                            continue
                        for f in names:
                            if f in name:
                                series[f].append(float(dur))
            shutil.rmtree(trace, ignore_errors=True)
            stats = {}
            for f in names:
                srt = sorted(series[f])
                stats[f] = {"n": len(srt), "med": round(srt[len(srt) // 2], 3),
                            "min": round(srt[0], 3),
                            "max": round(srt[-1], 3)} if srt else None
            rec = {"exp": "e12", "mode": mode, "session": s,
                   "card": current_card(), "K": 64, "stats": stats}
            if s == 0 and mode == args.modes[0]:
                h_static = e1_load_loop[(1,)](src, dst, 64, 63, 0, BLOCK, **opts)
                h_cb = cb_chain[(1,)](src, dst, 64, 0, BLOCK, **opts)
                rec["cce_diff"] = {
                    "e1_lines": len((h_static.asm.get("source") or "").splitlines()),
                    "cb_lines": len((h_cb.asm.get("source") or "").splitlines()),
                }
            out.write(**rec)
            print("session", s, mode, json.dumps(stats), flush=True)
    print("E12 DONE", flush=True)


if __name__ == "__main__":
    main()
