"""Common measurement utilities for the contiguous-ldst exploration experiments.

Every experiment on the 950PR server imports this module.  It provides:

* prof_sessions()  - run ABAB-interleaved kernels under torch_npu.profiler and
  collect per-kernel Duration(us) lists.
* stats()          - min / median / quartiles of a duration sample.
* pair_diff_ns()   - calibration-style per-pair (A-B) difference in ns.
* health_check()   - cross-card reference latency gate.  A card that "looks
  stable but runs slow" is exactly what this is meant to catch: the reference
  medians are compared against the best card measured in the same run.

All results go out as JSONL lines with an explicit `card` field so results from
different cards can never be silently mixed.
"""
import argparse
from collections import defaultdict
import csv
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)
import triton
import triton.language as tl


def current_card():
    """Best-effort id of the NPU this process is pinned to."""
    import os
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "?")
    try:
        return int(vis)
    except ValueError:
        return vis


def prof_session(launch_fns, name_filters, pairs, extra_warmup=6):
    """Interleave `launch_fns` under one profiler session.

    Each callable is launched once per round; `pairs` rounds are captured
    (plus wait/warmup rounds the schedule discards).  Returns
    {name_filter: [Duration(us), ...]}.

    launch_fns  : list of callables, e.g. [variant_A, control_B]
    name_filters: substrings matching each callable's kernel name, same order.
    """
    import torch_npu.profiler as prof
    total = pairs * len(launch_fns)
    trace = Path(tempfile.mkdtemp(prefix="exp_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=total),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for fn in launch_fns:            # pre-warm outside the captured rounds
            fn()
        torch.npu.synchronize()
        for _ in range(extra_warmup):
            for fn in launch_fns:
                fn()
        torch.npu.synchronize()
        for _ in range(pairs):
            for fn in launch_fns:
                fn()
                p.step()
    torch.npu.synchronize()
    rows = {f: [] for f in name_filters}
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for row in csv.DictReader(h):
                dur = row.get("Duration(us)")
                name = row.get("Name") or ""
                if not dur:
                    continue
                for f in name_filters:
                    if f in name:
                        rows[f].append(float(dur))
    shutil.rmtree(trace, ignore_errors=True)
    return rows


def stats_us(durs):
    """min/median/quartiles of a Duration(us) sample -> dict (us)."""
    if not durs:
        return None
    s = sorted(durs)
    n = len(s)
    return {
        "n": n,
        "min_us": round(s[0], 3),
        "q1_us": round(s[max(0, n // 4)], 3),
        "med_us": round(s[n // 2], 3),
        "q3_us": round(s[(3 * n) // 4], 3),
        "max_us": round(s[-1], 3),
    }


def prof_pairs_session(spec, rounds, extra_warmup=6):
    """One profiler session covering MULTIPLE probe/ctl pairs.

    spec: list of (probe_fn, probe_name, ctl_fn, ctl_name) tuples, launched
    in this order every round; each launch followed by prof.step().
    Returns {name: [Duration(us), ...]} — per-point diffs use the aligned
    i-th launches of probe/ctl (both captured `rounds` times).

    This exists because a profiler session costs ~25-30s (parse + schedule),
    so measuring all K/U points of one cell in a single session is ~10x
    faster than one session per point.
    """
    import torch_npu.profiler as prof
    total = rounds * len(spec)
    trace = Path(tempfile.mkdtemp(prefix="cals_"))
    with prof.profile(
            activities=[prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=1, warmup=2, active=total),
            on_trace_ready=prof.tensorboard_trace_handler(str(trace))) as p:
        for item in spec:              # compile/warm every kernel once
            item[0]()
            item[2]()
        torch.npu.synchronize()
        for _ in range(extra_warmup):
            for item in spec:
                item[0]()
                item[2]()
        torch.npu.synchronize()
        for _ in range(rounds):
            for probe, _, ctl, _ in spec:
                probe()
                p.step()
                ctl()
                p.step()
    torch.npu.synchronize()
    wanted = [t[1] for t in spec] + [t[3] for t in spec]
    rows = {f: [] for f in wanted}
    for path in sorted(trace.rglob("kernel_details.csv")):
        with path.open(newline="") as h:
            for row in csv.DictReader(h):
                dur = row.get("Duration(us)")
                name = row.get("Name") or ""
                if not dur:
                    continue
                for f in wanted:
                    if f in name:
                        rows[f].append(round(float(dur), 3))
    shutil.rmtree(trace, ignore_errors=True)
    return rows


def pair_diffs_from_session(spec, rows):
    """Per-point pair-diff (calibration-style) from a prof_pairs_session."""
    out = {}
    for probe, pname, ctl, cname in spec:
        da, db = rows.get(pname, []), rows.get(cname, [])
        out[pname] = pair_diff_ns(da, db)
    return out


def pair_diff_ns(durs_a, durs_b):
    """Median per-pair (A-B) difference in ns over min(len(A), len(B)) pairs.

    This mirrors probe_forms_single_shot.py: differences cancel block-level
    drift between consecutive launches.
    """
    n = min(len(durs_a), len(durs_b))
    if n == 0:
        return None
    diffs = sorted((durs_a[i] - durs_b[i]) * 1000.0 for i in range(n))
    med = diffs[n // 2]
    q1, q3 = diffs[n // 4], diffs[(3 * n) // 4]
    return {
        "npairs": n,
        "diff_med_ns": round(med, 1),
        "diff_q1_ns": round(q1, 1),
        "diff_q3_ns": round(q3, 1),
        "cv": round((q3 - q1) / max(1e-9, abs(med)), 4),
    }


class JsonlOut:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def write(self, **rec):
        rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        rec.setdefault("card", current_card())
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()


# --------------------------------------------------------------------------
# Health gate
# --------------------------------------------------------------------------
# A 4 KiB single-program store is the cheapest kernel that still goes through
# the whole compile/launch pipeline; a grid-56 streaming load touches the
# memory system at scale.  Both medians together catch "stable but slow".

@triton.jit
def ref_store4k(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane, tl.full((BLOCK,), 3.0, dst.dtype.element_ty))


@triton.jit
def ref_stream_load(src, out, iters, BLOCK: tl.constexpr, WINMASK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (WINMASK + 1) * BLOCK
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), src.dtype.element_ty)
    for step in range(iters):
        acc += tl.load(src + base + (step & WINMASK) * BLOCK + lane)
    tl.store(out + pid * BLOCK + lane, acc)


def health_check(out_path, pairs=12, stream_grid=56):
    """Measure both reference kernels; write one JSONL record."""
    src = torch.randn(64 * 1024 * 1024, device="npu")      # 256 MB fp32
    dst = torch.zeros(4096, device="npu")
    out4k = torch.zeros(56 * 1024, device="npu")

    opts = dict(compile_mode="simd", num_warps=1)
    f_small = lambda: ref_store4k[(1,)](dst, 1024, **opts)
    f_stream = lambda: ref_stream_load[(stream_grid,)](
        src, out4k, 512, 1024, 63, **opts)
    for _ in range(8):
        f_small()
        f_stream()
    torch.npu.synchronize()
    rows = prof_session([f_small, f_stream], ["ref_store4k", "ref_stream_load"],
                        pairs)
    rec = {
        "kind": "health",
        "ref_store4k": stats_us(rows["ref_store4k"]),
        "ref_stream": stats_us(rows["ref_stream_load"]),
    }
    JsonlOut(out_path).write(**rec)
    return rec


def gate_or_die(health_rec, history_path, ratio=1.50):
    """Raise if this card's small-kernel median exceeds ratio x the golden
    reference (0.61us, reproduced across many sessions) OR ratio x the
    per-card historical median.  The original min-over-history rule was
    poisoned by a single 0.4us fast outlier session."""
    GOLDEN_STORE_US = 0.61
    hist = defaultdict(list)
    p = Path(history_path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "health" and r.get("ref_store4k"):
                hist[r.get("card")].append(r["ref_store4k"]["med_us"])
    mine = health_rec["ref_store4k"]["med_us"]
    my_card = str(current_card())
    my_hist = hist.get(my_card, [])
    my_typical = statistics.median(my_hist) if my_hist else GOLDEN_STORE_US
    base = min(GOLDEN_STORE_US, my_typical)
    rec = {"kind": "gate", "my_med_us": mine, "golden_us": GOLDEN_STORE_US,
           "my_card_typical": my_typical, "ratio_vs_golden":
               round(mine / GOLDEN_STORE_US, 3), "cards_seen":
               sorted(hist.keys())}
    JsonlOut(history_path).write(**rec)
    if mine / base > ratio:
        raise SystemExit(
            "CARD HEALTH GATE FAILED: %.1fus is %.2fx the %.2fus reference "
            "(golden %.2f / card-%s typical %.2f). Move to another card."
            % (mine, mine / base, base, GOLDEN_STORE_US, my_card,
               my_typical))
    return rec


def std_opts(mode, warps):
    """Compile options matching the calibration probes."""
    o = dict(compile_mode=mode, num_warps=warps)
    if mode == "simt_only":
        o["superblock_factor"] = 1
    return o


def add_common_args(ap):
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--pairs", type=int, default=12,
                    help="profiler rounds per point")
    ap.add_argument("--health", default="", help="health history JSONL path")
    ap.add_argument("--skip-health", action="store_true")
    return ap
