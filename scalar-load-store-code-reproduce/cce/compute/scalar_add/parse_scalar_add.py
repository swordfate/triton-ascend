#!/usr/bin/env python3
"""Parse scalar-add CAModel dumps into a JSON summary.

Only loop-body hot instructions are used for the throughput numbers; epilogue /
initialization adds are reported separately so the table does not mix them.

Each case directory must contain core0.veccore0.instr_log.dump:
  SIMD: lines containing "ADD  dtype:F32"
  SIMT: lines containing "SIMT_FADD"
The SIMD case additionally uses core0.veccore0.ccu.scalar_issque.dump for the
push -> retire window.

Output: JSON list to stdout.
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ITERS = 64  # run_scalar_add_camodel.sh default


def _stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 3),
        "max": max(values),
    }


def _parse_lines(root: Path):
    instr = (root / "core0.veccore0.instr_log.dump").read_text(errors="ignore").splitlines()
    simd = []
    simt = []
    for line in instr:
        mt = re.search(r"\[info\] \[(\d+)\]", line)
        mp = re.search(r"PC: (0x[0-9a-fA-F]+)", line)
        mi = re.search(r"\(ID: (\d+)\)", line)
        if not (mt and mp and mi):
            continue
        rec = {
            "ts": int(mt.group(1)),
            "pc": mp.group(1).lower(),
            "id": int(mi.group(1)),
        }
        if "ADD  dtype:F32" in line:
            simd.append(rec)
        if "SIMT_FADD" in line:
            me = re.search(r"\[exec_time: (\d+)\]", line)
            ms = re.search(r"\[stallCyc:\s*(\d+)\]", line)
            rec["exec_time"] = int(me.group(1)) if me else None
            rec["stall_cyc"] = int(ms.group(1)) if ms else None
            simt.append(rec)
    return simd, simt


def _hot_split(events):
    counts = collections.Counter(e["pc"] for e in events)
    hot = {pc for pc, c in counts.items() if c >= 10}
    if not hot and counts:
        hot = set(counts)  # single-op / uniform cases have no repeated PC
    loop = sorted((e for e in events if e["pc"] in hot), key=lambda e: e["ts"])
    other = sorted((e for e in events if e["pc"] not in hot), key=lambda e: e["ts"])
    return loop, other, hot


def _interval_stats(loop):
    ts = [e["ts"] for e in loop]
    return _stats([b - a for a, b in zip(ts, ts[1:])])


def parse_simd(root: Path, events):
    loop, other, hot = _hot_split(events)
    windows = []
    pushes = {}
    retires = {}
    iss = (root / "core0.veccore0.ccu.scalar_issque.dump").read_text(errors="ignore").splitlines()
    for line in iss:
        m = re.search(r"\[info\] (\d+) \[Push Instr\] name=ADD pc=(0x[0-9a-fA-F]+) id=(\d+)", line)
        if m and m.group(2).lower() in hot:
            pushes[int(m.group(3))] = int(m.group(1))
        m = re.search(
            r"\[info\] (\d+) \[RETIRE INSTR\] instr\.name=ADD\s+instr\.pc=(0x[0-9a-fA-F]+)\s+instr\.id=(\d+)",
            line,
        )
        if m and m.group(2).lower() in hot:
            retires[int(m.group(3))] = int(m.group(1))
    for i, ts in pushes.items():
        if i in retires:
            windows.append(retires[i] - ts)
    span = loop[-1]["ts"] - loop[0]["ts"] if loop else 0
    return {
        "case": root.name,
        "kind": "simd_main_scalar_fadd",
        "iters": ITERS,
        "loop_add_count": len(loop),
        "epilogue_add_count": len(other),
        "total_add_count": len(loop) + len(other),
        "loop_pc_count": len(hot),
        "loop_span": span,
        "cycles_per_loop_add": round(span / (len(loop) - 1), 3) if len(loop) > 1 else None,
        "loop_adds_per_cycle": round((len(loop) - 1) / span, 3) if span > 0 else None,
        "loop_issue_interval": _interval_stats(loop),
        "push_to_retire_window": _stats(windows),
    }


def parse_simt(root: Path, events):
    loop, other, hot = _hot_split(events)
    span = loop[-1]["ts"] - loop[0]["ts"] if loop else 0
    execs = [e["exec_time"] for e in loop if e["exec_time"] is not None]
    stalls = [e["stall_cyc"] for e in loop if e["stall_cyc"] is not None]
    return {
        "case": root.name,
        "kind": "simt_warp_fadd_32t",
        "iters": ITERS,
        "loop_fadd_count": len(loop),
        "epilogue_fadd_count": len(other),
        "total_fadd_count": len(loop) + len(other),
        "loop_pc_count": len(hot),
        "loop_span": span,
        "cycles_per_warp_fadd": round(span / (len(loop) - 1), 3) if len(loop) > 1 else None,
        "warp_fadd_per_cycle": round((len(loop) - 1) / span, 3) if span > 0 else None,
        "loop_issue_interval": _interval_stats(loop),
        "exec_time": _stats(execs),
        "stall_cyc": _stats(stalls),
    }


def main():
    out = []
    for arg in sys.argv[1:]:
        root = Path(arg)
        simd, simt = _parse_lines(root)
        if simt:
            out.append(parse_simt(root, simt))
        else:
            out.append(parse_simd(root, simd))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
