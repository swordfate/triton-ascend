#!/usr/bin/env python3
"""Summarize CAModel-vs-board SYS_CNT probe logs.

Reads:
  results/board/<case>__<kernel>.log          # lines: REP <i> <kernel> v0=<ticks>
  results/camodel_ind_all/<case>__<kernel>.log # lines: LAUNCH <i> <kernel> v0=<ticks> ...
Writes:
  syscnt_compare.csv
  syscnt_compare.md
"""
import argparse
import csv
import pathlib
import re
import statistics

BOARD_RE = re.compile(r"REP\s+\d+\s+\S+\s+v0=(-?\d+)")
CAM_RE = re.compile(r"LAUNCH\s+\d+\s+(\S+)\s+v0=(-?\d+)")


def read_dir(path: pathlib.Path, pattern: re.Pattern, value_group: int = 1):
    out = {}
    if not path.is_dir():
        return out
    for f in sorted(path.glob("*.log")):
        m = re.match(r"(.+?)__(.+)\.log$", f.name)
        if not m:
            continue
        case, kernel = m.group(1), m.group(2)
        vals = [
            int(x.group(value_group))
            for line in f.read_text().splitlines()
            if (x := pattern.search(line))
        ]
        if vals:
            out[(case, kernel)] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parent))
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    board = read_dir(root / "results" / "board", BOARD_RE)
    cam = read_dir(root / "results" / "camodel_ind_all", CAM_RE, value_group=2)
    rows = []
    for key in sorted(set(board) | set(cam)):
        case, kernel = key
        bv = board.get(key, [])
        cv = cam.get(key, [])
        bmed = statistics.median(bv) if bv else None
        bmin = min(bv) if bv else None
        bmax = max(bv) if bv else None
        cval = statistics.median(cv) if cv else None
        diff = (cval - bmed) / bmed * 100.0 if (cval is not None and bmed) else None
        rows.append(
            dict(
                case=case,
                kernel=kernel,
                board_median=bmed,
                board_min=bmin,
                board_max=bmax,
                camodel=cval,
                diff_pct=diff,
            )
        )

    csv_path = root / "syscnt_compare.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "kernel", "board_median", "board_min", "board_max", "camodel", "diff_pct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["| case | kernel | board median (min) | CAModel | diff |", "|---|---|---:|---:|---:|"]
    for r in rows:
        def f(x):
            return "" if x is None else f"{x:.0f}"

        d = r["diff_pct"]
        diff_s = "" if d is None else f"{d:+.1f}%"
        md.append(
            f"| {r['case']} | {r['kernel']} | {f(r['board_median'])} ({f(r['board_min'])}) | "
            f"{f(r['camodel'])} | {diff_s} |"
        )
    (root / "syscnt_compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {csv_path}")
    print(f"wrote {root / 'syscnt_compare.md'}")


if __name__ == "__main__":
    main()
