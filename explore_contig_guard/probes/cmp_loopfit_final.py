#!/usr/bin/env python3
"""Compare loopfit predictions (RUNLF_PRED route dumps) against the final
review table's per-case errors - list PASS<->FAIL flips."""
import json
import os
import re
import sys
from pathlib import Path

FINAL_TABLE = Path("/root/work/triton-ascend/explore_contig_guard/"
                   "CONTIGUOUS_REVIEW_20261118.md")
MANIFEST = ("/root/w/cc-harness/costmodel_generalization/"
            "generalization_matrix_v140_all_100.json")


def load_docs(path):
    s = open(path, errors="ignore").read()
    dec = json.JSONDecoder()
    docs, i = [], 0
    while i < len(s):
        rest = s[i:].lstrip()
        if not rest:
            break
        off = len(s) - i - len(rest)
        try:
            obj, end = dec.raw_decode(rest)
        except json.JSONDecodeError:
            break
        docs.append(obj)
        i = off + end
    return docs


def main():
    # current predictions
    cur = {}
    for dirpath, _, files in os.walk("/tmp/RUNLF_PRED"):
        for f in files:
            if f.endswith(".route.json"):
                case = f.split("__")[0]
                docs = load_docs(os.path.join(dirpath, f))
                if not docs:
                    continue
                r = docs[-1]
                rt = r.get("effective_decision_kind") or "?"
                p = (r.get("candidate_elapsed_costs") or {}).get(rt)
                if p is not None:
                    cur[case] = (rt, p / 1000.0)
    # final-table errors: | # | case | ... | 终版预测 | 终版误差 | 终版 |
    fin = {}
    for line in FINAL_TABLE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*\d+\s*\|\s*(\S+)\s*\|.*\|([^|]*)\|([^|]*)\|"
                     r"\s*(PASS|FAIL|\*\*FAIL\*\*)\s*\|", line)
        if not m:
            m2 = re.match(r"\|\s*\d+\s*\|\s*(\S+)\s*\|", line)
            if m2 and ("PASS" in line or "FAIL" in line):
                cells = [c.strip() for c in line.split("|")]
                if len(cells) >= 12:
                    fin[m2.group(1)] = (cells[-3], cells[-2], cells[-1])
        else:
            fin[m.group(1)] = (m.group(2).strip(), m.group(3).strip(),
                               m.group(4).strip())

    def fnum(x):
        try:
            return float(re.sub(r"[^0-9.+-]", "", str(x)))
        except (TypeError, ValueError):
            return None

    flips = []
    for case, (rt, pred) in sorted(cur.items()):
        f = fin.get(case)
        if not f:
            continue
        fe = fnum(f[1])
        ce = None
        m = fin.get(case)
        # recompute current err from final-table measured column? use
        # score100-style: need meas; final table has 实测 col
        meas = None
        if m:
            meas = None
        # parse measured from the table's 实测 column instead
        for line in FINAL_TABLE.read_text(encoding="utf-8").splitlines():
            pass
        fe = fnum(f[1])
        ce_f = None
        # current err: recompute from meas parsed below
        if fe is None:
            continue
        # get measured from table col 4 (实测)
        # stored fin only has last3; reparse quickly
        ce = None
        flips.append((case, fe, rt, pred, ce_f, ce))
    # simpler: reparse full rows
    rows = {}
    for line in FINAL_TABLE.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 12 and cells[1].isdigit() is False and \
                cells[2] and not cells[2].startswith("-"):
            pass
    # do a robust reparse: find lines with 12 cells where cell[1] is int
    for line in FINAL_TABLE.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 12:
            continue
        try:
            int(cells[1])
        except ValueError:
            continue
        case = cells[2].strip('`')
        meas = cells[4]
        ferr = cells[10]
        fpred = cells[9]
        rows[case] = (meas, fpred, ferr, cells[11])

    print("%-46s %10s %10s %8s -> %8s  %s" %
          ("case", "meas", "final_pred", "final_err", "loopfit_err",
           "verdict"))
    n_pass = 0
    for case in sorted(rows):
        meas, fpred, ferr, fverdict = rows[case]
        try:
            mv = float(re.sub(r"[^0-9.]", "", meas))
        except ValueError:
            continue
        if case not in cur:
            print("%-46s %10.3f %10s %8s -> %8s  NO-ROUTE" %
                  (case, mv, fpred, ferr, "-"))
            continue
        rt, pred = cur[case]
        ce = 100.0 * (pred - mv) / mv
        verdict = "PASS" if abs(ce) < 10 else "FAIL"
        old_v = "PASS" if "PASS" in fverdict else "FAIL"
        mark = ""
        if verdict != old_v:
            mark = "  <<<< FLIP %s->%s" % (old_v, verdict)
        if verdict == "PASS":
            n_pass += 1
        print("%-46s %10.3f %10s %8s -> %8s  %s%s" %
              (case, mv, fpred, ferr, "%+.2f%%" % ce, verdict, mark))
    print("\nloopfit PASS:", n_pass, " scoreable:", len(cur))


main()
