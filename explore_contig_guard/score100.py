#!/usr/bin/env python3
"""score100.py <RUN_ROOT> - score a 100-case run dir against table_100case.md.

Finds each case's route.json under <root>/s*/ or <root>/ directly; takes the
LAST JSON doc in the file (fresh report), uses the decision kind's candidate
elapsed cost as the whole-kernel prediction.
"""
import json, os, re, sys

MANIFEST = "/root/w/cc-harness/costmodel_generalization/generalization_matrix_v140_all_100.json"


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
    cases = [c["name"] for c in json.load(open(MANIFEST))["cases"]]
    meas = {}
    for line in open("/root/work/table_100case.md", errors="ignore"):
        m = re.match(r"\| *\d+ \| (\S+) \| (.*?) \| *([\d.]+) \|", line)
        if m:
            meas[m.group(1)] = float(m.group(3))
    print("measured entries parsed:", len(meas), file=sys.stderr)

    root = sys.argv[1]
    # index all route.json files once
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".route.json"):
                case = f.split("__")[0]
                index.setdefault(case, os.path.join(dirpath, f))

    n_pass = n_fail = n_none = 0
    fails = []
    errs = []
    for case in cases:
        m = meas.get(case)
        f = index.get(case)
        p = None
        rt = "NO-RUN"
        if f:
            docs = load_docs(f)
            if docs:
                r = docs[-1]
                rt = r.get("effective_decision_kind") or "?"
                p = (r.get("candidate_elapsed_costs") or {}).get(rt)
                if p is not None:
                    p = p / 1000.0
        if m is None or p is None:
            n_none += 1
            continue
        e = 100.0 * (p - m) / m
        errs.append(abs(e))
        if abs(e) <= 10:
            n_pass += 1
        else:
            n_fail += 1
            fails.append((case, rt, m, p, e))
    print("PASS<=10%%: %d  FAIL: %d  N/A: %d  of %d" %
          (n_pass, n_fail, n_none, len(cases)))
    if errs:
        errs.sort()
        print("abs-err median %.2f%% mean %.2f%% max %.2f%%" %
              (errs[len(errs)//2], sum(errs)/len(errs), errs[-1]))
    for (case, rt, m, p, e) in fails:
        print("  FAIL %-46s %-18s meas=%9.3f pred=%9.3f err=%+7.2f%%" %
              (case, rt, m, p, e))


if __name__ == "__main__":
    main()
