#!/usr/bin/env python3
"""Acceptance self-check for the stride calibration round.

Runs every machine-checkable acceptance item from the handoff note and prints
a PASS/FAIL checklist:

  1. every delivered dataset has zero stride == 1 rows
     (and the 2-D dataset has zero fully contiguous rows);
  2. both path checks report mismatches = 0 and compile_failures = 0;
  3. every predictor reproduces its delivered errors point by point
     (max |predict - errors.pred_ns| < 1e-6 -> VERIFY OK);
  4. the feature reference agrees with the C++ mirror and the committed v2
     feature module (0 mismatch);
  5. optionally, re-running the default refit command reproduces every
     delivered JSON byte for byte (--with-refit).

Usage:
    python3 scripts/check_acceptance.py
    python3 scripts/check_acceptance.py --with-refit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
RESULTS = HERE / "results"
REFIT = RESULTS / "refit_no_stride1_v1"
DTYPE = RESULTS / "dtype_calibration_v1"

DATASETS = [
    REFIT / "load/dataset_simt_v8.csv",
    REFIT / "store/dataset.csv",
    DTYPE / "dataset.csv",
    DTYPE / "dataset_2d.csv",
]
PATH_CHECKS = [
    DTYPE / "ir_evidence/dtype_stride_path_check_template.json",
    DTYPE / "ir_evidence/stride2d_path_check.json",
]
PREDICTORS = [
    ("phase 0 v7 load", ["scripts/predict_strided_load_v7_simple.py", "--verify",
                         "--model", str(REFIT / "load/model_v7_simple_refit.json"),
                         "--errors", str(REFIT / "load/errors_all.csv")]),
    ("phase 0 v8 SIMT load", ["scripts/predict_strided_load_v8.py", "--verify",
                              "--model", str(REFIT / "load/model_v4_simple_refit.json"),
                              "--errors", str(REFIT / "load/errors_simt_v8.csv")]),
    ("phase 0 store", ["scripts/predict_strided_store_v1.py", "--verify",
                       "--model", str(REFIT / "store/model_store_v1_refit.json"),
                       "--dataset", str(REFIT / "store/dataset.csv"),
                       "--errors", str(REFIT / "store/errors_all.csv")]),
    ("phase A dtype", ["scripts/predict_dtype_model.py", "--verify"]),
    ("phase B 2D", ["scripts/predict_2d_model.py", "--verify"]),
    ("feature reference", ["scripts/stride_feature_reference.py", "--verify"]),
]

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def check_datasets():
    problems = []
    for path in DATASETS:
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            problems.append(f"empty {path}")
            continue
        if "stride" in rows[0]:
            bad = sum(1 for r in rows
                      if abs(float(r["stride"]) - 1.0) < 1e-12)
            if bad:
                problems.append(f"{path.name}: {bad} stride==1 rows")
        if path.name == "dataset_2d.csv":
            bad = sum(1 for r in rows if int(float(r["contiguous"])) == 1)
            if bad:
                problems.append(f"{path.name}: {bad} contiguous rows")
    total = sum(len(list(csv.DictReader(p.open(newline=""))))
                for p in DATASETS if p.exists())
    record("delivered datasets contain no stride==1 / contiguous rows",
           not problems, f"{total} rows across {len(DATASETS)} datasets"
           if not problems else "; ".join(problems))


def check_path_checks():
    problems = []
    for path in PATH_CHECKS:
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        data = json.loads(path.read_text())
        rows = data["results"]
        mismatches = sum(1 for r in rows if r.get("match") is False)
        failures = sum(1 for r in rows if not r.get("compiled"))
        if mismatches or failures:
            problems.append(f"{path.name}: mismatches={mismatches} "
                            f"compile_failures={failures}")
    record("path checks report 0 mismatches and 0 compile failures",
           not problems, "; ".join(problems) if problems else
           f"{len(PATH_CHECKS)} path checks")


def check_predictors():
    for name, argv in PREDICTORS:
        proc = subprocess.run([sys.executable, *argv], cwd=HERE,
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        record(f"predictor VERIFY OK: {name}", "VERIFY OK" in out,
               "" if "VERIFY OK" in out else out.strip().splitlines()[-1][:120]
               if out.strip() else "no output")


def check_refit_reproducible():
    def hashes():
        return {p.relative_to(REFIT).as_posix():
                hashlib.md5(p.read_bytes()).hexdigest()
                for p in sorted(REFIT.rglob("*")) if p.is_file()}
    before = hashes()
    proc = subprocess.run([sys.executable, "scripts/refit_no_stride1.py"],
                          cwd=HERE, capture_output=True, text=True)
    if proc.returncode != 0:
        record("refit default command reproduces deliverables", False,
               proc.stderr.strip().splitlines()[-1][:120])
        return
    after = hashes()
    diff = [k for k in before if before[k] != after.get(k)]
    record("refit default command reproduces deliverables bit for bit",
           not diff and set(before) == set(after),
           f"{len(before)} files" if not diff else f"changed: {diff[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-refit", action="store_true",
                    help="also re-run the default refit and compare hashes")
    args = ap.parse_args()

    check_datasets()
    check_path_checks()
    check_predictors()
    if args.with_refit:
        check_refit_reproducible()

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed != len(results):
        raise SystemExit("ACCEPTANCE CHECK FAILED")
    print("ACCEPTANCE OK")


if __name__ == "__main__":
    main()
