#!/usr/bin/env python3
"""Compile-only 100-case prediction driver: for each manifest case run its
bench driver with --route auto --report <name>__auto_f1.route.json (the
compiler writes the route dump; no measurement, no collector).  Shards run
in parallel across cards 1/2.

Usage: python3 run100_pred.py <SHARD> <OUTROOT>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

MANIFEST = ("/root/w/cc-harness/costmodel_generalization/"
            "generalization_matrix_v140_all_100.json")
HARNESS = "/root/w/cc-harness/costmodel_generalization"


def main():
    shard = int(sys.argv[1])
    outroot = Path(sys.argv[2])
    outroot.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(MANIFEST))["cases"][shard::8]
    env = dict(os.environ)
    env["PATH"] = "/home/c00946898/bishengir-install/bin:" + env.get("PATH", "")
    env["ASCEND_RT_VISIBLE_DEVICES"] = "1" if shard % 2 == 0 else "2"
    env["TRITON_CACHE_DIR"] = str(outroot / ("cache_s%d" % shard))
    for case in cases:
        name = "%s__auto_f1" % case["name"]
        report = outroot / (name + ".route.json")
        if report.exists():
            print("sh%d SKIP %s" % (shard, case["name"]), flush=True)
            continue
        cmd = [sys.executable, str(Path(HARNESS) / case["driver"]),
               "--route", "auto", "--factor", "1",
               "--warmup", "1", "--repeats", "1",
               "--report", str(report),
               *case.get("arguments", [])]
        try:
            r = subprocess.run(cmd, cwd=HARNESS, env=env, timeout=240,
                               capture_output=True)
            rc = r.returncode
        except Exception as exc:
            rc = -99
            print("sh%d ERR %s %s" % (shard, case["name"], str(exc)[:80]),
                  flush=True)
        print("sh%d %s rc=%d" % (shard, case["name"], rc), flush=True)
    print("sh%d DONE" % shard, flush=True)


main()
