#!/usr/bin/env python3
"""Parse CAModel OPPROF directories produced by run_camodel_compare.py.

For SIMD:
  - sum SCALAR pipe cycles and load/store-like cycles from instr_exe.csv
For SIMT:
  - invoke camodel/parse_camodel_counts.py and report active SIMT span/memory ops

Usage:
  python3 analyze_camodel_simd_simt.py --simd-dir DIR --simt-dir DIR \
      --cases direct_load_1,direct_load_4,direct_store_4,dep_load_4
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys


def simd_summary(opprof):
    total = 0.0
    scalar = 0.0
    scalar_load = 0.0
    scalar_store = 0.0
    files = 0
    for path in glob.glob(os.path.join(opprof, "simulator", "*", "*instr_exe.csv")):
        files += 1
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cycles = float(row.get("cycles") or 0)
                total += cycles
                if row.get("pipe") == "SCALAR":
                    scalar += cycles
                    name = (row.get("instr") or "").upper()
                    if name.startswith("LD_"):
                        scalar_load += cycles
                    elif name.startswith("ST_"):
                        scalar_store += cycles
    return {
        "files": files,
        "total_cycles": total,
        "scalar_pipe_cycles": scalar,
        "scalar_load_like_cycles": scalar_load,
        "scalar_store_like_cycles": scalar_store,
    }


def simt_summary(opprof, parser):
    tmp = "/tmp/_camodel_parsed.json"
    subprocess.run([sys.executable, parser, opprof, "-o", tmp],
                   check=True, stdout=subprocess.DEVNULL)
    data = json.load(open(tmp))
    span = 0
    memory_ops = 0
    int_alu_ops = 0
    active = 0
    for unit, info in data.get("per_unit", {}).items():
        sp = info.get("span") or {}
        if sp.get("delta"):
            active += 1
            span += int(sp["delta"])
            gc = info.get("group_counts", {})
            memory_ops += int(gc.get("memory", 0))
            int_alu_ops += int(gc.get("int_alu", 0))
    return {
        "active_units": active,
        "simt_span_cycles": span,
        "memory_ops": memory_ops,
        "int_alu_ops": int_alu_ops,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simd-dir", required=True)
    ap.add_argument("--simt-dir", required=True)
    ap.add_argument("--cases", required=True)
    args = ap.parse_args()

    parser = os.path.expanduser(
        "~/triton-ascend/third_party/ascend/costmodel/profiles/microbench/"
        "data_provider/camodel/parse_camodel_counts.py")
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    simd_dirs = sorted(glob.glob(os.path.join(args.simd_dir, "OPPROF_*")))
    simt_dirs = sorted(glob.glob(os.path.join(args.simt_dir, "OPPROF_*")))

    print("case,mode,total_cycles,scalar_pipe_cycles,scalar_load_like_cycles,"
          "scalar_store_like_cycles,active_units,simt_span_cycles,memory_ops,int_alu_ops")
    for case, sd, st in zip(cases, simd_dirs, simt_dirs):
        s = simd_summary(sd)
        t = simt_summary(st, parser)
        print(f"{case},simd,{s['total_cycles']:.0f},{s['scalar_pipe_cycles']:.0f},"
              f"{s['scalar_load_like_cycles']:.0f},{s['scalar_store_like_cycles']:.0f},"
              f",,,"
        )
        print(f"{case},simt_only,,,,,{t['active_units']},{t['simt_span_cycles']},"
              f"{t['memory_ops']},{t['int_alu_ops']}")


if __name__ == "__main__":
    main()
