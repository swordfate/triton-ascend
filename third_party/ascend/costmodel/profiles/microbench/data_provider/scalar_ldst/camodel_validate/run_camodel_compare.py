#!/usr/bin/env python3
"""Run a few Triton scalar cases under CAModel and report per-core block cycles.

Usage (on remote with CAModel environment):
    python3 run_camodel_compare.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[7]  # scalar_ldst -> ... -> repo root

CASES = [
    "direct_load_1",
    "direct_load_4",
    "direct_store_4",
    "dep_load_4",
]

START_RE = re.compile(
    r"\[info\]\s+\[(\d+)\]\s+\[block_start\]\s*:\s*AIV,\s*task_id=\d+,\s*core_id=(\d+),\s*block_id=(\d+)"
)
END_RE = re.compile(
    r"\[info\]\s+\[(\d+)\]\s+\[block_end\]\s*:\s*AIV,\s*task_id=\d+,\s*core_id=(\d+),\s*block_id=(\d+)"
)


def run_case(case):
    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    env.setdefault("TRITON_ASCEND_COMPILE_MODE", "simd")
    env.setdefault("TRITON_ASCEND_AUTO_SIMT_SCOPE", "off")
    # If not already set by caller, set the simulator environment used on this box.
    simlib = "/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend950PR_9589/lib"
    bishlib = "/home/c00946898/.conda/envs/bisheng_build/lib"
    bishlib2 = "/home/c00946898/bishengir-install/lib"
    env["LD_LIBRARY_PATH"] = ":".join(
        x for x in [simlib, bishlib, bishlib2, env.get("LD_LIBRARY_PATH", "")] if x)
    env["GE_INIT_DISABLE"] = "1"
    env["TASK_QUEUE_ENABLE"] = "0"
    cmd = [
        "msprof", "op", "simulator", "--kernel-name=scalar_camodel_cases",
        sys.executable, str(HERE / "scalar_camodel_cases.py"), case, "--grid", "2",
    ]
    proc = subprocess.run(cmd, cwd=HERE, env=env, text=True,
                          capture_output=True, timeout=600)
    text = proc.stdout + "\n" + proc.stderr
    starts = {}
    ends = {}
    for m in START_RE.finditer(text):
        key = (m.group(2), m.group(3))
        starts[key] = int(m.group(1))
    for m in END_RE.finditer(text):
        key = (m.group(2), m.group(3))
        ends[key] = int(m.group(1))
    durations = {}
    for key in sorted(starts):
        if key in ends:
            durations[key] = ends[key] - starts[key]
    return durations, proc.returncode


def main():
    mode = os.environ.get("TRITON_ASCEND_COMPILE_MODE", "simd")
    print(f"# mode={mode}", file=sys.stderr)
    print("case,mode,core_id,block_id,duration_cycles")
    for case in CASES:
        durations, rc = run_case(case)
        if rc != 0:
            print(f"# {case} failed rc={rc}", file=sys.stderr)
        for (core, block), dur in durations.items():
            print(f"{case},{mode},{core},{block},{dur}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
