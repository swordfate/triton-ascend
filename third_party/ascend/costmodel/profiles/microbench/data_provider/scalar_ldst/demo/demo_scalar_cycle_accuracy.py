#!/usr/bin/env python3
"""Demo: compare scalar cost-model cycle formula against live CCE measurements.

This script actually builds (if necessary) and runs the scalar_ldst CCE probes
on the current NPU, parses their output, and compares the result with the
formula used by StageCostModels.cpp.

Run:
    cd .../scalar_ldst
    source ~/env_ascend.sh
    conda activate wj_autoscope
    export ASCEND_RT_VISIBLE_DEVICES=1
    python3 demo/demo_scalar_cycle_accuracy.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve()
while not (REPO / ".git").exists() and REPO.parent != REPO:
    REPO = REPO.parent
if not (REPO / ".git").exists():
    raise SystemExit("Cannot locate repository root")

SCALAR_LDST = REPO / "third_party/ascend/costmodel/profiles/microbench/data_provider/scalar_ldst"
DEMO_DIR = SCALAR_LDST / "demo"
PROFILE_JSON = REPO / "third_party/ascend/costmodel/profiles/microbench/ascend_davidv100_v1.json"
SIMD_SIMT_JSON = REPO / "third_party/ascend/costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json"

PROBES = [
    "simd_scalar_gm_memory",
    "simd_scalar_gm_dep",
]

# Ops used to fit the committed JSON parameters.
CALIBRATION_OPS = {1, 2, 4, 8}


def run(cmd, **kwargs):
    print("+", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def compile_probe(name):
    env = os.environ.copy()
    # On the remote server, env_ascend.sh sets ccec and ASCEND_TOOLKIT_HOME.
    # If ccec is already in PATH this is harmless.
    bash = [
        "bash", "-lc",
        f"source ~/env_ascend.sh >/dev/null 2>&1 || true; "
        f"cd {SCALAR_LDST} && "
        f"INC=/home/c00946898/AscendNPU-IR/bishengir/lib/Template/include; "
        f"TK=\"$ASCEND_TOOLKIT_HOME\"; "
        f"ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 "
        f"-I\"$INC\" {name}.cce -o {name}.o && "
        f"g++ -O2 {name}_host.cpp -o {name}_host "
        f"-I\"$TK/x86_64-linux/pkg_inc\" -I\"$TK/include\" "
        f"-L\"$TK/lib64\" -lruntime -lascendcl",
    ]
    run(bash, env=env)


def run_probe(name):
    exe = SCALAR_LDST / f"{name}_host"
    if not exe.exists():
        compile_probe(name)
    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    proc = subprocess.run(
        [str(exe)],
        cwd=SCALAR_LDST,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def parse_memory_output(text):
    """Parse simd_scalar_gm_memory_host CSV after the header line."""
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("ops") or line.startswith("SIMD"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            ops = int(parts[0])
            load_cycles = float(parts[1])
            load_rate = float(parts[2])
            store_cycles = float(parts[3])
            store_rate = float(parts[4])
        except ValueError:
            continue
        rows[ops] = {
            "load_cycles": load_cycles,
            "load_rate": load_rate,
            "store_cycles": store_cycles,
            "store_rate": store_rate,
        }
    return rows


def parse_dep_output(text):
    """Parse simd_scalar_gm_dep_host CSV."""
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("ops") or line.startswith("SIMD"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            ops = int(parts[0])
            indep = float(parts[1])
            dep = float(parts[2])
            extra = float(parts[3])
        except ValueError:
            continue
        rows[ops] = {
            "independent_cycles": indep,
            "dependent_cycles": dep,
            "extra_per_edge_cycles": extra,
        }
    return rows


def read_profile_numbers():
    with PROFILE_JSON.open() as f:
        data = json.load(f)
    measurements = data["measurements"]
    load_rate = measurements["simd.scalar.load.throughput"]["value"]
    load_latency = measurements["simd.scalar.load.direct_latency"]["value"]
    store_rate = measurements["simd.scalar.store.throughput"]["value"]
    store_latency = measurements["simd.scalar.store.direct_latency"]["value"]

    with SIMD_SIMT_JSON.open() as f:
        sim = json.load(f)
    dep = sim["simd"]["stage_resources"]["scalar_memory"][
        "indirect_dependency_latency_system_cycles"]
    return load_rate, load_latency, store_rate, store_latency, dep


def predicted(count, rate, latency, dep_edges=0.0, dep_latency=0.0):
    return count / rate + latency + dep_edges * dep_latency


def print_table(name, measured, rate, latency):
    cal = {k: v for k, v in measured.items() if k in CALIBRATION_OPS}
    unseen = {k: v for k, v in measured.items() if k not in CALIBRATION_OPS}

    def emit(title, data):
        print(f"\n{title}")
        print(f"{'ops':>4} {'measured_cycles':>16} {'predicted_cycles':>17} {'error_%':>8}")
        for ops in sorted(data):
            meas = data[ops]
            pred = predicted(ops, rate, latency)
            err = (pred - meas) / meas * 100.0
            print(f"{ops:>4} {meas:>16.4f} {pred:>17.4f} {err:>7.2f}%")

    emit(f"{name} [calibration ops, used for fitting]", cal)
    emit(f"{name} [unseen validation ops]", unseen)


def main():
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    print(f"repo: {REPO}")
    print(f"scalar_ldst: {SCALAR_LDST}")
    print("Running live CCE probes ...")

    mem_text = run_probe("simd_scalar_gm_memory")
    dep_text = run_probe("simd_scalar_gm_dep")

    mem = parse_memory_output(mem_text)
    dep = parse_dep_output(dep_text)
    if not mem or not dep:
        print("CCE output parse failed; raw outputs:")
        print(mem_text)
        print(dep_text)
        raise SystemExit(1)

    load_rate, load_latency, store_rate, store_latency, dep_latency = read_profile_numbers()

    print("Committed profile values:")
    print(f"  SIMD load  rate={load_rate} latency={load_latency}")
    print(f"  SIMD store rate={store_rate} latency={store_latency}")
    print(f"  SIMD dependency={dep_latency}")

    print_table("SIMD scalar loads (live CCE)", {
        ops: v["load_cycles"] for ops, v in mem.items()
    }, load_rate, load_latency)

    print_table("SIMD scalar stores (live CCE)", {
        ops: v["store_cycles"] for ops, v in mem.items()
    }, store_rate, store_latency)

    print("\nSIMD scalar dependent-load chain (live CCE)")
    print(f"{'ops':>4} {'measured_dep':>14} {'predicted_dep':>15} {'error_%':>8}")
    for ops in sorted(dep):
        meas = dep[ops]["dependent_cycles"]
        pred = predicted(ops, load_rate, load_latency, dep_edges=ops, dep_latency=dep_latency)
        err = (pred - meas) / meas * 100.0
        print(f"{ops:>4} {meas:>14.4f} {pred:>15.4f} {err:>7.2f}%")


if __name__ == "__main__":
    main()
