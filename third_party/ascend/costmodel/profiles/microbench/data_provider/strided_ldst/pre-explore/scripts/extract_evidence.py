#!/usr/bin/env python3
"""Emit compact raw CAModel evidence for selected (run, case) pairs."""
import json
import re
import sys
from pathlib import Path

CORE = "core0.veccore0"


def read(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    run_dir = Path(sys.argv[1])
    wanted = [int(x) for x in sys.argv[2:]]
    manifest = json.loads((run_dir / "launch_manifest.json").read_text())
    profile = sorted(run_dir.glob("OPPROF_*"))[0]
    kernel_dir = profile / "strided_load_kernel"
    if not kernel_dir.is_dir():
        kernel_dir = profile
    for case in manifest["cases"]:
        idx = case["order"]
        if idx not in wanted:
            continue
        dump = kernel_dir / str(idx) / "dump"
        if not dump.is_dir():
            dump = kernel_dir / "dump"
        lo = int(case["x_ptr"], 16)
        hi = lo + int(case["x_bytes"])
        section(f'{run_dir.name} case {idx}: mode={case["mode"]} stride={case["stride"]} block={case["block"]} x=[{hex(lo)},{hex(hi)})')
        rvec = read(dump / f"{CORE}.rvec.dump")
        for line in rvec.splitlines():
            if "core: 0, subcore:" in line:
                print(line)
        instr_ts = [int(m.group(1)) for m in re.finditer(r"\[info\] \[(\d+)\]", read(dump / f"{CORE}.instr_log.dump"))]
        if instr_ts:
            print(f"instr_log: first={min(instr_ts)} last={max(instr_ts)} active={max(instr_ts)-min(instr_ts)}")
        print("\n-- MTE2 queue (push/retire) --")
        for line in read(dump / f"{CORE}.ccu.mte2_issque.dump").splitlines():
            if "Push Instr" in line or "RETIRE INSTR" in line:
                print(line)
        print("\n-- SIMD IFU load/store VF instructions --")
        for line in read(dump / f"{CORE}.rvec.simd.ifu.dump").splitlines():
            if re.search(r"RV_VLDI|RV_VSTI|Start SIMD VF", line):
                print(line)
        print("\n-- SIMT LSU (LDG/LDS/STG) --")
        for line in read(dump / f"{CORE}.rvec.simt.lsu.dump").splitlines():
            if re.search(r"SIMT_(LDG|LDS|STG|STS)", line):
                print(line)
        print("\n-- SIMT DC TagRam --")
        for line in read(dump / f"{CORE}.rvec.simt.dc.dump").splitlines():
            if "[TagRam]" in line and "cmd:" in line:
                print(line)
        print("\n-- BIU commands/returns inside input buffer range --")
        for line in read(dump / "core0.biu.brif.log.dump").splitlines():
            m = re.search(r"\[info\] (\d+): (send_rd_cmd|recv_biu_data),.*?addr: (0x[0-9a-fA-F]+), size: (\d+)", line)
            if not m:
                continue
            addr = int(m.group(3), 16)
            if lo <= addr < hi:
                print(line)


if __name__ == "__main__":
    main()
