#!/usr/bin/env python3
"""Quick parser for scratch load+vector-compute CAModel cases."""
import json
import pathlib
import re
import sys


def hexadd(a, b):
    return int(a, 16) + int(b, 16)


def ts(line):
    m = re.match(r"\[info\]\s+\[?(\d+)\]?", line)
    return int(m.group(1)) if m else None


def parse_simd(root):
    base = root / "dump" if (root / "dump").is_dir() else root
    instr = (base / "core0.veccore0.instr_log.dump").read_text().splitlines()
    sq = (base / "core0.veccore0.ccu.scalar_issque.dump").read_text().splitlines()
    # instruction metadata from instr_log
    ins = {}
    for line in instr:
        m = re.search(
            r"\[(\d+)\].*\(PC: (0x[0-9a-fA-F]+)\).*?\(ID: (\d+)\) (LD_XD_XN_IMM|ST_XD_XN_IMM)\s+(.*)",
            line,
        )
        if not m:
            continue
        t, pc, iid, name, rest = int(m.group(1)), m.group(2).lower(), int(m.group(3)), m.group(4), m.group(5)
        am = re.search(r"XN:X\d+=(0x[0-9a-fA-F]+), IMM:(0x[0-9a-fA-F]+|\d+)", rest)
        addr = hexadd(am.group(1), am.group(2)) if am else None
        exec_m = re.search(r"execTime:(0x[0-9a-fA-F]+)", rest)
        ins.setdefault(pc, []).append(
            {"time": t, "id": iid, "name": name, "addr": addr, "execTime": int(exec_m.group(1), 16) if exec_m else None}
        )
    # push/retire from scalar issue queue
    events = {pc: {"push": None, "retire": None} for pc in ins}
    for line in sq:
        m = re.search(r"\[info\]\s+(\d+)\s+\[Push Instr\] name=(\w+) pc=(0x[0-9A-Fa-f]+) id=(\d+)", line)
        if m:
            pc = m.group(3).lower()
            if pc in events:
                events[pc]["push"] = int(m.group(1))
        m = re.search(r"\[info\]\s+(\d+)\s+\[RETIRE INSTR\] instr.name=(\w+)\s+instr.pc=(0x[0-9A-Fa-f]+)\s+instr.id=(\d+)", line)
        if m:
            pc = m.group(3).lower()
            if pc in events:
                events[pc]["retire"] = int(m.group(1))

    loads = []  # GM scalar loads (address < 0x10000000 means UB)
    ub_stores = []
    out_stores = []
    for pc, metas in ins.items():
        for meta in metas:
            if meta["addr"] is None:
                continue
            if meta["addr"] < 0x10000000:  # UB staging
                if meta["name"] == "ST_XD_XN_IMM":
                    ub_stores.append({**meta, **events[pc], "pc": pc})
                continue
            if meta["name"] == "LD_XD_XN_IMM":
                loads.append({**meta, **events[pc], "pc": pc})
            elif meta["name"] == "ST_XD_XN_IMM":
                out_stores.append({**meta, **events[pc], "pc": pc})

    # VF block
    vf_record = None
    pb_push = None
    rv = {}
    for line in instr:
        if "PUSH_PB" in line and pb_push is None:
            pb_push = ts(line)
        m = re.search(r"\[(\d+)\].*VF\s+addr:.*vf_execute_time:\s*(\d+)", line)
        if m:
            vf_record = {"time": int(m.group(1)), "vf_execute_time": int(m.group(2))}
        for op in ["RV_VBR", "RV_VLDI", "RV_VADD", "RV_VSTI", "RV_SEND", "RV_PLT", "RV_SMOVI"]:
            if f") {op}" in line:
                rv.setdefault(op, []).append(ts(line))
    rv["pb_push"] = pb_push
    return {
        "case": root.name,
        "unit": "simd",
        "loads": loads,
        "ub_stores": ub_stores,
        "out_stores": out_stores,
        "vf_record": vf_record,
        "rv": rv,
    }


def parse_simt(root):
    base = root / "dump" if (root / "dump").is_dir() else root
    lsu = (base / "core0.veccore0.rvec.simt.lsu.dump").read_text().splitlines()
    instr = (base / "core0.veccore0.instr_log.dump").read_text().splitlines()
    ev = []
    for line in lsu:
        m = re.search(r"\[(\d+)\] (RECV|ISSUE|RETIRE)_INSTR name=(SIMT_LDG|SIMT_LDS|SIMT_STS) id=(\d+)", line)
        if m:
            ev.append({"time": int(m.group(1)), "kind": m.group(2), "name": m.group(3), "id": int(m.group(4))})
    fadd = []
    for line in instr:
        if "SIMT_FADD" in line:
            m = re.search(r"\[(\d+)\].*SIMT_FADD.*exec_time:\s*(\d+)", line)
            if m:
                fadd.append({"time": int(m.group(1)), "exec_time": int(m.group(2))})
    return {"case": root.name, "unit": "simt", "lsu": ev, "fadd": fadd}


def main():
    out = []
    for arg in sys.argv[1:]:
        root = pathlib.Path(arg)
        base = root / "dump" if (root / "dump").is_dir() else root
        simt_lsu = base / "core0.veccore0.rvec.simt.lsu.dump"
        is_simt = simt_lsu.exists() and "SIMT_LDG" in simt_lsu.read_text(errors="ignore")
        if is_simt:
            out.append(parse_simt(root))
        else:
            out.append(parse_simd(root))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
