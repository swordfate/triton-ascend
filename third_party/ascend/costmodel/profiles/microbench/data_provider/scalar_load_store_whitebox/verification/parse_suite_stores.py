#!/usr/bin/env python3
"""Extract store instruction issue/retire windows from padded-kernel OPPROF dumps.

Usage:
  parse_suite_stores.py <OPPROF_dir> simt|simd
"""
import re
import sys
from pathlib import Path


def read(p):
    try:
        return Path(p).read_text(errors="replace")
    except FileNotFoundError:
        return ""


def parse_simt(dump):
    text = read(dump / "core0.veccore0.rvec.simt.lsu.dump")
    events = {}
    for line in text.splitlines():
        m = re.search(r"\[(\d+)\] (RECV_INSTR|ISSUE_INSTR|RETIRE_INSTR) name=(\S+) id=(\d+)", line)
        if not m:
            continue
        t, kind, name, iid = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        if name != "SIMT_STG":
            continue
        r = events.setdefault(iid, {"name": name, "issue": [], "retire": []})
        if kind == "ISSUE_INSTR":
            r["issue"].append(t)
        elif kind == "RETIRE_INSTR":
            r["retire"].append(t)
    dc = {}
    text = read(dump / "core0.veccore0.rvec.simt.dc.dump")
    for line in text.splitlines():
        if "[TagRam]" not in line or "isaId:" not in line:
            continue
        m = re.search(r"cmd:(\w+).*?addr:(0x[0-9a-fA-F]+).*?isaId:(\d+)", line)
        if m and m.group(1) == "WRITE":
            dc.setdefault(int(m.group(3)), []).append(int(m.group(2), 16))
    out = []
    for iid, r in sorted(events.items(), key=lambda kv: min(kv[1]["issue"]) if kv[1]["issue"] else 10**12):
        iss = min(r["issue"]) if r["issue"] else None
        ret = max(r["retire"]) if r["retire"] else None
        out.append({
            "id": iid,
            "kind": "SIMT_STG",
            "issue": iss,
            "retire": ret,
            "active": (ret - iss) if iss is not None and ret is not None else None,
            "addr": hex(dc[iid][0]) if dc.get(iid) else None,
        })
    return out


def parse_simd(dump):
    instr = {}
    text = read(dump / "core0.veccore0.instr_log.dump")
    for line in text.splitlines():
        m = re.search(
            r"\[(\d+)\] \(PC: (0x[0-9a-fA-F]+)\)\s+(\S+)\s+:\s+\(Binary: [^)]*\)\s+\(ID: (\d+)\)\s+(\S+)(.*)",
            line,
        )
        if not m:
            continue
        name, rest = m.group(5), m.group(6)
        if name != "MOV_SRC_TO_DST_ALIGNv2" or "Dst:OUT" not in rest:
            continue
        iid = int(m.group(4))
        am = re.search(r"XD:\S+=?(0x[0-9a-fA-F]+)", rest)
        instr[iid] = {"id": iid, "t": int(m.group(1)), "pc": m.group(2).lower(), "addr": am.group(1) if am else None}
    queue = read(dump / "core0.veccore0.ccu.mte3_issque.dump")
    issue, retire = {}, {}
    for line in queue.splitlines():
        m = re.search(r"\[Push Instr\] name=(\S+) pc=(\S+) id=(\d+)", line)
        if m:
            tm = re.search(r"\[info\]\s+(\d+)", line)
            if tm:
                issue[int(m.group(3))] = int(tm.group(1))
            continue
        m = re.search(r"\[RETIRE INSTR\] instr\.name=(\S+)\s+instr\.pc=(\S+)\s+instr\.id=(\d+)", line)
        if m:
            tm = re.search(r"\[info\]\s+(\d+)", line)
            if tm:
                retire[int(m.group(3))] = int(tm.group(1))
    out = []
    for iid, it in sorted(instr.items(), key=lambda kv: kv[1]["t"]):
        iss = issue.get(iid)
        ret = retire.get(iid)
        out.append({
            "id": iid,
            "kind": "MTE3_MOV",
            "pc": it["pc"],
            "addr": it["addr"],
            "issue": iss,
            "retire": ret,
            "active": (ret - iss) if iss is not None and ret is not None else None,
        })
    return out


def main():
    dump = Path(sys.argv[1])
    mode = sys.argv[2]
    rows = parse_simt(dump) if mode == "simt" else parse_simd(dump)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
