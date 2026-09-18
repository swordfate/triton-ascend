#!/usr/bin/env python3
"""Print scalar load instruction timelines for a padded-kernel CAModel dump."""
import argparse
import re
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
        if m:
            dc.setdefault(int(m.group(3)), []).append((m.group(1), int(m.group(2), 16)))
    rows = []
    for iid, r in events.items():
        if r["name"] != "SIMT_LDG":
            continue
        iss = min(r["issue"]) if r["issue"] else None
        ret = max(r["retire"]) if r["retire"] else None
        entries = dc.get(iid, [])
        addr = entries[0][1] if entries else None
        rows.append((iss, iid, ret, addr, r["name"]))
    for row in sorted(rows):
        print(f"SIMT id={row[1]:4d} issue={row[0]} retire={row[2]} lat={row[2]-row[0]} addr={hex(row[3]) if row[3] else '?'}")


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
        t, pc, iid, name, rest = int(m.group(1)), m.group(2).lower(), int(m.group(4)), m.group(5), m.group(6)
        em = re.search(r"execTime:0x([0-9a-fA-F]+)", rest)
        hm = re.search(r"dcacheHit:(\d+)", rest)
        instr[iid] = {"t": t, "pc": pc, "name": name, "exec": int(em.group(1), 16) if em else None,
                      "hit": int(hm.group(1)) if hm else None}
    issue, retire = {}, {}
    text = read(dump / "core0.veccore0.ccu.scalar_issque.dump")
    for line in text.splitlines():
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
    dc = {}
    text = read(dump / "core0.veccore0.dcache_log.dump")
    for line in text.splitlines():
        m = re.search(
            r"calc_req_addr\. req pc:(0x[0-9a-fA-F]+).*?aligned_addr_:(0x[0-9a-fA-F]+)",
            line,
        )
        if m:
            dc.setdefault(m.group(1).lower(), []).append(int(m.group(2), 16))
    rows = []
    for iid, it in instr.items():
        if not it["name"].startswith(("LD_", "LDP_")):
            continue
        addrs = dc.get(it["pc"], [])
        data_addrs = [a for a in addrs if 0x100000000 <= a < 0x180000000]
        if not data_addrs:
            continue
        rows.append((issue.get(iid, 10**12), iid, retire.get(iid), it, data_addrs))
    for iss, iid, ret, it, addrs in sorted(rows):
        lat = (ret - iss) if ret is not None else None
        print(f"SIMD id={iid:4d} pc={it['pc']} name={it['name']} issue={iss} retire={ret} lat={lat} "
              f"exec={it['exec']} hit={it['hit']} addrs={[hex(a) for a in addrs]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("mode", choices=["simt", "simd"])
    args = ap.parse_args()
    dump = Path(args.dump)
    if args.mode == "simt":
        parse_simt(dump)
    else:
        parse_simd(dump)


if __name__ == "__main__":
    main()
