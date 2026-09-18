#!/usr/bin/env python3
"""Coarse scalar-load-only comparison: costmodel whitebox vs CAModel windows.

Predicted: sum of `load_per_iteration` over all `model=scalar_load` stages, for
           the factor-1 implementation of the requested mode.
Measured : core0.veccore0 scalar load instruction windows in one program.
  SIMD: first occurrence of each distinct PC among LD_*/LDP_* whose XN address
        is outside the parameter area (0x1022c...).
  SIMT: first occurrence of each distinct DC TagRam address among SIMT_LDG.
This is a scalar-load aggregate, not a per-stage mapping.
"""
import json
import pathlib
import re
import sys


def read(p):
    try:
        return pathlib.Path(p).read_text(errors="replace")
    except FileNotFoundError:
        return ""


def parse_simd(dump):
    instr = {}
    for line in read(dump / "core0.veccore0.instr_log.dump").splitlines():
        m = re.search(
            r"\[(\d+)\] \(PC: (0x[0-9a-fA-F]+)\)\s+(\S+)\s+:\s+"
            r"\(Binary: 0x[0-9a-fA-F]+\)\s+\(ID: (\d+)\)\s+(\S+)(.*)",
            line,
        )
        if not m:
            continue
        name = m.group(5)
        if not name.startswith(("LD_", "LDP_")):
            continue
        rest = m.group(6)
        am = re.search(r"XN:\w+=(0x[0-9a-fA-F]+)", rest)
        ub = re.search(r"accessUb:(\d+)", rest)
        ddr = re.search(r"accessDdr:(\d+)", rest)
        instr[int(m.group(4))] = {
            "id": int(m.group(4)),
            "t": int(m.group(1)),
            "pc": m.group(2).lower(),
            "name": name,
            "addr": am.group(1).lower() if am else None,
            "access_ub": int(ub.group(1)) if ub else None,
            "access_ddr": int(ddr.group(1)) if ddr else None,
        }
    issue, retire = {}, {}
    for line in read(dump / "core0.veccore0.ccu.scalar_issque.dump").splitlines():
        m = re.search(r"\[info\]\s+(\d+)\s+\[Push Instr\] name=(\S+) pc=(\S+) id=(\d+)", line)
        if m:
            issue[int(m.group(4))] = int(m.group(1))
            continue
        m = re.search(r"\[info\]\s+(\d+)\s+\[RETIRE INSTR\] instr\.name=(\S+)\s+instr\.pc=(\S+)\s+instr\.id=(\d+)", line)
        if m:
            retire[int(m.group(4))] = int(m.group(1))
    by_pc = {}
    for iid, it in instr.items():
        addr = it["addr"]
        if not addr or addr.startswith("0x1022c"):
            continue
        # keep only GM/DDR accesses; drop UB/stack staging loads
        if it.get("access_ddr") != 1 or it.get("access_ub") == 1:
            continue
        # CAModel assigns real HBM tensors high virtual addresses; compiler
        # scratch (0x208568/0x80000...) is below 0x1_0000_0000.
        if int(addr, 16) < 0x100000000:
            continue
        if iid not in issue or iid not in retire:
            continue
        rec = {"id": iid, "name": it["name"], "pc": it["pc"], "addr": addr,
               "issue": issue[iid], "retire": retire[iid], "window": retire[iid] - issue[iid]}
        prev = by_pc.get(it["pc"])
        if prev is None or rec["issue"] < prev["issue"]:
            by_pc[it["pc"]] = rec
    rows = sorted(by_pc.values(), key=lambda r: r["issue"])
    return rows


def parse_simt(dump):
    lsu = read(dump / "core0.veccore0.rvec.simt.lsu.dump").splitlines()
    events = {}
    for line in lsu:
        m = re.search(r"\[(\d+)\] (RECV_INSTR|ISSUE_INSTR|RETIRE_INSTR) name=(\S+) id=(\d+)", line)
        if not m or m.group(3) != "SIMT_LDG":
            continue
        iid = int(m.group(4))
        rec = events.setdefault(iid, {"id": iid, "issue": None, "retire": None})
        if m.group(2) == "ISSUE_INSTR":
            rec["issue"] = int(m.group(1))
        elif m.group(2) == "RETIRE_INSTR":
            rec["retire"] = int(m.group(1))
    addr = {}
    for line in read(dump / "core0.veccore0.rvec.simt.dc.dump").splitlines():
        if "[TagRam]" not in line or "isaId:" not in line:
            continue
        m = re.search(r"cmd:(\w+).*?addr:(0x[0-9a-fA-F]+).*?isaId:(\d+)", line)
        if m and m.group(1) == "READ":
            addr[int(m.group(3))] = m.group(2).lower()
    by_addr = {}
    for iid, ev in events.items():
        if ev["issue"] is None or ev["retire"] is None:
            continue
        a = addr.get(iid, f"id{iid}")
        rec = {"id": iid, "addr": a, "issue": ev["issue"], "retire": ev["retire"],
               "window": ev["retire"] - ev["issue"]}
        prev = by_addr.get(a)
        if prev is None or rec["issue"] < prev["issue"]:
            by_addr[a] = rec
    return sorted(by_addr.values(), key=lambda r: r["issue"])


def predicted(report, mode):
    stages = []
    for st in report["stage_model"]["logical_stages"]:
        if st.get("model") != "scalar_load":
            continue
        impl = None
        for cand in st.get("implementations", []):
            im = cand["implementation"]
            if im["mode"] == mode and im["superblock_factor"] == 1:
                impl = cand
                break
        if impl is None:
            continue
        stages.append({
            "id": st["id"],
            "K": st["workload"].get("scalar_load_count_per_iteration"),
            "indirect": st["workload"].get("indirect_scalar_load_count_per_iteration"),
            "load": impl["resource_system_cycles"].get("load_per_iteration", 0.0),
            "source": st.get("source_locations", []),
        })
    return stages


def main():
    report_path = pathlib.Path(sys.argv[1])
    dump = pathlib.Path(sys.argv[2])
    mode = sys.argv[3]
    report = json.load(open(report_path))
    rows = parse_simd(dump) if mode == "simd" else parse_simt(dump)
    pred = predicted(report, mode)
    out = {
        "mode": mode,
        "dump": str(dump),
        "predicted_stages": pred,
        "predicted_load_sum": sum(x["load"] for x in pred),
        "measured_rows": rows,
        "measured_load_sum": sum(x["window"] for x in rows),
    }
    out["err_pct"] = ((out["predicted_load_sum"] - out["measured_load_sum"])
                      / out["measured_load_sum"] * 100.0
                      if out["measured_load_sum"] else None)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
