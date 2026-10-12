#!/usr/bin/env python3
"""Parse CAModel dumps for the strided-load pre-explore experiment.

Input layout (as produced by run_camodel.sh):

  <run_dir>/
    launch_manifest.json          # launch order -> (mode, stride, buffer addrs)
    run.log                       # msopprof stdout
    OPPROF_.../<kernel>/<i>/{dump,simulator}

Only core0.veccore0 is inspected.  The parser deliberately keeps raw simulator
cycle stamps; it does not convert them to ns or SYS_CNT because that conversion
must be calibrated per simulator invocation (the reported ``duration_time``
and dump windows are not always aligned; see README.md discussion).
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CORE = "core0.veccore0"
HEX = r"0x[0-9a-fA-F]+"


def read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except (FileNotFoundError, IsADirectoryError):
        return ""


def to_int(value):
    if value is None:
        return None
    value = value.strip()
    if value.startswith("0x"):
        return int(value, 16)
    return int(value)


def hist_to_str(hist):
    return ",".join(f"{k}:{v}" for k, v in sorted(hist.items()))


def parse_rvec(dump):
    text = read_text(dump / f"{CORE}.rvec.dump")
    matches = re.findall(
        r"core: 0, subcore: (\d+), start: (\d+), tick: (\d+)", text
    )
    if not matches:
        return {}
    subcore, start, tick = matches[-1]
    return {"subcore": int(subcore), "start": int(start), "tick": int(tick),
            "duration": int(tick) - int(start)}


def parse_instr_span(dump):
    """First/last dynamic instruction timestamp from core0.veccore0 instr_log.

    The msopprof ``Core operator results`` duration is an active execution
    window; ``rvec.dump`` start/tick additionally includes core-task
    fetch/prologue/teardown.  ``instr_log`` min/max is the closest dump-side
    equivalent of the active window.
    """
    text = read_text(dump / f"{CORE}.instr_log.dump")
    ts = []
    for line in text.splitlines():
        m = re.search(r"\[info\] \[(\d+)\]", line)
        if m:
            ts.append(int(m.group(1)))
    if not ts:
        return {}
    return {"first": min(ts), "last": max(ts), "duration": max(ts) - min(ts), "count": len(ts)}


def parse_mte_queues(dump):
    """Return {queue: {instr_name: {push, retire, count}}} for MTE queues."""
    out = defaultdict(lambda: defaultdict(lambda: {"push": [], "retire": [], "pc": None}))
    for queue in ("mte1", "mte2", "mte3"):
        text = read_text(dump / f"{CORE}.ccu.{queue}_issque.dump")
        current = {}
        for line in text.splitlines():
            m = re.search(r"\[info\] (\d+) \[Push Instr\] name=(\S+) pc=(\S+) id=(\d+)", line)
            if m:
                t, name, pc, iid = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
                current[iid] = name
                rec = out[queue][name]
                rec["push"].append(t)
                rec["pc"] = pc
                continue
            m = re.search(r"\[info\] (\d+) \[RETIRE INSTR\] instr\.name=(\S+)\s+instr\.pc=(\S+)\s+instr\.id=(\d+)", line)
            if m:
                t, name, pc, iid = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
                out[queue][name]["retire"].append(t)
                out[queue][name]["pc"] = pc
    return {q: {k: v for k, v in names.items()} for q, names in out.items()}


def find_mte_load(mte_queues):
    """Pick the MTE instruction that moves the tile into UB.

    In the current lowering the shaped ``tl.load`` is an MTE2
    ``MOV_SRC_TO_DST_ALIGNv2``.  Keep a fallback for future names so the
    parser still produces evidence instead of silently dropping a case.
    """
    preferred = ("MOV_SRC_TO_DST_ALIGNv2", "MOV_SRC_TO_DST", "MOV")
    candidates = []
    for queue, instrs in mte_queues.items():
        for name, rec in instrs.items():
            if not rec["push"] or not rec["retire"]:
                continue
            if queue == "mte2":
                candidates.append((queue, name, rec))
    for pref in preferred:
        for queue, name, rec in candidates:
            if name == pref or name.startswith(pref):
                return {
                    "queue": queue,
                    "name": name,
                    "push": min(rec["push"]),
                    "retire": max(rec["retire"]),
                    "count": len(rec["push"]),
                    "pc": rec["pc"],
                }
    if candidates:
        queue, name, rec = candidates[0]
        return {"queue": queue, "name": name, "push": min(rec["push"]),
                "retire": max(rec["retire"]), "count": len(rec["push"]), "pc": rec["pc"]}
    return {}


def parse_simd_vf(dump):
    text = read_text(dump / f"{CORE}.rvec.simd.ifu.dump")
    starts = re.findall(r"\[info\] \[?(\d+)\]? Start SIMD VF: .*?instr_num: (\d+)", text)
    if not starts:
        starts = re.findall(r"\[info\] (\d+) Start SIMD VF: .*?instr_num: (\d+)", text)
    if not starts:
        return {}
    # Keep the first VF: BLOCK=32 currently materializes exactly one VF region.
    t, n = starts[0]
    return {"start": int(t), "instr_num": int(n), "vf_count": len(starts)}


def parse_simt_lsu(dump):
    text = read_text(dump / f"{CORE}.rvec.simt.lsu.dump")
    events = defaultdict(lambda: {"name": None, "recv": [], "issue": [], "retire": []})
    for line in text.splitlines():
        m = re.search(r"\[?(\d+)\]? (RECV_INSTR|ISSUE_INSTR|RETIRE_INSTR) name=(\S+) id=(\d+)", line)
        if not m:
            continue
        t, kind, name, iid = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        rec = events[iid]
        rec["name"] = name
        rec[kind.split("_")[0].lower()].append(t)
    out = {}
    for iid, rec in events.items():
        if not rec["issue"] or not rec["retire"]:
            continue
        key = f"{rec['name']}/{iid}"
        out[key] = {
            "name": rec["name"],
            "id": iid,
            "recv": min(rec["recv"]) if rec["recv"] else None,
            "issue": min(rec["issue"]),
            "retire": max(rec["retire"]),
            "active": max(rec["retire"]) - min(rec["issue"]),
        }
    return out


def parse_dc_tag(dump, isa_ids):
    text = read_text(dump / f"{CORE}.rvec.simt.dc.dump")
    reads = []
    all_reads = []
    for line in text.splitlines():
        if "[TagRam]" not in line or " cmd:" not in line:
            continue
        m = re.search(
            r"\[?(\d+)\]?\[TagRam\].*?cmd:(\w+), bank:(\d+), set:(\d+), "
            r"addr:(" + HEX + r"), size:(\d+), cacheLineIdx:(\d+), "
            r"tagRst:(\w+), isaId:(\d+), dispType:(\w+)",
            line,
        )
        if not m:
            continue
        rec = {
            "time": int(m.group(1)),
            "cmd": m.group(2),
            "bank": int(m.group(3)),
            "set": int(m.group(4)),
            "addr": int(m.group(5), 16),
            "size": int(m.group(6)),
            "cache_line_idx": int(m.group(7)),
            "tag_rst": m.group(8),
            "isa_id": int(m.group(9)),
            "disp_type": m.group(10),
        }
        if rec["cmd"] == "READ":
            all_reads.append(rec)
            if not isa_ids or rec["isa_id"] in isa_ids:
                reads.append(rec)
    sizes = Counter(r["size"] for r in reads)
    tag_rst = Counter(r["tag_rst"] for r in reads)
    return {
        "read_requests": len(reads),
        "read_sizes": dict(sizes),
        "read_bytes": sum(r["size"] for r in reads),
        "unique_addr_lines": len({r["addr"] // 128 for r in reads}),
        "unique_cache_line_idx": len({r["cache_line_idx"] for r in reads}),
        "tag_rst": dict(tag_rst),
        "first": min((r["time"] for r in reads), default=None),
        "last": max((r["time"] for r in reads), default=None),
        "all_dc_read_requests": len(all_reads),
    }


def parse_lsu_all(dump):
    """Aggregate SIMT LSU instruction counts per opcode for phase context."""
    text = read_text(dump / f"{CORE}.rvec.simt.lsu.dump")
    counts = Counter()
    for line in text.splitlines():
        m = re.search(r"(RECV_INSTR|ISSUE_INSTR|RETIRE_INSTR) name=(\S+) id=(\d+)", line)
        if m and m.group(1) == "ISSUE_INSTR":
            counts[m.group(2)] += 1
    return dict(counts)


def parse_biu(dump, lo, hi):
    text = read_text(dump / "core0.biu.brif.log.dump")
    sends = []
    recvs = []
    for line in text.splitlines():
        m = re.search(
            r"\[info\] (\d+): (send_rd_cmd|recv_biu_data),.*?addr: (" + HEX + r"), size: (\d+)",
            line,
        )
        if not m:
            continue
        t, kind, addr, size = int(m.group(1)), m.group(2), int(m.group(3), 16), int(m.group(4))
        if not (lo <= addr < hi):
            continue
        rec = {"time": t, "addr": addr, "size": size, "all_recved": "all_recved: 1" in line}
        if kind == "send_rd_cmd":
            port = re.search(r"port: (\S+),", line)
            rec["port"] = port.group(1) if port else None
            rec["gid"] = (lambda x: int(x.group(1)) if x else None)(
                re.search(r"gid: (\d+)", line)
            )
            sends.append(rec)
        else:
            recvs.append(rec)
    # Keep only likely data plane commands; instruction-cache fills use text
    # addresses such as 0x10d..., which never fall inside the input range.
    sizes = Counter(r["size"] for r in sends)
    unique_lines_64 = len({r["addr"] // 64 for r in sends})
    unique_lines_128 = len({r["addr"] // 128 for r in sends})
    return {
        "send_count": len(sends),
        "send_bytes": sum(r["size"] for r in sends),
        "send_sizes": dict(sizes),
        "send_unique_64B_lines": unique_lines_64,
        "send_unique_128B_lines": unique_lines_128,
        "send_ports": dict(Counter(r["port"] for r in sends)),
        "first_send": min((r["time"] for r in sends), default=None),
        "last_send": max((r["time"] for r in sends), default=None),
        "recv_count": len(recvs),
        "recv_bytes": sum(r["size"] for r in recvs),
        "first_recv": min((r["time"] for r in recvs), default=None),
        "last_recv": max((r["time"] for r in recvs), default=None),
        "all_recved_count": sum(1 for r in recvs if r["all_recved"]),
        "send_addr_first": sends[0]["addr"] if sends else None,
        "send_addr_last": sends[-1]["addr"] if sends else None,
    }


def parse_block_windows(run_log):
    """Parse ordered [block_start]/[block_end] pairs from msopprof stdout."""
    text = read_text(run_log)
    events = re.findall(r"\[(\d+)\] \[block_(start|end)\]\s+: AIV, task_id=(\d+), core_id=(\d+), block_id=(\d+)", text)
    starts = []
    ends = []
    for t, kind, task, core, block in events:
        (starts if kind == "start" else ends).append(int(t))
    pairs = []
    for s, e in zip(starts, ends):
        pairs.append({"start": s, "end": e, "duration": e - s})
    return pairs


def parse_reported_durations(run_log):
    """Extract core0.veccore0 duration_time(us) rows in order."""
    text = read_text(run_log)
    durations = []
    for m in re.finditer(r"Core operator results", text):
        section = text[m.end():m.end() + 600]
        rows = re.findall(r"core0\.veccore0\s+([0-9.]+)\s+([0-9.]+)", section)
        # Only accept the first table result after the header.
        if rows:
            durations.append({"duration_us": float(rows[0][0]), "running_us": float(rows[0][1])})
    return durations


def pair_load_and_biu(case, mte, vf, lsu, dc, biu):
    """Return (first_issue, last_retire, engine, instruction_count)."""
    if mte:
        return mte["push"], mte["retire"], f"{mte['queue']}:{mte['name']}", mte["count"]
    if lsu:
        # Use every GM LDG owned by the one tl.load tile.  For BLOCK=32 there
        # is exactly one; larger tiles may lower to several warp instructions.
        ldg = [v for v in lsu.values() if v["name"] == "SIMT_LDG"]
        entries = ldg if ldg else list(lsu.values())
        issue = min(v["issue"] for v in entries)
        retire = max(v["retire"] for v in entries)
        return issue, retire, entries[0]["name"], len(entries)
    return None, None, None, 0


def summarize_case(case, run_dir, kernel_dir, index, blocks, durations):
    mode = case["mode"]
    stride = int(case["stride"])
    block = int(case["block"])
    grid = int(case["grid"])
    x_ptr = int(case["x_ptr"], 16)
    x_bytes = int(case.get("x_bytes", 0))
    dump = kernel_dir / str(index) / "dump"
    if not dump.is_dir():
        dump = kernel_dir / "dump"  # single-launch flat layout
    rvec = parse_rvec(dump)
    instr = parse_instr_span(dump)
    mte_queues = parse_mte_queues(dump)
    mte_load = find_mte_load(mte_queues) if mode == "simd" else {}
    vf = parse_simd_vf(dump) if mode == "simd" else {}
    lsu = parse_simt_lsu(dump) if mode == "simt_only" else {}
    ldg_ids = {v["id"] for v in lsu.values() if v["name"] == "SIMT_LDG"}
    dc = parse_dc_tag(dump, ldg_ids) if mode == "simt_only" else {}
    lsu_counts = parse_lsu_all(dump) if mode == "simt_only" else {}
    biu = parse_biu(dump, x_ptr, x_ptr + max(x_bytes, 1))

    load_issue, load_retire, engine, load_instr_count = pair_load_and_biu(
        case, mte_load, vf, lsu, dc, biu
    )
    total = rvec.get("duration")
    active = instr.get("duration")
    result = {
        "index": index,
        "mode": mode,
        "stride": stride,
        "block": block,
        "grid": grid,
        "element_bytes": 4,
        "span_bytes": (block - 1) * stride * 4 + 4,
        "unique_lines_128_aligned": len({(x_ptr + i * stride * 4) // 128 for i in range(block)}),
        "rvec": rvec,
        "instr_span": instr,
        "total_cycles": total,
        "task_cycles": total,
        "instr_active_cycles": active,
        "load_engine": engine,
        "load_instr_count": load_instr_count,
        "load_issue": load_issue,
        "load_retire": load_retire,
        "load_active_cycles": (load_retire - load_issue) if load_issue is not None and load_retire is not None else None,
        "biu": biu,
        "biu_fill_span": None,
        "cache": dc,
        "simd_vf": vf,
        "simt_lsu_counts": lsu_counts,
        "mte_queues": {
            q: {name: {"push": min(r["push"]) if r["push"] else None,
                       "retire": max(r["retire"]) if r["retire"] else None,
                       "count": len(r["push"])}
                for name, r in instrs.items()}
            for q, instrs in mte_queues.items()
        },
    }
    # Phase decomposition.  Prefer the dynamic instruction window; fall back
    # to the AIV task window if the instruction log is unavailable.
    if active is not None and load_issue is not None:
        result["pre_load_cycles"] = load_issue - instr["first"]
    elif load_issue is not None and rvec:
        result["pre_load_cycles"] = load_issue - rvec["start"]
    if active is not None and load_retire is not None:
        result["post_load_cycles"] = instr["last"] - load_retire
    elif load_retire is not None and rvec:
        result["post_load_cycles"] = rvec["tick"] - load_retire
    if biu.get("first_recv") is not None and biu.get("first_send") is not None:
        result["biu_fill_span"] = biu["last_recv"] - biu["first_send"]
    if load_issue is not None and biu.get("first_send") is not None:
        result["load_issue_to_first_biu"] = biu["first_send"] - load_issue
    if biu.get("last_recv") is not None and load_retire is not None:
        result["last_biu_to_retire"] = load_retire - biu["last_recv"]
    if active:
        result["load_active_frac"] = (result["load_active_cycles"] or 0) / active
        result["biu_fill_frac"] = (result["biu_fill_span"] or 0) / active
    if index < len(blocks):
        result["aiv_block"] = blocks[index]
    if index < len(durations):
        result["reported_duration_us"] = durations[index]
        if active:
            result["dump_cycles_per_reported_us"] = active / durations[index]["duration_us"]
        if total:
            result["task_cycles_per_reported_us"] = total / durations[index]["duration_us"]
    return result


def write_csv(results, path):
    fields = [
        "index", "mode", "stride", "block", "span_bytes",
        "unique_lines_128_aligned", "task_cycles", "instr_active_cycles",
        "aiv_block_duration", "reported_duration_us",
        "dump_cycles_per_reported_us", "task_cycles_per_reported_us",
        "pre_load_cycles", "load_active_cycles", "post_load_cycles",
        "load_issue_to_first_biu", "biu_fill_span", "last_biu_to_retire",
        "load_engine", "load_instr_count",
        "biu_send_count", "biu_send_bytes", "biu_send_sizes",
        "biu_unique_128B_lines", "biu_first_send", "biu_last_send",
        "biu_recv_count", "biu_recv_bytes", "biu_first_recv", "biu_last_recv",
        "dc_read_requests", "dc_read_bytes", "dc_read_sizes", "dc_unique_lines",
        "dc_tag_rst", "dc_first", "dc_last",
        "simt_lsu_counts", "mte_queues",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["aiv_block_duration"] = (r.get("aiv_block") or {}).get("duration")
            row["biu_send_sizes"] = hist_to_str(r.get("biu", {}).get("send_sizes", {}))
            row["biu_unique_128B_lines"] = r.get("biu", {}).get("send_unique_128B_lines")
            row["biu_first_send"] = r.get("biu", {}).get("first_send")
            row["biu_last_send"] = r.get("biu", {}).get("last_send")
            row["biu_recv_count"] = r.get("biu", {}).get("recv_count")
            row["biu_recv_bytes"] = r.get("biu", {}).get("recv_bytes")
            row["biu_first_recv"] = r.get("biu", {}).get("first_recv")
            row["biu_last_recv"] = r.get("biu", {}).get("last_recv")
            row["dc_read_requests"] = r.get("cache", {}).get("read_requests")
            row["dc_read_bytes"] = r.get("cache", {}).get("read_bytes")
            row["dc_read_sizes"] = hist_to_str(r.get("cache", {}).get("read_sizes", {}))
            row["dc_unique_lines"] = r.get("cache", {}).get("unique_cache_line_idx")
            row["dc_tag_rst"] = hist_to_str(r.get("cache", {}).get("tag_rst", {}))
            row["dc_first"] = r.get("cache", {}).get("first")
            row["dc_last"] = r.get("cache", {}).get("last")
            row["simt_lsu_counts"] = hist_to_str(r.get("simt_lsu_counts", {}))
            row["mte_queues"] = json.dumps(r.get("mte_queues", {}), sort_keys=True)
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--kernel-name", default="strided_load_kernel")
    ap.add_argument("--profile", type=Path, default=None)
    ap.add_argument("-o", "--output-json", type=Path, default=Path("strided_camodel_parsed.json"))
    ap.add_argument("--output-csv", type=Path, default=Path("strided_camodel_parsed.csv"))
    ap.add_argument("--profile-name", default=None, help="OPPROF directory name (default: first found)")
    args = ap.parse_args()

    run_dir = args.run_dir
    manifest = json.loads(read_text(run_dir / "launch_manifest.json") or "{}")
    cases = manifest.get("cases", [])
    if not cases:
        raise SystemExit(f"no launch_manifest.json cases found in {run_dir}")
    if args.profile:
        profiles = [args.profile]
    else:
        profiles = sorted(run_dir.glob("OPPROF_*"))
    if args.profile_name:
        profiles = [p for p in profiles if p.name == args.profile_name]
    if not profiles:
        raise SystemExit("no OPPROF_* directory found")
    profile = profiles[0]
    kernel_dir = profile / args.kernel_name
    if not kernel_dir.is_dir():
        # flat single-launch layout
        kernel_dir = profile
    blocks = parse_block_windows(run_dir / "run.log")
    durations = parse_reported_durations(run_dir / "run.log")
    results = []
    for index, case in enumerate(cases):
        results.append(summarize_case(case, run_dir, kernel_dir, index, blocks, durations))
    output = {
        "run_dir": str(run_dir),
        "profile": str(profile),
        "kernel_dir": str(kernel_dir),
        "cases": results,
        "block_windows": blocks,
        "reported_durations": durations,
    }
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(results, args.output_csv)
    print(f"wrote {args.output_json} and {args.output_csv} ({len(results)} cases)")


if __name__ == "__main__":
    main()
