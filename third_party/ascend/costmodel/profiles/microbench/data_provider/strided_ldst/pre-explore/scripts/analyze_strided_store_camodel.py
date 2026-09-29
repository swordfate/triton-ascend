#!/usr/bin/env python3
"""Parse CAModel dumps for the shaped strided ``tl.store`` pre-explore.

This mirrors ``analyze_strided_camodel.py`` for stores.  It keeps raw
simulator timestamp cycles; it does not convert to ns/SYS_CNT.

Input layout produced by ``run_store_camodel.sh``:

  <run_dir>/
    launch_manifest.json
    run.log
    OPPROF_.../strided_store_kernel/<i>/{dump,simulator}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_strided_camodel import (  # noqa: E402
    CORE,
    HEX,
    parse_block_windows,
    parse_instr_span,
    parse_lsu_all,
    parse_mte_queues,
    parse_reported_durations,
    parse_rvec,
    parse_simt_lsu,
    read_text,
)


def hist_to_str(hist):
    return ",".join(f"{k}:{v}" for k, v in sorted(hist.items()))


def find_mte_store(mte_queues):
    """Pick the MTE3 UB->GM instruction (Triton SIMD shaped store)."""
    candidates = []
    for queue, instrs in mte_queues.items():
        for name, rec in instrs.items():
            if not rec["push"] or not rec["retire"]:
                continue
            if queue == "mte3":
                candidates.append((queue, name, rec))
    for queue, name, rec in candidates:
        if name == "MOV_SRC_TO_DST_ALIGNv2" or name.startswith("MOV_SRC_TO_DST"):
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
        return {
            "queue": queue,
            "name": name,
            "push": min(rec["push"]),
            "retire": max(rec["retire"]),
            "count": len(rec["push"]),
            "pc": rec["pc"],
        }
    return {}


def parse_wait_flag(dump):
    """SIMD store: WAIT_FLAG push/release around MTE3 instruction."""
    text = read_text(dump / f"{CORE}.ccu.mte3_issque.dump")
    pushes = []
    releases = []
    for line in text.splitlines():
        m = re.search(r"\[info\] (\d+) \[Push Instr\] name=WAIT_FLAG", line)
        if m:
            pushes.append(int(m.group(1)))
        m = re.search(r"\[info\] (\d+) \[release wait_flag\]", line)
        if m:
            releases.append(int(m.group(1)))
    return {
        "push": min(pushes) if pushes else None,
        "release": min(releases) if releases else None,
        "count": len(pushes),
    }


def parse_simd_vf_store(dump):
    """VF instructions that materialize the value vector in UB."""
    text = read_text(dump / f"{CORE}.rvec.simd.ifu.dump")
    starts = re.findall(r"\[info\] \[?(\d+)\]? Start SIMD VF: .*?instr_num: (\d+)", text)
    if not starts:
        starts = re.findall(r"\[info\] (\d+) Start SIMD VF: .*?instr_num: (\d+)", text)
    if not starts:
        return {}
    stores = []
    for line in text.splitlines():
        m = re.search(r"\[info\] \[?(\d+)\]? .*?RV_VSTI", line)
        if m:
            stores.append(int(m.group(1)))
    return {
        "vf_start": int(starts[0][0]),
        "instr_num": int(starts[0][1]),
        "vf_count": len(starts),
        "vsti_first": min(stores) if stores else None,
    }


def parse_dc_tag_store(dump, isa_ids):
    """SIMT DCache TagRam WRITE requests owned by SIMT_STG instructions."""
    text = read_text(dump / f"{CORE}.rvec.simt.dc.dump")
    writes = []
    all_writes = []
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
        if rec["cmd"] == "WRITE":
            all_writes.append(rec)
            if not isa_ids or rec["isa_id"] in isa_ids:
                writes.append(rec)
    sizes = Counter(r["size"] for r in writes)
    tag_rst = Counter(r["tag_rst"] for r in writes)
    return {
        "write_requests": len(writes),
        "write_sizes": dict(sizes),
        "write_bytes": sum(r["size"] for r in writes),
        "unique_addr_lines": len({r["addr"] // 128 for r in writes}),
        "unique_cache_line_idx": len({r["cache_line_idx"] for r in writes}),
        "tag_rst": dict(tag_rst),
        "first": min((r["time"] for r in writes), default=None),
        "last": max((r["time"] for r in writes), default=None),
        "all_dc_write_requests": len(all_writes),
    }


def parse_biu_write(dump, lo, hi):
    """BIU write path events for addresses inside the output buffer."""
    text = read_text(dump / "core0.biu.bwif.log.dump")
    send_aw = []
    recv_cmd = []
    recv_data = []
    send_data = []
    acks = []
    bsps = []
    for line in text.splitlines():
        # send_aw_cmd has wr_len and addr.
        m = re.search(
            r"\[info\] \[?(\d+)\]?: send_aw_cmd, port: (\S+), align_addr (" + HEX +
            r"), wr_len (\d+), addr: (" + HEX + r"),", line)
        if m:
            addr = int(m.group(5), 16)
            if lo <= addr < hi:
                send_aw.append({"time": int(m.group(1)), "addr": addr,
                                "size": int(m.group(4)), "port": m.group(2)})
            continue
        # command receive (SIMD mte / SIMT vdcache)
        m = re.search(
            r"\[info\] \[?(\d+)\]?: recv_(?:mte|simt)_wr_cmd, port: (\S+), addr: (" +
            HEX + r"),", line)
        if m:
            addr = int(m.group(3), 16)
            if lo <= addr < hi:
                recv_cmd.append({"time": int(m.group(1)), "addr": addr, "port": m.group(2)})
            continue
        # data movement / ack / response
        m = re.search(
            r"\[info\] \[?(\d+)\]?: (recv_wr_data|send_wr_data),.*?addr: (" + HEX +
            r"), size: (\d+)", line)
        if m:
            addr = int(m.group(3), 16)
            if lo <= addr < hi:
                rec = {"time": int(m.group(1)), "addr": addr, "size": int(m.group(4))}
                (recv_data if m.group(2) == "recv_wr_data" else send_data).append(rec)
            continue
        m = re.search(
            r"\[info\] \[?(\d+)\]?: (recv_wack|send_data_rsp|recv_brsp),.*?addr: ?(" +
            HEX + r")", line)
        if m:
            addr = int(m.group(3), 16)
            if lo <= addr < hi:
                rec = {"time": int(m.group(1)), "addr": addr, "kind": m.group(2)}
                if m.group(2) == "recv_wack":
                    acks.append(rec)
                else:
                    bsps.append(rec)
            continue
    send_sizes = Counter(r["size"] for r in send_aw)
    recv_sizes = Counter(r["size"] for r in recv_data)
    return {
        "send_aw_count": len(send_aw),
        "send_aw_bytes": sum(r["size"] for r in send_aw),
        "send_aw_sizes": dict(send_sizes),
        "send_aw_ports": dict(Counter(r["port"] for r in send_aw)),
        "send_aw_first": min((r["time"] for r in send_aw), default=None),
        "send_aw_last": max((r["time"] for r in send_aw), default=None),
        "recv_cmd_count": len(recv_cmd),
        "recv_cmd_first": min((r["time"] for r in recv_cmd), default=None),
        "recv_cmd_last": max((r["time"] for r in recv_cmd), default=None),
        "recv_data_count": len(recv_data),
        "recv_data_bytes": sum(r["size"] for r in recv_data),
        "recv_data_sizes": dict(recv_sizes),
        "recv_data_first": min((r["time"] for r in recv_data), default=None),
        "recv_data_last": max((r["time"] for r in recv_data), default=None),
        "send_data_count": len(send_data),
        "send_data_bytes": sum(r["size"] for r in send_data),
        "send_data_first": min((r["time"] for r in send_data), default=None),
        "send_data_last": max((r["time"] for r in send_data), default=None),
        "wack_count": len(acks),
        "wack_first": min((r["time"] for r in acks), default=None),
        "wack_last": max((r["time"] for r in acks), default=None),
        "response_last": max((r["time"] for r in bsps), default=None),
        "unique_128B_lines": len({r["addr"] // 128 for r in send_aw}),
    }


def pair_store(case, mte, lsu):
    """Return (first_issue, last_retire, engine, instruction_count)."""
    if mte:
        return mte["push"], mte["retire"], f"{mte['queue']}:{mte['name']}", mte["count"]
    if lsu:
        st = [v for v in lsu.values() if v["name"] == "SIMT_STG"]
        entries = st if st else list(lsu.values())
        return (min(v["issue"] for v in entries),
                max(v["retire"] for v in entries),
                entries[0]["name"], len(entries))
    return None, None, None, 0


def summarize_case(case, kernel_dir, index, blocks, durations):
    mode = case["mode"]
    stride = int(case["stride"])
    block = int(case["block"])
    grid = int(case["grid"])
    lo = int(case["out_ptr"], 16)
    hi = lo + max(int(case.get("out_bytes", 0)), 1)
    dump = kernel_dir / str(index) / "dump"
    if not dump.is_dir():
        dump = kernel_dir / "dump"
    rvec = parse_rvec(dump)
    instr = parse_instr_span(dump)
    mte_queues = parse_mte_queues(dump)
    mte = find_mte_store(mte_queues) if mode == "simd" else {}
    wait = parse_wait_flag(dump) if mode == "simd" else {}
    vf = parse_simd_vf_store(dump) if mode == "simd" else {}
    lsu = parse_simt_lsu(dump) if mode == "simt_only" else {}
    st_ids = {v["id"] for v in lsu.values() if v["name"] == "SIMT_STG"}
    dc = parse_dc_tag_store(dump, st_ids) if mode == "simt_only" else {}
    lsu_counts = parse_lsu_all(dump) if mode == "simt_only" else {}
    biu = parse_biu_write(dump, lo, hi)

    issue, retire, engine, instr_count = pair_store(case, mte, lsu)
    active = instr.get("duration")
    result = {
        "index": index,
        "mode": mode,
        "stride": stride,
        "block": block,
        "grid": grid,
        "element_bytes": 4,
        "out_ptr": hex(lo),
        "useful_bytes": block * 4,
        "span_bytes": (block - 1) * stride * 4 + 4,
        "unique_lines_128_aligned": len({(lo + i * stride * 4) // 128 for i in range(block)}),
        "rvec": rvec,
        "instr_span": instr,
        "task_cycles": rvec.get("duration"),
        "instr_active_cycles": active,
        "store_engine": engine,
        "store_instr_count": instr_count,
        "store_issue": issue,
        "store_retire": retire,
        "store_active_cycles": (retire - issue) if issue is not None and retire is not None else None,
        "biu": biu,
        "dc": dc,
        "wait_flag": wait,
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
    if active is not None and issue is not None:
        result["pre_store_cycles"] = issue - instr["first"]
    if active is not None and retire is not None:
        result["post_store_cycles"] = instr["last"] - retire
    first_biu = biu.get("send_aw_first")
    last_biu = max(
        biu.get("send_aw_last") or -10**18,
        biu.get("recv_data_last") or -10**18,
        biu.get("wack_last") or -10**18,
        biu.get("response_last") or -10**18,
    )
    if first_biu is None or last_biu < -10**17:
        last_biu = None
    result["biu_first_send"] = first_biu
    result["biu_last_event"] = last_biu
    if issue is not None and first_biu is not None:
        result["store_issue_to_first_biu"] = first_biu - issue
    if first_biu is not None and last_biu is not None:
        result["biu_write_span"] = last_biu - first_biu
    if retire is not None and last_biu is not None:
        result["biu_to_retire"] = retire - last_biu
    if index < len(blocks):
        result["aiv_block"] = blocks[index]
    if index < len(durations):
        result["reported_duration_us"] = durations[index]
    return result


def write_csv(results, path):
    fields = [
        "index", "mode", "stride", "block", "unique_lines_128_aligned",
        "task_cycles", "instr_active_cycles", "pre_store_cycles",
        "store_active_cycles", "post_store_cycles",
        "store_issue_to_first_biu", "biu_write_span", "biu_to_retire",
        "store_engine", "store_instr_count",
        "biu_send_aw_count", "biu_send_aw_bytes", "biu_send_aw_sizes",
        "biu_unique_128B_lines", "biu_send_aw_first", "biu_send_aw_last",
        "biu_recv_data_count", "biu_recv_data_bytes", "biu_recv_data_first",
        "biu_recv_data_last", "biu_wack_last", "biu_response_last",
        "dc_write_requests", "dc_write_bytes", "dc_write_sizes",
        "dc_unique_lines", "dc_tag_rst", "dc_first", "dc_last",
        "simt_lsu_counts", "mte_queues",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            biu = r.get("biu", {})
            dc = r.get("dc", {}) or {}
            row = dict(r)
            row.update({
                "biu_send_aw_count": biu.get("send_aw_count"),
                "biu_send_aw_bytes": biu.get("send_aw_bytes"),
                "biu_send_aw_sizes": hist_to_str(biu.get("send_aw_sizes", {})),
                "biu_unique_128B_lines": biu.get("unique_128B_lines"),
                "biu_send_aw_first": biu.get("send_aw_first"),
                "biu_send_aw_last": biu.get("send_aw_last"),
                "biu_recv_data_count": biu.get("recv_data_count"),
                "biu_recv_data_bytes": biu.get("recv_data_bytes"),
                "biu_recv_data_first": biu.get("recv_data_first"),
                "biu_recv_data_last": biu.get("recv_data_last"),
                "biu_wack_last": biu.get("wack_last"),
                "biu_response_last": biu.get("response_last"),
                "dc_write_requests": dc.get("write_requests"),
                "dc_write_bytes": dc.get("write_bytes"),
                "dc_write_sizes": hist_to_str(dc.get("write_sizes", {})),
                "dc_unique_lines": dc.get("unique_cache_line_idx"),
                "dc_tag_rst": hist_to_str(dc.get("tag_rst", {})),
                "dc_first": dc.get("first"),
                "dc_last": dc.get("last"),
                "simt_lsu_counts": hist_to_str(r.get("simt_lsu_counts", {})),
                "mte_queues": json.dumps(r.get("mte_queues", {}), sort_keys=True),
            })
            row["biu"] = None
            row["dc"] = None
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--kernel-name", default="strided_store_kernel")
    ap.add_argument("-o", "--output-json", type=Path, default=Path("strided_store_parsed.json"))
    ap.add_argument("--output-csv", type=Path, default=Path("strided_store_parsed.csv"))
    args = ap.parse_args()

    run_dir = args.run_dir
    manifest = json.loads(read_text(run_dir / "launch_manifest.json") or "{}")
    cases = manifest.get("cases", [])
    if not cases:
        raise SystemExit(f"no launch_manifest.json cases found in {run_dir}")
    profiles = sorted(run_dir.glob("OPPROF_*"))
    if not profiles:
        raise SystemExit("no OPPROF_* directory found")
    profile = profiles[0]
    kernel_dir = profile / args.kernel_name
    if not kernel_dir.is_dir():
        kernel_dir = profile
    blocks = parse_block_windows(run_dir / "run.log")
    durations = parse_reported_durations(run_dir / "run.log")
    results = [summarize_case(case, kernel_dir, i, blocks, durations)
               for i, case in enumerate(cases)]
    payload = {
        "run_dir": str(run_dir),
        "profile": str(profile),
        "kernel_dir": str(kernel_dir),
        "cases": results,
        "block_windows": blocks,
        "reported_durations": durations,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(results, args.output_csv)
    print(f"wrote {args.output_json} and {args.output_csv} ({len(results)} cases)")


if __name__ == "__main__":
    main()
