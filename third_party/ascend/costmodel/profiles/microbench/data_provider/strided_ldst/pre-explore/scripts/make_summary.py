#!/usr/bin/env python3
"""Join the block32/block128 parsed CAModel results into one summary CSV/MD."""
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent  # pre-explore root (scripts/ lives directly under it)
RUNS = [("block32", HERE / "results/block32/parsed.json"),
        ("block64", HERE / "results/block64/parsed.json"),
        ("block128", HERE / "results/block128/parsed.json")]
OUT_CSV = HERE / "results/strided_load_summary.csv"
OUT_MD = HERE / "results/strided_load_summary.md"

FIELDS = [
    "run", "mode", "block", "stride",
    "useful_bytes", "span_bytes", "unique_lines_128_aligned",
    "task_cycles", "instr_active_cycles",
    "pre_load_cycles", "load_active_cycles", "post_load_cycles",
    "issue_to_first_biu", "biu_fill_span", "biu_to_retire",
    "load_instr_count", "load_engine",
    "biu_send_count", "biu_send_bytes", "biu_send_sizes",
    "biu_recv_count", "biu_recv_bytes", "biu_fill_span_raw",
    "biu_unique_128B_lines",
    "dc_read_requests", "dc_read_bytes", "dc_read_sizes", "dc_unique_lines",
    "dc_tag_rst",
]


def sizes_to_str(hist):
    return ",".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: int(kv[0])))


def main():
    rows = []
    for run, path in RUNS:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for c in data["cases"]:
            biu = c.get("biu", {})
            dc = c.get("cache", {}) or {}
            row = {
                "run": run,
                "mode": c["mode"],
                "block": c["block"],
                "stride": c["stride"],
                "useful_bytes": c["block"] * 4,
                "span_bytes": c["span_bytes"],
                "unique_lines_128_aligned": c["unique_lines_128_aligned"],
                "task_cycles": c.get("task_cycles"),
                "instr_active_cycles": c.get("instr_active_cycles"),
                "pre_load_cycles": c.get("pre_load_cycles"),
                "load_active_cycles": c.get("load_active_cycles"),
                "post_load_cycles": c.get("post_load_cycles"),
                "issue_to_first_biu": c.get("load_issue_to_first_biu"),
                "biu_fill_span": c.get("biu_fill_span"),
                "biu_to_retire": c.get("last_biu_to_retire"),
                "load_instr_count": c.get("load_instr_count"),
                "load_engine": c.get("load_engine"),
                "biu_send_count": biu.get("send_count"),
                "biu_send_bytes": biu.get("send_bytes"),
                "biu_send_sizes": sizes_to_str(biu.get("send_sizes", {})),
                "biu_recv_count": biu.get("recv_count"),
                "biu_recv_bytes": biu.get("recv_bytes"),
                "biu_fill_span_raw": c.get("biu_fill_span"),
                "biu_unique_128B_lines": biu.get("send_unique_128B_lines"),
                "dc_read_requests": dc.get("read_requests"),
                "dc_read_bytes": dc.get("read_bytes"),
                "dc_read_sizes": sizes_to_str(dc.get("read_sizes", {})),
                "dc_unique_lines": dc.get("unique_cache_line_idx"),
                "dc_tag_rst": ",".join(f"{k}:{v}" for k, v in sorted(dc.get("tag_rst", {}).items())),
            }
            rows.append(row)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| run | mode | stride | block | useful B | lines | task cyc | active cyc | pre | load | post | load: issue->BIU | BIU fill | BIU->ret | BIU cmds | BIU sizes | DC reqs | DC sizes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in rows:
        lines.append(
            f'| {r["run"]} | {r["mode"]} | {r["stride"]} | {r["block"]} | {r["useful_bytes"]} | '
            f'{r["unique_lines_128_aligned"]} | {r["task_cycles"]} | {r["instr_active_cycles"]} | '
            f'{r["pre_load_cycles"]} | {r["load_active_cycles"]} | {r["post_load_cycles"]} | '
            f'{r["issue_to_first_biu"]} | {r["biu_fill_span"]} | {r["biu_to_retire"]} | '
            f'{r["biu_send_count"]} | {r["biu_send_sizes"]} | {r["dc_read_requests"] or "-"} | '
            f'{r["dc_read_sizes"] or "-"} |'
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV} and {OUT_MD} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
